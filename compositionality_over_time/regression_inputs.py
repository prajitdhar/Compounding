#!/usr/bin/env python3
"""Freeze completed Google wide feature Parquet files as verified R-ready CSVs.

This adapter reads only the small final benchmark feature tables, never the
corpus-sized feature_inputs. Source Parquet, feature caches and manifests remain
unchanged. A snapshot has a fixed scientific identity and cannot be extended or
overwritten with different inputs; select a new snapshot root for a new study.
"""
from __future__ import annotations

import argparse
import contextlib
import fcntl
import hashlib
import itertools
import json
import os
import re
import sys
from pathlib import Path
from types import SimpleNamespace

import pandas as pd

SOURCE_DIRECTORY = Path(__file__).resolve().parent / "feature_helpers"
if str(SOURCE_DIRECTORY) not in sys.path:
    sys.path.append(str(SOURCE_DIRECTORY))
from coha_feature_pipeline import read_ratings
from pipeline_source_info import sha256_file, write_json

SNAPSHOT_VERSION = 1
FEATURE_DEFINITION_VERSION = 3
BETA_POLICY = "clipped_unit_interval"
RATING_COLUMNS = ["modifier", "head", "avgModifier", "stdevModifier", "avgHead",
                  "stdevHead", "compositionality", "stdevHeadModifier", "is_adj",
                  "compound", "source", "is_original"]
ROW_KEYS = ["modifier", "head", "source", "compound", "is_adj", "is_original"]
DEFAULT_GRID = {"vocabs": [1000, 3000, 5000, 10000, 30000, 50000],
                "weightings": ["RAW", "PPMI"],
                "settings": ["Agnostic", "Pooled", "Aware"],
                "taggings": ["UnTagged", "Tagged"],
                "timespans": [10000, 10, 20, 50, 100],
                "cutoffs": [0, 10, 50, 100, 500, 1000]}
POLICIES = {"UnTagged": "all_lexical_bigrams", "Tagged": "adj_noun_or_noun_noun"}


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def feature_filename(kind, setting, weight, tagging, vocab, width, cutoff):
    setting = setting.removeprefix("Compound")
    return f"{kind}_Compound{setting}_withSetting_{weight}_{tagging}_ctx{vocab}_{width}_{cutoff}_na.csv"


def _grid(args, default=None):
    result = {}
    for name, values in (default or DEFAULT_GRID).items():
        requested = getattr(args, name, None)
        if requested is None:
            result[name] = list(values)
            continue
        tokens = [part for item in requested for part in re.split(r"[,\s]+", str(item)) if part]
        if name in ("vocabs", "timespans", "cutoffs"):
            try:
                tokens = [int(part) for part in tokens]
            except ValueError as error:
                raise ValueError(f"Invalid {name}") from error
            if any(value < (0 if name == "cutoffs" else 1) for value in tokens):
                raise ValueError(f"Invalid {name}")
        else:
            if name == "settings":
                tokens = [part.removeprefix("Compound") for part in tokens]
            if any(part not in DEFAULT_GRID[name] for part in tokens):
                raise ValueError(f"Invalid {name}: {tokens}")
        if not tokens:
            raise ValueError(f"Empty {name}")
        result[name] = list(dict.fromkeys(tokens))
    return result


def _files(grid):
    for vocab, weight, tagging, width, cutoff, setting in itertools.product(
            grid["vocabs"], grid["weightings"], grid["taggings"], grid["timespans"],
            grid["cutoffs"], grid["settings"]):
        for kind in (["features"] if width == 10000 else ["features", "temporal"]):
            yield feature_filename(kind, setting, weight, tagging, vocab, width, cutoff), {
                "kind": kind, "setting": setting, "weight": weight, "tagging": tagging,
                "vocab": vocab, "width": width, "cutoff": cutoff}


def _stable_record(path):
    stat = path.stat()
    return (stat.st_size, stat.st_mtime_ns, stat.st_ino)


def _hash_checked(path, expected=None):
    before = _stable_record(path)
    digest = sha256_file(path)
    if before != _stable_record(path):
        raise RuntimeError(f"File changed while being read: {path}")
    if expected is not None and digest != expected:
        raise ValueError(f"SHA256 mismatch: {path}")
    return digest


def _ratings():
    directory = Path(os.environ.get("RATINGS_DIR", "data/ratings")).expanduser()
    files = {"reddy90": directory / "reddy_90.txt", "cordeiro90": directory / "cordeiro_90.txt",
             "cordeiro100": directory / "cordeiro_100.txt"}
    hashes = {name: _hash_checked(path) for name, path in files.items()}
    frames = {tag: read_ratings(SimpleNamespace(**files, tag=tag == "Tagged"))
              for tag in POLICIES}
    return frames, hashes


def _rating_payload(frame, label):
    if frame.columns.duplicated().any() or not set(RATING_COLUMNS).issubset(frame.columns):
        raise ValueError(f"Missing/duplicate rating columns: {label}")
    payload = frame[RATING_COLUMNS].copy()
    for name in ("is_adj", "is_original"):
        normalized = payload[name].astype(str).str.lower()
        if not normalized.isin(["true", "false", "1", "0"]).all():
            raise ValueError(f"Invalid rating flags: {label}")
        payload[name] = normalized.isin(["true", "1"])
    if payload[ROW_KEYS].isna().any().any() or payload.duplicated(ROW_KEYS).any():
        raise ValueError(f"Missing/duplicate rating rows: {label}")
    return payload.sort_values(ROW_KEYS).reset_index(drop=True)


def _validate_frame(frame, expected, label):
    payload = _rating_payload(frame, label)
    try:
        pd.testing.assert_frame_equal(payload, _rating_payload(expected, "expected ratings"),
                                      check_dtype=False, check_exact=True)
    except AssertionError as error:
        raise ValueError(f"Rating roster or payload mismatch: {label}") from error
    features = [column for column in frame if column not in RATING_COLUMNS]
    if any(not re.fullmatch(r"[A-Za-z0-9_]+:\d+", column) for column in features):
        raise ValueError(f"Unexpected feature columns: {label}")
    if any(not pd.api.types.is_numeric_dtype(frame[column]) and not frame[column].isna().all()
           for column in features):
        raise ValueError(f"Non-numeric feature columns: {label}")
    for column in features:
        if column.startswith("beta:") and not frame[column].dropna().between(0.0, 1.0).all():
            raise ValueError(f"Unbounded beta in clipped-policy feature table: {label}, {column}")
    # All missing values remain missing. R is responsible for fold-local
    # preprocessing and whether a feature set has enough usable predictors.
    return payload


def _source_plan(root, grid, rating_hashes):
    root = root.resolve()
    mode = root if root.name == "local5" else root / "local5"
    if not mode.is_dir():
        raise ValueError(f"Missing Google local5 feature directory: {mode}")
    manifests, records, prepares, vocabularies = {}, {}, {}, set()
    for csv_name, parameters in _files(grid):
        kind, weight, tag = (parameters[name] for name in ("kind", "weight", "tagging"))
        vocab, width = parameters["vocab"], parameters["width"]
        manifest_name = f"{kind}_{weight}_{tag}_ctx{vocab}_{width}_manifest.json"
        if manifest_name not in manifests:
            path = mode / manifest_name
            if not path.is_file():
                raise ValueError(f"Missing completed feature manifest: {path}")
            before = _stable_record(path)
            value = json.loads(path.read_text())
            block = value.get("source_info", {})
            population = value.get("compound_population")
            expected_population = "adj_noun_and_noun_noun" if tag == "Tagged" else POLICIES[tag]
            if (value.get("schema_version") != 3 or value.get("format") != "parquet"
                    or value.get("feature_definition_version") != FEATURE_DEFINITION_VERSION
                    or value.get("beta_policy") != BETA_POLICY
                    or value.get("context_mode") != "local5"
                    or value.get("tagged_targets") != (tag == "Tagged")
                    or value.get("ppmi") != (weight == "PPMI")
                    or population != expected_population
                    or block.get("status") != "complete" or block.get("stage") != kind
                    or value.get("context_size") != vocab or value.get("temporal") != width):
                raise ValueError(f"Incompatible/incomplete feature manifest: {path}")
            scientific_fields = ("feature_definition_version", "beta_policy", "compound_population",
                                 "tagged_targets", "ppmi", "context_mode", "context_size",
                                 "temporal", "context_vocabulary")
            if any(block.get("parameters", {}).get(name) != value.get(name)
                   for name in scientific_fields):
                raise ValueError(f"Contradictory scientific source information: {path}")
            context = value.get("context_vocabulary", {})
            if (context.get("requested_size") != vocab or context.get("actual_size") != vocab
                    or context.get("fixed_across_time_bins_and_settings") is not True
                    or not context.get("sha256")):
                raise ValueError(f"Incompatible context vocabulary: {path}")
            vocabularies.add(context["sha256"])
            prepare = value.get("prepare_fingerprint")
            if not prepare or (tag in prepares and prepares[tag] != prepare):
                raise ValueError(f"Mixed feature preparations for {tag}: {path}")
            prepares[tag] = prepare
            # File basenames can differ from rating aliases; the writer records
            # rating inputs under their alias names.
            declared_ratings = {item.get("name"): item.get("sha256") for item in block.get("inputs", [])
                                if item.get("name") in rating_hashes}
            if declared_ratings and declared_ratings != rating_hashes:
                raise ValueError(f"Feature rating hashes differ from the current study: {path}")
            digest = _hash_checked(path)
            if before != _stable_record(path):
                raise RuntimeError(f"Manifest changed while planning snapshot: {path}")
            manifests[manifest_name] = {"path": str(path), "sha256": digest, "value": value}
        manifest = manifests[manifest_name]["value"]
        if parameters["cutoff"] not in manifest.get("cutoffs", []) or (
                f"Compound{parameters['setting']}" not in manifest.get("conditions", [])):
            raise ValueError(f"Requested setting/cutoff absent from {manifest_name}")
        source_name = Path(csv_name).with_suffix(".parquet").name
        matches = [item for item in manifest["source_info"].get("outputs", [])
                   if Path(item.get("path", "")).name == source_name]
        if len(matches) != 1 or not matches[0].get("sha256"):
            raise ValueError(f"Feature file not published by manifest: {source_name}")
        path = mode / source_name
        if not path.is_file() or path.stat().st_size != matches[0].get("bytes"):
            raise ValueError(f"Missing/wrong-size feature table: {path}")
        records[csv_name] = {"source": str(path), "source_sha256": matches[0]["sha256"],
                             "source_bytes": matches[0]["bytes"], "manifest": manifest_name,
                             **parameters}
    if len(vocabularies) != 1:
        raise ValueError("Feature tables use different frozen context vocabularies")
    identity = {"schema_version": SNAPSHOT_VERSION, "corpus": "google", "context_mode": "local5",
                "grid": grid, "feature_definition_version": FEATURE_DEFINITION_VERSION,
                "beta_policy": BETA_POLICY, "compound_pair_policy": POLICIES,
                "rating_sha256": rating_hashes, "context_vocabulary_sha256": next(iter(vocabularies)),
                "prepare_fingerprints": prepares, "files": records,
                "source_manifests": {name: {"source": value["path"], "sha256": value["sha256"]}
                                     for name, value in manifests.items()},
                "adapter_sha256": _hash_checked(Path(__file__)),
                "csv_encoding": "UTF-8; missing=NA; float_format=%.17g; no imputation"}
    return identity, manifests


@contextlib.contextmanager
def _lock(root):
    root.mkdir(parents=True, exist_ok=True)
    with (root / ".snapshot.lock").open("a") as stream:
        fcntl.flock(stream, fcntl.LOCK_EX)
        yield


def snapshot(args):
    root = Path(args.snapshot_root).resolve()
    grid = _grid(args)
    expected_ratings, rating_hashes = _ratings()
    identity, manifests = _source_plan(Path(args.features_root), grid, rating_hashes)
    fingerprint = _digest(identity)
    if getattr(args, "dry_run", False):
        print(f"Planned {len(identity['files'])} wide CSVs; snapshot identity {fingerprint}", flush=True)
        return identity
    # Do not write inside the source tree, including its ancestors.
    source = Path(args.features_root).resolve()
    if root == source or root in source.parents or source in root.parents:
        raise ValueError("Snapshot and source features must be separate directory trees")
    with _lock(root):
        state_path = root / "snapshot.json"
        if state_path.exists():
            state = json.loads(state_path.read_text())
            if state.get("fingerprint") != fingerprint:
                raise ValueError("Snapshot identity differs; use a new --snapshot-root (existing inputs are immutable)")
            if (root / "READY").exists():
                # Re-running conversion claims these are the same sources;
                # standalone verify deliberately does not need live sources.
                for record in identity["files"].values():
                    _hash_checked(Path(record["source"]), record["source_sha256"])
                verify_snapshot(root)
                print(f"Reused verified snapshot: {root}", flush=True)
                return state
        else:
            if (root / "local5").exists() and any((root / "local5").iterdir()):
                raise ValueError("Untracked snapshot files; use a new --snapshot-root")
            state = {"fingerprint": fingerprint, "identity": identity,
                     "compound_pair_policy": POLICIES, "files": {}}
            write_json(state_path, state)
        (root / "local5").mkdir(exist_ok=True)
        (root / ".checkpoints").mkdir(exist_ok=True)
        for name, value in manifests.items():
            destination = root / "source_manifests" / name
            if destination.exists() and _hash_checked(destination) != value["sha256"]:
                raise ValueError(f"Changed frozen source manifest: {destination}")
            if not destination.exists():
                destination.parent.mkdir(parents=True, exist_ok=True)
                # Preserve the source bytes/hash, not merely equivalent JSON.
                source_path = Path(value["path"])
                data = source_path.read_bytes()
                if hashlib.sha256(data).hexdigest() != value["sha256"]:
                    raise RuntimeError(f"Source manifest changed: {source_path}")
                temporary = destination.with_suffix(".json.tmp")
                temporary.write_bytes(data)
                temporary.replace(destination)
        total = len(identity["files"])
        source_snapshots = {}
        for index, (name, record) in enumerate(identity["files"].items(), 1):
            destination = root / "local5" / name
            receipt_path = root / ".checkpoints" / (name + ".json")
            path = Path(record["source"])
            before = _stable_record(path)
            source_snapshots[name] = before
            if name not in state["files"] and receipt_path.is_file():
                receipt = json.loads(receipt_path.read_text())
                if receipt.get("fingerprint") != fingerprint:
                    raise ValueError(f"Incompatible conversion checkpoint: {receipt_path}")
                state["files"][name] = receipt["file"]
            if name in state["files"]:
                _hash_checked(destination, state["files"][name]["sha256"])
                _hash_checked(path, record["source_sha256"])
            else:
                _hash_checked(path, record["source_sha256"])
                frame = pd.read_parquet(path)
                _validate_frame(frame, expected_ratings[record["tagging"]], path)
                if before != _stable_record(path):
                    raise RuntimeError(f"Source feature changed during conversion: {path}")
                temporary = destination.with_suffix(".csv.tmp")
                frame.to_csv(temporary, index=False, na_rep="NA", float_format="%.17g", lineterminator="\n")
                converted = {"sha256": _hash_checked(temporary),
                                         "bytes": temporary.stat().st_size, "rows": len(frame),
                                         "columns": list(frame.columns)}
                # Recover even if cancellation landed between CSV publication
                # and its receipt. Only adopt byte-identical conversion output.
                if destination.exists():
                    _hash_checked(destination, converted["sha256"])
                    temporary.unlink()
                else:
                    temporary.replace(destination)
                write_json(receipt_path, {"fingerprint": fingerprint, "file": converted})
                state["files"][name] = converted
            if index == 1 or index % 100 == 0 or index == total:
                write_json(state_path, state)
                print(f"Frozen {index}/{total} wide feature tables", flush=True)
        # Catch a running feature producer rewriting manifests or tables after
        # the plan. CSV payloads were checked against the same fixed ratings.
        for name, record in identity["source_manifests"].items():
            _hash_checked(Path(record["source"]), record["sha256"])
        for name, record in identity["files"].items():
            path = Path(record["source"])
            if _stable_record(path) != source_snapshots[name]:
                raise RuntimeError(f"Source feature changed: {path}")
        write_json(root / "READY", {"fingerprint": fingerprint, "snapshot_sha256": _hash_checked(state_path)})
        try:
            verify_snapshot(root)
        except (ValueError, RuntimeError, OSError):
            # A concurrent external edit must not leave a marker claiming a
            # usable newly published snapshot.
            (root / "READY").unlink(missing_ok=True)
            raise
        print(f"Regression snapshot ready: {root}", flush=True)
        return state


def verify_snapshot(root, selected_names=None):
    root = Path(root).resolve()
    ready_path, state_path = root / "READY", root / "snapshot.json"
    if not ready_path.is_file() or not state_path.is_file():
        raise ValueError(f"Snapshot is not READY: {root}; run snapshot first")
    ready, state = json.loads(ready_path.read_text()), json.loads(state_path.read_text())
    if ready.get("snapshot_sha256") != _hash_checked(state_path):
        raise ValueError("Snapshot manifest changed after READY")
    identity = state.get("identity", {})
    if (ready.get("fingerprint") != state.get("fingerprint") or _digest(identity) != state.get("fingerprint")
            or identity.get("schema_version") != SNAPSHOT_VERSION or identity.get("corpus") != "google"
            or identity.get("feature_definition_version") != FEATURE_DEFINITION_VERSION
            or identity.get("beta_policy") != BETA_POLICY
            or identity.get("compound_pair_policy") != POLICIES
            or set(state.get("files", {})) != set(identity.get("files", {}))):
        raise ValueError("Incompatible or incomplete regression snapshot")
    names = list(state["files"] if selected_names is None else selected_names)
    if not names or any(name not in state["files"] for name in names):
        raise ValueError("Requested grid is not covered by this snapshot")
    for name in names:
        if Path(name).name != name:
            raise ValueError("Invalid snapshot filename")
        record, path = state["files"][name], root / "local5" / name
        if not path.is_file() or path.stat().st_size != record["bytes"]:
            raise ValueError(f"Missing/wrong-size frozen CSV: {path}")
        _hash_checked(path, record["sha256"])
    for name, record in identity["source_manifests"].items():
        _hash_checked(root / "source_manifests" / name, record["sha256"])
    print(f"Verified {len(names)} frozen CSVs in {root}", flush=True)
    return state


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    children = parser.add_subparsers(dest="command", required=True)
    for command in ("snapshot", "verify"):
        child = children.add_parser(command)
        child.add_argument("--snapshot-root", required=True)
        for name in DEFAULT_GRID:
            child.add_argument(f"--{name}", nargs="+")
        if command == "snapshot":
            child.add_argument("--features-root", default="runs/compositionality/features")
            child.add_argument("--dry-run", action="store_true")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    if args.command == "snapshot":
        return snapshot(args)
    supplied = any(getattr(args, name) is not None for name in DEFAULT_GRID)
    if supplied:
        state = json.loads((Path(args.snapshot_root) / "snapshot.json").read_text())
        grid = _grid(args, state["identity"]["grid"])
        return verify_snapshot(args.snapshot_root, [name for name, _ in _files(grid)])
    return verify_snapshot(args.snapshot_root)


if __name__ == "__main__":
    try:
        main()
    except (ValueError, RuntimeError, OSError) as error:
        raise SystemExit(f"Regression inputs: {error}") from error
