"""Bounded, resumable cumulative Google features for frozen-model trajectories.

Read an existing completed raw preparation, never the 262 GB source matrices.
Deduplicate each population type's FIRST decade once in its original hash
bucket; additive occurrence counts and type introductions then support every
cumulative endpoint. Add RAW decade vectors before recomputing PPMI. Published
disjoint-bin features, PPMI values and productivity counts are never averaged.
"""

import argparse
import hashlib
import json
import os
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd

import features as base
from feature_math import static_features, setting_features, weight_matrix
from src.partitioned_parquet import connection_scope
from pipeline_source_info import software_record, write_json, sha256_file

ALGORITHM_VERSION = 1
TYPE_KINDS = (103, 104, 112, 113, 114)
ADDITIVE_KINDS = (0, 1, 2, 3, 4, 5, 6, 7, 8, 9)


def _material(value):
    return json.dumps(value, sort_keys=True)


def _sha_text(value):
    return hashlib.sha256(value.encode()).hexdigest()


def _sha_file(path):
    return sha256_file(path)


def _file(path):
    return {**base._snapshot(path), "sha256": _sha_file(path)}


def _same_file(record, *, hash_file=False):
    try:
        current = base._snapshot(record["path"])
        expected = {name: record[name] for name in ("path", "bytes", "mtime_ns")}
        return current == expected and (not hash_file or _sha_file(record["path"]) == record["sha256"])
    except (OSError, KeyError):
        return False


def _sql_files(paths):
    return "[" + ",".join("'" + base._quote(path) + "'" for path in paths) + "]"


def _modern_path(args):
    tag = "UnTagged"
    return Path(args.modern_root).resolve() / "local5" / (
        f"features_{args.condition}_withSetting_{args.weight}_{tag}"
        f"_ctx{args.context_size}_10000_{args.cutoff}_na.csv")


def _routed_inventory(root):
    """Read tiny routing receipts; verify each fragment only when its bucket runs."""
    buckets, inventory = {}, []
    directory = root / "counts" / "bigram_counts"
    markers = sorted(directory.glob("batch-*/complete.json"))
    if not markers:
        raise RuntimeError("Exact type-routing cache is absent; retain counts/bigram_counts from preparation")
    for marker in markers:
        receipt = json.loads(marker.read_text())
        if not receipt.get("files"):
            # Empty input batches legitimately have no fragments.
            if receipt.get("input_rows", 0) > 0:
                raise RuntimeError(f"Invalid count-routing receipt: {marker}")
        inventory.append(_file(marker))
        for record in receipt.get("files", []):
            path = Path(record["path"])
            if not path.resolve().is_relative_to(directory.resolve()):
                raise ValueError(f"Count fragment is outside its prepared cache: {path}")
            bucket = int(path.parent.name.split("=", 1)[1])
            buckets.setdefault(bucket, []).append(record)
    return buckets, inventory


def read_specification(args):
    """Validate small identities without rescanning source corpus or rewriting it."""
    root = Path(args.prepared_root).resolve()
    identity_path, complete_path = root / "identity.json", root / "complete.json"
    identity = json.loads(identity_path.read_text())
    complete = base._read_marker(complete_path)
    if not complete or complete.get("fingerprint") != base._digest(identity):
        raise RuntimeError("Existing Google preparation is incomplete or has an incompatible identity")
    if identity.get("version") != base.VERSION or identity.get("tagged_targets") is not False:
        raise ValueError("Cumulative UnTagged features require the current all-POS raw preparation")
    if identity.get("compound_population") != "all_lexical_bigrams":
        raise ValueError("Preparation must retain all lexical bigram POS combinations")
    if not {10, 10000}.issubset(identity.get("temporal_bins", [])):
        raise ValueError("Preparation must contain decade and pooled independent frequency summaries")
    for record in identity["inputs"].values():
        if not _same_file(record):
            raise RuntimeError(f"An extracted input changed since preparation: {record['path']}")
    for record in [identity["context_vocabulary"], *identity["ratings"]]:
        if _sha_file(record["path"]) != record["sha256"]:
            raise RuntimeError(f"A vocabulary or rating file changed since preparation: {record['path']}")
    vocabulary = pd.read_parquet(root / "contexts.parquet")
    if args.context_size > len(vocabulary) or vocabulary["rank"].tolist() != list(range(1, len(vocabulary) + 1)):
        raise ValueError("Requested context size must use a valid ranked prefix of the frozen vocabulary")
    targets = json.loads((root / "targets.json").read_text())
    pairs, words = pd.DataFrame(targets["pairs"]), pd.DataFrame(targets["words"])
    if pairs.target.tolist() != list(range(len(pairs))) or words.target.tolist() != list(range(len(words))):
        raise ValueError("Prepared target IDs must be contiguous")
    ratings_by_name = {record["name"]: record["path"] for record in identity["ratings"]}
    ratings = base.read_ratings(SimpleNamespace(tag=False, **ratings_by_name))
    expected_pairs = ratings[["modifier", "head"]].drop_duplicates().sort_values(
        ["modifier", "head"]).reset_index(drop=True)
    if not expected_pairs.equals(pairs[["modifier", "head"]]):
        raise ValueError("Cached targets differ from the rating roster")
    options = dict(complete["execution"])
    options.update(threads=args.threads, aggregation_workers=1,
                   memory_limit=args.memory_limit, max_temp_directory_size=args.max_temp_directory_size)
    specification = {"root": root, "pairs": pairs, "words": words, "vocabulary": vocabulary,
                "ratings": ratings, "identity": identity, "options": options}
    frequency_path = root / "frequencies/width-10/frequencies.parquet"
    # Read only twenty population-summary rows; plan must not create spill dirs.
    decade_rows = pd.read_parquet(frequency_path, columns=["time"],
                                 filters=[("kind", "==", 1), ("count", ">", 0)])
    available = sorted(int(year) for year in decade_rows.time.unique())
    if not available or any(year < 0 or year % 10 for year in available):
        raise ValueError("Preparation must contain nonnegative calendar-decade counts")
    endpoints = sorted(set(args.end_decades or available))
    if not set(endpoints).issubset(available) or available[-1] not in endpoints:
        raise ValueError(f"Endpoints must be prepared decades and include the final {available[-1]} endpoint")
    modern_path = _modern_path(args)
    modern = pd.read_csv(modern_path, keep_default_na=True)
    for name in ("source", "modifier", "head", "is_original", "is_adj"):
        if name not in modern:
            raise ValueError(f"Modern CSV is missing rating metadata: {name}")
    beta = modern.filter(regex=r"^beta:").to_numpy(dtype=float)
    if np.any(np.isfinite(beta) & ((beta < 0) | (beta > 1))):
        raise ValueError("Modern features have unbounded beta; use the corrected version-3 snapshot")
    snapshot_path = Path(args.modern_root) / "snapshot.json"
    if snapshot_path.exists():
        snapshot = json.loads(snapshot_path.read_text())
        snapshot_identity = snapshot.get("identity", {})
        if (snapshot_identity.get("beta_policy") != base.BETA_POLICY
                or snapshot_identity.get("feature_definition_version") != base.VERSION):
            raise ValueError("Frozen modern snapshot must identify clipped_unit_interval beta")
        if (snapshot_identity.get("context_vocabulary_sha256") != identity["context_vocabulary"]["sha256"]
                or snapshot_identity.get("compound_pair_policy", {}).get("UnTagged") != "all_lexical_bigrams"):
            raise ValueError("Modern snapshot and preparation use different vocabularies or compound populations")
        modern_record = snapshot.get("files", {}).get(modern_path.name, {})
        if modern_record.get("sha256") != _sha_file(modern_path):
            raise ValueError("Modern feature CSV differs from the verified frozen snapshot")
    buckets, inventory = _routed_inventory(root)
    modern_source = _file(modern_path)
    # Type routing/material is raw, so version-2 preparation is compatible with
    # version-3 bounded feature math. Do not pretend old software generated v3.
    material = {"algorithm_version": ALGORITHM_VERSION,
        "prepared_identity": _file(identity_path), "prepared_completion": _file(complete_path),
        "routing_inventory": inventory, "source_inputs": identity["inputs"],
        "context_vocabulary": identity["context_vocabulary"], "ratings": identity["ratings"],
        "source_software": identity["software"],
        "software": software_record([Path(__file__), base.DIRECTORY / "feature_math.py",
            Path(base.__file__), base.SOURCE_DIRECTORY / "coha_feature_pipeline.py",
            base.SOURCE_DIRECTORY / "coha_features.py", 
            base.DIRECTORY / "pipeline_source_info.py"], ["duckdb", "pyarrow", "pandas", "numpy", "scipy"]),
        "modern_source": modern_source, "context_mode": "local5", "context_size": args.context_size,
        "setting": args.condition.removeprefix("Compound"), "condition": args.condition,
        "ppmi": args.weight == "PPMI", "tagged": False, "cutoff": args.cutoff,
        "time_step": args.time_step, "feature_definition_version": base.VERSION,
        "beta_policy": base.BETA_POLICY}
    material_text = _material(material)
    specification.update(input_identity=_sha_text(material_text), input_identity_material=material_text,
        modern=modern, modern_source=modern_source, buckets=buckets, routing_inventory=inventory,
        frequency_path=frequency_path, available=available, endpoints=endpoints)
    specification["work"] = Path(args.work_root).resolve() / ("cumulative-" + specification["input_identity"][:24])
    output = Path(args.output_root).resolve()
    work = Path(args.work_root).resolve()
    for candidate in (output, work, Path(args.temp_directory).resolve()):
        for protected in (root, Path(args.modern_root).resolve(), *[Path(r["path"]).resolve().parent for r in identity["inputs"].values()]):
            if candidate == protected or candidate.is_relative_to(protected) or protected.is_relative_to(candidate):
                raise ValueError(f"New output/work/spill roots must not overlap prepared, modern or extracted inputs: {candidate}")
    if output == work or output.is_relative_to(work) or work.is_relative_to(output):
        raise ValueError("Cumulative snapshots and work cache must be separate directory trees")
    return specification


def _first_seen_query(paths):
    """Exact union types: first appearance, NOT the sum of decade type counts."""
    return f"""WITH first_seen AS MATERIALIZED (
        SELECT kind,target,partner,MIN(time)::BIGINT AS time
        FROM read_parquet({_sql_files(paths)}) WHERE kind IN (3,4,12) AND count>0
        GROUP BY kind,target,partner
    )
    SELECT CASE kind WHEN 3 THEN 103 WHEN 4 THEN 104 ELSE 112 END::INTEGER AS kind,
        ''::VARCHAR AS target,''::VARCHAR AS partner,time,COUNT(*)::BIGINT AS count
        FROM first_seen GROUP BY kind,time
    UNION ALL
    SELECT 113,target,'',time,COUNT(*)::BIGINT FROM first_seen
        WHERE kind=12 AND target IN (SELECT word FROM evaluation_words) GROUP BY target,time
    UNION ALL
    SELECT 114,partner,'',time,COUNT(*)::BIGINT FROM first_seen
        WHERE kind=12 AND partner IN (SELECT word FROM evaluation_words) GROUP BY partner,time"""


def _assert_sources(specification):
    for record in [specification["modern_source"], *specification["routing_inventory"]]:
        if not _same_file(record):
            raise RuntimeError(f"An input changed during the cumulative calculation: {record['path']}")
    for record in specification["identity"]["inputs"].values():
        if not _same_file(record):
            raise RuntimeError(f"An extracted input changed: {record['path']}")


def prepare(args):
    specification = read_specification(args)
    work = specification["work"]
    work.mkdir(parents=True, exist_ok=True)
    with base._lock(work / "prepare.lock"):
        marker = base._read_marker(work / "complete.json")
        if marker and marker.get("input_identity") == specification["input_identity"] and not args.force:
            print(f"Reused cumulative first-appearance cache: {work}", flush=True)
            return work
        parts = []
        for index, (bucket, records) in enumerate(sorted(specification["buckets"].items())):
            part = work / f"first-seen-{bucket:05d}.parquet"
            receipt = base._read_marker(part.with_suffix(".json"))
            if args.force or not receipt or receipt.get("input_identity") != specification["input_identity"]:
                for record in records:
                    if not _same_file(record):
                        raise RuntimeError(f"Prepared exact-type fragment changed: {record['path']}")
                with connection_scope(specification["options"], args.temp_directory) as connection:
                    base._setup(specification)(connection)
                    temporary = part.with_name(part.name + ".tmp")
                    query = _first_seen_query([record["path"] for record in records])
                    connection.execute(f"COPY ({query}) TO '{base._quote(temporary)}' "
                                       "(FORMAT PARQUET, COMPRESSION ZSTD)")
                    os.replace(temporary, part)
                write_json(part.with_suffix(".json"), {"input_identity": specification["input_identity"],
                           "files": [base._snapshot(part)]})
            parts.append(part)
            print(f"Type first appearances: bucket {index+1}/{len(specification['buckets'])}", flush=True)
        destination = work / "type_introductions.parquet"
        with connection_scope(specification["options"], args.temp_directory) as connection:
            query = f"""SELECT kind,target,partner,time,SUM(count)::BIGINT AS count
                FROM read_parquet({_sql_files(parts)}) GROUP BY kind,target,partner,time"""
            temporary = destination.with_name(destination.name + ".tmp")
            connection.execute(f"COPY ({query}) TO '{base._quote(temporary)}' (FORMAT PARQUET, COMPRESSION ZSTD)")
            os.replace(temporary, destination)
        _assert_sources(specification)
        write_json(work / "complete.json", {"input_identity": specification["input_identity"],
            "input_identity_material": specification["input_identity_material"],
            "meaning": "Distinct type introductions by first observed decade, after target POS pooling",
            "files": [base._snapshot(destination)]})
        return work


def cumulative_frequencies(specification, endpoint, args):
    with connection_scope(specification["options"], args.temp_directory) as connection:
        query = """SELECT kind,target,partner,0::BIGINT AS time,SUM(count)::BIGINT AS count
            FROM (SELECT kind,target,partner,time,count FROM read_parquet(?)
                  WHERE kind IN (0,1,2,3,4,5,6,7,8,9)
                  UNION ALL SELECT kind,target,partner,time,count FROM read_parquet(?))
            WHERE time <= ? GROUP BY kind,target,partner"""
        return connection.execute(query, [str(specification["frequency_path"]),
            str(specification["work"] / "type_introductions.parquet"), endpoint]).fetchdf()


def cumulative_static(specification, raw, background, frequencies, args):
    controls, occurrences, minimum, mod_ids, head_ids = base._controls(specification, frequencies, 0)
    compound = weight_matrix(raw[2], background[0], args.weight == "PPMI")
    general = weight_matrix(raw[5], background[1], args.weight == "PPMI")
    pooled_raw = raw[3] + raw[4]
    pooled = weight_matrix(pooled_raw, 2 * background[0], args.weight == "PPMI")
    if args.condition == "CompoundPooled":
        modifiers, heads = pooled, pooled
    elif args.condition == "CompoundAgnostic":
        modifiers, heads = general, general
    else:
        modifiers = weight_matrix(raw[3], background[0], args.weight == "PPMI")
        heads = weight_matrix(raw[4], background[0], args.weight == "PPMI")
    frame = specification["pairs"][["modifier", "head"]].copy()
    frame["time"] = 0
    frame = pd.concat([frame, static_features(compound, modifiers[mod_ids], heads[head_ids])], axis=1)
    frame = frame.merge(controls[args.condition], on=base.PAIR, how="left", validate="one_to_one")
    if args.condition != "CompoundAgnostic":
        roles = (pooled_raw, pooled_raw) if args.condition == "CompoundPooled" else (raw[3], raw[4])
        for index, ids in enumerate((mod_ids, head_ids)):
            label = "constituent" if args.condition == "CompoundPooled" else ("modifier", "head")[index]
            matrix = (modifiers, heads)[index]
            comparison = setting_features(matrix, general, roles[index], raw[5],
                occurrences[args.condition][index], occurrences["CompoundAgnostic"][index], label)
            values = comparison.iloc[ids].reset_index(drop=True)
            if args.condition == "CompoundPooled":
                values = values.add_suffix("_mod" if index == 0 else "_head")
            frame = pd.concat([frame, values], axis=1)
    return frame.loc[minimum > args.cutoff].reset_index(drop=True)


def _parity(wide, modern):
    keys = ["source", "modifier", "head", "is_original", "is_adj"]
    if wide.duplicated(keys).any() or modern.duplicated(keys).any() or len(wide) != len(modern):
        raise ValueError("Final cumulative and modern rating rosters differ")
    modern = modern.set_index(keys).sort_index()
    wide = wide.set_index(keys).sort_index()
    if not wide.index.equals(modern.index):
        raise ValueError("Final cumulative and modern rating keys differ")
    # The old writer omits all feature columns if no pair meets a high cutoff.
    # Explicit all-missing columns carry exactly the same scientific information.
    for column in set(wide.columns) ^ set(modern.columns):
        present = wide[column] if column in wide else modern[column]
        if ":0" not in column or not present.isna().all():
            raise ValueError(f"Final cumulative and modern columns differ: {column}")
        if column not in modern:
            modern[column] = np.nan
        if column not in wide:
            wide[column] = np.nan
    for column in modern:
        left, right = wide[column], modern[column]
        if pd.api.types.is_numeric_dtype(right):
            if not np.allclose(left.to_numpy(dtype=float), right.to_numpy(dtype=float),
                               equal_nan=True, atol=1e-6, rtol=0):
                raise ValueError(f"Final cumulative feature differs from pooled modern feature: {column}")
        elif not left.fillna("").equals(right.fillna("")):
            raise ValueError(f"Final cumulative rating metadata differs: {column}")


def _valid_manifest(path, specification, endpoint):
    try:
        value = json.loads(path.read_text())
        if value["input_identity"] != specification["input_identity"] or value["endpoint_decade"] != endpoint:
            return None
        if _sha_text(value["snapshot_identity_material"]) != value["snapshot_identity"]:
            return None
        if not all(_same_file(record, hash_file=True) for record in value["files"]):
            return None
        return value
    except (OSError, ValueError, KeyError, TypeError):
        return None


def _write_endpoint(specification, endpoint, frame, args):
    directory = Path(args.output_root).resolve() / "local5" / f"through_{endpoint}"
    directory.mkdir(parents=True, exist_ok=True)
    stem = f"features_{args.condition}_withSetting_{args.weight}_UnTagged_ctx{args.context_size}_10000_{args.cutoff}_na"
    parquet_path, csv_path = directory / (stem + ".parquet"), directory / (stem + ".csv")
    # Write both shared feature layouts, but keep publication atomic.
    temporary = directory / (stem + ".staging.parquet")
    base.write_features(frame, specification["ratings"], frame[base.PAIR], temporary)
    wide = pd.read_parquet(temporary)
    # Explicit eligibility/columns even when no rated targets have appeared yet.
    if "eligible:0" not in wide:
        for column in base._feature_columns(args.condition):
            if column not in base.PAIR:
                wide[column + ":0"] = np.nan
        wide["eligible:0"] = np.nan
        base._write_parquet(wide, temporary)
    if endpoint == specification["available"][-1]:
        _parity(wide, specification["modern"])
    csv_tmp = csv_path.with_name(csv_path.name + ".tmp")
    wide.to_csv(csv_tmp, index=False, float_format="%.17g", na_rep="NA")
    long_temporary = temporary.with_name(temporary.stem + "_long.parquet")
    long_path = parquet_path.with_name(parquet_path.stem + "_long.parquet")
    os.replace(temporary, parquet_path)
    os.replace(long_temporary, long_path)
    os.replace(csv_tmp, csv_path)
    files = [_file(path) for path in (csv_path, parquet_path, long_path)]
    snapshot_material = _material({"input_identity": specification["input_identity"],
        "endpoint_decade": endpoint, "end_year": endpoint + 9, "files": files})
    manifest = {"schema_version": 1, "stage": "google_cumulative_features",
        "input_identity": specification["input_identity"], "input_identity_material": specification["input_identity_material"],
        "snapshot_identity": _sha_text(snapshot_material), "snapshot_identity_material": snapshot_material,
        "endpoint_decade": endpoint, "end_year": endpoint + 9,
        "full_input_last_decade": specification["available"][-1],
        "meaning": "All extracted raw observations through the complete endpoint decade; not a disjoint time bin",
        "time_step": args.time_step, "context_mode": "local5", "context_size": args.context_size,
        "setting": args.condition.removeprefix("Compound"), "condition": args.condition,
        "cutoff": args.cutoff, "ppmi": args.weight == "PPMI", "tagged": False,
        "feature_definition_version": base.VERSION, "beta_policy": base.BETA_POLICY,
        "modern_source": specification["modern_source"], "files": files,
        "final_modern_parity_verified": endpoint == specification["available"][-1],
        "eligible_pairs": len(frame), "total_rating_rows": len(wide)}
    path = directory / "cumulative_manifest.json"
    write_json(path, manifest)
    return _file(path)


def run_features(args):
    specification = read_specification(args)
    work = specification["work"]
    prepared = base._read_marker(work / "complete.json")
    if not prepared or prepared.get("input_identity") != specification["input_identity"]:
        raise RuntimeError("Cumulative type preparation is incomplete; run prepare first")
    output = Path(args.output_root).resolve()
    output.mkdir(parents=True, exist_ok=True)
    with base._lock(output / "features.lock"):
        if not args.force and (output / "features_complete.json").exists():
            try:
                finished = check(args)
            except (RuntimeError, ValueError, OSError, KeyError):
                pass  # Repair only compatible missing/damaged epoch products below.
            else:
                print("Reused complete cumulative feature study", flush=True)
                return finished
        old_manifests = list((output / "local5").glob("through_*/cumulative_manifest.json"))
        for path in old_manifests:
            old = json.loads(path.read_text())
            if old.get("input_identity") != specification["input_identity"]:
                raise RuntimeError("Existing cumulative snapshots use incompatible settings; choose a new output root")
            if old.get("endpoint_decade") not in specification["endpoints"]:
                raise RuntimeError("Output contains endpoints outside the requested study; choose a new output root")
        write_json(output / "cumulative_plan.json", {"input_identity": specification["input_identity"],
            "endpoints": specification["endpoints"], "status": "planned"})
        manifests, raw, background, cursor = [], None, None, 0
        for endpoint in specification["endpoints"]:
            path = output / "local5" / f"through_{endpoint}" / "cumulative_manifest.json"
            saved = _valid_manifest(path, specification, endpoint)
            if saved and not args.force:
                manifests.append(_file(path))
                print(f"Reused cumulative endpoint through {endpoint+9}", flush=True)
                continue
            increment, bg_increment = base._load_vectors(specification, cursor, endpoint + 10 - cursor,
                                                          args.context_size, args)
            if raw is None:
                raw, background = increment, bg_increment
            else:
                raw = {kind: raw[kind] + increment[kind] for kind in raw}
                background = {kind: background[kind] + bg_increment[kind] for kind in background}
            cursor = endpoint + 10
            frequencies = cumulative_frequencies(specification, endpoint, args)
            frame = cumulative_static(specification, raw, background, frequencies, args)
            _assert_sources(specification)
            manifests.append(_write_endpoint(specification, endpoint, frame, args))
            print(f"Cumulative endpoint through {endpoint+9}: {len(frame)} eligible pairs", flush=True)
        # Verify last snapshot even on a completely reused run.
        last_csv = _modern_path(args).name
        _parity(pd.read_csv(output / "local5" / f"through_{specification['available'][-1]}" / last_csv),
                specification["modern"])
        _assert_sources(specification)
        complete = {"status": "complete", "input_identity": specification["input_identity"],
            "endpoints": specification["endpoints"], "manifests": manifests,
            "final_modern_parity_verified": True, "feature_definition_version": base.VERSION,
            "beta_policy": base.BETA_POLICY}
        write_json(output / "features_complete.json", complete)
        return complete


def check(args):
    specification = read_specification(args)
    output = Path(args.output_root).resolve()
    complete = json.loads((output / "features_complete.json").read_text())
    if complete.get("status") != "complete" or complete.get("input_identity") != specification["input_identity"]:
        raise RuntimeError("Cumulative study is incomplete or incompatible")
    if complete.get("endpoints") != specification["endpoints"] or len(complete.get("manifests", [])) != len(specification["endpoints"]):
        raise RuntimeError("Completed endpoint inventory differs from the requested study")
    for endpoint, record in zip(specification["endpoints"], complete["manifests"]):
        if not _same_file(record, hash_file=True) or not _valid_manifest(Path(record["path"]), specification, endpoint):
            raise RuntimeError(f"Invalid completed cumulative snapshot at endpoint {endpoint}")
    _parity(pd.read_csv(output / "local5" / f"through_{specification['available'][-1]}" / _modern_path(args).name),
            specification["modern"])
    print(f"Verified {len(specification['endpoints'])} cumulative endpoints and final modern-feature parity", flush=True)
    return complete


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("plan", "prepare", "features", "check"))
    for name in ("prepared-root", "modern-root", "work-root", "output-root", "temp-directory"):
        parser.add_argument("--" + name, required=True)
    parser.add_argument("--context-size", type=int, default=5000)
    parser.add_argument("--cutoff", type=int, default=0)
    parser.add_argument("--time-step", type=int, default=10)
    parser.add_argument("--end-decades", nargs="+", type=int)
    parser.add_argument("--condition", choices=base.CONDITIONS, default="CompoundPooled")
    parser.add_argument("--weight", choices=("RAW", "PPMI"), default="PPMI")
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--memory-limit", default="48GB")
    parser.add_argument("--max-temp-directory-size", default="64GB")
    parser.add_argument("--force", action="store_true", help="Recompute only this compatible study's own snapshots")
    args = parser.parse_args(argv)
    if args.context_size < 1 or args.cutoff < 0 or args.threads < 1 or args.time_step != 10:
        parser.error("Use positive context size/threads, nonnegative cutoff and decade endpoint step 10")
    return args


def main(argv=None):
    args = parse_args(argv)
    if args.command == "plan":
        specification = read_specification(args)
        plan = {"prepared_root": str(specification["root"]), "input_identity": specification["input_identity"],
            "endpoints": specification["endpoints"], "end_years": [year+9 for year in specification["endpoints"]],
            "type_buckets": len(specification["buckets"]), "work_directory": str(specification["work"]),
            "condition": args.condition, "weight": args.weight, "context_size": args.context_size,
            "no_source_matrix_rescan": True, "mode": "cumulative raw history; frozen ranked vocabulary"}
        print(json.dumps(plan, indent=2), flush=True)
        return plan
    return {"prepare": prepare, "features": run_features, "check": check}[args.command](args)


if __name__ == "__main__":
    main()
