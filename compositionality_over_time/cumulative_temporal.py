"""Scoped cumulative-prefix temporal predictors for the Google longitudinal study.

This implementation shares the static cumulative count/type cache and uses a
separate identity for the augmented features. It supports the configured
30000 / UnTagged / PPMI / Aware / 10 / cutoff-500 study. Temporal predictors are
cosines between independently PPMI-weighted prefixes through t and t-10, not
between disjoint decades and not between adjacent *requested* endpoints.
"""

import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

import cumulative_features as cumulative
from feature_math import cosine_rows, weight_matrix
from pipeline_source_info import software_record, write_json

base = cumulative.base
TEMPORAL_POLICY = "cumulative_prefix_cosine_lag10_v1"
TEMPORAL_LAG = 10
TEMPORAL_COLUMNS = ("change_comp", "change_mod", "change_head")


def _require_configuration(args):
    expected = {"context_size": 30000, "condition": "CompoundAware",
                "weight": "PPMI", "cutoff": 500, "time_step": 10}
    if any(getattr(args, name, None) != value for name, value in expected.items()):
        raise ValueError("Cumulative temporal features are scoped to context-size=30000, "
                         "UnTagged, condition=CompoundAware, weight=PPMI, cutoff=500, time-step=10")


def _policy():
    return {"temporal_policy": TEMPORAL_POLICY, "temporal_lag": TEMPORAL_LAG,
            "temporal_columns": list(TEMPORAL_COLUMNS)}


def read_specification(args):
    _require_configuration(args)
    specification = cumulative.read_specification(args)
    # Counts share the static cache identity; augmented outputs additionally
    # identify the temporal implementation and its parameters.
    specification["count_input_identity"] = specification["input_identity"]
    specification["count_input_identity_material"] = specification["input_identity_material"]
    material = {**json.loads(specification["input_identity_material"]), **_policy(),
                "count_input_identity": specification["count_input_identity"],
                "full_input_first_decade": specification["available"][0],
                "temporal_software": software_record([Path(__file__)],
                    ["duckdb", "pyarrow", "pandas", "numpy", "scipy"])}
    specification["input_identity_material"] = cumulative._material(material)
    specification["input_identity"] = cumulative._sha_text(specification["input_identity_material"])
    return specification


def prepare(args):
    _require_configuration(args)
    # Count preparation uses the static identity, not the augmented one.
    return cumulative.prepare(args)


def _lag_endpoint(specification, endpoint):
    lag = endpoint - TEMPORAL_LAG
    return lag if lag >= specification["available"][0] else None


def _weighted_prefix(specification, raw, background, frequencies, args):
    _, _, minimum, mod_ids, head_ids = base._controls(specification, frequencies, 0)
    # Aware roles use separate modifier/head matrices. All three populations
    # retain the COMPLETE bigram-context marginal before target/cutoff filtering.
    return {"minimum": minimum,
            "compound": weight_matrix(raw[2], background[0], ppmi=True),
            "modifier": weight_matrix(raw[3], background[0], ppmi=True)[mod_ids],
            "head": weight_matrix(raw[4], background[0], ppmi=True)[head_ids]}


def _temporal_values(current, previous, cutoff):
    size = len(current["minimum"])
    values = {column: np.full(size, np.nan) for column in TEMPORAL_COLUMNS}
    if previous is None:
        return pd.DataFrame(values)
    eligible = ((current["minimum"] > cutoff) & (previous["minimum"] > cutoff))
    for column, role in zip(TEMPORAL_COLUMNS, ("compound", "modifier", "head")):
        cosine = cosine_rows(current[role], previous[role])
        cosine[~eligible] = np.nan
        values[column] = cosine
    return pd.DataFrame(values)


def _paths(args, endpoint):
    directory = Path(args.output_root).resolve() / "local5" / f"through_{endpoint}"
    stem = cumulative._modern_path(args).stem
    return directory, [directory / (stem + suffix) for suffix in
                       (".csv", ".parquet", "_long.parquet")]


def _static_parity(wide, modern):
    temporal = [column + ":0" for column in TEMPORAL_COLUMNS]
    if not set(temporal).issubset(wide.columns):
        raise ValueError("Cumulative temporal output is missing required temporal columns")
    # No static or setting column is excluded from the numerical parity check.
    cumulative._parity(wide.drop(columns=temporal), modern)


def _snapshot_material(specification, endpoint, files):
    return cumulative._material({"input_identity": specification["input_identity"],
        "count_input_identity": specification["count_input_identity"],
        "endpoint_decade": endpoint, "end_year": endpoint + 9,
        "lag_endpoint_decade": _lag_endpoint(specification, endpoint), **_policy(), "files": files})


def _valid_manifest(path, specification, endpoint, args):
    try:
        value = json.loads(path.read_text())
        expected = {"schema_version": 1, "stage": "google_cumulative_temporal_features",
            "input_identity": specification["input_identity"],
            "input_identity_material": specification["input_identity_material"],
            "count_input_identity": specification["count_input_identity"],
            "endpoint_decade": endpoint, "end_year": endpoint + 9,
            "lag_endpoint_decade": _lag_endpoint(specification, endpoint), **_policy(),
            "full_input_first_decade": specification["available"][0],
            "full_input_last_decade": specification["available"][-1],
            "time_step": args.time_step, "context_mode": "local5", "context_size": args.context_size,
            "setting": args.condition.removeprefix("Compound"), "condition": args.condition,
            "cutoff": args.cutoff, "ppmi": True, "tagged": False,
            "feature_definition_version": base.VERSION, "beta_policy": base.BETA_POLICY,
            "modern_source": specification["modern_source"],
            "total_rating_rows": len(specification["ratings"]),
            "final_modern_parity_verified": endpoint == specification["available"][-1],
            "static_parity_excluded_columns": [column + ":0" for column in TEMPORAL_COLUMNS]}
        if any(value.get(name) != data for name, data in expected.items()):
            return None
        _, paths = _paths(args, endpoint)
        if [record["path"] for record in value["files"]] != [str(p) for p in paths]:
            return None
        material = _snapshot_material(specification, endpoint, value["files"])
        if (value["snapshot_identity_material"] != material
                or cumulative._sha_text(material) != value["snapshot_identity"]):
            return None
        if not all(cumulative._same_file(record, hash_file=True) for record in value["files"]):
            return None
        return value
    except (OSError, ValueError, KeyError, TypeError):
        return None


def _write_endpoint(specification, endpoint, frame, args):
    directory, paths = _paths(args, endpoint)
    csv_path, parquet_path, long_path = paths
    directory.mkdir(parents=True, exist_ok=True)
    temporary = parquet_path.with_name(parquet_path.stem + ".staging.parquet")
    base.write_features(frame, specification["ratings"], frame[base.PAIR], temporary)
    wide = pd.read_parquet(temporary)
    # The shared writer omits feature columns when no pair passes the cutoff.
    # Keep the entire rating roster and explicit missing temporal values then.
    if "eligible:0" not in wide:
        for column in [*base._feature_columns(args.condition), *TEMPORAL_COLUMNS]:
            if column not in base.PAIR:
                wide[column + ":0"] = np.nan
        wide["eligible:0"] = np.nan
        base._write_parquet(wide, temporary)
    if endpoint == specification["available"][-1]:
        _static_parity(wide, specification["modern"])
    csv_temporary = csv_path.with_name(csv_path.name + ".tmp")
    wide.to_csv(csv_temporary, index=False, float_format="%.17g", na_rep="NA")
    long_temporary = temporary.with_name(temporary.stem + "_long.parquet")
    os.replace(temporary, parquet_path)
    os.replace(long_temporary, long_path)
    os.replace(csv_temporary, csv_path)
    files = [cumulative._file(path) for path in paths]
    material = _snapshot_material(specification, endpoint, files)
    manifest = {"schema_version": 1, "stage": "google_cumulative_temporal_features",
        "input_identity": specification["input_identity"],
        "input_identity_material": specification["input_identity_material"],
        "count_input_identity": specification["count_input_identity"],
        "snapshot_identity": cumulative._sha_text(material), "snapshot_identity_material": material,
        "endpoint_decade": endpoint, "end_year": endpoint + 9,
        "lag_endpoint_decade": _lag_endpoint(specification, endpoint), **_policy(),
        "full_input_last_decade": specification["available"][-1],
        "full_input_first_decade": specification["available"][0],
        "meaning": "Cumulative history and its exact ten-year-lag cumulative prefix; not disjoint decades",
        "time_step": args.time_step, "context_mode": "local5", "context_size": args.context_size,
        "setting": args.condition.removeprefix("Compound"), "condition": args.condition,
        "cutoff": args.cutoff, "ppmi": True, "tagged": False,
        "feature_definition_version": base.VERSION, "beta_policy": base.BETA_POLICY,
        "modern_source": specification["modern_source"], "files": files,
        "final_modern_parity_verified": endpoint == specification["available"][-1],
        "static_parity_excluded_columns": [column + ":0" for column in TEMPORAL_COLUMNS],
        "eligible_pairs": len(frame), "total_rating_rows": len(wide)}
    path = directory / "cumulative_manifest.json"
    write_json(path, manifest)
    return cumulative._file(path)


def _compatible_output(output, specification):
    # Never let --force overwrite an old static study or another temporal one.
    paths = [output / "cumulative_plan.json", output / "features_complete.json"]
    paths.extend((output / "local5").glob("through_*/cumulative_manifest.json"))
    for path in paths:
        if not path.exists():
            continue
        value = json.loads(path.read_text())
        if value.get("input_identity") != specification["input_identity"]:
            raise RuntimeError("Existing cumulative output has an incompatible identity; choose a new output root")
        if "endpoints" in value and value["endpoints"] != specification["endpoints"]:
            raise RuntimeError("Existing cumulative output uses different endpoints; choose a new output root")
        if "endpoint_decade" in value and value["endpoint_decade"] not in specification["endpoints"]:
            raise RuntimeError("Existing cumulative output contains endpoints outside the requested study")


def run_features(args):
    specification = read_specification(args)
    prepared = base._read_marker(specification["work"] / "complete.json")
    if not prepared or prepared.get("input_identity") != specification["count_input_identity"]:
        raise RuntimeError("Cumulative type preparation is incomplete; run prepare first")
    output = Path(args.output_root).resolve()
    _compatible_output(output, specification)
    output.mkdir(parents=True, exist_ok=True)
    with base._lock(output / "features.lock"):
        _compatible_output(output, specification)
        complete_path = output / "features_complete.json"
        if not args.force and complete_path.exists():
            try:
                finished = check(args)
            except (RuntimeError, ValueError, OSError, KeyError, TypeError):
                pass
            else:
                print("Reused complete cumulative temporal study", flush=True)
                return finished
        # A failed repair must never leave a stale successful completion receipt.
        if complete_path.exists():
            complete_path.unlink()
        write_json(output / "cumulative_plan.json", {"input_identity": specification["input_identity"],
            "count_input_identity": specification["count_input_identity"], **_policy(),
            "endpoints": specification["endpoints"], "status": "planned"})
        manifests, missing = {}, []
        for endpoint in specification["endpoints"]:
            directory, _ = _paths(args, endpoint)
            path = directory / "cumulative_manifest.json"
            saved = _valid_manifest(path, specification, endpoint, args)
            if saved and not args.force:
                manifests[endpoint] = cumulative._file(path)
                print(f"Reused temporal endpoint through {endpoint + 9}", flush=True)
            else:
                missing.append(endpoint)
        needed = set(missing)
        needed.update(lag for endpoint in missing
                      if (lag := _lag_endpoint(specification, endpoint)) is not None)
        lag_needed = {lag for endpoint in missing
                      if (lag := _lag_endpoint(specification, endpoint)) is not None}
        raw, background, cursor, prefixes = None, None, 0, {}
        # Each raw interval is loaded at most once per invocation, including
        # lag prefixes skipped by a thinned requested endpoint list. Retain only
        # weighted target matrices needed by the exact following-lag endpoint.
        for endpoint in sorted(needed):
            increment, bg_increment = base._load_vectors(
                specification, cursor, endpoint + 10 - cursor, args.context_size, args)
            if raw is None:
                raw, background = increment, bg_increment
            else:
                raw = {kind: raw[kind] + increment[kind] for kind in raw}
                background = {kind: background[kind] + bg_increment[kind] for kind in background}
            cursor = endpoint + 10
            frequencies = cumulative.cumulative_frequencies(specification, endpoint, args)
            current = _weighted_prefix(specification, raw, background, frequencies, args)
            if endpoint in missing:
                frame = cumulative.cumulative_static(specification, raw, background, frequencies, args)
                lag = _lag_endpoint(specification, endpoint)
                values = _temporal_values(current, prefixes.get(lag), args.cutoff)
                values = pd.concat([specification["pairs"][["modifier", "head"]], values], axis=1)
                frame = frame.merge(values, on=["modifier", "head"], how="left", validate="one_to_one")
                cumulative._assert_sources(specification)
                manifests[endpoint] = _write_endpoint(specification, endpoint, frame, args)
                print(f"Temporal cumulative endpoint through {endpoint + 9}: {len(frame)} eligible pairs", flush=True)
                if lag is not None:
                    del prefixes[lag]
            if endpoint in lag_needed:
                prefixes[endpoint] = current
        _verify_final_parity(specification, args)
        cumulative._assert_sources(specification)
        complete = {"status": "complete", "input_identity": specification["input_identity"],
            "count_input_identity": specification["count_input_identity"], **_policy(),
            "endpoints": specification["endpoints"],
            "manifests": [manifests[endpoint] for endpoint in specification["endpoints"]],
            "final_modern_parity_verified": True, "feature_definition_version": base.VERSION,
            "beta_policy": base.BETA_POLICY}
        write_json(complete_path, complete)
        return complete


def _verify_final_parity(specification, args):
    _, paths = _paths(args, specification["available"][-1])
    _static_parity(pd.read_csv(paths[0]), specification["modern"])


def check(args):
    specification = read_specification(args)
    output = Path(args.output_root).resolve()
    complete = json.loads((output / "features_complete.json").read_text())
    expected = {"status": "complete", "input_identity": specification["input_identity"],
        "count_input_identity": specification["count_input_identity"], **_policy(),
        "endpoints": specification["endpoints"], "final_modern_parity_verified": True,
        "feature_definition_version": base.VERSION, "beta_policy": base.BETA_POLICY}
    if any(complete.get(name) != data for name, data in expected.items()):
        raise RuntimeError("Cumulative temporal study is incomplete or incompatible")
    if len(complete.get("manifests", [])) != len(specification["endpoints"]):
        raise RuntimeError("Completed temporal endpoint inventory differs from the requested study")
    for endpoint, record in zip(specification["endpoints"], complete["manifests"]):
        directory, _ = _paths(args, endpoint)
        expected_path = directory / "cumulative_manifest.json"
        if (record.get("path") != str(expected_path)
                or not cumulative._same_file(record, hash_file=True)
                or not _valid_manifest(expected_path, specification, endpoint, args)):
            raise RuntimeError(f"Invalid completed cumulative temporal snapshot at endpoint {endpoint}")
    _verify_final_parity(specification, args)
    cumulative._assert_sources(specification)
    print(f"Verified {len(specification['endpoints'])} temporal cumulative endpoints and full static-feature parity", flush=True)
    return complete


def parse_args(argv=None):
    values = list(sys.argv[1:] if argv is None else argv)
    if values and values[0] in ("plan", "prepare", "features", "check"):
        values[1:1] = ["--context-size", "30000", "--condition", "CompoundAware",
                       "--weight", "PPMI", "--cutoff", "500", "--time-step", "10"]
    args = cumulative.parse_args(values)
    _require_configuration(args)
    return args


def main(argv=None):
    args = parse_args(argv)
    if args.command == "plan":
        specification = read_specification(args)
        plan = {"prepared_root": str(specification["root"]), "input_identity": specification["input_identity"],
            "count_input_identity": specification["count_input_identity"], **_policy(),
            "endpoints": specification["endpoints"], "end_years": [year + 9 for year in specification["endpoints"]],
            "lag_endpoints": [_lag_endpoint(specification, year) for year in specification["endpoints"]],
            "type_buckets": len(specification["buckets"]), "work_directory": str(specification["work"]),
            "condition": args.condition, "weight": args.weight, "context_size": args.context_size,
            "no_source_matrix_rescan": True,
            "mode": "cumulative raw history; lag-10 cumulative-prefix cosines; frozen ranked vocabulary"}
        print(json.dumps(plan, indent=2), flush=True)
        return plan
    return {"prepare": prepare, "features": run_features, "check": check}[args.command](args)


if __name__ == "__main__":
    main()
