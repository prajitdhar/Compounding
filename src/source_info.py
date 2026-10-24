"""Corpus-independent artifact identities and the canonical feature specification.

Runtime manifests intentionally identify the actual local inputs used in a
run. Such manifests should be stored with private results, not committed to
the public source repository.
"""

import hashlib
from importlib import metadata
import json
from pathlib import Path
import platform
import sqlite3

from .io import atomic_path

SOURCE_INFO_SCHEMA_VERSION = 1
FEATURE_INPUT_SCHEMA_VERSION = 3

__all__ = [
    "SOURCE_INFO_SCHEMA_VERSION", "FEATURE_INPUT_SCHEMA_VERSION", "sha256_file",
    "artifact", "external_artifact", "software_record", "source_info",
    "write_json", "feature_table", "validate_feature_input_manifest",
    "canonical_time", "source_identity_paths",
]


def sha256_file(path):
    """Hash a file in bounded one-MiB chunks."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def artifact(path, name=None, sha256=None):
    path = Path(path)
    return {"name": name or path.name, "path": str(path.resolve()),
            "bytes": path.stat().st_size, "sha256": sha256 or sha256_file(path)}


def external_artifact(name, location, *, bytes=None, sha256=None):
    """Describe an external input without inventing its size or checksum."""
    return {"name": name, "path": str(location), "bytes": bytes, "sha256": sha256}


def source_identity_paths():
    """Return authoritative shared implementations to include in run hashes.

    Compatibility wrappers are not sufficient source identities on their own:
    include these files alongside a project's scientific code and wrappers.
    """
    directory = Path(__file__).resolve().parent
    return tuple(directory / name for name in (
        "source_info.py", "io.py", "progress.py", "partitioned_parquet.py",
    ))


def software_record(code_paths, packages=()):
    versions = {}
    for package in packages:
        try:
            versions[package] = metadata.version(package)
        except metadata.PackageNotFoundError:
            versions[package] = None
    paths = list(dict.fromkeys(Path(path).resolve() for path in
                              (*code_paths, *source_identity_paths())))
    # Different files with the same basename cannot be represented faithfully
    # by this schema. Repeating the same path remains backward-compatible.
    resolved = {}
    for path in paths:
        existing = resolved.setdefault(path.name, path.resolve())
        if existing != path.resolve():
            raise ValueError(f"Ambiguous code identity basename: {path.name}")
    return {"python": platform.python_version(), "packages": versions,
            "sqlite": sqlite3.sqlite_version,
            "code_sha256": {path.name: sha256_file(path) for path in paths}}


def source_info(stage, corpus, inputs, parameters, time, software, model=None,
               outputs=(), statistics=None, execution=None, status="complete"):
    """Create the common deterministic envelope without timestamps/self-hashes."""
    if status not in {"planned", "complete", "partial"}:
        raise ValueError(f"Unknown source information status {status!r}")
    return {"schema_version": SOURCE_INFO_SCHEMA_VERSION, "stage": stage,
            "status": status, "corpus": corpus, "inputs": list(inputs),
            "parameters": parameters, "time": time, "software": software,
            "model": model, "outputs": list(outputs),
            "statistics": statistics or {}, "execution": execution or {}}


def write_json(path, value):
    """Atomically publish stable, UTF-8, human-readable JSON."""
    with atomic_path(path) as temporary:
        temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def feature_table(path, columns, count_unit, *, required=True, name=None):
    record = artifact(path, name=name)
    record.update({"columns": list(columns), "count_unit": count_unit,
                   "required": bool(required), "format": Path(path).suffix.lstrip(".")})
    return record


def validate_feature_input_manifest(value):
    """Validate the all-lexical, independent-count Parquet feature specification."""
    version = value.get("feature_input_schema_version", value.get("schema_version"))
    if version != FEATURE_INPUT_SCHEMA_VERSION:
        raise ValueError(f"Unsupported feature-input schema version: {version!r}")
    if (value.get("format") != "parquet"
            or value.get("word_target_pos") != "all_lexical"
            or value.get("bigram_target_pos") != "all_lexical"):
        raise ValueError("Schema 3 requires all-lexical unigram/bigram Parquet inputs")
    required = {"unigrams", "unigram_counts", "bigrams", "bigram_counts"}
    table_files = value.get("table_files")
    if not isinstance(table_files, dict):
        raise ValueError("Feature-input manifest must contain a table_files object")
    missing = sorted(required.difference(table_files))
    if missing:
        raise ValueError(f'Missing required feature tables: {", ".join(missing)}')
    block = value.get("source_info", {})
    if block.get("status") != "complete" or block.get("stage") != "feature_inputs":
        raise ValueError("Feature-input source information must describe a completed feature_inputs stage")
    return value


def canonical_time(frame):
    """Accept time/year input columns without silently merging clocks."""
    frame = frame.copy()
    if "year" in frame:
        if "time" in frame:
            if not frame["time"].equals(frame["year"]):
                raise ValueError("Conflicting time and year columns")
            frame = frame.drop(columns="year")
        else:
            frame = frame.rename(columns={"year": "time"})
    if "time" not in frame:
        raise ValueError("Missing time column (year is also accepted)")
    return frame
