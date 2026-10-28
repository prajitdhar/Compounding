#!/usr/bin/env python3
"""Export pooled compound and constituent vectors for visualization.

Use heads/modifiers/compounds outputs from build_representations.py without
--temporal, plus its novel_compounds.csv.
Observed compounds take priority over additive future compounds. Output is TSV
or Parquet with common, novel, 0, ..., 299. Additive vectors are not normalized.
"""

import argparse
import json
import os
from pathlib import Path
import re
import tempfile

import numpy as np
import pandas as pd


def readable(token):
    return re.sub(r"_(?:NOUN|noun|n)$", "", token)


def table_columns(path):
    if path.suffix.lower() in {".parquet", ".pq"}:
        import pyarrow.parquet as pq
        return pq.ParquetFile(path).schema_arrow.names
    return pd.read_csv(path, sep="\t" if path.suffix.lower() == ".tsv" else ",",
                       nrows=0).columns.tolist()


def iter_table(path, keys, dimensions=None, batch_size=20000):
    names = table_columns(path)
    if dimensions is not None and {"decade", "time", "year"}.intersection(names):
        raise ValueError(f"{path}: temporal embeddings are unsupported; use pooled outputs "
                         "from build_representations.py without --temporal")
    coordinates = [f"dim_{i}" for i in range(dimensions)] if dimensions is not None else []
    columns = keys + coordinates
    missing = set(columns).difference(names)
    if missing:
        raise ValueError(f"{path}: missing required columns {sorted(missing)}; "
                         "requested dimensions are never silently reduced")
    if path.suffix.lower() in {".parquet", ".pq"}:
        import pyarrow.parquet as pq
        batches = (batch.to_pandas() for batch in
                   pq.ParquetFile(path).iter_batches(columns=columns, batch_size=batch_size))
    else:
        batches = pd.read_csv(path, sep="\t" if path.suffix.lower() == ".tsv" else ",",
                              usecols=columns, dtype={key: str for key in keys},
                              keep_default_na=False, chunksize=batch_size)
    for frame in batches:
        for key in keys:
            if frame[key].isna().any():
                raise ValueError(f"{path}: missing {key} lexical key")
            frame[key] = frame[key].astype(str)
            if frame[key].eq("").any() or frame[key].str.contains(r"\s", regex=True).any():
                raise ValueError(f"{path}: {key} must contain nonempty single-token lexical keys")
        if coordinates:
            frame[coordinates] = frame[coordinates].apply(pd.to_numeric, errors="raise")
            if not np.isfinite(frame[coordinates].to_numpy(dtype=np.float64)).all():
                raise ValueError(f"{path}: non-finite embedding coordinates")
        yield frame[columns]


def load_role(path, role, dimensions, batch_size):
    frames = list(iter_table(path, [role], dimensions, batch_size))
    if not frames:
        return pd.DataFrame(columns=[f"dim_{i}" for i in range(dimensions)],
                            index=pd.Index([], name=role), dtype=np.float64)
    frame = pd.concat(frames, ignore_index=True).set_index(role)
    if frame.index.duplicated().any():
        raise ValueError(f"{path}: duplicate {role} keys; use pooled embeddings")
    return frame


def atomic_text(path, value):
    fd, temporary = tempfile.mkstemp(prefix=".visualization-meta-", dir=path.parent)
    try:
        with os.fdopen(fd, "w") as stream:
            stream.write(value)
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


class DatasetWriter:
    def __init__(self, path, dimensions):
        self.path = path
        self.dimensions = dimensions
        self.parquet = path.suffix.lower() in {".parquet", ".pq"}
        fd, self.temporary = tempfile.mkstemp(prefix=".visualization-", dir=path.parent)
        os.close(fd)
        self.writer = None
        self.rows = 0

    def write(self, names, novel, vectors):
        if not names:
            return
        array = np.asarray(vectors, dtype=np.float64)
        if array.shape != (len(names), self.dimensions) or not np.isfinite(array).all():
            raise ValueError("Invalid output vector dimensions or non-finite additive coordinates")
        frame = pd.DataFrame(array, columns=[str(i) for i in range(self.dimensions)])
        frame.insert(0, "novel", np.full(len(names), novel, dtype=np.int64))
        frame.insert(0, "common", names)
        if self.parquet:
            import pyarrow as pa
            import pyarrow.parquet as pq
            table = pa.Table.from_pandas(frame, preserve_index=False)
            if self.writer is None:
                self.writer = pq.ParquetWriter(self.temporary, table.schema, compression="zstd")
            self.writer.write_table(table)
        else:
            frame.to_csv(self.temporary, sep="\t", index=False,
                         mode="a" if self.rows else "w", header=not self.rows, float_format="%.17g")
        self.rows += len(names)

    def close(self, publish=False):
        if self.writer is not None:
            self.writer.close()
            self.writer = None
        if publish:
            os.replace(self.temporary, self.path)
        if os.path.exists(self.temporary):
            os.unlink(self.temporary)


def run(args):
    paths = {key: Path(getattr(args, key)).expanduser() for key in
             ["heads", "modifiers", "compounds", "novel_compounds"]}
    for path in paths.values():
        if not path.is_file():
            raise ValueError(f"Source file unavailable: {path}")
    output = Path(args.output).expanduser()
    manifest_path = output.with_suffix(output.suffix + ".json")
    missing_path = output.with_suffix(output.suffix + ".missing.csv")
    for path in [output, manifest_path, missing_path]:
        if path.exists():
            raise ValueError(f"Output or report exists; choose a new output path: {path}")
    if args.dimensions < 1 or args.batch_size < 1:
        raise ValueError("dimensions and batch-size must be positive")
    heads = load_role(paths["heads"], "head", args.dimensions, args.batch_size)
    modifiers = load_role(paths["modifiers"], "modifier", args.dimensions, args.batch_size)
    # Validate the compound schema before creating any outputs.
    names = table_columns(paths["compounds"])
    if {"decade", "time", "year"}.intersection(names):
        raise ValueError("Temporal compound embeddings are unsupported; use pooled outputs")
    if not {"modifier", "head", *[f"dim_{i}" for i in range(args.dimensions)]}.issubset(names):
        raise ValueError("Compound table lacks tagged modifier/head keys or requested dimensions")
    output.parent.mkdir(parents=True, exist_ok=True)
    writer = DatasetWriter(output, args.dimensions)
    seen = set()
    counts = {"head_rows": 0, "observed_compound_rows": 0, "additive_novel_rows": 0,
              "duplicate_observed_identifiers": 0, "novel_input_rows": 0,
              "duplicate_novel_pairs": 0, "novel_observed_overlap": 0, "missing_novel_rows": 0}
    missing = []
    novel_pairs = set()
    committed = False
    try:
        for start in range(0, len(heads), args.batch_size):
            batch = heads.iloc[start:start + args.batch_size]
            readable_names, vectors = [], []
            for key, vector in zip(batch.index, batch.to_numpy(dtype=np.float64)):
                common = readable(key)
                if common in seen:
                    counts["duplicate_observed_identifiers"] += 1
                    continue
                seen.add(common)
                readable_names.append(common)
                vectors.append(vector)
            writer.write(readable_names, 0, vectors)
            counts["head_rows"] += len(readable_names)
        for batch in iter_table(paths["compounds"], ["modifier", "head"],
                                args.dimensions, args.batch_size):
            readable_names, vectors = [], []
            values = batch[[f"dim_{i}" for i in range(args.dimensions)]].to_numpy(dtype=np.float64)
            for modifier, head, vector in zip(batch.modifier, batch["head"], values):
                common = readable(modifier) + " " + readable(head)
                if common in seen:
                    counts["duplicate_observed_identifiers"] += 1
                    continue
                seen.add(common)
                readable_names.append(common)
                vectors.append(vector)
            writer.write(readable_names, 0, vectors)
            counts["observed_compound_rows"] += len(readable_names)
        for batch in iter_table(paths["novel_compounds"], ["modifier", "head"],
                                batch_size=args.batch_size):
            readable_names, vectors = [], []
            for modifier, head in zip(batch.modifier, batch["head"]):
                counts["novel_input_rows"] += 1
                pair = (modifier, head)
                if pair in novel_pairs:
                    counts["duplicate_novel_pairs"] += 1
                    continue
                novel_pairs.add(pair)
                common = readable(modifier) + " " + readable(head)
                if common in seen:
                    counts["novel_observed_overlap"] += 1
                    continue
                absent_modifier, absent_head = modifier not in modifiers.index, head not in heads.index
                if absent_modifier or absent_head:
                    missing.append({"modifier": modifier, "head": head, "common": common,
                                    "missing_modifier": absent_modifier, "missing_head": absent_head})
                    counts["missing_novel_rows"] += 1
                    continue
                vector = modifiers.loc[modifier].to_numpy(dtype=np.float64) + heads.loc[head].to_numpy(dtype=np.float64)
                seen.add(common)
                readable_names.append(common)
                vectors.append(vector)
            writer.write(readable_names, 1, vectors)
            counts["additive_novel_rows"] += len(readable_names)
        missing_frame = pd.DataFrame(missing, columns=["modifier", "head", "common",
                                                      "missing_modifier", "missing_head"])
        atomic_text(missing_path, missing_frame.to_csv(index=False))
        blocked = bool(missing) and not args.allow_missing_novel
        manifest = {"schema_version": 1, "status": "blocked_missing_novel" if blocked else "complete",
                    "dataset_type": "pooled compound and constituent visualization vectors",
                    "output": str(output.resolve()), "dimensions": args.dimensions,
                    "rows": writer.rows, "counts": counts,
                    "sources": {key: str(path.resolve()) for key, path in paths.items()},
                    "missing_report": str(missing_path.resolve()),
                    "allow_missing_novel": args.allow_missing_novel,
                    "method": "heads and observed compounds first; modifier + head for remaining novel pairs",
                    "normalization": "source vectors unchanged; additive vectors are not renormalized",
                    "lexical_keys": "tagged keys retained for lookup; noun POS suffix removed only in common",
                    "novel_label": "1 means additive representation retained after observed-priority deduplication",
                    "temporal_mode": "pooled embeddings only"}
        atomic_text(manifest_path, json.dumps(manifest, indent=2) + "\n")
        print(json.dumps({"output": str(output), "status": manifest["status"],
                          "dimensions": args.dimensions, "rows": writer.rows, **counts}, indent=2), flush=True)
        if blocked:
            raise ValueError(f"{len(missing):,} novel pairs lack constituent vectors; dataset not written. "
                             f"See {missing_path}. To explicitly accept this coverage, rerun with "
                             "--allow-missing-novel and a new output path.")
        if not writer.rows:
            raise ValueError("No visualization rows are available")
        writer.close(publish=True)
        committed = True
        return manifest
    finally:
        if not committed:
            writer.close()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for role in ["heads", "modifiers", "compounds"]:
        parser.add_argument("--" + role, type=Path, required=True)
    parser.add_argument("--novel-pairs", "--novel-compounds", dest="novel_compounds",
                        type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--dimensions", type=int, default=300)
    parser.add_argument("--batch-size", type=int, default=20000)
    parser.add_argument("--allow-missing-novel", action="store_true",
                        help="Explicitly accept missing novel pairs listed in the coverage report")
    return run(parser.parse_args(argv))


if __name__ == "__main__":
    main()
