"""Convert the three dense embedding CSVs to Parquet without loading them all into RAM.

Requires Python 3.8+ and pyarrow (install with: python -m pip install pyarrow).

Run from any directory:
    python csv_embeddings_to_parquet.py

By default, the 300 embedding columns are stored as float32, matching the
precision used to generate these CSVs. Use --float64 to retain every decimal
digit in the CSV instead. Existing Parquet files are never overwritten unless
--overwrite is given.
"""

import argparse
import csv
import os
from pathlib import Path
import sys
import tempfile


HERE = Path(__file__).resolve().parent
FILES = {
    "heads": ["head"],
    "modifiers": ["modifier"],
    "compounds": ["compound", "modifier", "head"],
}
DIMENSIONS = ["dim_{}".format(i) for i in range(300)]


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--directory", type=Path, default=HERE,
        help="Directory containing the CSVs; Parquet files go here too",
    )
    parser.add_argument(
        "--compression", choices=("zstd", "snappy"), default="zstd",
        help="Parquet compression (default: zstd for smaller transfers)",
    )
    parser.add_argument(
        "--float64", action="store_true",
        help="Keep full CSV decimal precision instead of using float32",
    )
    parser.add_argument(
        "--overwrite", action="store_true",
        help="Replace existing Parquet files",
    )
    return parser.parse_args()


def check_header(path, expected):
    if not path.is_file():
        raise FileNotFoundError(path)
    with path.open("r", encoding="utf-8", newline="") as source:
        actual = next(csv.reader(source), None)
    if actual != expected:
        raise ValueError("Unexpected columns in {}".format(path))


def convert(source, target, text_columns, args, pa, pacsv, pq):
    column_types = {name: pa.string() for name in text_columns}
    number_type = pa.float64() if args.float64 else pa.float32()
    column_types.update({name: number_type for name in DIMENSIONS})

    reader = pacsv.open_csv(
        str(source),
        read_options=pacsv.ReadOptions(block_size=64 * 1024 * 1024),
        convert_options=pacsv.ConvertOptions(column_types=column_types),
    )

    # Write in the destination directory so the final rename is atomic.
    fd, temp_name = tempfile.mkstemp(
        prefix="." + target.stem + "-", suffix=".parquet.tmp", dir=str(target.parent)
    )
    os.close(fd)
    row_count = 0
    next_report = 100_000
    try:
        writer = pq.ParquetWriter(
            temp_name, reader.schema, compression=args.compression,
            use_dictionary=(["modifier", "head"] if "compound" in text_columns else False),
        )
        try:
            for batch in reader:
                writer.write_batch(batch)
                row_count += batch.num_rows
                if row_count >= next_report:
                    print("{}: {:,} rows converted".format(source.name, row_count), flush=True)
                    next_report = row_count + 100_000
        finally:
            writer.close()

        metadata = pq.ParquetFile(temp_name).metadata
        if metadata.num_rows != row_count:
            raise RuntimeError("Parquet row count does not match the CSV stream")
        os.replace(temp_name, target)
    finally:
        if os.path.exists(temp_name):
            os.unlink(temp_name)

    print("{}: {:,} rows -> {} ({:.2f} GiB)".format(
        source.name, row_count, target.name, target.stat().st_size / 1024**3
    ), flush=True)


def main():
    args = parse_args()
    directory = args.directory.expanduser().resolve()
    jobs = []
    for kind, text_columns in FILES.items():
        source = directory / "dense_embeddings_{}.csv".format(kind)
        target = directory / "dense_embeddings_{}.parquet".format(kind)
        check_header(source, text_columns + DIMENSIONS)
        if target.exists() and not args.overwrite:
            raise FileExistsError("{} exists; use --overwrite to replace it".format(target))
        jobs.append((source, target, text_columns))

    if sys.version_info < (3, 8):
        raise SystemExit("Python 3.8 or newer is required for PyArrow")
    try:
        import pyarrow as pa
        import pyarrow.csv as pacsv
        import pyarrow.parquet as pq
    except ImportError as exc:
        raise SystemExit("Install PyArrow with: python -m pip install pyarrow") from exc

    if not pa.Codec.is_available(args.compression):
        raise SystemExit("{} codec unavailable; try --compression snappy".format(args.compression))

    for source, target, text_columns in jobs:
        convert(source, target, text_columns, args, pa, pacsv, pq)


if __name__ == "__main__":
    main()
