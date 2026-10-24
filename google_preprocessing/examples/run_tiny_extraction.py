#!/usr/bin/env python3
"""Offline extraction/reduction smoke example using constructed n-gram rows.

No corpus download, benchmark data, spaCy model, NER inference, or Slurm needed.
This is a software example with invented observations.
"""
import argparse
import gzip
from pathlib import Path
import subprocess
import sys
import tempfile

GOOGLE = Path(__file__).resolve().parents[1]

def run(output):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    source = output / '2-synthetic.txt.gz'
    if source.exists():
        raise FileExistsError('Choose a new example output directory')
    source.write_bytes(gzip.compress((GOOGLE / 'tests/test_data/2-sample.txt').read_bytes(), mtime=0))
    program = [sys.executable, str(GOOGLE / 'google_v3_pipeline.py')]
    subprocess.run([*program, 'extract', '--source', str(source), '--order', '2',
                    '--output', str(output)], check=True)
    subprocess.run([*program, 'reduce', '--output', str(output), '--table', 'bigrams',
        '--threads', '1', '--memory-limit', '128MB', '--partitions', '2',
        '--shuffle-buffer-mb', '8', '--aggregation-workers', '1'], check=True)
    import pyarrow.parquet as pq
    table = pq.read_table(output / 'reduced/bigrams.parquet')
    assert table.num_rows > 0
    assert 'count' in table.column_names
    print(f'Offline example passed: {table.num_rows} reduced bigram rows')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, help='New output directory; default uses temporary files')
    args = parser.parse_args()
    if args.output:
        run(args.output)
    else:
        with tempfile.TemporaryDirectory(prefix='compounding-google-example-') as directory:
            run(directory)
