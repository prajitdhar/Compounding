#!/bin/sh


LETTERS="se de"

for l in $LETTERS; do
python /data/dharp/compounds/Compounding/src/dataset_downloader-large.py --letter $l --cores 200 --chunksize 500000000
done
