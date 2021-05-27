#!/bin/sh


LETTERS="th punctuation wh"

for l in $LETTERS; do
python /data/dharp/compounds/Compounding/src/dataset_downloader-large.py --letter $l --cores 100 --chunksize 1000000000
done
