#!/bin/sh


LETTERS="as on we ma pr ar ip sh ca so hi bu al se de"

for l in $LETTERS; do
python /data/dharp/compounds/Compounding/src/dataset_downloader-large.py --letter $l --cores 30 --chunksize 200000000
done