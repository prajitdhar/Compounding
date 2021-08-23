#!/bin/sh


LETTERS="a_ ad ha wa he no wi fo re as on we ma pr ar ip sh ca so hi bu al se de by"

for l in $LETTERS; do
python /data/dharp/compounds/Compounding/src/dataset_downloader-large.py --letter $l --chunksize 100000000 --output /data/dharp/compounds/datasets/
done
