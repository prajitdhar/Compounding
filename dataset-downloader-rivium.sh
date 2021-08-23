#!/bin/sh


LETTERS="of it is in th co an be to wh"

for l in $LETTERS; do
python /data/dharp/compounds/Compounding/src/dataset_downloader-large.py --letter $l --chunksize 100000000 --output /data/dharp/compounds/datasets/
done
