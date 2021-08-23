#!/bin/sh


LETTERS="wo st fr di mo su at or yo me li pa do ex le pe po if ne fi un fa sa ch la lo ac ho mu go si en ev tr"

for l in $LETTERS; do
python /data/dharp/compounds/Compounding/src/dataset_downloader-large.py --letter $l --chunksize 100000000  --output /data/dharp/compounds/datasets/
done



