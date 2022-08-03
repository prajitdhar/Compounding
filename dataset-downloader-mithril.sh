#!/bin/sh



FILES=`awk '{print $2}' /data/dharp/compounds/datasets/mithril_fcat.txt`

for f in $FILES; do
curfile=`basename -s .pkl $f`
python /data/dharp/compounds/Compounding/src/google_downloader_v3.py --file $curfile --spath /data/dharp/compounds/datasets/;
done

