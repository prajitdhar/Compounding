#!/bin/sh



FILES=`awk '{print $2}' /data/dharp/compounds/datasets/new_rivium_fcat.txt`

for f in $FILES; do
echo $f
curfile=`basename -s .pkl $f`
    python /data/dharp/compounds/Compounding/src/google_downloader_v3.py --file $curfile --spath /data/dharp/compounds/datasets/googleV3 2>&1 | tee -a google_v3_rivium.txt;
done

