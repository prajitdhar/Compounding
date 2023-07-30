#!/bin/sh

FILES=`awk '{print $1}' /data/dharp/compounds/datasets/new_mithril_fcat_3.txt`



for f in $FILES; do
curfile=`basename -s .pkl $f`
python /data/dharp/compounds/Compounding/src/google_downloader_v3.py --file $curfile --spath /datanaco/dharp/compounds/datasets/googleV3/ 2>&1 | tee -a google_v3_mithril.txt;
done

