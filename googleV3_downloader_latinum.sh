#!/bin/sh

FILES=({0..2000})


for f in "${FILES[@]}"; do
echo $f
    python /data/dharp/compounds/Compounding/src/google_downloader_v3.py --file $f --spath /datanaco/dharp/compounds/datasets/googleV3/ 2>&1 | tee -a google_v3_latinum.txt;
done

