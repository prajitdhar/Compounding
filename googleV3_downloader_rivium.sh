#!/bin/sh

FILES=$(seq 71 1000)

for f in $FILES; do

    python /data/dharp/compounds/Compounding/src/google_downloader_v3.py --file $f
    
done