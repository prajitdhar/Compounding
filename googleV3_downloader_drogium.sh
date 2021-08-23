#!/bin/sh

FILES=$(seq 3001 4000)

for f in $FILES; do

    python /data/dharp/compounds/Compounding/src/google_downloader_v3.py --file $f
    
done