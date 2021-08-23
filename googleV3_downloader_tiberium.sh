#!/bin/sh

FILES=$(seq 2001 3000)

for f in $FILES; do

    python /data/dharp/compounds/Compounding/src/google_downloader_v3.py --file $f
    
done