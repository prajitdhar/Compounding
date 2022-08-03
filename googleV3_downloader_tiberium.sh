#!/bin/sh

FILES=$(seq 18001 18500)


for f in $FILES; do

    python /data/dharp/compounds/Compounding/src/google_compounder_v3.py --file $f --output /data/dharp/compounds/datasets/ 2>&1 | tee -a google_compounder_v3_rivium.txt
    
done