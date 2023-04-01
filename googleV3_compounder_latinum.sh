#!/bin/sh

FILES=$(seq 1 30)

for f in $FILES; do

    python /data/dharp/compounds/Compounding/src/google_compounder_v3.py --file /datanaco/dharp/compounds/datasets/googleV3/df_$f.parq --output /data/dharp/compounds/datasets/ 2>&1 | tee -a google_v3_compounder_latinum.txt
    
done