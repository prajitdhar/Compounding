#!/bin/sh

FILES=$(seq 103 114)

for f in $FILES; do

    python /data/dharp/compounds/Compounding/src/google_compounder_v3.py --word --file /data/dharp/compounds/datasets/entire_df_v3/df_$f.parq --output /data/dharp/compounds/datasets/ 2>&1 | tee -a google_v3_compounder_adamant.txt
    
done