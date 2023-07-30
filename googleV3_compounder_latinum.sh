#!/bin/sh

DECADES=$(seq 1980 10 2010)

for d in $DECADES; do

    python /data/dharp/compounds/Compounding/src/google_compounder_v3.py --data /datanaco/dharp/compounds/datasets/googleV3/ --output /data/dharp/compounds/datasets --decade $d --word 2>&1 | tee -a google_v3_compounder_word_latinum.txt
    
done