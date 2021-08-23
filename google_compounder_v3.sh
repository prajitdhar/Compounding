#!/bin/sh

FILES=$(ls /data/dharp/compounds/datasets/googleV3)

for f in $FILES; do
    echo $f
    python /data/dharp/compounds/Compounding/src/google_compounder_v3.py --data /data/dharp/compounds/datasets/googleV3/$f --output /data/dharp/compounds/datasets/v3_aware
    
done