#!/bin/sh


python /data/dharp/compounds/Compounding/src/google_compounder_v3.py --data /datanaco/dharp/compounds/datasets/googleV3/ --output /data/dharp/compounds/datasets --decade 1970 --word 2>&1 | tee -a google_v3_compounder_word_rivium.txt
    