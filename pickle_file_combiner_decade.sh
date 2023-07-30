#!/bin/sh

DECADES=`seq 2000 10 2010`


for dec in $DECADES;do

    python /data/dharp/compounds/Compounding/src/pickle_file_combiner.py --spath /datanaco/dharp/compounds/datasets/googleV3/ --dec $dec;
    done