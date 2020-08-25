#!/bin/sh

CUTOFF="20 50 100"
TIMESPAN="10 20 50 100"

for c in $CUTOFF; do for t in $TIMESPAN; do
	python dimreduce.py --inputdir ../../Compounding/coha_compounds --outputdir ../../Compounding/coha_compounds --temporal $t --cutoff $c --storedf --contextual
done
done
