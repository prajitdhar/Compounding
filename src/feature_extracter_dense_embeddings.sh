#!/bin/sh

CUTOFF="20 50 100"
TIMESPAN="10 20 50 100"

for c in $CUTOFF; do for t in $TIMESPAN; do
	python feature_extracter_dense_embeddings.py --inputdir ../../Compounding/coha_compounds --outputdir ../../Compounding/coha_compounds --temporal $t --cutoff $c --contextual
done
done
