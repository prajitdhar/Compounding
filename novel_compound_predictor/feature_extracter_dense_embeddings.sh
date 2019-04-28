#!/bin/sh

CUTOFF="20 50 100 200"
TIMESPAN="1 10 20 50 100"

for c in $CUTOFF; do for t in $TIMESPAN; do
	python3 feature_extracter_dense_embeddings.py --temporal $t --cutoff $c
done
done
