#!/bin/sh

CUTOFF="20 50 100"
TIMESPAN="0"

for c in $CUTOFF; do for t in $TIMESPAN; do
	python feature_extracter_dense_embeddings.py --temporal $t --cutoff $c --contextual
done
done
