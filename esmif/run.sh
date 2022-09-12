#! /bin/bash

python esmif.py \
    --r1 0 \
    --r2 2 \
    --tsv ./data/mutantion.tsv \
    --save ./data/result.tsv \
    --pdb ./data/5YH2.pdb
