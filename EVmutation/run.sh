#! /bin/bash

#python evmutant.py \
#    --model-params "example/PABP_YEAST.model_params" \
#    --mutantion "example/PABP_YEAST_Fields2013-singles.csv" \
#    --computer-mode "single" \
#    --save "example/result.tsv" 

python evmutant.py \
    --model-params "example/PABP_YEAST.model_params" \
    --mutantion "example/double.csv" \
    --computer-mode "multiple" \
    --save "example/resultdouble.tsv" 
