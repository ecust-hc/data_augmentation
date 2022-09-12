#! /bin/bash

#python esm1v.py \
#    --model-location esm1v_t33_650M_UR90S_1 esm1v_t33_650M_UR90S_2 esm1v_t33_650M_UR90S_3 esm1v_t33_650M_UR90S_4 esm1v_t33_650M_UR90S_5 \
#    --fasta ./data/BLAT_ECOLX_Ranganathan2015.fasta \
#    --dms-input ./data/more1.csv \
#    --mutation-col mutant \
#    --dms-output ./data/BLAT_ECOLX_Ranganathan2015_labeled2.csv \
#    --offset-idx 24 \
#    --scoring-strategy masked-marginals
#    --scoring-strategy pseudo-ppl


python esm1v.py \
   --model-location esm_msa1b_t12_100M_UR50S \
   --fasta ./data/BLAT_ECOLX_seq.fasta \
   --dms-input ./data/BLAT_ECOLX_Ranganathan2015.csv \
   --mutation-col mutant \
   --dms-output ./data/BLAT_ECOLX_Ranganathan2015_label.csv \
   --offset-idx 24 \
   --scoring-strategy masked-marginals \
   --msa-path ./data/BLAT_ECOLX_1_b0.5.a3m
