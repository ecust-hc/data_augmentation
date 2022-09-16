# Replace the following variables based on the paths where you store models and data
export checkpoint="../Tranception_Large"
export mutant_data_location="../ProteinGym_substitutions/A0A1I9GEU1_NEIME_Kennouche_2019.csv"
export scoring_location="../result/score.csv"
export target_seq="FTLIELMIVIAIVGILAAVALPAYQDYTARAQVSEAILLAEGQKSAVTEYYLNHGEWPGDNSSAGVATSADIKGKYVQSVTVANGVITAQMASSNVNNEIKSKKLSLWAKRQNGSVKWFCGQPVTRTTATATDVAAANGKTDDKINTKHLPSTCRDDSSAS"
#export inference_time_retrieval
#export MSA_folder=$PATH_TO_MSA
#export MSA_weights_folder=$PATH_TO_MSA_WEIGHTS_SUBSTITUTIONS 



python3 ../score_tranception_proteingym.py \
                --checkpoint ${checkpoint} \
                --mutant_data_location ${mutant_data_location} \
                --scoring_location ${scoring_location} \
                --target_seq ${target_seq}
#				--MSA_folder ${MSA_folder} \
#                --MSA_weights_folder ${MSA_weights_folder}
