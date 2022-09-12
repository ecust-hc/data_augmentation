
cd /home/huchao/work/data-augmentation/eve/
source /home/huchao/anaconda3/bin/activate /home/huchao/anaconda3/envs/evo

export MSA_data_location='./data/MSA/PTEN_HUMAN_b1.0.a2m'
export MSA_weights_location='./data/weights'
export VAE_checkpoint_location='./results/VAE_parameters'
export model_name_suffix='Jan1_PTEN_example'
export model_parameters_location='./EVE/default_model_params.json'
export training_logs_location='./logs/'
export protein_name='PTEN_HUMAN'

python ./train_VAE.py \
    --MSA_data_location ${MSA_data_location} \
    --protein_name ${protein_name} \
    --MSA_weights_location ${MSA_weights_location} \
    --VAE_checkpoint_location ${VAE_checkpoint_location} \
    --model_name_suffix ${model_name_suffix} \
    --model_parameters_location ${model_parameters_location} \
    --training_logs_location ${training_logs_location} 

export MSA_data_location='./data/MSA/PTEN_HUMAN_b1.0.a2m'
export MSA_weights_location='./data/weights'
export VAE_checkpoint_location='./results/VAE_parameters'
export model_name_suffix='Jan1_PTEN_example'
export model_parameters_location='./EVE/default_model_params.json'
export training_logs_location='./logs/'
export protein_name='PTEN_HUMAN'

export computation_mode='all_singles'
export all_singles_mutations_folder='./data/mutations'
export output_evol_indices_location='./results/evol_indices'
export num_samples_compute_evol_indices=20000
export batch_size=2048

python ./compute_evol_indices.py \
    --MSA_data_location ${MSA_data_location} \
    --protein_name ${protein_name} \
    --MSA_weights_location ${MSA_weights_location} \
    --VAE_checkpoint_location ${VAE_checkpoint_location} \
    --model_name_suffix ${model_name_suffix} \
    --model_parameters_location ${model_parameters_location} \
    --computation_mode ${computation_mode} \
    --all_singles_mutations_folder ${all_singles_mutations_folder} \
    --output_evol_indices_location ${output_evol_indices_location} \
    --num_samples_compute_evol_indices ${num_samples_compute_evol_indices} \
    --batch_size ${batch_size}
