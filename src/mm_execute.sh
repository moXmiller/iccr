#!/bin/bash

#SBATCH --mem-per-cpu=20000
#SBATCH -n 4
#SBATCH --gpus=rtx_3090:1
#SBATCH --time=4000
#SBATCH --job-name=lstm
#SBATCH --output=output_files/15_10_lstm.out

python_files_dir="/cluster/home/millerm/cf/garg_cf/src"
conda_env="/cluster/scratch/millerm/miniconda3_ths/envs/icl"
echo "paths read in"

source activate $conda_env
echo "conda environment activated"

module load eth_proxy
module load stack/2024-05
module load gcc/13.2.0
module load cuda/12.2.1
echo "modules loaded"

wandb online

echo "python execution started"

# python3 ${python_files_dir}/train.py --config conf/counterfactual.yaml --training.train_steps 1000000 --training.diversity 320000000 --wandb.name "GPT2_5d_1mio"

# python3 ${python_files_dir}/write_eval.py --data sde --family gpt2_sde --batch_size 8 --n_thetas 8 --diversity 2000000 --poisson 0 --lamb 40 --max_time 0.5 --dim_index 2 --training_loss masked_mse
# python3 ${python_files_dir}/write_eval.py --data sde --family gpt2_sde --batch_size 8 --n_thetas 8 --diversity 2000000 --lamb 40 --max_time 0.5 --training_loss masked_mse
# python3 ${python_files_dir}/write_eval.py --data sde --poisson 0 --family gpt2_sde --ode 1 --batch_size 8 --n_thetas 8 --diversity 2000000 --dim_index 2 --training_loss masked_mse

# python3 ${python_files_dir}/train.py --config conf/attentiononly.yaml --wandb.name "8l1hAO_1mio" --model.family gpt2_ao --training.train_steps 1000000 --training.diversity 320000000 
# python3 ${python_files_dir}/write_eval.py --family gpt2_ao --ao 1 --n_layer 8 --n_head 1 --train_steps 1000000 --diversity 320000000
# python3 ${python_files_dir}/train.py --config conf/attentiononly.yaml --wandb.name "8l1hAO_1mio_cont" --training.continuation 1 --model.family gpt2_ao_cont --training.train_steps 1000000 --training.diversity 320000000 
# python3 ${python_files_dir}/write_eval.py --family gpt2_ao_cont --ao 1 --n_layer 8 --n_head 1 --continuation 1 --train_steps 1000000 --diversity 320000000

# python3 ${python_files_dir}/train.py --config conf/counterfactual.yaml --wandb.name "GPT2_5d_1mio_cont" --training.continuation 1 --model.family gpt2_cont --training.train_steps 1000000 --training.diversity 320000000 
# python3 ${python_files_dir}/write_eval.py --family gpt2_cont --n_layer 12 --n_head 8 --continuation 1 --train_steps 1000000 --diversity 320000000

# python3 ${python_files_dir}/write_eval.py --data sde --family rnn_sde --batch_size 8 --n_thetas 8 --diversity 2000000 --n_layer 2 --training_loss masked_mse --position 2 --dim_index 2
# python3 ${python_files_dir}/write_sde_data.py --lamb 40 --max_time 0.5 --number_events 20
# python3 ${python_files_dir}/train.py --config conf/sde.yaml --training.poisson 1 --training.training_loss masked_mse --wandb.name sde_pois_20evts --training.number_events 20
# python3 ${python_files_dir}/train.py --config conf/sde.yaml --training.poisson 1 --training.training_loss masked_mse --model.family lstm_sde --wandb.name lstm_sde_2l_pois_20evts --model.n_layer 2 --training.number_events 20
# python3 ${python_files_dir}/train.py --config conf/sde.yaml --training.poisson 1 --training.training_loss masked_mse --model.family rnn_sde --wandb.name rnn_sde_2l_pois_20evts --model.n_layer 2 --training.number_events 20
# python3 ${python_files_dir}/train.py --config conf/sde.yaml --training.poisson 1 --training.training_loss masked_mse --model.family gru_sde --wandb.name gru_sde_2l_pois_20evts --model.n_layer 2 --training.number_events 20

for i in {15..101}
do
    echo "Iteration $i"
    python3 ${python_files_dir}/write_eval.py --train_steps 1000000 --o_dims 1 --diversity 64000000 --itr $i
done

# python3 ${python_files_dir}/write_eval.py --data sde --family gpt2_sde --batch_size 8 --n_thetas 8 --diversity 2000000 --training_loss masked_mse --number_events 20
# python3 ${python_files_dir}/write_eval.py --data sde --family lstm_sde --batch_size 8 --n_thetas 8 --diversity 2000000 --n_layer 2 --training_loss masked_mse --number_events 20
# python3 ${python_files_dir}/write_eval.py --data sde --family rnn_sde --batch_size 8 --n_thetas 8 --diversity 2000000 --n_layer 2 --training_loss masked_mse --number_events 20
# python3 ${python_files_dir}/write_eval.py --data sde --family gru_sde --batch_size 8 --n_thetas 8 --diversity 2000000 --n_layer 2 --training_loss masked_mse --number_events 20

# python3 ${python_files_dir}/train.py --config conf/counterfactual.yaml --training.curriculum.dims.start 1 --training.curriculum.dims.end 1 --model.n_dims 1 --training.diversity 6400000 --training.curriculum.points.start 30 --training.constant_z 14 --wandb.name "CF_constz14_1dim_100k" --training.train_steps 100000
# python3 ${python_files_dir}/train.py --config conf/counterfactual.yaml --training.curriculum.dims.start 2 --training.curriculum.dims.end 2 --model.n_dims 2 --training.diversity 12800000 --training.curriculum.points.start 30 --training.constant_z 14 --wandb.name "CF_constz14_2dim_100k" --training.train_steps 100000
# python3 ${python_files_dir}/train.py --config conf/counterfactual.yaml --training.curriculum.dims.start 3 --training.curriculum.dims.end 3 --model.n_dims 3 --training.diversity 19200000 --training.curriculum.points.start 30 --training.constant_z 14 --wandb.name "CF_constz14_3dim_100k" --training.train_steps 100000
# python3 ${python_files_dir}/train.py --config conf/counterfactual.yaml --training.curriculum.dims.start 4 --training.curriculum.dims.end 4 --model.n_dims 4 --training.diversity 25600000 --training.curriculum.points.start 30 --training.constant_z 14 --wandb.name "CF_constz14_4dim_100k" --training.train_steps 100000
# python3 ${python_files_dir}/train.py --config conf/counterfactual.yaml --training.curriculum.dims.start 1 --training.curriculum.dims.end 1 --model.n_dims 1 --training.diversity 6400000 --training.curriculum.points.start 30 --wandb.name "CF_min30_1dim_100k" --training.train_steps 100000


echo "python script executed"

# conda deactivate

echo "experiment completed"