#!/bin/bash

#SBATCH --mem-per-cpu=20000
#SBATCH -n 4
#SBATCH --gpus=rtx_3090:1
#SBATCH --time=2000
#SBATCH --job-name=iclr
#SBATCH --output=output_files/12_09_iclr.out

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

################## write_mse

python3 ${python_files_dir}/write_eval.py
python3 ${python_files_dir}/write_eval.py --family gpt2_ao --ao 1

python3 ${python_files_dir}/write_eval.py --family gpt2_mlp --n_layer 8 --n_head 1
python3 ${python_files_dir}/write_eval.py --n_layer 8 --n_head 1
python3 ${python_files_dir}/write_eval.py --n_layer 4 --n_head 1
python3 ${python_files_dir}/write_eval.py --n_layer 2 --n_head 1
python3 ${python_files_dir}/write_eval.py --family gpt2_ao --ao 1 --n_layer 8 --n_head 1
python3 ${python_files_dir}/write_eval.py --family gpt2_ao --ao 1 --n_layer 4 --n_head 1
python3 ${python_files_dir}/write_eval.py --family gpt2_ao --ao 1 --n_layer 2 --n_head 1

python3 ${python_files_dir}/write_eval.py --n_layer 4 --n_head 2
python3 ${python_files_dir}/write_eval.py --n_layer 2 --n_head 4
python3 ${python_files_dir}/write_eval.py --n_layer 1 --n_head 8
python3 ${python_files_dir}/write_eval.py --family gpt2_ao --ao 1 --n_layer 4 --n_head 2
python3 ${python_files_dir}/write_eval.py --family gpt2_ao --ao 1 --n_layer 2 --n_head 4
python3 ${python_files_dir}/write_eval.py --family gpt2_ao --ao 1 --n_layer 1 --n_head 8

python3 ${python_files_dir}/write_eval.py --family gpt2_ao --ao 1 --n_layer 8 --n_head 1 --transformation mullin
python3 ${python_files_dir}/write_eval.py --family gpt2_ao --ao 1 --n_layer 8 --n_head 1 --transformation tanh
python3 ${python_files_dir}/write_eval.py --family gpt2_ao --ao 1 --n_layer 8 --n_head 1 --transformation sigmoid
python3 ${python_files_dir}/write_eval.py --n_layer 8 --n_head 1 --transformation mullin
python3 ${python_files_dir}/write_eval.py --n_layer 8 --n_head 1 --transformation tanh
python3 ${python_files_dir}/write_eval.py --n_layer 8 --n_head 1 --transformation sigmoid

python3 ${python_files_dir}/write_eval.py --family gpt2_ao --ao 1 --n_layer 8 --n_head 1 --randomize_labels 1

python3 ${python_files_dir}/write_eval.py --family gpt2_ao --ao 1 --n_layer 8 --n_head 1 --block_setup 0

python3 ${python_files_dir}/write_eval.py --family lstm --n_layer 3
python3 ${python_files_dir}/write_eval.py --family rnn --n_layer 3
python3 ${python_files_dir}/write_eval.py --family gru --n_layer 3
python3 ${python_files_dir}/write_eval.py --family lstm --n_layer 2
python3 ${python_files_dir}/write_eval.py --family rnn --n_layer 2
python3 ${python_files_dir}/write_eval.py --family gru --n_layer 2

################## write_mse

python3 ${python_files_dir}/write_eval.py --family gpt2_ao --ao 1 --n_layer 8 --n_head 1 --train_steps 10000 --diversity 3200000
python3 ${python_files_dir}/write_eval.py --n_layer 8 --n_head 1 --train_steps 10000 --diversity 3200000
python3 ${python_files_dir}/write_eval.py --family gpt2_ao --ao 1 --n_layer 8 --n_head 1 --train_steps 20000 --diversity 6400000
python3 ${python_files_dir}/write_eval.py --n_layer 8 --n_head 1 --train_steps 20000 --diversity 6400000
python3 ${python_files_dir}/write_eval.py --family gpt2_ao --ao 1 --n_layer 8 --n_head 1 --train_steps 100000 --diversity 32000000
python3 ${python_files_dir}/write_eval.py --n_layer 8 --n_head 1 --train_steps 100000 --diversity 32000000
python3 ${python_files_dir}/write_eval.py --family gpt2_ao --ao 1 --n_layer 8 --n_head 1 --train_steps 150000 --diversity 48000000
python3 ${python_files_dir}/write_eval.py --n_layer 8 --n_head 1 --train_steps 150000 --diversity 48000000
python3 ${python_files_dir}/write_eval.py --family gpt2_ao --ao 1 --n_layer 8 --n_head 1 --train_steps 200000 --diversity 64000000
python3 ${python_files_dir}/write_eval.py --n_layer 8 --n_head 1 --train_steps 200000 --diversity 64000000

python3 ${python_files_dir}/write_eval.py --family gpt2_ao --ao 1 --o_dims 1 --diversity 3200000 --n_head 1 --n_layer 8
python3 ${python_files_dir}/write_eval.py --family gpt2_ao --ao 1 --o_dims 2 --diversity 6400000 --n_head 1 --n_layer 8
python3 ${python_files_dir}/write_eval.py --family gpt2_ao --ao 1 --o_dims 10 --diversity 32000000 --n_head 1 --n_layer 8
python3 ${python_files_dir}/write_eval.py --family gpt2_ao --ao 1 --o_dims 20 --diversity 64000000 --n_head 1 --n_layer 8
python3 ${python_files_dir}/write_eval.py --family gpt2_ao --ao 1 --o_dims 35 --diversity 112000000 --n_head 1 --n_layer 8
python3 ${python_files_dir}/write_eval.py --family gpt2_ao --ao 1 --o_dims 50 --diversity 160000000 --n_head 1 --n_layer 8
python3 ${python_files_dir}/write_eval.py --family gpt2_ao --ao 1 --o_dims 250 --diversity 800000000 --n_head 1 --n_layer 8

python3 ${python_files_dir}/write_eval.py --family gpt2_ao --ao 1 --o_dims 1 --diversity 3200000 --n_embd 64 --n_head 1 --n_layer 8
python3 ${python_files_dir}/write_eval.py --family gpt2_ao --ao 1 --o_dims 2 --diversity 6400000 --n_embd 128 --n_head 1 --n_layer 8
python3 ${python_files_dir}/write_eval.py --family gpt2_ao --ao 1 --o_dims 4 --diversity 12800000 --n_embd 256 --n_head 1 --n_layer 8

python3 ${python_files_dir}/write_eval.py --family gpt2_ao --ao 1 --o_dims 1 --diversity 3200000 --n_layer 1 --n_head 1
python3 ${python_files_dir}/write_eval.py --family gpt2_ao --ao 1 --o_dims 1 --diversity 3200000 --n_layer 2 --n_head 1
python3 ${python_files_dir}/write_eval.py --family gpt2_ao --ao 1 --o_dims 1 --diversity 3200000 --n_layer 4 --n_head 1

python3 ${python_files_dir}/write_eval.py --family gpt2_ao --ao 1 --min_examples 30 --n_layer 8 --n_head 1
python3 ${python_files_dir}/write_eval.py --family gpt2_ao --ao 1 --min_examples 30 --constant_z 0 --n_layer 8 --n_head 1
python3 ${python_files_dir}/write_eval.py --family gpt2_ao --ao 1 --min_examples 30 --constant_z 4 --n_layer 8 --n_head 1
python3 ${python_files_dir}/write_eval.py --family gpt2_ao --ao 1 --min_examples 30 --constant_z 9 --n_layer 8 --n_head 1
python3 ${python_files_dir}/write_eval.py --family gpt2_ao --ao 1 --min_examples 30 --constant_z 14 --n_layer 8 --n_head 1
python3 ${python_files_dir}/write_eval.py --family gpt2_ao --ao 1 --min_examples 30 --constant_z 19 --n_layer 8 --n_head 1
python3 ${python_files_dir}/write_eval.py --family gpt2_ao --ao 1 --min_examples 30 --constant_z 24 --n_layer 8 --n_head 1
python3 ${python_files_dir}/write_eval.py --family gpt2_ao --ao 1 --min_examples 30 --constant_z 29 --n_layer 8 --n_head 1

python3 ${python_files_dir}/write_eval.py --family gpt2_ao --ao 1 --training_loss mae --n_layer 8 --n_head 1
python3 ${python_files_dir}/write_eval.py --family gpt2_ao --ao 1 --training_loss mae --n_layer 8 --n_head 1 --eval_loss mae
python3 ${python_files_dir}/write_eval.py --family gpt2_ao --ao 1 --training_loss mse --n_layer 8 --n_head 1 --eval_loss mae

################## predict_bins

python3 ${python_files_dir}/write_eval.py --family gpt2_ao --ao 1 --n_layer 8 --n_head 1 --min_examples 30 --constant_z 19 --o_dims 1 --diversity 3200000 --n_bins 12 --training_loss rps
python3 ${python_files_dir}/write_eval.py --family gpt2_ao --ao 1 --n_layer 8 --n_head 1 --min_examples 30 --constant_z 19 --o_dims 1 --diversity 3200000 --n_bins 24 --training_loss rps
python3 ${python_files_dir}/write_eval.py --family gpt2_ao --ao 1 --n_layer 8 --n_head 1 --min_examples 30 --constant_z 19 --o_dims 1 --diversity 3200000 --n_bins 48 --training_loss rps
python3 ${python_files_dir}/write_eval.py --family gpt2_ao --ao 1 --n_layer 8 --n_head 1 --min_examples 30 --constant_z 19 --o_dims 1 --diversity 3200000 --n_bins 96 --training_loss rps
python3 ${python_files_dir}/write_eval.py --family gpt2_ao --ao 1 --n_layer 8 --n_head 1 --min_examples 30 --constant_z 19 --o_dims 1 --diversity 3200000 --n_bins 192 --training_loss rps
python3 ${python_files_dir}/write_eval.py --family gpt2_ao --ao 1 --n_layer 8 --n_head 1 --min_examples 30 --constant_z 19 --o_dims 1 --diversity 3200000 --n_bins 384 --training_loss rps
python3 ${python_files_dir}/write_eval.py --family gpt2_ao --ao 1 --n_layer 8 --n_head 1 --min_examples 30 --constant_z 19 --o_dims 1 --diversity 3200000 --n_bins 768 --training_loss rps
python3 ${python_files_dir}/write_eval.py --family gpt2_ao --ao 1 --n_layer 8 --n_head 1 --min_examples 30 --constant_z 19 --o_dims 1 --diversity 3200000 --n_bins 1536 --training_loss rps
python3 ${python_files_dir}/write_eval.py --family gpt2_ao --ao 1 --n_layer 8 --n_head 1 --min_examples 30 --constant_z 19 --o_dims 1 --diversity 3200000 --n_bins 12 --block_setup 0 --training_loss rps
python3 ${python_files_dir}/write_eval.py --family gpt2_ao --ao 1 --n_layer 8 --n_head 1 --min_examples 30 --constant_z 19 --o_dims 1 --diversity 3200000 --n_bins 24 --block_setup 0 --training_loss rps
python3 ${python_files_dir}/write_eval.py --family gpt2_ao --ao 1 --n_layer 8 --n_head 1 --min_examples 30 --constant_z 19 --o_dims 1 --diversity 3200000 --n_bins 48 --block_setup 0 --training_loss rps
python3 ${python_files_dir}/write_eval.py --family gpt2_ao --ao 1 --n_layer 8 --n_head 1 --min_examples 30 --constant_z 19 --o_dims 1 --diversity 3200000 --n_bins 96 --block_setup 0 --training_loss rps
python3 ${python_files_dir}/write_eval.py --family gpt2_ao --ao 1 --n_layer 8 --n_head 1 --min_examples 30 --constant_z 19 --o_dims 1 --diversity 3200000 --n_bins 192 --block_setup 0 --training_loss rps
python3 ${python_files_dir}/write_eval.py --family gpt2_ao --ao 1 --n_layer 8 --n_head 1 --min_examples 30 --constant_z 19 --o_dims 1 --diversity 3200000 --n_bins 384 --block_setup 0 --training_loss rps
python3 ${python_files_dir}/write_eval.py --family gpt2_ao --ao 1 --n_layer 8 --n_head 1 --min_examples 30 --constant_z 19 --o_dims 1 --diversity 3200000 --n_bins 768 --block_setup 0 --training_loss rps
python3 ${python_files_dir}/write_eval.py --family gpt2_ao --ao 1 --n_layer 8 --n_head 1 --min_examples 30 --constant_z 19 --o_dims 1 --diversity 3200000 --n_bins 1536 --block_setup 0 --training_loss rps

python3 ${python_files_dir}/write_eval.py --family gpt2_ao --ao 1 --n_layer 8 --n_head 1 --min_examples 30 --constant_z 19 --o_dims 1 --diversity 3200000 --n_bins 12 --training_loss brier
python3 ${python_files_dir}/write_eval.py --family gpt2_ao --ao 1 --n_layer 8 --n_head 1 --min_examples 30 --constant_z 19 --o_dims 1 --diversity 3200000 --n_bins 24 --training_loss brier
python3 ${python_files_dir}/write_eval.py --family gpt2_ao --ao 1 --n_layer 8 --n_head 1 --min_examples 30 --constant_z 19 --o_dims 1 --diversity 3200000 --n_bins 48 --training_loss brier
python3 ${python_files_dir}/write_eval.py --family gpt2_ao --ao 1 --n_layer 8 --n_head 1 --min_examples 30 --constant_z 19 --o_dims 1 --diversity 3200000 --n_bins 96 --training_loss brier
python3 ${python_files_dir}/write_eval.py --family gpt2_ao --ao 1 --n_layer 8 --n_head 1 --min_examples 30 --constant_z 19 --o_dims 1 --diversity 3200000 --n_bins 192 --training_loss brier
python3 ${python_files_dir}/write_eval.py --family gpt2_ao --ao 1 --n_layer 8 --n_head 1 --min_examples 30 --constant_z 19 --o_dims 1 --diversity 3200000 --n_bins 384 --training_loss brier
python3 ${python_files_dir}/write_eval.py --family gpt2_ao --ao 1 --n_layer 8 --n_head 1 --min_examples 30 --constant_z 19 --o_dims 1 --diversity 3200000 --n_bins 768 --training_loss brier
python3 ${python_files_dir}/write_eval.py --family gpt2_ao --ao 1 --n_layer 8 --n_head 1 --min_examples 30 --constant_z 19 --o_dims 1 --diversity 3200000 --n_bins 1536 --training_loss brier
python3 ${python_files_dir}/write_eval.py --family gpt2_ao --ao 1 --n_layer 8 --n_head 1 --min_examples 30 --constant_z 19 --o_dims 1 --diversity 3200000 --n_bins 12 --block_setup 0 --training_loss brier
python3 ${python_files_dir}/write_eval.py --family gpt2_ao --ao 1 --n_layer 8 --n_head 1 --min_examples 30 --constant_z 19 --o_dims 1 --diversity 3200000 --n_bins 24 --block_setup 0 --training_loss brier
python3 ${python_files_dir}/write_eval.py --family gpt2_ao --ao 1 --n_layer 8 --n_head 1 --min_examples 30 --constant_z 19 --o_dims 1 --diversity 3200000 --n_bins 48 --block_setup 0 --training_loss brier
python3 ${python_files_dir}/write_eval.py --family gpt2_ao --ao 1 --n_layer 8 --n_head 1 --min_examples 30 --constant_z 19 --o_dims 1 --diversity 3200000 --n_bins 96 --block_setup 0 --training_loss brier
python3 ${python_files_dir}/write_eval.py --family gpt2_ao --ao 1 --n_layer 8 --n_head 1 --min_examples 30 --constant_z 19 --o_dims 1 --diversity 3200000 --n_bins 192 --block_setup 0 --training_loss brier
python3 ${python_files_dir}/write_eval.py --family gpt2_ao --ao 1 --n_layer 8 --n_head 1 --min_examples 30 --constant_z 19 --o_dims 1 --diversity 3200000 --n_bins 384 --block_setup 0 --training_loss brier
python3 ${python_files_dir}/write_eval.py --family gpt2_ao --ao 1 --n_layer 8 --n_head 1 --min_examples 30 --constant_z 19 --o_dims 1 --diversity 3200000 --n_bins 768 --block_setup 0 --training_loss brier
python3 ${python_files_dir}/write_eval.py --family gpt2_ao --ao 1 --n_layer 8 --n_head 1 --min_examples 30 --constant_z 19 --o_dims 1 --diversity 3200000 --n_bins 1536 --block_setup 0 --training_loss brier

python3 ${python_files_dir}/write_eval.py --family gpt2_ao --ao 1 --n_layer 8 --n_head 1 --min_examples 30 --constant_z 19 --o_dims 1 --diversity 3200000 --n_bins 12 --training_loss cross_entropy
python3 ${python_files_dir}/write_eval.py --family gpt2_ao --ao 1 --n_layer 8 --n_head 1 --min_examples 30 --constant_z 19 --o_dims 1 --diversity 3200000 --n_bins 24 --training_loss cross_entropy
python3 ${python_files_dir}/write_eval.py --family gpt2_ao --ao 1 --n_layer 8 --n_head 1 --min_examples 30 --constant_z 19 --o_dims 1 --diversity 3200000 --n_bins 48 --training_loss cross_entropy
python3 ${python_files_dir}/write_eval.py --family gpt2_ao --ao 1 --n_layer 8 --n_head 1 --min_examples 30 --constant_z 19 --o_dims 1 --diversity 3200000 --n_bins 96 --training_loss cross_entropy
python3 ${python_files_dir}/write_eval.py --family gpt2_ao --ao 1 --n_layer 8 --n_head 1 --min_examples 30 --constant_z 19 --o_dims 1 --diversity 3200000 --n_bins 192 --training_loss cross_entropy
python3 ${python_files_dir}/write_eval.py --family gpt2_ao --ao 1 --n_layer 8 --n_head 1 --min_examples 30 --constant_z 19 --o_dims 1 --diversity 3200000 --n_bins 384 --training_loss cross_entropy
python3 ${python_files_dir}/write_eval.py --family gpt2_ao --ao 1 --n_layer 8 --n_head 1 --min_examples 30 --constant_z 19 --o_dims 1 --diversity 3200000 --n_bins 768 --training_loss cross_entropy
python3 ${python_files_dir}/write_eval.py --family gpt2_ao --ao 1 --n_layer 8 --n_head 1 --min_examples 30 --constant_z 19 --o_dims 1 --diversity 3200000 --n_bins 1536 --training_loss cross_entropy
python3 ${python_files_dir}/write_eval.py --family gpt2_ao --ao 1 --n_layer 8 --n_head 1 --min_examples 30 --constant_z 19 --o_dims 1 --diversity 3200000 --n_bins 12 --block_setup 0 --training_loss cross_entropy
python3 ${python_files_dir}/write_eval.py --family gpt2_ao --ao 1 --n_layer 8 --n_head 1 --min_examples 30 --constant_z 19 --o_dims 1 --diversity 3200000 --n_bins 24 --block_setup 0 --training_loss cross_entropy
python3 ${python_files_dir}/write_eval.py --family gpt2_ao --ao 1 --n_layer 8 --n_head 1 --min_examples 30 --constant_z 19 --o_dims 1 --diversity 3200000 --n_bins 48 --block_setup 0 --training_loss cross_entropy
python3 ${python_files_dir}/write_eval.py --family gpt2_ao --ao 1 --n_layer 8 --n_head 1 --min_examples 30 --constant_z 19 --o_dims 1 --diversity 3200000 --n_bins 96 --block_setup 0 --training_loss cross_entropy
python3 ${python_files_dir}/write_eval.py --family gpt2_ao --ao 1 --n_layer 8 --n_head 1 --min_examples 30 --constant_z 19 --o_dims 1 --diversity 3200000 --n_bins 192 --block_setup 0 --training_loss cross_entropy
python3 ${python_files_dir}/write_eval.py --family gpt2_ao --ao 1 --n_layer 8 --n_head 1 --min_examples 30 --constant_z 19 --o_dims 1 --diversity 3200000 --n_bins 384 --block_setup 0 --training_loss cross_entropy
python3 ${python_files_dir}/write_eval.py --family gpt2_ao --ao 1 --n_layer 8 --n_head 1 --min_examples 30 --constant_z 19 --o_dims 1 --diversity 3200000 --n_bins 768 --block_setup 0 --training_loss cross_entropy
python3 ${python_files_dir}/write_eval.py --family gpt2_ao --ao 1 --n_layer 8 --n_head 1 --min_examples 30 --constant_z 19 --o_dims 1 --diversity 3200000 --n_bins 1536 --block_setup 0 --training_loss cross_entropy

##################

python3 ${python_files_dir}/write_eval.py --family gpt2_ao --ao 1 --theta_dist uniform --diversity 1 --n_layer 8 --n_head 1
python3 ${python_files_dir}/write_eval.py --family gpt2_ao --ao 1 --theta_dist uniform --diversity 2 --n_layer 8 --n_head 1
python3 ${python_files_dir}/write_eval.py --family gpt2_ao --ao 1 --theta_dist uniform --diversity 5 --n_layer 8 --n_head 1
python3 ${python_files_dir}/write_eval.py --family gpt2_ao --ao 1 --theta_dist uniform --diversity 10 --n_layer 8 --n_head 1
python3 ${python_files_dir}/write_eval.py --family gpt2_ao --ao 1 --theta_dist uniform --diversity 20 --n_layer 8 --n_head 1
python3 ${python_files_dir}/write_eval.py --family gpt2_ao --ao 1 --theta_dist uniform --diversity 35 --n_layer 8 --n_head 1
python3 ${python_files_dir}/write_eval.py --family gpt2_ao --ao 1 --theta_dist uniform --diversity 50 --n_layer 8 --n_head 1
python3 ${python_files_dir}/write_eval.py --family gpt2_ao --ao 1 --theta_dist uniform --diversity 75 --n_layer 8 --n_head 1
python3 ${python_files_dir}/write_eval.py --family gpt2_ao --ao 1 --theta_dist uniform --diversity 100 --n_layer 8 --n_head 1
python3 ${python_files_dir}/write_eval.py --family gpt2_ao --ao 1 --theta_dist uniform --diversity 150 --n_layer 8 --n_head 1
python3 ${python_files_dir}/write_eval.py --family gpt2_ao --ao 1 --theta_dist uniform --diversity 200 --n_layer 8 --n_head 1
python3 ${python_files_dir}/write_eval.py --family gpt2_ao --ao 1 --theta_dist uniform --diversity 300 --n_layer 8 --n_head 1
python3 ${python_files_dir}/write_eval.py --family gpt2_ao --ao 1 --theta_dist uniform --diversity 500 --n_layer 8 --n_head 1
python3 ${python_files_dir}/write_eval.py --family gpt2_ao --ao 1 --theta_dist uniform --diversity 750 --n_layer 8 --n_head 1
python3 ${python_files_dir}/write_eval.py --family gpt2_ao --ao 1 --theta_dist uniform --diversity 1000 --n_layer 8 --n_head 1

python3 ${python_files_dir}/write_eval.py --family gpt2_ao --ao 1 --theta_dist norm --diversity 1 --n_layer 8 --n_head 1
python3 ${python_files_dir}/write_eval.py --family gpt2_ao --ao 1 --theta_dist norm --diversity 2 --n_layer 8 --n_head 1
python3 ${python_files_dir}/write_eval.py --family gpt2_ao --ao 1 --theta_dist norm --diversity 5 --n_layer 8 --n_head 1
python3 ${python_files_dir}/write_eval.py --family gpt2_ao --ao 1 --theta_dist norm --diversity 10 --n_layer 8 --n_head 1
python3 ${python_files_dir}/write_eval.py --family gpt2_ao --ao 1 --theta_dist norm --diversity 20 --n_layer 8 --n_head 1
python3 ${python_files_dir}/write_eval.py --family gpt2_ao --ao 1 --theta_dist norm --diversity 35 --n_layer 8 --n_head 1
python3 ${python_files_dir}/write_eval.py --family gpt2_ao --ao 1 --theta_dist norm --diversity 50 --n_layer 8 --n_head 1
python3 ${python_files_dir}/write_eval.py --family gpt2_ao --ao 1 --theta_dist norm --diversity 75 --n_layer 8 --n_head 1
python3 ${python_files_dir}/write_eval.py --family gpt2_ao --ao 1 --theta_dist norm --diversity 100 --n_layer 8 --n_head 1
python3 ${python_files_dir}/write_eval.py --family gpt2_ao --ao 1 --theta_dist norm --diversity 150 --n_layer 8 --n_head 1
python3 ${python_files_dir}/write_eval.py --family gpt2_ao --ao 1 --theta_dist norm --diversity 200 --n_layer 8 --n_head 1
python3 ${python_files_dir}/write_eval.py --family gpt2_ao --ao 1 --theta_dist norm --diversity 300 --n_layer 8 --n_head 1
python3 ${python_files_dir}/write_eval.py --family gpt2_ao --ao 1 --theta_dist norm --diversity 500 --n_layer 8 --n_head 1
python3 ${python_files_dir}/write_eval.py --family gpt2_ao --ao 1 --theta_dist norm --diversity 750 --n_layer 8 --n_head 1
python3 ${python_files_dir}/write_eval.py --family gpt2_ao --ao 1 --theta_dist norm --diversity 1000 --n_layer 8 --n_head 1

python3 ${python_files_dir}/write_eval.py --family gpt2_ao --ao 1 --theta_dist uniform --diversity 1 --n_layer 8 --n_head 1 --eval_theta_dist norm
python3 ${python_files_dir}/write_eval.py --family gpt2_ao --ao 1 --theta_dist uniform --diversity 2 --n_layer 8 --n_head 1 --eval_theta_dist norm
python3 ${python_files_dir}/write_eval.py --family gpt2_ao --ao 1 --theta_dist uniform --diversity 5 --n_layer 8 --n_head 1 --eval_theta_dist norm
python3 ${python_files_dir}/write_eval.py --family gpt2_ao --ao 1 --theta_dist uniform --diversity 10 --n_layer 8 --n_head 1 --eval_theta_dist norm
python3 ${python_files_dir}/write_eval.py --family gpt2_ao --ao 1 --theta_dist uniform --diversity 20 --n_layer 8 --n_head 1 --eval_theta_dist norm
python3 ${python_files_dir}/write_eval.py --family gpt2_ao --ao 1 --theta_dist uniform --diversity 35 --n_layer 8 --n_head 1 --eval_theta_dist norm
python3 ${python_files_dir}/write_eval.py --family gpt2_ao --ao 1 --theta_dist uniform --diversity 50 --n_layer 8 --n_head 1 --eval_theta_dist norm
python3 ${python_files_dir}/write_eval.py --family gpt2_ao --ao 1 --theta_dist uniform --diversity 75 --n_layer 8 --n_head 1 --eval_theta_dist norm
python3 ${python_files_dir}/write_eval.py --family gpt2_ao --ao 1 --theta_dist uniform --diversity 100 --n_layer 8 --n_head 1 --eval_theta_dist norm
python3 ${python_files_dir}/write_eval.py --family gpt2_ao --ao 1 --theta_dist uniform --diversity 150 --n_layer 8 --n_head 1 --eval_theta_dist norm
python3 ${python_files_dir}/write_eval.py --family gpt2_ao --ao 1 --theta_dist uniform --diversity 200 --n_layer 8 --n_head 1 --eval_theta_dist norm
python3 ${python_files_dir}/write_eval.py --family gpt2_ao --ao 1 --theta_dist uniform --diversity 300 --n_layer 8 --n_head 1 --eval_theta_dist norm
python3 ${python_files_dir}/write_eval.py --family gpt2_ao --ao 1 --theta_dist uniform --diversity 500 --n_layer 8 --n_head 1 --eval_theta_dist norm
python3 ${python_files_dir}/write_eval.py --family gpt2_ao --ao 1 --theta_dist uniform --diversity 750 --n_layer 8 --n_head 1 --eval_theta_dist norm
python3 ${python_files_dir}/write_eval.py --family gpt2_ao --ao 1 --theta_dist uniform --diversity 1000 --n_layer 8 --n_head 1 --eval_theta_dist norm

python3 ${python_files_dir}/write_eval.py --family gpt2_ao --ao 1 --theta_dist norm --diversity 1 --n_layer 8 --n_head 1 --eval_theta_dist norm
python3 ${python_files_dir}/write_eval.py --family gpt2_ao --ao 1 --theta_dist norm --diversity 2 --n_layer 8 --n_head 1 --eval_theta_dist norm
python3 ${python_files_dir}/write_eval.py --family gpt2_ao --ao 1 --theta_dist norm --diversity 5 --n_layer 8 --n_head 1 --eval_theta_dist norm
python3 ${python_files_dir}/write_eval.py --family gpt2_ao --ao 1 --theta_dist norm --diversity 10 --n_layer 8 --n_head 1 --eval_theta_dist norm
python3 ${python_files_dir}/write_eval.py --family gpt2_ao --ao 1 --theta_dist norm --diversity 20 --n_layer 8 --n_head 1 --eval_theta_dist norm
python3 ${python_files_dir}/write_eval.py --family gpt2_ao --ao 1 --theta_dist norm --diversity 35 --n_layer 8 --n_head 1 --eval_theta_dist norm
python3 ${python_files_dir}/write_eval.py --family gpt2_ao --ao 1 --theta_dist norm --diversity 50 --n_layer 8 --n_head 1 --eval_theta_dist norm
python3 ${python_files_dir}/write_eval.py --family gpt2_ao --ao 1 --theta_dist norm --diversity 75 --n_layer 8 --n_head 1 --eval_theta_dist norm
python3 ${python_files_dir}/write_eval.py --family gpt2_ao --ao 1 --theta_dist norm --diversity 100 --n_layer 8 --n_head 1 --eval_theta_dist norm
python3 ${python_files_dir}/write_eval.py --family gpt2_ao --ao 1 --theta_dist norm --diversity 150 --n_layer 8 --n_head 1 --eval_theta_dist norm
python3 ${python_files_dir}/write_eval.py --family gpt2_ao --ao 1 --theta_dist norm --diversity 200 --n_layer 8 --n_head 1 --eval_theta_dist norm
python3 ${python_files_dir}/write_eval.py --family gpt2_ao --ao 1 --theta_dist norm --diversity 300 --n_layer 8 --n_head 1 --eval_theta_dist norm
python3 ${python_files_dir}/write_eval.py --family gpt2_ao --ao 1 --theta_dist norm --diversity 500 --n_layer 8 --n_head 1 --eval_theta_dist norm
python3 ${python_files_dir}/write_eval.py --family gpt2_ao --ao 1 --theta_dist norm --diversity 750 --n_layer 8 --n_head 1 --eval_theta_dist norm
python3 ${python_files_dir}/write_eval.py --family gpt2_ao --ao 1 --theta_dist norm --diversity 1000 --n_layer 8 --n_head 1 --eval_theta_dist norm

################## write_mse

python3 ${python_files_dir}/write_eval.py --family gpt2_ao --ao 1 --o_dims 1 --diversity 3200000 --min_examples 30 --constant_z 14 --n_layer 8 --n_head 1
python3 ${python_files_dir}/write_eval.py --family gpt2_ao --ao 1 --o_dims 1 --diversity 3200000 --min_examples 30 --constant_z 14 --n_layer 4 --n_head 1
python3 ${python_files_dir}/write_eval.py --family gpt2_ao --ao 1 --o_dims 1 --diversity 3200000 --min_examples 30 --constant_z 14 --n_layer 2 --n_head 1
python3 ${python_files_dir}/write_eval.py --family gpt2_ao --ao 1 --o_dims 1 --diversity 3200000 --min_examples 30 --constant_z 14 --n_layer 1 --n_head 1

################## write_mse

python3 ${python_files_dir}/write_eval.py --family gpt2_ao --ao 1 --o_dims 1 --n_embd 8 --diversity 3200000 --min_examples 30 --constant_z 14 --n_layer 8 --n_head 1
python3 ${python_files_dir}/write_eval.py --family gpt2_ao --ao 1 --o_dims 1 --n_embd 16 --diversity 3200000 --min_examples 30 --constant_z 14 --n_layer 8 --n_head 1
python3 ${python_files_dir}/write_eval.py --family gpt2_ao --ao 1 --o_dims 1 --n_embd 32 --diversity 3200000 --min_examples 30 --constant_z 14 --n_layer 8 --n_head 1

python3 ${python_files_dir}/write_eval.py --family gpt2_ao --ao 1 --o_dims 1 --n_embd 8 --diversity 3200000 --min_examples 30 --constant_z 14 --n_layer 4
python3 ${python_files_dir}/write_eval.py --family gpt2_ao --ao 1 --o_dims 1 --n_embd 8 --diversity 3200000 --min_examples 30 --constant_z 14 --n_layer 2
python3 ${python_files_dir}/write_eval.py --family gpt2_ao --ao 1 --o_dims 1 --n_embd 8 --diversity 3200000 --min_examples 30 --constant_z 14 --n_layer 1
python3 ${python_files_dir}/write_eval.py --family gpt2_ao --ao 1 --o_dims 1 --n_embd 16 --diversity 3200000 --min_examples 30 --constant_z 14 --n_layer 4
python3 ${python_files_dir}/write_eval.py --family gpt2_ao --ao 1 --o_dims 1 --n_embd 16 --diversity 3200000 --min_examples 30 --constant_z 14 --n_layer 2
python3 ${python_files_dir}/write_eval.py --family gpt2_ao --ao 1 --o_dims 1 --n_embd 16 --diversity 3200000 --min_examples 30 --constant_z 14 --n_layer 1

################## write_mse

python3 ${python_files_dir}/write_eval.py --family gpt2_ao --ao 1 --theta_dist norm --diversity 3 --n_layer 8 --n_head 1
python3 ${python_files_dir}/write_eval.py --family gpt2_ao --ao 1 --theta_dist norm --diversity 4 --n_layer 8 --n_head 1
python3 ${python_files_dir}/write_eval.py --family gpt2_ao --ao 1 --theta_dist norm --diversity 6 --n_layer 8 --n_head 1
python3 ${python_files_dir}/write_eval.py --family gpt2_ao --ao 1 --theta_dist norm --diversity 7 --n_layer 8 --n_head 1
python3 ${python_files_dir}/write_eval.py --family gpt2_ao --ao 1 --theta_dist norm --diversity 8 --n_layer 8 --n_head 1
python3 ${python_files_dir}/write_eval.py --family gpt2_ao --ao 1 --theta_dist norm --diversity 9 --n_layer 8 --n_head 1

python3 ${python_files_dir}/write_eval.py --family gpt2_ao --ao 1 --theta_dist uniform --diversity 3 --n_layer 8 --n_head 1
python3 ${python_files_dir}/write_eval.py --family gpt2_ao --ao 1 --theta_dist uniform --diversity 4 --n_layer 8 --n_head 1
python3 ${python_files_dir}/write_eval.py --family gpt2_ao --ao 1 --theta_dist uniform --diversity 6 --n_layer 8 --n_head 1
python3 ${python_files_dir}/write_eval.py --family gpt2_ao --ao 1 --theta_dist uniform --diversity 7 --n_layer 8 --n_head 1
python3 ${python_files_dir}/write_eval.py --family gpt2_ao --ao 1 --theta_dist uniform --diversity 8 --n_layer 8 --n_head 1
python3 ${python_files_dir}/write_eval.py --family gpt2_ao --ao 1 --theta_dist uniform --diversity 9 --n_layer 8 --n_head 1

python3 ${python_files_dir}/write_eval.py --family gpt2_ao --ao 1 --theta_dist norm --diversity 3 --n_layer 8 --n_head 1 --eval_theta_dist norm
python3 ${python_files_dir}/write_eval.py --family gpt2_ao --ao 1 --theta_dist norm --diversity 4 --n_layer 8 --n_head 1 --eval_theta_dist norm
python3 ${python_files_dir}/write_eval.py --family gpt2_ao --ao 1 --theta_dist norm --diversity 6 --n_layer 8 --n_head 1 --eval_theta_dist norm
python3 ${python_files_dir}/write_eval.py --family gpt2_ao --ao 1 --theta_dist norm --diversity 7 --n_layer 8 --n_head 1 --eval_theta_dist norm
python3 ${python_files_dir}/write_eval.py --family gpt2_ao --ao 1 --theta_dist norm --diversity 8 --n_layer 8 --n_head 1 --eval_theta_dist norm
python3 ${python_files_dir}/write_eval.py --family gpt2_ao --ao 1 --theta_dist norm --diversity 9 --n_layer 8 --n_head 1 --eval_theta_dist norm

python3 ${python_files_dir}/write_eval.py --family gpt2_ao --ao 1 --theta_dist uniform --diversity 3 --n_layer 8 --n_head 1 --eval_theta_dist norm
python3 ${python_files_dir}/write_eval.py --family gpt2_ao --ao 1 --theta_dist uniform --diversity 4 --n_layer 8 --n_head 1 --eval_theta_dist norm
python3 ${python_files_dir}/write_eval.py --family gpt2_ao --ao 1 --theta_dist uniform --diversity 6 --n_layer 8 --n_head 1 --eval_theta_dist norm
python3 ${python_files_dir}/write_eval.py --family gpt2_ao --ao 1 --theta_dist uniform --diversity 7 --n_layer 8 --n_head 1 --eval_theta_dist norm
python3 ${python_files_dir}/write_eval.py --family gpt2_ao --ao 1 --theta_dist uniform --diversity 8 --n_layer 8 --n_head 1 --eval_theta_dist norm
python3 ${python_files_dir}/write_eval.py --family gpt2_ao --ao 1 --theta_dist uniform --diversity 9 --n_layer 8 --n_head 1 --eval_theta_dist norm

################## write_mse_sde

python3 ${python_files_dir}/write_eval.py --data sde --family gpt2_sde --batch_size 8 --n_thetas 8 --diversity 2000000 --poisson 0 --lamb 40 --max_time 0.5 --dim_index 2 --training_loss masked_mse
python3 ${python_files_dir}/write_eval.py --data sde --poisson 0 --family gpt2_ao_sde --ao 1 --batch_size 8 --n_thetas 8 --diversity 2000000 --dim_index 2 --training_loss masked_mse
python3 ${python_files_dir}/write_eval.py --data sde --poisson 0 --family gpt2_ao_sde --ao 1 --n_layer 8 --n_head 1 --batch_size 8 --n_thetas 8 --diversity 2000000 --dim_index 2 --training_loss masked_mse
python3 ${python_files_dir}/write_eval.py --data sde --poisson 0 --family gpt2_ao_sde --ao 1 --n_layer 4 --n_head 1 --batch_size 8 --n_thetas 8 --diversity 2000000 --dim_index 2 --training_loss masked_mse
python3 ${python_files_dir}/write_eval.py --data sde --poisson 0 --family gpt2_ao_sde --ao 1 --n_layer 2 --n_head 1 --batch_size 8 --n_thetas 8 --diversity 2000000 --dim_index 2 --training_loss masked_mse
python3 ${python_files_dir}/write_eval.py --data sde --poisson 0 --family gpt2_ao_sde --ao 1 --n_layer 1 --n_head 1 --batch_size 8 --n_thetas 8 --diversity 2000000 --dim_index 2 --training_loss masked_mse
python3 ${python_files_dir}/write_eval.py --data sde --poisson 0 --family gpt2_sde --n_layer 8 --n_head 1 --batch_size 8 --n_thetas 8 --diversity 2000000 --dim_index 2 --training_loss masked_mse
python3 ${python_files_dir}/write_eval.py --data sde --poisson 0 --family gpt2_sde --n_layer 4 --n_head 1 --batch_size 8 --n_thetas 8 --diversity 2000000 --dim_index 2 --training_loss masked_mse
python3 ${python_files_dir}/write_eval.py --data sde --poisson 0 --family gpt2_sde --n_layer 2 --n_head 1 --batch_size 8 --n_thetas 8 --diversity 2000000 --dim_index 2 --training_loss masked_mse
python3 ${python_files_dir}/write_eval.py --data sde --poisson 0 --family gpt2_sde --n_layer 1 --n_head 1 --batch_size 8 --n_thetas 8 --diversity 2000000 --dim_index 2 --training_loss masked_mse
python3 ${python_files_dir}/write_eval.py --data sde --poisson 0 --family lstm_sde --batch_size 8 --n_thetas 8 --diversity 2000000 --n_layer 2 --dim_index 2 --training_loss masked_mse
python3 ${python_files_dir}/write_eval.py --data sde --poisson 0 --family rnn_sde --batch_size 8 --n_thetas 8 --diversity 2000000 --n_layer 2 --dim_index 2 --training_loss masked_mse
python3 ${python_files_dir}/write_eval.py --data sde --poisson 0 --family gru_sde --batch_size 8 --n_thetas 8 --diversity 2000000 --n_layer 2 --dim_index 2 --training_loss masked_mse

python3 ${python_files_dir}/write_eval.py --data sde --poisson 0 --family gpt2_sde --ode 1 --batch_size 8 --n_thetas 8 --diversity 2000000 --dim_index 2 --training_loss masked_mse
python3 ${python_files_dir}/write_eval.py --data sde --poisson 0 --family gpt2_ao_sde --ao 1 --batch_size 8 --n_thetas 8 --diversity 2000000  --ode 1 --dim_index 2 --training_loss masked_mse
python3 ${python_files_dir}/write_eval.py --data sde --poisson 0 --family gpt2_ao_sde --ao 1 --n_layer 8 --n_head 1 --batch_size 8 --n_thetas 8 --diversity 2000000 --ode 1 --dim_index 2 --training_loss masked_mse
python3 ${python_files_dir}/write_eval.py --data sde --poisson 0 --family gpt2_ao_sde --ao 1 --n_layer 4 --n_head 1 --batch_size 8 --n_thetas 8 --diversity 2000000 --ode 1 --dim_index 2 --training_loss masked_mse
python3 ${python_files_dir}/write_eval.py --data sde --poisson 0 --family gpt2_ao_sde --ao 1 --n_layer 2 --n_head 1 --batch_size 8 --n_thetas 8 --diversity 2000000 --ode 1 --dim_index 2 --training_loss masked_mse
python3 ${python_files_dir}/write_eval.py --data sde --poisson 0 --family gpt2_ao_sde --ao 1 --n_layer 1 --n_head 1 --batch_size 8 --n_thetas 8 --diversity 2000000 --ode 1 --dim_index 2 --training_loss masked_mse
python3 ${python_files_dir}/write_eval.py --data sde --poisson 0 --family gpt2_sde --n_layer 8 --n_head 1 --batch_size 8 --n_thetas 8 --diversity 2000000 --ode 1 --dim_index 2 --training_loss masked_mse
python3 ${python_files_dir}/write_eval.py --data sde --poisson 0 --family gpt2_sde --n_layer 4 --n_head 1 --batch_size 8 --n_thetas 8 --diversity 2000000 --ode 1 --dim_index 2 --training_loss masked_mse
python3 ${python_files_dir}/write_eval.py --data sde --poisson 0 --family gpt2_sde --n_layer 2 --n_head 1 --batch_size 8 --n_thetas 8 --diversity 2000000 --ode 1 --dim_index 2 --training_loss masked_mse
python3 ${python_files_dir}/write_eval.py --data sde --poisson 0 --family gpt2_sde --n_layer 1 --n_head 1 --batch_size 8 --n_thetas 8 --diversity 2000000 --ode 1 --dim_index 2 --training_loss masked_mse
python3 ${python_files_dir}/write_eval.py --data sde --poisson 0 --family lstm_sde --batch_size 8 --n_thetas 8 --diversity 2000000 --ode 1 --n_layer 2 --dim_index 2 --training_loss masked_mse
python3 ${python_files_dir}/write_eval.py --data sde --poisson 0 --family rnn_sde --batch_size 8 --n_thetas 8 --diversity 2000000 --ode 1 --n_layer 2 --dim_index 2 --training_loss masked_mse
python3 ${python_files_dir}/write_eval.py --data sde --poisson 0 --family gru_sde --batch_size 8 --n_thetas 8 --diversity 2000000 --ode 1 --n_layer 2 --dim_index 2 --training_loss masked_mse

python3 ${python_files_dir}/write_eval.py --data sde --poisson 0 --family gpt2_ao_sde --ao 1 --lamb 20 --max_time 1 --n_layer 8 --n_head 1 --batch_size 8 --n_thetas 8 --diversity 2000000 --dim_index 4 --training_loss masked_mse
python3 ${python_files_dir}/write_eval.py --data sde --poisson 0 --family gpt2_ao_sde --ao 1 --lamb 10 --max_time 2 --n_layer 8 --n_head 1 --batch_size 8 --n_thetas 8 --diversity 2000000  --training_loss masked_mse
python3 ${python_files_dir}/write_eval.py --data sde --poisson 0 --family gpt2_ao_sde --ao 1 --lamb 20 --max_time 1 --n_layer 8 --n_head 1 --o_dims 1 --diversity 400000 --batch_size 8 --n_thetas 8 --training_loss masked_mse

################## write_mse_sde

python3 ${python_files_dir}/write_eval.py --data sde --family gpt2_sde --batch_size 8 --n_thetas 8 --diversity 2000000 --lamb 40 --max_time 0.5 --training_loss masked_mse
python3 ${python_files_dir}/write_eval.py --data sde --family gpt2_ao_sde --ao 1 --batch_size 8 --n_thetas 8 --diversity 2000000 --training_loss masked_mse
python3 ${python_files_dir}/write_eval.py --data sde --family gpt2_ao_sde --ao 1 --n_layer 8 --n_head 1 --batch_size 8 --n_thetas 8 --diversity 2000000 --training_loss masked_mse
python3 ${python_files_dir}/write_eval.py --data sde --family gpt2_ao_sde --ao 1 --n_layer 4 --n_head 1 --batch_size 8 --n_thetas 8 --diversity 2000000 --training_loss masked_mse
python3 ${python_files_dir}/write_eval.py --data sde --family gpt2_ao_sde --ao 1 --n_layer 2 --n_head 1 --batch_size 8 --n_thetas 8 --diversity 2000000 --training_loss masked_mse
python3 ${python_files_dir}/write_eval.py --data sde --family gpt2_ao_sde --ao 1 --n_layer 1 --n_head 1 --batch_size 8 --n_thetas 8 --diversity 2000000 --training_loss masked_mse
python3 ${python_files_dir}/write_eval.py --data sde --family gpt2_sde --n_layer 8 --n_head 1 --batch_size 8 --n_thetas 8 --diversity 2000000 --training_loss masked_mse
python3 ${python_files_dir}/write_eval.py --data sde --family gpt2_sde --n_layer 4 --n_head 1 --batch_size 8 --n_thetas 8 --diversity 2000000 --training_loss masked_mse
python3 ${python_files_dir}/write_eval.py --data sde --family gpt2_sde --n_layer 2 --n_head 1 --batch_size 8 --n_thetas 8 --diversity 2000000 --training_loss masked_mse
python3 ${python_files_dir}/write_eval.py --data sde --family gpt2_sde --n_layer 1 --n_head 1 --batch_size 8 --n_thetas 8 --diversity 2000000 --training_loss masked_mse
python3 ${python_files_dir}/write_eval.py --data sde --family lstm_sde --batch_size 8 --n_thetas 8 --diversity 2000000 --n_layer 2 --training_loss masked_mse
python3 ${python_files_dir}/write_eval.py --data sde --family rnn_sde --batch_size 8 --n_thetas 8 --diversity 2000000 --n_layer 2 --training_loss masked_mse
python3 ${python_files_dir}/write_eval.py --data sde --family gru_sde --batch_size 8 --n_thetas 8 --diversity 2000000 --n_layer 2 --training_loss masked_mse

python3 ${python_files_dir}/write_eval.py --data sde --family gpt2_ao_sde --ao 1 --lamb 20 --max_time 1 --n_layer 8 --n_head 1 --batch_size 8 --n_thetas 8 --diversity 2000000 --training_loss masked_mse
python3 ${python_files_dir}/write_eval.py --data sde --family gpt2_ao_sde --ao 1 --lamb 10 --max_time 2 --n_layer 8 --n_head 1 --batch_size 8 --n_thetas 8 --diversity 2000000 --training_loss masked_mse
python3 ${python_files_dir}/write_eval.py --data sde --family gpt2_ao_sde --ao 1 --lamb 20 --max_time 1 --n_layer 8 --n_head 1 --o_dims 1 --diversity 400000 --batch_size 8 --n_thetas 8 --training_loss masked_mse

################## write_mse / write_mse_sde

python3 ${python_files_dir}/write_eval.py --o_dims 1 --diversity 3200000 --min_examples 30 --constant_z 14
python3 ${python_files_dir}/write_eval.py --o_dims 2 --diversity 6400000 --min_examples 30 --constant_z 14
python3 ${python_files_dir}/write_eval.py --o_dims 3 --diversity 9600000 --min_examples 30 --constant_z 14
python3 ${python_files_dir}/write_eval.py --o_dims 4 --diversity 12800000 --min_examples 30 --constant_z 14
python3 ${python_files_dir}/write_eval.py --o_dims 1 --diversity 3200000 --min_examples 30

python3 ${python_files_dir}/write_eval.py --data sde --batch_size 8 --n_thetas 8 --diversity 2000000 --poisson 0 --training_loss masked_mse --family gpt2_ao_sde --n_layer 8 --n_head 1 --lamb 100 --max_time 0.2 --ao 1
python3 ${python_files_dir}/write_eval.py --data sde --batch_size 8 --n_thetas 8 --diversity 2000000 --poisson 0 --training_loss masked_mse --family gpt2_ao_sde --n_layer 8 --n_head 1 --lamb 200 --max_time 0.1 --ao 1
python3 ${python_files_dir}/write_eval.py --data sde --batch_size 8 --n_thetas 8 --diversity 2000000 --poisson 1 --training_loss masked_mse --family gpt2_ao_sde --n_layer 8 --n_head 1 --lamb 100 --max_time 0.2 --ao 1
python3 ${python_files_dir}/write_eval.py --data sde --batch_size 8 --n_thetas 8 --diversity 2000000 --poisson 1 --training_loss masked_mse --family gpt2_ao_sde --n_layer 8 --n_head 1 --lamb 200 --max_time 0.1 --ao 1
python3 ${python_files_dir}/write_eval.py --data sde --batch_size 8 --n_thetas 8 --diversity 2000000 --poisson 0 --training_loss masked_mse --family gpt2_sde --lamb 100 --max_time 0.2 --ao 1
python3 ${python_files_dir}/write_eval.py --data sde --batch_size 8 --n_thetas 8 --diversity 2000000 --poisson 0 --training_loss masked_mse --family gpt2_sde --lamb 200 --max_time 0.1 --ao 1
python3 ${python_files_dir}/write_eval.py --data sde --batch_size 8 --n_thetas 8 --diversity 2000000 --poisson 1 --training_loss masked_mse --family gpt2_sde --lamb 100 --max_time 0.2 --ao 1
python3 ${python_files_dir}/write_eval.py --data sde --batch_size 8 --n_thetas 8 --diversity 2000000 --poisson 1 --training_loss masked_mse --family gpt2_sde --lamb 200 --max_time 0.1 --ao 1

python3 ${python_files_dir}/write_eval.py --train_steps 1000000 --o_dims 1 --diversity 64000000
python3 ${python_files_dir}/write_eval.py --train_steps 1000000 --diversity 320000000

echo "python script executed"

conda deactivate

echo "experiment completed"