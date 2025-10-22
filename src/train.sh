#!/bin/bash

#SBATCH --mem-per-cpu=20000
#SBATCH -n 4
#SBATCH --gpus=rtx_3090:1
#SBATCH --time=2000
#SBATCH --job-name=96b
#SBATCH --output=output_files/file.out

python_files_dir="$HOME/project/src"
conda_env="$HOME/miniconda3/envs/icl"   # conda_env="/path/to/your/conda/env"
echo "paths set: $python_files_dir and $conda_env"

source activate $conda_env
echo "conda environment activated"

module load cuda/12.2.1 || echo "CUDA not available, skipping module load"
module load gcc/13.2.0 || echo "GCC not available, skipping module load"
echo "modules loaded"

wandb online

echo "python execution started"

##################

python3 ${python_files_dir}/train.py --config conf/counterfactual.yaml --wandb.name "GPT2"
python3 ${python_files_dir}/train.py --config conf/counterfactual.yaml --model.family gpt2_ao --wandb.name "GPT2AO"

python3 ${python_files_dir}/train.py --config conf/mlponly.yaml
python3 ${python_files_dir}/train.py --config conf/one_head.yaml --model.n_layer 8 --wandb.name "8l1h"
python3 ${python_files_dir}/train.py --config conf/one_head.yaml --model.n_layer 4 --wandb.name "4l1h"
python3 ${python_files_dir}/train.py --config conf/one_head.yaml --model.n_layer 2 --wandb.name "2l1h"
python3 ${python_files_dir}/train.py --config conf/attentiononly.yaml --model.n_layer 8 --wandb.name "8l1hAO"
python3 ${python_files_dir}/train.py --config conf/attentiononly.yaml --model.n_layer 4 --wandb.name "4l1hAO"
python3 ${python_files_dir}/train.py --config conf/attentiononly.yaml --model.n_layer 2 --wandb.name "2l1hAO"

python3 ${python_files_dir}/train.py --config conf/counterfactual.yaml --model.n_layer 4 --model.n_head 2 --wandb.name "4l2h"
python3 ${python_files_dir}/train.py --config conf/counterfactual.yaml --model.n_layer 2 --model.n_head 4 --wandb.name "2l4h"
python3 ${python_files_dir}/train.py --config conf/counterfactual.yaml --model.n_layer 1 --model.n_head 8 --wandb.name "1l8h"
python3 ${python_files_dir}/train.py --config conf/attentiononly.yaml --model.n_layer 4 --model.n_head 2 --wandb.name "4l2hAO"
python3 ${python_files_dir}/train.py --config conf/attentiononly.yaml --model.n_layer 2 --model.n_head 4 --wandb.name "2l4hAO"
python3 ${python_files_dir}/train.py --config conf/attentiononly.yaml --model.n_layer 1 --model.n_head 8 --wandb.name "1l8hAO"

python3 ${python_files_dir}/train.py --config conf/attentiononly.yaml --training.transformation mullin --wandb.name "8l1hAO_mullin"
python3 ${python_files_dir}/train.py --config conf/attentiononly.yaml --training.transformation tanh --wandb.name "8l1hAO_tanh"
python3 ${python_files_dir}/train.py --config conf/attentiononly.yaml --training.transformation sigmoid --wandb.name "8l1hAO_sigmoid"
python3 ${python_files_dir}/train.py --config conf/one_head.yaml --training.transformation mullin --wandb.name "8l1h_mullin"
python3 ${python_files_dir}/train.py --config conf/one_head.yaml --training.transformation tanh --wandb.name "8l1h_tanh"
python3 ${python_files_dir}/train.py --config conf/one_head.yaml --training.transformation sigmoid --wandb.name "8l1h_sigmoid"

python3 ${python_files_dir}/train.py --config conf/attentiononly.yaml --training.randomize_labels 1 --wandb.name "randlabels"

python3 ${python_files_dir}/train.py --config conf/attentiononly.yaml --model.block_setup 0 --wandb.name "AO_noblock"

python3 ${python_files_dir}/train.py --config conf/lstm.yaml
python3 ${python_files_dir}/train.py --config conf/rnn.yaml
python3 ${python_files_dir}/train.py --config conf/gru.yaml

python3 ${python_files_dir}/train.py --config conf/lstm.yaml --model.n_layer 2 --wandb.name "lstm_2l"
python3 ${python_files_dir}/train.py --config conf/rnn.yaml --model.n_layer 2 --wandb.name "rnn_2l"
python3 ${python_files_dir}/train.py --config conf/gru.yaml --model.n_layer 2 --wandb.name "gru_2l"

##################

python3 ${python_files_dir}/train.py --config conf/attentiononly.yaml --training.train_steps 10000 --training.diversity 3200000 --wandb.name "AO_10k"
python3 ${python_files_dir}/train.py --config conf/one_head.yaml --training.train_steps 10000 --training.diversity 3200000 --wandb.name "OH_10k"
python3 ${python_files_dir}/train.py --config conf/attentiononly.yaml --training.train_steps 20000 --training.diversity 6400000 --wandb.name "AO_20k"
python3 ${python_files_dir}/train.py --config conf/one_head.yaml --training.train_steps 20000 --training.diversity 6400000 --wandb.name "OH_20k"
python3 ${python_files_dir}/train.py --config conf/attentiononly.yaml --training.train_steps 100000 --training.diversity 32000000 --wandb.name "AO_100k"
python3 ${python_files_dir}/train.py --config conf/one_head.yaml --training.train_steps 100000 --training.diversity 32000000 --wandb.name "OH_100k"
python3 ${python_files_dir}/train.py --config conf/attentiononly.yaml --training.train_steps 150000 --training.diversity 48000000 --wandb.name "AO_150k"
python3 ${python_files_dir}/train.py --config conf/one_head.yaml --training.train_steps 150000 --training.diversity 48000000 --wandb.name "OH_150k"
python3 ${python_files_dir}/train.py --config conf/attentiononly.yaml --training.train_steps 200000 --training.diversity 64000000 --wandb.name "AO_200k"
python3 ${python_files_dir}/train.py --config conf/one_head.yaml --training.train_steps 200000 --training.diversity 64000000 --wandb.name "OH_200k"

python3 ${python_files_dir}/train.py --config conf/attentiononly.yaml --training.curriculum.dims.start 1 --training.curriculum.dims.end 1 --model.n_dims 1 --training.diversity 3200000 --wandb.name "AO_1dim"
python3 ${python_files_dir}/train.py --config conf/attentiononly.yaml --training.curriculum.dims.start 2 --training.curriculum.dims.end 2 --model.n_dims 2 --training.diversity 6400000 --wandb.name "AO_2dim"
python3 ${python_files_dir}/train.py --config conf/attentiononly.yaml --training.curriculum.dims.start 10 --training.curriculum.dims.end 10 --model.n_dims 10 --training.diversity 32000000 --wandb.name "AO_10dim"
python3 ${python_files_dir}/train.py --config conf/attentiononly.yaml --training.curriculum.dims.start 20 --training.curriculum.dims.end 20 --model.n_dims 20 --training.diversity 64000000 --wandb.name "AO_20dim"
python3 ${python_files_dir}/train.py --config conf/attentiononly.yaml --training.curriculum.dims.start 35 --training.curriculum.dims.end 35 --model.n_dims 35 --training.diversity 112000000 --wandb.name "AO_35dim"
python3 ${python_files_dir}/train.py --config conf/attentiononly.yaml --training.curriculum.dims.start 50 --training.curriculum.dims.end 50 --model.n_dims 50 --training.diversity 160000000 --wandb.name "AO_50dim"
python3 ${python_files_dir}/train.py --config conf/attentiononly.yaml --training.curriculum.dims.start 250 --training.curriculum.dims.end 250 --model.n_dims 250 --training.diversity 800000000 --wandb.name "AO_250dim"

python3 ${python_files_dir}/train.py --config conf/attentiononly.yaml --training.curriculum.dims.start 1 --training.curriculum.dims.end 1 --model.n_dims 1 --training.diversity 3200000 --model.n_embd 64 --wandb.name "AO_1dim_64emb"
python3 ${python_files_dir}/train.py --config conf/attentiononly.yaml --training.curriculum.dims.start 2 --training.curriculum.dims.end 2 --model.n_dims 2 --training.diversity 6400000 --model.n_embd 128 --wandb.name "AO_2dim_128emb"
python3 ${python_files_dir}/train.py --config conf/attentiononly.yaml --training.curriculum.dims.start 4 --training.curriculum.dims.end 4 --model.n_dims 4 --training.diversity 12800000 --model.n_embd 256 --wandb.name "AO_4dim_256emb"

python3 ${python_files_dir}/train.py --config conf/attentiononly.yaml --training.curriculum.dims.start 1 --training.curriculum.dims.end 1 --model.n_dims 1 --training.diversity 3200000 --model.n_layer 1 --wandb.name "1l_1dim"
python3 ${python_files_dir}/train.py --config conf/attentiononly.yaml --training.curriculum.dims.start 1 --training.curriculum.dims.end 1 --model.n_dims 1 --training.diversity 3200000 --model.n_layer 2 --wandb.name "2l_1dim"
python3 ${python_files_dir}/train.py --config conf/attentiononly.yaml --training.curriculum.dims.start 1 --training.curriculum.dims.end 1 --model.n_dims 1 --training.diversity 3200000 --model.n_layer 4 --wandb.name "4l_1dim"

python3 ${python_files_dir}/train.py --config conf/attentiononly.yaml --training.curriculum.points.start 30 --wandb.name "30-50"
python3 ${python_files_dir}/train.py --config conf/attentiononly.yaml --training.curriculum.points.start 30 --training.constant_z 0 --wandb.name "constz0"
python3 ${python_files_dir}/train.py --config conf/attentiononly.yaml --training.curriculum.points.start 30 --training.constant_z 4 --wandb.name "constz4"
python3 ${python_files_dir}/train.py --config conf/attentiononly.yaml --training.curriculum.points.start 30 --training.constant_z 9 --wandb.name "constz9"
python3 ${python_files_dir}/train.py --config conf/attentiononly.yaml --training.curriculum.points.start 30 --training.constant_z 14 --wandb.name "constz14"
python3 ${python_files_dir}/train.py --config conf/attentiononly.yaml --training.curriculum.points.start 30 --training.constant_z 19 --wandb.name "constz19"
python3 ${python_files_dir}/train.py --config conf/attentiononly.yaml --training.curriculum.points.start 30 --training.constant_z 24 --wandb.name "constz24"
python3 ${python_files_dir}/train.py --config conf/attentiononly.yaml --training.curriculum.points.start 30 --training.constant_z 29 --wandb.name "constz29"

python3 ${python_files_dir}/train.py --config conf/attentiononly.yaml --training.training_loss mae --wandb.name "mae"

##################

python3 ${python_files_dir}/train.py --config conf/sweepe.yaml
python3 ${python_files_dir}/train.py --config conf/sweepe.yaml --model.n_bins 24 --wandb.name "rps_24_block0"
python3 ${python_files_dir}/train.py --config conf/sweepe.yaml --model.n_bins 48 --wandb.name "rps_48_block0"
python3 ${python_files_dir}/train.py --config conf/sweepe.yaml --model.n_bins 96 --wandb.name "rps_96_block0"
python3 ${python_files_dir}/train.py --config conf/sweepe.yaml --model.n_bins 192 --wandb.name "rps_192_block0"
python3 ${python_files_dir}/train.py --config conf/sweepe.yaml --model.n_bins 384 --wandb.name "rps_384_block0"
python3 ${python_files_dir}/train.py --config conf/sweepe.yaml --model.n_bins 768 --wandb.name "rps_768_block0"
python3 ${python_files_dir}/train.py --config conf/sweepe.yaml --model.n_bins 1536 --wandb.name "rps_1536_block0"
python3 ${python_files_dir}/train.py --config conf/sweepe.yaml --model.n_bins 12 --model.block_setup 1 --wandb.name "rps_12_block1"
python3 ${python_files_dir}/train.py --config conf/sweepe.yaml --model.n_bins 24 --model.block_setup 1 --wandb.name "rps_24_block1"
python3 ${python_files_dir}/train.py --config conf/sweepe.yaml --model.n_bins 48 --model.block_setup 1 --wandb.name "rps_48_block1"
python3 ${python_files_dir}/train.py --config conf/sweepe.yaml --model.n_bins 96 --model.block_setup 1 --wandb.name "rps_96_block1"
python3 ${python_files_dir}/train.py --config conf/sweepe.yaml --model.n_bins 192 --model.block_setup 1 --wandb.name "rps_192_block1"
python3 ${python_files_dir}/train.py --config conf/sweepe.yaml --model.n_bins 384 --model.block_setup 1 --wandb.name "rps_384_block1"
python3 ${python_files_dir}/train.py --config conf/sweepe.yaml --model.n_bins 768 --model.block_setup 1 --wandb.name "rps_768_block1"
python3 ${python_files_dir}/train.py --config conf/sweepe.yaml --model.n_bins 1536 --model.block_setup 1 --wandb.name "rps_1536_block1"

python3 ${python_files_dir}/train.py --config conf/sweepe.yaml --model.n_bins 12 --training.training_loss brier --wandb.name "brier_12_block0"
python3 ${python_files_dir}/train.py --config conf/sweepe.yaml --model.n_bins 24 --training.training_loss brier --wandb.name "brier_24_block0"
python3 ${python_files_dir}/train.py --config conf/sweepe.yaml --model.n_bins 48 --training.training_loss brier --wandb.name "brier_48_block0"
python3 ${python_files_dir}/train.py --config conf/sweepe.yaml --model.n_bins 96 --training.training_loss brier --wandb.name "brier_96_block0"
python3 ${python_files_dir}/train.py --config conf/sweepe.yaml --model.n_bins 192 --training.training_loss brier --wandb.name "brier_192_block0"
python3 ${python_files_dir}/train.py --config conf/sweepe.yaml --model.n_bins 384 --training.training_loss brier --wandb.name "brier_384_block0"
python3 ${python_files_dir}/train.py --config conf/sweepe.yaml --model.n_bins 768 --training.training_loss brier --wandb.name "brier_768_block0"
python3 ${python_files_dir}/train.py --config conf/sweepe.yaml --model.n_bins 1536 --training.training_loss brier --wandb.name "brier_1536_block0"
python3 ${python_files_dir}/train.py --config conf/sweepe.yaml --model.n_bins 12 --model.block_setup 1 --training.training_loss brier --wandb.name "brier_12_block1"
python3 ${python_files_dir}/train.py --config conf/sweepe.yaml --model.n_bins 24 --model.block_setup 1 --training.training_loss brier --wandb.name "brier_24_block1"
python3 ${python_files_dir}/train.py --config conf/sweepe.yaml --model.n_bins 48 --model.block_setup 1 --training.training_loss brier --wandb.name "brier_48_block1"
python3 ${python_files_dir}/train.py --config conf/sweepe.yaml --model.n_bins 96 --model.block_setup 1 --training.training_loss brier --wandb.name "brier_96_block1"
python3 ${python_files_dir}/train.py --config conf/sweepe.yaml --model.n_bins 192 --model.block_setup 1 --training.training_loss brier --wandb.name "brier_192_block1"
python3 ${python_files_dir}/train.py --config conf/sweepe.yaml --model.n_bins 384 --model.block_setup 1 --training.training_loss brier --wandb.name "brier_384_block1"
python3 ${python_files_dir}/train.py --config conf/sweepe.yaml --model.n_bins 768 --model.block_setup 1 --training.training_loss brier --wandb.name "brier_768_block1"
python3 ${python_files_dir}/train.py --config conf/sweepe.yaml --model.n_bins 1536 --model.block_setup 1 --training.training_loss brier --wandb.name "brier_1536_block1"

python3 ${python_files_dir}/train.py --config conf/sweepe.yaml --model.n_bins 12 --training.training_loss cross_entropy --wandb.name "cross_entropy_12_block0"
python3 ${python_files_dir}/train.py --config conf/sweepe.yaml --model.n_bins 24 --training.training_loss cross_entropy --wandb.name "cross_entropy_24_block0"
python3 ${python_files_dir}/train.py --config conf/sweepe.yaml --model.n_bins 48 --training.training_loss cross_entropy --wandb.name "cross_entropy_48_block0"
python3 ${python_files_dir}/train.py --config conf/sweepe.yaml --model.n_bins 96 --training.training_loss cross_entropy --wandb.name "cross_entropy_96_block0"
python3 ${python_files_dir}/train.py --config conf/sweepe.yaml --model.n_bins 192 --training.training_loss cross_entropy --wandb.name "cross_entropy_192_block0"
python3 ${python_files_dir}/train.py --config conf/sweepe.yaml --model.n_bins 384 --training.training_loss cross_entropy --wandb.name "cross_entropy_384_block0"
python3 ${python_files_dir}/train.py --config conf/sweepe.yaml --model.n_bins 768 --training.training_loss cross_entropy --wandb.name "cross_entropy_768_block0"
python3 ${python_files_dir}/train.py --config conf/sweepe.yaml --model.n_bins 1536 --training.training_loss cross_entropy --wandb.name "cross_entropy_1536_block0"
python3 ${python_files_dir}/train.py --config conf/sweepe.yaml --model.n_bins 12 --model.block_setup 1 --training.training_loss cross_entropy --wandb.name "cross_entropy_12_block1"
python3 ${python_files_dir}/train.py --config conf/sweepe.yaml --model.n_bins 24 --model.block_setup 1 --training.training_loss cross_entropy --wandb.name "cross_entropy_24_block1"
python3 ${python_files_dir}/train.py --config conf/sweepe.yaml --model.n_bins 48 --model.block_setup 1 --training.training_loss cross_entropy --wandb.name "cross_entropy_48_block1"
python3 ${python_files_dir}/train.py --config conf/sweepe.yaml --model.n_bins 96 --model.block_setup 1 --training.training_loss cross_entropy --wandb.name "cross_entropy_96_block1"
python3 ${python_files_dir}/train.py --config conf/sweepe.yaml --model.n_bins 192 --model.block_setup 1 --training.training_loss cross_entropy --wandb.name "cross_entropy_192_block1"
python3 ${python_files_dir}/train.py --config conf/sweepe.yaml --model.n_bins 384 --model.block_setup 1 --training.training_loss cross_entropy --wandb.name "cross_entropy_384_block1"
python3 ${python_files_dir}/train.py --config conf/sweepe.yaml --model.n_bins 768 --model.block_setup 1 --training.training_loss cross_entropy --wandb.name "cross_entropy_768_block1"
python3 ${python_files_dir}/train.py --config conf/sweepe.yaml --model.n_bins 1536 --model.block_setup 1 --training.training_loss cross_entropy --wandb.name "cross_entropy_1536_block1"

##################

python3 ${python_files_dir}/train.py --config conf/attentiononly.yaml --training.theta_dist "uniform" --training.diversity 1 --wandb.name "uniform_1"
python3 ${python_files_dir}/train.py --config conf/attentiononly.yaml --training.theta_dist "uniform" --training.diversity 2 --wandb.name "uniform_2"
python3 ${python_files_dir}/train.py --config conf/attentiononly.yaml --training.theta_dist "uniform" --training.diversity 5 --wandb.name "uniform_5"
python3 ${python_files_dir}/train.py --config conf/attentiononly.yaml --training.theta_dist "uniform" --training.diversity 10 --wandb.name "uniform_10"
python3 ${python_files_dir}/train.py --config conf/attentiononly.yaml --training.theta_dist "uniform" --training.diversity 20 --wandb.name "uniform_20"
python3 ${python_files_dir}/train.py --config conf/attentiononly.yaml --training.theta_dist "uniform" --training.diversity 35 --wandb.name "uniform_35"
python3 ${python_files_dir}/train.py --config conf/attentiononly.yaml --training.theta_dist "uniform" --training.diversity 50 --wandb.name "uniform_50"
python3 ${python_files_dir}/train.py --config conf/attentiononly.yaml --training.theta_dist "uniform" --training.diversity 75 --wandb.name "uniform_75"
python3 ${python_files_dir}/train.py --config conf/attentiononly.yaml --training.theta_dist "uniform" --training.diversity 100 --wandb.name "uniform_100"
python3 ${python_files_dir}/train.py --config conf/attentiononly.yaml --training.theta_dist "uniform" --training.diversity 150 --wandb.name "uniform_150"
python3 ${python_files_dir}/train.py --config conf/attentiononly.yaml --training.theta_dist "uniform" --training.diversity 200 --wandb.name "uniform_200"
python3 ${python_files_dir}/train.py --config conf/attentiononly.yaml --training.theta_dist "uniform" --training.diversity 300 --wandb.name "uniform_300"
python3 ${python_files_dir}/train.py --config conf/attentiononly.yaml --training.theta_dist "uniform" --training.diversity 500 --wandb.name "uniform_500"
python3 ${python_files_dir}/train.py --config conf/attentiononly.yaml --training.theta_dist "uniform" --training.diversity 750 --wandb.name "uniform_750"
python3 ${python_files_dir}/train.py --config conf/attentiononly.yaml --training.theta_dist "uniform" --training.diversity 1000 --wandb.name "uniform_1000"

python3 ${python_files_dir}/train.py --config conf/attentiononly.yaml --training.theta_dist "norm" --training.diversity 1 --wandb.name "norm_1"
python3 ${python_files_dir}/train.py --config conf/attentiononly.yaml --training.theta_dist "norm" --training.diversity 2 --wandb.name "norm_2"
python3 ${python_files_dir}/train.py --config conf/attentiononly.yaml --training.theta_dist "norm" --training.diversity 5 --wandb.name "norm_5"
python3 ${python_files_dir}/train.py --config conf/attentiononly.yaml --training.theta_dist "norm" --training.diversity 10 --wandb.name "norm_10"
python3 ${python_files_dir}/train.py --config conf/attentiononly.yaml --training.theta_dist "norm" --training.diversity 20 --wandb.name "norm_20"
python3 ${python_files_dir}/train.py --config conf/attentiononly.yaml --training.theta_dist "norm" --training.diversity 35 --wandb.name "norm_35"
python3 ${python_files_dir}/train.py --config conf/attentiononly.yaml --training.theta_dist "norm" --training.diversity 50 --wandb.name "norm_50"
python3 ${python_files_dir}/train.py --config conf/attentiononly.yaml --training.theta_dist "norm" --training.diversity 75 --wandb.name "norm_75"
python3 ${python_files_dir}/train.py --config conf/attentiononly.yaml --training.theta_dist "norm" --training.diversity 100 --wandb.name "norm_100"
python3 ${python_files_dir}/train.py --config conf/attentiononly.yaml --training.theta_dist "norm" --training.diversity 150 --wandb.name "norm_150"
python3 ${python_files_dir}/train.py --config conf/attentiononly.yaml --training.theta_dist "norm" --training.diversity 200 --wandb.name "norm_200"
python3 ${python_files_dir}/train.py --config conf/attentiononly.yaml --training.theta_dist "norm" --training.diversity 300 --wandb.name "norm_300"
python3 ${python_files_dir}/train.py --config conf/attentiononly.yaml --training.theta_dist "norm" --training.diversity 500 --wandb.name "norm_500"
python3 ${python_files_dir}/train.py --config conf/attentiononly.yaml --training.theta_dist "norm" --training.diversity 750 --wandb.name "norm_750"
python3 ${python_files_dir}/train.py --config conf/attentiononly.yaml --training.theta_dist "norm" --training.diversity 1000 --wandb.name "norm_1000"

################## 11.09.

python3 ${python_files_dir}/train.py --config conf/attentiononly.yaml --training.curriculum.dims.start 1 --training.curriculum.dims.end 1 --model.n_dims 1 --training.diversity 3200000 --training.curriculum.points.start 30 --training.constant_z 14 --wandb.name "constz14_1dim"
python3 ${python_files_dir}/train.py --config conf/attentiononly.yaml --training.curriculum.dims.start 1 --training.curriculum.dims.end 1 --model.n_dims 1 --training.diversity 3200000 --training.curriculum.points.start 30 --training.constant_z 14 --wandb.name "constz14_1dim_4l" --model.n_layer 4
python3 ${python_files_dir}/train.py --config conf/attentiononly.yaml --training.curriculum.dims.start 1 --training.curriculum.dims.end 1 --model.n_dims 1 --training.diversity 3200000 --training.curriculum.points.start 30 --training.constant_z 14 --wandb.name "constz14_1dim_2l" --model.n_layer 2
python3 ${python_files_dir}/train.py --config conf/attentiononly.yaml --training.curriculum.dims.start 1 --training.curriculum.dims.end 1 --model.n_dims 1 --training.diversity 3200000 --training.curriculum.points.start 30 --training.constant_z 14 --wandb.name "constz14_1dim_1l" --model.n_layer 1

################## 14.09.

python3 ${python_files_dir}/train.py --config conf/attentiononly.yaml --training.curriculum.dims.start 1 --training.curriculum.dims.end 1 --model.n_dims 1 --model.n_embd 8 --training.diversity 3200000 --training.curriculum.points.start 30 --training.constant_z 14 --wandb.name "constz14_1dim_8embd"
python3 ${python_files_dir}/train.py --config conf/attentiononly.yaml --training.curriculum.dims.start 1 --training.curriculum.dims.end 1 --model.n_dims 1 --model.n_embd 16 --training.diversity 3200000 --training.curriculum.points.start 30 --training.constant_z 14 --wandb.name "constz14_1dim_16embd"
python3 ${python_files_dir}/train.py --config conf/attentiononly.yaml --training.curriculum.dims.start 1 --training.curriculum.dims.end 1 --model.n_dims 1 --model.n_embd 32 --training.diversity 3200000 --training.curriculum.points.start 30 --training.constant_z 14 --wandb.name "constz14_1dim_32embd"

python3 ${python_files_dir}/train.py --config conf/attentiononly.yaml --training.curriculum.dims.start 1 --training.curriculum.dims.end 1 --model.n_dims 1 --model.n_embd 8 --training.diversity 3200000 --training.curriculum.points.start 30 --training.constant_z 14 --wandb.name "constz14_1dim_4l_8embd" --model.n_layer 4
python3 ${python_files_dir}/train.py --config conf/attentiononly.yaml --training.curriculum.dims.start 1 --training.curriculum.dims.end 1 --model.n_dims 1 --model.n_embd 8 --training.diversity 3200000 --training.curriculum.points.start 30 --training.constant_z 14 --wandb.name "constz14_1dim_2l_8embd" --model.n_layer 2
python3 ${python_files_dir}/train.py --config conf/attentiononly.yaml --training.curriculum.dims.start 1 --training.curriculum.dims.end 1 --model.n_dims 1 --model.n_embd 8 --training.diversity 3200000 --training.curriculum.points.start 30 --training.constant_z 14 --wandb.name "constz14_1dim_1l_8embd" --model.n_layer 1
python3 ${python_files_dir}/train.py --config conf/attentiononly.yaml --training.curriculum.dims.start 1 --training.curriculum.dims.end 1 --model.n_dims 1 --model.n_embd 16 --training.diversity 3200000 --training.curriculum.points.start 30 --training.constant_z 14 --wandb.name "constz14_1dim_4l_16embd" --model.n_layer 4
python3 ${python_files_dir}/train.py --config conf/attentiononly.yaml --training.curriculum.dims.start 1 --training.curriculum.dims.end 1 --model.n_dims 1 --model.n_embd 16 --training.diversity 3200000 --training.curriculum.points.start 30 --training.constant_z 14 --wandb.name "constz14_1dim_2l_16embd" --model.n_layer 2
python3 ${python_files_dir}/train.py --config conf/attentiononly.yaml --training.curriculum.dims.start 1 --training.curriculum.dims.end 1 --model.n_dims 1 --model.n_embd 16 --training.diversity 3200000 --training.curriculum.points.start 30 --training.constant_z 14 --wandb.name "constz14_1dim_1l_16embd" --model.n_layer 1

################## 25.09.

python3 ${python_files_dir}/train.py --config conf/attentiononly.yaml --training.theta_dist "norm" --training.diversity 3 --wandb.name "norm_3"
python3 ${python_files_dir}/train.py --config conf/attentiononly.yaml --training.theta_dist "norm" --training.diversity 4 --wandb.name "norm_4"
python3 ${python_files_dir}/train.py --config conf/attentiononly.yaml --training.theta_dist "norm" --training.diversity 6 --wandb.name "norm_6"
python3 ${python_files_dir}/train.py --config conf/attentiononly.yaml --training.theta_dist "norm" --training.diversity 7 --wandb.name "norm_7"
python3 ${python_files_dir}/train.py --config conf/attentiononly.yaml --training.theta_dist "norm" --training.diversity 8 --wandb.name "norm_8"
python3 ${python_files_dir}/train.py --config conf/attentiononly.yaml --training.theta_dist "norm" --training.diversity 9 --wandb.name "norm_9"

python3 ${python_files_dir}/train.py --config conf/attentiononly.yaml --training.theta_dist "uniform" --training.diversity 3 --wandb.name "uniform_3"
python3 ${python_files_dir}/train.py --config conf/attentiononly.yaml --training.theta_dist "uniform" --training.diversity 4 --wandb.name "uniform_4"
python3 ${python_files_dir}/train.py --config conf/attentiononly.yaml --training.theta_dist "uniform" --training.diversity 6 --wandb.name "uniform_6"
python3 ${python_files_dir}/train.py --config conf/attentiononly.yaml --training.theta_dist "uniform" --training.diversity 7 --wandb.name "uniform_7"
python3 ${python_files_dir}/train.py --config conf/attentiononly.yaml --training.theta_dist "uniform" --training.diversity 8 --wandb.name "uniform_8"
python3 ${python_files_dir}/train.py --config conf/attentiononly.yaml --training.theta_dist "uniform" --training.diversity 9 --wandb.name "uniform_9"

################## sde

python3 ${python_files_dir}/train.py --config conf/sde.yaml --training.poisson 0 --training.training_loss masked_mse
python3 ${python_files_dir}/train.py --config conf/sde.yaml --training.poisson 0 --training.training_loss masked_mse --model.family gpt2_ao_sde --wandb.name GPT2AO_sde
python3 ${python_files_dir}/train.py --config conf/sde.yaml --training.poisson 0 --training.training_loss masked_mse --model.family gpt2_ao_sde --model.n_layer 8 --model.n_head 1 --wandb.name AO_8l1h_sde
python3 ${python_files_dir}/train.py --config conf/sde.yaml --training.poisson 0 --training.training_loss masked_mse --model.family gpt2_ao_sde --model.n_layer 4 --model.n_head 1 --wandb.name AO_4l1h_sde
python3 ${python_files_dir}/train.py --config conf/sde.yaml --training.poisson 0 --training.training_loss masked_mse --model.family gpt2_ao_sde --model.n_layer 2 --model.n_head 1 --wandb.name AO_2l1h_sde
python3 ${python_files_dir}/train.py --config conf/sde.yaml --training.poisson 0 --training.training_loss masked_mse --model.family gpt2_ao_sde --model.n_layer 1 --model.n_head 1 --wandb.name AO_1l1h_sde
python3 ${python_files_dir}/train.py --config conf/sde.yaml --training.poisson 0 --training.training_loss masked_mse --model.n_layer 8 --model.n_head 1 --wandb.name 8l1h_sde
python3 ${python_files_dir}/train.py --config conf/sde.yaml --training.poisson 0 --training.training_loss masked_mse --model.n_layer 4 --model.n_head 1 --wandb.name 4l1h_sde
python3 ${python_files_dir}/train.py --config conf/sde.yaml --training.poisson 0 --training.training_loss masked_mse --model.n_layer 2 --model.n_head 1 --wandb.name 2l1h_sde
python3 ${python_files_dir}/train.py --config conf/sde.yaml --training.poisson 0 --training.training_loss masked_mse --model.n_layer 1 --model.n_head 1 --wandb.name 1l1h_sde
python3 ${python_files_dir}/train.py --config conf/sde.yaml --training.poisson 0 --training.training_loss masked_mse --model.family lstm_sde --wandb.name lstm_sde_2l --model.n_layer 2
python3 ${python_files_dir}/train.py --config conf/sde.yaml --training.poisson 0 --training.training_loss masked_mse --model.family rnn_sde --wandb.name rnn_sde_2l --model.n_layer 2
python3 ${python_files_dir}/train.py --config conf/sde.yaml --training.poisson 0 --training.training_loss masked_mse --model.family gru_sde --wandb.name gru_sde_2l --model.n_layer 2

python3 ${python_files_dir}/train.py --config conf/sde.yaml --training.poisson 0 --training.ode 1 --training.training_loss masked_mse --wandb.name ode
python3 ${python_files_dir}/train.py --config conf/sde.yaml --training.poisson 0 --model.family gpt2_ao_sde --wandb.name GPT2AO_ode --training.ode 1 --training.training_loss masked_mse
python3 ${python_files_dir}/train.py --config conf/sde.yaml --training.poisson 0 --model.family gpt2_ao_sde --model.n_layer 8 --model.n_head 1 --wandb.name AO_8l1h_ode --training.ode 1 --training.training_loss masked_mse
python3 ${python_files_dir}/train.py --config conf/sde.yaml --training.poisson 0 --model.family gpt2_ao_sde --model.n_layer 4 --model.n_head 1 --wandb.name AO_4l1h_ode --training.ode 1 --training.training_loss masked_mse
python3 ${python_files_dir}/train.py --config conf/sde.yaml --training.poisson 0 --model.family gpt2_ao_sde --model.n_layer 2 --model.n_head 1 --wandb.name AO_2l1h_ode --training.ode 1 --training.training_loss masked_mse
python3 ${python_files_dir}/train.py --config conf/sde.yaml --training.poisson 0 --model.family gpt2_ao_sde --model.n_layer 1 --model.n_head 1 --wandb.name AO_1l1h_ode --training.ode 1 --training.training_loss masked_mse
python3 ${python_files_dir}/train.py --config conf/sde.yaml --training.poisson 0 --model.n_layer 8 --model.n_head 1 --wandb.name 8l1h_ode --training.ode 1 --training.training_loss masked_mse
python3 ${python_files_dir}/train.py --config conf/sde.yaml --training.poisson 0 --model.n_layer 4 --model.n_head 1 --wandb.name 4l1h_ode --training.ode 1 --training.training_loss masked_mse
python3 ${python_files_dir}/train.py --config conf/sde.yaml --training.poisson 0 --model.n_layer 2 --model.n_head 1 --wandb.name 2l1h_ode --training.ode 1 --training.training_loss masked_mse
python3 ${python_files_dir}/train.py --config conf/sde.yaml --training.poisson 0 --model.n_layer 1 --model.n_head 1 --wandb.name 1l1h_ode --training.ode 1 --training.training_loss masked_mse
python3 ${python_files_dir}/train.py --config conf/sde.yaml --training.poisson 0 --model.family lstm_sde --wandb.name lstm_ode_2l --training.ode 1 --training.training_loss masked_mse --model.n_layer 2
python3 ${python_files_dir}/train.py --config conf/sde.yaml --training.poisson 0 --model.family rnn_sde --wandb.name rnn_ode_2l --training.ode 1 --training.training_loss masked_mse --model.n_layer 2
python3 ${python_files_dir}/train.py --config conf/sde.yaml --training.poisson 0 --model.family gru_sde --wandb.name gru_ode_2l --training.ode 1 --training.training_loss masked_mse --model.n_layer 2

python3 ${python_files_dir}/train.py --config conf/sde.yaml --training.poisson 0 --training.training_loss masked_mse --model.family gpt2_ao_sde --training.lamb 20 --training.max_time 1 --model.n_layer 8 --model.n_head 1 --wandb.name AO_8l1h_sde_lamb20
python3 ${python_files_dir}/train.py --config conf/sde.yaml --training.poisson 0 --training.training_loss masked_mse --model.family gpt2_ao_sde --training.lamb 10 --training.max_time 2 --model.n_layer 8 --model.n_head 1 --wandb.name AO_8l1h_sde_lamb10
python3 ${python_files_dir}/train.py --config conf/sde.yaml --training.poisson 0 --training.training_loss masked_mse --model.family gpt2_ao_sde --training.lamb 20 --training.max_time 1 --model.n_layer 8 --model.n_head 1 --model.n_dims 1 --training.curriculum.dims.start 1 --training.curriculum.dims.end 1 --training.diversity 400000 --wandb.name GPT2AO_1dim_sde

################## poisson 1

python3 ${python_files_dir}/train.py --config conf/sde.yaml --training.poisson 1 --training.training_loss masked_mse --wandb.name sde_pois
python3 ${python_files_dir}/train.py --config conf/sde.yaml --training.poisson 1 --training.training_loss masked_mse --model.family gpt2_ao_sde --wandb.name GPT2AO_sde_pois
python3 ${python_files_dir}/train.py --config conf/sde.yaml --training.poisson 1 --training.training_loss masked_mse --model.family gpt2_ao_sde --model.n_layer 8 --model.n_head 1 --wandb.name AO_8l1h_sde_pois
python3 ${python_files_dir}/train.py --config conf/sde.yaml --training.poisson 1 --training.training_loss masked_mse --model.family gpt2_ao_sde --model.n_layer 4 --model.n_head 1 --wandb.name AO_4l1h_sde_pois
python3 ${python_files_dir}/train.py --config conf/sde.yaml --training.poisson 1 --training.training_loss masked_mse --model.family gpt2_ao_sde --model.n_layer 2 --model.n_head 1 --wandb.name AO_2l1h_sde_pois
python3 ${python_files_dir}/train.py --config conf/sde.yaml --training.poisson 1 --training.training_loss masked_mse --model.family gpt2_ao_sde --model.n_layer 1 --model.n_head 1 --wandb.name AO_1l1h_sde_pois
python3 ${python_files_dir}/train.py --config conf/sde.yaml --training.poisson 1 --training.training_loss masked_mse --model.n_layer 8 --model.n_head 1 --wandb.name 8l1h_sde_pois
python3 ${python_files_dir}/train.py --config conf/sde.yaml --training.poisson 1 --training.training_loss masked_mse --model.n_layer 4 --model.n_head 1 --wandb.name 4l1h_sde_pois
python3 ${python_files_dir}/train.py --config conf/sde.yaml --training.poisson 1 --training.training_loss masked_mse --model.n_layer 2 --model.n_head 1 --wandb.name 2l1h_sde_pois
python3 ${python_files_dir}/train.py --config conf/sde.yaml --training.poisson 1 --training.training_loss masked_mse --model.n_layer 1 --model.n_head 1 --wandb.name 1l1h_sde_pois
python3 ${python_files_dir}/train.py --config conf/sde.yaml --training.poisson 1 --training.training_loss masked_mse --model.family lstm_sde --wandb.name lstm_sde_2l_pois --model.n_layer 2
python3 ${python_files_dir}/train.py --config conf/sde.yaml --training.poisson 1 --training.training_loss masked_mse --model.family rnn_sde --wandb.name rnn_sde_2l_pois --model.n_layer 2
python3 ${python_files_dir}/train.py --config conf/sde.yaml --training.poisson 1 --training.training_loss masked_mse --model.family gru_sde --wandb.name gru_sde_2l_pois --model.n_layer 2

python3 ${python_files_dir}/train.py --config conf/sde.yaml --training.poisson 1 --training.training_loss masked_mse --model.family gpt2_ao_sde --training.lamb 20 --training.max_time 1 --model.n_layer 8 --model.n_head 1 --wandb.name AO_8l1h_sde_lamb20_pois
python3 ${python_files_dir}/train.py --config conf/sde.yaml --training.poisson 1 --training.training_loss masked_mse --model.family gpt2_ao_sde --training.lamb 10 --training.max_time 2 --model.n_layer 8 --model.n_head 1 --wandb.name AO_8l1h_sde_lamb10_pois
python3 ${python_files_dir}/train.py --config conf/sde.yaml --training.poisson 1 --training.training_loss masked_mse --model.family gpt2_ao_sde --training.lamb 20 --training.max_time 1 --model.n_layer 8 --model.n_head 1 --model.n_dims 1 --training.curriculum.dims.start 1 --training.curriculum.dims.end 1 --training.diversity 400000 --wandb.name GPT2AO_1dim_sde_pois

################## 06.10.

python3 ${python_files_dir}/train.py --config conf/counterfactual.yaml --training.curriculum.dims.start 1 --training.curriculum.dims.end 1 --model.n_dims 1 --training.diversity 3200000 --training.curriculum.points.start 30 --training.constant_z 14 --wandb.name "CF_constz14_1dim"
python3 ${python_files_dir}/train.py --config conf/counterfactual.yaml --training.curriculum.dims.start 2 --training.curriculum.dims.end 2 --model.n_dims 2 --training.diversity 6400000 --training.curriculum.points.start 30 --training.constant_z 14 --wandb.name "CF_constz14_2dim"
python3 ${python_files_dir}/train.py --config conf/counterfactual.yaml --training.curriculum.dims.start 3 --training.curriculum.dims.end 3 --model.n_dims 3 --training.diversity 9600000 --training.curriculum.points.start 30 --training.constant_z 14 --wandb.name "CF_constz14_3dim"
python3 ${python_files_dir}/train.py --config conf/counterfactual.yaml --training.curriculum.dims.start 4 --training.curriculum.dims.end 4 --model.n_dims 4 --training.diversity 12800000 --training.curriculum.points.start 30 --training.constant_z 14 --wandb.name "CF_constz14_4dim"
python3 ${python_files_dir}/train.py --config conf/counterfactual.yaml --training.curriculum.dims.start 1 --training.curriculum.dims.end 1 --model.n_dims 1 --training.diversity 3200000 --training.curriculum.points.start 30 --wandb.name "CF_min30_1dim"

python3 ${python_files_dir}/train.py --config conf/sde.yaml --training.poisson 0 --training.training_loss masked_mse --model.family gpt2_ao_sde --model.n_layer 8 --model.n_head 1 --wandb.name AO_8l1h_sde_lamb100 --training.lamb 100 --training.max_time 0.2
python3 ${python_files_dir}/train.py --config conf/sde.yaml --training.poisson 0 --training.training_loss masked_mse --model.family gpt2_ao_sde --model.n_layer 8 --model.n_head 1 --wandb.name AO_8l1h_sde_lamb200 --training.lamb 200 --training.max_time 0.1
python3 ${python_files_dir}/train.py --config conf/sde.yaml --training.poisson 1 --training.training_loss masked_mse --model.family gpt2_ao_sde --model.n_layer 8 --model.n_head 1 --wandb.name AO_8l1h_sde_lamb100_pois --training.lamb 100 --training.max_time 0.2
python3 ${python_files_dir}/train.py --config conf/sde.yaml --training.poisson 1 --training.training_loss masked_mse --model.family gpt2_ao_sde --model.n_layer 8 --model.n_head 1 --wandb.name AO_8l1h_sde_lamb200_pois --training.lamb 200 --training.max_time 0.1
python3 ${python_files_dir}/train.py --config conf/sde.yaml --training.poisson 0 --training.training_loss masked_mse --model.family gpt2_sde --wandb.name sde_lamb100 --training.lamb 100 --training.max_time 0.2
python3 ${python_files_dir}/train.py --config conf/sde.yaml --training.poisson 0 --training.training_loss masked_mse --model.family gpt2_sde --wandb.name sde_lamb200 --training.lamb 200 --training.max_time 0.1
python3 ${python_files_dir}/train.py --config conf/sde.yaml --training.poisson 1 --training.training_loss masked_mse --model.family gpt2_sde --wandb.name sde_lamb100_pois --training.lamb 100 --training.max_time 0.2
python3 ${python_files_dir}/train.py --config conf/sde.yaml --training.poisson 1 --training.training_loss masked_mse --model.family gpt2_sde --wandb.name sde_lamb200_pois --training.lamb 200 --training.max_time 0.1

python3 ${python_files_dir}/train.py --config conf/counterfactual.yaml --training.curriculum.dims.start 1 --training.curriculum.dims.end 1 --model.n_dims 1 --training.diversity 6400000 --training.curriculum.points.start 30 --training.constant_z 14 --wandb.name "CF_constz14_1dim_100k" --training.train_steps 100000
python3 ${python_files_dir}/train.py --config conf/counterfactual.yaml --training.curriculum.dims.start 2 --training.curriculum.dims.end 2 --model.n_dims 2 --training.diversity 12800000 --training.curriculum.points.start 30 --training.constant_z 14 --wandb.name "CF_constz14_2dim_100k" --training.train_steps 100000
python3 ${python_files_dir}/train.py --config conf/counterfactual.yaml --training.curriculum.dims.start 3 --training.curriculum.dims.end 3 --model.n_dims 3 --training.diversity 19200000 --training.curriculum.points.start 30 --training.constant_z 14 --wandb.name "CF_constz14_3dim_100k" --training.train_steps 100000
python3 ${python_files_dir}/train.py --config conf/counterfactual.yaml --training.curriculum.dims.start 4 --training.curriculum.dims.end 4 --model.n_dims 4 --training.diversity 25600000 --training.curriculum.points.start 30 --training.constant_z 14 --wandb.name "CF_constz14_4dim_100k" --training.train_steps 100000
python3 ${python_files_dir}/train.py --config conf/counterfactual.yaml --training.curriculum.dims.start 5 --training.curriculum.dims.end 5 --model.n_dims 5 --training.diversity 32000000 --training.curriculum.points.start 30 --training.constant_z 14 --wandb.name "CF_constz14_5dim_100k" --training.train_steps 100000
python3 ${python_files_dir}/train.py --config conf/counterfactual.yaml --training.curriculum.dims.start 1 --training.curriculum.dims.end 1 --model.n_dims 1 --training.diversity 6400000 --training.curriculum.points.start 30 --wandb.name "CF_min30_1dim_100k" --training.train_steps 100000

python3 ${python_files_dir}/train.py --config conf/attentiononly.yaml --wandb.name "8l1hAO_cont" --training.continuation 1 --model.family gpt2_ao_cont
python3 ${python_files_dir}/train.py --config conf/attentiononly.yaml --wandb.name "8l1hAO_min1" --model.family gpt2_ao --training.curriculum.points.start 1

python3 ${python_files_dir}/train.py --config conf/sde.yaml --training.poisson 0 --training.training_loss masked_mse --model.family gpt2_sde --wandb.name sde_lamb200_diffusion10 --training.lamb 200 --training.max_time 0.1 --training.diffusion 10
python3 ${python_files_dir}/train.py --config conf/sde.yaml --training.poisson 1 --training.training_loss masked_mse --model.family gpt2_sde --wandb.name sde_lamb100_pois_diffusion10 --training.lamb 100 --training.max_time 0.2 --training.diffusion 10
python3 ${python_files_dir}/train.py --config conf/sde.yaml --training.poisson 1 --training.training_loss masked_mse --model.family gpt2_sde --wandb.name sde_lamb200_pois_diffusion10 --training.lamb 200 --training.max_time 0.1 --training.diffusion 10

python3 ${python_files_dir}/train.py --config conf/counterfactual.yaml --training.train_steps 1000000 --training.curriculum.dims.start 1 --training.curriculum.dims.end 1 --model.n_dims 1 --training.diversity 64000000 --wandb.name "GPT2_1mio"
python3 ${python_files_dir}/train.py --config conf/counterfactual.yaml --training.train_steps 1000000 --training.diversity 320000000 --wandb.name "GPT2_5d_1mio"

echo "python script executed"

conda deactivate

echo "experiment completed"