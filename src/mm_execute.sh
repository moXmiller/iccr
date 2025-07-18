#!/bin/bash

#SBATCH --mem-per-cpu=20000
#SBATCH -n 4
#SBATCH --gpus=rtx_3090:1
#SBATCH --time=1200
#SBATCH --job-name=AO5cnt
#SBATCH --output=output_files/18_07_handwritten_embeddings_E_5_cont.out

python_files_dir="/cluster/home/millerm/cf/garg_cf/src"
conda_env="/cluster/scratch/millerm/miniconda3_jul/envs/icl"
echo "paths read in"

source activate $conda_env
echo "conda environment activated"

module load eth_proxy
module load stack/2024-05
module load gcc/13.2.0
module load cuda/12.2.1
echo "modules loaded"

echo "python execution started"
python3 ${python_files_dir}/train.py --config conf/sweepm.yaml # train.py --config conf/sweepg.yaml # mm_eval.py --model_size eightlayer --ao 1 
echo "python script executed"

conda deactivate

echo "experiment completed"