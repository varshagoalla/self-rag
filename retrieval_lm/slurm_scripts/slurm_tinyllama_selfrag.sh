#!/bin/bash
#SBATCH --job-name=tinyllama-selfrag
#SBATCH --account=your_account_number
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=4
#SBATCH --mem=0
#SBATCH --time=07:00:00
#SBATCH --output=logs/%x-%j.out

source ~/.bashrc
conda activate selfrag
module purge
module load CUDA/12.1.1

cd /scratch/user/varshagoalla/self-rag/retrieval_lm

mkdir -p logs

export HF_HOME=/scratch/user/varshagoalla/huggingface
export HUGGINGFACE_HUB_CACHE=$HF_HOME
export TRANSFORMERS_CACHE=$HF_HOME
export TRANSFORMERS_OFFLINE=1
export HF_HUB_OFFLINE=1

bash script_finetune_tinyllama.sh
