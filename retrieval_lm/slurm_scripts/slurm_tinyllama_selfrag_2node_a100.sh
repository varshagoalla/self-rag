#!/bin/bash
#SBATCH --job-name=tinyllama-selfrag-2n
#SBATCH --account=xxx
#SBATCH --partition=gpu
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:a100:2
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=15:00:00
#SBATCH --output=logs/%x-%j.out

source ~/.bashrc
conda activate selfrag
module purge
module load CUDA/12.1.1

cd /scratch/user/varshagoalla/self-rag/retrieval_lm
mkdir -p logs

export HF_HOME=/scratch/user/varshagoalla/huggingface
export HUGGINGFACE_HUB_CACHE=$HF_HOME
export TRANSFORMERS_OFFLINE=1
export HF_HUB_OFFLINE=1

export MASTER_ADDR=$(scontrol show hostnames "$SLURM_NODELIST" | head -n 1)
export MASTER_PORT=29501

echo "MASTER_ADDR=$MASTER_ADDR"
echo "MASTER_PORT=$MASTER_PORT"
scontrol show hostnames "$SLURM_NODELIST"

srun bash -c '
echo "HOSTNAME: $(hostname)"
nvidia-smi --query-gpu=name,index,memory.total --format=csv
python -c "import torch; print(\"cuda devices =\", torch.cuda.device_count()); print(\"bf16 supported =\", torch.cuda.is_bf16_supported() if torch.cuda.is_available() else False)"
bash script_finetune_tinyllama_multinode.sh
'
