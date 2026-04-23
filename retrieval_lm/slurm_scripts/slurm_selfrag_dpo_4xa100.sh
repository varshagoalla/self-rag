#!/bin/bash
#SBATCH --job-name=selfrag-dpo-4xA100
#SBATCH --account=xxx
#SBATCH --partition=gpu
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=2
#SBATCH --gres=gpu:a100:2
#SBATCH --cpus-per-task=4
#SBATCH --mem=0
#SBATCH --time=1:00:00
#SBATCH --output=logs/%x-%j.out

source ~/.bashrc
conda activate selfrag_dpo
module purge
module load CUDA/12.1.1

cd /scratch/user/varshagoalla/self-rag/retrieval_lm
mkdir -p logs

export HF_HOME=/scratch/user/varshagoalla/huggingface
export HUGGINGFACE_HUB_CACHE=$HF_HOME
export TRANSFORMERS_CACHE=$HF_HOME
export TRANSFORMERS_OFFLINE=1
export HF_HUB_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export MASTER_ADDR=$(scontrol show hostnames "$SLURM_NODELIST" | head -n 1)
export MASTER_PORT=29501

MODEL_PATH=/scratch/user/varshagoalla/huggingface/models--selfrag--selfrag_llama2_7b/snapshots/190261383b0779ff66d2f95a73c7ad267d94b820
TRAIN_FILE=../rl_data/train_rl_dpo.jsonl
OUTPUT_DIR=../output/selfrag_dpo_4xa100_run1

srun bash -c '
export RANK=${SLURM_PROCID}
export WORLD_SIZE=${SLURM_NTASKS}
export LOCAL_RANK=${SLURM_LOCALID}
export MASTER_ADDR='"${MASTER_ADDR}"'
export MASTER_PORT='"${MASTER_PORT}"'

python train_selfrag_dpo.py \
  --model-name-or-path "'"${MODEL_PATH}"'" \
  --train-file "'"${TRAIN_FILE}"'" \
  --output-dir "'"${OUTPUT_DIR}"'" \
  --num-train-epochs 1 \
  --per-device-train-batch-size 1 \
  --gradient-accumulation-steps 4 \
  --learning-rate 5e-6 \
  --max-length 640 \
  --max-prompt-length 320 \
  --lora-rank 16 \
  --lora-alpha 16 \
  --load-in-4bit \
  --gradient-checkpointing \
  --empty-cache-steps 100
'
