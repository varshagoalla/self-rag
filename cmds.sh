module load Anaconda3/2024.02-1
conda create -n selfrag python=3.10
conda activate selfrag
cd /scratch/user/varshagoalla/self-rag/
export HF_HOME=/scratch/user/varshagoalla/huggingface/
export HUGGINGFACE_HUB_CACHE=/scratch/user/varshagoalla/huggingface/
export TRANSFORMERS_CACHE=/scratch/user/varshagoalla/huggingface/
mkdir -p $HF_HOME
pip install -r requirements.txt 
module load GCC/14.2.0
module load CUDA/12.1.1
pip install --force-reinstall numpy==1.24.4
pip install vllm==0.2.6
mkdir ~/wheelhouse
wget https://github.com/Dao-AILab/flash-attention/releases/download/v2.3.6/flash_attn-2.3.6+cu122torch2.1cxx11abiFALSE-cp310-cp310-linux_x86_64.whl
python -c "from transformers import AutoTokenizer; AutoTokenizer.from_pretrained('selfrag/selfrag_llama2_7b', cache_dir='/scratch/user/varshagoalla/huggingface')"
python -c "from huggingface_hub import snapshot_download; snapshot_download(repo_id='selfrag/selfrag_llama2_7b', local_dir='/scratch/user/varshagoalla/huggingface/selfrag_llama2_7b', local_dir_use_symlinks=False)"
python -c "from huggingface_hub import snapshot_download; snapshot_download(repo_id='TinyLlama/TinyLlama-1.1B-Chat-v1.0')"



# Open an interactive session on the cluster with 1 GPU to build flash attention and test out the evaluation code.
srun --partition=gpu --nodes=1 --ntasks=1 --gres=gpu:1 --cpus-per-task=8 --mem=64G --time=04:00:00 --pty bash
conda activate selfrag
module load GCC/14.2.0
module load CUDA/12.1.1
export HF_HOME=/scratch/user/varshagoalla/huggingface/
export HUGGINGFACE_HUB_CACHE=/scratch/user/varshagoalla/huggingface/
export TRANSFORMERS_CACHE=/scratch/user/varshagoalla/huggingface/
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
cd ~/wheelhouse/
pip install ~/wheelhouse/flash_attn-2.3.6+cu122torch2.1cxx11abiFALSE-cp310-cp310-linux_x86_64.whl
python run_short_form.py --model_name /scratch/user/varshagoalla/huggingface/selfrag_llama2_7b --input_file ../eval_data/popqa_longtail_w_gs_first_500_samples.jsonl --mode adaptive_retrieval --max_new_tokens 100 --threshold 0.2 --output_file popqa_results.json --metric match --ndocs 10 --use_groundness --use_utility --use_seqscore  --dtype half --device cuda



cd output/tinyllama_selfrag_sft/epoch_0/


python merge_lora.py 
python run_short_form.py --model_name  /scratch/user/varshagoalla/self-rag/retrieval_lm/output/tinyllama_selfrag_sft_merged --input_file ../eval_data/popqa_longtail_w_gs_first_500_samples.jsonl --mode adaptive_retrieval --max_new_tokens 100 --threshold 0.2 --output_file popqa_results_tinyllama.json --metric match --ndocs 10 --use_groundness --use_utility --use_seqscore  --dtype half --device cuda