# Self-RAG PPO — Reinforcement Learning Experiments

This folder contains all code, trained model adapters, evaluation datasets, and analysis notebooks for the PPO-based fine-tuning experiments on top of `selfrag/selfrag_llama2_7b`.

---

## Overview

We trained three variants of PPO on Self-RAG:

| Variant | Training Script | Reward Signal | Data |
|---------|----------------|---------------|------|
| **PPO-Basic** | `ppo_basic_training.py` | Rule-based reflection tokens (utility, groundedness, relevance, format). Reward: `0.40×utility + 0.30×groundedness + 0.20×relevance + 0.10×format`. Batch-whitened before PPO update. | 20K stratified [Retrieval] examples |
| **PPO-Hard** | `ppo_hard_training.py` | Rule-based + anti-collapse penalties (repetition, length, coherence). Reward: `0.80 × reflection + 0.20 × anti-collapse`. | Hard-filtered examples ranked by judgment token count; 80/20 train/val split |
| **Critic-PPO** | `ppo_critic_training.py` | Frozen `selfrag/self_rag_critic` model evaluates utility, groundedness, and relevance externally — prevents self-scored reward hacking. | Same hard-example filter as PPO-Hard |

Evaluation was conducted on ARC Challenge, PopQA, and TriviaQA (short-form) and ASQA (long-form).

---

## File Structure

### Training Scripts

| File | Purpose |
|------|---------|
| `ppo_basic_training.py` | PPO Basic — full training script with rule-based reward from self-generated reflection tokens. Runs `evaluate_model()` automatically after training and saves `evaluation_results.json` to the output folder. |
| `ppo_hard_training.py` | PPO-Hard — extends Basic with hard-example filtering and anti-collapse penalties. Also auto-evaluates after training. |
| `ppo_critic_training.py` | Critic-PPO — uses a frozen critic model for reward scoring. Auto-evaluates after training and writes `evaluation_results_critic.json`. |
| `merge_lora.py` | Merges a trained LoRA adapter into the base model weights for full-model evaluation (required for Self-RAG eval scripts which cannot load LoRA adapters directly). |

> **Note on `evaluation_results.json`:** This file is generated automatically at the **end of training** by each training script's own built-in `evaluate_model()` function. It compares the PPO model against the baseline on the held-out test split using F1, containment, and reflection-token metrics. It is **not** produced by a separate eval script.

### Trained Models (`outputs/`)

| Folder | Variant | Data Size | Contents |
|--------|---------|-----------|---------|
| `outputs/basic_5k/` | PPO-Basic | 5K | Final adapter: `adapter_config.json`, `adapter_model.safetensors`, tokenizer, `evaluation_results.json`, `reward_history.json`, `reward_curves.png` |
| `outputs/hard_5k/` | PPO-Hard | 5K | Same structure as above |
| `outputs/hard_10k/` | PPO-Hard | 10K | Same structure as above |
| `outputs/critic_10k/` | Critic-PPO | 10K | Final adapter + `evaluation_results_critic.json` + reward files |
| `outputs/critic_20k/` | Critic-PPO | 20K | Same as above |

**Each complete output folder contains:**
- `adapter_config.json` + `adapter_model.safetensors` — LoRA adapter weights (load with PEFT)
- `tokenizer_config.json`, `tokenizer.model`, `added_tokens.json`, `special_tokens_map.json` — tokenizer
- `evaluation_results.json` / `evaluation_results_critic.json` — auto-saved post-training benchmark comparison (PPO vs base)
- `reward_history.json` — per-step reward logged throughout PPO training
- `reward_curves.png` — reward curve visualization

### Evaluation Datasets (`datasets/` and `longform_data/`)

Short-form evaluation outputs are stored in `datasets/`. Files follow the naming convention `{model}_{benchmark}.json`:

| File pattern | Description |
|---|---|
| `base_*.json` | Base `selfrag/selfrag_llama2_7b` outputs (ARC, PopQA, TriviaQA) |
| `ppo_v2_*.json` | PPO-Hard outputs |
| `critic_*.json` | Critic-PPO 10K outputs |
| `critic_20k_*.json` | Critic-PPO 20K outputs |

Long-form evaluation data is in `longform_data/`:

| File | Description |
|------|-------------|
| `asqa_eval_gtr_top100.json` | ASQA test set with GTR top-100 retrieved passages |
| `base_asqa.json` | Base model ASQA outputs |
| `critic10k_asqa.json` | Critic-PPO 10K ASQA outputs |

Training / test splits are in `selfrag_data/`:
- `balanced_test.json` — stratified held-out test set used in post-training eval
- `full_dataset.json` — full dataset before splitting

### Analysis Notebooks

| Notebook | Purpose |
|----------|---------|
| `Analyze_All_Models.ipynb` | Short-form benchmark comparison (ARC, PopQA, TriviaQA) across base, PPO-Hard, and Critic-PPO variants. Produces `all_models_benchmarks.png` and `all_models_distribution.png`. |
| `Analyze_LongForm_ASQA.ipynb` | Long-form ASQA evaluation — EM-Recall and ROUGE-L comparison between base and Critic-10K. Produces `longform_asqa_eval.png`. |
| `Analyze_PPO_vs_Base.ipynb` | Head-to-head PPO vs base model analysis with score distributions. |

---

## Running Evaluation on a New Benchmark

### Step 1 — Merge the LoRA adapter

```bash
python merge_lora.py \
    --base_model  /path/to/selfrag_llama2_7b \
    --lora_adapter outputs/critic_10k \
    --output_dir  outputs/critic_10k_merged
```

### Step 2 — Short-form evaluation (ARC / PopQA / TriviaQA)

```bash
python ../retrieval_lm/run_short_form.py \
    --model_name outputs/critic_10k_merged \
    --input_file /path/to/arc_challenge_test.jsonl \
    --output_file datasets/critic_10k_arc.json \
    --mode always_retrieve \
    --metric match
```

### Step 3 — Long-form evaluation (ASQA)

```bash
python ../retrieval_lm/run_long_form_static.py \
    --model_name outputs/critic_10k_merged \
    --input_file longform_data/asqa_eval_gtr_top100.json \
    --output_file longform_data/critic10k_asqa.json \
    --mode always_retrieve \
    --metric match,rouge
```

### Step 4 — Analyze results

Open `Analyze_All_Models.ipynb` or `Analyze_LongForm_ASQA.ipynb` to reproduce all plots and tables.

---

## Requirements

```bash
pip install -r requirements.txt
```

Key dependencies: `torch`, `transformers`, `peft`, `trl`, `bitsandbytes`, `datasets`, `accelerate`.
