#!/bin/bash


REPO_ROOT="/scratch/user/varshagoalla/self-rag"
EVAL_ROOT="$REPO_ROOT/eval_data"
MODEL_ROOT="${MODEL_ROOT:-$REPO_ROOT/output}"
SLURM_SCRIPT="${SLURM_SCRIPT:-$REPO_ROOT/retrieval_lm/slurm_eval_short_form.sh}"

MODELS=(
  "selfrag_dpo_4xa100_type_a_merged_ep2"
  "selfrag_dpo_4xa100_type_b_merged_ep2"
  "selfrag_dpo_4xa100_type_c_merged_ep2"
  "selfrag_dpo_mixed_20k_merged"
  "selfrag_dpo_mixed_20k_merged_ep2"
  "selfrag_dpo_mixed_25k_merged"
  "selfrag_dpo_mixed_25k_merged_ep2"
  "selfrag_dpo_mixed_all_merged"
  "selfrag_dpo_mixed_all_merged_ep2"
)

MATCH_DATASETS=(
  "$EVAL_ROOT/arc_challenge_processed.jsonl"
  "$EVAL_ROOT/popqa_longtail_w_gs.jsonl"
  "$EVAL_ROOT/triviaqa_test_w_gs.jsonl"
  "$EVAL_ROOT/health_claims_processed.jsonl"
)

for model_name in "${MODELS[@]}"; do
  model_path="$MODEL_ROOT/$model_name"

  for eval_file in "${MATCH_DATASETS[@]}"; do
    sbatch "$SLURM_SCRIPT" "$model_path" "$eval_file" match
  done

  sbatch "$SLURM_SCRIPT" "$model_path" "$EVAL_ROOT/health_claims_processed.jsonl" accuracy
done


