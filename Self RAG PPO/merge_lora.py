#!/usr/bin/env python3
"""
merge_lora.py — Merge a LoRA adapter into the base model weights.

vLLM (and some eval scripts) cannot load LoRA adapters directly.
This script merges the adapter in-place on CPU and saves a full model checkpoint.

Usage:
    python merge_lora.py \
        --base_model  /path/to/selfrag_llama2_7b \
        --lora_adapter /path/to/ppo_lora_output \
        --output_dir  /path/to/merged_model
"""

import os
import argparse
import torch

os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_HUB_OFFLINE"] = "1"

SELFRAG_TOKENS = [
    "[No Retrieval]", "[Retrieval]", "[Continue to Use Evidence]",
    "[Irrelevant]", "[Relevant]",
    "<paragraph>", "</paragraph>",
    "[Utility:1]", "[Utility:2]", "[Utility:3]", "[Utility:4]", "[Utility:5]",
    "[Fully supported]", "[Partially supported]", "[No support / Contradictory]",
]


def main():
    p = argparse.ArgumentParser(description="Merge LoRA adapter into base model for evaluation.")
    p.add_argument("--base_model",   required=True, help="Path to base model (e.g. selfrag_llama2_7b)")
    p.add_argument("--lora_adapter", required=True, help="Path to LoRA adapter directory")
    p.add_argument("--output_dir",   required=True, help="Where to save the merged model")
    args = p.parse_args()

    from transformers import AutoTokenizer, AutoModelForCausalLM
    from peft import PeftModel

    print(f"Loading base model: {args.base_model}")
    model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        torch_dtype=torch.float16,
        device_map="cpu",
        local_files_only=True,
    )

    print(f"Loading tokenizer: {args.base_model}")
    tokenizer = AutoTokenizer.from_pretrained(args.base_model, local_files_only=True, use_fast=False)

    new_tokens = [t for t in SELFRAG_TOKENS if t not in tokenizer.get_vocab()]
    if new_tokens:
        tokenizer.add_special_tokens({"additional_special_tokens": new_tokens})
        print(f"Added {len(new_tokens)} Self-RAG special tokens")

    model.resize_token_embeddings(len(tokenizer))

    print(f"Loading LoRA adapter: {args.lora_adapter}")
    model = PeftModel.from_pretrained(model, args.lora_adapter, local_files_only=True)

    print("Merging LoRA weights...")
    model = model.merge_and_unload()

    os.makedirs(args.output_dir, exist_ok=True)
    model.save_pretrained(args.output_dir, safe_serialization=True)
    tokenizer.save_pretrained(args.output_dir)

    print(f"\nDone! Merged model saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
