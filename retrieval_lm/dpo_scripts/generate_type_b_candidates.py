#!/usr/bin/env python

import argparse
import json
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def load_jsonl(path):
    rows = []
    with open(path) as f:
        for line in f:
            rows.append(json.loads(line))
    return rows


def save_jsonl(rows, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")


def build_retrieval_prefix(prompt, ctx):
    title = (ctx.get("title") or "").strip()
    text = (ctx.get("text") or "").strip()
    if not text:
        return None
    paragraph = f"{title}\n{text}".strip() if title else text
    return prompt + f"[Retrieval]<paragraph>{paragraph}</paragraph>"


def generate_for_prefix(model, tokenizer, prefix_text, max_new_tokens):
    inputs = tokenizer(prefix_text, return_tensors="pt").to(model.device)
    with torch.no_grad():
        generated = model.generate(
            **inputs,
            do_sample=False,
            max_new_tokens=max_new_tokens,
            pad_token_id=tokenizer.eos_token_id,
        )
    prompt_len = inputs["input_ids"].shape[1]
    continuation = tokenizer.decode(generated[0][prompt_len:], skip_special_tokens=False)
    return prefix_text + continuation


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name-or-path", required=True)
    parser.add_argument("--input-file", required=True)
    parser.add_argument("--retrieved-file", default=None)
    parser.add_argument("--output-file", required=True)
    parser.add_argument("--max-new-tokens", type=int, default=96)
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, use_fast=False)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        device_map="auto" if torch.cuda.is_available() else None,
    )
    model.eval()
    if hasattr(model, "generation_config"):
        model.generation_config.temperature = None
        model.generation_config.top_p = None

    rows = load_jsonl(args.input_file)
    if args.limit is not None:
        rows = rows[: args.limit]

    retrieved_by_id = {}
    if args.retrieved_file:
        retrieved_rows = load_jsonl(args.retrieved_file)
        for row in retrieved_rows:
            row_id = row.get("id")
            if row_id is not None:
                retrieved_by_id[row_id] = row

    output_rows = []
    for row in rows:
        if retrieved_by_id:
            retrieved_row = retrieved_by_id.get(row.get("id"))
            if retrieved_row is not None:
                row = {
                    **row,
                    "ctxs": retrieved_row.get("ctxs", row.get("ctxs", [])),
                    "top_contexts": retrieved_row.get("top_contexts", row.get("top_contexts", [])),
                }
        prompt = row["prompt"]
        ctxs = (row.get("ctxs") or row.get("top_contexts") or [])[: args.top_k]
        candidates = []
        candidate_passage_ranks = []
        used_ctxs = []

        for rank, ctx in enumerate(ctxs):
            prefix = build_retrieval_prefix(prompt, ctx)
            if prefix is None:
                continue
            candidate = generate_for_prefix(model, tokenizer, prefix, args.max_new_tokens)
            candidates.append(candidate)
            candidate_passage_ranks.append(rank)
            used_ctxs.append(ctx)

        enriched = dict(row)
        enriched["candidates"] = candidates
        enriched["candidate_types"] = ["retrieval"] * len(candidates)
        enriched["candidate_passage_ranks"] = candidate_passage_ranks
        enriched["candidate_ctxs"] = used_ctxs
        output_rows.append(enriched)

    save_jsonl(output_rows, Path(args.output_file))
    print(f"Saved Type B candidates for {len(output_rows)} prompts to {args.output_file}")


if __name__ == "__main__":
    main()
