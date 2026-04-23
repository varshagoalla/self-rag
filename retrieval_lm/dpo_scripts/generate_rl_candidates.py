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


def generate_candidates_for_prefix(
    model,
    tokenizer,
    prefix_text,
    num_candidates,
    max_new_tokens,
    do_sample,
    temperature,
    top_p,
):
    inputs = tokenizer(prefix_text, return_tensors="pt").to(model.device)
    generate_kwargs = {
        "max_new_tokens": max_new_tokens,
        "num_return_sequences": num_candidates,
        "pad_token_id": tokenizer.eos_token_id,
        "do_sample": do_sample,
    }
    if do_sample:
        generate_kwargs["temperature"] = temperature
        generate_kwargs["top_p"] = top_p

    with torch.no_grad():
        generated = model.generate(
            **inputs,
            **generate_kwargs,
        )

    prompt_len = inputs["input_ids"].shape[1]
    outputs = []
    for sequence in generated:
        continuation = tokenizer.decode(sequence[prompt_len:], skip_special_tokens=False)
        outputs.append(prefix_text + continuation)
    return outputs


def build_retrieval_prefix(prompt, paragraph):
    paragraph = (paragraph or "").strip()
    if not paragraph:
        return None
    return prompt + f"[Retrieval]<paragraph>{paragraph}</paragraph>"


def get_retrieved_paragraph(row):
    ctxs = row.get("ctxs") or row.get("top_contexts") or []
    if not ctxs:
        return None, "missing"
    first_ctx = ctxs[0]
    title = (first_ctx.get("title") or "").strip()
    text = (first_ctx.get("text") or "").strip()
    if not text:
        return None, "missing"
    paragraph = f"{title}\n{text}".strip() if title else text
    return paragraph, "retrieved_top1"


def get_oracle_paragraph(row):
    paragraph = (row.get("oracle_paragraph") or "").strip()
    if not paragraph:
        return None, "missing"
    return paragraph, "oracle"


def choose_retrieval_paragraph(row):
    oracle_label = row.get("retrieval_label")
    if oracle_label == "[Retrieval]":
        paragraph, source = get_oracle_paragraph(row)
        if paragraph:
            return paragraph, source
        return get_retrieved_paragraph(row)
    return get_retrieved_paragraph(row)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name-or-path", required=True)
    parser.add_argument("--input-file", required=True)
    parser.add_argument("--retrieved-file", default=None)
    parser.add_argument("--output-file", required=True)
    parser.add_argument("--num-candidates", type=int, default=2)
    parser.add_argument("--num-no-retrieval", type=int, default=1)
    parser.add_argument("--num-retrieval", type=int, default=1)
    parser.add_argument("--max-new-tokens", type=int, default=96)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--do-sample", action="store_true")
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    if args.num_no_retrieval + args.num_retrieval != args.num_candidates:
        raise ValueError("--num-no-retrieval + --num-retrieval must equal --num-candidates")

    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, use_fast=False)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        device_map="auto" if torch.cuda.is_available() else None,
    )
    model.eval()
    if not args.do_sample:
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
        paragraph, paragraph_source = choose_retrieval_paragraph(row)

        candidates = []
        candidate_types = []

        no_ret_prefix = prompt + "[No Retrieval]"
        candidates.extend(
            generate_candidates_for_prefix(
                model,
                tokenizer,
                no_ret_prefix,
                args.num_no_retrieval,
                args.max_new_tokens,
                args.do_sample,
                args.temperature,
                args.top_p,
            )
        )
        candidate_types.extend(["no_retrieval"] * args.num_no_retrieval)

        retrieval_prefix = build_retrieval_prefix(prompt, paragraph)
        if retrieval_prefix is not None and args.num_retrieval > 0:
            candidates.extend(
                generate_candidates_for_prefix(
                    model,
                    tokenizer,
                    retrieval_prefix,
                    args.num_retrieval,
                    args.max_new_tokens,
                    args.do_sample,
                    args.temperature,
                    args.top_p,
                )
            )
            candidate_types.extend(["retrieval"] * args.num_retrieval)

        enriched = dict(row)
        enriched["candidates"] = candidates[: args.num_candidates]
        enriched["candidate_types"] = candidate_types[: args.num_candidates]
        enriched["retrieval_paragraph_source"] = paragraph_source
        output_rows.append(enriched)

    save_jsonl(output_rows, Path(args.output_file))
    print(f"Saved candidates for {len(output_rows)} prompts to {args.output_file}")


if __name__ == "__main__":
    main()
