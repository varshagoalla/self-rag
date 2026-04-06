import argparse
import json

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


PROMPT_DICT = {
    "prompt_input": (
        "### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n### Response:\n"
    ),
    "prompt_no_input": (
        "### Instruction:\n{instruction}\n\n### Response:\n"
    ),
}


def build_prompt(example):
    if "instruction" not in example and "question" in example:
        example = dict(example)
        example["instruction"] = example["question"]
    if example.get("input", "") != "":
        return PROMPT_DICT["prompt_input"].format_map(example)
    return PROMPT_DICT["prompt_no_input"].format_map(example)


def is_retrieval_example(example):
    return (
        ("ctxs" in example and len(example["ctxs"]) > 0)
        or ("top_contexts" in example and len(example["top_contexts"]) > 0)
    )


def get_evidences(example, ndocs):
    if "ctxs" in example and example["ctxs"]:
        return example["ctxs"][:ndocs]
    if "top_contexts" in example and example["top_contexts"]:
        return example["top_contexts"][:ndocs]
    return []


def build_evidence_augmented_prompt(prompt, evidence):
    title = evidence.get("title", "")
    text = evidence.get("text", "")
    return f"{prompt}[Retrieval]<paragraph>{title}\n{text}</paragraph>"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--train_file", required=True)
    parser.add_argument("--num_examples", type=int, default=10)
    parser.add_argument("--max_new_tokens", type=int, default=200)
    parser.add_argument("--ndocs", type=int, default=3)
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model_path, use_fast=False)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=torch.float16,
        device_map="auto",
    )
    model.eval()

    examples = []
    skipped_missing_ctx = 0
    with open(args.train_file) as f:
        for line in f:
            example = json.loads(line)
            if not is_retrieval_example(example):
                continue
            if not get_evidences(example, args.ndocs):
                skipped_missing_ctx += 1
                continue
            examples.append(example)
            if len(examples) >= args.num_examples:
                break

    if not examples:
        raise ValueError("No matching examples found in the training file.")

    if skipped_missing_ctx:
        print(f"Skipped {skipped_missing_ctx} retrieval examples without ctxs/top_contexts.")

    for i, example in enumerate(examples, start=1):
        prompt = build_prompt(example)
        evidences = get_evidences(example, args.ndocs)

        print("=" * 80)
        print(f"Example {i}")
        print("-" * 80)
        for j, evidence in enumerate(evidences, start=1):
            generation_prompt = build_evidence_augmented_prompt(prompt, evidence)
            inputs = tokenizer(generation_prompt, return_tensors="pt").to(model.device)
            with torch.no_grad():
                output = model.generate(
                    **inputs,
                    max_new_tokens=args.max_new_tokens,
                    do_sample=False,
                )

            input_len = inputs["input_ids"].shape[1]
            generated_ids = output[0][input_len:]
            response_only = tokenizer.decode(generated_ids, skip_special_tokens=False)

            print(f"CTX {j}: {evidence.get('title', '')}")
            print("FULL INPUT TO MODEL:")
            print(generation_prompt)
            print("-" * 80)
            print("MODEL OUTPUT:")
            print(response_only)
            print("-" * 80)
        print()


if __name__ == "__main__":
    main()
