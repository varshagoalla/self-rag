#!/usr/bin/env python

import argparse
import json
import re
from pathlib import Path

from retrieval_lm.dpo_scripts.reward_utils import score_candidate, strip_prompt_prefix


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


def is_mostly_eos_tail(text):
    eos_count = text.count("</s>")
    stripped = text.replace("</s>", "").strip()
    return eos_count >= 5 and len(stripped) < 40


def preferred_candidate_is_valid(candidate, candidate_type, metrics):
    text = candidate or ""
    cleaned = metrics.get("cleaned_candidate", "").strip()
    if not cleaned:
        return False, "empty_cleaned_answer"
    if is_mostly_eos_tail(text):
        return False, "mostly_eos_tail"

    if candidate_type == "retrieval":
        if "[Retrieval]" not in text:
            return False, "missing_retrieval_token"
        if re.search(r"<paragraph>.*?</paragraph>", text, flags=re.DOTALL) is None:
            return False, "missing_paragraph_block"
    elif candidate_type == "no_retrieval":
        if "[No Retrieval]" not in text:
            return False, "missing_no_retrieval_token"

    return True, None


def preferred_answer_is_not_clearly_worse(chosen_metrics, rejected_metrics):
    chosen_match = chosen_metrics.get("answer_match", 0.0)
    rejected_match = rejected_metrics.get("answer_match", 0.0)
    chosen_f1 = chosen_metrics.get("answer_f1", 0.0)
    rejected_f1 = rejected_metrics.get("answer_f1", 0.0)

    if chosen_match < rejected_match:
        return False, "preferred_worse_answer_match"
    if chosen_match == 0.0 and rejected_match == 0.0 and chosen_f1 + 1e-8 < rejected_f1:
        return False, "preferred_worse_answer_f1"
    return True, None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-file", required=True)
    parser.add_argument("--output-file", required=True)
    parser.add_argument("--pair-type", required=True, choices=["type_a", "type_c"])
    args = parser.parse_args()

    rows = load_jsonl(args.input_file)
    dpo_rows = []
    skipped = 0
    skipped_reasons = {}
    retrieval_preferred = 0
    no_retrieval_preferred = 0
    for row in rows:
        candidates = row.get("candidates", [])
        candidate_types = row.get("candidate_types", [])
        if len(candidates) < 2:
            skipped += 1
            skipped_reasons["missing_candidates"] = skipped_reasons.get("missing_candidates", 0) + 1
            continue

        typed_candidates = {}
        for idx, candidate in enumerate(candidates):
            candidate_type = candidate_types[idx] if idx < len(candidate_types) else None
            metrics = score_candidate(candidate, row)
            typed_candidates[candidate_type] = {
                "text": candidate,
                "candidate_type": candidate_type,
                **metrics,
            }

        retrieval_candidate = typed_candidates.get("retrieval")
        no_retrieval_candidate = typed_candidates.get("no_retrieval")
        oracle_label = row.get("retrieval_label")

        if retrieval_candidate is None or no_retrieval_candidate is None:
            skipped += 1
            skipped_reasons["missing_branch_candidate"] = skipped_reasons.get("missing_branch_candidate", 0) + 1
            continue

        if oracle_label == "[Retrieval]":
            chosen = retrieval_candidate
            rejected = no_retrieval_candidate
            retrieval_preferred += 1
        elif oracle_label == "[No Retrieval]":
            chosen = no_retrieval_candidate
            rejected = retrieval_candidate
            no_retrieval_preferred += 1
        else:
            skipped += 1
            skipped_reasons["missing_oracle_label"] = skipped_reasons.get("missing_oracle_label", 0) + 1
            continue

        valid, reason = preferred_candidate_is_valid(
            chosen["text"], chosen.get("candidate_type"), chosen
        )
        if not valid:
            skipped += 1
            skipped_reasons[reason] = skipped_reasons.get(reason, 0) + 1
            continue

        valid, reason = preferred_answer_is_not_clearly_worse(chosen, rejected)
        if not valid:
            skipped += 1
            skipped_reasons[reason] = skipped_reasons.get(reason, 0) + 1
            continue

        if args.pair_type == "type_a":
            dpo_rows.append(
                {
                    "id": row.get("id"),
                    "dataset_name": row.get("dataset_name"),
                    "prompt": row["prompt"],
                    "chosen": strip_prompt_prefix(chosen["text"]).strip(),
                    "rejected": strip_prompt_prefix(rejected["text"]).strip(),
                    "pair_type": "type_a",
                    "chosen_type": chosen.get("candidate_type"),
                    "rejected_type": rejected.get("candidate_type"),
                    "reference_answer": row["reference_answer"],
                    "retrieval_label": row.get("retrieval_label"),
                    "support_label": row.get("support_label"),
                    "relevance_label": row.get("relevance_label"),
                }
            )

        elif args.pair_type == "type_c":
            oracle_output = (row.get("oracle_output") or "").strip()
            if oracle_output:
                oracle_completion = strip_prompt_prefix(oracle_output).strip()
                for generated in (retrieval_candidate, no_retrieval_candidate):
                    dpo_rows.append(
                        {
                            "id": row.get("id"),
                            "dataset_name": row.get("dataset_name"),
                            "prompt": row["prompt"],
                            "chosen": oracle_completion,
                            "rejected": strip_prompt_prefix(generated["text"]).strip(),
                            "pair_type": "type_c",
                            "chosen_type": "oracle",
                            "rejected_type": generated.get("candidate_type"),
                            "reference_answer": row["reference_answer"],
                            "retrieval_label": row.get("retrieval_label"),
                            "support_label": row.get("support_label"),
                            "relevance_label": row.get("relevance_label"),
                        }
                    )

    save_jsonl(dpo_rows, Path(args.output_file))
    print(f"Saved {len(dpo_rows)} DPO pairs to {args.output_file}")
    print(f"Skipped {skipped} prompts")
    print(f"Retrieval preferred: {retrieval_preferred}")
    print(f"No-retrieval preferred: {no_retrieval_preferred}")
    print(f"Skipped breakdown: {json.dumps(skipped_reasons, sort_keys=True)}")


if __name__ == "__main__":
    main()
