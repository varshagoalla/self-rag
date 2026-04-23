#!/usr/bin/env python

import argparse
import json
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


def is_valid_retrieval_candidate(candidate, metrics):
    text = candidate or ""
    if "[Retrieval]" not in text:
        return False
    if "<paragraph>" not in text or "</paragraph>" not in text:
        return False
    if not metrics.get("cleaned_candidate", "").strip():
        return False
    return True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-file", required=True)
    parser.add_argument("--output-file", required=True)
    parser.add_argument("--min-score-gap", type=float, default=0.15)
    args = parser.parse_args()

    rows = load_jsonl(args.input_file)
    dpo_rows = []
    skipped = 0
    skipped_reasons = {}

    for row in rows:
        candidates = row.get("candidates", [])
        passage_ranks = row.get("candidate_passage_ranks", [])
        if len(candidates) < 2:
            skipped += 1
            skipped_reasons["not_enough_retrieval_paths"] = skipped_reasons.get("not_enough_retrieval_paths", 0) + 1
            continue

        scored = []
        for idx, candidate in enumerate(candidates):
            metrics = score_candidate(candidate, row)
            if not is_valid_retrieval_candidate(candidate, metrics):
                continue
            scored.append(
                {
                    "text": candidate,
                    "passage_rank": passage_ranks[idx] if idx < len(passage_ranks) else idx,
                    **metrics,
                }
            )

        if len(scored) < 2:
            skipped += 1
            skipped_reasons["not_enough_valid_paths"] = skipped_reasons.get("not_enough_valid_paths", 0) + 1
            continue

        scored.sort(
            key=lambda x: (
                x["reward"],
                x["answer_match"],
                x["answer_f1"],
                -x["passage_rank"],
            ),
            reverse=True,
        )

        best = scored[0]
        worst = scored[-1]
        if best["reward"] - worst["reward"] < args.min_score_gap:
            skipped += 1
            skipped_reasons["score_gap_too_small"] = skipped_reasons.get("score_gap_too_small", 0) + 1
            continue

        dpo_rows.append(
            {
                "id": row.get("id"),
                "dataset_name": row.get("dataset_name"),
                "prompt": row["prompt"],
                "chosen": strip_prompt_prefix(best["text"]).strip(),
                "rejected": strip_prompt_prefix(worst["text"]).strip(),
                "pair_type": "type_b",
                "chosen_type": "retrieval",
                "rejected_type": "retrieval",
                "chosen_passage_rank": best["passage_rank"],
                "rejected_passage_rank": worst["passage_rank"],
                "reference_answer": row.get("reference_answer"),
                "retrieval_label": row.get("retrieval_label"),
                "support_label": row.get("support_label"),
                "relevance_label": row.get("relevance_label"),
            }
        )

    save_jsonl(dpo_rows, Path(args.output_file))
    print(f"Saved {len(dpo_rows)} Type B DPO pairs to {args.output_file}")
    print(f"Skipped {skipped} prompts")
    print(f"Skipped breakdown: {json.dumps(skipped_reasons, sort_keys=True)}")


if __name__ == "__main__":
    main()
