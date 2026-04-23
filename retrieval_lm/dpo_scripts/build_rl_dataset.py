#!/usr/bin/env python

import argparse
import json
import random
import re
from collections import Counter, defaultdict
from pathlib import Path


PROMPT_DICT = {
    "prompt_input": "### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n### Response:\n",
    "prompt_no_input": "### Instruction:\n{instruction}\n\n### Response:\n",
}

CONTROL_TOKENS = [
    "[Fully supported]",
    "[Partially supported]",
    "[No support / Contradictory]",
    "[No Retrieval]",
    "[Retrieval]",
    "[Irrelevant]",
    "[Relevant]",
    "<paragraph>",
    "</paragraph>",
    "[Utility:1]",
    "[Utility:2]",
    "[Utility:3]",
    "[Utility:4]",
    "[Utility:5]",
]


ALLOWED_DATASETS = {"nq", "fever", "obqa", "arc_easy", "asqa"}
CLOSED_DATASETS = {"fever", "obqa", "arc_easy"}

RETRIEVAL_SUBTYPE_WEIGHTS = {
    "fully_supported": 0.45,
    "partially_supported": 0.20,
    "no_support": 0.15,
    "irrelevant": 0.15,
    "other": 0.05,
}

NO_RETRIEVAL_SUBTYPE_WEIGHTS = {
    "utility_1": 0.15,
    "utility_2": 0.10,
    "utility_3": 0.05,
    "utility_4": 0.10,
    "utility_5": 0.55,
    "other": 0.05,
}


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


def extract_first_present(text, candidates):
    for candidate in candidates:
        if candidate in text:
            return candidate
    return None


def extract_paragraph_text(text):
    match = re.search(r"<paragraph>(.*?)</paragraph>", text, flags=re.DOTALL)
    if match is None:
        return None
    paragraph = " ".join(match.group(1).split())
    return paragraph if paragraph else None


def strip_paragraph_blocks(text):
    return re.sub(r"<paragraph>.*?</paragraph>", " ", text, flags=re.DOTALL)


def strip_control_tokens(text):
    text = strip_paragraph_blocks(text)
    for token in CONTROL_TOKENS:
        text = text.replace(token, " ")
    text = text.replace("</s>", " ")
    return " ".join(text.split())


def infer_reference_answer(output_text):
    return strip_control_tokens(output_text)


def build_behavior_labels(output_text):
    return {
        "retrieval_label": extract_first_present(output_text, ["[No Retrieval]", "[Retrieval]"]),
        "relevance_label": extract_first_present(output_text, ["[Relevant]", "[Irrelevant]"]),
        "support_label": extract_first_present(
            output_text,
            ["[Fully supported]", "[Partially supported]", "[No support / Contradictory]"],
        ),
        "utility_label": extract_first_present(
            output_text,
            ["[Utility:1]", "[Utility:2]", "[Utility:3]", "[Utility:4]", "[Utility:5]"],
        ),
    }


def make_prompt(row):
    example = {
        "instruction": row["instruction"],
        "input": row.get("input", ""),
    }
    template = PROMPT_DICT["prompt_input"] if example.get("input") else PROMPT_DICT["prompt_no_input"]
    return template.format_map(example)


def task_form(dataset_name):
    return "closed" if dataset_name in CLOSED_DATASETS else "open"


def retrieval_family(row):
    return "retrieval" if row.get("retrieval_label") == "[Retrieval]" else "no_retrieval"


def retrieval_subtype(row):
    if row.get("retrieval_label") != "[Retrieval]":
        return None
    if row.get("relevance_label") == "[Irrelevant]":
        return "irrelevant"
    support = row.get("support_label")
    if support == "[Fully supported]":
        return "fully_supported"
    if support == "[Partially supported]":
        return "partially_supported"
    if support == "[No support / Contradictory]":
        return "no_support"
    return "other"


def no_retrieval_subtype(row):
    if row.get("retrieval_label") != "[No Retrieval]":
        return None
    utility = row.get("utility_label")
    if utility == "[Utility:1]":
        return "utility_1"
    if utility == "[Utility:2]":
        return "utility_2"
    if utility == "[Utility:3]":
        return "utility_3"
    if utility == "[Utility:4]":
        return "utility_4"
    if utility == "[Utility:5]":
        return "utility_5"
    return "other"


def preferred_subtype(row):
    family = retrieval_family(row)
    return retrieval_subtype(row) if family == "retrieval" else no_retrieval_subtype(row)


def cap_weighted_targets(total, available_counts, preferred_weights):
    """Allocate `total` examples across subtypes, respecting availability."""
    if total <= 0 or not available_counts:
        return {name: 0 for name in available_counts}

    available_total = sum(available_counts.values())
    total = min(total, available_total)

    targets = {name: 0 for name in available_counts}
    if total == 0:
        return targets

    weighted = []
    for name, available in available_counts.items():
        weight = preferred_weights.get(name, preferred_weights.get("other", 0.0))
        exact = total * weight
        target = min(available, int(exact))
        targets[name] = target
        weighted.append((name, exact - int(exact)))

    assigned = sum(targets.values())
    remaining = total - assigned
    weighted.sort(key=lambda item: item[1], reverse=True)

    while remaining > 0:
        progressed = False
        for name, _ in weighted:
            if remaining <= 0:
                break
            if targets[name] >= available_counts[name]:
                continue
            targets[name] += 1
            remaining -= 1
            progressed = True
        if not progressed:
            break

    if remaining > 0:
        for name, available in sorted(available_counts.items(), key=lambda item: item[1] - targets[item[0]], reverse=True):
            if remaining <= 0:
                break
            room = available - targets[name]
            if room <= 0:
                continue
            take = min(room, remaining)
            targets[name] += take
            remaining -= take

    return targets


def redistribute_cell_targets(size, cell_buckets):
    """Balance across (task_form, retrieval_family) cells with graceful fallback."""
    cells = list(cell_buckets.keys())
    base = size // len(cells)
    targets = {cell: min(base, len(cell_buckets[cell])) for cell in cells}
    remainder = size - sum(targets.values())

    while remainder > 0:
        progressed = False
        for cell in cells:
            if remainder <= 0:
                break
            if targets[cell] >= len(cell_buckets[cell]):
                continue
            targets[cell] += 1
            remainder -= 1
            progressed = True
        if not progressed:
            break

    return targets


def sample_round_robin_by_dataset(rows, target_count, rng):
    by_dataset = defaultdict(list)
    for row in rows:
        by_dataset[row["dataset_name"]].append(row)
    for dataset_rows in by_dataset.values():
        rng.shuffle(dataset_rows)

    dataset_names = sorted(by_dataset.keys())
    sampled = []
    while len(sampled) < target_count:
        progressed = False
        for name in dataset_names:
            if len(sampled) >= target_count:
                break
            if not by_dataset[name]:
                continue
            sampled.append(by_dataset[name].pop())
            progressed = True
        if not progressed:
            break
    return sampled


def sample_split(rows, size, rng):
    rows = list(rows)
    rng.shuffle(rows)

    cell_buckets = defaultdict(list)
    for row in rows:
        cell = (task_form(row["dataset_name"]), retrieval_family(row))
        cell_buckets[cell].append(row)

    cell_targets = redistribute_cell_targets(
        size,
        {
            ("closed", "retrieval"): cell_buckets.get(("closed", "retrieval"), []),
            ("closed", "no_retrieval"): cell_buckets.get(("closed", "no_retrieval"), []),
            ("open", "retrieval"): cell_buckets.get(("open", "retrieval"), []),
            ("open", "no_retrieval"): cell_buckets.get(("open", "no_retrieval"), []),
        },
    )

    sampled = []
    used_ids = set()

    for cell, target in cell_targets.items():
        cell_rows = [row for row in cell_buckets.get(cell, []) if row["id"] not in used_ids]
        family = cell[1]
        subtype_fn = retrieval_subtype if family == "retrieval" else no_retrieval_subtype
        subtype_weights = RETRIEVAL_SUBTYPE_WEIGHTS if family == "retrieval" else NO_RETRIEVAL_SUBTYPE_WEIGHTS

        subtype_buckets = defaultdict(list)
        for row in cell_rows:
            subtype_buckets[subtype_fn(row)].append(row)

        available_counts = {name: len(bucket) for name, bucket in subtype_buckets.items()}
        subtype_targets = cap_weighted_targets(target, available_counts, subtype_weights)

        for subtype, subtype_target in subtype_targets.items():
            if subtype_target <= 0:
                continue
            subtype_rows = [row for row in subtype_buckets[subtype] if row["id"] not in used_ids]
            chosen = sample_round_robin_by_dataset(subtype_rows, subtype_target, rng)
            for row in chosen:
                if row["id"] in used_ids:
                    continue
                sampled.append(row)
                used_ids.add(row["id"])

        remaining = target - sum(1 for row in sampled if (task_form(row["dataset_name"]), retrieval_family(row)) == cell)
        if remaining > 0:
            leftovers = [row for row in cell_rows if row["id"] not in used_ids]
            chosen = sample_round_robin_by_dataset(leftovers, remaining, rng)
            for row in chosen:
                if row["id"] in used_ids:
                    continue
                sampled.append(row)
                used_ids.add(row["id"])

    if len(sampled) < size:
        leftovers = [row for row in rows if row["id"] not in used_ids]
        chosen = sample_round_robin_by_dataset(leftovers, size - len(sampled), rng)
        for row in chosen:
            if row["id"] in used_ids:
                continue
            sampled.append(row)
            used_ids.add(row["id"])

    return sampled, used_ids


def summarize(rows, name):
    print(f"{name} rows: {len(rows)}")
    print(f"{name} dataset mix:", Counter(row["dataset_name"] for row in rows))
    print(f"{name} task form mix:", Counter(task_form(row["dataset_name"]) for row in rows))
    print(f"{name} retrieval mix:", Counter(row.get("retrieval_label") for row in rows))
    print(f"{name} retrieval subtype mix:", Counter(retrieval_subtype(row) for row in rows if retrieval_family(row) == 'retrieval'))
    print(f"{name} no-retrieval utility mix:", Counter(no_retrieval_subtype(row) for row in rows if retrieval_family(row) == 'no_retrieval'))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-file", default="../train_data/train.jsonl")
    parser.add_argument("--output-dir", default="../rl_data")
    parser.add_argument("--train-size", type=int, default=12000)
    parser.add_argument("--valid-size", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    rng = random.Random(args.seed)
    rows = load_jsonl(args.input_file)

    filtered = []
    for row in rows:
        dataset_name = row.get("dataset_name")
        if dataset_name not in ALLOWED_DATASETS:
            continue

        reference_answer = infer_reference_answer(row["output"])
        if not reference_answer:
            continue

        labels = build_behavior_labels(row["output"])
        if labels.get("retrieval_label") not in {"[Retrieval]", "[No Retrieval]"}:
            continue
        record = {
            "id": row.get("id"),
            "dataset_name": dataset_name,
            "instruction": row["instruction"],
            "input": row.get("input", ""),
            "prompt": make_prompt(row),
            "oracle_output": row["output"],
            "oracle_paragraph": extract_paragraph_text(row["output"]),
            "reference_answer": reference_answer,
            "answers": [reference_answer],
            **labels,
        }
        filtered.append(record)

    train_rows, train_ids = sample_split(filtered, args.train_size, rng)
    valid_pool = [row for row in filtered if row["id"] not in train_ids]
    valid_rows, _ = sample_split(valid_pool, args.valid_size, rng)

    output_dir = Path(args.output_dir)
    save_jsonl(train_rows, output_dir / "train_rl.jsonl")
    save_jsonl(valid_rows, output_dir / "valid_rl.jsonl")

    print(f"Saved {len(train_rows)} train rows to {output_dir / 'train_rl.jsonl'}")
    print(f"Saved {len(valid_rows)} valid rows to {output_dir / 'valid_rl.jsonl'}")
    summarize(train_rows, "Train")
    summarize(valid_rows, "Valid")


if __name__ == "__main__":
    main()
