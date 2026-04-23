#!/usr/bin/env python

import argparse
import json
from pathlib import Path


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


def extract_query(row):
    dataset = row.get("dataset_name")
    instruction = (row.get("instruction") or "").strip()
    input_text = (row.get("input") or "").strip()

    if dataset == "fever" and input_text:
        return input_text
    if input_text:
        return input_text
    return instruction


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-file", required=True)
    parser.add_argument("--output-file", required=True)
    args = parser.parse_args()

    rows = load_jsonl(args.input_file)
    output_rows = []
    for row in rows:
        query = extract_query(row)
        output_rows.append(
            {
                "id": row.get("id"),
                "dataset_name": row.get("dataset_name"),
                "question": query,
            }
        )

    save_jsonl(output_rows, Path(args.output_file))
    print(f"Saved {len(output_rows)} retrieval queries to {args.output_file}")


if __name__ == "__main__":
    main()
