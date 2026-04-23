#!/usr/bin/env python

import argparse
import json
import re
import time
from pathlib import Path

from passage_retrieval import Retriever, load_data


def save_jsonl(rows, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")


def normalize_whitespace(text):
    return " ".join((text or "").split()).strip()


def extract_clean_query(row):
    dataset = row.get("dataset_name")
    raw_question = row.get("question") or ""
    raw_instruction = row.get("instruction") or ""
    source = raw_question if raw_question else raw_instruction
    source = source.strip()

    if dataset == "fever":
        if "## Input:" in source:
            source = source.split("## Input:", 1)[1]
        return normalize_whitespace(source)

    if dataset == "asqa":
        if "## Input:" in source:
            source = source.split("## Input:", 1)[1]
        return normalize_whitespace(source)

    if dataset == "nq":
        if "## Input:" in source:
            source = source.split("## Input:", 1)[1]
        return normalize_whitespace(source)

    if dataset in {"arc_easy", "obqa"}:
        if "## Input:" in source:
            source = source.split("## Input:", 1)[1]
        source = source.strip()
        option_match = re.search(r"\nA:\s", source)
        if option_match is not None:
            stem = source[: option_match.start()].strip()
            choices = source[option_match.start() :].strip()
            return f"{stem}\n{choices}".strip()
        return normalize_whitespace(source)

    return normalize_whitespace(source)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-file", required=True)
    parser.add_argument("--output-file", required=True)
    parser.add_argument("--passages", required=True)
    parser.add_argument("--passages-embeddings", required=True)
    parser.add_argument("--model-name-or-path", default="facebook/contriever-msmarco")
    parser.add_argument("--n-docs", type=int, default=5)
    parser.add_argument("--query-batch-size", type=int, default=128)
    parser.add_argument("--per-gpu-batch-size", type=int, default=64)
    parser.add_argument("--question-maxlength", type=int, default=512)
    parser.add_argument("--indexing-batch-size", type=int, default=1000000)
    parser.add_argument("--projection-size", type=int, default=768)
    parser.add_argument("--n-subquantizers", type=int, default=0)
    parser.add_argument("--n-bits", type=int, default=8)
    parser.add_argument("--save-or-load-index", action="store_true")
    parser.add_argument("--no-fp16", action="store_true")
    parser.add_argument("--lowercase", action="store_true")
    parser.add_argument("--normalize-text", action="store_true")
    args = parser.parse_args()

    retriever = Retriever(args)
    retriever.setup_retriever()

    rows = load_data(args.input_file)
    output_rows = []
    total_queries = 0
    retrieval_start = time.time()

    for start in range(0, len(rows), args.query_batch_size):
        batch_rows = rows[start : start + args.query_batch_size]
        batch_queries = []
        batch_kept_rows = []
        for row in batch_rows:
            query = extract_clean_query(row)
            if not query:
                continue
            batch_queries.append(query)
            batch_kept_rows.append(row)

        if not batch_queries:
            continue

        questions_embedding = retriever.embed_queries(args, batch_queries)
        top_ids_and_scores = retriever.index.search_knn(questions_embedding, args.n_docs)

        for row, (doc_ids, _scores) in zip(batch_kept_rows, top_ids_and_scores):
            docs = [retriever.passage_id_map[doc_id] for doc_id in doc_ids][: args.n_docs]
            enriched = dict(row)
            enriched["ctxs"] = docs
            output_rows.append(enriched)

        total_queries += len(batch_queries)
        elapsed = time.time() - retrieval_start
        print(
            f"Processed {total_queries}/{len(rows)} queries "
            f"({elapsed:.1f}s elapsed, batch_size={len(batch_queries)})",
            flush=True,
        )

    save_jsonl(output_rows, Path(args.output_file))
    print(f"Saved retrieval results for {len(output_rows)} queries to {args.output_file}")


if __name__ == "__main__":
    main()
