# Self-RAG DPO 

This directory contains the data artifacts used to preference-tune Self-RAG with DPO. Self-RAG is a 7B retrieval-augmented model trained to emit reflection/control tokens such as `[Retrieval]`, `[No Retrieval]`, `[Relevant]`, `[Fully supported]`, and `[Utility:*]`. Our project builds preference pairs from those retrieval-aware trajectories and uses QLoRA + DPO to improve the model's retrieval and answer behavior.

The data artifacts are in:

```text
self-rag/retrieval_lm/dpo_data
```

The scripts used to create them are archived in:

```text
self-rag/retrieval_lm/dpo_scripts
```

## Pipeline Overview

The full data pipeline is:

1. Curate a 12k source set from the original Self-RAG training data.
2. Extract clean retrieval queries.
3. Retrieve top passages with Contriever.
4. Generate retrieval/no-retrieval candidates with Self-RAG.
5. Build Type A and Type C DPO pairs from those candidates.
6. Generate retrieval-only candidates for Type B.
7. Build Type B DPO pairs from retrieval-only candidates.
8. Mix Type A/B/C into 20k, 25k, and full-data DPO datasets.
9. Train with `train_selfrag_dpo.py`.
10. Merge LoRA adapters and evaluate on short-form tasks.

## Stage 0: Original Self-RAG Training Data

The original Self-RAG generator training file at `train_data/train.jsonl` contains 145,619 examples across 11 datasets. 

| Dataset Name in `train.jsonl` | Paper Name | Category | Rows |
|---|---|---|---:|
| `gpt4_alpaca` | GPT-4 Alpaca | instruction-following | 26,168 |
| `stanford_alpaca` | Stanford Alpaca | instruction-following | 25,153 |
| `flan_v2` | FLAN-V2 | instruction-following | 17,817 |
| `wow` | Wizard of Wikipedia | knowledge-intensive | 17,367 |
| `nq` | Natural Questions | knowledge-intensive | 15,535 |
| `sharegpt` | ShareGPT | instruction-following | 13,406 |
| `fever` | FEVER | knowledge-intensive/fact verification | 9,966 |
| `oasst1` | Open Assistant 1 | instruction-following | 9,464 |
| `obqa` | OpenBookQA | knowledge-intensive MCQ | 4,699 |
| `asqa` | ASQA | knowledge-intensive ambiguous QA | 3,897 |
| `arc_easy` | ARC-Easy | knowledge-intensive MCQ | 2,147 |
| **Total** |  |  | **145,619** |

The instruction-following datasets preserve general instruction-following behavior. The knowledge-intensive datasets are more directly useful for training retrieval and reflection behavior. For this DPO project, we focused on the knowledge-intensive datasets that support retrieval/no-retrieval preference construction.

## Stage 1: Curate the 12k Source Set

Script:

```text
dpo_scripts/build_rl_dataset.py
```

Input:

```text
train_data/train.jsonl
```

Outputs:

```text
dpo_data/train_rl.jsonl
dpo_data/valid_rl.jsonl
```

This stage filters the original Self-RAG training data to five datasets:

```text
fever, nq, asqa, arc_easy, obqa
```


For each kept row, the script extracts:

- `prompt`: formatted from the original instruction/input
- `oracle_output`: the original Self-RAG target response
- `oracle_paragraph`: first `<paragraph>...</paragraph>` block when present
- `reference_answer`: oracle output after removing paragraphs and control tokens
- `retrieval_label`: `[Retrieval]` or `[No Retrieval]`
- `relevance_label`: `[Relevant]` or `[Irrelevant]`
- `support_label`: support/refutation token when present
- `utility_label`: `[Utility:1]` to `[Utility:5]`

Rows are kept only if they have a usable reference answer and a valid retrieval decision label.

Current `train_rl.jsonl` distribution:

| Dataset | Train Rows |
|---|---:|
| `fever` | 3,650 |
| `nq` | 3,267 |
| `asqa` | 2,733 |
| `arc_easy` | 1,209 |
| `obqa` | 1,141 |
| **Total** | **12,000** |

Retrieval decision balance:

| Retrieval Label | Train Rows |
|---|---:|
| `[Retrieval]` | 6,000 |
| `[No Retrieval]` | 6,000 |

Dataset by retrieval decision:

| Dataset | Retrieval | No Retrieval | Total |
|---|---:|---:|---:|
| `arc_easy` | 235 | 974 | 1,209 |
| `asqa` | 1,543 | 1,190 | 2,733 |
| `fever` | 2,658 | 992 | 3,650 |
| `nq` | 1,457 | 1,810 | 3,267 |
| `obqa` | 107 | 1,034 | 1,141 |

Sampling logic:

- The script first tries to balance across four cells: `closed + retrieval`, `closed + no_retrieval`, `open + retrieval`, and `open + no_retrieval`.
- For a 12k split, the intended target is roughly 3k examples per cell.
- This is best-effort: if a cell does not have enough examples after filtering, it is capped and the remaining quota is redistributed.
- Inside each cell, examples are grouped into subtypes and sampled using hand-set preferred weights.
- These weights are heuristics, not paper-derived constants or guaranteed final ratios.
- Within each subtype, rows are sampled round-robin across datasets after shuffling each dataset bucket.

Retrieval subtype preferences:

| Retrieval Subtype | Definition | Preferred Weight |
|---|---|---:|
| `fully_supported` | support label is `[Fully supported]` | 0.45 |
| `partially_supported` | support label is `[Partially supported]` | 0.20 |
| `no_support` | support label is `[No support / Contradictory]` | 0.15 |
| `irrelevant` | relevance label is `[Irrelevant]` | 0.15 |
| `other` | retrieval example outside the above buckets | 0.05 |

No-retrieval subtype preferences:

| No-Retrieval Subtype | Definition | Preferred Weight |
|---|---|---:|
| `utility_1` | utility label is `[Utility:1]` | 0.15 |
| `utility_2` | utility label is `[Utility:2]` | 0.10 |
| `utility_3` | utility label is `[Utility:3]` | 0.05 |
| `utility_4` | utility label is `[Utility:4]` | 0.10 |
| `utility_5` | utility label is `[Utility:5]` | 0.55 |
| `other` | no-retrieval example outside the above buckets | 0.05 |

Why these heuristics were used:

- For retrieval examples, we wanted mostly clean positive retrieval cases, while still keeping weak/noisy retrieval cases such as irrelevant or unsupported passages.
- For no-retrieval examples, `[Utility:5]` was preferred because these are usually strong cases where the model can answer without retrieval.
- Lower-utility no-retrieval examples were still retained for diversity.

The validation split is sampled after train from the remaining pool. Current `valid_rl.jsonl` distribution:

| Dataset | Valid Rows |
|---|---:|
| `fever` | 399 |
| `nq` | 305 |
| `arc_easy` | 100 |
| `obqa` | 99 |
| `asqa` | 97 |
| **Total** | **1,000** |

Example command:

```bash
python dpo_scripts/build_rl_dataset.py \
  --input-file train_data/train.jsonl \
  --output-dir dpo_data \
  --train-size 12000 \
  --valid-size 1000 \
  --seed 42
```

## Stage 2: Extract Retrieval Queries

Script:

```text
dpo_scripts/extract_retrieval_queries.py
```

Inputs:

```text
dpo_data/train_rl.jsonl
dpo_data/valid_rl.jsonl
```

Outputs:

```text
dpo_data/train_rl_retrieval_queries.jsonl
dpo_data/valid_rl_retrieval_queries.jsonl
```

This script creates query-only rows:

```text
id
dataset_name
question
```

At this stage, the script only extracts a raw query string from the example instruction/input and stores it in the `question` field. The more important dataset-specific cleanup, such as stripping instruction wrappers and keeping only the actual question/claim/choices, is done later inside `retrieve_queries.py` immediately before retrieval.

Example command:

```bash
python dpo_scripts/extract_retrieval_queries.py \
  --input-file dpo_data/train_rl.jsonl \
  --output-file dpo_data/train_rl_retrieval_queries.jsonl
```

## Stage 3: Retrieve Passages with Contriever

Script:

```text
dpo_scripts/retrieve_queries.py
```

Inputs:

```text
dpo_data/train_rl_retrieval_queries.jsonl
dpo_data/valid_rl_retrieval_queries.jsonl
```

Outputs:

```text
dpo_data/train_rl_retrieved.jsonl
dpo_data/valid_rl_retrieved.jsonl
```

This stage retrieves top passages with `facebook/contriever-msmarco` from the Wikipedia passage index. The output rows contain the original query metadata plus:

```text
ctxs
```

where `ctxs` is the list of retrieved passages.

`retrieve_queries.py` applies the dataset-specific query cleanup before sending queries to Contriever. This is the step where the raw `question` field is converted into the actual retrieval query:

- `fever`: claim only
- `nq`: question only
- `asqa`: question only
- `arc_easy` / `obqa`: question stem plus choices, without the generic instruction wrapper

Example command:

```bash
python dpo_scripts/retrieve_queries.py \
  --input-file dpo_data/train_rl_retrieval_queries.jsonl \
  --output-file dpo_data/train_rl_retrieved.jsonl \
  --model-name-or-path facebook/contriever-msmarco \
  --passages /path/to/psgs_w100.tsv \
  --passages-embeddings "/path/to/wikipedia_embeddings/*" \
  --n-docs 5 \
  --query-batch-size 128 \
  --per-gpu-batch-size 64 \
  --save-or-load-index
```

## Stage 4: Generate Type A/C Candidates

Script:

```text
dpo_scripts/generate_rl_candidates.py
```

Inputs:

```text
dpo_data/train_rl.jsonl
dpo_data/train_rl_retrieved.jsonl
```

Output:

```text
dpo_data/train_rl_candidates.jsonl
```

This stage runs Self-RAG generation to produce two candidate branches per prompt:

```text
no_retrieval candidate: prompt + [No Retrieval]
retrieval candidate:    prompt + [Retrieval]<paragraph>...</paragraph>
```

The retrieval paragraph is chosen using the hybrid rule:

- If the oracle label is `[Retrieval]`, use `oracle_paragraph`.
- If `oracle_paragraph` is missing, fall back to top-1 retrieved passage.
- If the oracle label is `[No Retrieval]`, use top-1 retrieved passage, because no oracle paragraph exists for no-retrieval oracle rows.

The script can join `ctxs` from `--retrieved-file` onto full RL rows by `id`, so the retrieved file does not need to be pre-merged with `train_rl.jsonl`.

Example command:

```bash
python dpo_scripts/generate_rl_candidates.py \
  --model-name-or-path /path/to/selfrag_llama2_7b \
  --input-file dpo_data/train_rl.jsonl \
  --retrieved-file dpo_data/train_rl_retrieved.jsonl \
  --output-file dpo_data/train_rl_candidates.jsonl
```

## Stage 5: Build Type A Data

Scripts:

```text
dpo_scripts/build_dpo_pairs.py
dpo_scripts/build_type_a_c_pairs.py
```

Input:

```text
dpo_data/train_rl_candidates.jsonl
```

Output:

```text
dpo_data/train_rl_dpo_type_a.jsonl
```

Type A compares generated retrieval vs generated no-retrieval for the same prompt.

Pair construction:

- If oracle label is `[Retrieval]`, `chosen` is the retrieval candidate and `rejected` is the no-retrieval candidate.
- If oracle label is `[No Retrieval]`, `chosen` is the no-retrieval candidate and `rejected` is the retrieval candidate.

Sanity filtering:

- The preferred candidate must contain the expected control token.
- Retrieval candidates must contain a `<paragraph>...</paragraph>` block.
- Empty or mostly EOS outputs are skipped.
- If the oracle-preferred branch is clearly worse than the rejected branch by answer match/F1, the pair is skipped.

Current output:

```text
train_rl_dpo_type_a.jsonl: 10,853 pairs
```

Example command:

```bash
python dpo_scripts/build_dpo_pairs.py \
  --input-file dpo_data/train_rl_candidates.jsonl \
  --output-file dpo_data/train_rl_dpo_type_a.jsonl \
  --pair-type type_a
```

## Stage 6: Build Type C Data

Scripts:

```text
dpo_scripts/build_dpo_pairs.py
dpo_scripts/build_type_a_c_pairs.py
```

Input:

```text
dpo_data/train_rl_candidates.jsonl
```

Output:

```text
dpo_data/train_rl_dpo_type_c.jsonl
```

Type C compares the original oracle Self-RAG response against generated candidates.

Pair construction:

- `chosen` is always the oracle response from `oracle_output`, with the prompt prefix stripped.
- `rejected` is the generated retrieval candidate.
- `rejected` is also the generated no-retrieval candidate.
- Therefore each kept prompt can produce two Type C pairs.

In the current construction, Type C is generated after the same candidate validity and answer-quality sanity filtering used for Type A. Since Type A kept 10,853 prompts, Type C has exactly twice that number:

```text
train_rl_dpo_type_c.jsonl: 21,706 pairs
```

Example wrapper command for Type A and Type C together:

```bash
python dpo_scripts/build_type_a_c_pairs.py \
  --input-file dpo_data/train_rl_candidates.jsonl \
  --output-dir dpo_data \
  --prefix train_rl_dpo
```

This writes:

```text
train_rl_dpo_type_a.jsonl
train_rl_dpo_type_c.jsonl
```

## Stage 7: Generate Type B Candidates

Script:

```text
dpo_scripts/generate_type_b_candidates.py
```

Inputs:

```text
dpo_data/train_rl.jsonl
dpo_data/train_rl_retrieved.jsonl
```

Intermediate output:

```text
dpo_data/train_rl_type_b_candidates.jsonl
```

This stage generates retrieval-only candidates. For each prompt, it takes the top-k retrieved passages and generates one retrieval-conditioned response per passage:

```text
prompt + [Retrieval]<paragraph>retrieved passage k</paragraph>
```

Default `top-k` is 3. The output stores:

```text
candidates
candidate_types
candidate_passage_ranks
candidate_ctxs
```

Example command:

```bash
python dpo_scripts/generate_type_b_candidates.py \
  --model-name-or-path /path/to/selfrag_llama2_7b \
  --input-file dpo_data/train_rl.jsonl \
  --retrieved-file dpo_data/train_rl_retrieved.jsonl \
  --output-file dpo_data/train_rl_type_b_candidates.jsonl \
  --top-k 3
```

## Stage 8: Build Type B Data

Script:

```text
dpo_scripts/build_type_b_pairs.py
```

Input:

```text
dpo_data/train_rl_type_b_candidates.jsonl
```

Output:

```text
dpo_data/train_rl_dpo_type_b.jsonl
```

Type B compares retrieval-conditioned candidates against each other.

Pair construction:

- All candidates are retrieval candidates.
- Invalid retrieval candidates are dropped.
- Each candidate is scored by `reward_utils.score_candidate`.
- The highest-scoring valid candidate becomes `chosen`.
- The lowest-scoring valid candidate becomes `rejected`.
- The pair is kept only if the reward gap is at least `--min-score-gap`, default `0.15`.

The reward includes:

- answer match
- answer F1
- support score
- relevance score
- utility score
- retrieval penalty
- formatting penalty

Current output:

```text
train_rl_dpo_type_b.jsonl: 5,991 pairs
```

Example command:

```bash
python dpo_scripts/build_type_b_pairs.py \
  --input-file dpo_data/train_rl_type_b_candidates.jsonl \
  --output-file dpo_data/train_rl_dpo_type_b.jsonl \
  --min-score-gap 0.15
```

## Stage 9: Mix Type A/B/C Data

Inputs:

```text
train_rl_dpo_type_a.jsonl
train_rl_dpo_type_b.jsonl
train_rl_dpo_type_c.jsonl
```

Outputs:

```text
train_rl_dpo_mixed_20k.jsonl
train_rl_dpo_mixed_25k.jsonl
train_rl_dpo_mixed_all.jsonl
```

The mixed datasets were created after typed pair construction:

- `train_rl_dpo_mixed_20k.jsonl`: all Type A + all Type B + sampled Type C to reach 20k
- `train_rl_dpo_mixed_25k.jsonl`: all Type A + all Type B + sampled Type C to reach 25k
- `train_rl_dpo_mixed_all.jsonl`: all Type A + all Type B + all Type C

The final mixed files were shuffled with a fixed seed before training.

Final pair counts:

| File | Type A | Type B | Type C | Total |
|---|---:|---:|---:|---:|
| `train_rl_dpo_type_a.jsonl` | 10,853 | 0 | 0 | 10,853 |
| `train_rl_dpo_type_b.jsonl` | 0 | 5,991 | 0 | 5,991 |
| `train_rl_dpo_type_c.jsonl` | 0 | 0 | 21,706 | 21,706 |
| `train_rl_dpo_mixed_20k.jsonl` | 10,853 | 5,991 | 3,156 | 20,000 |
| `train_rl_dpo_mixed_25k.jsonl` | 10,853 | 5,991 | 8,156 | 25,000 |
| `train_rl_dpo_mixed_all.jsonl` | 10,853 | 5,991 | 21,706 | 38,550 |

Dataset distribution for typed pair files:

| File | ARC Easy | ASQA | FEVER | NQ | OBQA |
|---|---:|---:|---:|---:|---:|
| `train_rl_dpo_type_a.jsonl` | 1,209 | 1,699 | 3,625 | 3,179 | 1,141 |
| `train_rl_dpo_type_b.jsonl` | 416 | 1,458 | 1,423 | 2,042 | 652 |
| `train_rl_dpo_type_c.jsonl` | 2,418 | 3,398 | 7,250 | 6,358 | 2,282 |

Dataset distribution for mixed files:

| File | ARC Easy | ASQA | FEVER | NQ | OBQA |
|---|---:|---:|---:|---:|---:|
| `train_rl_dpo_mixed_20k.jsonl` | 1,993 | 3,681 | 6,036 | 6,157 | 2,133 |
| `train_rl_dpo_mixed_25k.jsonl` | 2,577 | 4,392 | 7,770 | 7,601 | 2,660 |
| `train_rl_dpo_mixed_all.jsonl` | 4,043 | 6,555 | 12,298 | 11,579 | 4,075 |

## Stage 10: Train with DPO

Script:

```text
dpo_scripts/train_selfrag_dpo.py
```

Recommended training inputs:

```text
train_rl_dpo_type_a.jsonl
train_rl_dpo_type_b.jsonl
train_rl_dpo_type_c.jsonl
train_rl_dpo_mixed_20k.jsonl
train_rl_dpo_mixed_25k.jsonl
train_rl_dpo_mixed_all.jsonl
```

The training script uses `trl.DPOTrainer` with LoRA/QLoRA. Example:

```bash
python dpo_scripts/train_selfrag_dpo.py \
  --model-name-or-path /path/to/selfrag_llama2_7b \
  --train-file dpo_data/train_rl_dpo_mixed_20k.jsonl \
  --output-dir /path/to/output/selfrag_dpo_mixed_20k \
  --num-train-epochs 1 \
  --per-device-train-batch-size 4 \
  --gradient-accumulation-steps 8 \
  --learning-rate 5e-6 \
  --lora-rank 16 \
  --lora-alpha 16 \
  --load-in-4bit \
  --gradient-checkpointing
```

After training, LoRA adapters are merged into the base Self-RAG model before evaluation.

## Stage 11: Evaluation

After merging adapters, models were evaluated on short-form tasks:

```text
ARC Challenge
PopQA
TriviaQA
PubHealth / Health Claims
```

These eval datasets are not all exact training-set overlaps:

- ARC eval is `arc_challenge`, while the DPO source contains `arc_easy`.
- PubHealth/Health is a fact-verification eval dataset, while the DPO source contains FEVER.
- PopQA and TriviaQA are not in the DPO source subset.

The short-form evaluation script uses `match` as the main accuracy-style metric for these tasks. Scores are reported as averages in `[0, 1]`, so multiplying by 100 gives percentage accuracy/hit rate.

### Evaluation Results

The table below reports `match` as percentages.

| Model | ARC | PopQA | TriviaQA | Health |
|---|---:|---:|---:|---:|
| `base` | 67.15 | 54.97 | 66.18 | 71.94 |
| `type_a_ep1` | 67.24 | 55.11 | 66.22 | 72.14 |
| `type_b_ep1` | 67.15 | 54.97 | 66.18 | 72.14 |
| `type_c_ep1` | 67.15 | 55.18 | 66.21 | 72.04 |
| `type_a_ep2` | 67.06 | 54.90 | 66.24 | 72.14 |
| `type_b_ep2` | 67.24 | 54.82 | 66.24 | 72.04 |
| `type_c_ep2` | 67.06 | 54.90 | 66.42 | 72.04 |
| `mix20k_ep1` | 67.32 | 54.82 | 66.25 | 71.94 |
| `mix20k_ep2` | 67.32 | 54.90 | 66.21 | 71.94 |
| `mix25k_ep1` | 67.32 | 55.25 | 66.17 | 71.94 |
| `mix25k_ep2` | 67.32 | 55.04 | 66.14 | 72.04 |
| `mixall_ep1` | 67.15 | 54.97 | 66.22 | 71.94 |
| `mixall_ep2` | 67.24 | 55.18 | 66.28 | 72.14 |

Best result per dataset:

| Eval Dataset | Base | Best | Best Model(s) | Gain |
|---|---:|---:|---|---:|
| ARC | 67.15 | 67.32 | `mix20k_ep1`, `mix20k_ep2`, `mix25k_ep1`, `mix25k_ep2` | +0.17 |
| PopQA | 54.97 | 55.25 | `mix25k_ep1` | +0.29 |
| TriviaQA | 66.18 | 66.42 | `type_c_ep2` | +0.23 |
| Health | 71.94 | 72.14 | `type_a_ep1`, `type_b_ep1`, `type_a_ep2`, `mixall_ep2` | +0.20 |

Overall, DPO tuning gave small but consistent best-case gains over the base Self-RAG model. The gains are modest, usually below 0.3 percentage points, which suggests that the current preference data nudges the model in the right direction but does not substantially change short-form QA behavior. Mixed datasets are most competitive for ARC and PopQA, while Type C performs best on TriviaQA and Type A/B-style runs are strongest on Health.

## Data File Reference

| File | Rows | Description |
|---|---:|---|
| `train_rl.jsonl` | 12,000 | Source train examples with prompts, oracle outputs, extracted labels, reference answers, and oracle paragraphs. |
| `valid_rl.jsonl` | 1,000 | Source validation examples in the same format. |
| `train_rl_retrieval_queries.jsonl` | 12,000 | Query-only retrieval input from train examples. |
| `valid_rl_retrieval_queries.jsonl` | 1,000 | Query-only retrieval input from validation examples. |
| `train_rl_retrieved.jsonl` | 12,000 | Train retrieval results with `ctxs`. |
| `valid_rl_retrieved.jsonl` | 1,000 | Validation retrieval results with `ctxs`. |
| `train_rl_candidates.jsonl` | 12,000 | Retrieval/no-retrieval candidates for Type A/C. |
| `valid_rl_candidates.jsonl` | 1,000 | Validation retrieval/no-retrieval candidates. |
| `train_rl_dpo_type_a.jsonl` | 10,853 | Type A DPO pairs. |
| `train_rl_dpo_type_b.jsonl` | 5,991 | Type B DPO pairs. |
| `train_rl_dpo_type_c.jsonl` | 21,706 | Type C DPO pairs. |
| `train_rl_dpo_mixed_20k.jsonl` | 20,000 | Mixed 20k DPO file. |
| `train_rl_dpo_mixed_25k.jsonl` | 25,000 | Mixed 25k DPO file. |
| `train_rl_dpo_mixed_all.jsonl` | 38,550 | Full mixed DPO file. |
| `train_rl_dpo.jsonl` | 3,135 | Earlier combined DPO artifact kept for reference. |

## Schema Reference

Source rows contain:

```text
id
dataset_name
instruction
input
prompt
oracle_output
oracle_paragraph
reference_answer
answers
retrieval_label
relevance_label
support_label
utility_label
```

Retrieved rows contain:

```text
id
dataset_name
question
ctxs
```

DPO pair rows contain:

```text
id
dataset_name
prompt
chosen
rejected
pair_type
chosen_type
rejected_type
reference_answer
retrieval_label
support_label
relevance_label
```

Type B pair rows additionally contain:

```text
chosen_passage_rank
rejected_passage_rank
```
