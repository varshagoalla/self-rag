#!/usr/bin/env python3
"""
ppo_basic_training.py — PPO Fine-Tuning on top of Self-RAG (selfrag_llama2_7b)

Reward signal (rule-based, from the model's own reflection tokens):
  utility      0.40 — [Utility:1..5]
  groundedness 0.30 — [Fully supported / Partially / No support]
  relevance    0.20 — [Relevant / Irrelevant]
  format       0.10 — valid Self-RAG token structure

Usage:
  python ppo_basic_training.py --base_model /path/to/selfrag_llama2_7b \
      --train_data /path/to/train.json --output_dir /path/to/output
"""

import argparse
import json
import logging
import os
import re
import sys
from collections import Counter, defaultdict
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.backends.cuda

torch.backends.cuda.enable_math_sdp(True)
torch.backends.cuda.enable_flash_sdp(False)
torch.backends.cuda.enable_mem_efficient_sdp(False)

from datasets import Dataset
from tqdm import tqdm
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    GenerationConfig,
)
from peft import (
    LoraConfig,
    TaskType,
    get_peft_model,
    prepare_model_for_kbit_training,
    PeftModel,
)
from trl import (
    AutoModelForCausalLMWithValueHead,
    PPOConfig,
    PPOTrainer,
)

os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["HF_DATASETS_OFFLINE"] = "1"

logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
    stream=sys.stdout,
)
logger = logging.getLogger("ppo_basic")


# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────

SELFRAG_SPECIAL_TOKENS = [
    "[No Retrieval]", "[Retrieval]", "[Continue to Use Evidence]",
    "[Irrelevant]", "[Relevant]",
    "<paragraph>", "</paragraph>",
    "[Utility:1]", "[Utility:2]", "[Utility:3]", "[Utility:4]", "[Utility:5]",
    "[Fully supported]", "[Partially supported]", "[No support / Contradictory]",
]

PROMPT_NO_INPUT    = "### Instruction:\n{instruction}\n\n### Response:\n"
PROMPT_WITH_INPUT  = "### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n### Response:\n"


# ─────────────────────────────────────────────────────────────────────────────
# Argument parsing
# ─────────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Self-RAG PPO Basic — rule-based reward on reflection tokens",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--base_model",   default="selfrag/selfrag_llama2_7b")
    p.add_argument("--train_data",   default="data/train.json")
    p.add_argument("--test_data",    default="data/balanced_test.json")
    p.add_argument("--output_dir",   default="outputs/ppo_basic")
    p.add_argument("--ppo_model_dir", default=None)
    p.add_argument("--sft_adapter",  default=None)
    p.add_argument("--num_samples",  type=int,   default=20000)
    p.add_argument("--lora_r",       type=int,   default=32)
    p.add_argument("--lora_alpha",   type=int,   default=64)
    p.add_argument("--lora_dropout", type=float, default=0.05)
    p.add_argument("--ppo_steps",    type=int,   default=500)
    p.add_argument("--batch_size",   type=int,   default=8)
    p.add_argument("--mini_batch_size", type=int, default=2)
    p.add_argument("--ppo_epochs",   type=int,   default=4)
    p.add_argument("--learning_rate", type=float, default=1.41e-5)
    p.add_argument("--init_kl_coef", type=float, default=0.2)
    p.add_argument("--target_kl",    type=float, default=6.0)
    p.add_argument("--max_new_tokens", type=int, default=200)
    p.add_argument("--temperature",  type=float, default=0.7)
    p.add_argument("--top_p",        type=float, default=0.9)
    p.add_argument("--log_interval", type=int,   default=10)
    p.add_argument("--save_interval", type=int,  default=100)
    p.add_argument("--seed",         type=int,   default=42)
    p.add_argument("--eval_only",    action="store_true")
    p.add_argument("--no_4bit",      action="store_true")
    return p.parse_args()


# ─────────────────────────────────────────────────────────────────────────────
# Reward function
# ─────────────────────────────────────────────────────────────────────────────

class SelfRAGReward:
    """Rule-based reward from Self-RAG reflection tokens.

    R = 0.40×utility + 0.30×groundedness + 0.20×relevance + 0.10×format
    Rewards are batch-whitened before PPO update.
    """

    UTILITY_RE = re.compile(r'\[Utility:([1-5])\]')

    WEIGHTS = {
        "utility":      0.40,
        "groundedness": 0.30,
        "relevance":    0.20,
        "format":       0.10,
    }

    def compute(self, text: str) -> Dict[str, float]:
        if not text or len(text.strip()) < 3:
            return {"total": -0.5, "utility": 0.0, "groundedness": 0.0,
                    "relevance": 0.0, "format": 0.0}

        no_retrieval = "[No Retrieval]" in text

        m = self.UTILITY_RE.search(text)
        utility = (int(m.group(1)) - 1) / 4.0 if m else 0.0

        if no_retrieval:
            groundedness = 0.5
        elif "[Fully supported]" in text:
            groundedness = 1.0
        elif "[Partially supported]" in text:
            groundedness = 0.5
        elif "[No support / Contradictory]" in text:
            groundedness = 0.0
        else:
            groundedness = 0.0

        if no_retrieval:
            relevance = 0.5
        elif "[Relevant]" in text:
            relevance = 1.0
        elif "[Irrelevant]" in text:
            relevance = 0.0
        else:
            relevance = 0.5

        has_retrieval_decision = any(t in text for t in (
            "[No Retrieval]", "[Retrieval]", "[Continue to Use Evidence]"
        ))
        has_utility = bool(m)
        fmt = 0.5 * int(has_retrieval_decision) + 0.5 * int(has_utility)

        total = (
            self.WEIGHTS["utility"]      * utility
            + self.WEIGHTS["groundedness"] * groundedness
            + self.WEIGHTS["relevance"]    * relevance
            + self.WEIGHTS["format"]       * fmt
        )

        return {
            "total": float(total),
            "utility": utility,
            "groundedness": groundedness,
            "relevance": relevance,
            "format": fmt,
        }

    def __call__(self, text: str) -> float:
        return self.compute(text)["total"]


def whiten_rewards(rewards: List[torch.Tensor]) -> List[torch.Tensor]:
    """Normalize rewards to zero mean, unit variance within a batch."""
    r = torch.stack(rewards)
    r_whitened = (r - r.mean()) / (r.std() + 1e-8)
    return [r_whitened[i] for i in range(len(rewards))]


# ─────────────────────────────────────────────────────────────────────────────
# Data loading
# ─────────────────────────────────────────────────────────────────────────────

def load_and_filter_data(path: str, num_samples: int, seed: int = 42) -> List[Dict]:
    """Load training data, keep only [Retrieval] examples, stratified sample."""
    import random
    random.seed(seed)

    logger.info(f"Loading data from: {path}")
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    retrieval_examples = [
        ex for ex in data
        if ex.get("output") and "[Retrieval]" in ex["output"] and ex.get("instruction")
    ]
    logger.info(f"Filtered to {len(retrieval_examples)} [Retrieval] examples (from {len(data)} total)")

    by_source: Dict[str, List] = defaultdict(list)
    for ex in retrieval_examples:
        by_source[ex.get("dataset_name", "unknown")].append(ex)

    total_available = len(retrieval_examples)
    target = min(num_samples, total_available)
    sampled = []

    for source, examples in by_source.items():
        proportion = len(examples) / total_available
        n_draw = max(1, round(proportion * target))
        sampled.extend(random.sample(examples, min(n_draw, len(examples))))

    random.shuffle(sampled)
    sampled = sampled[:target]
    logger.info(f"Final PPO training set: {len(sampled)} examples")
    return sampled


def make_prompt(example: Dict) -> str:
    instruction = example.get("instruction", "")
    context = example.get("input", "")
    if context.strip():
        return PROMPT_WITH_INPUT.format(instruction=instruction, input=context)
    return PROMPT_NO_INPUT.format(instruction=instruction)


# ─────────────────────────────────────────────────────────────────────────────
# Model loading
# ─────────────────────────────────────────────────────────────────────────────

def build_tokenizer(model_path: str) -> AutoTokenizer:
    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False, local_files_only=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    new_tokens = [t for t in SELFRAG_SPECIAL_TOKENS if t not in tokenizer.get_vocab()]
    if new_tokens:
        tokenizer.add_special_tokens({"additional_special_tokens": new_tokens})
        logger.info(f"Added {len(new_tokens)} Self-RAG special tokens")
    return tokenizer


def build_ppo_model(
    base_model_path: str,
    tokenizer: AutoTokenizer,
    use_4bit: bool,
    lora_r: int,
    lora_alpha: int,
    lora_dropout: float,
    sft_adapter: Optional[str] = None,
) -> AutoModelForCausalLMWithValueHead:
    logger.info(f"Loading PPO base model from: {base_model_path}")
    load_kwargs = {
        "device_map": "auto",
        "local_files_only": True,
        "attn_implementation": "eager",
    }
    if use_4bit:
        load_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True,
        )
    else:
        load_kwargs["torch_dtype"] = torch.bfloat16

    base = AutoModelForCausalLM.from_pretrained(base_model_path, **load_kwargs)
    base.resize_token_embeddings(len(tokenizer))

    if use_4bit:
        base = prepare_model_for_kbit_training(base, use_gradient_checkpointing=True)

    if sft_adapter and os.path.isdir(sft_adapter):
        logger.info(f"Merging SFT adapter from: {sft_adapter}")
        base = PeftModel.from_pretrained(base, sft_adapter, local_files_only=True)
        base = base.merge_and_unload()
        if use_4bit:
            base = prepare_model_for_kbit_training(base, use_gradient_checkpointing=True)

    lora_cfg = LoraConfig(
        r=lora_r, lora_alpha=lora_alpha, lora_dropout=lora_dropout,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
        bias="none", task_type=TaskType.CAUSAL_LM,
    )
    base = get_peft_model(base, lora_cfg)
    ppo_model = AutoModelForCausalLMWithValueHead.from_pretrained(base)

    trainable = sum(p.numel() for p in ppo_model.parameters() if p.requires_grad)
    total     = sum(p.numel() for p in ppo_model.parameters())
    logger.info(f"Trainable: {trainable:,} / {total:,} ({100*trainable/total:.2f}%)")
    return ppo_model


def build_reference_model(
    base_model_path: str,
    tokenizer: AutoTokenizer,
    use_4bit: bool,
) -> AutoModelForCausalLMWithValueHead:
    logger.info("Loading frozen reference model...")
    load_kwargs = {
        "device_map": "auto",
        "local_files_only": True,
        "attn_implementation": "eager",
    }
    if use_4bit:
        load_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True,
        )
    else:
        load_kwargs["torch_dtype"] = torch.bfloat16

    ref_base = AutoModelForCausalLM.from_pretrained(base_model_path, **load_kwargs)
    ref_base.resize_token_embeddings(len(tokenizer))
    ref_model = AutoModelForCausalLMWithValueHead.from_pretrained(ref_base)
    for p in ref_model.parameters():
        p.requires_grad_(False)
    logger.info("Reference model frozen.")
    return ref_model


# ─────────────────────────────────────────────────────────────────────────────
# PPO Training loop
# ─────────────────────────────────────────────────────────────────────────────

def run_ppo(args: argparse.Namespace) -> None:
    os.makedirs(args.output_dir, exist_ok=True)
    tokenizer = build_tokenizer(args.base_model)

    raw_data = load_and_filter_data(args.train_data, args.num_samples, args.seed)

    def tokenise_query(ex: Dict) -> Dict:
        prompt = make_prompt(ex)
        ids = tokenizer.encode(prompt, truncation=True, max_length=256, add_special_tokens=True)
        return {"input_ids": ids, "query": prompt,
                "gold": ex.get("output", ""), "dataset_name": ex.get("dataset_name", "unknown")}

    records = [tokenise_query(ex) for ex in raw_data]
    ppo_dataset = Dataset.from_list(records)
    ppo_dataset.set_format(type="torch", columns=["input_ids"])

    ppo_model = build_ppo_model(
        args.base_model, tokenizer, use_4bit=not args.no_4bit,
        lora_r=args.lora_r, lora_alpha=args.lora_alpha, lora_dropout=args.lora_dropout,
        sft_adapter=getattr(args, "sft_adapter", None),
    )
    ref_model = build_reference_model(args.base_model, tokenizer, use_4bit=not args.no_4bit)

    ppo_config = PPOConfig(
        model_name=args.base_model,
        learning_rate=args.learning_rate,
        batch_size=args.batch_size,
        mini_batch_size=args.mini_batch_size,
        gradient_accumulation_steps=max(1, args.batch_size // args.mini_batch_size),
        ppo_epochs=args.ppo_epochs,
        init_kl_coef=args.init_kl_coef,
        target=args.target_kl,
        adap_kl_ctrl=True,
        seed=args.seed,
        log_with=None,
        remove_unused_columns=False,
    )

    ppo_trainer = PPOTrainer(
        config=ppo_config,
        model=ppo_model,
        ref_model=ref_model,
        tokenizer=tokenizer,
        dataset=ppo_dataset,
        data_collator=lambda data: {"input_ids": [d["input_ids"] for d in data]},
    )

    gen_cfg = GenerationConfig(
        max_new_tokens=args.max_new_tokens,
        do_sample=True,
        temperature=args.temperature,
        top_p=args.top_p,
        repetition_penalty=1.1,
        pad_token_id=tokenizer.eos_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )

    reward_fn = SelfRAGReward()
    reward_history    = []
    component_history = defaultdict(list)

    logger.info(f"Starting PPO: {len(ppo_dataset)} examples, {args.ppo_steps} steps")

    for step, batch in enumerate(tqdm(ppo_trainer.dataloader, total=args.ppo_steps, desc="PPO")):
        if step >= args.ppo_steps:
            break

        query_tensors: List[torch.Tensor] = batch["input_ids"]
        response_tensors: List[torch.Tensor] = ppo_trainer.generate(query_tensors, generation_config=gen_cfg)

        response_only = [r[len(q):] for q, r in zip(query_tensors, response_tensors)]
        decoded_responses = [tokenizer.decode(r, skip_special_tokens=False) for r in response_only]
        decoded_queries   = [tokenizer.decode(q, skip_special_tokens=True)  for q in query_tensors]

        raw_rewards = []
        details = []
        for resp in decoded_responses:
            d = reward_fn.compute(resp)
            raw_rewards.append(torch.tensor(d["total"], dtype=torch.float32))
            details.append(d)

        if any(torch.isnan(r) for r in raw_rewards):
            logger.warning(f"Step {step}: NaN reward, skipping")
            continue

        whitened_rewards = whiten_rewards(raw_rewards)
        stats = ppo_trainer.step(query_tensors, response_only, whitened_rewards)

        avg_raw_reward = float(torch.stack(raw_rewards).mean())
        reward_history.append(avg_raw_reward)
        for key in ("utility", "groundedness", "relevance", "format"):
            component_history[key].append(np.mean([d[key] for d in details]))

        if step % args.log_interval == 0:
            kl = stats.get("objective/kl", stats.get("ppo/mean_non_score_reward", 0.0))
            logger.info(
                f"Step {step:4d} | Reward={avg_raw_reward:+.4f} | "
                f"Util={component_history['utility'][-1]:.3f} | "
                f"Ground={component_history['groundedness'][-1]:.3f} | "
                f"Relev={component_history['relevance'][-1]:.3f} | "
                f"Fmt={component_history['format'][-1]:.3f} | KL={kl:.4f}"
            )
            logger.info(f"  Query   : {decoded_queries[0][:100]}")
            logger.info(f"  Response: {decoded_responses[0][:200]}")

        if step > 0 and step % args.save_interval == 0:
            _offset = getattr(args, "_resume_offset", 0)
            ckpt = os.path.join(args.output_dir, f"checkpoint_step{step + _offset}")
            os.makedirs(ckpt, exist_ok=True)
            getattr(ppo_model, "pretrained_model", ppo_model).save_pretrained(ckpt)
            tokenizer.save_pretrained(ckpt)
            logger.info(f"Saved checkpoint: {ckpt}")
            with open(os.path.join(args.output_dir, "reward_history.json"), "w") as f:
                json.dump({"reward_per_step": reward_history, "components": dict(component_history)}, f, indent=2)

    getattr(ppo_model, "pretrained_model", ppo_model).save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

    history_path = os.path.join(args.output_dir, "reward_history.json")
    with open(history_path, "w") as f:
        json.dump({"reward_per_step": reward_history, "components": dict(component_history)}, f, indent=2)

    logger.info(f"Training complete. Saved to: {args.output_dir}")


# ─────────────────────────────────────────────────────────────────────────────
# Evaluation
# ─────────────────────────────────────────────────────────────────────────────

def f1_score(pred: str, gold: str) -> float:
    pred_tokens = pred.lower().split()
    gold_tokens = gold.lower().split()
    if not pred_tokens or not gold_tokens:
        return 0.0
    common = Counter(pred_tokens) & Counter(gold_tokens)
    if not common:
        return 0.0
    prec = sum(common.values()) / len(pred_tokens)
    rec  = sum(common.values()) / len(gold_tokens)
    return 2 * prec * rec / (prec + rec + 1e-8)


def exact_match(pred: str, gold: str) -> float:
    return float(gold.lower().strip() in pred.lower())


def run_evaluation(
    baseline_model_path: str,
    ppo_model_dir: Optional[str],
    tokenizer: AutoTokenizer,
    test_data_path: str,
    use_4bit: bool,
    max_new_tokens: int = 200,
) -> None:
    logger.info("Running evaluation: Baseline vs PPO")

    with open(test_data_path, "r", encoding="utf-8") as f:
        test_data = json.load(f)
    logger.info(f"Loaded {len(test_data)} test examples")

    reward_fn  = SelfRAGReward()
    utility_re = re.compile(r'\[Utility:([1-5])\]')
    gen_cfg = GenerationConfig(
        max_new_tokens=max_new_tokens, do_sample=False,
        temperature=1.0, pad_token_id=tokenizer.eos_token_id,
    )

    def evaluate_model(model: AutoModelForCausalLM, name: str) -> Dict:
        model.eval()
        results = defaultdict(list)
        for ex in tqdm(test_data, desc=f"Eval {name}", leave=False):
            prompt = (PROMPT_WITH_INPUT.format(instruction=ex["instruction"], input=ex["input"])
                      if ex.get("input", "").strip()
                      else PROMPT_NO_INPUT.format(instruction=ex["instruction"]))
            gold = ex.get("output", "")
            inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=256).to(model.device)
            with torch.no_grad():
                out = model.generate(**inputs, generation_config=gen_cfg)
            pred = tokenizer.decode(out[0][inputs.input_ids.shape[1]:], skip_special_tokens=False)
            d = reward_fn.compute(pred)
            m = utility_re.search(pred)
            util_n = int(m.group(1)) if m else 0
            has_ret  = any(t in pred for t in ("[Retrieval]", "[No Retrieval]", "[Continue to Use Evidence]"))
            results["reward"].append(d["total"])
            results["utility"].append(d["utility"])
            results["groundedness"].append(d["groundedness"])
            results["relevance"].append(d["relevance"])
            results["high_utility"].append(float(util_n >= 4))
            results["fully_supported"].append(float("[Fully supported]" in pred))
            results["relevant"].append(float("[Relevant]" in pred))
            results["valid_format"].append(float(has_ret and bool(m)))
            results["token_f1"].append(f1_score(pred, gold))
            results["exact_match"].append(exact_match(pred, gold))
            results["dataset_name"].append(ex.get("dataset_name", "unknown"))
        return dict(results)

    load_kw = {"device_map": "auto", "local_files_only": True, "attn_implementation": "eager"}
    if use_4bit:
        load_kw["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16,
        )
    else:
        load_kw["torch_dtype"] = torch.bfloat16

    baseline = AutoModelForCausalLM.from_pretrained(baseline_model_path, **load_kw)
    baseline.resize_token_embeddings(len(tokenizer))
    baseline_results = evaluate_model(baseline, "Baseline")
    del baseline
    torch.cuda.empty_cache()

    ppo_results = None
    if ppo_model_dir and os.path.exists(ppo_model_dir):
        ppo_base = AutoModelForCausalLM.from_pretrained(baseline_model_path, **load_kw)
        ppo_base.resize_token_embeddings(len(tokenizer))
        ppo_model = PeftModel.from_pretrained(ppo_base, ppo_model_dir, local_files_only=True)
        ppo_results = evaluate_model(ppo_model, "PPO")
        del ppo_model, ppo_base
        torch.cuda.empty_cache()

    def mean(lst): return float(np.mean(lst)) if lst else 0.0

    metrics = [
        ("avg_reward", "reward"), ("% high_utility", "high_utility"),
        ("% fully_supp", "fully_supported"), ("% relevant", "relevant"),
        ("% valid_format", "valid_format"), ("token_f1", "token_f1"),
        ("exact_match", "exact_match"),
    ]

    print(f"\n{'Metric':<22} {'Baseline':>10}   {'PPO':>10}   {'Δ':>10}")
    print("-" * 60)
    for label, key in metrics:
        b = mean(baseline_results[key])
        p = mean(ppo_results[key]) if ppo_results else None
        delta_str = f"  {p - b:+.4f}" if p is not None else ""
        ppo_str = f"{p:.4f}" if p is not None else "  N/A"
        print(f"  {label:<20}   {b:.4f}     {ppo_str}{delta_str}")

    if ppo_results:
        out = {
            "baseline": {k: mean(v) for k, v in baseline_results.items() if k != "dataset_name"},
            "ppo":      {k: mean(v) for k, v in ppo_results.items()      if k != "dataset_name"},
        }
        results_path = os.path.join(ppo_model_dir or ".", "evaluation_results.json")
        with open(results_path, "w") as f:
            json.dump(out, f, indent=2)
        logger.info(f"Saved: {results_path}")


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    args = parse_args()
    if args.eval_only:
        tokenizer = build_tokenizer(args.base_model)
        run_evaluation(args.base_model, args.ppo_model_dir, tokenizer,
                       args.test_data, not args.no_4bit, args.max_new_tokens)
    else:
        run_ppo(args)
        tokenizer = build_tokenizer(args.base_model)
        run_evaluation(args.base_model, args.output_dir, tokenizer,
                       args.test_data, not args.no_4bit, args.max_new_tokens)


if __name__ == "__main__":
    main()
