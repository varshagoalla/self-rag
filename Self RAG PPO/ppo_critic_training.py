#!/usr/bin/env python3
"""
ppo_critic_training.py — PPO with Frozen Critic Reward on Hard Examples

Combines hard-example selection (from ppo_hard_training) with a frozen
selfrag/self_rag_critic model as the reward signal instead of self-scored tokens.

Reward (0.80 × critic_score + 0.20 × anti-collapse):
  utility      0.15 — critic P([Utility:N])
  groundedness 0.30 — critic P([Fully supported])
  relevance    0.25 — critic P([Relevant])
  format       0.10 — rule-based token structure check
  + repetition penalty | length penalty | coherence bonus

Usage:
  python ppo_critic_training.py \
      --base_model   /path/to/selfrag_llama2_7b \
      --critic_model /path/to/self_rag_critic \
      --train_data   /path/to/train.json \
      --num_samples  5000
"""

import argparse
import json
import logging
import os
import re
from collections import Counter
from typing import Dict, List

import selfrag_ppo_critic as _base

logger = logging.getLogger("ppo_critic")


# ─────────────────────────────────────────────────────────────────────────────
# Hard-example data filter (same logic as ppo_hard_training)
# ─────────────────────────────────────────────────────────────────────────────

JUDGMENT_TOKENS = [
    "[Relevant]", "[Irrelevant]",
    "[Fully supported]", "[Partially supported]", "[No support / Contradictory]",
    "[Utility:1]", "[Utility:2]", "[Utility:3]", "[Utility:4]", "[Utility:5]",
]


def _count_judgment_tokens(text: str) -> int:
    return sum(text.count(tok) for tok in JUDGMENT_TOKENS)


def load_and_filter_data(path: str, num_samples: int, seed: int = 42) -> List[Dict]:
    """Keep only [Retrieval] examples, rank by judgment token count, 80/20 hard/easy split."""
    import random
    random.seed(seed)

    logger.info(f"Loading data from: {path}")
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    retrieval_examples = [
        ex for ex in data
        if ex.get("output") and "[Retrieval]" in ex["output"] and ex.get("instruction")
    ]
    logger.info(f"After [Retrieval] filter: {len(retrieval_examples)} (from {len(data)} total)")

    scored = sorted(
        [(_count_judgment_tokens(ex["output"]), ex) for ex in retrieval_examples],
        key=lambda x: x[0], reverse=True,
    )

    n_hard = min(int(num_samples * 0.80), len(scored))
    n_easy = min(int(num_samples * 0.20), max(0, len(scored) - n_hard))

    hard_pool = scored[:n_hard]
    easy_pool = scored[n_hard:]

    random.shuffle(hard_pool)
    sampled_hard = [ex for _, ex in hard_pool[:n_hard]]
    sampled_easy = ([ex for _, ex in random.sample(easy_pool, min(n_easy, len(easy_pool)))]
                    if easy_pool and n_easy > 0 else [])

    sampled = sampled_hard + sampled_easy
    random.shuffle(sampled)

    logger.info(f"Final: {len(sampled)} examples ({len(sampled_hard)} hard, {len(sampled_easy)} easy)")
    return sampled


# ─────────────────────────────────────────────────────────────────────────────
# Critic reward + anti-collapse penalties
# ─────────────────────────────────────────────────────────────────────────────

class CriticRewardV2(_base.CriticReward):
    """Frozen-critic reward + repetition/length/coherence anti-collapse.

    Final = 0.80 × critic_score + 0.20 × anti_collapse_score
    """

    CRITIC_WEIGHTS = {
        "utility":      0.15,
        "groundedness": 0.30,
        "relevance":    0.25,
        "format":       0.10,
    }

    FORMAT_PENALTY:     float = -0.30
    REPETITION_PENALTY: float = -0.30
    SHORT_PENALTY:      float = -0.25
    LONG_BONUS:         float =  0.05
    COHERENCE_BONUS:    float =  0.10
    CRITIC_WEIGHT:      float =  0.80
    ANTI_COLLAPSE_WEIGHT: float = 0.20

    UTILITY_RE = re.compile(r'\[Utility:([1-5])\]')

    def __init__(self, *args, **kwargs):
        kwargs["weights"] = self.CRITIC_WEIGHTS
        super().__init__(*args, **kwargs)

    def _repetition_score(self, text: str) -> float:
        """Returns 0.0 (no repetition) to -0.30 (heavy) based on repeated 4-grams."""
        words = text.lower().split()
        if len(words) < 8:
            return 0.0
        ngrams = [" ".join(words[i:i+4]) for i in range(len(words) - 3)]
        counts = Counter(ngrams)
        repeated = sum(1 for c in counts.values() if c >= 3)
        ratio = repeated / len(counts) if counts else 0.0
        if ratio > 0.20: return self.REPETITION_PENALTY
        if ratio > 0.10: return self.REPETITION_PENALTY * 0.5
        if ratio > 0.05: return self.REPETITION_PENALTY * 0.2
        return 0.0

    def _length_score(self, text: str) -> float:
        """Returns -0.25 (too short) to +0.05 (good length)."""
        n = len(text.strip())
        if n < 30:   return self.SHORT_PENALTY
        if n < 50:   return self.SHORT_PENALTY * 0.4
        if n < 100:  return 0.0
        if n <= 400: return self.LONG_BONUS
        return 0.0

    def _coherence_score(self, text: str) -> float:
        """Bonus for complete Self-RAG structure (retrieval + utility + relevance/grounding)."""
        has_ret  = any(t in text for t in ("[No Retrieval]", "[Retrieval]", "[Continue to Use Evidence]"))
        has_util = bool(self.UTILITY_RE.search(text))
        has_rel  = "[Relevant]" in text or "[Irrelevant]" in text
        has_gnd  = any(t in text for t in ("[Fully supported]", "[Partially supported]", "[No support / Contradictory]"))
        if has_ret and has_util:
            return self.COHERENCE_BONUS if (has_rel or has_gnd) else self.COHERENCE_BONUS * 0.5
        return 0.0

    def compute(self, instruction: str, response_text: str) -> Dict[str, float]:
        if not response_text or len(response_text.strip()) < 3:
            return {"total": -0.5, "utility": 0.0, "groundedness": 0.0,
                    "relevance": 0.0, "format": 0.0,
                    "repetition": 0.0, "length": self.SHORT_PENALTY, "coherence": 0.0}

        passage, answer = self._extract_parts(response_text)
        if not answer:
            return {"total": -0.5, "utility": 0.0, "groundedness": 0.0,
                    "relevance": 0.0, "format": 0.0,
                    "repetition": 0.0, "length": self._length_score(response_text), "coherence": 0.0}

        relevance    = self._relevance_score(instruction, passage)
        groundedness = self._groundedness_score(instruction, passage, answer)
        utility      = self._utility_score(instruction, answer)
        fmt          = self._format_score(response_text)

        critic_score = (self.CRITIC_WEIGHTS["utility"]      * utility
                        + self.CRITIC_WEIGHTS["groundedness"] * groundedness
                        + self.CRITIC_WEIGHTS["relevance"]    * relevance
                        + self.CRITIC_WEIGHTS["format"]       * fmt)

        has_ret  = any(t in response_text for t in ("[No Retrieval]", "[Retrieval]", "[Continue to Use Evidence]"))
        has_util = bool(self.UTILITY_RE.search(response_text))

        if not has_ret and not has_util:
            rep = self._repetition_score(response_text)
            lng = self._length_score(response_text)
            return {"total": self.FORMAT_PENALTY + rep + lng,
                    "utility": utility, "groundedness": groundedness,
                    "relevance": relevance, "format": fmt,
                    "repetition": rep, "length": lng, "coherence": 0.0}

        rep = self._repetition_score(response_text)
        lng = self._length_score(response_text)
        coh = self._coherence_score(response_text)
        anti = max(-1.0, min(1.0, rep + lng + coh))
        total = self.CRITIC_WEIGHT * critic_score + self.ANTI_COLLAPSE_WEIGHT * anti

        return {"total": float(total), "utility": float(utility),
                "groundedness": float(groundedness), "relevance": float(relevance),
                "format": float(fmt), "repetition": float(rep),
                "length": float(lng), "coherence": float(coh)}

    def __call__(self, instruction: str, response_text: str) -> float:
        return self.compute(instruction, response_text)["total"]


# ─────────────────────────────────────────────────────────────────────────────
# Hyperparameters
# ─────────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Self-RAG Critic-PPO — frozen critic reward on hard examples",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--base_model",   default="selfrag/selfrag_llama2_7b")
    p.add_argument("--critic_model", default="selfrag/self_rag_critic")
    p.add_argument("--train_data",   default="data/train.json")
    p.add_argument("--test_data",    default="data/balanced_test.json")
    p.add_argument("--output_dir",   default="outputs/ppo_critic")
    p.add_argument("--ppo_model_dir", default=None)
    p.add_argument("--sft_adapter",  default=None)
    p.add_argument("--num_samples",  type=int,   default=5000)
    p.add_argument("--lora_r",       type=int,   default=32)
    p.add_argument("--lora_alpha",   type=int,   default=64)
    p.add_argument("--lora_dropout", type=float, default=0.05)
    p.add_argument("--ppo_steps",    type=int,   default=625)
    p.add_argument("--batch_size",   type=int,   default=8)
    p.add_argument("--mini_batch_size", type=int, default=2)
    p.add_argument("--ppo_epochs",   type=int,   default=1)
    p.add_argument("--learning_rate", type=float, default=5e-6)
    p.add_argument("--init_kl_coef", type=float, default=0.5)
    p.add_argument("--target_kl",    type=float, default=2.0)
    p.add_argument("--max_new_tokens", type=int, default=200)
    p.add_argument("--temperature",  type=float, default=0.7)
    p.add_argument("--top_p",        type=float, default=0.9)
    p.add_argument("--w_utility",      type=float, default=0.15)
    p.add_argument("--w_groundedness", type=float, default=0.30)
    p.add_argument("--w_relevance",    type=float, default=0.25)
    p.add_argument("--w_format",       type=float, default=0.10)
    p.add_argument("--log_interval",  type=int, default=10)
    p.add_argument("--save_interval", type=int, default=100)
    p.add_argument("--seed",          type=int, default=42)
    p.add_argument("--eval_only",     action="store_true")
    p.add_argument("--no_4bit",       action="store_true")
    return p.parse_args()


# ─────────────────────────────────────────────────────────────────────────────
# Apply patches
# ─────────────────────────────────────────────────────────────────────────────

_base.load_and_filter_data = load_and_filter_data
_base.CriticReward         = CriticRewardV2
_base.parse_args           = parse_args


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    _base.main()
