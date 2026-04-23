import re
from typing import Dict, List, Optional

from metrics import match, qa_f1_score
from utils import control_tokens


UTILITY_SCORE = {
    "[Utility:1]": 0.0,
    "[Utility:2]": 0.25,
    "[Utility:3]": 0.5,
    "[Utility:4]": 0.75,
    "[Utility:5]": 1.0,
}

SUPPORT_SCORE = {
    "[Fully supported]": 1.0,
    "[Partially supported]": 0.5,
    "[No support / Contradictory]": 0.0,
}

RELEVANCE_SCORE = {
    "[Relevant]": 1.0,
    "[Irrelevant]": 0.0,
}


def extract_first_present(text: str, candidates: List[str]) -> Optional[str]:
    for candidate in candidates:
        if candidate in text:
            return candidate
    return None


def strip_paragraph_blocks(text: str) -> str:
    return re.sub(r"<paragraph>.*?</paragraph>", " ", text, flags=re.DOTALL)


def strip_prompt_prefix(text: str) -> str:
    if "### Response:" in text:
        return text.split("### Response:", 1)[1]
    return text


def extract_paragraph_text(text: str) -> Optional[str]:
    match = re.search(r"<paragraph>(.*?)</paragraph>", text, flags=re.DOTALL)
    if match is None:
        return None
    paragraph = " ".join(match.group(1).split())
    return paragraph if paragraph else None


def strip_control_tokens(text: str) -> str:
    text = strip_prompt_prefix(text)
    text = strip_paragraph_blocks(text)
    for token in control_tokens:
        text = text.replace(token, " ")
    text = text.replace("</s>", " ")
    return " ".join(text.split())


def infer_reference_answer(output_text: str) -> str:
    return strip_control_tokens(output_text)


def build_behavior_labels(output_text: str) -> Dict[str, Optional[str]]:
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


def score_candidate(candidate_text: str, example: Dict) -> Dict[str, float]:
    cleaned_candidate = strip_control_tokens(candidate_text)
    answers = example.get("answers") or [example["reference_answer"]]
    answers = [answer for answer in answers if answer]
    answer_match = float(match(cleaned_candidate.lower(), [answer.lower() for answer in answers])) if answers else 0.0
    answer_f1 = max((qa_f1_score(cleaned_candidate, answer) for answer in answers), default=0.0)

    candidate_retrieval = extract_first_present(candidate_text, ["[No Retrieval]", "[Retrieval]"])
    candidate_relevance = extract_first_present(candidate_text, ["[Relevant]", "[Irrelevant]"])
    candidate_support = extract_first_present(
        candidate_text,
        ["[Fully supported]", "[Partially supported]", "[No support / Contradictory]"],
    )
    candidate_utility = extract_first_present(
        candidate_text,
        ["[Utility:1]", "[Utility:2]", "[Utility:3]", "[Utility:4]", "[Utility:5]"],
    )

    support_score = SUPPORT_SCORE.get(candidate_support, 0.0)
    relevance_score = RELEVANCE_SCORE.get(candidate_relevance, 0.0)
    utility_score = UTILITY_SCORE.get(candidate_utility, 0.0)

    retrieval_penalty = 0.0
    oracle_retrieval = example.get("retrieval_label")
    if oracle_retrieval == "[No Retrieval]" and candidate_retrieval == "[Retrieval]":
        retrieval_penalty += 0.2
    if oracle_retrieval == "[Retrieval]" and candidate_retrieval == "[No Retrieval]":
        retrieval_penalty += 0.1
    if candidate_retrieval == "[Retrieval]" and candidate_relevance == "[Irrelevant]":
        retrieval_penalty += 0.2
    if candidate_retrieval == "[Retrieval]" and candidate_support == "[No support / Contradictory]":
        retrieval_penalty += 0.2

    format_penalty = 0.0
    if candidate_retrieval == "[Retrieval]" and "<paragraph>" not in candidate_text:
        format_penalty += 0.2
    if not cleaned_candidate:
        format_penalty += 0.3

    reward = (
        1.0 * answer_match
        + 0.3 * answer_f1
        + 0.4 * support_score
        + 0.2 * relevance_score
        + 0.1 * utility_score
        - retrieval_penalty
        - format_penalty
    )

    return {
        "reward": reward,
        "answer_match": answer_match,
        "answer_f1": answer_f1,
        "support_score": support_score,
        "relevance_score": relevance_score,
        "utility_score": utility_score,
        "retrieval_penalty": retrieval_penalty,
        "format_penalty": format_penalty,
        "cleaned_candidate": cleaned_candidate,
    }
