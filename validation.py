import json
import itertools
import numpy as np
from typing import List, Dict, Tuple
from sklearn.metrics import cohen_kappa_score

from openai import OpenAI

# Setting Up the LLM Client (using GitHub Models) 
client = OpenAI(
    base_url="https://models.inference.ai.azure.com",
    api_key="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",
)

# Choose a small/cheap model for judging
MODEL_NAME = "gpt-4o-mini"

# Loading the Event Extractions
def load_event_extractions(path: str = "/data/final_dataset/events_extracted.json") -> List[Dict]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    print(f"Loaded {len(data)} event extractions.")
    return data

# Split Lincoln documents from the others
def split_lincoln_vs_others(rows: List[Dict]) -> Dict[str, Dict[str, List[Dict]]]:
    per_event: Dict[str, Dict[str, List[Dict]]] = {}

    for row in rows:
        event_id = row["event_id"]
        source_type = row.get("source_type")

        # In your schema: non-Book = Lincoln primary source
        is_lincoln = source_type != "Book"
        role = "lincoln" if is_lincoln else "others"

        per_event.setdefault(event_id, {"lincoln": [], "others": []})
        per_event[event_id][role].append(row)

    for eid, groups in per_event.items():
        print(f"Event {eid}: {len(groups['lincoln'])} Lincoln docs, {len(groups['others'])} Other docs")

    return per_event

# Build the comparison pairs
def build_comparison_pairs(per_event: Dict[str, Dict[str, List[Dict]]]) -> List[Tuple[Dict, Dict]]:
    pairs: List[Tuple[Dict, Dict]] = []
    for eid, groups in per_event.items():
        for lincoln_row, other_row in itertools.product(groups["lincoln"], groups["others"]):
            pairs.append((lincoln_row, other_row))
    print(f"Total comparison pairs: {len(pairs)}")
    return pairs

# Judge prompt setup
def build_judge_prompt(event_name: str, lincoln_extraction: Dict, other_extraction: Dict, style: str = "cot") -> str:
    lincoln_claims = "\n- ".join(lincoln_extraction.get("claims", [])) or "(no claims)"
    other_claims = "\n- ".join(other_extraction.get("claims", [])) or "(no claims)"

    base = f"""
You are an expert historian comparing two descriptions of the same historical event.

Event: {event_name}

Source A: Abraham Lincoln (primary source)
Claims:
- {lincoln_claims}

Source B: Secondary historian/author
Claims:
- {other_claims}

Your tasks:

1. Give a consistency score (0–100) describing how consistent Source B is with Source A.
2. Identify difference types (zero or more): "factual", "interpretive", "omission".

Return STRICT JSON:
{{
  "consistency_score": <integer>,
  "difference_types": ["factual" | "interpretive" | "omission"],
  "reasoning": "<short explanation>"
}}
""".strip()

    if style == "zero_shot":
        return base

    if style == "cot":
        return base + """

Think step by step, then output ONLY the JSON.
"""

    if style == "few_shot":
        example = r"""

Example 1:
JSON:
{
 "consistency_score": 95,
 "difference_types": [],
 "reasoning": "Both sources agree on the key facts and differ only slightly in wording."
}

Example 2:
JSON:
{
 "consistency_score": 20,
 "difference_types": ["factual"],
 "reasoning": "They disagree on the outcome and timing of the event."
}

Now produce JSON for this case.
"""
        return base + "\n\n" + example

    return base

# Call the judge model
def call_judge_llm(
    event_name: str,
    lincoln_row: Dict,
    other_row: Dict,
    style: str = "cot",
    temperature: float = 0.0,
) -> Dict:
    lincoln_ex = lincoln_row["extraction"]
    other_ex = other_row["extraction"]

    prompt = build_judge_prompt(event_name, lincoln_ex, other_ex, style)

    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[{"role": "user", "content": prompt}],
        temperature=temperature,
        max_tokens=400,
    )

    raw = response.choices[0].message.content.strip()

    try:
        start = raw.find("{")
        end = raw.rfind("}")
        json_str = raw[start:end+1]
        data = json.loads(json_str)
    except Exception:
        data = {
            "consistency_score": None,
            "difference_types": [],
            "reasoning": raw,
        }

    data.setdefault("consistency_score", None)
    data.setdefault("difference_types", [])
    data.setdefault("reasoning", "")

    return data

# Evaluating different prompt styles
def eval_prompt_strategies(
    pairs: List[Tuple[Dict, Dict]],
    max_pairs: int = 8,
    styles=("zero_shot", "cot", "few_shot"),
) -> List[Dict]:

    subset = pairs[:max_pairs]
    results: List[Dict] = []

    for (lincoln_row, other_row) in subset:
        event_id = lincoln_row["event_id"]
        event_name = lincoln_row["event_name"]

        for style in styles:
            print(f"[Ablation] event={event_id}, style={style}")
            out = call_judge_llm(event_name, lincoln_row, other_row, style)
            score = out.get("consistency_score")

            results.append({
                "event_id": event_id,
                "style": style,
                "score": score
            })

    return results


def summarize_prompt_results(prompt_results: List[Dict]):
    styles = sorted(set(r["style"] for r in prompt_results))
    summary = []

    for style in styles:
        scores = [
            r["score"]
            for r in prompt_results
            if r["style"] == style and r["score"] is not None
        ]
        if not scores:
            continue

        mean_score = float(np.mean(scores))
        std_score = float(np.std(scores))
        summary.append({
            "style": style,
            "n": len(scores),
            "mean": mean_score,
            "std": std_score,
        })

    print("Prompt style summary:")
    for row in summary:
        print(
            f"  {row['style']}: n={row['n']}, "
            f"mean={row['mean']:.1f}, std={row['std']:.1f}"
        )

    return summary

# Test the model for self-consistency
def eval_self_consistency(
    pairs: List[Tuple[Dict, Dict]],
    style: str = "cot",
    runs: int = 5,
    max_pairs: int = 5,
    temperature: float = 0.7,
):
    subset = pairs[:max_pairs]
    results = []

    for (lincoln_row, other_row) in subset:
        event_id = lincoln_row["event_id"]
        event_name = lincoln_row["event_name"]

        scores = []
        print(f"[Self Consistency] {event_id}")

        for _ in range(runs):
            out = call_judge_llm(event_name, lincoln_row, other_row, style, temperature)
            s = out.get("consistency_score")
            if isinstance(s, (int, float)):
                scores.append(s)

        results.append({
            "event_id": event_id,
            "scores": scores,
            "mean": float(np.mean(scores)) if scores else None,
            "std": float(np.std(scores)) if scores else None,
        })

    return results


def summarize_self_consistency(selfcons: List[Dict]):
    stds = [row["std"] for row in selfcons if row["std"] is not None]
    means = [row["mean"] for row in selfcons if row["mean"] is not None]

    if not stds:
        print("No self-consistency data.")
        return None

    overall_mean_std = float(np.mean(stds))
    max_std = float(np.max(stds))
    min_std = float(np.min(stds))

    print(f"Self-consistency summary across {len(stds)} pairs:")
    print(f"  avg std of scores = {overall_mean_std:.2f}")
    print(f"  min std = {min_std:.2f}, max std = {max_std:.2f}")

    return {
        "avg_std": overall_mean_std,
        "min_std": min_std,
        "max_std": max_std,
        "stds": stds,
        "means": means,
    }

# Calculate Cohen's kappa
def model_label_from_judge_output(judge_output: Dict, threshold: int = 60) -> str:
    score = judge_output.get("consistency_score")
    if score is None:
        return "Unknown"
    return "Consistent" if score >= threshold else "Contradictory"


def compute_kappa(
    labeled_pairs: List[Tuple[Dict, Dict]],
    human_labels: List[str],
    style: str = "cot",
    threshold: int = 60,
):
    model_labels = []

    for (lincoln_row, other_row) in labeled_pairs:
        event_name = lincoln_row["event_name"]
        out = call_judge_llm(event_name, lincoln_row, other_row, style)
        model_labels.append(model_label_from_judge_output(out, threshold))
    kappa = cohen_kappa_score(human_labels, model_labels)
    print("Cohen's Kappa =", kappa)
    return kappa

# Execute a sample portion of the evaluation pipeline
rows = load_event_extractions("data/final_dataset/events_extracted.json")
per_event = split_lincoln_vs_others(rows)
pairs = build_comparison_pairs(per_event)

print("\n=========== PROMPT STRATEGY TEST ===========")

prompt_results = eval_prompt_strategies(pairs, max_pairs=5)
print(prompt_results)
prompt_summary = summarize_prompt_results(prompt_results)

def pretty_print_pair(lincoln_row: Dict, other_row: Dict, max_claims: int = 3):
    print("=== Event:", lincoln_row["event_id"], "-", lincoln_row["event_name"], "===")
    print("\n[LINCOLN CLAIMS]")
    for c in lincoln_row["extraction"].get("claims", [])[:max_claims]:
        print(" -", c)

    print("\n[OTHER AUTHOR CLAIMS]")
    for c in other_row["extraction"].get("claims", [])[:max_claims]:
        print(" -", c)
    print("\n")

sample_pairs_1 = pairs[:5]+pairs[-5:]
for lincoln_row, other_row in sample_pairs_1:
    pretty_print_pair(lincoln_row, other_row)

print("\n=========== SELF CONSISTENCY TEST ===========")
selfcons = eval_self_consistency(pairs, max_pairs=3)
print(selfcons)
selfcons_summary = summarize_self_consistency(selfcons)

print("\n=========== COHEN'S KAPPA ===========")
sample_pairs = pairs[:10]
human_labels = ["Consistent", "Contradictory", "Contradictory", "Consistent", "Contradictory",
                "Consistent", "Contradictory", "Consistent", "Consistent", "Contradictory"]
kappa = compute_kappa(sample_pairs, human_labels)
print("Kappa:", kappa)
