import os
from openai import OpenAI
from dotenv import load_dotenv
import json
from typing import List, Dict

from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import torch

load_dotenv("./env")

client = OpenAI(
    base_url="https://models.inference.ai.azure.com",
    api_key="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",
)

# Defined historical events for extraction
EVENTS = [
    {
        "id": "election_night_1860",
        "name": "Election Night 1860",
        "keywords": ["election night", "election", "1860", "november 1860"],
    },
    {
        "id": "fort_sumter",
        "name": "Fort Sumter decision",
        "keywords": ["fort sumter", "sumter", "charleston harbor"],
    },
    {
        "id": "gettysburg_address",
        "name": "Gettysburg Address",
        "keywords": ["gettysburg", "four score", "cemetery", "dedication"],
    },
    {
        "id": "second_inaugural",
        "name": "Second Inaugural Address",
        "keywords": ["second inaugural", "inaugural address", "with malice toward none"],
    },
    {
        "id": "ford_assassination",
        "name": "Ford's Theatre assassination",
        "keywords": ["ford's theatre", "assassination", "booth"],
    },
]

# Data loading functions
def load_json(path: str) -> List[Dict]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_all_sources() -> List[Dict]:
    """
    Load Gutenberg and LoC datasets from data/final_dataset.
    """
    project_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(project_dir, "data", "final_dataset")

    gutenberg_path = os.path.join(data_dir, "gutenberg_authors.json")
    loc_path = os.path.join(data_dir, "lincoln_loc.json")

    sources: List[Dict] = []

    if os.path.exists(gutenberg_path):
        print(f"Loading Gutenberg dataset from {gutenberg_path}")
        sources.extend(load_json(gutenberg_path))
    else:
        print(f"Gutenberg dataset NOT found at {gutenberg_path}")

    if os.path.exists(loc_path):
        print(f"Loading LoC dataset from {loc_path}")
        sources.extend(load_json(loc_path))
    else:
        print(f"LoC dataset NOT found at {loc_path}")

    print(f"📚 Total documents loaded: {len(sources)}\n")
    return sources

# Extract relevant text excerpts for each event
def find_snippets_for_event(text: str, keywords: List[str], max_chars: int = 2000) -> str:
    """
    Simple strategy to avoid feeding the whole book:

    - Search for each keyword in the text.
    - For each hit, grab a window of text around it.
    - Concatenate all windows (up to max_chars) as the LLM context.
    """
    if not text:
        return ""

    lower = text.lower()
    snippets: List[str] = []
    window_size = 400 

    for kw in keywords:
        kw_lower = kw.lower()
        start = 0
        while True:
            idx = lower.find(kw_lower, start)
            if idx == -1:
                break

            w_start = max(0, idx - window_size)
            w_end = min(len(text), idx + len(kw_lower) + window_size)
            snippet = text[w_start:w_end].strip()
            if snippet:
                snippets.append(snippet)

            start = idx + len(kw_lower)

    if not snippets:
        return ""

# Deduplicate similar snippet entries
    seen = set()
    unique_snippets: List[str] = []
    for s in snippets:
        if s not in seen:
            seen.add(s)
            unique_snippets.append(s)

    combined = "\n\n--- SNIPPET SEPARATOR ---\n\n".join(unique_snippets)

    if len(combined) > max_chars:
        combined = combined[:max_chars] + "\n\n[TRUNCATED FOR CONTEXT WINDOW]\n"

    return combined

# Model call for one event-document pair
def call_llm_for_event(document: dict, event: dict, excerpts: str) -> dict:
    """
    Use OpenAI-compatible completions API instead of a local model.
    """

    if not excerpts:
        return {
            "event": event["id"],
            "author": document.get("from") or document.get("author") or None,
            "claims": [],
            "temporal_details": {"date": None, "time": None},
            "tone": None,
        }

    prompt = f"""
You are a careful historian extracting information about a specific historical event.

Event: {event["name"]}
Event ID: {event["id"]}

Document metadata:
- id: {document.get("id")}
- title: {document.get("title")}
- document_type: {document.get("document_type")}
- date: {document.get("date")}
- place: {document.get("place")}
- from: {document.get("from")}
- to: {document.get("to")}

EXCERPTS:
<<<
{excerpts}
>>>

Your task:
1. List concrete factual claims found in these excerpts.
2. Identify any explicit dates or times.
3. Assign a tone ("Supportive", "Critical", "Neutral", etc.)

Return STRICT JSON:

{{
  "event": "{event["id"]}",
  "author": "{document.get("from") or document.get("author") or None}",
  "claims": [],
  "temporal_details": {{"date": null, "time": null}},
  "tone": null
}}
"""

    response = client.chat.completions.create(
        model="gpt-4.1",
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
        max_tokens=500,
    )

    raw = response.choices[0].message.content.strip()

    try:
        start = raw.find("{")
        end = raw.rfind("}")
        json_str = raw[start:end+1]
        data = json.loads(json_str)
    except Exception:
        data = {
            "event": event["id"],
            "author": document.get("from") or document.get("author") or None,
            "claims": [raw],
            "temporal_details": {"date": None, "time": None},
            "tone": None,
        }

    return data

# Full extraction loop
def extract_events_for_all_documents() -> List[Dict]:
    """
    For each document and each event:
      - find relevant snippets (for Books)
      - or use full content (for non-Books)
      - call the LLM
      - store one row *only if there are claims*
    """
    sources = load_all_sources()
    all_results: List[Dict] = []

    for doc in sources:
        doc_id = doc.get("id")
        print(f"Processing document: {doc_id} - {doc.get('title')}")

        content = doc.get("content", "")
        doc_type = doc.get("document_type")
        is_book = (doc_type == "Book")

        for event in EVENTS:
            print(f"  → Event: {event['id']}")

            if is_book:
                excerpts = find_snippets_for_event(content, event["keywords"])
            else:
                max_chars = 2000
                excerpts = content[:max_chars] if content else ""

            if not excerpts:
                print("    (No relevant snippets found; returning empty extraction.)")
            else:
                print(f"    Found snippets (length={len(excerpts)} chars). Calling LLaMA...")

            extraction = call_llm_for_event(
                document=doc,
                event=event,
                excerpts=excerpts,
            )

            claims = extraction.get("claims")
            if claims is None or (isinstance(claims, list) and len(claims) == 0):
                print("    → No claims returned; skipping this event-document pair.")
                continue

            row = {
                "source_id": doc_id,
                "source_title": doc.get("title"),
                "source_type": doc.get("document_type"),
                "event_id": event["id"],
                "event_name": event["name"],
                "extraction": extraction,
            }
            all_results.append(row)

    return all_results

def save_event_extractions(results: List[Dict], out_path: str) -> None:
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\n Saved event extractions to {out_path}")


def main():
    results = extract_events_for_all_documents()
    project_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(project_dir, "data", "final_dataset")
    out_path = os.path.join(data_dir, "events_extracted.json")
    save_event_extractions(results, out_path)


if __name__ == "__main__":
    main()