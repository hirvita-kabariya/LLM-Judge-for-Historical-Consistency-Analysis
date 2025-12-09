# LLM-Judge-for-Historical-Consistency-Analysis
Automated LLM-Based Pipeline for Historical Claim Extraction &amp; Consistency Evaluation

## Overview
- Collects five Project Gutenberg biographies (secondary) and five Library of Congress letters/speeches (primary).
- Normalizes all sources to a shared JSON schema while preserving raw text.
- Extracts claims, temporal details, and tone for five key events via an LLM with context-window controls.
- Runs an LLM judge to score consistency (0–100) between Lincoln and other authors, classifying differences as factual, interpretive, or omissions.
- Provides prompt ablations, self-consistency checks, and Cohen’s kappa vs. manual labels.

## Project Structure
- `gutenberg_books.py` — download + normalize Gutenberg books → `data/final_dataset/gutenberg_authors.json`.
- `lincoln_loc.py` — download + normalize LoC items (with rate-limit sleep) → `data/final_dataset/lincoln_loc.json`.
- `event_extraction.py` — snippet search, prompting, and JSON outputs → `data/final_dataset/events_extracted.json`.
- `validation.py` — judge prompts (zero-shot, CoT, few-shot), robustness, self-consistency, and Cohen’s kappa.
- `data/raw/` — downloaded source files.
- `data/final_dataset/` — normalized datasets and extraction results.

## Setup
1. Python 3.10+ recommended.
2. Install dependencies:
   ```bash
   pip install openai python-dotenv transformers torch scikit-learn numpy requests
   ```
3. Auth: the OpenAI client is configured for GitHub Models. Set `OPENAI_API_KEY` (or update the client init). `event_extraction.py` will also read `./env` if present.

## Data Acquisition
Run collectors (writes raw to `data/raw/...` and normalized JSON to `data/final_dataset/...`):
```bash
python gutenberg_books.py
python lincoln_loc.py
```
Schema: `id`, `title`, `reference`, `document_type`, `date`, `place`, `from`, `to`, `content`. LoC requests pause 1s to respect limits.

## Event Extraction
Events covered: Election Night 1860, Fort Sumter decision, Gettysburg Address, Second Inaugural Address, Ford’s Theatre assassination.

Method:
- Books: keyword search with 400-char windows around hits; merged and truncated to ~2000 chars.
- LoC items: first 2000 chars used directly.
- Prompt asks for claims, temporal details, and tone; enforces strict JSON.

Run:
```bash
python event_extraction.py
```
Output: `data/final_dataset/events_extracted.json` (one row per (document, event) with claims).

## Consistency Evaluation & Validation
- Pairing: non-Book = Lincoln primary; Book = secondary. All primary×secondary pairs per event are judged.
- Outputs: consistency score (0–100), `difference_types` (`factual`, `interpretive`, `omission`), and short reasoning.
- Prompt styles: `zero_shot`, `cot` (default), `few_shot`.

Experiments (in `validation.py`):
- Prompt robustness: `eval_prompt_strategies` (default up to 8 pairs) and `summarize_prompt_results`.
- Self-consistency: `eval_self_consistency` (default 5 runs, temp 0.7) and `summarize_self_consistency`.
- Inter-rater agreement: `compute_kappa` vs. your `human_labels` (threshold 60 for Consistent/Contradictory).

Run full suite:
```bash
python validation.py
```
Before kappa, set `human_labels` in `validation.py` to your manual annotations.

## Results (current configuration)
- Prompt ablation (approx.): zero-shot ≈ 65±12.6, CoT ≈ 62±11.7, few-shot ≈ 60±11.0.
- Self-consistency (5 runs, temp>0): avg std ≈ 10 (min ≈ 7.5, max ≈ 13.3).
- Cohen’s kappa vs. 10 human labels: ~0.58 (moderate agreement).

## Known Limitations
- Keyword windowing can miss indirectly phrased mentions or capture digressions; consider semantic search/reranking.
- Some extractions are vague when prose is reflective; stronger prompt constraints or post-filters may help.
- Hardcoded API keys in scripts; switch to environment-based secrets for safety.

## Extensibility
- Swap keyword windows for embedding search or RAG chunking to improve recall.
- Tune `max_chars`/window sizes per model context limits.
- Persist judge outputs (JSON/CSV) in `validation.py` instead of stdout if needed.
- Expand events or sources by extending the `EVENTS` list and adding new collectors.

aphical alignment/contradiction with moderate human agreement; further gains likely from better snippet retrieval, prompt refinement, and expanded human labeling.

