# OptiSet-like Methodology Spec

This document defines the methodology for the **OptiSet-like** experiments in this repository.
The goal is to make the setup transparent, reproducible, and easy to explain to advisors/reviewers.

## Scope and thesis alignment

- **Main comparison:** `policygradient` (true set utility) vs `plrank_surrogate` (PL-Rank-1 with singleton/DCG surrogate).
- **Policy family:** Plackett-Luce over passage scores.
- **Utility:** generator-based baseline-subtracted negative mean token NLL.
- **Primary reporting:** held-out utility vs wall-clock and vs total generator forward passes (Oosterhuis-style).
- **Out-of-scope for this stage:** PL-Rank-2/3, full OptiSet Expand-then-Refine data synthesis pipeline.

---

## 1) Dataset and splits

- QA data source: `hotpot_qa` with `distractor` config.
- Candidate pools are precomputed and cached per `(data_seed, retriever, corpus, top_n)`.
- `data_seed` controls deterministic train/val example identity selection.
- `seed` controls training randomness (model init/sampling), independent from `data_seed`.
- Each run writes:
  - `train_ids.json`
  - `val_ids.json`
  - SHA256 of each ID list in `provenance.json`
- Split sanity checks (hard fail):
  - no empty train/val split
  - no duplicates inside train/val
  - no train/val overlap

---

## 2) Corpus format and default corpus choice

Default OptiSet-like corpus choice is **DPR Wikipedia passages**:

- Preferred source: `facebook/wiki_dpr` with a `psgs_w100.*` config. For lightweight local iteration, use `psgs_w100.nq.exact.no_embeddings` (same passages without embedding payloads).
- Iteration subset: `500k` to `1M` passages for local/Colab cost control.

Normalized corpus record format (JSONL):

```json
{"doc_id":"...", "title":"...", "text":"..."}
```

Passage interpretation in this repo:

- one row is one candidate retrievable passage
- `title` is optional but preserved when available
- passage chunking/length is inherited from source corpus preprocessing

---

## 3) Retriever protocol

- **Main retriever:** Contriever (`facebook/contriever-msmarco`), `top_n=20`.
- **Ablation retriever:** BM25, `top_n=20`.
- Candidate pool retrieval is deterministic for fixed:
  - corpus file
  - retriever config
  - `data_seed`
  - train/val size
- For each QA example, pool stores top-20 passages plus metadata.

---

## 4) Candidate-pool cache schema and provenance

Pool JSONL schema:

- `example_id`
- `split` (`train` / `val`)
- `question`
- `gold_answer`
- `passages` (top-N retrieved texts, ordered)
- optional `doc_ids`, `scores`, `retriever`, `top_n`

Run-time provenance:

- `candidate_pool_sha256` is computed from `--candidate_pool_path`.
- Optional deterministic enforcement:
  - pass `--expected_pool_sha256`
  - mismatch => hard fail.
- Run writes `provenance.json` with:
  - pool path/hash + hash-check status
  - seeds/config
  - split hashes
  - baseline metrics
  - compute accounting definition

---

## 5) Generator utility definition

Per-example utility uses a seq2seq generator (`google/flan-t5-small` unless explicitly changed):

1. Build prompt from `(question, selected passages)`.
2. Compute tokenized labels for gold answer.
3. Mask target padding as `-100`.
4. Utility raw value:
   - `u_raw(S) = - mean_token_NLL(q, S, answer)`
5. Baseline-subtracted mode (default):
   - `u(S) = u_raw(S) - u_raw(empty_context)`
   - `empty_context` is explicit placeholder `<none>` in the same prompt template.

Truncation handling:

- Prompt/token truncation is counted on every utility forward pass.
- Passages are token-capped at load time for prompting:
  - `passages_raw`: original retrieved text
  - `passages_for_prompt`: capped with generator tokenizer (`--max_passage_tokens_for_prompt`, default 128)
  - all generator-facing paths (preflight, training, held-out eval, top-K baseline, answer generation) use `passages_for_prompt`.
- Preflight gate uses prompt truncation rate from top-K baseline utility calls.
- If rate exceeds threshold (`--max_prompt_truncation_rate`, default `0.20`), run aborts with guidance.

---

## 6) Compute accounting definitions

`total_generator_forward_passes` includes all utility-related generator forwards:

- training utility evaluations
- empty-context baseline evaluations
- singleton precompute evaluations (surrogate path)
- held-out utility evaluations

`online_generator_forward_passes`:

- `online = total - singleton_precompute_forward_passes`

Backward-compat note:

- legacy column `cumulative_reward_forward_passes` is retained and equals `cumulative_total_generator_forward_passes`.

---

## 7) Evaluation outputs and headline metrics

Per-seed outputs:

- `train_set_utility_policygradient.csv`
- `train_set_utility_plrank_surrogate.csv`
- `summary_table.csv` (includes top-K retriever baseline row)
- `README.txt`
- `provenance.json`
- `preflight_report.json` (for preflight runs)

Aggregated outputs (multi-seed):

- utility-vs-time curve with common cutoff
- utility-vs-total-forward-passes curve with common cutoff
- optional utility-vs-online-forward-passes curve
- normalized AUC (nAUC) metrics under common budgets
- fixed-budget utility point metrics

Headline metrics for efficiency claims:

1. `nAUC utility vs time`
2. `nAUC utility vs total forward passes`
3. utility at shared fixed total-pass budget

---

## Sanity gates (must pass before expensive runs)

1. Pool hash check (if expected hash provided) must pass.
2. Split files and split hashes must be written (`train_ids.json`, `val_ids.json`).
3. Top-K retriever baseline under true utility must be computed/logged.
4. Prompt truncation preflight gate must pass.

If any gate fails, run is considered **diagnostic-only** and not thesis-evidence.

---

## Worst-case plan if OptiSet code never arrives

What we can still claim defensibly:

- We evaluate PL policy-gradient methods in an **OptiSet-like setting**:
  - retrieval-built candidate pools (`k=20`)
  - Wikipedia passage corpus (DPR-style instantiation)
  - generator-based baseline-subtracted utility
  - compute-budget matched reporting.

What may differ from OptiSet implementation details:

- exact prompt templates and generation stack
- corpus preprocessing/chunking and retrieval index details
- absence of full Expand-then-Refine candidate set synthesis/training paradigm.

How differences are documented:

- report corpus/retriever/model hashes and configs
- disclose truncation rates and prompt limits
- label setup as **OptiSet-like**, not exact reproduction.
