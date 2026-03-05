# Retrieval Pools

This folder stores cached retrieval candidate pools used by `scripts/set_utility_experiment.py`.

Each retrieval pool JSONL record has:

- `example_id`
- `split` (`train` or `val`)
- `question`
- `gold_answer`
- `passages` (top-N retrieved texts)
- optional: `doc_ids`, `scores`, `retriever`, `top_n`

## Corpus source options

- **Hotpot corpus (stepping stone / sanity):**
  - built from HotpotQA context fields via `scripts/build_passage_corpus_hotpotqa.py`
  - retrieves over a small corpus constructed from dataset-provided passages
- **Wikipedia corpus (OptiSet-like):**
  - normalized from an external Wikipedia passage file via `scripts/prepare_wikipedia_passage_corpus.py`
  - retrieves over a larger passage corpus external to per-example context

## Retriever options

Both retrievers are supported by `scripts/build_retrieval_pool.py` via `--retriever`:

- BM25: `--retriever bm25` (default)
- Contriever: `--retriever contriever` (uses `facebook/contriever-msmarco` by default)

## Reproducibility and caching

- Retrieval pool construction is deterministic for a given seed and corpus.
- Scripts write cache files named by retriever/top-N/train/val/seed/corpus.
- If output exists, scripts reuse cached results unless `--overwrite` is set.

## Scope note

These retrieval pools provide **OptiSet-like ingredients** (retriever-built candidate pools + generator-based utility), but this repository does **not** implement full OptiSet Expand-then-Refine candidate-set generation.
