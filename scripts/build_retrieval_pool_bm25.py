import argparse
import collections
import json
import os
import pathlib
import re

import numpy as np
from datasets import load_dataset


TOKEN_RE = re.compile(r'\w+')


def _tokenize(text):
  return TOKEN_RE.findall(text.lower())


def _read_corpus(corpus_path, max_docs=None):
  docs = []
  with open(corpus_path, 'r') as handle:
    for i, line in enumerate(handle):
      if max_docs is not None and i >= max_docs:
        break
      if not line.strip():
        continue
      row = json.loads(line)
      docs.append({
          'doc_id': str(row['doc_id']),
          'title': str(row.get('title', '')),
          'text': str(row['text']),
      })
  if not docs:
    raise ValueError('Corpus is empty: %s' % corpus_path)
  return docs


def _build_bm25_index(docs):
  postings = {}
  doc_lens = np.zeros(len(docs), dtype=np.float64)
  for idx, doc in enumerate(docs):
    toks = _tokenize(doc['text'])
    doc_lens[idx] = len(toks)
    tf = collections.Counter(toks)
    for term, freq in tf.items():
      postings.setdefault(term, []).append((idx, float(freq)))
  avgdl = float(np.mean(doc_lens))
  return postings, doc_lens, avgdl


def _bm25_retrieve(query, docs, postings, doc_lens, avgdl, top_n, k1=0.9, b=0.4):
  q_counts = collections.Counter(_tokenize(query))
  n_docs = len(docs)
  scores = np.zeros(n_docs, dtype=np.float64)
  for term, qtf in q_counts.items():
    plist = postings.get(term, None)
    if not plist:
      continue
    df = float(len(plist))
    idf = np.log(1.0 + (n_docs - df + 0.5) / (df + 0.5))
    for doc_idx, tf in plist:
      denom = tf + k1 * (1.0 - b + b * doc_lens[doc_idx] / avgdl)
      scores[doc_idx] += qtf * idf * (tf * (k1 + 1.0) / denom)

  if top_n >= n_docs:
    cand = np.arange(n_docs)
  else:
    cand = np.argpartition(scores, -top_n)[-top_n:]
  cand = sorted(cand.tolist(), key=lambda i: (-scores[i], docs[i]['doc_id']))
  cand = cand[:top_n]
  return cand, scores[cand]


def _load_hotpot_questions(train_examples, val_examples, seed):
  raw = load_dataset('hotpot_qa', 'distractor', split='train').shuffle(seed=seed)
  picked = raw.select(range(train_examples + val_examples))
  rows = []
  for i, ex in enumerate(picked):
    split = 'train' if i < train_examples else 'val'
    rows.append({
        'example_id': '%s_%d' % (split, i if split == 'train' else i - train_examples),
        'split': split,
        'question': ex['question'],
        'gold_answer': ex['answer'],
    })
  return rows


def _default_output_path(corpus_path, top_n, train_examples, val_examples, seed):
  corpus_tag = pathlib.Path(corpus_path).stem
  return os.path.join(
      'data', 'retrieval_pools',
      'hotpotqa_bm25_top%d_train%d_val%d_seed%d_%s.jsonl' % (
          top_n, train_examples, val_examples, seed, corpus_tag))


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument('--corpus_path', required=True, type=str)
  parser.add_argument('--output_path', type=str, default=None)
  parser.add_argument('--top_n', type=int, default=20)
  parser.add_argument('--train_examples', type=int, default=200)
  parser.add_argument('--val_examples', type=int, default=80)
  parser.add_argument('--seed', type=int, default=42)
  parser.add_argument('--max_corpus_docs', type=int, default=None)
  parser.add_argument('--overwrite', action='store_true')
  args = parser.parse_args()

  output_path = args.output_path or _default_output_path(
      args.corpus_path, args.top_n, args.train_examples, args.val_examples, args.seed)
  out_dir = os.path.dirname(output_path)
  if out_dir:
    os.makedirs(out_dir, exist_ok=True)
  if os.path.exists(output_path) and not args.overwrite:
    print('Using cached retrieval pool at %s (pass --overwrite to rebuild).' % output_path)
    return

  docs = _read_corpus(args.corpus_path, max_docs=args.max_corpus_docs)
  postings, doc_lens, avgdl = _build_bm25_index(docs)
  examples = _load_hotpot_questions(args.train_examples, args.val_examples, args.seed)

  with open(output_path, 'w') as handle:
    for ex in examples:
      top_idx, top_scores = _bm25_retrieve(
          ex['question'],
          docs,
          postings,
          doc_lens,
          avgdl,
          top_n=args.top_n)
      out = {
          'example_id': ex['example_id'],
          'split': ex['split'],
          'question': ex['question'],
          'gold_answer': ex['gold_answer'],
          'passages': [('%s: %s' % (docs[i]['title'], docs[i]['text']) if docs[i].get('title') else docs[i]['text']) for i in top_idx],
          'doc_ids': [docs[i]['doc_id'] for i in top_idx],
          'scores': [float(s) for s in top_scores],
          'retriever': 'bm25',
          'top_n': args.top_n,
      }
      handle.write(json.dumps(out) + '\n')

  print('Wrote retrieval pool to %s' % output_path)
  print('Examples: %d, top_n: %d, corpus docs: %d' % (len(examples), args.top_n, len(docs)))


if __name__ == '__main__':
  main()
