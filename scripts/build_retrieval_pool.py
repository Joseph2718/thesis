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


def _passage_text(doc):
  if doc.get('title'):
    return '%s: %s' % (doc['title'], doc['text'])
  return doc['text']


# ---------- BM25 ----------

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


# ---------- Contriever ----------

def _build_contriever_index(docs, model_name, batch_size, device):
  from sentence_transformers import SentenceTransformer
  print('Loading Contriever model: %s (device=%s)' % (model_name, device or 'auto'))
  model = SentenceTransformer(model_name, device=device)
  texts = [_passage_text(d) for d in docs]
  print('Encoding %d corpus passages (batch_size=%d)...' % (len(texts), batch_size))
  embeddings = model.encode(texts, batch_size=batch_size, show_progress_bar=True,
                            normalize_embeddings=True)
  embeddings = np.asarray(embeddings, dtype=np.float32)
  return model, embeddings


def _contriever_retrieve(query, model, corpus_embeddings, top_n):
  q_emb = model.encode([query], normalize_embeddings=True)
  q_emb = np.asarray(q_emb, dtype=np.float32)
  scores = (corpus_embeddings @ q_emb.T).squeeze(1)
  if top_n >= scores.shape[0]:
    cand = np.arange(scores.shape[0])
  else:
    cand = np.argpartition(scores, -top_n)[-top_n:]
  cand = sorted(cand.tolist(), key=lambda i: -scores[i])
  cand = cand[:top_n]
  return cand, scores[cand].tolist()


# ---------- Common ----------

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


def _default_output_path(retriever, corpus_path, top_n, train_examples, val_examples, seed):
  corpus_tag = pathlib.Path(corpus_path).stem
  return os.path.join(
      'data', 'retrieval_pools',
      'hotpotqa_%s_top%d_train%d_val%d_seed%d_%s.jsonl' % (
          retriever, top_n, train_examples, val_examples, seed, corpus_tag))


def main():
  parser = argparse.ArgumentParser(
      description='Build a retrieval pool (BM25 or Contriever) for HotpotQA.')
  parser.add_argument('--corpus_path', required=True, type=str)
  parser.add_argument('--retriever', type=str, default='bm25',
                      choices=['bm25', 'contriever'],
                      help='Retrieval method: bm25 or contriever.')
  parser.add_argument('--contriever_model', type=str, default='facebook/contriever-msmarco',
                      help='HuggingFace model name for Contriever retrieval.')
  parser.add_argument('--encode_batch_size', type=int, default=256,
                      help='Batch size for Contriever corpus encoding.')
  parser.add_argument('--device', type=str, default=None,
                      help='Device for Contriever (None=auto, cuda, cpu, mps).')
  parser.add_argument('--output_path', type=str, default=None)
  parser.add_argument('--top_n', type=int, default=20)
  parser.add_argument('--train_examples', type=int, required=True)
  parser.add_argument('--val_examples', type=int, required=True)
  parser.add_argument('--seed', type=int, default=42)
  parser.add_argument('--max_corpus_docs', type=int, default=None)
  parser.add_argument('--overwrite', action='store_true')
  args = parser.parse_args()

  output_path = args.output_path or _default_output_path(
      args.retriever, args.corpus_path, args.top_n,
      args.train_examples, args.val_examples, args.seed)
  out_dir = os.path.dirname(output_path)
  if out_dir:
    os.makedirs(out_dir, exist_ok=True)
  if os.path.exists(output_path) and not args.overwrite:
    print('Using cached retrieval pool at %s (pass --overwrite to rebuild).' % output_path)
    return

  docs = _read_corpus(args.corpus_path, max_docs=args.max_corpus_docs)
  examples = _load_hotpot_questions(args.train_examples, args.val_examples, args.seed)

  if args.retriever == 'bm25':
    postings, doc_lens, avgdl = _build_bm25_index(docs)
    print('BM25 index built: %d docs, avgdl=%.1f' % (len(docs), avgdl))

    def retrieve(query):
      idx, scores = _bm25_retrieve(query, docs, postings, doc_lens, avgdl, args.top_n)
      return idx, [float(s) for s in scores]

  elif args.retriever == 'contriever':
    model, corpus_embs = _build_contriever_index(
        docs, args.contriever_model, args.encode_batch_size, args.device)

    def retrieve(query):
      return _contriever_retrieve(query, model, corpus_embs, args.top_n)

  print('Retrieving top-%d for %d examples...' % (args.top_n, len(examples)))
  with open(output_path, 'w') as handle:
    for i, ex in enumerate(examples):
      top_idx, top_scores = retrieve(ex['question'])
      out = {
          'example_id': ex['example_id'],
          'split': ex['split'],
          'question': ex['question'],
          'gold_answer': ex['gold_answer'],
          'passages': [_passage_text(docs[j]) for j in top_idx],
          'doc_ids': [docs[j]['doc_id'] for j in top_idx],
          'scores': top_scores,
          'retriever': args.retriever,
          'top_n': args.top_n,
      }
      handle.write(json.dumps(out) + '\n')
      if (i + 1) % 100 == 0:
        print('  %d / %d queries done' % (i + 1, len(examples)))

  print('Wrote retrieval pool to %s' % output_path)
  print('Retriever: %s, examples: %d, top_n: %d, corpus docs: %d' % (
      args.retriever, len(examples), args.top_n, len(docs)))


if __name__ == '__main__':
  main()
