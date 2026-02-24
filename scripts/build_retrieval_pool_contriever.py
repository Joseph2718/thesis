import argparse
import json
import os
import pathlib
import re

import numpy as np
import torch
from datasets import load_dataset
from transformers import AutoModel, AutoTokenizer


TOKEN_RE = re.compile(r'\s+')


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


def _normalize_text(text):
  return TOKEN_RE.sub(' ', text.strip())


def _mean_pool(last_hidden_state, attention_mask):
  mask = attention_mask.unsqueeze(-1).float()
  summed = (last_hidden_state * mask).sum(dim=1)
  denom = mask.sum(dim=1).clamp(min=1e-9)
  return summed / denom


def _encode_texts(texts, tokenizer, model, device, batch_size=64, max_len=256):
  embs = []
  with torch.no_grad():
    for s in range(0, len(texts), batch_size):
      batch = texts[s:s + batch_size]
      tok = tokenizer(batch,
                      return_tensors='pt',
                      padding=True,
                      truncation=True,
                      max_length=max_len).to(device)
      out = model(**tok)
      pooled = _mean_pool(out.last_hidden_state, tok['attention_mask'])
      pooled = torch.nn.functional.normalize(pooled, p=2, dim=1)
      embs.append(pooled.cpu().numpy().astype(np.float32))
  return np.concatenate(embs, axis=0)


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
      'hotpotqa_contriever_top%d_train%d_val%d_seed%d_%s.jsonl' % (
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
  parser.add_argument('--model_name', type=str, default='facebook/contriever-msmarco')
  parser.add_argument('--batch_size', type=int, default=64)
  parser.add_argument('--max_len', type=int, default=256)
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
  examples = _load_hotpot_questions(args.train_examples, args.val_examples, args.seed)

  device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
  tokenizer = AutoTokenizer.from_pretrained(args.model_name)
  model = AutoModel.from_pretrained(args.model_name).to(device)
  model.eval()

  corpus_texts = [_normalize_text((doc['title'] + ' ' + doc['text']).strip()) for doc in docs]
  corpus_emb = _encode_texts(corpus_texts, tokenizer, model, device, batch_size=args.batch_size, max_len=args.max_len)
  query_emb = _encode_texts(
      [_normalize_text(ex['question']) for ex in examples],
      tokenizer, model, device, batch_size=args.batch_size, max_len=args.max_len)

  with open(output_path, 'w') as handle:
    for ex_i, ex in enumerate(examples):
      scores = np.matmul(corpus_emb, query_emb[ex_i])
      if args.top_n >= len(docs):
        cand = np.arange(len(docs))
      else:
        cand = np.argpartition(scores, -args.top_n)[-args.top_n:]
      cand = sorted(cand.tolist(), key=lambda i: (-scores[i], docs[i]['doc_id']))[:args.top_n]
      out = {
          'example_id': ex['example_id'],
          'split': ex['split'],
          'question': ex['question'],
          'gold_answer': ex['gold_answer'],
          'passages': [('%s: %s' % (docs[i]['title'], docs[i]['text']) if docs[i].get('title') else docs[i]['text']) for i in cand],
          'doc_ids': [docs[i]['doc_id'] for i in cand],
          'scores': [float(scores[i]) for i in cand],
          'retriever': 'contriever',
          'top_n': args.top_n,
      }
      handle.write(json.dumps(out) + '\n')

  print('Wrote retrieval pool to %s' % output_path)
  print('Examples: %d, top_n: %d, corpus docs: %d' % (len(examples), args.top_n, len(docs)))


if __name__ == '__main__':
  main()
