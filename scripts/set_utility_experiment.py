import argparse
import csv
import hashlib
import json
import os
import pathlib
import random
import re
import string
import sys
import time

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import torch
from datasets import load_dataset
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
  sys.path.insert(0, str(PROJECT_ROOT))

import algorithms.PLRank as plr
import algorithms.tensorflowloss as tfl
import utils.plackettluce as pl

EMPTY_CONTEXT_PLACEHOLDER = '<none>'


def sha256_file(path):
  hasher = hashlib.sha256()
  with open(path, 'rb') as handle:
    while True:
      chunk = handle.read(1024 * 1024)
      if not chunk:
        break
      hasher.update(chunk)
  return hasher.hexdigest()


def sha256_json_payload(payload):
  # Stable JSON serialization ensures deterministic split/provenance hashes.
  blob = json.dumps(payload, sort_keys=True, separators=(',', ':')).encode('utf-8')
  return hashlib.sha256(blob).hexdigest()


def safe_rate(numerator, denominator):
  if denominator <= 0:
    return 0.0
  return float(numerator) / float(denominator)


class DenseEncoder:
  """Encodes text into dense embeddings using a sentence-transformers model."""

  def __init__(self, model_name='facebook/contriever', device_str=None):
    from sentence_transformers import SentenceTransformer
    self.model = SentenceTransformer(model_name, device=device_str)
    self.embed_dim = self.model.get_sentence_embedding_dimension()

  def encode(self, texts, batch_size=64):
    return self.model.encode(
        texts, batch_size=batch_size,
        show_progress_bar=False, normalize_embeddings=True)

  def feature_dim(self):
    return 4 * self.embed_dim + 1

  def featurize(self, query, passages):
    q_emb = self.encode([query])[0]
    if not passages:
      return np.zeros((0, self.feature_dim()), dtype=np.float32)
    p_embs = self.encode(passages)
    q_tile = np.repeat(q_emb[np.newaxis, :], p_embs.shape[0], axis=0)
    interaction = q_tile * p_embs
    absdiff = np.abs(q_tile - p_embs)
    cos_sim = (p_embs @ q_emb).reshape(-1, 1)
    return np.concatenate([q_tile, p_embs, interaction, absdiff, cos_sim], axis=1).astype(np.float32)


def normalize_answer(text):
  text = text.lower().strip()
  text = re.sub(r'\b(a|an|the)\b', ' ', text)
  text = ''.join(ch for ch in text if ch not in set(string.punctuation))
  text = re.sub(r'\s+', ' ', text).strip()
  return text


def compute_em_f1(prediction, gold):
  pred_norm = normalize_answer(prediction)
  gold_norm = normalize_answer(gold)
  em = float(pred_norm == gold_norm)

  pred_tokens = pred_norm.split()
  gold_tokens = gold_norm.split()
  if not pred_tokens and not gold_tokens:
    return em, 1.0
  if not pred_tokens or not gold_tokens:
    return em, 0.0

  gold_counts = {}
  for tok in gold_tokens:
    gold_counts[tok] = gold_counts.get(tok, 0) + 1
  overlap = 0
  for tok in pred_tokens:
    if gold_counts.get(tok, 0) > 0:
      overlap += 1
      gold_counts[tok] -= 1
  if overlap == 0:
    return em, 0.0
  precision = overlap / float(len(pred_tokens))
  recall = overlap / float(len(gold_tokens))
  f1 = 2.0 * precision * recall / (precision + recall)
  return em, f1


class UtilityEvaluator(object):
  def __init__(self,
               tokenizer,
               generator_model,
               device,
               utility_mode='baseline_subtracted',
               max_input_len=512,
               max_target_len=64,
               exact_truncation_check=False):
    self.tokenizer = tokenizer
    self.generator_model = generator_model
    self.device = device
    self.utility_mode = utility_mode
    self.max_input_len = max_input_len
    self.max_target_len = max_target_len
    self.exact_truncation_check = bool(exact_truncation_check)
    self.reward_forward_passes = 0
    self.prompt_truncation_count = 0
    self.target_truncation_count = 0
    self._empty_baseline_cache = {}

  def _tokenize_prompt(self, prompt):
    # Try overflow-aware truncation metadata first (fast tokenizer path).
    was_truncated = False
    try:
      meta = self.tokenizer(
          prompt,
          truncation=True,
          max_length=self.max_input_len,
          return_overflowing_tokens=True)
      if 'num_truncated_tokens' in meta:
        num_trunc = meta['num_truncated_tokens']
        if isinstance(num_trunc, (list, tuple)):
          num_trunc = num_trunc[0] if num_trunc else 0
        was_truncated = int(num_trunc) > 0
      elif 'overflowing_tokens' in meta:
        overflowing = meta['overflowing_tokens']
        if isinstance(overflowing, (list, tuple)) and overflowing:
          first = overflowing[0]
          if isinstance(first, (list, tuple)):
            was_truncated = len(first) > 0
          else:
            was_truncated = len(overflowing) > 0
    except Exception:
      was_truncated = False

    toks = self.tokenizer(
        prompt,
        return_tensors='pt',
        truncation=True,
        max_length=self.max_input_len).to(self.device)
    if not was_truncated:
      if self.exact_truncation_check:
        full_ids = self.tokenizer(
            prompt,
            return_tensors='pt',
            truncation=False).input_ids
        was_truncated = int(full_ids.shape[1]) > int(self.max_input_len)
      else:
        was_truncated = toks.input_ids.shape[1] >= self.max_input_len
    return toks, was_truncated

  def _tokenize_labels(self, gold_answer):
    unpadded = self.tokenizer(gold_answer,
                              return_tensors='pt',
                              truncation=True,
                              max_length=self.max_target_len).input_ids
    was_truncated = unpadded.shape[1] >= self.max_target_len
    labels = self.tokenizer(gold_answer,
                            return_tensors='pt',
                            truncation=True,
                            max_length=self.max_target_len,
                            padding='max_length').input_ids.to(self.device)
    labels = labels.masked_fill(labels == self.tokenizer.pad_token_id, -100)
    return labels, was_truncated

  def compute_neg_nll_utility(self, query, passages, gold_answer):
    prompt = build_rag_input(query, passages)
    model_inputs, prompt_truncated = self._tokenize_prompt(prompt)
    labels, target_truncated = self._tokenize_labels(gold_answer)
    if prompt_truncated:
      self.prompt_truncation_count += 1
    if target_truncated:
      self.target_truncation_count += 1
    with torch.no_grad():
      outputs = self.generator_model(**model_inputs, labels=labels)
    self.reward_forward_passes += 1
    # Utility is negative average token NLL (equivalently negative log-PPL up to exp).
    return float(-outputs.loss.item())

  def compute_set_utility(self, query, passages, gold_answer, baseline_key):
    utility_s = self.compute_neg_nll_utility(query, passages, gold_answer)
    if self.utility_mode == 'raw_neg_nll':
      return utility_s
    if baseline_key not in self._empty_baseline_cache:
      self._empty_baseline_cache[baseline_key] = self.compute_neg_nll_utility(query, [], gold_answer)
    return utility_s - self._empty_baseline_cache[baseline_key]


def build_rag_input(query, passages):
  if passages:
    context = '\n'.join(['[%d] %s' % (i + 1, passage) for i, passage in enumerate(passages)])
  else:
    context = EMPTY_CONTEXT_PLACEHOLDER
  return 'question: %s\ncontext:\n%s\nanswer:' % (query, context)


def cap_passage_for_prompt(tokenizer, passage, max_passage_tokens):
  if max_passage_tokens is None or max_passage_tokens <= 0:
    return passage
  token_ids = tokenizer.encode(
      passage,
      add_special_tokens=False,
      truncation=True,
      max_length=int(max_passage_tokens))
  return tokenizer.decode(token_ids, skip_special_tokens=True)


def annotate_prompt_passages(examples, tokenizer, max_passage_tokens):
  total_raw_chars = 0
  total_capped_chars = 0
  total_passages = 0
  for ex in examples:
    raw_passages = list(ex['passages'])
    capped_passages = [
        cap_passage_for_prompt(tokenizer, passage, max_passage_tokens)
        for passage in raw_passages
    ]
    ex['passages_raw'] = raw_passages
    ex['passages_for_prompt'] = capped_passages
    total_passages += len(raw_passages)
    total_raw_chars += sum(len(p) for p in raw_passages)
    total_capped_chars += sum(len(p) for p in capped_passages)
  return {
      'total_passages': total_passages,
      'mean_raw_chars': (float(total_raw_chars) / float(total_passages)) if total_passages else 0.0,
      'mean_capped_chars': (float(total_capped_chars) / float(total_passages)) if total_passages else 0.0,
  }


def get_prompt_passages(ex, indices):
  src = ex.get('passages_for_prompt', ex['passages'])
  return [src[i] for i in indices]


def generate_answer(query,
                    passages,
                    tokenizer,
                    generator_model,
                    device,
                    max_input_len=512,
                    max_new_tokens=32):
  prompt = build_rag_input(query, passages)
  model_inputs = tokenizer(prompt,
                           return_tensors='pt',
                           truncation=True,
                           max_length=max_input_len).to(device)
  with torch.no_grad():
    out_tokens = generator_model.generate(
        **model_inputs,
        max_new_tokens=max_new_tokens,
        do_sample=False)
  return tokenizer.decode(out_tokens[0], skip_special_tokens=True)


def featurize_passages(query, passages, encoder=None):
  if encoder is not None:
    return encoder.featurize(query, passages)
  q_tokens = set(re.findall(r'\w+', query.lower()))
  q_len = max(len(q_tokens), 1)
  feats = []
  for passage in passages:
    p_tokens = re.findall(r'\w+', passage.lower())
    p_set = set(p_tokens)
    overlap = len(q_tokens.intersection(p_set)) / float(q_len)
    overlap_norm_by_passage = len(q_tokens.intersection(p_set)) / float(max(len(p_set), 1))
    length_feature = min(np.log(1.0 + len(p_tokens)) / 8.0, 1.0)
    feats.append([overlap, overlap_norm_by_passage, length_feature, 1.0])
  return np.asarray(feats, dtype=np.float32)


def parse_hotpot_example(example, max_passages, encoder=None):
  question = example['question']
  answer = example['answer']
  titles = example['context']['title']
  sentences = example['context']['sentences']
  passages = []
  for title, sent_list in zip(titles, sentences):
    text = ' '.join(sent_list).strip()
    if not text:
      continue
    passages.append('%s: %s' % (title, text))
    if len(passages) >= max_passages:
      break
  return {
      'question': question,
      'answer': answer,
      'passages': passages,
      'features': featurize_passages(question, passages, encoder=encoder),
  }


def load_hotpot_subset(max_passages, train_examples, val_examples, seed, encoder=None):
  raw = load_dataset('hotpot_qa', 'distractor', split='train')
  raw = raw.shuffle(seed=seed)
  parsed = []
  for ex in raw:
    parsed_ex = parse_hotpot_example(ex, max_passages=max_passages, encoder=encoder)
    if len(parsed_ex['passages']) >= 2:
      parsed.append(parsed_ex)
    if len(parsed) >= train_examples + val_examples:
      break
  if len(parsed) < train_examples + val_examples:
    raise ValueError('Not enough parsed examples for requested train/val split.')
  train = parsed[:train_examples]
  val = parsed[train_examples:train_examples + val_examples]
  for i, ex in enumerate(train):
    ex['example_id'] = 'train_%d' % i
  for i, ex in enumerate(val):
    ex['example_id'] = 'val_%d' % i
  return train, val


def load_examples_from_candidate_pool(candidate_pool_path, max_passages, train_examples, val_examples, encoder=None):
  train = []
  val = []
  fallback_rows = []
  with open(candidate_pool_path, 'r') as handle:
    for line in handle:
      if not line.strip():
        continue
      row = json.loads(line)
      passages = row['passages'][:max_passages]
      if len(passages) < 2:
        continue
      question = row['question']
      answer = row.get('gold_answer', row.get('answer', ''))
      split = row.get('split', None)
      ex_id = row.get('example_id', '')
      if split is None and ex_id.startswith('train_'):
        split = 'train'
      elif split is None and ex_id.startswith('val_'):
        split = 'val'
      feats = featurize_passages(question, passages, encoder=encoder)
      raw_scores = row.get('scores', None)
      if raw_scores is not None:
        retriever_scores = np.asarray(raw_scores[:len(passages)], dtype=np.float32)
        if retriever_scores.max() > retriever_scores.min():
          retriever_scores = (retriever_scores - retriever_scores.min()) / (retriever_scores.max() - retriever_scores.min())
        feats = np.concatenate([feats, retriever_scores.reshape(-1, 1)], axis=1)
      example = {
          'question': question,
          'answer': answer,
          'passages': passages,
          'features': feats,
          'example_id': ex_id,
      }
      if split == 'train':
        if not ex_id:
          example['example_id'] = 'train_%d' % len(train)
        train.append(example)
      elif split in ('val', 'validation'):
        if not ex_id:
          example['example_id'] = 'val_%d' % len(val)
        val.append(example)
      else:
        fallback_rows.append(example)

  # If split labels are absent, preserve deterministic file order:
  # first train_examples rows are train, next val_examples are validation.
  if fallback_rows:
    needed = train_examples + val_examples
    if len(fallback_rows) < needed:
      raise ValueError('Candidate pool missing split labels and has too few rows (%d < %d).' % (len(fallback_rows), needed))
    if len(train) == 0 and len(val) == 0:
      train = fallback_rows[:train_examples]
      val = fallback_rows[train_examples:train_examples + val_examples]
      for i, ex in enumerate(train):
        if not ex['example_id']:
          ex['example_id'] = 'train_%d' % i
      for i, ex in enumerate(val):
        if not ex['example_id']:
          ex['example_id'] = 'val_%d' % i
    else:
      raise ValueError('Candidate pool mixes split-labeled and unlabeled rows; expected one style.')

  if len(train) < train_examples or len(val) < val_examples:
    raise ValueError('Candidate pool has insufficient rows for requested train/val sizes (%d/%d available).' % (len(train), len(val)))
  return train[:train_examples], val[:val_examples]


def init_scorer(input_dim, hidden_units, seed, dropout=0.0):
  tf.keras.utils.set_random_seed(seed)
  layers = []
  for h in hidden_units:
    layers.append(tf.keras.layers.Dense(h, activation='sigmoid', dtype=tf.float32))
    if dropout > 0:
      layers.append(tf.keras.layers.Dropout(dropout))
  layers.append(tf.keras.layers.Dense(1, activation=None, dtype=tf.float32))
  model = tf.keras.Sequential(layers)
  model.build((None, input_dim))
  return model


def _transform_singleton_gains(raw_gains, transform):
  if transform == 'shift_min':
    return raw_gains - np.min(raw_gains)
  if transform == 'clip_zero':
    return np.maximum(raw_gains, 0.0)
  if transform == 'rank_normalize':
    order = np.argsort(raw_gains)
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(order.shape[0], dtype=np.float64)
    if order.shape[0] <= 1:
      return np.zeros_like(raw_gains, dtype=np.float64)
    return ranks / float(order.shape[0] - 1)
  raise ValueError('Unknown singleton gain transform: %s' % transform)


def precompute_singleton_utilities_raw(examples, utility_evaluator):
  raw_singletons = []
  for ex in examples:
    prompt_passages = ex.get('passages_for_prompt', ex['passages'])
    doc_utils = np.zeros(len(prompt_passages), dtype=np.float64)
    for i, passage in enumerate(prompt_passages):
      doc_utils[i] = utility_evaluator.compute_set_utility(
          ex['question'],
          [passage],
          ex['answer'],
          baseline_key=ex['example_id'])
    raw_singletons.append(np.asarray(doc_utils, dtype=np.float64))
  return raw_singletons


def transform_singleton_utilities(raw_singletons, singleton_gain_transform):
  gains = []
  for raw_utils in raw_singletons:
    transformed = _transform_singleton_gains(
        np.asarray(raw_utils, dtype=np.float64),
        singleton_gain_transform)
    gains.append(np.asarray(transformed, dtype=np.float64))
  return gains


def _doc_counts_for_examples(examples):
  return [len(ex.get('passages_for_prompt', ex['passages'])) for ex in examples]


def _doc_count_stats(doc_counts):
  if not doc_counts:
    return {'min': 0, 'mean': 0.0, 'max': 0}
  arr = np.asarray(doc_counts, dtype=np.float64)
  return {
      'min': int(np.min(arr)),
      'mean': float(np.mean(arr)),
      'max': int(np.max(arr)),
  }


def _save_singleton_cache_npz(raw_singletons, npz_path):
  lengths = np.asarray([int(np.asarray(x).shape[0]) for x in raw_singletons], dtype=np.int32)
  if lengths.size == 0:
    flat = np.zeros((0,), dtype=np.float64)
  else:
    flat = np.concatenate([np.asarray(x, dtype=np.float64) for x in raw_singletons], axis=0)
  np.savez_compressed(npz_path, flat_utilities=flat, lengths=lengths)


def _load_singleton_cache_npz(npz_path):
  with np.load(npz_path) as data:
    flat = np.asarray(data['flat_utilities'], dtype=np.float64)
    lengths = np.asarray(data['lengths'], dtype=np.int32)
  raw_singletons = []
  offset = 0
  for length in lengths:
    length_int = int(length)
    if length_int < 0:
      raise ValueError('Invalid negative singleton length in cache: %d' % length_int)
    raw_singletons.append(np.asarray(flat[offset:offset + length_int], dtype=np.float64))
    offset += length_int
  if offset != flat.shape[0]:
    raise ValueError('Singleton cache flat/length mismatch (%d != %d).' % (offset, flat.shape[0]))
  return raw_singletons


def _validate_singleton_cache_payload(raw_singletons, expected_doc_counts):
  if len(raw_singletons) != len(expected_doc_counts):
    raise ValueError(
        'Singleton cache example count mismatch (%d cached vs %d expected).' % (
            len(raw_singletons), len(expected_doc_counts)))
  for i, (arr, expected_count) in enumerate(zip(raw_singletons, expected_doc_counts)):
    if int(np.asarray(arr).shape[0]) != int(expected_count):
      raise ValueError(
          'Singleton cache doc count mismatch at example %d (%d cached vs %d expected).' % (
              i, int(np.asarray(arr).shape[0]), int(expected_count)))


def precompute_singleton_gains(examples, k, utility_evaluator, singleton_gain_transform):
  del k  # Singleton utilities are independent of cutoff K.
  raw_singletons = precompute_singleton_utilities_raw(examples, utility_evaluator)
  return transform_singleton_utilities(raw_singletons, singleton_gain_transform)


def evaluate_heldout(model,
                     eval_examples,
                     k,
                     utility_evaluator,
                     tokenizer,
                     generator_model,
                     device):
  util_vals = []
  em_vals = []
  f1_vals = []
  for ex in eval_examples:
    scores = model(ex['features'], training=False).numpy()[:, 0]
    ranking = np.argsort(-scores)
    topk = ranking[:min(k, ranking.shape[0])]
    selected_passages = get_prompt_passages(ex, topk)
    util_vals.append(utility_evaluator.compute_set_utility(
        ex['question'],
        selected_passages,
        ex['answer'],
        baseline_key=ex['example_id']))
    pred_answer = generate_answer(
        ex['question'],
        selected_passages,
        tokenizer,
        generator_model,
        device)
    em, f1 = compute_em_f1(pred_answer, ex['answer'])
    em_vals.append(em)
    f1_vals.append(f1)
  return float(np.mean(util_vals)), float(np.mean(em_vals)), float(np.mean(f1_vals))


def evaluate_topk_retriever_baseline(eval_examples, k, utility_evaluator, tokenizer, generator_model, device):
  """Evaluate non-learned baseline: top-K by retriever score (passage order in pool)."""
  util_vals = []
  em_vals = []
  f1_vals = []
  for ex in eval_examples:
    topk = list(range(min(k, len(ex.get('passages_for_prompt', ex['passages'])))))
    selected_passages = get_prompt_passages(ex, topk)
    util_vals.append(utility_evaluator.compute_set_utility(
        ex['question'],
        selected_passages,
        ex['answer'],
        baseline_key=ex['example_id']))
    pred_answer = generate_answer(
        ex['question'],
        selected_passages,
        tokenizer,
        generator_model,
        device)
    em, f1 = compute_em_f1(pred_answer, ex['answer'])
    em_vals.append(em)
    f1_vals.append(f1)
  return float(np.mean(util_vals)), float(np.mean(em_vals)), float(np.mean(f1_vals))


def train_one_method(method_name,
                     model,
                     optimizer,
                     train_examples,
                     singleton_gains,
                     k,
                     num_samples,
                     minibatch_queries,
                     max_steps,
                     eval_every,
                     eval_examples,
                     utility_evaluator,
                     tokenizer,
                     generator_model,
                     device,
                     output_csv,
                     precompute_forward_passes=0,
                     debug_learnability=False):
  rank_weights = np.ones(k, dtype=np.float64)
  if minibatch_queries < 1:
    raise ValueError('minibatch_queries must be >= 1, got %d' % minibatch_queries)
  train_q_stream = list(range(len(train_examples)))
  rng = np.random.RandomState(13)
  rng.shuffle(train_q_stream)
  train_q_cursor = 0
  prev_params = None
  debug_steps = []

  with open(output_csv, 'w', newline='') as handle:
    writer = csv.DictWriter(handle, fieldnames=[
        'method', 'step', 'batch_utility', 'heldout_utility', 'heldout_em_approx',
        'heldout_f1_approx', 'reward_forward_passes_step', 'cumulative_reward_forward_passes',
        'cumulative_total_generator_forward_passes', 'cumulative_online_generator_forward_passes',
        'cumulative_time_ms'])
    writer.writeheader()

    start = time.perf_counter()
    for step in range(1, max_steps + 1):
      prev_reward_fw = utility_evaluator.reward_forward_passes
      batch_qids = []
      for _ in range(minibatch_queries):
        if train_q_cursor >= len(train_q_stream):
          rng.shuffle(train_q_stream)
          train_q_cursor = 0
        batch_qids.append(int(train_q_stream[train_q_cursor]))
        train_q_cursor += 1

      if debug_learnability:
        cur_params = np.concatenate([w.numpy().ravel() for w in model.trainable_variables])
        if prev_params is None:
          prev_params = cur_params.copy()

      with tf.GradientTape() as tape:
        query_losses = []
        query_utilities = []
        debug_feature_stats = []
        debug_score_stats = []
        debug_n_docs = []
        debug_reward_vectors = []
        debug_gain_vectors = []
        debug_doc_weight_vectors = []

        for qid in batch_qids:
          ex = train_examples[qid]
          features = ex['features']
          n_docs = features.shape[0]
          cutoff = min(k, n_docs)
          scores_tf = model(features, training=False)
          np_scores = scores_tf.numpy()[:, 0]

          query_util = None
          if method_name == 'policygradient':
            sampled_rankings = pl.gumbel_sample_rankings(np_scores, num_samples, cutoff=cutoff)[0]
            sampled_rewards = np.array(
                [utility_evaluator.compute_set_utility(
                    ex['question'],
                    get_prompt_passages(ex, ranking[:cutoff]),
                    ex['answer'],
                    baseline_key=ex['example_id'])
                 for ranking in sampled_rankings],
                dtype=np.float64)
            centered_rewards = sampled_rewards - np.mean(sampled_rewards)
            labels_dummy = np.zeros(n_docs, dtype=np.float64)
            query_loss = tf.cast(tfl.policy_gradient(
                rank_weights[:cutoff],
                labels_dummy,
                scores_tf,
                sampled_rankings=sampled_rankings,
                sampled_rewards=centered_rewards), tf.float32)
            query_util = float(np.mean(sampled_rewards))
            if debug_learnability:
              debug_reward_vectors.append(sampled_rewards)
          elif method_name == 'plrank_surrogate':
            gains = singleton_gains[qid]
            doc_weights = plr.PL_rank_1(
                rank_weights[:cutoff],
                gains,
                np_scores,
                n_samples=num_samples)
            query_loss = -tf.reduce_sum(scores_tf[:, 0] * tf.constant(doc_weights, dtype=tf.float32))
            ranking = np.argsort(-np_scores)
            query_util = utility_evaluator.compute_set_utility(
                ex['question'],
                get_prompt_passages(ex, ranking[:cutoff]),
                ex['answer'],
                baseline_key=ex['example_id'])
            if debug_learnability:
              debug_gain_vectors.append(gains)
              debug_doc_weight_vectors.append(doc_weights)
          else:
            raise ValueError('Unknown method: %s' % method_name)

          query_losses.append(query_loss)
          query_utilities.append(float(query_util))
          if debug_learnability:
            debug_n_docs.append(int(n_docs))
            debug_feature_stats.append((
                float(np.mean(features)),
                float(np.std(features)),
                float(np.min(features)),
                float(np.max(features)),
                float(np.mean(features == 0) * 100.0)))
            debug_score_stats.append((
                float(np.mean(np_scores)),
                float(np.std(np_scores)),
                float(np.min(np_scores)),
                float(np.max(np_scores))))

        loss = tf.add_n(query_losses) / float(len(query_losses))
        batch_utility = float(np.mean(query_utilities))

      grads = tape.gradient(loss, model.trainable_variables)
      optimizer.apply_gradients(zip(grads, model.trainable_variables))

      if debug_learnability:
        grad_norm = float(np.sqrt(sum(float(tf.reduce_sum(g**2)) for g in grads if g is not None)))
        new_params = np.concatenate([w.numpy().ravel() for w in model.trainable_variables])
        param_delta = float(np.linalg.norm(new_params - prev_params))
        prev_params = new_params.copy()
        step_debug = {
            'method': method_name,
            'step': step,
            'qid': batch_qids[0],
            'qids': [int(q) for q in batch_qids],
            'minibatch_size': len(batch_qids),
            'n_docs_mean': float(np.mean(debug_n_docs)),
            'n_docs_min': int(np.min(debug_n_docs)),
            'n_docs_max': int(np.max(debug_n_docs)),
            'feature_mean': float(np.mean([s[0] for s in debug_feature_stats])),
            'feature_std': float(np.mean([s[1] for s in debug_feature_stats])),
            'feature_min': float(np.min([s[2] for s in debug_feature_stats])),
            'feature_max': float(np.max([s[3] for s in debug_feature_stats])),
            'feature_pct_zero': float(np.mean([s[4] for s in debug_feature_stats])),
            'score_mean': float(np.mean([s[0] for s in debug_score_stats])),
            'score_std': float(np.mean([s[1] for s in debug_score_stats])),
            'score_min': float(np.min([s[2] for s in debug_score_stats])),
            'score_max': float(np.max([s[3] for s in debug_score_stats])),
            'loss': float(loss.numpy()),
            'grad_norm': grad_norm,
            'param_delta_norm': param_delta,
            'batch_utility': float(batch_utility),
        }
        if method_name == 'policygradient':
          reward_concat = np.concatenate(debug_reward_vectors)
          step_debug['reward_mean'] = float(np.mean(reward_concat))
          step_debug['reward_std'] = float(np.std(reward_concat))
          step_debug['reward_min'] = float(np.min(reward_concat))
          step_debug['reward_max'] = float(np.max(reward_concat))
        elif method_name == 'plrank_surrogate':
          gain_concat = np.concatenate(debug_gain_vectors)
          doc_weight_concat = np.concatenate(debug_doc_weight_vectors)
          step_debug['singleton_gains_mean'] = float(np.mean(gain_concat))
          step_debug['singleton_gains_std'] = float(np.std(gain_concat))
          step_debug['singleton_gains_min'] = float(np.min(gain_concat))
          step_debug['singleton_gains_max'] = float(np.max(gain_concat))
          step_debug['doc_weights_mean'] = float(np.mean(doc_weight_concat))
          step_debug['doc_weights_std'] = float(np.std(doc_weight_concat))
        debug_steps.append(step_debug)
        print('[DEBUG %s step=%d mb=%d] feat_std=%.4f | score_std=%.4f | loss=%.6f | grad_norm=%.6f | param_delta=%.6f | utility=%.4f' % (
            method_name, step, len(batch_qids), step_debug['feature_std'],
            step_debug['score_std'], step_debug['loss'], grad_norm, param_delta, batch_utility))

      heldout_utility = ''
      heldout_em_approx = ''
      heldout_f1_approx = ''
      if step % eval_every == 0:
        heldout_utility, heldout_em_approx, heldout_f1_approx = evaluate_heldout(
            model,
            eval_examples,
            k,
            utility_evaluator,
            tokenizer,
            generator_model,
            device)
      step_reward_forward_passes = utility_evaluator.reward_forward_passes - prev_reward_fw
      cumulative_total_forward_passes = utility_evaluator.reward_forward_passes
      cumulative_online_forward_passes = cumulative_total_forward_passes - int(precompute_forward_passes)

      writer.writerow({
          'method': method_name,
          'step': step,
          'batch_utility': batch_utility,
          'heldout_utility': heldout_utility,
          'heldout_em_approx': heldout_em_approx,
          'heldout_f1_approx': heldout_f1_approx,
          'reward_forward_passes_step': step_reward_forward_passes,
          'cumulative_reward_forward_passes': cumulative_total_forward_passes,
          'cumulative_total_generator_forward_passes': cumulative_total_forward_passes,
          'cumulative_online_generator_forward_passes': cumulative_online_forward_passes,
          'cumulative_time_ms': (time.perf_counter() - start) * 1000.0,
      })
      handle.flush()

  return debug_steps


def read_eval_points(csv_path):
  points = []
  with open(csv_path, newline='') as handle:
    reader = csv.DictReader(handle)
    for row in reader:
      if row['heldout_utility'] != '':
        points.append({
            'step': int(row['step']),
            'time_ms': float(row['cumulative_time_ms']),
            'reward_fw_cum': int(float(row['cumulative_reward_forward_passes'])),
            'utility': float(row['heldout_utility']),
            'em_approx': float(row['heldout_em_approx']),
            'f1_approx': float(row['heldout_f1_approx']),
        })
  return points


def time_to_threshold(points, threshold):
  for point in points:
    if point['utility'] >= threshold:
      return point['step'], point['time_ms']
  return None, None


def plot_convergence(pg_points, pl_points, out_steps, out_time):
  plt.figure(figsize=(7.0, 4.3))
  plt.plot([p['step'] for p in pg_points], [p['utility'] for p in pg_points], label='PolicyGradient')
  plt.plot([p['step'] for p in pl_points], [p['utility'] for p in pl_points], label='PL-Rank surrogate')
  plt.xlabel('Step')
  plt.ylabel('Held-out set utility')
  plt.title('Set Utility Convergence vs Steps')
  plt.legend()
  plt.tight_layout()
  plt.savefig(out_steps, dpi=160)
  plt.close()

  plt.figure(figsize=(7.0, 4.3))
  plt.plot([p['time_ms'] for p in pg_points], [p['utility'] for p in pg_points], label='PolicyGradient')
  plt.plot([p['time_ms'] for p in pl_points], [p['utility'] for p in pl_points], label='PL-Rank surrogate')
  plt.xlabel('Cumulative wall-clock time (ms)')
  plt.ylabel('Held-out set utility')
  plt.title('Set Utility Convergence vs Time')
  plt.legend()
  plt.tight_layout()
  plt.savefig(out_time, dpi=160)
  plt.close()


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument('--output_dir', type=str, default='runs/set_utility_experiment')
  parser.add_argument('--candidate_pool_path', type=str, default=None,
                      help='Optional JSONL retrieval pool. If set, uses retrieved passages instead of Hotpot context passages.')
  parser.add_argument('--generator_model', type=str, default='google/flan-t5-small')
  parser.add_argument('--train_examples', type=int, required=True)
  parser.add_argument('--val_examples', type=int, required=True)
  parser.add_argument('--max_passages', type=int, default=20)
  parser.add_argument('--k', type=int, default=5)
  parser.add_argument('--max_steps', type=int, default=40)
  parser.add_argument('--eval_every', type=int, default=10)
  parser.add_argument('--num_samples', type=int, default=2)
  parser.add_argument('--minibatch_queries', type=int, default=1,
                      help='Number of training queries per optimization step.')
  parser.add_argument('--seed', type=int, default=42)
  parser.add_argument('--learning_rate', type=float, default=0.01)
  parser.add_argument('--utility_mode', type=str, default='baseline_subtracted',
                      choices=['baseline_subtracted', 'raw_neg_nll'])
  parser.add_argument('--singleton_gain_transform', type=str, default='shift_min',
                      choices=['shift_min', 'clip_zero', 'rank_normalize'])
  parser.add_argument('--feature_mode', type=str, default='lexical',
                      choices=['lexical', 'dense'],
                      help='Passage featurization: lexical (4 hand-crafted) or dense (pretrained encoder embeddings).')
  parser.add_argument('--encoder_model', type=str,
                      default='facebook/contriever',
                      help='Sentence-transformers model for dense features (default: contriever for OptiSet-like setup).')
  parser.add_argument('--hidden_units', type=str, default=None,
                      help='Comma-separated hidden layer sizes (default: 64,32 for dense, 32,32 for lexical).')
  parser.add_argument('--dropout', type=float, default=0.0,
                      help='Dropout rate between hidden layers (0 = no dropout).')
  parser.add_argument('--data_seed', type=int, default=42,
                      help='Seed for data subset selection (held fixed across runs to isolate optimizer variance).')
  parser.add_argument('--debug_learnability', type=int, default=0,
                      help='Enable learnability diagnostics (1=on). Prints feature/score/grad/reward stats per step.')
  parser.add_argument('--feature_normalize', type=str, default='none',
                      choices=['none', 'per_query_zscore'],
                      help='Feature normalization: none or per_query_zscore (z-score within each query\'s docs).')
  parser.add_argument('--max_passage_tokens_for_prompt', type=int, default=128,
                      help='Token cap per passage for generator prompts (using generator tokenizer).')
  parser.add_argument('--max_input_len', type=int, default=512,
                      help='Max generator input length used by utility evaluator.')
  parser.add_argument('--max_target_len', type=int, default=64,
                      help='Max generator target length used by utility evaluator.')
  parser.add_argument('--singleton_cache_dir', type=str, default='runs/singleton_cache',
                      help='Directory for cached per-example singleton utilities.')
  parser.add_argument('--compute_singletons', type=int, default=0, choices=[0, 1],
                      help='If 1, compute and write singleton utility cache on MISS. If 0, refuse on MISS.')
  parser.add_argument('--preflight_only', type=int, default=0,
                      help='Run sanity checks only (no training). Computes pool hash, top-K baseline, and truncation rates.')
  parser.add_argument('--max_prompt_truncation_rate', type=float, default=0.20,
                      help='Abort if prompt truncation rate exceeds this threshold (evaluated in preflight baseline).')
  parser.add_argument('--expected_pool_sha256', type=str, default='',
                      help='Optional expected SHA256 for --candidate_pool_path. If provided and mismatched, abort.')
  args = parser.parse_args()

  os.makedirs(args.output_dir, exist_ok=True)
  preflight_only = bool(args.preflight_only)
  expected_pool_sha256 = args.expected_pool_sha256.strip().lower()
  candidate_pool_sha256 = ''
  pool_hash_match = 'not_checked'
  if args.candidate_pool_path:
    candidate_pool_sha256 = sha256_file(args.candidate_pool_path)
    pool_hash_match = 'computed_only'
    print('Candidate pool SHA256: %s' % candidate_pool_sha256)
    if expected_pool_sha256:
      pool_hash_match = str(candidate_pool_sha256 == expected_pool_sha256)
      if candidate_pool_sha256 != expected_pool_sha256:
        raise ValueError(
            'Candidate pool hash mismatch.\n'
            '  expected: %s\n'
            '  observed: %s\n'
            'Refusing to continue because pool determinism check failed.'
            % (expected_pool_sha256, candidate_pool_sha256))
      print('Candidate pool hash check: PASS')

  device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
  print('--- Hardware ---')
  print('PyTorch generator device: %s' % device)
  print('CUDA available: %s' % torch.cuda.is_available())
  if torch.cuda.is_available():
    print('CUDA device: %s' % torch.cuda.get_device_name(0))
  tf_gpus = tf.config.list_physical_devices('GPU')
  print('TF physical GPUs: %s' % (tf_gpus if tf_gpus else 'none (CPU only for scorer)'))
  print('----------------')

  encoder = None
  if args.feature_mode == 'dense':
    enc_device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print('Loading dense encoder: %s (device=%s)' % (args.encoder_model, enc_device))
    encoder = DenseEncoder(model_name=args.encoder_model, device_str=enc_device)
    print('Encoder embedding dim: %d -> feature dim: %d' % (encoder.embed_dim, encoder.feature_dim()))

  if args.hidden_units is not None:
    hidden_units = [int(x) for x in args.hidden_units.split(',')]
  elif args.feature_mode == 'dense':
    hidden_units = [256, 128]
  else:
    hidden_units = [32, 32]

  tokenizer = AutoTokenizer.from_pretrained(args.generator_model)
  generator_model = AutoModelForSeq2SeqLM.from_pretrained(args.generator_model).to(device)
  generator_model.eval()
  utility_evaluator_pg = UtilityEvaluator(
      tokenizer=tokenizer,
      generator_model=generator_model,
      device=device,
      utility_mode=args.utility_mode,
      max_input_len=args.max_input_len,
      max_target_len=args.max_target_len,
      exact_truncation_check=False)
  utility_evaluator_pl = UtilityEvaluator(
      tokenizer=tokenizer,
      generator_model=generator_model,
      device=device,
      utility_mode=args.utility_mode,
      max_input_len=args.max_input_len,
      max_target_len=args.max_target_len,
      exact_truncation_check=False)

  # --- Data loading uses data_seed (fixed across runs) ---
  random.seed(args.data_seed)
  np.random.seed(args.data_seed)
  torch.manual_seed(args.data_seed)

  if args.candidate_pool_path:
    train_examples, val_examples = load_examples_from_candidate_pool(
        candidate_pool_path=args.candidate_pool_path,
        max_passages=args.max_passages,
        train_examples=args.train_examples,
        val_examples=args.val_examples,
        encoder=encoder)
  else:
    train_examples, val_examples = load_hotpot_subset(
        max_passages=args.max_passages,
        train_examples=args.train_examples,
        val_examples=args.val_examples,
        seed=args.data_seed,
        encoder=encoder)

  train_cap_stats = annotate_prompt_passages(
      train_examples,
      tokenizer=tokenizer,
      max_passage_tokens=args.max_passage_tokens_for_prompt)
  val_cap_stats = annotate_prompt_passages(
      val_examples,
      tokenizer=tokenizer,
      max_passage_tokens=args.max_passage_tokens_for_prompt)
  print('Applied passage token cap for prompts: max_passage_tokens_for_prompt=%d' % (
      args.max_passage_tokens_for_prompt))
  print('Prompt passage chars (train): mean_raw=%.1f mean_capped=%.1f over %d passages' % (
      train_cap_stats['mean_raw_chars'], train_cap_stats['mean_capped_chars'], train_cap_stats['total_passages']))
  print('Prompt passage chars (val): mean_raw=%.1f mean_capped=%.1f over %d passages' % (
      val_cap_stats['mean_raw_chars'], val_cap_stats['mean_capped_chars'], val_cap_stats['total_passages']))

  if args.feature_normalize == 'per_query_zscore':
    eps = 1e-6
    for examples in [train_examples, val_examples]:
      for ex in examples:
        feats = ex['features']
        if feats.shape[0] > 1:
          mu = feats.mean(axis=0, keepdims=True)
          sigma = feats.std(axis=0, keepdims=True) + eps
          ex['features'] = ((feats - mu) / sigma).astype(np.float32)
    print('Applied per_query_zscore normalization (eps=%.0e)' % eps)

  input_dim = train_examples[0]['features'].shape[1]
  print('data_seed: %d (data subset), seed: %d (training randomness)' % (args.data_seed, args.seed))
  print('Feature mode: %s, input_dim: %d, feature_normalize: %s, hidden_units: %s, dropout: %.2f' % (
      args.feature_mode, input_dim, args.feature_normalize, hidden_units, args.dropout))

  if args.debug_learnability:
    all_feats = np.concatenate([ex['features'] for ex in train_examples[:5]], axis=0)
    print('[DEBUG] Feature sanity (first 5 train examples, all passages):')
    print('  shape: %s' % (all_feats.shape,))
    print('  mean=%.6f  std=%.6f  min=%.6f  max=%.6f' %
          (all_feats.mean(), all_feats.std(), all_feats.min(), all_feats.max()))
    print('  pct_zero=%.2f%%' % (np.mean(all_feats == 0) * 100))
    for d in range(min(5, all_feats.shape[1])):
      print('  dim[%d] mean=%.6f std=%.6f' % (d, all_feats[:, d].mean(), all_feats[:, d].std()))
    if all_feats.shape[1] > 5:
      print('  ... (%d more feature dims)' % (all_feats.shape[1] - 5))

  train_ids = [ex['example_id'] for ex in train_examples]
  val_ids = [ex['example_id'] for ex in val_examples]
  if not train_ids or not val_ids:
    raise ValueError('Missing train_ids or val_ids; refusing to run without fixed data splits.')
  if len(set(train_ids)) != len(train_ids):
    raise ValueError('Duplicate IDs detected in train split.')
  if len(set(val_ids)) != len(val_ids):
    raise ValueError('Duplicate IDs detected in validation split.')
  if set(train_ids).intersection(set(val_ids)):
    raise ValueError('Train/val ID overlap detected; split reproducibility is invalid.')

  train_ids_path = os.path.join(args.output_dir, 'train_ids.json')
  val_ids_path = os.path.join(args.output_dir, 'val_ids.json')
  with open(train_ids_path, 'w') as f:
    json.dump(train_ids, f, indent=2)
  with open(val_ids_path, 'w') as f:
    json.dump(val_ids, f, indent=2)
  if not os.path.exists(train_ids_path) or not os.path.exists(val_ids_path):
    raise ValueError('Missing train_ids.json/val_ids.json after write; aborting.')
  train_ids_sha256 = sha256_json_payload(train_ids)
  val_ids_sha256 = sha256_json_payload(val_ids)
  print('train_ids SHA256: %s' % train_ids_sha256)
  print('val_ids SHA256:   %s' % val_ids_sha256)

  utility_evaluator_baseline = UtilityEvaluator(
      tokenizer=tokenizer,
      generator_model=generator_model,
      device=device,
      utility_mode=args.utility_mode,
      max_input_len=args.max_input_len,
      max_target_len=args.max_target_len,
      exact_truncation_check=preflight_only)
  baseline_util, baseline_em, baseline_f1 = evaluate_topk_retriever_baseline(
      val_examples, args.k, utility_evaluator_baseline, tokenizer, generator_model, device)
  print('Top-K retriever baseline: utility=%.4f  EM=%.4f  F1=%.4f' % (baseline_util, baseline_em, baseline_f1))
  baseline_prompt_trunc_rate = safe_rate(
      utility_evaluator_baseline.prompt_truncation_count,
      utility_evaluator_baseline.reward_forward_passes)
  baseline_target_trunc_rate = safe_rate(
      utility_evaluator_baseline.target_truncation_count,
      utility_evaluator_baseline.reward_forward_passes)
  print('Baseline truncation rates: prompt=%.2f%% target=%.2f%% (%d/%d prompt truncations)' % (
      100.0 * baseline_prompt_trunc_rate,
      100.0 * baseline_target_trunc_rate,
      utility_evaluator_baseline.prompt_truncation_count,
      utility_evaluator_baseline.reward_forward_passes))

  if baseline_prompt_trunc_rate > args.max_prompt_truncation_rate:
    raise ValueError(
        'Prompt truncation sanity gate failed.\n'
        '  observed prompt truncation rate: %.2f%%\n'
        '  allowed maximum: %.2f%%\n'
        'Suggested fixes: reduce --max_passages, shorten passages in corpus/pool, lower prompt length.'
        % (100.0 * baseline_prompt_trunc_rate, 100.0 * args.max_prompt_truncation_rate))

  if candidate_pool_sha256:
    pool_sha256_for_singleton_cache = candidate_pool_sha256
  else:
    # For static Hotpot context pool, tie cache identity to deterministic split IDs.
    pool_sha256_for_singleton_cache = sha256_json_payload({
        'source': 'hotpot_qa_context_static_pool',
        'train_ids_sha256': train_ids_sha256,
        'val_ids_sha256': val_ids_sha256,
    })

  singleton_cache_key_fields = {
      'pool_sha256': pool_sha256_for_singleton_cache,
      'generator_model': args.generator_model,
      'max_passage_tokens_for_prompt': args.max_passage_tokens_for_prompt,
      'max_input_len': args.max_input_len,
      'max_target_len': args.max_target_len,
      'k': args.k,
      'split': 'train',
  }
  singleton_cache_key = sha256_json_payload(singleton_cache_key_fields)
  singleton_cache_path = os.path.join(args.singleton_cache_dir, singleton_cache_key)
  singleton_cache_npz_path = os.path.join(singleton_cache_path, 'singleton_utilities.npz')
  singleton_cache_metadata_path = os.path.join(singleton_cache_path, 'metadata.json')
  singleton_cache_compute_enabled = bool(args.compute_singletons)
  singleton_cache_expected_doc_counts = _doc_counts_for_examples(train_examples)
  singleton_cache_status = 'miss'
  singleton_cache_note = ''
  singleton_cache_metadata = None
  singleton_cache_write_created = False
  raw_singleton_utilities = None
  print('Singleton cache path: %s' % singleton_cache_path)

  if os.path.exists(singleton_cache_npz_path) and os.path.exists(singleton_cache_metadata_path):
    try:
      with open(singleton_cache_metadata_path) as f:
        singleton_cache_metadata = json.load(f)
      if singleton_cache_metadata.get('cache_key_fields') != singleton_cache_key_fields:
        raise ValueError('cache key fields mismatch')
      if singleton_cache_metadata.get('utility_mode') != args.utility_mode:
        raise ValueError('utility_mode mismatch (%s cached vs %s requested)' % (
            singleton_cache_metadata.get('utility_mode'), args.utility_mode))
      raw_singleton_utilities = _load_singleton_cache_npz(singleton_cache_npz_path)
      _validate_singleton_cache_payload(raw_singleton_utilities, singleton_cache_expected_doc_counts)
      singleton_cache_status = 'hit'
      print('Singleton cache status: HIT')
    except Exception as exc:
      singleton_cache_status = 'miss'
      singleton_cache_note = 'incompatible cache ignored: %s' % str(exc)
      raw_singleton_utilities = None
      print('Singleton cache status: MISS (%s)' % singleton_cache_note)
  else:
    singleton_cache_status = 'miss'
    print('Singleton cache status: MISS')

  if (raw_singleton_utilities is None) and (not singleton_cache_compute_enabled):
    raise ValueError(
        'Singleton cache MISS at %s.\n'
        'Refusing to compute singleton utilities because --compute_singletons=0.\n'
        'Re-run with --compute_singletons 1 to create cache.' % singleton_cache_path)

  provenance = {
      'generator_model': args.generator_model,
      'candidate_pool_path': args.candidate_pool_path or 'hotpot_qa_context_static_pool',
      'candidate_pool_sha256': candidate_pool_sha256,
      'expected_pool_sha256': expected_pool_sha256,
      'pool_hash_match': pool_hash_match,
      'data_seed': args.data_seed,
      'training_seed': args.seed,
      'train_examples': args.train_examples,
      'val_examples': args.val_examples,
      'max_passages': args.max_passages,
      'max_passage_tokens_for_prompt': args.max_passage_tokens_for_prompt,
      'max_input_len': args.max_input_len,
      'max_target_len': args.max_target_len,
      'k': args.k,
      'num_samples': args.num_samples,
      'minibatch_queries': args.minibatch_queries,
      'utility_mode': args.utility_mode,
      'feature_mode': args.feature_mode,
      'feature_normalize': args.feature_normalize,
      'singleton_gain_transform': args.singleton_gain_transform,
      'singleton_cache': {
          'singleton_cache_dir': args.singleton_cache_dir,
          'cache_key_sha256': singleton_cache_key,
          'cache_path': singleton_cache_path,
          'cache_status': singleton_cache_status,
          'cache_note': singleton_cache_note,
          'empty_context_baseline_handling': (
              'Computed on-the-fly by UtilityEvaluator with per-example baseline_key caching.'),
          'cache_key_fields': singleton_cache_key_fields,
          'cache_metadata_path': singleton_cache_metadata_path,
          'cache_npz_path': singleton_cache_npz_path,
          'compute_singletons': int(singleton_cache_compute_enabled),
      },
      'prompt_passage_cap_stats': {
          'train': train_cap_stats,
          'val': val_cap_stats,
      },
      'train_ids_path': train_ids_path,
      'val_ids_path': val_ids_path,
      'train_ids_sha256': train_ids_sha256,
      'val_ids_sha256': val_ids_sha256,
      'topk_retriever_baseline': {
          'heldout_utility': baseline_util,
          'heldout_em_approx': baseline_em,
          'heldout_f1_approx': baseline_f1,
          'cumulative_total_generator_forward_passes': utility_evaluator_baseline.reward_forward_passes,
          'cumulative_online_generator_forward_passes': utility_evaluator_baseline.reward_forward_passes,
          'singleton_precompute_forward_passes': 0,
      },
      'preflight': {
          'prompt_truncation_count': utility_evaluator_baseline.prompt_truncation_count,
          'target_truncation_count': utility_evaluator_baseline.target_truncation_count,
          'total_generator_forward_passes': utility_evaluator_baseline.reward_forward_passes,
          'prompt_truncation_rate': baseline_prompt_trunc_rate,
          'target_truncation_rate': baseline_target_trunc_rate,
          'max_prompt_truncation_rate': args.max_prompt_truncation_rate,
      },
      'compute_accounting_definition': {
          'total_generator_forward_passes': 'All generator forward passes used for utility accounting (training utility + empty-context baseline + singleton precompute + held-out eval).',
          'online_generator_forward_passes': 'total_generator_forward_passes - singleton_precompute_forward_passes.',
      },
      'policygradient_training_note': (
          'PolicyGradient uses a per-query multi-sample baseline: sampled rewards are centered by '
          'subtracting the mean reward across the num_samples rankings for the same query before '
          'computing the policy-gradient loss.'),
  }
  provenance_path = os.path.join(args.output_dir, 'provenance.json')
  with open(provenance_path, 'w') as f:
    json.dump(provenance, f, indent=2)
  preflight_report_path = os.path.join(args.output_dir, 'preflight_report.json')
  with open(preflight_report_path, 'w') as f:
    json.dump(provenance, f, indent=2)

  pl_singleton_precompute_forward_passes = 0
  if raw_singleton_utilities is None:
    print('Singleton cache status: MISS -> computing singleton utilities because --compute_singletons=1')
    os.makedirs(singleton_cache_path, exist_ok=True)
    singleton_started_utc = time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())
    singleton_compute_start_fw = utility_evaluator_pl.reward_forward_passes
    raw_singleton_utilities = precompute_singleton_utilities_raw(
        train_examples,
        utility_evaluator=utility_evaluator_pl)
    pl_singleton_precompute_forward_passes = (
        utility_evaluator_pl.reward_forward_passes - singleton_compute_start_fw)
    _validate_singleton_cache_payload(raw_singleton_utilities, singleton_cache_expected_doc_counts)
    _save_singleton_cache_npz(raw_singleton_utilities, singleton_cache_npz_path)
    singleton_finished_utc = time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())
    singleton_cache_metadata = {
        'cache_key_fields': singleton_cache_key_fields,
        'cache_key_sha256': singleton_cache_key,
        'utility_mode': args.utility_mode,
        'pool_sha256': pool_sha256_for_singleton_cache,
        'pool_path_used': args.candidate_pool_path or 'hotpot_qa_context_static_pool',
        'train_ids_sha256': train_ids_sha256,
        'val_ids_sha256': val_ids_sha256,
        'num_examples': len(train_examples),
        'doc_counts': singleton_cache_expected_doc_counts,
        'doc_count_stats': _doc_count_stats(singleton_cache_expected_doc_counts),
        'generator_model': args.generator_model,
        'max_passage_tokens_for_prompt': args.max_passage_tokens_for_prompt,
        'max_input_len': args.max_input_len,
        'max_target_len': args.max_target_len,
        'k': args.k,
        'singleton_precompute_forward_passes_when_created': int(pl_singleton_precompute_forward_passes),
        'created_utc': singleton_started_utc,
        'written_utc': singleton_finished_utc,
    }
    with open(singleton_cache_metadata_path, 'w') as f:
      json.dump(singleton_cache_metadata, f, indent=2)
    singleton_cache_status = 'computed'
    singleton_cache_write_created = True
    print('Singleton cache write complete: %s' % singleton_cache_path)
  else:
    print('Singleton cache status: HIT (loaded singleton utilities; no singleton precompute this run)')
    pl_singleton_precompute_forward_passes = 0

  singleton_gains = transform_singleton_utilities(
      raw_singleton_utilities,
      singleton_gain_transform=args.singleton_gain_transform)
  provenance['singleton_cache']['cache_status'] = singleton_cache_status
  provenance['singleton_cache']['cache_note'] = singleton_cache_note
  provenance['singleton_cache']['cache_write_created'] = bool(singleton_cache_write_created)
  provenance['singleton_cache']['singleton_precompute_forward_passes_this_run'] = int(
      pl_singleton_precompute_forward_passes)
  if singleton_cache_metadata is not None:
    provenance['singleton_cache']['cache_metadata_summary'] = {
        'num_examples': singleton_cache_metadata.get('num_examples'),
        'doc_count_stats': singleton_cache_metadata.get('doc_count_stats'),
        'created_utc': singleton_cache_metadata.get('created_utc'),
        'written_utc': singleton_cache_metadata.get('written_utc'),
    }

  if args.debug_learnability:
    all_gains = np.concatenate([singleton_gains[i] for i in range(min(5, len(singleton_gains)))])
    print('[DEBUG] Singleton gains (first 5 queries):')
    print('  mean=%.6f  std=%.6f  min=%.6f  max=%.6f' %
          (all_gains.mean(), all_gains.std(), all_gains.min(), all_gains.max()))
    print('  pct_zero=%.2f%%  nonzero=%d/%d' %
          (np.mean(all_gains == 0) * 100, np.count_nonzero(all_gains), len(all_gains)))

  if preflight_only:
    with open(preflight_report_path, 'w') as f:
      json.dump(provenance, f, indent=2)
    print('Preflight PASS: wrote %s' % preflight_report_path)
    return

  # --- Re-seed with training seed before model init and training ---
  random.seed(args.seed)
  np.random.seed(args.seed)
  tf.keras.utils.set_random_seed(args.seed)
  torch.manual_seed(args.seed)

  scorer_pg = init_scorer(input_dim=input_dim, hidden_units=hidden_units, seed=args.seed, dropout=args.dropout)
  scorer_pl = init_scorer(input_dim=input_dim, hidden_units=hidden_units, seed=args.seed, dropout=args.dropout)
  shared_weights = [w.copy() for w in scorer_pg.get_weights()]
  scorer_pl.set_weights([w.copy() for w in shared_weights])

  opt_pg = tf.keras.optimizers.SGD(learning_rate=args.learning_rate)
  opt_pl = tf.keras.optimizers.SGD(learning_rate=args.learning_rate)

  pg_csv = os.path.join(args.output_dir, 'train_set_utility_policygradient.csv')
  pl_csv = os.path.join(args.output_dir, 'train_set_utility_plrank_surrogate.csv')
  debug = bool(args.debug_learnability)
  pg_debug = train_one_method(
      method_name='policygradient',
      model=scorer_pg,
      optimizer=opt_pg,
      train_examples=train_examples,
      singleton_gains=singleton_gains,
      k=args.k,
      num_samples=args.num_samples,
      minibatch_queries=args.minibatch_queries,
      max_steps=args.max_steps,
      eval_every=args.eval_every,
      eval_examples=val_examples,
      utility_evaluator=utility_evaluator_pg,
      tokenizer=tokenizer,
      generator_model=generator_model,
      device=device,
      output_csv=pg_csv,
      precompute_forward_passes=0,
      debug_learnability=debug)
  pl_debug = train_one_method(
      method_name='plrank_surrogate',
      model=scorer_pl,
      optimizer=opt_pl,
      train_examples=train_examples,
      singleton_gains=singleton_gains,
      k=args.k,
      num_samples=args.num_samples,
      minibatch_queries=args.minibatch_queries,
      max_steps=args.max_steps,
      eval_every=args.eval_every,
      eval_examples=val_examples,
      utility_evaluator=utility_evaluator_pl,
      tokenizer=tokenizer,
      generator_model=generator_model,
      device=device,
      output_csv=pl_csv,
      precompute_forward_passes=pl_singleton_precompute_forward_passes,
      debug_learnability=debug)

  if debug and (pg_debug or pl_debug):
    debug_path = os.path.join(args.output_dir, 'debug_stats.json')
    with open(debug_path, 'w') as f:
      json.dump({'policygradient': pg_debug, 'plrank_surrogate': pl_debug}, f, indent=2)
    print('Wrote debug diagnostics to %s' % debug_path)

  pg_points = read_eval_points(pg_csv)
  pl_points = read_eval_points(pl_csv)
  out_steps = os.path.join(args.output_dir, 'convergence_set_utility_vs_steps.png')
  out_time = os.path.join(args.output_dir, 'convergence_set_utility_vs_time.png')
  plot_convergence(pg_points, pl_points, out_steps, out_time)

  best_util = max([p['utility'] for p in pg_points + pl_points])
  threshold = 0.95 * best_util
  pg_step, pg_ms = time_to_threshold(pg_points, threshold)
  pl_step, pl_ms = time_to_threshold(pl_points, threshold)

  summary_csv = os.path.join(args.output_dir, 'summary_table.csv')
  with open(summary_csv, 'w', newline='') as handle:
    writer = csv.DictWriter(handle, fieldnames=[
        'method', 'final_heldout_utility', 'final_heldout_em_approx', 'final_heldout_f1_approx',
        'threshold_utility', 'steps_to_threshold', 'time_ms_to_threshold',
        'cumulative_total_generator_forward_passes', 'cumulative_online_generator_forward_passes',
        'cumulative_reward_forward_passes', 'singleton_precompute_forward_passes',
        'prompt_truncation_count', 'target_truncation_count'])
    writer.writeheader()
    pg_final = pg_points[-1]
    pl_final = pl_points[-1]
    writer.writerow({
        'method': 'policygradient',
        'final_heldout_utility': pg_final['utility'],
        'final_heldout_em_approx': pg_final['em_approx'],
        'final_heldout_f1_approx': pg_final['f1_approx'],
        'threshold_utility': threshold,
        'steps_to_threshold': 'N/A' if pg_step is None else pg_step,
        'time_ms_to_threshold': 'N/A' if pg_ms is None else pg_ms,
        'cumulative_total_generator_forward_passes': utility_evaluator_pg.reward_forward_passes,
        'cumulative_online_generator_forward_passes': utility_evaluator_pg.reward_forward_passes,
        'cumulative_reward_forward_passes': utility_evaluator_pg.reward_forward_passes,
        'singleton_precompute_forward_passes': 0,
        'prompt_truncation_count': utility_evaluator_pg.prompt_truncation_count,
        'target_truncation_count': utility_evaluator_pg.target_truncation_count,
    })
    writer.writerow({
        'method': 'plrank_surrogate',
        'final_heldout_utility': pl_final['utility'],
        'final_heldout_em_approx': pl_final['em_approx'],
        'final_heldout_f1_approx': pl_final['f1_approx'],
        'threshold_utility': threshold,
        'steps_to_threshold': 'N/A' if pl_step is None else pl_step,
        'time_ms_to_threshold': 'N/A' if pl_ms is None else pl_ms,
        'cumulative_total_generator_forward_passes': utility_evaluator_pl.reward_forward_passes,
        'cumulative_online_generator_forward_passes': (
            utility_evaluator_pl.reward_forward_passes - pl_singleton_precompute_forward_passes),
        'cumulative_reward_forward_passes': utility_evaluator_pl.reward_forward_passes,
        'singleton_precompute_forward_passes': pl_singleton_precompute_forward_passes,
        'prompt_truncation_count': utility_evaluator_pl.prompt_truncation_count,
        'target_truncation_count': utility_evaluator_pl.target_truncation_count,
    })
    writer.writerow({
        'method': 'topk_retriever_baseline',
        'final_heldout_utility': baseline_util,
        'final_heldout_em_approx': baseline_em,
        'final_heldout_f1_approx': baseline_f1,
        'threshold_utility': '',
        'steps_to_threshold': 'N/A',
        'time_ms_to_threshold': 'N/A',
        'cumulative_total_generator_forward_passes': utility_evaluator_baseline.reward_forward_passes,
        'cumulative_online_generator_forward_passes': utility_evaluator_baseline.reward_forward_passes,
        'cumulative_reward_forward_passes': utility_evaluator_baseline.reward_forward_passes,
        'singleton_precompute_forward_passes': 0,
        'prompt_truncation_count': utility_evaluator_baseline.prompt_truncation_count,
        'target_truncation_count': utility_evaluator_baseline.target_truncation_count,
    })

  readme_path = os.path.join(args.output_dir, 'README.txt')
  with open(readme_path, 'w') as handle:
    handle.write('Generator model: %s\n' % args.generator_model)
    if args.candidate_pool_path:
      handle.write('Candidate pool source: %s\n' % args.candidate_pool_path)
      handle.write('Candidate pool SHA256: %s\n' % candidate_pool_sha256)
      handle.write('Using retrieved passages as fixed top-N candidate pool.\n')
    else:
      handle.write('Static candidate pool note: HotpotQA context passages are used as a fixed top-N candidate pool.\n')
      handle.write('This is not a full retriever pipeline over Wikipedia.\n')
    handle.write('Pool hash check status: %s\n' % pool_hash_match)
    handle.write('Feature mode: %s\n' % args.feature_mode)
    if args.feature_mode == 'dense':
      handle.write('Encoder model: %s\n' % args.encoder_model)
      handle.write('Feature dim: %d (embed_dim + cosine_sim)\n' % input_dim)
    handle.write('Hidden units: %s, dropout: %.2f\n' % (hidden_units, args.dropout))
    handle.write('Empty-context baseline uses explicit placeholder: %s\n' % EMPTY_CONTEXT_PLACEHOLDER)
    handle.write('Utility mode: %s\n' % args.utility_mode)
    handle.write('Utility definition: negative mean token NLL; baseline_subtracted uses u(S)=u_raw(S)-u_raw(empty).\n')
    handle.write(
        'PolicyGradient variance reduction: per-query multi-sample baseline; sampled rewards are centered '
        'by subtracting their within-query mean across num_samples rankings.\n')
    handle.write('Singleton gain transform: %s (default/production: shift_min; preserves per-query relative differences while enforcing nonnegativity).\n' % args.singleton_gain_transform)
    handle.write('EM/F1 are approximate normalized metrics (not the official Hotpot script).\n')
    handle.write('Data seed: %d (fixed for data subset selection). Training seed: %d.\n' % (args.data_seed, args.seed))
    handle.write('Passage prompt token cap (generator tokenizer): %d\n' % args.max_passage_tokens_for_prompt)
    handle.write('Utility evaluator token limits: max_input_len=%d, max_target_len=%d\n' % (
        args.max_input_len, args.max_target_len))
    handle.write('Singleton cache path: %s\n' % singleton_cache_path)
    handle.write('Singleton cache status: %s\n' % singleton_cache_status)
    handle.write('Singleton cache key SHA256: %s\n' % singleton_cache_key)
    handle.write('Singleton cache write-created this run: %s\n' % str(bool(singleton_cache_write_created)))
    handle.write('Singleton empty-context baseline handling: computed on-the-fly with per-example baseline_key cache in UtilityEvaluator.\n')
    handle.write('train_ids.json SHA256: %s\n' % train_ids_sha256)
    handle.write('val_ids.json SHA256: %s\n' % val_ids_sha256)
    handle.write('Preflight baseline prompt truncation: %.2f%% (threshold %.2f%%)\n' % (
        100.0 * baseline_prompt_trunc_rate, 100.0 * args.max_prompt_truncation_rate))
    handle.write('Compute accounting:\n')
    handle.write('- total_generator_forward_passes includes training utility + empty-context baseline + singleton precompute + held-out eval.\n')
    handle.write('- online_generator_forward_passes = total_generator_forward_passes - singleton_precompute_forward_passes.\n')

  provenance['run_results'] = {
      'policygradient': {
          'cumulative_total_generator_forward_passes': utility_evaluator_pg.reward_forward_passes,
          'cumulative_online_generator_forward_passes': utility_evaluator_pg.reward_forward_passes,
          'singleton_precompute_forward_passes': 0,
          'prompt_truncation_count': utility_evaluator_pg.prompt_truncation_count,
          'target_truncation_count': utility_evaluator_pg.target_truncation_count,
      },
      'plrank_surrogate': {
          'cumulative_total_generator_forward_passes': utility_evaluator_pl.reward_forward_passes,
          'cumulative_online_generator_forward_passes': (
              utility_evaluator_pl.reward_forward_passes - pl_singleton_precompute_forward_passes),
          'singleton_precompute_forward_passes': pl_singleton_precompute_forward_passes,
          'prompt_truncation_count': utility_evaluator_pl.prompt_truncation_count,
          'target_truncation_count': utility_evaluator_pl.target_truncation_count,
      },
  }
  with open(provenance_path, 'w') as f:
    json.dump(provenance, f, indent=2)

  print('Wrote set-utility experiment outputs to %s' % args.output_dir)
  print('Generator model: %s' % args.generator_model)
  print('Reward forward passes: PG=%d, PL-surrogate=%d' % (
      utility_evaluator_pg.reward_forward_passes, utility_evaluator_pl.reward_forward_passes))
  print('Prompt truncation counts: PG=%d, PL-surrogate=%d' % (
      utility_evaluator_pg.prompt_truncation_count, utility_evaluator_pl.prompt_truncation_count))
  print('Note: EM/F1 reported here are approximate metrics.')
  print('PolicyGradient threshold: steps=%s time_ms=%s' % (
      'N/A' if pg_step is None else pg_step,
      'N/A' if pg_ms is None else '%.3f' % pg_ms))
  print('PL-Rank surrogate threshold: steps=%s time_ms=%s' % (
      'N/A' if pl_step is None else pl_step,
      'N/A' if pl_ms is None else '%.3f' % pl_ms))


if __name__ == '__main__':
  main()
