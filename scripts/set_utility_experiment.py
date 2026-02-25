import argparse
import csv
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


class DenseEncoder:
  """Encodes text into dense embeddings using a sentence-transformers model."""

  def __init__(self, model_name='sentence-transformers/all-MiniLM-L6-v2', device_str='cpu'):
    from sentence_transformers import SentenceTransformer
    self.model = SentenceTransformer(model_name, device=device_str)
    self.embed_dim = self.model.get_sentence_embedding_dimension()

  def encode(self, texts, batch_size=64):
    return self.model.encode(
        texts, batch_size=batch_size,
        show_progress_bar=False, normalize_embeddings=True)

  def featurize(self, query, passages):
    q_emb = self.encode([query])[0]
    if not passages:
      return np.zeros((0, self.embed_dim + 1), dtype=np.float32)
    p_embs = self.encode(passages)
    interaction = q_emb[np.newaxis, :] * p_embs
    cos_sim = (p_embs @ q_emb).reshape(-1, 1)
    return np.concatenate([interaction, cos_sim], axis=1).astype(np.float32)


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
               max_target_len=64):
    self.tokenizer = tokenizer
    self.generator_model = generator_model
    self.device = device
    self.utility_mode = utility_mode
    self.max_input_len = max_input_len
    self.max_target_len = max_target_len
    self.reward_forward_passes = 0
    self.prompt_truncation_count = 0
    self.target_truncation_count = 0
    self._empty_baseline_cache = {}

  def _tokenize_prompt(self, prompt):
    toks = self.tokenizer(prompt,
                          return_tensors='pt',
                          truncation=True,
                          max_length=self.max_input_len).to(self.device)
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


def precompute_singleton_gains(examples, k, utility_evaluator, singleton_gain_transform):
  rank_weights = np.ones(k, dtype=np.float64)
  gains = []
  for ex in examples:
    doc_gains = np.zeros(len(ex['passages']), dtype=np.float64)
    for i, passage in enumerate(ex['passages']):
      utility = utility_evaluator.compute_set_utility(
          ex['question'],
          [passage],
          ex['answer'],
          baseline_key=ex['example_id'])
      doc_gains[i] = utility * rank_weights[0]
    transformed = _transform_singleton_gains(doc_gains, singleton_gain_transform)
    gains.append(np.asarray(transformed, dtype=np.float64))
  return gains


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
    selected_passages = [ex['passages'][idx] for idx in topk]
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
    topk = list(range(min(k, len(ex['passages']))))
    selected_passages = [ex['passages'][idx] for idx in topk]
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
                     max_steps,
                     eval_every,
                     eval_examples,
                     utility_evaluator,
                     tokenizer,
                     generator_model,
                     device,
                     output_csv):
  rank_weights = np.ones(k, dtype=np.float64)
  train_q_stream = list(range(len(train_examples)))
  rng = np.random.RandomState(13)

  with open(output_csv, 'w', newline='') as handle:
    writer = csv.DictWriter(handle, fieldnames=[
        'method', 'step', 'batch_utility', 'heldout_utility', 'heldout_em_approx',
        'heldout_f1_approx', 'reward_forward_passes_step', 'cumulative_reward_forward_passes',
        'cumulative_time_ms'])
    writer.writeheader()

    start = time.perf_counter()
    for step in range(1, max_steps + 1):
      prev_reward_fw = utility_evaluator.reward_forward_passes
      qid = train_q_stream[(step - 1) % len(train_q_stream)]
      if step % len(train_q_stream) == 1:
        rng.shuffle(train_q_stream)

      ex = train_examples[qid]
      features = ex['features']
      n_docs = features.shape[0]
      cutoff = min(k, n_docs)

      with tf.GradientTape() as tape:
        scores_tf = model(features, training=False)
        np_scores = scores_tf.numpy()[:, 0]

        if method_name == 'policygradient':
          sampled_rankings = pl.gumbel_sample_rankings(np_scores, num_samples, cutoff=cutoff)[0]
          sampled_rewards = np.array(
              [utility_evaluator.compute_set_utility(
                  ex['question'],
                  [ex['passages'][doc_idx] for doc_idx in ranking[:cutoff]],
                  ex['answer'],
                  baseline_key=ex['example_id'])
               for ranking in sampled_rankings],
              dtype=np.float64)
          labels_dummy = np.zeros(n_docs, dtype=np.float64)
          loss = tf.cast(tfl.policy_gradient(
              rank_weights[:cutoff],
              labels_dummy,
              scores_tf,
              sampled_rankings=sampled_rankings,
              sampled_rewards=sampled_rewards), tf.float32)
          batch_utility = float(np.mean(sampled_rewards))
        elif method_name == 'plrank_surrogate':
          gains = singleton_gains[qid]
          doc_weights = plr.PL_rank_1(
              rank_weights[:cutoff],
              gains,
              np_scores,
              n_samples=num_samples)
          loss = -tf.reduce_sum(scores_tf[:, 0] * tf.constant(doc_weights, dtype=tf.float32))
          ranking = np.argsort(-np_scores)
          batch_utility = utility_evaluator.compute_set_utility(
              ex['question'],
              [ex['passages'][doc_idx] for doc_idx in ranking[:cutoff]],
              ex['answer'],
              baseline_key=ex['example_id'])
        else:
          raise ValueError('Unknown method: %s' % method_name)

      grads = tape.gradient(loss, model.trainable_variables)
      optimizer.apply_gradients(zip(grads, model.trainable_variables))

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

      writer.writerow({
          'method': method_name,
          'step': step,
          'batch_utility': batch_utility,
          'heldout_utility': heldout_utility,
          'heldout_em_approx': heldout_em_approx,
          'heldout_f1_approx': heldout_f1_approx,
          'reward_forward_passes_step': step_reward_forward_passes,
          'cumulative_reward_forward_passes': utility_evaluator.reward_forward_passes,
          'cumulative_time_ms': (time.perf_counter() - start) * 1000.0,
      })
      handle.flush()


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
  parser.add_argument('--train_examples', type=int, default=200)
  parser.add_argument('--val_examples', type=int, default=80)
  parser.add_argument('--max_passages', type=int, default=20)
  parser.add_argument('--k', type=int, default=5)
  parser.add_argument('--max_steps', type=int, default=40)
  parser.add_argument('--eval_every', type=int, default=10)
  parser.add_argument('--num_samples', type=int, default=2)
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
                      default='sentence-transformers/all-MiniLM-L6-v2',
                      help='Sentence-transformers model for dense features.')
  parser.add_argument('--hidden_units', type=str, default=None,
                      help='Comma-separated hidden layer sizes (default: 64,32 for dense, 32,32 for lexical).')
  parser.add_argument('--dropout', type=float, default=0.0,
                      help='Dropout rate between hidden layers (0 = no dropout).')
  parser.add_argument('--data_seed', type=int, default=42,
                      help='Seed for data subset selection (held fixed across runs to isolate optimizer variance).')
  args = parser.parse_args()

  os.makedirs(args.output_dir, exist_ok=True)

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
    print('Encoder embedding dim: %d -> feature dim: %d' % (encoder.embed_dim, encoder.embed_dim + 1))

  if args.hidden_units is not None:
    hidden_units = [int(x) for x in args.hidden_units.split(',')]
  elif args.feature_mode == 'dense':
    hidden_units = [64, 32]
  else:
    hidden_units = [32, 32]

  tokenizer = AutoTokenizer.from_pretrained(args.generator_model)
  generator_model = AutoModelForSeq2SeqLM.from_pretrained(args.generator_model).to(device)
  generator_model.eval()
  utility_evaluator_pg = UtilityEvaluator(
      tokenizer=tokenizer,
      generator_model=generator_model,
      device=device,
      utility_mode=args.utility_mode)
  utility_evaluator_pl = UtilityEvaluator(
      tokenizer=tokenizer,
      generator_model=generator_model,
      device=device,
      utility_mode=args.utility_mode)

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

  input_dim = train_examples[0]['features'].shape[1]
  print('data_seed: %d (data subset), seed: %d (training randomness)' % (args.data_seed, args.seed))
  print('Feature mode: %s, input_dim: %d, hidden_units: %s, dropout: %.2f' % (
      args.feature_mode, input_dim, hidden_units, args.dropout))

  train_ids = [ex['example_id'] for ex in train_examples]
  val_ids = [ex['example_id'] for ex in val_examples]
  with open(os.path.join(args.output_dir, 'train_ids.json'), 'w') as f:
    json.dump(train_ids, f)
  with open(os.path.join(args.output_dir, 'val_ids.json'), 'w') as f:
    json.dump(val_ids, f)

  utility_evaluator_baseline = UtilityEvaluator(
      tokenizer=tokenizer,
      generator_model=generator_model,
      device=device,
      utility_mode=args.utility_mode)
  baseline_util, baseline_em, baseline_f1 = evaluate_topk_retriever_baseline(
      val_examples, args.k, utility_evaluator_baseline, tokenizer, generator_model, device)
  print('Top-K retriever baseline: utility=%.4f  EM=%.4f  F1=%.4f' % (baseline_util, baseline_em, baseline_f1))

  singleton_gains = precompute_singleton_gains(
      train_examples,
      k=args.k,
      utility_evaluator=utility_evaluator_pl,
      singleton_gain_transform=args.singleton_gain_transform)
  pl_singleton_precompute_forward_passes = utility_evaluator_pl.reward_forward_passes

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
  train_one_method(
      method_name='policygradient',
      model=scorer_pg,
      optimizer=opt_pg,
      train_examples=train_examples,
      singleton_gains=singleton_gains,
      k=args.k,
      num_samples=args.num_samples,
      max_steps=args.max_steps,
      eval_every=args.eval_every,
      eval_examples=val_examples,
      utility_evaluator=utility_evaluator_pg,
      tokenizer=tokenizer,
      generator_model=generator_model,
      device=device,
      output_csv=pg_csv)
  train_one_method(
      method_name='plrank_surrogate',
      model=scorer_pl,
      optimizer=opt_pl,
      train_examples=train_examples,
      singleton_gains=singleton_gains,
      k=args.k,
      num_samples=args.num_samples,
      max_steps=args.max_steps,
      eval_every=args.eval_every,
      eval_examples=val_examples,
      utility_evaluator=utility_evaluator_pl,
      tokenizer=tokenizer,
      generator_model=generator_model,
      device=device,
      output_csv=pl_csv)

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
        'cumulative_reward_forward_passes': utility_evaluator_baseline.reward_forward_passes,
        'singleton_precompute_forward_passes': 0,
        'prompt_truncation_count': utility_evaluator_baseline.prompt_truncation_count,
        'target_truncation_count': utility_evaluator_baseline.target_truncation_count,
    })

  readme_path = os.path.join(args.output_dir, 'README.txt')
  with open(readme_path, 'w') as handle:
    if args.candidate_pool_path:
      handle.write('Candidate pool source: %s\n' % args.candidate_pool_path)
      handle.write('Using retrieved passages as fixed top-N candidate pool.\n')
    else:
      handle.write('Static candidate pool note: HotpotQA context passages are used as a fixed top-N candidate pool.\n')
      handle.write('This is not a full retriever pipeline over Wikipedia.\n')
    handle.write('Feature mode: %s\n' % args.feature_mode)
    if args.feature_mode == 'dense':
      handle.write('Encoder model: %s\n' % args.encoder_model)
      handle.write('Feature dim: %d (embed_dim + cosine_sim)\n' % input_dim)
    handle.write('Hidden units: %s, dropout: %.2f\n' % (hidden_units, args.dropout))
    handle.write('Empty-context baseline uses explicit placeholder: %s\n' % EMPTY_CONTEXT_PLACEHOLDER)
    handle.write('Utility mode: %s\n' % args.utility_mode)
    handle.write('Utility definition: negative mean token NLL; baseline_subtracted uses u(S)=u_raw(S)-u_raw(empty).\n')
    handle.write('Singleton gain transform: %s (default/production: shift_min; preserves per-query relative differences while enforcing nonnegativity).\n' % args.singleton_gain_transform)
    handle.write('EM/F1 are approximate normalized metrics (not the official Hotpot script).\n')
    handle.write('Data seed: %d (fixed for data subset selection). Training seed: %d.\n' % (args.data_seed, args.seed))

  print('Wrote set-utility experiment outputs to %s' % args.output_dir)
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
