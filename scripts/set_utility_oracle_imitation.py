import argparse
import csv
import hashlib
import json
import os
import pathlib
import random
import sys
import time

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
  sys.path.insert(0, str(PROJECT_ROOT))

import scripts.set_utility_experiment as sue


def precompute_singleton_utilities(examples, utility_evaluator):
  # Reuse existing singleton precompute logic from set_utility_experiment.
  # shift_min is a per-query constant shift, so order/sign differences are preserved.
  return sue.precompute_singleton_gains(
      examples=examples,
      k=1,
      utility_evaluator=utility_evaluator,
      singleton_gain_transform='shift_min')


def set_utility_for_indices(ex, indices, utility_evaluator):
  selected_passages = sue.get_prompt_passages(ex, indices)
  return utility_evaluator.compute_set_utility(
      ex['question'],
      selected_passages,
      ex['answer'],
      baseline_key=ex['example_id'])


def oracle_greedy_indices(ex, k, utility_evaluator):
  n_docs = len(ex.get('passages_for_prompt', ex['passages']))
  if n_docs == 0:
    return [], 0.0
  remaining = list(range(n_docs))
  chosen = []
  chosen_utility = 0.0
  for _ in range(min(k, n_docs)):
    best_doc = None
    best_u = None
    for doc_idx in remaining:
      trial = chosen + [doc_idx]
      trial_u = set_utility_for_indices(ex, trial, utility_evaluator)
      if best_u is None or trial_u > best_u:
        best_u = trial_u
        best_doc = doc_idx
    chosen.append(best_doc)
    remaining.remove(best_doc)
    chosen_utility = float(best_u)
  return chosen, chosen_utility


def overlap_at_k(indices_a, indices_b, k):
  if k <= 0:
    return 0.0
  a = set(indices_a[:k])
  b = set(indices_b[:k])
  return float(len(a.intersection(b))) / float(k)


def singleton_ndcg_at_k(pred_indices, singleton_utils, k):
  if k <= 0:
    return 0.0
  gains = np.asarray(singleton_utils, dtype=np.float64)
  if gains.shape[0] == 0:
    return 0.0
  # Shift to nonnegative gains for a stable ranking-based diagnostic.
  gains = gains - np.min(gains)
  pred = np.asarray(pred_indices[:k], dtype=np.int32)
  ideal = np.argsort(-gains)[:k]
  discounts = 1.0 / np.log2(np.arange(2, k + 2, dtype=np.float64))
  dcg = np.sum(gains[pred] * discounts[:pred.shape[0]])
  idcg = np.sum(gains[ideal] * discounts[:ideal.shape[0]])
  if idcg <= 0:
    return 0.0
  return float(dcg / idcg)


def evaluate_fixed_methods(eval_examples, eval_singletons, k, utility_evaluator):
  baseline_vals = []
  singleton_oracle_vals = []
  greedy_oracle_vals = []

  for ex, singleton_utils in zip(eval_examples, eval_singletons):
    n_docs = len(singleton_utils)
    qk = min(k, n_docs)

    baseline_idx = list(range(qk))
    singleton_idx = np.argsort(-singleton_utils)[:qk]
    _, greedy_u = oracle_greedy_indices(ex, qk, utility_evaluator)

    baseline_vals.append(set_utility_for_indices(ex, baseline_idx, utility_evaluator))
    singleton_oracle_vals.append(set_utility_for_indices(ex, singleton_idx, utility_evaluator))
    greedy_oracle_vals.append(float(greedy_u))

  return {
      'baseline': np.asarray(baseline_vals, dtype=np.float64),
      'oracle_singleton_topk': np.asarray(singleton_oracle_vals, dtype=np.float64),
      'oracle_greedy': np.asarray(greedy_oracle_vals, dtype=np.float64),
  }


def evaluate_supervised_model(model, eval_examples, eval_singletons, k, utility_evaluator):
  utility_vals = []
  overlap_vals = []
  ndcg_vals = []
  for ex, singleton_utils in zip(eval_examples, eval_singletons):
    scores = model(ex['features'], training=False).numpy()[:, 0]
    ranking = np.argsort(-scores)
    qk = min(k, ranking.shape[0])
    pred_topk = ranking[:qk]
    oracle_topk = np.argsort(-singleton_utils)[:qk]
    utility_vals.append(set_utility_for_indices(ex, pred_topk, utility_evaluator))
    overlap_vals.append(overlap_at_k(pred_topk, oracle_topk, qk))
    ndcg_vals.append(singleton_ndcg_at_k(pred_topk, singleton_utils, qk))
  utility_vals = np.asarray(utility_vals, dtype=np.float64)
  overlap_vals = np.asarray(overlap_vals, dtype=np.float64)
  ndcg_vals = np.asarray(ndcg_vals, dtype=np.float64)
  return {
      'utility_vals': utility_vals,
      'overlap_vals': overlap_vals,
      'ndcg_vals': ndcg_vals,
      'mean_utility': float(np.mean(utility_vals)) if utility_vals.size else 0.0,
      'std_utility': float(np.std(utility_vals)) if utility_vals.size else 0.0,
      'mean_overlap_at_k': float(np.mean(overlap_vals)) if overlap_vals.size else 0.0,
      'mean_ndcg_singleton_at_k': float(np.mean(ndcg_vals)) if ndcg_vals.size else 0.0,
  }


def _listwise_softmax_loss(scores_tf, singleton_utils, temperature):
  logits = scores_tf[:, 0]
  temp = max(float(temperature), 1e-6)
  target_logits = tf.constant(singleton_utils / temp, dtype=tf.float32)
  target_probs = tf.nn.softmax(target_logits)
  pred_log_probs = tf.nn.log_softmax(logits)
  return -tf.reduce_sum(target_probs * pred_log_probs)


def _pairwise_logistic_loss(scores_tf, singleton_utils, max_pairs, rng):
  n_docs = singleton_utils.shape[0]
  pair_triplets = []
  for i in range(n_docs):
    for j in range(i + 1, n_docs):
      if singleton_utils[i] == singleton_utils[j]:
        continue
      sign = 1.0 if singleton_utils[i] > singleton_utils[j] else -1.0
      pair_triplets.append((i, j, sign))
  if not pair_triplets:
    return tf.constant(0.0, dtype=tf.float32)
  if max_pairs > 0 and len(pair_triplets) > max_pairs:
    sampled = rng.choice(len(pair_triplets), size=max_pairs, replace=False)
    pair_triplets = [pair_triplets[idx] for idx in sampled]
  pair_losses = []
  logits = scores_tf[:, 0]
  for i, j, sign in pair_triplets:
    diff = logits[i] - logits[j]
    pair_losses.append(tf.nn.softplus(-sign * diff))
  return tf.reduce_mean(tf.stack(pair_losses))


def train_supervised_oracle_imitation(model,
                                      optimizer,
                                      train_examples,
                                      train_singletons,
                                      eval_examples,
                                      eval_singletons,
                                      k,
                                      max_steps,
                                      eval_every,
                                      minibatch_queries,
                                      supervised_loss,
                                      target_temperature,
                                      pairwise_max_pairs,
                                      utility_evaluator,
                                      precompute_forward_passes,
                                      output_csv):
  if minibatch_queries < 1:
    raise ValueError('minibatch_queries must be >= 1, got %d' % minibatch_queries)

  train_q_stream = list(range(len(train_examples)))
  rng = np.random.RandomState(17)
  rng.shuffle(train_q_stream)
  train_q_cursor = 0
  eval_points = []

  with open(output_csv, 'w', newline='') as handle:
    writer = csv.DictWriter(handle, fieldnames=[
        'method', 'step', 'batch_loss', 'heldout_utility', 'heldout_overlap_at_k',
        'heldout_ndcg_singleton_at_k', 'reward_forward_passes_step',
        'cumulative_reward_forward_passes', 'cumulative_total_generator_forward_passes',
        'cumulative_online_generator_forward_passes', 'cumulative_time_ms'])
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

      with tf.GradientTape() as tape:
        query_losses = []
        for qid in batch_qids:
          ex = train_examples[qid]
          singleton_utils = np.asarray(train_singletons[qid], dtype=np.float64)
          scores_tf = model(ex['features'], training=True)
          if supervised_loss == 'listwise':
            query_loss = _listwise_softmax_loss(scores_tf, singleton_utils, target_temperature)
          elif supervised_loss == 'pairwise':
            query_loss = _pairwise_logistic_loss(scores_tf, singleton_utils, pairwise_max_pairs, rng)
          else:
            raise ValueError('Unknown supervised_loss: %s' % supervised_loss)
          query_losses.append(query_loss)
        loss = tf.add_n(query_losses) / float(len(query_losses))

      grads = tape.gradient(loss, model.trainable_variables)
      optimizer.apply_gradients(zip(grads, model.trainable_variables))

      heldout_utility = ''
      heldout_overlap = ''
      heldout_ndcg = ''
      if step % eval_every == 0:
        eval_stats = evaluate_supervised_model(
            model=model,
            eval_examples=eval_examples,
            eval_singletons=eval_singletons,
            k=k,
            utility_evaluator=utility_evaluator)
        heldout_utility = eval_stats['mean_utility']
        heldout_overlap = eval_stats['mean_overlap_at_k']
        heldout_ndcg = eval_stats['mean_ndcg_singleton_at_k']
        eval_points.append({
            'step': step,
            'time_ms': (time.perf_counter() - start) * 1000.0,
            'utility': float(heldout_utility),
            'overlap_at_k': float(heldout_overlap),
            'ndcg_singleton_at_k': float(heldout_ndcg),
        })

      step_reward_forward_passes = utility_evaluator.reward_forward_passes - prev_reward_fw
      cumulative_total_forward_passes = utility_evaluator.reward_forward_passes
      cumulative_online_forward_passes = cumulative_total_forward_passes - int(precompute_forward_passes)
      writer.writerow({
          'method': 'supervised_oracle_imitation',
          'step': step,
          'batch_loss': float(loss.numpy()),
          'heldout_utility': heldout_utility,
          'heldout_overlap_at_k': heldout_overlap,
          'heldout_ndcg_singleton_at_k': heldout_ndcg,
          'reward_forward_passes_step': step_reward_forward_passes,
          'cumulative_reward_forward_passes': cumulative_total_forward_passes,
          'cumulative_total_generator_forward_passes': cumulative_total_forward_passes,
          'cumulative_online_generator_forward_passes': cumulative_online_forward_passes,
          'cumulative_time_ms': (time.perf_counter() - start) * 1000.0,
      })
      handle.flush()

  return eval_points


def plot_val_utility(eval_points, baseline_utility, out_path):
  if not eval_points:
    return
  plt.figure(figsize=(7.0, 4.3))
  plt.plot([p['step'] for p in eval_points], [p['utility'] for p in eval_points],
           label='Supervised oracle imitation')
  plt.axhline(y=baseline_utility, color='gray', linestyle='--', label='Retriever top-K baseline')
  plt.xlabel('Step')
  plt.ylabel('Held-out set utility')
  plt.title('Supervised Oracle Imitation vs Baseline')
  plt.legend()
  plt.tight_layout()
  plt.savefig(out_path, dpi=160)
  plt.close()


def read_rl_final_utility_from_csv(csv_path):
  if not csv_path:
    return None
  if not os.path.exists(csv_path):
    return None
  points = sue.read_eval_points(csv_path)
  if not points:
    return None
  return float(points[-1]['utility'])


def infer_bottleneck(baseline_mean, supervised_mean, supervised_overlap, margin, overlap_threshold):
  if supervised_mean <= baseline_mean + margin:
    return (
        'representation_or_policy_class_bottleneck',
        'Supervised scorer does not clear baseline margin; features/model likely limit learnable signal.')
  if supervised_overlap < overlap_threshold:
    return (
        'mixed_signal_bottleneck',
        'Utility improves but oracle overlap remains low; representation likely only partially captures oracle signal.')
  return (
      'rl_optimization_bottleneck_likely',
      'Supervised scorer beats baseline with good oracle overlap; RL variance/exploration likely the main blocker.')


def _sha256_json_payload(payload):
  blob = json.dumps(payload, sort_keys=True, separators=(',', ':')).encode('utf-8')
  return hashlib.sha256(blob).hexdigest()


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument('--output_dir', type=str, default='runs/set_utility_oracle_imitation')
  parser.add_argument('--candidate_pool_path', type=str, default=None,
                      help='Optional JSONL retrieval pool. If set, uses retrieved passages instead of Hotpot context passages.')
  parser.add_argument('--generator_model', type=str, default='google/flan-t5-small')
  parser.add_argument('--train_examples', type=int, default=200)
  parser.add_argument('--val_examples', type=int, default=80)
  parser.add_argument('--max_passages', type=int, default=20)
  parser.add_argument('--k', type=int, default=5)
  parser.add_argument('--max_steps', type=int, default=200)
  parser.add_argument('--eval_every', type=int, default=20)
  parser.add_argument('--minibatch_queries', type=int, default=16)
  parser.add_argument('--seed', type=int, default=42)
  parser.add_argument('--learning_rate', type=float, default=0.001)
  parser.add_argument('--utility_mode', type=str, default='baseline_subtracted',
                      choices=['baseline_subtracted', 'raw_neg_nll'])
  parser.add_argument('--feature_mode', type=str, default='dense',
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
                      help='Seed for data subset selection.')
  parser.add_argument('--feature_normalize', type=str, default='per_query_zscore',
                      choices=['none', 'per_query_zscore'],
                      help='Feature normalization: none or per_query_zscore (z-score within each query\'s docs).')
  parser.add_argument('--max_passage_tokens_for_prompt', type=int, default=64,
                      help='Token cap per passage for generator prompts (using generator tokenizer).')
  parser.add_argument('--expected_pool_sha256', type=str, default='',
                      help='Optional expected SHA256 for --candidate_pool_path. If provided and mismatched, abort.')
  parser.add_argument('--supervised_loss', type=str, default='listwise', choices=['listwise', 'pairwise'])
  parser.add_argument('--target_temperature', type=float, default=1.0,
                      help='Temperature for listwise target softmax over singleton utilities.')
  parser.add_argument('--pairwise_max_pairs', type=int, default=64,
                      help='Max sampled pairs per query when --supervised_loss pairwise.')
  parser.add_argument('--rl_csv_path', type=str, default='',
                      help='Optional path to existing policygradient CSV to include in diagnostic table.')
  parser.add_argument('--supervised_success_margin', type=float, default=0.01,
                      help='Minimum utility lift over baseline to call supervised imitation successful.')
  parser.add_argument('--supervised_overlap_threshold', type=float, default=0.6,
                      help='Mean overlap@K threshold used in bottleneck diagnosis.')
  args = parser.parse_args()

  os.makedirs(args.output_dir, exist_ok=True)

  expected_pool_sha256 = args.expected_pool_sha256.strip().lower()
  candidate_pool_sha256 = ''
  pool_hash_match = 'not_checked'
  if args.candidate_pool_path:
    candidate_pool_sha256 = sue.sha256_file(args.candidate_pool_path)
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
    encoder = sue.DenseEncoder(model_name=args.encoder_model, device_str=enc_device)
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
  utility_evaluator = sue.UtilityEvaluator(
      tokenizer=tokenizer,
      generator_model=generator_model,
      device=device,
      utility_mode=args.utility_mode,
      exact_truncation_check=False)

  random.seed(args.data_seed)
  np.random.seed(args.data_seed)
  torch.manual_seed(args.data_seed)
  if args.candidate_pool_path:
    train_examples, val_examples = sue.load_examples_from_candidate_pool(
        candidate_pool_path=args.candidate_pool_path,
        max_passages=args.max_passages,
        train_examples=args.train_examples,
        val_examples=args.val_examples,
        encoder=encoder)
  else:
    train_examples, val_examples = sue.load_hotpot_subset(
        max_passages=args.max_passages,
        train_examples=args.train_examples,
        val_examples=args.val_examples,
        seed=args.data_seed,
        encoder=encoder)

  train_cap_stats = sue.annotate_prompt_passages(
      train_examples,
      tokenizer=tokenizer,
      max_passage_tokens=args.max_passage_tokens_for_prompt)
  val_cap_stats = sue.annotate_prompt_passages(
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
  train_ids_sha256 = _sha256_json_payload(train_ids)
  val_ids_sha256 = _sha256_json_payload(val_ids)
  print('train_ids SHA256: %s' % train_ids_sha256)
  print('val_ids SHA256:   %s' % val_ids_sha256)

  print('Precomputing singleton utilities for train/val ...')
  train_singletons = precompute_singleton_utilities(train_examples, utility_evaluator)
  val_singletons = precompute_singleton_utilities(val_examples, utility_evaluator)
  singleton_precompute_forward_passes = utility_evaluator.reward_forward_passes
  print('Singleton precompute forward passes: %d' % singleton_precompute_forward_passes)

  print('Evaluating retriever baseline + oracle diagnostics ...')
  fixed = evaluate_fixed_methods(
      eval_examples=val_examples,
      eval_singletons=val_singletons,
      k=args.k,
      utility_evaluator=utility_evaluator)
  baseline_mean = float(np.mean(fixed['baseline']))
  baseline_std = float(np.std(fixed['baseline']))
  singleton_mean = float(np.mean(fixed['oracle_singleton_topk']))
  singleton_std = float(np.std(fixed['oracle_singleton_topk']))
  greedy_mean = float(np.mean(fixed['oracle_greedy']))
  greedy_std = float(np.std(fixed['oracle_greedy']))
  print('Retriever top-K baseline: utility=%.6f +/- %.6f' % (baseline_mean, baseline_std))
  print('Oracle singleton-topK:    utility=%.6f +/- %.6f  (delta=%.6f)' % (
      singleton_mean, singleton_std, singleton_mean - baseline_mean))
  print('Oracle greedy:            utility=%.6f +/- %.6f  (delta=%.6f)' % (
      greedy_mean, greedy_std, greedy_mean - baseline_mean))

  random.seed(args.seed)
  np.random.seed(args.seed)
  tf.keras.utils.set_random_seed(args.seed)
  torch.manual_seed(args.seed)
  model = sue.init_scorer(input_dim=input_dim, hidden_units=hidden_units, seed=args.seed, dropout=args.dropout)
  optimizer = tf.keras.optimizers.Adam(learning_rate=args.learning_rate)

  train_csv = os.path.join(args.output_dir, 'train_set_utility_supervised_oracle.csv')
  eval_points = train_supervised_oracle_imitation(
      model=model,
      optimizer=optimizer,
      train_examples=train_examples,
      train_singletons=train_singletons,
      eval_examples=val_examples,
      eval_singletons=val_singletons,
      k=args.k,
      max_steps=args.max_steps,
      eval_every=args.eval_every,
      minibatch_queries=args.minibatch_queries,
      supervised_loss=args.supervised_loss,
      target_temperature=args.target_temperature,
      pairwise_max_pairs=args.pairwise_max_pairs,
      utility_evaluator=utility_evaluator,
      precompute_forward_passes=singleton_precompute_forward_passes,
      output_csv=train_csv)

  final_supervised = evaluate_supervised_model(
      model=model,
      eval_examples=val_examples,
      eval_singletons=val_singletons,
      k=args.k,
      utility_evaluator=utility_evaluator)
  print('Final supervised scorer: utility=%.6f +/- %.6f | overlap@K=%.4f | singleton-NDCG@K=%.4f' % (
      final_supervised['mean_utility'],
      final_supervised['std_utility'],
      final_supervised['mean_overlap_at_k'],
      final_supervised['mean_ndcg_singleton_at_k']))

  bottleneck_code, bottleneck_note = infer_bottleneck(
      baseline_mean=baseline_mean,
      supervised_mean=final_supervised['mean_utility'],
      supervised_overlap=final_supervised['mean_overlap_at_k'],
      margin=args.supervised_success_margin,
      overlap_threshold=args.supervised_overlap_threshold)

  rl_final_utility = read_rl_final_utility_from_csv(args.rl_csv_path)

  plot_path = os.path.join(args.output_dir, 'convergence_supervised_oracle_vs_steps.png')
  plot_val_utility(eval_points, baseline_mean, plot_path)

  diagnostic_csv = os.path.join(args.output_dir, 'diagnostic_table.csv')
  with open(diagnostic_csv, 'w', newline='') as handle:
    writer = csv.DictWriter(handle, fieldnames=[
        'method', 'mean_utility', 'std_utility', 'delta_vs_baseline',
        'mean_overlap_at_k', 'mean_ndcg_singleton_at_k', 'notes'])
    writer.writeheader()
    writer.writerow({
        'method': 'topk_retriever_baseline',
        'mean_utility': baseline_mean,
        'std_utility': baseline_std,
        'delta_vs_baseline': 0.0,
        'mean_overlap_at_k': '',
        'mean_ndcg_singleton_at_k': '',
        'notes': 'Top-K by retriever ordering from fixed candidate pool.',
    })
    writer.writerow({
        'method': 'oracle_singleton_topk',
        'mean_utility': singleton_mean,
        'std_utility': singleton_std,
        'delta_vs_baseline': singleton_mean - baseline_mean,
        'mean_overlap_at_k': '',
        'mean_ndcg_singleton_at_k': '',
        'notes': 'Ranks docs by singleton utility u({d}) and takes top-K.',
    })
    writer.writerow({
        'method': 'oracle_greedy',
        'mean_utility': greedy_mean,
        'std_utility': greedy_std,
        'delta_vs_baseline': greedy_mean - baseline_mean,
        'mean_overlap_at_k': '',
        'mean_ndcg_singleton_at_k': '',
        'notes': 'Greedy add-by-utility oracle on true set utility u(S).',
    })
    writer.writerow({
        'method': 'supervised_oracle_imitation',
        'mean_utility': final_supervised['mean_utility'],
        'std_utility': final_supervised['std_utility'],
        'delta_vs_baseline': final_supervised['mean_utility'] - baseline_mean,
        'mean_overlap_at_k': final_supervised['mean_overlap_at_k'],
        'mean_ndcg_singleton_at_k': final_supervised['mean_ndcg_singleton_at_k'],
        'notes': 'Same scorer/policy class as RL, trained to imitate singleton oracle signal.',
    })
    if rl_final_utility is not None:
      writer.writerow({
          'method': 'policygradient_from_csv',
          'mean_utility': rl_final_utility,
          'std_utility': '',
          'delta_vs_baseline': rl_final_utility - baseline_mean,
          'mean_overlap_at_k': '',
          'mean_ndcg_singleton_at_k': '',
          'notes': 'Final held-out utility parsed from --rl_csv_path.',
      })

  report = {
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
      'k': args.k,
      'feature_mode': args.feature_mode,
      'feature_normalize': args.feature_normalize,
      'hidden_units': hidden_units,
      'dropout': args.dropout,
      'utility_mode': args.utility_mode,
      'supervised_loss': args.supervised_loss,
      'target_temperature': args.target_temperature,
      'pairwise_max_pairs': args.pairwise_max_pairs,
      'max_steps': args.max_steps,
      'eval_every': args.eval_every,
      'minibatch_queries': args.minibatch_queries,
      'train_ids_path': train_ids_path,
      'val_ids_path': val_ids_path,
      'train_ids_sha256': train_ids_sha256,
      'val_ids_sha256': val_ids_sha256,
      'prompt_passage_cap_stats': {
          'train': train_cap_stats,
          'val': val_cap_stats,
      },
      'diagnostic_summary': {
          'topk_retriever_baseline_mean_utility': baseline_mean,
          'oracle_singleton_topk_mean_utility': singleton_mean,
          'oracle_greedy_mean_utility': greedy_mean,
          'supervised_oracle_imitation_mean_utility': final_supervised['mean_utility'],
          'supervised_oracle_imitation_mean_overlap_at_k': final_supervised['mean_overlap_at_k'],
          'supervised_oracle_imitation_mean_ndcg_singleton_at_k': final_supervised['mean_ndcg_singleton_at_k'],
          'policygradient_from_csv_mean_utility': rl_final_utility,
      },
      'diagnosis': {
          'bottleneck_code': bottleneck_code,
          'bottleneck_note': bottleneck_note,
          'success_margin': args.supervised_success_margin,
          'overlap_threshold': args.supervised_overlap_threshold,
      },
      'compute_accounting_definition': {
          'total_generator_forward_passes': (
              'All generator forward passes used for utility accounting '
              '(singleton precompute + oracle diagnostics + held-out eval).'),
          'online_generator_forward_passes': (
              'total_generator_forward_passes - singleton_precompute_forward_passes.'),
      },
      'run_results': {
          'cumulative_total_generator_forward_passes': utility_evaluator.reward_forward_passes,
          'cumulative_online_generator_forward_passes': (
              utility_evaluator.reward_forward_passes - singleton_precompute_forward_passes),
          'singleton_precompute_forward_passes': singleton_precompute_forward_passes,
          'prompt_truncation_count': utility_evaluator.prompt_truncation_count,
          'target_truncation_count': utility_evaluator.target_truncation_count,
      },
      'artifacts': {
          'train_csv': train_csv,
          'diagnostic_table_csv': diagnostic_csv,
          'convergence_plot': plot_path,
      },
  }
  report_path = os.path.join(args.output_dir, 'diagnostic_report.json')
  with open(report_path, 'w') as f:
    json.dump(report, f, indent=2)

  readme_path = os.path.join(args.output_dir, 'README.txt')
  with open(readme_path, 'w') as handle:
    handle.write('Generator model: %s\n' % args.generator_model)
    if args.candidate_pool_path:
      handle.write('Candidate pool source: %s\n' % args.candidate_pool_path)
      handle.write('Candidate pool SHA256: %s\n' % candidate_pool_sha256)
    handle.write('Pool hash check status: %s\n' % pool_hash_match)
    handle.write('Utility mode: %s\n' % args.utility_mode)
    handle.write('Feature mode: %s, feature_normalize: %s\n' % (args.feature_mode, args.feature_normalize))
    handle.write('Hidden units: %s, dropout: %.2f\n' % (hidden_units, args.dropout))
    handle.write('Supervised loss: %s\n' % args.supervised_loss)
    if args.supervised_loss == 'listwise':
      handle.write('Listwise target temperature: %.4f\n' % args.target_temperature)
    else:
      handle.write('Pairwise max pairs/query: %d\n' % args.pairwise_max_pairs)
    handle.write('Minibatch queries: %d, max_steps: %d, eval_every: %d\n' % (
        args.minibatch_queries, args.max_steps, args.eval_every))
    handle.write('Passage prompt token cap (generator tokenizer): %d\n' % args.max_passage_tokens_for_prompt)
    handle.write('train_ids.json SHA256: %s\n' % train_ids_sha256)
    handle.write('val_ids.json SHA256: %s\n' % val_ids_sha256)
    handle.write('\nDiagnostic summary (held-out utility):\n')
    handle.write('- topk_retriever_baseline: %.6f\n' % baseline_mean)
    handle.write('- oracle_singleton_topk: %.6f (delta %.6f)\n' % (
        singleton_mean, singleton_mean - baseline_mean))
    handle.write('- oracle_greedy: %.6f (delta %.6f)\n' % (
        greedy_mean, greedy_mean - baseline_mean))
    handle.write('- supervised_oracle_imitation: %.6f (delta %.6f)\n' % (
        final_supervised['mean_utility'], final_supervised['mean_utility'] - baseline_mean))
    if rl_final_utility is not None:
      handle.write('- policygradient_from_csv: %.6f (delta %.6f)\n' % (
          rl_final_utility, rl_final_utility - baseline_mean))
    handle.write('\nRepresentation-vs-RL diagnosis:\n')
    handle.write('- bottleneck_code: %s\n' % bottleneck_code)
    handle.write('- note: %s\n' % bottleneck_note)
    handle.write('\nCompute accounting:\n')
    handle.write('- total_generator_forward_passes includes singleton precompute + oracle diagnostics + held-out eval.\n')
    handle.write('- online_generator_forward_passes = total_generator_forward_passes - singleton_precompute_forward_passes.\n')

  print('Wrote supervised oracle imitation outputs to %s' % args.output_dir)
  print('Diagnostic report: %s' % report_path)
  print('Diagnostic table:  %s' % diagnostic_csv)
  print('Convergence plot:  %s' % plot_path)
  print('Diagnosis: %s | %s' % (bottleneck_code, bottleneck_note))


if __name__ == '__main__':
  main()
