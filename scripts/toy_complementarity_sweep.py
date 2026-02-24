import argparse
import csv
import os
import pathlib
import sys
import time

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
  sys.path.insert(0, str(PROJECT_ROOT))

import algorithms.PLRank as plr
import algorithms.tensorflowloss as tfl
import utils.dataset as dataset
import utils.plackettluce as pl


def _init_model(input_dim, hidden_units, seed):
  tf.keras.utils.set_random_seed(seed)
  layers = [tf.keras.layers.Dense(h, activation='sigmoid', dtype=tf.float32) for h in hidden_units]
  layers.append(tf.keras.layers.Dense(1, activation=None, dtype=tf.float32))
  model = tf.keras.Sequential(layers)
  model.build((None, input_dim))
  return model


def _select_queries(split_obj, max_queries, seed):
  all_qids = np.arange(split_obj.num_queries())
  rng = np.random.RandomState(seed)
  rng.shuffle(all_qids)
  return all_qids[:min(max_queries, all_qids.shape[0])]


def _prepare_query_records(split_obj, qids, label_vector):
  records = []
  for qid in qids:
    labels = split_obj.query_values_from_vector(qid, label_vector).astype(np.float64, copy=False)
    feat = split_obj.query_feat(qid).astype(np.float32, copy=False)
    norms = np.linalg.norm(feat.astype(np.float64), axis=1, keepdims=True)
    norms[norms == 0.0] = 1.0
    normalized = feat.astype(np.float64) / norms
    sim = np.matmul(normalized, normalized.T)
    sim = np.clip(sim, 0.0, 1.0)
    np.fill_diagonal(sim, 0.0)
    records.append({
        'qid': int(qid),
        'feat': feat,
        'labels': labels,
        'sim': sim,
        'singleton_gains': np.maximum(labels, 0.0),  # u_lambda({d}) = rho_d; already nonnegative for 0-4 labels
    })
  return records


class ToySetUtilityCounter(object):
  def __init__(self, lam, cutoff):
    self.lam = float(lam)
    self.cutoff = int(cutoff)
    self.calls = 0

  def utility(self, qrecord, ranking):
    self.calls += 1
    topk = ranking[:min(self.cutoff, ranking.shape[0])]
    relevance = float(np.sum(qrecord['labels'][topk]))
    if topk.shape[0] <= 1:
      redundancy = 0.0
    else:
      pair_sim = qrecord['sim'][np.ix_(topk, topk)]
      redundancy = float(np.sum(np.triu(pair_sim, 1)))
    return relevance - self.lam * redundancy


def _evaluate_heldout(model, val_records, utility_counter):
  util_vals = []
  for rec in val_records:
    scores = model(rec['feat'], training=False).numpy()[:, 0]
    ranking = np.argsort(-scores)
    util_vals.append(utility_counter.utility(rec, ranking))
  return float(np.mean(util_vals))


def _train_method(method_name,
                  model,
                  optimizer,
                  train_records,
                  val_records,
                  utility_counter,
                  cutoff,
                  num_samples,
                  max_steps,
                  eval_every):
  rank_weights_dcg = 1.0 / np.log2(np.arange(cutoff) + 2.0)
  qstream = np.arange(len(train_records))
  rng = np.random.RandomState(17)
  curve_rows = []
  heldout_hist = []
  start = time.perf_counter()

  for step in range(1, max_steps + 1):
    if (step - 1) % len(qstream) == 0:
      rng.shuffle(qstream)
    rec = train_records[qstream[(step - 1) % len(qstream)]]
    n_docs = rec['labels'].shape[0]
    q_cutoff = min(cutoff, n_docs)

    with tf.GradientTape() as tape:
      scores_tf = model(rec['feat'], training=False)
      np_scores = scores_tf.numpy()[:, 0]

      if method_name == 'policygradient':
        sampled_rankings = pl.gumbel_sample_rankings(np_scores, num_samples, cutoff=q_cutoff)[0]
        sampled_rewards = np.asarray(
            [utility_counter.utility(rec, ranking) for ranking in sampled_rankings],
            dtype=np.float64)
        labels_dummy = np.zeros(n_docs, dtype=np.float64)
        loss = tf.cast(tfl.policy_gradient(
            rank_weights_dcg[:q_cutoff],
            labels_dummy,
            scores_tf,
            sampled_rankings=sampled_rankings,
            sampled_rewards=sampled_rewards), tf.float32)
        batch_utility = float(np.mean(sampled_rewards))
      elif method_name == 'plrank_surrogate':
        doc_weights = plr.PL_rank_1(
            rank_weights_dcg[:q_cutoff],
            rec['singleton_gains'],
            np_scores,
            n_samples=num_samples)
        loss = -tf.reduce_sum(scores_tf[:, 0] * tf.constant(doc_weights, dtype=tf.float32))
        ranking = np.argsort(-np_scores)
        batch_utility = utility_counter.utility(rec, ranking)
      else:
        raise ValueError('Unknown method: %s' % method_name)

    grads = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))

    heldout_utility = ''
    if step % eval_every == 0:
      heldout_utility = _evaluate_heldout(model, val_records, utility_counter)
      heldout_hist.append(heldout_utility)

    curve_rows.append({
        'method': method_name,
        'step': step,
        'batch_utility': batch_utility,
        'heldout_utility': heldout_utility,
        'reward_calls_cumulative': utility_counter.calls,
        'time_ms_cumulative': (time.perf_counter() - start) * 1000.0,
    })

  final_heldout = heldout_hist[-1] if heldout_hist else np.nan
  best_heldout = float(np.max(heldout_hist)) if heldout_hist else np.nan
  return curve_rows, {
      'method': method_name,
      'final_heldout_utility': final_heldout,
      'best_heldout_utility': best_heldout,
      'runtime_ms': curve_rows[-1]['time_ms_cumulative'],
      'reward_calls_total': utility_counter.calls,
  }


def _write_curve_csv(path, rows):
  with open(path, 'w', newline='') as handle:
    writer = csv.DictWriter(handle, fieldnames=[
        'seed', 'method', 'step', 'batch_utility', 'heldout_utility',
        'reward_calls_cumulative', 'time_ms_cumulative'])
    writer.writeheader()
    for row in rows:
      writer.writerow(row)


def _mean_std(vals):
  arr = np.asarray(vals, dtype=np.float64)
  return float(np.mean(arr)), float(np.std(arr))


def _plot_aggregate(aggregate_rows, out_dir):
  lambdas = np.asarray([row['lambda'] for row in aggregate_rows], dtype=np.float64)
  gap_final = np.asarray([row['utility_gap_final'] for row in aggregate_rows], dtype=np.float64)
  gap_best = np.asarray([row['utility_gap_best'] for row in aggregate_rows], dtype=np.float64)

  plt.figure(figsize=(7.0, 4.4))
  plt.plot(lambdas, gap_final, marker='o', label='Final utility gap (PG - PL)')
  plt.plot(lambdas, gap_best, marker='s', label='Best utility gap (PG - PL)')
  plt.axhline(0.0, color='black', linewidth=1.0, linestyle='--')
  plt.xlabel('Lambda')
  plt.ylabel('Utility gap')
  plt.title('Utility Gap vs Complementarity Strength')
  plt.legend()
  plt.tight_layout()
  plt.savefig(os.path.join(out_dir, 'utility_gap_vs_lambda.png'), dpi=160)
  plt.close()

  plt.figure(figsize=(7.0, 4.4))
  plt.plot(lambdas, [row['pg_final_mean'] for row in aggregate_rows], marker='o', label='PolicyGradient final')
  plt.plot(lambdas, [row['pl_final_mean'] for row in aggregate_rows], marker='s', label='PL-Rank surrogate final')
  plt.xlabel('Lambda')
  plt.ylabel('Final held-out utility')
  plt.title('Final Utility vs Lambda')
  plt.legend()
  plt.tight_layout()
  plt.savefig(os.path.join(out_dir, 'utility_vs_lambda.png'), dpi=160)
  plt.close()

  plt.figure(figsize=(7.0, 4.4))
  plt.plot(lambdas, [row['pg_runtime_mean'] for row in aggregate_rows], marker='o', label='PolicyGradient runtime')
  plt.plot(lambdas, [row['pl_runtime_mean'] for row in aggregate_rows], marker='s', label='PL-Rank surrogate runtime')
  plt.xlabel('Lambda')
  plt.ylabel('Runtime (ms)')
  plt.title('Runtime vs Lambda')
  plt.legend()
  plt.tight_layout()
  plt.savefig(os.path.join(out_dir, 'runtime_vs_lambda.png'), dpi=160)
  plt.close()


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument('--output_dir', default='runs/toy_complementarity_sweep')
  parser.add_argument('--dataset', default='MSLR-WEB10k')
  parser.add_argument('--dataset_info_path', default='local_dataset_info.txt')
  parser.add_argument('--fold_id', type=int, default=1)
  parser.add_argument('--lambdas', nargs='+', type=float, default=[0.0, 0.1, 0.3, 1.0])
  parser.add_argument('--seeds', nargs='+', type=int, default=[41, 42, 43])
  parser.add_argument('--max_steps', type=int, default=200)
  parser.add_argument('--eval_every', type=int, default=10)
  parser.add_argument('--num_samples', type=int, default=2)
  parser.add_argument('--cutoff', type=int, default=5)
  parser.add_argument('--learning_rate', type=float, default=0.01)
  parser.add_argument('--max_train_queries', type=int, default=300)
  parser.add_argument('--max_val_queries', type=int, default=100)
  args = parser.parse_args()

  os.makedirs(args.output_dir, exist_ok=True)

  data = dataset.get_dataset_from_json_info(args.dataset, args.dataset_info_path)
  fold_id = (args.fold_id - 1) % data.num_folds()
  data = data.get_data_folds()[fold_id]
  data.read_data()

  train_labels = data.train.label_vector.astype(np.float64, copy=False)
  val_labels = data.validation.label_vector.astype(np.float64, copy=False)

  aggregate_rows = []
  for lam in args.lambdas:
    lam_dir = os.path.join(args.output_dir, 'lambda_%s' % str(lam).replace('.', 'p'))
    os.makedirs(lam_dir, exist_ok=True)

    per_seed_summary_rows = []
    per_seed_curve_rows = []

    for seed in args.seeds:
      train_qids = _select_queries(data.train, args.max_train_queries, seed=seed)
      val_qids = _select_queries(data.validation, args.max_val_queries, seed=seed + 1000)
      train_records = _prepare_query_records(data.train, train_qids, train_labels)
      val_records = _prepare_query_records(data.validation, val_qids, val_labels)

      tf.keras.utils.set_random_seed(seed)
      init_model = _init_model(train_records[0]['feat'].shape[1], [32, 32], seed)
      shared_weights = [w.copy() for w in init_model.get_weights()]
      model_pg = _init_model(train_records[0]['feat'].shape[1], [32, 32], seed)
      model_pl = _init_model(train_records[0]['feat'].shape[1], [32, 32], seed)
      model_pg.set_weights([w.copy() for w in shared_weights])
      model_pl.set_weights([w.copy() for w in shared_weights])

      opt_pg = tf.keras.optimizers.SGD(learning_rate=args.learning_rate)
      opt_pl = tf.keras.optimizers.SGD(learning_rate=args.learning_rate)
      util_pg = ToySetUtilityCounter(lam=lam, cutoff=args.cutoff)
      util_pl = ToySetUtilityCounter(lam=lam, cutoff=args.cutoff)

      curve_pg, summ_pg = _train_method(
          'policygradient', model_pg, opt_pg, train_records, val_records, util_pg,
          args.cutoff, args.num_samples, args.max_steps, args.eval_every)
      curve_pl, summ_pl = _train_method(
          'plrank_surrogate', model_pl, opt_pl, train_records, val_records, util_pl,
          args.cutoff, args.num_samples, args.max_steps, args.eval_every)

      summ_pg['seed'] = seed
      summ_pl['seed'] = seed
      per_seed_summary_rows.extend([summ_pg, summ_pl])
      for row in curve_pg:
        row['seed'] = seed
      for row in curve_pl:
        row['seed'] = seed
      per_seed_curve_rows.extend(curve_pg + curve_pl)

    # Write per-lambda artifacts.
    with open(os.path.join(lam_dir, 'summary.csv'), 'w', newline='') as handle:
      writer = csv.DictWriter(handle, fieldnames=[
          'seed', 'method', 'final_heldout_utility', 'best_heldout_utility', 'runtime_ms', 'reward_calls_total'])
      writer.writeheader()
      for row in per_seed_summary_rows:
        writer.writerow(row)
      # mean rows for convenience
      for method in ('policygradient', 'plrank_surrogate'):
        rows = [r for r in per_seed_summary_rows if r['method'] == method]
        writer.writerow({
            'seed': 'mean',
            'method': method,
            'final_heldout_utility': np.mean([r['final_heldout_utility'] for r in rows]),
            'best_heldout_utility': np.mean([r['best_heldout_utility'] for r in rows]),
            'runtime_ms': np.mean([r['runtime_ms'] for r in rows]),
            'reward_calls_total': np.mean([r['reward_calls_total'] for r in rows]),
        })
    _write_curve_csv(os.path.join(lam_dir, 'curve.csv'), per_seed_curve_rows)

    pg_rows = [r for r in per_seed_summary_rows if r['method'] == 'policygradient']
    pl_rows = [r for r in per_seed_summary_rows if r['method'] == 'plrank_surrogate']
    pg_final_mean, pg_final_std = _mean_std([r['final_heldout_utility'] for r in pg_rows])
    pl_final_mean, pl_final_std = _mean_std([r['final_heldout_utility'] for r in pl_rows])
    pg_best_mean, pg_best_std = _mean_std([r['best_heldout_utility'] for r in pg_rows])
    pl_best_mean, pl_best_std = _mean_std([r['best_heldout_utility'] for r in pl_rows])
    pg_runtime_mean, pg_runtime_std = _mean_std([r['runtime_ms'] for r in pg_rows])
    pl_runtime_mean, pl_runtime_std = _mean_std([r['runtime_ms'] for r in pl_rows])
    pg_calls_mean, pg_calls_std = _mean_std([r['reward_calls_total'] for r in pg_rows])
    pl_calls_mean, pl_calls_std = _mean_std([r['reward_calls_total'] for r in pl_rows])

    aggregate_rows.append({
        'lambda': lam,
        'pg_final_mean': pg_final_mean,
        'pg_final_std': pg_final_std,
        'pl_final_mean': pl_final_mean,
        'pl_final_std': pl_final_std,
        'pg_best_mean': pg_best_mean,
        'pg_best_std': pg_best_std,
        'pl_best_mean': pl_best_mean,
        'pl_best_std': pl_best_std,
        'utility_gap_final': pg_final_mean - pl_final_mean,
        'utility_gap_best': pg_best_mean - pl_best_mean,
        'pg_runtime_mean': pg_runtime_mean,
        'pg_runtime_std': pg_runtime_std,
        'pl_runtime_mean': pl_runtime_mean,
        'pl_runtime_std': pl_runtime_std,
        'runtime_ratio_pg_over_pl': pg_runtime_mean / pl_runtime_mean if pl_runtime_mean != 0 else np.nan,
        'pg_reward_calls_mean': pg_calls_mean,
        'pg_reward_calls_std': pg_calls_std,
        'pl_reward_calls_mean': pl_calls_mean,
        'pl_reward_calls_std': pl_calls_std,
        'reward_call_ratio_pg_over_pl': pg_calls_mean / pl_calls_mean if pl_calls_mean != 0 else np.nan,
    })

  # Write aggregate table and plots.
  aggregate_csv = os.path.join(args.output_dir, 'aggregate.csv')
  with open(aggregate_csv, 'w', newline='') as handle:
    writer = csv.DictWriter(handle, fieldnames=[
        'lambda',
        'pg_final_mean', 'pg_final_std', 'pl_final_mean', 'pl_final_std',
        'pg_best_mean', 'pg_best_std', 'pl_best_mean', 'pl_best_std',
        'utility_gap_final', 'utility_gap_best',
        'pg_runtime_mean', 'pg_runtime_std', 'pl_runtime_mean', 'pl_runtime_std', 'runtime_ratio_pg_over_pl',
        'pg_reward_calls_mean', 'pg_reward_calls_std', 'pl_reward_calls_mean', 'pl_reward_calls_std', 'reward_call_ratio_pg_over_pl',
    ])
    writer.writeheader()
    for row in aggregate_rows:
      writer.writerow(row)

  _plot_aggregate(aggregate_rows, args.output_dir)
  print('Wrote toy complementarity sweep results to %s' % args.output_dir)
  print('Aggregate table: %s' % aggregate_csv)


if __name__ == '__main__':
  main()
