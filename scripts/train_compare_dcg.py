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
import utils.evaluate as evl


def _init_model_float32(hidden_units):
  layers = [tf.keras.layers.Dense(x, activation='sigmoid', dtype=tf.float32)
            for x in hidden_units]
  layers.append(tf.keras.layers.Dense(1, activation=None, dtype=tf.float32))
  return tf.keras.Sequential(layers)


def _make_query_stream(num_queries, required, seed):
  rng = np.random.RandomState(seed)
  stream = []
  while len(stream) < required:
    perm = rng.permutation(num_queries).tolist()
    stream.extend(perm)
  return np.asarray(stream[:required], dtype=np.int32)


def _deterministic_query_dcg(q_scores, q_labels, q_metric_weights):
  order = np.argsort(-q_scores)
  cutoff = min(order.shape[0], q_metric_weights.shape[0])
  ranking = order[:cutoff]
  return float(np.sum(q_metric_weights[:cutoff] * q_labels[ranking]))


def _collect_eval_points(csv_path):
  points = []
  with open(csv_path, newline='') as handle:
    reader = csv.DictReader(handle)
    for row in reader:
      heldout = row['heldout_dcg']
      if heldout != '':
        points.append((int(row['step']),
                       float(row['cumulative_time_ms']),
                       float(heldout)))
  return points


def _steps_time_to_threshold(points, threshold):
  for step, cum_ms, heldout in points:
    if heldout >= threshold:
      return step, cum_ms
  return None, None


def _run_training_job(estimator,
                      output_csv_path,
                      data_fold,
                      fold_id,
                      train_labels,
                      heldout_labels,
                      metric_weights,
                      ideal_heldout_metrics,
                      max_steps,
                      eval_every,
                      num_samples,
                      cutoff,
                      batch_queries,
                      learning_rate,
                      hidden_units,
                      seed,
                      initial_weights,
                      num_features):
  assert estimator in ('policygradient', 'plrank')
  plrank_variant = 'PL_rank_1'
  objective_name = 'dcg'
  assert objective_name == 'dcg', 'This script is DCG-only.'
  assert plrank_variant == 'PL_rank_1', 'This script must use PL-Rank-1.'

  tf.keras.utils.set_random_seed(seed)
  model = _init_model_float32(hidden_units)
  model.build((None, num_features))
  model.set_weights([w.copy() for w in initial_weights])
  for cur_w, init_w in zip(model.get_weights(), initial_weights):
    assert np.array_equal(cur_w, init_w), 'Model weights do not match shared initialization.'
  optimizer = tf.keras.optimizers.SGD(learning_rate=learning_rate)

  n_train_queries = data_fold.train.num_queries()
  qid_stream = _make_query_stream(
      n_train_queries,
      required=max_steps * batch_queries,
      seed=seed + fold_id * 1000)

  with open(output_csv_path, 'w', newline='') as handle:
    writer = csv.DictWriter(
        handle,
        fieldnames=['estimator', 'step', 'batch_dcg', 'heldout_dcg', 'cumulative_time_ms'])
    writer.writeheader()

    train_start = time.perf_counter()
    for step in range(1, max_steps + 1):
      qids = qid_stream[(step - 1) * batch_queries: step * batch_queries]
      batch_dcg_vals = []
      with tf.GradientTape() as tape:
        total_loss = tf.constant(0.0, dtype=tf.float32)
        for qid in qids:
          q_labels = data_fold.train.query_values_from_vector(qid, train_labels).astype(np.float32, copy=False)
          q_feat = data_fold.train.query_feat(qid).astype(np.float32, copy=False)
          q_tf_scores = model(q_feat, training=False)
          q_np_scores = q_tf_scores.numpy()[:, 0]
          q_cutoff = min(cutoff, q_labels.shape[0])
          q_metric_weights = metric_weights[:q_cutoff]
          batch_dcg_vals.append(_deterministic_query_dcg(q_np_scores, q_labels, q_metric_weights))

          if estimator == 'policygradient':
            q_loss = tf.cast(tfl.policy_gradient(
                q_metric_weights,
                q_labels,
                q_tf_scores,
                n_samples=num_samples), tf.float32)
          else:
            doc_weights = plr.PL_rank_1(
                q_metric_weights,
                q_labels,
                q_np_scores,
                n_samples=num_samples)
            q_loss = -tf.reduce_sum(q_tf_scores[:, 0] * tf.constant(doc_weights, dtype=tf.float32))

          total_loss += q_loss

      gradients = tape.gradient(total_loss, model.trainable_variables)
      optimizer.apply_gradients(zip(gradients, model.trainable_variables))

      heldout_dcg = ''
      if step % eval_every == 0:
        heldout_dcg = evl.evaluate_max_likelihood(
            data_fold.validation,
            model,
            metric_weights,
            heldout_labels,
            ideal_heldout_metrics)[0]

      writer.writerow({
          'estimator': estimator,
          'step': step,
          'batch_dcg': float(np.mean(batch_dcg_vals)),
          'heldout_dcg': heldout_dcg,
          'cumulative_time_ms': (time.perf_counter() - train_start) * 1000.0,
      })
      handle.flush()


def _plot_curves(policy_csv, plrank_csv, out_steps_path, out_time_path):
  pg = _collect_eval_points(policy_csv)
  pl = _collect_eval_points(plrank_csv)

  plt.figure(figsize=(7.5, 4.5))
  plt.plot([x[0] for x in pg], [x[2] for x in pg], label='policygradient')
  plt.plot([x[0] for x in pl], [x[2] for x in pl], label='plrank')
  plt.xlabel('Training step')
  plt.ylabel('Held-out DCG')
  plt.title('Held-out DCG vs Steps')
  plt.legend()
  plt.tight_layout()
  plt.savefig(out_steps_path, dpi=160)
  plt.close()

  plt.figure(figsize=(7.5, 4.5))
  plt.plot([x[1] for x in pg], [x[2] for x in pg], label='policygradient')
  plt.plot([x[1] for x in pl], [x[2] for x in pl], label='plrank')
  plt.xlabel('Cumulative wall-clock time (ms)')
  plt.ylabel('Held-out DCG')
  plt.title('Held-out DCG vs Time')
  plt.legend()
  plt.tight_layout()
  plt.savefig(out_time_path, dpi=160)
  plt.close()

  return pg, pl


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument('--dataset', type=str, default='MSLR-WEB10k')
  parser.add_argument('--dataset_info_path', type=str, default='local_dataset_info.txt')
  parser.add_argument('--fold_id', type=int, default=1)
  parser.add_argument('--cutoff', type=int, default=5)
  parser.add_argument('--num_samples', type=int, default=1)
  parser.add_argument('--batch_queries', type=int, default=1)
  parser.add_argument('--max_steps', type=int, default=200)
  parser.add_argument('--eval_every', type=int, default=10)
  parser.add_argument('--learning_rate', type=float, default=0.01)
  parser.add_argument('--seed', type=int, default=42)
  parser.add_argument('--hidden_units', type=int, nargs='+', default=[32, 32])
  parser.add_argument('--policy_csv', type=str, default='runs/train_dcg_policygradient.csv')
  parser.add_argument('--plrank_csv', type=str, default='runs/train_dcg_plrank.csv')
  parser.add_argument('--plot_steps', type=str, default='runs/dcg_vs_steps.png')
  parser.add_argument('--plot_time', type=str, default='runs/dcg_vs_time.png')
  parser.add_argument('--summary_path', type=str, default='runs/train_dcg_threshold_summary.txt')
  args = parser.parse_args()

  for path in (args.policy_csv, args.plrank_csv, args.plot_steps, args.plot_time, args.summary_path):
    out_dir = os.path.dirname(path)
    if out_dir:
      os.makedirs(out_dir, exist_ok=True)

  data = dataset.get_dataset_from_json_info(args.dataset, args.dataset_info_path)
  fold_id = (args.fold_id - 1) % data.num_folds()
  data_fold = data.get_data_folds()[fold_id]
  data_fold.read_data()

  train_labels = (2 ** data_fold.train.label_vector - 1).astype(np.float32)
  heldout_labels = (2 ** data_fold.validation.label_vector - 1).astype(np.float32)
  max_ranking_size = min(args.cutoff, data_fold.max_query_size())
  metric_weights = (1.0 / np.log2(np.arange(max_ranking_size) + 2)).astype(np.float32)
  ideal_heldout_metrics = evl.ideal_metrics(data_fold.validation, metric_weights, heldout_labels)

  print('Optimizer: SGD')
  print('Learning rate: %.2f' % args.learning_rate)

  tf.keras.utils.set_random_seed(args.seed)
  shared_init_model = _init_model_float32(args.hidden_units)
  shared_init_model.build((None, data_fold.num_features))
  shared_init_weights = [w.copy() for w in shared_init_model.get_weights()]
  sanity_policy_model = _init_model_float32(args.hidden_units)
  sanity_plrank_model = _init_model_float32(args.hidden_units)
  sanity_policy_model.build((None, data_fold.num_features))
  sanity_plrank_model.build((None, data_fold.num_features))
  sanity_policy_model.set_weights([w.copy() for w in shared_init_weights])
  sanity_plrank_model.set_weights([w.copy() for w in shared_init_weights])
  for w_a, w_b in zip(sanity_policy_model.get_weights(), sanity_plrank_model.get_weights()):
    assert np.array_equal(w_a, w_b), 'Initial weights differ between estimators.'

  _run_training_job(
      estimator='policygradient',
      output_csv_path=args.policy_csv,
      data_fold=data_fold,
      fold_id=fold_id,
      train_labels=train_labels,
      heldout_labels=heldout_labels,
      metric_weights=metric_weights,
      ideal_heldout_metrics=ideal_heldout_metrics,
      max_steps=args.max_steps,
      eval_every=args.eval_every,
      num_samples=args.num_samples,
      cutoff=max_ranking_size,
      batch_queries=args.batch_queries,
      learning_rate=args.learning_rate,
      hidden_units=args.hidden_units,
      seed=args.seed,
      initial_weights=shared_init_weights,
      num_features=data_fold.num_features)

  _run_training_job(
      estimator='plrank',
      output_csv_path=args.plrank_csv,
      data_fold=data_fold,
      fold_id=fold_id,
      train_labels=train_labels,
      heldout_labels=heldout_labels,
      metric_weights=metric_weights,
      ideal_heldout_metrics=ideal_heldout_metrics,
      max_steps=args.max_steps,
      eval_every=args.eval_every,
      num_samples=args.num_samples,
      cutoff=max_ranking_size,
      batch_queries=args.batch_queries,
      learning_rate=args.learning_rate,
      hidden_units=args.hidden_units,
      seed=args.seed,
      initial_weights=shared_init_weights,
      num_features=data_fold.num_features)

  pg_points, pl_points = _plot_curves(
      args.policy_csv,
      args.plrank_csv,
      args.plot_steps,
      args.plot_time)

  all_heldout = [p[2] for p in pg_points] + [p[2] for p in pl_points]
  threshold = 0.95 * float(np.max(all_heldout))
  pg_step, pg_time = _steps_time_to_threshold(pg_points, threshold)
  pl_step, pl_time = _steps_time_to_threshold(pl_points, threshold)

  summary_lines = [
      'Threshold (95%% of best held-out DCG): %.6f' % threshold,
      'PL-Rank reached threshold in %s steps / %s ms' % (
          'N/A' if pl_step is None else str(pl_step),
          'N/A' if pl_time is None else '%.3f' % pl_time),
      'PolicyGradient reached threshold in %s steps / %s ms' % (
          'N/A' if pg_step is None else str(pg_step),
          'N/A' if pg_time is None else '%.3f' % pg_time),
  ]
  with open(args.summary_path, 'w') as handle:
    handle.write('\n'.join(summary_lines) + '\n')

  for line in summary_lines:
    print(line)
  print('Wrote %s, %s, %s, %s' % (
      args.policy_csv, args.plrank_csv, args.plot_steps, args.plot_time))


if __name__ == '__main__':
  main()
