import argparse
import csv
import time

import numpy as np
import tensorflow as tf

import algorithms.PLRank as plr
import algorithms.tensorflowloss as tfl
import utils.dataset as dataset
import utils.experiment_utils as exu
import utils.nnmodel as nn
import utils.plackettluce as pl


def _select_fixed_query_ids(data_split, n_queries, seed):
  available = np.arange(data_split.num_queries())
  rng = np.random.RandomState(seed)
  rng.shuffle(available)
  return available[:min(n_queries, available.shape[0])]


def _sample_rankings_per_query(query_ids,
                               model,
                               data_split,
                               gains_vector,
                               metric_weights,
                               cutoff,
                               num_samples,
                               replicate_seed):
  sampled = {}
  for qid in query_ids:
    q_feat = data_split.query_feat(qid)
    q_labels = data_split.query_values_from_vector(qid, gains_vector)
    q_tf_scores = tf.cast(model(q_feat, training=False), tf.float32)
    q_np_scores = q_tf_scores.numpy()[:, 0]
    q_cutoff = min(cutoff, q_labels.shape[0])
    np.random.seed(replicate_seed + 7919 * int(qid))
    sampled[qid] = pl.gumbel_sample_rankings(
                    q_np_scores,
                    num_samples,
                    cutoff=q_cutoff)[0]
  return sampled


def _estimate_grad_norm(estimator,
                        query_ids,
                        sampled_rankings,
                        model,
                        data_split,
                        gains_vector,
                        metric_weights,
                        cutoff):
  start = time.perf_counter()
  with tf.GradientTape() as tape:
    total_loss = tf.constant(0.0, dtype=tf.float32)
    for qid in query_ids:
      q_feat = data_split.query_feat(qid)
      q_labels = data_split.query_values_from_vector(qid, gains_vector)
      q_tf_scores = tf.cast(model(q_feat, training=False), tf.float32)
      q_np_scores = q_tf_scores.numpy()[:, 0]
      q_cutoff = min(cutoff, q_labels.shape[0])
      q_metric_weights = metric_weights[:q_cutoff]
      q_sampled_rankings = sampled_rankings[qid]

      if estimator == 'policygradient':
        q_loss = tf.cast(tfl.policy_gradient(
                    q_metric_weights,
                    q_labels,
                    q_tf_scores,
                    sampled_rankings=q_sampled_rankings), tf.float32)
      elif estimator == 'plrank':
        doc_weights = plr.PL_rank_1(
                        q_metric_weights,
                        q_labels,
                        q_np_scores,
                        sampled_rankings=q_sampled_rankings)
        q_loss = -tf.reduce_sum(q_tf_scores[:, 0] * tf.constant(doc_weights, dtype=tf.float32))
      else:
        raise ValueError('Unknown estimator: %s' % estimator)

      total_loss += q_loss
  gradients = tape.gradient(total_loss, model.trainable_variables)
  runtime_ms = (time.perf_counter() - start) * 1000.0
  return exu.compute_global_grad_norm(gradients), runtime_ms


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument('output_csv', type=str,
                      help='Output CSV path for variance measurements.')
  parser.add_argument('--dataset', type=str, default='TinyToy',
                      help='Dataset name from dataset info json.')
  parser.add_argument('--dataset_info_path', type=str, default='local_dataset_info.txt',
                      help='Path to dataset info json.')
  parser.add_argument('--fold_id', type=int, default=1,
                      help='Fold number to select (1-indexed).')
  parser.add_argument('--cutoff', type=int, default=5,
                      help='Maximum ranking cutoff K.')
  parser.add_argument('--minibatch_queries', type=int, default=8,
                      help='Number of fixed queries in minibatch.')
  parser.add_argument('--replicates', type=int, default=200,
                      help='Number of stochastic gradient replicates.')
  parser.add_argument('--num_samples', type=int, default=1,
                      help='Number of sampled rankings per query per replicate.')
  parser.add_argument('--seed', type=int, default=0,
                      help='Base random seed.')
  args = parser.parse_args()

  data = dataset.get_dataset_from_json_info(args.dataset, args.dataset_info_path)
  fold_id = (args.fold_id - 1) % data.num_folds()
  data = data.get_data_folds()[fold_id]
  data.read_data()

  model_params = {'hidden units': [32, 32], 'learning_rate': 0.001}
  model = nn.init_model(model_params)

  max_ranking_size = min(args.cutoff, data.max_query_size())
  metric_weights = 1.0 / np.log2(np.arange(max_ranking_size) + 2)
  train_gains = 2 ** data.train.label_vector - 1

  objective_name = 'dcg'
  assert objective_name == 'dcg', 'Variance experiment must remain pure DCG.'
  plrank_variant = 'PL_rank_1'
  assert plrank_variant == 'PL_rank_1', 'Variance experiment must use PL-Rank-1.'

  query_ids = _select_fixed_query_ids(data.train, args.minibatch_queries, args.seed)
  estimators = ('policygradient', 'plrank')

  with open(args.output_csv, 'w', newline='') as handle:
    writer = csv.DictWriter(
        handle,
        fieldnames=['estimator', 'replicate_id', 'grad_norm', 'runtime_ms', 'seed'])
    writer.writeheader()

    for replicate_id in range(args.replicates):
      replicate_seed = args.seed + replicate_id
      sampled_rankings = _sample_rankings_per_query(
                          query_ids,
                          model,
                          data.train,
                          train_gains,
                          metric_weights,
                          max_ranking_size,
                          args.num_samples,
                          replicate_seed)
      baseline_rankings = {qid: ranks.copy() for qid, ranks in sampled_rankings.items()}
      for estimator in estimators:
        for qid in query_ids:
          assert np.array_equal(sampled_rankings[qid], baseline_rankings[qid]), (
              'Sampled rankings changed between estimators for qid=%s' % qid)
        grad_norm, runtime_ms = _estimate_grad_norm(
                                  estimator,
                                  query_ids,
                                  sampled_rankings,
                                  model,
                                  data.train,
                                  train_gains,
                                  metric_weights,
                                  max_ranking_size)
        writer.writerow({
            'estimator': estimator,
            'replicate_id': replicate_id,
            'grad_norm': grad_norm,
            'runtime_ms': runtime_ms,
            'seed': replicate_seed,
        })
      handle.flush()

  print('Wrote variance measurements to %s' % args.output_csv)


if __name__ == '__main__':
  main()
