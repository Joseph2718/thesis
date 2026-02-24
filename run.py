# Copyright (C) H.R. Oosterhuis 2021.
# Distributed under the MIT License (see the accompanying README.md and LICENSE files).

import argparse
import numpy as np
import time
import tensorflow as tf
import json
import os

import algorithms.PLRank as plr
import algorithms.pairwise as pw
import algorithms.lambdaloss as ll
import algorithms.tensorflowloss as tfl
import utils.plackettluce as pl
import utils.dataset as dataset
import utils.nnmodel as nn
import utils.evaluate as evl
import utils.experiment_utils as exu

parser = argparse.ArgumentParser()
parser.add_argument("output_path", type=str,
                    help="Path to output model.")
parser.add_argument("--fold_id", type=int,
                    help="Fold number to select, modulo operator is applied to stay in range.",
                    default=1)
parser.add_argument("--dataset", type=str,
                    default="Webscope_C14_Set1",
                    help="Name of dataset.")
parser.add_argument("--dataset_info_path", type=str,
                    default="local_dataset_info.txt",
                    help="Path to dataset info file.")
parser.add_argument("--cutoff", type=int,
                    help="Maximum number of items that can be displayed.",
                    default=5)
parser.add_argument("--num_samples", required=True,
                    help="Number of samples for gradient estimation ('dynamic' applies the dynamic strategy).")
parser.add_argument("--num_eval_samples", type=int,
                    help="Number of samples for metric calculation in evaluation.",
                    default=10**2)
parser.add_argument("--loss", type=str, required=True,
                    help="Name of the loss to use (PL_rank_1/PL_rank_2/lambdaloss/pairwise/policygradient/placementpolicygradient).")
parser.add_argument("--timed", action='store_true',
                    help="Turns off evaluation so method can be timed.")
parser.add_argument("--vali", action='store_true',
                    help="Results calculated on the validation set.")
parser.add_argument("--reward_type", type=str, default="existing",
                    choices=["existing", "toy_set"],
                    help="Reward definition to use for stochastic ranking losses.")
parser.add_argument("--objective", type=str, default="auto",
                    choices=["auto", "set_utility", "dcg", "dcg_surrogate_from_toy_set"],
                    help=("Training objective. set_utility is only supported for "
                          "policygradient/placementpolicygradient; PL-Rank style "
                          "losses require dcg or dcg_surrogate_from_toy_set."))
parser.add_argument("--reward_lambda", type=float, default=0.0,
                    help="Redundancy penalty weight for toy_set reward.")
parser.add_argument("--max_steps", type=int, default=None,
                    help="Optional hard cap on optimization steps for smoke tests.")

args = parser.parse_args()

cutoff = args.cutoff
num_samples = args.num_samples
num_eval_samples = args.num_eval_samples
timed_run = args.timed
validation_results = args.vali
reward_type = args.reward_type
objective = exu.resolve_objective(args.loss, reward_type, args.objective)
reward_lambda = args.reward_lambda
max_steps = args.max_steps
exu.validate_objective_for_loss(args.loss, objective)

if num_samples == 'dynamic':
  dynamic_samples = True
else:
  dynamic_samples = False
  num_samples = int(num_samples)

if timed_run:
  if args.dataset == 'Webscope_C14_Set1':
    n_epochs = 40
    max_time = 8000
  elif args.dataset == 'MSLR-WEB10k':
    n_epochs = 40
  elif args.dataset == 'MSLR-WEB30k':
    n_epochs = 40
    max_time = 9000
  elif args.dataset == 'istella':
    n_epochs = 40
    max_time = 15000
  else:
    n_epochs = 20
    max_time = 3600
else:
  if args.dataset == 'Webscope_C14_Set1':
    n_epochs = 40
  elif args.dataset == 'MSLR-WEB10k':
    n_epochs = 40
  elif args.dataset == 'MSLR-WEB30k':
    n_epochs = 40
  elif args.dataset == 'istella':
    n_epochs = 40
  else:
    n_epochs = 20

data = dataset.get_dataset_from_json_info(
                  args.dataset,
                  args.dataset_info_path,
                )
fold_id = (args.fold_id-1)%data.num_folds()
data = data.get_data_folds()[fold_id]

start = time.time()
data.read_data()
print('Time past for reading data: %d seconds' % (time.time() - start))

max_ranking_size = np.min((cutoff, data.max_query_size()))

model_params = {'hidden units': [32, 32],
                'learning_rate': 0.001,}

model = nn.init_model(model_params)
optimizer = tf.keras.optimizers.SGD(learning_rate=model_params['learning_rate'])

results = []
if args.loss in ('policygradient', 'placementpolicygradient'):
  estimator_name = 'reinforce'
else:
  estimator_name = 'plrank'
run_timestamp = time.strftime('%Y%m%d_%H%M%S')
run_csv_path = os.path.join('runs', '%s_%s_%s_%s.csv' % (run_timestamp, objective, reward_type, estimator_name))
step_logger = exu.CSVLogger(run_csv_path,
                            ['step',
                             'epoch',
                             'step_time_sec',
                             'cumulative_reward_eval_count',
                             'reward',
                             'grad_norm',
                             'estimator',
                             'loss',
                             'objective',
                             'reward_type',
                             'num_samples'])

metric_weights = 1./np.log2(np.arange(max_ranking_size) + 2)
train_labels = 2**data.train.label_vector-1
vali_labels = 2**data.validation.label_vector-1
test_labels = 2**data.test.label_vector-1
ideal_train_metrics = evl.ideal_metrics(data.train, metric_weights, train_labels)
ideal_vali_metrics = evl.ideal_metrics(data.validation, metric_weights, vali_labels)
ideal_test_metrics = evl.ideal_metrics(data.test, metric_weights, test_labels)

real_start_time = time.time()
total_train_time = 0
last_total_train_time = time.time()
method_train_time = 0

n_queries = data.train.num_queries()
if dynamic_samples:
  num_samples = 10
  float_num_samples = 10.
  add_per_step = 90./(n_queries*40.)
  max_num_samples = 100
steps = 0
next_check = 0
reward_eval_count = 0
stop_training = False
for epoch_i in range(n_epochs):
  query_permutation = np.random.permutation(n_queries)
  for qid in query_permutation:
    step_start_time = time.time()
    step_reward = np.nan
    step_reward_evals = 0
    grad_norm = 0.0

    q_labels =  data.train.query_values_from_vector(
                              qid, train_labels)
    q_feat = data.train.query_feat(qid)
    q_ideal_metric = ideal_train_metrics[qid]

    if q_ideal_metric != 0:
      q_metric_weights = metric_weights #/q_ideal_metric #uncomment for NDCG
      with tf.GradientTape() as tape:
        q_tf_scores = model(q_feat)
        q_np_scores = q_tf_scores.numpy()[:,0]
        q_cutoff = min(max_ranking_size, q_labels.shape[0])
        q_metric_weights = metric_weights[:q_cutoff]

        if objective == 'set_utility':
          q_labels_train = q_labels
        else:
          q_labels_train = exu.get_decomposable_gains(
                              objective,
                              q_metric_weights,
                              q_labels,
                              q_feat,
                              reward_lambda=reward_lambda)

        def compute_set_reward(ranking):
          if reward_type == 'toy_set':
            return exu.compute_toy_set_reward(
                          q_metric_weights,
                          q_labels,
                          q_feat,
                          ranking,
                          reward_lambda=reward_lambda,
                          topk=q_cutoff)
          return exu.compute_existing_reward(
                        q_metric_weights,
                        q_labels,
                        ranking,
                        topk=q_cutoff)

        last_method_train_time = time.time()
        if args.loss == 'policygradient':
          if objective == 'set_utility':
            sampled_rankings = pl.gumbel_sample_rankings(
                                        q_np_scores,
                                        num_samples,
                                        cutoff=q_cutoff)[0]
            sampled_rewards = np.array(
                                [compute_set_reward(ranking)
                                 for ranking in sampled_rankings],
                                dtype=np.float64)
            step_reward = float(np.mean(sampled_rewards))
            step_reward_evals += sampled_rewards.shape[0]
            loss = tfl.policy_gradient(
                                      q_metric_weights,
                                      q_labels_train,
                                      q_tf_scores,
                                      sampled_rankings=sampled_rankings,
                                      sampled_rewards=sampled_rewards
                                      )
          else:
            loss = tfl.policy_gradient(
                                      q_metric_weights,
                                      q_labels_train,
                                      q_tf_scores,
                                      n_samples=num_samples
                                      )
            step_reward_evals += num_samples
            sampled_ranking = pl.gumbel_sample_rankings(
                                        q_np_scores,
                                        1,
                                        cutoff=q_cutoff)[0][0]
            step_reward = compute_set_reward(sampled_ranking)
            step_reward_evals += 1
          method_train_time += time.time() - last_method_train_time
        elif args.loss == 'placementpolicygradient':
          if objective == 'set_utility':
            sampled_rankings = pl.gumbel_sample_rankings(
                                        q_np_scores,
                                        num_samples,
                                        cutoff=q_cutoff)[0]
            sampled_following_rewards = np.array(
                                [exu.compute_following_reward_vector(
                                                            q_metric_weights,
                                                            q_labels,
                                                            q_feat,
                                                            ranking,
                                                            reward_type=reward_type,
                                                            reward_lambda=reward_lambda,
                                                            topk=q_cutoff)
                                 for ranking in sampled_rankings],
                                dtype=np.float64)
            sampled_rewards = sampled_following_rewards[:, 0] if sampled_following_rewards.size > 0 else np.zeros(sampled_rankings.shape[0], dtype=np.float64)
            step_reward = float(np.mean(sampled_rewards))
            step_reward_evals += sampled_following_rewards.shape[0] * q_cutoff
            loss = tfl.placement_policy_gradient(
                                      q_metric_weights,
                                      q_labels_train,
                                      q_tf_scores,
                                      sampled_rankings=sampled_rankings,
                                      sampled_following_rewards=sampled_following_rewards
                                      )
          else:
            loss = tfl.placement_policy_gradient(
                                      q_metric_weights,
                                      q_labels_train,
                                      q_tf_scores,
                                      n_samples=num_samples
                                      )
            step_reward_evals += num_samples
            sampled_ranking = pl.gumbel_sample_rankings(
                                        q_np_scores,
                                        1,
                                        cutoff=q_cutoff)[0][0]
            step_reward = compute_set_reward(sampled_ranking)
            step_reward_evals += 1
          method_train_time += time.time() - last_method_train_time
        else:
          if args.loss == 'pairwise':
            doc_weights = pw.pairwise(q_labels_train,
                                      q_np_scores,
                                      )
          elif args.loss == 'lambdaloss':
            doc_weights = ll.lambdaloss(
                                      q_metric_weights,
                                      q_labels_train,
                                      q_np_scores,
                                      n_samples=num_samples
                                      )
          elif args.loss == 'PL_rank_1':
            doc_weights = plr.PL_rank_1(
                                      q_metric_weights,
                                      q_labels_train,
                                      q_np_scores,
                                      n_samples=num_samples)
          elif args.loss == 'PL_rank_2':
            doc_weights = plr.PL_rank_2(
                                      q_metric_weights,
                                      q_labels_train,
                                      q_np_scores,
                                      n_samples=num_samples)
          else:
            raise NotImplementedError('Unknown loss %s' % args.loss)
          method_train_time += time.time() - last_method_train_time
          sampled_ranking = pl.gumbel_sample_rankings(
                                      q_np_scores,
                                      1,
                                      cutoff=q_cutoff)[0][0]
          step_reward = compute_set_reward(sampled_ranking)
          step_reward_evals += 1

          loss = -tf.reduce_sum(q_tf_scores[:,0] * doc_weights)

      gradients = tape.gradient(loss, model.trainable_variables)
      grad_norm = exu.compute_global_grad_norm(gradients)
      optimizer.apply_gradients(zip(gradients, model.trainable_variables))

    steps += 1
    reward_eval_count += step_reward_evals
    step_logger.log({
        'step': steps,
        'epoch': steps/float(n_queries),
        'step_time_sec': time.time() - step_start_time,
        'cumulative_reward_eval_count': reward_eval_count,
        'reward': step_reward,
        'grad_norm': grad_norm,
        'estimator': estimator_name,
        'loss': args.loss,
        'objective': objective,
        'reward_type': reward_type,
        'num_samples': num_samples,
    })
    if dynamic_samples:
      float_num_samples = 10 + steps*add_per_step
      num_samples = min(int(np.round(float_num_samples)), max_num_samples)
    cur_epoch = steps/float(n_queries)
    if timed_run and (cur_epoch > next_check or (time.time() - real_start_time) > max_time):
      total_train_time += time.time() - last_total_train_time
      results.append({'steps': steps,
                      'epoch': next_check,
                      'train time': method_train_time,
                      'total time': total_train_time,
                      'num_samples': num_samples})
      print('%0.02f method-time: %s total-time: %s' % (cur_epoch,
            method_train_time/cur_epoch, total_train_time/cur_epoch))
      next_check += n_epochs/10000.
      last_total_train_time = time.time()
    elif cur_epoch >= next_check:
      total_train_time += time.time() - last_total_train_time
      if validation_results:
        cur_result = evl.compute_results(data.validation,
                                    model, metric_weights,
                                    vali_labels, ideal_vali_metrics,
                                    num_eval_samples)
      else:
        cur_result = evl.compute_results(data.test,
                                    model, metric_weights,
                                    test_labels, ideal_test_metrics,
                                    num_eval_samples)  
      cur_epoch = steps/float(n_queries)
      results.append({'steps': steps,
                      'epoch': next_check,
                      'train time': method_train_time,
                      'total time': total_train_time,
                      'result': cur_result,
                      'num_samples': num_samples})
      print('%0.02f expected_metric: %s deterministic_metric: %s method-time: %s total-time: %s' % (cur_epoch,
            cur_result['normalized expectation'], cur_result['normalized maximum likelihood'],
            method_train_time/cur_epoch, total_train_time/cur_epoch))
      next_check += n_epochs/100.
      last_total_train_time = time.time()

    if timed_run and (time.time() - real_start_time) > max_time:
      stop_training = True
      break
    if max_steps is not None and steps >= max_steps:
      stop_training = True
      break
  if stop_training:
    break

output = {
  'dataset': args.dataset,
  'fold number': args.fold_id,
  'run name': args.loss.replace('_', ' '),
  'loss': args.loss.replace('_', ' '),
  'model hyperparameters': model_params,
  'results': results,
  'number of samples': num_samples,
  'number of evaluation samples': num_eval_samples,
  'cutoff': cutoff,
  'objective': objective,
  'reward type': reward_type,
  'reward lambda': reward_lambda,
  'estimator': estimator_name,
  'csv log path': run_csv_path,
}
if dynamic_samples:
  output['number of samples'] = 'dynamic'

step_logger.close()
print('Wrote per-step logs to %s' % run_csv_path)
print('Writing results to %s' % args.output_path)
with open(args.output_path, 'w') as f:
  json.dump(output, f)
