import csv
import os

import numpy as np
import tensorflow as tf

SET_UTILITY_LOSSES = ('policygradient', 'placementpolicygradient')
DECOMPOSABLE_LOSSES = ('pairwise', 'lambdaloss', 'PL_rank_1', 'PL_rank_2')


def resolve_objective(loss_name, reward_type, objective):
  if objective != 'auto':
    return objective
  if loss_name in DECOMPOSABLE_LOSSES:
    return 'dcg'
  if reward_type == 'toy_set':
    return 'set_utility'
  return 'dcg'


def validate_objective_for_loss(loss_name, objective):
  if objective == 'set_utility' and loss_name in DECOMPOSABLE_LOSSES:
    raise ValueError(
        ('Loss "%s" with objective "set_utility" is unsupported because this '
         'estimator requires decomposable per-document gains. '
         'Use --objective dcg or --objective dcg_surrogate_from_toy_set.')
        % loss_name)


def compute_existing_reward(rank_weights, labels, ranking, topk=None):
  max_cutoff = min(rank_weights.shape[0], ranking.shape[0])
  if topk is not None:
    max_cutoff = min(max_cutoff, topk)
  if max_cutoff <= 0:
    return 0.0
  prefix = ranking[:max_cutoff]
  return float(np.sum(rank_weights[:max_cutoff] * labels[prefix]))


def _mean_pairwise_cosine(feature_matrix, doc_indices):
  n_docs = doc_indices.shape[0]
  if n_docs < 2:
    return 0.0
  vectors = feature_matrix[doc_indices]
  norms = np.linalg.norm(vectors, axis=1, keepdims=True)
  norms[norms == 0.] = 1.
  normalized = vectors / norms
  cosine = np.matmul(normalized, normalized.T)
  triu = np.triu_indices(n_docs, k=1)
  return float(np.mean(cosine[triu]))


def compute_toy_set_reward(rank_weights, labels, query_features, ranking, reward_lambda=0.0, topk=None):
  max_cutoff = min(rank_weights.shape[0], ranking.shape[0])
  if topk is not None:
    max_cutoff = min(max_cutoff, topk)
  if max_cutoff <= 0:
    return 0.0
  prefix = ranking[:max_cutoff]
  relevance = float(np.sum(rank_weights[:max_cutoff] * labels[prefix]))
  redundancy = _mean_pairwise_cosine(query_features, prefix)
  return relevance - reward_lambda * redundancy


def compute_toy_singleton_gains(rank_weights, labels, query_features, reward_lambda=0.0):
  n_docs = labels.shape[0]
  gains = np.zeros(n_docs, dtype=np.float64)
  for doc_i in range(n_docs):
    singleton_ranking = np.array([doc_i], dtype=np.int32)
    gains[doc_i] = compute_toy_set_reward(
                      rank_weights,
                      labels,
                      query_features,
                      singleton_ranking,
                      reward_lambda=reward_lambda,
                      topk=1)
  return gains


def compute_toy_doc_relevance(labels, query_features, reward_lambda=0.0):
  if reward_lambda == 0.0:
    return labels.astype(np.float64, copy=True)
  n_docs = labels.shape[0]
  if n_docs <= 1:
    return labels.astype(np.float64, copy=True)

  norms = np.linalg.norm(query_features, axis=1, keepdims=True)
  norms[norms == 0.] = 1.
  normalized = query_features / norms
  cosine = np.matmul(normalized, normalized.T)
  np.fill_diagonal(cosine, 0.0)
  redundancy_per_doc = np.sum(cosine, axis=1) / float(n_docs - 1)
  return labels.astype(np.float64) - reward_lambda * redundancy_per_doc


def compute_following_reward_vector(rank_weights, labels, query_features, ranking, reward_type='existing', reward_lambda=0.0, topk=None):
  max_cutoff = min(rank_weights.shape[0], ranking.shape[0])
  if topk is not None:
    max_cutoff = min(max_cutoff, topk)
  if max_cutoff <= 0:
    return np.zeros(0, dtype=np.float64)

  if reward_type == 'existing':
    reward_fn = compute_existing_reward
  elif reward_type == 'toy_set':
    reward_fn = compute_toy_set_reward
  else:
    raise ValueError('Unknown reward type: %s' % reward_type)

  prefix = ranking[:max_cutoff]
  if reward_type == 'toy_set':
    full_reward = reward_fn(rank_weights, labels, query_features, prefix, reward_lambda=reward_lambda, topk=max_cutoff)
  else:
    full_reward = reward_fn(rank_weights, labels, prefix, topk=max_cutoff)

  following = np.zeros(max_cutoff, dtype=np.float64)
  for k in range(max_cutoff):
    if k == 0:
      prev_reward = 0.0
    else:
      prefix_k = prefix[:k]
      if reward_type == 'toy_set':
        prev_reward = reward_fn(rank_weights, labels, query_features, prefix_k, reward_lambda=reward_lambda, topk=k)
      else:
        prev_reward = reward_fn(rank_weights, labels, prefix_k, topk=k)
    following[k] = full_reward - prev_reward
  return following


def get_decomposable_gains(objective, rank_weights, labels, query_features, reward_lambda=0.0):
  if objective == 'dcg':
    return labels.astype(np.float64, copy=True)
  if objective == 'dcg_surrogate_from_toy_set':
    return compute_toy_singleton_gains(rank_weights, labels, query_features, reward_lambda=reward_lambda)
  raise ValueError('Objective %s does not define decomposable per-document gains.' % objective)


def compute_global_grad_norm(gradients):
  sq_norms = []
  for grad in gradients:
    if grad is None:
      continue
    sq_norms.append(tf.reduce_sum(tf.square(grad)))
  if not sq_norms:
    return 0.0
  return float(tf.sqrt(tf.add_n(sq_norms)).numpy())


class CSVLogger(object):
  def __init__(self, path, fieldnames):
    self.path = path
    os.makedirs(os.path.dirname(path), exist_ok=True)
    self._handle = open(path, 'w', newline='')
    self._writer = csv.DictWriter(self._handle, fieldnames=fieldnames)
    self._writer.writeheader()

  def log(self, row_dict):
    self._writer.writerow(row_dict)
    self._handle.flush()

  def close(self):
    self._handle.close()
