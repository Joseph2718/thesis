import csv
import os

import numpy as np
import tensorflow as tf


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
