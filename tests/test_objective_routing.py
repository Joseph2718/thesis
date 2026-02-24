import unittest

import numpy as np
import tensorflow as tf

import algorithms.PLRank as plr
import algorithms.tensorflowloss as tfl
import utils.experiment_utils as exu


class ObjectiveRoutingTests(unittest.TestCase):

  def setUp(self):
    self.rank_weights = np.array([1.0, 1.0 / np.log2(3.0)], dtype=np.float64)
    self.labels = np.array([3.0, 1.0, 0.0], dtype=np.float64)
    self.query_features = np.array([[1.0, 0.0],
                                    [0.8, 0.2],
                                    [0.0, 1.0]], dtype=np.float64)
    self.sampled_rankings = np.array([[0, 1],
                                      [1, 2]], dtype=np.int32)
    self.scores_tf = tf.constant([[1.2],
                                  [0.3],
                                  [-0.5]], dtype=tf.float64)
    self.scores_np = self.scores_tf.numpy()[:, 0]

  def test_plrank_refuses_raw_set_utility(self):
    with self.assertRaisesRegex(ValueError, 'requires decomposable per-document gains'):
      exu.validate_objective_for_loss('PL_rank_1', 'set_utility')

  def test_plrank_runs_with_dcg_surrogate_from_toy_set(self):
    gains = exu.get_decomposable_gains(
              'dcg_surrogate_from_toy_set',
              self.rank_weights,
              self.labels,
              self.query_features,
              reward_lambda=0.25)
    weights = plr.PL_rank_1(
                self.rank_weights,
                gains,
                self.scores_np,
                n_samples=4)
    self.assertEqual(weights.shape, self.labels.shape)
    self.assertTrue(np.all(np.isfinite(weights)))

  def test_methods_1_and_2_run_with_toy_set_utility(self):
    sampled_rewards = np.array(
        [exu.compute_toy_set_reward(self.rank_weights,
                                    self.labels,
                                    self.query_features,
                                    ranking,
                                    reward_lambda=0.25,
                                    topk=self.rank_weights.shape[0])
         for ranking in self.sampled_rankings],
        dtype=np.float64)
    reinforce_loss = tfl.policy_gradient(
                        self.rank_weights,
                        self.labels,
                        self.scores_tf,
                        sampled_rankings=self.sampled_rankings,
                        sampled_rewards=sampled_rewards)

    sampled_following_rewards = np.array(
        [exu.compute_following_reward_vector(
              self.rank_weights,
              self.labels,
              self.query_features,
              ranking,
              reward_type='toy_set',
              reward_lambda=0.25,
              topk=self.rank_weights.shape[0])
         for ranking in self.sampled_rankings],
        dtype=np.float64)
    placement_loss = tfl.placement_policy_gradient(
                        self.rank_weights,
                        self.labels,
                        self.scores_tf,
                        sampled_rankings=self.sampled_rankings,
                        sampled_following_rewards=sampled_following_rewards)

    self.assertTrue(np.isfinite(float(reinforce_loss.numpy())))
    self.assertTrue(np.isfinite(float(placement_loss.numpy())))


if __name__ == '__main__':
  unittest.main()
