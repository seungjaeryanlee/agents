# coding=utf-8
# Copyright 2018 The TF-Agents Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""A DQN Agent with RND intrionsic rewards.

Implements the DQN algorithm from

"Human level control through deep reinforcement learning"
  Mnih et al., 2015
  https://deepmind.com/research/dqn/

Implements the RND algorithm from

"Exploration by Random Network Distillation"
  Burda et al., 2018
  https://arxiv.org/abs/1810.12894
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections

import gin
import tensorflow as tf

from tf_agents.agents import tf_agent
from tf_agents.agents.dqn import dqn_agent
from tf_agents.agents.dqn.dqn_agent import element_wise_huber_loss, element_wise_squared_loss
from tf_agents.policies import boltzmann_policy
from tf_agents.policies import epsilon_greedy_policy
from tf_agents.policies import greedy_policy
from tf_agents.policies import q_policy
from tf_agents.trajectories import trajectory
from tf_agents.utils import common
from tf_agents.utils import eager_utils
from tf_agents.utils import nest_utils
from tf_agents.utils import value_ops

# TODO(seungjaeryanlee): Definition of those element wise losses should not belong to
# this file. Move them to utils/common or utils/losses. Look at dqn_agent for
# similar examples: element_wise_huber_loss and element_wise_squared_loss.
def mean_squared_loss(x, y):
  return tf.reduce_mean(
    input_tensor=tf.compat.v1.losses.mean_squared_error(
      x, y, reduction=tf.compat.v1.losses.Reduction.NONE),
    axis=1)


class RndDqnLossInfo(collections.namedtuple('RndDqnLossInfo',
                                              ('td_loss',
                                               'td_error',
                                               'rnd_losses',
                                               'avg_rnd_loss'))):
  pass


def compute_td_targets(next_q_values, rewards, discounts):
  return tf.stop_gradient(rewards + discounts * next_q_values)


@gin.configurable
class RndDqnAgent(dqn_agent.DqnAgent):
  """A RNDDQN Agent.

  Implements the DQN algorithm from

  "Human level control through deep reinforcement learning"
    Mnih et al., 2015
    https://deepmind.com/research/dqn/

  Implements the RND algorithm from

  "Exploration by Random Network Distillation"
    Burda et al., 2018
    https://arxiv.org/abs/1810.12894

  This agent also implements n-step updates. See "Rainbow: Combining
  Improvements in Deep Reinforcement Learning" by Hessel et al., 2017, for a
  discussion on its benefits: https://arxiv.org/abs/1710.02298
  """

  def __init__(
      self,
      time_step_spec,
      action_spec,
      q_network,
      optimizer,
      rnd_network,
      rnd_optimizer,
      rnd_loss_fn=None,
      epsilon_greedy=0.1,
      n_step_update=1,
      boltzmann_temperature=None,
      emit_log_probability=False,
      # Params for target network updates
      target_update_tau=1.0,
      target_update_period=1,
      # Params for training.
      td_errors_loss_fn=None,
      gamma=1.0,
      reward_scale_factor=1.0,
      gradient_clipping=None,
      # Params for debugging
      debug_summaries=False,
      summarize_grads_and_vars=False,
      train_step_counter=None,
      name=None):
    """Creates a RNDDQN Agent.

    Args:
      time_step_spec: A `TimeStep` spec of the expected time_steps.
      action_spec: A nest of BoundedTensorSpec representing the actions.
      q_network: A tf_agents.network.Network to be used by the agent. The
        network will be called with call(observation, step_type).
      optimizer: The optimizer to use for training.
      rnd_network: A tf_agents.network.Network to be used to calculate RND intrinsic reward.
      rnd_optimizer: The optimizer to use for training RND network.
      rnd_loss_fn: A function for computing the RND errors loss. If None, a
        default value of mean_squared_loss is used.
      epsilon_greedy: probability of choosing a random action in the default
        epsilon-greedy collect policy (used only if a wrapper is not provided to
        the collect_policy method).
      n_step_update: The number of steps to consider when computing TD error and
        TD loss. Defaults to single-step updates. Note that this requires the
        user to call train on Trajectory objects with a time dimension of
        `n_step_update + 1`. However, note that we do not yet support
        `n_step_update > 1` in the case of RNNs (i.e., non-empty
        `q_network.state_spec`).
      boltzmann_temperature: Temperature value to use for Boltzmann sampling of
        the actions during data collection. The closer to 0.0, the higher the
        probability of choosing the best action.
      emit_log_probability: Whether policies emit log probabilities or not.
      target_update_tau: Factor for soft update of the target networks.
      target_update_period: Period for soft update of the target networks.
      td_errors_loss_fn: A function for computing the TD errors loss. If None, a
        default value of element_wise_huber_loss is used. This function takes as
        input the target and the estimated Q values and returns the loss for
        each element of the batch.
      gamma: A discount factor for future rewards.
      reward_scale_factor: Multiplicative scale for the reward.
      gradient_clipping: Norm length to clip gradients.
      debug_summaries: A bool to gather debug summaries.
      summarize_grads_and_vars: If True, gradient and network variable summaries
        will be written during training.
      train_step_counter: An optional counter to increment every time the train
        op is run.  Defaults to the global_step.
      name: The name of this agent. All variables in this module will fall
        under that name. Defaults to the class name.

    Raises:
      ValueError: If the action spec contains more than one action or action
        spec minimum is not equal to 0.
      NotImplementedError: If `q_network` has non-empty `state_spec` (i.e., an
        RNN is provided) and `n_step_update > 1`.
    """
    super(RndDqnAgent, self).__init__(
        time_step_spec,
        action_spec,
        q_network,
        optimizer,
        epsilon_greedy=epsilon_greedy,
        n_step_update=n_step_update,
        boltzmann_temperature=boltzmann_temperature,
        emit_log_probability=emit_log_probability,
        # Params for target network updates
        target_update_tau=target_update_tau,
        target_update_period=target_update_period,
        # Params for training.
        td_errors_loss_fn=td_errors_loss_fn,
        gamma=gamma,
        reward_scale_factor=reward_scale_factor,
        gradient_clipping=gradient_clipping,
        # Params for debugging
        debug_summaries=debug_summaries,
        summarize_grads_and_vars=summarize_grads_and_vars,
        train_step_counter=train_step_counter,
        name=name)

    self._rnd_network = rnd_network
    self._target_rnd_network = self._rnd_network.copy(name='TargetRNDNetwork')
    self._rnd_optimizer = rnd_optimizer
    self._rnd_loss_fn = rnd_loss_fn or mean_squared_loss

  # Use @common.function in graph mode or for speeding up.
  def _train(self, experience, weights):
    # TODO(seungjaeryanlee): Would making this persistent change performance?
    with tf.GradientTape(persistent=True) as tape:
      loss_info = self._loss(
          experience,
          td_errors_loss_fn=self._td_errors_loss_fn,
          gamma=self._gamma,
          reward_scale_factor=self._reward_scale_factor,
          weights=weights)
    tf.debugging.check_numerics(loss_info[0], 'Loss is inf or nan')
    variables_to_train = self._q_network.trainable_weights
    assert list(variables_to_train), "No variables in the agent's q_network."
    grads = tape.gradient(loss_info.loss, variables_to_train)
    # Tuple is used for py3, where zip is a generator producing values once.
    grads_and_vars = tuple(zip(grads, variables_to_train))
    if self._gradient_clipping is not None:
      grads_and_vars = eager_utils.clip_gradient_norms(grads_and_vars,
                                                       self._gradient_clipping)

    if self._summarize_grads_and_vars:
      eager_utils.add_variables_summaries(grads_and_vars,
                                          self.train_step_counter)
      eager_utils.add_gradients_summaries(grads_and_vars,
                                          self.train_step_counter)

    self._optimizer.apply_gradients(grads_and_vars,
                                    global_step=self.train_step_counter)

    self._update_target()

    # Train RND Network
    rnd_variables_to_train = self._rnd_network.trainable_weights
    rnd_grads = tape.gradient(loss_info.extra.avg_rnd_loss, rnd_variables_to_train)
    rnd_grads_and_vars = tuple(zip(rnd_grads, rnd_variables_to_train))
    if self._summarize_grads_and_vars:
      eager_utils.add_variables_summaries(rnd_grads_and_vars,
                                          self.train_step_counter)
      eager_utils.add_gradients_summaries(rnd_grads_and_vars,
                                          self.train_step_counter)

    self._rnd_optimizer.apply_gradients(rnd_grads_and_vars,
                                        global_step=self.train_step_counter)

    return loss_info


  # TODO(seungjaeryanlee) A lot of this is redundant...
  def _loss(self,
            experience,
            td_errors_loss_fn=element_wise_huber_loss,
            rnd_loss_fn=mean_squared_loss,
            gamma=1.0,
            reward_scale_factor=1.0,
            weights=None):
    """Computes loss for DQN training.

    Args:
      experience: A batch of experience data in the form of a `Trajectory`. The
        structure of `experience` must match that of `self.policy.step_spec`.
        All tensors in `experience` must be shaped `[batch, time, ...]` where
        `time` must be equal to `self.train_sequence_length` if that
        property is not `None`.
      td_errors_loss_fn: A function(td_targets, predictions) to compute the
        element wise loss.
      rnd_loss_fn: A function(rnd_targets, predictions) to compute the
        experience wise loss.
      gamma: Discount for future rewards.
      reward_scale_factor: Multiplicative factor to scale extrinsic rewards.
      weights: Optional scalar or elementwise (per-batch-entry) importance
        weights.  The output td_loss will be scaled by these weights, and
        the final scalar loss is the mean of these values.

    Returns:
      loss: An instance of `DqnLossInfo`.
    Raises:
      ValueError:
        if the number of actions is greater than 1.
    """
    # Check that `experience` includes two outer dimensions [B, T, ...]. This
    # method requires a time dimension to compute the loss properly.
    self._check_trajectory_dimensions(experience)

    if self._n_step_update == 1:
      time_steps, actions, next_time_steps = self._experience_to_transitions(
          experience)
    else:
      # To compute n-step returns, we need the first time steps, the first
      # actions, and the last time steps. Therefore we extract the first and
      # last transitions from our Trajectory.
      first_two_steps = tf.nest.map_structure(lambda x: x[:, :2], experience)
      last_two_steps = tf.nest.map_structure(lambda x: x[:, -2:], experience)
      time_steps, actions, _ = self._experience_to_transitions(first_two_steps)
      _, _, next_time_steps = self._experience_to_transitions(last_two_steps)

    # Compute RND loss
    # RND loss should be calculated first, since it is used as intrinsic rewards
    with tf.name_scope('rnd_loss'):
      rnd_prediction, _ = self._rnd_network(time_steps.observation,
                                            time_steps.step_type)
      rnd_target, _ = self._target_rnd_network(time_steps.observation,
                                               time_steps.step_type)

      # rnd_losses serve as intrinsic rewards and have shape (BATCH_SIZE, )
      rnd_losses = rnd_loss_fn(rnd_prediction, rnd_target)
      # avg_rnd_loss is a scalar used to train RND network
      avg_rnd_loss = tf.reduce_mean(input_tensor=rnd_losses)

      with tf.name_scope('Losses/'):
        tf.compat.v2.summary.scalar(
            name='rnd_loss', data=avg_rnd_loss, step=self.train_step_counter)

      if self._summarize_grads_and_vars:
        with tf.name_scope('Variables/'):
          for var in self._rnd_network.trainable_weights:
            tf.compat.v2.summary.histogram(
                name=var.name.replace(':', '_'),
                data=var,
                step=self.train_step_counter)

      if self._debug_summaries:
        common.generate_tensor_summaries('rnd_loss', rnd_losses,
                                         self.train_step_counter)

    with tf.name_scope('loss'):
      actions = tf.nest.flatten(actions)[0]
      q_values, _ = self._q_network(time_steps.observation,
                                    time_steps.step_type)

      # Handle action_spec.shape=(), and shape=(1,) by using the
      # multi_dim_actions param.
      multi_dim_actions = tf.nest.flatten(self._action_spec)[0].shape.ndims > 0
      q_values = common.index_with_actions(
          q_values,
          tf.cast(actions, dtype=tf.int32),
          multi_dim_actions=multi_dim_actions)

      next_q_values = self._compute_next_q_values(next_time_steps)

      if self._n_step_update == 1:
        # Special case for n = 1 to avoid a loss of performance.
        td_targets = compute_td_targets(
            next_q_values,
            rewards=reward_scale_factor * next_time_steps.reward + rnd_losses,
            discounts=gamma * next_time_steps.discount)
      else:
        # When computing discounted return, we need to throw out the last time
        # index of both reward and discount, which are filled with dummy values
        # to match the dimensions of the observation.
        rewards = reward_scale_factor * experience.reward[:, :-1] + rnd_losses
        discounts = gamma * experience.discount[:, :-1]

        # TODO(b/134618876): Properly handle Trajectories that include episode
        # boundaries with nonzero discount.

        td_targets = value_ops.discounted_return(
            rewards=rewards,
            discounts=discounts,
            final_value=next_q_values,
            time_major=False,
            provide_all_returns=False)

      valid_mask = tf.cast(~time_steps.is_last(), tf.float32)
      td_error = valid_mask * (td_targets - q_values)

      td_loss = valid_mask * td_errors_loss_fn(td_targets, q_values)

      if nest_utils.is_batched_nested_tensors(
          time_steps, self.time_step_spec, num_outer_dims=2):
        # Do a sum over the time dimension.
        td_loss = tf.reduce_sum(input_tensor=td_loss, axis=1)

      if weights is not None:
        td_loss *= weights

      # Average across the elements of the batch.
      # Note: We use an element wise loss above to ensure each element is always
      #   weighted by 1/N where N is the batch size, even when some of the
      #   weights are zero due to boundary transitions. Weighting by 1/K where K
      #   is the actual number of non-zero weight would artificially increase
      #   their contribution in the loss. Think about what would happen as
      #   the number of boundary samples increases.
      loss = tf.reduce_mean(input_tensor=td_loss)

      with tf.name_scope('Losses/'):
        tf.compat.v2.summary.scalar(
            name='loss', data=loss, step=self.train_step_counter)

      if self._summarize_grads_and_vars:
        with tf.name_scope('Variables/'):
          for var in self._q_network.trainable_weights:
            tf.compat.v2.summary.histogram(
                name=var.name.replace(':', '_'),
                data=var,
                step=self.train_step_counter)

      if self._debug_summaries:
        diff_q_values = q_values - next_q_values
        common.generate_tensor_summaries('td_error', td_error,
                                         self.train_step_counter)
        common.generate_tensor_summaries('td_loss', td_loss,
                                         self.train_step_counter)
        common.generate_tensor_summaries('q_values', q_values,
                                         self.train_step_counter)
        common.generate_tensor_summaries('next_q_values', next_q_values,
                                         self.train_step_counter)
        common.generate_tensor_summaries('diff_q_values', diff_q_values,
                                         self.train_step_counter)

    return tf_agent.LossInfo(loss, RndDqnLossInfo(td_loss=td_loss,
                                                  td_error=td_error,
                                                  rnd_losses=rnd_losses,
                                                  avg_rnd_loss=avg_rnd_loss))


@gin.configurable
class RndDdqnAgent(RndDqnAgent):
  """A RND Double DQN Agent.

  Implements the Double-DQN algorithm from

  "Deep Reinforcement Learning with Double Q-learning"
   Hasselt et al., 2015
   https://arxiv.org/abs/1509.06461

  Implements the RND algorithm from

  "Exploration by Random Network Distillation"
    Burda et al., 2018
    https://arxiv.org/abs/1810.12894

  """

  # NOTE Q) Same as DdqnAgent: How can I remove the redundancy?
  def _compute_next_q_values(self, next_time_steps):
    """Compute the q value of the next state for TD error computation.

    Args:
      next_time_steps: A batch of next timesteps

    Returns:
      A tensor of Q values for the given next state.
    """
    # TODO(b/117175589): Add binary tests for DDQN.
    next_q_values, _ = self._q_network(next_time_steps.observation,
                                       next_time_steps.step_type)
    best_next_actions = tf.cast(
        tf.argmax(input=next_q_values, axis=-1), dtype=tf.int32)
    next_target_q_values, _ = self._target_q_network(
        next_time_steps.observation, next_time_steps.step_type)
    multi_dim_actions = best_next_actions.shape.ndims > 1
    return common.index_with_actions(
        next_target_q_values,
        best_next_actions,
        multi_dim_actions=multi_dim_actions)
