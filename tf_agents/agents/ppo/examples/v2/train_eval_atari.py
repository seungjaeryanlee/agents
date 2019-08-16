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

r"""Train and Eval PPO on Atari environments.

To run on Atari Pong without RND:

```bash
tensorboard --logdir $HOME/tmp/ppo/gym/PongDeterministic-v0/ --port 2223 &

python tf_agents/agents/ppo/examples/v2/train_eval_atari.py \
  --root_dir=$HOME/tmp/ppo/gym/PongDeterministic-v0/ \
  --logtostderr
```

To run on Atari Pong with RND:

```bash
tensorboard --logdir $HOME/tmp/rndppo/gym/PongDeterministic-v0/ --port 2223 &

python tf_agents/agents/ppo/examples/v2/train_eval_atari.py \
  --root_dir=$HOME/tmp/rndppo/gym/PongDeterministic-v0/ \
  --logtostderr --use_rnd
```
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import time

from absl import app
from absl import flags
from absl import logging

import gin
import tensorflow as tf

from tf_agents.agents.ppo import ppo_agent
from tf_agents.drivers import dynamic_episode_driver
from tf_agents.environments import parallel_py_environment
from tf_agents.environments import suite_atari
from tf_agents.environments import tf_py_environment
from tf_agents.environments.suite_atari import DEFAULT_ATARI_GYM_WRAPPERS_WITH_STACKING
from tf_agents.eval import metric_utils
from tf_agents.metrics import tf_metrics
from tf_agents.networks import actor_distribution_network
from tf_agents.networks import actor_distribution_rnn_network
from tf_agents.networks import encoding_network
from tf_agents.networks import value_network
from tf_agents.networks import value_rnn_network
from tf_agents.replay_buffers import tf_uniform_replay_buffer
from tf_agents.utils import common
from train_eval import train_eval


flags.DEFINE_string('root_dir', os.getenv('TEST_UNDECLARED_OUTPUTS_DIR'),
                    'Root directory for writing logs/summaries/checkpoints.')
flags.DEFINE_string('env_name', 'PongDeterministic-v0', 'Name of an environment')
flags.DEFINE_integer('replay_buffer_capacity', 18001,
                     'Replay buffer capacity per env.')
flags.DEFINE_integer('num_parallel_environments', 16,
                     'Number of environments to run in parallel')
flags.DEFINE_integer('num_environment_steps', 2000000000,
                     'Number of environment steps to run before finishing.')
flags.DEFINE_integer('num_epochs', 4,
                     'Number of epochs for computing policy updates.')
flags.DEFINE_integer(
    'collect_episodes_per_iteration', 16,
    'The number of episodes to take in the environment before '
    'each update. This is the total across all parallel '
    'environments.')
flags.DEFINE_integer('num_eval_episodes', 30,
                     'The number of episodes to run eval on.')
flags.DEFINE_boolean('use_rnns', False,
                     'If true, use RNN for policy and value function.')
flags.DEFINE_boolean('use_rnd', False,
                     'If true, use RND for reward shaping.')
flags.DEFINE_integer('norm_init_episodes', 5,
                    'The number of episodes to initialize the normalizers.')
FLAGS = flags.FLAGS


def main(_):
  logging.set_verbosity(logging.INFO)
  tf.compat.v1.enable_v2_behavior()
  train_eval(
      FLAGS.root_dir,
      env_name=FLAGS.env_name,
      env_load_fn=suite_atari.load,
      gym_env_wrappers=DEFAULT_ATARI_GYM_WRAPPERS_WITH_STACKING,
      use_rnns=FLAGS.use_rnns,
      use_rnd=FLAGS.use_rnd,
      num_environment_steps=FLAGS.num_environment_steps,
      collect_episodes_per_iteration=FLAGS.collect_episodes_per_iteration,
      num_parallel_environments=FLAGS.num_parallel_environments,
      replay_buffer_capacity=FLAGS.replay_buffer_capacity,
      norm_init_episodes=FLAGS.norm_init_episodes,
      num_epochs=FLAGS.num_epochs,
      num_eval_episodes=FLAGS.num_eval_episodes)


if __name__ == '__main__':
  flags.mark_flag_as_required('root_dir')
  app.run(main)
