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

"""Common utilities for TF-Agents Environments."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tf_agents.environments import py_environment
from tf_agents.environments import tf_environment
from tf_agents.environments import tf_py_environment
from tf_agents.specs import array_spec


def get_tf_env(environment):
  """Ensures output is a tf_environment, wrapping py_environments if needed."""
  if environment is None:
    raise ValueError('`environment` cannot be None')
  if isinstance(environment, py_environment.PyEnvironment):
    tf_env = tf_py_environment.TFPyEnvironment(environment)
  elif isinstance(environment, tf_environment.TFEnvironment):
    tf_env = environment
  else:
    raise ValueError(
        '`environment` %s must be an instance of '
        '`tf_environment.TFEnvironment` or `py_environment.PyEnvironment`.' %
        environment)
  return tf_env
