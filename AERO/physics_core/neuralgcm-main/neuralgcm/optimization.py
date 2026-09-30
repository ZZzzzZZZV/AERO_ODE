# Copyright 2024 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import collections
import re
from typing import Sequence

import gin
import optax


gin.external_configurable(optax.adabelief, module='optax')
gin.external_configurable(optax.adam, module='optax')
gin.external_configurable(optax.adamw, module='optax')

gin.external_configurable(optax.constant_schedule, module='optax')
gin.external_configurable(optax.join_schedules, module='optax')
gin.external_configurable(optax.piecewise_constant_schedule, module='optax')
gin.external_configurable(optax.exponential_decay, module='optax')
gin.external_configurable(
    optax.warmup_exponential_decay_schedule, module='optax'
)


class OptimizerError(Exception):
  pass


@gin.configurable
def optimizer(value):
  return value


OptState = collections.namedtuple('OptState', ['state', 'params'])


@gin.register
def piecewise_constant_schedule_specified_by_rates(
    rates: Sequence[float],
    boundaries: Sequence[int],
) -> optax.Schedule:

  return optax.join_schedules(
      schedules=[optax.constant_schedule(r) for r in rates],
      boundaries=boundaries,
  )


@gin.register
def delayed_constant_schedule(
    turn_on_step: int,
    rate: float,
) -> optax.Schedule:

  return piecewise_constant_schedule_specified_by_rates(
      rates=[0., rate],
      boundaries=[turn_on_step],
  )


@gin.register
def top_level_multi_adam(
    top_level_keys: Sequence[str] = (),
    learning_rates: Sequence[optax.ScalarOrSchedule] = (),
    default_learning_rate: optax.ScalarOrSchedule = 1e-4,
    b1: float = 0.9,
    b2: float = 0.95,
    eps: float = 1e-6,
    raise_if_keys_not_found: bool = True,
) -> optax.GradientTransformation:

  if len(top_level_keys) != len(learning_rates):
    raise ValueError(
        f'{top_level_keys=} had different length than {learning_rates=}'
    )
  if '' in top_level_keys:
    raise ValueError('An empty string "" was found in `top_level_keys`.')

  default_label = 'DEFAULT_LABEL'
  if default_label in top_level_keys:
    raise ValueError(f'{default_label=} should not be in `top_level_keys`')

  def find_matching_top_level_key(param_name: str) -> str:

    prefix = 'REGEX_'
    matches = []
    for k in top_level_keys:
      if k.startswith(prefix) and re.search(k.lstrip(prefix), param_name):
        matches.append(k)
      elif k == param_name:
        matches.append(k)
    if not matches:
      return default_label
    elif len(matches) == 1:
      return matches[0]
    else:
      raise ValueError(
          f'{param_name=} had more than 1 ({len(matches)}) match '
          f'({matches}). Only one `top_level_keys` should match, or else we '
          'cannot choose a unique learning rate for these parameters.'
      )

  def get_prefix_labels(params):


    labels = {
        param_name: find_matching_top_level_key(param_name)
        for param_name in params
    }
    top_level_keys_that_matched = [
        k for k in labels.values() if k != default_label
    ]
    missing_keys = set(top_level_keys).difference(top_level_keys_that_matched)
    if raise_if_keys_not_found and missing_keys:
      raise OptimizerError(
          f'{missing_keys=} not found in params: {sorted(params)}'
      )
    return labels

  def make_adam(lr):
    return optax.adam(lr, b1=b1, b2=b2, eps=eps)

  return optax.multi_transform(
      transforms={
          k: make_adam(lr) for k, lr in zip(top_level_keys, learning_rates)
      }
      | {default_label: make_adam(default_learning_rate)},
      param_labels=get_prefix_labels,
  )
