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
from typing import Any, Optional, Sequence

import gin
import haiku as hk
import jax
import numpy as np


Constant = gin.external_configurable(hk.initializers.Constant)
VarianceScaling = gin.external_configurable(hk.initializers.VarianceScaling)
Orthogonal = gin.external_configurable(hk.initializers.Orthogonal)


def _compute_fans(
    shape: Sequence[int],
    fan_in_axes: Optional[Sequence[int]] = None,
) -> tuple[int, int]:


  if len(shape) < 1:
    fan_in = fan_out = 1
  elif len(shape) == 1:
    fan_in = fan_out = shape[0]
  elif len(shape) == 2:
    fan_in, fan_out = shape
  else:
    if fan_in_axes is not None:

      fan_in = np.prod([shape[i] for i in fan_in_axes])
      fan_out = np.prod([s for i, s in enumerate(shape)
                         if i not in fan_in_axes])
    else:


      receptive_field_size = np.prod(shape[:-2])
      fan_in = shape[-2] * receptive_field_size
      fan_out = shape[-1] * receptive_field_size
  return fan_in, fan_out


@gin.register
class ReducingVarianceScaling(hk.initializers.Initializer):


  def __init__(
      self,
      scale=1.0,
      mode='fan_in',
      distribution='truncated_normal',
      fan_in_axes=None,
  ):

    if scale < 0.0:
      raise ValueError('`scale` must be a positive float.')
    if mode not in {'fan_in', 'fan_out', 'fan_avg'}:
      raise ValueError('Invalid `mode` argument:', mode)
    distribution = distribution.lower()
    if distribution not in {'normal', 'truncated_normal', 'uniform'}:
      raise ValueError('Invalid `distribution` argument:', distribution)
    self.scale = scale
    self.mode = mode
    self.distribution = distribution
    self.fan_in_axes = fan_in_axes

  def __call__(self, shape: Sequence[int], dtype: Any) -> jax.Array:
    scale = self.scale
    fan_in, fan_out = _compute_fans(shape, self.fan_in_axes)
    if self.mode == 'fan_in':
      scale /= max(1.0, fan_in) ** 2
    elif self.mode == 'fan_out':
      scale /= max(1.0, fan_out) ** 2
    else:
      scale /= max(1.0, (fan_in + fan_out) / 2.0) ** 2

    if self.distribution == 'truncated_normal':
      stddev = np.sqrt(scale)


      distribution_stddev = np.asarray(.87962566103423978, dtype=dtype)
      stddev = stddev / distribution_stddev
      return hk.initializers.TruncatedNormal(stddev=stddev)(shape, dtype)
    elif self.distribution == 'normal':
      stddev = np.sqrt(scale)
      return hk.initializers.RandomNormal(stddev=stddev)(shape, dtype)
    else:
      limit = np.sqrt(3.0 * scale)
      uniform_init = hk.initializers.RandomUniform(minval=-limit, maxval=limit)
      return uniform_init(shape, dtype)
