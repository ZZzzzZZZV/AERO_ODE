# Copyright 2023 Google LLC

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     https://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import functools
from typing import Callable

from dinosaur import spherical_harmonic
from dinosaur import typing

import jax
import jax.numpy as jnp
import numpy as np


def _preserves_shape(target, scaling):
  target_shape = np.shape(target)
  return target_shape == np.broadcast_shapes(target_shape, scaling.shape)


def _make_filter_fn(scaling, name=None):
  rescale = lambda x: scaling * x if _preserves_shape(x, scaling) else x
  return functools.partial(
      jax.tree_util.tree_map, jax.named_call(rescale, name=name))


def exponential_filter(
    grid: spherical_harmonic.Grid,
    attenuation: float | typing.Array = 16,
    order: int | typing.Array = 18,
    cutoff: float = 0,
) -> Callable[[typing.PyTreeState], typing.PyTreeState]:


  _, total_wavenumber = grid.modal_axes

  k = total_wavenumber / total_wavenumber.max()
  a = attenuation
  c = cutoff
  p = order

  scaling = jnp.exp((k > c) * (-a * (((k - c) / (1 - c)) ** (2 * p))))
  return _make_filter_fn(scaling, "exponential_filter")


def horizontal_diffusion_filter(
    grid: spherical_harmonic.Grid,
    scale: float | typing.Array,
    order: int = 1,
) -> Callable[[typing.PyTreeState], typing.PyTreeState]:

  eigenvalues = grid.laplacian_eigenvalues
  scaling = jnp.exp(-scale * (-eigenvalues) ** order)
  return _make_filter_fn(scaling, "horizontal_diffusion_filter")
