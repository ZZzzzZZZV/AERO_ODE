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

from __future__ import annotations

import dataclasses
import functools
from typing import Sequence

from dinosaur import jax_numpy_utils
from dinosaur import typing

import jax
from jax import lax
import jax.numpy as jnp
import numpy as np


Array = typing.Array


einsum = functools.partial(jnp.einsum, precision=lax.Precision.HIGHEST)


def _slice_shape_along_axis(
    x: np.ndarray,
    axis: int,
    slice_width: int = 1,
) -> tuple[int, ...]:

  x_shape = list(x.shape)
  x_shape[axis] = slice_width
  return tuple(x_shape)


@dataclasses.dataclass(frozen=True)
class SigmaCoordinates:


  boundaries: np.ndarray

  def __init__(self, boundaries: Sequence[float] | np.ndarray):
    object.__setattr__(self, 'boundaries', np.asarray(boundaries))
    if not (
        np.isclose(self.boundaries[0], 0) and np.isclose(self.boundaries[-1], 1)
    ):
      raise ValueError(
          'Expected boundaries[0] = 0, boundaries[-1] = 1, '
          f'got boundaries = {self.boundaries}'
      )
    if not all(np.diff(self.boundaries) > 0):
      raise ValueError(
          'Expected `boundaries` to be monotonically increasing, '
          f'got boundaries = {self.boundaries}'
      )

  @property
  def internal_boundaries(self) -> np.ndarray:
    return self.boundaries[1:-1]

  @property
  def centers(self) -> np.ndarray:
    return (self.boundaries[1:] + self.boundaries[:-1]) / 2

  @property
  def layer_thickness(self) -> np.ndarray:
    return np.diff(self.boundaries)

  @property
  def center_to_center(self) -> np.ndarray:
    return np.diff(self.centers)

  @property
  def layers(self) -> int:
    return len(self.boundaries) - 1

  @classmethod
  def equidistant(cls, layers: int) -> SigmaCoordinates:
    boundaries = np.linspace(0, 1, layers + 1)
    return cls(boundaries)

  def asdict(self):
    return {k: v.tolist() for k, v in dataclasses.asdict(self).items()}

  def __hash__(self):
    return hash(tuple(self.centers.tolist()))

  def __eq__(self, other):
    return isinstance(other, SigmaCoordinates) and np.array_equal(
        self.centers, other.centers
    )


@jax.named_call
def centered_difference(
    x: np.ndarray, coordinates: SigmaCoordinates, axis: int = -3
) -> np.ndarray:

  if coordinates.layers != x.shape[axis]:
    raise ValueError(
        '`x.shape[axis]` must be equal to `coordinates.layers`; '
        f'got {x.shape[axis]} and {coordinates.layers}.'
    )

  dx = jax_numpy_utils.diff(x, axis=axis)
  dx_axes = range(dx.ndim)
  inv_d𝜎 = 1 / coordinates.center_to_center
  inv_d𝜎_axes = [dx_axes[axis]]
  return einsum(dx, dx_axes, inv_d𝜎, inv_d𝜎_axes, dx_axes, precision='float32')


@jax.named_call
def cumulative_sigma_integral(
    x: Array,
    coordinates: SigmaCoordinates,
    axis: int = -3,
    downward: bool = True,
    cumsum_method: str = 'dot',
    sharding: jax.sharding.NamedSharding | None = None,
) -> jax.Array:

  if coordinates.layers != x.shape[axis]:
    raise ValueError(
        '`x.shape[axis]` must be equal to `coordinates.layers`;'
        f'got {x.shape[axis]} and {coordinates.layers}.'
    )
  x_axes = range(x.ndim)
  d𝜎 = coordinates.layer_thickness
  d𝜎_axes = [x_axes[axis]]
  xd𝜎 = einsum(x, x_axes, d𝜎, d𝜎_axes, x_axes)
  if downward:
    return jax_numpy_utils.cumsum(
        xd𝜎, axis, method=cumsum_method, sharding=sharding
    )
  else:
    return jax_numpy_utils.reverse_cumsum(
        xd𝜎, axis, method=cumsum_method, sharding=sharding
    )


@jax.named_call
def sigma_integral(
    x: Array,
    coordinates: SigmaCoordinates,
    axis: int = -3,
    keepdims: bool = True,
) -> jax.Array:

  if coordinates.layers != x.shape[axis]:
    raise ValueError(
        '`x.shape[axis]` must be equal to `coordinates.layers`;'
        f'got {x.shape[axis]} and {coordinates.layers}.'
    )
  x_axes = range(x.ndim)
  d𝜎 = coordinates.layer_thickness
  d𝜎_axes = [x_axes[axis]]
  xd𝜎 = einsum(x, x_axes, d𝜎, d𝜎_axes, x_axes)
  return xd𝜎.sum(axis=axis, keepdims=keepdims)


@jax.named_call
def cumulative_log_sigma_integral(
    x: Array,
    coordinates: SigmaCoordinates,
    axis: int = -3,
    downward: bool = True,
    cumsum_method: str = 'dot',
) -> jax.Array:

  if coordinates.layers != x.shape[axis]:
    raise ValueError(
        '`x.shape[axis]` must be equal to `coordinates.layers`;'
        f'got {x.shape[axis]} and {coordinates.layers}.'
    )


  x_last = lax.slice_in_dim(x, -1, None, axis=axis)
  x_interpolated = (
      lax.slice_in_dim(x, 1, None, axis=axis)
      + lax.slice_in_dim(x, 0, -1, axis=axis)
  ) / 2
  integrand = jnp.concatenate([x_interpolated, x_last], axis=axis)
  integrand_axes = range(integrand.ndim)
  log𝜎 = jnp.log(coordinates.centers)
  dlog𝜎 = jnp.diff(log𝜎, append=0)
  dlog𝜎_axes = [integrand_axes[axis]]
  xd𝜎 = einsum(integrand, integrand_axes, dlog𝜎, dlog𝜎_axes, integrand_axes)
  if downward:
    return jax_numpy_utils.cumsum(xd𝜎, axis, method=cumsum_method)
  else:
    return jax_numpy_utils.reverse_cumsum(xd𝜎, axis, method=cumsum_method)


@jax.named_call
def centered_vertical_advection(
    w: Array,
    x: Array,
    coordinates: SigmaCoordinates,
    axis: int = -3,
    w_boundary_values: tuple[Array, Array] | None = None,
    dx_dsigma_boundary_values: tuple[Array, Array] | None = None,
) -> jnp.ndarray:

  if w_boundary_values is None:
    w_slc_shape = _slice_shape_along_axis(w, axis)
    w_boundary_values = (
        jnp.zeros(w_slc_shape, dtype=jax.dtypes.canonicalize_dtype(w.dtype)),
        jnp.zeros(w_slc_shape, dtype=jax.dtypes.canonicalize_dtype(w.dtype)),
    )
  if dx_dsigma_boundary_values is None:
    x_slc_shape = _slice_shape_along_axis(x, axis)
    dx_dsigma_boundary_values = (
        jnp.zeros(x_slc_shape, dtype=jax.dtypes.canonicalize_dtype(x.dtype)),
        jnp.zeros(x_slc_shape, dtype=jax.dtypes.canonicalize_dtype(x.dtype)),
    )

  w_boundary_top, w_boundary_bot = w_boundary_values
  w = jnp.concatenate([w_boundary_top, w, w_boundary_bot], axis=axis)

  x_diff = centered_difference(x, coordinates, axis)
  x_diff_boundary_top, x_diff_boundary_bot = dx_dsigma_boundary_values
  x_diff = jnp.concatenate(
      [x_diff_boundary_top, x_diff, x_diff_boundary_bot], axis=axis
  )

  w_times_x_diff = w * x_diff
  return -0.5 * (
      lax.slice_in_dim(w_times_x_diff, 1, None, axis=axis)
      + lax.slice_in_dim(w_times_x_diff, 0, -1, axis=axis)
  )


@jax.named_call
def upwind_vertical_advection(
    w: Array,
    x: Array,
    coordinates: SigmaCoordinates,
    axis: int = -3,
) -> jnp.ndarray:

  w_slc_shape = _slice_shape_along_axis(w, axis)
  w_boundary_values = (
      jnp.zeros(w_slc_shape, dtype=jax.dtypes.canonicalize_dtype(w.dtype)),
      jnp.zeros(w_slc_shape, dtype=jax.dtypes.canonicalize_dtype(w.dtype)),
  )

  x_slc_shape = _slice_shape_along_axis(x, axis)
  dx_dsigma_boundary_values = (
      jnp.zeros(x_slc_shape, dtype=jax.dtypes.canonicalize_dtype(x.dtype)),
      jnp.zeros(x_slc_shape, dtype=jax.dtypes.canonicalize_dtype(x.dtype)),
  )


  x_diff = centered_difference(x, coordinates, axis)

  w_boundary_top, w_boundary_bot = w_boundary_values
  w_up = jnp.concatenate([w_boundary_top, w], axis=axis)
  w_down = jnp.concatenate([w, w_boundary_bot], axis=axis)

  x_diff_boundary_top, x_diff_boundary_bot = dx_dsigma_boundary_values
  x_diff_up = jnp.concatenate([x_diff_boundary_top, x_diff], axis=axis)
  x_diff_down = jnp.concatenate([x_diff, x_diff_boundary_bot], axis=axis)

  return -(jnp.maximum(w_up, 0) * x_diff_up +
           jnp.minimum(w_down, 0) * x_diff_down)


