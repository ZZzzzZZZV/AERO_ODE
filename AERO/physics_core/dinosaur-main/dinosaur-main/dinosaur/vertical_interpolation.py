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
import importlib
from typing import Any, Callable, Dict, Sequence, Union

import dinosaur
from dinosaur import pytree_utils
from dinosaur import sigma_coordinates
from dinosaur import typing
import jax
import jax.numpy as jnp
import numpy as np


Array = typing.Array
InterpolateFn = Callable[[Array, Array, Array], Array]


def vectorize_vertical_interpolation(
    interpolate_fn: InterpolateFn,
) -> InterpolateFn:


  interpolate_fn = jax.vmap(interpolate_fn, (-1, None, -1), out_axes=-1)
  interpolate_fn = jax.vmap(interpolate_fn, (-1, None, -1), out_axes=-1)

  interpolate_fn = jax.vmap(interpolate_fn, (0, None, None), out_axes=0)

  return jnp.vectorize(interpolate_fn, signature='(a,x,y),(b),(b,x,y)->(a,x,y)')


def _dot_interp(x, xp, fp):


  n = len(xp)
  i = jnp.arange(n)
  dx = xp[1:] - xp[:-1]
  delta = x - xp[:-1]
  w = delta / dx
  w_left = jnp.pad(1 - w, [(0, 1)])
  w_right = jnp.pad(w, [(1, 0)])
  u = jnp.searchsorted(xp, x, side='right', method='compare_all')
  u = jnp.clip(u, 1, n - 1)
  weights = w_left * (i == (u - 1)) + w_right * (i == u)
  weights = jnp.where(x < xp[0], i == 0, weights)
  weights = jnp.where(x > xp[-1], i == (n - 1), weights)
  return jnp.dot(weights, fp, precision='highest')


@jax.jit
def interp(
    x: typing.Numeric, xp: typing.Array, fp: typing.Array
) -> jnp.ndarray:

  on_tpu = all(device.platform == 'tpu' for device in jax.local_devices())
  if on_tpu:
    return _dot_interp(x, xp, fp)
  else:
    return jnp.interp(x, xp, fp)


def _extrapolate_left(y):
  delta = y[1] - y[0]
  return jnp.concatenate([jnp.array([y[0] - delta]), y])


def _extrapolate_right(y):
  delta = y[-1] - y[-2]
  return jnp.concatenate([y, jnp.array([y[-1] + delta])])


def _extrapolate_both(y):
  return _extrapolate_left(_extrapolate_right(y))


def _linear_interp_with_safe_extrap(x, xp, fp, n=1):

  for _ in range(n):
    xp = _extrapolate_both(xp)
    fp = _extrapolate_both(fp)
  return jnp.interp(x, xp, fp, left=np.nan, right=np.nan)


@jax.jit
def linear_interp_with_linear_extrap(
    x: typing.Numeric, xp: typing.Array, fp: typing.Array
) -> jnp.ndarray:

  n = len(xp)
  i = jnp.arange(n)
  dx = xp[1:] - xp[:-1]
  delta = x - xp[:-1]
  w = delta / dx
  w_left = jnp.pad(1 - w, [(0, 1)])
  w_right = jnp.pad(w, [(1, 0)])
  u = jnp.searchsorted(xp, x, side='right', method='compare_all')
  u = jnp.clip(u, 1, n - 1)
  weights = w_left * (i == (u - 1)) + w_right * (i == u)
  return jnp.dot(weights, fp, precision='highest')


@dataclasses.dataclass(frozen=True)
class PressureCoordinates:


  centers: np.ndarray

  def __init__(self, centers: Union[Sequence[float], np.ndarray]):
    object.__setattr__(self, 'centers', np.asarray(centers))
    if not all(np.diff(self.centers) > 0):
      raise ValueError(
          'Expected `centers` to be monotonically increasing, '
          f'got centers = {self.centers}'
      )

  @property
  def layers(self) -> int:
    return len(self.centers)

  def asdict(self) -> Dict[str, Any]:
    return {k: v.tolist() for k, v in dataclasses.asdict(self).items()}

  def __hash__(self):
    return hash(tuple(self.centers.tolist()))

  def __eq__(self, other):
    return isinstance(other, PressureCoordinates) and np.array_equal(
        self.centers, other.centers
    )


@dataclasses.dataclass(frozen=True)
class HybridCoordinates:


  a_boundaries: np.ndarray
  b_boundaries: np.ndarray

  def __post_init__(self):
    if len(self.a_boundaries) != len(self.b_boundaries):
      raise ValueError(
          'Expected `a_boundaries` and `b_boundaries` to have the same length, '
          f'got {len(self.a_boundaries)} and {len(self.b_boundaries)}.'
      )

  @classmethod
  def _from_resource_csv(cls, path: str) -> HybridCoordinates:
    levels_csv = importlib.resources.files(dinosaur).joinpath(path)
    with levels_csv.open() as f:
      a_in_pa, b = np.loadtxt(f, skiprows=1, usecols=(1, 2), delimiter='\t').T
    a = a_in_pa / 100


    assert 100 < a.max() < 1000
    return cls(a_boundaries=a, b_boundaries=b)

  @classmethod
  def ECMWF137(cls) -> HybridCoordinates:

    return cls._from_resource_csv('data/ecmwf137_hybrid_levels.csv')

  @classmethod
  def UFS127(cls) -> HybridCoordinates:

    return cls._from_resource_csv('data/ufs127_hybrid_levels.csv')

  @property
  def layers(self) -> int:
    return len(self.a_boundaries) - 1

  def __hash__(self):
    return hash(
        (tuple(self.a_boundaries.tolist()), tuple(self.b_boundaries.tolist()))
    )

  def __eq__(self, other):
    return (
        isinstance(other, HybridCoordinates)
        and np.array_equal(self.a_boundaries, other.a_boundaries)
        and np.array_equal(self.b_boundaries, other.b_boundaries)
    )

  def get_sigma_boundaries(
      self, surface_pressure: typing.Numeric
  ) -> typing.Array:

    return self.a_boundaries / surface_pressure + self.b_boundaries

  def get_sigma_centers(self, surface_pressure: typing.Numeric) -> typing.Array:

    boundaries = self.get_sigma_boundaries(surface_pressure)
    return (boundaries[1:] + boundaries[:-1]) / 2

  def to_approx_sigma_coords(
      self, layers: int, surface_pressure: float = 1013.25
  ) -> sigma_coordinates.SigmaCoordinates:

    original_bounds = self.get_sigma_boundaries(surface_pressure)
    interpolated_bounds = jax.vmap(jnp.interp, (0, None, None))(
        jnp.linspace(0, 1, num=layers + 1),
        jnp.linspace(0, 1, num=self.layers + 1),
        original_bounds,
    )
    interpolated_bounds = np.array(interpolated_bounds)


    interpolated_bounds[0] = 0.0
    interpolated_bounds[-1] = 1.0
    return sigma_coordinates.SigmaCoordinates(interpolated_bounds)


@functools.partial(jax.jit, static_argnums=0)
def get_surface_pressure(
    pressure_levels: PressureCoordinates,
    geopotential: typing.Array,
    orography: typing.Array,
    gravity_acceleration: float,
) -> typing.Array:


  relative_height = orography * gravity_acceleration - geopotential

  @functools.partial(jnp.vectorize, signature='(z,x,y),(z)->(1,x,y)')
  @functools.partial(jax.vmap, in_axes=(-1, None), out_axes=-1)
  @functools.partial(jax.vmap, in_axes=(-1, None), out_axes=-1)
  def find_intercept(rh, levels):
    return linear_interp_with_linear_extrap(0.0, rh, levels)[np.newaxis]

  return find_intercept(relative_height, pressure_levels.centers)


def vertical_interpolation(
    x: typing.Array,
    xp: typing.Array,
    fp: typing.Array,
) -> typing.Array:

  return interp(x, jnp.asarray(xp), fp)


@functools.partial(jax.jit, static_argnums=(1, 2, 4))
def interp_pressure_to_sigma(
    fields: typing.Pytree,
    pressure_coords: PressureCoordinates,
    sigma_coords: sigma_coordinates.SigmaCoordinates,
    surface_pressure: typing.Array,
    interpolate_fn: InterpolateFn = (
        vectorize_vertical_interpolation(_linear_interp_with_safe_extrap)
    ),
) -> typing.Pytree:

  desired = sigma_coords.centers[:, np.newaxis, np.newaxis] * surface_pressure
  regrid = lambda x: interpolate_fn(desired, pressure_coords.centers, x)

  def cond_fn(x) -> bool:
    shape = jnp.shape(x)
    return len(shape) >= 3 and shape[-3] == pressure_coords.centers.shape[0]

  return pytree_utils.tree_map_where(
      condition_fn=cond_fn,
      f=regrid,
      g=lambda x: x,
      x=fields,
  )


@functools.partial(jax.jit, static_argnums=(1, 2, 4))
def interp_sigma_to_pressure(
    fields: typing.Pytree,
    pressure_coords: PressureCoordinates,
    sigma_coords: sigma_coordinates.SigmaCoordinates,
    surface_pressure: typing.Array,
    interpolate_fn: InterpolateFn = (
        vectorize_vertical_interpolation(_linear_interp_with_safe_extrap)
    ),
) -> typing.Pytree:

  desired = (
      pressure_coords.centers[:, np.newaxis, np.newaxis] / surface_pressure
  )
  regrid = lambda x: interpolate_fn(desired, sigma_coords.centers, x)
  return pytree_utils.tree_map_over_nonscalars(regrid, fields)


@functools.partial(jax.jit, static_argnums=(1, 2))
def interp_hybrid_to_sigma(
    fields: typing.Pytree,
    hybrid_coords: HybridCoordinates,
    sigma_coords: sigma_coordinates.SigmaCoordinates,
    surface_pressure: typing.Array,
) -> typing.Pytree:


  @jax.jit
  @functools.partial(jnp.vectorize, signature='(x,y),(a),(b,x,y)->(a,x,y)')
  @functools.partial(jax.vmap, in_axes=(-1, None, -1), out_axes=-1)
  @functools.partial(jax.vmap, in_axes=(-1, None, -1), out_axes=-1)
  def regrid(surface_pressure, target_sigmas, field):
    source_sigmas = hybrid_coords.get_sigma_centers(surface_pressure)
    return jax.vmap(_linear_interp_with_safe_extrap, in_axes=(0, None, None))(
        target_sigmas, source_sigmas, field
    )

  return pytree_utils.tree_map_over_nonscalars(
      lambda x: regrid(surface_pressure, sigma_coords.centers, x), fields
  )


def _interval_overlap(
    source_bounds: typing.Array, target_bounds: typing.Array
) -> jnp.ndarray:


  upper = jnp.minimum(
      target_bounds[1:, jnp.newaxis], source_bounds[jnp.newaxis, 1:]
  )
  lower = jnp.maximum(
      target_bounds[:-1, jnp.newaxis], source_bounds[jnp.newaxis, :-1]
  )
  return jnp.maximum(upper - lower, 0)


def conservative_regrid_weights(
    source_bounds: typing.Array, target_bounds: typing.Array
) -> jnp.ndarray:

  weights = _interval_overlap(source_bounds, target_bounds)
  weights /= jnp.sum(weights, axis=1, keepdims=True)
  assert weights.shape == (target_bounds.size - 1, source_bounds.size - 1)
  return weights


@functools.partial(jax.jit, static_argnums=(1, 2))
def regrid_hybrid_to_sigma(
    fields: typing.Pytree,
    hybrid_coords: HybridCoordinates,
    sigma_coords: sigma_coordinates.SigmaCoordinates,
    surface_pressure: typing.Array,
) -> typing.Pytree:


  @jax.jit
  @functools.partial(jnp.vectorize, signature='(x,y),(a),(b,x,y)->(c,x,y)')
  @functools.partial(jax.vmap, in_axes=(-1, None, -1), out_axes=-1)
  @functools.partial(jax.vmap, in_axes=(-1, None, -1), out_axes=-1)
  def regrid(surface_pressure, sigma_bounds, field):
    assert sigma_bounds.shape == (sigma_coords.layers + 1,)
    if field.shape[0] != hybrid_coords.layers:
      raise ValueError(
          f'Source has {hybrid_coords.layers} layers, but field has'
          f' {fields.shape[0]}'
      )
    hybrid_bounds = hybrid_coords.get_sigma_boundaries(surface_pressure)
    weights = conservative_regrid_weights(hybrid_bounds, sigma_bounds)
    result = jnp.einsum('ab,b->a', weights, field, precision='float32')
    assert result.shape[0] == sigma_coords.layers
    return result

  return pytree_utils.tree_map_over_nonscalars(
      lambda x: regrid(surface_pressure, sigma_coords.boundaries, x), fields
  )


@dataclasses.dataclass(frozen=True)
class Regridder:


  source_grid: HybridCoordinates
  target_grid: sigma_coordinates.SigmaCoordinates

  def __call__(
      self, field: typing.Array, surface_pressure: typing.Array
  ) -> jnp.ndarray:
    raise NotImplementedError


class ConservativeRegridder(Regridder):


  def __call__(
      self, field: typing.Array, surface_pressure: typing.Array
  ) -> jnp.ndarray:
    return regrid_hybrid_to_sigma(
        field, self.source_grid, self.target_grid, surface_pressure
    )


class BilinearRegridder(Regridder):


  def __call__(
      self, field: typing.Array, surface_pressure: typing.Array
  ) -> jnp.ndarray:
    return interp_hybrid_to_sigma(
        field, self.source_grid, self.target_grid, surface_pressure
    )
