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
from __future__ import annotations
import dataclasses
from typing import Callable, Optional, Sequence

from dinosaur import coordinate_systems
from dinosaur import horizontal_interpolation
from dinosaur import pytree_utils
from dinosaur import spherical_harmonic
from dinosaur import typing
import frozendict
import gin
import jax
import jax.core
from jax.interpreters import ad
from jax.interpreters import batching
from jax.interpreters import mlir
import jax.numpy as jnp
import numpy as np


tree_map = jax.tree_util.tree_map
tree_leaves = jax.tree_util.tree_leaves
Pytree = typing.Pytree
TrajectoryRepresentations = typing.TrajectoryRepresentations


N_VARS = 5


TIME = 'time'
LEVEL = 'level'
LONGITUDINAL_WAVENUMBER = 'longitudinal_wavenumber'
TOTAL_WAVENUMBER = 'total_wavenumber'
LONGITUDINAL = 'longitudinal'
LATITUDINAL = 'latitudinal'


SPATIAL_AXES = (-2, -1)
TIME_AXIS = 0
LEVEL_AXIS = 1
ALL_AXES = (TIME_AXIS, LEVEL_AXIS) + SPATIAL_AXES


MODAL_AXIS_INDICES = frozendict.frozendict({
    TIME: TIME_AXIS,
    LEVEL: LEVEL_AXIS,
    LONGITUDINAL_WAVENUMBER: 2,
    TOTAL_WAVENUMBER: 3,
})


NODAL_AXIS_INDICES = frozendict.frozendict({
    TIME: TIME_AXIS,
    LEVEL: LEVEL_AXIS,
    LONGITUDINAL: 2,
    LATITUDINAL: 3,
})


class ShapeError(Exception):
  pass


@dataclasses.dataclass
class TrajectorySpec:


  trajectory_length: int
  max_trajectory_length: int
  steps_per_save: int
  coords: coordinate_systems.CoordinateSystem
  data_coords: coordinate_systems.CoordinateSystem

  def __post_init__(self):
    if self.trajectory_length > self.max_trajectory_length:
      raise ValueError(
          f'{self.trajectory_length=} > {self.max_trajectory_length=}.'
      )


@dataclasses.dataclass
class TrajectoryShape:


  n_times: int
  n_levels: int
  n_longitudinal_wavenumbers: int
  n_total_wavenumbers: int
  n_longitude_nodes: int
  n_latitude_nodes: int

  def assert_compliant(self, trajectory: typing.Pytree, is_nodal: bool) -> None:

    if is_nodal:
      expected_shape = (
          self.n_times,
          self.n_levels,
          self.n_longitude_nodes,
          self.n_latitude_nodes,
      )
    else:
      expected_shape = (
          self.n_times,
          self.n_levels,
          self.n_longitudinal_wavenumbers,
          self.n_total_wavenumbers,
      )

    is_compliant = tree_map(lambda x: np.shape(x) == expected_shape, trajectory)
    if not all(tree_leaves(is_compliant)):
      shapes = tree_map(np.shape, trajectory)
      raise ShapeError(
          f'Some `trajectory` shapes were non-compliant ({is_nodal=}). '
          f'{expected_shape=}. Found {shapes=}. '
          f'This TrajectoryShape is {self}.'
      )


def nodal_surface_mean(
    x: typing.Array, coords: coordinate_systems.CoordinateSystem
) -> typing.Array:

  if x.shape[-2:] != coords.horizontal.nodal_shape[-2:]:
    raise ValueError(f'Input to nodal_surface_mean: {x.shape=}, while expected '
                     f'spatial shape is {coords.horizontal.nodal_shape=}.')
  surface_area = 4 * jnp.pi * coords.horizontal.radius**2

  return coords.horizontal.integrate(x) / surface_area


def modal_surface_mean(
    x: typing.Array, coords: coordinate_systems.CoordinateSystem
) -> typing.Array:

  if x.shape[-2:] != coords.horizontal.modal_shape[-2:]:
    raise ValueError(f'Input to modal_surface_mean: {x.shape=}, while expected '
                     f'modal shape is {coords.horizontal.modal_shape=}.')


  surface_area = 4 * jnp.pi * coords.horizontal.radius**2


  return jnp.sum(x, axis=SPATIAL_AXES) / surface_area


def extract_time_slice(trajectory: Pytree, time_slice: slice) -> Pytree:
  return pytree_utils.slice_along_axis(trajectory, TIME_AXIS, time_slice)


def extract_time_step(trajectory: Pytree, time_step: int) -> Pytree:
  return extract_time_slice(trajectory, slice(time_step, time_step + 1))


def extract_vertical_slice(
    trajectory: Pytree,
    coords: coordinate_systems.CoordinateSystem,
    level: int,
) -> Pytree:
  i = coords.vertical.centers.tolist().index(level)
  index = slice(i, i + 1)
  trajectory = pytree_utils.slice_along_axis(trajectory, LEVEL_AXIS, index)
  return trajectory


def filter_sim_time(trajectory: Pytree) -> Pytree:
  if isinstance(trajectory, dict):
    trajectory = dict(trajectory)
    trajectory.pop('sim_time', None)
  return trajectory


def filter_sim_time_and_diagnostics(trajectory: Pytree) -> Pytree:
  if isinstance(trajectory, dict):
    trajectory = dict(trajectory)
    trajectory.pop('sim_time', None)
    trajectory.pop('diagnostics', None)
  return trajectory


def extract_variable(
    trajectory: TrajectoryRepresentations,
    trajectory_spec: TrajectorySpec,
    time_step: int | slice | None = None,
    level: int | None = None,
    getter: Callable[[Pytree], Pytree] = filter_sim_time,
    is_nodal: bool = True,
    is_encoded: bool = False,
) -> Pytree:

  if is_encoded:
    coords = trajectory_spec.coords
  else:
    coords = trajectory_spec.data_coords
  trajectory = trajectory.get_representation(
      is_nodal=is_nodal, is_encoded=is_encoded
  )
  trajectory = getter(trajectory)
  if time_step is not None:
    if isinstance(time_step, slice):
      trajectory = extract_time_slice(trajectory, time_step)
    else:
      trajectory = extract_time_step(trajectory, time_step)
  if level is not None:
    trajectory = extract_vertical_slice(trajectory, coords, level)
  return trajectory


def replace_with_linear_trucation(
    trajectory_spec: TrajectorySpec,
) -> TrajectorySpec:

  grid = trajectory_spec.data_coords.horizontal
  max_wavenumber = grid.longitude_wavenumbers - 1
  assert max_wavenumber + 2 == grid.total_wavenumbers
  gaussian_nodes = grid.longitude_nodes // 4
  assert gaussian_nodes == grid.latitude_nodes // 2


  new_horizontal = spherical_harmonic.Grid.construct(
      max_wavenumber=2 * gaussian_nodes - 1,
      gaussian_nodes=gaussian_nodes,
      latitude_spacing=grid.latitude_spacing,
      radius=grid.radius,
  )


  return dataclasses.replace(
      trajectory_spec,
      data_coords=dataclasses.replace(
          trajectory_spec.data_coords,
          horizontal=new_horizontal,
      ),
  )


def trajectory_4d_shape(
    trajectory_spec: TrajectorySpec,
    keep_levels: Optional[Sequence[float]] = None,
) -> TrajectoryShape:

  if keep_levels is None:
    n_levels = trajectory_spec.data_coords.vertical.layers
  else:
    n_levels = sum(bool(i) for i in keep_levels)
    if n_levels > trajectory_spec.data_coords.vertical.layers:
      raise ValueError(
          f'{n_levels=} implied by `keep_levels` was greater than '
          f'{trajectory_spec.data_coords.vertical.layers=}'
      )
  grid = trajectory_spec.data_coords.horizontal
  n_m, n_l = grid.modal_shape
  return TrajectoryShape(
      n_times=trajectory_spec.trajectory_length,
      n_levels=n_levels,
      n_longitudinal_wavenumbers=n_m,
      n_total_wavenumbers=n_l,
      n_longitude_nodes=grid.longitude_nodes,
      n_latitude_nodes=grid.latitude_nodes,
  )


def pmean_all_axes(x: jax.Array) -> jax.Array:

  return _pmean_all_axes_p.bind(x)


def _pmean_all_axes_impl(x):
  return x


def _pmean_all_axes_batch(args, batch_axes):
  (x,) = args
  (batch_axis,) = batch_axes
  y = jnp.broadcast_to(x.mean(axis=batch_axes, keepdims=True), x.shape)
  return _pmean_all_axes_p.bind(y), batch_axis


_pmean_all_axes_p = jax.core.Primitive('pmean_all_axes')
_pmean_all_axes_p.def_impl(_pmean_all_axes_impl)
_pmean_all_axes_p.def_abstract_eval(_pmean_all_axes_impl)
batching.primitive_batchers[_pmean_all_axes_p] = _pmean_all_axes_batch
ad.deflinear(_pmean_all_axes_p, lambda cotangent: [pmean_all_axes(cotangent)])
mlir.register_lowering(
    _pmean_all_axes_p,
    mlir.lower_fun(_pmean_all_axes_impl, multiple_results=False),
)


@dataclasses.dataclass
class AggregationTransform:


  trajectory_spec: TrajectorySpec
  out_trajectory_spec: TrajectorySpec
  is_nodal: bool
  is_encoded: bool

  def __call__(self, inputs: Pytree) -> Pytree:
    raise NotImplementedError


AggregationTransformConstructor = Callable[..., AggregationTransform]


@gin.register
class AggregateIdentity(AggregationTransform):

  def __init__(
      self,
      trajectory_spec: TrajectorySpec,
      is_nodal: bool,
      is_encoded: bool,
  ):
    super().__init__(trajectory_spec, trajectory_spec, is_nodal, is_encoded)

  def __call__(self, inputs: Pytree) -> Pytree:
    return inputs


@gin.register
class SumVariables(AggregationTransform):


  def __init__(
      self,
      trajectory_spec: TrajectorySpec,
      is_nodal: bool,
      is_encoded: bool,
  ):
    super().__init__(trajectory_spec, trajectory_spec, is_nodal, is_encoded)

  def __call__(self, inputs: Pytree) -> Pytree:
    return sum(jax.tree_util.tree_leaves(inputs))


@gin.register
class RegriddingAggregation(AggregationTransform):


  def __init__(
      self,
      trajectory_spec: TrajectorySpec,
      is_nodal: bool,
      is_encoded: bool,
      target_grid: coordinate_systems.CoordinateSystem,
      scale_by_area: bool = False,
  ):
    if not is_nodal:
      raise ValueError('AggregateHorizontal is only supported on nodal data')
    if is_encoded:
      source_coords = trajectory_spec.coords
      coords = dataclasses.replace(source_coords, horizontal=target_grid)
      out_trajectory_spec = dataclasses.replace(trajectory_spec, coords=coords)
    else:
      source_coords = trajectory_spec.data_coords
      coords = dataclasses.replace(source_coords, horizontal=target_grid)
      out_trajectory_spec = dataclasses.replace(
          trajectory_spec, data_coords=coords)
    super().__init__(trajectory_spec, out_trajectory_spec, is_nodal, is_encoded)
    self.regrid_fn = horizontal_interpolation.ConservativeRegridder(
        source_coords.horizontal, coords.horizontal)


    lower_lon_boundaries = horizontal_interpolation._periodic_lower_bounds(
        coords.horizontal.longitudes, 2 * np.pi)
    upper_lon_boundaries = horizontal_interpolation._periodic_upper_bounds(
        coords.horizontal.longitudes, 2 * np.pi)
    lat_boundaries = horizontal_interpolation._latitude_cell_bounds(
        coords.horizontal.latitudes)
    lon_weights = upper_lon_boundaries - lower_lon_boundaries
    lat_weights = jnp.sin(lat_boundaries[1:]) - jnp.sin(lat_boundaries[:-1])
    self.weights = lat_weights[np.newaxis, :] * lon_weights[:, np.newaxis]
    self.scale_by_area = scale_by_area

  def __call__(self, inputs: Pytree) -> Pytree:
    if self.scale_by_area:
      return tree_map(lambda x: self.regrid_fn(x) * self.weights, inputs)
    else:
      return tree_map(self.regrid_fn, inputs)


@gin.register
class TimeWindowSum(AggregationTransform):


  def __init__(
      self,
      trajectory_spec: TrajectorySpec,
      is_nodal: bool,
      is_encoded: bool,
      time_window_size: int,
  ):
    trajectory_length = trajectory_spec.trajectory_length
    if trajectory_length % time_window_size != 0:
      raise ValueError(f'Cannot aggregate {trajectory_length=} '
                       f'into {time_window_size=} sections.')
    new_length = trajectory_spec.trajectory_length // time_window_size
    out_trajectory_spec = dataclasses.replace(
        trajectory_spec,
        trajectory_length=new_length,
        steps_per_save=trajectory_spec.steps_per_save * time_window_size)
    super().__init__(trajectory_spec, out_trajectory_spec, is_nodal, is_encoded)
    eye = np.eye(trajectory_length)


    self.time_axis_weights = sum(
        [np.roll(eye, i, 0) for i in range(time_window_size)]
    )[:, ::time_window_size]

  def __call__(self, inputs: Pytree) -> Pytree:
    def _aggregate_time(x: jax.Array):
      return jnp.einsum(
          'tk,...thml->...khml', self.time_axis_weights, x, precision='float32')
    return tree_map(_aggregate_time, inputs)
