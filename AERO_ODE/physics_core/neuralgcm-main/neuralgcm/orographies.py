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
from typing import Any, Callable, Mapping, Optional, Sequence
from dinosaur import coordinate_systems
from dinosaur import primitive_equations
from dinosaur import scales
from dinosaur import typing
from dinosaur import xarray_utils
import gin
import haiku as hk
import jax.numpy as jnp
import numpy as np


units = scales.units
OrographyModule = Callable[..., typing.Array]
FilterModule = Callable[..., typing.PyTreeFilterFn]


@gin.register
class ClippedOrography(hk.Module):


  def __init__(
      self,
      coords: coordinate_systems.CoordinateSystem,
      dt: float,
      physics_specs: Any,
      aux_features: typing.AuxFeatures,
      wavenumbers_to_clip: int = 1,
      name: Optional[str] = None,
  ):
    del dt, physics_specs
    super().__init__(name=name)
    self.coords = coords
    self.wavenumbers_to_clip = wavenumbers_to_clip
    self.nodal_orography = aux_features.get(
        xarray_utils.OROGRAPHY, np.zeros(coords.horizontal.nodal_shape))

  def __call__(self) -> typing.Array:

    return primitive_equations.truncated_modal_orography(
        self.nodal_orography, self.coords, self.wavenumbers_to_clip)


@gin.register
class FilteredCustomOrography(hk.Module):


  def __init__(
      self,
      coords: coordinate_systems.CoordinateSystem,
      dt: float,
      physics_specs: Any,
      aux_features: typing.AuxFeatures,
      orography_data_path: str,
      filter_modules: Sequence[FilterModule] = tuple(),
      renaming_dict: Optional[Mapping[str, str]] = None,
      name: Optional[str] = None,
  ):
    super().__init__(name=name)
    ds = xarray_utils.ds_from_path_or_aux(orography_data_path, aux_features)
    if renaming_dict is not None:
      ds = ds.rename(renaming_dict)
    nodal_orography = xarray_utils.nodal_orography_from_ds(ds)

    self.nodal_orography = physics_specs.nondimensionalize(
        nodal_orography * units.meter)
    self.coords = coords


    self.input_coords = xarray_utils.coordinate_system_from_dataset(
        ds, truncation=xarray_utils.LINEAR, spmd_mesh=coords.spmd_mesh,
        spherical_harmonics_impl=self.coords.horizontal.spherical_harmonics_impl
    )
    self.filter_fns = [
        module(coords, dt, physics_specs, aux_features)
        for module in filter_modules]

  def __call__(self) -> typing.Array:

    return primitive_equations.filtered_modal_orography(
        self.nodal_orography, self.coords, self.input_coords, self.filter_fns)


@gin.register
class LearnedOrography(hk.Module):


  def __init__(
      self,
      coords: coordinate_systems.CoordinateSystem,
      dt: float,
      physics_specs: Any,
      aux_features: typing.AuxFeatures,
      base_orography_module: OrographyModule,
      correction_scale: float,
      name: Optional[str] = None,
  ):
    super().__init__(name=name)
    self.coords = coords
    self.base_orography_fn = base_orography_module(
        coords, dt, physics_specs, aux_features)
    self.scale = correction_scale


    self.correction = hk.get_parameter(
        'orography', (coords.horizontal.mask.sum(),), jnp.float32,
        init=hk.initializers.Constant(0.0))

  def __call__(self) -> typing.Array:

    mask = self.coords.horizontal.mask
    correction_2d = jnp.zeros(self.coords.horizontal.modal_shape)
    correction_2d = correction_2d.at[mask].set(self.correction)
    return self.base_orography_fn() + correction_2d * self.scale
