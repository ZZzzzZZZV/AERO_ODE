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
import dataclasses
import functools
import re
from typing import Any, Callable, Dict, Mapping, Optional, Sequence, Tuple
from dinosaur import coordinate_systems
from dinosaur import pytree_utils
from dinosaur import sigma_coordinates
from dinosaur import typing

import gin
import haiku as hk
import jax
import jax.numpy as jnp
from neuralgcm import filters
import numpy as np
import tensorflow_probability.substrates.jax as tfp


KeyWithCosLatFactor = typing.KeyWithCosLatFactor
TransformModule = typing.TransformModule


@gin.register
class EmptyTransform(hk.Module):


  def __init__(self, *args, name: Optional[str] = None):
    del args
    super().__init__(name=name)

  def __call__(self, inputs: ...) -> typing.Pytree:
    return {}


@gin.register
class IdentityTransform(hk.Module):


  def __init__(self, *args, name: Optional[str] = None, **kwargs):
    del args, kwargs
    super().__init__(name=name)

  def __call__(self, inputs: typing.PyTreeState) -> typing.PyTreeState:
    return inputs


@gin.register
class ShiftAndNormalize(hk.Module):


  def __init__(
      self,
      coords: coordinate_systems.CoordinateSystem,
      dt: float,
      physics_specs: Any,
      aux_features: typing.AuxFeatures,
      shifts: typing.Pytree,
      scales: typing.Pytree,
      features_to_exclude: Sequence[str] = tuple(),
      global_scale: Optional[float] = None,
      name: Optional[str] = None,
  ):
    del coords, dt, physics_specs, aux_features
    super().__init__(name=name)
    self.shifts = shifts
    if global_scale is not None:
      scales = jax.tree_util.tree_map(lambda x: x * global_scale, scales)
    self.scales = scales

  def __call__(self, inputs):
    inputs, from_dict_fn = pytree_utils.as_dict(inputs)
    shifts = pytree_utils.replace_with_matching_or_default(
        inputs, self.shifts, default=None, check_used_all_replace_keys=False)
    scales = pytree_utils.replace_with_matching_or_default(
        inputs, self.scales, default=None, check_used_all_replace_keys=False)


    result = jax.tree_util.tree_map(
        lambda x, y, z: (x - y) / z, inputs, shifts, scales)
    return from_dict_fn(result)


@gin.register
class InverseShiftAndNormalize(hk.Module):


  def __init__(
      self,
      coords: coordinate_systems.CoordinateSystem,
      dt: float,
      physics_specs: Any,
      aux_features: typing.AuxFeatures,
      shifts: typing.Pytree,
      scales: typing.Pytree,
      global_scale: Optional[float] = None,
      name: Optional[str] = None,
  ):
    del coords, dt, physics_specs, aux_features
    super().__init__(name=name)
    self.shifts = shifts
    if global_scale is not None:
      scales = jax.tree_util.tree_map(lambda x: x * global_scale, scales)
    self.scales = scales

  def __call__(self, inputs):
    inputs, from_dict_fn = pytree_utils.as_dict(inputs)
    shifts = pytree_utils.replace_with_matching_or_default(
        inputs, self.shifts, default=None, check_used_all_replace_keys=False)
    scales = pytree_utils.replace_with_matching_or_default(
        inputs, self.scales, default=None, check_used_all_replace_keys=False)


    result = jax.tree_util.tree_map(
        lambda x, y, z: (None if x is None else x * z + y),
        inputs,
        shifts,
        scales,
        is_leaf=lambda x: x is None,
    )
    return from_dict_fn(result)


@gin.register
class ToModalWithDivCurlTransform(hk.Module):


  def __init__(
      self,
      coords: coordinate_systems.CoordinateSystem,
      dt: float,
      physics_specs: Any,
      aux_features: typing.AuxFeatures,
      name: Optional[str] = None,
  ):
    del dt, physics_specs, aux_features
    super().__init__(name=name)
    self.coords = coords

  def __call__(self, inputs: typing.Pytree) -> typing.Pytree:
    if 'u' not in inputs or 'v' not in inputs:
      raise ValueError('Inputs to ToModalWithDivCurlTransform must include `u, '
                       f'v`, got keys: {inputs.keys()}')
    sec_lat = 1 / self.coords.horizontal.cos_lat
    u, v = inputs.pop('u'), inputs.pop('v')


    inputs['u'] = u * sec_lat
    inputs['v'] = v * sec_lat
    to_modal_fn = lambda x: (self.coords.horizontal.to_modal(x)
                             if x is not None else None)
    modal_outputs = jax.tree_util.tree_map(to_modal_fn, inputs)
    u, v = modal_outputs.pop('u'), modal_outputs.pop('v')
    modal_outputs['divergence'] = self.coords.horizontal.div_cos_lat((u, v))
    modal_outputs['vorticity'] = self.coords.horizontal.curl_cos_lat((u, v))
    return modal_outputs


@gin.register
class ToModalDiffOperators(hk.Module):


  def __init__(
      self,
      coords: coordinate_systems.CoordinateSystem,
      dt: float,
      physics_specs: Any,
      aux_features: typing.AuxFeatures,
      name: Optional[str] = None,
  ):
    del dt, physics_specs, aux_features
    super().__init__(name=name)
    self.coords = coords

  def __call__(
      self,
      inputs: Mapping[typing.KeyWithCosLatFactor, typing.Array],
  ) -> Mapping[typing.KeyWithCosLatFactor, typing.Array]:
    features = {}
    for k, value in inputs.items():
      name, cos_lat_order = k.name, k.factor_order
      d_value_dlon, d_value_dlat = self.coords.horizontal.cos_lat_grad(value)
      laplacian_value = self.coords.horizontal.laplacian(value)
      dlon_key = typing.KeyWithCosLatFactor(name + '_dlon', cos_lat_order + 1)
      dlat_key = typing.KeyWithCosLatFactor(name + '_dlat', cos_lat_order + 1)
      del2_key = typing.KeyWithCosLatFactor(name + '_del2', cos_lat_order)
      features[dlon_key] = d_value_dlon
      features[dlat_key] = d_value_dlat
      features[del2_key] = laplacian_value
    return features


@gin.register
class ModalToNodalTransform(hk.Module):


  def __init__(
      self,
      coords: coordinate_systems.CoordinateSystem,
      dt: float,
      physics_specs: Any,
      aux_features: typing.AuxFeatures,
      name: Optional[str] = None,
  ):
    del dt, physics_specs, aux_features
    super().__init__(name=name)
    self.coords = coords

  def __call__(self, inputs: typing.PyTreeState) -> typing.PyTreeState:
    return self.coords.horizontal.to_nodal(inputs)


@gin.register
class NodalToModalTransform(hk.Module):


  def __init__(
      self,
      coords: coordinate_systems.CoordinateSystem,
      dt: float,
      physics_specs: Any,
      aux_features: typing.AuxFeatures,
      name: Optional[str] = None,
  ):
    del dt, physics_specs, aux_features
    super().__init__(name=name)
    self.coords = coords

  def __call__(self, inputs: typing.PyTreeState) -> typing.PyTreeState:
    return self.coords.horizontal.to_modal(inputs)


@gin.register
class ClipTransform(hk.Module):


  def __init__(
      self,
      coords: coordinate_systems.CoordinateSystem,
      dt: float,
      physics_specs: Any,
      aux_features: typing.AuxFeatures,
      wavenumbers_to_clip: int = 1,
      name: Optional[str] = None,
  ):

    del dt, physics_specs, aux_features
    super().__init__(name=name)
    self.coords = coords
    self.wavenumbers_to_clip = wavenumbers_to_clip

  def __call__(self, inputs: typing.PyTreeState) -> typing.PyTreeState:
    return self.coords.horizontal.clip_wavenumbers(
        inputs, self.wavenumbers_to_clip
    )


@gin.register
class SinhArcsinhTransform(hk.Module):


  def __init__(
      self,
      coords: coordinate_systems.CoordinateSystem,
      dt: float,
      physics_specs: Any,
      aux_features: typing.AuxFeatures,
      tailweight: float = 1.0,
      name: Optional[str] = None,
  ):
    del coords, dt, physics_specs, aux_features
    super().__init__(name=name)
    self.tailweight = tailweight

  def __call__(self, inputs: typing.PyTreeState) -> typing.PyTreeState:
    f = tfp.bijectors.SinhArcsinh(tailweight=self.tailweight).forward
    return jax.tree_util.tree_map(f, inputs)


@gin.register
class NondimensionalizeTransform(hk.Module):


  def __init__(
      self,
      coords: coordinate_systems.CoordinateSystem,
      dt: float,
      physics_specs: Any,
      aux_features: typing.AuxFeatures,
      input_coords: coordinate_systems.CoordinateSystem,
      inputs_to_units_mapping: Dict[str, str],
      name: Optional[str] = None,
  ):

    del coords, dt, aux_features, input_coords
    super().__init__(name=name)
    self.inputs_to_units_mapping = inputs_to_units_mapping
    self.nondimensionalize = physics_specs.nondimensionalize

  def __call__(self, inputs: typing.PyTreeState) -> typing.PyTreeState:
    inputs, from_dict_fn = pytree_utils.as_dict(inputs)
    inputs_to_units_mapping = pytree_utils.replace_with_matching_or_default(
        inputs, self.inputs_to_units_mapping, default=None,
        check_used_all_replace_keys=False,
    )
    nondim_fn = lambda x, y: self.nondimensionalize(x * typing.Quantity(y))
    result = jax.tree_util.tree_map(nondim_fn, inputs, inputs_to_units_mapping)
    return from_dict_fn(result)


@gin.register
class RedimensionalizeTransform(hk.Module):


  def __init__(
      self,
      coords: coordinate_systems.CoordinateSystem,
      dt: float,
      physics_specs: Any,
      aux_features: typing.AuxFeatures,
      output_coords: coordinate_systems.CoordinateSystem,
      inputs_to_units_mapping: Dict[str, str],
      name: Optional[str] = None,
  ):

    del coords, dt, aux_features, output_coords
    super().__init__(name=name)
    self.inputs_to_units_mapping = inputs_to_units_mapping
    self.dimensionalize = physics_specs.dimensionalize

  def __call__(self, inputs: typing.PyTreeState) -> typing.PyTreeState:
    inputs, from_dict_fn = pytree_utils.as_dict(inputs)


    physics_core_inputs = {}
    regular_inputs = {}
    for key, value in inputs.items():
      if key.startswith('_physics_'):
        physics_core_inputs[key] = value
      else:
        regular_inputs[key] = value


    inputs_to_units_mapping = pytree_utils.replace_with_matching_or_default(
        regular_inputs, self.inputs_to_units_mapping, default=None,
        check_used_all_replace_keys=False,
    )
    dim_fn = lambda x, y: self.dimensionalize(x, typing.Quantity(y)).m if y is not None else x
    result = jax.tree_util.tree_map(dim_fn, regular_inputs, inputs_to_units_mapping)


    for key, value in physics_core_inputs.items():

      original_key = key[len('_physics_'):]
      if original_key in self.inputs_to_units_mapping:
        unit = self.inputs_to_units_mapping[original_key]

        if isinstance(unit, dict):

          result[key] = jax.tree_util.tree_map(
              lambda v, u: self.dimensionalize(v, typing.Quantity(u)).m if u is not None else v,
              value, unit
          )
        else:

          result[key] = self.dimensionalize(value, typing.Quantity(unit)).m
      else:

        result[key] = value

    return from_dict_fn(result)


@gin.register
class SequentialTransform(hk.Module):


  def __init__(
      self,
      coords: coordinate_systems.CoordinateSystem,
      dt: float,
      physics_specs: Any,
      aux_features: typing.AuxFeatures,
      transform_modules: Sequence[TransformModule],
      name: Optional[str] = None,
  ):
    super().__init__(name=name)
    self.transform_fns = [module(coords, dt, physics_specs, aux_features)
                          for module in transform_modules]

  def __call__(self, inputs: typing.PyTreeState) -> typing.PyTreeState:
    for transform_fn in self.transform_fns:
      inputs = transform_fn(inputs)
    return inputs


@gin.register
class LevelScale(hk.Module):


  def __init__(
      self,
      coords: coordinate_systems.CoordinateSystem,
      dt: float,
      physics_specs: Any,
      aux_features: typing.AuxFeatures,
      scales: Sequence[float],
      keys_to_scale: Sequence[str] = tuple(),
      name: Optional[str] = None,
  ):
    del coords, dt, physics_specs, aux_features
    super().__init__(name=name)
    self.scale_fn = functools.partial(
        coordinate_systems.scale_levels_for_matching_keys,
        scales=np.asarray(scales),
        keys_to_scale=keys_to_scale)

  def __call__(self, inputs: typing.Pytree) -> typing.Pytree:
    return self.scale_fn(inputs)


@gin.register
class InverseLevelScale(hk.Module):


  def __init__(
      self,
      coords: coordinate_systems.CoordinateSystem,
      dt: float,
      physics_specs: Any,
      aux_features: typing.AuxFeatures,
      scales: Sequence[float],
      keys_to_scale: Sequence[str] = tuple(),
      name: Optional[str] = None,
  ):
    del coords, dt, physics_specs, aux_features
    super().__init__(name=name)
    self.scale_fn = functools.partial(
        coordinate_systems.scale_levels_for_matching_keys,
        scales=1/np.asarray(scales),
        keys_to_scale=keys_to_scale)

  def __call__(self, inputs: typing.Pytree) -> typing.Pytree:
    return self.scale_fn(inputs)


@gin.register
class HardClip(hk.Module):


  def __init__(
      self,
      coords: coordinate_systems.CoordinateSystem,
      dt: float,
      physics_specs: Any,
      aux_features: typing.AuxFeatures,
      max_value: float,
      name: Optional[str] = None,
  ):
    del coords, dt, physics_specs, aux_features
    super().__init__(name=name)
    self.clip_fn = functools.partial(
        jnp.clip, min=-max_value, max=max_value)

  def __call__(self, inputs: typing.Pytree) -> typing.Pytree:
    return jax.tree_util.tree_map(self.clip_fn, inputs)


@gin.register
class SoftClip(hk.Module):


  def __init__(
      self,
      coords: coordinate_systems.CoordinateSystem,
      dt: float,
      physics_specs: Any,
      aux_features: typing.AuxFeatures,
      max_value: float,
      hinge_softness: float = 1.0,
      name: Optional[str] = None,
  ):
    del coords, dt, physics_specs, aux_features
    if max_value < 0 or hinge_softness < 0:
      raise ValueError('max_value and hinge_softness must be positive, '
                       f'{max_value=}, {hinge_softness=}')
    super().__init__(name=name)
    low = -max_value
    high = max_value
    hinge = hinge_softness
    softplus_fn = lambda x: hinge * jax.nn.softplus(x / hinge)
    self.clip_fn = lambda x: (
        -softplus_fn(high - low - softplus_fn(x - low)) *
        (high - low) / (softplus_fn(high - low)) + high)

  def __call__(self, inputs: typing.Pytree) -> typing.Pytree:
    return jax.tree_util.tree_map(self.clip_fn, inputs)


@gin.register
class ToModalDiffOperatorsWithFiltering(hk.Module):


  def __init__(
      self,
      coords: coordinate_systems.CoordinateSystem,
      dt: float,
      physics_specs: Any,
      aux_features: typing.AuxFeatures,
      filter_attenuations: Tuple[float, ...] = tuple(),
      name: Optional[str] = None,
  ):
    super().__init__(name=name)
    self.coords = coords
    self.attenuations = filter_attenuations
    feature_filters = []
    for attenuation in filter_attenuations:
      feature_filters.append(
          filters.DataExponentialFilter(
              coords, dt, physics_specs, aux_features,
              order=1, attenuation=attenuation))
    self.feature_filters = feature_filters

  def __call__(
      self,
      inputs: Mapping[KeyWithCosLatFactor, typing.Array],
  ) -> Mapping[KeyWithCosLatFactor, typing.Array]:
    features = {}
    for k, value in inputs.items():
      name, cos_lat_order = k.name, k.factor_order
      for filter_fn, att in zip(self.feature_filters, self.attenuations):
        filtered_value = filter_fn(value)
        d_value_dlon, d_value_dlat = self.coords.horizontal.cos_lat_grad(
            filtered_value)
        laplacian_value = self.coords.horizontal.laplacian(filtered_value)


        dlon_key = KeyWithCosLatFactor(
            name + f'_dlon_{att}', cos_lat_order + 1, att)
        dlat_key = KeyWithCosLatFactor(
            name + f'_dlat_{att}', cos_lat_order + 1, att)
        del2_key = KeyWithCosLatFactor(
            name + f'_del2_{att}', cos_lat_order, att)
        features[dlon_key] = d_value_dlon
        features[dlat_key] = d_value_dlat
        features[del2_key] = laplacian_value
    return features


@gin.register
class TruncateSigmaLevels(hk.Module):


  def __init__(
      self,
      coords: coordinate_systems.CoordinateSystem,
      dt: float,
      physics_specs: Any,
      aux_features: typing.AuxFeatures,
      sigma_ranges: dict[str, Tuple[float, float]],
      name: Optional[str] = None,
  ):
    super().__init__(name=name)
    del dt, physics_specs, aux_features
    self.sigma_ranges = sigma_ranges
    self.sigma_levels = coords.vertical.centers

  def __call__(self, inputs: typing.PyTreeState) -> typing.PyTreeState:


    def _slice_fn(x, sigma_range):

      sigma_min_slice, sigma_max_slice = sigma_range
      lower_index = np.argmax((self.sigma_levels - sigma_min_slice) > 0)
      if sigma_max_slice > np.max(self.sigma_levels):
        upper_index = len(self.sigma_levels)
      else:
        upper_index = np.argmin((self.sigma_levels - sigma_max_slice) < 0)
      return x[slice(lower_index, upper_index), ...]

    def recurse_and_replace(x: dict[str, Any],
                            y: dict[str, Any],
                            default=None) -> dict[str, Any]:

      return {
          k: (
              y.get(k, default)
              if not isinstance(v, dict)
              else recurse_and_replace(v, y, default)
          )
          for k, v in x.items()
      }

    inputs, from_dict_fn = pytree_utils.as_dict(inputs)
    sigma_ranges_extended = recurse_and_replace(
        inputs, self.sigma_ranges, default=(0, 1)
    )
    outputs = jax.tree_util.tree_map(_slice_fn, inputs, sigma_ranges_extended)
    return from_dict_fn(outputs)


@gin.register
class TakeSurfaceAdjacentSigmaLevel(hk.Module):


  def __init__(
      self,
      coords: coordinate_systems.CoordinateSystem,
      dt: float,
      physics_specs: Any,
      aux_features: typing.AuxFeatures,
      name: Optional[str] = None,
  ):
    super().__init__(name=name)
    del coords, dt, physics_specs, aux_features

  def __call__(self, inputs: typing.PyTreeState) -> typing.PyTreeState:


    def _slice_fn(x):
      return x[slice(-1, None), ...]

    inputs, from_dict_fn = pytree_utils.as_dict(inputs)
    outputs = jax.tree_util.tree_map(_slice_fn, inputs)
    return from_dict_fn(outputs)


@gin.register
@dataclasses.dataclass
class FeatureSelector:

  regex_patterns: str

  def __call__(
      self,
      inputs: Dict[str, typing.Array],
  ) -> Dict[str, typing.Array]:
    outputs = {}
    for k, v in inputs.items():
      if re.fullmatch(self.regex_patterns, k):
        outputs[k] = v
    return outputs


@gin.register
class BroadcastTransform:


  def __init__(self, *args, **kwargs):
    del args, kwargs

  def __call__(self, inputs: typing.Pytree) -> typing.Pytree:
    leaves, tree_def = jax.tree_util.tree_flatten(inputs)
    leaves = jnp.broadcast_arrays(*leaves)
    return jax.tree_util.tree_unflatten(tree_def, leaves)


@gin.register
class SquashLevelsTransform:


  def __init__(
      self,
      coords: coordinate_systems.CoordinateSystem,
      dt: float,
      physics_specs: Any,
      aux_features: typing.AuxFeatures,
      low_cutoffs: Sequence[float] = (0.05, 0.1),
      high_cutoffs: Sequence[float] = (0.86, 0.965),
  ):
    del dt, physics_specs, aux_features
    if not isinstance(coords.vertical, sigma_coordinates.SigmaCoordinates):
      raise ValueError(f'Cannot apply sigma_squash on {coords.vertical=}')
    sigma = coords.vertical.centers
    if len(low_cutoffs) != 2:
      raise ValueError(f'{len(low_cutoffs)=} but should have been 2.')
    if len(high_cutoffs) != 2:
      raise ValueError(f'{len(high_cutoffs)=} but should have been 2.')

    low_func = (sigma - low_cutoffs[0]) / (low_cutoffs[1] - low_cutoffs[0])
    high_func = (high_cutoffs[1] - sigma) / (high_cutoffs[1] - high_cutoffs[0])


    lower_bound = np.minimum(1., np.minimum(low_func, high_func))
    self._sigma_squash = np.maximum(0., lower_bound)[:, np.newaxis, np.newaxis]

  def __call__(self, inputs: typing.Pytree) -> typing.Pytree:
    def squash_per_level_only(x):
      shape = jnp.shape(x)
      ndim = len(shape)
      if ndim >= 3 and shape[-3] > 1:
        return x * self._sigma_squash
      elif ndim in {2, 3}:
        return x * self._sigma_squash[-1]
      else:
        return x
    return jax.tree_util.tree_map(squash_per_level_only, inputs)


@gin.register
def add_prefix(features: dict[str, Any], prefix: str) -> dict[str, Any]:

  return {prefix + k: v for k, v in features.items()}


def straight_through(
    f: Callable[[typing.Array], typing.Array],
) -> Callable[[typing.Array], typing.Array]:

  def straight_through_f(x):
    zero = x - jax.lax.stop_gradient(x)
    return zero + jax.lax.stop_gradient(f(x))
  return straight_through_f
