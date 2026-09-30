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

import collections
import dataclasses
import functools
from typing import Any, Callable, Dict, Optional, Sequence, Union
from dinosaur import coordinate_systems
from dinosaur import layer_coordinates
from dinosaur import scales
from dinosaur import sigma_coordinates
from dinosaur import spherical_harmonic
from dinosaur import time_integration
from dinosaur import typing
from dinosaur import xarray_utils
import gin
import haiku as hk
import jax.numpy as jnp

from neuralgcm import correctors
from neuralgcm import decoders
from neuralgcm import embeddings
from neuralgcm import encoders
from neuralgcm import equations
from neuralgcm import features
from neuralgcm import filters
from neuralgcm import forcings
from neuralgcm import gin_utils
from neuralgcm import layers
from neuralgcm import mappings
from neuralgcm import model_utils
from neuralgcm import physics_specifications
from neuralgcm import steps
from neuralgcm import stochastic
from neuralgcm import towers
from neuralgcm import transforms
import numpy as np
import xarray


DEFAULT_REFERENCE_TEMPERATURE = 288
DEFAULT_REFERENCE_DATETIME_STR = '1979-01-01T00'

Array = typing.Array
AuxFeatures = typing.AuxFeatures
DataState = typing.DataState
PyTreeState = typing.PyTreeState
ModelState = typing.ModelState
ForcingData = typing.ForcingData
Forcing = typing.Forcing
Numeric = typing.Numeric
QuantityOrStr = Union[str, scales.Quantity]


data_to_xarray = gin.external_configurable(
    xarray_utils.data_to_xarray, 'data_to_xarray'
)

primitive_eq_to_xarray = gin.external_configurable(
    xarray_utils.data_to_xarray, 'primitive_eq_to_xarray'
)
data_to_xarray_with_renaming = gin.external_configurable(
    xarray_utils.data_to_xarray_with_renaming, 'data_to_xarray_with_renaming'
)
dynamic_covariate_data_to_xarray = gin.external_configurable(
    xarray_utils.dynamic_covariate_data_to_xarray,
    'dynamic_covariate_data_to_xarray',
)


xarray_to_shallow_water = gin.external_configurable(
    xarray_utils.xarray_to_shallow_water_eq_data, 'xarray_to_shallow_water'
)
xarray_to_primitive_eq = gin.external_configurable(
    xarray_utils.xarray_to_primitive_eq_data, 'xarray_to_primitive_eq'
)
xarray_to_primitive_eq_with_time = gin.external_configurable(
    xarray_utils.xarray_to_primitive_equations_with_time_data,
    'xarray_to_primitive_eq_with_time',
)
xarray_to_weatherbench_data = gin.external_configurable(
    xarray_utils.xarray_to_weatherbench_data, 'xarray_to_weatherbench_data'
)
xarray_to_data_with_renaming = gin.external_configurable(
    xarray_utils.xarray_to_data_with_renaming, 'xarray_to_data_with_renaming'
)
xarray_to_dynamic_covariate_data = gin.external_configurable(
    xarray_utils.xarray_to_dynamic_covariate_data,
    'xarray_to_dynamic_covariate_data',
)
xarray_to_state_and_dynamic_covariate_data = gin.external_configurable(
    xarray_utils.xarray_to_state_and_dynamic_covariate_data,
    'xarray_to_state_and_dynamic_covariate_data',
)
coordinate_system_from_dataset = gin.external_configurable(
    xarray_utils.coordinate_system_from_dataset,
    'coordinate_system_from_dataset',
    allowlist=['truncation', 'spherical_harmonics_impl'])


Grid = gin.external_configurable(
    spherical_harmonic.Grid, denylist=['spmd_mesh']
)
GridWithWavenumbers = gin.external_configurable(
    spherical_harmonic.Grid.with_wavenumbers, 'GridWithWavenumbers'
)
GridT21 = gin.external_configurable(spherical_harmonic.Grid.T21, 'GridT21')
GridT31 = gin.external_configurable(spherical_harmonic.Grid.T31, 'GridT31')
GridT42 = gin.external_configurable(spherical_harmonic.Grid.T42, 'GridT42')
GridT85 = gin.external_configurable(spherical_harmonic.Grid.T85, 'GridT85')
GridT106 = gin.external_configurable(spherical_harmonic.Grid.T106, 'GridT106')
GridT119 = gin.external_configurable(spherical_harmonic.Grid.T119, 'GridT119')
GridT170 = gin.external_configurable(spherical_harmonic.Grid.T170, 'GridT170')
GridT213 = gin.external_configurable(spherical_harmonic.Grid.T213, 'GridT213')
GridTL31 = gin.external_configurable(spherical_harmonic.Grid.TL31, 'GridTL31')
GridTL63 = gin.external_configurable(spherical_harmonic.Grid.TL63, 'GridTL63')
GridTL95 = gin.external_configurable(spherical_harmonic.Grid.TL95, 'GridTL95')
GridTL127 = gin.external_configurable(
    spherical_harmonic.Grid.TL127, 'GridTL127'
)
GridTL159 = gin.external_configurable(
    spherical_harmonic.Grid.TL159, 'GridTL159'
)
GridTL179 = gin.external_configurable(
    spherical_harmonic.Grid.TL179, 'GridTL179'
)
GridTL255 = gin.external_configurable(
    spherical_harmonic.Grid.TL255, 'GridTL255'
)
RealSphericalHarmonics = gin.external_configurable(
    spherical_harmonic.RealSphericalHarmonics
)
RealSphericalHarmonicsWithZeroImag = gin.external_configurable(
    spherical_harmonic.RealSphericalHarmonicsWithZeroImag,
    denylist=['spmd_mesh'],
)


ComplexSphericalHarmonics = gin.external_configurable(
    spherical_harmonic.ComplexSphericalHarmonics
)
LayerCoordinates = gin.external_configurable(layer_coordinates.LayerCoordinates)
SigmaCoordinates = gin.external_configurable(sigma_coordinates.SigmaCoordinates)
SigmaCoordinatesEquidistant = gin.external_configurable(
    sigma_coordinates.SigmaCoordinates.equidistant,
    'SigmaCoordinatesEquidistant',
)
CoordinateSystem = gin.external_configurable(
    coordinate_systems.CoordinateSystem, denylist=['spmd_mesh']
)


centered_vertical_advection = gin.external_configurable(
    sigma_coordinates.centered_vertical_advection
)
upwind_vertical_advection = gin.external_configurable(
    sigma_coordinates.upwind_vertical_advection
)


@dataclasses.dataclass(frozen=True)
class ModelSpecs(collections.abc.Mapping):


  coords: coordinate_systems.CoordinateSystem
  dt: float
  physics_specs: Any
  aux_features: typing.AuxFeatures

  def __len__(self):
    return len(dataclasses.fields(self))

  def __iter__(self):
    return iter(f.name for f in dataclasses.fields(self))

  def __getitem__(self, key):
    return getattr(self, key)


@gin.configurable(
    allowlist=(
        'model_time_step',
        'custom_coords',
        'reference_temperature',
        'reference_datetime_str',
    )
)
def get_model_specs(
    data_coords: coordinate_systems.CoordinateSystem,
    physics_specs: Any,
    aux_features: typing.AuxFeatures,
    model_time_step: Optional[Union[float, QuantityOrStr]] = None,
    custom_coords: Optional[coordinate_systems.CoordinateSystem] = None,
    reference_temperature: Optional[float | Sequence] = None,
    reference_datetime_str: Optional[str] = None,
) -> ModelSpecs:

  if model_time_step is None:
    raise ValueError('must provide model_time_step or outer_time_step')

  if custom_coords is None:
    coords = data_coords
  else:
    coords = dataclasses.replace(custom_coords, spmd_mesh=data_coords.spmd_mesh)

  if aux_features.get(xarray_utils.REF_TEMP_KEY) is None:
    if reference_temperature is None:
      ones = np.ones(coords.vertical.layers, np.float32)
      ref_temps = DEFAULT_REFERENCE_TEMPERATURE * ones
      aux_features[xarray_utils.REF_TEMP_KEY] = ref_temps
    else:
      ones = np.ones(coords.vertical.layers, np.float32)
      ref_temps = np.asarray(reference_temperature)
      if ref_temps.ndim == 1 and ref_temps.shape[0] != coords.vertical.layers:
        raise ValueError(
            '`ref_temps` must be a scalar or a sequence with '
            f'{coords.vertical.layers=} elements, got {ref_temps.shape=}'
        )
      ref_temps = ref_temps * ones
      aux_features[xarray_utils.REF_TEMP_KEY] = ref_temps
  else:
    if reference_temperature is not None:
      raise ValueError(
          'reference temperature already specified in aux_features'
      )

  if aux_features.get(xarray_utils.REFERENCE_DATETIME_KEY) is None:
    if reference_datetime_str is None:
      reference_datetime = np.datetime64(DEFAULT_REFERENCE_DATETIME_STR)
      aux_features[xarray_utils.REFERENCE_DATETIME_KEY] = reference_datetime
    else:
      reference_datetime = np.datetime64(reference_datetime_str)
      aux_features[xarray_utils.REFERENCE_DATETIME_KEY] = reference_datetime
  else:
    if reference_datetime_str is not None:
      raise ValueError('reference datetime already specified in aux_data')

  if isinstance(model_time_step, (str, scales.Quantity)):
    dt = physics_specs.nondimensionalize(scales.Quantity(model_time_step))
  else:
    dt = model_time_step

  return ModelSpecs(
      coords=coords,
      dt=dt,
      physics_specs=physics_specs,
      aux_features=aux_features,
  )


def _identity(x):
  return x


class DynamicalSystem(hk.Module):


  def __init__(
      self,
      coords: coordinate_systems.CoordinateSystem,
      dt: float,
      physics_specs: Any,
      aux_features: Dict[str, Any],
      input_coords: coordinate_systems.CoordinateSystem,
      output_coords: coordinate_systems.CoordinateSystem,
      name: Optional[str] = None,
  ):
    super().__init__(name=name)
    self.coords = coords
    self.dt = dt
    self.physics_specs = physics_specs
    self.aux_features = aux_features
    self.input_coords = input_coords
    self.output_coords = output_coords

  def encode(self, x: PyTreeState, forcing: Forcing) -> PyTreeState:

    raise NotImplementedError('Model subclass did not define encode')

  def decode(self, x: PyTreeState, forcing: Forcing) -> PyTreeState:

    raise NotImplementedError('Model subclass did not define decode')

  def advance(self, x: PyTreeState, forcing: Forcing) -> PyTreeState:

    raise NotImplementedError('Model subclass did not define advance')

  def forcing_fn(self, forcing_data: ForcingData, sim_time: Numeric) -> Forcing:

    raise NotImplementedError('Model subclass did not define forcing_fn')

  def trajectory(
      self,
      x: ...,
      outer_steps: int,
      inner_steps: int = 1,
      *,
      forcing_data: ForcingData,
      start_with_input: bool = False,
      post_process_fn: Callable = _identity,
  ) -> ...:


    def step_fn(x: PyTreeState) -> PyTreeState:

      if isinstance(x, typing.ModelState):
        sim_time = getattr(x.state, 'sim_time', None)
      else:
        sim_time = getattr(x, 'sim_time', None)
      forcing = self.forcing_fn(forcing_data, sim_time)
      x, forcing = self.coords.with_dycore_sharding((x, forcing))
      y = self.advance(x, forcing)
      y = self.coords.with_dycore_sharding(y)
      return y

    return trajectory_from_step(
        step_fn,
        outer_steps,
        inner_steps,
        start_with_input=start_with_input,
        post_process_fn=post_process_fn,
    )(x)


@gin.configurable
class ModularStepModel(DynamicalSystem):


  def __init__(
      self,
      coords: coordinate_systems.CoordinateSystem,
      dt: float,
      physics_specs: Any,
      aux_features: Dict[str, Any],
      input_coords: coordinate_systems.CoordinateSystem,
      output_coords: coordinate_systems.CoordinateSystem,
      advance_module: ... = gin.REQUIRED,
      encoder_module: ... = gin.REQUIRED,
      decoder_module: ... = gin.REQUIRED,
      forcing_module: ... = forcings.NoForcing,
      name: Optional[str] = None,
  ):
    super().__init__(
        coords,
        dt,
        physics_specs,
        aux_features,
        input_coords,
        output_coords,
        name=name,
    )
    self.advance_fn = advance_module(coords, dt, physics_specs, aux_features)
    self.encoder_fn = encoder_module(
        coords, dt, physics_specs, aux_features, input_coords
    )
    self.decoder_fn = decoder_module(
        coords, dt, physics_specs, aux_features, output_coords
    )
    self.forcing_fn = forcing_module(coords, dt, physics_specs, aux_features)

  def encode(self, x: PyTreeState, forcing: Forcing) -> PyTreeState:
    return self.encoder_fn(x, forcing)

  def decode(self, x: PyTreeState, forcing: Forcing) -> PyTreeState:
    return self.decoder_fn(x, forcing)

  def advance(self, x: PyTreeState, forcing: Forcing) -> PyTreeState:
    return self.advance_fn(x, forcing)


@gin.configurable
class StochasticModularStepModel(DynamicalSystem):


  def __init__(
      self,
      coords: coordinate_systems.CoordinateSystem,
      dt: float,
      physics_specs: Any,
      aux_features: Dict[str, Any],
      input_coords: coordinate_systems.CoordinateSystem,
      output_coords: coordinate_systems.CoordinateSystem,
      advance_module: ... = gin.REQUIRED,
      encoder_module: ... = gin.REQUIRED,
      decoder_module: ... = gin.REQUIRED,
      forcing_module: ... = forcings.NoForcing,
      name: Optional[str] = None,
  ):
    super().__init__(
        coords,
        dt,
        physics_specs,
        aux_features,
        input_coords,
        output_coords,
        name=name,
    )
    self.advance_fn = advance_module(coords, dt, physics_specs, aux_features)
    self.encoder_fn = encoder_module(
        coords, dt, physics_specs, aux_features, input_coords
    )
    self.decoder_fn = decoder_module(
        coords, dt, physics_specs, aux_features, output_coords
    )
    self.forcing_fn = forcing_module(coords, dt, physics_specs, aux_features)

  def encode(
      self,
      x: DataState,
      forcing: Forcing,
  ) -> ModelState:

    model_state = self.encoder_fn(x, forcing=forcing)


    return self.advance_fn.finalize_state(model_state, forcing)

  def decode(self, x: ModelState, forcing: Forcing) -> typing.Pytree:


    return self.decoder_fn(x, forcing=forcing)

  def advance(
      self,
      x: ModelState,
      forcing: Forcing,
  ) -> ModelState:

    return self.advance_fn(x, forcing)


@gin.configurable(
    allowlist=(
        'checkpoint_step',
        'checkpoint_multistep',
        'checkpoint_post_process',
    )
)
def trajectory_from_step(
    step_fn: Callable,
    outer_steps: int,
    inner_steps: int,
    *,
    start_with_input: bool,
    post_process_fn: Callable,
    checkpoint_step: bool = True,
    checkpoint_multistep: bool = False,
    checkpoint_post_process: bool = True,
) -> Callable:

  if checkpoint_step:
    step_fn = hk.remat(step_fn)

  if checkpoint_post_process:
    post_process_fn = hk.remat(post_process_fn)

  if checkpoint_multistep:
    def outer_scan_fn(f, init, xs, length=None):
      return hk.scan(hk.remat(f), init, xs, length=length)
  else:
    outer_scan_fn = hk.scan

  return time_integration.trajectory_from_step(
      step_fn,
      outer_steps,
      inner_steps,
      start_with_input=start_with_input,
      post_process_fn=post_process_fn,
      inner_scan_fn=hk.scan,
      outer_scan_fn=outer_scan_fn,
  )


@gin.configurable(allowlist=('model_cls', 'to_xarray_fn', 'from_xarray_fn'))
class WhirlModel:


  def __init__(
      self,
      coords: coordinate_systems.CoordinateSystem,
      dt: float,
      physics_specs: Any,
      aux_features: Optional[AuxFeatures] = None,
      input_coords: Optional[coordinate_systems.CoordinateSystem] = None,
      output_coords: Optional[coordinate_systems.CoordinateSystem] = None,
      model_cls: Callable[[], DynamicalSystem] = gin.REQUIRED,
      to_xarray_fn: Optional[Callable[..., xarray.Dataset]] = None,
      from_xarray_fn: Optional[Callable[..., DataState]] = None,
  ):

    if aux_features is None:
      aux_features = {}
    if input_coords is None:
      input_coords = coords
    if output_coords is None:
      output_coords = coords
    self._coords = coords
    self._data_coords = input_coords
    specs = ModelSpecs(coords, dt, physics_specs, aux_features)
    model_cls = functools.partial(
        model_cls,
        coords=coords,
        dt=dt,
        physics_specs=physics_specs,
        aux_features=aux_features,
        input_coords=input_coords,
        output_coords=output_coords,
    )

    def forcing_fwd(forcing_data, sim_time):
      return model_cls().forcing_fn(forcing_data, sim_time)

    forcing_fn = hk.transform(forcing_fwd).apply
    encode_fwd = lambda x, forcing: model_cls().encode(x, forcing)
    encode_fn = hk.transform(encode_fwd).apply
    decode_fwd = lambda x, forcing: model_cls().decode(x, forcing)
    decode_fn = hk.transform(decode_fwd).apply
    advance_fwd = lambda x, forcing: model_cls().advance(x, forcing)
    advance_fn = hk.transform(advance_fwd).apply
    if to_xarray_fn is not None:
      to_xarray_fn = functools.partial(to_xarray_fn, coords=output_coords)
    self.forcing_fn = forcing_fn
    self.encode_fn = encode_fn
    self.decode_fn = decode_fn
    self.advance_fn = advance_fn
    self.specs = specs
    self.model_cls = model_cls
    self.to_xarray_fn = to_xarray_fn
    self.from_xarray_fn = from_xarray_fn

  @property
  def coords(self) -> coordinate_systems.CoordinateSystem:
    return self._coords

  @property
  def data_coords(self) -> coordinate_systems.CoordinateSystem:
    return self._data_coords

  def init_params(
      self,
      rng: Array,
      input_trajectory: typing.DataState,
      forcing_data: ForcingData,
  ) -> typing.Params:


    def fwd(x):
      model = self.model_cls()
      decode = model_utils.with_forcing(
          model.decode, model.forcing_fn, forcing_data
      )
      advance = model_utils.with_forcing(
          model.advance, model.forcing_fn, forcing_data
      )
      encode = model_utils.with_forcing(
          model.encode, model.forcing_fn, forcing_data
      )
      return decode(advance(encode(x)))

    hk_model = hk.transform(fwd)
    return hk_model.init(rng, input_trajectory)


def get_whirl_model(
    data_ds: xarray.Dataset,
    model_config_str: str,
    additional_gin_bindings: Optional[list[str]] = None,
) -> WhirlModel:

  if additional_gin_bindings is None:
    additional_gin_bindings = []

  try:
    data_aux_features = xarray_utils.aux_features_from_xarray(data_ds)
  except KeyError:
    data_aux_features = {}

  if 'physics_config_str' in data_ds.attrs:
    physics_config_str = data_ds.attrs['physics_config_str']
  else:
    physics_config_str = ''

  gin.enter_interactive_mode()
  gin.clear_config()
  gin_utils.parse_gin_config(
      physics_config_str,
      model_config_str,
      override_physics_configs_from_data=True,
      gin_bindings=additional_gin_bindings,
  )

  data_coords = coordinate_system_from_dataset(data_ds)
  physics_specs = physics_specifications.get_physics_specs()
  model_specs = get_model_specs(data_coords, physics_specs, data_aux_features)
  return WhirlModel(
      coords=model_specs.coords,
      dt=model_specs.dt,
      physics_specs=model_specs.physics_specs,
      aux_features=model_specs.aux_features,
      input_coords=data_coords,
      output_coords=data_coords,
  )


_ECMWF_CUTOFFS = {


    'low_cutoffs': (0.05, 0.1),

    'high_cutoffs': (0.86, 0.965),
}


def _piecewise_squasher(
    sigma: Array,
    low_cutoffs: Sequence[float],
    high_cutoffs: Sequence[float],
) -> Array:

  if sigma.ndim != 1:
    raise ValueError(f'{sigma.shape=} but should have been a 1-D array')
  if len(low_cutoffs) != 2:
    raise ValueError(f'{len(low_cutoffs)=} but should have been 2.')
  if len(high_cutoffs) != 2:
    raise ValueError(f'{len(high_cutoffs)=} but should have been 2.')

  low_func = (sigma - low_cutoffs[0]) / (low_cutoffs[1] - low_cutoffs[0])
  high_func = (high_cutoffs[1] - sigma) / (high_cutoffs[1] - high_cutoffs[0])


  lower_bound = jnp.minimum(1.0, jnp.minimum(low_func, high_func))
  return jnp.maximum(0.0, lower_bound)[:, jnp.newaxis, jnp.newaxis]
