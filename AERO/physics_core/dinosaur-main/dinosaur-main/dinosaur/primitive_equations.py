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

from typing import Callable, Mapping, Sequence

from dinosaur import coordinate_systems
from dinosaur import jax_numpy_utils
from dinosaur import scales
from dinosaur import sigma_coordinates
from dinosaur import spherical_harmonic
from dinosaur import time_integration
from dinosaur import typing
from dinosaur import vertical_interpolation

import jax
from jax import lax
import jax.numpy as jnp
import numpy as np
import tree_math

units = scales.units

Array = typing.Array
Numeric = typing.Numeric
Quantity = typing.Quantity

OrographyInitFn = Callable[..., Array]

SCALE = scales.DEFAULT_SCALE
GRAVITY_ACCELERATION = SCALE.nondimensionalize(scales.GRAVITY_ACCELERATION)
IDEAL_GAS_CONSTANT = SCALE.nondimensionalize(scales.IDEAL_GAS_CONSTANT)
WATER_VAPOR_GAS_CONSTANT = SCALE.nondimensionalize(
    scales.IDEAL_GAS_CONSTANT_H20
)
WATER_VAPOR_CP = SCALE.nondimensionalize(scales.WATER_VAPOR_CP)
KAPPA = SCALE.nondimensionalize(scales.KAPPA)


einsum = functools.partial(jnp.einsum, precision=jax.lax.Precision.HIGHEST)


@tree_math.struct
class State:


  vorticity: Array
  divergence: Array
  temperature_variation: Array
  log_surface_pressure: Array
  tracers: Mapping[str, Array] = dataclasses.field(default_factory=dict)


@tree_math.struct
class StateWithTime:


  vorticity: Array
  divergence: Array
  temperature_variation: Array
  log_surface_pressure: Array
  sim_time: float
  tracers: Mapping[str, Array] = dataclasses.field(default_factory=dict)


class StateShapeError(Exception):
  pass


def validate_state_shape(
    state: State, coords: coordinate_systems.CoordinateSystem
):

  if state.vorticity.shape != coords.modal_shape:
    raise StateShapeError(
        f'Expected vorticity shape {coords.modal_shape}; '
        f'got shape {state.vorticity.shape}.'
    )
  if state.divergence.shape != coords.modal_shape:
    raise StateShapeError(
        f'Expected divergence shape {coords.modal_shape}; '
        f'got shape {state.divergence.shape}.'
    )
  if state.temperature_variation.shape != coords.modal_shape:
    raise StateShapeError(
        f'Expected temperature_variation shape {coords.modal_shape}; '
        f'got shape {state.temperature_variation.shape}.'
    )
  if state.log_surface_pressure.shape != coords.surface_modal_shape:
    raise StateShapeError(
        f'Expected log_surface_pressure shape {coords.surface_modal_shape}; '
        f'got shape {state.log_surface_pressure.shape}.'
    )
  for tracer_name, array in state.tracers.items():
    if array.shape[-3:] != coords.modal_shape:
      raise StateShapeError(
          f'Expected tracer {tracer_name} shape {coords.modal_shape}; '
          f'got shape {array.shape}.'
      )


@tree_math.struct
class DiagnosticState:


  vorticity: Array
  divergence: Array
  temperature_variation: Array
  cos_lat_u: tuple[Array, Array]
  sigma_dot_explicit: Array
  sigma_dot_full: Array
  cos_lat_grad_log_sp: Array
  u_dot_grad_log_sp: Array
  tracers: Mapping[str, Array]


@jax.named_call
def compute_diagnostic_state(
    state: State,
    coords: coordinate_systems.CoordinateSystem,
) -> DiagnosticState:


  def to_nodal_fn(x):
    return coords.horizontal.to_nodal(x)

  nodal_vorticity = to_nodal_fn(state.vorticity)
  nodal_divergence = to_nodal_fn(state.divergence)
  nodal_temperature_variation = to_nodal_fn(state.temperature_variation)
  tracers = to_nodal_fn(state.tracers)
  nodal_cos_lat_u = jax.tree_util.tree_map(
      to_nodal_fn,
      spherical_harmonic.get_cos_lat_vector(
          state.vorticity, state.divergence, coords.horizontal, clip=False
      ),
  )
  cos_lat_grad_log_sp = coords.horizontal.cos_lat_grad(
      state.log_surface_pressure, clip=False
  )
  nodal_cos_lat_grad_log_sp = to_nodal_fn(cos_lat_grad_log_sp)
  nodal_u_dot_grad_log_sp = sum(jax.tree_util.tree_map(
      lambda x, y: x * y * coords.horizontal.sec2_lat,
      nodal_cos_lat_u, nodal_cos_lat_grad_log_sp))
  f_explicit = sigma_coordinates.cumulative_sigma_integral(
      nodal_u_dot_grad_log_sp, coords.vertical)
  f_full = sigma_coordinates.cumulative_sigma_integral(
      nodal_divergence + nodal_u_dot_grad_log_sp, coords.vertical)

  sum_𝜎 = np.cumsum(coords.vertical.layer_thickness)[:, np.newaxis, np.newaxis]
  sigma_dot_explicit = lax.slice_in_dim(
      sum_𝜎 * lax.slice_in_dim(f_explicit, -1, None) - f_explicit, 0, -1
  )
  sigma_dot_full = lax.slice_in_dim(
      sum_𝜎 * lax.slice_in_dim(f_full, -1, None) - f_full, 0, -1
  )
  return DiagnosticState(
      vorticity=nodal_vorticity,
      divergence=nodal_divergence,
      temperature_variation=nodal_temperature_variation,
      cos_lat_u=nodal_cos_lat_u,
      sigma_dot_explicit=sigma_dot_explicit,
      sigma_dot_full=sigma_dot_full,
      cos_lat_grad_log_sp=nodal_cos_lat_grad_log_sp,
      u_dot_grad_log_sp=nodal_u_dot_grad_log_sp,
      tracers=tracers,
  )


def _vertical_interp(x, xp, fp):


  assert x.ndim in {1, 3} and xp.ndim in {1, 3}
  interpolate_fn = vertical_interpolation.interp
  in_axes = (-1 if x.ndim == 3 else None, -1 if xp.ndim == 3 else None, -1)
  interpolate_fn = jax.vmap(interpolate_fn, in_axes, out_axes=-1)
  interpolate_fn = jax.vmap(interpolate_fn, in_axes, out_axes=-1)
  interpolate_fn = jax.vmap(interpolate_fn, (0, None, None), out_axes=0)
  return interpolate_fn(x, xp, fp)


def compute_vertical_velocity(
    state: State, coords: coordinate_systems.CoordinateSystem
) -> jax.Array:

  sigma_dot_boundaries = compute_diagnostic_state(state, coords).sigma_dot_full
  assert sigma_dot_boundaries.ndim == 3


  sigma_dot_padded = jnp.pad(sigma_dot_boundaries, [(1, 1), (0, 0), (0, 0)])
  return 0.5 * (sigma_dot_padded[1:] + sigma_dot_padded[:-1])


def semi_lagrangian_vertical_advection_step(
    state: State, coords: coordinate_systems.CoordinateSystem, dt: float
) -> State:

  velocity = compute_vertical_velocity(state, coords)
  target = coords.vertical.centers
  source = target[:, jnp.newaxis, jnp.newaxis] - dt * velocity

  def interpolate(x):
    if x.ndim < 3 or x.shape[0] == 1:
      return x

    x = coords.horizontal.to_nodal(x)
    x = _vertical_interp(target, source, x)
    x = coords.horizontal.to_modal(x)
    return x

  return jax.tree_util.tree_map(interpolate, state)


@dataclasses.dataclass(frozen=True)
class PrimitiveEquationsSpecs:


  radius: float
  angular_velocity: float
  gravity_acceleration: float
  ideal_gas_constant: float
  water_vapor_gas_constant: float
  water_vapor_isobaric_heat_capacity: float
  kappa: float
  scale: scales.Scale

  @property
  def R(self) -> float:

    return self.ideal_gas_constant

  @property
  def R_vapor(self) -> float:

    return self.water_vapor_gas_constant

  @property
  def g(self) -> float:

    return self.gravity_acceleration

  @property
  def Cp(self) -> float:

    return self.ideal_gas_constant / self.kappa

  @property
  def Cp_vapor(self) -> float:

    return self.water_vapor_isobaric_heat_capacity

  def nondimensionalize(self, quantity: Quantity) -> Numeric:

    return self.scale.nondimensionalize(quantity)

  def nondimensionalize_timedelta64(self, timedelta: np.timedelta64) -> Numeric:

    base_unit = 's'
    return self.scale.nondimensionalize(
        timedelta / np.timedelta64(1, base_unit) * units(base_unit)
    )

  def dimensionalize(self, value: Numeric, unit: units.Unit) -> Quantity:

    return self.scale.dimensionalize(value, unit)

  def dimensionalize_timedelta64(self, value: Numeric) -> np.timedelta64:

    base_unit = 's'
    dt = self.scale.dimensionalize(value, units(base_unit)).m
    if isinstance(dt, np.ndarray):
      return dt.astype(f'timedelta64[{base_unit}]')
    else:
      return np.timedelta64(int(dt), base_unit)

  @classmethod
  def from_si(
      cls,
      radius_si: Quantity = scales.RADIUS,
      angular_velocity_si: Quantity = scales.ANGULAR_VELOCITY,
      gravity_acceleration_si: Quantity = scales.GRAVITY_ACCELERATION,
      ideal_gas_constant_si: Quantity = scales.IDEAL_GAS_CONSTANT,
      water_vapor_gas_constant_si: Quantity = scales.IDEAL_GAS_CONSTANT_H20,
      water_vapor_isobaric_heat_capacity_si: Quantity = scales.WATER_VAPOR_CP,
      kappa_si: Quantity = scales.KAPPA,
      scale: scales.Scale = scales.DEFAULT_SCALE,
  ) -> PrimitiveEquationsSpecs:

    return cls(
        scale.nondimensionalize(radius_si),
        scale.nondimensionalize(angular_velocity_si),
        scale.nondimensionalize(gravity_acceleration_si),
        scale.nondimensionalize(ideal_gas_constant_si),
        scale.nondimensionalize(water_vapor_gas_constant_si),
        scale.nondimensionalize(water_vapor_isobaric_heat_capacity_si),
        scale.nondimensionalize(kappa_si),
        scale,
    )


def get_sigma_ratios(
    coordinates: sigma_coordinates.SigmaCoordinates,
) -> np.ndarray:

  alpha = np.diff(np.log(coordinates.centers), append=0) / 2
  alpha[-1] = -np.log(coordinates.centers[-1])
  return alpha


def get_geopotential_weights(
    coordinates: sigma_coordinates.SigmaCoordinates,
    ideal_gas_constant: float = IDEAL_GAS_CONSTANT,
) -> np.ndarray:


  alpha = get_sigma_ratios(coordinates)
  weights = np.zeros([coordinates.layers, coordinates.layers])
  for j in range(coordinates.layers):
    weights[j, j] = alpha[j]
    for k in range(j + 1, coordinates.layers):
      weights[j, k] = alpha[k] + alpha[k - 1]
  return ideal_gas_constant * weights


def get_geopotential_diff(
    temperature: Array,
    coordinates: sigma_coordinates.SigmaCoordinates,
    ideal_gas_constant: float = IDEAL_GAS_CONSTANT,
    method: str = 'dense',
    sharding: jax.sharding.NamedSharding | None = None,
) -> jax.Array:

  if method == 'dense':
    weights = get_geopotential_weights(coordinates, ideal_gas_constant)
    return _vertical_matvec(weights, temperature)
  elif method == 'sparse':
    alpha = ideal_gas_constant * get_sigma_ratios(coordinates)
    alpha2 = np.concatenate([[0], alpha[1:] + alpha[:-1]])
    return (
        jax_numpy_utils.reverse_cumsum(
            alpha2[:, np.newaxis, np.newaxis] * temperature,
            axis=0,
            sharding=sharding,
        )
        + (alpha - alpha2)[:, np.newaxis, np.newaxis] * temperature
    )
  else:
    raise ValueError(f'unknown {method=} for get_geopotential_diff')


_CONSTANT_NORMALIZATION_FACTOR = 3.5449077


def _add_constant(x: jnp.ndarray, c: float | Array) -> jnp.ndarray:

  return x.at[..., 0, 0].add(_CONSTANT_NORMALIZATION_FACTOR * c)


def get_geopotential(
    temperature_variation: Array,
    reference_temperature: Array,
    orography: Array,
    coordinates: sigma_coordinates.SigmaCoordinates,
    gravity_acceleration: float = GRAVITY_ACCELERATION,
    ideal_gas_constant: float = IDEAL_GAS_CONSTANT,
    sharding: jax.sharding.NamedSharding | None = None,
) -> jnp.ndarray:

  surface_geopotential = orography * gravity_acceleration
  temperature = _add_constant(temperature_variation, reference_temperature)
  geopotential_diff = get_geopotential_diff(
      temperature, coordinates, ideal_gas_constant, sharding=sharding
  )
  return surface_geopotential + geopotential_diff


def get_geopotential_with_moisture(
    temperature: typing.Array,
    specific_humidity: typing.Array,
    nodal_orography: typing.Array,
    coordinates: sigma_coordinates.SigmaCoordinates,
    gravity_acceleration: float = GRAVITY_ACCELERATION,
    ideal_gas_constant: float = IDEAL_GAS_CONSTANT,
    water_vapor_gas_constant: float = WATER_VAPOR_GAS_CONSTANT,
    sharding: jax.sharding.NamedSharding | None = None,
) -> jnp.ndarray:

  gas_const_ratio = water_vapor_gas_constant / ideal_gas_constant
  surface_geopotential = nodal_orography * gravity_acceleration
  virtual_temp = temperature * (1 + (gas_const_ratio - 1) * specific_humidity)
  geopotential_diff = get_geopotential_diff(
      virtual_temp, coordinates, ideal_gas_constant, sharding=sharding
  )
  return surface_geopotential + geopotential_diff


def get_temperature_implicit_weights(
    coordinates: sigma_coordinates.SigmaCoordinates,
    reference_temperature: np.ndarray,
    kappa: float = KAPPA,
) -> np.ndarray:

  if (
      reference_temperature.ndim != 1
      or reference_temperature.shape[-1] != coordinates.layers
  ):
    raise ValueError(
        '`reference_temp` must be a vector of length `coordinates.layers`; '
        f'got shape {reference_temperature.shape} and '
        f'{coordinates.layers} layers.'
    )


  p = np.tril(np.ones([coordinates.layers, coordinates.layers]))


  alpha = get_sigma_ratios(coordinates)[..., np.newaxis]
  p_alpha = p * alpha
  p_alpha_shifted = np.roll(p_alpha, 1, axis=0)
  p_alpha_shifted[0] = 0
  h0 = (
      kappa
      * reference_temperature[..., np.newaxis]
      * (p_alpha + p_alpha_shifted)
      / coordinates.layer_thickness[..., np.newaxis]
  )


  temp_diff = np.diff(reference_temperature)
  thickness_sum = (
      coordinates.layer_thickness[:-1] + coordinates.layer_thickness[1:]
  )

  k0 = np.concatenate((temp_diff / thickness_sum, [0]), axis=0)[..., np.newaxis]

  thickness_cumulative = np.cumsum(coordinates.layer_thickness)[..., np.newaxis]

  k1 = p - thickness_cumulative

  k = k0 * k1


  k_shifted = np.roll(k, 1, axis=0)
  k_shifted[0] = 0

  return (h0 - k - k_shifted) * coordinates.layer_thickness


def get_temperature_implicit(
    divergence: Array,
    coordinates: sigma_coordinates.SigmaCoordinates,
    reference_temperature: np.ndarray,
    kappa: float = KAPPA,
    method: str = 'dense',
    sharding: jax.sharding.NamedSharding | None = None,
) -> jax.Array:

  weights = -get_temperature_implicit_weights(
      coordinates, reference_temperature, kappa
  )

  if method == 'dense':
    return _vertical_matvec(weights, divergence)
  elif method == 'sparse':
    diag_weights = np.diag(weights)
    up_weights = np.concatenate([[0], weights[1:, 0]])
    down_weights = np.concatenate([weights[:-1, -1], [0]])
    up_divergence = (
        jax_numpy_utils.cumsum(divergence, axis=0, sharding=sharding)
        - divergence
    )
    result = (
        up_weights[:, np.newaxis, np.newaxis] * up_divergence
        + diag_weights[:, np.newaxis, np.newaxis] * divergence
    )
    if (down_weights != 0).any():

      down_divergence = (
          jax_numpy_utils.reverse_cumsum(divergence, axis=0, sharding=sharding)
          - divergence
      )
      result += down_weights[:, np.newaxis, np.newaxis] * down_divergence
    return result
  else:
    raise ValueError(f'unknown {method=} for get_temperature_implicit')


@jax.named_call
def _vertical_matvec(a: Array, x: Array) -> jax.Array:

  return einsum('gh,...hml->...gml', a, x)


@jax.named_call
def _vertical_matvec_per_wavenumber(a: Array, x: Array) -> jax.Array:


  return einsum('lgh,...hml->...gml', a, x)


def _get_implicit_term_matrix(
    eta,
    coords,
    reference_temperature,
    kappa=KAPPA,
    ideal_gas_constant=IDEAL_GAS_CONSTANT,
) -> np.ndarray:


  eye = np.eye(coords.vertical.layers)[np.newaxis]
  lam = coords.horizontal.laplacian_eigenvalues
  g = get_geopotential_weights(coords.vertical, ideal_gas_constant)
  r = ideal_gas_constant
  h = get_temperature_implicit_weights(
      coords.vertical, reference_temperature, kappa
  )
  t = reference_temperature[:, np.newaxis]
  thickness = coords.vertical.layer_thickness[np.newaxis, np.newaxis, :]


  l = coords.horizontal.modal_shape[1]
  j = k = coords.vertical.layers

  row0 = np.concatenate(
      [
          np.broadcast_to(eye, [l, j, k]),
          eta * np.einsum('l,jk->ljk', lam, g),
          eta * r * np.einsum('l,jo->ljo', lam, t),
      ],
      axis=2,
  )
  row1 = np.concatenate(
      [
          eta * np.broadcast_to(h[np.newaxis], [l, j, k]),
          np.broadcast_to(eye, [l, j, k]),
          np.zeros([l, j, 1]),
      ],
      axis=2,
  )
  row2 = np.concatenate(
      [
          np.broadcast_to(eta * thickness, [l, 1, k]),
          np.zeros([l, 1, k]),
          np.ones([l, 1, 1]),
      ],
      axis=2,
  )
  return np.concatenate((row0, row1, row2), axis=1)


def div_sec_lat(
    m_component: Array, n_component: Array, grid: spherical_harmonic.Grid
) -> Array:


  m_component = grid.to_modal(m_component * grid.sec2_lat)
  n_component = grid.to_modal(n_component * grid.sec2_lat)
  return grid.div_cos_lat((m_component, n_component), clip=False)


def truncated_modal_orography(
    orography: Array,
    coords: coordinate_systems.CoordinateSystem,
    wavenumbers_to_clip: int = 1,
) -> Array:

  grid = coords.horizontal
  expected_shape = grid.nodal_shape
  if orography.shape != expected_shape:
    raise ValueError(f'Expected nodal orography with shape={expected_shape}')
  return grid.clip_wavenumbers(grid.to_modal(orography), n=wavenumbers_to_clip)


def filtered_modal_orography(
    orography: Array,
    coords: coordinate_systems.CoordinateSystem,
    input_coords: coordinate_systems.CoordinateSystem | None = None,
    filter_fns: Sequence[typing.PostProcessFn] = tuple(),
) -> Array:

  if input_coords is None:
    input_coords = coords
  expected_shape = input_coords.horizontal.nodal_shape
  if orography.shape != expected_shape:
    raise ValueError(f'Expected nodal orography with shape={expected_shape}')
  interpolate_fn = coordinate_systems.get_spectral_interpolate_fn(
      input_coords, coords, expect_same_vertical=False
  )
  modal_orography = interpolate_fn(input_coords.horizontal.to_modal(orography))
  for filter_fn in filter_fns:
    modal_orography = filter_fn(modal_orography)
  return modal_orography


@dataclasses.dataclass
class PrimitiveEquations(time_integration.ImplicitExplicitODE):

  reference_temperature: np.ndarray
  orography: Array
  coords: coordinate_systems.CoordinateSystem
  physics_specs: PrimitiveEquationsSpecs
  vertical_matmul_method: str | None = None
  vertical_advection: Callable[..., jax.Array] = (
      sigma_coordinates.centered_vertical_advection
  )
  include_vertical_advection: bool = True

  @property
  def coriolis_parameter(self) -> Array:

    _, sin_lat = self.coords.horizontal.nodal_mesh
    return 2 * self.physics_specs.angular_velocity * sin_lat

  @property
  def T_ref(self) -> Array:

    return self.reference_temperature[..., np.newaxis, np.newaxis]

  @jax.named_call
  def _vertical_tendency(self, w: Array, x: Array) -> Array:

    return self.vertical_advection(w, x, self.coords.vertical)

  @jax.named_call
  def _t_omega_over_sigma_sp(
      self, temperature_field: Array, g_term: Array, v_dot_grad_log_sp: Array
  ) -> Array:

    f = sigma_coordinates.cumulative_sigma_integral(
        g_term, self.coords.vertical, sharding=self.coords.dycore_sharding
    )
    alpha = get_sigma_ratios(self.coords.vertical)
    alpha = alpha[:, np.newaxis, np.newaxis]
    del_𝜎 = self.coords.vertical.layer_thickness[:, np.newaxis, np.newaxis]
    padding = [(1, 0), (0, 0), (0, 0)]
    g_part = (alpha * f + jnp.pad(alpha * f, padding)[:-1, ...]) / del_𝜎
    return temperature_field * (v_dot_grad_log_sp - g_part)

  @jax.named_call
  def kinetic_energy_tendency(self, aux_state: DiagnosticState) -> Array:

    nodal_cos_lat_u2 = jnp.stack(aux_state.cos_lat_u) ** 2
    kinetic = nodal_cos_lat_u2.sum(0) * self.coords.horizontal.sec2_lat / 2
    return -self.coords.horizontal.laplacian(
        self.coords.horizontal.to_modal(kinetic)
    )

  @jax.named_call
  def orography_tendency(self) -> Array:


    return -self.physics_specs.g * self.coords.horizontal.laplacian(
        self.orography
    )

  @jax.named_call
  def curl_and_div_tendencies(
      self,
      aux_state: DiagnosticState,
  ) -> tuple[Array, Array]:

    sec2_lat = self.coords.horizontal.sec2_lat

    u, v = aux_state.cos_lat_u
    total_vorticity = aux_state.vorticity + self.coriolis_parameter

    nodal_vorticity_u = -v * total_vorticity * sec2_lat
    nodal_vorticity_v = u * total_vorticity * sec2_lat

    d𝜎_dt = aux_state.sigma_dot_full
    if self.include_vertical_advection:

      sigma_dot_u = -self._vertical_tendency(d𝜎_dt, u)
      sigma_dot_v = -self._vertical_tendency(d𝜎_dt, v)
    else:
      sigma_dot_u = 0
      sigma_dot_v = 0
    rt = self.physics_specs.R * aux_state.temperature_variation
    grad_log_ps_u, grad_log_ps_v = aux_state.cos_lat_grad_log_sp
    vertical_term_u = (sigma_dot_u + rt * grad_log_ps_u) * sec2_lat
    vertical_term_v = (sigma_dot_v + rt * grad_log_ps_v) * sec2_lat
    combined_u = self.coords.horizontal.to_modal(
        nodal_vorticity_u + vertical_term_u
    )
    combined_v = self.coords.horizontal.to_modal(
        nodal_vorticity_v + vertical_term_v
    )

    dζ_dt = -self.coords.horizontal.curl_cos_lat(
        (combined_u, combined_v), clip=False
    )
    d𝛅_dt = -self.coords.horizontal.div_cos_lat(
        (combined_u, combined_v), clip=False
    )
    return (dζ_dt, d𝛅_dt)

  @jax.named_call
  def nodal_temperature_vertical_tendency(
      self,
      aux_state: DiagnosticState,
  ) -> Array | float:


    sigma_dot_explicit = aux_state.sigma_dot_explicit
    sigma_dot_full = aux_state.sigma_dot_full
    temperature_variation = aux_state.temperature_variation
    if self.include_vertical_advection:
      tendency = self._vertical_tendency(sigma_dot_full, temperature_variation)
    else:
      tendency = 0
    if np.unique(self.T_ref.ravel()).size > 1:

      tendency += self._vertical_tendency(sigma_dot_explicit, self.T_ref)
    return tendency

  @jax.named_call
  def horizontal_scalar_advection(
      self,
      scalar: Array,
      aux_state: DiagnosticState,
  ) -> tuple[Array, Array]:

    u, v = aux_state.cos_lat_u
    nodal_terms = scalar * aux_state.divergence
    modal_terms = -div_sec_lat(u * scalar, v * scalar, self.coords.horizontal)
    return nodal_terms, modal_terms

  @jax.named_call
  def nodal_temperature_adiabatic_tendency(
      self, aux_state: DiagnosticState
  ) -> Array:

    g_explicit = aux_state.u_dot_grad_log_sp
    g_full = g_explicit + aux_state.divergence
    mean_t_part = self._t_omega_over_sigma_sp(
        self.T_ref, g_explicit, aux_state.u_dot_grad_log_sp
    )
    variation_t_part = self._t_omega_over_sigma_sp(
        aux_state.temperature_variation, g_full, aux_state.u_dot_grad_log_sp
    )
    return self.physics_specs.kappa * (mean_t_part + variation_t_part)

  @jax.named_call
  def nodal_log_pressure_tendency(self, aux_state: DiagnosticState) -> Array:


    g = aux_state.u_dot_grad_log_sp
    return -sigma_coordinates.sigma_integral(g, self.coords.vertical)

  @jax.named_call
  def explicit_terms(self, state: State) -> State:

    aux_state = compute_diagnostic_state(state, self.coords)

    vorticity_tendency, divergence_dot = self.curl_and_div_tendencies(aux_state)
    kinetic_energy_tendency = self.kinetic_energy_tendency(aux_state)
    orography_tendency = self.orography_tendency()
    horizontal_tendency_fn = functools.partial(
        self.horizontal_scalar_advection, aux_state=aux_state
    )
    dT_dt_horizontal_nodal, dT_dt_horizontal_modal = horizontal_tendency_fn(
        aux_state.temperature_variation
    )
    tracers_horizontal_nodal_and_modal = jax.tree_util.tree_map(
        horizontal_tendency_fn, aux_state.tracers
    )

    dT_dt_vertical = self.nodal_temperature_vertical_tendency(aux_state)
    dT_dt_adiabatic = self.nodal_temperature_adiabatic_tendency(aux_state)
    log_sp_tendency = self.nodal_log_pressure_tendency(aux_state)
    sigma_dot_full = aux_state.sigma_dot_full
    if self.include_vertical_advection:
      vertical_tendency_fn = functools.partial(
          self._vertical_tendency, sigma_dot_full
      )
    else:
      vertical_tendency_fn = lambda x: 0
    tracers_vertical_nodal = jax.tree_util.tree_map(
        vertical_tendency_fn, aux_state.tracers
    )

    to_modal_fn = self.coords.horizontal.to_modal
    divergence_tendency = (
        divergence_dot + kinetic_energy_tendency + orography_tendency
    )
    temperature_tendency = (
        to_modal_fn(dT_dt_horizontal_nodal + dT_dt_vertical + dT_dt_adiabatic)
        + dT_dt_horizontal_modal
    )
    log_surface_pressure_tendency = to_modal_fn(log_sp_tendency)
    tracers_tendency = jax.tree_util.tree_map(
        lambda x, y_z: to_modal_fn(x + y_z[0]) + y_z[1],
        tracers_vertical_nodal,
        tracers_horizontal_nodal_and_modal,
    )
    tendency = State(
        vorticity=vorticity_tendency,
        divergence=divergence_tendency,
        temperature_variation=temperature_tendency,
        log_surface_pressure=log_surface_pressure_tendency,
        tracers=tracers_tendency,
    )


    return self.coords.horizontal.clip_wavenumbers(tendency)

  @jax.named_call
  def implicit_terms(self, state: State) -> State:

    method = self.vertical_matmul_method
    if method is None:
      mesh = self.coords.spmd_mesh
      method = 'sparse' if mesh is not None and mesh.shape['z'] > 1 else 'dense'

    geopotential_diff = get_geopotential_diff(
        state.temperature_variation,
        self.coords.vertical,
        self.physics_specs.R,
        method=method,
        sharding=self.coords.dycore_sharding,
    )
    rt_log_p = (
        self.physics_specs.ideal_gas_constant
        * self.T_ref
        * state.log_surface_pressure
    )
    vorticity_implicit = jnp.zeros_like(state.vorticity)
    divergence_implicit = -self.coords.horizontal.laplacian(
        geopotential_diff + rt_log_p
    )
    temperature_variation_implicit = get_temperature_implicit(
        state.divergence,
        self.coords.vertical,
        self.reference_temperature,
        self.physics_specs.kappa,
        method=method,
        sharding=self.coords.dycore_sharding,
    )
    log_surface_pressure_implicit = -_vertical_matvec(
        self.coords.vertical.layer_thickness[np.newaxis], state.divergence
    )
    tracers_implicit = jax.tree_util.tree_map(jnp.zeros_like, state.tracers)
    return State(
        vorticity=vorticity_implicit,
        divergence=divergence_implicit,
        temperature_variation=temperature_variation_implicit,
        log_surface_pressure=log_surface_pressure_implicit,
        tracers=tracers_implicit,
    )

  @jax.named_call
  def implicit_inverse(
      self,
      state: State,
      step_size: float,
      method: str = 'split',
  ) -> State:

    if isinstance(step_size, jax.core.Tracer):


      raise TypeError(
          f'`step_size` must be concrete but a Tracer was passed: {step_size}. '
          'This error is likely caused by '
          '`jax.jit(primitive.inverse_terms)(state, eta). Instead, do '
          '`jax.jit(lambda s: primitive.inverse_terms(s, eta=eta))(state)`.'
      )

    implicit_matrix = _get_implicit_term_matrix(
        step_size,
        self.coords,
        self.reference_temperature,
        self.physics_specs.kappa,
        self.physics_specs.R,
    )
    assert implicit_matrix.dtype == np.float64


    layers = self.coords.vertical.layers
    div = slice(0, layers)
    temp = slice(layers, 2 * layers)
    logp = slice(2 * layers, 2 * layers + 1)
    temp_logp = slice(layers, 2 * layers + 1)

    def named_vertical_matvec(name):
      return jax.named_call(_vertical_matvec_per_wavenumber, name=name)

    if method == 'split':


      inverse = np.linalg.inv(implicit_matrix)
      assert not np.isnan(inverse).any()

      inverted_divergence = (
          named_vertical_matvec('div_from_div')(
              inverse[:, div, div], state.divergence
          )
          + named_vertical_matvec('div_from_temp')(
              inverse[:, div, temp], state.temperature_variation
          )
          + named_vertical_matvec('div_from_logp')(
              inverse[:, div, logp], state.log_surface_pressure
          )
      )
      inverted_temperature_variation = (
          named_vertical_matvec('temp_from_div')(
              inverse[:, temp, div], state.divergence
          )
          + named_vertical_matvec('temp_from_temp')(
              inverse[:, temp, temp], state.temperature_variation
          )
          + named_vertical_matvec('temp_from_logp')(
              inverse[:, temp, logp], state.log_surface_pressure
          )
      )
      inverted_log_surface_pressure = (
          named_vertical_matvec('logp_from_div')(
              inverse[:, logp, div], state.divergence
          )
          + named_vertical_matvec('logp_from_temp')(
              inverse[:, logp, temp], state.temperature_variation
          )
          + named_vertical_matvec('logp_from_logp')(
              inverse[:, logp, logp], state.log_surface_pressure
          )
      )

    elif method == 'stacked':


      inverse = np.linalg.inv(implicit_matrix)
      assert not np.isnan(inverse).any()
      stacked_state = jnp.concatenate([
          state.divergence,
          state.temperature_variation,
          state.log_surface_pressure,
      ])
      stacked_inverse = named_vertical_matvec('inverse')(inverse, stacked_state)
      inverted_divergence = stacked_inverse[div]
      inverted_temperature_variation = stacked_inverse[temp]
      inverted_log_surface_pressure = stacked_inverse[logp]

    elif method == 'blockwise':


      GH = (
          implicit_matrix[:, div, temp_logp]
          @ implicit_matrix[:, temp_logp, div]
      )
      div_inverse = np.linalg.inv(np.eye(layers) - GH)

      η = step_size
      λ = self.coords.horizontal.laplacian_eigenvalues

      gt = get_geopotential_diff(
          state.temperature_variation,
          self.coords.vertical,
          self.physics_specs.R,
          method='sparse',
          sharding=self.coords.dycore_sharding,
      )
      div_from_temp = η * λ[np.newaxis, np.newaxis, :] * gt

      div_from_logp = named_vertical_matvec('div_from_logp')(
          implicit_matrix[:, div, logp], state.log_surface_pressure
      )
      inverted_divergence = named_vertical_matvec('div_solve')(
          div_inverse, state.divergence - div_from_temp - div_from_logp
      )
      HG = (
          implicit_matrix[:, temp_logp, div]
          @ implicit_matrix[:, div, temp_logp]
      )
      temp_logp_inverse = np.linalg.inv(np.eye(layers + 1) - HG)

      hd = -get_temperature_implicit(
          state.divergence,
          self.coords.vertical,
          self.reference_temperature,
          self.physics_specs.kappa,
          method='sparse',
          sharding=self.coords.dycore_sharding,
      )
      temp_from_div = η * hd
      temp_part = state.temperature_variation - temp_from_div

      logp_from_div = named_vertical_matvec('logp_from_div')(
          implicit_matrix[:, logp, div], state.divergence
      )
      logp_part = state.log_surface_pressure - logp_from_div

      inverted_temperature_variation = named_vertical_matvec(
          'temp_solve_from_temp'
      )(temp_logp_inverse[:, :-1, :-1], temp_part) + named_vertical_matvec(
          'temp_solve_from_logp'
      )(
          temp_logp_inverse[:, :-1:, -1:], logp_part
      )
      inverted_log_surface_pressure = named_vertical_matvec(
          'logp_solve_from_temp'
      )(temp_logp_inverse[:, -1:, :-1], temp_part) + named_vertical_matvec(
          'logp_solve_from_temp'
      )(
          temp_logp_inverse[:, -1:, -1:], logp_part
      )
    else:
      raise ValueError(f'invalid {method=}')

    inverted_vorticity = state.vorticity
    inverted_tracers = state.tracers

    return State(
        inverted_vorticity,
        inverted_divergence,
        inverted_temperature_variation,
        inverted_log_surface_pressure,
        inverted_tracers,
    )


@dataclasses.dataclass
class PrimitiveEquationsWithTime(PrimitiveEquations):


  def _time_and_state(
      self, state_with_time: StateWithTime
  ) -> tuple[float, State]:
    state_with_time_dict = state_with_time.asdict()
    sim_time = state_with_time_dict.pop('sim_time')
    state_without_time = State(**state_with_time_dict)
    return sim_time, state_without_time

  def explicit_terms(self, state: StateWithTime) -> StateWithTime:

    _, state_without_time = self._time_and_state(state)
    explicit_terms = super().explicit_terms(state_without_time)
    return StateWithTime(**explicit_terms.asdict(), sim_time=1.0)

  def implicit_terms(self, state: StateWithTime) -> StateWithTime:

    _, state_without_time = self._time_and_state(state)
    implicit_terms = super().implicit_terms(state_without_time)
    return StateWithTime(**implicit_terms.asdict(), sim_time=0.0)

  def implicit_inverse(
      self,
      state: StateWithTime,
      step_size: float,
  ) -> StateWithTime:

    sim_time, state_without_time = self._time_and_state(state)
    inverted = super().implicit_inverse(state_without_time, step_size)
    return StateWithTime(**inverted.asdict(), sim_time=sim_time)


@dataclasses.dataclass
class MoistPrimitiveEquations(PrimitiveEquationsWithTime):


  def _get_specific_humidity(self, aux_state: DiagnosticState) -> Array:

    if 'specific_humidity' not in aux_state.tracers:
      raise ValueError(
          '`specific_humidity` is not found in tracers: '
          f'{aux_state.tracers.keys()}.'
      )
    return aux_state.tracers['specific_humidity']

  def _virtual_temperature(self, aux_state, moisture_contribution):

    return (
        self.physics_specs.R
        * aux_state.temperature_variation
        * (1 + moisture_contribution)
    )

  @jax.named_call
  def curl_and_div_tendencies(
      self,
      aux_state: DiagnosticState,
  ) -> tuple[Array, Array]:

    gas_const_ratio = self.physics_specs.R_vapor / self.physics_specs.R
    q = self._get_specific_humidity(aux_state)
    moisture_contribution = (gas_const_ratio - 1) * q
    sec2_lat = self.coords.horizontal.sec2_lat

    u, v = aux_state.cos_lat_u
    total_vorticity = aux_state.vorticity + self.coriolis_parameter

    nodal_vorticity_u = -v * total_vorticity * sec2_lat
    nodal_vorticity_v = u * total_vorticity * sec2_lat

    d𝜎_dt = aux_state.sigma_dot_full
    if self.include_vertical_advection:

      sigma_dot_u = -self._vertical_tendency(d𝜎_dt, u)
      sigma_dot_v = -self._vertical_tendency(d𝜎_dt, v)
    else:
      sigma_dot_u = 0
      sigma_dot_v = 0

    rTv = self._virtual_temperature(
        aux_state, moisture_contribution
    )
    grad_log_ps_u, grad_log_ps_v = aux_state.cos_lat_grad_log_sp
    vertical_term_u = (sigma_dot_u + rTv * grad_log_ps_u) * sec2_lat
    vertical_term_v = (sigma_dot_v + rTv * grad_log_ps_v) * sec2_lat
    combined_u = self.coords.horizontal.to_modal(
        nodal_vorticity_u + vertical_term_u
    )
    combined_v = self.coords.horizontal.to_modal(
        nodal_vorticity_v + vertical_term_v
    )

    dζ_dt = -self.coords.horizontal.curl_cos_lat(
        (combined_u, combined_v), clip=False
    )
    d𝛅_dt = -self.coords.horizontal.div_cos_lat(
        (combined_u, combined_v), clip=False
    )
    return (dζ_dt, d𝛅_dt)

  @jax.named_call
  def nodal_temperature_adiabatic_tendency(
      self, aux_state: DiagnosticState
  ) -> Array:

    gas_const_ratio = self.physics_specs.R_vapor / self.physics_specs.R
    heat_capacity_ratio = self.physics_specs.Cp_vapor / self.physics_specs.Cp
    g_explicit = aux_state.u_dot_grad_log_sp
    g_full = g_explicit + aux_state.divergence
    q = self._get_specific_humidity(aux_state)
    mean_t_part = self._t_omega_over_sigma_sp(
        self.T_ref, g_explicit, aux_state.u_dot_grad_log_sp
    )


    variation_temperature_component = aux_state.temperature_variation * (
        (1 + (gas_const_ratio - 1) * q) / (1 + (heat_capacity_ratio - 1) * q)
    )
    humidity_reference_component = self.T_ref * (
        ((gas_const_ratio - heat_capacity_ratio) * q)
        / (1 + (heat_capacity_ratio - 1) * q)
    )
    variation_and_humidity_terms = (
        variation_temperature_component + humidity_reference_component
    )
    variation_and_Tv_part = self._t_omega_over_sigma_sp(
        variation_and_humidity_terms, g_full, aux_state.u_dot_grad_log_sp
    )
    return self.physics_specs.kappa * (mean_t_part + variation_and_Tv_part)

  @jax.named_call
  def divergence_tendency_due_to_humidity(
      self,
      state: State,
      aux_state: DiagnosticState,
  ) -> Array:

    method = self.vertical_matmul_method
    if method is None:
      mesh = self.coords.spmd_mesh
      method = 'sparse' if mesh is not None and mesh.shape['z'] > 1 else 'dense'

    q = self._get_specific_humidity(aux_state)
    physics_specs = self.physics_specs


    nodal_laplacian_lsp = self.coords.horizontal.to_nodal(
        self.coords.horizontal.laplacian(state.log_surface_pressure)
    )
    nodal_laplacian_correction_term = (
        q
        * nodal_laplacian_lsp
        * self.T_ref
        * (physics_specs.R_vapor - physics_specs.R)
    )


    q_modal = self._get_specific_humidity(state)
    cos_lat_grad_q = self.coords.horizontal.cos_lat_grad(q_modal, clip=False)
    nodal_cos_lat_grad_q = self.coords.horizontal.to_nodal(cos_lat_grad_q)
    coefficient = self.T_ref * (physics_specs.R_vapor - physics_specs.R)
    nodal_dot_term = (
        coefficient
        * self.coords.horizontal.sec2_lat
        * (
            nodal_cos_lat_grad_q[0] * aux_state.cos_lat_grad_log_sp[0]
            + nodal_cos_lat_grad_q[1] * aux_state.cos_lat_grad_log_sp[1]
        )
    )


    temperature = aux_state.temperature_variation + self.T_ref
    temperature_diff = (
        q * temperature * (physics_specs.R_vapor / physics_specs.R - 1)
    )
    geopotential_diff = get_geopotential_diff(
        temperature_diff,
        self.coords.vertical,
        physics_specs.R,
        method=method,
        sharding=self.coords.dycore_sharding,
    )

    return -self.coords.horizontal.laplacian(
        self.coords.horizontal.to_modal(geopotential_diff)
    ) - self.coords.horizontal.to_modal(
        nodal_dot_term + nodal_laplacian_correction_term
    )

  @jax.named_call
  def vorticity_tendency_due_to_humidity(
      self,
      state: State,
      aux_state: DiagnosticState,
  ) -> Array:
    physics_specs = self.physics_specs
    q_modal = self._get_specific_humidity(state)
    cos_lat_grad_q = self.coords.horizontal.cos_lat_grad(q_modal, clip=False)
    nodal_cos_lat_grad_q = self.coords.horizontal.to_nodal(cos_lat_grad_q)
    nodal_cos_lat_grad_log_sp = aux_state.cos_lat_grad_log_sp
    coefficient = self.T_ref * (physics_specs.R_vapor - physics_specs.R)
    nodal_curl_term = (
        coefficient
        * self.coords.horizontal.sec2_lat
        * (
            nodal_cos_lat_grad_log_sp[0] * nodal_cos_lat_grad_q[1]
            - nodal_cos_lat_grad_log_sp[1] * nodal_cos_lat_grad_q[0]
        )
    )
    return self.coords.horizontal.to_modal(nodal_curl_term)

  @jax.named_call
  def explicit_terms(self, state: StateWithTime) -> StateWithTime:

    _, state_without_time = self._time_and_state(state)
    aux_state = compute_diagnostic_state(state_without_time, self.coords)

    vorticity_dot, divergence_dot = self.curl_and_div_tendencies(aux_state)
    humidity_vort_correction_tendency = self.vorticity_tendency_due_to_humidity(
        state_without_time, aux_state
    )
    kinetic_energy_tendency = self.kinetic_energy_tendency(aux_state)
    orography_tendency = self.orography_tendency()
    humidity_div_correction_tendency = self.divergence_tendency_due_to_humidity(
        state_without_time, aux_state
    )
    horizontal_tendency_fn = functools.partial(
        self.horizontal_scalar_advection, aux_state=aux_state
    )
    dT_dt_horizontal_nodal, dT_dt_horizontal_modal = horizontal_tendency_fn(
        aux_state.temperature_variation
    )
    tracers_horizontal_nodal_and_modal = jax.tree_util.tree_map(
        horizontal_tendency_fn, aux_state.tracers
    )

    dT_dt_vertical = self.nodal_temperature_vertical_tendency(aux_state)
    dT_dt_adiabatic = self.nodal_temperature_adiabatic_tendency(aux_state)
    log_sp_tendency = self.nodal_log_pressure_tendency(aux_state)
    sigma_dot_full = aux_state.sigma_dot_full
    if self.include_vertical_advection:
      vertical_tendency_fn = functools.partial(
          self._vertical_tendency, sigma_dot_full
      )
    else:
      vertical_tendency_fn = lambda x: 0
    tracers_vertical_nodal = jax.tree_util.tree_map(
        vertical_tendency_fn, aux_state.tracers
    )

    to_modal_fn = self.coords.horizontal.to_modal
    vorticity_tendency = vorticity_dot + humidity_vort_correction_tendency
    divergence_tendency = (
        divergence_dot
        + kinetic_energy_tendency
        + orography_tendency
        + humidity_div_correction_tendency
    )
    temperature_tendency = (
        to_modal_fn(dT_dt_horizontal_nodal + dT_dt_vertical + dT_dt_adiabatic)
        + dT_dt_horizontal_modal
    )
    log_surface_pressure_tendency = to_modal_fn(log_sp_tendency)
    tracers_tendency = jax.tree_util.tree_map(
        lambda x, y_z: to_modal_fn(x + y_z[0]) + y_z[1],
        tracers_vertical_nodal,
        tracers_horizontal_nodal_and_modal,
    )
    explicit_terms = State(
        vorticity=vorticity_tendency,
        divergence=divergence_tendency,
        temperature_variation=temperature_tendency,
        log_surface_pressure=log_surface_pressure_tendency,
        tracers=tracers_tendency,
    )


    explicit_terms = self.coords.horizontal.clip_wavenumbers(explicit_terms)
    return StateWithTime(**explicit_terms.asdict(), sim_time=1.0)


@dataclasses.dataclass
class MoistPrimitiveEquationsWithCloudMoisture(MoistPrimitiveEquations):


  def _virtual_temperature(self, aux_state, moisture_contribution):

    return (
        self.physics_specs.R
        * aux_state.temperature_variation
        * (
            1
            + moisture_contribution
            - self._get_cloud_water(aux_state)
            - self._get_cloud_ice(aux_state)
        )
    )

  def _get_cloud_water(self, aux_state: DiagnosticState) -> Array:

    if 'specific_cloud_liquid_water_content' not in aux_state.tracers:
      raise ValueError(
          '`specific_cloud_liquid_water_content` is not found in tracers: '
          f'{aux_state.tracers.keys()}.'
      )
    return aux_state.tracers['specific_cloud_liquid_water_content']

  def _get_cloud_ice(self, aux_state: DiagnosticState) -> Array:

    if 'specific_cloud_ice_water_content' not in aux_state.tracers:
      raise ValueError(
          '`specific_cloud_ice_water_content` is not found in tracers: '
          f'{aux_state.tracers.keys()}.'
      )
    return aux_state.tracers['specific_cloud_ice_water_content']


