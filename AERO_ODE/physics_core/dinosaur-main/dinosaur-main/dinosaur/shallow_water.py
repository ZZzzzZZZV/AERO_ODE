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
from typing import Callable, Sequence, Type

from dinosaur import coordinate_systems
from dinosaur import scales
from dinosaur import spherical_harmonic
from dinosaur import time_integration
from dinosaur import typing

import jax
import jax.numpy as jnp
import numpy as np
import tree_math

units = scales.units

Array = typing.Array
Numeric = typing.Numeric
Quantity = typing.Quantity

FilterFn = typing.FilterFn
InverseFn = typing.InverseFn
StateFn = typing.StateFn
StepFn = typing.StepFn

SCALE = scales.DEFAULT_SCALE


einsum = functools.partial(jnp.einsum, precision=jax.lax.Precision.HIGHEST)


@tree_math.struct
class State:

  vorticity: Array
  divergence: Array
  potential: Array


@dataclasses.dataclass(frozen=True)
class ShallowWaterSpecs:

  densities: Array
  radius: float
  angular_velocity: float
  gravity_acceleration: float
  scale: scales.Scale

  @property
  def g(self) -> float:

    return self.gravity_acceleration

  @property
  def num_layers(self) -> int:

    return self.densities.size

  def nondimensionalize(self, quantity: Quantity) -> Numeric:

    return self.scale.nondimensionalize(quantity)

  def dimensionalize(self, value: Numeric, unit: units.Unit) -> Quantity:

    return self.scale.dimensionalize(value, unit)

  @classmethod
  def from_si(
      cls: Type[ShallowWaterSpecs],
      densities: Quantity = np.ones(1) * scales.WATER_DENSITY,
      radius_si: Quantity = scales.RADIUS,
      angular_velocity_si: Quantity = scales.ANGULAR_VELOCITY,
      gravity_acceleration_si: Quantity = scales.GRAVITY_ACCELERATION,
      scale: scales.Scale = scales.DEFAULT_SCALE) -> ShallowWaterSpecs:

    return cls(scale.nondimensionalize(densities),
               scale.nondimensionalize(radius_si),
               scale.nondimensionalize(angular_velocity_si),
               scale.nondimensionalize(gravity_acceleration_si),
               scale)


def state_to_nodal(state: State, grid: spherical_harmonic.Grid) -> State:

  return jax.tree.map(
      lambda x: grid.to_nodal(grid.clip_wavenumbers(x)), state)


def state_to_modal(state: State, grid: spherical_harmonic.Grid) -> State:

  return jax.tree.map(grid.to_modal, state)


def get_density_ratios(density: Array) -> np.ndarray:

  ratios = np.minimum(density / density[..., np.newaxis], 1)
  np.fill_diagonal(ratios, 0)
  return ratios


def get_coriolis(grid: spherical_harmonic.Grid) -> np.ndarray:

  _, sin_lat = grid.nodal_mesh
  return sin_lat


@dataclasses.dataclass
class ShallowWaterEquations(time_integration.ImplicitExplicitODE):

  coords: coordinate_systems.CoordinateSystem
  physics_specs: ShallowWaterSpecs
  orography: Array
  reference_potential: Array

  @property
  def coriolis_parameter(self) -> Array:

    _, sin_lat = self.coords.horizontal.nodal_mesh
    return 2 * self.physics_specs.angular_velocity * sin_lat

  @property
  def density_ratios(self) -> Array:

    return get_density_ratios(self.physics_specs.densities)

  @property
  def ref_potential(self) -> Array:

    return self.reference_potential[..., np.newaxis, np.newaxis]

  def explicit_terms(self, state: State) -> State:


    u = jnp.stack(spherical_harmonic.get_cos_lat_vector(
        state.vorticity, state.divergence, self.coords.horizontal))


    nodal_u = self.coords.horizontal.to_nodal(u)
    nodal_state = state_to_nodal(state, self.coords.horizontal)

    total_vorticity = nodal_state.vorticity + self.coriolis_parameter

    sec2_lat = self.coords.horizontal.sec2_lat
    nodal_b = nodal_u * total_vorticity * sec2_lat
    nodal_g = nodal_u * nodal_state.potential * sec2_lat
    nodal_e = (nodal_u * nodal_u).sum(0) * sec2_lat / 2


    bge_nodal = jnp.concatenate(
        [nodal_b, nodal_g, jnp.expand_dims(nodal_e, axis=0)], axis=0)
    bge = self.coords.horizontal.to_modal(bge_nodal)
    b, g, e = jnp.split(bge, [2, 4], axis=0)
    e = jnp.squeeze(e, axis=0)


    p = einsum('ab,...bml->...aml',
               self.density_ratios,
               state.potential)
    if self.orography is not None:
      p = p + self.orography

    explicit_vorticity = self.coords.horizontal.clip_wavenumbers(
        -self.coords.horizontal.div_cos_lat(b)
    )
    explicit_divergence = self.coords.horizontal.clip_wavenumbers(
        -self.coords.horizontal.laplacian(p + e) +
        self.coords.horizontal.curl_cos_lat(b)
    )
    explicit_potential = self.coords.horizontal.clip_wavenumbers(
        -self.coords.horizontal.div_cos_lat(g)
    )
    return State(explicit_vorticity, explicit_divergence, explicit_potential)

  def implicit_terms(self, state: State) -> State:

    return State(
        vorticity=jnp.zeros_like(state.vorticity),
        divergence=-self.coords.horizontal.laplacian(state.potential),
        potential=-self.ref_potential * state.divergence)

  def implicit_inverse(self, state: State, step_size: float) -> State:

    inverse_schur_complement = 1 / (
        1 - step_size ** 2 * self.ref_potential *
        self.coords.horizontal.laplacian_eigenvalues)
    return State(
        vorticity=state.vorticity,
        divergence=inverse_schur_complement * (
            state.divergence -
            step_size * self.coords.horizontal.laplacian(state.potential)
        ),
        potential=inverse_schur_complement * (
            -step_size * self.ref_potential * state.divergence + state.potential
        )
    )


def shallow_water_leapfrog_step(
    coords: coordinate_systems.CoordinateSystem,
    dt: float,
    physics_specs: ShallowWaterSpecs,
    mean_potential: np.ndarray,
    orography: Array | None = None,
    alpha: float = 0.5
) -> typing.TimeStepFn:

  shallow_water_ode = ShallowWaterEquations(
      coords, physics_specs, orography, mean_potential)
  return time_integration.semi_implicit_leapfrog(shallow_water_ode, dt, alpha)


def shallow_water_leapfrog_trajectory(
    coords: coordinate_systems.CoordinateSystem,
    dt: float,
    physics_specs: ShallowWaterSpecs,
    inner_steps: int,
    outer_steps: int,
    mean_potential: np.ndarray,
    orography: Array | None = None,
    filters: Sequence[typing.PyTreeStepFilterFn] = (),
    alpha: float = 0.5,
) -> typing.TrajectoryFn:

  step_fn = shallow_water_leapfrog_step(
      coords, dt, physics_specs, mean_potential, orography, alpha)
  step_fn = time_integration.step_with_filters(step_fn, filters)
  post_process_fn = lambda x: x[0]
  trajectory_fn = time_integration.trajectory_from_step(
      step_fn, outer_steps, inner_steps, post_process_fn=post_process_fn)
  return trajectory_fn


def default_filters(
    grid: spherical_harmonic.Grid,
    dt: float,
) -> Sequence[typing.PyTreeStepFilterFn]:

  return (
      time_integration.exponential_leapfrog_step_filter(grid, dt),
      time_integration.robert_asselin_leapfrog_filter(0.05),
  )
