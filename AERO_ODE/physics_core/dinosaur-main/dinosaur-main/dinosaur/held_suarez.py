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

from dinosaur import coordinate_systems
from dinosaur import primitive_equations
from dinosaur import scales
from dinosaur import time_integration
from dinosaur import typing
import jax
import jax.numpy as jnp
import numpy as np

units = scales.units
Quantity = units.Quantity


class HeldSuarezForcing(time_integration.ExplicitODE):


  def __init__(
      self,
      coords: coordinate_systems.CoordinateSystem,
      physics_specs: primitive_equations.PrimitiveEquationsSpecs,
      reference_temperature: typing.Array,
      p0: Quantity = 1e5 * units.pascal,
      sigma_b: Quantity = 0.7,
      kf: Quantity = 1 / (1 * units.day),
      ka: Quantity = 1 / (40 * units.day),
      ks: Quantity = 1 / (4 * units.day),
      minT: Quantity = 200 * units.degK,
      maxT: Quantity = 315 * units.degK,
      dTy: Quantity = 60 * units.degK,
      dThz: Quantity = 10 * units.degK,
  ):

    self.coords = coords
    self.physics_specs = physics_specs
    self.reference_temperature = reference_temperature
    self.p0 = physics_specs.nondimensionalize(p0)
    self.sigma_b = sigma_b
    self.kf = physics_specs.nondimensionalize(kf)
    self.ka = physics_specs.nondimensionalize(ka)
    self.ks = physics_specs.nondimensionalize(ks)
    self.minT = physics_specs.nondimensionalize(minT)
    self.maxT = physics_specs.nondimensionalize(maxT)
    self.dTy = physics_specs.nondimensionalize(dTy)
    self.dThz = physics_specs.nondimensionalize(dThz)

    self.sigma = self.coords.vertical.centers
    _, sin_lat = self.coords.horizontal.nodal_mesh
    self.lat = np.arcsin(sin_lat)

  def kv(self):
    kv_coeff = self.kf * (
        np.maximum(0, (self.sigma - self.sigma_b) / (1 - self.sigma_b))
    )
    return kv_coeff[:, np.newaxis, np.newaxis]

  def kt(self):
    cutoff = np.maximum(0, (self.sigma - self.sigma_b) / (1 - self.sigma_b))
    return self.ka + (self.ks - self.ka) * (
        cutoff[:, np.newaxis, np.newaxis] * np.cos(self.lat) ** 4
    )

  def equilibrium_temperature(self, nodal_surface_pressure):
    p_over_p0 = (
        self.sigma[:, np.newaxis, np.newaxis] * nodal_surface_pressure / self.p0
    )
    temperature = p_over_p0**self.physics_specs.kappa * (
        self.maxT
        - self.dTy * np.sin(self.lat) ** 2
        - self.dThz * jnp.log(p_over_p0) * np.cos(self.lat) ** 2
    )
    return jnp.maximum(self.minT, temperature)

  def explicit_terms(
      self, state: primitive_equations.State
  ) -> primitive_equations.State:

    aux_state = primitive_equations.compute_diagnostic_state(
        state=state, coords=self.coords
    )


    nodal_velocity_tendency = jax.tree.map(
        lambda x: -self.kv() * x / self.coords.horizontal.cos_lat**2,
        aux_state.cos_lat_u,
    )


    nodal_temperature = (
        self.reference_temperature[:, np.newaxis, np.newaxis]
        + aux_state.temperature_variation
    )
    nodal_log_surface_pressure = self.coords.horizontal.to_nodal(
        state.log_surface_pressure
    )
    nodal_surface_pressure = jnp.exp(nodal_log_surface_pressure)
    Teq = self.equilibrium_temperature(nodal_surface_pressure)
    nodal_temperature_tendency = -self.kt() * (nodal_temperature - Teq)


    temperature_tendency = self.coords.horizontal.to_modal(
        nodal_temperature_tendency
    )
    velocity_tendency = self.coords.horizontal.to_modal(nodal_velocity_tendency)
    vorticity_tendency = self.coords.horizontal.curl_cos_lat(velocity_tendency)
    divergence_tendency = self.coords.horizontal.div_cos_lat(velocity_tendency)


    log_surface_pressure_tendency = jnp.zeros_like(state.log_surface_pressure)

    return primitive_equations.State(
        vorticity=vorticity_tendency,
        divergence=divergence_tendency,
        temperature_variation=temperature_tendency,
        log_surface_pressure=log_surface_pressure_tendency,
    )
