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
from typing import Sequence, Union
from dinosaur import primitive_equations
from dinosaur import scales
from dinosaur import shallow_water
import gin
import numpy as np


QuantityOrStr = Union[str, scales.Quantity]


GET_DEFAULT_SCALE = gin.external_configurable(
    lambda: scales.DEFAULT_SCALE, name='GET_DEFAULT_SCALE')
GET_ATMOSPHERIC_SCALE = gin.external_configurable(
    lambda: scales.ATMOSPHERIC_SCALE, name='GET_ATMOSPHERIC_SCALE')


@gin.configurable
def get_physics_specs(construct_fn=gin.REQUIRED):

  return construct_fn()


@gin.register
def shallow_water_specs_constructor(
    density_vals: Union[Sequence[float], np.ndarray],
    density_units: QuantityOrStr = scales.WATER_DENSITY,
    radius_si: QuantityOrStr = scales.RADIUS,
    angular_velocity_si: QuantityOrStr = scales.ANGULAR_VELOCITY,
    gravity_acceleration_si: QuantityOrStr = scales.GRAVITY_ACCELERATION,
    scale: scales.Scale = scales.DEFAULT_SCALE
) -> shallow_water.ShallowWaterSpecs:

  densities = np.asarray(density_vals) * scales.Quantity(density_units)
  return shallow_water.ShallowWaterSpecs.from_si(
      densities=densities,
      radius_si=scales.Quantity(radius_si),
      angular_velocity_si=scales.Quantity(angular_velocity_si),
      gravity_acceleration_si=scales.Quantity(gravity_acceleration_si),
      scale=scale)


@gin.register
def primitive_eq_specs_constructor(
    radius_si: QuantityOrStr = scales.RADIUS,
    angular_velocity_si: QuantityOrStr = scales.ANGULAR_VELOCITY,
    gravity_acceleration_si: QuantityOrStr = scales.GRAVITY_ACCELERATION,
    ideal_gas_constant_si: QuantityOrStr = scales.IDEAL_GAS_CONSTANT,
    water_vapor_gas_constant_si: QuantityOrStr = scales.IDEAL_GAS_CONSTANT_H20,
    water_vapor_isobaric_heat_capacity_si: QuantityOrStr = (
        scales.WATER_VAPOR_CP),
    kappa_si: QuantityOrStr = scales.KAPPA,
    scale: scales.Scale = scales.DEFAULT_SCALE,
) -> primitive_equations.PrimitiveEquationsSpecs:

  return primitive_equations.PrimitiveEquationsSpecs.from_si(
      radius_si=scales.Quantity(radius_si),
      angular_velocity_si=scales.Quantity(angular_velocity_si),
      gravity_acceleration_si=scales.Quantity(gravity_acceleration_si),
      ideal_gas_constant_si=scales.Quantity(ideal_gas_constant_si),
      water_vapor_gas_constant_si=scales.Quantity(water_vapor_gas_constant_si),
      water_vapor_isobaric_heat_capacity_si=scales.Quantity(
          water_vapor_isobaric_heat_capacity_si),
      kappa_si=scales.Quantity(kappa_si),
      scale=scale)
