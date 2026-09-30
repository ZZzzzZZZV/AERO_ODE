# Copyright 2024 Google LLC

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
from typing import Any, Callable, Generic, TypeVar

import jax
from neuralgcm.experimental import scales
import numpy as np
import tree_math


Dtype = jax.typing.DTypeLike | Any
Array = np.ndarray | jax.Array
Numeric = float | int | Array
Timestep = np.timedelta64 | float
PRNGKeyArray = jax.Array
units = scales.units
Quantity = scales.Quantity


PyTreeState = TypeVar('PyTreeState')
Pytree = Any


@tree_math.struct
class ModelState(Generic[PyTreeState]):


  prognostics: PyTreeState
  diagnostics: Pytree = dataclasses.field(default_factory=dict)
  randomness: Pytree = dataclasses.field(default_factory=dict)


@jax.tree_util.register_pytree_node_class
@dataclasses.dataclass
class Randomness:


  prng_key: jax.Array
  prng_step: int = 0
  core: Pytree = None

  def tree_flatten(self):

    leaves = (self.prng_key, self.prng_step, self.core)
    aux_data = ()
    return leaves, aux_data

  @classmethod
  def tree_unflatten(cls, aux_data, leaves):

    return cls(*leaves, *aux_data)


@jax.tree_util.register_pytree_node_class
@dataclasses.dataclass
class Timedelta:


  days: Numeric = 0
  seconds: Numeric = 0


  def __post_init__(self):
    days_delta, seconds = divmod(self.seconds, 24 * 60 * 60)
    self.days = self.days + days_delta
    self.seconds = seconds

  @classmethod
  def from_timedelta64(cls, values: np.timedelta64 | np.ndarray) -> Timedelta:
    seconds = values // np.timedelta64(1, 's')


    return Timedelta(0, seconds)

  def to_timedelta64(self) -> np.timedelta64 | np.ndarray:
    seconds = np.int64(self.days) * 24 * 60 * 60 + np.int64(self.seconds)
    return seconds * np.timedelta64(1, 's')

  def __add__(self, other):
    if not isinstance(other, Timedelta):
      return NotImplemented
    days = self.days + other.days
    seconds = self.seconds + other.seconds
    return Timedelta(days, seconds)

  def __neg__(self):
    return Timedelta(-self.days, -self.seconds)

  def __sub__(self, other):
    if not isinstance(other, Timedelta):
      return NotImplemented
    return self + (-other)

  def __mul__(self, other):
    if not isinstance(other, Numeric):
      return NotImplemented
    return Timedelta(self.days * other, self.seconds * other)

  __rmul__ = __mul__


  def tree_flatten(self):
    leaves = (self.days, self.seconds)
    aux_data = None
    return leaves, aux_data

  @classmethod
  def tree_unflatten(cls, aux_data, leaves):
    assert aux_data is None


    result = object.__new__(cls)
    result.days, result.seconds = leaves
    return result


_UNIX_EPOCH = np.datetime64('1970-01-01T00:00:00', 's')


@jax.tree_util.register_pytree_node_class
@dataclasses.dataclass
class Timestamp:


  delta: Timedelta

  @classmethod
  def from_datetime64(cls, values: np.datetime64 | np.ndarray) -> Timestamp:
    return cls(Timedelta.from_timedelta64(values - _UNIX_EPOCH))

  def to_datetime64(self) -> np.timedelta64 | np.ndarray:
    return self.delta.to_timedelta64() + _UNIX_EPOCH

  def __add__(self, other):
    if not isinstance(other, Timedelta):
      return NotImplemented
    return Timestamp(self.delta + other)

  __radd__ = __add__

  def __sub__(self, other):
    if isinstance(other, Timestamp):
      return self.delta - other.delta
    elif isinstance(other, Timedelta):
      return Timestamp(self.delta - other)
    else:
      return NotImplemented

  def tree_flatten(self):
    leaves = (self.delta,)
    aux_data = None
    return leaves, aux_data

  @classmethod
  def tree_unflatten(cls, aux_data, leaves):
    assert aux_data is None
    return cls(*leaves)


PostProcessFn = Callable[..., Any]


@dataclasses.dataclass(eq=True, order=True, frozen=True)
class KeyWithCosLatFactor:


  name: str
  factor_order: int
