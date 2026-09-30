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
from typing import Callable
from dinosaur import typing
import jax
import jax.numpy as jnp
import metrics_util


Pytree = typing.Pytree
TrajectoryRepresentations = typing.TrajectoryRepresentations

tree_leaves = jax.tree_util.tree_leaves
tree_map = jax.tree_util.tree_map


@dataclasses.dataclass
class Evaluator:


  def evaluate(
      self,
      prediction: TrajectoryRepresentations,
      target: TrajectoryRepresentations,
  ) -> Pytree:

    raise NotImplementedError()


@dataclasses.dataclass
class EvaluateFunctionWrapper(Evaluator):


  def __init__(
      self,
      evaluate_fn: Callable[
          [TrajectoryRepresentations, TrajectoryRepresentations], Pytree
      ],
  ):
    self._evaluate_fn = evaluate_fn

  def evaluate(
      self,
      prediction: TrajectoryRepresentations,
      target: TrajectoryRepresentations,
  ) -> Pytree:
    return self._evaluate_fn(prediction, target)


class MetricRuntimeError(Exception):
  pass


@dataclasses.dataclass
class Metric(Evaluator):


  trajectory_spec: metrics_util.TrajectorySpec
  is_nodal: bool = dataclasses.field(default=True, kw_only=True)
  is_encoded: bool = dataclasses.field(default=False, kw_only=True)

  def get_representation(self, x: TrajectoryRepresentations) -> Pytree:
    x_rep = x.get_representation(
        is_nodal=self.is_nodal, is_encoded=self.is_encoded
    )
    if x_rep is None:
      raise MetricRuntimeError(
          'Desired representation of `x` was None. '
          f'{self.is_nodal=}, {self.is_encoded=}'
      )
    return x_rep

  def surface_mean(self, trajectory: Pytree) -> Pytree:
    if self.is_encoded:
      coords = self.trajectory_spec.coords
    else:
      coords = self.trajectory_spec.data_coords
    if self.is_nodal:


      fn = lambda x: metrics_util.nodal_surface_mean(x, coords)
    else:
      fn = lambda x: metrics_util.modal_surface_mean(x, coords)
    return tree_map(fn, trajectory)

  def mean_per_variable(self, trajectory: Pytree) -> Pytree:

    return tree_map(jnp.mean, self.surface_mean(trajectory))


class ScalarMetric(Metric):
  pass


@dataclasses.dataclass
class Loss(ScalarMetric):


  trajectory_spec: metrics_util.TrajectorySpec
  is_nodal: bool = dataclasses.field(default=True, kw_only=True)
  is_encoded: bool = dataclasses.field(default=False, kw_only=True)
  time_step: int | slice | None = dataclasses.field(default=None, kw_only=True)

  def evaluate_per_variable(
      self,
      prediction: TrajectoryRepresentations,
      target: TrajectoryRepresentations,
  ) -> Pytree:
    raise NotImplementedError()

  def evaluate(
      self,
      prediction: TrajectoryRepresentations,
      target: TrajectoryRepresentations,
  ) -> jnp.ndarray:
    error_per_variable = self.evaluate_per_variable(prediction, target)
    return sum(tree_leaves(error_per_variable))

  def debug_loss_terms_instance(self) -> EvaluateFunctionWrapper:


    def evaluate_fn(
        prediction: TrajectoryRepresentations,
        target: TrajectoryRepresentations,
    ) -> Pytree:


      loss_per_variable = self.evaluate_per_variable(prediction, target)


      sum_of_all_terms = sum(tree_leaves(loss_per_variable))
      relative_loss = tree_map(
          lambda x: x / sum_of_all_terms, loss_per_variable
      )
      return {'relative_loss': relative_loss}

    return EvaluateFunctionWrapper(evaluate_fn)
