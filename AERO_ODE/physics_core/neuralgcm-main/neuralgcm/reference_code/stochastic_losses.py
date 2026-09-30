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
import abc
from typing import Callable, Optional, Sequence
from dinosaur import typing
import gin
import jax
import jax.numpy as jnp
from neuralgcm import model_utils
import numpy as np

import linear_transforms
import metrics_base
import metrics_util


Pytree = typing.Pytree
TrajectoryRepresentations = typing.TrajectoryRepresentations

AggregationTransformConstructor = metrics_util.AggregationTransformConstructor

tree_leaves = jax.tree_util.tree_leaves
tree_map = jax.tree_util.tree_map


def replicate(
    x: Pytree,
    axis_name: str = 'batch',
    times: Optional[int] = None,
) -> Pytree:

  if times is None:
    times = jax.local_device_count()

  def _replicate(_):
    return x

  return jax.pmap(_replicate, axis_name)(np.ones(times))


class EnergyLikeLoss(metrics_base.Loss, abc.ABC):


  def __init__(
      self,
      trajectory_spec: metrics_util.TrajectorySpec,
      components: Sequence[linear_transforms.LinearTransformConstructor],
      time_step: Optional[int | slice] = None,
      level: Optional[int] = None,
      getter: Callable[[Pytree], Pytree] = (
          metrics_util.filter_sim_time_and_diagnostics
      ),
      beta: float = 1.0,
      ensemble_term_weight: float = 0.5,
      is_nodal: bool = True,
      is_encoded: bool = False,
      coarsen_aggregation: AggregationTransformConstructor = (
          metrics_util.AggregateIdentity),
      vector_norm_squared_aggregation: AggregationTransformConstructor = (
          metrics_util.AggregateIdentity),
  ):

    self.coarsen_fn = coarsen_aggregation(
        trajectory_spec, is_nodal=is_nodal, is_encoded=is_encoded)
    self.vector_norm_squared_fn = vector_norm_squared_aggregation(
        self.coarsen_fn.out_trajectory_spec, is_nodal=is_nodal,
        is_encoded=is_encoded)

    super().__init__(
        self.vector_norm_squared_fn.out_trajectory_spec,
        is_nodal=is_nodal, is_encoded=is_encoded)
    self.components = components
    self.time_step = time_step
    self.level = level
    self.getter = getter

    self.transform = linear_transforms.ComposedTransformForLoss(
        trajectory_spec, self.components
    )
    self._beta = beta
    self._ensemble_term_weight = ensemble_term_weight

  def a_minus_cb(self, a: Pytree, c: float, b: Pytree) -> Pytree:

    return tree_map(lambda a_i, b_i: a_i - c * b_i, a, b)

  def ca_minus_b(self, c: float, a: Pytree, b: Pytree) -> Pytree:

    return tree_map(lambda a_i, b_i: c * a_i - b_i, a, b)

  def component_mean(self, tree: Pytree) -> jax.Array:

    leaf_means = tree_leaves(self.mean_per_variable(tree))
    return sum(leaf_means) / len(leaf_means)

  def ensemble_mean(self, tree: Pytree) -> Pytree:
    return jax.lax.pmean(tree, 'ensemble')

  def _prepare(self, trajectory: TrajectoryRepresentations) -> Pytree:


    trajectory = metrics_util.extract_variable(
        trajectory,
        self.trajectory_spec,
        self.time_step,
        self.level,
        self.getter,
        self.is_nodal,
        self.is_encoded,
    )
    return trajectory

  def evaluate(
      self,
      prediction: TrajectoryRepresentations,
      target: TrajectoryRepresentations,
  ) -> Pytree:

    pv2ss = self._per_variable_spread_skill_errors(prediction, target)
    return self._spread_skill_and_loss(
        x_minus_y=pv2ss['x_minus_y'],
        x_minus_xprime=pv2ss['x_minus_xprime'],
    )['loss']

  def debug_loss_terms_instance(self) -> metrics_base.EvaluateFunctionWrapper:


    def evaluate_fn(
        prediction: TrajectoryRepresentations,
        target: TrajectoryRepresentations,
    ) -> Pytree:


      pv2ss = self._per_variable_spread_skill_errors(prediction, target)
      overall_spread_skill_loss = self._spread_skill_and_loss(
          x_minus_y=pv2ss['x_minus_y'],
          x_minus_xprime=pv2ss['x_minus_xprime'],
      )

      all_vars = pv2ss['x_minus_y'].keys()

      per_variable_terms = {
          var: self._spread_skill_and_loss(
              x_minus_y=pv2ss['x_minus_y'][var],
              x_minus_xprime=pv2ss['x_minus_xprime'][var],
          )
          for var in all_vars
      }


      per_variable_losses = {
          var: per_variable_terms[var]['loss'] for var in all_vars
      }
      sum_of_losses = sum(per_variable_losses.values())
      per_variable_relative_losses = tree_map(
          lambda x: x / sum_of_losses, per_variable_losses
      )
      return {
          'relative_loss': per_variable_relative_losses,
          'overall': overall_spread_skill_loss,
          'per_variable_spread': {
              var: per_variable_terms[var]['spread'] for var in all_vars
          },
          'per_variable_skill': {
              var: per_variable_terms[var]['skill'] for var in all_vars
          },
      }

    return metrics_base.EvaluateFunctionWrapper(evaluate_fn)

  def _per_variable_spread_skill_errors(
      self,
      prediction: TrajectoryRepresentations,
      target: TrajectoryRepresentations,
  ) -> Pytree:

    ensemble_size = jax.lax.psum(1, 'ensemble')
    if ensemble_size != 2:
      raise ValueError(f'{ensemble_size=} is not 2')

    prediction = self.transform(self._prepare(prediction), target)
    target = self.transform(self._prepare(target), target)

    x_minus_y = tree_map(jnp.subtract, prediction, target)

    xprime = jax.lax.pshuffle(prediction, 'ensemble', (1, 0))
    x_minus_xprime = tree_map(jnp.subtract, prediction, xprime)

    return {
        'x_minus_y': x_minus_y,
        'x_minus_xprime': x_minus_xprime,
        'prediction': prediction,
    }

  @abc.abstractmethod
  def _spread_skill_and_loss(
      self,
      x_minus_y: Pytree,
      x_minus_xprime: Pytree,
  ) -> dict[str, jax.Array]:
    pass


@gin.register(
    denylist=['coarsen_aggregation', 'vector_norm_squared_aggregation'])
class CRPSLoss(EnergyLikeLoss):


  def _spread_skill_and_loss(
      self,
      x_minus_y: Pytree,
      x_minus_xprime: Pytree,
  ) -> dict[str, jax.Array]:

    a_minus_cb = self.a_minus_cb
    ensemble_mean = self.ensemble_mean
    component_mean = self.component_mean

    def abs_beta(tree: Pytree) -> Pytree:
      return tree_map(lambda x: jnp.abs(x) ** self._beta, tree)


    skill = component_mean(ensemble_mean(abs_beta(x_minus_y)))


    spread = component_mean(ensemble_mean(abs_beta(x_minus_xprime)))


    crps = component_mean(
        ensemble_mean(
            a_minus_cb(
                abs_beta(x_minus_y),
                self._ensemble_term_weight,
                abs_beta(x_minus_xprime),
            )
        )
    )
    return {'spread': spread, 'skill': skill, 'loss': crps}


@gin.register(
    denylist=['coarsen_aggregation', 'vector_norm_squared_aggregation'])
class EnergyScoreLoss(EnergyLikeLoss):


  def _spread_skill_and_loss(
      self,
      x_minus_y: Pytree,
      x_minus_xprime: Pytree,
  ) -> dict[str, jax.Array]:

    a_minus_cb = self.a_minus_cb
    ensemble_mean = self.ensemble_mean
    component_mean = self.component_mean

    def sqrt_beta(x: jax.Array) -> jax.Array:
      return model_utils.safe_sqrt(x) ** self._beta

    def square(tree: Pytree) -> Pytree:
      return tree_map(jnp.square, tree)


    skill = ensemble_mean(sqrt_beta(component_mean(square(x_minus_y))))


    spread = ensemble_mean(sqrt_beta(component_mean(square(x_minus_xprime))))


    es_straightforward = a_minus_cb(skill, self._ensemble_term_weight, spread)
    es = es_straightforward


    return {'spread': spread, 'skill': skill, 'loss': es}

