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
from typing import Any, Callable, Sequence, Tuple
from dinosaur import coordinate_systems
from dinosaur import pytree_utils
from dinosaur import typing
import haiku as hk
import jax
import jax.numpy as jnp

Array = typing.Array
DynamicalSystem = Any
Pytree = typing.Pytree

tree_map = jax.tree_util.tree_map
tree_leaves = jax.tree_util.tree_leaves


def with_preprocessing(
    fn: Callable[..., Pytree],
    preprocess_fn: Callable,
) -> Callable[..., Pytree]:

  @functools.wraps(fn)
  def apply_fn(x, *args, **kwargs):
    return fn(preprocess_fn(x), *args, **kwargs)

  return apply_fn


def with_post_processing(
    fn: Callable[..., Pytree],
    post_process_fn: Callable,
) -> Callable[..., Pytree]:

  @functools.wraps(fn)
  def apply_fn(*args, **kwargs):
    return post_process_fn(*fn(*args, **kwargs))

  return apply_fn


def with_forcing(
    fn: Callable[..., Pytree],
    forcing_fn: typing.ForcingFn,
    forcing_data: typing.ForcingData,
) -> Callable[..., Pytree]:


  @functools.wraps(fn)
  def wrapped(x, forcing_fn=forcing_fn):

    if dataclasses.is_dataclass(x):
      if isinstance(x, typing.ModelState):
        sim_time = dataclasses.asdict(x.state).get('sim_time', None)
      else:
        sim_time = dataclasses.asdict(x).get('sim_time', None)
    else:
      sim_time = x.get('sim_time', None)

    if sim_time is not None:
      sim_time = jax.numpy.asarray(sim_time)
      if sim_time.ndim:
        forcing_fn = jax.vmap(forcing_fn, in_axes=(None, 0))
    forcing = forcing_fn(forcing_data, sim_time)
    return fn(x, forcing=forcing)
  return wrapped


def with_split_input(
    fn: Callable[..., Pytree],
    split_index: int,
    time_axis: int = 0,
) -> Callable[..., Pytree]:

  @functools.wraps(fn)
  def apply_fn(x, *args, **kwargs):
    init, _ = pytree_utils.split_along_axis(x, split_index, axis=time_axis)
    return fn(init, *args, **kwargs)

  return apply_fn


def with_input_included(
    trajectory_fn: typing.TrajectoryFn,
    time_axis: int = 0,
    num_last_input_frames_to_trim: int = 0,
) -> typing.TrajectoryFn:

  if num_last_input_frames_to_trim > 0:
    num_last_input_frames_to_trim = -num_last_input_frames_to_trim
  else:
    num_last_input_frames_to_trim = None
  inputs_time_slice = slice(None, num_last_input_frames_to_trim)
  @functools.wraps(trajectory_fn)
  def _trajectory(x, *args, **kwargs):
    final, unroll = trajectory_fn(x, *args, **kwargs)
    x_concat = pytree_utils.slice_along_axis(x, time_axis, inputs_time_slice)
    return final, pytree_utils.concat_along_axis([x_concat, unroll], time_axis)

  return _trajectory


def MITgcm_with_inputs_and_forcing(
    model: DynamicalSystem,
    num_init_frames: int,
    start_with_input: bool = False,
) -> typing.TrajectoryFn:

  def _trajectory_fn(x, forcing_data, outer_steps, inner_steps=1):


    trajectory_fn = functools.partial(
        model.trajectory,
        outer_steps=outer_steps,
        inner_steps=inner_steps,
        forcing_data=forcing_data,
        start_with_input=start_with_input)

    encode_fn = with_forcing(model.encode, model.forcing_fn, forcing_data)
    trajectory_fn = with_preprocessing(trajectory_fn, encode_fn)
    trajectory_fn = with_split_input(trajectory_fn, num_init_frames)
    return trajectory_fn(x)

  return _trajectory_fn


def trajectory_with_inputs_and_forcing_and_stop_gradients(
    model: DynamicalSystem,
    num_init_frames: int,
    start_with_input: bool = False,
    stop_gradient_outer_steps: Sequence[int] = (),
) -> typing.TrajectoryFn:

  stop_gradient_outer_steps = list(sorted(stop_gradient_outer_steps))
  if num_init_frames != 1:
    raise ValueError(f'{num_init_frames=} is not supported yet.')

  if stop_gradient_outer_steps and min(stop_gradient_outer_steps) <= 0:
    raise ValueError(
        f'{stop_gradient_outer_steps=} contained non-positive values'
    )

  expand_dim0 = lambda tree: tree_map(lambda x_i: x_i[jnp.newaxis], tree)
  concat_dim0 = lambda trees: pytree_utils.concat_along_axis(trees, axis=0)
  slice_dim0 = lambda tree, idx: pytree_utils.slice_along_axis(
      tree, axis=0, idx=idx
  )

  def concat_trajectories_with_stop_grads(
      x, forcing_data, outer_steps, inner_steps=1
  ):
    if (
        stop_gradient_outer_steps
        and max(stop_gradient_outer_steps) > outer_steps
    ):
      raise ValueError(
          f'{stop_gradient_outer_steps=} contained values > {outer_steps=}'
      )
    outer_steps_seq = list(stop_gradient_outer_steps)
    if not outer_steps_seq or outer_steps_seq[-1] != outer_steps:
      outer_steps_seq.append(outer_steps)


    final_state, first_leg = trajectory_with_inputs_and_forcing(
        model,
        num_init_frames=num_init_frames,
        start_with_input=start_with_input,
    )(
        x,
        forcing_data=forcing_data,
        outer_steps=outer_steps_seq[0],
        inner_steps=inner_steps,
    )


    sections = [
        first_leg,
    ]


    trajectory_fn = functools.partial(
        model.trajectory,
        inner_steps=inner_steps,
        forcing_data=forcing_data,
        start_with_input=start_with_input,
    )
    for i in range(1, len(outer_steps_seq)):


      assert set(outer_steps_seq[:-1]).issubset(stop_gradient_outer_steps)
      stop_grad_at_start = outer_steps_seq[i - 1] in stop_gradient_outer_steps

      initial_state = final_state


      final_state, this_leg = trajectory_fn(
          jax.lax.stop_gradient(initial_state)
          if stop_grad_at_start
          else initial_state,
          outer_steps=outer_steps_seq[i] - outer_steps_seq[i - 1],
      )

      if stop_grad_at_start and start_with_input:

        this_leg = concat_dim0([
            expand_dim0(initial_state),
            slice_dim0(this_leg, idx=slice(1, None)),
        ])
      sections.append(this_leg)

    return final_state, concat_dim0(sections)

  return concat_trajectories_with_stop_grads


def decoded_trajectory_with_forcing(
    model: DynamicalSystem,
    start_with_input: bool = False,
) -> typing.TrajectoryFn:

  def _trajectory_fn(x, forcing_data, outer_steps, inner_steps=1):


    trajectory_fn = functools.partial(
        model.trajectory,
        forcing_data=forcing_data,
        post_process_fn=with_forcing(model.decode,
                                     model.forcing_fn, forcing_data),
        start_with_input=start_with_input)
    return trajectory_fn(x, outer_steps, inner_steps)

  return _trajectory_fn


def decoded_trajectory_with_inputs_and_forcing(
    model: DynamicalSystem,
    num_init_frames: int,
    start_with_input: bool = False,
) -> typing.TrajectoryFn:

  def _trajectory_fn(x, forcing_data, outer_steps, inner_steps=1):


    trajectory_fn = decoded_trajectory_with_forcing(model, start_with_input)
    trajectory_fn = functools.partial(
        trajectory_fn,
        forcing_data=forcing_data,
        outer_steps=outer_steps,
        inner_steps=inner_steps)

    trajectory_fn = with_preprocessing(
        trajectory_fn, with_forcing(model.encode,
                                    model.forcing_fn, forcing_data))

    trajectory_fn = with_input_included(
        trajectory_fn, num_last_input_frames_to_trim=int(start_with_input))

    trajectory_fn = with_split_input(trajectory_fn, num_init_frames)
    return trajectory_fn(x)

  return _trajectory_fn


def process_trajectory(
    input_trajectory: Pytree,
    process_fn: Callable[[Pytree], Pytree],
) -> Pytree:

  step_fn = lambda c, x: tuple([None, hk.remat(process_fn)(x)])
  _, out = hk.scan(step_fn, None, xs=input_trajectory)
  return out


def _maybe_to_nodal_with_physics_sharding(x, /, coords):
  x = coordinate_systems.maybe_to_nodal(x, coords)
  x = coords.with_physics_sharding(x)
  return x


def _maybe_to_modal_with_physics_sharding(x, /, coords):
  x = coordinate_systems.maybe_to_modal(x, coords)
  x = coords.with_physics_sharding(x)
  return x


def compute_prediction_representations(
    predicted_trajectory: typing.Pytree,
    forcing_data: typing.ForcingData,
    model: DynamicalSystem,
) -> typing.TrajectoryRepresentations:

  decode_fn = with_forcing(model.decode, model.forcing_fn, forcing_data)
  data_to_nodal = functools.partial(
      _maybe_to_nodal_with_physics_sharding, coords=model.output_coords)
  data_to_modal = functools.partial(
      _maybe_to_modal_with_physics_sharding, coords=model.output_coords)
  model_to_nodal = functools.partial(
      _maybe_to_nodal_with_physics_sharding, coords=model.coords)
  model_to_modal = functools.partial(
      _maybe_to_modal_with_physics_sharding, coords=model.coords)
  predicted_data_trajectory = process_trajectory(
      predicted_trajectory, decode_fn)

  if isinstance(predicted_trajectory, typing.ModelState):
    predicted_trajectory = predicted_trajectory.state
  if dataclasses.is_dataclass(predicted_trajectory):

    predicted_trajectory = predicted_trajectory.asdict()
  return typing.TrajectoryRepresentations(
      data_nodal_trajectory=process_trajectory(
          predicted_data_trajectory, data_to_nodal),
      data_modal_trajectory=process_trajectory(
          predicted_data_trajectory, data_to_modal),
      model_nodal_trajectory=process_trajectory(
          predicted_trajectory, model_to_nodal),
      model_modal_trajectory=process_trajectory(
          predicted_trajectory, model_to_modal),
  )


def compute_target_representations(
    target_trajectory: typing.Pytree,
    forcing_data: typing.ForcingData,
    model: DynamicalSystem,
) -> typing.TrajectoryRepresentations:

  encode_slice_fn = with_forcing(model.encode, model.forcing_fn, forcing_data)
  encode_fn = lambda tree: encode_slice_fn(
      jax.tree_util.tree_map(lambda x: jnp.expand_dims(x, 0), tree))
  data_to_nodal = functools.partial(
      _maybe_to_nodal_with_physics_sharding, coords=model.output_coords)
  data_to_modal = functools.partial(
      _maybe_to_modal_with_physics_sharding, coords=model.output_coords)
  model_to_nodal = functools.partial(
      _maybe_to_nodal_with_physics_sharding, coords=model.coords)
  model_to_modal = functools.partial(
      _maybe_to_modal_with_physics_sharding, coords=model.coords)
  target_model_trajectory = process_trajectory(
      target_trajectory, encode_fn)
  if isinstance(target_model_trajectory, typing.ModelState):
    target_model_trajectory = target_model_trajectory.state
  if dataclasses.is_dataclass(target_model_trajectory):

    target_model_trajectory = target_model_trajectory.asdict()
  return typing.TrajectoryRepresentations(
      data_nodal_trajectory=process_trajectory(
          target_trajectory, data_to_nodal),
      data_modal_trajectory=process_trajectory(
          target_trajectory, data_to_modal),
      model_nodal_trajectory=process_trajectory(
          target_model_trajectory, model_to_nodal),
      model_modal_trajectory=process_trajectory(
          target_model_trajectory, model_to_modal),
  )


def compute_prediction_and_target_representations(
    predicted_model_trajectory: typing.Pytree,
    target_data_trajectory: typing.Pytree,
    forcing_data: typing.ForcingData,
    model: DynamicalSystem,
) -> Tuple[typing.TrajectoryRepresentations, typing.TrajectoryRepresentations]:

  prediction_representations = compute_prediction_representations(
      predicted_model_trajectory, forcing_data, model)
  target_representations = compute_target_representations(
      target_data_trajectory, forcing_data, model)
  return prediction_representations, target_representations


@jax.custom_jvp
def safe_sqrt(x: Array) -> jax.Array:

  return jnp.sqrt(x)


@safe_sqrt.defjvp
def safe_sqrt_jvp(
    primals: Array,
    tangents: Array,
) -> tuple[jax.Array, jax.Array]:
  (x,) = primals
  (x_dot,) = tangents
  primal_out = safe_sqrt(x)
  eps = jnp.finfo(x.dtype).eps
  safe_x = jnp.where(x > eps, x, 1.0)
  tangent_out = jnp.where(x > eps, x_dot / (2 * safe_sqrt(safe_x)), 0)
  return primal_out, tangent_out
