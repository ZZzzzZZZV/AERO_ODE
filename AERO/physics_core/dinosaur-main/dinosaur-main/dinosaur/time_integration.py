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
import math
from typing import Any, Callable, Optional, Sequence, TypeVar, Union

from dinosaur import filtering
from dinosaur import spherical_harmonic
from dinosaur import typing

import jax
import jax.numpy as jnp
import numpy as np
import tree_math


tree_map = jax.tree_util.tree_map


State = typing.State
StateFn = typing.StateFn
InverseFn = typing.InverseFn
StepFn = typing.StepFn
FilterFn = typing.FilterFn
PyTreeState = typing.PyTreeState
PyTreeTermsFn = typing.PyTreeTermsFn
PyTreeInverseFn = typing.PyTreeInverseFn
TimeStepFn = typing.TimeStepFn
PyTreeStepFilterFn = typing.PyTreeStepFilterFn
PostProcessFn = typing.PostProcessFn


class ExplicitODE:


  def explicit_terms(self, state: PyTreeState) -> PyTreeState:

    raise NotImplementedError

  @classmethod
  def from_functions(
      cls,
      explicit_terms: PyTreeTermsFn,
  ) -> ExplicitODE:

    explicit_ode = cls()
    explicit_ode.explicit_terms = explicit_terms
    return explicit_ode


class ImplicitExplicitODE:


  def explicit_terms(self, state: PyTreeState) -> PyTreeState:

    raise NotImplementedError

  def implicit_terms(self, state: PyTreeState) -> PyTreeState:

    raise NotImplementedError

  def implicit_inverse(
      self, state: PyTreeState, step_size: float,
  ) -> PyTreeState:

    raise NotImplementedError

  @classmethod
  def from_functions(
      cls,
      explicit_terms: PyTreeTermsFn,
      implicit_terms: PyTreeTermsFn,
      implicit_inverse: PyTreeInverseFn,
  ) -> ImplicitExplicitODE:

    explicit_implicit_ode = cls()
    explicit_implicit_ode.explicit_terms = explicit_terms
    explicit_implicit_ode.implicit_terms = implicit_terms
    explicit_implicit_ode.implicit_inverse = implicit_inverse
    return explicit_implicit_ode


@dataclasses.dataclass
class TimeReversedImExODE(ImplicitExplicitODE):

  forward_eq: ImplicitExplicitODE

  def explicit_terms(self, state: PyTreeState) -> PyTreeState:
    forward_term = self.forward_eq.explicit_terms(state)
    return tree_map(jnp.negative, forward_term)

  def implicit_terms(self, state: PyTreeState) -> PyTreeState:
    forward_term = self.forward_eq.implicit_terms(state)
    return tree_map(jnp.negative, forward_term)

  def implicit_inverse(
      self, state: PyTreeState, step_size: float,
  ) -> PyTreeState:
    return self.forward_eq.implicit_inverse(state, -step_size)


def compose_equations(
    equations: Sequence[Union[ImplicitExplicitODE, ExplicitODE]],
) -> ImplicitExplicitODE:

  implicit_explicit_eqs = list(
      filter(lambda x: isinstance(x, ImplicitExplicitODE), equations))
  if len(implicit_explicit_eqs) != 1:
    raise ValueError('compose_equations supports at most 1 ImplicitExplicitODE '
                     f'got {len(implicit_explicit_eqs)}')
  (implicit_explicit_equation,) = implicit_explicit_eqs
  assert isinstance(implicit_explicit_equation, ImplicitExplicitODE)

  def explicit_fn(x: PyTreeState) -> PyTreeState:
    explicit_tendencies = [fn.explicit_terms(x) for fn in equations]
    return tree_map(
        lambda *args: sum([x for x in args if x is not None]),
        *explicit_tendencies)

  return ImplicitExplicitODE.from_functions(
      explicit_fn, implicit_explicit_equation.implicit_terms,
      implicit_explicit_equation.implicit_inverse)


def backward_forward_euler(
    equation: ImplicitExplicitODE,
    time_step: float,
) -> TimeStepFn:


  dt = time_step
  F = tree_math.unwrap(equation.explicit_terms)
  G_inv = tree_math.unwrap(equation.implicit_inverse, vector_argnums=0)

  @tree_math.wrap
  def step_fn(u0):
    g = u0 + dt * F(u0)
    u1 = G_inv(g, dt)
    return u1
  return step_fn


def semi_implicit_leapfrog(
    equation: ImplicitExplicitODE,
    time_step: float,
    alpha: float = 0.5,
) -> TimeStepFn:

  explicit_fn = tree_math.unwrap(equation.explicit_terms)
  implicit_fn = tree_math.unwrap(equation.implicit_terms)
  inverse_fn = tree_math.unwrap(equation.implicit_inverse, vector_argnums=0)

  def step_fn(u: PyTreeState) -> PyTreeState:

    previous, current = u
    previous, current = tree_math.Vector(previous), tree_math.Vector(current)
    explicit_current = explicit_fn(current)
    implicit_previous = implicit_fn(previous)
    intermediate = previous + 2 * time_step * (
        explicit_current + (1 - alpha) * implicit_previous)
    eta = 2 * time_step * alpha
    future = inverse_fn(intermediate, eta)
    return (current.tree, future.tree)

  return step_fn


def crank_nicolson_rk2(
    equation: ImplicitExplicitODE,
    time_step: float,
) -> TimeStepFn:

  dt = time_step
  F = tree_math.unwrap(equation.explicit_terms)
  G = tree_math.unwrap(equation.implicit_terms)
  G_inv = tree_math.unwrap(equation.implicit_inverse, vector_argnums=0)

  @tree_math.wrap
  def step_fn(u0):
    g = u0 + 0.5 * dt * G(u0)
    h1 = F(u0)
    u1 = G_inv(g + dt * h1, 0.5 * dt)
    h2 = 0.5 * (F(u1) + h1)
    u2 = G_inv(g + dt * h2, 0.5 * dt)
    return u2
  return step_fn


def low_storage_runge_kutta_crank_nicolson(
    alphas: Sequence[float],
    betas: Sequence[float],
    gammas: Sequence[float],
    equation: ImplicitExplicitODE,
    time_step: float,
) -> TimeStepFn:

  α = alphas
  β = betas
  γ = gammas
  dt = time_step
  F = tree_math.unwrap(equation.explicit_terms)
  G = tree_math.unwrap(equation.implicit_terms)
  G_inv = tree_math.unwrap(equation.implicit_inverse, vector_argnums=0)

  if len(alphas) - 1 != len(betas) != len(gammas):
    raise ValueError('number of RK coefficients does not match')

  @tree_math.wrap
  def step_fn(u):
    h = 0
    for k in range(len(β)):
      h = F(u) + β[k] * h
      µ = 0.5 * dt * (α[k + 1] - α[k])
      u = G_inv(u + γ[k] * dt * h + µ * G(u), µ)
    return u
  return step_fn


def crank_nicolson_rk3(
    equation: ImplicitExplicitODE,
    time_step: float,
) -> TimeStepFn:

  return low_storage_runge_kutta_crank_nicolson(
      alphas=[0, 1/3, 3/4, 1],
      betas=[0, -5/9, -153/128],
      gammas=[1/3, 15/16, 8/15],
      equation=equation,
      time_step=time_step,
  )


def crank_nicolson_rk4(
    equation: ImplicitExplicitODE,
    time_step: float,
) -> TimeStepFn:


  return low_storage_runge_kutta_crank_nicolson(
      alphas=[0, 0.1496590219993, 0.3704009573644, 0.6222557631345, 0.9582821306748, 1],
      betas=[0, -0.4178904745, -1.192151694643, -1.697784692471, -1.514183444257],
      gammas=[0.1496590219993, 0.3792103129999, 0.8229550293869, 0.6994504559488, 0.1530572479681],
      equation=equation,
      time_step=time_step,
  )


@dataclasses.dataclass
class ImExButcherTableau:

  a_ex: Sequence[Sequence[float]]
  a_im: Sequence[Sequence[float]]
  b_ex: Sequence[float]
  b_im: Sequence[float]

  def __post_init__(self):
    if len({len(self.a_ex) + 1,
            len(self.a_im) + 1,
            len(self.b_ex),
            len(self.b_im)}) > 1:
      raise ValueError('inconsistent Butcher tableau')


def imex_runge_kutta(
    tableau: ImExButcherTableau,
    equation: ImplicitExplicitODE,
    time_step: float,
) -> TimeStepFn:


  dt = time_step
  F = tree_math.unwrap(equation.explicit_terms)
  G = tree_math.unwrap(equation.implicit_terms)
  G_inv = tree_math.unwrap(equation.implicit_inverse, vector_argnums=0)

  a_ex = tableau.a_ex
  a_im = tableau.a_im
  b_ex = tableau.b_ex
  b_im = tableau.b_im

  num_steps = len(b_ex)

  @tree_math.wrap
  def step_fn(y0):
    f = [None] * num_steps
    g = [None] * num_steps

    f[0] = F(y0)
    g[0] = G(y0)

    for i in range(1, num_steps):
      ex_terms = dt * sum(a_ex[i-1][j] * f[j] for j in range(i) if a_ex[i-1][j])
      im_terms = dt * sum(a_im[i-1][j] * g[j] for j in range(i) if a_im[i-1][j])
      Y_star = y0 + ex_terms + im_terms
      Y = G_inv(Y_star, dt * a_im[i-1][i])
      if any(a_ex[j][i] for j in range(i, num_steps - 1)) or b_ex[i]:
        f[i] = F(Y)
      if any(a_im[j][i] for j in range(i, num_steps - 1)) or b_im[i]:
        g[i] = G(Y)

    ex_terms = dt * sum(b_ex[j] * f[j] for j in range(num_steps) if b_ex[j])
    im_terms = dt * sum(b_im[j] * g[j] for j in range(num_steps) if b_im[j])
    y_next = y0 + ex_terms + im_terms

    return y_next

  return step_fn


def imex_rk_sil3(
    equation: ImplicitExplicitODE,
    time_step: float,
) -> TimeStepFn:

  return imex_runge_kutta(
      tableau=ImExButcherTableau(
          a_ex=[[1/3], [1/6, 1/2], [1/2, -1/2, 1]],
          a_im=[[1/6, 1/6], [1/3, 0, 1/3], [3/8, 0, 3/8, 1/4]],
          b_ex=[1/2, -1/2, 1, 0],
          b_im=[3/8, 0, 3/8, 1/4],
      ),
      equation=equation,
      time_step=time_step,
  )


def robert_asselin_leapfrog_filter(r: float) -> PyTreeStepFilterFn:

  def _filter(u: PyTreeState, u_next: PyTreeState) -> PyTreeState:
    previous, current = u
    _, future = u_next
    filtered_current = tree_map(
        lambda p, c, f: (1 - 2 * r) * c + r * (p + f),
        previous, current, future)
    return (filtered_current, future)

  return _filter


def runge_kutta_step_filter(
    state_filter: PyTreeTermsFn,
) -> PyTreeStepFilterFn:


  def _filter(u: PyTreeState, u_next: PyTreeState) -> PyTreeState:
    del u
    return state_filter(u_next)

  return _filter


def leapfrog_step_filter(
    state_filter: PyTreeTermsFn,
) -> PyTreeStepFilterFn:


  def _filter(u: PyTreeState, u_next: PyTreeState) -> PyTreeState:
    del u
    current, future = u_next
    future = state_filter(future)
    return (current, future)

  return _filter


def exponential_step_filter(
    grid: spherical_harmonic.Grid,
    dt: float,
    tau: float = 0.010938,
    order: int = 18,
    cutoff: float = 0,
):

  filter_fn = filtering.exponential_filter(grid, dt / tau, order, cutoff)
  return runge_kutta_step_filter(filter_fn)


def exponential_leapfrog_step_filter(
    grid: spherical_harmonic.Grid,
    dt: float,
    tau: float = 0.010938,
    order: int = 18,
    cutoff: float = 0,
):

  filter_fn = filtering.exponential_filter(grid, dt / tau, order, cutoff)
  return leapfrog_step_filter(filter_fn)


def horizontal_diffusion_step_filter(
    grid: spherical_harmonic.Grid,
    dt: float,
    tau: float,
    order: int = 1,
):

  eigenvalues = grid.laplacian_eigenvalues
  scale = dt / (tau * abs(eigenvalues[-1]) ** order)
  filter_fn = filtering.horizontal_diffusion_filter(grid, scale, order)
  return runge_kutta_step_filter(filter_fn)


def step_with_filters(
    step_fn: TimeStepFn,
    filters: Sequence[PyTreeStepFilterFn],
) -> TimeStepFn:

  def _step_fn(u: PyTreeState) -> PyTreeState:
    u_next = step_fn(u)
    for filter_fn in filters:
      u_next = filter_fn(u, u_next)
    return u_next

  return _step_fn


def repeated(
    fn: TimeStepFn,
    steps: int,
    scan_fn: typing.ScanFn = jax.lax.scan
) -> TimeStepFn:

  if steps == 1:
    return fn
  def f_repeated(x_initial: PyTreeState) -> PyTreeState:
    g = lambda x, _: (fn(x), None)
    x_final, _ = scan_fn(g, x_initial, xs=None, length=steps)
    return x_final
  return f_repeated


def trajectory_from_step(
    step_fn: TimeStepFn,
    outer_steps: int,
    inner_steps: int,
    *,
    start_with_input: bool = False,
    post_process_fn: PostProcessFn = lambda x: x,
    outer_scan_fn: typing.ScanFn = jax.lax.scan,
    inner_scan_fn: typing.ScanFn = jax.lax.scan,
) -> Callable[[PyTreeState], tuple[PyTreeState, Any]]:

  if inner_steps != 1:
    step_fn = repeated(step_fn, inner_steps, inner_scan_fn)

  def step(carry_in, _):
    carry_out = step_fn(carry_in)
    frame = carry_in if start_with_input else carry_out
    return carry_out, post_process_fn(frame)

  def multistep(x):
    return outer_scan_fn(step, x, xs=None, length=outer_steps)

  return multistep


Carry = TypeVar('Carry')
Input = TypeVar('Input')
Output = TypeVar('Output')
Func = TypeVar('Func', bound=Callable)


def nested_checkpoint_scan(
    f: Callable[[Carry, Input], tuple[Carry, Output]],
    init: Carry,
    xs: Input,
    length: Optional[int] = None,
    *,
    nested_lengths: Sequence[int],
    scan_fn: typing.ScanFn = jax.lax.scan,
    checkpoint_fn: Callable[[Func], Func] = jax.checkpoint,
) -> tuple[Carry, Output]:


  if length is not None and length != math.prod(nested_lengths):
    raise ValueError(f'inconsistent {length=} and {nested_lengths=}')

  def nested_reshape(x):
    x = jnp.asarray(x)
    new_shape = tuple(nested_lengths) + x.shape[1:]
    return x.reshape(new_shape)

  sub_xs = tree_map(nested_reshape, xs)
  return _inner_nested_scan(f, init, sub_xs, nested_lengths, scan_fn,
                            checkpoint_fn)


def _inner_nested_scan(f, init, xs, lengths, scan_fn, checkpoint_fn):

  if len(lengths) == 1:
    return scan_fn(f, init, xs, lengths[0])

  @checkpoint_fn
  def sub_scans(carry, xs):
    return _inner_nested_scan(f, carry, xs, lengths[1:], scan_fn, checkpoint_fn)

  carry, out = scan_fn(sub_scans, init, xs, lengths[0])
  stacked_out = tree_map(jnp.concatenate, out)
  return carry, stacked_out


def accumulate_repeated(
    step_fn: StateFn,
    weights: jnp.ndarray,
    state: State,
    scan_fn: typing.ScanFn = jax.lax.scan,
) -> State:

  def f(carry, weight):
    state, averaged = carry
    state = step_fn(state)
    averaged = tree_map(lambda s, a: a + weight * s, state, averaged)
    return (state, averaged), None

  zeros = tree_map(jnp.zeros_like, state)
  init = (state, zeros)
  (_, averaged), _ = scan_fn(f, init, weights)
  return averaged


def _dfi_lanczos_weights(
    time_span: float, cutoff_period: float, dt: float,
) -> np.ndarray:

  N = round(time_span / (2 * dt))
  n = np.arange(1, N + 1)
  w = np.sinc(n / (N + 1)) * np.sinc(n * time_span / (cutoff_period * N))
  return w


def digital_filter_initialization(
    equation: ImplicitExplicitODE,
    ode_solver: Callable[[ImplicitExplicitODE, float], StateFn],
    filters: Sequence[PyTreeStepFilterFn],
    time_span: float,
    cutoff_period: float,
    dt: float,
) -> StateFn:

  def f(state):
    forward_step = step_with_filters(ode_solver(equation, dt), filters)
    backward_step = step_with_filters(
        ode_solver(TimeReversedImExODE(equation), dt), filters)

    weights = _dfi_lanczos_weights(time_span, cutoff_period, dt)
    init_weight = 1.0
    total_weight = init_weight + 2 * weights.sum()

    init_weight /= total_weight
    weights /= total_weight

    init_term = tree_map(lambda x: x * init_weight, state)
    forward_term = accumulate_repeated(forward_step, weights, state)
    backward_term = accumulate_repeated(backward_step, weights, state)
    return tree_map(lambda *xs: sum(xs), init_term, forward_term, backward_term)
  return f


def maybe_fix_sim_time_roundoff(
    state: typing.PyTreeState,
    dt: float,
) ->typing.PyTreeState:

  if hasattr(state, 'sim_time'):
    state.sim_time = dt * jnp.round(state.sim_time / dt)
  return state
