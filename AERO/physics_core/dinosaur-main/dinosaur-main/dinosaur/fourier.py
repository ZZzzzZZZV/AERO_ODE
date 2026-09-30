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

from dinosaur import jax_numpy_utils

import jax
import jax.numpy as jnp
import numpy as np
import scipy.linalg


def real_basis(wavenumbers: int, nodes: int) -> np.ndarray:

  if nodes < wavenumbers:
    raise ValueError(
        '`real_basis` requires nodes >= wavenumbers; '
        f'got m = {nodes} and n = {wavenumbers}.'
    )

  dft = scipy.linalg.dft(nodes)[:, :wavenumbers] / np.sqrt(np.pi)

  cos = np.real(dft[:, 1:])
  sin = -np.imag(dft[:, 1:])

  f = np.empty(shape=[nodes, 2 * wavenumbers - 1], dtype=np.float64)
  f[:, 0] = 1 / np.sqrt(2 * np.pi)
  f[:, 1::2] = cos
  f[:, 2::2] = sin
  return f


def real_basis_derivative(
    u: jnp.ndarray | jax.Array, /, axis: int = -1
) -> jax.Array:

  if u.shape[axis] % 2 != 1:
    raise ValueError(f'{u.shape=} along {axis=} is not odd')
  if axis >= 0:
    raise ValueError('axis must be negative')
  i = jnp.arange(u.shape[axis]).reshape((-1,) + (1,) * (-1 - axis))
  j = (i + 1) // 2
  u_down = jax_numpy_utils.shift(u, -1, axis)
  u_up = jax_numpy_utils.shift(u, +1, axis)
  return j * jnp.where(i % 2, u_down, -u_up)


def real_basis_with_zero_imag(wavenumbers: int, nodes: int) -> np.ndarray:

  if nodes < wavenumbers:
    raise ValueError(
        '`real_basis` requires nodes >= wavenumbers; '
        f'got m = {nodes} and n = {wavenumbers}.'
    )

  dft = scipy.linalg.dft(nodes)[:, :wavenumbers] / np.sqrt(np.pi)

  cos = np.real(dft[:, 1:])
  sin = -np.imag(dft[:, 1:])

  f = np.empty(shape=[nodes, 2 * wavenumbers], dtype=np.float64)
  f[:, 0] = 1 / np.sqrt(2 * np.pi)
  f[:, 1] = 0
  f[:, 2::2] = cos
  f[:, 3::2] = sin
  return f


def real_basis_derivative_with_zero_imag(
    u: jnp.ndarray | jax.Array, axis: int = -1, frequency_offset: int = 0
) -> jax.Array:

  if u.shape[axis] % 2:
    raise ValueError(f'{u.shape=} along {axis=} is not even')
  if axis >= 0:
    raise ValueError('axis must be negative')
  i = jnp.arange(u.shape[axis]).reshape((-1,) + (1,) * (-1 - axis))
  j = frequency_offset + i // 2
  u_down = jax_numpy_utils.shift(u, -1, axis)
  u_up = jax_numpy_utils.shift(u, +1, axis)
  return j * jnp.where((i + 1) % 2, u_down, -u_up)


def complex_basis(wavenumbers: int, nodes: int) -> np.ndarray:

  if wavenumbers > nodes // 2 + 1:
    raise ValueError(
        '`wavenumbers` must be no greater than `nodes // 2 + 1`;'
        f'got wavenumbers = {wavenumbers}, nodes = {nodes}.'
    )
  basis = scipy.linalg.dft(nodes).conj()[:, :wavenumbers] / np.sqrt(np.pi)
  basis[:, 0] /= np.sqrt(2)
  return basis


def complex_basis_derivative(
    u: jnp.ndarray | jax.Array, axis: int = -1
) -> jax.Array:

  if axis >= 0:
    raise ValueError('axis must be negative')
  k = jnp.arange(u.shape[axis]).reshape((-1,) + (1,) * (-1 - axis))
  return 1j * k * u


def quadrature_nodes(nodes: int) -> tuple[np.ndarray, np.ndarray]:

  xs = np.linspace(0, 2 * np.pi, nodes, endpoint=False)
  weights = 2 * np.pi / nodes
  return xs, weights
