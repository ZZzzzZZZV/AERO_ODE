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

import functools

import numpy as np
import scipy.special as sps


def _evaluate_rhombus(n_l: int,
                      n_m: int,
                      x: np.ndarray,
                      truncation='rhombus') -> np.ndarray:

  y = np.sqrt(1 - x * x)
  p = np.zeros((n_l, n_m, len(x)))
  p[0, 0] = p[0, 0] + 1 / np.sqrt(2)
  for m in range(1, n_m):
    p[0, m] = -np.sqrt(1 + 1 / (2 * m)) * y * p[0, m - 1]
  m_max = n_m
  for k in range(1, n_l):
    if truncation == 'triangle':
      m_max = min(n_m, n_l - k)
    m = np.arange(m_max).reshape((-1, 1))
    m2 = np.square(m)
    mk2 = np.square(m + k)
    mkp2 = np.square(m + k - 1)
    a = np.sqrt((4 * mk2 - 1) / (mk2 - m2))
    b = np.sqrt((mkp2 - m2) / (4 * mkp2 - 1))
    p[k, :m_max] = a * (x * p[k - 1, :m_max] - b * p[k - 2, :m_max])
  return p


def evaluate(n_m: int,
             n_l: int,
             x: np.ndarray) -> np.ndarray:

  if n_m > n_l:
    raise ValueError(f'Expected n_m <= n_l; got n_m = {n_m} and n_l = {n_l}.')
  r = np.transpose(
      _evaluate_rhombus(n_l=n_l, n_m=n_m, x=x, truncation='triangle'),
      (1, 2, 0))
  p = np.zeros((n_m, len(x), n_l))
  for m in range(n_m):
    p[m, :, m:n_l] = r[m, :, 0:n_l - m]
  return p


def gauss_legendre_nodes(n: int) -> tuple[np.ndarray, np.ndarray]:

  return sps.roots_legendre(n)


@functools.lru_cache(maxsize=128)
def equiangular_nodes(n: int) -> tuple[np.ndarray, np.ndarray]:

  spacing = np.pi / n
  theta = np.linspace(-np.pi / 2 + spacing / 2,
                      np.pi / 2 - spacing / 2,
                      n)
  x = np.sin(theta)
  w = _compute_weights(x)
  return x, w


@functools.lru_cache(maxsize=128)
def equiangular_nodes_with_poles(n: int) -> tuple[np.ndarray, np.ndarray]:

  theta = np.linspace(-np.pi / 2, np.pi / 2, n)
  x = np.sin(theta)
  w = _compute_weights(x)
  return x, w


def _compute_weights(x: np.ndarray) -> np.ndarray:


  legendre = evaluate(n_m=1, n_l=x.shape[0], x=x)[0].T
  z = np.zeros_like(x)
  z[0] = 1
  w = np.linalg.solve(legendre, z)


  return w / w.sum() * 2
