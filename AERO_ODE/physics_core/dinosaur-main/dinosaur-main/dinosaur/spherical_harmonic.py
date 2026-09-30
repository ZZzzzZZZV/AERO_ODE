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
import math
from typing import Any, Callable

from dinosaur import associated_legendre
from dinosaur import fourier
from dinosaur import jax_numpy_utils
from dinosaur import pytree_utils
from dinosaur import typing
import jax
from jax import lax
from jax.experimental import shard_map
import jax.numpy as jnp
import numpy as np


Array = typing.Array
ArrayOrArrayTuple = typing.ArrayOrArrayTuple


einsum = functools.partial(jnp.einsum, precision=jax.lax.Precision.HIGHEST)


LATITUDE_SPACINGS = dict(
    gauss=associated_legendre.gauss_legendre_nodes,
    equiangular=associated_legendre.equiangular_nodes,
    equiangular_with_poles=associated_legendre.equiangular_nodes_with_poles,
)


def get_latitude_nodes(n: int, spacing: str) -> tuple[np.ndarray, np.ndarray]:

  get_nodes = LATITUDE_SPACINGS.get(spacing)
  if get_nodes is None:
    raise ValueError(
        f'Unknown spacing: {spacing}'
        f'available spacings are {list(LATITUDE_SPACINGS.keys())}'
    )
  return get_nodes(n)


@dataclasses.dataclass
class _SphericalHarmonicBasis:


  f: np.ndarray
  p: np.ndarray
  w: np.ndarray


@dataclasses.dataclass(frozen=True)
class SphericalHarmonics:


  longitude_wavenumbers: int = 0
  total_wavenumbers: int = 0
  longitude_nodes: int = 0
  latitude_nodes: int = 0
  latitude_spacing: str = 'gauss'

  @property
  def nodal_axes(self) -> tuple[np.ndarray, np.ndarray]:

    raise NotImplementedError

  @property
  def nodal_shape(self) -> tuple[int, int]:

    raise NotImplementedError

  @property
  def nodal_padding(self) -> tuple[int, int]:

    raise NotImplementedError

  @property
  def modal_axes(self) -> tuple[np.ndarray, np.ndarray]:

    raise NotImplementedError

  @property
  def modal_shape(self) -> tuple[int, int]:

    raise NotImplementedError

  @property
  def modal_padding(self) -> tuple[int, int]:

    raise NotImplementedError

  @property
  def modal_dtype(self) -> np.dtype:

    raise NotImplementedError

  @property
  def mask(self) -> np.ndarray:

    raise NotImplementedError

  @property
  def basis(self) -> _SphericalHarmonicBasis:

    raise NotImplementedError

  def inverse_transform(self, x: jax.Array) -> jax.Array:

    raise NotImplementedError

  def transform(self, x: jax.Array) -> jax.Array:

    raise NotImplementedError

  def longitudinal_derivative(self, x: Array) -> Array:

    raise NotImplementedError


class RealSphericalHarmonics(SphericalHarmonics):


  @functools.cached_property
  def nodal_axes(self) -> tuple[np.ndarray, np.ndarray]:
    longitude, _ = fourier.quadrature_nodes(self.longitude_nodes)
    sin_latitude, _ = get_latitude_nodes(
        self.latitude_nodes, self.latitude_spacing
    )
    return longitude, sin_latitude

  @functools.cached_property
  def nodal_shape(self) -> tuple[int, int]:
    return (self.longitude_nodes, self.latitude_nodes)

  @functools.cached_property
  def nodal_padding(self) -> tuple[int, int]:
    return (0, 0)

  @functools.cached_property
  def modal_axes(self) -> tuple[np.ndarray, np.ndarray]:
    m_pos = np.arange(1, self.longitude_wavenumbers)
    m_pos_neg = np.stack([m_pos, -m_pos], axis=1).ravel()
    lon_wavenumbers = np.concatenate([[0], m_pos_neg])
    tot_wavenumbers = np.arange(self.total_wavenumbers)
    return lon_wavenumbers, tot_wavenumbers

  @functools.cached_property
  def modal_shape(self) -> tuple[int, int]:
    return (2 * self.longitude_wavenumbers - 1, self.total_wavenumbers)

  @functools.cached_property
  def modal_padding(self) -> tuple[int, int]:
    return (0, 0)

  @functools.cached_property
  def modal_dtype(self) -> np.dtype:
    return np.dtype(np.float32)

  @functools.cached_property
  def mask(self) -> np.ndarray:
    m, l = np.meshgrid(*self.modal_axes, indexing='ij')
    return abs(m) <= l

  @functools.cached_property
  def basis(self) -> _SphericalHarmonicBasis:


    f = fourier.real_basis(
        wavenumbers=self.longitude_wavenumbers,
        nodes=self.longitude_nodes,
    )
    _, wf = fourier.quadrature_nodes(self.longitude_nodes)
    x, wp = get_latitude_nodes(self.latitude_nodes, self.latitude_spacing)
    w = wf * wp
    p = associated_legendre.evaluate(
        n_m=self.longitude_wavenumbers, n_l=self.total_wavenumbers, x=x
    )


    p = np.repeat(p, 2, axis=0)


    p = p[1:]
    return _SphericalHarmonicBasis(f=f, p=p, w=w)

  def inverse_transform(self, x):
    p = self.basis.p
    f = self.basis.f
    px = jax.named_call(einsum, name='inv_legendre')('mjl,...ml->...mj', p, x)


    fpx = jax.named_call(einsum, name='inv_fourier')('im,...mj->...ij', f, px)
    return fpx

  def transform(self, x):
    w = self.basis.w
    f = self.basis.f
    p = self.basis.p
    wx = w * x
    fwx = jax.named_call(einsum, name='fwd_fourier')('im,...ij->...mj', f, wx)
    pfwx = jax.named_call(einsum, name='fwd_legendre')(
        'mjl,...mj->...ml', p, fwx
    )
    return pfwx

  def longitudinal_derivative(self, x: Array) -> Array:
    return fourier.real_basis_derivative(x, axis=-2)


def _round_to_multiple(x: int, multiple: int) -> int:
  return multiple * math.ceil(x / multiple)


P = jax.sharding.PartitionSpec
shmap = shard_map.shard_map


@jax.named_call
def _unstack_m(x: jax.Array, /, mesh: jax.sharding.Mesh | None) -> jax.Array:


  def unstack(x):
    shape = x.shape[:-2] + (2, x.shape[-2] // 2) + x.shape[-1:]
    return jnp.reshape(x, shape, order='F')

  if mesh is None:
    return unstack(x)

  assert x.ndim in {2, 3}, x.shape
  z = None if x.shape[0] == 1 else 'z'
  in_spec = P(z, 'x', 'y') if x.ndim == 3 else P('x', 'y')
  out_spec = P(z, None, 'x', 'y') if x.ndim == 3 else P(None, 'x', 'y')
  return shmap(unstack, mesh, (in_spec,), out_spec)(x)


@jax.named_call
def _stack_m(x: jax.Array, /, mesh: jax.sharding.Mesh | None) -> jax.Array:


  def stack(x):
    shape = x.shape[:-3] + (-1,) + x.shape[-1:]
    return jnp.reshape(x, shape, order='F')

  if mesh is None:
    return stack(x)

  assert x.ndim in {3, 4}, x.shape
  z = None if x.shape[0] == 1 else 'z'
  in_spec = P(z, None, 'x', 'y') if x.ndim == 4 else P(None, 'x', 'y')
  out_spec = P(z, 'x', 'y') if x.ndim == 4 else P('x', 'y')
  return shmap(stack, mesh, (in_spec,), out_spec)(x)


@jax.named_call
def _fourier_derivative_for_real_basis_with_zero_imag(
    x: jax.Array, /, mesh: jax.sharding.Mesh | None
) -> jax.Array:


  if mesh is None:
    return fourier.real_basis_derivative_with_zero_imag(x, axis=-2)


  def differentiate(u):
    axis = -2
    frequency_offset = u.shape[axis] // 2 * lax.axis_index('x')
    return fourier.real_basis_derivative_with_zero_imag(
        u, axis, frequency_offset
    )

  assert x.ndim in {2, 3}, x.shape
  z = None if x.shape[0] == 1 else 'z'
  spec = P(z, 'x', 'y') if x.ndim == 3 else P('x', 'y')


  return shmap(differentiate, mesh, (spec,), spec, check_rep=False)(x)


def _transform_einsum(
    subscripts: str,
    lhs: np.ndarray | jax.Array,
    rhs: jax.Array,
    mesh: jax.sharding.Mesh | None,
    reverse_einsum_arg_order: bool | None,
    precision: str,
) -> jax.Array:

  out_ndim = len(
      jax.eval_shape(functools.partial(jnp.einsum, subscripts), lhs, rhs).shape
  )
  in_subscripts, _ = subscripts.split('->')
  _, rhs_subscripts = in_subscripts.split(',')

  if rhs.ndim == len(rhs_subscripts.replace('...', '')):

    subscripts = subscripts.replace('...', '')
    in_spec = P(None, 'x', 'y') if rhs.ndim == 3 else P('x', 'y')
    out_spec = P(None, 'x', 'y') if out_ndim == 3 else P('x', 'y')
  elif rhs.ndim == len(rhs_subscripts.replace('...', 'z')):

    subscripts = subscripts.replace('...', 'z')
    z = None if rhs.shape[0] == 1 else 'z'
    in_spec = P(z, None, 'x', 'y') if rhs.ndim == 4 else P(z, 'x', 'y')
    out_spec = P(z, None, 'x', 'y') if out_ndim == 4 else P(z, 'x', 'y')
  else:
    raise ValueError(f'only 0 or 1 dimensions supported for ...: {subscripts}')

  return jax_numpy_utils.sharded_einsum(
      subscripts,
      lhs,
      rhs,
      mesh=mesh,
      rhs_spec=in_spec,
      out_spec=out_spec,
      reverse_arg_order=bool(reverse_einsum_arg_order),
      precision=precision,
  )


@dataclasses.dataclass(frozen=True)
class RealSphericalHarmonicsWithZeroImag(SphericalHarmonics):


  spmd_mesh: jax.sharding.Mesh | None = None
  base_shape_multiple: int | None = None
  reverse_einsum_arg_order: bool | None = None
  stacked_fourier_transforms: bool | None = None
  transform_precision: str = 'tensorfloat32'

  def __post_init__(self):
    model_parallelism = self.spmd_mesh is not None and any(
        self.spmd_mesh.shape[dim] > 1 for dim in 'zxy'
    )

    if self.base_shape_multiple is None:
      shape_multiple = 8 if model_parallelism else 1
      object.__setattr__(self, 'base_shape_multiple', shape_multiple)

    if self.reverse_einsum_arg_order is None:
      object.__setattr__(self, 'reverse_einsum_arg_order', model_parallelism)

    if self.stacked_fourier_transforms is None:


      unstacked_matmuls = math.ceil(self.longitude_wavenumbers / 128)
      stacked_matmuls = 2 * math.ceil(self.longitude_wavenumbers / 256)
      stack = stacked_matmuls <= unstacked_matmuls
      object.__setattr__(self, 'stacked_fourier_transforms', stack)

  @functools.cached_property
  def nodal_limits(self) -> tuple[int, int]:
    return (self.longitude_nodes, self.latitude_nodes)

  @functools.cached_property
  def modal_limits(self) -> tuple[int, int]:
    return (2 * self.longitude_wavenumbers, self.total_wavenumbers)

  def _mesh_shape(self) -> tuple[int, int]:
    if self.spmd_mesh is not None:
      return (self.spmd_mesh.shape['x'], self.spmd_mesh.shape['y'])
    else:
      return (1, 1)

  @functools.cached_property
  def nodal_shape(self) -> tuple[int, int]:
    base = self.base_shape_multiple or 1
    x_shards, y_shards = self._mesh_shape()
    shape_multiples = (base * x_shards, base * y_shards)
    return tuple(map(_round_to_multiple, self.nodal_limits, shape_multiples))

  @functools.cached_property
  def modal_shape(self) -> tuple[int, int]:
    base = self.base_shape_multiple or 1
    x_shards, y_shards = self._mesh_shape()

    shape_multiples = (2 * base * x_shards, base * y_shards)
    return tuple(map(_round_to_multiple, self.modal_limits, shape_multiples))

  @functools.cached_property
  def nodal_padding(self) -> tuple[int, int]:
    return tuple(x - y for x, y in zip(self.nodal_shape, self.nodal_limits))

  @functools.cached_property
  def modal_padding(self) -> tuple[int, int]:
    return tuple(x - y for x, y in zip(self.modal_shape, self.modal_limits))

  @functools.cached_property
  def nodal_axes(self) -> tuple[np.ndarray, np.ndarray]:
    nodal_pad_x, nodal_pad_y = self.nodal_padding
    longitude, _ = fourier.quadrature_nodes(self.longitude_nodes)
    longitude = np.pad(longitude, [(0, nodal_pad_x)])
    sin_latitude, _ = get_latitude_nodes(
        self.latitude_nodes, self.latitude_spacing
    )
    sin_latitude = np.pad(sin_latitude, [(0, nodal_pad_y)])
    return longitude, sin_latitude

  @functools.cached_property
  def modal_axes(self) -> tuple[np.ndarray, np.ndarray]:
    modal_pad_x, modal_pad_y = self.modal_padding
    m_pos = np.arange(1, self.longitude_wavenumbers)
    m_pos_neg = np.stack([m_pos, -m_pos], axis=1).ravel()
    lon_wavenumbers = np.pad(
        np.concatenate([[0, 0], m_pos_neg]), [(0, modal_pad_x)]
    )
    tot_wavenumbers = np.pad(
        np.arange(self.total_wavenumbers), [(0, modal_pad_y)]
    )
    return lon_wavenumbers, tot_wavenumbers

  @functools.cached_property
  def modal_dtype(self) -> np.dtype:
    return np.dtype(np.float32)

  @functools.cached_property
  def mask(self) -> np.ndarray:
    m, l = np.meshgrid(*self.modal_axes, indexing='ij')
    i, j = np.meshgrid(*(np.arange(s) for s in self.modal_shape), indexing='ij')
    i_lim, j_lim = self.modal_limits
    return (abs(m) <= l) & (i != 1) & (i < i_lim) & (j < j_lim)

  @functools.cached_property
  def basis(self) -> _SphericalHarmonicBasis:
    nodal_pad_x, nodal_pad_y = self.nodal_padding
    modal_pad_x, modal_pad_y = self.modal_padding

    f = fourier.real_basis_with_zero_imag(
        wavenumbers=self.longitude_wavenumbers,
        nodes=self.longitude_nodes,
    )
    f = np.pad(f, [(0, nodal_pad_x), (0, modal_pad_x)])
    if self.stacked_fourier_transforms:
      f = np.reshape(f, (-1, 2, f.shape[-1] // 2), order='F')

    _, wf = fourier.quadrature_nodes(self.longitude_nodes)
    x, wp = get_latitude_nodes(self.latitude_nodes, self.latitude_spacing)
    w = wf * wp
    w = np.pad(w, [(0, nodal_pad_y)])

    p = associated_legendre.evaluate(
        n_m=self.longitude_wavenumbers, n_l=self.total_wavenumbers, x=x
    )
    p = np.pad(p, [(0, modal_pad_x // 2), (0, nodal_pad_y), (0, modal_pad_y)])

    return _SphericalHarmonicBasis(f=f, p=p, w=w)

  def inverse_transform(self, x):
    p = self.basis.p
    f = self.basis.f
    mesh = self.spmd_mesh
    einsum_args = (self.reverse_einsum_arg_order, self.transform_precision)


    x = _unstack_m(x, mesh)
    x = jax.named_call(_transform_einsum, name='inv_legendre')(
        'mjl,...sml->...smj', p, x, mesh, *einsum_args
    )
    if self.stacked_fourier_transforms:
      x = jax.named_call(_transform_einsum, name='inv_fourier')(
          'ism,...smj->...ij', f, x, mesh, *einsum_args
      )
    else:
      x = _stack_m(x, mesh)
      x = jax.named_call(_transform_einsum, name='inv_fourier')(
          'im,...mj->...ij', f, x, mesh, *einsum_args
      )
    return x

  def transform(self, x):
    w = self.basis.w
    f = self.basis.f
    p = self.basis.p
    mesh = self.spmd_mesh
    einsum_args = (self.reverse_einsum_arg_order, self.transform_precision)

    x = w * x
    if self.stacked_fourier_transforms:
      x = jax.named_call(_transform_einsum, name='fwd_fourier')(
          'ism,...ij->...smj', f, x, mesh, *einsum_args
      )
    else:
      x = jax.named_call(_transform_einsum, name='fwd_fourier')(
          'im,...ij->...mj', f, x, mesh, *einsum_args
      )
      x = _unstack_m(x, mesh)
    x = jax.named_call(_transform_einsum, name='fwd_legendre')(
        'mjl,...smj->...sml', p, x, mesh, *einsum_args
    )
    x = _stack_m(x, mesh)
    return x

  def longitudinal_derivative(self, x: Array) -> Array:
    return _fourier_derivative_for_real_basis_with_zero_imag(x, self.spmd_mesh)


@dataclasses.dataclass(frozen=True)
class ComplexSphericalHarmonics(SphericalHarmonics):


  @functools.cached_property
  def nodal_axes(self) -> tuple[np.ndarray, np.ndarray]:
    longitude, _ = fourier.quadrature_nodes(self.longitude_nodes)
    sin_latitude, _ = get_latitude_nodes(
        self.latitude_nodes, self.latitude_spacing
    )
    return longitude, sin_latitude

  @functools.cached_property
  def nodal_shape(self) -> tuple[int, int]:
    return (self.longitude_nodes, self.latitude_nodes)

  @functools.cached_property
  def nodal_padding(self) -> tuple[int, int]:
    return (0, 0)

  @functools.cached_property
  def modal_axes(self) -> tuple[np.ndarray, np.ndarray]:
    lon_wavenumbers = np.arange(self.longitude_wavenumbers)
    tot_wavenumbers = np.arange(self.total_wavenumbers)
    return lon_wavenumbers, tot_wavenumbers

  @functools.cached_property
  def modal_shape(self) -> tuple[int, int]:
    return (self.longitude_wavenumbers, self.total_wavenumbers)

  @functools.cached_property
  def modal_padding(self) -> tuple[int, int]:
    return (0, 0)

  @functools.cached_property
  def modal_dtype(self) -> np.dtype:
    return np.dtype(np.complex64)

  @functools.cached_property
  def mask(self) -> np.ndarray:
    m, l = np.meshgrid(*self.modal_axes, indexing='ij')
    return m <= l

  @functools.cached_property
  def basis(self) -> _SphericalHarmonicBasis:
    f = fourier.complex_basis(
        wavenumbers=self.longitude_wavenumbers,
        nodes=self.longitude_nodes,
    )
    _, wf = fourier.quadrature_nodes(self.longitude_nodes)
    x, wp = get_latitude_nodes(self.latitude_nodes, self.latitude_spacing)
    w = wf * wp
    p = associated_legendre.evaluate(
        n_m=self.longitude_wavenumbers, n_l=self.total_wavenumbers, x=x
    )
    return _SphericalHarmonicBasis(f=f, p=p, w=w)

  def inverse_transform(self, x):
    p = self.basis.p
    f = self.basis.f
    px = jax.named_call(einsum, name='inv_legendre')('mjl,...ml->...mj', p, x)
    fpx_from_real = jax.named_call(einsum, name='inv_fourier_from_real')(
        'im,...mj->...ij', jnp.real(f), jnp.real(px)
    )
    fpx_from_imag = jax.named_call(einsum, name='inv_fourier_from_imag')(
        'im,...mj->...ij', -jnp.imag(f), jnp.imag(px)
    )
    return fpx_from_real + fpx_from_imag

  def transform(self, x):
    w = self.basis.w
    f = self.basis.f
    p = self.basis.p
    wx = w * x
    fwx = jax.named_call(einsum, name='fwd_fourier')(
        'im,...ij->...mj', jnp.conj(f), wx
    )
    pfwx = jax.named_call(einsum, name='fwd_legendre')(
        'mjl,...mj->...ml', p, fwx
    )
    return pfwx

  def longitudinal_derivative(self, x: Array) -> Array:
    return fourier.complex_basis_derivative(x, axis=-2)


def _vertical_pad(
    field: jax.Array, mesh: jax.sharding.Mesh | None
) -> tuple[jax.Array, int | None]:
  if field.ndim < 3 or field.shape[0] == 1 or mesh is None:
    return field, None
  assert field.ndim == 3, field.shape
  z_multiple = mesh.shape['z']
  z_padding = _round_to_multiple(field.shape[0], z_multiple) - field.shape[0]
  return jnp.pad(field, [(0, z_padding), (0, 0), (0, 0)]), z_padding


def _vertical_crop(field: jax.Array, padding: int | None) -> jax.Array:
  if not padding:
    return field
  assert field.ndim == 3, field.shape
  return jax.lax.slice_in_dim(field, 0, -padding, axis=0)


def _with_vertical_padding(
    f: Callable[[jax.Array], jax.Array], mesh: jax.sharding.Mesh | None
) -> Callable[[jax.Array], jax.Array]:

  def g(x):
    x, padding = _vertical_pad(x, mesh)
    return _vertical_crop(f(x), padding)

  return g


SPHERICAL_HARMONICS_IMPL_KEY = 'spherical_harmonics_impl'
SPMD_MESH_KEY = 'spmd_mesh'


SphericalHarmonicsImpl = Callable[..., SphericalHarmonics]


@dataclasses.dataclass(frozen=True)
class Grid:


  longitude_wavenumbers: int = 0
  total_wavenumbers: int = 0
  longitude_nodes: int = 0
  latitude_nodes: int = 0
  latitude_spacing: str = 'gauss'
  longitude_offset: float = 0.0
  radius: float | None = None
  spherical_harmonics_impl: SphericalHarmonicsImpl = RealSphericalHarmonics
  spmd_mesh: jax.sharding.Mesh | None = None

  def __post_init__(self):
    if self.radius is None:
      object.__setattr__(self, 'radius', 1.0)

    if self.latitude_spacing not in LATITUDE_SPACINGS:
      raise ValueError(
          f'Unsupported `latitude_spacing` "{self.latitude_spacing}". '
          f'Supported values are: {list(LATITUDE_SPACINGS)}.'
      )

    if self.spmd_mesh is not None:
      if not {'x', 'y'} <= set(self.spmd_mesh.axis_names):
        raise ValueError(
            "mesh is missing one or more of the required axis names 'x' and "
            f"'y': {self.spmd_mesh}"
        )
      assert isinstance(
          self.spherical_harmonics, RealSphericalHarmonicsWithZeroImag
      )

  @classmethod
  def with_wavenumbers(
      cls,
      longitude_wavenumbers: int,
      dealiasing: str = 'quadratic',
      latitude_spacing: str = 'gauss',
      longitude_offset: float = 0.0,
      spherical_harmonics_impl: SphericalHarmonicsImpl = RealSphericalHarmonics,
      radius: float | None = None,
  ) -> Grid:


    order = {'linear': 2, 'quadratic': 3, 'cubic': 4}[dealiasing]
    longitude_nodes = order * longitude_wavenumbers + 1
    latitude_nodes = math.ceil(longitude_nodes / 2)

    return cls(
        longitude_wavenumbers=longitude_wavenumbers,
        total_wavenumbers=longitude_wavenumbers + 1,
        longitude_nodes=longitude_nodes,
        latitude_nodes=latitude_nodes,
        latitude_spacing=latitude_spacing,
        longitude_offset=longitude_offset,
        spherical_harmonics_impl=spherical_harmonics_impl,
        radius=radius,
    )

  @classmethod
  def construct(
      cls,
      max_wavenumber: int,
      gaussian_nodes: int,
      latitude_spacing: str = 'gauss',
      longitude_offset: float = 0.0,
      radius: float | None = None,
      spherical_harmonics_impl: SphericalHarmonicsImpl = RealSphericalHarmonics,
  ) -> Grid:


    return cls(
        longitude_wavenumbers=max_wavenumber + 1,
        total_wavenumbers=max_wavenumber + 2,
        longitude_nodes=4 * gaussian_nodes,
        latitude_nodes=2 * gaussian_nodes,
        latitude_spacing=latitude_spacing,
        longitude_offset=longitude_offset,
        spherical_harmonics_impl=spherical_harmonics_impl,
        radius=radius,
    )


  @classmethod
  def T21(cls, **kwargs) -> Grid:
    return cls.construct(max_wavenumber=21, gaussian_nodes=16, **kwargs)

  @classmethod
  def T31(cls, **kwargs) -> Grid:
    return cls.construct(max_wavenumber=31, gaussian_nodes=24, **kwargs)

  @classmethod
  def T42(cls, **kwargs) -> Grid:
    return cls.construct(max_wavenumber=42, gaussian_nodes=32, **kwargs)

  @classmethod
  def T85(cls, **kwargs) -> Grid:
    return cls.construct(max_wavenumber=85, gaussian_nodes=64, **kwargs)

  @classmethod
  def T106(cls, **kwargs) -> Grid:
    return cls.construct(max_wavenumber=106, gaussian_nodes=80, **kwargs)

  @classmethod
  def T119(cls, **kwargs) -> Grid:
    return cls.construct(max_wavenumber=119, gaussian_nodes=90, **kwargs)

  @classmethod
  def T170(cls, **kwargs) -> Grid:
    return cls.construct(max_wavenumber=170, gaussian_nodes=128, **kwargs)

  @classmethod
  def T213(cls, **kwargs) -> Grid:
    return cls.construct(max_wavenumber=213, gaussian_nodes=160, **kwargs)

  @classmethod
  def T340(cls, **kwargs) -> Grid:
    return cls.construct(max_wavenumber=340, gaussian_nodes=256, **kwargs)

  @classmethod
  def T425(cls, **kwargs) -> Grid:
    return cls.construct(max_wavenumber=425, gaussian_nodes=320, **kwargs)


  @classmethod
  def TL31(cls, **kwargs) -> Grid:
    return cls.construct(max_wavenumber=31, gaussian_nodes=16, **kwargs)

  @classmethod
  def TL47(cls, **kwargs) -> Grid:
    return cls.construct(max_wavenumber=47, gaussian_nodes=24, **kwargs)

  @classmethod
  def TL63(cls, **kwargs) -> Grid:
    return cls.construct(max_wavenumber=63, gaussian_nodes=32, **kwargs)

  @classmethod
  def TL95(cls, **kwargs) -> Grid:
    return cls.construct(max_wavenumber=95, gaussian_nodes=48, **kwargs)

  @classmethod
  def TL127(cls, **kwargs) -> Grid:
    return cls.construct(max_wavenumber=127, gaussian_nodes=64, **kwargs)

  @classmethod
  def TL159(cls, **kwargs) -> Grid:
    return cls.construct(max_wavenumber=159, gaussian_nodes=80, **kwargs)

  @classmethod
  def TL179(cls, **kwargs) -> Grid:
    return cls.construct(max_wavenumber=179, gaussian_nodes=90, **kwargs)

  @classmethod
  def TL255(cls, **kwargs) -> Grid:
    return cls.construct(max_wavenumber=255, gaussian_nodes=128, **kwargs)

  @classmethod
  def TL639(cls, **kwargs) -> Grid:
    return cls.construct(max_wavenumber=639, gaussian_nodes=320, **kwargs)

  @classmethod
  def TL1279(cls, **kwargs) -> Grid:
    return cls.construct(max_wavenumber=1279, gaussian_nodes=640, **kwargs)


  def asdict(self) -> dict[str, Any]:

    items = dataclasses.asdict(self)
    items[SPHERICAL_HARMONICS_IMPL_KEY] = self.spherical_harmonics_impl.__name__
    if self.spmd_mesh is not None:
      items[SPMD_MESH_KEY] = ','.join(
          f'{k}={v}' for k, v in self.spmd_mesh.shape.items()
      )
    else:
      items[SPMD_MESH_KEY] = ''
    return items


  @functools.cached_property
  def spherical_harmonics(self) -> SphericalHarmonics:

    if self.spmd_mesh is not None:
      kwargs = dict(spmd_mesh=self.spmd_mesh)
    else:
      kwargs = dict()
    return self.spherical_harmonics_impl(
        longitude_wavenumbers=self.longitude_wavenumbers,
        total_wavenumbers=self.total_wavenumbers,
        longitude_nodes=self.longitude_nodes,
        latitude_nodes=self.latitude_nodes,
        latitude_spacing=self.latitude_spacing,
        **kwargs,
    )

  @property
  def longitudes(self) -> np.ndarray:
    return self.nodal_axes[0]

  @property
  def latitudes(self) -> np.ndarray:
    return np.arcsin(self.nodal_axes[1])

  @functools.cached_property
  def nodal_axes(self) -> tuple[np.ndarray, np.ndarray]:

    lon, sin_lat = self.spherical_harmonics.nodal_axes
    return lon + self.longitude_offset, sin_lat

  @functools.cached_property
  def nodal_shape(self) -> tuple[int, int]:
    return self.spherical_harmonics.nodal_shape

  @functools.cached_property
  def nodal_padding(self) -> tuple[int, int]:
    return self.spherical_harmonics.nodal_padding

  @functools.cached_property
  def nodal_mesh(self) -> tuple[np.ndarray, np.ndarray]:
    return np.meshgrid(*self.nodal_axes, indexing='ij')

  @functools.cached_property
  def modal_axes(self) -> tuple[np.ndarray, np.ndarray]:

    return self.spherical_harmonics.modal_axes

  @functools.cached_property
  def modal_shape(self) -> tuple[int, int]:
    return self.spherical_harmonics.modal_shape

  @functools.cached_property
  def modal_padding(self) -> tuple[int, int]:
    return self.spherical_harmonics.modal_padding

  @functools.cached_property
  def mask(self) -> np.ndarray:

    return self.spherical_harmonics.mask

  @functools.cached_property
  def modal_mesh(self) -> tuple[np.ndarray, np.ndarray]:

    return np.meshgrid(*self.spherical_harmonics.modal_axes, indexing='ij')

  @functools.cached_property
  def cos_lat(self) -> jnp.ndarray:
    _, sin_lat = self.nodal_axes
    return np.sqrt(1 - sin_lat**2)

  @functools.cached_property
  def sec2_lat(self) -> jnp.ndarray:
    _, sin_lat = self.nodal_axes
    return 1 / (1 - sin_lat**2)

  @functools.cached_property
  def laplacian_eigenvalues(self) -> np.ndarray:
    _, l = self.modal_axes
    return -l * (l + 1) / (self.radius**2)


  @jax.named_call
  def to_nodal(self, x: typing.Pytree) -> typing.Pytree:

    f = _with_vertical_padding(
        self.spherical_harmonics.inverse_transform, self.spmd_mesh
    )
    return pytree_utils.tree_map_over_nonscalars(f, x)

  @jax.named_call
  def to_modal(self, z: typing.Pytree) -> typing.Pytree:

    f = _with_vertical_padding(
        self.spherical_harmonics.transform, self.spmd_mesh
    )
    return pytree_utils.tree_map_over_nonscalars(f, z)

  @jax.named_call
  def laplacian(self, x: Array) -> jnp.ndarray:

    return x * self.laplacian_eigenvalues

  @jax.named_call
  def inverse_laplacian(self, x: Array) -> jnp.ndarray:

    with np.errstate(divide='ignore', invalid='ignore'):
      inverse_eigenvalues = 1 / self.laplacian_eigenvalues
    inverse_eigenvalues[0] = 0
    inverse_eigenvalues[self.total_wavenumbers :] = 0
    assert not np.isnan(inverse_eigenvalues).any()
    return x * inverse_eigenvalues

  @jax.named_call
  def clip_wavenumbers(self, x: typing.Pytree, n: int = 1) -> typing.Pytree:

    if n <= 0:
      raise ValueError(f'`n` must be >= 0; got {n}.')

    def clip(x):


      num_zeros = n + self.modal_padding[-1]
      mask = jnp.ones(self.modal_shape[-1], x.dtype).at[-num_zeros:].set(0)
      return x * mask

    return pytree_utils.tree_map_over_nonscalars(clip, x)

  @functools.cached_property
  def _derivative_recurrence_weights(self) -> tuple[np.ndarray, np.ndarray]:
    m, l = self.modal_mesh
    a = np.sqrt(self.mask * (l**2 - m**2) / (4 * l**2 - 1))
    a[:, 0] = 0
    b = np.sqrt(self.mask * ((l + 1) ** 2 - m**2) / (4 * (l + 1) ** 2 - 1))
    b[:, -1] = 0
    return a, b

  @jax.named_call
  def d_dlon(self, x: Array) -> Array:

    return _with_vertical_padding(
        self.spherical_harmonics.longitudinal_derivative, self.spmd_mesh
    )(x)

  @jax.named_call
  def cos_lat_d_dlat(self, x: Array) -> Array:

    _, l = self.modal_mesh
    a, b = self._derivative_recurrence_weights
    x_lm1 = jax_numpy_utils.shift(((l + 1) * a) * x, -1, axis=-1)
    x_lp1 = jax_numpy_utils.shift((-l * b) * x, +1, axis=-1)
    return x_lm1 + x_lp1

  @jax.named_call
  def sec_lat_d_dlat_cos2(self, x: Array) -> Array:

    _, l = self.modal_mesh
    a, b = self._derivative_recurrence_weights
    x_lm1 = jax_numpy_utils.shift(((l - 1) * a) * x, -1, axis=-1)
    x_lp1 = jax_numpy_utils.shift((-(l + 2) * b) * x, +1, axis=-1)
    return x_lm1 + x_lp1

  @jax.named_call
  def cos_lat_grad(self, x: Array, clip: bool = True) -> Array:


    raw = self.d_dlon(x) / self.radius, self.cos_lat_d_dlat(x) / self.radius
    if clip:
      return self.clip_wavenumbers(raw)
    return raw

  @jax.named_call
  def k_cross(self, v: ArrayOrArrayTuple) -> Array:

    return -v[1], v[0]

  @jax.named_call
  def div_cos_lat(
      self,
      v: ArrayOrArrayTuple,
      clip: bool = True,
  ) -> Array:

    raw = (self.d_dlon(v[0]) + self.sec_lat_d_dlat_cos2(v[1])) / self.radius
    if clip:
      return self.clip_wavenumbers(raw)
    return raw

  @jax.named_call
  def curl_cos_lat(
      self,
      v: ArrayOrArrayTuple,
      clip: bool = True,
  ) -> Array:

    raw = (self.d_dlon(v[1]) - self.sec_lat_d_dlat_cos2(v[0])) / self.radius
    if clip:
      return self.clip_wavenumbers(raw)
    return raw

  @property
  def quadrature_weights(self) -> np.ndarray:

    return np.broadcast_to(self.spherical_harmonics.basis.w, self.nodal_shape)

  @jax.named_call
  def integrate(self, z: Array) -> Array:

    w = self.spherical_harmonics.basis.w * self.radius**2
    return einsum('y,...xy->...', w, z)


@jax.named_call
def get_cos_lat_vector(
    vorticity: Array,
    divergence: Array,
    grid: Grid,
    clip: bool = True,
) -> Array:

  stream_function = grid.inverse_laplacian(vorticity)
  velocity_potential = grid.inverse_laplacian(divergence)
  return jax.tree_util.tree_map(
      lambda x, y: x + y,
      grid.cos_lat_grad(velocity_potential, clip=clip),
      grid.k_cross(grid.cos_lat_grad(stream_function, clip=clip)),
  )


@functools.partial(jax.jit, static_argnames=('grid', 'clip'))
def uv_nodal_to_vor_div_modal(
    grid: Grid,
    u_nodal: Array,
    v_nodal: Array,
    clip: bool = True,
) -> tuple[Array, Array]:

  u_over_cos_lat = grid.to_modal(u_nodal / grid.cos_lat)
  v_over_cos_lat = grid.to_modal(v_nodal / grid.cos_lat)

  vorticity = grid.curl_cos_lat((u_over_cos_lat, v_over_cos_lat), clip=clip)
  divergence = grid.div_cos_lat((u_over_cos_lat, v_over_cos_lat), clip=clip)
  return vorticity, divergence


@functools.partial(jax.jit, static_argnames=('grid', 'clip'))
def vor_div_to_uv_nodal(
    grid: Grid,
    vorticity: Array,
    divergence: Array,
    clip: bool = True,
) -> tuple[Array, Array]:

  u_cos_lat, v_cos_lat = get_cos_lat_vector(
      vorticity, divergence, grid, clip=clip
  )
  u_nodal = grid.to_nodal(u_cos_lat) / grid.cos_lat
  v_nodal = grid.to_nodal(v_cos_lat) / grid.cos_lat
  return u_nodal, v_nodal
