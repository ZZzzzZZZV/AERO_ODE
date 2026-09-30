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
from typing import Callable, Protocol, Sequence

from flax import nnx
import jax
import jax.numpy as jnp
from neuralgcm.experimental import typing


default_w_init = nnx.initializers.lecun_normal()
default_b_init = nnx.initializers.zeros_init()


class UnaryLayer(Protocol):


  def __init__(self, input_size: int, output_size: int, **kwargs):
    ...

  def __call__(self, inputs: jax.Array) -> jax.Array:
    ...

  @property
  def input_size(self) -> int:
    ...

  @property
  def output_size(self) -> int:
    ...


UnaryLayerFactory = Callable[..., UnaryLayer]


class Mlp(nnx.Module):


  def __init__(
      self,
      input_size: int,
      output_size: int,
      *,
      intermediate_sizes: Sequence[int],
      activation: Callable[[jax.Array], jax.Array] = jax.nn.relu,
      w_init: nnx.initializers.Initializer = default_w_init,
      b_init: nnx.initializers.Initializer = default_b_init,
      use_bias: bool = True,
      w_init_final: nnx.initializers.Initializer | None = None,
      b_init_final: nnx.initializers.Initializer | None = None,
      activate_final: bool = False,
      dtype: typing.Dtype = jnp.float32,
      rngs: nnx.Rngs,
  ):

    self.input_size = input_size
    self.output_size = output_size
    output_sizes = tuple(intermediate_sizes) + (output_size,)
    if w_init_final is None:
      w_init_final = w_init
    if b_init_final is None:
      b_init_final = b_init
    self.activation = activation
    self.activate_final = activate_final
    output_sizes = tuple(output_sizes)
    input_sizes = (input_size,) + output_sizes[:-1]
    layers = []
    for i, (d_in, d_out) in enumerate(zip(input_sizes, output_sizes)):
      layers.append(
          nnx.Linear(
              in_features=d_in,
              out_features=d_out,
              kernel_init=w_init if i < len(input_sizes) - 1 else w_init_final,
              bias_init=b_init if i < len(input_sizes) - 1 else b_init_final,
              use_bias=use_bias,
              param_dtype=dtype,
              rngs=rngs,
          )
      )
    self.layers = tuple(layers)

  def __call__(
      self,
      inputs: jax.Array,
  ) -> jax.Array:

    num_layers = len(self.layers)
    out = inputs
    for i, layer in enumerate(self.layers):
      out = layer(out)
      if i < (num_layers - 1) or self.activate_final:
        out = self.activation(out)
    return out


class MlpUniform(Mlp):


  def __init__(
      self,
      input_size: int,
      output_size: int,
      *,
      hidden_size: int,
      n_hidden_layers: int,
      activation: Callable[[jax.Array], jax.Array] = jax.nn.gelu,
      w_init: nnx.initializers.Initializer = default_w_init,
      b_init: nnx.initializers.Initializer = default_b_init,
      use_bias: bool = True,
      w_init_final: nnx.initializers.Initializer | None = None,
      b_init_final: nnx.initializers.Initializer | None = None,
      activate_final: bool = False,
      dtype: typing.Dtype = jnp.float32,
      rngs: nnx.Rngs,
  ):
    intermediate_sizes = (hidden_size,) * n_hidden_layers
    super().__init__(
        input_size=input_size,
        output_size=output_size,
        intermediate_sizes=intermediate_sizes,
        activation=activation,
        w_init=w_init,
        b_init=b_init,
        use_bias=use_bias,
        w_init_final=w_init_final,
        b_init_final=b_init_final,
        activate_final=activate_final,
        dtype=dtype,
        rngs=rngs,
    )


def conv_dilated_ncw(
    inputs,
    kernel,
    strides,
    padding_lax,
    lhs_dilation,
    rhs_dilation,
    dimension_numbers,
    feature_group_count,
    precision,
):

  del dimension_numbers
  return jax.lax.conv_general_dilated(
      inputs,
      kernel,
      strides,
      padding_lax,
      lhs_dilation=lhs_dilation,
      rhs_dilation=rhs_dilation,
      dimension_numbers=('NCW', 'WIO', 'NCW'),
      feature_group_count=feature_group_count,
      precision=precision,
  )


class ConvLevel(nnx.Conv):


  def __init__(
      self,
      input_size: int,
      output_size: int,
      *,
      kernel_size: int,
      strides: int = 1,
      padding: str = 'SAME',
      input_dilation: int = 1,
      kernel_dilation: int = 1,
      use_bias: bool = True,
      precision: nnx.nnx.nn.linear.PrecisionLike = None,
      w_init: nnx.initializers.Initializer = default_w_init,
      b_init: nnx.initializers.Initializer = default_b_init,
      rngs: nnx.Rngs,
  ):
    self.input_size = input_size
    self.output_size = output_size

    ncw_b_init = lambda key, shape, dtype: b_init(key, shape + (1,), dtype)
    super().__init__(
        in_features=input_size,
        out_features=output_size,
        kernel_size=kernel_size,
        strides=strides,
        padding=padding,
        input_dilation=input_dilation,
        kernel_dilation=kernel_dilation,
        use_bias=use_bias,
        precision=precision,
        kernel_init=w_init,
        bias_init=ncw_b_init,
        conv_general_dilated=conv_dilated_ncw,
        rngs=rngs,
    )


class CnnLevel(nnx.Module):


  def __init__(
      self,
      input_size: int,
      output_size: int,
      *,
      channels: Sequence[int],
      kernel_sizes: int | Sequence[int],
      strides: int | Sequence[int] = 1,
      padding: str = 'SAME',
      input_dilation: int = 1,
      kernel_dilations: int | Sequence[int] = 1,
      use_bias: bool = True,
      precision: nnx.nnx.nn.linear.PrecisionLike = None,
      w_init: nnx.initializers.Initializer = default_w_init,
      b_init: nnx.initializers.Initializer = default_b_init,
      w_init_final: nnx.initializers.Initializer = default_w_init,
      b_init_final: nnx.initializers.Initializer = default_b_init,
      activation: Callable[[jax.Array], jax.Array] = jax.nn.gelu,
      activate_final: bool = False,
      rngs: nnx.Rngs,
  ):
    self.input_size = input_size
    self.output_size = output_size
    n_hidden = len(channels)
    n_total = n_hidden + 1
    if isinstance(kernel_sizes, int):
      kernel_sizes = [kernel_sizes] * n_total
    if isinstance(kernel_dilations, int):
      kernel_dilations = [kernel_dilations] * n_total
    channels = list(channels) + [output_size]
    if len(set([len(channels), len(kernel_sizes), len(kernel_dilations)])) != 1:
      raise ValueError(
          f'Missing kernel | dilation specs for {n_total=} '
          f'layers, got {kernel_sizes=}, {kernel_dilations=}.'
      )
    w_inits = [w_init] * n_hidden + [w_init_final]
    b_inits = [b_init] * n_hidden + [b_init_final]
    params = zip(channels, kernel_sizes, kernel_dilations, w_inits, b_inits)
    self.layers = []
    din = input_size
    for dout, kernel, dilation, w_init_i, b_init_i in params:
      self.layers.append(
          ConvLevel(
              input_size=din,
              output_size=dout,
              kernel_size=kernel,
              strides=strides,
              input_dilation=input_dilation,
              kernel_dilation=dilation,
              padding=padding,
              use_bias=use_bias,
              w_init=w_init_i,
              b_init=b_init_i,
              precision=precision,
              rngs=rngs,
          )
      )
      din = dout
    self.activation = activation
    self.activate_final = activate_final

  def __call__(self, inputs: jax.Array) -> jax.Array:
    out = inputs
    num_layers = len(self.layers)
    for i, layer in enumerate(self.layers):
      out = layer(out)
      if i < (num_layers - 1) or self.activate_final:
        out = self.activation(out)
    return out
