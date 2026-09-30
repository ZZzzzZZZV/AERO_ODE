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
from typing import Callable, Optional, Sequence, Tuple
from dinosaur import typing
import gin
import haiku as hk
import jax
import jax.numpy as jnp

from neuralgcm import initializers

Array = typing.Array
GatingFactory = typing.GatingFactory
TowerFactory = typing.TowerFactory
MLP = gin.external_configurable(hk.nets.MLP)


relu = gin.external_configurable(jax.nn.relu)
gelu = gin.external_configurable(jax.nn.gelu)
silu = gin.external_configurable(jax.nn.silu)


@gin.register(denylist=['output_size'])
class MlpUniform(hk.nets.MLP):


  def __init__(
      self,
      output_size: int,
      num_hidden_units: int = gin.REQUIRED,
      num_hidden_layers: int = gin.REQUIRED,
      w_init: Optional[hk.initializers.Initializer] = None,
      b_init: Optional[hk.initializers.Initializer] = None,
      with_bias: bool = True,
      activation: Callable[[jnp.ndarray], jnp.ndarray] = jax.nn.relu,
      activate_final: bool = False,
      w_init_final: Optional[hk.initializers.Initializer] = None,
      b_init_final: Optional[hk.initializers.Initializer] = None,
      name: Optional[str] = None,
  ):
    hidden_output_sizes = [num_hidden_units] * num_hidden_layers
    super().__init__(
        hidden_output_sizes,
        w_init=w_init,
        b_init=b_init,
        with_bias=with_bias,
        activation=activation,
        activate_final=True,
        name=name,
    )
    self.linear_final = hk.Linear(
        output_size=output_size,
        w_init=w_init_final,
        b_init=b_init_final,
        with_bias=with_bias,
        name='linear_%d' % num_hidden_layers,
    )
    self.activate_linear_final = activate_final

  def __call__(
      self,
      inputs: jax.Array,
      dropout_rate: Optional[float] = None,
      rng: Optional[jax.Array] = None,
  ) -> jax.Array:
    out = super().__call__(inputs, dropout_rate=dropout_rate, rng=rng)
    out = self.linear_final(out)
    if self.activate_linear_final:
      out = self.activation(out)
    return out


@gin.register(denylist=['output_size'])
class ConvLonLat(hk.Module):


  def __init__(
      self,
      output_size: int,
      kernel_shape: Tuple[int, int] = gin.REQUIRED,
      with_bias: bool = True,
      name: Optional[str] = None,
  ):
    super().__init__(name=name)
    self._padding = []
    for kernel_size in kernel_shape:
      pad_left = kernel_size // 2
      self._padding.append((pad_left, kernel_size - pad_left - 1))

    self._conv_module = hk.Conv2D(
        output_channels=output_size,
        kernel_shape=kernel_shape,
        with_bias=with_bias,
        padding='VALID',
        data_format='NCHW',
    )


  def __call__(self, inputs: Array) -> Array:


    inputs = jnp.pad(inputs, [(0, 0), self._padding[0], (0, 0)], mode='wrap')


    inputs = jnp.pad(
        inputs, [(0, 0), (0, 0), self._padding[1]], mode='constant'
    )
    return self._conv_module(inputs)


@gin.register
class ConvLevel(hk.Conv1D):


  def __init__(
      self,
      output_channels: int,
      kernel_shape: int,
      dilation_rate: int = 1,
      padding: str = 'SAME',
      with_bias: bool = True,
      w_init: Optional[hk.initializers.Initializer] = None,
      b_init: Optional[hk.initializers.Initializer] = None,
      data_format: str = 'NCW',
      name: Optional[str] = None,
  ):
    super().__init__(
        output_channels=output_channels,
        kernel_shape=kernel_shape,
        rate=dilation_rate,
        padding=padding,
        with_bias=with_bias,
        w_init=w_init,
        b_init=b_init,
        data_format=data_format,
        name=name,
    )


@gin.register
class VerticalConvNet(hk.Module):


  def __init__(
      self,
      output_size: int,
      channels: Sequence[int],
      kernel_shapes: int | Sequence[int],
      dilation_rates: int | Sequence[int],
      padding: str = 'SAME',
      with_bias: bool = True,
      w_init: Optional[hk.initializers.Initializer] = None,
      b_init: Optional[hk.initializers.Initializer] = None,
      data_format: str = 'NCW',
      activation: Callable[[jnp.ndarray], jnp.ndarray] = jax.nn.relu,
      activate_final: bool = False,
      w_init_final: Optional[hk.initializers.Initializer] = None,
      b_init_final: Optional[hk.initializers.Initializer] = None,
      name: Optional[str] = None,
  ):
    super().__init__(name=name)
    n_hidden = len(channels)
    if isinstance(kernel_shapes, int):
      kernel_shapes = [kernel_shapes] * (n_hidden + 1)
    if isinstance(dilation_rates, int):
      dilation_rates = [dilation_rates] * (n_hidden + 1)
    channels = list(channels) + [output_size]
    if len(set([len(channels), len(kernel_shapes), len(dilation_rates)])) != 1:
      raise ValueError(
          f'Missing kernel|dilation specs for {n_hidden + 1} '
          f'layers, got {kernel_shapes=}, {dilation_rates=}.'
      )
    w_inits = [w_init] * n_hidden + [w_init_final]
    b_inits = [b_init] * n_hidden + [b_init_final]
    params = zip(channels, kernel_shapes, dilation_rates, w_inits, b_inits)
    self.layers = []
    for c, kernel, dilation, w_init_i, b_init_i in params:
      self.layers.append(
          ConvLevel(
              output_channels=c,
              kernel_shape=kernel,
              dilation_rate=dilation,
              padding=padding,
              with_bias=with_bias,
              w_init=w_init_i,
              b_init=b_init_i,
              data_format=data_format,
          )
      )
    self.activation = activation
    self.activate_final = activate_final

  def __call__(self, inputs: Array) -> Array:
    out = inputs
    num_layers = len(self.layers)
    for i, layer in enumerate(self.layers):
      out = layer(out)
      if i < (num_layers - 1) or self.activate_final:
        out = self.activation(out)
    return out


@gin.register
class LevelTransformer(hk.Module):


  def __init__(
      self,
      output_size: int,
      latent_size: int,
      n_layers: int,
      num_heads: int,
      key_size: int,
      widening_factor: int = 2,
      activation: Callable[[jnp.ndarray], jnp.ndarray] = jax.nn.gelu,
      input_projection_net: TowerFactory = hk.Linear,
      skip_final_projection: bool = False,
      gating_module: GatingFactory = lambda: lambda x, y: x + y,
      name: Optional[str] = None,
  ):
    super().__init__(name=name)
    value_size, reminder = divmod(latent_size, num_heads)
    if reminder != 0:
      raise ValueError(f'{latent_size=} is not divisible by {num_heads=}.')

    self.output_size = output_size
    self.latent_size = latent_size
    self.n_layers = n_layers
    self.num_heads = num_heads
    self.key_size = key_size
    self.value_size = value_size
    self.wide_latent_size = widening_factor * latent_size
    self.activation = activation
    self.w_init = hk.initializers.VarianceScaling(2 / self.n_layers)
    self.gating_fn = gating_module()

    if input_projection_net is not None:
      self.project_input_fn = input_projection_net(latent_size)
    else:

      def skip_with_check_fn(inputs):
        _, d = inputs.shape
        if d != latent_size:
          raise ValueError(
              f'{inputs.shape=} not compatible with {latent_size=}'
              ' Specify projection module in the transformer.'
          )
        return inputs

      self.project_input_fn = skip_with_check_fn
    if skip_final_projection:
      if output_size != self.latent_size:
        raise ValueError(
            f'Unable to skip projection for {output_size=}, '
            f'{self.latent_size=}.'
        )
      self.final_projection = lambda x: x
    else:
      self.final_projection = hk.Linear(output_size)

  @hk.transparent
  def layer_norm(self, x: jnp.ndarray) -> jnp.ndarray:

    ln = hk.LayerNorm(axis=-1, create_scale=True, create_offset=True)
    return ln(x)

  def __call__(
      self,
      inputs: Array,
      latents: Optional[Array] = None,
      positional_encoding: Optional[Array] = None,
  ) -> Array:

    inputs = jnp.transpose(inputs)
    h = self.project_input_fn(inputs)
    if latents is not None:
      latents = jnp.transpose(latents)
    if positional_encoding is not None:
      init_query_input = jnp.transpose(positional_encoding)
      special_query_stage = 0
    else:
      special_query_stage = -1
    h_dense = None
    last_layer_id = self.n_layers - 1
    for layer_id in range(self.n_layers - 1):

      h = self.gating_fn(h, h_dense) if h_dense is not None else h

      h_norm = self.layer_norm(h)
      attn_block = hk.MultiHeadAttention(
          num_heads=self.num_heads,
          key_size=self.key_size,
          value_size=self.value_size,
          model_size=self.latent_size,
          w_init=self.w_init,
      )

      h_attn = attn_block(
          query=init_query_input if layer_id == special_query_stage else h_norm,
          key=latents if latents is not None else h_norm,
          value=latents if latents is not None else h_norm,
      )

      h = self.gating_fn(h, h_attn)
      if layer_id != last_layer_id:
        dense_block = hk.Sequential([
            hk.Linear(self.wide_latent_size, w_init=self.w_init),
            self.activation,
            hk.Linear(self.latent_size, w_init=self.w_init),
        ])
        h_dense = dense_block(self.layer_norm(h))

    h_dense = self.final_projection(h)
    h_dense = jnp.transpose(h_dense)
    return h_dense
