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
from typing import Callable, Optional, Sequence
from dinosaur import pytree_utils
from dinosaur import typing
import gin
import haiku as hk
import jax

from neuralgcm import transforms


Array = typing.Array
Tower = Callable[[int], Callable[..., Array]]
MappingModule = Callable[[typing.Pytree], typing.Pytree]


@gin.register(denylist=['output_shapes'])
class NodalMapping(hk.Module):


  def __init__(
      self,
      output_shapes: typing.Pytree,
      tower_factory: Tower = gin.REQUIRED,
      name: Optional[str] = None,
  ):
    super().__init__(name=name)
    feature_axis = -3
    output_size = sum([x[feature_axis]
                       for x in jax.tree_util.tree_leaves(output_shapes)])

    self.tower = tower_factory(output_size)
    self.output_shapes = output_shapes
    self.feature_axis = feature_axis

  def __call__(self, inputs: typing.Pytree) -> typing.Pytree:
    array = pytree_utils.pack_pytree(inputs, self.feature_axis)
    if array.ndim != 3:
      raise ValueError(f'Expected input array with ndim=3, got {array.shape=}')
    outputs = self.tower(array)
    if outputs.ndim != 3:
      raise ValueError(f'Expected outputs with ndim=3, got {outputs.shape=}')
    return pytree_utils.unpack_to_pytree(
        outputs, self.output_shapes, self.feature_axis)


@gin.register(denylist=['output_shapes'])
class NodalVolumeMapping(hk.Module):


  def __init__(
      self,
      output_shapes: typing.Pytree,
      tower_factory: Tower = gin.REQUIRED,
      name: Optional[str] = None
  ):
    super().__init__(name=name)
    feature_axis = 0
    output_size = len(jax.tree_util.tree_leaves(output_shapes))


    self.tower = tower_factory(output_size)
    self.output_shapes = output_shapes
    self.feature_axis = feature_axis

  def __call__(self, inputs: typing.Pytree) -> typing.Pytree:
    array = pytree_utils.stack_pytree(inputs, axis=self.feature_axis)
    if array.ndim != 4:
      raise ValueError(f'Expected input array with ndim=4, got {array.shape=}')
    outputs = self.tower(array)
    if outputs.ndim != 4:
      raise ValueError(f'Expected outputs with ndim=4, got {outputs.shape=}')
    return pytree_utils.unstack_to_pytree(
        outputs, self.output_shapes, axis=self.feature_axis
    )


@gin.register
class NodalVolumeTransformerMapping(hk.Module):


  def __init__(
      self,
      output_shapes: typing.Pytree,
      encoder_transformer_tower_factory: Tower = gin.REQUIRED,
      decoder_transformer_tower_factory: Tower = gin.REQUIRED,
      latent_size: int = gin.REQUIRED,
      encoder_inputs_selection_module=gin.REQUIRED,
      decoder_inputs_selection_module=transforms.EmptyTransform,
      encoder_pos_encoding_module=transforms.EmptyTransform,
      decoder_pos_encoding_module=transforms.EmptyTransform,
      name: Optional[str] = None
  ):
    super().__init__(name=name)
    feature_axis = 0
    output_size = len(jax.tree_util.tree_leaves(output_shapes))
    self.encoder_tower = encoder_transformer_tower_factory(latent_size)
    self.decoder_tower = decoder_transformer_tower_factory(output_size)
    self.output_shapes = output_shapes
    self.feature_axis = feature_axis
    self.get_encoder_inputs_fn = encoder_inputs_selection_module()
    self.get_decode_inputs_fn = decoder_inputs_selection_module()
    self.encoder_positional_encodings_fn = encoder_pos_encoding_module()
    self.decoder_positional_encodings_fn = decoder_pos_encoding_module()

  def __call__(self, inputs: typing.Pytree) -> typing.Pytree:
    enc_inputs = self.get_encoder_inputs_fn(inputs)
    dec_inputs = self.get_decode_inputs_fn(inputs)
    enc_array = pytree_utils.stack_pytree(enc_inputs, axis=self.feature_axis)
    dec_array = pytree_utils.stack_pytree(dec_inputs, axis=self.feature_axis)
    enc_pos_encoding = pytree_utils.stack_pytree(
        self.encoder_positional_encodings_fn(inputs), axis=self.feature_axis)
    dec_pos_encoding = pytree_utils.stack_pytree(
        self.decoder_positional_encodings_fn(inputs), axis=self.feature_axis)
    input_ndims = set(
        x.ndim
        for x in [enc_array, dec_array, enc_pos_encoding, dec_pos_encoding]
        if x is not None)
    if input_ndims != {4}:
      raise ValueError(f'Expected all inputs have ndim=4, got {input_ndims=}')
    latents = self.encoder_tower(enc_array, None, enc_pos_encoding)

    decoder_latents = None if dec_array is None else latents

    dec_array = dec_array if dec_array is not None else latents
    outputs = self.decoder_tower(dec_array, decoder_latents, dec_pos_encoding)
    if outputs.ndim != 4:
      raise ValueError(f'Expected outputs with ndim=4, got {outputs.shape=}')
    return pytree_utils.unstack_to_pytree(
        outputs, self.output_shapes, axis=self.feature_axis
    )


@gin.register(denylist=['output_shapes'])
class ParallelMapping(hk.Module):


  def __init__(
      self,
      output_shapes: typing.Pytree,
      mappings: Sequence[MappingModule] = gin.REQUIRED,
      name: Optional[str] = None,
  ):
    super().__init__(name=name)
    self.mapping_fns = [m(output_shapes) for m in mappings]

  def __call__(self, inputs: typing.Pytree) -> typing.Pytree:
    results = [mapping_fn(inputs) for mapping_fn in self.mapping_fns]
    return jax.tree_util.tree_map(lambda *args: sum(args), *results)
