# Copyright 2022 Google LLC.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Attention module library."""

import functools
from typing import Any, Dict, Iterable, Mapping, Optional, Tuple, Union
from einops import rearrange, repeat
from flax import linen as nn
import jax
import jax.numpy as jnp
from statm.modules import misc

Shape = Tuple[int]

DType = Any
Array = jnp.ndarray
ArrayTree = Union[Array, Iterable["ArrayTree"], Mapping[str, "ArrayTree"]]  # pytype: disable=not-supported-yet
ProcessorState = ArrayTree
PRNGKey = Array
NestedDict = Dict[str, Any]


class SlotAttention(nn.Module):
    """Slot Attention module.

  Note: This module uses pre-normalization by default.
  """

    num_iterations: int = 1
    qkv_size: Optional[int] = None
    mlp_size: Optional[int] = None
    epsilon: float = 1e-8
    num_heads: int = 1

    @nn.compact
    def __call__(self, slots: Array, inputs: Array,
                 padding_mask: Optional[Array] = None,
                 train: bool = False) -> Array:
        """Slot Attention module forward pass."""
        del padding_mask, train  # Unused.

        qkv_size = self.qkv_size or slots.shape[-1]
        head_dim = qkv_size // self.num_heads
        dense = functools.partial(nn.DenseGeneral,
                                  axis=-1, features=(self.num_heads, head_dim),
                                  use_bias=False)

        # Shared modules.
        dense_q = dense(name="general_dense_q_0")
        layernorm_q = nn.LayerNorm()
        inverted_attention = InvertedDotProductAttention(
            norm_type="mean", multi_head=self.num_heads > 1)
        gru = misc.GRU()

        if self.mlp_size is not None:
            mlp = misc.MLP(hidden_size=self.mlp_size, layernorm="pre", residual=True)  # type: ignore

        inputs = nn.LayerNorm()(inputs)
        k = dense(name="general_dense_k_0")(inputs)
        v = dense(name="general_dense_v_0")(inputs)

        # Multiple rounds of attention.
        for _ in range(self.num_iterations):

            # Inverted dot-product attention.
            slots_n = layernorm_q(slots)
            q = dense_q(slots_n)  # q.shape = (..., n_inputs, slot_size).
            updates = inverted_attention(query=q, key=k, value=v)

            # Recurrent update.
            slots = gru(slots, updates)

            # Feedforward block with pre-normalization. 原始的slot-attention有一个mlp，这里没有 hidden_size=128 or 256
            if self.mlp_size is not None:
                slots = mlp(slots)

        return slots


class InvertedDotProductAttention(nn.Module):
    """Inverted version of dot-product attention (softmax over query axis)."""

    norm_type: Optional[str] = "mean"  # mean, layernorm, or None
    multi_head: bool = False
    epsilon: float = 1e-8
    dtype: DType = jnp.float32
    precision: Optional[jax.lax.Precision] = None

    @nn.compact
    def __call__(self, query: Array, key: Array, value: Array,
                 train: bool = False) -> Array:
        """Computes inverted dot-product attention.

    Args:
      query: Queries with shape of `[batch..., q_num, qk_features]`.
      key: Keys with shape of `[batch..., kv_num, qk_features]`.
      value: Values with shape of `[batch..., kv_num, v_features]`.
      train: Indicating whether we're training or evaluating.

    Returns:
      Output of shape `[batch_size..., n_queries, v_features]`
    """
        del train  # Unused.

        attn = GeneralizedDotProductAttention(
            inverted_attn=True,
            renormalize_keys=True if self.norm_type == "mean" else False,
            epsilon=self.epsilon,
            dtype=self.dtype,
            precision=self.precision)

        # Apply attention mechanism.
        output = attn(query=query, key=key, value=value)

        if self.multi_head:
            # Multi-head aggregation. Equivalent to concat + dense layer.
            output = nn.DenseGeneral(features=output.shape[-1], axis=(-2, -1))(output)
        else:
            # Remove head dimension.
            output = jnp.squeeze(output, axis=-2)

        if self.norm_type == "layernorm":
            output = nn.LayerNorm()(output)

        return output


class GeneralizedDotProductAttention(nn.Module):
    """Multi-head dot-product attention with customizable normalization axis.

  This module supports logging of attention weights in a variable collection.
  """

    dtype: DType = jnp.float32
    precision: Optional[jax.lax.Precision] = None
    epsilon: float = 1e-8
    inverted_attn: bool = False
    renormalize_keys: bool = False
    attn_weights_only: bool = False

    @nn.compact
    def __call__(self, query: Array, key: Array, value: Array,
                 train: bool = False, **kwargs) -> Array:
        """Computes multi-head dot-product attention given query, key, and value.

    Args:
      query: Queries with shape of `[batch..., q_num, num_heads, qk_features]`.
      key: Keys with shape of `[batch..., kv_num, num_heads, qk_features]`.
      value: Values with shape of `[batch..., kv_num, num_heads, v_features]`.
      train: Indicating whether we're training or evaluating.
      **kwargs: Additional keyword arguments are required when used as attention
        function in nn.MultiHeadDotProductAttention, but they will be ignored
        here.

    Returns:
      Output of shape `[batch..., q_num, num_heads, v_features]`.
    """

        assert query.ndim == key.ndim == value.ndim, (
            "Queries, keys, and values must have the same rank.")
        assert query.shape[:-3] == key.shape[:-3] == value.shape[:-3], (
            "Query, key, and value batch dimensions must match.")
        assert query.shape[-2] == key.shape[-2] == value.shape[-2], (
            "Query, key, and value num_heads dimensions must match.")
        assert key.shape[-3] == value.shape[-3], (
            "Key and value cardinality dimensions must match.")
        assert query.shape[-1] == key.shape[-1], (
            "Query and key feature dimensions must match.")

        if kwargs.get("bias") is not None:
            raise NotImplementedError(
                "Support for masked attention is not yet implemented.")

        if "dropout_rate" in kwargs:
            if kwargs["dropout_rate"] > 0.:
                raise NotImplementedError("Support for dropout is not yet implemented.")

        # Temperature normalization.
        qk_features = query.shape[-1]
        query = query / jnp.sqrt(qk_features).astype(self.dtype)

        # attn.shape = (batch..., num_heads, q_num, kv_num)
        attn = jnp.einsum("...qhd,...khd->...hqk", query, key,
                          precision=self.precision)

        if self.inverted_attn:
            attention_axis = -2  # Query axis.
        else:
            attention_axis = -1  # Key axis.

        # Softmax normalization (by default over key axis).
        attn = jax.nn.softmax(attn, axis=attention_axis).astype(self.dtype)

        # Defines intermediate for logging.
        if not train:
            self.sow("intermediates", "attn", attn)

        if self.renormalize_keys:
            # Corresponds to value aggregation via weighted mean (as opposed to sum).
            normalizer = jnp.sum(attn, axis=-1, keepdims=True) + self.epsilon
            attn = attn / normalizer

        if self.attn_weights_only:
            return attn

        # Aggregate values using a weighted sum with weights provided by `attn`.
        return jnp.einsum("...hqk,...khd->...qhd", attn, value,
                          precision=self.precision)


class Transformer(nn.Module):
    """Transformer with multiple blocks."""

    num_heads: int
    qkv_size: int
    mlp_size: int
    num_layers: int
    pre_norm: bool = False

    @nn.compact
    def __call__(self, queries: Array, inputs: Optional[Array] = None,
                 padding_mask: Optional[Array] = None,
                 train: bool = False) -> Array:
        x = queries
        for lyr in range(self.num_layers):
            x = TransformerBlock(
                num_heads=self.num_heads, qkv_size=self.qkv_size,
                mlp_size=self.mlp_size, pre_norm=self.pre_norm,
                name=f"TransformerBlock{lyr}")(  # pytype: disable=wrong-arg-types
                x, inputs, padding_mask, train)
        return x


class TransformerBlock(nn.Module):
    """Transformer decoder block."""

    num_heads: int
    qkv_size: int
    mlp_size: int
    pre_norm: bool = False

    @nn.compact
    def __call__(self, queries: Array, inputs: Optional[Array] = None,
                 padding_mask: Optional[Array] = None,
                 train: bool = False) -> Array:
        del padding_mask  # Unused.
        assert queries.ndim == 3

        attention_fn = GeneralizedDotProductAttention()

        attn = functools.partial(
            nn.MultiHeadDotProductAttention,
            num_heads=self.num_heads,
            qkv_features=self.qkv_size,
            attention_fn=attention_fn)

        mlp = misc.MLP(hidden_size=self.mlp_size)  # type: ignore

        if self.pre_norm:
            # Self-attention on queries.
            x = nn.LayerNorm()(queries)
            x = attn()(inputs_q=x, inputs_kv=x, deterministic=not train)
            x = x + queries

            # Cross-attention on inputs.
            if inputs is not None:
                assert inputs.ndim == 3
                y = nn.LayerNorm()(x)
                y = attn()(inputs_q=y, inputs_kv=inputs, deterministic=not train)
                y = y + x
            else:
                y = x

            # MLP
            z = nn.LayerNorm()(y)
            z = mlp(z, train)
            z = z + y
        else:
            # Self-attention on queries.
            x = queries
            x = attn()(inputs_q=x, inputs_kv=x, deterministic=not train)
            x = x + queries
            x = nn.LayerNorm()(x)

            # Cross-attention on inputs.
            if inputs is not None:
                assert inputs.ndim == 3
                y = attn()(inputs_q=x, inputs_kv=inputs, deterministic=not train)
                y = y + x
                y = nn.LayerNorm()(y)
            else:
                y = x

            # MLP.
            z = mlp(y, train)
            z = z + y
            z = nn.LayerNorm()(z)
        return z


class TimeSpaceTransformerBlock(nn.Module):
    """
    Spatial attention first, followed by temporal attention.
    """

    num_heads: int
    qkv_size: int
    mlp_size: int
    pre_norm: bool = False

    @nn.compact
    def __call__(self, queries: Array, inputs: Array,
                 padding_mask: Optional[Array] = None,
                 train: bool = False) -> Array:
        del padding_mask  # Unused.
        assert queries.ndim == 3  # (batch,o,128)
        assert inputs.ndim == 4  # (batch,t,o,128)
        B, O, D = queries.shape
        attention_fn = GeneralizedDotProductAttention()

        space_attn = functools.partial(
            nn.MultiHeadDotProductAttention,
            num_heads=self.num_heads,
            qkv_features=self.qkv_size,
            attention_fn=attention_fn)
        time_attn = functools.partial(
            nn.MultiHeadDotProductAttention,
            num_heads=self.num_heads,
            qkv_features=self.qkv_size,
            attention_fn=attention_fn)

        mlp = misc.MLP(hidden_size=self.mlp_size)  # type: ignore

        if self.pre_norm:
            # space attention.
            xs = nn.LayerNorm()(queries)
            xs = space_attn()(inputs_q=xs, inputs_kv=xs, deterministic=not train)
            xs = xs + queries

            # Time attention
            xt = nn.LayerNorm()(xs)
            xt = jnp.expand_dims(xt, axis=1)  # bod->b1od
            xt = rearrange(xt, 'b t o d -> (b o) t d')
            xt_buffer = inputs
            xt_buffer = rearrange(xt_buffer, 'b t o d -> (b o) t d')
            xt = time_attn()(inputs_q=xt, inputs_kv=xt_buffer, deterministic=not train)  # (b*o,1,128)
            xt = xt.reshape(B, O, D) + xs

            # MLP.
            y = xt
            z = nn.LayerNorm()(y)
            z = mlp(z, train)
            z = z + y
        else:
            # space attention.
            xs = queries
            xs = space_attn()(inputs_q=xs, inputs_kv=xs, deterministic=not train)
            xs = xs + queries
            xs = nn.LayerNorm()(xs)

            # Time attention
            xt = jnp.expand_dims(xs, axis=1)  # bod->b1od
            xt = rearrange(xt, 'b t o d -> (b o) t d')
            xt_buffer = inputs
            xt_buffer = rearrange(xt_buffer, 'b t o d -> (b o) t d')
            xt = time_attn()(inputs_q=xt, inputs_kv=xt_buffer, deterministic=not train)  # (b*o,1,128)
            xt = xt.reshape(B, O, D) + xs
            xt = nn.LayerNorm()(xt)

            y = xt
            # MLP.
            z = mlp(y, train)
            z = z + y
            z = nn.LayerNorm()(z)
        return z


class TimeSpaceTransformerBlock2(nn.Module):
    """
     SOTA in our paper.
     Spatial attention add temporal attention.
    """

    num_heads: int
    qkv_size: int
    mlp_size: int
    pre_norm: bool = False

    @nn.compact
    def __call__(self, queries: Array, inputs: Array,
                 padding_mask: Optional[Array] = None,
                 train: bool = False) -> Array:
        del padding_mask  # Unused.
        assert queries.ndim == 3  # (batch,o,128)
        assert inputs.ndim == 4  # (batch,t,o,128)
        B, O, D = queries.shape
        attention_fn = GeneralizedDotProductAttention()

        attn = functools.partial(
            nn.MultiHeadDotProductAttention,
            num_heads=self.num_heads,
            qkv_features=self.qkv_size,
            attention_fn=attention_fn)

        mlp = misc.MLP(hidden_size=self.mlp_size)  # type: ignore

        if self.pre_norm:
            # Space attention.
            xs = nn.LayerNorm()(queries)
            xs = attn()(inputs_q=xs, inputs_kv=xs, deterministic=not train)
            xs = xs + queries

            # Time attention
            xt = nn.LayerNorm()(queries)
            xt = jnp.expand_dims(xt, axis=1)  # bod->b1od
            xt = rearrange(xt, 'b t o d -> (b o) t d')
            xt_buffer = nn.LayerNorm()(inputs)
            xt_buffer = rearrange(xt_buffer, 'b t o d -> (b o) t d')
            xt = attn()(inputs_q=xt, inputs_kv=xt_buffer, deterministic=not train)  # (b*o,1,128)
            xt = xt.reshape(B, O, D) + queries

            y = xt + xs
            # MLP.
            z = nn.LayerNorm()(y)
            z = mlp(z, train)
            z = z + y

        else:
            # Space attention.
            xs = queries
            xs = attn()(inputs_q=xs, inputs_kv=xs, deterministic=not train)
            xs = xs + queries
            xs = nn.LayerNorm()(xs)

            # Time attention
            xt = jnp.expand_dims(queries, axis=1)  # bod->b1od
            xt = rearrange(xt, 'b t o d -> (b o) t d')
            xt_buffer = inputs
            xt_buffer = rearrange(xt_buffer, 'b t o d -> (b o) t d')
            xt = attn()(inputs_q=xt, inputs_kv=xt_buffer, deterministic=not train)  # (b*o,1,128)
            xt = xt.reshape(B, O, D) + queries
            xt = nn.LayerNorm()(xt)

            y = xt + xs
            # MLP.
            z = mlp(y, train)
            z = z + y
            z = nn.LayerNorm()(z)
        return z


class TimeSpaceTransformerBlock3(nn.Module):
    """
    Temporal attention first, followed by spatial attention.
    """

    num_heads: int
    qkv_size: int
    mlp_size: int
    pre_norm: bool = False

    @nn.compact
    def __call__(self, queries: Array, inputs: Array,
                 padding_mask: Optional[Array] = None,
                 train: bool = False) -> Array:
        del padding_mask  # Unused.
        assert queries.ndim == 3  # (batch,o,128)
        assert inputs.ndim == 4  # (batch,t,o,128)
        B, O, D = queries.shape
        attention_fn = GeneralizedDotProductAttention()

        attn = functools.partial(
            nn.MultiHeadDotProductAttention,
            num_heads=self.num_heads,
            qkv_features=self.qkv_size,
            attention_fn=attention_fn)

        mlp = misc.MLP(hidden_size=self.mlp_size)  # type: ignore

        if self.pre_norm:
            # Time attention
            xt = jnp.expand_dims(queries, axis=1)  # bod->b1od
            xt = nn.LayerNorm()(xt)
            xt_buffer = nn.LayerNorm()(inputs)
            xt = rearrange(xt, 'b t o d -> (b o) t d')
            xt_buffer = rearrange(xt_buffer, 'b t o d -> (b o) t d')
            xt = attn()(inputs_q=xt, inputs_kv=xt_buffer, deterministic=not train)  # (b*o,1,128)
            xt = xt.reshape(B, O, D) + queries

            # Space attention.
            xs = nn.LayerNorm()(xt)  # (b,o,128)
            xs = attn()(inputs_q=xs, inputs_kv=xs, deterministic=not train)
            xs = xs + xt

            y = xs
            # MLP
            z = nn.LayerNorm()(y)
            z = mlp(z, train)
            z = z + y
        else:
            # Time attention
            xt = jnp.expand_dims(queries, axis=1)  # bod->b1od
            xt = rearrange(xt, 'b t o d -> (b o) t d')
            xt_buffer = inputs
            xt_buffer = rearrange(xt_buffer, 'b t o d -> (b o) t d')
            xt = attn()(inputs_q=xt, inputs_kv=xt_buffer, deterministic=not train)  # (b*o,1,128)
            xt = xt.reshape(B, O, D) + queries
            xt = nn.LayerNorm()(xt)

            # Space attention.
            xs = xt
            xs = attn()(inputs_q=xs, inputs_kv=xs, deterministic=not train)
            xs = xs + xt
            xs = nn.LayerNorm()(xs)

            y = xs
            # MLP.
            z = mlp(y, train)
            z = z + y
            z = nn.LayerNorm()(z)
        return z


class TimeSpaceTransformerBlock4(nn.Module):
    """
    Spatial attention followed by temporal attention,
    with a single residual connection applied after both.
    """

    num_heads: int
    qkv_size: int
    mlp_size: int
    pre_norm: bool = False

    @nn.compact
    def __call__(self, queries: Array, inputs: Array,
                 padding_mask: Optional[Array] = None,
                 train: bool = False) -> Array:
        del padding_mask  # Unused.
        assert queries.ndim == 3  # (batch,o,128)
        assert inputs.ndim == 4  # (batch,t,o,128)
        B, O, D = queries.shape
        attention_fn = GeneralizedDotProductAttention()

        attn = functools.partial(
            nn.MultiHeadDotProductAttention,
            num_heads=self.num_heads,
            qkv_features=self.qkv_size,
            attention_fn=attention_fn)

        mlp = misc.MLP(hidden_size=self.mlp_size)  # type: ignore

        if self.pre_norm:
            xs = nn.LayerNorm()(queries)
            xs = attn()(inputs_q=xs, inputs_kv=xs, deterministic=not train)
            xs = xs

            # Time attention
            xt = jnp.expand_dims(queries, axis=1)  # bod->b1od
            xt = nn.LayerNorm()(xt)
            xt_buffer = nn.LayerNorm()(inputs)
            xt = rearrange(xt, 'b t o d -> (b o) t d')
            xt_buffer = rearrange(xt_buffer, 'b t o d -> (b o) t d')
            xt = attn()(inputs_q=xt, inputs_kv=xt_buffer, deterministic=not train)  # (b*o,1,128)
            xt = xt.reshape(B, O, D)

            y = xt + xs + queries
            # MLP.
            z = nn.LayerNorm()(y)
            z = mlp(z, train)
            z = z + y
        else:
            # Space attention.
            xs = queries
            xs = attn()(inputs_q=xs, inputs_kv=xs, deterministic=not train)

            # Time attention
            xt = jnp.expand_dims(queries, axis=1)  # bod->b1od
            xt = rearrange(xt, 'b t o d -> (b o) t d')
            xt_buffer = inputs
            xt_buffer = rearrange(xt_buffer, 'b t o d -> (b o) t d')
            xt = attn()(inputs_q=xt, inputs_kv=xt_buffer, deterministic=not train)  # (b*o,1,128)
            xt = xt.reshape(B, O, D)

            y = nn.LayerNorm()(xt + xs + queries)
            # MLP.
            z = mlp(y, train)
            z = z + y
            z = nn.LayerNorm()(z)
        return z


class TimeSpaceTransformerBlock5(nn.Module):
    """
    All attention
    The current slot attends to all slots stored in the buffer via attention.
    """

    num_heads: int
    qkv_size: int
    mlp_size: int
    pre_norm: bool = False

    @nn.compact
    def __call__(self, queries: Array, inputs: Array,
                 padding_mask: Optional[Array] = None,
                 train: bool = False) -> Array:
        del padding_mask  # Unused.
        assert queries.ndim == 3  # (batch,o,128)
        assert inputs.ndim == 4  # (batch,t,o,128)
        B, O, D = queries.shape
        attention_fn = GeneralizedDotProductAttention()

        attn = functools.partial(
            nn.MultiHeadDotProductAttention,
            num_heads=self.num_heads,
            qkv_features=self.qkv_size,
            attention_fn=attention_fn)

        mlp = misc.MLP(hidden_size=self.mlp_size)  # type: ignore

        if self.pre_norm:
            # Space attention.
            xs = nn.LayerNorm()(queries)  # (b,o,128)
            xs = attn()(inputs_q=xs, inputs_kv=xs, deterministic=not train)
            xs = xs + queries

            # Time attention
            xt = jnp.expand_dims(queries, axis=1)  # bod->b1od
            xt = nn.LayerNorm()(xt)
            xt_buffer = nn.LayerNorm()(inputs)
            xt = rearrange(xt, 'b t o d -> (b o) t d')
            xt_buffer = rearrange(xt_buffer, 'b t o d -> (b o) t d')
            xt = attn()(inputs_q=xt, inputs_kv=xt_buffer, deterministic=not train)  # (b*o,1,128)
            xt = xt.reshape(B, O, D) + queries

            y = xs + xt

            # MLP
            z = nn.LayerNorm()(y)
            z = mlp(z, train)
            z = z + y
        else:
            # Space attention.
            xs = queries
            xs = attn()(inputs_q=xs, inputs_kv=xs, deterministic=not train)
            xs = xs + queries
            xs = nn.LayerNorm()(xs)

            # Time attention
            xt = jnp.expand_dims(queries, axis=1)  # bod->b1od
            xt = rearrange(xt, 'b t o d -> b (t o)  d')
            xt_buffer = inputs
            xt_buffer = rearrange(xt_buffer, 'b t o d -> b (t o) d')
            xt = attn()(inputs_q=xt, inputs_kv=xt_buffer, deterministic=not train)  # (b,o,128)
            xt = xt.reshape(B, O, D) + queries
            xt = nn.LayerNorm()(xt)

            y = xt + xs
            # MLP.
            z = mlp(y, train)
            z = z + y
            z = nn.LayerNorm()(z)
        return z
