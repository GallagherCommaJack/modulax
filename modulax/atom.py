import math
from typing import Tuple, Union

import jax
import jax.numpy as jnp
from jax.sharding import NamedSharding
from jax.sharding import PartitionSpec as P

from .abstract import Module, ShardingConfig, ShardingMode


class Conv2D(Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: Union[str, Tuple[int, int]] = "same",
        mass: float = 1,
    ):
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.children = []
        self.mass = mass
        self.sensitivity = 1
        self.length = 1
        self.scale = math.sqrt(out_channels / in_channels) / (kernel_size**2)

    def init_opt_state(self, key: jax.Array, params: jax.Array) -> jax.Array:
        k_u = jax.random.split(key)[1]
        return jax.random.normal(
            k_u,
            (
                self.kernel_size,
                self.kernel_size,
                self.in_channels,
            ),
            dtype=jnp.float32,
        )

    def init_params(self, key: jax.Array) -> jax.Array:
        k_w = jax.random.split(key)[0]
        return jax.nn.initializers.delta_orthogonal()(
            k_w,
            (self.kernel_size, self.kernel_size, self.out_channels, self.in_channels),
            dtype=jnp.float32,
        )

    def __call__(self, rng: jax.Array, params: jax.Array, x: jax.Array):
        return jax.lax.conv_general_dilated(
            x,
            params,
            window_strides=(self.stride, self.stride),
            padding=self.padding,
            preferred_element_type=x.dtype,
            dimension_numbers=jax.lax.ConvDimensionNumbers(
                lhs_spec=(0, 3, 1, 2),
                rhs_spec=(2, 3, 0, 1),
                out_spec=(0, 3, 1, 2),
            ),
        )

    def normalize(
        self,
        update: jax.Array,
        state: jax.Array,
    ) -> Tuple[jax.Array, jax.Array]:
        weight = update
        u = state
        v = jnp.einsum("hwi,hwoi->hwo", u, weight)
        v /= jnp.linalg.norm(v, axis=-1, keepdims=True)
        u = jnp.einsum("hwo,hwoi->hwi", v, weight)
        weight /= jnp.linalg.norm(u, axis=-1, keepdims=True)
        return weight, u


LinearState = jax.Array
LinearParams = jax.Array


class Linear(Module):
    def __init__(
        self,
        out_features: int,
        in_features: int,
        mass: float = 1,
        num_specnorm_vecs: int = 8,
        eps: float = 1e-6,
    ):
        super().__init__()
        self.mass = mass
        self.sensitivity = 1
        self.length = 1
        self.out_features = out_features
        self.in_features = in_features
        self.scale = math.sqrt(out_features / in_features)
        self.num_specnorm_vecs = num_specnorm_vecs
        self.children = []
        self.eps = eps

    def init_opt_state(self, key: jax.Array, params: LinearParams) -> LinearState:
        return jax.random.normal(
            key,
            (
                self.num_specnorm_vecs,
                self.in_features,
            ),
            dtype=jnp.float32,
        )

    def _shard_opt_state(
        self, opt_state: LinearState, config: ShardingConfig, mode: ShardingMode
    ) -> LinearState:
        spec = [None, None]
        if mode == "mp" or mode == "both":
            spec[1] = config.mp_axis
        return NamedSharding(config.mesh, P(*spec))

    def init_params(self, key: jax.Array) -> LinearParams:
        w = jax.nn.initializers.orthogonal()(
            key,
            (self.out_features, self.in_features),
            dtype=jnp.float32,
        )
        return w

    def _shard_params(
        self, params: LinearParams, config: ShardingConfig, mode: ShardingMode
    ) -> LinearParams:
        spec = [None, None]
        if mode == "fsdp" or mode == "both":
            spec[0] = config.fsdp_axis
        if mode == "mp" or mode == "both":
            spec[1] = config.mp_axis
        return NamedSharding(config.mesh, P(*spec))

    def __call__(
        self,
        rng: jax.Array,
        params: jax.Array,
        x: jax.Array,
    ) -> jax.Array:
        return self.scale * jnp.einsum(
            "...i,oi->...o",
            x,
            params,
            preferred_element_type=x.dtype,
        )

    def normalize(
        self,
        update: jax.Array,
        state: jax.Array,
    ) -> Tuple[jax.Array, jax.Array]:
        weight = update
        u = state
        v = jnp.einsum("ki,oi->ko", u, weight)
        v /= jnp.linalg.norm(v)
        u = jnp.einsum("ko,oi->ki", v, weight)
        n = jnp.max(jnp.linalg.norm(u, axis=-1))
        weight /= n + self.eps
        return weight, u
