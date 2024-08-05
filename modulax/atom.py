import math
from typing import Tuple, TypedDict

import jax
import jax.numpy as jnp

from .abstract import Module, SimpleModule


class Conv2D(Module[jax.Array, jax.Array, jax.Array]):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: str = "same" | Tuple[int, int],
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


class Linear(Module[jax.Array, jax.Array, jax.Array]):
    def __init__(self, out_features, in_features, mass=1):
        super().__init__()
        self.mass = mass
        self.sensitivity = 1
        self.length = 1
        self.out_features = out_features
        self.in_features = in_features
        self.scale = math.sqrt(out_features / in_features)
        self.children = []

    def init_opt_state(self, key: jax.Array, params: jax.Array) -> jax.Array:
        k_u = jax.random.split(key)[1]
        return jax.random.normal(k_u, (self.in_features,), dtype=jnp.float32)

    def init_params(self, key: jax.Array) -> jax.Array:
        k_w = jax.random.split(key)[0]
        return jax.nn.initializers.orthogonal(column_axis=0)(
            k_w,
            (self.out_features, self.in_features),
            dtype=jnp.float32,
        )

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
        v = jnp.einsum("i,oi->o", u, weight)
        v /= jnp.linalg.norm(v)
        u = jnp.einsum("o,oi->i", v, weight)
        weight /= jnp.linalg.norm(u)
        return weight, u


class ShampooLinearState(TypedDict):
    i: jax.Array
    l: jax.Array
    l_inv: jax.Array
    r: jax.Array
    r_inv: jax.Array


class ShampooLinear(Module[ShampooLinearState, jax.Array]):
    def __init__(
        self,
        out_features,
        in_features,
        mass=1,
        update_preconditioner: int = 10,
    ):
        super().__init__()
        self.mass = mass
        self.sensitivity = 1
        self.length = 1
        self.out_features = out_features
        self.in_features = in_features
        self.scale = math.sqrt(out_features / in_features)
        self.update_preconditioner = update_preconditioner
        self.children = []

    def init_opt_state(self, key: jax.Array, params: jax.Array) -> ShampooLinearState:
        return {
            "i": jnp.zeros((), dtype=jnp.uint32),
            "l": jnp.eye(self.out_features),
            "l_inv": jnp.eye(self.out_features),
            "r": jnp.eye(self.in_features),
            "r_inv": jnp.eye(self.in_features),
        }

    def init_params(self, key: jax.Array) -> jax.Array:
        return jax.nn.initializers.orthogonal(column_axis=0)(
            key,
            (self.out_features, self.in_features),
            dtype=jnp.float32,
        )

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
        state: ShampooLinearState,
    ) -> Tuple[jax.Array, ShampooLinearState]:
        r = state["l"] + update @ update.T  # oo
        l = state["r"] + update.T @ update  # ii
        i = state["i"]
        l_inv = jax.lax.cond(
            i % self.update_preconditioner == 0,
            jnp.linalg.matrix_power(l, -1 / 4),
            state["r_inv"],
        )
        r_inv = jax.lax.cond(
            i % self.update_preconditioner == 0,
            jnp.linalg.matrix_power(r, -1 / 4),
            state["l_inv"],
        )
        update = r_inv @ update @ l_inv
        return update, {
            "i": i + 1,
            "l": l,
            "l_inv": l_inv,
            "r": r,
            "r_inv": r_inv,
        }