import math
from typing import Tuple

import jax
import jax.numpy as jnp

from .abstract import Module


class Linear(Module[jax.Array, jax.Array, jax.Array, jax.Array]):
    def __init__(self, out_features, in_features, mass=1):
        super().__init__()
        self.mass = mass
        self.sensitivity = 1
        self.length = 1
        self.out_features = out_features
        self.in_features = in_features
        self.scale = math.sqrt(out_features / in_features)
        self.children = []

    def __call__(
        self,
        rng: jax.Array,
        x: jax.Array,
        state: jax.Array,
        params: jax.Array,
    ) -> Tuple[jax.Array, jax.Array]:
        return self.scale * jnp.einsum(
            "...i,oi->...o",
            x,
            params,
            preferred_element_type=x.dtype,
        )

    def init(self, rng: jax.Array):
        k_w, k_u = jax.random.split(rng, 2)
        weight = jax.nn.initializers.orthogonal(column_axis=0)(
            k_w,
            (self.out_features, self.in_features),
            dtype=jnp.float32,
        )
        u = jax.random.normal(k_u, (self.in_features,), dtype=jnp.float32)

        return u, weight

    def normalize(
        self,
        rng: jax.Array,
        state: jax.Array,
        update: jax.Array,
        target_norm: jax.Array,
    ) -> Tuple[jax.Array, jax.Array]:
        weight = update
        u = state
        v = jnp.einsum("i,oi->o", u, weight)
        v /= jnp.linalg.norm(v)
        u = jnp.einsum("o,oi->i", v, weight)
        weight *= target_norm / jnp.linalg.norm(u)
        return u, weight

    def regularize(
        self,
        rng: jax.Array,
        state: jax.Array,
        params: jax.Array,
        strength: jax.Array,
    ):
        params = params * (1 - strength)
        return state, params

    def print_submodules(self):
        print(
            f"Linear module of shape {(self.out_features, self.in_features)} and mass {self.mass}."
        )


class Conv2D(Module[jax.Array, jax.Array, jax.Array, jax.Array]):
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

    def init(self, rng: jax.Array):
        k_w, k_u = jax.random.split(rng, 2)
        weight = jax.nn.initializers.delta_orthogonal()(
            k_w,
            (self.kernel_size, self.kernel_size, self.out_channels, self.in_channels),
            dtype=jnp.float32,
        )
        u = jax.random.normal(
            k_u,
            (
                self.kernel_size,
                self.kernel_size,
                self.in_channels,
            ),
            dtype=jnp.float32,
        )

    def __call__(
        self, rng: jax.Array, x: jax.Array, state: jax.Array, params: jax.Array
    ):
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
        rng: jax.Array,
        state: jax.Array,
        update: jax.Array,
        target_norm: jax.Array,
    ) -> Tuple[jax.Array, jax.Array]:
        weight = update
        u = state
        v = jnp.einsum("hwi,hwoi->hwo", u, weight)
        v /= jnp.linalg.norm(v, axis=-1, keepdims=True)
        u = jnp.einsum("hwo,hwoi->hwi", v, weight)
        weight *= target_norm / jnp.linalg.norm(u, axis=-1, keepdims=True)
        return u, weight

    def regularize(
        self,
        rng: jax.Array,
        state: jax.Array,
        params: jax.Array,
        strength: jax.Array,
    ):
        params = params * (1 - strength)
        return state, params

    def print_submodules(self):
        print(
            f"Conv2D module of shape {(self.kernel_size, self.kernel_size, self.out_channels, self.in_channels)} and mass {self.mass}."
        )


class ShampooLinearState(TypedDict):
    i: jax.Array
    l: jax.Array
    l_inv: jax.Array
    r: jax.Array
    r_inv: jax.Array


class ShampooLinear(Module[ShampooLinearState, jax.Array, jax.Array, jax.Array]):
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

    def __call__(
        self,
        rng: jax.Array,
        x: jax.Array,
        state: jax.Array,
        params: jax.Array,
    ) -> Tuple[jax.Array, jax.Array]:
        return self.scale * jnp.einsum(
            "...i,oi->...o",
            x,
            params,
            preferred_element_type=x.dtype,
        )

    def init(self, rng: jax.Array):
        weight = jax.nn.initializers.orthogonal(column_axis=0)(
            rng,
            (self.out_features, self.in_features),
            dtype=jnp.float32,
        )
        l = jnp.eye(self.out_features)
        l_inv = jnp.eye(self.out_features)
        r = jnp.eye(self.in_features)
        r_inv = jnp.eye(self.in_features)
        i = jnp.zeros(
            (),
            dtype=jnp.uint32,
        )

        return {
            "i": i,
            "l": l,
            "l_inv": l_inv,
            "r": r,
            "r_inv": r_inv,
        }, weight

    def normalize(
        self,
        rng: jax.Array,
        state: jax.Array,
        update: jax.Array,
        target_norm: jax.Array,
    ) -> Tuple[ShampooLinearState, jax.Array]:
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
        update = target_norm * r_inv @ update @ l_inv
        return {
            "i": i + 1,
            "l": l,
            "l_inv": l_inv,
            "r": r,
            "r_inv": r_inv,
        }, update

    def regularize(
        self,
        rng: jax.Array,
        state: jax.Array,
        params: jax.Array,
        strength: jax.Array,
    ):
        params = params * (1 - strength)
        return state, params

    def print_submodules(self):
        print(
            f"Linear module of shape {(self.out_features, self.in_features)} and mass {self.mass}."
        )
