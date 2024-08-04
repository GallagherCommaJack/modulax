from .abstract import Module
import jax.numpy as jnp
import math
import jax
import jax.numpy as jnp
import einx

from typing import Tuple, NamedTuple, TypedDict


class LinearState(TypedDict):
    u: jax.Array
    v: jax.Array


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
        state: LinearState,
        params: jax.Array,
    ) -> Tuple[jax.Array, LinearState]:
        return self.scale * einx.dot("... i, o i -> ... o", x, params)

    def init(self, rng: jax.Array):
        k_w, k_u, k_v = jax.random.split(rng, 3)
        weight = jax.nn.initializers.orthogonal(column_axis=0)(
            k_w,
            (self.out_features, self.in_features),
            dtype=jnp.float32,
        )
        u = jax.random.normal(k_u, (self.out_features,), dtype=jnp.float32)
        v = jax.random.normal(k_v, (self.in_features,), dtype=jnp.float32)

        return {"u": u, "v": v}, weight

    def normalize(
        self,
        rng: jax.Array,
        state: LinearState,
        params: jax.Array,
        target_norm: jax.Array,
    ) -> Tuple[LinearState, jax.Array]:
        weight = params
        u = state["u"]
        v = state["v"]
        v = weight @ u
        v /= jnp.linalg.norm(v)
        u = weight.T @ v
        u *= target_norm / jnp.linalg.norm(u)
        return {"u": u, "v": v}, weight

    def regularize(self, rng: jax.Array, state: LinearState, params: jax.Array, strength: jax.Array):
        params = params * (1 - strength)
        return state, params

    def print_submodules(self):
        print(
            f"Linear module of shape {(self.out_features, self.in_features)} and mass {self.mass}."
        )
