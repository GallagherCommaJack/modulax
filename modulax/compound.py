from typing import Optional, Tuple, TypedDict

import jax
import jax.numpy as jnp
from jax import Array
from jax.typing import ArrayLike

from .abstract import Module
from .atom import Linear, LinearState, LinearParams


class TransformerStream(TypedDict):
    x: Array
    cond: Array
    kv_mask: Optional[Array]
    pairwise_mask: Optional[Array]


class AdaRMSNorm(Module[LinearState, LinearParams, Tuple[Array, Array], Array]):
    def __init__(
        self,
        d_cond: int,
        d_model: int,
        mass: float = 1,
        eps: float = 1e-6,
        base: float = 1.0,
        num_specnorm_vecs: int = 8,
    ):
        m = Linear(
            input_features=d_cond,
            output_features=d_model,
            mass=mass,
            num_specnorm_vecs=num_specnorm_vecs,
        )
        self.mass = m.mass
        self.children = [m]
        self.sensitivity = m.sensitivity
        self.eps = eps
        self.base = base

    def init_opt_state(self, key: Array, params: Array) -> LinearState:
        return self.children[0].init_opt_state(key, params)

    def init_params(self, key: Array) -> Array:
        return jax.tree.map(
            jnp.zeros_like,
            self.children[0].init_params(key),
        )

    def scale_updates(
        self, opt_state: LinearState, update: Array, target_norm: ArrayLike
    ) -> Tuple[LinearState, Array]:
        return self.children[0].scale_updates(opt_state, update, target_norm)

    def regularize(
        self, params: Array, opt_state: LinearState
    ) -> Tuple[Array, LinearState]:
        return self.children[0].regularize(params, opt_state)

    def normalize(
        self, update: Array, opt_state: LinearState
    ) -> Tuple[Array, LinearState]:
        return self.children[0].normalize(update, opt_state)

    def __call__(self, params: Array, rng: Array, x: Tuple[Array, Array]) -> Array:
        x, cond = x
        x /= jnp.sqrt(jnp.mean(jnp.square(x), axis=-1, keepdims=True) + self.eps)
        scales = self.base + self.children[0](params, rng, cond)
        return x * scales
