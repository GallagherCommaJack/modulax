from typing import Optional, Tuple, TypedDict

import jax.numpy as jnp
from jax import Array

from .abstract import WrapperModule
from .atom import Linear
from .bond import AddHeads, FunctionalAttention, RemoveHeads


class TransformerStream(TypedDict):
    x: Array
    cond: Array
    kv_mask: Optional[Array]
    pairwise_mask: Optional[Array]


class AdaRMSNorm(WrapperModule):
    def __init__(
        self,
        d_cond: int,
        d_model: int,
        mass: float = 1,
        eps: float = 1e-6,
        base: float = 1.0,
        num_specnorm_vecs: int = 8,
    ):
        super().__init__(
            Linear(
                input_features=d_cond,
                output_features=d_model,
                mass=mass,
                num_specnorm_vecs=num_specnorm_vecs,
            )
        )
        self.eps = eps
        self.base = base

    def __call__(self, params: Array, rng: Array, x: Tuple[Array, Array]) -> Array:
        x, cond = x
        x /= jnp.sqrt(jnp.mean(jnp.square(x), axis=-1, keepdims=True) + self.eps)
        scales = self.base + self.children[0](params, rng, cond)
        return x * scales


def Attention(num_heads: int, d_model: int, d_qk: int, d_v: int):
    Q = AddHeads(num_heads) @ Linear(num_heads * d_qk, d_model)
    K = AddHeads(num_heads) @ Linear(num_heads * d_qk, d_model)
    V = AddHeads(num_heads) @ Linear(num_heads * d_v, d_model)
    O = Linear(d_model, num_heads * d_v) @ RemoveHeads()
    return O @ FunctionalAttention() @ (Q, K, V)
