import math
from typing import Generic, NamedTuple, Optional, Tuple

import einx
import jax
import jax.numpy as jnp

from .abstract import Module, X, Y


class AttentionInputs(NamedTuple):
    q: jax.Array
    k: jax.Array
    v: jax.Array
    kv_mask: Optional[jax.Array] = None
    pairwise_mask: Optional[jax.Array] = None


class Bond(Module[None, None, X, Y], Generic[X, Y]):
    """A module with no weights."""

    def __init__(self):
        super().__init__()
        self.mass: float = 0
        self.sensitivity: float = 1
        self.length: int = 0
        self.children: list["Module"] = []

    def init_opt_state(self, key: jax.Array, params: None) -> None:
        return None

    def init_params(self, key: jax.Array) -> None:
        return None

    def normalize(self, update: None, state: None) -> Tuple[None, None]:
        return None, None

    def regularize(self, params: None, state: None) -> Tuple[None, None]:
        return None, None

    def __call__(self, rng: jax.Array, params: None, x: jax.Array) -> jax.Array:
        return x


class Identity(Bond[X, X], Generic[X]):
    """Identity module."""

    def __call__(self, rng: jax.Array, params: None, x: X) -> X:
        return x


class Flatten(Bond[jax.Array, jax.Array]):
    """Flatten all non-batch dimensions."""

    def __call__(self, rng: jax.Array, params: None, x: jax.Array) -> jax.Array:
        return einx.rearrange("b ... -> b (...)", x)


class AddHeads(Bond[jax.Array, jax.Array]):
    """Reshapes an input to have heads."""

    def __init__(self, num_heads: int):
        super().__init__()
        self.num_heads: int = num_heads

    def __call__(self, rng: jax.Array, params: None, x: jax.Array) -> jax.Array:
        return einx.rearrange("b t (h d) -> b h t d", x, h=self.num_heads)


class RemoveHeads(Bond[jax.Array, jax.Array]):
    """Inverse of AddHeads."""

    def __call__(self, rng: jax.Array, params: None, x: jax.Array) -> jax.Array:
        return einx.rearrange("b h t d -> b t (h d)", x)


class Abs(Bond[jax.Array, jax.Array]):
    """Absolute value nonlinearity."""

    def __call__(self, rng: jax.Array, params: None, x: jax.Array) -> jax.Array:
        return jnp.abs(x)


class ReLU(Bond[jax.Array, jax.Array]):
    """ReLU nonlinearity."""

    def __init__(self):
        super().__init__()
        self.sensitivity: float = 1 / math.sqrt(2)

    def __call__(self, rng: jax.Array, params: None, x: jax.Array) -> jax.Array:
        return jax.nn.relu(x)


def ScaledReLU() -> Bond[jax.Array, jax.Array]:
    """ReLU scaled to have sensitivity one."""
    return math.sqrt(2) * ReLU()


class GELU(Bond[jax.Array, jax.Array]):
    """GELU nonlinearity."""

    def __init__(self):
        super().__init__()
        self.sensitivity: float = 1 / math.sqrt(2)

    def __call__(self, rng: jax.Array, params: None, x: jax.Array) -> jax.Array:
        return jax.nn.gelu(x)


def ScaledGELU() -> Bond[jax.Array, jax.Array]:
    """GELU scaled to have sensitivity 1."""
    return math.sqrt(2) * GELU()


class MeanSubtract(Bond[jax.Array, jax.Array]):
    """Mean subtraction."""

    def __init__(self, axis: int = -1):
        super().__init__()
        self.axis: int = axis

    def __call__(self, rng: jax.Array, params: None, x: jax.Array) -> jax.Array:
        return x - jnp.mean(x, axis=self.axis, keepdims=True)


class RMSDivide(Bond[jax.Array, jax.Array]):
    """Normalize to have unit RMS norm."""

    def __init__(self, axis: int = -1):
        super().__init__()
        self.axis: int = axis

    def __call__(self, rng: jax.Array, params: None, x: jax.Array) -> jax.Array:
        return x / jnp.sqrt(jnp.mean(jnp.square(x), axis=self.axis, keepdims=True))


def LayerNorm(axis: int = -1) -> Bond[jax.Array, jax.Array]:
    """Mean subtraction followed by RMS normalization."""
    return RMSDivide(axis) @ MeanSubtract(axis)


class Mean(Bond[jax.Array, jax.Array]):
    """Take the mean over a specified dimension."""

    def __init__(self, axis: int):
        super().__init__()
        self.axis: int = axis

    def __call__(self, rng: jax.Array, params: None, x: jax.Array) -> jax.Array:
        return jnp.mean(x, axis=self.axis)


class AvgPool(Bond[jax.Array, jax.Array]):
    """Average pooling that adapts to different input sizes."""

    def __init__(self, output_size: Tuple[int, int] = (1, 1)):
        super().__init__()
        self.output_size: Tuple[int, int] = output_size

    def __call__(self, rng: jax.Array, params: None, x: jax.Array) -> jax.Array:
        return jax.image.resize(x, x.shape[:2] + self.output_size, method="average")


class FunctionalAttention(Bond[AttentionInputs, jax.Array]):
    """The part of attention that doesn't involve weights."""

    def __init__(self, causal: bool = False):
        super().__init__()
        self.causal: bool = causal

    def __call__(self, rng: jax.Array, params: None, x: AttentionInputs) -> jax.Array:
        q, k, v = x.q, x.k, x.v
        attn_weights = einx.dot("b h q d, b h k d -> b h q k", q, k) / jnp.sqrt(
            q.shape[-1]
        )

        if self.causal:
            mask = jnp.tril(jnp.ones([q.shape[-2], k.shape[-2]], dtype=jnp.bool_))
            attn_weights = einx.where(
                "q k, b h q k, -> b h q k", mask, attn_weights, float("-inf")
            )
        if x.kv_mask is not None:
            attn_weights = einx.where(
                "b k, b h q k, -> b h q k",
                x.kv_mask,
                attn_weights,
                float("-inf"),
            )
        if x.pairwise_mask is not None:
            attn_weights = einx.where(
                "b q k, b h q k, -> b h q k",
                x.pairwise_mask,
                attn_weights,
                float("-inf"),
            )

        attn_weights = einx.softmax("b h q [k]", attn_weights)
        return einx.dot("b h q k,b h k d->b h q d", attn_weights, v)
