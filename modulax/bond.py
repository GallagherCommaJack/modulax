import math
from typing import Tuple, Optional
import einx
import jax
import jax.numpy as jnp

from .abstract import Module


class Bond(Module[None, None, jax.Array, jax.Array]):
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


class Identity(Bond):
    """Identity module."""

    def __call__(self, rng: jax.Array, params: None, x: jax.Array) -> jax.Array:
        return x


class Flatten(Bond):
    """Flatten all non-batch dimensions."""

    def __call__(self, rng: jax.Array, params: None, x: jax.Array) -> jax.Array:
        return jnp.reshape(x, (x.shape[0], -1))


class AddHeads(Bond):
    """Reshapes an input to have heads."""

    def __init__(self, num_heads: int):
        super().__init__()
        self.num_heads: int = num_heads

    def __call__(self, rng: jax.Array, params: None, x: jax.Array) -> jax.Array:
        B, T, C = x.shape
        return jnp.reshape(x, (B, T, self.num_heads, C // self.num_heads)).transpose(0, 2, 1, 3)


class RemoveHeads(Bond):
    """Inverse of AddHeads."""

    def __call__(self, rng: jax.Array, params: None, x: jax.Array) -> jax.Array:
        B, nh, T, hs = x.shape
        return jnp.reshape(x.transpose(0, 2, 1, 3), (B, T, nh * hs))


class Enumerate(Bond):
    """Replace each column with its column index. Used to make position embeddings."""

    def __call__(self, rng: jax.Array, params: None, x: jax.Array) -> jax.Array:
        return jnp.arange(0, x.shape[1], dtype=jnp.int32)


class Abs(Bond):
    """Absolute value nonlinearity."""

    def __call__(self, rng: jax.Array, params: None, x: jax.Array) -> jax.Array:
        return jnp.abs(x)


class ReLU(Bond):
    """ReLU nonlinearity."""

    def __init__(self):
        super().__init__()
        self.sensitivity: float = 1 / math.sqrt(2)

    def __call__(self, rng: jax.Array, params: None, x: jax.Array) -> jax.Array:
        return jax.nn.relu(x)


def ScaledReLU() -> Bond:
    """ReLU scaled to have sensitivity one."""
    return math.sqrt(2) * ReLU()


class GELU(Bond):
    """GELU nonlinearity."""

    def __init__(self):
        super().__init__()
        self.sensitivity: float = 1 / math.sqrt(2)

    def __call__(self, rng: jax.Array, params: None, x: jax.Array) -> jax.Array:
        return jax.nn.gelu(x)


def ScaledGELU() -> Bond:
    """GELU scaled to have sensitivity 1."""
    return math.sqrt(2) * GELU()


class MeanSubtract(Bond):
    """Mean subtraction."""

    def __init__(self, axis: int = -1):
        super().__init__()
        self.axis: int = axis

    def __call__(self, rng: jax.Array, params: None, x: jax.Array) -> jax.Array:
        return x - jnp.mean(x, axis=self.axis, keepdims=True)


class RMSDivide(Bond):
    """Normalize to have unit RMS norm."""

    def __init__(self, axis: int = -1):
        super().__init__()
        self.axis: int = axis

    def __call__(self, rng: jax.Array, params: None, x: jax.Array) -> jax.Array:
        return x / jnp.sqrt(jnp.mean(jnp.square(x), axis=self.axis, keepdims=True))


def LayerNorm(axis: int = -1) -> Bond:
    """Mean subtraction followed by RMS normalization."""
    return RMSDivide(axis) @ MeanSubtract(axis)


class Mean(Bond):
    """Take the mean over a specified dimension."""

    def __init__(self, axis: int):
        super().__init__()
        self.axis: int = axis

    def __call__(self, rng: jax.Array, params: None, x: jax.Array) -> jax.Array:
        return jnp.mean(x, axis=self.axis)


class AvgPool(Bond):
    """Average pooling that adapts to different input sizes."""

    def __init__(self, output_size: Tuple[int, int] = (1, 1)):
        super().__init__()
        self.output_size: Tuple[int, int] = output_size

    def __call__(self, rng: jax.Array, params: None, x: jax.Array) -> jax.Array:
        return jax.image.resize(x, x.shape[:2] + self.output_size, method="average")


class FunctionalAttention(Bond):
    """The part of attention that doesn't involve weights."""

    def __init__(self, causal: bool = False):
        super().__init__()
        self.causal: bool = causal

    def __call__(self, rng: jax.Array, params: None, x: Tuple[jax.Array, jax.Array, jax.Array, Optional[jax.Array], Optional[jax.Array]]) -> jax.Array:
        q, k, v, kv_mask, pairwise_mask = x
        attn_weights = einx.dot("b h q d, b h k d -> b h q k", q, k) / jnp.sqrt(
            q.shape[-1]
        )

        if self.causal:
            mask = jnp.tril(jnp.ones([q.shape[-2], k.shape[-2]], dtype=jnp.bool_))
            attn_weights = einx.where(
                "q k, b h q k, -> b h q k", mask, attn_weights, float("-inf")
            )
        if kv_mask is not None:
            attn_weights = einx.where(
                "b k, b h q k, -> b h q k",
                kv_mask,
                attn_weights,
                float("-inf"),
            )
        if pairwise_mask is not None:
            attn_weights = einx.where(
                "b q k, b h q k, -> b h q k",
                pairwise_mask,
                attn_weights,
                float("-inf"),
            )

        attn_weights = einx.softmax("b h q [k]", attn_weights)
        return einx.dot("b h q k,b h k d->b h q d", attn_weights, v)