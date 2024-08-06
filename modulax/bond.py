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


class FourierFeatures(Bond[jax.Array, Tuple[jax.Array, jax.Array]]):
    def __init__(self, d_out: int, min_wavelength: float, max_wavelength: float):
        assert min_wavelength > 0, f"{min_wavelength=} must be positive"
        assert max_wavelength > 0, f"{max_wavelength=} must be positive"
        super().__init__()
        self.d_out: int = d_out
        self.min_wavelength: float = min_wavelength
        self.max_wavelength: float = max_wavelength

    def __call__(
        self, rng: jax.Array, params: None, x: jax.Array
    ) -> Tuple[jax.Array, jax.Array]:
        d_in = x.shape[-1]
        assert self.d_out % d_in == 0, f"{self.d_out=} must be divisible by {d_in=}"
        assert (
            self.d_out // d_in
        ) % 2 == 0, f"{self.d_out=} must be an even multiple of {d_in=}"
        num_bands = self.d_out // d_in
        num_freqs = num_bands // 2
        freqs = jnp.exp(
            jnp.linspace(
                jnp.log(self.min_wavelength),
                jnp.log(self.max_wavelength),
                num_freqs,
            )
        )
        y = einx.multiply("b ... d, f -> b ... (d f)", x, freqs)
        return jnp.sin(y), jnp.cos(y)


def rotate_half(x: jax.Array) -> jax.Array:
    x1, x2 = einx.rearrange("... (d h) -> h ... d", x, h=2)
    y = jnp.stack([-x2, x1], axis=0)
    x = einx.rearrange("h ... d -> ... (d h)", y)
    return x


class RotaryPositionEmbedding(Bond[Tuple[jax.Array, jax.Array], jax.Array]):
    def __init__(
        self,
        min_wavelength: float,
        max_wavelength: float,
        rotate_frac: float = 0.5,
    ):
        super().__init__()
        self.min_wavelength: float = min_wavelength
        self.max_wavelength: float = max_wavelength
        self.rotate_frac: float = rotate_frac

    def __call__(
        self, rng: jax.Array, params: None, x: Tuple[jax.Array, jax.Array]
    ) -> jax.Array:
        embs, pos = x
        *_, n_head, _, d_head = embs.shape
        *_, _, d_pos = pos.shape
        d_model = n_head * d_head
        d_rot = int(d_model * self.rotate_frac)
        assert (
            d_rot % d_pos == 0
        ), f"{d_rot=} must be divisible by {d_pos=}. {d_model=}, {self.rotate_frac=}"
        n_bands = d_rot // d_pos
        n_freqs = n_bands // 2
        assert n_freqs % n_head == 0, f"{n_freqs=} must be divisible by {n_head=}"
        freqs = jnp.exp(
            jnp.linspace(
                jnp.log(self.min_wavelength),
                jnp.log(self.max_wavelength),
                n_freqs,
            )
        )
        freqs_per_head = einx.rearrange("(f h) -> h f", freqs, h=n_head)
        pos_freqs = einx.multiply("... s d, h f -> ... h s (d f)", pos, freqs_per_head)
        embs_left, embs_right = jnp.split(embs, [pos_freqs.shape[-1]], axis=-1)
        embs_left = embs_left * jnp.sin(pos_freqs) + embs_right * jnp.cos(pos_freqs)
        return jnp.concatenate([embs_left, embs_right], axis=-1)


class Tuplefy(Bond[jax.Array, Tuple[jax.Array, ...]]):
    def __init__(self, axis: int = 0):
        super().__init__()
        self.axis: int = axis

    def __call__(
        self, rng: jax.Array, params: None, x: jax.Array
    ) -> Tuple[jax.Array, ...]:
        x = jnp.moveaxis(x, self.axis, 0)
        return tuple(x)
