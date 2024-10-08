import math
from typing import NamedTuple, Optional, Tuple, Union

import einx
import jax
import jax.numpy as jnp

from .abstract import Module


class FullAttentionInputs(NamedTuple):
    q: jax.Array
    k: jax.Array
    v: jax.Array
    kv_mask: Optional[jax.Array] = None
    pairwise_mask: Optional[jax.Array] = None


AttentionInputs = Union[FullAttentionInputs, Tuple[jax.Array, jax.Array, jax.Array]]


class Bond(Module):
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
        return einx.rearrange("b ... -> b (...)", x)


class AddHeads(Bond):
    """Reshapes an input to have heads."""

    def __init__(self, num_heads: int):
        super().__init__()
        self.num_heads: int = num_heads

    def __call__(self, rng: jax.Array, params: None, x: jax.Array) -> jax.Array:
        return einx.rearrange("b t (h d) -> b h t d", x, h=self.num_heads)


class RemoveHeads(Bond):
    """Inverse of AddHeads."""

    def __call__(self, rng: jax.Array, params: None, x: jax.Array) -> jax.Array:
        return einx.rearrange("b h t d -> b t (h d)", x)


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

    def __call__(self, rng: jax.Array, params: None, x: AttentionInputs) -> jax.Array:
        if isinstance(x, FullAttentionInputs):
            q, k, v = x.q, x.k, x.v
            kv_mask = x.kv_mask
            pairwise_mask = x.pairwise_mask
        else:
            q, k, v = x
            kv_mask = None
            pairwise_mask = None
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


class FourierFeatures(Bond):
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


class RotaryPositionEmbedding(Bond):
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


class Tuplefy(Bond):
    def __init__(self, axis: int = 0):
        super().__init__()
        self.axis: int = axis

    def __call__(
        self, rng: jax.Array, params: None, x: jax.Array
    ) -> Tuple[jax.Array, ...]:
        x = jnp.moveaxis(x, self.axis, 0)
        return tuple(x)