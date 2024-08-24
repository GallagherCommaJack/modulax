from typing import Tuple, TypeVar

import jax
import jax.numpy as jnp
from flax import nnx

Params = TypeVar("Params")
OptState = TypeVar("OptState")


class ModulaParam(nnx.Param):
    sensitivity: float
    mass: float

    def __init__(self, value: jax.Array, sensitivity: float = 1.0, mass: float = 1.0):
        self.sensitivity = sensitivity
        self.mass = mass
        super().__init__(value)

    def init_opt_state(self, key: jax.Array) -> OptState:
        return None

    def scale_updates(
        self,
        update: Params,
        opt_state: OptState,
        target_norm: jax.typing.ArrayLike = 1.0,
    ) -> Tuple[Params, OptState]:
        return (
            jax.tree.map(lambda u: u * target_norm, update),
            opt_state,
        )

    def regularize(
        self, update: Params, opt_state: OptState
    ) -> Tuple[Params, OptState]:
        return update, opt_state

    def normalize(self, update: Params, opt_state: OptState) -> Tuple[Params, OptState]:
        return jax.tree.map(lambda u: u / self.sensitivity, update), opt_state


class SpectralMatrix(ModulaParam):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        mass: float = 1.0,
        sensitivity: float = 1.0,
        num_specnorm_vecs: int = 8,
        eps: float = 1e-6,
        weight_decay: float = 0.1,
        *,
        rngs: nnx.Rngs,
    ):
        super().__init__(
            value=nnx.initializers.orthogonal(column_axis=-2)(
                rngs.params(),
                (out_features, in_features),
            ),
            sensitivity=sensitivity,
            mass=mass,
        )
        self.num_specnorm_vecs = num_specnorm_vecs
        self.eps = eps
        self.weight_decay = weight_decay

    def init_opt_state(self, key: jax.Array) -> jax.Array:
        vecs = nnx.initializers.orthogonal()(
            key, (self.num_specnorm_vecs, self.in_features)
        )
        return vecs

    def normalize(
        self, update: jax.Array, opt_state: jax.Array
    ) -> Tuple[jax.Array, jax.Array]:
        v = jnp.einsum("ki,oi->ko", opt_state, update)
        v /= jnp.linalg.norm(v, axis=-1, keepdims=True)
        u = jnp.einsum("ko,oi->ki", v, update)
        un = jnp.linalg.norm(u, axis=-1, keepdims=True)
        u /= un
        update /= jnp.max(un) + self.eps
        return update, u

    def regularize(
        self, update: jax.Array, opt_state: jax.Array
    ) -> Tuple[jax.Array, jax.Array]:
        update -= self.weight_decay * self.value
        return update, opt_state
