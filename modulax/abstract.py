import contextlib
from abc import ABC
from typing import (
    Dict,
    List,
    Literal,
    NamedTuple,
    Optional,
    Sequence,
    Tuple,
    TypeVar,
    Union,
)

import jax
import jax.numpy as jnp
import jax.sharding
import numpy as np
import optax
from jax.sharding import Mesh, NamedSharding, PartitionSpec

OptState = TypeVar("OptState")
Params = TypeVar("Params")
X = TypeVar("X")
Y = TypeVar("Y")
Z = TypeVar("Z")
W = TypeVar("W")
V = TypeVar("V")

OptStateF = TypeVar("OptStateF")
OptStateG = TypeVar("OptStateG")
OptStateH = TypeVar("OptStateH")
OptStateI = TypeVar("OptStateI")
ParamsF = TypeVar("ParamsF")
ParamsG = TypeVar("ParamsG")
ParamsH = TypeVar("ParamsH")
ParamsI = TypeVar("ParamsI")


class ShardingConfig(NamedTuple):
    mesh: Mesh
    fsdp_axis: Union[None, str, Sequence[str]] = "dp"
    dp_axis: Union[None, str, Sequence[str]] = "dp"
    mp_axis: Union[None, str, Sequence[str]] = None


_SHARDING_CONFIG = None


@contextlib.contextmanager
def with_sharding_config(config: ShardingConfig):
    global _SHARDING_CONFIG
    old_config = _SHARDING_CONFIG
    _SHARDING_CONFIG = config
    yield
    _SHARDING_CONFIG = old_config


def get_sharding_config() -> ShardingConfig:
    if _SHARDING_CONFIG is None:
        raise ValueError("sharding config not set")
    return _SHARDING_CONFIG


ShardingMode = Literal["none", "fsdp", "mp", "both"]

_SHARDING_MODE = "none"


@contextlib.contextmanager
def with_sharding_mode(mode: ShardingMode):
    global _SHARDING_MODE
    old_mode = _SHARDING_MODE
    _SHARDING_MODE = mode
    yield
    _SHARDING_MODE = old_mode


def get_sharding_mode() -> ShardingMode:
    return _SHARDING_MODE


class Module(ABC):
    mass: float
    sensitivity: float
    length: int
    children: List["Module"]

    def init_params(self, key: jax.Array) -> Params:
        return None

    def init_opt_state(self, key: jax.Array, params: Params) -> OptState:
        return None

    def scale_updates(
        self,
        opt_state: OptState,
        update: Params,
        target_norm: jax.typing.ArrayLike,
    ) -> Tuple[OptState, Params]:
        return opt_state, update

    def regularize(
        self,
        params: Params,
        opt_state: OptState,
    ) -> Tuple[Params, OptState]:
        # default to weight decay
        return params, opt_state

    def normalize(
        self,
        update: Params,
        opt_state: OptState,
    ) -> Tuple[Params, OptState]:
        return update, opt_state

    def optax_update_fn(
        self,
        reg_strength: jax.typing.ArrayLike = 0.0,
        target_norm: jax.typing.ArrayLike = 1.0,
        flip_sign: bool = True,
    ) -> optax.TransformUpdateFn:
        def update_fn(update, opt_state, params=None):
            update, opt_state = self.normalize(update, opt_state)
            if params is not None and reg_strength > 0:
                params, reg_update = self.regularize(params, opt_state)
                update = jax.tree.map(
                    lambda u, r: u + reg_strength * r, update, reg_update
                )
            opt_state, update = self.scale_updates(
                opt_state,
                update,
                target_norm=target_norm,
            )
            if flip_sign:
                update = jax.tree.map(lambda u: -u, update)
            return update, opt_state

        return update_fn

    def apply_update(
        self,
        opt_state: OptState,
        params: Params,
        update: Params,
        reg_strength: jax.typing.ArrayLike = 0.0,
        target_norm: jax.typing.ArrayLike = 1.0,
    ) -> Tuple[OptState, Params]:
        update, opt_state = self.optax_update_fn(
            reg_strength, target_norm, flip_sign=False
        )(update, opt_state, params)
        params = jax.tree.map(
            lambda p, u: jnp.asarray(p - u, dtype=p.dtype),
            params,
            update,
        )
        return opt_state, params

    def tare(self, absolute=1, relative=None):
        if relative is not None:
            self.mass *= relative
            for child in self.children:
                child.tare(relative=relative)
        else:
            self.tare(relative=absolute / self.mass)

    def shard_params(
        self,
        params: Params,
        *,
        config: Optional[ShardingConfig] = None,
        mode: Optional[ShardingMode] = None,
    ) -> Params:
        if mode is None:
            mode = get_sharding_mode()
        if config is None:
            config = get_sharding_config()
        return self._shard_params(params, config, mode)

    def _shard_params(
        self, params: Params, config: ShardingConfig, mode: ShardingMode
    ) -> Params:
        return jax.tree.map(
            lambda _: NamedSharding(config.mesh, PartitionSpec()),
            params,
        )

    def shard_opt_state(
        self,
        opt_state: OptState,
        *,
        config: Optional[ShardingConfig] = None,
        mode: Optional[ShardingMode] = None,
    ) -> OptState:
        if mode is None:
            mode = get_sharding_mode()
        if config is None:
            config = get_sharding_config()
        return self._shard_opt_state(opt_state, config, mode)

    def _shard_opt_state(
        self, opt_state: OptState, config: ShardingConfig, mode: ShardingMode
    ) -> OptState:
        return jax.tree.map(
            lambda _: NamedSharding(config.mesh, PartitionSpec()),
            opt_state,
        )

    def __call__(self, rng: jax.Array, params: Params, x: X) -> Y:
        raise NotImplementedError(
            f"__call__ not implemented for {self.__class__.__name__}"
        )

    def __matmul__(self, other):
        if isinstance(other, tuple):
            other = TupleModule(*other)
        elif isinstance(other, dict):
            other = DictModule(other)

        children = []
        if isinstance(self, CompositeModule):
            children.extend(self.children)
        else:
            children.append(self)
        if isinstance(other, CompositeModule):
            children.extend(other.children)
        else:
            children.append(other)

        return CompositeModule(*children)

    def __rmatmul__(self, other):
        if isinstance(other, tuple):
            other = TupleModule(*other)
        elif isinstance(other, dict):
            other = DictModule(other)
        else:
            raise ValueError("cannot multiply a module by a non-module")
        return other @ self

    def __add__(self, other: Union[int, float, "Module"]):
        if isinstance(other, (int, float)):
            return self @ Add(other)
        else:
            return (self, other) @ Sum()

    def __radd__(self, other: Union[int, float]):
        return Add(other) @ self

    def __mul__(self, other: Union[int, float, "Module"]):
        if isinstance(other, (int, float)):
            assert other != 0, "cannot multiply a module by zero"
            return self @ Mul(other)
        else:
            return TupleModule(self, other) @ Prod()

    def __rmul__(self, other: Union[int, float]):
        assert other != 0, "cannot multiply a module by zero"
        return Mul(other) @ self

    def __truediv__(self, other: Union[int, float]):
        assert other != 0, "cannot divide a module by zero"
        return self * (1 / other)

    def __pow__(self, other: int):
        assert other >= 0 and other % 1 == 0, "nonnegative integer powers only"
        return Pow(other, self)


class CompositeModule(Module):
    children: List[Module]

    def __init__(self, *modules: Module):
        self.children = list(modules)
        self.mass = sum(m.mass for m in self.children)
        self.sensitivity = np.prod([m.sensitivity for m in self.children])
        self.length = sum(m.length for m in self.children)

    def init_opt_state(self, key: jax.Array, params):
        ks = jax.random.split(key, len(self.children))
        return tuple(
            child.init_opt_state(k, p) for child, k, p in zip(self.children, ks, params)
        )

    def init_params(self, key: jax.Array):
        ks = jax.random.split(key, len(self.children))
        return tuple(child.init_params(k) for child, k in zip(self.children, ks))

    def scale_updates(
        self,
        opt_state,
        update,
        target_norm: jax.Array,
    ):
        scaled_states = []
        scaled_updates = []
        remaining_sensitivity = 1.0
        for i, (child, state, upd) in enumerate(zip(self.children, opt_state, update)):
            if i < len(self.children) - 1:
                child_target_norm = (
                    target_norm * child.mass / self.mass / remaining_sensitivity
                )
                remaining_sensitivity *= child.sensitivity
            else:
                child_target_norm = target_norm * child.mass / self.mass

            s, u = child.scale_updates(state, upd, target_norm=child_target_norm)
            scaled_states.append(s)
            scaled_updates.append(u)

        return tuple(scaled_states), tuple(scaled_updates)

    def normalize(
        self,
        update,
        opt_state,
    ):
        normalized_updates = []
        normalized_states = []
        for child, u, s in zip(self.children, update, opt_state):
            nu, ns = child.normalize(u, s)
            normalized_updates.append(nu)
            normalized_states.append(ns)
        return tuple(normalized_updates), tuple(normalized_states)

    def regularize(
        self,
        params,
        opt_state,
    ):
        regularized_params = []
        regularized_states = []
        for child, p, s in zip(self.children, params, opt_state):
            rp, rs = child.regularize(p, s)
            regularized_params.append(rp)
            regularized_states.append(rs)
        return tuple(regularized_params), tuple(regularized_states)

    def __call__(
        self,
        rng: jax.Array,
        params,
        x,
    ):
        rngs = jax.random.split(rng, len(self.children))
        for child, p, r in zip(self.children[::-1], params[::-1], rngs):
            x = child(r, p, x)
        return x

    def _shard_params(
        self, params: Params, config: ShardingConfig, mode: ShardingMode
    ) -> Params:
        return tuple(
            child._shard_params(p, config, mode)
            for child, p in zip(self.children, params)
        )

    def _shard_opt_state(
        self, opt_state: OptState, config: ShardingConfig, mode: ShardingMode
    ) -> OptState:
        return tuple(
            child._shard_opt_state(s, config, mode)
            for child, s in zip(self.children, opt_state)
        )


class TupleModule(Module):
    def __init__(self, *modules: Module):
        self.mass = sum(m.mass for m in modules)
        self.sensitivity = sum(m.sensitivity for m in modules)
        self.length = sum(m.length for m in modules)
        self.children = list(modules)

    def init_opt_state(self, key: jax.Array, params):
        keys = jax.random.split(key, len(self.children))
        return tuple(
            child.init_opt_state(k, p)
            for child, k, p in zip(self.children, keys, params)
        )

    def init_params(self, key: jax.Array):
        keys = jax.random.split(key, len(self.children))
        return tuple(child.init_params(k) for child, k in zip(self.children, keys))

    def scale_updates(
        self,
        opt_state,
        update,
        target_norm: jax.Array,
    ):
        scaled_states_and_updates = [
            child.scale_updates(s, u, target_norm=target_norm * child.mass / self.mass)
            for child, s, u in zip(self.children, opt_state, update)
        ]
        return tuple(zip(*scaled_states_and_updates))

    def normalize(
        self,
        update,
        opt_state,
    ):
        normalized = [
            child.normalize(u, s)
            for child, u, s in zip(self.children, update, opt_state)
        ]
        return tuple(zip(*normalized))

    def regularize(
        self,
        params,
        opt_state,
    ):
        regularized = [
            child.regularize(p, s)
            for child, p, s in zip(self.children, params, opt_state)
        ]
        return tuple(zip(*regularized))

    def __call__(
        self,
        rng: jax.Array,
        params,
        x,
    ):
        keys = jax.random.split(rng, len(self.children))
        return tuple(child(k, p, x) for child, k, p in zip(self.children, keys, params))

    def __getitem__(self, idx: int):
        return self.children[idx]

    def __len__(self):
        return len(self.children)

    def __iter__(self):
        return iter(self.children)

    def _shard_params(
        self, params: Params, config: ShardingConfig, mode: ShardingMode
    ) -> Params:
        return tuple(
            child._shard_params(p, config, mode)
            for child, p in zip(self.children, params)
        )

    def _shard_opt_state(
        self, opt_state: OptState, config: ShardingConfig, mode: ShardingMode
    ) -> OptState:
        return tuple(
            child._shard_opt_state(s, config, mode)
            for child, s in zip(self.children, opt_state)
        )


class DictModule(Module):
    def __init__(self, modules: Dict[str, Module]):
        self.modules = modules
        self.mass = sum(m.mass for m in self.modules.values())
        self.sensitivity = sum(m.sensitivity for m in self.modules.values())
        self.length = sum(m.length for m in self.modules.values())
        self.children = list(self.modules.values())

    def init_opt_state(
        self, key: jax.Array, params: Dict[str, Params]
    ) -> Dict[str, OptState]:
        keys = jax.random.split(key, len(self.modules))
        return {
            k: m.init_opt_state(keys[i], params[k])
            for i, (k, m) in enumerate(self.modules.items())
        }

    def init_params(self, key: jax.Array) -> Dict[str, Params]:
        keys = jax.random.split(key, len(self.modules))
        return {
            k: m.init_params(keys[i]) for i, (k, m) in enumerate(self.modules.items())
        }

    def scale_updates(
        self,
        opt_state: Dict[str, OptState],
        update: Dict[str, Params],
        target_norm: jax.Array,
    ) -> Tuple[Dict[str, OptState], Dict[str, Params]]:
        scaled_states, scaled_updates = {}, {}
        for k, m in self.modules.items():
            s, u = m.scale_updates(
                opt_state[k], update[k], target_norm * m.mass / self.mass
            )
            scaled_states[k], scaled_updates[k] = s, u
        return scaled_states, scaled_updates

    def normalize(
        self,
        update: Dict[str, Params],
        opt_state: Dict[str, OptState],
    ) -> Tuple[Dict[str, Params], Dict[str, OptState]]:
        normalized_updates, normalized_states = {}, {}
        for k, m in self.modules.items():
            u, s = m.normalize(update[k], opt_state[k])
            normalized_updates[k], normalized_states[k] = u, s
        return normalized_updates, normalized_states

    def regularize(
        self,
        params: Dict[str, Params],
        opt_state: Dict[str, OptState],
    ) -> Tuple[Dict[str, Params], Dict[str, OptState]]:
        regularized_params, regularized_states = {}, {}
        for k, m in self.modules.items():
            p, s = m.regularize(params[k], opt_state[k])
            regularized_params[k], regularized_states[k] = p, s
        return regularized_params, regularized_states

    def __call__(
        self,
        rng: jax.Array,
        params: Dict[str, Params],
        x: X,
    ) -> Dict[str, Y]:
        keys = jax.random.split(rng, len(self.modules))
        return {
            k: m(keys[i], params[k], x) for i, (k, m) in enumerate(self.modules.items())
        }

    def __getitem__(self, key: str) -> Module:
        return self.modules[key]

    def __len__(self):
        return len(self.modules)

    def __iter__(self):
        return iter(self.modules)

    def items(self):
        return self.modules.items()

    def keys(self):
        return self.modules.keys()

    def values(self):
        return self.modules.values()

    def _shard_params(
        self, params: Dict[str, Params], config: ShardingConfig, mode: ShardingMode
    ) -> Dict[str, Params]:
        return {
            k: m._shard_params(params[k], config, mode) for k, m in self.modules.items()
        }

    def _shard_opt_state(
        self, opt_state: Dict[str, OptState], config: ShardingConfig, mode: ShardingMode
    ) -> Dict[str, OptState]:
        return {
            k: m._shard_opt_state(opt_state[k], config, mode)
            for k, m in self.modules.items()
        }


class Sum(Module):
    def __init__(self):
        super().__init__()
        self.mass = 0
        self.sensitivity = 1
        self.length = 0
        self.children = []

    def init_opt_state(self, key, params):
        return None

    def init_params(self, key):
        return None

    def __call__(self, rng, params, x: Tuple[X, ...]):
        return jax.tree.map(lambda *xs: sum(xs), *x)


class Add(Module):
    def __init__(self, alpha: float):
        self.alpha = alpha
        self.mass = 0
        self.sensitivity = abs(alpha)
        self.length = 0
        self.children = []

    def init_opt_state(self, key, params):
        return None

    def init_params(self, key):
        return None

    def __call__(self, rng, params, x: X):
        return jax.tree.map(lambda x: self.alpha + x, x)


class Mul(Module):
    def __init__(self, alpha: float):
        self.alpha = alpha
        self.mass = 0
        self.sensitivity = abs(alpha)
        self.length = 0
        self.children = []

    def init_opt_state(self, key, params):
        return None

    def init_params(self, key):
        return None

    def __call__(self, rng, params, x: X):
        return jax.tree.map(lambda x: self.alpha * x, x)


class Prod(Module):
    def __init__(self):
        self.mass = 0
        self.sensitivity = 1
        self.length = 0
        self.children = []

    def init_opt_state(self, key, params):
        return None

    def init_params(self, key):
        return None

    def __call__(self, rng, params, x: Tuple[X, ...]):
        if len(x) == 0:
            return 1
        _acc = x[0]
        for x in x[1:]:
            _acc = jax.tree.map(lambda acc, x: acc * x, _acc, x)
        return _acc


class Pow(Module):
    def __init__(
        self,
        depth: int,
        module: Module,
        reverse: bool = False,
        unroll: int = 1,
        _split_transpose: bool = False,
    ):
        self.depth = depth
        self.module = module
        self.mass = module.mass * depth
        self.sensitivity = module.sensitivity**depth
        self.length = module.length
        self.children = [module]
        self.reverse = reverse
        self.unroll = unroll
        self._split_transpose = _split_transpose

    def init_opt_state(self, key: jax.Array, params: Params) -> OptState:
        ks = jax.random.split(key, self.depth)
        return jax.vmap(self.module.init_opt_state)(ks, params)

    def init_params(self, key: jax.Array) -> Params:
        ks = jax.random.split(key, self.depth)
        return jax.vmap(self.module.init_params)(ks)

    def scale_updates(
        self,
        opt_state: OptState,
        update: Params,
        target_norm: jax.Array,
    ) -> Tuple[OptState, Params]:
        target_norms = np.array(
            [
                target_norm / self.module.sensitivity**i / self.depth
                for i in reversed(range(self.depth))
            ]
        )
        opt_state, update = jax.vmap(self.module.scale_updates)(
            opt_state, update, target_norms
        )
        return opt_state, update

    def regularize(
        self,
        params: Params,
        opt_state: OptState,
    ) -> Tuple[Params, OptState]:
        return jax.vmap(self.module.regularize)(params, opt_state)

    def normalize(
        self,
        update: Params,
        opt_state: OptState,
    ) -> Tuple[Params, OptState]:
        return jax.vmap(self.module.normalize)(update, opt_state)

    def __call__(
        self,
        rng: jax.Array,
        params: Params,
        x: X,
    ) -> Tuple[X, X]:
        ks = jax.random.split(rng, self.depth)

        def scan_body(x: X, loop_params: Tuple[jax.Array, Params]):
            r, p = loop_params
            x = self.module(r, p, x)
            return x, x

        y, ys = jax.lax.scan(
            scan_body,
            x,
            (ks, params),
            self.depth,
            reverse=self.reverse,
            unroll=self.unroll,
            _split_transpose=self._split_transpose,
        )

        return y, ys


class VMap(Module):
    def __init__(
        self,
        n: int,
        module: Module,
        tuple_mode: bool = True,
    ):
        self.n = n
        self.module = module
        self.mass = module.mass * n
        self.sensitivity = module.sensitivity**n
        self.length = module.length
        self.children = [module]
        self.tuple_mode = tuple_mode

    def init_opt_state(self, key: jax.Array, params: Params) -> OptState:
        ks = jax.random.split(key, self.n)
        return jax.vmap(self.module.init_opt_state)(ks, params)

    def init_params(self, key: jax.Array) -> Params:
        ks = jax.random.split(key, self.n)
        return jax.vmap(self.module.init_params)(ks)

    def scale_updates(
        self,
        opt_state: OptState,
        update: Params,
        target_norm: jax.Array,
    ) -> Tuple[OptState, Params]:
        opt_state, update = jax.vmap(self.module.scale_updates)(
            opt_state, update, target_norm / self.n
        )
        return opt_state, update

    def regularize(
        self,
        params: Params,
        opt_state: OptState,
    ) -> Tuple[Params, OptState]:
        return jax.vmap(self.module.regularize)(params, opt_state)

    def normalize(
        self,
        update: Params,
        opt_state: OptState,
    ) -> Tuple[Params, OptState]:
        return jax.vmap(self.module.normalize)(update, opt_state)

    def __call__(
        self,
        rng: jax.Array,
        params: Params,
        x: X,
    ) -> Y:
        ks = jax.random.split(rng, self.n)
        if self.tuple_mode:
            x = jax.tree.map(lambda x: x[None], x)
        y = jax.vmap(self.module)(ks, params, x)
        return y


class WrapperModule(Module):
    def __init__(self, inner: Module):
        self.inner = inner
        self.mass = inner.mass
        self.sensitivity = inner.sensitivity
        self.length = inner.length
        self.children = [inner]

    def init_opt_state(self, key: jax.Array, params: Params) -> OptState:
        return self.inner.init_opt_state(key, params)

    def init_params(self, key: jax.Array) -> Params:
        return self.inner.init_params(key)

    def scale_updates(
        self,
        opt_state: OptState,
        update: Params,
        target_norm: jax.Array,
    ) -> Tuple[OptState, Params]:
        return self.inner.scale_updates(opt_state, update, target_norm)

    def regularize(
        self,
        params: Params,
        opt_state: OptState,
    ) -> Tuple[Params, OptState]:
        return self.inner.regularize(params, opt_state)

    def normalize(
        self,
        update: Params,
        opt_state: OptState,
    ) -> Tuple[Params, OptState]:
        return self.inner.normalize(update, opt_state)

    def __call__(
        self,
        rng: jax.Array,
        params: Params,
        x: X,
    ) -> Y:
        return self.inner(rng, params, x)

    def _shard_params(
        self, params: Params, config: ShardingConfig, mode: ShardingMode
    ) -> Params:
        return self.inner._shard_params(params, config, mode)

    def _shard_opt_state(
        self, opt_state: OptState, config: ShardingConfig, mode: ShardingMode
    ) -> OptState:
        return self.inner._shard_opt_state(opt_state, config, mode)
