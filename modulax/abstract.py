from abc import ABC, abstractmethod
import optax
import functools as ft
import numpy as np
from typing import Generic, List, Tuple, TypeVar

import jax
import jax.numpy as jnp

OptState = TypeVar("OptState")
Params = TypeVar("Params")
X = TypeVar("X")
Y = TypeVar("Y")
Z = TypeVar("Z")

OptStateF = TypeVar("OptStateF")
OptStateG = TypeVar("OptStateG")
ParamsF = TypeVar("ParamsF")
ParamsG = TypeVar("ParamsG")

CompositeOptState = Tuple[OptStateF, OptStateG]
CompositeParams = Tuple[ParamsF, ParamsG]


class Module(ABC, Generic[OptState, Params, X, Y]):
    mass: float
    sensitivity: float
    length: int
    children: List["Module"]

    @abstractmethod
    def init(self, key: jax.Array) -> Tuple[OptState, Params]: ...

    def scale_updates(
        self,
        opt_state: OptState,
        update: Params,
        target_norm: jax.typing.ArrayLike,
    ) -> Tuple[OptState, Params]:
        return opt_state, update

    def regularize(
        self,
        opt_state: OptState,
        params: Params,
    ) -> Tuple[OptState, Params]:
        # default to weight decay
        return opt_state, params

    def normalize(
        self,
        opt_state: OptState,
        update: Params,
    ) -> Tuple[OptState, Params]:
        return opt_state, update

    def optax_update_fn(
        self,
        reg_strength: jax.typing.ArrayLike = 0.0,
        target_norm: jax.typing.ArrayLike = 1.0,
        flip_sign: bool = True,
    ) -> optax.TransformUpdateFn:
        def update_fn(update, opt_state, params=None):
            opt_state, update = self.normalize(opt_state, update)
            if params is not None and reg_strength > 0:
                opt_state, reg_update = self.regularize(opt_state, params)
                update = jax.tree.map(
                    lambda u, r: u + reg_strength * r, update, reg_update
                )
            opt_state, update = self.update_scale(
                opt_state,
                params,
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

    @abstractmethod
    def __call__(self, rng: jax.Array, x: X, params: Params) -> Y: ...

    def tare(self, absolute=1, relative=None):
        if relative is not None:
            self.mass *= relative
            for child in self.children:
                child.tare(relative=relative)
        else:
            self.tare(relative=absolute / self.mass)

    def __matmul__(self, other):
        return CompositeModule(self, other)

    def __rmatmul__(self, other):
        return other @ self

    def __add__(self, other: int | float | "Module[OptStateF, ParamsF, X, Y]"):
        if isinstance(other, (int, float)):
            return self @ Add(other)
        else:
            return (self, other) @ Sum()

    def __radd__(self, other: int | float):
        return Add(other) @ self

    def __mul__(self, other: int | float | "Module[OptStateF, ParamsF, X, Y]"):
        if isinstance(other, (int, float)):
            assert other != 0, "cannot multiply a module by zero"
            return self @ Mul(other)
        else:
            return TupleModule(self, other) @ Prod()

    def __rmul__(self, other: int | float):
        assert other != 0, "cannot multiply a module by zero"
        return Mul(other) @ self

    def __truediv__(self, other: int | float):
        assert other != 0, "cannot divide a module by zero"
        return self * (1 / other)

    def __pow__(self, other: int):
        assert other >= 0 and other % 1 == 0, "nonnegative integer powers only"
        return Pow(other, self)


class CompositeModule(
    Module[CompositeOptState, CompositeParams, X, Z],
    Generic[OptStateF, OptStateG, ParamsF, ParamsG, X, Y, Z],
):
    children: List[Module[OptStateF, ParamsF, X, Y] | Module[OptStateG, ParamsG, Y, Z]]

    def __init__(
        self,
        f: Module[OptStateF, ParamsF, X, Y],
        g: Module[OptStateG, ParamsG, Y, Z],
    ):
        self.mass = f.mass + g.mass
        self.sensitivity = f.sensitivity * g.sensitivity
        self.length = f.length + g.length
        self.children = [f, g]

    def init(self, key: jax.Array) -> Tuple[CompositeOptState, CompositeParams]:
        ks = jax.random.split(key, len(self.children))
        sps = [child.init(k) for child, k in zip(self.children, ks)]
        ss, ps = zip(*sps)
        return ss, ps

    def scale_updates(
        self,
        opt_state: CompositeOptState,
        update: CompositeParams,
        target_norm: jax.Array,
    ) -> Tuple[CompositeOptState, CompositeParams]:
        sf, sg = opt_state
        uf, ug = update
        f, g = self.children

        if self.mass > 0:
            sf, scale_f = f.scale_updates(
                sf,
                uf,
                target_norm=target_norm * f.mass / self.mass / g.sensitivity,
            )
            sg, scale_g = g.scale_updates(
                sg, ug, target_norm=target_norm * g.mass / self.mass
            )
        return (sf, sg), (scale_f, scale_g)

    def normalize(
        self,
        opt_state: CompositeOptState,
        update: CompositeParams,
    ) -> Tuple[CompositeOptState, CompositeParams]:
        sf, sg = opt_state
        uf, ug = update
        f, g = self.children
        sf, uf = f.normalize(sf, uf)
        sg, ug = g.normalize(sg, ug)
        return (sf, sg), (uf, ug)

    def regularize(
        self,
        opt_state: CompositeOptState,
        params: CompositeParams,
    ) -> Tuple[CompositeOptState, CompositeParams]:
        sf, sg = opt_state
        uf, ug = params
        f, g = self.children
        sf, uf = f.regularize(sf, uf)
        sg, ug = g.regularize(sg, ug)
        return (sf, sg), (uf, ug)

    def __call__(
        self,
        rng: jax.Array,
        x: X,
        params: CompositeParams,
    ) -> Z:
        pf, pg = params
        rf, rg = jax.random.split(rng)
        y = self.children[0](rf, x, pf)
        z = self.children[1](rg, y, pg)
        return z


class TupleModule(
    Module[CompositeOptState, CompositeParams, X, Tuple[Y, Z]],
    Generic[OptStateF, OptStateG, ParamsF, ParamsG, X, Y, Z],
):
    def __init__(
        self,
        f: Module[OptStateF, ParamsF, X, Y],
        g: Module[OptStateG, ParamsG, X, Z],
    ):
        self.mass = f.mass + g.mass
        self.sensitivity = f.sensitivity + g.sensitivity
        self.length = f.length + g.length
        self.children = [f, g]

    def init(self, key: jax.Array) -> Tuple[CompositeOptState, CompositeParams]:
        kf, kg = jax.random.split(key)
        sf, pf = self.children[0].init(kf)
        sg, pg = self.children[1].init(kg)
        return (sf, sg), (pf, pg)

    def scale_updates(
        self,
        opt_state: CompositeOptState,
        update: CompositeParams,
        target_norm: jax.Array,
    ) -> Tuple[CompositeOptState, CompositeParams]:
        sf, sg = opt_state
        uf, ug = update
        mf, mg = self.children
        sf, uf = mf.scale_updates(
            sf,
            uf,
            target_norm=target_norm * mf.mass / self.mass,
        )
        sg, ug = mg.scale_updates(
            sg,
            ug,
            target_norm=target_norm * mg.mass / self.mass,
        )
        return (sf, sg), (uf, ug)

    def normalize(
        self,
        opt_state: CompositeOptState,
        update: CompositeParams,
    ):
        sf, sg = opt_state
        uf, ug = update
        f, g = self.children
        sf, uf = f.normalize(sf, uf)
        sg, ug = g.normalize(sg, ug)
        return (sf, sg), (uf, ug)

    def regularize(
        self,
        opt_state: CompositeOptState,
        params: CompositeParams,
    ) -> Tuple[CompositeOptState, CompositeParams]:
        sf, sg = opt_state
        uf, ug = params
        f, g = self.children
        sf, uf = f.regularize(sf, uf)
        sg, ug = g.regularize(sg, ug)
        return (sf, sg), (uf, ug)

    def __call__(
        self,
        rng: jax.Array,
        x: X,
        params: CompositeParams,
    ) -> Tuple[Y, Z]:
        pf, pg = params
        rf, rg = jax.random.split(rng)
        y = self.children[0](rf, x, pf)
        z = self.children[1](rg, x, pg)
        return y, z


class Sum(Module[None, None, Tuple[X, ...], X], Generic[X]):
    def __init__(self):
        super().__init__()
        self.mass = 0
        self.sensitivity = 1
        self.length = 0
        self.children = []

    def init(self, key):
        return None, None

    def __call__(self, rng, x: Tuple[X, ...], params):
        return sum(x)


class Add(Module[None, None, X, X], Generic[X]):
    def __init__(self, alpha: float):
        self.alpha = alpha
        self.mass = 0
        self.sensitivity = abs(alpha)
        self.length = 0
        self.children = []

    def init(self, key):
        return None, None

    def __call__(self, rng, x: X, params):
        return jax.tree.map(lambda x: self.alpha + x, x)


class Mul(Module[None, None, X, X], Generic[X]):
    def __init__(self, alpha: float):
        self.alpha = alpha
        self.mass = 0
        self.sensitivity = abs(alpha)
        self.length = 0
        self.children = []

    def init(self, key):
        return None, None

    def __call__(self, rng, x: X, params):
        return jax.tree.map(lambda x: self.alpha * x, x)


class Prod(Module[None, None, Tuple[X, ...], X], Generic[X]):
    def __init__(self):
        self.mass = 0
        self.sensitivity = 1
        self.length = 0
        self.children = []

    def init(self, key):
        return None, None

    def __call__(self, rng, x: Tuple[X, ...], params):
        assert len(x) > 0
        _acc = x[0]
        for x in x[1:]:
            _acc = jax.tree.map(lambda x, y: x * y, _acc, x)
        return _acc


class Pow(Module[OptState, Params, X, Tuple[X, X]], Generic[OptState, Params, X]):
    def __init__(
        self,
        depth: int,
        module: Module[OptState, Params, X, X],
        reverse: bool = False,
        unroll: int = 1,
        _split_transpose: bool = False,
    ):
        self.depth = depth
        self.module = module
        self.mass = module.mass * depth
        self.sensitivity = module.sensitivity**depth
        self.length = module.length * depth
        self.children = [module]
        self.reverse = reverse
        self.unroll = unroll
        self._split_transpose = _split_transpose

    def init(self, key: jax.Array) -> Tuple[OptState, Params]:
        ks = jax.random.split(key, self.depth)
        ss, ps = jax.vmap(self.module.init)(ks)
        return ss, ps

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
        opt_state: OptState,
        params: Params,
    ) -> Tuple[OptState, Params]:
        return jax.vmap(self.module.regularize)(opt_state, params)

    def normalize(
        self,
        opt_state: OptState,
        update: Params,
    ) -> Tuple[OptState, Params]:
        return jax.vmap(self.module.normalize)(opt_state, update)

    def __call__(
        self,
        rng: jax.Array,
        x: X,
        params: Params,
    ) -> Tuple[X, X]:
        ks = jax.random.split(rng, self.depth)

        def scan_body(x: X, loop_params: Tuple[jax.Array, Params]):
            r, p = loop_params
            x = self.module(r, x, p)
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
