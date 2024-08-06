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


class Module(ABC, Generic[OptState, Params, X, Y]):
    mass: float
    sensitivity: float
    length: int
    children: List["Module"]

    @abstractmethod
    def init_opt_state(self, key: jax.Array, params: Params) -> OptState: ...

    @abstractmethod
    def init_params(self, key: jax.Array) -> Params: ...

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
    def __call__(self, rng: jax.Array, params: Params, x: X) -> Y: ...

    def tare(self, absolute=1, relative=None):
        if relative is not None:
            self.mass *= relative
            for child in self.children:
                child.tare(relative=relative)
        else:
            self.tare(relative=absolute / self.mass)

    def __matmul__(self, other):
        if isinstance(other, tuple):
            if len(other) == 2:
                other = TupleModule(*other)
            elif len(other) == 3:
                other = TripleModule(*other)
            elif len(other) == 4:
                other = QuadrupleModule(*other)
            else:
                raise ValueError(
                    f"cannot multiply a module by a tuple of length {len(other)}"
                )
        return CompositeModule(self, other)

    def __rmatmul__(self, other):
        if isinstance(other, tuple):
            if len(other) == 2:
                other = TupleModule(*other)
            elif len(other) == 3:
                other = TripleModule(*other)
            elif len(other) == 4:
                other = QuadrupleModule(*other)
            else:
                raise ValueError(
                    f"cannot multiply a module by a tuple of length {len(other)}"
                )
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
    Module[Tuple[OptStateF, OptStateG], Tuple[ParamsF, ParamsG], X, Z],
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

    def init_opt_state(
        self, key: jax.Array, params: Tuple[ParamsF, ParamsG]
    ) -> Tuple[OptStateF, OptStateG]:
        ks = jax.random.split(key, len(self.children))
        return tuple(
            child.init_opt_state(k, p) for child, k, p in zip(self.children, ks, params)
        )

    def init_params(self, key: jax.Array) -> Tuple[ParamsF, ParamsG]:
        ks = jax.random.split(key, len(self.children))
        return tuple(child.init_params(k) for child, k in zip(self.children, ks))

    def scale_updates(
        self,
        opt_state: Tuple[OptStateF, OptStateG],
        update: Tuple[ParamsF, ParamsG],
        target_norm: jax.Array,
    ) -> Tuple[Tuple[OptStateF, OptStateG], Tuple[ParamsF, ParamsG]]:
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
        update: Tuple[ParamsF, ParamsG],
        opt_state: Tuple[OptStateF, OptStateG],
    ) -> Tuple[Tuple[ParamsF, ParamsG], Tuple[OptStateF, OptStateG]]:
        uf, ug = update
        sf, sg = opt_state
        f, g = self.children
        uf, sf = f.normalize(uf, sf)
        ug, sg = g.normalize(ug, sg)
        return (uf, ug), (sf, sg)

    def regularize(
        self,
        params: Tuple[ParamsF, ParamsG],
        opt_state: Tuple[OptStateF, OptStateG],
    ) -> Tuple[Tuple[ParamsF, ParamsG], Tuple[OptStateF, OptStateG]]:
        uf, ug = params
        sf, sg = opt_state
        f, g = self.children
        uf, sf = f.regularize(uf, sf)
        ug, sg = g.regularize(ug, sg)
        return (uf, ug), (sf, sg)

    def __call__(
        self,
        rng: jax.Array,
        params: Tuple[ParamsF, ParamsG],
        x: X,
    ) -> Z:
        pf, pg = params
        rf, rg = jax.random.split(rng)
        y = self.children[0](rf, pf, x)
        z = self.children[1](rg, pg, y)
        return z


class TupleModule(
    Module[Tuple[OptStateF, OptStateG], Tuple[ParamsF, ParamsG], X, Tuple[Y, Z]],
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

    def init_opt_state(
        self, key: jax.Array, params: Tuple[ParamsF, ParamsG]
    ) -> Tuple[OptStateF, OptStateG]:
        kf, kg = jax.random.split(key)
        pf, pg = params
        sf = self.children[0].init_opt_state(kf, pf)
        sg = self.children[1].init_opt_state(kg, pg)
        return (sf, sg)

    def init_params(self, key: jax.Array) -> Tuple[ParamsF, ParamsG]:
        kf, kg = jax.random.split(key)
        pf = self.children[0].init_params(kf)
        pg = self.children[1].init_params(kg)
        return (pf, pg)

    def scale_updates(
        self,
        opt_state: Tuple[OptStateF, OptStateG],
        update: Tuple[ParamsF, ParamsG],
        target_norm: jax.Array,
    ) -> Tuple[Tuple[OptStateF, OptStateG], Tuple[ParamsF, ParamsG]]:
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
        update: Tuple[ParamsF, ParamsG],
        opt_state: Tuple[OptStateF, OptStateG],
    ):
        uf, ug = update
        sf, sg = opt_state
        f, g = self.children
        uf, sf = f.normalize(uf, sf)
        ug, sg = g.normalize(ug, sg)
        return (uf, ug), (sf, sg)

    def regularize(
        self,
        params: Tuple[ParamsF, ParamsG],
        opt_state: Tuple[OptStateF, OptStateG],
    ) -> Tuple[Tuple[ParamsF, ParamsG], Tuple[OptStateF, OptStateG]]:
        uf, ug = params
        sf, sg = opt_state
        f, g = self.children
        uf, sf = f.regularize(uf, sf)
        ug, sg = g.regularize(ug, sg)
        return (uf, ug), (sf, sg)

    def __call__(
        self,
        rng: jax.Array,
        params: Tuple[ParamsF, ParamsG],
        x: X,
    ) -> Tuple[Y, Z]:
        pf, pg = params
        rf, rg = jax.random.split(rng)
        y = self.children[0](rf, pf, x)
        z = self.children[1](rg, pg, x)
        return y, z


class TripleModule(
    Module[
        Tuple[OptStateF, OptStateG, OptStateH],
        Tuple[ParamsF, ParamsG, ParamsH],
        X,
        Tuple[Y, Z, W],
    ],
    Generic[OptStateF, OptStateG, OptStateH, ParamsF, ParamsG, ParamsH, X, Y, Z, W],
):
    def __init__(
        self,
        f: Module[OptStateF, ParamsF, X, Y],
        g: Module[OptStateG, ParamsG, X, Z],
        h: Module[OptStateH, ParamsH, X, W],
    ):
        self.mass = f.mass + g.mass + h.mass
        self.sensitivity = f.sensitivity + g.sensitivity + h.sensitivity
        self.length = f.length + g.length + h.length
        self.children = [f, g, h]

    def init_opt_state(
        self, key: jax.Array, params: Tuple[ParamsF, ParamsG, ParamsH]
    ) -> Tuple[OptStateF, OptStateG, OptStateH]:
        kf, kg, kh = jax.random.split(key, 3)
        pf, pg, ph = params
        sf = self.children[0].init_opt_state(kf, pf)
        sg = self.children[1].init_opt_state(kg, pg)
        sh = self.children[2].init_opt_state(kh, ph)
        return (sf, sg, sh)

    def init_params(self, key: jax.Array) -> Tuple[ParamsF, ParamsG, ParamsH]:
        kf, kg, kh = jax.random.split(key, 3)
        pf = self.children[0].init_params(kf)
        pg = self.children[1].init_params(kg)
        ph = self.children[2].init_params(kh)
        return (pf, pg, ph)

    def scale_updates(
        self,
        opt_state: Tuple[OptStateF, OptStateG, OptStateH],
        update: Tuple[ParamsF, ParamsG, ParamsH],
        target_norm: jax.Array,
    ) -> Tuple[
        Tuple[OptStateF, OptStateG, OptStateH], Tuple[ParamsF, ParamsG, ParamsH]
    ]:
        sf, sg, sh = opt_state
        uf, ug, uh = update
        mf, mg, mh = self.children
        sf, uf = mf.scale_updates(sf, uf, target_norm=target_norm * mf.mass / self.mass)
        sg, ug = mg.scale_updates(sg, ug, target_norm=target_norm * mg.mass / self.mass)
        sh, uh = mh.scale_updates(sh, uh, target_norm=target_norm * mh.mass / self.mass)
        return (sf, sg, sh), (uf, ug, uh)

    def normalize(
        self,
        update: Tuple[ParamsF, ParamsG, ParamsH],
        opt_state: Tuple[OptStateF, OptStateG, OptStateH],
    ):
        uf, ug, uh = update
        sf, sg, sh = opt_state
        f, g, h = self.children
        uf, sf = f.normalize(uf, sf)
        ug, sg = g.normalize(ug, sg)
        uh, sh = h.normalize(uh, sh)
        return (uf, ug, uh), (sf, sg, sh)

    def regularize(
        self,
        params: Tuple[ParamsF, ParamsG, ParamsH],
        opt_state: Tuple[OptStateF, OptStateG, OptStateH],
    ) -> Tuple[
        Tuple[ParamsF, ParamsG, ParamsH], Tuple[OptStateF, OptStateG, OptStateH]
    ]:
        uf, ug, uh = params
        sf, sg, sh = opt_state
        f, g, h = self.children
        uf, sf = f.regularize(uf, sf)
        ug, sg = g.regularize(ug, sg)
        uh, sh = h.regularize(uh, sh)
        return (uf, ug, uh), (sf, sg, sh)

    def __call__(
        self,
        rng: jax.Array,
        params: Tuple[ParamsF, ParamsG, ParamsH],
        x: X,
    ) -> Tuple[Y, Z, W]:
        pf, pg, ph = params
        rf, rg, rh = jax.random.split(rng, 3)
        y = self.children[0](rf, pf, x)
        z = self.children[1](rg, pg, x)
        w = self.children[2](rh, ph, x)
        return y, z, w


class QuadrupleModule(
    Module[
        Tuple[OptStateF, OptStateG, OptStateH, OptStateI],
        Tuple[ParamsF, ParamsG, ParamsH, ParamsI],
        X,
        Tuple[Y, Z, W, V],
    ],
    Generic[
        OptStateF,
        OptStateG,
        OptStateH,
        OptStateI,
        ParamsF,
        ParamsG,
        ParamsH,
        ParamsI,
        X,
        Y,
        Z,
        W,
        V,
    ],
):
    def __init__(
        self,
        f: Module[OptStateF, ParamsF, X, Y],
        g: Module[OptStateG, ParamsG, X, Z],
        h: Module[OptStateH, ParamsH, X, W],
        i: Module[OptStateI, ParamsI, X, V],
    ):
        self.mass = f.mass + g.mass + h.mass + i.mass
        self.sensitivity = f.sensitivity + g.sensitivity + h.sensitivity + i.sensitivity
        self.length = f.length + g.length + h.length + i.length
        self.children = [f, g, h, i]

    def init_opt_state(
        self, key: jax.Array, params: Tuple[ParamsF, ParamsG, ParamsH, ParamsI]
    ) -> Tuple[OptStateF, OptStateG, OptStateH, OptStateI]:
        kf, kg, kh, ki = jax.random.split(key, 4)
        pf, pg, ph, pi = params
        sf = self.children[0].init_opt_state(kf, pf)
        sg = self.children[1].init_opt_state(kg, pg)
        sh = self.children[2].init_opt_state(kh, ph)
        si = self.children[3].init_opt_state(ki, pi)
        return (sf, sg, sh, si)

    def init_params(self, key: jax.Array) -> Tuple[ParamsF, ParamsG, ParamsH, ParamsI]:
        kf, kg, kh, ki = jax.random.split(key, 4)
        pf = self.children[0].init_params(kf)
        pg = self.children[1].init_params(kg)
        ph = self.children[2].init_params(kh)
        pi = self.children[3].init_params(ki)
        return (pf, pg, ph, pi)

    def scale_updates(
        self,
        opt_state: Tuple[OptStateF, OptStateG, OptStateH, OptStateI],
        update: Tuple[ParamsF, ParamsG, ParamsH, ParamsI],
        target_norm: jax.Array,
    ) -> Tuple[
        Tuple[OptStateF, OptStateG, OptStateH, OptStateI],
        Tuple[ParamsF, ParamsG, ParamsH, ParamsI],
    ]:
        sf, sg, sh, si = opt_state
        uf, ug, uh, ui = update
        mf, mg, mh, mi = self.children
        sf, uf = mf.scale_updates(sf, uf, target_norm=target_norm * mf.mass / self.mass)
        sg, ug = mg.scale_updates(sg, ug, target_norm=target_norm * mg.mass / self.mass)
        sh, uh = mh.scale_updates(sh, uh, target_norm=target_norm * mh.mass / self.mass)
        si, ui = mi.scale_updates(si, ui, target_norm=target_norm * mi.mass / self.mass)
        return (sf, sg, sh, si), (uf, ug, uh, ui)

    def normalize(
        self,
        update: Tuple[ParamsF, ParamsG, ParamsH, ParamsI],
        opt_state: Tuple[OptStateF, OptStateG, OptStateH, OptStateI],
    ):
        uf, ug, uh, ui = update
        sf, sg, sh, si = opt_state
        f, g, h, i = self.children
        uf, sf = f.normalize(uf, sf)
        ug, sg = g.normalize(ug, sg)
        uh, sh = h.normalize(uh, sh)
        ui, si = i.normalize(ui, si)
        return (uf, ug, uh, ui), (sf, sg, sh, si)

    def regularize(
        self,
        params: Tuple[ParamsF, ParamsG, ParamsH, ParamsI],
        opt_state: Tuple[OptStateF, OptStateG, OptStateH, OptStateI],
    ) -> Tuple[
        Tuple[ParamsF, ParamsG, ParamsH, ParamsI],
        Tuple[OptStateF, OptStateG, OptStateH, OptStateI],
    ]:
        uf, ug, uh, ui = params
        sf, sg, sh, si = opt_state
        f, g, h, i = self.children
        uf, sf = f.regularize(uf, sf)
        ug, sg = g.regularize(ug, sg)
        uh, sh = h.regularize(uh, sh)
        ui, si = i.regularize(ui, si)
        return (uf, ug, uh, ui), (sf, sg, sh, si)

    def __call__(
        self,
        rng: jax.Array,
        params: Tuple[ParamsF, ParamsG, ParamsH, ParamsI],
        x: X,
    ) -> Tuple[Y, Z, W, V]:
        pf, pg, ph, pi = params
        rf, rg, rh, ri = jax.random.split(rng, 4)
        y = self.children[0](rf, pf, x)
        z = self.children[1](rg, pg, x)
        w = self.children[2](rh, ph, x)
        v = self.children[3](ri, pi, x)
        return y, z, w, v


class Sum(Module[None, None, Tuple[X, ...], X], Generic[X]):
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
        return sum(x)


class Add(Module[None, None, X, X], Generic[X]):
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


class Mul(Module[None, None, X, X], Generic[X]):
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


class Prod(Module[None, None, Tuple[X, ...], X], Generic[X]):
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


class VMap(Module[OptState, Params, X, Y], Generic[OptState, Params, X, Y]):
    def __init__(
        self,
        n: int,
        module: Module[OptState, Params, X, X],
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
    ) -> Tuple[X, X]:
        ks = jax.random.split(rng, self.n)
        if self.tuple_mode:
            x = jax.tree.map(lambda x: x[None], x)
        y = jax.vmap(self.module)(ks, params, x)
        return y
