from abc import ABC, abstractmethod
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

    @abstractmethod
    def regularize(
        self,
        key: jax.Array,
        opt_state: OptState,
        params: Params,
        strength: jax.Array,
    ) -> Tuple[OptState, Params]: ...

    @abstractmethod
    def normalize(
        self,
        key: jax.Array,
        opt_state: OptState,
        update: Params,
        target_norm: jax.Array,
    ) -> Tuple[OptState, Params]: ...

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


class SimpleModule(Module[None, Params, X, Y], Generic[Params, X, Y]):
    """A module with no internal optimizer state."""

    @abstractmethod
    def simple_init(self, key: jax.Array) -> Params: ...

    def init(self, key: jax.Array) -> Tuple[None, Params]:
        return None, self.simple_init(key)

    @abstractmethod
    def simple_regularize(
        self,
        key: jax.Array,
        params: Params,
        strength: jax.Array,
    ) -> Params: ...

    def regularize(
        self,
        key: jax.Array,
        opt_state: None,
        params: Params,
        strength: jax.Array,
    ) -> Tuple[None, Params]:
        return None, self.simple_regularize(key, params, strength)

    @abstractmethod
    def simple_normalize(
        self,
        key: jax.Array,
        params: Params,
        target_norm: jax.Array,
    ) -> Params: ...

    def normalize(
        self,
        key: jax.Array,
        opt_state: None,
        update: Params,
        target_norm: jax.Array,
    ) -> Tuple[None, Params]:
        return None, self.simple_normalize(key, update, target_norm)


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

    def normalize(
        self,
        key: jax.Array,
        opt_state: CompositeOptState,
        update: CompositeParams,
        target_norm: jax.Array,
    ):
        if self.mass > 0:
            ks = jax.random.split(key, len(self.children))
            ss, us = []
            sensitivity = 1.0
            for child, k, opt_state, update in zip(
                self.children, ks, opt_state, update
            ):
                s, u = child.normalize(
                    k,
                    opt_state,
                    update,
                    child.mass / self.mass * target_norm / sensitivity,
                )
                sensitivity *= child.sensitivity
                ss.append(s)
                us.append(u)
            opt_state = tuple(ss)
            update = tuple(us)
        else:
            update = jax.tree.map(jnp.zeros_like, update)
        return opt_state, update

    def regularize(
        self,
        key: jax.Array,
        opt_state: CompositeOptState,
        params: CompositeParams,
        strength: jax.Array,
    ) -> Tuple[CompositeOptState, CompositeParams]:
        if self.mass > 0:
            ks = jax.random.split(key, len(self.children))
            ss, ps = []
            sensitivity = 1.0
            for child, k, s, p in zip(self.children, ks, opt_state, params):
                s, p = child.regularize(
                    k,
                    s,
                    p,
                    child.mass / self.mass * strength / sensitivity,
                )
                sensitivity *= child.sensitivity
                ss.append(s)
                ps.append(p)
            opt_state = tuple(ss)
            params = tuple(ps)
        return opt_state, params

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
        self.sensitivity = f.sensitivity * g.sensitivity
        self.length = f.length + g.length
        self.children = [f, g]

    def init(self, key: jax.Array) -> Tuple[CompositeOptState, CompositeParams]:
        kf, kg = jax.random.split(key)
        sf, pf = self.children[0].init(kf)
        sg, pg = self.children[1].init(kg)
        return (sf, sg), (pf, pg)

    def normalize(
        self,
        key: jax.Array,
        opt_state: CompositeOptState,
        update: CompositeParams,
        target_norm: jax.Array,
    ):
        if self.mass > 0:
            sf, sg = opt_state
            pf, pg = update
            kf, kg = jax.random.split(key)
            mf, mg = self.children
            sf, pf = mf.normalize(
                kf, sf, pf, target_norm=target_norm * mf.mass / self.mass
            )
            sg, pg = mg.normalize(
                kg, sg, pg, target_norm=target_norm * mg.mass / self.mass
            )
            return (sf, sg), (pf, pg)
        else:
            return opt_state, jax.tree.map(jnp.zeros_like, update)

    def regularize(
        self,
        key: jax.Array,
        opt_state: CompositeOptState,
        params: CompositeParams,
        strength: jax.Array,
    ) -> Tuple[CompositeOptState, CompositeParams]:
        if self.mass > 0:
            kf, kg = jax.random.split(key)
            sf, sg = opt_state
            pf, pg = params
            mf, mg = self.children
            sf, pf = mf.regularize(kf, sf, pf, strength=strength * mf.mass / self.mass)
            sg, pg = mg.regularize(kg, sg, pg, strength=strength * mg.mass / self.mass)
            return (sf, sg), (pf, pg)
        else:
            return opt_state, params

    def __call__(
        self,
        rng: jax.Array,
        x: X,
        params: CompositeParams,
    ) -> Tuple[Tuple[Y, Z], CompositeOptState]:
        pf, pg = params
        rf, rg = jax.random.split(rng)
        y = self.children[0](rf, x, pf)
        z = self.children[1](rg, x, pg)
        return (y, z), None


class Sum(Module[None, None, Tuple[X, ...], X], Generic[X]):
    def __init__(self):
        super().__init__()
        self.mass = 0
        self.sensitivity = 1
        self.length = 0
        self.children = []

    def init(self, key):
        return None, None

    def normalize(self, key, opt_state, update, target_norm):
        return None, None

    def regularize(self, key, opt_state, params, strength):
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

    def normalize(self, key, opt_state, update, target_norm):
        return None, None

    def regularize(self, key, opt_state, params, strength):
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

    def normalize(self, key, opt_state, update, target_norm):
        return None, None

    def regularize(self, key, opt_state, params, strength):
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

    def normalize(self, key, opt_state, update, target_norm):
        return None, None

    def regularize(self, key, opt_state, params, strength):
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
        share_params: bool = False,
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

    def regularize(
        self,
        key: jax.Array,
        opt_state: OptState,
        params: Params,
        strength: jax.Array,
    ) -> Tuple[OptState, Params]:
        ks = jax.random.split(key, self.depth)

        def scan_body(strength, spk):
            k, s, p = spk
            s, p = self.module.regularize(k, s, p, strength / self.depth)
            return strength / self.module.sensitivity, (s, p)

        _, (ss, ps) = jax.lax.scan(
            scan_body,
            jnp.asarray(strength, dtype=jnp.float32),
            (ks, opt_state, params),
            self.depth,
            reverse=not self.reverse,
        )
        return ss, ps

    def normalize(
        self,
        key: jax.Array,
        opt_state: OptState,
        update: Params,
        target_norm: jax.Array,
    ) -> Tuple[OptState, Params]:
        ks = jax.random.split(key, self.depth)

        def scan_body(target_norm, spk):
            k, s, p = spk
            s, p = self.module.normalize(k, s, p, target_norm / self.depth)
            return target_norm / self.module.sensitivity, (s, p)

        _, (ss, ps) = jax.lax.scan(
            scan_body,
            jnp.asarray(target_norm, dtype=jnp.float32),
            (ks, opt_state, update),
            self.depth,
            reverse=not self.reverse,
        )
        return ss, ps

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
