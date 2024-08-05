import copy
import jax.numpy as jnp
import jax
from abc import ABC, abstractmethod
from typing import TypeVar, Tuple, Generic, List

State = TypeVar("State")
Params = TypeVar("Params")
X = TypeVar("X")
Y = TypeVar("Y")
Z = TypeVar("Z")

StateF = TypeVar("StateF")
StateG = TypeVar("StateG")
ParamsF = TypeVar("ParamsF")
ParamsG = TypeVar("ParamsG")

CompositeState = Tuple[StateF, StateG]
CompositeParams = Tuple[ParamsF, ParamsG]


class Module(ABC, Generic[State, Params, X, Y]):
    mass: float
    sensitivity: float
    length: int
    children: List["Module"]

    @abstractmethod
    def init(self, key: jax.Array) -> Tuple[State, Params]: ...

    @abstractmethod
    def regularize(
        self,
        key: jax.Array,
        state: State,
        params: Params,
        strength: jax.Array,
    ) -> Tuple[State, Params]: ...

    @abstractmethod
    def normalize(
        self,
        key: jax.Array,
        state: State,
        update: Params,
        target_norm: jax.Array,
    ) -> Tuple[State, Params]: ...

    @abstractmethod
    def __call__(
        self, rng: jax.Array, x: X, state: State, params: Params
    ) -> Tuple[Y, State]: ...

    def tare(self, absolute=1, relative=None):
        if relative is not None:
            self.mass *= relative
            for child in self.children:
                child.tare(relative=relative)
        else:
            self.tare(relative=absolute / self.mass)

    # def print_submodules(self):
    #     for child in self.children:
    #         child.print_submodules()

    def __matmul__(self, other):
        return CompositeModule(self, other)

    def __rmatmul__(self, other):
        return other @ self

    def __add__(self, other: int | float | "Module[StateF, ParamsF, X, Y]"):
        if isinstance(other, (int, float)):
            return self @ Add(other)
        else:
            return (self, other) @ Sum()

    def __radd__(self, other: int | float):
        return Add(other) @ self

    def __mul__(self, other: int | float | "Module[StateF, ParamsF, X, Y]"):
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
    Module[CompositeState, CompositeParams, X, Z],
    Generic[StateF, StateG, ParamsF, ParamsG, X, Y, Z],
):
    children: List[Module[StateF, ParamsF, X, Y] | Module[StateG, ParamsG, Y, Z]]

    def __init__(
        self,
        f: Module[StateF, ParamsF, X, Y],
        g: Module[StateG, ParamsG, Y, Z],
    ):
        self.mass = f.mass + g.mass
        self.sensitivity = f.sensitivity * g.sensitivity
        self.length = f.length + g.length
        self.children = [f, g]

    def init(self, key: jax.Array) -> Tuple[CompositeState, CompositeParams]:
        kf, kg = jax.random.split(key)
        sf, pf = self.children[0].init(kf)
        sg, pg = self.children[1].init(kg)
        return (sf, sg), (pf, pg)

    def normalize(
        self,
        key: jax.Array,
        state: CompositeState,
        update: CompositeParams,
        target_norm: jax.Array,
    ):
        if self.mass > 0:
            sf, sg = state
            pf, pg = update
            kf, kg = jax.random.split(key)
            m0, m1 = self.children
            sf, pf = m0.normalize(
                kf,
                sf,
                pf,
                target_norm=m0.mass / self.mass * target_norm / m1.sensitivity,
            )
            sg, pg = m1.normalize(
                kg,
                sg,
                pg,
                target_norm=m1.mass / self.mass * target_norm,
            )
            return (sf, sg), (pf, pg)
        else:
            return state, jax.tree.map(jnp.zeros_like, update)

    def regularize(
        self,
        key: jax.Array,
        state: CompositeState,
        params: CompositeParams,
        strength: jax.Array,
    ) -> Tuple[CompositeState, CompositeParams]:
        kf, kg = jax.random.split(key)
        sf, sg = state
        pf, pg = params
        mf, mg = self.children
        if self.mass > 0:
            sf, pf = mf.regularize(
                kf,
                sf,
                pf,
                strength=mf.mass / self.mass * strength / mg.sensitivity,
            )
            sg, pg = mg.regularize(
                kg,
                sg,
                pg,
                strength=mg.mass / self.mass * strength,
            )
        return (sf, sg), (pf, pg)

    def __call__(
        self,
        rng: jax.Array,
        x: X,
        state: CompositeState,
        params: CompositeParams,
    ) -> Tuple[Z, CompositeState]:
        sf, sg = state
        pf, pg = params
        rf, rg = jax.random.split(rng)
        y, sf = self.children[0](rf, x, sf, pf)
        z, sg = self.children[1](rg, y, sg, pg)
        return z, (sf, sg)


class TupleModule(
    Module[CompositeState, CompositeParams, X, Tuple[Y, Z]],
    Generic[StateF, StateG, ParamsF, ParamsG, X, Y, Z],
):
    def __init__(
        self,
        f: Module[StateF, ParamsF, X, Y],
        g: Module[StateG, ParamsG, X, Z],
    ):
        self.mass = f.mass + g.mass
        self.sensitivity = f.sensitivity * g.sensitivity
        self.length = f.length + g.length
        self.children = [f, g]

    def init(self, key: jax.Array) -> Tuple[CompositeState, CompositeParams]:
        kf, kg = jax.random.split(key)
        sf, pf = self.children[0].init(kf)
        sg, pg = self.children[1].init(kg)
        return (sf, sg), (pf, pg)

    def normalize(
        self,
        key: jax.Array,
        state: CompositeState,
        update: CompositeParams,
        target_norm: jax.Array,
    ):
        if self.mass > 0:
            sf, sg = state
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
            return state, jax.tree.map(jnp.zeros_like, update)

    def regularize(
        self,
        key: jax.Array,
        state: CompositeState,
        params: CompositeParams,
        strength: jax.Array,
    ) -> Tuple[CompositeState, CompositeParams]:
        if self.mass > 0:
            kf, kg = jax.random.split(key)
            sf, sg = state
            pf, pg = params
            mf, mg = self.children
            sf, pf = mf.regularize(kf, sf, pf, strength=strength * mf.mass / self.mass)
            sg, pg = mg.regularize(kg, sg, pg, strength=strength * mg.mass / self.mass)
            return (sf, sg), (pf, pg)
        else:
            return state, params

    def __call__(
        self,
        rng: jax.Array,
        x: X,
        state: CompositeState,
        params: CompositeParams,
    ) -> Tuple[Tuple[Y, Z], CompositeState]:
        sf, sg = state
        pf, pg = params
        rf, rg = jax.random.split(rng)
        y, sf = self.children[0](rf, x, sf, pf)
        z, sg = self.children[1](rg, y, sg, pg)
        return (y, z), (sf, sg)


class Sum(Module[None, None, Tuple[X, ...], X], Generic[X]):
    def __init__(self):
        super().__init__()
        self.mass = 0
        self.sensitivity = 1
        self.length = 0
        self.children = []

    def init(self, key):
        return None, None

    def normalize(self, key, state, update, target_norm):
        return None, None

    def regularize(self, key, state, params, strength):
        return None, None

    def __call__(self, rng, x: Tuple[X, ...], state, params):
        return sum(x), None


class Add(Module[None, None, X, X], Generic[X]):
    def __init__(self, alpha: float):
        self.alpha = alpha
        self.mass = 0
        self.sensitivity = abs(alpha)
        self.length = 0
        self.children = []

    def init(self, key):
        return None, None

    def normalize(self, key, state, update, target_norm):
        return None, None

    def regularize(self, key, state, params, strength):
        return None, None

    def __call__(self, rng, x: X, state, params):
        return jax.tree.map(lambda x: self.alpha + x, x), None


class Mul(Module[None, None, X, X], Generic[X]):
    def __init__(self, alpha: float):
        self.alpha = alpha
        self.mass = 0
        self.sensitivity = abs(alpha)
        self.length = 0
        self.children = []

    def init(self, key):
        return None, None

    def normalize(self, key, state, update, target_norm):
        return None, None

    def regularize(self, key, state, params, strength):
        return None, None

    def __call__(self, rng, x: X, state, params):
        return jax.tree.map(lambda x: self.alpha * x, x), None


class Prod(Module[None, None, Tuple[X, ...], X], Generic[X]):
    def __init__(self):
        self.mass = 0
        self.sensitivity = 1
        self.length = 0
        self.children = []

    def init(self, key):
        return None, None

    def normalize(self, key, state, update, target_norm):
        return None, None

    def regularize(self, key, state, params, strength):
        return None, None

    def __call__(self, rng, x: Tuple[X, ...], state, params):
        assert len(x) > 0
        _acc = x[0]
        for x in x[1:]:
            _acc = jax.tree.map(lambda x, y: x * y, _acc, x)
        return _acc, None


class Pow(Module[State, Params, X, Tuple[X, X]], Generic[State, Params, X]):
    def __init__(
        self,
        depth: int,
        module: Module[State, Params, X, X],
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
        self.share_params = share_params

    def init(self, key: jax.Array) -> Tuple[State, Params]:
        if self.share_params:
            return self.module.init(key)
        else:
            ks = jax.random.split(key, self.depth)
            ss, ps = jax.vmap(self.module.init)(ks)
        return ss, ps

    def regularize(
        self, key: jax.Array, state: State, params: Params, strength: jax.Array
    ) -> Tuple[State, Params]:
        if self.share_params:
            return self.module.regularize(key, state, params, strength / self.depth)
        else:
            ks = jax.random.split(key, self.depth)
            ss, ps = jax.vmap(
                lambda k, s, p: self.module.regularize(k, s, p, strength / self.depth)
            )(ks, state, params)
            return ss, ps

    def normalize(
        self,
        key: jax.Array,
        state: State,
        update: Params,
        target_norm: jax.Array,
    ) -> Tuple[State, Params]:
        if self.share_params:
            return self.module.normalize(key, state, update, target_norm / self.depth)
        else:
            ks = jax.random.split(key, self.depth)
            ss, ps = jax.vmap(
                lambda k, s, p: self.module.normalize(k, s, p, target_norm / self.depth)
            )(ks, state, update)
            return ss, ps

    def __call__(
        self,
        rng: jax.Array,
        x: X,
        state: State,
        params: Params,
    ) -> Tuple[Tuple[X, X], State]:
        ks = jax.random.split(rng, self.depth)

        def share_params_scan_body(x: Tuple[X, State], loop_params: jax.Array):
            x, s = x
            r = loop_params
            x, s = self.module(r, x, s, params)
            return (x, s), x

        def scan_body(x: X, loop_params: Tuple[jax.Array, State, Params]):
            r, s, p = loop_params
            x, s = self.module(r, x, s, p)
            return x, (x, s)

        if self.share_params:
            (y, state_), ys = jax.lax.scan(
                share_params_scan_body,
                (x, state),
                (ks, params),
                self.depth,
            )
        else:
            y, (ys, state_) = jax.lax.scan(
                scan_body,
                x,
                (ks, state, params),
                self.depth,
                reverse=self.reverse,
                unroll=self.unroll,
                _split_transpose=self._split_transpose,
            )

        return (y, ys), state_
