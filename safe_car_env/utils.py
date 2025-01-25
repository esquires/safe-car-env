import collections
import dataclasses
import enum
import math
import numbers
from pathlib import Path
from types import ModuleType
from typing import Any, Optional, Protocol, Sequence, TypeVar

import gym
import numpy as np
import numpy.typing as npt
import torch
import yaml
from torch import Tensor

NpFloat = npt.NDArray[np.float32]
NpInt = npt.NDArray[np.int32]
NpAny = npt.NDArray[Any]

T = TypeVar("T", NpAny, Tensor, float, int)
TL = TypeVar("TL", NpAny, Tensor, list[numbers.Number])
TF = TypeVar("TF", NpFloat, Tensor, float)


def interp(x: T, x_low: Any, x_high: Any, y_low: Any, y_high: Any) -> T:
    # copied from rlgear.utils
    # https://github.com/esquires/rlgear/blob/master/rlgear/utils.py#L858
    if x_low == x_high:
        return y_low  # type: ignore
    else:
        pct = (x - x_low) / (x_high - x_low)
        return y_low + pct * (y_high - y_low)  # type: ignore


class NumType(enum.Enum):
    SCALAR = 0
    NUMPY = 1
    TORCH = 2


class MathCalcs:
    def __init__(self, use_scalar: bool, allow_np_cast: bool):
        self.use_scalar = use_scalar
        self.allow_np_cast = allow_np_cast

    def __repr__(self) -> str:
        return f"MathCalcs(use_scalar={self.use_scalar}, cast_np={self.allow_np_cast})"

    @staticmethod
    def get_type(v: T) -> NumType:
        if isinstance(v, Tensor):
            return NumType.TORCH
        elif isinstance(v, numbers.Number):
            return NumType.SCALAR
        else:
            return NumType.NUMPY

    def np_cast(self, val: T) -> tuple[T | NpAny, bool]:
        if (
            self.allow_np_cast
            and isinstance(val, Tensor)
            and val.shape[0] == 1
            and not val.requires_grad
        ):
            # for this particular case it is faster to work with numpy arrays.
            return val.numpy().squeeze(0), True
        else:
            return val, False

    def scalar_cast(
        self, val: T, _type: Optional[NumType] = None
    ) -> tuple[Tensor | NpAny | float, bool]:
        if _type is None:
            _type = self.get_type(val)
        if _type == NumType.SCALAR:
            return val, True

        elif isinstance(val, np.ndarray) and val.shape[0] == 1:
            return float(val[0]), True
        else:
            return val, False

    def get_module(
        self, val: T, _type: Optional[NumType] = None
    ) -> ModuleType:
        if _type is None:
            _type = self.get_type(val)
        match _type:  # NOQA
            case NumType.SCALAR:
                return math
            case NumType.NUMPY:
                return np
            case NumType.TORCH:
                return torch

    def clamp(
        self,
        val: TL,
        low: Optional[T | float] = None,
        high: Optional[T | float] = None,
        _type: Optional[NumType] = None,
    ) -> TL:
        if _type is None:
            _type = self.get_type(val)
        if _type == NumType.SCALAR:
            if low is None:
                assert high is not None
                return min(val, high)
            elif high is None:
                assert low is not None
                return max(val, low)
            else:
                return max(min(val, high), low)
        else:
            m = self.get_module(val)
            return m.clip(val, low, high)

    def min(self, vals: TL) -> TL:
        _type = self.get_type(vals[0])
        match _type:
            case NumType.SCALAR:
                return min(vals)
            case NumType.TORCH:
                return torch.min(vals, dim=1, keepdim=True).values
            case _:
                return np.min(vals)

    def max(self, vals: TL) -> TL:
        _type = self.get_type(vals[0])
        match _type:
            case NumType.SCALAR:
                return max(vals)
            case NumType.TORCH:
                return torch.max(vals, dim=1, keepdim=True).values
            case _:
                return np.max(vals)

    def abs(self, val: TL, _type: Optional[NumType] = None) -> TL:
        if _type is None:
            _type = self.get_type(val)

        if _type == NumType.SCALAR:
            return abs(val)
        elif _type == NumType.TORCH:
            return torch.abs(val)  # type: ignore
        else:
            return np.abs(val)  # type: ignore

    def sign(self, val: T, _type: Optional[NumType] = None) -> T:
        if _type is None:
            _type = self.get_type(val)

        if _type == NumType.SCALAR:
            if val > 0:
                return 1.0  # type: ignore
            elif val < 0:
                return -1.0  # type: ignore
            else:
                return 0.0  # type: ignore
        elif _type == NumType.NUMPY:
            return np.sign(val)  # type: ignore
        else:
            return torch.sign(val)  # type: ignore

    def symlog(self, val: T, _type: Optional[NumType] = None) -> T:
        # see dreamer v3
        # https://arxiv.org/abs/2301.04104
        if _type is None:
            _type = self.get_type(val)

        m = self.get_module(val, _type)
        return self.sign(val, _type) * m.log(self.abs(val, _type) + 1)

    def split(self, val: T, n: int, N: Optional[int] = None) -> tuple[T, ...]:
        if N is None:
            N = n

        if isinstance(val, Tensor):
            assert val.shape[1] == N, f"val has dim {val.shape[1]} but expected {N}"
            return torch.chunk(val, n, dim=1)
        elif self.use_scalar and n == N:
            return val
        elif isinstance(val, np.ndarray):
            assert val.shape[0] == N, f"val has dim {val.shape[0]} but expected {N}"
            return np.split(val, n)  # type: ignore
        else:
            # e.g. a list
            return val

    def concat(self, vals: list[T]) -> T:
        _type = self.get_type(vals[0])
        match _type:
            case NumType.SCALAR:
                return vals
            case NumType.TORCH:
                return torch.cat(vals, dim=1)
            case _:
                return np.hstack(vals)

    def get_element(
        self, state_or_val: T, idx: int, n: int, _type: Optional[NumType] = None
    ) -> tuple[T, bool]:
        scalar, cast_success = self.scalar_cast(state_or_val, _type)
        if cast_success:
            return scalar, True
        elif self.use_scalar and not isinstance(state_or_val, Tensor):
            return state_or_val[idx], True
        elif isinstance(state_or_val, Tensor) and state_or_val.shape[1] == 1:
            return state_or_val, False
        else:
            return self.split(state_or_val, n)[idx], False


class ActuatorLimits:
    def __init__(self, low: npt.ArrayLike, high: npt.ArrayLike, calcs: MathCalcs):
        self.low = np.asarray(low, dtype=np.float32)
        self.high = np.asarray(high, dtype=np.float32)
        self.normalized_low = -1
        self.normalized_high = 1
        self.calcs = calcs

    def __repr__(self) -> str:
        return f"ActuatorLimits(low={self.low}, high={self.high}, calcs={self.calcs})"

    def normalize(self, action: T) -> T:
        low, high = self._get_lims(action)
        pct = (action - low) / (high - low)
        return self.normalized_low + pct * (self.normalized_high - self.normalized_low)

    def get_idx(self, idx: int) -> NpFloat:
        return np.array([self.low[idx], self.high[idx]])

    def set_idx(self, idx: int, val: Sequence[float]) -> None:
        self.low[idx] = val[0]
        self.high[idx] = val[1]

    def denormalize(self, action: T) -> T:
        low, high = self._get_lims(action)
        pct = (action - self.normalized_low) / (
            self.normalized_high - self.normalized_low
        )
        pct = (action - (-1)) / 2
        return low + pct * (high - low)

    def _get_lims(self, action: T) -> tuple[T, T]:
        if isinstance(action, Tensor):
            low = torch.from_numpy(self.low)
            high = torch.from_numpy(self.high)

            if action.device.type == "cuda":
                low = low.cuda()
                high = high.cuda()
            return low, high

        else:
            return self.low, self.high

    def clamp(self, action: T) -> T:
        low, high = self._get_lims(action)
        return self.calcs.clamp(action, low, high)


class EpisodeStats:
    def __init__(self) -> None:
        self.stats: dict[str, list[float]] = collections.defaultdict(list)

    def reset(self) -> None:
        self.stats.clear()

    def update(self, episode_stats: dict[str, float]) -> None:
        for key, val in episode_stats.items():
            self.stats[key].append(val)

    def get_mean(self) -> dict[str, float]:
        return {k: float(np.mean(v)) for k, v in self.stats.items()}


def angle_wrap(ang_rad: T) -> T:

    # this is valid for NpFloat and Tensor provided that they are a single element
    while ang_rad > np.pi:
        ang_rad -= 2 * np.pi

    while ang_rad < -np.pi:
        ang_rad += 2 * np.pi

    return ang_rad


class Dynamics(Protocol):
    def get_next_state(self, state: T, action: T) -> T: ...


class BarrierFunction(Protocol):
    def get_h(self, state: T) -> T: ...


class BarrierConstraint:
    def __init__(
        self,
        lmbda: float,
        dynamics: Dynamics,
        barrier_function: BarrierFunction,
        calcs: MathCalcs,
    ):
        self.lmbda = lmbda
        self.dynamics = dynamics
        self.barrier_function = barrier_function
        self.calcs = calcs

    def get_constraint(self, state: T, action: TF, h: Optional[T] = None) -> T:
        state, state_to_np = self.calcs.np_cast(state)
        action, action_to_np = self.calcs.np_cast(action)
        assert (state_to_np and action_to_np) or (not state_to_np and not action_to_np)
        if h is None:
            h = self.barrier_function.get_h(state)

        if state_to_np:
            h = self.calcs.scalar_cast(h, NumType.NUMPY)[0]

        next_state = self.dynamics.get_next_state(state, action)
        next_h = self.barrier_function.get_h(next_state)
        constraint = next_h - (1 - self.lmbda) * h

        if state_to_np:
            h = torch.tensor([[h]], dtype=torch.float32)
        # print(h, action, next_state, next_h, constraint)
        return constraint


def get_default_params() -> dict[str, Any]:
    yaml_file = Path(__file__).parent / "params.yaml"

    with open(yaml_file, "r", encoding="UTF-8") as f:
        params = yaml.safe_load(f)
    return params  # type: ignore


@dataclasses.dataclass
class TimeTracker:
    max_timesteps: int
    num_timesteps: int


def make_box(scale: float, n: int) -> gym.spaces.Box:
    lim = (scale * np.ones(n)).astype(np.float32)
    return gym.spaces.Box(-lim, lim)
