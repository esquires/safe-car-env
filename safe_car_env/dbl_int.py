import dataclasses
from pathlib import Path
from typing import Any, Optional

import gym
import numpy as np
from safe_car_env.utils import (
    ActuatorLimits,
    MathCalcs,
    NpFloat,
    NumType,
    T,
    TimeTracker,
    get_default_params,
    make_box,
)

InfoDict = dict[str, float]


class A_tilde:
    def __init__(self, a_minus: float, a_plus: Optional[float], calcs: MathCalcs):
        assert a_minus < 0
        if a_plus is not None:
            assert 0 < a_plus
        self.a_minus = a_minus
        self.a_plus = a_plus
        self.calcs = calcs

    def get_a(self, v: T, _type: Optional[NumType] = None) -> T | float:
        if self.a_plus is None:
            return self.a_minus

        v, v_cast = self.calcs.scalar_cast(v, _type)  # type: ignore
        if v_cast:
            return self.a_minus if v >= 0 else self.a_plus
        else:
            m = self.calcs.get_module(v, _type)
            a = m.where(v >= 0, self.a_minus, self.a_plus)
            return a  # type: ignore

    def __repr__(self) -> str:
        return (
            f"A_tilde(a_minus={self.a_minus}, a_plus={self.a_plus}, calcs={self.calcs})"
        )


class DblIntSafeController:
    def __init__(self, dt: float, a_tilde: A_tilde, setpoint: float, calcs: MathCalcs):
        self.dt = dt
        self.a_tilde = a_tilde
        self.setpoint = setpoint
        self.calcs = calcs

    def __repr__(self) -> str:
        return (
            f"DblIntSafeController("
            f"dt={self.dt}, "
            f"a_tilde={self.a_tilde}, "
            f"setpoint={self.setpoint}, "
            f"calcs={self.calcs}"
            ")"
        )

    def get_control(self, state: T, _type: Optional[NumType] = None) -> T:
        v, v_cast = self.calcs.get_element(state, 1, 2, _type)

        if v_cast:
            err = v - self.setpoint
            if err >= 0:
                return max(-err / self.dt, self.a_tilde.a_minus)  # type: ignore
            else:
                return min(-err / self.dt, self.a_tilde.a_plus)  # type: ignore
        else:

            m = self.calcs.get_module(state)

            err = v - self.setpoint

            u = m.where(
                err >= 0,
                self.calcs.clamp(-err / self.dt, low=self.a_tilde.a_minus),
                self.calcs.clamp(-err / self.dt, high=self.a_tilde.a_plus),
            )
            return u


class DblIntDynamics:
    def __init__(self, dt: float, act_lims: ActuatorLimits, calcs: MathCalcs):
        self.dt = dt
        self.act_lims = act_lims
        self.calcs = calcs

    def get_next_state(self, state: T, action: T) -> T:
        p, v = self.calcs.split(state, 2)
        action = self.act_lims.clamp(action)

        action = self.calcs.scalar_cast(action)[0]  # type: ignore

        next_p = p + self.dt * v
        next_v = v + self.dt * action

        return self.calcs.concat([next_p, next_v])  # type: ignore

    def __repr__(self) -> str:
        return f"DblIntDynamics(dt={self.dt}, act_lims={self.act_lims})"


# pylint: disable=too-many-instance-attributes,abstract-method
class DblIntEnv(gym.Env):
    # pylint: disable=too-many-positional-arguments
    def __init__(
        self,
        dynamics: DblIntDynamics,
        init_pos_bounds: tuple[float, float],
        init_vel_bounds: tuple[float, float],
        max_timesteps: int,
        viewer: Optional[Any],
    ):
        self.dynamics = dynamics
        self.init_pos_bounds = init_pos_bounds
        self.init_vel_bounds = init_vel_bounds
        self.time_tracker = TimeTracker(max_timesteps, 0)
        self.viewer = viewer

        self.observation_space = make_box((1.0), 3)
        self.action_space = make_box((1.0), 1)

        self.state = np.array([0.0, 0.0], dtype=np.float32)
        self.closest = np.inf

        self.safe_controller_obs_dim = 2
        self.denormalize = self.dynamics.act_lims.denormalize
        self.normalize = self.dynamics.act_lims.normalize
        self.unsafe_counts: list[bool] = []

        self.rho_low = -1
        self.rho_high = 5
        self.enable_viewer = True

    # pylint: disable=unused-argument
    def reset(self, **kwargs: Any) -> tuple[NpFloat, InfoDict]:
        def _sample() -> NpFloat:
            _unif = np.random.uniform
            return np.hstack(
                (_unif(*self.init_pos_bounds), _unif(*self.init_vel_bounds)),
                dtype=np.float32,
            )

        if self.viewer is not None:
            self.viewer.reset()

        self.time_tracker.num_timesteps = 0

        self.state = _sample()

        self.closest = self.state[0]
        obs, info = self.get_obs()
        return obs, info

    def step(self, action: NpFloat) -> tuple[NpFloat, float, bool, InfoDict]:
        action = np.clip(float(action), -1, 1)

        if self.enable_viewer and self.viewer is not None:
            t_beg_step = self.time_tracker.num_timesteps * self.dynamics.dt
            obs = self.get_obs()[0]
            done = (
                self.time_tracker.num_timesteps + 1
            ) >= self.time_tracker.max_timesteps or self.state[0] < -1
            self.viewer.render(self.state, action, t_beg_step, done, obs)

        self.time_tracker.num_timesteps += 1
        done = (
            self.time_tracker.num_timesteps >= self.time_tracker.max_timesteps
            or self.state[0] < -1
        )

        self.state = np.array(
            self.dynamics.get_next_state(self.state, action), dtype=np.float32
        )

        if self.state[0] < self.closest:
            rew = self.closest - self.state[0]
            self.closest = self.state[0]
        else:
            rew = 0.0

        obs, info = self.get_obs()
        if done:
            info["episode_unsafe"] = self.closest < 0
        return obs, rew, done, info

    def get_obs(self) -> tuple[NpFloat, InfoDict]:
        info = {}
        info["safe_controller_obs"] = self.state.copy()
        info["constraint_violation"] = float(self.closest < 0)  # type: ignore
        info["cost"] = float(self.closest < 0)  # type: ignore
        info["rho"] = self.state[0]

        t = self.time_tracker.num_timesteps * self.dynamics.dt
        return np.hstack((self.state, t), dtype=np.float32), info  # type: ignore


@dataclasses.dataclass
class DblIntVars:
    dynamics: DblIntDynamics
    safe_controller: DblIntSafeController
    env: DblIntEnv
    params: dict[str, Any]
    calcs: MathCalcs


# pylint: disable=too-many-locals
def make(
    use_scalar: bool,
    allow_np_cast: bool,
    allow_gui: bool,
    log_dir: Optional[Path],
    params: Optional[dict[str, Any]],
) -> DblIntVars:

    if params is None:
        params = get_default_params()["dbl-int"]

    calcs = MathCalcs(use_scalar, allow_np_cast)
    act_lims = ActuatorLimits(calcs=calcs, **params["actuator_lims"])

    dt = params["dynamics"]["dt"]
    dynamics = DblIntDynamics(dt, act_lims, calcs)
    a_tilde = A_tilde(calcs=calcs, **params["a_tilde"])
    ctrl = DblIntSafeController(dt, a_tilde, 0.0, calcs)

    if allow_gui and params["do_render"] and log_dir is not None:
        from safe_car_env.dbl_int_viewer import DblIntViewer, DblIntViewerParams
        from safe_car_env.viewer_utils import FrameSaver

        dbl_int_viewer_params = DblIntViewerParams(**params["viewer"])
        frame_saver = FrameSaver(log_dir, 5, 60) if params["save_frames"] else None
        viewer = DblIntViewer(act_lims, dbl_int_viewer_params, frame_saver)
    else:
        viewer = None

    env = DblIntEnv(dynamics, viewer=viewer, **params["env"])

    return DblIntVars(dynamics, ctrl, env, params, calcs)
