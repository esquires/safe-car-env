# pylint: disable=cyclic-import
import dataclasses
import enum
import math
from pathlib import Path
from typing import Any, Callable, Generic, Literal, Optional, Self, Type

import gym
import numpy as np
import torch
import unitpy
from safe_car_env.dbl_int import A_tilde, DblIntSafeController
from safe_car_env.utils import (
    TF,
    ActuatorLimits,
    EpisodeStats,
    MathCalcs,
    NpFloat,
    NumType,
    T,
    TimeTracker,
    get_default_params,
    make_box,
)

InfoDict = dict[str, float]


def mph_to_mps(mph: float) -> float:
    return (mph * unitpy.Unit("mile per hour")).to("m/s").value


@dataclasses.dataclass
class CarParams:
    width: float
    length_rear: float
    length_front: float

    def get_beta(self, u2: TF, calcs: MathCalcs, _type: Optional[NumType] = None) -> TF:
        lr = self.length_rear
        lf = self.length_front

        if _type is None:
            _type = calcs.get_type(_type)
        _tan, _atan = self._get_funcs(_type)

        beta = _atan(_tan(u2) * lr / (lf + lr))
        return beta

    def get_beta_inv(
        self, val: T, calcs: MathCalcs, _type: Optional[NumType] = None
    ) -> T:
        lr = self.length_rear
        lf = self.length_front

        if _type is None:
            _type = calcs.get_type(_type)
        _tan, _atan = self._get_funcs(_type)
        beta_inv = _tan(_atan(((lf + lr) / lr) * val))
        return beta_inv

    @staticmethod
    def _get_funcs(_type: NumType) -> tuple[Callable, Callable]:
        if _type == NumType.SCALAR:
            return math.tan, math.atan
        elif _type == NumType.NUMPY:
            return np.tan, np.arctan
        else:
            return torch.tan, torch.arctan


@dataclasses.dataclass
class SafetySpec:
    D_lead: float
    tau: float
    v_lim: float


@dataclasses.dataclass
class AllVehState:
    a: NpFloat
    b: NpFloat
    c: NpFloat


@dataclasses.dataclass
class SafetySummary:  # pylint: disable=too-many-instance-attributes
    offset: float
    in_lane1: bool
    in_lane2: bool
    off_road: bool
    dist_off_road: float

    dist_a: float
    dist_b: float
    dist_thresh: float

    too_close_a: bool
    too_close_b: bool

    crash_a: bool
    crash_b: bool

    too_fast: bool

    violation: bool

    @classmethod
    def from_states(  # pylint: disable=too-many-locals,too-many-positional-arguments
        cls: Type[Self],
        spec: SafetySpec,
        params: CarParams,
        W_lane: float,
        states: AllVehState,
        calcs: MathCalcs,
    ) -> "SafetySummary":
        x_c, y_c, v_c, psi_c = states.c

        offset = get_car_offset(params, psi_c, calcs)
        in_lane1 = y_c - offset <= W_lane
        in_lane2 = y_c + offset >= W_lane
        off_road = y_c - offset < 0 or y_c + offset > 2 * W_lane
        dist_off_road = min(y_c - offset, 2 * W_lane - y_c - offset)

        dist_a = states.a[0] - x_c
        dist_b = states.b[0] - x_c
        dist_thresh = spec.D_lead + spec.tau * calcs.clamp(v_c, low=0.0)

        too_close_a = in_lane1 and dist_a < dist_thresh
        too_close_b = in_lane2 and dist_b < dist_thresh

        crash_a = in_lane1 and dist_a < spec.D_lead
        crash_b = in_lane2 and dist_b < spec.D_lead

        too_fast = v_c > spec.v_lim

        violation = off_road or too_close_a or too_close_b or too_fast

        return cls(
            offset=offset,
            in_lane1=in_lane1,
            in_lane2=in_lane2,
            off_road=off_road,
            dist_off_road=dist_off_road,
            dist_a=dist_a,
            dist_b=dist_b,
            dist_thresh=dist_thresh,
            too_close_a=too_close_a,
            too_close_b=too_close_b,
            crash_a=crash_a,
            crash_b=crash_b,
            too_fast=too_fast,
            violation=violation,
        )


class CarDynamics:
    def __init__(
        self,
        dt: float,
        car_params: CarParams,
        act_lims: ActuatorLimits,
        calcs: MathCalcs,
    ):
        self.dt = dt
        self.car_params = car_params
        self.act_lims = act_lims
        self.calcs = calcs

    def __repr__(self) -> str:
        return (
            "CarDynamics("
            f"dt={self.dt}, "
            f"car_params={self.car_params}, "
            f"act_lims={self.act_lims}, "
            f"calcs={self.calcs}"
            ")"
        )

    # pylint: disable=too-many-locals
    def get_next_state(self, state: T, action: Optional[T]) -> T:
        """
        model is from equation (2 a-e) of
        Polack, Philip, et al. "The kinematic bicycle model: A
        consistent model for planning feasible trajectories for
        autonomous vehicles?." 2017 IEEE intelligent vehicles
        symposium (IV). IEEE, 2017.
        https://ieeexplore.ieee.org/abstract/document/7995816
        """

        x, y, v, psi = self.calcs.split(state, 4)
        action = self.act_lims.clamp(action)
        u1, u2 = self.calcs.split(action, 2)

        _type = self.calcs.get_type(x)
        _sin, _cos = self.get_trig_funcs(_type)

        beta_u2 = self.car_params.get_beta(u2, self.calcs, _type)

        next_x = self.get_next_x(x, v, psi, beta_u2, _cos)
        next_y = self.get_next_y(y, v, psi, beta_u2, _sin)
        next_v = self.get_next_v(v, u1)
        next_psi = self.get_next_psi(v, psi, beta_u2, _sin)

        next_state = self.calcs.concat([next_x, next_y, next_v, next_psi])
        return next_state

    @staticmethod
    def get_trig_funcs(_type: NumType) -> tuple[Callable, Callable]:
        if _type == NumType.SCALAR:
            return math.sin, math.cos
        elif _type == NumType.TORCH:
            return torch.sin, torch.cos
        else:
            return np.sin, np.cos

    def get_next_x(self, x: T, v: T, psi: T, beta_u2: TF, _cos: Callable) -> T:
        return x + self.dt * v * _cos(psi + beta_u2)

    def get_next_y(self, y: T, v: T, psi: T, beta_u2: TF, _sin: Callable) -> T:
        return y + self.dt * v * _sin(psi + beta_u2)

    def get_next_v(self, v: T, u1: T) -> T:
        return v + self.dt * u1

    def get_next_psi(self, v: T, psi: T, beta_u2: TF, _sin: Callable) -> T:
        lr = self.car_params.length_rear
        return psi + self.dt * (v / lr) * _sin(beta_u2)


def get_car_offset(
    car_params: CarParams, psi: T, calcs: MathCalcs, _type: Optional[NumType] = None
) -> T:
    L = car_params.length_front
    W = car_params.width

    if _type is None:
        _type = calcs.get_type(psi)

    if _type == NumType.SCALAR:
        _abs, _sin, _cos = abs, math.sin, math.cos
    elif _type == NumType.NUMPY:
        _abs, _sin, _cos = np.abs, np.sin, np.cos
    else:
        _abs, _sin, _cos = torch.abs, torch.sin, torch.cos

    offset = L * _abs(_sin(psi)) + (W / 2) * _abs(_cos(psi))
    return offset


class LaneRho(Generic[T]):
    def __init__(self, L1: T, H1: T, L2: T, H2: T):
        self.L1: T = L1
        self.H1: T = H1
        self.L2: T = L2
        self.H2: T = H2

    def __repr__(self) -> str:
        calcs = MathCalcs(True, True)
        out = calcs.concat([self.L1, self.H1, self.L2, self.H2])
        return out.__repr__()

    def __getitem__(self, idx: int) -> T:
        match idx:  # NOQA
            case 0:
                return self.L1
            case 1:
                return self.H1
            case 2:
                return self.L2
            case 3:
                return self.H2
            case _:
                raise IndexError("only indexes 0-3 are supported")

    @classmethod
    def make(  # pylint: disable=too-many-positional-arguments
        cls,
        y: T,
        psi: T,
        W_lane: float,
        car_params: CarParams,
        calcs: MathCalcs,
        _type: Optional[NumType] = None,
    ) -> Self:
        offset = get_car_offset(car_params, psi, calcs, _type)

        rho_L1 = y - offset
        rho_H1 = W_lane - offset - y
        rho_L2 = y - offset - W_lane
        rho_H2 = 2 * W_lane - offset - y

        return cls(rho_L1, rho_H1, rho_L2, rho_H2)


class LeadCarMode(enum.Enum):
    BRAKE = 0
    CRUISE = 1
    ACCELERATE = 2


class LeadCar:
    def __init__(  # pylint: disable=too-many-positional-arguments
        self,
        lane_idx: Literal["a", "b"],
        ctrl: DblIntSafeController,
        W_lane: float,
        speed_range_mph: tuple[float, float],
        cruise_interval: float,
        start_x_lims: tuple[float, float],
        calcs: MathCalcs,
    ):
        self.lane_idx = lane_idx
        self.W_lane = W_lane

        self.speed_range_mph = np.array(speed_range_mph)
        self.speed_range_mps = (
            (self.speed_range_mph * unitpy.Unit("mile per hour")).to("m/s").value
        )
        self.cruise_interval = cruise_interval
        self.start_x_lims = start_x_lims
        self.calcs = calcs

        self.ctrl = ctrl
        self.mode = LeadCarMode.CRUISE
        self.time_end_cruise = np.inf

    def __repr__(self) -> str:
        return (
            "LeadCar("
            f"lane_idx={self.lane_idx}, "
            f"ctrl={self.ctrl}, "
            f"W_lane={self.W_lane}, "
            f"speed_range_mph={self.speed_range_mph}, "
            f"cruise_interval={self.cruise_interval}, "
            f"start_x_lims={self.start_x_lims}, "
            f"calcs={self.calcs}"
            ")"
        )

    def reset(self, state: NpFloat) -> None:
        state[0] = np.random.uniform(*self.start_x_lims)
        state[1] = self.W_lane * (0.5 if self.lane_idx == "a" else 1.5)
        state[2] = np.random.uniform(*self.speed_range_mps)

        self.mode = LeadCarMode.CRUISE
        self.ctrl.setpoint = state[2]
        self.time_end_cruise = np.random.uniform(*self.cruise_interval)

    def step(self, state: NpFloat, x_follow: float, t: float) -> NpFloat:
        x, _, v, _ = state
        vel_close = abs(v - self.ctrl.setpoint) < 1

        if x < x_follow:
            # print('spawning')
            self.reset(state)
            state[0] += x_follow

        if self.mode == LeadCarMode.CRUISE and t >= self.time_end_cruise:
            self.mode = LeadCarMode.BRAKE
            self.ctrl.setpoint = self.ctrl.setpoint * np.random.uniform(0, 1)

        elif self.mode == LeadCarMode.BRAKE and vel_close:
            self.mode = LeadCarMode.ACCELERATE
            self.ctrl.setpoint = np.random.uniform(*self.speed_range_mps)

        elif self.mode == LeadCarMode.ACCELERATE and vel_close:
            self.mode = LeadCarMode.CRUISE
            self.time_end_cruise = t + np.random.uniform(*self.cruise_interval)

        action = self.ctrl.get_control(v, NumType.SCALAR)
        return action


# pylint: disable=too-many-instance-attributes,abstract-method
class CarEnv(gym.Env):

    # pylint: disable=too-many-positional-arguments
    def __init__(
        self,
        car_dynamics: CarDynamics,
        lead_car_a: LeadCar,
        lead_car_b: LeadCar,
        W_lane: float,
        safety_spec: SafetySpec,
        speed_tgt_mph: float,
        max_timesteps: int,
        viewer: Optional["CarEnvViewer"],  # NOQA
        calcs: MathCalcs,
    ):
        self.car_dynamics = car_dynamics
        self.lead_car_a = lead_car_a
        self.lead_car_b = lead_car_b
        self.W_lane = W_lane
        self.safety_spec = safety_spec
        self.speed_tgt_mph = speed_tgt_mph
        self.speed_tgt = (speed_tgt_mph * unitpy.Unit("mile per hour")).to("m/s").value
        self.viewer = viewer
        self.calcs = calcs

        self.episode_stats = EpisodeStats()

        self.states = AllVehState(*[np.zeros(4, dtype=np.float32) for _ in range(3)])

        self.time_tracker = TimeTracker(max_timesteps, 0)

        self.observation_space = make_box((1.0), 8)
        self.action_space = make_box((1.0), 2)

        self.safe_controller_obs_dim = 12
        self.denormalize = self.car_dynamics.act_lims.denormalize
        self.normalize = self.car_dynamics.act_lims.normalize
        self.unsafe_counts: list[bool] = []
        self.speed_deviations: list[float] = []

        self.rho_low = -0.1
        self.rho_high = 0.5

    # pylint: disable=unused-argument
    def reset(self, **kwargs: Any) -> tuple[NpFloat, InfoDict]:

        if self.viewer is not None:
            self.viewer.reset()

        self.lead_car_a.reset(self.states.a)
        self.lead_car_b.reset(self.states.b)

        offset = self.car_dynamics.car_params.width
        eps = 1e-3 * self.W_lane  # to avoid floating point errors
        y_c = np.random.uniform(offset + eps, 2 * self.W_lane - offset - eps)
        self.states.c[:] = [0.0, y_c, 0.95 * self.speed_tgt, 0]

        self.time_tracker.num_timesteps = 0
        self.speed_deviations = []

        safety_summary = SafetySummary.from_states(
            self.safety_spec,
            self.car_dynamics.car_params,
            self.W_lane,
            self.states,
            self.calcs,
        )
        info = self._update_constraint_info(safety_summary, {})
        info["rho"] = self.get_rho(safety_summary)
        self.unsafe_counts = [info["constraint_violation"]]

        obs = self.get_obs()
        self.tot_rew = 0.0

        # self.psi_lims = []

        return obs, info

    def get_rho(self, safety_summary: SafetySummary) -> float:
        rho_off_road = safety_summary.dist_off_road / self.W_lane
        v_c = self.states.c[2]
        rho_speed = (self.safety_spec.v_lim - v_c) / 10

        rel_dist_a = safety_summary.dist_a - safety_summary.dist_thresh
        rel_dist_b = safety_summary.dist_b - safety_summary.dist_thresh
        rho_dist = min(rel_dist_a, rel_dist_b) / safety_summary.dist_thresh

        rho = min(rho_off_road, rho_speed, rho_dist)
        return rho

    def step(self, action: NpFloat) -> tuple[NpFloat, float, bool, InfoDict]:
        action = np.clip(action, -1, 1)
        orig_action = action.copy()
        action = self.denormalize(action)
        self.time_tracker.num_timesteps += 1
        dynamics = self.car_dynamics
        t = self.time_tracker.num_timesteps * dynamics.dt
        # print(self.t, action, self.state_c[2])

        # state_a, state_b, state_c = split(self.all_veh_state, 3, 12)
        x_c = self.states.c[0]
        # print('psi: ', np.rad2deg(self.states.c[3]))
        action_a = np.hstack((self.lead_car_a.step(self.states.a, x_c, t), 0))
        action_b = np.hstack((self.lead_car_b.step(self.states.b, x_c, t), 0))

        # all_action = np.hstack((action_a, action_b, action))
        self.states.a = dynamics.get_next_state(self.states.a, action_a)
        self.states.b = dynamics.get_next_state(self.states.b, action_b)

        # don't allow negative speeds since assumedly we aren't allowing the
        # car to change gears
        min_allowed_acc = max(dynamics.act_lims.low[0], -self.states.c[2] / dynamics.dt)
        action[0] = max(min_allowed_acc, action[0])

        self.states.c = dynamics.get_next_state(self.states.c, action)
        self.speed_deviations.append(abs(self.speed_tgt - self.states.c[2]))
        # self.psi_lims.append(abs(np.rad2deg(self.states.c[3])))

        safety_summary = SafetySummary.from_states(
            self.safety_spec,
            self.car_dynamics.car_params,
            self.W_lane,
            self.states,
            self.calcs,
        )

        done, info = self.get_done(safety_summary)
        rew = self.get_rew(action[0])
        self.tot_rew += rew

        if done:
            # print(self.tot_rew)
            # print(max(self.psi_lims))
            info["pct_unsafe"] = float(np.mean(self.unsafe_counts))
            info["speed_deviation"] = float(np.mean(self.speed_deviations))

            info.update({f"episode_{k}": v for k, v in info.items()})
            self.episode_stats.update(info)

        obs = self.get_obs()
        info = self._update_constraint_info(safety_summary, info)
        info["rho"] = self.get_rho(safety_summary)

        if self.viewer is not None:
            self.viewer.render(
                self.states, orig_action, safety_summary, t, done, info, obs
            )

        return obs, rew, done, info

    def get_rew(self, acc: float) -> float:
        v_c = self.states.c[2]
        speed_deviation = abs(v_c - self.speed_tgt)
        rew = 1 - speed_deviation / self.speed_tgt
        return float(rew)

    def get_done(
        self, safety_summary: SafetySummary
    ) -> tuple[bool, InfoDict]:
        info = {}
        exit_time = self.time_tracker.num_timesteps >= self.time_tracker.max_timesteps
        crash = safety_summary.crash_a or safety_summary.crash_b
        done = crash or safety_summary.off_road or exit_time

        if done:
            info["exit_crash"] = float(crash)
            info["exit_off_road"] = float(safety_summary.off_road)
            info["exit_t_thresh"] = float(exit_time)

        return done, info

    def _update_constraint_info(
        self, safety_summary: SafetySummary, info: InfoDict
    ) -> InfoDict:
        info["safe_controller_obs"] = np.hstack(
            (self.states.a, self.states.b, self.states.c), dtype=np.float32
        )
        info["constraint_violation"] = float(safety_summary.violation)
        info["cost"] = float(safety_summary.violation)

        return info

    def get_obs(self) -> NpFloat:
        # state_a, state_b, state_c = split(self.all_veh_state, 3, 12)
        x_c, y_c, v_c, psi_c = self.states.c

        dist_a = self.states.a[0] - x_c
        dist_b = self.states.b[0] - x_c
        start_x = self.lead_car_a.start_x_lims[1]

        def _normalize_dist(_d: float) -> float:
            _half = start_x / 2
            return (_d - _half) / _half

        obs = np.array(
            [
                _normalize_dist(dist_a),
                _normalize_dist(dist_b),
                (y_c - self.W_lane) / self.W_lane,
                (self.states.a[2] - self.speed_tgt) / self.speed_tgt,
                (self.states.b[2] - self.speed_tgt) / self.speed_tgt,
                (v_c - self.speed_tgt) / self.speed_tgt,
                np.sin(psi_c),
                np.cos(psi_c),
            ],
            dtype=np.float32,
        )

        return obs


@dataclasses.dataclass
class CarVars:
    car_dynamics: CarDynamics
    ctrl_dbl_int: DblIntSafeController
    safety_spec: SafetySpec
    env: CarEnv
    params: dict[str, Any]
    calcs: MathCalcs


# pylint: disable=too-many-locals
def make(
    use_scalar: bool,
    allow_np_cast: bool,
    allow_gui: bool,
    log_dir: Optional[Path],
    params: Optional[dict[str, Any]],
) -> CarVars:

    if params is None:
        params = get_default_params()["car"]

    calcs = MathCalcs(use_scalar, allow_np_cast)
    car_params = CarParams(**params["car_params"])

    act_lims = ActuatorLimits(calcs=calcs, **params["actuator_lims"])
    act_lims.set_idx(1, np.deg2rad(act_lims.get_idx(1)))

    dt = params["dynamics"]["dt"]
    car_dynamics = CarDynamics(dt, car_params, act_lims, calcs)

    safety_spec_dict = params["safety_spec"].copy()
    safety_spec_dict["v_lim"] = (
        (safety_spec_dict["v_lim_miles_per_hour"] * unitpy.Unit("mile per hour"))
        .to("m/s")
        .value
    )
    del safety_spec_dict["v_lim_miles_per_hour"]
    safety_spec = SafetySpec(**safety_spec_dict)
    a_tilde = A_tilde(calcs=calcs, **params["a_tilde"])
    assert a_tilde.a_minus >= act_lims.low[0] and a_tilde.a_plus <= act_lims.high[0]

    W_lane = params["W_lane"]

    ctrl_dbl_int = DblIntSafeController(dt, a_tilde, setpoint=0, calcs=calcs)

    def _make_lead_car(
        _lane_idx: Literal["a", "b"]
    ) -> tuple[DblIntSafeController, LeadCar]:
        _u1_min = act_lims.low[0]
        _u1_max = act_lims.high[0]
        _ctrl = DblIntSafeController(dt, A_tilde(_u1_min, _u1_max, calcs), 0, calcs)
        return _ctrl, LeadCar(
            _lane_idx, _ctrl, W_lane, calcs=calcs, **params["env"]["lead_veh"]
        )

    lead_ctrl_a, lead_car_a = _make_lead_car("a")  # pylint: disable=unused-variable
    lead_ctrl_b, lead_car_b = _make_lead_car("b")  # pylint: disable=unused-variable

    if allow_gui and params["env"]["do_render"]:
        from safe_car_env.car_viewer import (
            CarEnvViewer,
            CarEnvViewerParams,
            LegendParams,
        )
        from safe_car_env.viewer_utils import FrameSaver

        legend_params = LegendParams(**params["legend_params"])
        car_env_viewer_params = CarEnvViewerParams(
            legend_params=legend_params,
            **params["viewer"]
        )
        frame_saver = (
            FrameSaver(log_dir, 5, 60) if params["env"]["save_frames"] else None
        )
        viewer = CarEnvViewer(
            W_lane,
            safety_spec,
            lead_car_a.start_x_lims[1],
            act_lims,
            car_env_viewer_params,
            frame_saver,
        )
    else:
        viewer = None

    env = CarEnv(
        car_dynamics=car_dynamics,
        lead_car_a=lead_car_a,
        lead_car_b=lead_car_b,
        W_lane=W_lane,
        safety_spec=safety_spec,
        speed_tgt_mph=params["env"]["speed_tgt_mph"],
        max_timesteps=params["env"]["max_timesteps"],
        viewer=viewer,
        calcs=calcs,
    )

    return CarVars(car_dynamics, ctrl_dbl_int, safety_spec, env, params, calcs)
