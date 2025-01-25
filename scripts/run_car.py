import argparse
from pathlib import Path

import numpy as np
import unitpy
from safe_car_env.car import make
from safe_car_env.dbl_int import DblIntSafeController
from safe_car_env.utils import NpFloat, get_default_params


class Controller:
    def __init__(self, vel_ctrl: DblIntSafeController):
        self.vel_ctrl = vel_ctrl

    def get_control(self, obs: NpFloat) -> NpFloat:
        # double integrator is just position and velocity
        # and the controller ignores position so give it a
        # state of (0, vel)
        _, _, state_c = self.vel_ctrl.calcs.split(obs, 3, 12)
        v_c = state_c[2]
        dbl_int_state = np.array([0, v_c])
        u1 = self.vel_ctrl.get_control(dbl_int_state)

        # heading control
        u2 = 0.0
        return np.array([u1, u2])


def main() -> None:

    parser = argparse.ArgumentParser()
    parser.add_argument("--tgt_speed", required=True, type=float, help="miles per hour")
    args = parser.parse_args()

    log_dir = Path("/tmp/car")

    params = get_default_params()["car"]
    params["env"]["do_render"] = True
    env_vars = make(
        use_scalar=True,  # optimization for floats, np arrays and torch tensors
        allow_np_cast=True,  # optimization for floats, np arrays and torch tensors
        allow_gui=True,
        log_dir=log_dir,  # this is where videos will be output if a gui is used
        params=params,
    )

    env = env_vars.env
    ctrl = Controller(env_vars.ctrl_dbl_int)
    tgt_speed_meters_per_sec = (
        (args.tgt_speed * unitpy.Unit("mile per hour")).to("m/s").value
    )
    ctrl.vel_ctrl.setpoint = tgt_speed_meters_per_sec

    # typically in RL the observation is a normalized version
    # of the state. A non-RL controller does not need this
    # normalized state so rather than denormalizing so that the
    # non-RL controller can run, the environment also provides
    # the unnormalized observation in the info dict with
    # key "safe_controller_obs"
    info = env.reset()[1]
    obs = info["safe_controller_obs"]

    while True:
        action = ctrl.get_control(obs)

        # the environment expects a normalized action between -1 and 1
        action = env.normalize(action)

        done, info = env.step(action)[2:]
        obs = info["safe_controller_obs"]

        if done:
            info = env.reset()[1]
            obs = info["safe_controller_obs"]


if __name__ == "__main__":
    main()
