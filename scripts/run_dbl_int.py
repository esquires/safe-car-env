import argparse
from pathlib import Path

from safe_car_env.dbl_int import make
from safe_car_env.utils import get_default_params


def main() -> None:

    parser = argparse.ArgumentParser()
    parser.add_argument("--tgt_speed", required=True, type=float, help="meters per sec")
    args = parser.parse_args()

    log_dir = Path("/tmp/dbl-int")

    params = get_default_params()["dbl-int"]
    params["do_render"] = True
    env_vars = make(
        use_scalar=True,  # optimization for floats, np arrays and torch tensors
        allow_np_cast=True,  # optimization for floats, np arrays and torch tensors
        allow_gui=True,
        log_dir=log_dir,  # this is where videos will be output if a gui is used
        params=params,
    )

    env = env_vars.env
    ctrl = env_vars.safe_controller
    ctrl.setpoint = args.tgt_speed

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
