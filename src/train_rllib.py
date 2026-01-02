import argparse
import time

import ray
from ray.rllib.algorithms.ppo import PPOConfig


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--env", type=str, default="CartPole-v1")
    p.add_argument("--stop-timesteps", type=int, default=200_000)
    p.add_argument("--num-env-runners", type=int, default=4)  # parallel samplers
    p.add_argument("--num-gpus", type=int, default=0)
    p.add_argument("--address", type=str, default=None, help="Ray address, e.g. auto")
    args = p.parse_args()

    # Connect to Ray cluster if provided, otherwise local Ray runtime.
    if args.address:
        ray.init(address=args.address)
    else:
        ray.init()

    config = (
        PPOConfig()
        .environment(args.env)
        # "env_runners" is where you scale sampling parallelism
        .env_runners(num_env_runners=args.num_env_runners)
        .resources(num_gpus=args.num_gpus)
    )

    algo = config.build()

    t0 = time.time()
    total_ts = 0
    it = 0

    while total_ts < args.stop_timesteps:
        result = algo.train()
        it += 1

        # RLlib result dict contains timesteps counters (field names can vary by version)
        # Try these common keys:
        total_ts = (
            result.get("num_env_steps_sampled_lifetime")
            or result.get("num_env_steps_sampled")
            or result.get("timesteps_total")
            or total_ts
        )

        print(
            f"iter={it} total_timesteps={total_ts} "
            f"episode_reward_mean={result.get('episode_reward_mean')}"
        )

    dt = time.time() - t0
    print(f"DONE in {dt:.2f}s -> approx {total_ts/dt:.1f} timesteps/s")

    algo.stop()
    ray.shutdown()


if __name__ == "__main__":
    main()