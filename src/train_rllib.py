# import argparse
# import time

# import ray
# from ray.rllib.algorithms.ppo import PPOConfig


# def main():
#     p = argparse.ArgumentParser()
#     p.add_argument("--env", type=str, default="CartPole-v1")
#     p.add_argument("--stop-timesteps", type=int, default=200_000)
#     p.add_argument("--num-env-runners", type=int, default=4)  # parallel samplers
#     p.add_argument("--num-gpus", type=int, default=0)
#     p.add_argument("--address", type=str, default=None, help="Ray address, e.g. auto")
#     args = p.parse_args()

#     # Connect to Ray cluster if provided, otherwise local Ray runtime.
#     if args.address:
#         ray.init(address=args.address)
#     else:
#         ray.init()

#     config = (
#         PPOConfig()
#         .environment(args.env)
#         # "env_runners" is where you scale sampling parallelism
#         .env_runners(num_env_runners=args.num_env_runners)
#         .resources(num_gpus=args.num_gpus)
#     )

#     algo = config.build()

#     t0 = time.time()
#     total_ts = 0
#     it = 0

#     while total_ts < args.stop_timesteps:
#         result = algo.train()
#         it += 1

#         # RLlib result dict contains timesteps counters (field names can vary by version)
#         # Try these common keys:
#         total_ts = (
#             result.get("num_env_steps_sampled_lifetime")
#             or result.get("num_env_steps_sampled")
#             or result.get("timesteps_total")
#             or total_ts
#         )

#         print(
#             f"iter={it} total_timesteps={total_ts} "
#             f"episode_reward_mean={result.get('episode_reward_mean')}"
#         )

#     dt = time.time() - t0
#     print(f"DONE in {dt:.2f}s -> approx {total_ts/dt:.1f} timesteps/s")

#     algo.stop()
#     ray.shutdown()


# if __name__ == "__main__":
#     main()

import argparse
import ray
from ray.rllib.algorithms.ppo import PPOConfig
import csv
import os
import time

def parse_args():
    parser = argparse.ArgumentParser(description="RLlib PPO Training")

    parser.add_argument(
        "--env",
        type=str,
        default="CartPole-v1",
        help="Gymnasium environment name"
    )

    parser.add_argument(
        "--num-env-runners",
        type=int,
        default=4,
        help="Number of parallel rollout workers"
    )

    parser.add_argument(
        "--iterations",
        type=int,
        default=20,
        help="Number of training iterations"
    )

    parser.add_argument(
        "--use-gpu",
        action="store_true",
        help="Enable GPU training if available"
    )

    return parser.parse_args()


def main():
    args = parse_args()

    # Connect to Ray
    ray.init(ignore_reinit_error=True)

    config = (
        PPOConfig()
        .environment(args.env)
        .framework("torch")
        .env_runners(num_env_runners=args.num_env_runners)
        .resources(num_gpus=1 if args.use_gpu else 0)

    )

    algo = config.build()
    os.makedirs("results", exist_ok=True)

    with open("results/rewards.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["iteration", "episode_reward_mean", "iteration_time_sec"])

        for i in range(args.iterations):
            start = time.time()
            result = algo.train()
            reward = result["env_runners"]["episode_return_mean"]
            duration = time.time() - start

            writer.writerow([i + 1, reward, duration])
            
            print(
            f"Iter {i+1} | "
            f"Reward: {reward:.2f} | "
            f"Time: {duration:.2f}s"
            )

    algo.stop()
    ray.shutdown()


if __name__ == "__main__":
    main()
