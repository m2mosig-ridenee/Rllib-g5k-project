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
