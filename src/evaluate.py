import csv
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")

iterations = []
rewards = []

with open("results/rewards.csv", "r") as f:
    reader = csv.DictReader(f)
    for row in reader:
        iterations.append(int(row["iteration"]))
        rewards.append(float(row["episode_reward_mean"]))

plt.figure(figsize=(8, 5))
plt.plot(iterations, rewards, marker="o")
plt.xlabel("Training Iteration")
plt.ylabel("Episode Reward Mean")
plt.title("RLlib PPO Training")
plt.grid(True)
plt.tight_layout()
plt.savefig("results/episode_reward_plot.png")