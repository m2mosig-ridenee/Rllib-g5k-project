# Report — RLlib Experiments on Grid'5000 (Single-node vs Multi-node)

**Student:** Eya Ridene

**Execution platform:** Grid'5000 (Grenoble – dahu cluster)

**RL library:** Ray RLlib

**Algorithm:** PPO (Proximal Policy Optimization)

**Environment:** CartPole-v1  / Pendulum-v1

---

## 1) Goal

The objective of this work is to demonstrate **distributed RL training execution** with Ray RLlib on Grid'5000 and to evaluate the **performance impact of scaling across nodes**.

We run the **same RL training workload** twice:

- **1 node (single-machine execution)**  

  Ray is started locally on the compute node. All RLlib components (driver + sampling actors) run on the same physical machine.

- **2 nodes (distributed execution using a Ray cluster)**  

  We start a **Ray head** process on one node and a **Ray worker** process on the second node, then run RLlib while connecting to the cluster. RLlib distributes its **sampling actors** (EnvRunners) across both nodes.

We focus on:

- Using an existing RLlib algorithm (PPO)
- Training on Gymnasium environments
- Varying the number of parallel environment runners
- Comparing execution on 1 node vs 2 nodes

The main question addressed is:

> **Does increasing parallelism and distributing sampling across multiple nodes reduce training time and improve execution performance?**

---

## 2) Implementation overview

Two Python scripts were developed:

### 2.1 Training script (train_rllib.py)

This script trains a PPO agent using RLlib.

**Key characteristics:**

- Configurable environment (default: CartPole-v1)
- Configurable parallelism via `--num-env-runners`
- Optional GPU usage (`--use-gpu`)
- Fixed number of training iterations (`--iterations`)
- Automatic Ray initialization, compatible with both local and cluster execution
- Logging of results to a CSV file

At each training iteration, the script records:

- Mean episode reward
- Duration of the training iteration (seconds)

These metrics are saved to:

- `results/rewards.csv`

This design separates training from analysis, enabling reproducible evaluation.

### 2.2 Evaluation and visualization (evaluate.py)

This script performs offline evaluation:

- Reads `results/rewards.csv`
- Extracts:
  - Training iteration index
  - Mean episode reward
- Generates a reward-vs-iteration plot
- Saves the figure as:
  - `results/episode_reward_plot.png`

The script uses a headless plotting backend, making it compatible with remote cluster execution.
---

## 3) Experiment setup

### 3.1 Common settings

The following settings are shared across all experiments:

- **Algorithm:** PPO (RLlib)
- **Framework:** PyTorch
- **Training length:** Fixed number of iterations
- **Metrics collected:**
  - Mean episode reward
  - Iteration execution time

The only intended differences between experiments are:

- Number of compute nodes (1 vs 2)
- Number of environment runners (`num_env_runners`)
- Environment used (CartPole-v1, Pendulum-v1)

### 3.2 Single-node execution

- One Grid'5000 compute node is reserved
- Ray is started locally on that node
- All RLlib components (driver + environment runners) run on the same machine

**Example command:**

```bash
python train_rllib.py --env CartPole-v1 --num-env-runners 4 --iterations 20
```

In this configuration:

- Parallelism is limited to the CPUs of a single node
- Environment runners execute locally

### 3.3 Two-node execution (Ray cluster)

We reserve two compute nodes. We start:

- a **Ray head** process on the first node (cluster coordinator)
- a **Ray worker** process on the second node (adds CPU resources to the cluster)

Then the RLlib training script is launched on the head node while connecting to the cluster.

**Example command:**

```bash
python train_rllib.py --env CartPole-v1 --num-env-runners 8 --iterations 20
```

In this configuration:

- Environment runners are distributed across both nodes
- Increased parallelism is used to exploit additional CPU resources
- The training code itself remains unchanged; only the execution context differs.

---

## 4) Results

Experiments were conducted using PPO with RLlib on two Gymnasium environments: CartPole-v1 and Pendulum-v1.

For each environment, training was executed on a single node and on two nodes using a Ray cluster.

At each training iteration, we recorded the mean episode reward and the iteration execution time.

### 4.1 CartPole-v1

#### Learning behavior

For both single-node and two-node configurations, the PPO agent successfully learns the CartPole task.

In the single-node setup, the mean episode reward increases steadily from low initial values and reaches near-optimal performance (reward close to 500) after approximately 15–20 training iterations.

In the two-node setup, learning follows a similar trend; however, higher rewards are often reached in earlier iterations.

This difference does not indicate improved learning quality per se, but rather results from increased sampling parallelism. With more environment runners, each training iteration processes a larger number of environment interactions, allowing the agent to observe more data per iteration.

#### Performance comparison

Iteration execution time is consistently lower in the two-node configuration.

| Setup | Nodes | num_env_runners | Avg. iteration time (s) |
|-------|-------|-----------------|-------------------------|
| Single node | 1 | 4 | ~12–13 |
| Two nodes | 2 | 8 | ~10–11 |

This reduction in iteration duration demonstrates that distributed execution improves training efficiency, even for a lightweight environment such as CartPole.

**Plot placeholders:**

<img src="results/cartpole_1node.png" alt="CartPole 1 node Results" width="400"/>
<img src="results/cartpole_2nodes.png" alt="CartPole 2 nodes Results" width="400"/>

### 4.2 Pendulum-v1

#### Learning behavior

Pendulum-v1 is a continuous-control environment with negative rewards.

In both configurations, rewards fluctuate between approximately –1150 and –950 throughout training. A slight improvement trend is observed, but no clear convergence is reached within the selected number of training iterations.

Learning behavior is similar between the single-node and two-node executions, indicating that distributed sampling does not alter the underlying learning dynamics of PPO for this task.

#### Performance comparison

Iteration execution time shows a clear improvement when using two nodes.

| Setup | Nodes | num_env_runners | Avg. iteration time (s) |
|-------|-------|-----------------|-------------------------|
| Single node | 1 | 4 | ~14–15 |
| Two nodes | 2 | 8 | ~10–11 |

The speedup is more pronounced than for CartPole, which can be attributed to the higher computational cost of the Pendulum environment.

**Plot placeholders:**

<img src="results/pendulum_1node.png" alt="Pendulum 1 node Results" width="400"/>
<img src="results/pendulum_2nodes.png" alt="Pendulum 2 nodes Results" width="400"/>

### 4.3 Discussion and summary

Across both environments, distributed execution leads to reduced iteration execution time without negatively impacting learning behavior.

While the two-node configuration often reaches higher rewards in fewer training iterations, this effect is explained by a higher number of environment interactions per iteration rather than improved sample efficiency.

Performance gains remain modest for CartPole due to its lightweight nature, whereas Pendulum benefits more clearly from distributed sampling. These results confirm that RLlib effectively exploits parallel and distributed execution on Grid'5000, with benefits depending on environment complexity.

---

## 5) Conclusion

This work demonstrated the use of Ray RLlib for running reinforcement learning experiments on the Grid’5000 cluster. Using PPO on CartPole-v1 and Pendulum-v1, we compared single-node and two-node executions.

The results show that distributing environment sampling across multiple nodes reduces training iteration time, while preserving similar learning behavior and final performance. Performance gains are modest for lightweight environments and more noticeable for heavier ones, highlighting the practical trade-offs of distributed reinforcement learning.