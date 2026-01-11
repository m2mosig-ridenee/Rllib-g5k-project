# RLlib on Grid'5000 — Parallel Reinforcement Learning (1 node vs 2 nodes)

**Course:** Parallel Programming for the AI Era (2025–2026)

**Student:** Eya Ridene (individual work)

**Deadline:** 12/01/2026

**Repository:** https://github.com/m2mosig-ridenee/Rllib-g5k-project

## 1) Project goal

The objective of this project is to deploy and evaluate parallel reinforcement learning using **Ray RLlib** on the **Grid'5000** platform.

We compare:

- **Single-node** execution
- **Two-node** distributed execution using a Ray cluster (head + worker)

The comparison focuses on:

- **Learning behavior** (episode reward evolution)
- **Wall-clock training time** per iteration
- **Impact of distributed rollout collection**

## 2) Reinforcement learning tasks

### Main environment

- **Library:** Gymnasium
- **Environment:** CartPole-v1
- **Algorithm:** PPO (Proximal Policy Optimization)

<!-- > CartPole-v1 is lightweight and converges quickly. It is therefore useful to highlight the limits of scalability when computation is small compared to communication overhead. -->

### Additional environment testing (for more scalability analysis)

- Pendulum-v1 (continuous control, heavier computation)

These environments better illustrate the benefits of parallel execution.

## 3) Repository structure

```
├── README.md
├── report.md
├── requirements.txt
├── process_illustration/        # Snapshots of commands/execution
├── results/
│   ├── *.csv                    # Training metrics (episode rewards, iteration times)
│   └── *.png                    # Generated plots (episode reward visualizations)
├── scripts/
│   ├── g5k_single_node.sh       # Script for single-node execution on Grid'5000
│   └── g5k_two_nodes.sh         # Script for two-node distributed execution
└── src/
    ├── train_rllib.py           # Training script with timing and CSV logging
    └── evaluate.py              # Offline evaluation and plotting from CSV

```

## 4) Accessing Grid'5000

```bash
ssh <login>@access.grid5000.fr
```

Then connect to Grenoble site frontend :

```bash
ssh grenoble
```

## 5) Environment setup (on compute nodes)

### 5.1 Create a Python virtual environment

```bash
python3 -m venv ~/rllib-venv
source ~/rllib-venv/bin/activate

pip install --upgrade pip
pip install "ray[rllib]" torch gymnasium pydantic
```

Clone the repository if needed:

```bash
git clone https://github.com/m2mosig-ridenee/Rllib-g5k-project
cd Rllib-g5k-project
```

## 6) Single-node experiment

### 6.1 Reserve one node (interactive job)

```bash
oarsub -I -l nodes=1,walltime=02:00:00 --project lab-2025-uga-mosig-hpcai
```

You are connected to a compute node.

### 6.2 Run training

```bash
source ~/rllib-venv/bin/activate
cd ~/Rllib-g5k-project

python src/train_rllib.py \
  --env CartPole-v1 \
  --num-env-runners 4 \
  --iterations 20
```

The script logs:

- episode reward mean
- wall-clock time per iteration

Results are stored in for later analysis:

- `results/rewards.csv`

## 7) Two-node distributed experiment (Ray cluster)

### 7.1 Reserve two nodes

```bash
oarsub -I -l nodes=2,walltime=02:00:00 --project lab-2025-uga-mosig-hpcai
```

### 7.2 Identify nodes

```bash
sort $OAR_NODEFILE | uniq
```

- **HEAD:** current node
- **WORKER:** the other node

### 7.3 Start Ray head (on HEAD)

```bash
source ~/rllib-venv/bin/activate
ray stop || true
ray start --head --port=6379
```

### 7.4 Start Ray worker (on WORKER)

```bash
ssh <WORKER_HOSTNAME>
source ~/rllib-venv/bin/activate
ray stop || true
ray start --address="<HEAD_HOSTNAME>:6379"
exit
```

### 7.5 Run training on the cluster (from HEAD only)

```bash
source ~/rllib-venv/bin/activate
cd ~/Rllib-g5k-project

python src/train_rllib.py \
  --env CartPole-v1 \
  --num-env-runners 8 \
  --iterations 20
```

### 7.6 Stop Ray

Stop on worker:

```bash
ssh "$WORKER"    # or oarsh "$WORKER"
ray stop
exit
```

Stop on head:

```bash
ray stop
```

## 8) Plotting results

Plots are generated offline from CSV logs using a non-interactive backend.

```bash
python src/evaluate.py
```

This produces:

- `results/episode_reward_plot.png`

## 9) Performance analysis approach

- Learning curves are compared qualitatively
- Average iteration time is compared quantitatively
- Speedup is discussed with respect to workload size

<!-- **Observed behavior:**

- CartPole-v1 shows limited or no speedup due to overhead
- Pendulum-v1 benefits from distributed execution -->

## 10) Alternative way to run (automation scripts)

Instead of typing all commands manually, the experiments can be launched using **bash scripts** (e.g., `*.sh`) that:

- create/activate the Python virtual environment
- install dependencies
- start/stop Ray (for multi-node)
- run the same `train_rllib.py` command

This method is useful for:

- reproducibility (same steps each time)
- avoiding copy/paste mistakes
- running multiple experiments consistently

The repository includes scripts in the `scripts/` directory:

- `g5k_single_node.sh` — automates the single-node experiment
- `g5k_two_nodes.sh` — automates the two-node Ray cluster setup and training

These scripts can be submitted directly via `oarsub` for batch execution:

```bash
oarsub -l nodes=1,walltime=02:00:00 --project lab-2025-uga-mosig-hpcai scripts/g5k_single_node.sh
oarsub -l nodes=2,walltime=02:00:00 --project lab-2025-uga-mosig-hpcai scripts/g5k_two_nodes.sh
```

---

## 11) Notes on reproducibility

- Reinforcement learning is stochastic
- Different runs may produce different learning curves
- Trends and averages are used rather than single-run conclusions

---

## 10) References

- Ray RLlib documentation (PPO + scaling via EnvRunners)
- Gymnasium environments documentation
- Grid'5000 "Getting Started" guide