#!/usr/bin/env bash
set -euo pipefail

# 1) Get unique node list from OAR
NODES=($(sort -u "$OAR_NODEFILE"))
HEAD="${NODES[0]}"
WORKERS=("${NODES[@]:1}")
ENV_NAME="${ENV_NAME:-CartPole-v1}"
ITERS="${ITERS:-20}"
ENV_RUNNERS="${ENV_RUNNERS:-8}"

echo "HEAD=$HEAD"
echo "WORKERS=${WORKERS[*]}"

source "$HOME/.venvs/rllib/bin/activate"

# 2) Start Ray head on the first node
HEAD_IP=$(hostname -I | awk '{print $1}')
ray stop --force || true
ray start --head --node-ip-address="$HEAD_IP" --port=6379

# 3) Start Ray workers on the other nodes
for w in "${WORKERS[@]}"; do
  ssh "$w" "source $HOME/.venvs/rllib/bin/activate && ray stop --force || true && ray start --address=$HEAD_IP:6379"
done

# 4) Run training connected to the cluster
python src/train_rllib.py \
  --env "$ENV_NAME" \
  --iterations "$ITERS" \
  --num-env-runners "$ENV_RUNNERS"

# 5) Cleanup
for w in "${WORKERS[@]}"; do
  ssh "$w" "ray stop --force" || true
done
ray stop --force || true