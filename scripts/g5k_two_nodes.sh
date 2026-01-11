#!/usr/bin/env bash
set -euo pipefail

# Get unique node list from OAR
NODES=($(sort -u "$OAR_NODEFILE"))
HEAD="${NODES[0]}"
WORKERS=("${NODES[@]:1}")

echo "HEAD=$HEAD"
echo "WORKERS=${WORKERS[*]}"

# Start Ray head on the first node (this script runs on the first node already)
HEAD_IP=$(hostname -I | awk '{print $1}')
ray stop --force || true
ray start --head --node-ip-address="$HEAD_IP" --port=6379

# Start Ray workers on the other nodes
for w in "${WORKERS[@]}"; do
  ssh "$w" "ray stop --force || true; ray start --address=$HEAD_IP:6379"
done

# Run training connected to the cluster
source .venv/bin/activate
/usr/bin/time -p python train_rllib.py --env CartPole-v1 --num-env-runners 8 --stop-timesteps 200000 --address auto

# Cleanup
for w in "${WORKERS[@]}"; do
  ssh "$w" "ray stop --force" || true
done
ray stop --force || true