#!/usr/bin/env bash
set -euo pipefail

# Configs
REPO_URL="https://github.com/m2mosig-ridenee/rllib-g5k-project.git"
REPO_DIR="$HOME/rllib-g5k-project"
VENV_DIR="$HOME/.venvs/rllib"
ENV_NAME="${ENV_NAME:-CartPole-v1}"
ITERS="${ITERS:-20}"
ENV_RUNNERS="${ENV_RUNNERS:-4}"

echo "Node: $(hostname)"
echo "Env: $ENV_NAME"
echo "Iterations: $ITERS"
echo "Env runners: $ENV_RUNNERS"

# 1) Create venv if needed
if [ ! -d "$VENV_DIR" ]; then
  python3 -m venv "$VENV_DIR"
fi

source "$VENV_DIR/bin/activate"

# 2) Install deps
pip install -U pip
pip install "ray[rllib]" torch gymnasium pydantic

# 3) Get code
if [ ! -d "$REPO_DIR/.git" ]; then
  git clone "$REPO_URL" "$REPO_DIR"
else
  git -C "$REPO_DIR" pull --rebase
fi

cd "$REPO_DIR"

# 4) Run single-node training
echo "Running single-node training..."
python src/train_rllib.py \
  --env "$ENV_NAME" \
  --iterations "$ITERS" \
  --num-env-runners "$ENV_RUNNERS"