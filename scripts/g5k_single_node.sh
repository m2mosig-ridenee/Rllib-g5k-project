#!/usr/bin/env bash
set -euo pipefail

# ====== CONFIG: change this to your repo URL ======
REPO_URL="https://github.com/m2mosig-ridenee/rllib-g5k-project.git"
REPO_DIR="$HOME/rllib-g5k-project"
VENV_DIR="$HOME/.venvs/rllib"

ITERS="${ITERS:-30}"
ENV_RUNNERS="${ENV_RUNNERS:-4}"

echo "Node: $(hostname)"
echo "Using venv: $VENV_DIR"
echo "Repo: $REPO_DIR"
echo

# 1) Create venv if needed
if [ ! -d "$VENV_DIR" ]; then
  python3 -m venv "$VENV_DIR"
fi
# shellcheck disable=SC1090
source "$VENV_DIR/bin/activate"

# 2) Install deps (safe to re-run)
pip install -U pip
pip install "ray[rllib]" torch gymnasium

# 3) Get code
if [ ! -d "$REPO_DIR/.git" ]; then
  git clone "$REPO_URL" "$REPO_DIR"
else
  git -C "$REPO_DIR" pull --rebase
fi

cd "$REPO_DIR"

# 4) Run single-node training
echo "Running single-node training..."
python src/train_rllib.py --iters "$ITERS" --num-env-runners "$ENV_RUNNERS"