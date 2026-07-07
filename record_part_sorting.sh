#!/usr/bin/env bash
# Record Part Sorting episodes with the Walker S2 Isaac Sim robot.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

if [[ -x /isaac-sim/python.sh ]]; then
  ISAAC_PYTHON=(/isaac-sim/python.sh)
elif [[ -x /home/chris/isaacsim/python.sh ]]; then
  ISAAC_PYTHON=(/home/chris/isaacsim/python.sh)
else
  echo "Isaac Sim python.sh not found (tried /isaac-sim and ~/isaacsim)." >&2
  exit 1
fi

RESOURCES="${SCRIPT_DIR}/assets/resources"
mkdir -p "$RESOURCES"
if [[ ! -e "${RESOURCES}/WalkerS2-Model" && -d "${SCRIPT_DIR}/../WalkerS2-Model" ]]; then
  ln -sf "../../../WalkerS2-Model" "${RESOURCES}/WalkerS2-Model"
fi

# Writable pip cache (avoids /root/.cache permission warnings in Docker).
export PIP_CACHE_DIR="${SCRIPT_DIR}/.cache/pip"
mkdir -p "$PIP_CACHE_DIR"

export PYTHONPATH="${SCRIPT_DIR}${PYTHONPATH:+:$PYTHONPATH}"

if [[ -d /workspace/WalkerS2-Model ]]; then
  export ZOLLENT_REPO_ROOT=/workspace
elif [[ -d "${SCRIPT_DIR}/../WalkerS2-Model" ]]; then
  export ZOLLENT_REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
fi

URDF_REL="WalkerS2-Model/walker_s2_description_hand3_v1_left_hand3_v1_right/walker_s2_description_hand3_v1_left_hand3_v1_right.urdf"
URDF_CANDIDATES=(
  "${ZOLLENT_REPO_ROOT:-}/${URDF_REL}"
  "${SCRIPT_DIR}/assets/resources/${URDF_REL}"
)
URDF_FOUND=""
for candidate in "${URDF_CANDIDATES[@]}"; do
  if [[ -f "$candidate" ]]; then
    URDF_FOUND="$candidate"
    break
  fi
done
if [[ -z "$URDF_FOUND" ]]; then
  echo "Missing URDF: ${URDF_REL}" >&2
  echo "Extract the official combined URDF zip under WalkerS2-Model/." >&2
  echo "If using Docker, restart the container via ./run.sh so WalkerS2-Model is mounted." >&2
  exit 1
fi

# Refresh editable install only when project metadata changed (skip slow pip on every run).
EDITABLE_STAMP="${SCRIPT_DIR}/.cache/editable-install.stamp"
mkdir -p "${SCRIPT_DIR}/.cache"
if [[ ! -f "$EDITABLE_STAMP" ]] \
   || [[ pyproject.toml -nt "$EDITABLE_STAMP" ]] \
   || [[ setup.py -nt "$EDITABLE_STAMP" ]]; then
  "${ISAAC_PYTHON[@]}" -m pip install -e . --no-deps -q
  touch "$EDITABLE_STAMP"
fi

DATASET_ROOT="${DATASET_ROOT:-${SCRIPT_DIR}/datasets/part_sorting_walker_s2}"
DATASET_REPO_ID="${DATASET_REPO_ID:-local/part_sorting_walker_s2}"
NUM_EPISODES="${NUM_EPISODES:-5}"
EPISODE_TIME_S="${EPISODE_TIME_S:-60}"
RESET_TIME_S="${RESET_TIME_S:-10}"
FPS="${FPS:-20}"

exec "${ISAAC_PYTHON[@]}" -m lerobot.scripts.lerobot_record \
  --robot.type=walker_s2_sim \
  --robot.headless=false \
  --teleop.type=walker_s2_keyboard \
  --task=Part_Sorting \
  --dataset.root="$DATASET_ROOT" \
  --dataset.repo_id="$DATASET_REPO_ID" \
  --dataset.num_episodes="$NUM_EPISODES" \
  --dataset.episode_time_s="$EPISODE_TIME_S" \
  --dataset.reset_time_s="$RESET_TIME_S" \
  --dataset.fps="$FPS" \
  --dataset.single_task="Part_Sorting" \
  --dataset.video=true \
  --dataset.push_to_hub=false \
  --dataset.streaming_encoding=true \
  --dataset.encoder_threads=2 \
  --display_data=false \
  "$@"
