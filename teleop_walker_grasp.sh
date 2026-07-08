#!/usr/bin/env bash
# Launch the standalone Walker S2 fixed-grasp teleop demo.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

if [[ -n "${ISAAC_SIM_PYTHON:-}" ]]; then
  ISAAC_PYTHON="$ISAAC_SIM_PYTHON"
elif [[ -x /isaac-sim/python.sh ]]; then
  ISAAC_PYTHON=/isaac-sim/python.sh
elif [[ -x "${HOME}/isaacsim/python.sh" ]]; then
  ISAAC_PYTHON="${HOME}/isaacsim/python.sh"
else
  echo "Isaac Sim python.sh was not found." >&2
  echo "Set ISAAC_SIM_PYTHON=/path/to/isaacsim/python.sh." >&2
  exit 1
fi

DEFAULT_URDF="${SCRIPT_DIR}/assets/resources/walker_s2_description_hand3_v1_left_hand3_v1_right/walker_s2_description_hand3_v1_left_hand3_v1_right.urdf"
URDF_PATH="${WALKER_S2_URDF:-$DEFAULT_URDF}"

if [[ ! -f "$URDF_PATH" ]]; then
  echo "Walker S2 URDF was not found: $URDF_PATH" >&2
  echo "Set WALKER_S2_URDF=/absolute/path/to/the/robot.urdf." >&2
  exit 1
fi

exec "$ISAAC_PYTHON" scripts/isaac_walker_fixed_grasp_demo.py \
  --urdf "$URDF_PATH" \
  "$@"
