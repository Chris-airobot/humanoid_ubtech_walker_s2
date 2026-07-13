#!/usr/bin/env bash
# Export the Walker S2 + Hand3 URDF to a robot-only USD and validate it.
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

exec "$ISAAC_PYTHON" scripts/export_walker_s2_usd.py "$@"
