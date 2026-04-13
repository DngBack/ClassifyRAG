#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
APP_FILE="$ROOT_DIR/streamlit_app.py"

if [[ ! -f "$APP_FILE" ]]; then
  echo "Missing app file: $APP_FILE" >&2
  exit 1
fi

choose_port() {
  for p in $(seq 7860 7870); do
    if ! ss -ltn "( sport = :$p )" 2>/dev/null | awk 'NR>1 {found=1} END {exit found ? 0 : 1}'; then
      echo "$p"
      return 0
    fi
  done
  return 1
}

is_port_free() {
  local p="$1"
  ! ss -ltn "( sport = :$p )" 2>/dev/null | awk 'NR>1 {found=1} END {exit found ? 0 : 1}'
}

PORT="${1:-}"
if [[ -z "$PORT" ]]; then
  if ! PORT="$(choose_port)"; then
    echo "No free port in range 7860-7870" >&2
    exit 1
  fi
else
  if ! is_port_free "$PORT"; then
    echo "Port $PORT is busy, searching free port in 7860-7870..." >&2
    if ! PORT="$(choose_port)"; then
      echo "No free port in range 7860-7870" >&2
      exit 1
    fi
  fi
fi

echo "Starting Streamlit on port $PORT"
exec streamlit run "$APP_FILE" --server.port "$PORT" --server.address 0.0.0.0
