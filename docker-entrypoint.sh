#!/usr/bin/env bash
set -euo pipefail

python -m uvicorn api.main:app --host 0.0.0.0 --port 8000 &
api_pid=$!

python -m streamlit run app/streamlit_app.py \
  --server.address 0.0.0.0 \
  --server.port 8501 \
  --server.headless true &
streamlit_pid=$!

cleanup() {
  kill "$api_pid" "$streamlit_pid" 2>/dev/null || true
}
trap cleanup EXIT INT TERM

wait -n "$api_pid" "$streamlit_pid"
