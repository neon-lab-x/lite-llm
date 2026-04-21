#!/usr/bin/env bash
# Run prepare_data.py in a loop with auto-restart on crash.
# SIGKILL (OOM, platform kill) cannot be caught by Python, so this wrapper
# uses the script's built-in resume logic to pick up where it left off.
#
# Usage:
#   export HF_ENDPOINT=https://hf-mirror.com
#   export HF_TOKEN=hf_xxx
#   bash scripts/production/run_prepare_loop.sh          # full run
#   bash scripts/production/run_prepare_loop.sh --scale 0.01  # smoke test

set -euo pipefail

SCRIPT="scripts/production/prepare_data.py"
PYTHON="/root/lite-llm/.venv/bin/python"
LOG="prepare_$(date +%Y%m%d_%H%M%S).log"
MAX_RESTARTS=100

echo "=== prepare_data auto-restart loop ==="
echo "  Log: $LOG"
echo "  Max restarts: $MAX_RESTARTS"
echo ""

restart=0
while [ $restart -lt $MAX_RESTARTS ]; do
    restart=$((restart + 1))
    echo ""
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] === Attempt $restart/$MAX_RESTARTS ===" | tee -a "$LOG"

    $PYTHON "$@" "$SCRIPT" 2>&1 | tee -a "$LOG"
    exit_code=${PIPESTATUS[0]}

    if [ $exit_code -eq 0 ]; then
        echo ""
        echo "[$(date '+%Y-%m-%d %H:%M:%S')] Completed successfully." | tee -a "$LOG"
        exit 0
    fi

    echo ""
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] Exit code: $exit_code" | tee -a "$LOG"
    echo "  Restarting in 30s (resume will skip completed shards)..." | tee -a "$LOG"
    sleep 30
done

echo "[$(date '+%Y-%m-%d %H:%M:%S')] Exceeded $MAX_RESTARTS restarts. Giving up." | tee -a "$LOG"
exit 1
