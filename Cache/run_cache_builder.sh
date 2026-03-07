#!/bin/bash
# run_cache.sh — restarts build_cache.py automatically after rate limit boots
# Stops when a run completes with zero failures (exit code 0)

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CACHE_SCRIPT="$SCRIPT_DIR/build_cache.py"

echo "$(date) — cache runner started"
echo "Script: $CACHE_SCRIPT"
echo ""

while true; do
    echo "$(date) — starting cache build..."
    python "$CACHE_SCRIPT"
    EXIT_CODE=$?

    if [ $EXIT_CODE -eq 0 ]; then
        echo ""
        echo "$(date) — cache build completed with no failures. Done."
        break
    else
        echo ""
        echo "$(date) — failures detected (exit code $EXIT_CODE), restarting in 60s..."
        sleep 60
    fi
done