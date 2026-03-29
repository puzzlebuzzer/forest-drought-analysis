#!/bin/bash
# run_cache_builder.sh
#
# Runs build_sentinel_cache.py and restarts it immediately on any 403.
# Restart gets a fresh STAC connection + fresh signed URLs, which is the
# only reliable way to push past Planetary Computer rate limits.
#
# Progress is preserved across restarts — the script skips already-cached
# scenes on each run via the manifest.
#
# Usage (from project root):
#   bash Cache/run_cache_builder.sh                          # north, suffix _3_24
#   bash Cache/run_cache_builder.sh --aoi south
#   bash Cache/run_cache_builder.sh --aoi north --cache-suffix _3_24
#
# Logs restart events to Cache/restart_log_<aoi>.txt

set -euo pipefail

# ── Defaults ──────────────────────────────────────────────────────────────────
AOI="north"
CACHE_SUFFIX="_3_24"

# ── Parse args ────────────────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
    case "$1" in
        --aoi)           AOI="$2";          shift 2 ;;
        --cache-suffix)  CACHE_SUFFIX="$2"; shift 2 ;;
        *) echo "Unknown arg: $1"; exit 1 ;;
    esac
done

# ── Paths ─────────────────────────────────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"

if [ "$AOI" = "north" ]; then
    CACHE_NAME="GWNF_cache${CACHE_SUFFIX}"
    PROGRESS_FILE="$REPO_ROOT/../AOI/NorthAOI/${CACHE_NAME}/s2/indices/build_progress_north.txt"
else
    CACHE_NAME="Smoky_cache${CACHE_SUFFIX}"
    PROGRESS_FILE="$REPO_ROOT/../AOI/SouthAOI/${CACHE_NAME}/s2/indices/build_progress_south.txt"
fi

LOG_FILE="/tmp/cache_restart_log_${AOI}.txt"

echo "========================================"    | tee -a "$LOG_FILE"
echo "$(date '+%Y-%m-%d %H:%M:%S') — runner started" | tee -a "$LOG_FILE"
echo "AOI: $AOI  |  suffix: $CACHE_SUFFIX"        | tee -a "$LOG_FILE"
echo "Progress file: $PROGRESS_FILE"              | tee -a "$LOG_FILE"
echo "========================================"    | tee -a "$LOG_FILE"

ATTEMPT=0

while true; do
    ATTEMPT=$((ATTEMPT + 1))
    echo "" | tee -a "$LOG_FILE"
    echo "$(date '+%Y-%m-%d %H:%M:%S') — run #${ATTEMPT} starting..." | tee -a "$LOG_FILE"

    # Clear any stale 403 status from a previous run so we don't false-trigger
    if [ -f "$PROGRESS_FILE" ]; then
        sed -i 's/sleeping (403)[^\n]*/checking.../g' "$PROGRESS_FILE" 2>/dev/null || true
    fi

    # Start the build script in the background (unbuffered output)
    PYTHONUNBUFFERED=1 python "$SCRIPT_DIR/build_sentinel_cache.py" \
        --aoi "$AOI" --cache-suffix "$CACHE_SUFFIX" &
    BUILD_PID=$!

    KILLED=0

    # Poll progress file every 20s while script is running
    while kill -0 "$BUILD_PID" 2>/dev/null; do
        sleep 20
        if [ -f "$PROGRESS_FILE" ] && grep -q "sleeping (403)" "$PROGRESS_FILE" 2>/dev/null; then
            echo "$(date '+%Y-%m-%d %H:%M:%S') — 403 detected, killing PID $BUILD_PID and restarting..." \
                | tee -a "$LOG_FILE"
            kill "$BUILD_PID" 2>/dev/null
            wait "$BUILD_PID" 2>/dev/null || true
            KILLED=1
            break
        fi
    done

    if [ "$KILLED" -eq 0 ]; then
        # Script exited on its own — check exit code
        wait "$BUILD_PID"
        EXIT_CODE=$?
        if [ "$EXIT_CODE" -eq 0 ]; then
            echo "$(date '+%Y-%m-%d %H:%M:%S') — completed successfully. Done." \
                | tee -a "$LOG_FILE"
            break
        else
            echo "$(date '+%Y-%m-%d %H:%M:%S') — exited with code $EXIT_CODE, restarting in 30s..." \
                | tee -a "$LOG_FILE"
            sleep 30
        fi
    else
        # Brief pause so PC isn't hammered immediately
        echo "$(date '+%Y-%m-%d %H:%M:%S') — waiting 15s before restart..." | tee -a "$LOG_FILE"
        sleep 15
    fi
done
