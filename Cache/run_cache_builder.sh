#!/bin/bash
#
# run_cache_builder.sh
#
# Re-runs a cache builder whenever it exits non-zero. This is intended to pair
# with the cache builders' fail-fast 403 behavior so a fresh Python process can
# resume from the manifests after a Planetary Computer rate-limit stop.
#
# Usage from repo root:
#   bash Python/Cache/run_cache_builder.sh sentinel --aoi south
#   bash Python/Cache/run_cache_builder.sh landsat --aoi north --indices NDVI NDMI
#
# Exit behavior:
#   - exit code 0 from the Python builder: stop normally
#   - any non-zero exit code: wait briefly and re-run

set -euo pipefail

if [[ $# -lt 1 ]]; then
    echo "Usage: bash Python/Cache/run_cache_builder.sh <sentinel|landsat> [builder args...]"
    exit 1
fi

BUILDER_KIND="$1"
shift

case "$BUILDER_KIND" in
    sentinel)
        BUILDER_SCRIPT="build_sentinel_cache.py"
        ;;
    landsat)
        BUILDER_SCRIPT="build_landsat_cache.py"
        ;;
    *)
        echo "Unknown builder '$BUILDER_KIND'. Use 'sentinel' or 'landsat'."
        exit 1
        ;;
esac

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"
LOG_FILE="/tmp/cache_builder_runner_${BUILDER_KIND}.log"
RESTART_SLEEP=20
ATTEMPT=0

echo "========================================" | tee -a "$LOG_FILE"
echo "$(date '+%Y-%m-%d %H:%M:%S') — runner started" | tee -a "$LOG_FILE"
echo "Builder: $BUILDER_KIND" | tee -a "$LOG_FILE"
echo "Script:  $SCRIPT_DIR/$BUILDER_SCRIPT" | tee -a "$LOG_FILE"
echo "Args:    $*" | tee -a "$LOG_FILE"
echo "========================================" | tee -a "$LOG_FILE"

while true; do
    ATTEMPT=$((ATTEMPT + 1))
    echo "" | tee -a "$LOG_FILE"
    echo "$(date '+%Y-%m-%d %H:%M:%S') — run #$ATTEMPT starting" | tee -a "$LOG_FILE"

    set +e
    (
        cd "$REPO_ROOT"
        PYTHONUNBUFFERED=1 python "./Cache/$BUILDER_SCRIPT" "$@"
    )
    EXIT_CODE=$?
    set -e

    if [[ "$EXIT_CODE" -eq 0 ]]; then
        echo "$(date '+%Y-%m-%d %H:%M:%S') — builder exited cleanly; runner stopping" | tee -a "$LOG_FILE"
        exit 0
    fi

    echo "$(date '+%Y-%m-%d %H:%M:%S') — builder exited with code $EXIT_CODE; restarting in ${RESTART_SLEEP}s" | tee -a "$LOG_FILE"
    sleep "$RESTART_SLEEP"
done
