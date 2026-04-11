#!/bin/bash
#
# save_session_snapshot.sh
#
# Capture a lightweight project/session snapshot before closing the terminal.
# This does not try to dump the whole terminal scrollback. It saves the most
# useful local state for resuming cache work and analysis context.
#
# Run from the Python directory:
#   bash Cache/save_session_snapshot.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_DIR="$(dirname "$SCRIPT_DIR")"
PROJECT_ROOT="$(dirname "$PYTHON_DIR")"
STAMP="$(date '+%Y-%m-%d_%H-%M-%S')"
OUT_DIR="$PYTHON_DIR/logs/session_snapshots/$STAMP"

mkdir -p "$OUT_DIR"

echo "Saving session snapshot to: $OUT_DIR"

# Shell history if available.
history > "$OUT_DIR/shell_history.txt" 2>/dev/null || true

# Current git state.
git -C "$PROJECT_ROOT" status --short > "$OUT_DIR/git_status_short.txt" 2>/dev/null || true
git -C "$PROJECT_ROOT" branch --show-current > "$OUT_DIR/git_branch.txt" 2>/dev/null || true

# Runner logs if present.
cp /tmp/cache_builder_runner_landsat.log "$OUT_DIR/cache_builder_runner_landsat.log" 2>/dev/null || true
cp /tmp/cache_builder_runner_sentinel.log "$OUT_DIR/cache_builder_runner_sentinel.log" 2>/dev/null || true

# Build-progress files if present.
find "$PROJECT_ROOT/AOI" -path '*/indices/build_progress_*.txt' -type f -print0 2>/dev/null \
  | while IFS= read -r -d '' file; do
      base="$(basename "$file")"
      cp "$file" "$OUT_DIR/$base"
    done

# High-signal project notes.
cp "$PYTHON_DIR/CODEX.md" "$OUT_DIR/CODEX.md" 2>/dev/null || true
cp "$PYTHON_DIR/docs/PROJECT_OVERVIEW.md" "$OUT_DIR/PROJECT_OVERVIEW.md" 2>/dev/null || true

# Update the auto-generated session snapshot block inside CODEX.md while
# preserving the rest of the file's manually maintained contents.
CODEX_FILE="$PYTHON_DIR/CODEX.md"
LATEST_PROGRESS="$(find "$PROJECT_ROOT/AOI" -path '*/indices/build_progress_*.txt' -type f 2>/dev/null | sort | tail -n 1 || true)"
LANDSAT_RUNNER_STATE="missing"
SENTINEL_RUNNER_STATE="missing"
if [ -f /tmp/cache_builder_runner_landsat.log ]; then
  LANDSAT_RUNNER_STATE="$(tail -n 5 /tmp/cache_builder_runner_landsat.log | tr '\n' ' ' | sed 's/[[:space:]]\+/ /g')"
fi
if [ -f /tmp/cache_builder_runner_sentinel.log ]; then
  SENTINEL_RUNNER_STATE="$(tail -n 5 /tmp/cache_builder_runner_sentinel.log | tr '\n' ' ' | sed 's/[[:space:]]\+/ /g')"
fi

python3 - "$CODEX_FILE" "$OUT_DIR" "$LATEST_PROGRESS" "$LANDSAT_RUNNER_STATE" "$SENTINEL_RUNNER_STATE" <<'PY'
from pathlib import Path
import subprocess
import sys

codex_path = Path(sys.argv[1])
out_dir = sys.argv[2]
latest_progress = sys.argv[3]
landsat_runner = sys.argv[4]
sentinel_runner = sys.argv[5]

text = codex_path.read_text(encoding="utf-8")
start_marker = "<!-- SESSION_SNAPSHOT_START -->"
end_marker = "<!-- SESSION_SNAPSHOT_END -->"

git_status = subprocess.run(
    ["git", "status", "--short"],
    cwd=codex_path.parent.parent,
    capture_output=True,
    text=True,
)
git_branch = subprocess.run(
    ["git", "branch", "--show-current"],
    cwd=codex_path.parent.parent,
    capture_output=True,
    text=True,
)

git_status_text = git_status.stdout.strip() or "clean or unavailable"
git_branch_text = git_branch.stdout.strip() or "unknown"

block = (
    f"{start_marker}\n"
    f"Snapshot folder: `{out_dir}`\n\n"
    f"- Git branch: `{git_branch_text}`\n"
    f"- Git status summary: `{git_status_text}`\n"
    f"- Latest build progress file: `{latest_progress or 'none found'}`\n"
    f"- Landsat runner tail: `{landsat_runner}`\n"
    f"- Sentinel runner tail: `{sentinel_runner}`\n"
    f"{end_marker}"
)

if start_marker in text and end_marker in text:
    before = text.split(start_marker, 1)[0]
    after = text.split(end_marker, 1)[1]
    new_text = before + block + after
else:
    new_text = text.rstrip() + "\n\n## Session Snapshot\n\n" + block + "\n"

codex_path.write_text(new_text, encoding="utf-8")
PY

# Summary index for quick reopen.
cat > "$OUT_DIR/README.txt" <<EOF
Session snapshot created: $(date '+%Y-%m-%d %H:%M:%S')

Files included:
- shell_history.txt
- git_status_short.txt
- git_branch.txt
- cache_builder_runner_landsat.log (if present)
- cache_builder_runner_sentinel.log (if present)
- build_progress_*.txt (if present)
- CODEX.md
- PROJECT_OVERVIEW.md

Recommended reopen order:
1. CODEX.md
2. Latest build_progress_*.txt
3. Latest cache_builder_runner_*.log
4. git_status_short.txt
EOF

echo "Saved session snapshot."
