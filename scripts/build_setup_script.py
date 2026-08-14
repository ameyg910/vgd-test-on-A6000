"""Emit a single self-contained shell script that materialises the Phase 0 monorepo.

Run from the monorepo; writes ``phase0_setup.sh`` to the requested output path.
"""

from __future__ import annotations

import stat
import sys
from pathlib import Path
from typing import List

ROOT = Path(__file__).resolve().parents[1]
DELIM = "PDIFF_FILE_EOF"

INCLUDE_SUFFIXES = {".py", ".toml", ".yaml", ".md"}
EXCLUDE_PARTS = {"__pycache__", "artifacts", "figures", ".mypy_cache", ".pytest_cache"}
EXCLUDE_NAMES = {"build_setup_script.py"}

HEADER = r'''#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# Phase 0 (W1): port the screening task into the prescriptive-diffusion monorepo.
#
# Usage, from the root of your screening-task checkout (the directory that
# contains guided_diffusion/):
#
#     bash phase0_setup.sh                 # creates ../prescriptive-diffusion
#     bash phase0_setup.sh /path/to/target # or an explicit target
#     bash phase0_setup.sh --verify        # also run pytest + mypy at the end
#
# The script is idempotent: re-running overwrites the generated files and leaves
# your screening checkout untouched. It never writes outside the target dir,
# except for reading your existing checkpoints.
# ---------------------------------------------------------------------------
set -euo pipefail

VERIFY=0
FORCE=0
TARGET=""
for arg in "$@"; do
  case "$arg" in
    --verify) VERIFY=1 ;;
    --force) FORCE=1 ;;
    -h|--help) sed -n '2,18p' "$0"; exit 0 ;;
    *) TARGET="$arg" ;;
  esac
done

SOURCE_REPO="$(pwd)"
if [ ! -d "$SOURCE_REPO/guided_diffusion" ]; then
  echo "error: run this from your screening-task checkout (no guided_diffusion/ here)" >&2
  echo "       current directory: $SOURCE_REPO" >&2
  exit 1
fi

if [ -z "$TARGET" ]; then
  TARGET="$(cd .. && pwd)/prescriptive-diffusion"
fi
mkdir -p "$TARGET"
TARGET="$(cd "$TARGET" && pwd)"

# A non-empty target is almost always a mistake: merging into an unrelated
# project produces a tree that imports but is not this project.
STRAY="$(cd "$TARGET" && ls -A 2>/dev/null | grep -v -E '^(\.git|\.venv|venv|artifacts|figures|logs|\.cache|\.mypy_cache|\.pytest_cache|data)$' || true)"
if [ -n "$STRAY" ] && [ "$FORCE" -eq 0 ]; then
  echo "error: target is not empty. Refusing to merge into an existing tree." >&2
  echo "       unexpected entries:" >&2
  echo "$STRAY" | sed 's/^/         /' >&2
  echo "       move them aside, or re-run with --force to overwrite." >&2
  exit 1
fi

if [ "$TARGET" = "$SOURCE_REPO" ]; then
  echo "error: target must differ from the screening checkout (it would clobber" >&2
  echo "       pyproject.toml, README.md and tests/). Pass a different path." >&2
  exit 1
fi

echo "source (screening): $SOURCE_REPO"
echo "target (monorepo):  $TARGET"
echo

write() {  # write <relative-path>; content arrives on stdin
  local path="$TARGET/$1"
  mkdir -p "$(dirname "$path")"
  cat > "$path"
  echo "  wrote $1"
}

echo "[1/4] writing source files"
'''

FOOTER = r'''
echo
echo "[2/4] staging trained checkpoints"
mkdir -p "$TARGET/artifacts/checkpoints"
for ckpt in denoiser_toy2d.pt mode_classifier.pt; do
  if [ -f "$SOURCE_REPO/artifacts/$ckpt" ]; then
    cp "$SOURCE_REPO/artifacts/$ckpt" "$TARGET/artifacts/checkpoints/$ckpt"
    echo "  copied $ckpt"
  else
    echo "  WARNING: $SOURCE_REPO/artifacts/$ckpt not found."
    echo "           Re-create it with: python -m scripts.train_toy --epochs 250"
    echo "           then copy it to $TARGET/artifacts/checkpoints/"
  fi
done

echo
echo "[3/4] generating parity goldens from the screening package"
# Frozen outputs of the OLD code, so the parity test runs without it present.
if ( cd "$TARGET" && PYTHONPATH="$TARGET" PDIFF_SCREENING_REPO="$SOURCE_REPO" \
       python3 -m scripts.make_parity_goldens ); then
  echo "  goldens written to tests/data/screening_goldens.pt"
else
  echo "  WARNING: golden generation failed; the parity test will skip."
  echo "           Retry with: PDIFF_SCREENING_REPO=$SOURCE_REPO python -m scripts.make_parity_goldens"
fi

echo
echo "[4/4] done"
if [ "$VERIFY" -eq 1 ]; then
  echo
  echo "running pytest"
  ( cd "$TARGET" && python3 -m pytest -q )
  echo "running mypy --strict"
  ( cd "$TARGET" && python3 -m mypy ) || true
fi

cat <<NEXT

Next steps
----------
  cd $TARGET
  pip install -e ".[dev,viz]"     # torch, pydantic, structlog, pytest, mypy
  python -m pytest -q             # expect 57 passed
  python -m mypy                  # expect: no issues found
  python -m experiments.toy_2d.run --stage all   # ~15 min, reproduces screening results

Per-user scratch on the A6000 boxes (no code change needed):
  export PDIFF_ARTIFACT_DIR=/scratch/\$USER/prescriptive-diffusion/artifacts
  export PDIFF_CHECKPOINT_DIR=/scratch/\$USER/prescriptive-diffusion/checkpoints

What changed vs. the screening task is documented in docs/porting_notes.md.
NEXT
echo
echo "target: $TARGET"
'''


def collect() -> List[Path]:
    """Every source file that belongs in the generated script."""
    files: List[Path] = []
    for path in sorted(ROOT.rglob("*")):
        if not path.is_file():
            continue
        if set(path.relative_to(ROOT).parts) & EXCLUDE_PARTS:
            continue
        if path.name in EXCLUDE_NAMES:
            continue
        if path.suffix not in INCLUDE_SUFFIXES and path.name != "conftest.py":
            continue
        files.append(path)
    return files


def main(output: Path) -> Path:
    """Write the setup script and return its path."""
    files = collect()
    chunks = [HEADER]
    for path in files:
        rel = path.relative_to(ROOT).as_posix()
        text = path.read_text()
        if DELIM in text:
            raise SystemExit(f"delimiter collision in {rel}")
        if not text.strip():
            chunks.append(f'mkdir -p "$TARGET/$(dirname {rel})" && : > "$TARGET/{rel}"'
                          f' && echo "  wrote {rel}"\n')
            continue
        if not text.endswith("\n"):
            text += "\n"
        chunks.append(f"write {rel} <<'{DELIM}'\n{text}{DELIM}\n")
    chunks.append(FOOTER)

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("".join(chunks))
    output.chmod(output.stat().st_mode | stat.S_IEXEC)
    print(f"wrote {output} ({output.stat().st_size / 1024:.0f} KB, {len(files)} files)")
    return output


if __name__ == "__main__":
    main(Path(sys.argv[1] if len(sys.argv) > 1 else "phase0_setup.sh"))
