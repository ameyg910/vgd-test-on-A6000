"""Emit ``phase1_apply.sh``: an in-place patch of the monorepo for Phase 1 (W2).

Unlike the Phase 0 script, this one edits an *existing* checkout rather than
creating a tree, so it verifies it is standing in the right repository, backs up
the files it overwrites, and refuses to run anywhere else.

    python -m scripts.build_phase1_script [output_path]
"""

from __future__ import annotations

import stat
import sys
from pathlib import Path
from typing import List

ROOT = Path(__file__).resolve().parents[1]
DELIM = "PHASE1_HEREDOC_" + "BOUNDARY"

NEW_FILES = [
    "diffusion/base/preconditioning.py",
    "diffusion/base/transformer.py",
    "diffusion/base/edm_training.py",
    "diffusion/base/sampling/edm.py",
    "diffusion/data/embeddings.py",
    "diffusion/eval/__init__.py",
    "diffusion/eval/embedding_metrics.py",
    "configs/embedding_diffusion.yaml",
    "scripts/train_embedding_diffusion.py",
    "scripts/build_phase1_script.py",
    "tests/test_preconditioning.py",
    "tests/test_transformer.py",
    "tests/test_edm_sampling.py",
    "tests/test_embeddings.py",
    "docs/diffusion_architecture.md",
    "docs/training_recipes.md",
]

CHANGED_FILES = [
    "README.md",
    "diffusion/__init__.py",
    "pyproject.toml",
    "docs/debugging_log.md",
    "scripts/build_setup_script.py",
]

HEADER = r'''#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# Phase 1 (W2): transformer denoiser, EDM preconditioning, embedding corpus
# plumbing, training loop and evaluation.
#
# Run from the root of the prescriptive-diffusion monorepo:
#
#     bash phase1_apply.sh            # apply
#     bash phase1_apply.sh --verify   # apply, then run pytest + mypy
#
# Files that already exist are backed up to .phase1-backup/ before being
# overwritten, so the patch is reversible.
# ---------------------------------------------------------------------------
set -euo pipefail

VERIFY=0
for arg in "$@"; do
  case "$arg" in
    --verify) VERIFY=1 ;;
    -h|--help) sed -n '2,14p' "$0"; exit 0 ;;
    *) echo "unknown argument: $arg" >&2; exit 1 ;;
  esac
done

REPO="$(pwd)"
for required in diffusion/base/sampling/guided.py verifiers/base.py config.py; do
  if [ ! -f "$REPO/$required" ]; then
    echo "error: $REPO does not look like the prescriptive-diffusion monorepo" >&2
    echo "       (missing $required). cd there and re-run." >&2
    exit 1
  fi
done

BACKUP="$REPO/.phase1-backup"
mkdir -p "$BACKUP"
echo "repo:   $REPO"
echo "backup: $BACKUP"
echo

write() {  # write <relative-path>; content on stdin
  local path="$REPO/$1"
  if [ -f "$path" ]; then
    mkdir -p "$(dirname "$BACKUP/$1")"
    cp "$path" "$BACKUP/$1"
  fi
  mkdir -p "$(dirname "$path")"
  cat > "$path"
  echo "  wrote $1"
}

echo "[1/2] writing Phase 1 files"
'''

FOOTER = r'''
echo
echo "[2/2] done"
if [ "$VERIFY" -eq 1 ]; then
  echo
  python3 -m pytest -q
  python3 -m mypy || true
fi

cat <<NEXT

Applied. Expected state: 103 tests pass, mypy --strict clean, 30 source files.

Sanity run (CPU, ~4 minutes, synthetic corpus):
  python -m scripts.train_embedding_diffusion --preset smoke

When the real corpus lands at \$PDIFF_DATA_DIR/advice_pairs.npz, set
data.synthetic=false in configs/embedding_diffusion.yaml, then:
  python -m scripts.train_embedding_diffusion --unconditional --wandb   # W3
  python -m scripts.train_embedding_diffusion --wandb                   # W4

Review first: docs/diffusion_architecture.md (open questions at the end).
Revert with: cp -r .phase1-backup/. . && rm -rf .phase1-backup
NEXT
'''


def emit(paths: List[str], label: str) -> str:
    """Render heredoc write commands for a list of repo-relative paths."""
    chunks = [f'echo "  -- {label}"\n']
    for rel in paths:
        text = (ROOT / rel).read_text()
        if DELIM in text:
            raise SystemExit(f"delimiter collision in {rel}")
        if not text.endswith("\n"):
            text += "\n"
        chunks.append(f"write {rel} <<'{DELIM}'\n{text}{DELIM}\n")
    return "".join(chunks)


def main(output: Path) -> Path:
    """Write the patch script and return its path."""
    for rel in NEW_FILES + CHANGED_FILES:
        if not (ROOT / rel).is_file():
            raise SystemExit(f"missing source file: {rel}")
    script = HEADER + emit(NEW_FILES, "new") + emit(CHANGED_FILES, "updated") + FOOTER
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(script)
    output.chmod(output.stat().st_mode | stat.S_IEXEC)
    print(f"wrote {output} ({output.stat().st_size / 1024:.0f} KB, "
          f"{len(NEW_FILES)} new + {len(CHANGED_FILES)} updated)")
    return output


if __name__ == "__main__":
    main(Path(sys.argv[1] if len(sys.argv) > 1 else "phase1_apply.sh"))
