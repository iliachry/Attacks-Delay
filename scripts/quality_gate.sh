#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

echo "=== [Gate 1] Ruff Static Analysis & Formatting ==="
UV_LINK_MODE=copy uvx ruff check src/ tests/ scripts/

echo "=== [Gate 2] Pytest Unit, SymPy & SimPy Verification Suite ==="
UV_LINK_MODE=copy uv run pytest

echo "=== [Gate 3] README Drift & CLI Contract Verification ==="
python3 ~/Projects/sovereign-kb/src/sovereign_kb/readme_check.py "$REPO_ROOT"

echo "=== [Gate 4] Goldwood Index Triple-9+ Audit (PAPER / APP / VIDEO) ==="
~/.local/bin/kb gwi audit "$REPO_ROOT" --profile paper --strict-lenses
~/.local/bin/kb gwi audit "$REPO_ROOT" --profile app --strict-lenses
~/.local/bin/kb gwi audit "$REPO_ROOT/artifacts/walkthrough_compressed.mp4" --profile video --probe-audio
