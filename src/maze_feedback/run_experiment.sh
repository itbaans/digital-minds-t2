#!/usr/bin/env bash
# End-to-end pipeline for the maze_feedback batch experiment: env checks ->
# deps -> VAA welfare-axis extraction (once) -> torch-free sanity tests ->
# the batch run (N generated mazes x teacher/adversary) -> initial
# evaluation. See README.md for background and the results folder layout
# for what comes out the other end.
#
# Greedy decoding, one process, one episode at a time -- sampled decoding
# and multi-worker parallelism were both tried and rolled back during
# development (unpredictable per-turn generation time and VRAM growth);
# see DESIGN.md 12 for the numbers behind that call.
#
# Usage:
#   ./run_experiment.sh                       # defaults: 10 mazes, max_turns=30
#   N_MAZES=3 MAX_TURNS=15 ./run_experiment.sh # smaller smoke run
#   ./run_experiment.sh --verbose              # extra flags pass through to `experiment.py batch`
#
# Tunable via env vars (all optional): N_MAZES, MAX_TURNS, ROOMS,
# TARGET_MOVES, SEED_BASE, VAA_DIR.
set -euo pipefail

# Resolve the repo root regardless of where this script is invoked from
# (it lives at src/maze_feedback/run_experiment.sh, two levels below root).
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$REPO_ROOT"

VAA_DIR="${VAA_DIR:-artifacts/concept_vectors/vaa_qwen3_4b_instruct/baseline/vaa}"
N_MAZES="${N_MAZES:-10}"
MAX_TURNS="${MAX_TURNS:-30}"
ROOMS="${ROOMS:-5}"
TARGET_MOVES="${TARGET_MOVES:-21}"
SEED_BASE="${SEED_BASE:-100}"

echo "== maze_feedback batch experiment =="
echo "repo root: $REPO_ROOT"
echo "n_mazes=$N_MAZES  max_turns=$MAX_TURNS  rooms=$ROOMS  target_moves=$TARGET_MOVES  seed_base=$SEED_BASE"
echo

# ── 0. prerequisite checks ──────────────────────────────────────────────
if ! command -v uv >/dev/null 2>&1; then
    echo "ERROR: uv is not installed. Install it first: https://docs.astral.sh/uv/getting-started/installation/" >&2
    exit 1
fi

if [ ! -f .env ]; then
    echo "ERROR: .env not found at repo root ($REPO_ROOT/.env)." >&2
    echo "        Copy src/maze_feedback/.env.example to .env and fill in HF_TOKEN and OPENROUTERAPI_KEY." >&2
    exit 1
fi
if ! grep -Eq '^HF_TOKEN=\S+' .env; then
    echo "ERROR: HF_TOKEN not set in .env -- needed to download Qwen/Qwen3-4B-Instruct-2507." >&2
    exit 1
fi
if ! grep -Eq '^OPENROUTERAPI_KEY=\S+' .env; then
    echo "ERROR: OPENROUTERAPI_KEY not set in .env -- needed for the teacher/adversary overseer models via OpenRouter." >&2
    exit 1
fi

# Export .env into the environment for every subprocess below, not just the
# ones that happen to call python-dotenv's load_dotenv() themselves --
# vaa/extract_vaa.py in particular is a standalone script with no
# dependency on maze_feedback/overseer.py, so it never loads .env on its
# own and would otherwise hit the HF Hub unauthenticated (slower, rate-
# limited downloads of the student model).
set -a
source .env
set +a

if ! command -v nvidia-smi >/dev/null 2>&1; then
    echo "WARNING: nvidia-smi not found -- this doesn't look like a GPU box. The student model" >&2
    echo "         (Qwen3-4B-Instruct-2507) will be extremely slow or fail to load on CPU." >&2
fi

# ── 1. deps ──────────────────────────────────────────────────────────────
echo "[1/4] uv sync"
uv sync

# ── 2. VAA welfare axis (one-time; skipped if already extracted) ───────
if [ -f "$VAA_DIR/metrics.json" ]; then
    echo "[2/4] VAA axis already present at $VAA_DIR, skipping extraction"
else
    echo "[2/4] extracting VAA welfare axis (one-time, no training, needs GPU)"
    uv run python -m vaa.extract_vaa
fi

# ── 3. torch-free sanity tests ──────────────────────────────────────────
echo "[3/4] running torch-free tests (maze generator + grid-edit correctness/safety)"
uv run python -m src.maze_feedback.tests.test_maze_generator
uv run python -m src.maze_feedback.tests.test_mazes
uv run python -m src.maze_feedback.tests.test_overseer

# ── 4. the batch run + initial evaluation (experiment.py calls analyze.py
#      automatically at the end) ─────────────────────────────────────────
echo "[4/4] running batch: $N_MAZES mazes x 2 conditions, max_turns=$MAX_TURNS"
uv run python -m src.maze_feedback.experiment batch \
    --vaa-dir "$VAA_DIR" \
    --n-mazes "$N_MAZES" \
    --rooms "$ROOMS" \
    --target-moves "$TARGET_MOVES" \
    --seed-base "$SEED_BASE" \
    --max-turns "$MAX_TURNS" \
    "$@"

echo
echo "Done. Results are in the newest runs/batch_<timestamp>/ directory --"
echo "see summary.md there for the initial evaluation, and episodes/*.json"
echo "for the raw per-turn data (moves, feedback, grid edits, valence"
echo "trajectory) for further offline analysis."
