# zebra_logic — ZebraLogic benchmark

Benchmarks a model on ZebraLogic grid-logic puzzles (small/easy sizes by
default). This is a preliminary step for a larger not-yet-defined experiment
built around zebra-style logic puzzles — this package currently only covers
the benchmarking piece.

## Key decision: which dataset, and why

`allenai/ZebraLogicBench` (the "official" public repo) masks every solution
cell (`"___"`) — it's a held-out leaderboard set. The real answers live in a
**gated** `allenai/ZebraLogicBench-private` repo we don't have approved
access to (confirmed: 401 GatedRepoError).

We use **`WildEval/ZebraLogic`** instead — the original/earlier release of
the same benchmark (identical puzzle text and ids, verified against a sample
from the allenai version) with solutions intact, so we can self-score
locally.

## Faithfulness to the public benchmark

Prompt template (`prompts.py`) and scoring logic (`scoring.py`) are
reproduced verbatim from the benchmark's own eval code,
[ZeroEval](https://github.com/yuchenlin/ZeroEval) (`src/templates/
ZEBRA_GRID.py`, `src/_TEMPLATES.py`, `src/evaluation/zebra_grid_eval.py`),
so numbers are comparable to the public leaderboard rather than an artifact
of our own prompt wording:
- One-shot example with reasoning + JSON solution format.
- Answer extraction: brace-matching for the LAST complete top-level JSON
  object in the raw output (tolerant of preamble/trailing text).
- Scoring: case-insensitive, whitespace-stripped exact string match per
  cell. **Puzzle Acc** = fraction of puzzles with every cell correct.
  **Cell Acc** = fraction of individual cells correct.

Size-difficulty buckets (`data.py`: `SMALL_SIZES`, `MEDIUM_SIZES`,
`LARGE_SIZES`, `XL_SIZES`, `EASY_SIZES`, `HARD_SIZES`) are copied verbatim
from the same eval code. "Small" = 8 size buckets (2×2 through 4×2), 320 of
the 1000 puzzles.

## Files

```
data.py         load puzzles from WildEval/ZebraLogic, size-group constants (no torch)
prompts.py      ZeroEval's exact one-shot prompt template (no torch)
scoring.py      JSON extraction + puzzle/cell-level scoring (no torch)
runner.py       model loading + generation via the repo's model_utils (needs torch)
benchmark.py    CLI: run a model over a size group, print + save results
tests/          torch-free tests for parsing/prompt/scoring logic
```

## Run

```bash
uv sync    # picks up huggingface_hub from pyproject.toml

python -m src.zebra_logic.tests.test_scoring    # no GPU

# quick check (10 random small puzzles)
python -m src.zebra_logic.benchmark --sizes small --n 10 --verbose

# full small-size benchmark (320 puzzles)
python -m src.zebra_logic.benchmark --sizes small
```

Default model is the repo's `DEFAULT_BASE_MODEL` (currently
Qwen3-4B-Instruct-2507); override with `--model`. `--max-new-tokens 1024`
default should be plenty for small puzzles; larger sizes will likely need
more.
