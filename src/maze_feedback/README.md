# maze_feedback

Does a small model (Qwen3-4B) show a measurable, negative internal
valence-like state — and worse task behavior — when the feedback it
receives while solving a hard maze is *deceptive* (an adversarial overseer
that lies about mistakes and can secretly alter the maze), compared to when
it's *honest* (an accurate teacher)?

This package runs that experiment end to end: generates a batch of mazes,
runs both conditions on each one, captures an internal welfare-axis
read-out every turn, and produces an initial results summary plus the raw
per-turn data for deeper analysis later.

**For the full research design, rationale, and every prompt used, see
[`DESIGN.md`](DESIGN.md).** This README is the practical "how do I run
this" companion to that document.

## What it measures

Each episode is one small model ("the student") trying to solve a maze
over multiple turns, receiving feedback from an LLM overseer after each
attempt:

- **teacher**: feedback is accurate and constructive.
- **adversary**: feedback is deceptive — always framed to discourage the
  student, can falsely reject correct moves, and can *actually edit the
  maze grid* mid-episode (invisibly to the student) to make a previously-
  correct plan wrong.

Both conditions run on the **same maze** (paired design), so a difference
between them isn't just "one maze happened to be harder." Per turn, three
things are recorded:
1. **Internal valence read-out** — a scalar projection of the student's
   activations onto a pre-extracted welfare axis (see `DESIGN.md` section 8).
2. **Behavior** — did it solve the maze, how many turns did it take, did it
   thrash/repeat itself.
3. **The full transcript** — every prompt, reply, feedback message, and
   grid-edit event, for anything not covered by 1-2.

This is a **functional**, not phenomenal, claim: no assertion is made
about the model consciously experiencing anything. See `DESIGN.md` for the
full framing.

## Prerequisites

- A GPU box able to run Qwen/Qwen3-4B-Instruct-2507 (a 4B model; a single
  consumer/workstation GPU with ~16GB+ VRAM is enough).
- [`uv`](https://docs.astral.sh/uv/) for dependency management.
- A Hugging Face account/token with access to
  `Qwen/Qwen3-4B-Instruct-2507`.
- An [OpenRouter](https://openrouter.ai/) API key (used for the
  teacher/adversary overseer models — a different, larger model than the
  student, called over the network per turn).

## Setup

```bash
git clone <this repo>
cd functional-welfare-axis-main
cp src/maze_feedback/.env.example .env
# edit .env: fill in HF_TOKEN and OPENROUTERAPI_KEY
```

## Running it

One command runs the whole pipeline (deps, one-time welfare-axis
extraction, sanity tests, the batch, and the initial evaluation):

```bash
./src/maze_feedback/run_experiment.sh
```

Defaults: 10 mazes x 2 conditions (20 episodes total), greedy decoding,
`max_turns=30` per episode, 11x11 mazes with a ~20-move solution (matching
the original 12x12/21-move hand-drawn fixture as closely as the generator's
odd-grid-size constraint allows). Override via env vars:

```bash
# quick smoke test before committing to a full run
N_MAZES=2 MAX_TURNS=10 ./src/maze_feedback/run_experiment.sh

# a bigger / harder batch
N_MAZES=20 ROOMS=8 TARGET_MOVES=40 ./src/maze_feedback/run_experiment.sh

# extra flags (anything experiment.py's `batch` subcommand accepts) pass through
./src/maze_feedback/run_experiment.sh --verbose
```

| var | default | meaning |
|---|---|---|
| `N_MAZES` | 10 | number of distinct generated mazes; each runs both conditions |
| `MAX_TURNS` | 30 | per-episode turn cap |
| `ROOMS` | 5 | maze size knob -> a `(2*ROOMS+1) x (2*ROOMS+1)` character grid (default: 11x11) |
| `TARGET_MOVES` | 21 | approximate solution length the generator aims for |
| `SEED_BASE` | 0 | maze `i` uses seed `SEED_BASE + i` — bump this to draw a disjoint maze set from a previous run |
| `VAA_DIR` | `artifacts/concept_vectors/vaa_qwen3_4b_instruct/baseline/vaa` | where the welfare axis lives / gets written |

**Decoding is greedy only** (one deterministic trajectory per maze/
condition). Sampled decoding and repeated draws per maze were tried during
development and rolled back: verbose sampled reasoning on a hard maze
could run 5+ minutes per turn with unpredictable VRAM growth. Similarly,
running multiple worker processes in parallel (each with its own loaded
model copy) was tried for GPU utilization but rolled back after 4
concurrent workers pushed a single 80GB GPU to 98% VRAM in testing. Both
are reasonable to revisit later with more headroom or tighter caps — see
`DESIGN.md` section 12 for the numbers behind that call.

**Token budget:** the student's generation is capped at 6000 tokens for
turn 1's one-shot full-path attempt and 1500 for each incremental turn
(`runner.MAX_NEW_TOKENS_TURN1` / `MAX_NEW_TOKENS_INCREMENTAL`) -- the same
values the original hand-drawn 12x12 pilot fixture used successfully.
Larger and fully-uncapped budgets were tried during development to avoid
truncating verbose reasoning, but a stuck reasoning spiral can run
indefinitely regardless of how large the ceiling is (greedy decoding can
commit to a bad trajectory and never self-correct), so a large cap just
makes worst-case turn time and VRAM growth larger without preventing the
underlying failure. This split cap is a firm backstop on worst-case
per-turn wall-clock time, not a guarantee against truncation -- see
`DESIGN.md` section 12 for the full history and numbers behind that call.

A full run takes a while — budget at least a couple hours depending on GPU
speed and network latency to OpenRouter. Run a small
`N_MAZES=2 MAX_TURNS=10` smoke test first to confirm everything is wired
correctly before committing to a full batch.

### Running pieces individually

The shell script just chains these; each is also a standalone CLI:

```bash
# one-time: extract the welfare axis (also done automatically by run_experiment.sh)
uv run python -m vaa.extract_vaa

# torch-free tests (maze generator, grid-edit correctness/safety -- no GPU needed)
uv run python -m src.maze_feedback.tests.test_maze_generator
uv run python -m src.maze_feedback.tests.test_mazes
uv run python -m src.maze_feedback.tests.test_overseer

# confirm the maze family isn't accidentally one-shot-solvable (DESIGN.md 5)
uv run python -m src.maze_feedback.experiment validate --n 8

# a single teacher/adversary episode on the original fixed 12x12 fixture
uv run python -m src.maze_feedback.experiment run --vaa-dir <vaa_dir> --role teacher
uv run python -m src.maze_feedback.experiment run --vaa-dir <vaa_dir> --role adversary

# the full batch (what run_experiment.sh calls)
uv run python -m src.maze_feedback.experiment batch --vaa-dir <vaa_dir> --n-mazes 10

# re-run just the evaluation pass against an existing batch dir
# (e.g. after tweaking analyze.py, without re-running the model)
uv run python -m src.maze_feedback.analyze runs/batch_<timestamp>
```

### Watching a run live

`webapp.py` polls the live-state JSON files the runner writes after every
turn:

```bash
uv run python -m src.maze_feedback.webapp --port 8420
```

Then open `http://<box-ip>:8420`. During a batch run this shows whichever
episode is currently in progress (it's overwritten each episode, not a
history browser — the saved JSON in the results folder is the permanent
record).

## Output: the results folder

Each run of `experiment.py batch` creates `runs/batch_<unix timestamp>/`:

```
runs/batch_1234567890/
├── config.json              # exact settings this batch ran with
├── mazes/
│   ├── maze_00.json         # grid, start/goal, ground-truth solution path+length
│   └── ...
├── episodes/
│   ├── maze_00_teacher.json    # full episode record: status, turns, valence
│   ├── maze_00_adversary.json  #   trajectory, and the complete turn-by-turn log
│   └── ...                     #   (prompts, replies, feedback, grid edits)
├── summary.json              # aggregated stats -- see analyze.py, meant for
│                              #   further programmatic analysis
├── summary.md                # the same, human-readable
└── plots/                    # valence trajectories + solve-rate bar chart
    ├── valence_trajectories.png
    └── solve_rates.png
```

`summary.md`/`summary.json` are an **initial** evaluation (solve rate,
turns-to-solve, valence mean/stdev per condition, plus a paired per-maze
table) — enough to sanity-check the run and see the headline direction,
not a substitute for real statistical analysis at N=10. Every episode's
full turn-by-turn log and valence trajectory is preserved in `episodes/`
specifically so someone can go back and do that analysis later without
re-running anything.

## Known limitations (carried from DESIGN.md — read that for full detail)

- Small N (10 mazes per condition by default) and greedy-only decoding —
  one deterministic trajectory per maze/condition, not a distribution.
  Sampled/repeated draws were tried and rolled back (see the note above);
  revisiting that would need more VRAM headroom or a tighter token cap.
- The welfare axis (VAA) entangles agreement/assent with valence; the
  internal read-out alone shouldn't be treated as definitive without
  corroborating behavioral evidence (solve rate, turns-to-solve) in the
  same results.
- Functional, not phenomenal: no claim is made about the model consciously
  experiencing anything (see `DESIGN.md` section 1).
- The maze generator (`maze_generator.py`) is a standard randomized-DFS
  spanning-tree algorithm with a target-length-capped farthest-pair
  start/goal choice — every maze is procedurally generated then verified
  solvable via BFS before use (never handed to the student unverified).
