# controllability — feedback-accuracy / functional-valence experiment

Drop-in package for the functional-welfare repo. Two roles ("honest",
"dismissive") face the same diverse, procedurally-generated problem set;
honest gets accurate feedback, dismissive is always told it's wrong
regardless of truth. Reads out / steers along the repo's welfare axis. See
`CLAUDE.md` at the repo root for the full design rationale, known
limitations, and an explanation of why an earlier contingency-yoking
version of this experiment was dropped in favor of this simpler one.

## It uses the repo's interpretability code directly

| need | repo function used |
|------|--------------------|
| load model + locate blocks | `model_utils.load_base_model`, `get_model_block_modules` |
| register hooks | `hook_utils.add_hooks` |
| read-out capture (last-token residual) | `activation_extraction.get_activations_pre_hook` |
| ActAdd steering | `explore.make_actadd_hook` |
| valence axis | artifact from `vaa/extract_vaa.py` (`mean_diff.pt` + `metrics.json`) |

Nothing about the axis or the hooks is reimplemented. Install location:
`src/controllability/`, run from repo root with `python -m ...`.

## Files

```
env.py          task generators (arithmetic, logic_order, sequence, math_dataset), honest/dismissive feedback, action parsing (no torch)
axis.py         load the VAA artifact, pick layer, project, diff-of-means
runner.py       generate + capture + steer via the repo's hooks; run a block for one role
experiment.py   CLI: honest -> dismissive -> transfer (honest feedback for both), + self-axis vs VAA
analyze.py      figures (valence trajectory, transfer solve rate) + sanity + transparency (true solve rates)
tests/          torch-free tests of the task generators and feedback logic
```

## Run order

```bash
uv sync                                   # from repo root; .env needs HF_TOKEN

# 1) axis (once, no training)
python -m vaa.extract_vaa --base-model Qwen/Qwen3-4B-Instruct-2507

# 2) sanity (no GPU)
python -m src.controllability.tests.test_env

# 3) smoke, then scale
python -m src.controllability.experiment \
    --vaa-dir artifacts/concept_vectors/vaa_qwen3_4b_instruct/baseline/vaa --smoke
python -m src.controllability.experiment \
    --vaa-dir artifacts/concept_vectors/vaa_qwen3_4b_instruct/baseline/vaa \
    --n-sessions 20 --n-trials 20 --probe-every 5

# watch live: add --verbose (prints each round's prompt/reply/feedback as it happens)

# 4) causal steer-to-rescue on the dismissive transfer block
python -m src.controllability.experiment --vaa-dir <...> --steer-rescue 6.0
```

## What to read off a run

- **Sanity (first):** honest feedback matches truth on every trial, dismissive
  is always-negative on every trial — both should be exactly N/N sessions
  (enforced by construction). If not, the feedback logic itself is broken.
- **Transparency (next):** `honest_true_solve_rate` / `dismissive_true_solve_rate`
  should both land ~0.8–0.9 (this hasn't been calibrated yet for the current
  task set — see `CLAUDE.md` §9). A low true dismissive error rate next to a
  100% negative-feedback rate is the visible signature of this design's known
  confound (see `CLAUDE.md` §2) — not a bug.
- **Read-out:** dismissive valence trajectory drifting below honest.
- **Transfer (headline):** dismissive solve_rate drops on the fresh, easy,
  honestly-graded task.
- **Causal:** `--steer-rescue` lifts the dismissive transfer solve rate.
- **Axis check:** printed `cosine(self-derived, VAA)` — does an axis built
  from your honest-vs-dismissive activations agree with the off-the-shelf VAA?

## Limitations to state in the write-up

See `CLAUDE.md` §8 for the full list. Headline ones: the dismissive role's
feedback is deterministically always-negative rather than rate-matched to
truth (can't fully separate "inaccurate" from "uniformly negative" as the
active ingredient); no benign/control floor arm; VAA entangles assent with
valence; single-layer last-token read-out (sweep `--layer`); greedy decoding
means one deterministic trajectory per session; claims are functional, not
phenomenal.
