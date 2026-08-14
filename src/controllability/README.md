# controllability — learned-helplessness experiment (Design 2)

Drop-in package for the functional-welfare repo. It manipulates **contingency**
between a Qwen3-4B agent's actions and outcomes while holding outcomes and
vocabulary identical across arms, and reads out / steers along the repo's
welfare axis.

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
env.py          task, arms, triadic yoking, action parsing (no torch)
axis.py         load the VAA artifact, pick layer, project, diff-of-means
runner.py       generate + capture + steer via the repo's hooks; run a block
experiment.py   CLI: master -> yoked -> control -> transfer, + self-axis vs VAA
analyze.py      figures (valence trajectory, transfer solve rate) + sanity
tests/          torch-free tests of the yoking logic
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

# 4) causal steer-to-rescue on the yoked transfer block
python -m src.controllability.experiment --vaa-dir <...> --steer-rescue 6.0
```

## What to read off a run

- **Sanity (first):** corr(action,outcome) ~1 master, ~0 yoked, outcomes_matched true.
- **Read-out:** yoked valence trajectory drifting below master despite identical outcomes.
- **Transfer (headline):** yoked solve_rate drops on the fresh solvable task.
- **Causal:** --steer-rescue lifts the yoked transfer solve rate.
- **Axis check:** printed cosine(self-derived, VAA) — does an axis built from your
  contingent-vs-yoked activations agree with the off-the-shelf VAA?

## Limitations to state in the write-up

VAA entangles assent with valence; single-layer last-token read-out (sweep --layer);
claims are functional, not phenomenal.
