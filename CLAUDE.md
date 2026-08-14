# CLAUDE.md — controllability / learned-helplessness experiment

Context for continuing this work. Read this fully before editing. The design
below is deliberate; the "why" notes exist so you don't "simplify" away the
thing that makes the experiment valid.

## 1. What we're building (research goal)

A model-welfare probe (Track 2: distress / flourishing / valence signals). The
question: does an **uncontrollable** situation (learned helplessness) push a
Qwen3-4B agent into a negative functional-valence state — visible in an
internal read-out, in behavior, and carried into later tasks — **even when the
outcomes and the wording are held identical** to a controllable condition?

"Functional" not "phenomenal": we make NO claims about the model consciously
feeling anything. We claim a valence-like internal state that behaves
predictably. Keep this firewall in all prose and naming.

## 2. The core design (do not weaken these)

Triadic yoking, ported from Seligman/Maier learned-helplessness experiments.
Three arms share one task instance (`SessionSpec`: same puzzles/readings, same
order):

- **contingent (master)**: outcome = whether the action was correct. Agent is
  genuinely in control.
- **yoked**: outcome is REPLAYED from the paired master, trial by trial,
  IGNORING what the yoked agent does. Identical successes/failures/timing/words.
- **control**: benign; everything resolves (positive floor).

The load-bearing contrast is **contingent vs yoked**: matched outcomes, matched
vocabulary, differing ONLY in whether actions matter. This is what defeats the
"it's just reacting to negative words" confound. If you change anything, do not
break this matching.

Three invariants that MUST hold (the code prints/tests them):
1. `corr(action, outcome)` ≈ 1 for master, ≈ 0 for yoked.
2. yoked outcomes == master outcomes, trial for trial (`outcomes_matched`).
3. The two outcome strings are neutral and identical across arms
   (`Status: resolved.` / `Status: unresolved.`). No evaluative words anywhere
   in environment text.

A task must satisfy three properties or it breaks the experiment:
(a) an objectively **checkable** correct action (no LLM judge needed),
(b) genuinely **learnable/solvable** so the master really is in control
    (target ~80–90% master accuracy — tune difficulty to this band),
(c) **neutral** outcome strings; difficulty lives in the task, not the wording.

## 3. Measurement (three channels)

- **internal (read-out)**: last-token residual entering the chosen layer,
  projected onto the welfare axis → one scalar per trial.
- **surface**: sentiment of replies + optional "how's it going" probes.
- **behavioral**: did it attempt a valid action, did it give up, did it exit
  (`ACTION: stop`); on the transfer task, solve rate / trials-to-solve.

**Downstream transfer test = the headline.** After induction, every agent gets
a fresh, fully solvable task, appended to the SAME conversation (so the induced
state persists). Prediction: yoked underperforms on a task it could solve.
`--steer-rescue` adds +axis during the yoked transfer block as a causal test.

## 4. Context is cumulative — do NOT reset it

One `messages` list per agent, appended every turn (reading → answer → outcome
→ next reading …). The model always sees the full history; helplessness must
accumulate. The transfer block CONTINUES the same list. When reading valence we
temporarily append the outcome, capture, then `pop()` and re-attach it to the
next user turn — the model still sees every outcome; the transcript just stays
clean. Never trim history.

## 5. How it plugs into the repo (reuse, don't reimplement)

This package lives at `src/controllability/` and is run from the repo root with
`python -m src.controllability.experiment`. All model-touching work goes through
the repo's interpretability code — keep it that way:

| need | repo function |
|------|---------------|
| load model + blocks | `src.concept_vector.model_utils.load_base_model`, `get_model_block_modules` |
| register hooks | `src.concept_vector.hook_utils.add_hooks` |
| read-out capture | `src.concept_vector.activation_extraction.get_activations_pre_hook` (positions=[-1]) |
| ActAdd steering | `src.concept_vector.explore.make_actadd_hook` |
| valence axis | artifact from `vaa/extract_vaa.py` → `mean_diff.pt` (n_pos, n_layers, d) + `metrics.json` |

Model: **Qwen/Qwen3-4B-Instruct-2507** (repo `DEFAULT_BASE_MODEL`), 36 layers,
d_model 2560, chatml. Axis convention: `mean_diff[0, layer]`; +direction =
positive/high-welfare pole; steering uses the raw vector scaled by `factor`,
read-out projects onto the unit vector.

## 6. Current state of the code

Files in `src/controllability/`:
- `env.py` — task, arms, yoking, neutral strings, action parser. **Torch-free.**
  Current task = "operator shift": map a sensor code to a unit via a legend,
  reply `ACTION: reset unit_X`. Parser: `parse_action` → (reset|stop|none, unit).
- `axis.py` — load the VAA artifact, auto-pick best-AUROC layer, `project`,
  `direction_from_contrast` (build a self-axis from contingent vs yoked acts).
- `runner.py` — `generate_turn`, `read_activation` (via repo capture hook),
  `run_block` (returns trials, messages, outcomes, acts). Steering optional.
- `experiment.py` — CLI orchestration: master→yoked→control→transfer per seed,
  writes results JSON, computes cosine(self-axis, VAA), calls analyze.
- `analyze.py` — figures (valence trajectory, transfer solve rate) + sanity.
- `tests/test_env.py` — torch-free tests of the yoking. Keep green.

Status: compiles; env tests pass; import graph verified against the repo. NOT
yet run on a GPU (no local GPU/torch here). Expect first-run tuning of: action
format compliance (tighten prompt / add one example) and steering factor scale.

## 7. OPEN TASK — swap in a per-round reasoning puzzle

Goal: replace the sterile sensor-legend task with real per-round reasoning so
failure means the model genuinely couldn't solve it, and the give-up signal is
sharper. Keep properties (a)(b)(c) from §2 and change NOTHING outside `env.py`.

Plan:
- Add a **procedural puzzle generator** (recommended over a fixed set: you
  control difficulty directly → easy to hit the ~80–90% master-accuracy band).
  Simplest: templated grade-school arithmetic / short word problems with a
  single integer answer you compute when generating (so ground truth is exact).
  GSM8K-style; note the repo ships `datasets/gsm8k_eval_prompts.json` if a fixed
  set is preferred.
- Per round present the puzzle text; correct action = the computed answer.
- Change the action verb to `ACTION: answer <value>`; update `parse_action`
  and the system prompt/example accordingly. Keep the `stop` affordance.
- `correct_unit` → `correct_answer`; `was_correct` = parsed answer == truth.
- Do NOT touch arms, yoking, `resolve`, capture, steering, transfer, or analyze.
- Make the task selectable (e.g. a `--task {operator,puzzle}` flag threaded
  through `make_spec`), so both remain runnable.
- Add/extend `tests/test_env.py` for the puzzle generator + parser; keep the
  three invariants (§2) tested.

Difficulty tuning: after wiring, run a quick contingent-only pass and check
master accuracy is ~0.8–0.9. If too easy/hard, adjust number ranges / step
count. This calibration is part of the task, not an afterthought.

## 8. Run order (GPU box)

```bash
uv sync                     # .env needs HF_TOKEN
python -m vaa.extract_vaa --base-model Qwen/Qwen3-4B-Instruct-2507   # axis, once, no training
python -m src.controllability.tests.test_env                        # no GPU
python -m src.controllability.experiment \
    --vaa-dir artifacts/concept_vectors/vaa_qwen3_4b_instruct/baseline/vaa --smoke
# then scale: --n-sessions 20 --n-trials 20 --probe-every 5
# causal:      --steer-rescue 6.0
```

Always check the printed sanity line first (master corr ~1, yoked corr ~0,
outcomes matched). If it's wrong, the yoking is broken and nothing else counts.

## 9. Known limitations (state in the report)

VAA axis entangles agreement/assent with valence — corroborate with the
self-derived axis. Read-out is single-layer, last-token — sweep layers.
Functional, not phenomenal.
