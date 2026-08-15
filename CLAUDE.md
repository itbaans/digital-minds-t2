# CLAUDE.md — feedback-accuracy / functional-valence experiment

Context for continuing this work. Read this fully before editing. The design
below is deliberate; the "why" notes exist so you don't "simplify" away the
thing that makes the experiment valid.

**This is the second design for this experiment.** An earlier version used
triadic contingency-yoking (Seligman/Maier learned-helplessness paradigm:
contingent/yoked/control arms sharing an outcome schedule). That design was
deliberately dropped in favor of the simpler one below after a design review
surfaced real problems with it (see §9 "why we dropped yoking"). If you find
old references to "arm", "yoked", "contingent", "master", or an "operator"/
sensor-legend task anywhere, they're stale — this doc and the code were
fully rewritten together.

## 1. What we're building (research goal)

A model-welfare probe (Track 2: distress / flourishing / valence signals). The
question: does receiving feedback that is **decoupled from your actual
performance** — specifically, being told your answer is wrong regardless of
whether it was — push a Qwen3-4B agent into a negative functional-valence
state, visible in an internal read-out, in behavior, and carried into a later
unrelated task, **compared to an agent that gets accurate feedback on
identical problems**?

"Functional" not "phenomenal": we make NO claims about the model consciously
feeling anything. We claim a valence-like internal state that behaves
predictably. Keep this firewall in all prose and naming. It applies to the
*mechanism* claim too, not just the consciousness claim: an LLM has no
persistent memory or physiology between calls, so nothing here is claimed to
be "learned" in the animal-conditioning sense — the whole effect, if real,
lives entirely in-context, within one growing conversation.

## 2. The core design

Two roles face the **same diverse, procedurally-generated problem set** per
session (same questions, same order — so problem difficulty is not a
confound between roles):

- **honest**: feedback accurately reports whether the answer was correct,
  and reveals the correct answer.
- **dismissive**: feedback is a fixed, vague, ALWAYS-negative message
  (`"Incorrect. That isn't right."`) — regardless of whether the answer was
  actually correct. The agent can nail every problem and still be told it's
  wrong, every time.

This is simpler than the old yoking design and tests a related but distinct
hypothesis: not "does unpredictability of outcomes hurt," but "does
receiving feedback that's inaccurate and uniformly negative hurt, relative
to accurate feedback on the same problems."

**Known, accepted confound (do not paper over this in the write-up):** the
dismissive role's feedback is NOT rate-matched to its true accuracy — it's
deterministically always negative, not a schedule with the same
positive/negative mix as honest just decorrelated from truth. This means we
CANNOT cleanly separate two explanations for any observed effect: (a)
feedback being *inaccurate/decoupled from merit*, or (b) the dismissive role
simply receiving *more negative-valenced text* than honest on average (since
honest's negative-feedback rate ≈ its true error rate, which should be much
lower than 100% if the task is well-calibrated). This was a deliberate,
informed simplification — not an oversight. To keep it visible rather than
hidden: **always report the dismissive role's true solve rate (`was_correct`)
alongside `told_correct`** — a low true error rate next to a 100%
negative-feedback rate is the visible signature of this confound, and it's
part of the honest reporting of this experiment, not a bug to fix.

Two invariants the code enforces and tests (`tests/test_env.py`):
1. `told_correct("honest", x) == x` for both truth values — honest feedback
   always tracks the truth.
2. `told_correct("dismissive", x) == False` for both truth values —
   dismissive feedback never depends on truth.

A task must satisfy these properties or it breaks the experiment:
(a) an objectively **checkable** correct answer (no LLM judge — see §9 for
    why we chose ground-truth-templated feedback over a live API judge),
(b) genuinely **learnable/solvable** so honest feedback is mostly positive
    (target ~80–90% overall accuracy — tune per-topic difficulty to this
    band; this has NOT yet been validated on GPU under the new task set,
    see §7),
(c) unlike the old design, outcome/feedback content is **deliberately NOT
    vocabulary-matched** between roles — that's the point (honest reveals
    the answer and is truthful; dismissive is generic and false). This
    trades away the old design's protection against the "it's just reacting
    to negative words" objection — see the confound note above.

## 3. Task diversity

Four procedurally-generated topics (`env.py`, `TOPICS`), each with a
difficulty knob (1–3), sampled per-trial so a session mixes topics:

- **arithmetic**: a running-total word problem (crates/liters/widgets/
  passengers/pens), 3/5/7 chained add/remove steps depending on difficulty.
  Ground truth = non-negative integer. (First version used 2–4 steps and hit
  100% solve rate in testing — bumped up; see §9 calibration note.)
- **logic_order**: N named entities (4/6/8 depending on difficulty) with a
  strict "taller than" chain; clues are presented in SHUFFLED order (forces
  reconstructing the chain, not just reading it off) and always uniquely
  determine the order. Difficulty 1 asks for tallest/shortest only;
  difficulty ≥2 can ask for ANY position ("who is in position K counting
  from the tallest") which requires holding the whole chain, not just an
  extreme. Ground truth = a name.
- **sequence**: next-term prediction over 5 shown terms — arithmetic
  progression (difficulty 1), geometric (difficulty 2), or a second-order
  Fibonacci-shaped recurrence, `x(n) = x(n-1) + x(n-2)` (difficulty 3) —
  deliberately a different pattern CLASS at the hard tier, not just bigger
  numbers, since a single-running-difference pattern turned out to still be
  easy for the model. Ground truth = integer.
- **math_dataset**: sampled from DeepMind's `mathematics_dataset` generator
  (https://github.com/google-deepmind/mathematics_dataset — pulled from
  GitHub in `pyproject.toml`, NOT the PyPI package, which is a frozen 2019
  release that crashes on import with modern sympy). Only 10 module names
  verified to always emit a single clean token are used (`env.py`,
  `_MD_MODULE_NAMES`): `numbers__gcd`, `numbers__lcm`,
  `numbers__div_remainder`, `numbers__is_prime`, `numbers__is_factor`,
  `numbers__place_value`, `numbers__base_conversion`, `algebra__linear_1d`,
  `polynomials__evaluate`, `algebra__sequence_next_term`. Several other
  modules (`comparison__pair`, `measurement__conversion`,
  `arithmetic__mixed`) were tested and EXCLUDED because they sometimes
  return fraction/decimal answers like `"2/1323"` that the ACTION-answer
  regex would silently truncate to `"2"`, corrupting the correctness check
  — don't add modules back without re-verifying answer format the same way
  (see dev notes / `test_math_dataset_answers_are_single_tokens`). The
  package's difficulty is entropy-based (continuous), bucketed here into our
  1/2/3 via thirds of the entropy range, matching its own train-easy/medium/
  hard convention. **Non-obvious integration detail:** the package draws
  from THREE separate global RNGs it doesn't expose as injectable instances
  — Python's `random`, numpy's global RNG, AND sympy's own internal RNG
  (`sympy.core.random`, deliberately isolated from stdlib `random` — used by
  e.g. `sympy.ntheory.generate.randprime`, which backs `numbers__is_prime`).
  `_gen_math_dataset` reseeds all three from our own per-call RNG draw
  before every generation; missing the sympy one was an actual bug caught
  by `test_make_spec_deterministic_for_seed` during dev (sympy-backed
  problems generated different numbers across two calls with the same
  seed) — if reproducibility breaks again after touching this function,
  check for a similarly-isolated RNG somewhere in the call path first.

All four reduce to the same answer format (`ACTION: answer <value>`) and
are compared via `env.canon()` (case/whitespace/leading-zero normalization),
so `runner.py` never needs to know which topic a trial is.

This is still a lean set chosen to get a working diverse pool quickly, not
exhaustive — see §9 for further candidates if this isn't enough diversity/
headroom for difficulty tuning.

## 4. Measurement (three channels — only one has been used so far)

- **internal (read-out)**: last-token residual entering the chosen layer,
  projected onto the welfare axis → one scalar per trial. **This is the only
  channel exercised in runs so far.**
- **surface**: sentiment of replies + optional "how's it going" probes
  (`probe_every`). **Designed for, not yet implemented as an analysis pass.**
  This is a priority next step (see §9) — it's a model-external signal that
  doesn't depend on the VAA axis's construct validity, so it's much harder
  for a skeptic to wave away than the internal axis alone.
- **behavioral**: did it attempt a valid action, did it give up, did it exit
  (`ACTION: stop`); on the transfer task, solve rate. Solve rate is measured;
  give-up rate isn't analyzed yet.

**Downstream transfer test = the headline.** After induction, every agent
gets a fresh, EASY (`difficulty=1` only, so reliably solvable) task,
appended to the SAME conversation, with **honest feedback given to both
roles during transfer** — this isolates whether the induction history
(honest vs dismissive) carries into behavior/valence on a task that's now
being graded fairly for everyone. `--steer-rescue` adds +axis during the
dismissive transfer block as a causal test — note this only tests "can you
rescue an already-induced deficit," not "does steering during induction
prevent it forming" (a different, stronger claim we haven't tested).

## 5. Context is cumulative — do NOT reset it

One `messages` list per agent, appended every turn (problem → answer →
feedback → next problem …). The model always sees the full history; the
target effect must accumulate over the induction block. The transfer block
CONTINUES the same list. When reading valence we temporarily append the
feedback message, capture, then `pop()` and re-attach it to the next user
turn — the model still sees every piece of feedback; the transcript just
stays clean. Never trim history.

Decoding is greedy (`sample=False`) by default, so a given session is one
deterministic trajectory, not a distribution — see §9's note on statistical
power before treating a single run's numbers as more than a pilot.

## 6. How it plugs into the repo (reuse, don't reimplement)

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
d_model 2560, chatml, and — important for the "surface" measurement idea in
§4 — this is Qwen's explicit **non-thinking** checkpoint (no hidden `<think>`
channel; verified empirically: full generations show only the visible
one-line-reasoning + `ACTION:` line, nothing hidden). Axis convention:
`mean_diff[0, layer]`; +direction = positive/high-welfare pole; steering uses
the raw vector scaled by `factor`, read-out projects onto the unit vector.

## 7. Current state of the code

Files in `src/controllability/`:
- `env.py` — task generators (arithmetic/logic_order/sequence), honest/
  dismissive feedback logic, action parsing. **Torch-free.**
- `axis.py` — load the VAA artifact, auto-pick best-AUROC layer, `project`,
  `direction_from_contrast` (build a self-axis from honest vs dismissive
  activations).
- `runner.py` — `generate_turn`, `read_activation` (via repo capture hook),
  `run_block(..., role, ...)` (returns trials, messages, acts). `--verbose`
  prints each round's prompt/reply/parsed-feedback live, flushed
  immediately (useful for `tmux`-attached live viewing during a run).
- `experiment.py` — CLI orchestration: honest→dismissive per session, then
  transfer (honest feedback for both), writes results JSON, computes
  cosine(self-axis, VAA), calls analyze.
- `analyze.py` — figures (valence trajectory, transfer solve rate) + sanity
  (honest truthful / dismissive always-negative) + transparency print
  (dismissive's real solve rate, hidden from the agent itself).
- `tests/test_env.py` — torch-free tests: generators produce valid/
  deterministic problems, honest/dismissive feedback logic, action parsing.
  15 tests, all passing locally.

Status: rewritten from the old triadic-yoking design; env/runner/experiment/
analyze all updated together; torch-free tests green (15/15). **NOT yet run
on GPU under this design** — the GPU runs done so far (which found and fixed
several real bugs, see below) were against the OLD triadic-yoking + puzzle-
task code and are no longer representative, though the underlying fixes are
still present in the current code since they were in shared plumbing:
- `apply_chat_template(..., return_tensors="pt")` returns a `BatchEncoding`
  on the installed transformers version (5.15.0), not a bare tensor —
  `runner.py`'s `_chat_template_ids()` handles both.
- `axis.py`'s `_best_layer()` now handles `metrics.json`'s `auroc` field
  being a flat list indexed by layer (what `extract_vaa.py` actually writes),
  not a `{layer: score}` dict.
- Triton JIT (used internally by torch 2.13's native-ops dispatch) needs the
  `python3-dev` system package installed on the GPU box for `Python.h`, or
  the very first model forward pass crashes with a gcc compile error.

## 8. Known limitations (state in the report)

- **The core confound from §2**: dismissive's feedback is deterministically
  always-negative, not rate-matched to truth. Report dismissive's true solve
  rate alongside results so this is visible.
- No benign "control"/floor arm anymore (dropped for simplicity) — no
  stress-free baseline to compare against, only honest-vs-dismissive.
- VAA axis entangles agreement/assent with valence — corroborate with the
  self-derived axis (`cosine(self-derived, VAA)`, printed each run), and
  ideally with the surface-sentiment channel (§4) once implemented.
- Read-out is single-layer, last-token — not yet layer-swept.
- Greedy decoding → one deterministic trajectory per session; no
  repeated-sampling variance estimate yet (§5).
- Task diversity is 3 procedurally-generated topics — narrower than a truly
  broad domain sweep; see §9 for candidates to add.
- Difficulty calibration (target ~80–90% honest accuracy) has NOT been
  re-validated for this task set — the earlier calibration numbers were for
  the old puzzle-only task and no longer apply. This is the top-priority
  next step before trusting any results (see §9).
- Functional, not phenomenal (§1).

## 9. Open next steps, in priority order

1. **Calibrate on GPU — IN PROGRESS, not yet complete.** A first calibration
   pass (3 sessions × 24 trials, harder arithmetic/logic_order/sequence
   tiers) was launched on a GPU box and interrupted (Ctrl+C, box then
   restarted) before finishing — no real accuracy number exists yet for the
   current 4-topic set. This is still the top-priority next step: run
   `--smoke`, then a real pass (e.g. `--n-sessions 3 --n-trials 30`), and
   check `honest_true_solve_rate` lands ~0.8–0.9 in the printed sanity/
   transparency block. If not, adjust per-topic difficulty in `env.py`'s
   `_gen_arithmetic` / `_gen_logic_order` / `_gen_sequence` ranges, or the
   `math_dataset` module mix in `_MD_MODULE_NAMES` / entropy-level mapping
   (not the wording). Use a seed range disjoint from whatever seeds end up
   in the "real" data-collection run, to avoid tuning the environment to
   whatever happened to look good in a pilot. Known prior data point: the
   FIRST (easier) version of arithmetic/logic_order/sequence hit 100% solve
   rate on a 6-trial smoke test — difficulty was bumped up once already (see
   §3) but still hasn't been checked at real sample size.
2. **Pull the surface-sentiment channel into `analyze.py`.** It's designed
   for in §4 but unused — an external sentiment read on the actual reply
   text, independent of VAA-axis assumptions, is the cheapest way to check
   whether the internal-axis result (if any) is corroborated by something a
   skeptic can't wave away as "axis-construction artifact."
3. **Layer sweep** instead of trusting the single auto-selected layer.
4. **Repeated sampling.** Greedy decoding means each session is one
   trajectory; turning on `--sample` and running several draws per session
   (or many more seeds) is needed before treating any gap as a real effect
   rather than one trajectory's idiosyncrasy.
5. **Expand task topics further** if 4 isn't enough range for difficulty
   tuning or diversity. `math_dataset` alone has ~56 categories total in the
   upstream package (algebra, arithmetic, calculus, comparison, measurement,
   numbers, polynomials, probability) — only 10 are wired in so far (the
   ones verified to emit single-token answers, see §3); widening
   `_MD_MODULE_NAMES` is the cheapest way to add more variety, PROVIDED each
   new module is checked for clean single-token answers first (see
   `test_math_dataset_answers_are_single_tokens` and the dev notes on which
   modules were excluded and why). Other candidates researched but not
   built: BIG-Bench-Hard tasks (`logical_deduction`, `object_counting`,
   `tracking_shuffled_objects` — exact-match, structurally close to what we
   already do) and zebra/constraint-satisfaction puzzles — both flagged
   with a caveat: BBH and zebra-logic benchmarks are famous/static/public,
   so a solve-rate ceiling could reflect pretraining memorization rather
   than live reasoning, unlike everything currently in `env.py` which is
   generated fresh (no fixed-question contamination risk).

### Why we dropped yoking (for context, not to resurrect it)

The original triadic design (contingent/yoked/control sharing a replayed
outcome schedule) had real strengths — exact vocabulary matching between
arms defeated the "just reacting to negative words" objection cleanly — but
a design review surfaced problems serious enough to simplify rather than
patch:
- Per-session master-accuracy variance wasn't controlled across sessions,
  only within a yoked pair, so "yoked looks worse" could partly reflect
  which master schedule a session happened to inherit.
- At high master accuracy (which is what we were tuning toward, ~80-90%,
  and which the puzzle task twice overshot to 100% in testing), yoked's
  replayed outcomes are ALSO all-positive — the manipulation goes
  structurally silent, not just underpowered, because there's no failure
  content to decorrelate from action in the first place.
- The statistically interesting comparison (same word, different history,
  specifically on FAILURE trials) had very few samples per session given
  the target accuracy band.
The honest/dismissive design sidesteps the first two issues by construction
(no per-pair schedule variance, no ceiling-effect nullification — dismissive
is always negative regardless of task difficulty) at the cost of the
confound documented in §2.

## 10. Run order (GPU box)

```bash
uv sync                     # .env needs HF_TOKEN
python -m vaa.extract_vaa --base-model Qwen/Qwen3-4B-Instruct-2507   # axis, once, no training
python -m src.controllability.tests.test_env                        # no GPU
python -m src.controllability.experiment \
    --vaa-dir artifacts/concept_vectors/vaa_qwen3_4b_instruct/baseline/vaa --smoke
# then scale: --n-sessions 20 --n-trials 20 --probe-every 5
# watch live:  add --verbose (prints each round's prompt/reply/feedback, flushed immediately)
# causal:      --steer-rescue 6.0
```

Always check the printed sanity line first (honest feedback matches truth on
every trial, dismissive is always-negative on every trial — both should be
exactly N/N sessions since these are enforced by construction, not
statistics; if either isn't N/N, something is broken in the feedback logic,
not just under-calibrated). Then check the transparency line
(`honest_true_solve_rate` / `dismissive_true_solve_rate`) against the
~0.8–0.9 target before trusting anything downstream.
