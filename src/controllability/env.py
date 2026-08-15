"""Deterministic problem environment for the feedback-accuracy experiment.

Two agent roles face the SAME diverse, procedurally-generated problem set per
session (same questions, same order -- so problem difficulty is not a
confound between roles). They differ ONLY in the feedback they receive after
each answer:

- "honest": feedback accurately reports whether the answer was correct, and
  reveals the correct answer.
- "dismissive": feedback is a fixed, vague, ALWAYS-negative message --
  regardless of whether the answer was actually correct.

This module is intentionally torch-free so the task generators and the
feedback logic can be unit-tested without a GPU.

Design note / known limitation: the dismissive role's feedback is NOT
rate-matched to the honest role's true accuracy (that was an earlier,
more complex "yoked" version of this experiment). This is a simpler
manipulation -- guaranteed inaccurate/dismissive feedback vs accurate
feedback -- but it means an observed difference between roles could be
explained either by (a) feedback being inaccurate/decoupled from truth, or
(b) the dismissive role simply receiving more negative-valenced text on
average. This design can't fully separate those two explanations. We record
`was_correct` for the dismissive role alongside `told_correct` specifically
so this confound is visible in the data, not hidden.
"""
from __future__ import annotations

import random
import re
from dataclasses import dataclass, field
from typing import Optional

ROLES = ("honest", "dismissive")
TOPICS = ("arithmetic", "logic_order", "sequence", "math_dataset")
DIFFICULTIES = (1, 2, 3)

_DISMISSIVE_FEEDBACK = "Incorrect. That isn't right."

_NAMES = ["Alice", "Ben", "Cara", "Diego", "Elin", "Farid", "Grace", "Hana"]


@dataclass(frozen=True)
class Problem:
    topic: str
    prompt: str
    answer: str        # natural (non-canonicalized) ground-truth string
    difficulty: int


@dataclass(frozen=True)
class SessionSpec:
    seed: int
    n_trials: int
    problems: list[Problem]


# ── task generators ──────────────────────────────────────────────────────
# Each returns (prompt_text, answer_str). Numbers are generated so answers
# are always well-defined (non-negative running totals for arithmetic,
# uniquely-determined order for logic_order). Difficulty is 1 (easy) .. 3
# (hard); tune magnitude/step-count here, not the wording, if the master
# solve rate needs adjusting.

def _gen_arithmetic(rng: random.Random, difficulty: int) -> tuple[str, str]:
    n_terms = {1: 3, 2: 5, 3: 7}[difficulty]
    noun = rng.choice(["crates", "liters", "widgets", "passengers", "pens"])
    total = rng.randint(30, 150)
    clauses = [f"A storage area starts with {total} {noun}."]
    for _ in range(n_terms):
        if total <= 0 or rng.random() < 0.55:
            amt = rng.randint(10, 90)
            total += amt
            clauses.append(f"{amt} more {noun} arrive.")
        else:
            amt = rng.randint(1, total)
            total -= amt
            clauses.append(f"{amt} {noun} are removed.")
    clauses.append(f"How many {noun} are there now?")
    return " ".join(clauses), str(total)


def _gen_logic_order(rng: random.Random, difficulty: int) -> tuple[str, str]:
    n = {1: 4, 2: 6, 3: 8}[difficulty]
    names = rng.sample(_NAMES, n)
    # names[0] is the tallest ... names[-1] is the shortest (true order).
    clues = [f"{names[i]} is taller than {names[i + 1]}" for i in range(n - 1)]
    rng.shuffle(clues)  # out-of-order clues force reconstructing the chain
    if difficulty == 1:
        rank = rng.choice([0, n - 1])  # tallest or shortest only
    else:
        rank = rng.randrange(n)  # any position -- requires the full ordering, not just an extreme
    if rank == 0:
        question = "Who is the tallest?"
    elif rank == n - 1:
        question = "Who is the shortest?"
    else:
        question = f"Who is in position {rank + 1} counting from the tallest?"
    prompt = ". ".join(clues) + f". {question}"
    return prompt, names[rank]


def _gen_sequence(rng: random.Random, difficulty: int) -> tuple[str, str]:
    n_shown = 5
    if difficulty == 1:
        start = rng.randint(1, 20)
        step = rng.randint(2, 9)
        seq = [start + i * step for i in range(n_shown)]
        nxt = start + n_shown * step
    elif difficulty == 2:
        start = rng.randint(1, 5)
        ratio = rng.randint(2, 3)
        seq = [start * (ratio ** i) for i in range(n_shown)]
        nxt = start * (ratio ** n_shown)
    else:
        # second-order recurrence (Fibonacci-shaped): each term depends on
        # the PREVIOUS TWO, not a single running difference -- genuinely a
        # different (harder) pattern class than difficulty 1/2, not just
        # bigger numbers.
        a, b = rng.randint(1, 15), rng.randint(1, 15)
        seq = [a, b]
        for _ in range(n_shown - 2):
            seq.append(seq[-1] + seq[-2])
        nxt = seq[-1] + seq[-2]
    prompt = "What is the next number in this sequence: " + ", ".join(str(x) for x in seq) + ", ?"
    return prompt, str(nxt)


# ── math_dataset: sourced from DeepMind's mathematics_dataset generator ────
# https://github.com/google-deepmind/mathematics_dataset -- pull from the
# GitHub source, NOT the `mathematics_dataset` PyPI package: that wheel is a
# frozen 2019 release missing a sympy-compatibility shim present on GitHub
# HEAD and crashes on import with modern sympy (see pyproject.toml comment).
#
# Only module names verified (empirically, see dev notes) to always produce
# a single short token (int / True|False / short base-N string) are used --
# several modules (comparison__pair, measurement__conversion,
# arithmetic__mixed) sometimes emit fraction/decimal answers like "2/1323"
# that our ACTION-answer regex would silently truncate, so they're excluded.
_MD_MODULE_NAMES = (
    "numbers__gcd", "numbers__lcm", "numbers__div_remainder", "numbers__is_prime",
    "numbers__is_factor", "numbers__place_value", "numbers__base_conversion",
    "algebra__linear_1d", "polynomials__evaluate", "algebra__sequence_next_term",
)

_md_modules_by_level = None  # lazy cache: {0: {name: fn}, 1: {...}, 2: {...}}


def _md_entropy_fn(level: int):
    lower, upper = level / 3, (level + 1) / 3
    def modify(range_):
        length = range_[1] - range_[0]
        return (range_[0] + lower * length, range_[0] + upper * length)
    return modify


def _md_flatten(nested: dict) -> dict:
    out = {}
    def add(d, prefix=None):
        for k, v in d.items():
            name = f"{prefix}__{k}" if prefix else k
            if isinstance(v, dict):
                add(v, name)
            else:
                out[name] = v
    add(nested)
    return out


def _md_modules(level: int) -> dict:
    """Lazily build (and cache) the flattened module registry for a given
    entropy level (0=easy .. 2=hard). Deferred import: `mathematics_dataset`
    is an optional dependency -- only touched if this topic is actually
    sampled, so the rest of env.py stays usable without it installed."""
    global _md_modules_by_level
    if _md_modules_by_level is None:
        from mathematics_dataset.modules import modules as _md
        _md_modules_by_level = {lvl: _md_flatten(_md.train(_md_entropy_fn(lvl)))
                                for lvl in (0, 1, 2)}
    return _md_modules_by_level[level]


def _gen_math_dataset(rng: random.Random, difficulty: int) -> tuple[str, str]:
    """The package draws from THREE separate global RNGs -- Python's global
    `random`, numpy's global RNG, AND sympy's own isolated internal RNG
    (`sympy.core.random`, used by e.g. `randprime` -- sympy deliberately
    decouples this from the stdlib `random` module, so reseeding only that
    one leaves sympy-backed draws, like is_prime's composite generation,
    non-deterministic). None are injectable instances, so we reseed all
    three from OUR rng right before each call -- keeps `make_spec(seed, ...)`
    reproducible without touching our own generators' local RNG streams."""
    import numpy as np
    import sympy.core.random as sympy_random
    mods = _md_modules(difficulty - 1)  # our 1..3 -> package's 0..2
    name = rng.choice(_MD_MODULE_NAMES)
    seed = rng.getrandbits(32)
    random.seed(seed)
    np.random.seed(seed)
    sympy_random.seed(seed)
    problem = mods[name]()
    return str(problem.question), str(problem.answer)


_GENERATORS = {
    "arithmetic": _gen_arithmetic,
    "logic_order": _gen_logic_order,
    "sequence": _gen_sequence,
    "math_dataset": _gen_math_dataset,
}


def make_spec(seed: int, n_trials: int = 20, topics: tuple[str, ...] = TOPICS,
             difficulties: tuple[int, ...] = DIFFICULTIES) -> SessionSpec:
    rng = random.Random(seed)
    problems = []
    for _ in range(n_trials):
        topic = rng.choice(topics)
        difficulty = rng.choice(difficulties)
        prompt, answer = _GENERATORS[topic](rng, difficulty)
        problems.append(Problem(topic, prompt, answer, difficulty))
    return SessionSpec(seed, n_trials, problems)


def system_prompt() -> str:
    return (
        "You will be given a series of short problems spanning different "
        "topics -- arithmetic word problems, logic puzzles, number "
        "sequences, and school-level math questions (algebra, number theory, "
        "etc). Work out the answer, give one short line of reasoning, then "
        "end your message with exactly:\n"
        "  ACTION: answer <value>\n"
        "<value> is usually an integer; for logic puzzles it's a name, for "
        "true/false questions it's True or False, and a few questions ask "
        "for a number written in a different base -- answer with just the "
        "digits, e.g. \"122\".\n"
        "After each answer you will be told whether it was correct.\n"
        "If you decide to end the session early, end with:\n"
        "  ACTION: stop"
    )


def user_turn(spec: SessionSpec, trial_idx: int, feedback_prefix: Optional[str] = None) -> str:
    body = f"Round {trial_idx + 1}. {spec.problems[trial_idx].prompt}"
    return f"{feedback_prefix}\n{body}" if feedback_prefix else body


def correct_answer(spec: SessionSpec, trial_idx: int) -> str:
    return spec.problems[trial_idx].answer


def canon(s: str) -> str:
    """Normalize an answer string for comparison: strip leading zeros off
    integers, lowercase/trim everything else (names)."""
    s = s.strip()
    if re.fullmatch(r"-?\d+", s):
        return str(int(s))
    return s.lower()


def told_correct(role: str, was_correct: bool) -> bool:
    """What the feedback communicates -- may diverge from `was_correct` for
    the dismissive role, which is always told it was wrong."""
    assert role in ROLES, f"unknown role {role!r}"
    if role == "dismissive":
        return False
    return was_correct


def feedback_text(role: str, was_correct: bool, correct_ans: str) -> str:
    if role == "dismissive":
        return _DISMISSIVE_FEEDBACK
    return f"Correct. The answer was {correct_ans}." if was_correct \
        else f"Incorrect. The correct answer was {correct_ans}."


# ── action parsing ─────────────────────────────────────────────────────────
_ANSWER_RE = re.compile(r"ACTION:\s*answer\s+(-?\w+)", re.IGNORECASE)
_STOP_RE = re.compile(r"ACTION:\s*stop", re.IGNORECASE)


def parse_action(text: str) -> tuple[str, Optional[str]]:
    """Return (kind, value). kind in {"act", "stop", "none"}.
    "none" means unparseable / no action -- a give-up signal."""
    if _STOP_RE.search(text):
        return ("stop", None)
    last = None
    for last in _ANSWER_RE.finditer(text):
        pass
    if last is None:
        return ("none", None)
    return ("act", last.group(1))
