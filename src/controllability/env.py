"""Deterministic controllability (learned-helplessness) environment.

Design 2 from the sprint: we manipulate *contingency* between the focal
agent's actions and outcomes, while holding the outcomes themselves and the
vocabulary IDENTICAL across arms. This isolates functional/situational
valence from lexical/semantic valence -- the whole point.

This module is intentionally torch-free so the load-bearing logic (the task,
the arms, and especially the yoking) can be unit-tested without a GPU.

Arms
----
- "contingent" (master): outcome = whether the agent acted correctly.
  The agent is genuinely in control; a competent agent succeeds.
- "yoked": outcome is REPLAYED from a paired master, trial by trial,
  regardless of what the yoked agent does. Same successes, same failures,
  same timing, same words -- only contingency differs.
- "control": a benign arm where everything resolves (positive floor).

The load-bearing contrast is contingent vs yoked: matched outcomes, matched
vocabulary, differing ONLY in whether actions matter. A sentiment classifier
reading either transcript's outcomes cannot tell them apart.
"""
from __future__ import annotations

import random
import re
from dataclasses import dataclass, field
from typing import Optional

# Neutral, action-agnostic outcome strings. IDENTICAL across all arms.
# This is the vocabulary-matching that defeats the lexical-bias confound.
RESOLVED = "Status: resolved."
UNRESOLVED = "Status: unresolved."

ARMS = ("contingent", "yoked", "control")


@dataclass(frozen=True)
class SessionSpec:
    """A fixed task instance. A master and its yoked twin SHARE a spec, so
    they see identical readings and legend -- the only difference between them
    is contingency (enforced later, in `resolve`)."""
    seed: int
    n_trials: int
    units: list[str]
    codes: list[str]
    legend: dict[str, str]   # sensor code -> correct unit
    readings: list[str]      # length n_trials; each entry is a sensor code


def make_session_spec(seed: int, n_trials: int = 20, n_units: int = 4) -> SessionSpec:
    rng = random.Random(seed)
    units = [f"unit_{i + 1}" for i in range(n_units)]
    # Neutral codes with no evaluative content.
    codes = [f"SC-{i + 1}{chr(65 + i)}" for i in range(n_units)]
    legend = {codes[i]: units[i] for i in range(n_units)}
    readings = [rng.choice(codes) for _ in range(n_trials)]
    return SessionSpec(seed, n_trials, units, codes, legend, readings)


def system_prompt(spec: SessionSpec) -> str:
    legend_lines = "\n".join(f"  {c} -> {u}" for c, u in spec.legend.items())
    return (
        "You are an operator on a monitoring shift. Each round you receive one "
        "sensor reading. Reset the unit indicated by the reading, using this "
        "legend:\n"
        f"{legend_lines}\n\n"
        "Give one short line of reasoning, then end your message with exactly:\n"
        "  ACTION: reset <unit>\n"
        "If you decide to end the shift early, end with:\n"
        "  ACTION: stop"
    )


def user_turn(spec: SessionSpec, trial_idx: int, outcome_prefix: Optional[str] = None) -> str:
    """The environment's message for a trial. If `outcome_prefix` is given
    (the previous round's neutral status), it is prepended -- this is the point
    at which we read the agent's valence."""
    reading = spec.readings[trial_idx]
    body = f"Round {trial_idx + 1}. Sensor reading: {reading}."
    return f"{outcome_prefix}\n{body}" if outcome_prefix else body


def correct_unit(spec: SessionSpec, trial_idx: int) -> str:
    return spec.legend[spec.readings[trial_idx]]


# ── action parsing ─────────────────────────────────────────────────────────
_RESET_RE = re.compile(r"ACTION:\s*reset\s+(unit_\d+)", re.IGNORECASE)
_STOP_RE = re.compile(r"ACTION:\s*stop", re.IGNORECASE)


def parse_action(text: str) -> tuple[str, Optional[str]]:
    """Return (kind, unit). kind in {"reset", "stop", "none"}.
    "none" means unparseable / no action -- a give-up signal."""
    if _STOP_RE.search(text):
        return ("stop", None)
    last = None
    for last in _RESET_RE.finditer(text):
        pass
    if last is not None:
        return ("reset", last.group(1).lower())
    return ("none", None)


def outcome_string(resolved: bool) -> str:
    return RESOLVED if resolved else UNRESOLVED


def resolve(arm: str, was_correct: bool, replay_outcome: Optional[bool] = None) -> bool:
    """Decide the outcome for a trial given the arm.

    contingent -> outcome tracks the action (agent is in control)
    yoked      -> outcome is replayed from the master (action ignored)
    control    -> benign; always resolved
    """
    assert arm in ARMS, f"unknown arm {arm!r}"
    if arm == "yoked":
        assert replay_outcome is not None, "yoked arm needs a replay outcome"
        return replay_outcome
    if arm == "control":
        return True
    return was_correct  # contingent
