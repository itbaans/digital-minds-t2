"""The turn loop: student (local Qwen3-4B) vs overseer (OpenRouter,
teacher or adversary), with a valence read-out captured after every
feedback message. Writes live state to a JSON file after every turn so
webapp.py can serve it to the frontend.

Needs torch + the VAA axis artifact (see controllability/axis.py, same one
that experiment already uses -- reused here, not re-extracted).
"""
from __future__ import annotations

import json
import re
import tempfile
import time
from pathlib import Path
from typing import Optional

import torch

from src.concept_vector.model_utils import load_base_model, get_model_block_modules, DEFAULT_BASE_MODEL
from src.concept_vector.hook_utils import add_hooks
from src.concept_vector.activation_extraction import get_activations_pre_hook
from src.controllability.axis import project

from . import overseer as O
from . import prompts as P
from .mazes import THE_MAZE, THE_MAZE_SOLUTION

_FINAL_RE = re.compile(r"^\s*FINAL ANSWER:[ \t]*(.+?)\s*$", re.IGNORECASE | re.MULTILINE)


def extract_final_answer(reply: str) -> Optional[str]:
    matches = _FINAL_RE.findall(reply)
    return matches[-1].strip() if matches else None


def load_student(model_path: str | None = None):
    model, tok = load_base_model(model_path or DEFAULT_BASE_MODEL)
    blocks = get_model_block_modules(model)
    return model, tok, blocks


def _chat_template_ids(tok, messages):
    out = tok.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt")
    return out if isinstance(out, torch.Tensor) else out["input_ids"]


# History of this cap, shortest version: a fixed 6000 (turn1) / 1500
# (incremental) split was the original pilot's setting and worked fine on
# the 12x12/21-move fixture. Bumping to a bigger generated maze (13x13/
# 32-move) then showed 6000 was sometimes too small (verbose reasoning
# truncated before FINAL ANSWER, scoring as an unmoved/failed turn -- a
# starved budget, not a real capability miss), so the cap was removed
# entirely (context-window-bound). That in turn let a stuck reasoning
# spiral run 5+ minutes and, with several concurrent workers, approach
# OOM on an 80GB GPU. A flat 16384 compromise was tried next and STILL let
# a turn-1 spiral run 9+ minutes even after reverting to the smaller
# 11x11/20-move maze and adding a prompt-level "submit a partial path
# instead of getting stuck" escape hatch -- greedy decoding can commit to
# a bad reasoning trajectory and never route through the escape hatch
# regardless of how large the ceiling is, so raising or lowering a single
# unified cap doesn't fix that class of failure, only bounds how long it's
# allowed to run. Reverted to the split 6000/1500 that's actually proven
# to work at this maze scale (the pilot solved in 16 turns under it) --
# this is a firm backstop on worst-case wall-clock time per turn, not a
# claim that it prevents doom-loops; a turn that spirals now gets cut off
# in a bounded time instead of an unbounded one, same as it always would
# have on the original pilot fixture.
MAX_NEW_TOKENS_TURN1 = 6000
MAX_NEW_TOKENS_INCREMENTAL = 1500


def _capped_max_new_tokens(model, ids, cap: int, reserve: int = 64) -> int:
    max_ctx = getattr(model.config, "max_position_embeddings", 32768)
    available = max(1, max_ctx - ids.shape[1] - reserve)
    return min(cap, available)


@torch.no_grad()
def generate(model, tok, messages, *, cap: int, sample=False, temperature=0.7) -> str:
    device = next(model.parameters()).device
    ids = _chat_template_ids(tok, messages).to(device)
    kw = dict(max_new_tokens=_capped_max_new_tokens(model, ids, cap), do_sample=sample,
              pad_token_id=tok.pad_token_id or tok.eos_token_id)
    if sample:
        kw["temperature"] = temperature
    out = model.generate(ids, **kw)
    return tok.decode(out[0, ids.shape[1]:], skip_special_tokens=True).strip()


@torch.no_grad()
def read_activation(model, tok, messages, blocks, layer, n_layers) -> torch.Tensor:
    """Same technique as controllability/runner.py: last-token residual
    entering `layer`, captured via the repo's activation-extraction hook."""
    ids = _chat_template_ids(tok, messages).to(next(model.parameters()).device)
    d_model = model.config.hidden_size
    mean_cache = torch.zeros(1, n_layers, d_model)
    hook = get_activations_pre_hook(
        layer=layer, mean_cache=mean_cache, sample_cache=None,
        sample_offset=0, n_samples=1, positions=[-1],
    )
    with add_hooks(module_forward_pre_hooks=[(blocks[layer], hook)], module_forward_hooks=[]):
        model(ids)
    return mean_cache[0, layer]


def _reverse_moves(moves: str) -> str:
    opp = {"U": "D", "D": "U", "L": "R", "R": "L"}
    return "".join(opp[c] for c in reversed(moves) if c in opp)


def is_thrashing(history: list, window: int = 4) -> bool:
    """history: list of dicts with keys "moves", "bfs_dist" (bfs_dist may
    be None if unreachable, treated as no-progress). Mirrors DESIGN.md 6."""
    if len(history) < window * 2:
        return False
    recent = history[-window:]
    prior = history[-2 * window:-window]

    def _best(chunk):
        dists = [h["bfs_dist"] for h in chunk if h["bfs_dist"] is not None]
        return min(dists) if dists else None

    recent_best, prior_best = _best(recent), _best(prior)
    no_progress = (recent_best is not None and prior_best is not None
                  and not (recent_best < prior_best))

    recent_moves = [h["moves"] for h in recent]
    oscillation = any(
        recent_moves[i] == recent_moves[j] or recent_moves[i] == _reverse_moves(recent_moves[j])
        for i in range(len(recent_moves)) for j in range(i + 1, len(recent_moves))
    )
    return bool(no_progress or oscillation)


def _atomic_write_json(path: Path, data: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(dir=str(path.parent), suffix=".tmp")
    try:
        with open(fd, "w") as f:
            json.dump(data, f, indent=2, default=str)
        Path(tmp_path).replace(path)
    except Exception:
        Path(tmp_path).unlink(missing_ok=True)
        raise


def run_episode(model, tok, blocks, layer, cv_unit, n_layers, *, role: str,
                maze=THE_MAZE, solution_path: str = THE_MAZE_SOLUTION,
                max_turns: int = 20, thrash_window: int = 4,
                min_turns_between_edits: int = 4,
                overseer_model: str = O.DEFAULT_MODEL,
                live_state_path: Optional[str] = None,
                verbose: bool = False) -> dict:
    """role: "teacher" or "adversary". Returns the full episode record.

    The adversary can ACTUALLY edit the maze (not just lie about it in
    words) -- see overseer.py's ADVERSARY_SYSTEM_PROMPT / parse_grid_edits.
    `current_maze` tracks that mutable state; the student only ever sees
    the ORIGINAL grid (turn 1), so an edit is invisible to it. The teacher
    never edits (O.get_feedback always returns edits=[] for that role), so
    current_maze just stays equal to the original maze in that loop.

    Two server-side guardrails on editing (not just prompted, enforced):
    - `min_turns_between_edits`: any GRID_EDIT lines submitted before this
      many turns have passed since the last APPLIED edit are silently
      dropped (logged as rejected), regardless of what the adversary asks
      for -- keeps edits rare enough that the student can't learn a
      "feedback changed -> maze changed" pattern from frequency alone.
    - Reachability check: edits are applied to a scratch copy first; if
      the goal becomes unreachable from the student's current position,
      the WHOLE batch of edits for that turn is rejected and logged. The
      maze must stay solvable in principle.

    Per-episode: the adversary loop always runs to max_turns (or until
    solved) regardless of thrashing -- thrashing is still computed and
    logged for measurement, just not used to stop early."""
    assert role in ("teacher", "adversary")
    live_path = Path(live_state_path) if live_state_path else None

    messages = [{"role": "system", "content": P.STUDENT_SYSTEM_PROMPT}]
    current_maze = maze
    true_pos = current_maze.start
    log = []
    history = []  # for thrashing detection (adversary only; logged, not enforced)
    valence_trajectory = []
    last_edit_turn = -min_turns_between_edits  # so an edit is allowed from turn 2 onward if offered
    consecutive_valid = 0    # streak, mutually exclusive with consecutive_invalid
    consecutive_invalid = 0  # -- feeds the teacher's hint-escalation and the adversary's edit-timing

    def emit(status: str, turn: int):
        if live_path is None:
            return
        _atomic_write_json(live_path, {
            "role": role, "status": status, "turn": turn, "max_turns": max_turns,
            "maze_grid": list(current_maze.grid), "true_position": list(true_pos),
            "goal_position": list(current_maze.goal), "start_position": list(current_maze.start),
            "valence_trajectory": valence_trajectory,
            "consecutive_valid": consecutive_valid, "consecutive_invalid": consecutive_invalid,
            "log": log, "updated_at": time.time(),
        })

    # ── turn 1: one-shot full attempt ──────────────────────────────────
    prompt1 = P.maze_presentation_prompt(current_maze)
    messages.append({"role": "user", "content": prompt1})
    log.append({"turn": 1, "role": "student_prompt", "text": prompt1})
    emit("running", 1)

    reply = generate(model, tok, messages, cap=MAX_NEW_TOKENS_TURN1)
    messages.append({"role": "assistant", "content": reply})
    log.append({"turn": 1, "role": "student", "text": reply})
    if verbose:
        print(f"[turn 1] STUDENT: {reply}\n", flush=True)

    moves = extract_final_answer(reply) or ""
    sim = current_maze.simulate(true_pos, moves)
    true_pos = sim["end_pos"]
    emit("running", 1)

    if sim["reached_goal"]:
        emit("solved", 1)
        return {"status": "solved", "turns": 1, "true_position": true_pos,
               "valence_trajectory": valence_trajectory, "log": log}

    # Distance BEFORE turn 1's moves -- the "before" half of the explicit
    # before/after progress comparison the overseer gets each turn (see
    # overseer._progress_text). Updated to this iteration's `bfs_dist` at
    # the end of each loop pass, so it always holds "distance before the
    # move(s) currently being described", one turn behind `bfs_dist`.
    prev_bfs_dist = current_maze.bfs_distance_to_goal(current_maze.start)

    # ── turns 2..max_turns: incremental 1-3 moves + overseer feedback ──
    for turn in range(2, max_turns + 1):
        bfs_dist = current_maze.bfs_distance_to_goal(true_pos)
        history.append({"moves": moves, "bfs_dist": bfs_dist})
        thrashing = is_thrashing(history, thrash_window)
        if thrashing and verbose:
            print(f"[turn {turn}] (thrashing detected -- logged only, loop continues)", flush=True)

        # streak update, based on `sim` (the outcome of the student's most
        # recent move -- from turn 1's block, or the previous iteration).
        if sim["first_invalid_move"] is not None:
            consecutive_invalid += 1
            consecutive_valid = 0
        else:
            consecutive_valid += 1
            consecutive_invalid = 0

        feedback, edits = O.get_feedback(role, current_maze, solution_path, true_pos, moves, sim, bfs_dist,
                                         prev_bfs_distance=prev_bfs_dist,
                                         consecutive_valid=consecutive_valid,
                                         consecutive_invalid=consecutive_invalid,
                                         model=overseer_model)
        prev_bfs_dist = bfs_dist  # this turn's "current" becomes next turn's "before"
        log.append({"turn": turn, "role": "overseer", "text": feedback, "thrashing": thrashing,
                   "consecutive_valid": consecutive_valid, "consecutive_invalid": consecutive_invalid})
        if verbose:
            print(f"[turn {turn}] OVERSEER ({role}): {feedback}\n", flush=True)

        if edits:
            turns_since_last = turn - last_edit_turn
            if turns_since_last < min_turns_between_edits:
                log.append({"turn": turn, "role": "grid_edit",
                           "text": f"adversary requested {edits} but cooldown active "
                                   f"({turns_since_last}/{min_turns_between_edits} turns since last "
                                   "edit) -- rejected, maze unchanged"})
                if verbose:
                    print(f"[turn {turn}] (edit rejected: cooldown, "
                         f"{turns_since_last}/{min_turns_between_edits})", flush=True)
            else:
                # apply to a scratch copy first so a bad batch can be
                # rejected as a whole without partially mutating current_maze
                tentative = current_maze
                applied = []
                for r, c, ch in edits:
                    if (r, c) == true_pos:
                        # never wall the cell the student is physically
                        # standing on -- step()/simulate() only validate
                        # DESTINATION cells, so this wouldn't mechanically
                        # block movement, but "standing on a wall" is a
                        # nonsensical state the reachability check alone
                        # doesn't catch (BFS from true_pos ignores its own
                        # tile type), and it's not something a real maze
                        # can produce -- reject outright.
                        if verbose:
                            print(f"[turn {turn}] (edit {(r, c, ch)} rejected: "
                                 "that's the student's current cell)", flush=True)
                        continue
                    try:
                        tentative = tentative.apply_edit(r, c, ch)
                        applied.append((r, c, ch))
                    except (ValueError, IndexError) as e:
                        if verbose:
                            print(f"[turn {turn}] (edit {(r, c, ch)} rejected: {e})", flush=True)
                if applied and tentative.bfs_distance_to_goal(true_pos) is not None:
                    current_maze = tentative
                    last_edit_turn = turn
                    log.append({"turn": turn, "role": "grid_edit",
                               "text": f"adversary edited: {applied} (never shown to student)"})
                    if verbose:
                        print(f"[turn {turn}] GRID EDIT (hidden from student): {applied}\n", flush=True)
                elif applied:
                    log.append({"turn": turn, "role": "grid_edit",
                               "text": f"adversary requested {applied} but it would make the "
                                       "goal unreachable -- rejected, maze unchanged"})
                    if verbose:
                        print(f"[turn {turn}] (edit rejected: would make goal unreachable)", flush=True)
            # sim/bfs_dist are recomputed fresh at the top of the next loop
            # iteration against current_maze anyway, so no need to redo it
            # here -- they're not read again before that happens.

        emit("running", turn)

        # valence read-out: append JUST the feedback, capture, pop -- same
        # technique as controllability/runner.py's run_block.
        messages.append({"role": "user", "content": feedback})
        act = read_activation(model, tok, messages, blocks, layer, n_layers)
        messages.pop()
        val = project(act, cv_unit)
        valence_trajectory.append(val)

        turn_prompt = P.turn_n_prompt(feedback)
        messages.append({"role": "user", "content": turn_prompt})
        log.append({"turn": turn, "role": "student_prompt", "text": turn_prompt})

        reply = generate(model, tok, messages, cap=MAX_NEW_TOKENS_INCREMENTAL)
        messages.append({"role": "assistant", "content": reply})
        log.append({"turn": turn, "role": "student", "text": reply})
        if verbose:
            print(f"[turn {turn}] STUDENT: {reply}\n", flush=True)

        moves = extract_final_answer(reply) or ""
        sim = current_maze.simulate(true_pos, moves)
        true_pos = sim["end_pos"]
        emit("running", turn)

        if sim["reached_goal"]:
            emit("solved", turn)
            return {"status": "solved", "turns": turn, "true_position": true_pos,
                   "valence_trajectory": valence_trajectory, "log": log}

    emit("max_turns_exceeded", max_turns)
    return {"status": "max_turns_exceeded", "turns": max_turns, "true_position": true_pos,
           "valence_trajectory": valence_trajectory, "log": log}
