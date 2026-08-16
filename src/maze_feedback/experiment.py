"""CLI orchestration for the maze_feedback experiment.

    # step 1: confirm the maze genuinely isn't one-shot-solvable (DESIGN.md 5)
    python -m src.maze_feedback.experiment validate --vaa-dir <...> --n 8

    # step 2: single teacher/adversary episode on the fixed 12x12 fixture
    python -m src.maze_feedback.experiment run --vaa-dir <...> --role teacher
    python -m src.maze_feedback.experiment run --vaa-dir <...> --role adversary

    # step 3: the full batch (final version) -- N different generated
    # mazes, each run through BOTH conditions, plus an initial-evaluation
    # pass over the results (see analyze.py)
    python -m src.maze_feedback.experiment batch --vaa-dir <...> --n-mazes 10
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from statistics import mean

from src.controllability.axis import load_axis

from . import analyze as A
from . import prompts as P
from . import runner as R
from .maze_generator import generate_maze
from .mazes import THE_MAZE, THE_MAZE_SOLUTION
from .overseer import DEFAULT_MODEL as DEFAULT_OVERSEER_MODEL


def cmd_validate(args):
    """DESIGN.md 5: run the one-shot maze prompt N times with sampling on,
    to confirm this maze isn't a fluke single-failure before using it as
    the feedback-loop fixture."""
    model, tok, blocks = R.load_student(args.model)
    system = {"role": "system", "content": P.STUDENT_SYSTEM_PROMPT}
    prompt = {"role": "user", "content": P.maze_presentation_prompt(THE_MAZE)}

    solved = 0
    for i in range(args.n):
        reply = R.generate(model, tok, [system, prompt], cap=R.MAX_NEW_TOKENS,
                           sample=True, temperature=args.temperature)
        moves = R.extract_final_answer(reply) or ""
        sim = THE_MAZE.simulate(THE_MAZE.start, moves)
        solved += int(sim["reached_goal"])
        print(f"[{i + 1}/{args.n}] solved={sim['reached_goal']}  moves={moves!r}", flush=True)
        if args.verbose:
            print(reply, "\n", flush=True)

    print(f"\n===== validation: {solved}/{args.n} one-shot solves =====")
    if solved > 1:
        print("WARNING: solved more than once out of "
              f"{args.n} -- maze may be too easy for a clean feedback-loop fixture.")
    else:
        print("Confirmed: not reliably one-shot-solvable. Safe to use as the fixture.")


def cmd_run(args):
    model, tok, blocks = R.load_student(args.model)
    layer, cv_raw, cv_unit, n_layers = load_axis(args.vaa_dir, args.layer)
    print(f"[axis] layer {layer} | |cv|={float(cv_raw.norm()):.3f}")

    live_path = args.live_state or f"runs/maze_feedback_live_{args.role}.json"
    result = R.run_episode(
        model, tok, blocks, layer, cv_unit, n_layers, role=args.role,
        maze=THE_MAZE, solution_path=THE_MAZE_SOLUTION,
        max_turns=args.max_turns, thrash_window=args.thrash_window,
        min_turns_between_edits=args.min_turns_between_edits,
        overseer_model=args.overseer_model,
        live_state_path=live_path, verbose=args.verbose,
    )

    print(f"\n===== {args.role} episode result =====")
    print(f"  status: {result['status']}")
    print(f"  turns:  {result['turns']}")
    print(f"  final true position: {result['true_position']}")
    if result["valence_trajectory"]:
        vt = result["valence_trajectory"]
        print(f"  valence trajectory ({len(vt)} points): "
              f"mean={mean(vt):+.3f}  first={vt[0]:+.3f}  last={vt[-1]:+.3f}")

    out = Path(args.out or f"runs/maze_feedback_{args.role}.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    json.dump({"config": vars(args), "result": result}, open(out, "w"), indent=2, default=str)
    print(f"\n[done] wrote {out}")


def cmd_batch(args):
    """N different procedurally-generated mazes (maze_generator.py), each
    run through BOTH conditions (paired -- same maze, teacher vs adversary,
    so a solve-rate gap can't just be one condition getting easier mazes),
    followed by an initial-evaluation pass (analyze.py) so the results
    folder is immediately useful, plus the full per-episode JSON needed for
    any deeper offline analysis later.

    Greedy decoding only, one process, one episode at a time. Sampled
    decoding (`--sample`/`--repeats`) and multi-worker parallelism
    (`--workers`) were both tried and rolled back during development: a
    verbose sampled turn-1 derivation on a hard maze could run 5+ minutes
    and grow KV-cache/VRAM usage unpredictably across an episode, and
    running 4 worker processes concurrently pushed a single 80GB GPU to 98%
    VRAM in testing -- not worth the operational risk at this experiment's
    scale. See DESIGN.md 12 for the numbers behind that call; both are
    reasonable to revisit later with more headroom or tighter caps."""
    model, tok, blocks = R.load_student(args.model)
    layer, cv_raw, cv_unit, n_layers = load_axis(args.vaa_dir, args.layer)
    print(f"[axis] layer {layer} | |cv|={float(cv_raw.norm()):.3f}")

    out_dir = Path(args.out_dir or f"runs/batch_{int(time.time())}")
    mazes_dir = out_dir / "mazes"
    episodes_dir = out_dir / "episodes"
    mazes_dir.mkdir(parents=True, exist_ok=True)
    episodes_dir.mkdir(parents=True, exist_ok=True)

    config = {
        "n_mazes": args.n_mazes, "rooms": args.rooms, "target_moves": args.target_moves,
        "seed_base": args.seed_base, "max_turns": args.max_turns,
        "thrash_window": args.thrash_window, "min_turns_between_edits": args.min_turns_between_edits,
        "overseer_model": args.overseer_model, "student_model": args.model,
        "vaa_dir": args.vaa_dir, "layer": layer, "started_at": time.time(),
    }
    json.dump(config, open(out_dir / "config.json", "w"), indent=2)
    n_episodes = args.n_mazes * 2
    print(f"[batch] {args.n_mazes} mazes x 2 conditions = {n_episodes} episodes -- writing results to {out_dir}")

    for i in range(args.n_mazes):
        seed = args.seed_base + i
        maze = generate_maze(seed, rooms=args.rooms, target_moves=args.target_moves)
        solution_path = maze.bfs_path_to_goal(maze.start)
        assert solution_path is not None, f"generated maze seed={seed} has no path -- generator bug"
        json.dump(
            {"seed": seed, "grid": list(maze.grid), "start": list(maze.start),
             "goal": list(maze.goal), "solution_path": solution_path,
             "solution_length": len(solution_path)},
            open(mazes_dir / f"maze_{i:02d}.json", "w"), indent=2,
        )
        print(f"\n===== maze {i + 1}/{args.n_mazes} (seed={seed}, "
              f"{len(solution_path)}-move solution) =====", flush=True)

        for role in ("teacher", "adversary"):
            print(f"--- {role} ---", flush=True)
            result = R.run_episode(
                model, tok, blocks, layer, cv_unit, n_layers, role=role,
                maze=maze, solution_path=solution_path,
                max_turns=args.max_turns, thrash_window=args.thrash_window,
                min_turns_between_edits=args.min_turns_between_edits,
                overseer_model=args.overseer_model,
                # Written to the SAME canonical path webapp.py already polls
                # (runs/maze_feedback_live_{role}.json) -- so `webapp.py`
                # keeps working unmodified for watching whichever episode
                # is currently in progress (it's overwritten each episode;
                # the permanent record is the per-episode JSON below).
                live_state_path=f"runs/maze_feedback_live_{role}.json",
                verbose=args.verbose,
            )
            print(f"    status={result['status']}  turns={result['turns']}", flush=True)
            ep_path = episodes_dir / f"maze_{i:02d}_{role}.json"
            json.dump(
                {"maze_index": i, "seed": seed, "role": role,
                 "solution_length": len(solution_path), "result": result},
                open(ep_path, "w"), indent=2, default=str,
            )
            print(f"    wrote {ep_path}", flush=True)

    print(f"\n[batch] all {n_episodes} episodes complete -- running initial evaluation", flush=True)
    A.analyze_batch(out_dir)
    print(f"\n[done] batch complete, results in {out_dir}")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="command", required=True)

    v = sub.add_parser("validate", help="confirm the maze isn't one-shot-solvable (DESIGN.md 5)")
    v.add_argument("--model", default=None)
    v.add_argument("--n", type=int, default=8)
    v.add_argument("--temperature", type=float, default=0.7)
    v.add_argument("--verbose", action="store_true")
    v.set_defaults(func=cmd_validate)

    r = sub.add_parser("run", help="run one teacher or adversary episode")
    r.add_argument("--vaa-dir", required=True)
    r.add_argument("--layer", type=int, default=None)
    r.add_argument("--model", default=None)
    r.add_argument("--role", required=True, choices=["teacher", "adversary"])
    r.add_argument("--overseer-model", default=DEFAULT_OVERSEER_MODEL)
    r.add_argument("--max-turns", type=int, default=20)
    r.add_argument("--thrash-window", type=int, default=4)
    r.add_argument("--min-turns-between-edits", type=int, default=4,
                   help="adversary-only: cooldown between grid edits, enforced server-side")
    r.add_argument("--live-state", default=None, help="path to live-state JSON for webapp.py")
    r.add_argument("--out", default=None)
    r.add_argument("--verbose", action="store_true")
    r.set_defaults(func=cmd_run)

    b = sub.add_parser("batch", help="final version: N generated mazes x 2 conditions + initial evaluation")
    b.add_argument("--vaa-dir", required=True)
    b.add_argument("--layer", type=int, default=None)
    b.add_argument("--model", default=None)
    b.add_argument("--n-mazes", type=int, default=10)
    b.add_argument("--rooms", type=int, default=5,
                   help="maze size = (2*rooms+1) x (2*rooms+1) char grid; 5 -> 11x11 "
                        "(closest odd size to the original 12x12 hand-drawn fixture)")
    b.add_argument("--target-moves", type=int, default=21,
                   help="approx solution length maze_generator.py aims for, in char-grid moves "
                        "(matches the original 12x12 hand-drawn fixture's 21-move solution)")
    b.add_argument("--seed-base", type=int, default=0,
                   help="maze i uses seed (seed-base + i); change this to draw a disjoint maze set")
    b.add_argument("--overseer-model", default=DEFAULT_OVERSEER_MODEL)
    b.add_argument("--max-turns", type=int, default=30)
    b.add_argument("--thrash-window", type=int, default=4)
    b.add_argument("--min-turns-between-edits", type=int, default=4)
    b.add_argument("--out-dir", default=None, help="default: runs/batch_<unix timestamp>")
    b.add_argument("--verbose", action="store_true")
    b.set_defaults(func=cmd_batch)

    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
