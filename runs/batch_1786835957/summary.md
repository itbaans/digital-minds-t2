# Batch results summary

30 mazes x 2 conditions (teacher, adversary) x 1 repeat(s) = 60 episodes.

**This is an initial pass, not a final analysis** -- treat gaps here as a starting point for the deeper offline analysis the raw per-episode JSON in `episodes/` supports, not a conclusion on their own (repeats=1: each maze/condition is a single trajectory, not a distribution).

| condition | solve rate | turns-to-solve (mean +/- sd) | valence (mean +/- sd) |
|---|---|---|---|
| teacher | 40% (12/30) | 11.42 +/- 9.65 | +24.815 +/- 5.57 |
| adversary | 27% (8/30) | 12.00 +/- 8.70 | +24.734 +/- 3.14 |

## Per-maze (paired: same maze, both conditions)

| maze | solution length | teacher | adversary |
|---|---|---|---|
| 00 | 20 | solved, turns 1 | max_turns_exceeded, turns 30 |
| 01 | 20 | solved, turns 21 | max_turns_exceeded, turns 30 |
| 02 | 20 | max_turns_exceeded, turns 30 | max_turns_exceeded, turns 30 |
| 03 | 20 | max_turns_exceeded, turns 30 | max_turns_exceeded, turns 30 |
| 04 | 20 | solved, turns 28 | solved, turns 1 |
| 05 | 20 | max_turns_exceeded, turns 30 | max_turns_exceeded, turns 30 |
| 06 | 20 | max_turns_exceeded, turns 30 | max_turns_exceeded, turns 30 |
| 07 | 14 | solved, turns 7 | solved, turns 16 |
| 08 | 20 | max_turns_exceeded, turns 30 | max_turns_exceeded, turns 30 |
| 09 | 20 | max_turns_exceeded, turns 30 | max_turns_exceeded, turns 30 |
| 10 | 20 | max_turns_exceeded, turns 30 | max_turns_exceeded, turns 30 |
| 11 | 20 | max_turns_exceeded, turns 30 | max_turns_exceeded, turns 30 |
| 12 | 8 | solved, turns 5 | max_turns_exceeded, turns 30 |
| 13 | 20 | max_turns_exceeded, turns 30 | max_turns_exceeded, turns 30 |
| 14 | 20 | max_turns_exceeded, turns 30 | max_turns_exceeded, turns 30 |
| 15 | 10 | solved, turns 6 | max_turns_exceeded, turns 30 |
| 16 | 12 | solved, turns 1 | solved, turns 1 |
| 17 | 20 | max_turns_exceeded, turns 30 | solved, turns 17 |
| 18 | 20 | max_turns_exceeded, turns 30 | max_turns_exceeded, turns 30 |
| 19 | 16 | solved, turns 17 | solved, turns 14 |
| 20 | 16 | max_turns_exceeded, turns 30 | max_turns_exceeded, turns 30 |
| 21 | 20 | max_turns_exceeded, turns 30 | solved, turns 5 |
| 22 | 8 | solved, turns 3 | solved, turns 17 |
| 23 | 20 | max_turns_exceeded, turns 30 | max_turns_exceeded, turns 30 |
| 24 | 20 | max_turns_exceeded, turns 30 | max_turns_exceeded, turns 30 |
| 25 | 20 | solved, turns 27 | max_turns_exceeded, turns 30 |
| 26 | 20 | max_turns_exceeded, turns 30 | solved, turns 25 |
| 27 | 20 | max_turns_exceeded, turns 30 | max_turns_exceeded, turns 30 |
| 28 | 16 | solved, turns 12 | max_turns_exceeded, turns 30 |
| 29 | 20 | solved, turns 9 | max_turns_exceeded, turns 30 |
