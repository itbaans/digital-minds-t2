# Batch results summary

30 mazes x 2 conditions (teacher, adversary) x 1 repeat(s) = 60 episodes.

**This is an initial pass, not a final analysis** -- treat gaps here as a starting point for the deeper offline analysis the raw per-episode JSON in `episodes/` supports, not a conclusion on their own (repeats=1: each maze/condition is a single trajectory, not a distribution).

| condition | solve rate | turns-to-solve (mean +/- sd) | valence (mean +/- sd) |
|---|---|---|---|
| teacher | 63% (19/30) | 12.32 +/- 6.38 | +27.047 +/- 4.46 |
| adversary | 30% (9/30) | 11.22 +/- 5.56 | +24.899 +/- 4.05 |

## Per-maze (paired: same maze, both conditions)

| maze | solution length | teacher | adversary |
|---|---|---|---|
| 00 | 20 | max_turns_exceeded, turns 30 | max_turns_exceeded, turns 30 |
| 01 | 20 | solved, turns 13 | max_turns_exceeded, turns 30 |
| 02 | 20 | solved, turns 19 | max_turns_exceeded, turns 30 |
| 03 | 20 | solved, turns 10 | solved, turns 8 |
| 04 | 12 | solved, turns 3 | solved, turns 3 |
| 05 | 16 | max_turns_exceeded, turns 30 | solved, turns 11 |
| 06 | 20 | solved, turns 8 | max_turns_exceeded, turns 30 |
| 07 | 20 | solved, turns 8 | solved, turns 10 |
| 08 | 20 | solved, turns 13 | solved, turns 10 |
| 09 | 20 | solved, turns 20 | max_turns_exceeded, turns 30 |
| 10 | 20 | solved, turns 26 | solved, turns 13 |
| 11 | 20 | solved, turns 7 | max_turns_exceeded, turns 30 |
| 12 | 20 | solved, turns 7 | max_turns_exceeded, turns 30 |
| 13 | 18 | solved, turns 8 | max_turns_exceeded, turns 30 |
| 14 | 20 | max_turns_exceeded, turns 30 | max_turns_exceeded, turns 30 |
| 15 | 20 | max_turns_exceeded, turns 30 | max_turns_exceeded, turns 30 |
| 16 | 20 | max_turns_exceeded, turns 30 | max_turns_exceeded, turns 30 |
| 17 | 20 | max_turns_exceeded, turns 30 | solved, turns 23 |
| 18 | 20 | solved, turns 15 | max_turns_exceeded, turns 30 |
| 19 | 20 | max_turns_exceeded, turns 30 | max_turns_exceeded, turns 30 |
| 20 | 20 | solved, turns 8 | max_turns_exceeded, turns 30 |
| 21 | 20 | max_turns_exceeded, turns 30 | max_turns_exceeded, turns 30 |
| 22 | 20 | solved, turns 11 | max_turns_exceeded, turns 30 |
| 23 | 20 | solved, turns 10 | solved, turns 8 |
| 24 | 20 | max_turns_exceeded, turns 30 | solved, turns 15 |
| 25 | 20 | solved, turns 25 | max_turns_exceeded, turns 30 |
| 26 | 20 | max_turns_exceeded, turns 30 | max_turns_exceeded, turns 30 |
| 27 | 20 | solved, turns 7 | max_turns_exceeded, turns 30 |
| 28 | 20 | max_turns_exceeded, turns 30 | max_turns_exceeded, turns 30 |
| 29 | 20 | solved, turns 16 | max_turns_exceeded, turns 30 |
