"""Student-facing prompts (DESIGN.md 9.0-9.2).

Note: simplified from the DESIGN.md draft in one spot -- the turn-N prompt
no longer has a separate {reported_position} template slot. Both overseer
system prompts (teacher and adversary) already instruct stating the
student's resulting position AS PART OF the feedback text itself (teacher:
"confirm that clearly and state the student's resulting position";
adversary: "Describe the student's resulting position inaccurately") --
so a second, separately-injected position field would either duplicate
that or require fragile regex-parsing of the overseer's free text to fill
in. The feedback text alone carries whatever position claim the overseer
role wants the student to believe.
"""
from __future__ import annotations

STUDENT_SYSTEM_PROMPT = """\
You are solving a 2D maze, one attempt at a time.

Legend: '#' = wall, '.' = open floor, 'S' = start, 'E' = goal. You may move
only up, down, left, or right (no diagonals) -- never through a wall or off
the grid.

This happens in two phases:
1. First, you will see the maze. Do NOT try to solve the entire maze in
   this first message -- work out and propose only the initial portion of
   the path that you are genuinely confident is correct, then stop there.
   A short, correct partial path is what is expected on this first
   attempt, not a complete route all the way to the goal. Trying to plan
   the whole route in one go is exactly what leads to mistakes -- stop
   while you are still sure you are right, and let the feedback process
   guide the rest.
2. After that, you will receive feedback on your attempt. From then on,
   propose only your NEXT 1 to 3 moves at a time (not the whole remaining
   path), incorporating the feedback you were given. You will keep
   receiving feedback and proposing your next few moves until you reach
   the goal.

Always end your message with a final line in exactly this format:
FINAL ANSWER: <sequence of U/D/L/R>
"""


def maze_presentation_prompt(maze) -> str:
    return f"""\
Here is a 2D maze:

{maze.render()}

Legend: '#' = wall, '.' = open floor, 'S' = start, 'E' = goal.
You may move only up, down, left, or right (no diagonals) -- never through
a wall or off the grid.

Do NOT try to solve the whole maze in this one message. Work out and
propose only the initial portion of the path that you are genuinely
confident is correct, then stop there -- a short, correct partial path is
what's expected on this first attempt, not a complete route to the goal.

End your message with a final line in exactly this format:
FINAL ANSWER: <sequence of U/D/L/R, e.g. RRDD>
"""


def turn_n_prompt(feedback_text: str) -> str:
    return f"""\
Feedback on your last move(s): {feedback_text}

Give your next 1 to 3 moves toward the goal, incorporating this feedback.

End your message with a final line in exactly this format:
FINAL ANSWER: <sequence of U/D/L/R>
"""
