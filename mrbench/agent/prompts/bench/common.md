## The question

A team of robot arms is mounted round one table, with things on it and a goal. The question is how big a team the
goal needs:

- which arms are needed -- the fewest that can do the whole task -- and which can stay idle;
- whether a thing on the table can stand in for an arm (some things look as if they could and cannot);
- or that the task cannot be done with what is here, and what is missing.

## The answer: `submit_team_plan`

- `feasible`; `arms_used` (the fewest arms that do it; empty if it cannot be done); `idle` (every other arm, each
  with why);
- `things_used` (each with the job it does and, if it takes over an arm's job, which); `things_rejected` (things that
  might seem to do a job and do not, with the evidence);
- `missing` (if it cannot be done: which job has no doer, and what kind of arm or thing would do it);
- `steps` (who does what, and what happens at the same time); `key_constraints` (what forces the number of arms,
  with numbers).

Every arm appears exactly once, in `arms_used` or in `idle`.

## True of every arm and thing

- Positions and sizes are in metres, in the world frame `briefing` describes; it also gives the arms, the things and
  the facts that do not show (a spring, a tight fit).
- Masses in the briefing are rough (25 % lower to 30 % higher); `heft` measures one.
- A hand grips a thing only if the thing fits between its open fingers -- unless a second hand presses on the thing's
  far side.
- Every session has a limited number of tool calls; every reply says how many are left.
