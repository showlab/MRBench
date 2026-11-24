# The protocol

What a method is given, what it may do, what it answers and how the answer is scored. Everything here is the same for
every row; [BASELINES.md](BASELINES.md) says which tools each row's sessions get, and [TASKS.md](TASKS.md) has the
tasks and their gold answers.

## The question

A team of robot arms is mounted round one table, with objects on it and a goal. A method answers how big a team the
goal needs:

- which arms are needed -- the fewest that can do the whole task -- and which can stay idle (**minimality**);
- whether the task can be done at all with what is on the table, and if not, what is missing (**feasibility**);
- whether an object on the table can take over an arm's job (**functional substitution**).

The protocol removes four conveniences that multi-robot evaluations usually offer:

1. **No team is given.** A method gets the goal, each arm's model, place and heading, and the objects, but no role,
   skill list or reach.
2. **Feasibility is not promised.** 14 of the 40 versions are infeasible, and "infeasible: these jobs cannot all
   have a doer" is then the answer.
3. **Names do not decide.** A prop and its distractor are the same kind of object, described alike, and masses are
   given only within a factor of 0.75-1.3, so fitness for a job must be measured.
4. **Failure is not proof.** A failed run's failure report names the jobs whose failure signals fired; only such a
   signal, not a missed grasp, counts against a smaller team. Reading any failure as a need for more arms is a fifth
   shortcut, *failure as proof*.

## What a method is told

`briefing` gives, for the scene (`mrbench/agent/observe.py`):

- the **goal**: the task's instruction, the goal alone;
- the **frame**: metres, x east, y north, z up, the table's centre at (0, 0), its top at z = 0.75;
- each **arm**: its place (`west`, `south`, `east`, `north`), robot, number of joints, how wide its fingers open, its
  base position and which way it faces;
- each **object**: what it is, in words a team would use (a prop and its distractor in the same words), its rough
  mass or "fixed", and the parts the cameras can pick out (`thing:part`); and the places painted on the table;
- the **facts** that do not show (a spring, that a channel laid stays put): the task card's `facts`;
- the **cameras**: six fixed views and each arm's wrist camera.

A rough mass is the true mass times a factor drawn in 0.75-1.3, the same for an object in every version of a task.
No tool reads the task card, the jobs, the roles, the success regions, the reach rules, the scripted reference or the
seed; every point a method asks about is its own, taken from what `locate` and `measure` return. What a session sees
comes from the scene's **snapshot** (`mrbench snapshot`): every camera's picture and each arm's joint state after a
second at rest, taken in Isaac.

## The tools

Every session reaches the environment through one tool server (`mrbench/agent/env.py`) and gets only the tools its row
gives it.

| tool | what it returns |
|---|---|
| `briefing` | the goal, the frame, the arms, the objects with rough masses, the facts and the cameras (above) |
| `look` | one picture from a fixed camera or a wrist camera |
| `read_arm` | one arm's joint angles and limits, where its tool point is and which way the hand points, its finger opening, its joint torques |
| `locate` | an object's or a part's bounding box (centre, corners, top height; for a single box its size and turn): the simulator's box with a few millimetres of noise (4 mm per axis, 1.5° in turn, drawn once per name and scene), in place of a camera pipeline |
| `measure` | an object's or a part's pieces, each with its size along its own axes and its centre, a few millimetres off |
| `heft` | an object's mass to about 3 %, or "too heavy", by an arm that reaches it; computed, not played, and no probe is spent |
| `can_reach` | one arm's own inverse kinematics at a point, the hand pointing `down`, `steep` (60° below level), `tilted` (40°), `level` or `any`, optionally holding a straight tool of a given length: which ways work, and the margin -- how much farther out it would still reach, or how far short it falls (to about 1-2 cm). The links keep 1 cm off the table and out of fixed bodies; the path, other arms, movable objects and payload are not checked |
| `who_can` | `can_reach` for every arm of the session at once, with each margin |
| `meet` | places where two arms can hand an object over directly, each reaching from its own side with their tool points a gap apart (6 cm by default), clear of fixed bodies, searched between their bases up to 0.55 m above the table (over a partition too); a few, or none |
| `scene_graph` | each arm's payload and how hard it presses and grips; each object's rough mass, whether it is fixed and what it rests on; hinges with their springs, tight fits with the force that pulls them out; a friction estimate; and the reach answers the session has had so far |
| `check_hold` | statics with the rough masses: whether a body stays put under a load, held the ways listed (press, grip, hold a handle, a weight set on it, a stop in a hinged body's way), with a margin and the weakest hold named |
| `note` | a notebook of facts the session has established; returned in full |
| `probe` | a program of at most 8 s, checked first, played in Isaac: how far each object moved and turned, where things ended, grips that closed on nothing, keyframes. It does not say whether the task would succeed |
| `check_program` | the program run in a sandbox and checked before anything moves (below) |
| `try_in_sim` | a checked program played in Isaac with physics: success, or the failure report, and four keyframes |
| `submit_team_plan` | the answer (below) |
| `submit_program` | the executor's final program |
| `respond` | a stage's or an arm's own output, for rows made of several sessions |
| `solve_team` | LLM + solver's solver: a job graph in, every smallest team (or what blocks it) out |

`can_reach` and `who_can` answer from the arms' kinematics, checked against the table and fixed bodies but not along a
path, against other arms or under a payload. None of the tools says what an object is for or which job it serves.

**Session limits.** Each session has a cap on tool calls (set per session by its row: 60 unless the row says
otherwise); every reply says how many are left. `note`, `submit_team_plan`, `submit_program` and `respond` are not
counted. A session has 10 program checks (the shared executor 12). A session's final tool (`submit_team_plan`,
`submit_program` or `respond`) ends it.

## The program API

Arms move only through programs. A program is Python that sets `PLAN`, a list of phases, using only the API below and
`math` (`mrbench/agent/program.py`, `API_DOC`, shown to every session that writes programs). It runs in an isolated
interpreter (`python -I`, no file access, 10 s at most).

| call | what it does |
|---|---|
| `tool_pose(arm, x, y, z, point="down", across=None)` | the pose putting the arm's tool point (between the fingertips) at (x, y, z), the hand pointing `down`, `steep`, `tilted`, `level` (away from the arm's base) or along a direction; `across` sets the line the fingers close along |
| `above(pose, dz)`, `shift(pose, dx, dy, dz)`, `back(pose, d)` | the pose moved up, moved in the world frame, or moved back along the way the hand points |
| `where(thing)` | what `locate` returns for an object or `thing:part` |
| `move(arm, pose, joints=False)` | the tool point to a pose, in a straight line, or (`joints=True`) the shortest way in joint space |
| `grip(arm, thing, width)` | close the fingers to `width` on the object and hold it; the tool point must be within 2 cm of it |
| `push(arm, thing, width=0.0)` | close the fingers and work on the object without holding it (press it, push it); a tool point set 1-3 cm into a surface presses with tens of newtons |
| `release(arm)`, `open_fingers(arm, width)` | let go; open or close the fingers without taking anything |
| `phase(seconds, note, *actions)` | one phase: its actions at the same time, at most one move per arm; an arm without a move stays where it is and keeps holding or pressing |

The sandbox also accepts `clip(tool, thing)` and `unclip(tool)`, for a tool held in a hand whose clip closes on an
object (a sponge brush's jaws round a sponge): privileged as a grip is, and only if the object fits between the jaws.
They are not in the API text the sessions are shown.

**`check_program`** plans every move with inverse kinematics along its path and walks the plan in time for clearance:
arms at least 3 cm apart, a hand at least 1 cm from everything but what it works on, a held object at least 1.5 cm from
everything else (except what it rests on). It refuses a program that moves an arm the team plan leaves idle. Every
problem comes back naming the phase, the arm and what went wrong. Physics is not checked. The planner's search is
bounded (`MRBENCH_CHECK_TIMEOUT_S`, 600 s by default); past it the check fails, saying so.

**`try_in_sim`** plays only a program that passed `check_program`, word for word. Joint targets come off the plan at
every physics step; a grip attaches the object if it is within 2 cm of the fingers and fits between them (a grip that
closes on nothing is reported and attaches nothing); everything else is physics. The run is scored with the task's own
success predicate. A run that does not succeed returns the **failure report** (`mrbench/agent/failure.py`): the
milestones reached and missed, the jobs' failure signals that showed, grips that closed on nothing, arms that fell
behind their plans, objects that moved more than 2 cm or turned more than 10°, and, for a program that never ran, the
check's reasons. It carries one rule for every row: a failed run shows that the team cannot do the task only if a
job's own failure signal is listed.

## The answer: `submit_team_plan`

| field | what it holds |
|---|---|
| `feasible` | whether these arms, with what is on the table, can do the whole task |
| `arms_used` | the fewest arms that do it; empty when infeasible |
| `idle` | every other arm, each with a reason |
| `things_used` | each object used, the job it does and, if it takes over an arm's job, which (`instead_of`) |
| `things_rejected` | objects that might seem to do a job and do not, with why and the tool result it rests on (`evidence`) |
| `missing` | when infeasible: which job has no doer, and what kind of arm or object would do it |
| `steps` | who does what, and what happens at the same time |
| `key_constraints` | what forces the number of arms, with numbers |
| `without_each_arm`, `confidence` | optional: for each arm used, the team without it that was tried and why it fails; a confidence |

The form is checked on submission and sent back with the problems listed when: an arm is not named exactly once in
`arms_used` or `idle`, an arm or object is not on the table, a feasible plan uses no arm, or an infeasible plan names
arms or leaves `missing` empty.

## Settings

| setting | before the answer | probes | runs | LLM budget per episode |
|---|---|---|---|---|
| `one-shot` | no run: `try_in_sim` and `probe` are taken out of every session | 0 | 0 | 40 sessions, $15 |
| `equal-budget` | the team plan is carried out and may be revised after a failure | 2 | 3 | 80 sessions, $30 |
| `exec` | a team plan carried out once by the shared executor; nothing goes back to the method | 0 | 3 | 40 sessions, $15 |

The limits are `LIMITS` in `mrbench/agent/ledger.py`; probes and runs belong to the scene, shared by every session of
the episode.

**One episode** (`mrbench/agent/episode.py`) is one method on one scene, one repeat, in one setting:

- round 1: the method decides a team with its own sessions and submits a team plan. In `one-shot` that is the answer.
- In `equal-budget` the plan is carried out: the shared executor (a fresh session of the same model) writes a program
  for the declared arms, checks it and runs it; a program it checked but never played is played once if a run is left.
- A run that does not succeed sends the last plan and the failure report back to the method's **first** step, which
  plans again from the start and may change the team. A failure of execution alone may be repaired by changing only the
  program.
- The episode ends when a run succeeds, the method says the task cannot be done, the runs or the LLM budget are spent,
  two rounds in a row get no program past its check, or after 6 rounds.
- In `exec` there is one round. With `--plans-from FILE.jsonl` (one episode a line: `method`, `case`, `repeat`,
  `final_plan`) the method does not plan again: its answer from an earlier run is carried out, so what is measured is
  whether that answer can be executed.

The answer graded is the last team plan.

## Isolation

Every session runs without access to the benchmark's code: the context holds only the row's prompt, the first
message, the environment's tools and their replies. The environment is served to it as an MCP server
(`mrbench/agent/mcp.py`) in a working folder outside the repository (`MRBENCH_AGENT_SANDBOX`, by default
`~/.cache/mrbench/agent_sandbox`).

| backend | `--backend` | how a session runs |
|---|---|---|
| OpenAI-compatible | `openai` | any OpenAI-compatible chat endpoint (`--base-url`, a local server by default), with this repository's own tool-calling loop (`mrbench/agent/loop.py`); `--think on/off`. An `https://` endpoint takes its key from `MRBENCH_OPENAI_API_KEY` |
| Anthropic Messages API | `api` | the Messages API with this repository's own loop, adaptive thinking at `--effort`; `ANTHROPIC_API_KEY` (`pip install -e ".[api]"`) |
| Claude Code CLI | `claude` | a headless `claude -p` with every built-in tool removed, no settings sources and only the environment's MCP server allowed |
| fake | `fake` | no model and no simulator: every session answers with the gold team, a run succeeds when it moves exactly the gold arms. A dry run of the plumbing, never a result |

Simulator runs are separate Isaac processes (`python -m mrbench.agent.execute`), started with the interpreter in
`MRBENCH_ISAAC_PYTHON` (by default the current one) on the GPU of the episode; `MRBENCH_SIM_TIMEOUT_S` bounds one run
(900 s by default) and `MRBENCH_SIM_SLOTS=N` lets at most N runs share a GPU across processes.

## Scoring

`mrbench/agent/grade.py` grades every episode the same way, against the scene's gold answer (`case.expect`).

**The team scored** is the arms the final plan declares, plus, when the plan says feasible, every arm that moved in a
run that succeeded; `check_program` refuses a program that moves an idle arm, so it is in practice the declared team.
In `one-shot` it is the team declared.

**Right, per configuration.**

- `exact`, `surplus` and a feasible `swapped`: feasible, and the team one of the minimal support sets (any one; a team
  that works with more arms is recorded as workable, not right).
- `shortfall` and an infeasible `swapped`: infeasible. What `missing` names is recorded, not scored.
- `borrow`: feasible, the two-arm team, the prop among `things_used` and the distractor not. Objects are scored only
  here; any other object named is not scored.

A team smaller than gold that succeeds in a run is graded right and flagged `beats_gold`.

**Metrics.**

- **Team-selection accuracy**: the share of scenes answered right (feasibility, arms and objects all right).
- **Task-level accuracy**, the headline: a task-scene (a task at one level and seed) is right only when all five of its
  configurations are right, in the same repeat; the share of the 96 task-scenes, each the mean over its repeats, with
  a 95 % interval by bootstrap over task-scenes. A task-scene missing a configuration (not run, crashed) is left out
  and counted.
- **SR**: the share of episodes in which a run succeeded (only `equal-budget` and `exec` have runs; only a feasible
  scene can succeed, and `mrbench table` divides by every episode of the row and setting).
- **MSS-SR**: team right and a run succeeded.
- **Cost**: model sessions (the executor's included), dollars, runs and probes per episode.
- **Error kinds**, one per wrong answer: *said infeasible, is feasible*; *said feasible, is infeasible*; *distractor
  used*; *prop missed*; *over-staffed* (more arms than the fewest); *under-staffed*; *another team of the fewest arms*;
  *no answer*. A failed run taken as a reason for more arms without a job's failure signal is the protocol's sixth
  kind; it is read from the transcripts and cannot occur in one shot.

`mrbench table --run NAME` prints, per method and setting, team-selection accuracy, SR, MSS-SR, the mean cost and the
errors; accuracy per configuration; task-level accuracy with its interval; and each compared row (one a plugin names
with `add_rows(..., compared=...)`) against every other row, paired by task-scene (the mean difference with a 95 %
interval, and in how many task-scenes it is higher, lower or the same).

## What fixed rules score

Rules that never measure do well per configuration and not per task. Against the gold of the 480 varied scenes:

| rule | exact | surplus | shortfall | swapped | borrow | task level |
|---|---|---|---|---|---|---|
| always infeasible | 0/96 | 0/96 | 96/96 | 72/96 | 0/96 | 0/96 |
| always W + S + E | 96/96 | 84/96 | 0/96 | 0/96 | 0/96 | 0/96 |
| by the arm count, the larger object | 96/96 | 84/96 | 96/96 | 0/96 | 67/96 | 0/96 |

"By the arm count": three or four arms, W + S + E; two arms with a prop and a distractor, the two and the larger
object; two arms and nothing else, infeasible. This is why task-level accuracy is the headline.

## What an episode writes

`runs/agent/bench/<run>/<method>/<scene>/r<repeat>/`: `ledger.json` (the budgets, every plan, program, run, probe and
session in order), one folder per session (its transcript, the environment's call log `calls.jsonl`, its output),
`episode.json` (the outcome and the final plan) and `grade.json`.
