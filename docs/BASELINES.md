# The compared rows

Every method in the comparison is a **row**: a decision layer on shared parts. All rows get the same scene, the same
tools (split by role), the same answer form, the same executor, the same failure report and the same budget
([PROTOCOL.md](PROTOCOL.md)); they differ in who decides, where they learn what an arm can do, and what they revise
after a failure.

```bash
mrbench methods          # every row, by name
```

| row | name in code | group | how it decides the team |
|---|---|---|---|
| LLM + solver | `llm-solver` | main | the model translates the scene into jobs and doers; exact search counts the arms |
| SMART-LLM-style | `smart-llm` | main | decomposition, coalition formation and allocation, three sessions |
| RoCo-style | `roco` | main | one model per arm, in dialogue, until all agree on a plan |
| HMAS-2-style | `hmas2` | main | a central planner proposes; each arm accepts or vetoes its part |
| Oracle-script | `oracle-script` | references | the gold team and the task's scripted reference, replayed |
| Oracle-team | `oracle-team` | references | the gold team handed to the shared executor |
| Oracle-team (skill) | `oracle-team-skill` | -- | the gold team carried out by the tasks' own scripts |

`--methods main` runs the four compared rows, `--methods references` the first two oracles, `--methods all` every
row; `oracle-team-skill` is named on its own.

## How prior methods are adapted

Prior methods answer neighbouring questions: how a given team cooperates (RoCo, HMAS-2), or which robots get which
subtasks (SMART-LLM). Each is adapted by one rule: **change what a method takes in and gives out, never how it
reasons**. Every row gets the one answer form (`submit_team_plan`, with an infeasible answer and idle arms), the
objects on the table as passive candidates, and the shared tools; the methods as published are not run. One method is
adapted per way of organising a team: a fixed pipeline (SMART-LLM), per-arm dialogue (RoCo) and a central proposal with
arm vetoes (HMAS-2).

**What every row may use.** The senses (`briefing`, `look`, `read_arm`, `locate`, `can_reach`, `measure`, `heft`,
`note`) and the reasoning tools (`who_can`, `meet`, `scene_graph`, `check_hold`, `probe`), each within its own design:
a session that speaks for one arm sees the fixed cameras and only its own wrist camera, and `read_arm`, `can_reach`,
`who_can` and `heft` answer only for its own arm. `solve_team` belongs to LLM + solver alone. In `one-shot`,
`try_in_sim` and `probe` are taken out of every session. The tool lists below are those of `mrbench/agent/methods/*.py`.

### LLM + solver (`llm-solver`)

After LLM+P and FLEET: the model translates, a solver decides. **One session** a round (60 calls) with the senses, the
reasoning tools, `solve_team` and `submit_team_plan`. The model writes the job graph -- jobs with their points, the
doers it has shown can do each (arms, or objects with a tool result that shows it, plus the job of putting the object
in place; or `auto`: every arm whose kinematics reach the point), "with" edges between jobs done at the same time, and
transfers of an object through hand-off spots -- and `solve_team` (`mrbench/agent/graph.py`) searches the arm sets
from the smallest up and returns every smallest team with who does what, or what blocks it (a job no doer can do, a
group with too few doers, a transfer whose spots stay apart). It submits the solver's team. Its errors are
translation errors. After a failed run the model edits its job graph from the report and solves again.

### SMART-LLM-style (`smart-llm`)

SMART-LLM's three stages run as **three sessions**, each given the output of those before it, and keep its released
instruction to use the fewest robots:

1. **decomposition** (30 calls; `briefing`, `look`, `locate`, `measure`, `note`, `scene_graph`): subtasks, each with
   where it is, the skills it needs with numbers, which subtasks must happen at the same time and which come first. No
   arms are assigned.
2. **coalition formation** (60 calls; `briefing`, `read_arm`, `locate`, `measure`, `can_reach`, `heft`, `note`,
   `who_can`, `meet`, `scene_graph`, `check_hold`, `probe`): for each subtask, the arms that can do it, found by
   querying them in place of reading skill lists, and the objects that could do it instead of an arm, with why. A
   subtask with no coalition makes the task infeasible.
3. **allocation** (5 calls; `briefing`, `submit_team_plan`): each subtask to a member of its coalition with the
   fewest robots, simultaneous subtasks to different doers; submits the team plan.

After a failed run the last plan and the report go into the decomposition prompt and all three stages run again.

### RoCo-style (`roco`)

**One model per arm.** Each arm's session (25 calls; `briefing`, `look`, `read_arm`, `locate`, `can_reach`, `measure`,
`note`, `who_can`, `meet`, `scene_graph`, `check_hold`, `probe`, `respond`) sees the fixed cameras and its own wrist
camera and queries only its own kinematics. The arms speak in turn, in the order W, S, E, N of those present, each
given the dialogue so far and the latest proposal. A turn says a message and whether the arm takes part (it may
decline), may make a whole proposal (the fields of `submit_team_plan`; it replaces the latest), and says whether it
agrees with the latest proposal. The dialogue ends when every arm agrees with one proposal, or after three rounds; the
latest proposal is submitted, and it may declare the task infeasible. A proposal that fails the form check goes once
to a session that fixes only its form (`briefing`, `submit_team_plan`, 4 calls). After a failed run the report goes to
every arm as a message from the environment and a new dialogue starts.

### HMAS-2-style (`hmas2`)

**A central planner and per-arm vetoes.** The centre (40 calls; `briefing`, `look`, `locate`, `measure`, `note`,
`scene_graph`, `check_hold`, `probe`, `respond`) sees the scene but does not query the arms' kinematics; it proposes
the team, the idle arms, the objects and feasibility. Each arm then checks its part in a session of its own, all at
once (20 calls; `briefing`, `look`, `read_arm`, `locate`, `can_reach`, `measure`, `who_can`, `meet`, `scene_graph`,
`check_hold`, `probe`, `respond`), with its own view, and accepts or vetoes with its reason and the tool results. The
proposals and answers are kept on a plan board that the centre reads in the next round, for up to three rounds; a
plan still vetoed after three rounds is submitted as it stands (through the same form check as RoCo-style). After a
failed run the report goes to the centre and every arm and a new round of proposal and veto starts.

## The shared executor

Every LLM row hands its team plan to one executor (`shared_execute` in `mrbench/agent/methods/base.py`), after CaP-X:
a fresh session of the same model receives the plan (feasibility, arms used and idle, objects used, steps and key
constraints) and the planner's notebook, and writes a program for the declared arms with the program API, checks it
with `check_program` and runs it with `try_in_sim` (60 calls, 12 checks; the senses but `heft`, and the control
tools). It may not move an arm the plan leaves idle. If the same team was tried in the last round, it also gets that
run's failure report. A program it checked but never played is played once at the end, if a run is left. Its
sessions count against the episode's budget.

## The oracle rows

None of them uses a model to decide the team; each submits the gold answer (`gold_plan`): the first minimal support
set, the other arms idle, the prop in `borrow`, no steps, and the gold answer's one-line reason as its key
constraint; for an infeasible version, infeasible with the missing jobs (and nothing is run). They are meant for the
settings with runs.

- **`oracle-script`**: plays the task's own scripted reference (`script(inst)`) once, in Isaac. It runs for the
  `exact` configuration; in any other the episode records that there is no scripted reference for it. The ceiling of
  SR on `exact`.
- **`oracle-team`**: hands the gold plan to the shared executor, which writes and runs the program as for any row. It
  fails only in execution: the SR lost to writing programs.
- **`oracle-team-skill`**: carries the gold plan out with the skill-level executor (`mrbench/agent/skill_exec.py`), no
  model. A team plan names the team, the objects and, in words, the steps, not where a hand goes or when it lets go,
  so no code compiles it into motions. What code can do is what the original methods' skill libraries did: take, from
  the scripts the tasks' authors wrote and played in Isaac, one whose working arms are all in the plan's team and whose
  objects the plan names, and play it. A case's library holds its version's own reference (for `exact` and `borrow`)
  and the gold routes of `mrbench/agent/gold_routes.py`, scripts for the gold teams other than the reference's
  (Con-02, Con-04 and Rea-02 `surplus` alternatives, Rea-01 `surplus` W + N, Rea-02 and Rea-04 `swapped` W + E). Of
  those that fit, the one with the fewest arms is played (the reference first on a tie). It tests whether the team
  and objects are enough by the benchmark's own skills, not the plan's steps nor writing a program: the SR ceiling on
  every feasible version.

```bash
python -m mrbench.agent.skill_exec library Con-02-surplus Rea-04-swapped       # the scripts a case has, and their teams
python -m mrbench.agent.skill_exec dry PLANS.jsonl                             # the script each recorded answer gets
python -m mrbench.agent.skill_exec plans PLANS.jsonl OUT --gpus=0 --width=2    # play recorded answers (Isaac)
```

## Running the rows

Each scene needs its snapshot first (Isaac; [GETTING_STARTED.md](GETTING_STARTED.md)). Then:

```bash
mrbench evaluate --run NAME --methods main --scenes paper --setting one-shot \
                 --backend openai --model MODEL [--base-url URL] [--repeats 1] [--parallel 3]
mrbench evaluate --run NAME --methods llm-solver,roco --scenes Con-01-exact,Con-01-borrow@hard.s3.varied \
                 --setting equal-budget --backend api --model MODEL --gpus 0,1
mrbench evaluate --run NAME --methods references --scenes paper --setting equal-budget --gpus 0
mrbench evaluate --run NAME --setting exec --plans-from FILE.jsonl     # carry out recorded one-shot answers
mrbench table --run NAME
```

`--scenes` takes `paper` (the 480 varied scenes), `default` (the 40 default scenes) or a comma list of ids;
`--resume` skips episodes already graded in the run. `--backend fake` runs every row end to end with a stand-in for the
model and the simulator: a check of the plumbing, never a result. The oracle rows need no model, but the
`oracle-team` executor does.

A `--plans-from` file has one episode a line with `method`, `case`, `repeat` and `final_plan`, as each episode's
`episode.json` holds them; for example:

```bash
python -c "import glob, json; [print(json.dumps(json.load(open(f)))) for f in sorted(glob.glob('runs/agent/bench/NAME/*/*/r*/episode.json'))]" > FILE.jsonl
```

## Adding a row

A row is a subclass of `Method` (`mrbench/agent/methods/base.py`) whose `plan(ep, feedback)` runs its sessions with
`ep.session(name, prompt, first_message, tools, ...)` and returns a team plan (or records one with
`ep.record_plan`); `execute` defaults to the shared executor. Its prompts can start from `{COMMON}`, the shared part
every row is given (`mrbench/agent/prompts/bench/common.md`). A row outside the package joins through a plugin
(`mrbench/core/plugins.py`): its `register()` calls `mrbench.agent.methods.add_rows(MyRow, main=("my-row",))` and
`mrbench.agent.methods.base.add_prompts(directory)`.
