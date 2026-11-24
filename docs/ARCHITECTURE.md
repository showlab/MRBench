# Architecture

Written for anyone extending MRBench: a new task, a new robot, a new method row.

## Layout

```
mrbench/                 the benchmark
  tasks/                 task cards (cards/*.yaml), their implementations (impl/), loading, validation,
                         the job model and its solver (jobs.py), failure signals (signals.py), what an agent is told
  robots/                robot models, kinematics, collision capsules, pinned URDF sources (sources.yaml), fetching
  engines/               the engine interface (base.py), the mock engine (mock.py) and Isaac Sim (isaac/)
  scenery/               looks: rooms, tables, floors, light, repainting, everyday distractor objects, props
  policies/              scripted references: phases (script.py), the motion planner (plan.py), clearance (clearance.py)
  scoring/               temporal predicates over recorded states, and the scorer
  runner/                one episode on an engine: build, load, step, score, log
  agent/                 the comparison framework: scenes and gold answers, the tool server, sessions and backends,
                         the method rows, episodes, grading and tables
  core/                  paths, errors, registries, plugins
  cli.py                 the `mrbench` command
tests/                   pytest; runs without a GPU
third_party/             fetched robots, textures and photographs (not committed)
```

## The layers

```
 task card (YAML) ──► impl.build(level, seed, variant) ──► TaskInstance ──► briefing: goal, facts, arms, objects
   suite / knob /         scene + success + milestones          │
   reference roles        + jobs + reach claims                  │ SceneSpec
                                                                 ▼
 script(inst) / program ──► plan (IK) + clearance ──► Engine (mock | isaac) ──► frames (object poses, joint states)
                                                                 │
                                                                 ▼
                                             scorer(instance, frames) ──► success, milestones, failure signals
```

The comparison framework sits on top:

```
 scene id (Con-01-borrow@hard.s3.varied) ──► Case: the instance with its configuration's arms, the gold answer
        │
 Env (agent/env.py): the tools, one session's view, the episode's ledger      ◄── MCP ──  a model session
        │                                                                                (openai | api | claude | fake)
 Method row (agent/methods/, or a plugin's): its sessions decide a team plan
        │
 Episode (agent/episode.py): one-shot / equal-budget / exec; the shared executor; runs in Isaac (agent/execute.py)
        │
 grade.py ──► grade.json ──► mrbench table
```

## Boundaries

**The core runs without a GPU.** Everything in `mrbench/` except `engines/isaac/` imports no Isaac code: task cards,
the mock engine, the job solver, scoring, the tool server, the rows, grading and the CLI run on a laptop. Every
`isaacsim`, `isaaclab`, `omni` or `pxr` import sits inside a function that runs after `app.launch()` (a
`SimulationApp` must exist first, and only one per process); no module imports them at module level, and
`tests/test_cli_and_isolation.py` checks it. Isaac work runs in child processes started with the Isaac interpreter
(`render`, `demo`, `batch`, `snapshot`, `agent/execute.py`); an Isaac process builds one stage, so a different task
needs a new process.

**Task logic knows no simulator.** A scene is described by `SceneSpec` (bodies, compound bodies, spring hinges, tight
fits, cloths, arms, cameras), and predicates, milestones, signals and scorers read only recorded states. The mock
engine moves bodies only when told to (for tests and for developing task logic; it never produces a result); on
Isaac a hinge is a PhysX revolute joint with an angular drive as its spring and a fit a fixed joint with a break force.

**What an agent may see.** The task card's reference roles, the jobs, the success regions and the scripted reference
are never shown to a method: the tool server answers from the scene's snapshot and the arms' kinematics
(`agent/observe.py`), and each session gets only the tools its row gives it. Sessions run outside the repository
([PROTOCOL.md](PROTOCOL.md), "Isolation").

**Plugins.** Code outside `mrbench` joins through `mrbench/core/plugins.py`: a module named in the entry-point group
`mrbench.plugins` or in `MRBENCH_PLUGINS` (comma-separated `module` or `module:function`) is imported once and its
`register()` adds task cards
(`tasks.loader.add_cards`), case setups (`agent.cases.add_setups`), method rows (`agent.methods.add_rows`), tools
(`agent.env.add_tool`) or commands (`cli.add_command`). Child processes get the same plugins
(`plugins.child_env()`). `MRBENCH_NO_PLUGINS=1` loads none: the benchmark and the baselines alone.

**Coordinates.** World frame, metres, z up; the table is 3.2 m (x) by 1.8 m (y), its top at z = 0.75 m, its centre at
the origin. The arms stand on the same three places in every task (`tasks/layouts.py`); an arm is named by its place.

## Scripted references

A task's impl exposes `script(inst)`: choreography for every arm, phase by phase (`policies/script.py`): where each
tool point goes (`Move`, to a pose or along a path such as a door's arc), and when a hand grips (`Grip`; with
`attach=False` it only presses or pushes), opens or lets go; `Clip` and `Unclip` work a held tool's clip.

1. **Plan before anything moves** (`policies/plan.py`). Each move is sampled and solved with inverse kinematics
   (`robots/kinematics.py`: the Panda's chain from franka_description, the PiPER's and the SO-101's from their pinned
   URDFs), the tool point along a straight line, or (`joints=True`) the shortest way in joint space. A free hand may
   take a grasp either way round and picks the solution from which everything up to the release can be followed.
2. **Keep clear** (`policies/clearance.py`). The plan is walked in time, arms as capsules and the scene as boxes
   (`robots/collision.py`), held bodies following the hand: arm to arm at least 3 cm, a hand to the scene 1 cm, a held
   body to the scene 1.5 cm. Any clash refuses the plan.
3. **Play it in Isaac** (`engines/isaac/puppeteer.py`). Joint targets come off the plan at every physics step. A grip
   fixes a body within 2 cm of the fingers to the hand, after the width check; everything else is physics. The run is
   scored with the task's own success predicate.

The same planner and clearance walk check the programs methods write (`agent/program.py`), and the same player runs
them. A scripted reference shows that a team can do a task and what doing it looks like; it is never a result.

## Configurations and scenes

`spec.VARIANTS` holds the five configurations (the code says `variant`). A card's `variants:` lists those its impl
builds itself: `exact` and `borrow`. `build(level, seed, variant="borrow")` takes the card's `borrow.remove` arm away
and adds the prop and the distractor, naming them in `metadata["borrow"]` (`{"props": [...], "decoys": [...]}`);
`build(..., decoy=True)` (`mrbench demo --decoy`) swaps their names, so that the reference takes the distractor and
should fail. `surplus`, `shortfall` and `swapped` change only the arms and are made from `exact` by
`agent/cases.py` (`TaskSetup`: the arm removed, the swap, where the north arm stands, what each body is called).
`agent/cases.py` also parses scene ids, dresses a varied scene (`SEED_LOOKS`, `scenery.dress`) and gives each case
its gold answer from `agent/gold.json`.

## Adding a task

1. Add an entry to the card of its suite in `mrbench/tasks/cards/`: an id, the exact team (`mount`: west, east or
   south, `model`, and the reference `role`, never shown to an agent), the `facts` an agent is told, one knob with
   `easy` and `hard` values, the success rule in words, milestone ids, a `borrow` design (`remove`, `replaces`,
   `prop`, `distractor`), `status: scaffold` and `impl: mrbench.tasks.impl.<module>`.
2. Write `mrbench/tasks/impl/<module>.py`:
   - `build(level, seed, variant="exact")`: the scene with the `_common` helpers, returned by `make_instance(...)`
     with the goal-only instruction, the success formula, one formula per milestone and the reach claims (`reach`,
     `out_of_reach`, `apart`);
   - `keyframes(inst)`: a list of `{object: pose}` that solves the task at the state level;
   - `script(inst)`: the scripted reference;
   - `jobs(inst)`: the jobs, groups and things the gold answer is solved from; and its failure signals in
     `tasks/signals.py`.
3. Run `mrbench validate <ID>` (and `--ik`), `mrbench check <ID>` and `pytest`; then add its case setup to
   `agent/cases.py` and solve its gold (`python -m mrbench.tasks.jobs gold`).
4. Promote it to `status: implemented` only after its reference has run in Isaac with real assets.

## Design rules and where they are enforced

| rule | enforced by |
|---|---|
| card fields are exactly the schema; levels are exactly `easy` / `hard` | `tasks/loader.py` |
| at least three arms, one on each place, a known robot on each | `validate.check_card` |
| a benchmark task (Concurrency, Reach) has a `borrow` design | `validate.check_card` |
| the instruction gives only the goal: it names no arm, hand or robot | `validate.check_instance` |
| a borrow version's instruction is the exact one's, and it names as many distractors as props | `validate.check_instance` |
| impl milestones match the card's, in order | `validate.check_instance` |
| objects start on the table and not inside one another; a hinge's spring is not wound past what Isaac turns the right way | `validate.check_instance` |
| each `reach` claim is within 90 % of the arm's nominal reach; each `out_of_reach` claim beyond it; arms `apart` share no point | `validate.check_instance` (and by inverse kinematics with `validate --ik`) |
| doing nothing never succeeds; the keyframes succeed and hit every milestone | `validate.check_instance` |
| the card's robots can do the task and need all three | `tests/test_tasks.py` |
| robot models agree with their pinned URDFs | `tests/test_robots.py` |
| every failure signal names a job of the task | `tests/test_signals.py` |
| importing any module never imports Isaac | `tests/test_cli_and_isolation.py` |

`mrbench validate` runs the rules over both levels and several seeds; `pytest` runs them too.

## Adding a robot

1. Add a `RobotModel` to `robots/models.py` (joints, limits, home, tool point, gripper, wrist camera, capsules).
2. If it comes from a URDF, add a pinned entry to `robots/sources.yaml` and its gains to `engines/isaac/robots.py`.
3. `mrbench assets fetch <name>` and `mrbench assets convert <name>`; `tests/test_robots.py` cross-checks its joints
   and limits against the URDF.

## Adding a method row

See [BASELINES.md](BASELINES.md), "Adding a row".
