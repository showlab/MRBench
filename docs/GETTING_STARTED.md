# Getting started

How to install MRBench, look at its tasks, play a scripted reference, and evaluate a method on the benchmark's scenes.

## 1. Install the core

The core -- task cards, the mock engine, the job solver and gold answers, the tool server, the method rows, grading and
the command line -- needs Python 3.10 or 3.11 and no GPU:

```bash
git clone <this repository> MRBench && cd MRBench
python3.10 -m venv .venv && source .venv/bin/activate      # or: uv venv --python 3.10 .venv
pip install -e ".[dev]"                                    # add ,api for --backend api: pip install -e ".[dev,api]"
mrbench validate                                           # every design rule, no GPU
pytest -q
```

Keep this environment at `.venv` in the repository: the tool server of every model session runs with
`.venv/bin/python` when it exists.

```bash
mrbench tasks                         # the eight tasks
mrbench show Con-01                   # one card and a built instance
mrbench brief Con-01 --pretty         # its instruction, facts, objects and arms, as JSON
mrbench methods                       # the method rows
mrbench check Con-04 --levels easy    # plan the scripted reference and check its clearance (no Isaac)
```

## 2. Install Isaac Sim

Simulator runs, pictures and snapshots need **Isaac Sim 4.5.0** and **Isaac Lab 2.1**, with Python 3.10, in an
environment of their own (the dependency set is large and pins hard). You need an RTX GPU with at least 12 GiB free
and an NVIDIA driver supported by Isaac Sim 4.5. The `isaacsim` 4.5 wheels need glibc 2.34 or newer (Ubuntu 22.04+;
`mrbench doctor` prints the version); on an older system use NVIDIA's standalone Isaac Sim 4.5 package with Isaac Lab's
`isaaclab.sh`, or the `nvcr.io/nvidia/isaac-sim:4.5.0` container.

With pip, from the repository's root (about 20 GB, mostly the Isaac Sim wheels):

```bash
python3.10 -m venv .venv-isaac && source .venv-isaac/bin/activate
python -m pip install torch==2.5.1                     # before isaacsim, which would pull a newer one
python -m pip install "isaacsim[all,extscache]==4.5.0" --extra-index-url https://pypi.nvidia.com
git clone --depth 1 --branch v2.1.0 https://github.com/isaac-sim/IsaacLab ../IsaacLab
python -m pip install "setuptools<81"                  # flatdict's build needs pkg_resources
python -m pip install flatdict==4.0.1 --no-build-isolation
python -m pip install -e ../IsaacLab/source/isaaclab -e ../IsaacLab/source/isaaclab_assets
python -m pip install -e ".[isaac,dev]"                # pins warp-lang==1.5.0, the build Kit ships
export OMNI_KIT_ACCEPT_EULA=YES                        # after reading NVIDIA's terms
```

The first Isaac launch on a machine compiles shaders and takes several minutes. Pick a GPU with `--gpus` (or
`MRBENCH_ISAAC_GPU`); `mrbench doctor` reports which has the most free memory.

## 3. Fetch the assets

In the Isaac environment:

```bash
mrbench assets fetch       # the PiPER and SO-101 URDFs and meshes, at pinned commits
mrbench assets convert     # URDF -> USD (starts Isaac headless once)
mrbench assets looks       # the table and floor textures and the studio HDRI (Poly Haven, CC0), sha256-pinned
mrbench assets status      # what is there
mrbench doctor             # Python, GPU, Isaac Sim, assets
```

Downloads go to `third_party/`; set `MRBENCH_THIRD_PARTY` to keep them elsewhere (a local disk). The Panda, the room,
the table legs' material and the YCB objects load from NVIDIA's asset server over HTTP on every run.

## 4. A first scripted reference

```bash
mrbench check Con-04                                   # plan and clearance at both levels (no Isaac)
mrbench demo Con-04 --levels easy --gpus 0             # Isaac: a video of the reference, scored by the task's predicate
mrbench demo Con-04 --configuration borrow --decoy     # the borrow reference with the distractor: it should fail
mrbench render Con-04 --gpus 0                         # the start and goal pictures of docs/media/tasks/
mrbench batch Con-04 --configurations exact borrow --seeds 0-2 --gpus 0 -o runs/batch/try   # many scenes
```

`mrbench validate --ik` checks every reach claim with the arms' inverse kinematics. [TASKS.md](TASKS.md) describes
every task, its reference and its gold answers.

## 5. Take the snapshots

A method sees a scene through its snapshot: every camera's picture and each arm's joint state after a second at rest.
Take them once, in the Isaac environment:

```bash
mrbench snapshot --scenes paper --gpus 0          # the 480 varied scenes -> runs/agent/cases/<scene>/
mrbench snapshot --scenes default --gpus 0,1      # the 40 default scenes
mrbench snapshot --scenes Con-01-exact,Con-01-borrow@hard.s3.varied
```

Scenes already taken are skipped (`--force` takes them again).

## 6. Evaluate a method

In the core environment, with the snapshots in place:

```bash
mrbench evaluate --run first --methods main --scenes paper --setting one-shot \
                 --backend openai --model MODEL --base-url http://127.0.0.1:8000/v1 --parallel 3
```

- `--methods`: `main` (the four compared rows), `references` (`oracle-script`, `oracle-team`), `all`, or a comma
  list (`llm-solver,roco`).
- `--scenes`: `paper` (the 480 varied scenes), `default` (the 40 default scenes) or a comma list of ids.
- `--setting`: `one-shot` (the answer without any run), `equal-budget` (2 probes and 3 runs per scene, with revision
  after a failure) or `exec` (below). [PROTOCOL.md](PROTOCOL.md) has the rules.
- `--backend`: `openai` (any OpenAI-compatible endpoint; `MRBENCH_OPENAI_API_KEY` for an `https://` one), `api` (the
  Anthropic Messages API; `ANTHROPIC_API_KEY`, `--effort`), `claude` (the Claude Code CLI with its own tools off) or
  `fake` (a dry run of the plumbing, never a result).
- `--repeats`, `--parallel` (episodes at once), `--resume` (skip episodes already graded in the run).

`one-shot` needs no simulator beyond the snapshots. `equal-budget` and `exec` play programs in Isaac: point
`MRBENCH_ISAAC_PYTHON` at the Isaac environment's interpreter and spread the runs with `--gpus`:

```bash
export MRBENCH_ISAAC_PYTHON=$PWD/.venv-isaac/bin/python OMNI_KIT_ACCEPT_EULA=YES
mrbench evaluate --run budget --methods main --scenes paper --setting equal-budget \
                 --backend api --model MODEL --gpus 0,1
```

**Carrying out recorded answers.** The `exec` setting hands a one-shot answer to the shared executor once (at most 3
runs, nothing goes back to the method). `--plans-from` takes a JSONL of earlier episodes (one `episode.json` a line):

```bash
python -c "import glob, json; [print(json.dumps(json.load(open(f)))) for f in sorted(glob.glob('runs/agent/bench/first/*/*/r*/episode.json'))]" > first.jsonl
mrbench evaluate --run first-exec --setting exec --plans-from first.jsonl --backend openai --model MODEL --gpus 0
```

[BASELINES.md](BASELINES.md) describes each row.

## 7. The table

```bash
mrbench table --run first
```

It prints, per method and setting, team-selection accuracy, SR, MSS-SR, the mean cost and the errors by kind;
accuracy per configuration; task-level accuracy (all five configurations of a task-scene right) with its 95 %
interval; and, for a row a plugin names as compared, that row against each other row, paired by task-scene. Each
episode is in
`runs/agent/bench/<run>/<method>/<scene>/r<repeat>/`: its ledger, every session's transcript and tool calls, the
outcome and `grade.json`.
