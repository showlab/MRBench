"""Runs the compared methods on scenes and tabulates them (docs/BASELINES.md, docs/PROTOCOL.md).

    mrbench methods
    mrbench evaluate --run NAME [--methods main|references|all|llm-solver,roco] \
        [--scenes paper|default|all|Con-01-exact,Con-01-borrow@hard.s3.varied,...] \
        [--setting one-shot|equal-budget|exec] [--repeats 1] [--parallel 3] \
        [--backend openai|api|claude|fake] [--model MODEL] [--base-url URL] [--gpus 0,1] [--resume]
    mrbench evaluate --run NAME --setting exec --plans-from EPISODES.jsonl
    mrbench table --run NAME

(``python -m mrbench.agent.bench methods|run|table`` is the same.)  ``--scenes paper`` is the paper's 480 varied
scenes: the eight tasks in their five configurations, at both levels and seeds 0-5; ``default`` is each of the 40
versions in its default scene (easy, seed 0, the card's look).  Episodes land in
``runs/agent/bench/<run>/<method>/<scene>/r<k>/`` (``episode.py``), each with its ``grade.json``.  A scene needs
its snapshot in ``runs/agent/cases/<scene>`` (``mrbench snapshot``).  ``--backend fake`` runs every row with a
stand-in for the model and the simulator (``fakes.py``): a dry run of the plumbing, not a result.

``--setting exec`` carries a team plan out in one round (the shared executor's session, three runs at most), and a
failed run does not go back to the method.  With ``--plans-from`` the team plan is the one the method gave in an
earlier run -- a JSONL with an episode a line (``method``, ``case``, ``repeat``, ``final_plan``) -- so the question
is whether that answer can be carried out.
"""

from __future__ import annotations

import argparse
import itertools
import json
import shutil
import sys
from concurrent.futures import ThreadPoolExecutor

from .cases import cases, scene_ids
from .episode import BENCH_DIR, Episode, run_episode
from .grade import grade_dir, table
from .ledger import SETTINGS
from .llm import SANDBOX
from .methods import get, main_rows, reference_rows, registry
from .methods.base import Recorded

#: case groups a plugin adds for ``--cases`` (``add_case_group``): name -> which case ids belong
CASE_GROUPS: dict = {}


def add_case_group(name: str, pick) -> None:
    """``--cases NAME`` runs the cases ``pick(case_id)`` says yes to (``core/plugins.py``)."""
    CASE_GROUPS[name] = pick


def make_backend(a):
    if a.backend == "fake":
        from .fakes import backend

        return backend()
    if a.backend == "openai":
        from .llm import OpenAICompat

        return OpenAICompat(a.model, a.base_url, a.think == "on")
    if a.backend == "api":
        from .llm import ClaudeAPI

        return ClaudeAPI(a.model, a.effort)
    from .llm import ClaudeCLI

    return ClaudeCLI(a.model, a.effort)


def scenes(spec: str) -> list[str]:
    """The scene ids ``--scenes`` names: ``paper`` (the 480 varied scenes), ``default`` (the 40 default scenes),
    ``all`` (both), a plugin's group, or a comma list of ids."""
    every = [f"{t}-{v}" for t, v in cases()]
    bench = [c for c in every if c.startswith(("Con-", "Rea-"))]
    groups = {"default": bench, "paper": scene_ids(bench), "all": bench + scene_ids(bench),
              **{name: [c for c in every if pick(c)] for name, pick in CASE_GROUPS.items()}}
    return groups.get(spec) or spec.split(",")


#: each row as the paper names and describes it
ROWS = {
    "llm-solver": "LLM + solver: the model translates the scene into jobs and doers; exact search counts the arms",
    "smart-llm": "SMART-LLM-style: decomposition, coalition formation and allocation, three sessions",
    "roco": "RoCo-style: one model per arm, in dialogue, until all agree on a plan",
    "hmas2": "HMAS-2-style: a central planner proposes; each arm accepts or vetoes its part",
    "oracle-script": "Oracle-script: the gold team's scripted reference, replayed (exact)",
    "oracle-team": "Oracle-team: the gold team handed to the shared executor",
    "oracle-team-skill": "Oracle-team (skill): the gold team carried out by the tasks' own scripted skills",
}


def cmd_methods(a) -> int:
    for name, m in registry().items():
        description = (m.__doc__ or "").strip().splitlines()
        print(f"{name:18s} {ROWS.get(name) or (description[0] if description else m.__name__)}")
    return 0


def cmd_run(a) -> int:
    if a.backend != "fake" and not a.model:
        raise SystemExit("--model is needed with --backend " + a.backend)
    methods = {"all": list(registry()), "main": list(main_rows()), "references": list(reference_rows())}.get(a.methods) \
        or a.methods.split(",")
    ids = scenes(a.scenes)
    recorded: dict = {}  # method -> (case, repeat) -> the team plan it gave before; a later line replaces an earlier
    for f in filter(None, (a.plans_from or "").split(",")):
        for line in open(f):
            g = json.loads(line)
            recorded.setdefault(g["method"], {})[g["case"], g["repeat"]] = g.get("final_plan")
    gpus = itertools.cycle(a.gpus.split(","))
    jobs = [(m, c, k, next(gpus)) for m in methods for c in ids for k in range(1, a.repeats + 1)]
    if a.resume:  # a graded episode stays; one cut off midway starts again from nothing
        jobs = [j for j in jobs if not (BENCH_DIR / a.run / j[0] / j[1] / f"r{j[2]}" / "grade.json").exists()]

    def one(job):
        m, c, k, gpu = job
        root = BENCH_DIR / a.run / m / c / f"r{k}"
        if a.resume:
            for d in (root, SANDBOX / a.run / m / c / f"r{k}"):
                shutil.rmtree(d, ignore_errors=True)
        try:
            ep = Episode(m, c, a.setting, k, a.run, make_backend(a), gpu, root=root)
            run_episode(Recorded(get(m), recorded[m]) if m in recorded else get(m), ep)
            g = grade_dir(ep.root)
        except Exception as e:  # noqa: BLE001 -- one episode's crash is reported, the others go on
            import traceback

            root.mkdir(parents=True, exist_ok=True)
            (root / "crash.log").write_text(traceback.format_exc())
            g = {"method": m, "case": c, "repeat": k, "crash": f"{type(e).__name__}: {e}"}
        print(json.dumps({key: g.get(key) for key in ("method", "case", "repeat", "ended", "mss_right", "success",
                                                      "team", "error", "cost_usd", "crash")}), flush=True)
        return g

    with ThreadPoolExecutor(a.parallel) as pool:
        rows = list(pool.map(one, jobs))
    print(table([r for r in rows if "crash" not in r]))
    return 0


def cmd_table(a) -> int:
    rows = [json.loads(f.read_text()) for f in sorted((BENCH_DIR / a.run).glob("*/*/r*/grade.json"))]
    print(table(rows))
    return 0


def add_parsers(sub, aliases=()) -> None:
    """``methods``, ``evaluate`` and ``table``, on ``mrbench``'s subparsers (``mrbench run`` plays scripted
    references) or this module's (where ``run`` is ``evaluate``)."""
    sub.add_parser("methods", help="the method rows").set_defaults(fn=cmd_methods)
    r = sub.add_parser("evaluate", aliases=list(aliases), help="run methods on scenes and grade them")
    r.add_argument("--run", required=True, help="the run's name (runs/agent/bench/<run>/)")
    r.add_argument("--methods", default="main", help="main (the compared rows), references, all, or a comma list")
    r.add_argument("--scenes", "--cases", dest="scenes", default="paper",
                   help="paper (the 480 varied scenes), default (the 40 default scenes), all, or a comma list of ids")
    r.add_argument("--setting", default="one-shot", choices=SETTINGS)
    r.add_argument("--plans-from", help="JSONL of earlier episodes: the methods named in it carry out the team plans "
                                         "they gave there instead of planning again")
    r.add_argument("--repeats", type=int, default=1)
    r.add_argument("--parallel", type=int, default=3)
    r.add_argument("--resume", action="store_true", help="skip the episodes already graded in this run")
    r.add_argument("--gpus", default="0", help="Isaac runs: GPUs to spread the episodes over")
    r.add_argument("--backend", default="openai", choices=["openai", "api", "claude", "fake"],
                   help="openai: an OpenAI-compatible endpoint (--base-url); api: the Anthropic Messages API; "
                        "claude: the Claude Code CLI with its own tools off; fake: a dry run")
    r.add_argument("--model", default=None, help="the model name the backend expects (not needed with fake)")
    r.add_argument("--effort", default="high", help="reasoning effort (api, claude)")
    r.add_argument("--base-url", default="http://127.0.0.1:8000/v1", help="openai: the endpoint")
    r.add_argument("--think", default="on", choices=["on", "off"], help="openai: ask for thinking")
    r.set_defaults(fn=cmd_run)
    t = sub.add_parser("table", help="tabulate a run")
    t.add_argument("--run", required=True)
    t.set_defaults(fn=cmd_table)


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="python -m mrbench.agent.bench")
    add_parsers(p.add_subparsers(dest="cmd", required=True), aliases=("run",))
    a = p.parse_args(argv)
    return a.fn(a)


if __name__ == "__main__":
    sys.exit(main())
