"""The skill-level executor: a team plan carried out by the task's own hand-written scripts, no LLM
(docs/BASELINES.md).

A team plan (``submit_team_plan``) says the team, the things and, in words, the steps -- not where a hand goes,
how it grips or when it lets go.  So no code can compile a plan into motions.  What code can do without a model is
what the original baselines' skill libraries did: take, from the scripts the task's authors wrote and played in Isaac
-- each version's reference (``script(inst)``) and the gold routes of ``gold_routes.py`` -- one
whose working arms are all in the plan's team and whose things the plan names, and play it.

So a run succeeds when the plan's team and things are enough by the benchmark's own skills, and the skill works in
Isaac.  It does not test the plan's steps, nor whether the method could write a program: it is the ceiling of the
``exec`` setting (perfect execution of a sufficient team), played.

    python -m mrbench.agent.skill_exec library CASE ...          # the scripts a case has, and their teams
    python -m mrbench.agent.skill_exec dry PLANS.jsonl [--cases=a,b] [--rows=a,b]   # the script each answer gets
    python -m mrbench.agent.skill_exec gold OUT CASE ...          # the gold team of each case, played (Isaac)
    python -m mrbench.agent.skill_exec skill OUT CASE:NAME ...    # named skills, each played once (Isaac)
    python -m mrbench.agent.skill_exec plans PLANS.jsonl OUT [--cases=a,b] [--rows=a,b]  # recorded answers (Isaac)

Isaac runs: ``--gpus=0`` (default), ``--width=N`` at a time (default 2), ``MRBENCH_SIM_TIMEOUT_S`` a run (default
5400), ``MRBENCH_ISAAC_PYTHON``.  Results: ``OUT/results.jsonl``, a run's report in ``OUT/<row>/<case>/``.
"""

from __future__ import annotations

import functools
import importlib
import importlib.util
import json
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from ..core.paths import REPO_ROOT


#: the gold routes taken as skills, with the team each proves (its name and docs/TASKS.md); an arm the route moves
#: that is not in its team only makes room (it moves aside first).  Not taken: ``Con-04-borrow-decoyvalve`` and
#: ``Con-04-shortfall-attempt``, which show that a distractor or too few arms fail
ROUTE_TEAMS = {
    "Con-02-surplus": ("east", "north", "west"),
    "Con-02-surplus-nsw": ("north", "south", "west"),
    "Rea-02-swapped": ("east", "west"),
    "Rea-02-surplus-nsw": ("north", "south", "west"),
    "Rea-01-surplus-nw": ("north", "west"),
    "Rea-04-swapped": ("east", "west"),
    "Con-04-surplus-nsw": ("north", "south", "west"),
    "Con-04-surplus-nes": ("east", "north", "south"),
    "Con-04-surplus-enw": ("east", "north", "west"),
}
REFERENCE = "reference"


@dataclass(frozen=True)
class Skill:
    """A script a case has: ``name`` is "reference" or a route's name."""

    name: str
    team: frozenset
    things: tuple
    #: arms it moves only to make room
    parked: frozenset = frozenset()


@functools.lru_cache(maxsize=1)
def _routes() -> dict:
    from . import gold_routes

    return gold_routes.ROUTES


def _version(case) -> str:
    return f"{case.task}-{case.variant}"


def script_for(case, name: str) -> list:
    """The phases of one of a case's skills, built for its scene."""
    if name == REFERENCE:
        return importlib.import_module(case.inst.spec.impl).script(case.inst)
    if name not in ROUTE_TEAMS or "-".join(name.split("-")[:3]) != _version(case):
        raise ValueError(f"no skill {name!r} for {_version(case)}")
    return _routes()[name](case.inst)


def library(case) -> list[Skill]:
    """The skills a case has: its version's reference (the arms it moves are its team; a borrow version's prop and a
    case's blocks are its things, as in ``gold_plan``) and the gold routes for the version."""
    from .program import acting_arms

    e = case.expect
    things = tuple(e.things) + (tuple(case.blocks) if e.blocks else ())
    try:
        out = [Skill(REFERENCE, frozenset(acting_arms(script_for(case, REFERENCE))), things)]
    except Exception:  # noqa: BLE001 -- a version without a reference of its own
        out = []
    for name, team in ROUTE_TEAMS.items():
        if "-".join(name.split("-")[:3]) == _version(case):
            movers = frozenset(acting_arms(script_for(case, name)))
            out.append(Skill(name, frozenset(team), (), movers - frozenset(team)))
    return out


def choose(case, plan: dict) -> tuple[Optional[Skill], str]:
    """The skill a team plan gets: of those whose team is in the plan's team and whose things the plan names, the
    one with the fewest arms (the reference first on a tie).  None, and why, when there is none."""
    if not plan.get("feasible"):
        return None, "the plan says the task cannot be done"
    team = set(plan.get("arms_used") or [])
    named = {(t.get("thing") or "").partition(":")[0] for t in plan.get("things_used") or []}
    lib = library(case)
    fits = [s for s in lib if s.team <= team and set(s.things) <= named]
    if fits:
        return min(fits, key=lambda s: (len(s.team), s.name != REFERENCE, s.name)), ""
    why = []
    for s in lib:
        lack = sorted(s.team - team)
        miss = sorted(set(s.things) - named)
        why.append(f"{s.name} ({'+'.join(sorted(s.team))}{' with ' + ', '.join(s.things) if s.things else ''}): "
                   + "; ".join(([f"arms not in the team: {', '.join(lack)}"] if lack else [])
                               + ([f"things not named: {', '.join(miss)}"] if miss else [])))
    return None, "no skill fits -- " + " | ".join(why) if why else "the case has no skill"


# ---------------------------------------------------------------------- playing (Isaac, in a child process)


def _wait_for_gpu(gpu: str, mib: int = 8000) -> None:
    ok = 0
    while ok < 2:
        q = subprocess.run(["nvidia-smi", "-i", gpu, "--query-gpu=memory.free", "--format=csv,noheader,nounits"],
                           capture_output=True, text=True)
        ok = ok + 1 if q.returncode == 0 and int(q.stdout.split()[0]) >= mib else 0
        time.sleep(10)


def play(case_id: str, skill: str, out: Path, gpu: str) -> dict:
    """One run of a case's skill in Isaac (``execute.py`` with ``@skill:NAME``); its report."""
    isaac = os.environ.get("MRBENCH_ISAAC_PYTHON", sys.executable)
    timeout = float(os.environ.get("MRBENCH_SIM_TIMEOUT_S", 5400))
    from ..core import plugins

    env = {**os.environ, "CUDA_VISIBLE_DEVICES": gpu, **plugins.child_env(), "OMNI_KIT_ACCEPT_EULA": "YES",
           "OPENBLAS_NUM_THREADS": "1", "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"}
    out.mkdir(parents=True, exist_ok=True)
    _wait_for_gpu(gpu)
    t0 = time.time()
    with (out.parent / f"{out.name}.log").open("w") as f:
        try:
            subprocess.run([isaac, "-m", "mrbench.agent.execute", case_id, f"@skill:{skill}", str(out)], env=env,
                           stdout=f, stderr=f, stdin=subprocess.DEVNULL, timeout=timeout, cwd=REPO_ROOT)
        except subprocess.TimeoutExpired:
            pass
    rep_f = out / "report.json"
    rep = json.loads(rep_f.read_text()) if rep_f.exists() else {"played": False, "error": "did not start or timed out"}
    rep["wall_s"] = round(time.time() - t0)
    return rep


def _jobs_run(jobs: list[dict], out: Path, gpus: list[str], width: int) -> None:
    """Each job {row, case, plan, (mss_right)}: the skill chosen, played; one line a job in ``results.jsonl``."""
    from .cases import case_from_id

    out.mkdir(parents=True, exist_ok=True)

    def one(k_job):
        k, job = k_job
        case = case_from_id(job["case"])
        if job.get("skill"):  # a named skill, played as it is
            skill, why = next(s for s in library(case) if s.name == job["skill"]), ""
        else:
            skill, why = choose(case, job["plan"])
        rec = {"row": job["row"], "case": job["case"], "team": sorted(job["plan"].get("arms_used") or []),
               "said_feasible": bool(job["plan"].get("feasible")), "mss_right": job.get("mss_right"),
               "skill": skill.name if skill else None, "skill_team": sorted(skill.team) if skill else None,
               "why": why}
        if skill is not None:
            where = out / job["row"] / job["case"] / (job["skill"] if job.get("skill") else "")
            rep = play(job["case"], skill.name, where, gpus[k % len(gpus)])
            rec.update(played=bool(rep.get("played")), success=bool(rep.get("success")),
                       milestones=rep.get("milestones_hit"), missed_grips=rep.get("missed_grips"),
                       error=rep.get("error", ""), wall_s=rep.get("wall_s"))
        else:
            rec.update(played=False, success=False)
        with (out / "results.jsonl").open("a") as f:
            f.write(json.dumps(rec) + "\n")
        print(f"{time.strftime('%H:%M:%S')} {rec['row']} {rec['case']}: {rec['skill']} success={rec['success']}",
              flush=True)

    with ThreadPoolExecutor(width) as pool:
        list(pool.map(one, enumerate(jobs)))


def gold_plan_of(case) -> dict:
    """``methods.base.gold_plan`` for a case, without an episode."""
    from types import SimpleNamespace

    from .methods.base import gold_plan

    return gold_plan(SimpleNamespace(case=case, arms=[a.slot for a in case.inst.scene.arms]))


def _recorded(path: Path, cases: Optional[set], rows: Optional[set]) -> list[dict]:
    """Recorded answers, one a line: {method, case, plan | final_plan, mss_right}."""
    out = []
    for line in path.read_text().splitlines():
        g = json.loads(line)
        if (cases and g["case"] not in cases) or (rows and g["method"] not in rows):
            continue
        out.append({"row": g["method"], "case": g["case"], "plan": g.get("plan") or g.get("final_plan") or {},
                    "mss_right": g.get("mss_right")})
    return out


def main(argv=None) -> int:
    from .cases import case_from_id

    argv = list(sys.argv[1:] if argv is None else argv)
    opts = dict(a[2:].split("=", 1) for a in argv if a.startswith("--") and "=" in a)
    args = [a for a in argv if not a.startswith("--")]
    sel = lambda k: set(opts[k].split(",")) if k in opts else None  # noqa: E731
    gpus, width = opts.get("gpus", "0").split(","), int(opts.get("width", 2))
    cmd = args[0] if args else ""
    if cmd == "library":
        for c in args[1:]:
            for s in library(case_from_id(c)):
                print(f"{c}\t{s.name}\t{'+'.join(sorted(s.team))}\t{','.join(s.things)}"
                      f"\t{('parks ' + '+'.join(sorted(s.parked))) if s.parked else ''}")
        return 0
    if cmd == "dry":
        for job in _recorded(Path(args[1]), sel("cases"), sel("rows")):
            skill, why = choose(case_from_id(job["case"]), job["plan"])
            print(json.dumps({"row": job["row"], "case": job["case"], "team": sorted(job["plan"].get("arms_used") or []),
                              "mss_right": job["mss_right"], "skill": skill.name if skill else None, "why": why}))
        return 0
    if cmd == "gold":
        jobs = [{"row": "oracle-team-skill", "case": c, "plan": gold_plan_of(case_from_id(c)), "mss_right": True}
                for c in args[2:]]
        _jobs_run(jobs, Path(args[1]), gpus, width)
        return 0
    if cmd == "skill":  # OUT CASE:NAME ...: named skills, each played once (its team as the plan)
        jobs = []
        for a in args[2:]:
            c, _, name = a.partition(":")
            team = next(s.team for s in library(case_from_id(c)) if s.name == name)
            jobs.append({"row": "skill", "case": c, "skill": name, "plan": {"feasible": True, "arms_used": sorted(team)}})
        _jobs_run(jobs, Path(args[1]), gpus, width)
        return 0
    if cmd == "plans":
        _jobs_run(_recorded(Path(args[1]), sel("cases"), sel("rows")), Path(args[2]), gpus, width)
        return 0
    print(__doc__)
    return 2


if __name__ == "__main__":
    sys.exit(main())
