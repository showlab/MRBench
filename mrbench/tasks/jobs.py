"""Jobs: what a task asks to be done, who on the table can do each, and the fewest arms that can do them all.

A task instance has **jobs** (its impl's ``jobs(inst)``).  A job has **sites** -- named points of the instance (a body,
a region, a cloth's key point "towel:end_l", resolved as ``validate --ik`` resolves a reach claim), explicit points
(x, y, z), or a choice of points (``AnyOf``: a bar held anywhere along it) -- and a **group**: the jobs of one group
are done at the same time (something held while another hand works).  Two kinds:

* ``act`` -- one doer reaches every site;
* ``carry`` -- a body from its first site to its second: one doer that reaches both, or a chain of doers in which
  each two in turn meet -- a place both reach from their own sides, their tool points ``meet_slack`` apart across the
  thing, in the air or on the table, clear of every fixed body (``reachability.shared_point``, the search ``validate
  --ik`` makes for an ``apart`` claim, so walls and partitions count).

The **doers** are the arms on the table and the **things** a task declares (``Thing``): a thing does the jobs it is
declared for once it is in place, and putting it there is a job, or jobs, for an arm (its ``placing``).  A declared
thing that does no job is a distractor; its ``why`` is the one property that keeps it from working.

An arm **can** do a site when inverse kinematics with margins gets its tool point there -- the rule by which ``validate
--ik --slack`` breaks an "out of reach" claim (``reachability.reaches``, and ``near_points``: a body's surface within
a grip's reach, a place ``slack`` short of its middle), with the job's ``slack``.  A claim that an arm cannot do a job
is therefore a claim ``validate --ik`` makes, and the margins never make an arm less capable than its middle point.

A **team** (a set of arms) is feasible when the jobs can be assigned: each job a capable doer from the team, or a
usable thing (whose placing jobs are then assigned to the team too), and no doer on two jobs of one group.  The answer
(``solve``): every feasible team of the fewest arms -- the minimal support sets, each accepted -- with the things they
use and the things left out; or, when no team is feasible, the jobs that have no doer.  The search is exhaustive over
the subsets of the arms (at most four) and the assignments.

The model is kinematic: a job is decided by its sites, its group and who reaches them (a hand that reaches a lid
holds it).  The IK results are kept in ``tasks/facts/<task>.json``,
keyed by a digest of everything that decides them (the arm's place, its robot's kinematics and collision capsules,
the reach check's settings, the points, the fixed bodies: ``robot_key``), so that the tests can replay them without
solving IK again, and a change to any of those asks for them again:

    python -m mrbench.tasks.jobs show Con-01 [--variant V --level L --seed S]   # who can do each job, the answer
    python -m mrbench.tasks.jobs facts Con-01 ...                               # solve and keep every version's IK
    python -m mrbench.tasks.jobs gold [--check]                                 # the agent cases' gold (agent/gold.json)
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import importlib
import itertools
import json
import os
import sys
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Optional, Union

from ..core.errors import MRBenchError

KINDS = ("act", "carry")
#: where the IK results are kept, a file a task
FACTS_DIR = Path(__file__).parent / "facts"
#: a site's point is at least this far over the table (``reachability.check``'s rule for a named target)
LOW = 0.04


@dataclass(frozen=True)
class AnyOf:
    """A site reached by reaching any one of ``points`` (each a name or an (x, y, z))."""

    points: tuple


#: a name, an (x, y, z), a choice of them, or None: nothing there to take (no arm can do the job)
Site = Union[str, tuple, AnyOf, None]


@dataclass(frozen=True)
class Job:
    #: what must be done, in words; unique in its instance
    name: str
    kind: str = "act"
    #: act: every site one doer must reach; carry: (from, to)
    sites: tuple = ()
    #: jobs with the same group are done at the same time ("" for none)
    group: str = ""
    #: the margin a hand has at this job's sites (``reachability.near_points``): how far short of a place's middle it
    #: still does the job; a body's surface within a grip's reach counts whenever slack > 0
    slack: float = 0.0
    #: carry: the body carried
    body: str = ""

    def __post_init__(self):
        if self.kind not in KINDS:
            raise ValueError(f"job {self.name!r}: kind {self.kind!r} is not one of {KINDS}")
        if self.kind == "carry" and len(self.sites) != 2:
            raise ValueError(f"job {self.name!r}: a carry has two sites, from and to")


@dataclass(frozen=True)
class Thing:
    #: the body on the table
    name: str
    #: the jobs (by name) it does, all at once, once it is in place; () -- a distractor: it does none
    does: tuple[str, ...] = ()
    #: what putting it in place asks of the arms
    placing: tuple[Job, ...] = ()
    #: why it works, or, for a distractor, the one property that keeps it from working
    why: str = ""


@dataclass(frozen=True)
class Jobs:
    jobs: tuple[Job, ...]
    things: tuple[Thing, ...] = ()
    #: two hands passing a thing: their tool points this far apart across it (``reachability.shared_point``)
    meet_slack: float = 0.05

    def __post_init__(self):
        names = [j.name for j in self.all_jobs()]
        if len(set(names)) != len(names):
            raise ValueError(f"two jobs share a name: {names}")
        for t in self.things:
            unknown = set(t.does) - {j.name for j in self.jobs}
            if unknown:
                raise ValueError(f"{t.name} is said to do {sorted(unknown)}, which are not jobs")

    def all_jobs(self) -> tuple[Job, ...]:
        return self.jobs + tuple(p for t in self.things for p in t.placing)

    def job(self, name: str) -> Job:
        return next(j for j in self.all_jobs() if j.name == name)


@dataclass(frozen=True)
class Answer:
    feasible: bool
    #: every feasible team of the fewest arms (empty when none is)
    minimal: tuple[frozenset, ...] = ()
    #: the things those teams use, and the declared things they leave out
    things_used: tuple[str, ...] = ()
    things_rejected: tuple[str, ...] = ()
    #: infeasible: the jobs no doer can do (or, if each has one, the jobs of a group no team can staff at once)
    missing: tuple[str, ...] = ()
    #: for each minimal team, in order: job -> who does it (arms, or a thing)
    plans: tuple[dict, ...] = field(default=(), compare=False)
    #: the feasible teams of more arms than the fewest (right, not minimal)
    workable: tuple[frozenset, ...] = ()


class MissingFact(MRBenchError):
    """An IK result the facts file does not have (offline): solve it again with ``python -m mrbench.tasks.jobs facts``."""


# ------------------------------------------------------------------------------------------------ the IK results
def _digest(obj) -> str:
    return hashlib.sha1(json.dumps(obj, sort_keys=True, separators=(",", ":")).encode()).hexdigest()[:20]


def _r(v, nd: int = 4):
    return [round(float(x), nd) + 0.0 for x in v]


class Facts:
    """IK results by digest: those in ``FACTS_DIR``, and those solved here.  ``offline``: never solve (a result not
    kept is a ``MissingFact``)."""

    def __init__(self, offline: bool = False, path: Path = FACTS_DIR):
        self.offline, self.path = offline, path
        self.mem: dict[str, int] = {}
        self.new: dict[str, dict[str, int]] = {}
        self.used: dict[str, set[str]] = {}
        self._loaded: set[str] = set()

    def _file(self, task: str) -> Path:
        return self.path / f"{task}.json"

    def _load(self, task: str) -> None:
        if task not in self._loaded:
            self._loaded.add(task)
            f = self._file(task)
            if f.exists():
                self.mem.update(json.loads(f.read_text())["facts"])

    def get(self, task: str, key: str, solve, what: str) -> int:
        self._load(task)
        self.used.setdefault(task, set()).add(key)
        if key in self.mem:
            return self.mem[key]
        if self.offline:
            raise MissingFact(f"{task}: no IK result kept for {what}; solve it again: python -m mrbench.tasks.jobs "
                              f"facts {task}")
        v = int(bool(solve()))
        self.mem[key] = v
        self.new.setdefault(task, {})[key] = v
        return v

    def save(self) -> list[Path]:
        """Add what was solved here to the facts files (under a lock: several processes may save at once)."""
        out = []
        for task, new in self.new.items():
            if not new:
                continue
            f = self._file(task)
            f.parent.mkdir(parents=True, exist_ok=True)
            with _locked(f.parent):
                kept = json.loads(f.read_text())["facts"] if f.exists() else {}
                kept.update(new)
                _write(f, task, kept)
            out.append(f)
            new.clear()
        return out

    def prune(self, task: str) -> int:
        """Keep in ``task``'s file only the results asked for here (after solving every version: the others are
        left from scenes or jobs that changed); how many went."""
        f = self._file(task)
        with _locked(f.parent):
            kept = json.loads(f.read_text())["facts"]
            _write(f, task, {k: v for k, v in kept.items() if k in self.used.get(task, ())})
        return len(kept) - len(self.used.get(task, ()) & set(kept))


def _write(f: Path, task: str, facts: dict) -> None:
    """``f`` replaced whole (a reader never sees it half written)."""
    tmp = f.with_name(f".{f.name}.{os.getpid()}")
    tmp.write_text(json.dumps({"task": task, "facts": dict(sorted(facts.items()))}, indent=0) + "\n")
    os.replace(tmp, f)


@contextlib.contextmanager
def _locked(directory: Path):
    """The facts directory locked against the other processes saving (on the directory itself: nothing to clean up)."""
    import fcntl

    fd = os.open(directory, os.O_RDONLY)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX)
        yield
    finally:
        fcntl.flock(fd, fcntl.LOCK_UN)
        os.close(fd)


@lru_cache(maxsize=None)
def robot_key(model: str) -> str:
    """What else decides an IK result: the robot's joints, limits, home and tool point, its collision capsules and how
    far its hand opens, and the reach check's own settings -- so that a change to any of them asks for the result
    again instead of reusing it."""
    from ..robots import collision as col
    from ..robots import get_model
    from ..robots import kinematics as kin
    from . import reachability as rc

    m = get_model(model)
    chain = [[_r(j.origin.flatten()), None if j.axis is None else _r(j.axis)] for j in kin.chain(model)]
    return _digest([chain, [list(map(float, lim)) for lim in m.arm_limits], _r(m.home), m.tcp_offset_m,
                    repr(col._LINKS[model]), round(col.grip_span_m(m), 4),
                    [rc.TABLE_GAP, list(rc.PITCHES), rc.SEEDS, rc.GRIP_REACH, list(rc.APART_Z), list(rc.APART_ALONG),
                     list(rc.APART_ACROSS)]])


class Capability:
    """Who in an instance's scene can do what: its arms' reach at sites, and where two arms meet, by IK (``Facts``)."""

    def __init__(self, inst, facts: Optional[Facts] = None):
        from ..robots import collision as col

        self.inst, self.sc = inst, inst.scene
        self.task = inst.spec.id
        self.facts = facts if facts is not None else Facts()
        fixed = [(o.name, b) for o in self.sc.objects if not o.dynamic for b in col.boxes_of(o, o.pose)]
        self.fixed = [b for _, b in fixed]
        self.scene_key = _digest([[n, _r(b.center), _r(b.rot.flatten()), _r(b.half)] for n, b in fixed] +
                                 [_r(self.sc.table_size), round(self.sc.table_top_z, 4)])

    def _arm(self, slot: str) -> list:
        a = self.sc.arm(slot)
        return [a.model, _r(a.pos), round(a.yaw_deg, 3), robot_key(a.model)]

    def point(self, target) -> tuple:
        p = list(self.sc.point(target)) if isinstance(target, str) else [float(v) for v in target]
        p[2] = max(p[2], self.sc.table_top_z + LOW)
        return tuple(p)

    def reach(self, slot: str, site: Site, slack: float = 0.0) -> bool:
        """Whether ``slot`` can do ``site`` (any of an ``AnyOf``'s points), with ``slack``."""
        if site is None:
            return False
        return any(self._reach(slot, s, slack) for s in (site.points if isinstance(site, AnyOf) else (site,)))

    def _reach(self, slot: str, target, slack: float) -> bool:
        from .reachability import near_points, reaches

        p = self.point(target)
        near = near_points(self.sc, slot, target, slack) if slack > 0 else []
        key = "r" + _digest([self._arm(slot), _r(p), [_r(q) for q in near], self.scene_key])
        what = f"{slot} ({self.sc.arm(slot).model}) at {target if isinstance(target, str) else _r(target, 3)}"
        return bool(self.facts.get(self.task, key, lambda: reaches(self.sc, slot, p) or
                                   any(reaches(self.sc, slot, q, hand_free=True) for q in near), what))

    def meet(self, a: str, b: str, slack: float) -> bool:
        """Whether ``a`` and ``b`` share a place to pass a thing at (``reachability.shared_point``)."""
        from .reachability import shared_point

        a, b = sorted((a, b), key=lambda s: json.dumps(self._arm(s)))
        key = "m" + _digest([self._arm(a), self._arm(b), round(slack, 4), self.scene_key])
        return bool(self.facts.get(self.task, key, lambda: shared_point(self.sc, a, b, self.fixed, slack) is not None,
                                   f"where {a} ({self.sc.arm(a).model}) and {b} ({self.sc.arm(b).model}) meet"))

    # -------------------------------------------------------------------------------------- jobs
    def options(self, job: Job, team, meet_slack: float, every: bool = False) -> list[frozenset]:
        """The ways ``team`` can do ``job``, each the set of arms it takes; with ``every`` all of them, else the first
        (an act: each capable arm; a carry: each arm that reaches both ends, else each chain of arms that meet)."""
        if job.kind == "act":
            out = []
            for a in team:
                if all(self.reach(a, s, job.slack) for s in job.sites):
                    out.append(frozenset([a]))
                    if not every:
                        break
            return out
        start, end = job.sites
        out = [frozenset([a]) for a in team if self.reach(a, start, job.slack) and self.reach(a, end, job.slack)]
        if out and not every:
            return out[:1]
        starts = [a for a in team if self.reach(a, start, job.slack)]
        ends = {a for a in team if self.reach(a, end, job.slack)}
        for k in range(2, len(team) + 1):
            for chain in itertools.permutations(team, k):
                if chain[0] not in starts or chain[-1] not in ends or frozenset(chain) in out:
                    continue
                if any(frozenset(o) < frozenset(chain) for o in out):
                    continue  # a shorter chain of the same arms does it
                if all(self.meet(x, y, meet_slack) for x, y in zip(chain, chain[1:])):
                    out.append(frozenset(chain))
                    if not every:
                        return out
        return out


def declared(inst) -> Jobs:
    """The instance's jobs, from its impl's ``jobs(inst)``."""
    mod = importlib.import_module(inst.spec.impl)
    if not hasattr(mod, "jobs"):
        raise MRBenchError(f"{inst.spec.id}: its impl declares no jobs (a jobs(inst) function)")
    return mod.jobs(inst)


def _staff(decl: Jobs, cap: Capability, team, used) -> Optional[dict]:
    """An assignment of the jobs to ``team`` and the things ``used`` (job -> doers), or None."""
    covered = {j: t.name for t in used for j in t.does}
    todo = [j for j in decl.jobs if j.name not in covered] + [p for t in used for p in t.placing]
    plan: dict = {}
    groups: dict[str, list] = {}
    for j in decl.jobs:
        if j.name in covered and j.group:
            groups.setdefault(j.group, []).append((j.name, [frozenset([covered[j.name]])]))
        elif j.name in covered:
            plan[j.name] = frozenset([covered[j.name]])
    for j in todo:
        opts = cap.options(j, team, decl.meet_slack, every=bool(j.group))
        if not opts:
            return None
        if j.group:
            groups.setdefault(j.group, []).append((j.name, opts))
        else:
            plan[j.name] = opts[0]
    for g, entries in groups.items():
        pick = _distinct(entries)
        if pick is None:
            return None
        plan.update(pick)
    return plan


def _distinct(entries, taken=frozenset()) -> Optional[dict]:
    """One option a job, no doer in two (backtracking): {job: doers}, or None."""
    if not entries:
        return {}
    (name, opts), rest = entries[0], entries[1:]
    for o in opts:
        if o & taken:
            continue
        r = _distinct(rest, taken | o)
        if r is not None:
            return {name: o, **r}
    return None


def assign(decl: Jobs, cap: Capability, team) -> Optional[tuple[dict, tuple[str, ...]]]:
    """(job -> doers, the things used) for ``team``, using as few things as can be, or None if it cannot."""
    usable = [t for t in decl.things if t.does]
    for k in range(len(usable) + 1):
        for used in itertools.combinations(usable, k):
            plan = _staff(decl, cap, team, used)
            if plan is not None:
                return plan, tuple(t.name for t in used)
    return None


#: a doer that is not there: one more hand, able to do one job (``missing``)
ANOTHER = "(another hand)"


def missing(decl: Jobs, cap: Capability, arms) -> tuple[str, ...]:
    """What the arms lack: the jobs no doer can do (a job a usable thing does counts as done when its placing can be
    done); or, when each has a doer but no assignment staffs a group at once, the jobs one more hand would do to make
    it feasible -- any one of them, the ones fewest arms can do first."""
    usable = [t for t in decl.things if t.does and all(cap.options(p, arms, decl.meet_slack) for p in t.placing)]
    done_by_thing = {j for t in usable for j in t.does}
    out = tuple(j.name for j in decl.jobs if j.name not in done_by_thing and not cap.options(j, arms, decl.meet_slack))
    if out:
        return out
    entries = {g: [(j, cap.options(j, arms, decl.meet_slack, every=True) +
                    [frozenset([t.name]) for t in usable if j.name in t.does]) for j in decl.jobs if j.group == g]
               for g in dict.fromkeys(j.group for j in decl.jobs if j.group)}
    short = []
    for g, es in entries.items():
        if _distinct([(j.name, o) for j, o in es]) is not None:
            continue
        for j, opts in sorted(es, key=lambda e: len(e[1])):
            more = [(k.name, o + [frozenset([ANOTHER])] if k is j else o) for k, o in es]
            if _distinct(more) is not None:
                short.append(j.name)
    return tuple(short)


def solve(inst, facts: Optional[Facts] = None, decl: Optional[Jobs] = None, cap: Optional[Capability] = None) -> Answer:
    """The fewest-arm teams that can do the instance's jobs (exhaustive), or the jobs that have no doer.  ``cap``:
    another account of who reaches what (a test's)."""
    decl = decl or declared(inst)
    cap = cap or Capability(inst, facts)
    arms = [a.slot for a in inst.scene.arms]
    found, larger = [], []
    for k in range(1, len(arms) + 1):
        for team in itertools.combinations(arms, k):
            r = assign(decl, cap, team)
            if r is not None:
                (larger if found and len(team) > len(found[0][0]) else found).append((frozenset(team), *r))
    if found:
        used = tuple(sorted({t for _, _, ts in found for t in ts}))
        rejected = tuple(t.name for t in decl.things if t.name not in used)
        plans = tuple({j: sorted(d) for j, d in p.items()} for _, p, _ in found)
        return Answer(True, tuple(t for t, _, _ in found), used, rejected, (), plans,
                      tuple(t for t, _, _ in larger))
    return Answer(False, (), (), tuple(t.name for t in decl.things), missing(decl, cap, arms))


def who_can(inst, facts: Optional[Facts] = None, decl: Optional[Jobs] = None) -> dict[str, list[str]]:
    """Each job (and each thing's placing job) -> every way the instance's arms can do it ("west", "west+south")."""
    decl = decl or declared(inst)
    cap = Capability(inst, facts)
    arms = [a.slot for a in inst.scene.arms]
    return {j.name: ["+".join(sorted(o)) for o in cap.options(j, arms, decl.meet_slack, every=True)]
            for j in decl.all_jobs()}


# ------------------------------------------------------------------------------------------------ command line
def _versions(task: str, variants, levels, seeds):
    """The agent cases' scenes (``agent.cases``: each version's team), as (variant, level, seed, instance)."""
    from ..agent.cases import make_case, setups

    for v in variants:
        if v not in setups()[task].expect and v not in gold_variants(task):
            continue
        for lv in levels:
            for s in seeds:
                yield v, lv, s, make_case(task, v, lv, s, check_gold=False).inst


def gold_variants(task: str) -> tuple[str, ...]:
    """The versions of ``task`` the solver gives the gold of: its agent cases' (``agent.cases.GOLD_VARIANTS``)."""
    from ..agent.cases import GOLD_VARIANTS, has_variant

    return tuple(v for v in GOLD_VARIANTS if has_variant(task, v))


def _fmt(a: Answer) -> str:
    if not a.feasible:
        return f"infeasible; missing: {'; '.join(a.missing)}"
    teams = " | ".join("+".join(sorted(t)) for t in a.minimal)
    things = f"; using {', '.join(a.things_used)}" if a.things_used else ""
    rej = f"; not {', '.join(a.things_rejected)}" if a.things_rejected else ""
    return f"{teams}{things}{rej}"


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="python -m mrbench.tasks.jobs", description=__doc__.split("\n\n")[0])
    sub = p.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("show", help="who can do each job, and the answer")
    s.add_argument("task")
    s.add_argument("--variant", default="exact")
    s.add_argument("--level", default="easy")
    s.add_argument("--seed", type=int, default=0)
    s.add_argument("--offline", action="store_true", help="only the IK results kept")
    f = sub.add_parser("facts", help="solve every version's jobs and keep the IK results")
    f.add_argument("tasks", nargs="+")
    f.add_argument("--variants", nargs="+", default=None)
    f.add_argument("--levels", nargs="+", default=["easy", "hard"])
    f.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4, 5])
    f.add_argument("--prune", action="store_true", help="then keep only the results these versions use (give every "
                                                         "version, level and seed: the defaults)")
    g = sub.add_parser("gold", help="the agent cases' gold, from the solver (agent/gold.json)")
    g.add_argument("--check", action="store_true", help="only compare (offline), do not write")
    a = p.parse_args(argv)

    if a.cmd == "show":
        from ..agent.cases import make_case

        facts = Facts(offline=a.offline)
        inst = make_case(a.task, a.variant, a.level, a.seed, check_gold=False).inst
        decl = declared(inst)
        print(f"{a.task} {a.variant} {a.level} s{a.seed}: arms " +
              ", ".join(f"{x.slot} ({x.model})" for x in inst.scene.arms))
        for j, ways in who_can(inst, facts, decl).items():
            print(f"  {j}: {', '.join(ways) or 'no one'}")
        print("  answer:", _fmt(solve(inst, facts, decl)))
        facts.save()
        return 0
    if a.cmd == "facts":
        facts = Facts()
        for t in a.tasks:
            for v, lv, s, inst in _versions(t, a.variants or gold_variants(t), a.levels, a.seeds):
                print(f"{t} {v} {lv} s{s}: {_fmt(solve(inst, facts))}", flush=True)
                facts.save()
            if a.prune:
                print(f"{t}: {facts.prune(t)} result(s) no version uses dropped")
        return 0
    from ..agent import gold as gold_mod

    return gold_mod.main(check=a.check)


if __name__ == "__main__":  # run as mrbench.tasks.jobs, not as __main__: the impls' AnyOf must be this module's
    from mrbench.tasks import jobs as _jobs

    sys.exit(_jobs.main())
