"""One episode: one method on one case, one repeat, in one setting -- the loop every compared method shares
(docs/PROTOCOL.md, "Settings"):

    round 1: the method decides a team (its own procedure, its own sessions)        -> a team plan
             one-shot: that is the answer.  equal budget: the team plan is carried out --
             by the shared executor, or by the method's own -- and played in the simulator
    round n: a run that did not succeed sends its failure report back to the method's FIRST step, which plans again
             from the start with the report and its last plan, and may change the team

until a run succeeds, the method says the task cannot be done, the runs or the LLM budget are spent, or the
method gives up.  The answer graded is the last team plan; the team is the arms declared plus the arms that acted in
the run that succeeded (``grade.py``).

Everything lands in ``runs/agent/bench/<run>/<method>/<case>/r<repeat>/``: ``ledger.json`` (budgets, every plan,
program, run and session), one folder per session (its transcript and the environment's call log), and
``episode.json`` with the outcome.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Optional

from ..core.paths import REPO_ROOT
from . import failure
from .cases import Case, case_from_id
from .env import EnvConfig
from .ledger import Ledger
from .llm import SANDBOX, Backend, SessionResult, SessionSpec
from .observe import Observer

if TYPE_CHECKING:
    from .methods.base import Method

CASES_DIR = REPO_ROOT / "runs" / "agent" / "cases"
BENCH_DIR = REPO_ROOT / "runs" / "agent" / "bench"
#: rounds at most (each round after the first follows a failed run, so the run budget bounds them as well)
MAX_ROUNDS = 6
#: rounds in a row in which no program got as far as a run (none passed its check): then the episode ends
MAX_DRY_ROUNDS = 2


class OutOfBudget(Exception):
    """The episode's LLM budget (sessions or dollars) is spent."""


@dataclass
class ExecResult:
    """What carrying out a team plan came to."""
    success: bool
    #: the failure report of the last run (or of the last check, if nothing ran); None after a success
    report: Optional[dict] = None
    #: the arms that acted in the run that succeeded, or in the last run
    movers: list = field(default_factory=list)
    note: str = ""


@dataclass
class Episode:
    method: str
    case_id: str
    setting: str
    repeat: int
    run: str
    backend: Backend
    gpu: str = "0"
    root: Path = field(default=None)
    snapshot: Path = field(default=None)

    def __post_init__(self):
        self.root = Path(self.root or BENCH_DIR / self.run / self.method / self.case_id / f"r{self.repeat}")
        self.snapshot = Path(self.snapshot or CASES_DIR / self.case_id)
        self.case: Case = case_from_id(self.case_id)
        self.arms = [a.slot for a in self.case.inst.scene.arms]
        self.round = 0
        self.ledger: Ledger = None

    # ------------------------------------------------------------------ what a method uses
    def start(self, limits: Optional[dict] = None) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        self.ledger = Ledger.create(self.root / "ledger.json", self.setting, limits, method=self.method,
                                    case=self.case_id, repeat=self.repeat, backend=self.backend.label)

    @property
    def can_try(self) -> bool:
        return self.setting in ("equal-budget", "exec") and self.ledger.left("runs") > 0

    def config(self, name: str, tools, **kw) -> EnvConfig:
        tools = list(tools)
        if self.setting == "one-shot":  # nothing is played before the team is submitted
            tools = [t for t in tools if t not in ("try_in_sim", "probe")]
        return EnvConfig(case=self.case_id, snapshot=str(self.snapshot), log=str(self.root / name),
                         ledger=str(self.ledger.path), tools=tools, gpu=self.gpu, session=name, **kw)

    def session(self, name: str, system: str, first: str, tools=None, backend: Optional[Backend] = None,
                **env_kw) -> SessionResult:
        """Run one session (``tools`` None: a plain completion).  Counts it and its cost against the LLM budget."""
        name = f"round{self.round}/{name}"
        if self.ledger.take("sessions") is None or self.ledger.exhausted() == "cost_usd":
            raise OutOfBudget(self.ledger.exhausted() or "sessions")
        cfg = self.config(name, tools, **env_kw) if tools is not None else None
        spec = SessionSpec(name, system, first, cfg)
        box = SANDBOX / self.run / self.method / self.case_id / f"r{self.repeat}" / name
        r = (backend or self.backend).run(spec, self.root / name, box)
        self.ledger.spend("cost_usd", r.cost_usd)
        self.ledger.add("sessions", {"name": name, "backend": (backend or self.backend).label, "seconds": r.seconds,
                                     "turns": r.turns, "cost_usd": r.cost_usd, "tool_calls": r.tool_calls,
                                     "exit": r.exit, "output": sorted(r.output)})
        return r

    def observer(self) -> Observer:
        """The shared observation, in-process, for a row that is a rule: what ``briefing`` gives."""
        return Observer(self.case, self.snapshot)

    def record_plan(self, plan: dict, by: str) -> None:
        """A team plan a method decided without a session's ``submit_team_plan`` (a rule, an oracle)."""
        from .env import normalise_plan, validate_plan

        plan = normalise_plan(plan)
        problems = validate_plan(plan, self.arms, {t["name"] for t in self.observer().things()})
        if problems:
            raise ValueError(f"{by}: {problems}")
        self.ledger.add("plans", {"session": f"round{self.round}/{by}", "plan": plan})

    def latest_plan(self, since_round: Optional[int] = None) -> Optional[dict]:
        plans = self.ledger.records("plans")
        if since_round is not None:
            plans = [p for p in plans if p["session"].startswith(f"round{since_round}/")]
        return plans[-1]["plan"] if plans else None

    def runs(self, since_round: Optional[int] = None) -> list[dict]:
        rs = self.ledger.records("runs")
        if since_round is not None:
            rs = [r for r in rs if r["session"].startswith(f"round{since_round}/")]
        return rs

    def result_of_round(self) -> ExecResult:
        """What this round's runs came to: a success, or the last run's failure report."""
        rs = self.runs(self.round)
        for r in rs:
            if r["success"]:
                return ExecResult(True, None, r.get("movers", []))
        if rs:
            last = rs[-1]
            rep = json.loads(Path(last["report"]).read_text()) if Path(last["report"]).exists() else None
            return ExecResult(False, failure.report(self.case, rep), last.get("movers", []))
        progs = [p for p in self.ledger.records("programs") if p["session"].startswith(f"round{self.round}/")]
        if progs:
            chk = Path(progs[-1]["plan"]).with_name(f"check_{progs[-1]['k']}.json")
            check = json.loads(chk.read_text()) if chk.exists() else None
            return ExecResult(False, failure.report(self.case, None, check), [], "no run: the program did not pass")
        return ExecResult(False, failure.report(self.case, None), [], "no program")


# ---------------------------------------------------------------------- the loop
def run_episode(method: "Method", ep: Episode) -> dict:
    ep.start(method.limits(ep.setting))
    t0 = time.time()
    feedback: Optional[dict] = None
    outcome = {"method": method.name, "case": ep.case_id, "setting": ep.setting, "repeat": ep.repeat,
               "backend": ep.backend.label, "rounds": 0, "success": False, "ended": ""}
    if ep.setting == "one-shot" and not method.one_shot:
        outcome["ended"] = "not run: the method cannot answer without trying"
        (ep.root / "episode.json").write_text(json.dumps(outcome, indent=1))
        return outcome
    dry = 0
    try:
        for rnd in range(1, (1 if ep.setting == "exec" else min(MAX_ROUNDS, method.max_rounds)) + 1):
            ep.round = outcome["rounds"] = rnd
            plan = method.plan(ep, feedback)
            if plan is None:
                outcome["ended"] = "the method gave no team plan"
                break
            if ep.setting == "one-shot" or not method.tries:
                outcome["ended"] = "the team plan is the answer (nothing tried)"
                break
            if not plan.get("feasible"):
                outcome["ended"] = "the method says the task cannot be done"
                break
            if not ep.can_try:
                outcome["ended"] = "no runs left"
                break
            res = method.execute(ep, plan, feedback)
            outcome["success"] = res.success
            if res.success:
                outcome["ended"] = "a run succeeded"
                outcome["movers"] = res.movers
                break
            dry = 0 if ep.runs(rnd) else dry + 1
            if dry >= MAX_DRY_ROUNDS:
                outcome["ended"] = f"{dry} rounds in a row without a run: no program passed its check"
                break
            feedback = {"round": rnd, "plan": plan, "report": res.report, "note": res.note}
            if not ep.can_try:
                outcome["ended"] = "no runs left"
                break
        else:
            outcome["ended"] = "carried out once: no run succeeded" if ep.setting == "exec" else "round limit"
    except OutOfBudget as e:
        outcome["ended"] = f"LLM budget spent ({e})"
    if not outcome["success"]:
        outcome["success"] = any(r["success"] for r in ep.runs())  # a method that tries inside its own sessions
    method.conclude(ep, outcome)
    final = ep.latest_plan()
    outcome.update({"final_plan": final, "seconds": round(time.time() - t0, 1), "ledger": str(ep.ledger.path)})
    (ep.root / "episode.json").write_text(json.dumps(outcome, indent=1))
    return outcome


# ---------------------------------------------------------------------- shared pieces of prompts
def feedback_text(feedback: Optional[dict]) -> str:
    """What a method's first step is told after a failed run: the last plan and the failure report."""
    if not feedback:
        return ""
    shown = {k: feedback["plan"].get(k) for k in ("feasible", "arms_used", "idle", "things_used", "steps")}
    return ("\n\n## The last round\n\nYour team plan in the last round:\n" + json.dumps(shown, indent=1)
            + "\n\n" + failure.text(feedback["report"] or {}) + ("\n" + feedback["note"] if feedback.get("note") else "")
            + "\n\nPlan again from the start with this in mind; you may keep or change the team.")


