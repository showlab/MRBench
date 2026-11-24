"""The reference rows of docs/BASELINES.md that need no LLM to decide the team: Oracle-script, Oracle-team and
Oracle-team with the skill-level executor.
"""

from __future__ import annotations

from typing import Optional

from ..env import isaac_runner
from ..episode import Episode, ExecResult
from .base import Method, gold_plan


class OracleScript(Method):
    """The gold team and the task's hand-written reference, replayed: the SR ceiling.  The reference exists for the
    task's own scene (``exact``); ``oracle-team-skill`` covers the other versions."""

    name = "oracle-script"

    def plan(self, ep: Episode, feedback: Optional[dict]) -> Optional[dict]:
        if feedback is not None:  # the reference is what it is: one run tells
            return None
        plan = gold_plan(ep)
        ep.record_plan(plan, "gold")
        return plan

    def execute(self, ep: Episode, plan: dict, feedback: Optional[dict]) -> ExecResult:
        if ep.case.variant != "exact":
            return ExecResult(False, None, [], f"no scripted reference for the {ep.case.variant} version yet")
        if ep.ledger.take("runs") is None:
            return ExecResult(False, None, [], "no runs left")
        out = ep.root / f"round{ep.round}" / "script" / "sim_script"
        out.parent.mkdir(parents=True, exist_ok=True)
        runner = getattr(ep.backend, "sim", None) or isaac_runner
        from pathlib import Path

        rep = runner(ep.case_id, Path("@script"), out, ep.gpu, None)
        ep.ledger.add("runs", {"session": f"round{ep.round}/script", "k": 0, "success": bool(rep.get("success")),
                               "played": bool(rep.get("played")), "report": str(out / "report.json"),
                               "plan": "@script", "movers": list(plan.get("arms_used", [])),
                               "milestones_hit": rep.get("milestones_hit")})
        return ep.result_of_round()


class OracleTeamSkill(OracleScript):
    """The gold team, carried out by the skill-level executor (``skill_exec.py``, docs/BASELINES.md): the
    version's own reference or a gold route whose team is the gold one -- the SR ceiling on every version, not only
    ``exact``.  No LLM."""

    name = "oracle-team-skill"

    def execute(self, ep: Episode, plan: dict, feedback: Optional[dict]) -> ExecResult:
        from pathlib import Path

        from ..skill_exec import choose

        skill, why = choose(ep.case, plan)
        if skill is None:
            return ExecResult(False, None, [], why)
        if ep.ledger.take("runs") is None:
            return ExecResult(False, None, [], "no runs left")
        out = ep.root / f"round{ep.round}" / "skill" / f"sim_{skill.name}"
        out.parent.mkdir(parents=True, exist_ok=True)
        runner = getattr(ep.backend, "sim", None) or isaac_runner
        rep = runner(ep.case_id, Path(f"@skill:{skill.name}"), out, ep.gpu, None)
        ep.ledger.add("runs", {"session": f"round{ep.round}/skill", "k": 0, "success": bool(rep.get("success")),
                               "played": bool(rep.get("played")), "report": str(out / "report.json"),
                               "plan": f"@skill:{skill.name}", "movers": sorted(skill.team),
                               "milestones_hit": rep.get("milestones_hit")})
        return ep.result_of_round()


class OracleTeam(Method):
    """The gold team, given; the shared executor writes and tries the program: the SR lost in execution alone."""

    name = "oracle-team"

    def plan(self, ep: Episode, feedback: Optional[dict]) -> Optional[dict]:
        plan = gold_plan(ep)
        ep.record_plan(plan, "gold")
        return plan
