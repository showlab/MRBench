"""A stand-in for the model and the simulator, to run every row end to end without either: each session looks at
the scene, answers with the gold team (``methods.base.gold_plan``) in whatever form the session asks for, and writes a
program that keeps the idle arms still; the simulator says a run succeeds when its program moves exactly the gold
arms.  For tests and for ``bench --backend fake`` (a dry run of the plumbing, not a result)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

from .env import Env
from .llm import FakeBackend, SessionSpec


def _gold(env: Env) -> dict:
    e = env.case.expect
    arms = env.arms
    if not e.feasible:
        return {"feasible": False, "arms_used": [], "idle": [{"arm": a, "reason": "fake"} for a in arms],
                "missing": e.missing or "fake", "things_used": []}
    team = sorted(e.minimal[0])
    return {"feasible": True, "arms_used": team, "idle": [{"arm": a, "reason": "fake"} for a in arms if a not in team],
            "missing": "", "things_used": [{"thing": b, "job": "fake"} for b in env.case.blocks] if e.blocks else [],
            "without_each_arm": [{"arm": a, "tried": "fake", "failed_because": "fake", "evidence": "fake"}
                                 for a in team]}


def _program(env: Env, team: list[str]) -> str:
    """A program in which each arm of ``team`` opens its fingers a little (it acts; nothing needs inverse kinematics)."""
    acts = ", ".join(f"open_fingers('{a}', 0.03)" for a in team)
    return f"PLAN = [phase(1.0, 'open', {acts})]" if team else "PLAN = []"


def player(spec: SessionSpec, env: Optional[Env]) -> str:
    if env is None:
        return "{}"
    tools = env.cfg.tools
    if "briefing" in tools:
        env.logged_call("briefing", {})
    gold = _gold(env)
    if "check_program" in tools:
        team = (env.team_plan() or gold).get("arms_used", [])
        code = _program(env, team)
        env.logged_call("check_program", {"code": code})
        if "try_in_sim" in tools:
            env.logged_call("try_in_sim", {"code": code})
        env.logged_call("submit_program", {"code": code})
        if "submit_team_plan" in tools and not env.ledger.records("plans"):
            env.logged_call("submit_team_plan", gold)
        return "DONE"
    if "submit_team_plan" in tools:
        env.logged_call("submit_team_plan", gold)
        return "DONE"
    if "respond" in tools:
        props = (env.cfg.respond_schema or {}).get("properties", {})
        if "subtasks" in props:
            data = {"subtasks": [{"id": "s1", "what": "the task", "skills": ["reach"]}]}
        elif "coalitions" in props:
            data = {"coalitions": [{"subtask": "s1", "arms": gold["arms_used"], "things": []}]}
        elif "accept" in props:
            data = {"accept": True, "reason": "fake"}
        elif "taking_part" in props:
            data = {"taking_part": env.cfg.view in gold["arms_used"], "proposal": gold, "agree": True}
        else:
            data = gold
        env.logged_call("respond", {"message": "fake", "data": data})
    return "DONE"


def sim(case_id: str, plan_json: Path, out: Path, gpu: str, probe_s) -> dict:
    """Succeeds when the program moves exactly the gold arms (the oracle's ``@script`` always does)."""
    from .cases import case_from_id
    from .program import acting_arms, to_script

    case = case_from_id(case_id)
    gold = set(case.expect.minimal[0]) if case.expect.minimal else set()
    if str(plan_json) == "@script":
        movers = set(gold)
    elif str(plan_json).startswith("@skill:"):  # the skill-level executor: the skill's own team
        from .skill_exec import library

        movers = set(next(s.team for s in library(case) if s.name == str(plan_json)[len("@skill:"):]))
    else:
        movers = acting_arms(to_script(json.loads(plan_json.read_text())))
    ok = bool(gold) and movers == gold and probe_s is None
    names = [m.name for m in case.inst.milestones]
    rep = {"case": case_id, "played": True, "success": ok, "milestones_hit": names if ok else [],
           "first_hit_s": {}, "missed_grips": [], "moved_m": {}, "worst": {}, "phases": [], "keyframes": []}
    out.mkdir(parents=True, exist_ok=True)
    (out / "report.json").write_text(json.dumps(rep, indent=1))
    return rep


def backend() -> FakeBackend:
    return FakeBackend(player, sim=sim, label="fake")
