"""What a compared method is, and the pieces the rows share: the prompts' common part and the shared executor.

A method decides a team (``plan``) and, in the equal-budget setting, has it carried out (``execute``; by default the
shared executor, which only the declared arms move in).  ``episode.run_episode`` loops the two.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

from ...core.paths import REPO_ROOT
from .. import failure
from ..env import CONTROL, SENSES, isaac_runner
from ..episode import Episode, ExecResult

PROMPTS = Path(__file__).parent.parent / "prompts" / "bench"
#: where ``prompt`` looks for a prompt, in order: the package's, then those a plugin adds (``add_prompts``)
PROMPT_DIRS: list[Path] = [PROMPTS]
#: the first message of a session that decides the team
FIRST = "Plan the team for the task on this table. Start with `briefing`."


def add_prompts(directory: Path) -> None:
    """Prompts of a plugin's rows (``core/plugins.py``), found after the package's own."""
    PROMPT_DIRS.append(Path(directory))


def prompt(name: str, **fill) -> str:
    """A prompt from ``prompts/bench``, with ``{COMMON}`` (the question and the answer every row is given),
    ``{API_DOC}`` and any other ``{KEY}`` filled in."""
    from ..program import API_DOC

    found = [d / f"{name}.md" for d in PROMPT_DIRS if (d / f"{name}.md").is_file()]
    if not found:
        raise FileNotFoundError(f"no prompt {name!r} in {', '.join(map(str, PROMPT_DIRS))}")
    text = found[0].read_text()
    fill = {"COMMON": (PROMPTS / "common.md").read_text().strip(), "API_DOC": API_DOC.strip(), **fill}
    for k, v in fill.items():
        text = text.replace("{" + k + "}", str(v))
    return text


class Method:
    #: the row's name in the tables
    name = ""
    #: whether it can answer without trying (one-shot); a method that cannot is not run in that setting
    one_shot = True
    #: whether its team plan is carried out at all (a rule's may not be)
    tries = True
    #: rounds at most
    max_rounds = 6

    def limits(self, setting: str) -> Optional[dict]:
        return None

    def plan(self, ep: Episode, feedback: Optional[dict]) -> Optional[dict]:
        raise NotImplementedError

    def execute(self, ep: Episode, plan: dict, feedback: Optional[dict]) -> ExecResult:
        return shared_execute(ep, plan, feedback)

    def conclude(self, ep: Episode, outcome: dict) -> None:
        """After the loop: a method may record its final answer (one that tries teams says infeasible when none succeeded)."""


class Recorded(Method):
    """A row with the answers it gave in an earlier run (``bench --plans-from``): the team plan is read back, not
    made again, and carried out the row's own way.  ``plans``: (case, repeat) -> the team plan, None where the row
    gave none.  A failed run does not go back to the row: its sessions are not there to be told."""

    def __init__(self, row: Method, plans: dict):
        self.row, self.plans, self.name = row, plans, row.name
        self.one_shot, self.tries, self.max_rounds = row.one_shot, row.tries, 1

    def limits(self, setting: str) -> Optional[dict]:
        return self.row.limits(setting)

    def plan(self, ep: Episode, feedback: Optional[dict]) -> Optional[dict]:
        plan = self.plans.get((ep.case_id, ep.repeat)) if feedback is None else None
        if plan is not None:
            ep.record_plan(plan, "recorded")
        return plan

    def execute(self, ep: Episode, plan: dict, feedback: Optional[dict]) -> ExecResult:
        return self.row.execute(ep, plan, feedback)


# ---------------------------------------------------------------------- the shared executor
EXECUTOR_TOOLS = [t for t in SENSES if t != "heft"] + list(CONTROL)


def shown_plan(plan: dict) -> dict:
    return {k: plan.get(k) for k in ("feasible", "arms_used", "idle", "things_used", "steps", "key_constraints")}


def shared_execute(ep: Episode, plan: dict, feedback: Optional[dict], name: str = "executor") -> ExecResult:
    """The shared stage 2 for any row's team plan: a fresh session writes the program, checks it and tries it; only the
    declared arms may move.  A program it never played is played once here, if a run is left."""
    first = "The team plan:\n" + json.dumps(shown_plan(plan), indent=1)
    if plan.get("notes"):
        first += "\n\nThe planner's notebook:\n" + "\n".join(f"- {n}" for n in plan["notes"])
    if feedback and sorted(feedback["plan"].get("arms_used", [])) == sorted(plan.get("arms_used", [])):
        first += "\n\nThe same team was tried in the last round. " + failure.text(feedback["report"] or {})
    first += "\n\nWrite the program. Start with `briefing`."
    ep.session(name, prompt("executor"), first, EXECUTOR_TOOLS, team_plan=plan, final=["submit_program"],
               calls=60, checks=12)
    if not ep.runs(ep.round) and ep.can_try:
        play_last_program(ep, name)
    return ep.result_of_round()


def play_last_program(ep: Episode, name: str) -> Optional[dict]:
    """Play this round's submitted program (or the last that passed its check) once, as a run of the case."""
    progs = [p for p in ep.ledger.records("programs") if p["session"].startswith(f"round{ep.round}/") and p["ok"]]
    if not progs:
        return None
    out_f = ep.root / f"round{ep.round}" / name / "output.json"
    final_k = json.loads(out_f.read_text()).get("submit_program", {}).get("check") if out_f.exists() else None
    pick = next((p for p in progs if p["k"] == final_k and p["session"].endswith(name)), progs[-1])
    if ep.ledger.take("runs") is None:
        return None
    plan_f = Path(pick["plan"])
    out = plan_f.parent / f"sim_{pick['k']}"
    runner = getattr(ep.backend, "sim", None) or isaac_runner
    rep = runner(ep.case_id, plan_f, out, ep.gpu, None)
    ep.ledger.add("runs", {"session": f"round{ep.round}/{name}-played", "k": pick["k"],
                           "success": bool(rep.get("success")), "played": bool(rep.get("played")),
                           "report": str(out / "report.json"), "plan": str(plan_f), "movers": pick.get("movers", []),
                           "milestones_hit": rep.get("milestones_hit")})
    return rep


# ---------------------------------------------------------------------- gold answers (the oracles)
#: a team plan proposed in a discussion (RoCo-, HMAS-2-style), as `respond` takes it
PROPOSAL = {"type": "object", "description": "the team plan as you propose it (the fields of submit_team_plan)",
            "properties": {"feasible": {"type": "boolean"}, "arms_used": {"type": "array", "items": {"type": "string"}},
                           "idle": {"type": "array", "items": {"type": "object"}},
                           "things_used": {"type": "array", "items": {"type": "object"}},
                           "things_rejected": {"type": "array", "items": {"type": "object"}},
                           "missing": {"type": "string"}, "steps": {"type": "array", "items": {"type": "object"}},
                           "key_constraints": {"type": "array", "items": {"type": "string"}}},
            "required": ["feasible", "arms_used", "idle", "missing"]}


def record_proposal(ep: Episode, proposal: dict, by: str) -> Optional[dict]:
    """A proposal from a discussion becomes the team plan; one that does not pass the form check is sent back
    once to be fixed through ``submit_team_plan``."""
    from ..env import normalise_plan, validate_plan

    plan = normalise_plan(proposal)
    problems = validate_plan(plan, ep.arms, {t["name"] for t in ep.observer().things()})
    if not problems:
        ep.record_plan(plan, by)
        return plan
    r = ep.session(f"{by}-submit", prompt("fix_plan"),
                   "The plan agreed:\n" + json.dumps(proposal, indent=1) + "\n\nIt was not accepted:\n- "
                   + "\n- ".join(problems) + "\n\nSubmit it, fixed, with submit_team_plan.",
                   ["briefing", "submit_team_plan"], calls=4)
    return r.output.get("submit_team_plan")


def gold_plan(ep: Episode) -> dict:
    """The right answer from the case's ``Expect``."""
    e = ep.case.expect
    if not e.feasible:
        return {"feasible": False, "arms_used": [], "idle": [{"arm": a, "reason": "gold"} for a in ep.arms],
                "missing": e.missing or "gold: infeasible", "things_used": [], "steps": [], "key_constraints": [e.why]}
    team = sorted(e.minimal[0])
    things = [{"thing": b, "job": "gold"} for b in ep.case.blocks] if e.blocks else []
    things += [{"thing": t, "job": "gold"} for t in e.things]  # the eight's borrow versions: the prop
    return {"feasible": True, "arms_used": team, "idle": [{"arm": a, "reason": "gold"} for a in ep.arms
                                                          if a not in team],
            "things_used": things, "missing": "", "steps": [], "key_constraints": [e.why]}


__all__ = ["Method", "prompt", "PROPOSAL", "record_proposal", "shared_execute", "play_last_program", "gold_plan", "EXECUTOR_TOOLS", "REPO_ROOT"]
