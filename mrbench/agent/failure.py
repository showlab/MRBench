"""The failure report after a run that did not succeed: the same for every compared method (docs/BASELINES.md).

What it holds: the milestones reached and missed; the jobs' own failure signals that showed (a door shut with the
box inside), for a task that declares them (``tasks/signals.py``: the eight benchmark tasks); what went wrong in the
program -- grips that closed on nothing, arms that fell behind their plans; how far things moved; and, for a program
that never ran, the check's reasons (inverse kinematics, clearance).

The rule it states, for every method alike: a run that fails shows that a smaller team cannot do the task only if
a job's own failure signal was reported; a missed grip, an arm behind its plan or a clash is the program's fault --
one failed program is not every program.
"""

from __future__ import annotations

from typing import Optional

RULE = ("A failed run shows that this team cannot do the task only if a job's own failure signal is listed; a grip "
        "that closed on nothing, an arm behind its plan or a clash is the program's fault, not the team's.")
#: a thing counts as moved, in the report, past this far or this turn
MOVED_M, TURNED_DEG = 0.02, 10.0


def job_signals(case, rep: dict) -> Optional[list]:
    """The jobs' own failure signals that showed in this run (``signals_hit``, scored with the run), if its task
    declares them (None: it declares none, or the run's report does not list them)."""
    from ..tasks import signals

    if not signals.declared(case.inst) or "signals_hit" not in rep:
        return None
    return list(rep["signals_hit"])


def report(case, rep: Optional[dict], check: Optional[dict] = None) -> dict:
    """The failure report for one run's ``report.json`` (``execute.py``), or for a program the check refused."""
    milestones = [m.name for m in case.inst.milestones]
    if not rep or not rep.get("played"):
        out = {"success": False, "played": False, "rule": RULE}
        if rep and rep.get("error"):
            out["not_played_because"] = rep["error"][:1500]
        if check:
            out["check_problems"] = check.get("problems", [])[:12]
        return out
    hit = rep.get("milestones_hit") or []
    signals = job_signals(case, rep)
    moved = {k: v for k, v in (rep.get("worst") or {}).items()
             if v.get("max_moved_m", 0) > MOVED_M or v.get("max_turned_deg", 0) > TURNED_DEG}
    out = {"success": bool(rep.get("success")), "played": True,
           "milestones_reached": [m for m in milestones if m in hit],
           "milestones_missed": [m for m in milestones if m not in hit],
           "first_reached_s": rep.get("first_hit_s"),
           "job_failure_signals": signals if signals is not None else "none known for this task or this run",
           "grips_that_closed_on_nothing": rep.get("missed_grips") or [],
           "arms_behind_plan_at_phase_end": [p for p in rep.get("phases", []) if p.get("off_plan_m")][:8],
           "things_moved_or_turned": moved, "rule": RULE}
    return out


def text(r: dict) -> str:
    """The report as the prompt shows it."""
    import json

    return "The last run did not succeed. Failure report:\n" + json.dumps(r, indent=1)
