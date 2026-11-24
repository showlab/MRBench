"""Grades an episode (docs/PROTOCOL.md, "Scoring"), the same for every row.

* **The team is what was declared plus what was used**: the arms in the final team plan, plus the arms that acted in
  a run that succeeded; an arm counts as used if it moved in that run's program.
* **MSS right**: feasible and infeasible told apart (a team smaller than gold that succeeded in a run counts as
  right, flagged ``beats_gold``); for a feasible case, the team is one of the fewest-arm sets and
  the things are right (the blocks used where the fewest need them, and not where they do not; in the eight's
  borrow versions, the prop used and the distractor not); for an infeasible one, said so, with what is missing
  (whether it names the right job is recorded for a judge, not scored here).
* **SR**, **MSS-SR** (the team right and a run succeeded), and the cost: sessions, dollars, runs, probes, seconds.
* The error classes (docs/PROTOCOL.md, "Metrics"): over- or under-staffed, feasibility wrong either way, distractor used, prop missed.

The right answers come from the case's ``Expect`` (``cases.py``); for the eight, the solver's over the tasks' declared
jobs (``tasks/jobs.py``, docs/PROTOCOL.md).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

from .cases import case_from_id


def _used_arms(ledger: dict) -> set:
    used = set()
    for r in ledger.get("runs", []):
        if r.get("success"):
            used |= set(r.get("movers") or [])
    return used


def grade(outcome: dict, ledger: dict) -> dict:
    case = case_from_id(outcome["case"])
    e, variant = case.expect, case.variant
    plan: Optional[dict] = outcome.get("final_plan")
    u = ledger["used"]
    g = {"method": outcome["method"], "case": outcome["case"], "variant": variant, "setting": outcome["setting"],
         "repeat": outcome["repeat"], "ended": outcome.get("ended"), "rounds": outcome.get("rounds"),
         "success": bool(outcome.get("success")), "sessions": u["sessions"], "cost_usd": round(u["cost_usd"], 3),
         "runs": u["runs"], "probes": u["probes"], "seconds": outcome.get("seconds"),
         "expected": {"feasible": e.feasible, "minimal": [sorted(s) for s in e.minimal], "blocks": e.blocks,
                      "missing": e.missing}}
    if outcome.get("ended", "").startswith("not run"):
        g["answered"] = None
        return g
    if not plan:
        g.update({"answered": False, "mss_right": False, "error": "no answer"})
        return g
    declared = set(plan.get("arms_used") or [])
    team = declared | (_used_arms(ledger) if plan.get("feasible") else set())
    things = {t.get("thing", "").partition(":")[0] for t in plan.get("things_used") or []}
    blocks_used = any(t.startswith("block") for t in things)
    g.update({"answered": True, "said_feasible": bool(plan.get("feasible")), "declared": sorted(declared),
              "team": sorted(team), "things_used": sorted(things), "missing_said": plan.get("missing", "")})
    g["feasible_right"] = bool(plan.get("feasible")) == e.feasible
    things_right = True if e.blocks is None else (blocks_used == e.blocks)
    if e.things or e.rejected:  # the eight's borrow versions: the prop used, the distractor not
        things_right = set(e.things) <= things and not set(e.rejected) & things
    if e.feasible and plan.get("feasible"):
        fewest = min(len(s) for s in e.minimal)
        g["team_right"] = frozenset(team) in e.minimal
        g["things_right"] = things_right
        g["mss_right"] = g["team_right"] and things_right
        g["workable"] = g["mss_right"] or frozenset(team) in e.workable
        g["extra_arms"] = len(team) - fewest
        if g["success"] and len(team) < fewest:
            # docs/PROTOCOL.md, "Scoring": a team smaller than gold that succeeds in a run counts as right
            g.update({"beats_gold": True, "team_right": True, "mss_right": things_right, "workable": True})
    else:
        g["team_right"] = g["mss_right"] = g["feasible_right"] and not e.feasible
        g["things_right"] = not blocks_used if not plan.get("feasible") else things_right
    g["mss_sr"] = bool(g["mss_right"] and g["success"])
    g["error"] = error_class(g, e, blocks_used)
    return g


def error_class(g: dict, e, blocks_used: bool) -> str:
    if g["mss_right"]:
        return ""
    if not g["feasible_right"]:
        return "said infeasible, is feasible" if e.feasible else "said feasible, is infeasible"
    said = set(g.get("things_used") or ())
    if set(e.rejected) & said:
        return "distractor used"
    if e.things and not set(e.things) <= said:
        return "prop missed"
    if e.blocks and not blocks_used:
        return "prop missed"
    if e.blocks is False and blocks_used:
        return "distractor used" if e.feasible else "thing used"
    if g.get("extra_arms", 0) > 0:
        return "over-staffed"
    if g.get("extra_arms", 0) < 0:
        return "under-staffed"
    return "another team of the fewest arms" if g.get("extra_arms") == 0 else "wrong"


def grade_dir(ep_dir: Path) -> dict:
    outcome = json.loads((ep_dir / "episode.json").read_text())
    ledger = json.loads((ep_dir / "ledger.json").read_text())
    g = grade(outcome, ledger)
    (ep_dir / "grade.json").write_text(json.dumps(g, indent=1))
    return g


def table(rows: list[dict]) -> str:
    """Per method x setting: team-selection accuracy, SR, MSS-SR and the mean cost; and per method the errors.  Then the accuracy per
    configuration, the task level (``task_level``) and each compared row (``methods.compared_rows``) against each other
    row (``paired``)."""
    from collections import defaultdict

    from .methods import compared_rows

    groups = defaultdict(list)
    for r in rows:
        if r.get("answered") is None:
            continue
        groups[(r["method"], r["setting"])].append(r)
    lines = ["| method | setting | episodes | team acc. | SR | MSS-SR | sessions | $ | runs | errors |",
             "|---|---|---|---|---|---|---|---|---|---|"]
    for (m, s), rs in sorted(groups.items()):
        n = len(rs)
        errs = defaultdict(int)
        for r in rs:
            if r.get("error"):
                errs[r["error"]] += 1
        mean = lambda k: sum(r.get(k) or 0 for r in rs) / n  # noqa: E731
        lines.append(f"| {m} | {s} | {n} | {sum(bool(r.get('mss_right')) for r in rs)}/{n} | "
                     f"{sum(r['success'] for r in rs)}/{n} | {sum(bool(r.get('mss_sr')) for r in rs)}/{n} | "
                     f"{mean('sessions'):.1f} | {mean('cost_usd'):.2f} | {mean('runs'):.1f} | "
                     f"{', '.join(f'{k} {v}' for k, v in sorted(errs.items())) or '-'} |")
    return "\n\n".join(t for t in ("\n".join(lines), by_version(rows), task_level(rows),
                                   *(paired(rows, b) for b in compared_rows())) if t)


def by_version(rows: list[dict]) -> str:
    """Team-selection accuracy per method x setting and configuration."""
    from collections import defaultdict

    from .cases import GOLD_VARIANTS

    right, n = defaultdict(int), defaultdict(int)
    for r in rows:
        if r.get("answered") is None:
            continue
        k = (r["method"], r["setting"], r["variant"])
        n[k] += 1
        right[k] += bool(r.get("mss_right"))
    seen = {v for (_, _, v) in n}
    variants = [v for v in GOLD_VARIANTS if v in seen] + sorted(seen - set(GOLD_VARIANTS))
    lines = ["| method | setting | " + " | ".join(variants) + " |", "|---|---|" + "---|" * len(variants)]
    for m, s in sorted({(m, s) for (m, s, _) in n}):
        lines.append(f"| {m} | {s} | " + " | ".join(f"{right[m, s, v]}/{n[m, s, v]}" if n[m, s, v] else "-"
                                                    for v in variants) + " |")
    return "\n".join(lines)


def _task_scenes(rows: list[dict]) -> tuple[dict, dict]:
    """Per method x setting: each task scene -- its task, level, seed and look -- with whether all its versions with
    gold were right, one entry per repeat; and how many (scene, repeat)s were left out for a version missing."""
    from collections import defaultdict

    from .cases import parse_case
    from .gold import variants

    ok = defaultdict(dict)  # (method, setting, task scene, repeat) -> {version: right}
    for r in rows:
        if r.get("answered") is None:
            continue
        task, variant, level, seed, look = parse_case(r["case"])
        ok[r["method"], r["setting"], (task, level, seed, look), r["repeat"]][variant] = bool(r.get("mss_right"))
    wanted = {}
    scenes = defaultdict(lambda: defaultdict(list))  # (method, setting) -> task scene -> [right per repeat]
    missing = defaultdict(int)
    for (m, s, scene, _), got in ok.items():
        task = scene[0]
        if task not in wanted:
            try:
                wanted[task] = set(variants(task))
            except Exception:  # noqa: BLE001 -- a task without gold: the versions run are the ones asked
                wanted[task] = set()
        need = wanted[task] or set(got)
        if not need <= set(got):
            missing[m, s] += 1
            continue
        scenes[m, s][scene].append(all(got[v] for v in need))
    return scenes, missing


def _interval(xs: list[float], rng, draws: int) -> tuple[float, float]:
    boot = sorted(sum(rng.choices(xs, k=len(xs))) / len(xs) for _ in range(draws))
    return boot[int(0.025 * draws)], boot[int(0.975 * draws) - 1]


def task_level(rows: list[dict], draws: int = 2000) -> str:
    """The headline (docs/PROTOCOL.md, 5): a task in a scene -- its level, seed and look -- is right in a repeat only
    if every version of it with gold is right there.  Per method x setting, the share of task scenes right (each
    scene's mean over its repeats), with a 95% interval by bootstrap over the scenes; a scene missing a version in a
    repeat (not run, or crashed) is left out and counted."""
    import random

    scenes, missing = _task_scenes(rows)
    rng = random.Random(0)
    lines = ["| method | setting | task scenes | task level | 95% interval | left out |", "|---|---|---|---|---|---|"]
    for m, s in sorted(set(scenes) | set(missing)):
        means = [sum(v) / len(v) for v in scenes[m, s].values()]
        if not means:
            lines.append(f"| {m} | {s} | 0 | - | - | {missing[m, s]} |")
            continue
        lo, hi = _interval(means, rng, draws)
        lines.append(f"| {m} | {s} | {len(means)} | {sum(means) / len(means):.3f} | {lo:.3f}-{hi:.3f} | "
                     f"{missing[m, s]} |")
    return "\n".join(lines)


def paired(rows: list[dict], base: str, draws: int = 2000) -> str:
    """Rows compared paired by scene (docs/PROTOCOL.md, 5): per setting, ``base`` against each other method on the
    task scenes both have whole -- the mean difference of their task levels (each scene's mean over its repeats), a
    95% interval by bootstrap over those scenes, and in how many scenes ``base`` was higher, lower or the same.
    Empty when ``base`` has no task scenes."""
    import random

    scenes, _ = _task_scenes(rows)
    rng = random.Random(0)
    mean = lambda v: sum(v) / len(v)  # noqa: E731
    lines = [f"| setting | method | task scenes in common | {base} minus it | 95% interval | {base} higher / lower / "
             "same |", "|---|---|---|---|---|---|"]
    for m, s in sorted(k for k in scenes if k[0] != base and (base, k[1]) in scenes):
        mine, theirs = scenes[base, s], scenes[m, s]
        common = sorted(set(mine) & set(theirs))
        if not common:
            lines.append(f"| {s} | {m} | 0 | - | - | - |")
            continue
        d = [mean(mine[c]) - mean(theirs[c]) for c in common]
        lo, hi = _interval(d, rng, draws)
        lines.append(f"| {s} | {m} | {len(d)} | {mean(d):+.3f} | {lo:+.3f} to {hi:+.3f} | "
                     f"{sum(x > 0 for x in d)} / {sum(x < 0 for x in d)} / {sum(x == 0 for x in d)} |")
    return "\n".join(lines) if len(lines) > 2 else ""
