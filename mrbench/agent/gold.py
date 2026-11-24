"""The gold of the benchmark's eight, from the solver over their declared jobs (``tasks/jobs.py``): for every version
(``cases.GOLD_VARIANTS`` a task has), the answer in each scene of both levels and seeds 0-5 -- which must agree -- kept
in ``gold.json``, from which ``cases.SETUPS`` takes each case's ``Expect``.

    python -m mrbench.tasks.jobs gold            # solve (with the IK results kept) and write gold.json
    python -m mrbench.tasks.jobs gold --check    # only compare, offline

A borrow design's things are kept by their part in it ("prop", "decoy"): which body is which is drawn from the seed.
What is missing is kept by level (a level may have more jobs: Rea-01's three pieces of fruit at hard).
"""

from __future__ import annotations

import json
import sys

from ..tasks.jobs import Answer, Facts, solve
from .cases import GOLD, has_variant, make_case

LEVELS, SEEDS = ("easy", "hard"), (0, 1, 2, 3, 4, 5)


def benchmark() -> tuple[str, ...]:
    from ..scenery.sweep import BENCHMARK

    return BENCHMARK


def variants(task: str) -> tuple[str, ...]:
    from .cases import GOLD_VARIANTS

    return tuple(v for v in GOLD_VARIANTS if has_variant(task, v))


def as_roles(a: Answer, inst) -> dict:
    """The answer as kept: teams as sorted lists, things by their part in the borrow design."""
    b = inst.metadata.get("borrow") or {}
    role = {**{n: "prop" for n in b.get("props", ())}, **{n: "decoy" for n in b.get("decoys", ())}}
    return {"feasible": a.feasible, "minimal": sorted(sorted(t) for t in a.minimal),
            "workable": sorted(sorted(t) for t in a.workable),
            "things": [role.get(n, n) for n in a.things_used], "rejected": [role.get(n, n) for n in a.things_rejected],
            "missing": list(a.missing)}


def solve_case(task: str, variant: str, level: str, seed: int, facts: Facts) -> dict:
    c = make_case(task, variant, level, seed, check_gold=False)
    return as_roles(solve(c.inst, facts), c.inst)


def compute(tasks=None, facts: Facts = None) -> tuple[dict, list[str]]:
    """(task -> version -> its gold, every scene whose answer differs from the version's first scene).  What is missing
    is, at each level, the jobs every scene of it names (a scene may name more: a job one arm reaches only within its
    margin there -- ``scenes``)."""
    gold, differ = {}, []
    for t, v, per in scenes(tasks, facts):
        first = per[0][2]
        differ += [f"{t}-{v}@{lv}.s{s}: {a}; easy.s0: {first}" for lv, s, a, _ in per if a != first]
        missing = {lv: [j for j in next(m for l2, _, _, m in per if l2 == lv)
                        if all(j in m for l2, _, _, m in per if l2 == lv)] for lv in LEVELS}
        gold.setdefault(t, {})[v] = {**first, "missing": missing}
    return gold, differ


def scenes(tasks=None, facts: Facts = None):
    """(task, version, [(level, seed, its answer less what is missing, what is missing), ...]) of every version."""
    facts = facts or Facts(offline=True)
    for t in tasks or benchmark():
        for v in variants(t):
            per = []
            for lv in LEVELS:
                for s in SEEDS:
                    a = solve_case(t, v, lv, s, facts)
                    per.append((lv, s, a, a.pop("missing")))
            yield t, v, per


def main(check: bool = False) -> int:
    facts = Facts(offline=check)
    gold, differ = compute(facts=facts)
    facts.save()
    for d in differ:
        print(f"- the scenes differ: {d}")
    if check:
        kept = json.loads(GOLD.read_text()) if GOLD.exists() else {}
        diffs = [f"{t}-{v}: kept {kept.get(t, {}).get(v)}, solved {g}" for t, vs in gold.items() for v, g in vs.items()
                 if kept.get(t, {}).get(v) != g]
        for d in diffs:
            print(f"- {d}")
        print("gold.json is the solver's" if not (diffs or differ) else f"{len(diffs)} answer(s) differ")
        return 1 if diffs or differ else 0
    GOLD.write_text(json.dumps(gold, indent=1, sort_keys=True) + "\n")
    print(f"wrote {GOLD} ({sum(len(v) for v in gold.values())} answers)")
    return 1 if differ else 0


if __name__ == "__main__":
    sys.exit(main("--check" in sys.argv))
