"""Graphs for the team decision: the scene graph the agent reads, and the job graph it writes for ``solve_team``.

The two reasons one arm is not enough are two graph problems:

* **at the same time** (concurrency-bound): jobs joined by a "with" edge must be done by different doers -- arms, or
  things standing in for an arm (a weight, a stop).  Choosing doers is colouring that conflict graph with the doers
  each job allows; an arm may do any number of jobs that are not joined;
* **far apart** (reach-bound): a transfer moves a thing from one spot to another through hand-off spots; it can be
  done by a set of arms if the spots are connected when two spots are joined by every arm that reaches both.

``solve_team`` searches the arm sets from the smallest up and returns every smallest one that does every job and
every transfer, with who does what; if none does, what blocks it (a job no doer can do, a group of jobs at the same
time with too few doers, a transfer whose spots stay apart) -- the "excuse" Gobelbecker et al. ask for.  The agent
writes the graph, from what it has established with the tools; the solver only searches.
"""

from __future__ import annotations

from itertools import combinations
from typing import Optional

from .physics import facts


def scene_graph(case, obs, reach_log: list[dict]) -> dict:
    """Things (what, mass, fixed or what it rests on), joints (springs, tight fits), the arms' capacities, and the
    reach edges the agent has established so far with ``check_reach``."""
    f = facts(case, obs)
    places = [{"name": t["name"], "what": t["what"]} for t in obs.things() if t["mass"] == "-"]
    edges = [{"arm": r["arm"], "point": r["point"],
              "reaches": [k for k, v in r["reachable"].items() if v.startswith("yes")]} for r in reach_log]
    return {**f, "places": places, "reach_edges_so_far": edges}


def solve_team(case, obs, jobs: list[dict], transfers: list[dict], cache: Optional[dict] = None) -> dict:
    cache = {} if cache is None else cache
    arms = [a.slot for a in case.inst.scene.arms]
    things = {t["name"] for t in obs.things() if t["mass"] not in ("-", "fixed")}
    ids = [j["id"] for j in jobs]
    if len(set(ids)) != len(ids):
        return {"error": "job ids must be different"}
    # who may do each job: arms the agent names (or, with "auto", every arm whose kinematics reach the job's point),
    # and things standing in for an arm
    allowed: dict[str, set] = {}
    notes = []
    for j in jobs:
        doers = j.get("doers", "auto")
        if doers == "auto" or doers == ["auto"]:
            if "at" not in j:
                return {"error": f"job {j['id']}: 'auto' doers need the point 'at'"}
            allowed[j["id"]] = {a for a in arms if obs.reaches(a, j["at"])}
        else:
            bad = [d for d in doers if d not in arms and d not in things]
            if bad:
                return {"error": f"job {j['id']}: no arm or movable thing called {', '.join(bad)}"}
            allowed[j["id"]] = set(doers)
    conflict = {i: set() for i in ids}
    for j in jobs:
        for k in j.get("with", []):
            if k not in conflict:
                return {"error": f"job {j['id']}: 'with' names no job {k!r}"}
            conflict[j["id"]].add(k)
            conflict[k].add(j["id"])

    def assign(team: set) -> Optional[dict]:
        order = sorted(ids, key=lambda i: (len(allowed[i] & (team | things)), -len(conflict[i])))
        got: dict[str, str] = {}
        used_things: set = set()

        def go(n: int) -> bool:
            if n == len(order):
                return True
            i = order[n]
            for d in sorted(allowed[i] & (team | things)):
                if any(got.get(k) == d for k in conflict[i]):
                    continue
                if d in things and d in used_things:
                    continue  # a thing stands in for one job: it stays where it was put
                got[i] = d
                if d in things:
                    used_things.add(d)
                if go(n + 1):
                    return True
                del got[i]
                used_things.discard(d)
            return False

        return dict(got) if go(0) else None

    def connected(team: set, t: dict) -> Optional[list]:
        spots = [t["from"], *t.get("via", []), t["to"]]
        n = len(spots)
        link = {i: [] for i in range(n)}
        for a in team:
            reach = [i for i in range(n) if obs.reaches(a, spots[i])]
            for i, k in combinations(reach, 2):
                link[i].append((k, a))
                link[k].append((i, a))
        prev = {0: None}
        todo = [0]
        while todo:
            i = todo.pop(0)
            for k, a in link[i]:
                if k not in prev:
                    prev[k] = (i, a)
                    todo.append(k)
        if n - 1 not in prev:
            return None
        legs, k = [], n - 1
        while prev[k] is not None:
            i, a = prev[k]
            legs.append({"arm": a, "from": spots[i], "to": spots[k]})
            k = i
        return legs[::-1]

    found = []
    for size in range(0, len(arms) + 1):
        for team in combinations(arms, size):
            s = set(team)
            got = assign(s)
            if got is None:
                continue
            relays = {}
            for t in transfers:
                legs = connected(s, t)
                if legs is None:
                    break
                relays[t["id"]] = legs
            else:
                used = {d for d in got.values() if d in arms} | {leg["arm"] for v in relays.values() for leg in v}
                if used != s:
                    continue  # a smaller set already does it
                found.append({"arms": sorted(s), "idle": sorted(set(arms) - s), "jobs": got,
                              "things_used": sorted({d for d in got.values() if d in things}), "relays": relays})
        if found:
            break
    if found:
        return {"feasible": True, "fewest_arms": len(found[0]["arms"]), "smallest_teams": found[:6], "notes": notes}
    apart = [t for t in transfers if connected(set(arms), t) is None]
    return {"feasible": False, "why": _excuse(arms, things, ids, allowed, conflict, apart, obs, cache)}


def _excuse(arms, things, ids, allowed, conflict, apart, obs, cache) -> list[str]:
    """What blocks every team: the smallest reasons, with the doers each needs."""
    out = []
    for i in ids:
        if not allowed[i] & (set(arms) | things):
            out.append(f"job {i}: no arm or thing on the table can do it")
    # groups of jobs all at the same time (cliques) with fewer doers than jobs
    for size in range(len(ids), 1, -1):
        for group in combinations(ids, size):
            if all(k in conflict[i] for i, k in combinations(group, 2)):
                doers = set().union(*(allowed[i] for i in group)) & (set(arms) | things)
                if len(doers) < size:
                    out.append(f"jobs {', '.join(group)} happen at the same time but only {len(doers)} doer(s) can do "
                               f"them ({', '.join(sorted(doers)) or 'none'}): one more arm or thing that can do one "
                               f"of them is missing")
    for t in apart:
        spots = [t["from"], *t.get("via", []), t["to"]]
        reach = {a: [s for s in spots if obs.reaches(a, s)] for a in arms}
        out.append(f"transfer {t['id']}: even with every arm the spots do not join up; who reaches which spot: "
                   + "; ".join(f"{a}: {len(r)} of {len(spots)}" for a, r in reach.items()))
    return out or ["no single reason: try fewer constraints"]
