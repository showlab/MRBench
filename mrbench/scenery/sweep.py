"""Where on the table a task's scripted reference works: the cells under its arms and the bodies they hold.

    python -m mrbench.scenery.sweep [TASK ...]       # plan each reference and write mrbench/scenery/sweeps/

For each benchmark task, each version it has built and each level (seed 0), the reference is planned and walked
every ``STEP``: a cell of the table (``CELL`` square) is swept when any part of an arm (``robots.collision``'s
capsules, the fingers wide open), or of a rigid body a hand holds, comes within ``LOW`` of the table over it.  The
seed moves things a centimetre or two; ``distractors`` keeps a margin round the cells.  Each file names the sha of
the impl it was planned from: ``test_scenery`` says when an impl has changed since, and its sweeps must be made
again.  Needs numpy; the planning takes a minute or so a version.
"""

from __future__ import annotations

import hashlib
import json
import math
import sys
from pathlib import Path

#: a cell's side, how low over the table a part sweeps it, and how often the plan is sampled (s)
CELL, LOW, STEP = 0.02, 0.30, 0.1
DIR = Path(__file__).parent / "sweeps"
BENCHMARK = ("Con-01", "Con-02", "Con-03", "Con-04", "Rea-01", "Rea-02", "Rea-03", "Rea-04")


def impl_sha(task: str) -> str:
    """The sha of ``task``'s impl file, which its sweeps were planned from."""
    import importlib

    from ..tasks import get

    return hashlib.sha1(Path(importlib.import_module(get(task).impl).__file__).read_bytes()).hexdigest()[:12]


def cell_of(x: float, y: float) -> tuple[int, int]:
    return math.floor(x / CELL), math.floor(y / CELL)


def _segment_cells(a, b, r, out: set) -> None:
    """The cells within ``r`` of the table-plane segment ``a``-``b``."""
    (ax, ay), (bx, by) = a, b
    lo_x, hi_x = math.floor((min(ax, bx) - r) / CELL), math.floor((max(ax, bx) + r) / CELL)
    lo_y, hi_y = math.floor((min(ay, by) - r) / CELL), math.floor((max(ay, by) + r) / CELL)
    dx, dy = bx - ax, by - ay
    n2 = dx * dx + dy * dy
    for i in range(lo_x, hi_x + 1):
        for j in range(lo_y, hi_y + 1):
            px, py = (i + 0.5) * CELL, (j + 0.5) * CELL
            s = 0.0 if n2 == 0 else max(0.0, min(1.0, ((px - ax) * dx + (py - ay) * dy) / n2))
            if math.hypot(px - ax - s * dx, py - ay - s * dy) <= r + CELL / 2:
                out.add((i, j))


def sweep(inst) -> set[tuple[int, int]]:
    """The cells ``inst``'s scripted reference sweeps."""
    import importlib

    from ..policies.plan import plan as make_plan
    from ..policies.script import Clip, Grip, Release, Unclip
    from ..policies.clearance import FIT_TOL, GRIP_REACH
    from ..robots import collision as col
    from ..robots import kinematics as kin

    np = kin.np
    sc = inst.scene
    plan = make_plan(inst, importlib.import_module(inst.spec.impl).script(inst))
    arms = {a.slot: a for a in sc.arms}
    bases = {s: kin.transform(a.pos, kin.rpy_matrix(0.0, 0.0, math.radians(a.yaw_deg))) for s, a in arms.items()}
    cloths = {c.name for c in sc.cloths}
    poses = {o.name: o.pose for o in sc.dynamic_objects}
    held: dict[str, tuple[str, "np.ndarray"]] = {}
    clips: dict[str, tuple[str, "np.ndarray"]] = {}
    events = sorted(plan.events, key=lambda e: e[0])
    low = sc.table_top_z + LOW
    out: set = set()
    ei = 0
    for k in range(int(plan.duration / STEP) + 1):
        t = k * STEP
        q = {s: plan.arms[s].at(t) for s in arms}
        tool = {s: bases[s] @ kin.tcp(arms[s].model, q[s]) for s in arms}
        while ei < len(events) and events[ei][0] <= t + 1e-9:
            et, act = events[ei]
            if isinstance(act, Grip) and act.attach and act.target not in cloths:
                held[act.slot] = (act.target, np.linalg.inv(tool[act.slot]) @ kin.pose_matrix(poses[act.target]))
            elif isinstance(act, Release) and act.slot in held:
                body, rel = held.pop(act.slot)
                poses[body] = kin.matrix_pose(tool[act.slot] @ rel)
            elif isinstance(act, (Clip, Unclip)):
                owner = next((s for s, (name, _) in held.items() if name == act.tool), None)
                if owner is not None:
                    for slot, (body, rel) in held.items():
                        poses[body] = kin.matrix_pose(bases[slot] @ kin.tcp(
                            arms[slot].model, plan.arms[slot].at(et)) @ rel)
                    tm = kin.pose_matrix(poses[act.tool])
                    if isinstance(act, Clip):
                        ok, _ = col.clip_takes(sc.obj(act.tool), poses[act.tool], sc.obj(act.target),
                                               poses[act.target], GRIP_REACH, FIT_TOL)
                        if ok:
                            clips[act.tool] = (act.target, np.linalg.inv(tm) @ kin.pose_matrix(poses[act.target]))
                    elif act.tool in clips:
                        body, rel = clips.pop(act.tool)
                        poses[body] = kin.matrix_pose(tm @ rel)
            ei += 1
        for s, a in arms.items():
            for c in col.arm_capsules(a.model, q[s], 1.0, bases[s]):
                if min(c.a[2], c.b[2]) - c.r < low:
                    _segment_cells(c.a[:2], c.b[:2], c.r, out)
        for s, (body, rel) in held.items():
            poses[body] = kin.matrix_pose(tool[s] @ rel)
        for carrier, (body, rel) in clips.items():
            poses[body] = kin.matrix_pose(kin.pose_matrix(poses[carrier]) @ rel)
        carried = {body for body, _ in held.values()}
        carried.update(body for carrier, (body, _) in clips.items() if carrier in carried)
        for body in carried:
            for b in col.boxes_of(sc.obj(body), poses[body]):
                corners = [b.center + b.rot @ (b.half * np.array(sgn)) for sgn in
                           ((x, y, z) for x in (-1, 1) for y in (-1, 1) for z in (-1, 1))]
                if min(p[2] for p in corners) < low:
                    xs, ys = [p[0] for p in corners], [p[1] for p in corners]
                    for i in range(math.floor(min(xs) / CELL), math.floor(max(xs) / CELL) + 1):
                        for j in range(math.floor(min(ys) / CELL), math.floor(max(ys) / CELL) + 1):
                            out.add((i, j))
    return out


def path(task: str, variant: str, level: str) -> Path:
    return DIR / f"{task}-{variant}-{level}.json"


def write(task: str) -> list[Path]:
    from ..tasks import build, get

    DIR.mkdir(exist_ok=True)
    written = []
    for variant in get(task).variants:
        for level in ("easy", "hard"):
            cells = sweep(build(task, level, 0, variant=variant))
            p = path(task, variant, level)
            p.write_text(json.dumps({"task": task, "variant": variant, "level": level, "impl_sha": impl_sha(task),
                                     "cell_m": CELL, "low_m": LOW, "rows": runs(cells)}, separators=(",", ":")))
            written.append(p)
            print(f"{p.name}: {len(cells)} cells", flush=True)
    return written


def runs(cells: set) -> dict[str, list[list[int]]]:
    """The cells as runs along x, row by row: {"j": [[i_first, i_last], ...]} (a sweep's file is a few KB)."""
    rows: dict[str, list[list[int]]] = {}
    for j in sorted({c[1] for c in cells}):
        xs = sorted(i for i, jj in cells if jj == j)
        out = [[xs[0], xs[0]]]
        for i in xs[1:]:
            if i == out[-1][1] + 1:
                out[-1][1] = i
            else:
                out.append([i, i])
        rows[str(j)] = out
    return rows


def cells(task: str) -> set[tuple[int, int]]:
    """Every cell any built version of ``task`` sweeps at either level (empty if none were made)."""
    out: set = set()
    for p in DIR.glob(f"{task}-*.json"):
        for j, spans in json.loads(p.read_text())["rows"].items():
            out |= {(i, int(j)) for a, b in spans for i in range(a, b + 1)}
    return out


if __name__ == "__main__":
    for t in sys.argv[1:] or BENCHMARK:
        write(t)
