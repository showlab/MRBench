"""Stage 2: the agent's program -- phases of arm actions -- run in a sandbox, then checked before anything moves.

The agent writes Python against a small arm API (``API_DOC``, shown to it as is) that builds ``PLAN``, a list of
phases.  The code runs in a separate, isolated interpreter (``python -I``) with only that API and ``math``, under a
time limit; what comes back is the plan as JSON.  It is then checked as PRoC3S checks a program before a real run:
planned with inverse kinematics along every path (``policies/plan.py``) and walked for clearance between the arms and
with the scene (``policies/clearance.py``).  Failures come back step by step, naming the phase, the arm and what
went wrong, the clashes grouped and counted.  Physics is not checked here: ``execute.py`` plays a program in Isaac.
A session's check bounds the planner's search (``check(..., timeout_s=)``, ``env.CHECK_TIMEOUT_S``): past it the
check fails, saying the search ran out of time, instead of leaving the session without an answer.
"""

from __future__ import annotations

import json
import subprocess
import sys
from collections import Counter

from ..policies.script import Clip, Grip, Move, Open, Phase, Release, Unclip

TIME_LIMIT_S = 10
#: clashes reported, most frequent first
MAX_CLASHES = 6

API_DOC = '''
All positions are in the world frame (metres; x east, y north, z up). A *pose* is the tool point -- the point between
the fingertips -- with the hand's direction: [x, y, z, qw, qx, qy, qz].

tool_pose(arm, x, y, z, point="down", across=None) -> pose
    The pose that puts `arm`'s tool point at (x, y, z), its hand pointing `point`:
      "down" (straight down), "steep" (60 deg below level), "tilted" (40 deg), "level" (horizontal) -- the last three
      pointing away from the arm's base, toward (x, y) -- or a direction (dx, dy, dz).
    across: the direction (dx, dy, dz) the fingers open and close along, made square to the pointing; by default
      level and square to the line from the base. A bar the fingers close on must lie square to it.
    Example: grasp = tool_pose("west", -0.30, -0.20, 0.80, "down", across=(0, 1, 0))  # fingers open north-south

above(pose, dz) -> pose        the same pose moved up by dz (down if negative)
shift(pose, dx=0, dy=0, dz=0)  the same pose moved in the world frame
back(pose, d) -> pose          the same pose moved d back along the way the hand points (where to wait before going in)
where(thing) -> dict           what `locate` returns for a thing or "thing:part" (centre, box_min, box_max, top_z ...)

move(arm, pose, joints=False)  the tool point goes to `pose`, in a straight line (joints=False) or the shortest way in
                               joint space (joints=True: for moving across, when the hand's path does not matter)
grip(arm, thing, width)        close the fingers to `width` metres apart on `thing` and hold it: from then on it moves
                               with the hand. The tool point must be within 2 cm of the thing.
push(arm, thing, width=0.0)    close the fingers to `width` and work on `thing` without holding it (press it down,
                               push it). To press with force, put the tool point 1-3 cm *into* the surface: the
                               arm's joint stiffness turns that into a push of tens of newtons.
release(arm)                   let go (opens the fingers fully: if that would hit something, follow with open_fingers)
open_fingers(arm, width)       open or close the fingers to `width` metres without taking anything

phase(seconds, note, *actions) one phase: its actions, at most one move per arm, all at the same time. An arm with no
                               move in a phase stays where it is (and keeps holding or pressing).

Your code must set PLAN = [phase(...), phase(...), ...]. Arms start at their home poses (see read_arm). Only `math`
can be imported.
'''

#: the sandbox: the API, then the agent's code, then PLAN printed as JSON
_SANDBOX = r'''
import json, math, sys
_IN = json.loads(sys.stdin.read())
_ARMS, _THINGS = _IN["arms"], _IN["things"]

def _norm(v):
    n = math.sqrt(sum(x * x for x in v)); return [x / n for x in v]
def _cross(a, b):
    return [a[1]*b[2]-a[2]*b[1], a[2]*b[0]-a[0]*b[2], a[0]*b[1]-a[1]*b[0]]
def _quat(approach, across):
    z = _norm(approach); d = sum(a*b for a, b in zip(across, z))
    y = _norm([a - d*b for a, b in zip(across, z)]); x = _cross(y, z)
    m = [[x[0], y[0], z[0]], [x[1], y[1], z[1]], [x[2], y[2], z[2]]]
    tr = m[0][0] + m[1][1] + m[2][2]
    if tr > 0:
        s = math.sqrt(tr + 1.0) * 2
        return [0.25*s, (m[2][1]-m[1][2])/s, (m[0][2]-m[2][0])/s, (m[1][0]-m[0][1])/s]
    i = max(range(3), key=lambda k: m[k][k]); j, k = (i+1) % 3, (i+2) % 3
    s = math.sqrt(1.0 + m[i][i] - m[j][j] - m[k][k]) * 2
    q = [0.0]*4; q[0] = (m[k][j]-m[j][k])/s; q[i+1] = 0.25*s; q[j+1] = (m[j][i]+m[i][j])/s; q[k+1] = (m[k][i]+m[i][k])/s
    return q
def _rot(q, v):
    w, x, y, z = q
    return [v[0]*(1-2*(y*y+z*z)) + v[1]*2*(x*y-w*z) + v[2]*2*(x*z+w*y),
            v[0]*2*(x*y+w*z) + v[1]*(1-2*(x*x+z*z)) + v[2]*2*(y*z-w*x),
            v[0]*2*(x*z-w*y) + v[1]*2*(y*z+w*x) + v[2]*(1-2*(x*x+y*y))]
_PITCH = {"down": 90.0, "steep": 60.0, "tilted": 40.0, "level": 0.0}
def tool_pose(arm, x, y, z, point="down", across=None):
    if arm not in _ARMS: raise ValueError(f"no arm {arm!r}; arms: {', '.join(_ARMS)}")
    bx, by = _ARMS[arm]["base"][:2]
    rx, ry = x - bx, y - by; n = math.hypot(rx, ry) or 1.0
    if isinstance(point, str):
        if point not in _PITCH: raise ValueError(f"point must be one of {list(_PITCH)} or a direction")
        p = math.radians(_PITCH[point])
        approach = [rx/n*math.cos(p), ry/n*math.cos(p), -math.sin(p)]
    else:
        approach = list(point)
    acr = list(across) if across is not None else [-ry/n, rx/n, 0.0]
    return [float(x), float(y), float(z)] + _quat(approach, acr)
def above(pose, dz): return [pose[0], pose[1], pose[2] + dz] + list(pose[3:])
def shift(pose, dx=0.0, dy=0.0, dz=0.0): return [pose[0] + dx, pose[1] + dy, pose[2] + dz] + list(pose[3:])
def back(pose, d):
    zz = _rot(pose[3:], [0.0, 0.0, 1.0]); return [pose[0] - d*zz[0], pose[1] - d*zz[1], pose[2] - d*zz[2]] + list(pose[3:])
def where(thing):
    if thing not in _THINGS: raise KeyError(f"nothing called {thing!r}; things: {', '.join(_THINGS)}")
    return dict(_THINGS[thing])
def _arm(a):
    if a not in _ARMS: raise ValueError(f"no arm {a!r}; arms: {', '.join(_ARMS)}")
    return a
def _pose(p):
    p = [float(v) for v in p]
    if len(p) != 7: raise ValueError("a pose is [x, y, z, qw, qx, qy, qz]")
    return p
def move(arm, pose, joints=False): return {"type": "move", "arm": _arm(arm), "to": _pose(pose), "joints": bool(joints)}
def grip(arm, thing, width): return {"type": "grip", "arm": _arm(arm), "thing": str(thing), "width": float(width), "attach": True}
def push(arm, thing, width=0.0): return {"type": "grip", "arm": _arm(arm), "thing": str(thing), "width": float(width), "attach": False}
def release(arm): return {"type": "release", "arm": _arm(arm)}
def open_fingers(arm, width): return {"type": "open", "arm": _arm(arm), "width": float(width)}
def clip(tool, thing): return {"type": "clip", "tool": str(tool), "thing": str(thing)}
def unclip(tool): return {"type": "unclip", "tool": str(tool)}
def phase(seconds, note, *actions):
    moved = [a["arm"] for a in actions if a["type"] == "move"]
    dup = {a for a in moved if moved.count(a) > 1}
    if dup: raise ValueError(f"phase {note!r}: more than one move for {', '.join(sorted(dup))}")
    return {"seconds": float(seconds), "note": str(note), "actions": list(actions)}
_ALLOWED = {"math"}
_real_import = __import__
def _import(name, *a, **k):
    if name.split(".")[0] not in _ALLOWED: raise ImportError(f"only math can be imported, not {name}")
    return _real_import(name, *a, **k)
import builtins
_G = {"__builtins__": {**{k: getattr(builtins, k) for k in dir(builtins) if k not in ("open", "exec", "eval", "compile", "input", "breakpoint", "exit", "quit")}, "__import__": _import},
      "math": math, "tool_pose": tool_pose, "above": above, "shift": shift, "back": back, "where": where,
      "move": move, "grip": grip, "push": push, "release": release, "open_fingers": open_fingers,
      "clip": clip, "unclip": unclip, "phase": phase}
exec(compile(_IN["code"], "<program>", "exec"), _G)
if "PLAN" not in _G: raise NameError("the program must set PLAN")
print("\n__PLAN__" + json.dumps(_G["PLAN"]))
'''


def run_sandbox(code: str, arms: dict, things: dict) -> tuple[list | None, str]:
    """(the plan as JSON, what the program printed or its error)."""
    try:
        p = subprocess.run([sys.executable, "-I", "-c", _SANDBOX], input=json.dumps(
            {"code": code, "arms": arms, "things": things}), capture_output=True, text=True, timeout=TIME_LIMIT_S)
    except subprocess.TimeoutExpired:
        return None, f"the program ran longer than {TIME_LIMIT_S} s"
    out, _, plan = p.stdout.partition("\n__PLAN__")
    if p.returncode != 0 or not plan:
        err = p.stderr.strip().splitlines()
        return None, (out[-1500:] + "\n" if out.strip() else "") + "\n".join(err[-8:])
    return json.loads(plan), out[-1500:]


def to_script(plan: list) -> list[Phase]:
    out = []
    for ph in plan:
        acts = []
        for a in ph["actions"]:
            if a["type"] == "move":
                acts.append(Move(a["arm"], tuple(a["to"]), a["joints"]))
            elif a["type"] == "grip":
                acts.append(Grip(a["arm"], a["thing"], a["width"], a["attach"]))
            elif a["type"] == "release":
                acts.append(Release(a["arm"]))
            elif a["type"] == "open":
                acts.append(Open(a["arm"], a["width"]))
            elif a["type"] == "clip":
                acts.append(Clip(a["tool"], a["thing"]))
            elif a["type"] == "unclip":
                acts.append(Unclip(a["tool"]))
        out.append(Phase(ph["seconds"], tuple(acts), ph["note"]))
    return out


def acting_arms(script: list[Phase]) -> set[str]:
    """Arms with explicit actions; tool catches belong to the hand already holding them."""
    return {a.slot for ph in script for a in ph.actions if isinstance(a, (Move, Grip, Open, Release))}


def check(inst, script: list[Phase], timeout_s: float | None = None) -> dict:
    """Plan the script and walk it for clearance: ``{"ok": bool, "problems": [...], "plan": Plan | None}``.
    ``timeout_s`` bounds the planner's search (``policies.plan.PlanTimeout``: a failed check); None, no bound."""
    from ..policies import clearance
    from ..policies.plan import PlanError, plan

    names = {o.name for o in inst.scene.objects} | {c.name for c in inst.scene.cloths}
    bad = [f"phase {i} ({ph.note or 'no note'}): no thing {a.target!r} to grip" for i, ph in enumerate(script)
           for a in ph.actions if isinstance(a, Grip) and a.target not in names]
    rigid = {o.name for o in inst.scene.dynamic_objects}
    for i, ph in enumerate(script):
        for a in ph.actions:
            if isinstance(a, (Clip, Unclip)):
                targets = (a.tool, a.target) if isinstance(a, Clip) else (a.tool,)
                bad += [f"phase {i}: no dynamic rigid body {n!r} for the clip action" for n in targets if n not in rigid]
                if isinstance(a, Clip) and a.tool == a.target:
                    bad.append(f"phase {i}: a tool cannot clip itself")
    if bad:
        return {"ok": False, "problems": bad, "plan": None}
    try:
        p = plan(inst, script, timeout_s=timeout_s)
    except PlanError as e:
        return {"ok": False, "problems": [f"inverse kinematics: {e}".replace(f"{inst.spec.id}: ", "")], "plan": None}
    clashes = clearance.check(inst, p)
    wide = clearance.too_wide(inst, p)
    counts = Counter((c.phase, c.a, c.b) for c in clashes)
    first = {}
    for c in clashes:
        first.setdefault((c.phase, c.a, c.b), c)
    problems = [f"{n} time(s): {first[k]}" for k, n in counts.most_common(MAX_CLASHES)]
    if len(counts) > MAX_CLASHES:
        problems.append(f"... and {len(counts) - MAX_CLASHES} more kinds of clash")
    return {"ok": not clashes and not wide, "problems": wide + problems, "plan": p, "warnings": p.warnings}
