"""Pictures of every task, rendered in Isaac Sim: the start and the goal, at both levels.

    mrbench render                       # every task, seed 0 -> docs/media/tasks/<ID>.jpg
    mrbench render Rea-01 Con-03 --gpus 0,1    # some tasks, spread over two GPUs

One Isaac process per (task, level): the two levels of a task have different
objects, and the Isaac engine builds one stage per process.  Each process

1. builds the scene, holds every arm at home for ``SETTLE_S`` of physics and
   renders the **start**;
2. for a task whose impl names a ``MIDDLE`` keyframe, puts the objects where
   the oracle has them then and renders the **middle**;
3. puts every object at its last oracle keyframe pose -- physics paused, the
   arms still at home -- and renders the **goal**.

The goal picture shows what success looks like, not that the arms can reach
it: nothing has moved an arm yet.  Each process also writes what the settle
showed (objects and cloths that slid, turned, sank or fell; arms that sagged off home) to
``<raw>/<ID>-<level>.json``; ``compose`` then lays the four pictures of a task
side by side.  Isaac is imported only inside :func:`render_level`.
"""

from __future__ import annotations

import json
import math
import os
import subprocess
import sys
from pathlib import Path
from typing import Mapping, Optional, Sequence

from .engines.base import Frame, Vec3
from .robots import get_model
from .tasks.instance import TaskInstance, keyframes
from .tasks.layouts import TABLE_TOP_Z

LEVELS = ("easy", "hard")
#: "middle" only for a task whose impl names a keyframe to show (``MIDDLE``)
STATES = ("start", "middle", "goal")
#: fixed cameras, (eye, target, focal length mm) in the world frame; the table is
#: 3.2 x 1.8 m, its top at z = 0.75, and every task mounts its arms in the same three
#: places.  "main" looks steeply down from high over the south side, over the south arm,
#: at the whole length of the table and both far ends (from the south-east, the south arm
#: would hide the west one); "top" is nearly orthographic, for checking where things are -- below
#: the room's ceiling (4.2 m up).
MAIN_VIEW_POSE: tuple[Vec3, Vec3, float] = ((0.0, -2.0, 3.0), (0.0, 0.1, 0.8), 16.5)
TOP_VIEW: tuple[Vec3, Vec3, float] = ((0.0, -0.001, 4.0), (0.0, 0.0, TABLE_TOP_Z), 18.4)
#: the views every task is rendered from; the main one is the picture in docs/TASKS.md
MAIN_VIEW, VIEW_NAMES = "main", ("main", "top")


def default_views() -> dict[str, tuple[Vec3, Vec3, float]]:
    return {"main": MAIN_VIEW_POSE, "top": TOP_VIEW}


SIZE = (960, 720)
#: physics run before the start picture, arms held at home: long enough for
#: anything unstable (a sagging arm, an object spawned in a collider) to show
SETTLE_S = 1.0
#: what the settle may change before the report flags it
MOVED_M, TURNED_DEG, SAGGED_RAD = 0.01, 5.0, 0.05
PANEL_W = 540


def middle_poses(inst: TaskInstance) -> Optional[dict]:
    """The state the impl's ``MIDDLE`` keyframe reaches -- the doors held open, the towel
    over the hanger -- for a task whose start and goal alone do not show what it is about."""
    import importlib

    i = getattr(importlib.import_module(inst.spec.impl), "MIDDLE", None)
    if i is None:
        return None
    out: dict = {}
    for kf in keyframes(inst)[: i + 1]:
        out.update(kf)
    return out


def goal_poses(inst: TaskInstance) -> dict:
    """Every object's pose (a cloth's points) at the end of the oracle: the last keyframe that places it."""
    out: dict = {}
    for kf in keyframes(inst):
        out.update(kf)
    return out


def _turned_deg(q0: Sequence[float], q1: Sequence[float]) -> float:
    """How far a body turned between two orientations."""
    dot = abs(sum(a * b for a, b in zip(q0, q1)))
    return math.degrees(2 * math.acos(min(1.0, dot)))


def settle_report(inst: TaskInstance, first: Frame, last: Frame) -> dict:
    """What holding still for ``SETTLE_S`` did to the scene."""
    objects = {}
    for o in inst.scene.dynamic_objects:
        p0, p1 = first.objects[o.name], last.objects[o.name]
        moved = math.dist(p0[:3], p1[:3])
        objects[o.name] = {"moved_m": round(moved, 4), "dz_m": round(p1[2] - p0[2], 4),
                           "turned_deg": round(_turned_deg(p0[3:], p1[3:]), 2)}
    arms = {}
    for a in inst.scene.arms:
        m = get_model(a.model)
        q = last.arms[a.slot]["q"]
        err = [abs(v - h) for v, h in zip(q, m.home)]
        arms[a.slot] = {"max_home_error_rad": round(max(err), 4), "joint": m.arm_joints[err.index(max(err))],
                        "gripper": last.arms[a.slot]["gripper"]}
    flags = [f"{n} moved {v['moved_m'] * 100:.1f} cm" for n, v in objects.items() if v["moved_m"] > MOVED_M]
    flags += [f"{n} turned {v['turned_deg']:.0f} deg" for n, v in objects.items() if v["turned_deg"] > TURNED_DEG]
    flags += [f"{n} below the table top" for n in objects if last.objects[n][2] < TABLE_TOP_Z]
    for c in inst.scene.cloths:  # a cloth: how far its marks moved, and none through the table
        moved = max(math.dist(a, b) for a, b in zip(first.cloths[c.name], last.cloths[c.name]))
        lowest = min(p[2] for p in last.cloths[c.name])
        objects[c.name] = {"moved_m": round(moved, 4), "lowest_above_table_m": round(lowest - TABLE_TOP_Z, 4)}
        if moved > MOVED_M:
            flags.append(f"{c.name} moved {moved * 100:.1f} cm")
        if lowest < TABLE_TOP_Z:
            flags.append(f"{c.name} below the table top")
    flags += [f"{s} is {v['max_home_error_rad']:.2f} rad off home at {v['joint']}"
              for s, v in arms.items() if v["max_home_error_rad"] > SAGGED_RAD]
    props = props_moved(inst, first, last)
    flags += [f"{n} ({inst.scene.props[[p.name for p in inst.scene.props].index(n)].kind}) moved {m * 100:.1f} cm"
              for n, m in props.items() if m > MOVED_M]
    return {"settle_s": SETTLE_S, "objects": objects, "arms": arms, "props_moved_m": props, "flags": flags}


def props_moved(inst: TaskInstance, first: Frame, last: Frame) -> dict:
    """How far each prop's middle, and each distractor's (``scenery.distractors``), moved between two frames (m)."""
    out = {p.name: round(math.dist(first.props[p.name][:3], last.props[p.name][:3]), 4) for p in inst.scene.props}
    out.update({n: round(math.dist(first.objects[n][:3], last.objects[n][:3]), 4)
                for n in inst.metadata.get("distractors") or {}})
    return out


def raw_path(raw: Path, task: str, level: str, state: str, view: str) -> Path:
    return raw / f"{task}-{level}-{state}-{view}.png"


def render_level(task: str, level: str, seed: int, raw: Path,
                 views: Optional[Mapping[str, tuple[Vec3, Vec3, float]]] = None, size: tuple[int, int] = SIZE,
                 look: Optional[str] = None, variant: str = "exact") -> dict:
    """In this process: render the start and the goal of one task-level, in ``look`` (``scenery.parse_look``;
    None: the default look), its ``variant`` version (written as ``tasks.tag``: "Con-01-borrow-easy-...").  Needs
    Isaac, and leaves it running: end the process with ``os._exit`` afterwards."""
    from PIL import Image

    from .engines import make_engine
    from .policies import make_policy
    from .tasks import build, tag

    inst = build(task, level, seed, look, variant=variant)
    name = tag(task, variant)
    # never closed: Kit's shutdown hangs once cameras have rendered, so the caller
    # leaves the process with os._exit after this returns (see cli.cmd_render)
    views = views or default_views()
    engine = make_engine("isaac", headless=True, cameras=dict(views), camera_size=size)
    frames = [engine.load(inst.scene, seed)]
    hold = make_policy("do_nothing")
    hold.reset(inst, seed)
    for _ in range(int(round(SETTLE_S / engine.control_dt))):
        frames.append(engine.step(hold.act(frames[-1])))
    pictures = {"start": engine.render()}
    report = settle_report(inst, frames[0], frames[-1])
    cloths = {c.name for c in inst.scene.cloths}

    def put(poses: dict) -> None:
        for name, value in poses.items():
            (engine.set_cloth_points if name in cloths else engine.set_object_pose)(name, value)

    middle = middle_poses(inst)
    if middle is not None:
        put(middle)
        pictures["middle"] = engine.render()
    put(goal_poses(inst))
    pictures["goal"] = engine.render()

    raw.mkdir(parents=True, exist_ok=True)
    for state, by_view in pictures.items():
        for view, rgb in by_view.items():
            Image.fromarray(rgb).save(raw_path(raw, name, level, state, view))
    from .scenery import describe

    report.update(task=task, level=level, seed=seed, variant=variant, instruction=inst.instruction,
                  look=describe(inst.scene.look), props=[p.kind for p in inst.scene.props])
    (raw / f"{name}-{level}.json").write_text(json.dumps(report, indent=1))
    return report


def compose(task: str, raw: Path, out: Path, view: str = MAIN_VIEW, panel_w: int = PANEL_W) -> Path:
    """One strip per task and view: easy start, (easy middle,) easy goal, hard start, ...
    ``<ID>.jpg`` for the main view, ``<ID>-<view>.jpg`` for the others."""
    from PIL import Image, ImageDraw, ImageFont

    states = [s for s in STATES if all(raw_path(raw, task, lv, s, view).exists() for lv in LEVELS)]
    panels = []
    for level in LEVELS:
        for state in states:
            img = Image.open(raw_path(raw, task, level, state, view)).convert("RGB")
            img = img.resize((panel_w, round(img.height * panel_w / img.width)), Image.LANCZOS)
            draw = ImageDraw.Draw(img)
            label = f"{level} · {state}"
            try:
                font = ImageFont.truetype("DejaVuSans.ttf", 18)
            except OSError:
                font = ImageFont.load_default()
            x0, y0, x1, y1 = draw.textbbox((8, 5), label, font=font)
            draw.rectangle((0, 0, x1 + 8, y1 + 6), fill=(40, 40, 40))
            draw.text((8, 5), label, fill=(255, 255, 255), font=font)
            panels.append(img)
    strip = Image.new("RGB", (sum(p.width for p in panels), panels[0].height))
    x = 0
    for p in panels:
        strip.paste(p, (x, 0))
        x += p.width
    out.mkdir(parents=True, exist_ok=True)
    path = out / (f"{task}.jpg" if view == MAIN_VIEW else f"{task}-{view}.jpg")
    strip.save(path, quality=88)
    return path


def run_children(jobs: Sequence[tuple[str, list[str]]], gpus: Sequence[str], log_dir: Path,
                 timeout_s: Optional[float] = None) -> list[str]:
    """Run ``python -m mrbench <argv>`` for each (name, argv), at most one per GPU at a time,
    each pinned to its GPU, logging to ``<log_dir>/<name>.log``; with ``timeout_s``, a child running longer is
    killed (on a crowded GPU a child can hang for hours).  Returns the names that failed."""
    import time

    jobs = list(jobs)
    log_dir.mkdir(parents=True, exist_ok=True)
    running: dict[str, tuple[subprocess.Popen, str]] = {}  # gpu -> (process, name)
    started: dict[str, float] = {}
    failed: list[str] = []
    while jobs or running:
        for gpu in [g for g in gpus if g not in running]:
            if not jobs:
                break
            name, argv = jobs.pop(0)
            env = {**os.environ, "MRBENCH_ISAAC_GPU": gpu, "CUDA_VISIBLE_DEVICES": gpu}
            with open(log_dir / f"{name.replace('/', '-')}.log", "w") as log:
                running[gpu] = (subprocess.Popen([sys.executable, "-m", "mrbench", *argv], env=env, stdout=log,
                                                 stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL), name)
            started[gpu] = time.time()
        time.sleep(1.0)
        for gpu, (proc, name) in list(running.items()):
            if proc.poll() is None and timeout_s is not None and time.time() - started[gpu] > timeout_s:
                proc.kill()
                proc.wait()
                print(f"{name} on GPU {gpu}: killed after {timeout_s:.0f} s", flush=True)
            if proc.poll() is None:
                continue
            del running[gpu]
            print(f"{name} on GPU {gpu}: {'ok' if proc.returncode == 0 else f'FAILED (rc={proc.returncode})'}",
                  flush=True)
            if proc.returncode != 0:
                failed.append(name)
    return failed


def render_all(tasks: Sequence[str], seed: int, raw: Path, out: Path, gpus: Sequence[str],
               levels: Sequence[str] = LEVELS, log_dir: Optional[Path] = None, look: Optional[str] = None,
               variant: str = "exact") -> int:
    """One child process per (task, level), at most one per GPU at a time; then compose."""
    from .tasks import tag

    jobs = [(f"{tag(t, variant)}/{lv}", ["render", t, "--levels", lv, "--seed", str(seed), "--raw", str(raw),
                                         "--child", "--variant", variant] + (["--look", look] if look else []))
            for t in tasks for lv in levels]
    failed = run_children(jobs, gpus, log_dir or raw)
    for t in tasks:
        name = tag(t, variant)
        for view in VIEW_NAMES:
            if all(raw_path(raw, name, lv, s, view).exists() for lv in LEVELS for s in ("start", "goal")):
                print(f"{t}: {compose(name, raw, out, view)}")
    for f in failed:
        print(f"failed: {f}", file=sys.stderr)
    return 1 if failed else 0
