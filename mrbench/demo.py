"""Videos of the scripted reference doing a task, in Isaac Sim with physics running.

    mrbench demo Con-01                        # both levels -> docs/media/videos/Con-01-<level>.mp4
    mrbench demo Con-01 Con-04 --gpus 0,1

A task that has a scripted reference exposes ``script(inst)`` in its impl module
(``policies/script.py`` says what a script is).  One Isaac process per task-level:
it builds the scene, plays the script (``engines/isaac/puppeteer.py``), scores what
physically happened with the task's own success formula, and writes the video and
``<out>/<ID>-<level>.json`` with the score.  The reference is privileged -- a grip
attaches what is gripped -- so its success shows the task can be done, never a result.
"""

from __future__ import annotations

import importlib
import json
from pathlib import Path
from typing import Sequence

from .render import SIZE, default_views, run_children


def has_script(task: str) -> bool:
    from .tasks import get

    return hasattr(importlib.import_module(get(task).impl), "script")


def demo_level(task: str, level: str, seed: int, out: Path, settle_s: float = 0.5, look=None,
               variant: str = "exact", decoy: bool = False) -> dict:
    """In this process: play the task's script and record it, in ``look`` (``scenery.parse_look``; None: the
    default look), its ``variant`` version (with ``decoy``, the borrow version's distractor in its prop's place).
    Needs Isaac, and leaves it running: end the process with ``os._exit`` afterwards."""
    from .tasks import build, tag

    inst = build(task, level, seed, look, variant=variant, decoy=decoy)
    return play(inst, out, f"{tag(task, variant, decoy)}-{level}", settle_s, decoy=decoy)


def play(inst, out: Path, name: str, settle_s: float = 0.5, decoy: bool = False, video: bool = True,
         measured_clearance: bool = False) -> dict:
    """In this process: play an instance's scripted reference in Isaac and record it, as ``<out>/<name>.mp4`` and
    ``.json`` (the score, milestone times, closest approaches, missed grips, how far each arm was off its plan);
    without ``video``, no camera at all and no ``.mp4`` (faster: the score alone).  Needs Isaac, and leaves it
    running: end the process with ``os._exit`` afterwards."""
    from .reference import play_script
    from .scoring import score

    script = importlib.import_module(inst.spec.impl).script(inst)
    pup, frames = play_script(inst, script, settle_s, video)
    result = score(inst, frames)
    clip = pup.save_video(out / f"{name}.mp4") if video else None
    from .policies.clearance import closest
    from .render import props_moved
    from .scenery import describe

    report = {"task": inst.spec.id, "level": inst.level, "seed": inst.seed, "variant": inst.variant, "decoy": decoy,
              "look": describe(inst.scene.look),
              "sim_time_s": round(pup.t, 2), "video": clip.name if clip else None, **result.to_json(),
              **budget(inst, result), "closest": closest(inst, frames),
              "missed_grips": pup.missed, "props_moved_m": props_moved(inst, frames[0], frames[-1]),
              "phases": pup.trace}
    if decoy or inst.metadata.get("decoy"):
        report["decoy_protocol"] = decoy_protocol(inst, getattr(pup, "clashes", []), result, pup.missed,
                                                  getattr(pup, "safe_stop", None))
    if measured_clearance:
        from .reference_audit import audit_clearance
        from .policies.script import phase

        report['clearance_audit'] = audit_clearance(inst, frames, [phase(settle_s, '')] + script)
    (out / f"{name}.json").write_text(json.dumps(report, indent=1))
    return report


#: the distractor-run protocol (``mrbench demo --decoy``)
DECOY_PROTOCOL = "v1"


def decoy_protocol(inst, clashes, result, missed, safe_stop=None) -> dict:
    """What a distractor run's report says about it (``mrbench demo --decoy``): every plan clash (played anyway), which
    of them involve the distractor (named as the prop it stands in for) and which do not; it counts as a valid
    negative only if the task did not succeed and no clash that does not involve the distractor goes into anything
    (a negative gap).  Why the attempt failed -- the distractor's declared defect -- is read from the run itself."""
    names = set((inst.metadata.get("borrow") or {}).get("props", ()))
    if safe_stop is not None:  # the plan's clashes after the run stopped were never played
        clashes = [c for c in clashes if c.t <= safe_stop]
    body = lambda n: n.split(".")[0]  # noqa: E731
    mine = [c for c in clashes if body(c.a) in names or body(c.b) in names]
    other = [c for c in clashes if c not in mine]
    into = [c for c in other if c.gap < 0]
    return {"version": DECOY_PROTOCOL, "distractor": sorted(names),
            "clashes_with_distractor": [str(c) for c in mine], "other_clashes": [str(c) for c in other],
            "other_clashes_into": [str(c) for c in into], "missed_grips": len(missed or ()), "safe_stop_s": safe_stop,
            "valid_negative": (not result.success) and not into}


def budget(inst, result) -> dict:
    """The version's time budget, and whether the success came within it (``success`` alone says only that it came
    while the script played, however long that was)."""
    horizon = inst.spec.horizon(inst.variant)
    t = (getattr(result, "first_hit_s", None) or {}).get("success")
    return {"horizon_s": horizon, "success_within_horizon": bool(result.success and t is not None and t <= horizon)}


def demo_all(tasks: Sequence[str], levels: Sequence[str], seed: int, out: Path, gpus: Sequence[str],
             look=None, variant: str = "exact", decoy: bool = False) -> int:
    from .tasks import tag

    extra = (["--look", look] if look else []) + ["--variant", variant] + (["--decoy"] if decoy else [])
    jobs = [(f"{tag(t, variant, decoy)}/{lv}", ["demo", t, "--levels", lv, "--seed", str(seed), "-o", str(out),
                                                "--child"] + extra) for t in tasks for lv in levels]
    failed = run_children(jobs, gpus, out / "logs")
    for t in tasks:
        for lv in levels:
            p = out / f"{tag(t, variant, decoy)}-{lv}.json"
            if p.exists():
                r = json.loads(p.read_text())
                print(f"{t}/{lv}: success={r['success']} milestones={r['milestones_hit']} video={r['video']}")
    return 1 if failed else 0

