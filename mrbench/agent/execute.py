"""Plays an agent's program on a case in Isaac, with physics, and scores what happened with the task's own success
formula.  Isaac process only:

    python -m mrbench.agent.execute CASE PLAN_JSON OUT_DIR [--probe SECONDS]

``PLAN_JSON`` may be ``@script``: the task's own scripted reference (``script(inst)``), for the Oracle-script row;
it exists for a case whose scene is the task's own (``exact``).  ``@skill:NAME``: one of the case's skills
(``skill_exec.py``: its version's reference, or a gold route), for the skill-level executor.  ``--probe`` marks a probe (``env.Env._probe``): the
same play, read back for how things moved.

Writes ``OUT_DIR/run.mp4``, ``OUT_DIR/report.json`` (success, milestones, how far each arm ended from its plan at
every phase, grips that closed on nothing, where each thing ended) and a few keyframes ``OUT_DIR/key_<k>.jpg`` taken
at phase ends.  Grips are privileged, as in the scripted references: a grip fixes what is between the fingers to the
hand.
"""

from __future__ import annotations

import json
import math
import os
import subprocess
import sys
from pathlib import Path

#: keyframes kept for the agent
KEYFRAMES = 4


def execute(case_id: str, plan_json: Path, out: Path) -> dict:
    from ..engines import make_engine
    from ..engines.isaac.puppeteer import Puppeteer
    from ..policies.plan import PlanError
    from ..policies.script import phase
    from ..scoring import score
    from .cases import case_from_id
    from .program import to_script
    from .snapshot import SIZE, VIEWS

    case = case_from_id(case_id)
    if str(plan_json) == "@script":
        import importlib

        if case.variant != "exact":
            raise ValueError(f"the scripted reference is for the task's own scene, not its {case.variant} version")
        script = importlib.import_module(case.inst.spec.impl).script(case.inst)
    elif str(plan_json).startswith("@skill:"):  # the skill-level executor (skill_exec.py): any version's script
        from .skill_exec import script_for

        script = script_for(case, str(plan_json)[len("@skill:"):])
    else:
        script = to_script(json.loads(plan_json.read_text()))
    engine = make_engine("isaac", headless=True, cameras={"main": VIEWS["main"]}, camera_size=SIZE, wrist_cameras=True)
    engine.load(case.inst.scene, case.seed)
    engine.render()
    pup = Puppeteer(engine, case.inst, camera="main")
    out.mkdir(parents=True, exist_ok=True)
    try:
        frames = pup.play([phase(0.5, "")] + script)
    except PlanError as e:
        report = {"case": case_id, "played": False, "error": str(e)}
        (out / "report.json").write_text(json.dumps(report, indent=1))
        return report
    result = score(case.inst, frames)
    video = pup.save_video(out / "run.mp4")
    first, last = frames[0], frames[-1]
    moved = {k: round(sum((a - b) ** 2 for a, b in zip(last.objects[k][:3], first.objects[k][:3])) ** 0.5, 3)
             for k in last.objects}
    # the most each thing was ever off where it started: how far, and how far turned (deg), and when
    worst = {}
    for k in first.objects:
        q0 = first.objects[k][3:]
        far = max(((sum((a - b) ** 2 for a, b in zip(f.objects[k][:3], first.objects[k][:3])) ** 0.5, f.t)
                   for f in frames), key=lambda x: x[0])
        turn = max(((2 * math.degrees(math.acos(min(1.0, abs(sum(a * b for a, b in zip(f.objects[k][3:], q0)))))),
                     f.t) for f in frames), key=lambda x: x[0])
        worst[k] = {"max_moved_m": round(far[0], 3), "at_s": round(far[1], 1), "max_turned_deg": round(turn[0], 1),
                    "turned_at_s": round(turn[1], 1)}
    (out / "frames.json").write_text(json.dumps([f.to_json() for f in frames[::2]]))
    phases = [{"t": tr["t"], "note": tr["note"],
               "off_plan_m": {s: d[0] for s, d in tr["off_plan"].items() if d[0] > 0.02}} for tr in pup.trace]
    times = [tr["t"] for tr in pup.trace]
    picks = [times[round(i * (len(times) - 1) / (KEYFRAMES - 1))] for i in range(KEYFRAMES)] if times else []
    keys = []
    for k, t in enumerate(dict.fromkeys(picks)):
        f = out / f"key_{k}.jpg"
        subprocess.run(["ffmpeg", "-nostdin", "-y", "-loglevel", "error", "-ss", f"{max(0.0, t - 0.05):.2f}", "-i",
                        str(video), "-frames:v", "1", "-q:v", "3", str(f)], stdin=subprocess.DEVNULL, check=False)
        if f.exists():
            keys.append({"t": t, "file": f.name})
    report = {"case": case_id, "played": True, **result.to_json(), "sim_time_s": round(pup.t, 2),
              "missed_grips": pup.missed, "moved_m": moved, "worst": worst, "phases": phases, "keyframes": keys, "video": video.name}
    (out / "report.json").write_text(json.dumps(report, indent=1))
    return report


def main(argv) -> int:
    code = 0
    try:
        r = execute(argv[1], Path(argv[2]), Path(argv[3]))
        print(f"executed {argv[1]}: success={r.get('success')} milestones={r.get('milestones_hit')}", flush=True)
    except Exception as e:  # noqa: BLE001 -- report and leave; Kit must not be shut down normally
        import traceback

        traceback.print_exc()
        Path(argv[3]).mkdir(parents=True, exist_ok=True)
        (Path(argv[3]) / "report.json").write_text(json.dumps({"case": argv[1], "played": False,
                                                                "error": f"{type(e).__name__}: {e}"}))
        code = 1
    sys.stdout.flush()
    os._exit(code)


if __name__ == "__main__":
    sys.exit(main(sys.argv))
