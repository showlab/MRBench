"""Renders a case in Isaac and records what the team senses at the start: every camera's picture, and each arm's joint
angles, finger opening and joint torques after a second at rest.  Isaac process only (``python -m mrbench.agent.snapshot
TASK VARIANT OUT``); it leaves with ``os._exit`` (Kit hangs on shutdown once cameras have rendered).

    python -m mrbench.agent.snapshot all [--gpus 0,1] [--cases Con-01-exact,...] [--force]

takes every case's snapshot that is not there yet into ``runs/agent/cases/<case>``, one Isaac process per case, as
many at once as GPUs given.  ``--scenes`` takes each case in each scene of ``cases.scene_ids`` as well
(``Con-01-exact@hard.s2.varied``).
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

#: the fixed cameras: (eye, target, focal length mm) in the world frame.  "main" and "top" are the docs' views; the
#: others look from each side of the table and from low over the south-east corner at the middle.
VIEWS = {
    "main": ((0.0, -2.0, 3.0), (0.0, 0.1, 0.8), 16.5),
    "top": ((0.0, -0.001, 4.0), (0.0, 0.0, 0.75), 18.4),
    "front": ((0.75, -1.45, 1.35), (-0.05, -0.12, 0.85), 18.0),
    "west": ((-2.3, -0.55, 1.65), (-0.1, -0.05, 0.85), 16.0),
    "east": ((2.3, -0.55, 1.65), (0.1, -0.05, 0.85), 16.0),
    "north": ((0.0, 2.1, 1.85), (0.0, -0.1, 0.85), 16.0),
}
SIZE, WRIST_SIZE = (960, 720), (480, 360)
#: physics run with the arms held at home before anything is read
SETTLE_S = 1.0


def snapshot(case_id: str, out: Path) -> dict:
    from PIL import Image

    from ..engines import make_engine
    from ..policies import make_policy
    from .cases import case_from_id

    case = case_from_id(case_id)
    engine = make_engine("isaac", headless=True, cameras=dict(VIEWS), camera_size=SIZE, wrist_cameras=True,
                         wrist_size=WRIST_SIZE)
    frame = engine.load(case.inst.scene, case.seed)
    hold = make_policy("do_nothing")
    hold.reset(case.inst, case.seed)
    for _ in range(int(round(SETTLE_S / engine.control_dt))):
        frame = engine.step(hold.act(frame))
    pictures = engine.render()
    out.mkdir(parents=True, exist_ok=True)
    for view, rgb in pictures.items():
        Image.fromarray(rgb).save(out / f"{view}.jpg", quality=90)
    arms = {}
    for a in case.inst.scene.arms:
        arm_ids, _ = engine._joint_ids[a.slot]
        data = engine.scene[a.slot].data
        arms[a.slot] = {**frame.arms[a.slot], "torque": data.applied_torque[0, arm_ids].tolist(),
                        "torque_limit": data.joint_effort_limits[0, arm_ids].tolist()
                        if hasattr(data, "joint_effort_limits") else None}
    state = {"case": case_id, "t": frame.t, "arms": arms, "objects": {k: list(v) for k, v in frame.objects.items()},
             "views": sorted(pictures)}
    (out / "state.json").write_text(json.dumps(state, indent=1))
    return state


def snapshot_all(case_ids, gpus, force: bool = False) -> int:
    """Each case's snapshot in its own Isaac process, spread over ``gpus``: how many failed."""
    import subprocess
    from concurrent.futures import ThreadPoolExecutor

    from ..core import plugins
    from ..core.paths import REPO_ROOT

    root = REPO_ROOT / "runs" / "agent" / "cases"
    todo = [c for c in case_ids if force or not (root / c / "state.json").exists()]
    free = list(gpus)

    def one(case_id):
        gpu = free.pop()
        try:
            (root / case_id).mkdir(parents=True, exist_ok=True)
            env = {**os.environ, "CUDA_VISIBLE_DEVICES": gpu, **plugins.child_env()}
            with (root / case_id / "snapshot.log").open("w") as log:
                code = subprocess.run([sys.executable, "-m", "mrbench.agent.snapshot", case_id, str(root / case_id)],
                                      env=env, stdout=log, stderr=log, timeout=900).returncode
            ok = code == 0 and (root / case_id / "state.json").exists()
            print(f"{case_id} on GPU {gpu}: {'ok' if ok else 'FAILED (see its snapshot.log)'}", flush=True)
            return ok
        finally:
            free.append(gpu)

    with ThreadPoolExecutor(len(gpus)) as pool:
        results = list(pool.map(one, todo))
    print(f"{len(todo) - sum(results)} of {len(todo)} failed; {len(case_ids) - len(todo)} were there already")
    return len(todo) - sum(results)


def main(argv) -> int:
    if len(argv) > 1 and argv[1] == "all":
        import argparse

        from .cases import cases, scene_ids

        p = argparse.ArgumentParser(prog="python -m mrbench.agent.snapshot all")
        p.add_argument("--gpus", default=os.environ.get("MRBENCH_ISAAC_GPU", "0"))
        p.add_argument("--cases", default="all")
        p.add_argument("--force", action="store_true", help="take them again even if they are there")
        p.add_argument("--scenes", action="store_true", help="each case in each varied scene too (cases.scene_ids)")
        a = p.parse_args(argv[2:])
        ids = [f"{t}-{v}" for t, v in cases()] if a.cases == "all" else a.cases.split(",")
        if a.scenes:
            ids = ids + scene_ids(ids)
        return 1 if snapshot_all(ids, a.gpus.split(","), a.force) else 0
    case_id, out = argv[1], Path(argv[2])
    try:
        state = snapshot(case_id, out)
        print(f"snapshot {state['case']}: {', '.join(state['views'])}", flush=True)
        code = 0
    except Exception as e:  # noqa: BLE001 -- report and leave; Kit must not be shut down normally
        import traceback

        traceback.print_exc()
        print(f"snapshot failed: {e}", flush=True)
        code = 1
    os._exit(code)


if __name__ == "__main__":
    sys.exit(main(sys.argv))
