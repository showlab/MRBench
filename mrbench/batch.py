"""Many seeded scenes of the benchmark's tasks, each played with its scripted reference in Isaac: how often the
reference succeeds, task by task and version by version, and which scenes it fails on (the scenes the benchmark's
episodes are drawn from must each be doable).

    mrbench batch Con-01 Rea-02 --seeds 0-2 --gpus 0,1 -o runs/batch/try         # Isaac: one process per scene
    mrbench batch --seeds 0-29 --robots random --masses random --gpus 0,1 -o runs/batch/seeds
    mrbench batch --seeds 0-2 --dry -o runs/batch/dry      # no Isaac: plan each reference, check its clearance

A scene is one (task, version, level, seed); each writes ``<out>/<name>.json`` (and, played, ``.mp4``), and a scene
already there is not played again (``--force`` plays it again).  ``<out>/summary.md`` has the success table and the
list of failures.  What changes with the seed:

* where things lie: each task's own seeded jitter, as ``tasks.build`` makes the scene;
* with ``--masses random``: every movable body's mass, times a factor in MASS_RANGE drawn from the seed;
* with ``--robots random``: which robot stands on each mount -- one of the combinations that can do the task and
  still need every arm (``tasks.combos.combinations``), drawn from the seed.  The scripted references were written
  for the card's robots, so such a scene tests the reference as much as the scene;
* with ``--look random``: the table, the room and the light (``scenery``);
* with ``--sizes random``: the size of a task's free object (``SIZES``: the ball, the fruit, the tomato, the
  sponge), times a factor drawn from the seed.  What the level's knob sets (a box's width, a slot's clearance, a
  gap) and what the design fits round it stay as they are.
"""

from __future__ import annotations

import contextlib
import dataclasses
import json
import random
from pathlib import Path
from typing import Optional, Sequence

#: a movable body's mass is scaled by a factor drawn from this range (``--masses random``)
MASS_RANGE = (0.8, 1.25)
#: what ``--sizes random`` changes in a task: its impl's module constants (the scene and its scripted reference
#: both read them), given the impl module and the factor drawn -- a free object's size and what is derived from it
#: when the module is imported -- and the factor's range.
SIZES = {
    # the ball the end hands lift between them (22 cm across; 10 cm of room round it in the box): no smaller (at
    # 0.95-0.97 the west forearm, going down beside it, comes within 1 cm of the box's west lip)
    "Con-03": ((1.0, 1.05), lambda m, k: {"BALL_R": m.BALL_R * k}),
    # the fruit a hand takes (its middle over its foot, and how far the fingers close on it): no larger (at 1.05-1.1
    # a fruit carried at hard passes within 1.5 cm of another)
    "Rea-01": ((0.9, 1.03), lambda m, k: {"FRUIT_R": m.FRUIT_R * k, "FRUIT_Z": m.FRUIT_Z + m.FRUIT_R * (k - 1),
                                          "FRUIT_GRIP": m.FRUIT_GRIP + 2 * m.FRUIT_R * (k - 1)}),
    # the tomato that rolls down the channels: no larger (at 1.04-1.1 the west hand, lifting the stop, comes within
    # 1 cm of it)
    "Rea-02": ((0.9, 1.03), lambda m, k: {"TOMATO_R": m.TOMATO_R * k}),
    # the sponge (and how far a hand opens and closes on its width): no smaller (thinner, the south hand taking it
    # comes within 1 cm of the table)
    "Rea-04": ((1.0, 1.1), lambda m, k: {"SPONGE": tuple(s * k for s in m.SPONGE),
                                          "GRIP_W": m.GRIP_W + m.SPONGE[1] * (k - 1),
                                          "SHAPE_W": m.SHAPE_W + m.SPONGE[1] * (k - 1)}),
}
#: the benchmark's eight
BENCHMARK = ("Con-01", "Con-02", "Con-03", "Con-04", "Rea-01", "Rea-02", "Rea-03", "Rea-04")


@dataclasses.dataclass(frozen=True)
class Scene:
    task: str
    variant: str
    level: str
    seed: int

    @property
    def name(self) -> str:
        from .tasks import tag

        return f"{tag(self.task, self.variant)}-{self.level}-s{self.seed}"


def has_version(task: str, variant: str) -> bool:
    """Whether ``task`` has ``variant`` to play: one its card lists as built, or the surplus version its agent case
    sets up (the card's team and a fourth arm, which the exact reference leaves at home)."""
    from .tasks import get

    if variant == "surplus":
        from .agent.cases import setups

        return task in setups() and "surplus" in setups()[task].expect
    if variant == "swapped":  # the exact team, one place's robot replaced as the agent case has it
        from .agent.cases import has_variant

        return has_variant(task, "swapped")
    return variant in get(task).variants


def scenes(tasks: Sequence[str], variants: Sequence[str], levels: Sequence[str], seeds: Sequence[int]) -> list[Scene]:
    """Every scene to play: the versions each task has (the others are left out)."""
    return [Scene(t, v, lv, s) for t in tasks for v in variants if has_version(t, v) for lv in levels for s in seeds]


def _rng(what: str, task: str, seed: int) -> random.Random:
    from .tasks import get

    return random.Random(f"batch/{what}/{get(task).seed_id}/{seed}")


_COMBOS: dict = {}


def robot_choices(task: str) -> list[dict]:
    """The robot combinations (mount -> model) that can do ``task`` and still need every arm."""
    if task not in _COMBOS:
        from .tasks.combos import combinations

        _COMBOS[task] = [c for c, can, needs in combinations(task) if can and needs]
    return _COMBOS[task]


def randomize(inst, robots: str = "card", masses: str = "card"):
    """The instance with its robots and masses drawn from its seed (``card``: as the card has them), and what was
    drawn."""
    sc = inst.scene
    info: dict = {}
    if robots == "random":
        choices = robot_choices(inst.spec.id)
        if choices:
            combo = _rng("robots", inst.spec.id, inst.seed).choice(choices)
            sc = dataclasses.replace(sc, arms=tuple(dataclasses.replace(a, model=combo.get(a.slot, a.model))
                                                    for a in sc.arms))
        info["robots"] = {a.slot: a.model for a in sc.arms}
    if masses == "random":
        rng = _rng("masses", inst.spec.id, inst.seed)
        scale = {o.name: rng.uniform(*MASS_RANGE) for o in sc.objects if o.dynamic}
        sc = dataclasses.replace(sc, objects=tuple(dataclasses.replace(o, mass_kg=round(o.mass_kg * scale[o.name], 4))
                                                   if o.name in scale else o for o in sc.objects))
        info["mass_scale"] = {k: round(v, 3) for k, v in scale.items()}
    return dataclasses.replace(inst, scene=sc), info


@contextlib.contextmanager
def sized(task: str, seed: int, sizes: str = "card"):
    """While it lasts, ``task``'s free object has the size drawn from ``seed`` (``SIZES``; ``card``: as it is):
    yields what was drawn.  The impl's constants are put back after."""
    if sizes != "random" or task not in SIZES:
        yield {}
        return
    import importlib

    from .tasks import get

    m = importlib.import_module(get(task).impl)
    (lo, hi), change = SIZES[task]
    k = _rng("sizes", task, seed).uniform(lo, hi)
    new = change(m, k)
    old = {name: getattr(m, name) for name in new}
    for name, value in new.items():
        setattr(m, name, value)
    try:
        yield {"size_scale": round(k, 3)}
    finally:
        for name, value in old.items():
            setattr(m, name, value)


def build_scene(scene: Scene, robots: str = "card", masses: str = "card", look: Optional[str] = None):
    from .tasks import build

    if scene.variant == "surplus":  # the exact scene and the agent case's fourth arm, idle at home
        from .agent.cases import EXTRA_ARM, setups

        inst = build(scene.task, scene.level, scene.seed, look)
        # a case's own place for the fourth arm, else EXTRA_ARM
        arms = (*inst.scene.arms, getattr(setups()[scene.task], "extra", None) or EXTRA_ARM)
        inst = dataclasses.replace(inst, scene=dataclasses.replace(inst.scene, arms=arms))
    elif scene.variant == "swapped":  # the exact scene and its reference, one place's robot replaced
        from .agent.cases import setups

        inst = build(scene.task, scene.level, scene.seed, look)
        slot, model = setups()[scene.task].swap
        arms = tuple(dataclasses.replace(a, model=model) if a.slot == slot else a for a in inst.scene.arms)
        inst = dataclasses.replace(inst, scene=dataclasses.replace(inst.scene, arms=arms))
    else:
        inst = build(scene.task, scene.level, scene.seed, look, variant=scene.variant)
    return randomize(inst, robots, masses)


def play_scene(scene: Scene, out: Path, robots: str = "card", masses: str = "card", look: Optional[str] = None,
               dry: bool = False, video: bool = True, sizes: str = "card") -> dict:
    """One scene, in this process: its scripted reference played in Isaac (leaves Isaac running: end the process
    with ``os._exit``), or with ``dry`` only planned and checked for clearance (``agent.program.check``, as ``mrbench
    check``).  Writes and returns its report."""
    import importlib

    out.mkdir(parents=True, exist_ok=True)
    with sized(scene.task, scene.seed, sizes) as drawn:  # the scene and its reference both see the sizes drawn
        inst, info = build_scene(scene, robots, masses, look)
        base = {"scene": scene.name, "task": scene.task, "variant": scene.variant, "level": scene.level,
                "seed": scene.seed, "robots_drawn": robots, "masses_drawn": masses, "sizes_drawn": sizes, **info,
                **drawn}
        if dry:
            from .agent import program

            r = program.check(inst, importlib.import_module(inst.spec.impl).script(inst))
            rep = {**base, "dry": True, "success": bool(r["ok"]), "problems": r["problems"][:20]}
        else:
            from .demo import play

            # the scene's own fields last: a surplus scene's instance is the exact version's, and says so
            rep = {**play(inst, out, scene.name, video=video), **base}
    (out / f"{scene.name}.json").write_text(json.dumps(rep, indent=1))
    return rep


def run(todo: Sequence[Scene], out: Path, gpus: Sequence[str], robots: str = "card", masses: str = "card",
        look: Optional[str] = None, dry: bool = False, force: bool = False, workers: int = 4, video: bool = True,
        timeout_s: Optional[float] = None, sizes: str = "card") -> int:
    """Play every scene not played yet -- in Isaac one process per scene, at most one per GPU at a time, each killed
    after ``timeout_s`` (its scene then reported as not run); dry, in a pool of ``workers`` processes -- then write the
    summary.  Returns how many failed to run (not to succeed)."""
    out.mkdir(parents=True, exist_ok=True)
    todo = [s for s in todo if force or not (out / f"{s.name}.json").exists()]
    if dry:
        from concurrent.futures import ProcessPoolExecutor

        failed = 0
        with ProcessPoolExecutor(max(1, workers)) as pool:
            futures = {pool.submit(play_scene, s, out, robots, masses, look, True, True, sizes): s for s in todo}
            for f, s in futures.items():
                try:
                    r = f.result()
                    print(f"{s.name}: {'clear' if r['success'] else 'NOT clear'}", flush=True)
                except Exception as e:  # noqa: BLE001 -- a scene that does not build is reported, the rest go on
                    failed += 1
                    (out / f"{s.name}.json").write_text(json.dumps({"scene": s.name, "task": s.task, "variant": s.variant,
                                                                    "level": s.level, "seed": s.seed, "dry": True,
                                                                    "success": False, "error": repr(e)}, indent=1))
                    print(f"{s.name}: error {e!r}", flush=True)
    else:
        from .render import run_children

        extra = ["--robots", robots, "--masses", masses, "--sizes", sizes, "-o", str(out), "--child"] \
            + (["--look", look] if look else []) + ([] if video else ["--no-video"])
        jobs = [(s.name, ["batch", s.task, "--variants", s.variant, "--levels", s.level, "--seeds", str(s.seed)] + extra)
                for s in todo]
        gone = run_children(jobs, gpus, out / "logs", timeout_s)
        # a process whose Kit did not start (a stale /dev/shm/carb-RStringInternals-<pid>, its pid reused) is played
        # once more
        again = [j for j in jobs if j[0] in gone and _kit_did_not_start(out / "logs", j[0])]
        if again:
            print(f"{len(again)} scene(s) whose Isaac did not start: once more", flush=True)
            gone = (set(gone) - {j[0] for j in again}) | set(run_children(again, gpus, out / "logs", timeout_s))
        for s in todo:  # a scene whose process failed or was killed: reported as such, to be played again
            if s.name in gone and not (out / f"{s.name}.json").exists():
                (out / f"{s.name}.json").write_text(json.dumps({"scene": s.name, "task": s.task, "variant": s.variant,
                                                                "level": s.level, "seed": s.seed, "success": False,
                                                                "error": "the run failed or was killed (see logs/)"},
                                                               indent=1))
        failed = len(gone)
    print(summarize(out))
    return failed


#: what Kit prints when it cannot make its shared memory at start (and the process then aborts)
KIT_NO_SHM = "Failed to create shared memory"


def _kit_did_not_start(logs: Path, name: str) -> bool:
    log = logs / f"{name.replace('/', '-')}.log"
    try:
        return KIT_NO_SHM in log.read_text(errors="replace")
    except OSError:
        return False


def summarize(out: Path) -> str:
    """The success table (task, version, level: scenes, successes) and the failures, from the reports in ``out``;
    also written to ``out/summary.md`` and ``summary.json``."""
    reps = []
    for f in sorted(out.glob("*.json")):
        if f.name == "summary.json":
            continue
        try:
            r = json.loads(f.read_text())
        except json.JSONDecodeError:
            continue
        if "scene" in r:
            reps.append(r)
    rows: dict = {}
    for r in reps:
        k = (r["task"], r["variant"], r["level"])
        n, ok, inb = rows.get(k, (0, 0, 0))
        # within the version's time budget (a report with no budget: counted as success alone says)
        rows[k] = (n + 1, ok + bool(r.get("success")), inb + bool(r.get("success_within_horizon", r.get("success"))))
    lines = ["| task | version | level | scenes | successes | rate | within budget |", "|---|---|---|---|---|---|---|"]
    for (t, v, lv), (n, ok, inb) in sorted(rows.items()):
        lines.append(f"| {t} | {v} | {lv} | {n} | {ok} | {ok / n:.0%} | {inb} |")
    fails = [r for r in reps if not r.get("success")]
    lines += ["", f"failures: {len(fails)} of {len(reps)}"]
    for r in fails:
        why = r.get("error") or "; ".join(r.get("problems", [])[:2]) or (
            f"milestones {r.get('milestones_hit', [])}" + (f", missed grips {r['missed_grips']}" if r.get("missed_grips")
                                                           else ""))
        lines.append(f"- {r['scene']}: {why}" + (f" (robots {r['robots']})" if r.get("robots") else ""))
    text = "\n".join(lines)
    (out / "summary.md").write_text(text + "\n")
    (out / "summary.json").write_text(json.dumps({"rows": [{"task": t, "variant": v, "level": lv, "scenes": n,
                                                            "successes": ok, "within_budget": inb} for (t, v, lv), (n, ok, inb) in
                                                           sorted(rows.items())],
                                                  "failures": [r["scene"] for r in fails]}, indent=1))
    return text
