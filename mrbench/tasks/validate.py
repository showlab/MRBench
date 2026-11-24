"""Design rules every task card / impl must satisfy.  ``mrbench validate`` and the
test-suite both run these, so a rule that is not checked here does not exist."""

from __future__ import annotations

import itertools
import math
import re

from ..core.errors import MRBenchError
from ..robots import MODELS, get_model
from .instance import TaskInstance, build
from .layouts import MOUNTS
from .loader import all_specs
from .spec import BENCHMARK_SUITES, LEVELS, MIN_ARMS_PER_TASK, TaskSpec

#: Usable fraction of the nominal reach for things an arm must reach.
REACH_MARGIN = 0.9
#: How far a hinge's spring may be wound anywhere in its travel (preload + travel, degrees).  Isaac drives a hinge
#: toward its spring's target, ``preload_deg`` past shut, the shorter way round: wound 180 or more, the spring turns
#: the body the other way: a pedal wound 168 + 14 stays pressed down when let go.  The margin is for a body pushed a
#: little past its stop.
HINGE_WOUND_MAX_DEG = 170.0
#: how far two bodies (one of them movable) may start inside each other: their collision boxes are measured, and a
#: sphere is measured as its box (Rea-02's tomato on its sloping tray "overlaps" 1-2 mm)
OVERLAP_TOL = 0.003
CHECK_SEEDS = (0, 1, 2)
#: An instruction gives only the goal: it names no arm and no robot (who does what is the agent's to work out).
#: Places on the table may still be named by the side they are on ("the far west end").
ROLE_WORDS = re.compile(r"\b(arms?|hands?|franka|panda|piper|so-?101|robots?)\b", re.IGNORECASE)


def check_card(spec: TaskSpec) -> list[str]:
    p = []
    if len(spec.arms) < MIN_ARMS_PER_TASK:
        p.append(f"{spec.id}: {len(spec.arms)} arms, need >= {MIN_ARMS_PER_TASK}")
    slots = [a.slot for a in spec.arms]
    if len(set(slots)) != len(slots):
        p.append(f"{spec.id}: duplicate arm slots {slots}")
    mounts = [a.mount for a in spec.arms]
    if len(set(mounts)) != len(mounts):
        p.append(f"{spec.id}: two arms on one mount {mounts}")
    p += [f"{spec.id}: no mount {m!r} (the mounts are {', '.join(MOUNTS)})" for m in mounts if m not in MOUNTS]
    p += [f"{spec.id}: unknown robot model {a.model!r}" for a in spec.arms if a.model not in MODELS]
    if len(set(spec.milestones)) != len(spec.milestones):
        p.append(f"{spec.id}: duplicate milestone ids")
    if spec.suite in BENCHMARK_SUITES and spec.borrow is None:
        p.append(f"{spec.id}: a {spec.suite} task needs a borrow design (which job a prop can do instead of an arm)")
    return p


def check_instance(inst: TaskInstance) -> list[str]:
    from ..runner.episode import replay_keyframes, run_episode  # runner imports tasks; keep lazy

    tag = f"{inst.spec.id}/{inst.level}/s{inst.seed}"
    p = []
    names = [m.name for m in inst.milestones]
    if tuple(names) != inst.spec.milestones:
        p.append(f"{tag}: impl milestones {names} != card {list(inst.spec.milestones)}")
    gone = inst.spec.borrow.remove if inst.variant == "borrow" else None
    if [a.slot for a in inst.scene.arms] != [a.slot for a in inst.spec.arms if a.slot != gone]:
        p.append(f"{tag}: scene arms do not match the card" + (f" (less {gone}, for its borrow version)" if gone else ""))
    if inst.variant == "borrow":
        b = inst.metadata.get("borrow") or {}
        names = {o.name for o in inst.scene.objects}
        props, decoys = list(b.get("props", ())), list(b.get("decoys", ()))
        if not props or len(props) != len(decoys):
            p.append(f"{tag}: metadata['borrow'] must name its props and as many distractors")
        p += [f"{tag}: borrow names {n}, which is not in the scene" for n in props + decoys if n not in names]
        exact = build(inst.spec.id, inst.level, inst.seed)
        if inst.instruction != exact.instruction:
            p.append(f"{tag}: the borrow version's instruction differs from the exact one's")
    named = sorted({m.group(0).lower() for m in ROLE_WORDS.finditer(inst.instruction)})
    if named:
        p.append(f"{tag}: the instruction names {', '.join(named)}; it must give only the goal")

    sx, sy = inst.scene.table_size[0] / 2, inst.scene.table_size[1] / 2
    for o in inst.scene.objects:
        if abs(o.pos[0]) > sx or abs(o.pos[1]) > sy:
            p.append(f"{tag}: object {o.name} starts off the table at {o.pos[:2]}")
    for c in inst.scene.cloths:
        if any(abs(x) > sx or abs(y) > sy for x, y, _ in c.flat_points()):
            p.append(f"{tag}: cloth {c.name} does not lie wholly on the table")
    dynamic = {o.name for o in inst.scene.dynamic_objects}
    for c in inst.scene.cloths:
        p += [f"{tag}: cloth {c.name} is tied to {t.body}, which is not a dynamic body" for t in c.ties
              if t.body not in dynamic]
    for h in inst.scene.hinges:
        if not inst.scene.obj(h.body).dynamic:
            p.append(f"{tag}: hinge {h.name} turns {h.body}, which is static")
        wound = h.preload_deg + h.open_max_deg
        if h.stiffness > 0 and wound >= HINGE_WOUND_MAX_DEG:
            p.append(f"{tag}: hinge {h.name}'s spring is wound {wound:.0f} degrees at its stop (preload "
                     f"{h.preload_deg:.0f} + travel {h.open_max_deg:.0f}); under {HINGE_WOUND_MAX_DEG:.0f}, or Isaac "
                     f"turns the body the other way past 180")
    # no body starts inside another: two bodies that overlap are pushed apart by the physics as it starts, and may
    # move together (two wedges whose crossbars overlap 2 cm do, in Isaac)
    from ..robots import collision as col

    for o1, o2 in itertools.combinations(inst.scene.objects, 2):
        if o1.dynamic or o2.dynamic:
            gap = min(col.box_box(a, b) for a in col.boxes_of(o1, o1.pose) for b in col.boxes_of(o2, o2.pose))
            if gap < -OVERLAP_TOL:
                p.append(f"{tag}: {o1.name} and {o2.name} start {-100 * gap:.1f} cm inside each other")

    def dist(slot: str, target: str) -> tuple[float, float]:
        a = inst.scene.arm(slot)
        pt = inst.scene.point(target)
        return math.dist(a.pos[:2], pt[:2]), get_model(a.model).reach_m

    for slot, target in inst.reach:
        d, r = dist(slot, target)
        if d > REACH_MARGIN * r:
            p.append(f"{tag}: {slot} must reach {target} but it is {d:.2f} m away (usable {REACH_MARGIN * r:.2f} m)")
    for slot, target in inst.out_of_reach:
        d, r = dist(slot, target)
        if d <= r:
            p.append(f"{tag}: {slot} must NOT reach {target} but it is only {d:.2f} m away (reach {r:.2f} m)")
    for a, b in inst.apart:
        pa, pb = inst.scene.arm(a), inst.scene.arm(b)
        d, r = math.dist(pa.pos[:2], pb.pos[:2]), get_model(pa.model).reach_m + get_model(pb.model).reach_m
        if d <= r:
            p.append(f"{tag}: {a} and {b} must not meet but their reaches overlap ({d:.2f} m apart, {r:.2f} m)")

    try:
        from ..engines.mock import MockEngine
        from ..policies import make_policy

        base = run_episode(inst, MockEngine(), make_policy("do_nothing"), horizon_s=2.0)
        if base.success:
            p.append(f"{tag}: succeeds without doing anything")
        if base.signals_hit:  # (``tasks/signals.py``: a job's failure shows only once it was tried)
            p.append(f"{tag}: doing nothing shows failure signals {base.signals_hit}")
        oracle = replay_keyframes(inst, MockEngine())
        if not oracle.success or oracle.progress < 1.0:
            p.append(f"{tag}: oracle keyframes do not solve the task "
                     f"(success={oracle.success}, milestones={oracle.milestones_hit})")
        if oracle.signals_hit:
            p.append(f"{tag}: the oracle keyframes show failure signals {oracle.signals_hit}")
    except MRBenchError as e:
        p.append(f"{tag}: {e}")
    return p


def check_props(inst: TaskInstance) -> list[str]:
    """Props, which no named look brings but ``clutter=`` puts on: every set's, on this task, each where
    ``spot_ok`` lets it stand -- out of every arm's reach, clear of the task, on the table -- no two
    overlapping, and the task unchanged by them."""
    from ..scenery import CLUTTER, dress, parse_look
    from ..scenery.props import footprint, overlap, spot_ok

    tag = f"{inst.spec.id}/{inst.level}/s{inst.seed}"
    p = []
    for name in CLUTTER:
        look = parse_look(f"default,clutter={name}:2")
        dressed = dress(inst, look)
        if dressed.scene.objects != inst.scene.objects or dressed.scene.regions != inst.scene.regions:
            p.append(f"{tag}: the {name} props changed the task's bodies or regions")
        props = dressed.scene.props
        if not props:
            p.append(f"{tag}: no room for any {name} group")
        boxes = [footprint(pr.kind, pr.pos[0], pr.pos[1], pr.yaw_deg) for pr in props]
        for i, pr in enumerate(props):
            if not spot_ok(inst.scene, pr.kind, pr.pos[0], pr.pos[1], pr.yaw_deg):
                p.append(f"{tag}: the {name} props stand {pr.kind} where it may not ({pr.pos[:2]})")
            p += [f"{tag}: the {name} props stand {pr.kind} on {props[j].kind}" for j in range(i)
                  if overlap(boxes[i], boxes[j])]
    return p


def validate(task_ids=None, variant: str = "exact") -> list[str]:
    """Every design rule, on the tasks' ``variant`` configuration: of the tasks named, one that is not built in it is
    a problem; of all tasks, those without it are left out."""
    specs = all_specs()
    problems: list[str] = []
    for tid, spec in specs.items():
        if task_ids and tid not in task_ids:
            continue
        problems += check_card(spec)
        if spec.status == "planned":
            continue
        if variant not in spec.variants:
            if task_ids:
                problems.append(f"{tid}: no {variant} version yet (built: {', '.join(spec.variants)})")
            continue
        for level in LEVELS:
            for seed in CHECK_SEEDS:
                try:
                    inst = build(tid, level, seed, variant=variant)
                    problems += check_instance(inst)
                    if seed == CHECK_SEEDS[0]:
                        problems += check_props(inst)
                except MRBenchError as e:
                    problems.append(f"{tid}/{level}/s{seed}: build failed: {e}")
    return problems
