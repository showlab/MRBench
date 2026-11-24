"""Measured clearance audit for physical reference recordings (CPU only).

Keep the ordinary raw minima in the report. This companion audit distinguishes
the existing cloth-pinching/table allowance from unintended contact; it uses
the same margins and action windows as policies.clearance.check.
"""

import math


def _cloth_windows(scene, script):
    from .policies.script import Grip, Release

    cloths = {c.name for c in scene.cloths}
    t, starts, events = 0.0, [], []
    for ph in script:
        starts.append(t)
        events.extend((t, a) for a in ph.actions if isinstance(a, (Grip, Release)))
        t += ph.seconds
    windows = []
    for when, act in events:
        if not isinstance(act, Grip) or act.target not in cloths:
            continue
        earlier = [s for s in starts if s < when - 1e-9]
        begin = earlier[-2] if len(earlier) >= 2 else 0.0
        end = next((at for at, a in events if at > when and isinstance(a, Release) and a.slot == act.slot), t)
        windows.append((act.slot, begin, end + 1.5))
    return windows


def audit_clearance(inst, frames, script):
    from .policies.clearance import ARM_GAP, ENV_GAP, HAND, PINCH_INTO_TABLE
    from .robots import collision as col, kinematics as kin

    scene = inst.scene
    windows = _cloth_windows(scene, script)
    arms = {a.slot: a for a in scene.arms}
    bases = {s: kin.transform(a.pos, kin.rpy_matrix(0, 0, math.radians(a.yaw_deg))) for s, a in arms.items()}
    statics = [col.table_box(scene)] + [b for o in scene.objects if not o.dynamic for b in col.boxes_of(o, o.pose)]
    minima = {}

    def note(key, gap, need, t, first, second):
        if key not in minima or gap < minima[key]['gap_m']:
            minima[key] = dict(gap_m=float(gap), required_m=need, at_s=round(t, 4),
                               first=first, second=second)

    for f in frames:
        caps = {s: col.arm_capsules(a.model, f.arms[s]['q'], max(0, f.arms[s]['gripper']), bases[s])
                for s, a in arms.items()}
        slots = list(arms)
        for i, s in enumerate(slots):
            for other in slots[i + 1:]:
                gap, a, b = min((col.capsule_capsule(a, b), a.name, b.name) for a in caps[s] for b in caps[other])
                note(f'{s}~{other}', gap, ARM_GAP, f.t, f'{s}.{a}', f'{other}.{b}')
            pinching = any(slot == s and lo <= f.t <= hi for slot, lo, hi in windows)
            boxes = statics + [col.prop_box(p, f.props.get(p.name)) for p in scene.props]
            for cap in caps[s]:
                for box in boxes:
                    if cap.on_table and box.name == 'table':
                        continue
                    allowed = pinching and cap.name in HAND and box.name == 'table'
                    key = f'{s}~cloth_table' if allowed else f'{s}~scene'
                    note(key, col.capsule_box(cap, box), -PINCH_INTO_TABLE if allowed else ENV_GAP,
                         f.t, f'{s}.{cap.name}', box.name)
    passed = all(v['gap_m'] >= v['required_m'] for v in minima.values())
    return dict(passed=passed, policy='Unchanged preflight margins; cloth hand/table contact only in its action window',
                minima=minima, cloth_contact_windows=[dict(arm=s, begin_s=lo, end_s=hi) for s, lo, hi in windows])
