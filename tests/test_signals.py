"""The jobs' own failure signals (``mrbench/tasks/signals.py``): their names are the tasks' jobs, a run in which
nothing happens or the task is done shows none, and a run in which a job is tried and fails shows its signal -- in
the score and in the failure report.  Synthetic frames only (no engine)."""

import importlib
import math
from types import SimpleNamespace

import pytest

from mrbench import tasks
from mrbench.agent import failure
from mrbench.engines.base import Frame, ObjectSpec, RegionSpec, SceneSpec, pose7
from mrbench.scoring.predicates import InRegion, Lifted, Then
from mrbench.scoring.scorer import score
from mrbench.tasks import signals

BENCHMARK = ("Con-01", "Con-02", "Con-03", "Con-04", "Rea-01", "Rea-02", "Rea-03", "Rea-04")


def qmul(a, b):
    aw, ax, ay, az = a
    bw, bx, by, bz = b
    return (aw * bw - ax * bx - ay * by - az * bz, aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx, aw * bz + ax * by - ay * bx + az * bw)


def turned(pose, deg, axis=(1.0, 0.0, 0.0)):
    """``pose`` turned by ``deg`` about ``axis`` (a hinge's angle is all ``Opened`` reads)."""
    h = math.radians(deg) / 2
    return (*pose[:3], *qmul(pose[3:], (math.cos(h), *(math.sin(h) * a for a in axis))))


def moved(pose, xyz):
    return (*xyz, *pose[3:])


def run(inst, *states, dt=0.25, hold=4):
    """Frames: each state (body -> pose, over the start) held for ``hold`` frames ``dt`` apart."""
    out, t = [], 0.0
    for st in states:
        for _ in range(hold):
            out.append(Frame(t=t, objects={**{o.name: o.pose for o in inst.scene.objects if o.dynamic}, **st}))
            t += dt
    return out


def shown(inst, frames):
    return score(inst, frames).signals_hit


def test_then_counts_only_after_a():
    scene = SceneSpec(arms=(), objects=(ObjectSpec("a", "box", (0.04,) * 3, (0, 0, 0.02)),),
                      regions=(RegionSpec("goal", (0.5, 0.0, 0.02), (0.05, 0.05, 0.05)),))
    home, there, up = (0, 0, 0.02), (0.5, 0, 0.02), (0, 0, 0.3)

    def fs(*ps):
        return [Frame(t=i * 0.5, objects={"a": pose7(p)}) for i, p in enumerate(ps)]

    lifted, back = Lifted("a", 0.1), ~Lifted("a", 0.1) & ~InRegion("a", "goal")
    assert Then(lifted, back).first(fs(home, home, home), scene) is None  # at home from the start: not "back"
    assert Then(lifted, back).first(fs(home, up, home), scene) == 2
    assert Then(lifted, back).first(fs(home, up, there), scene) is None
    assert Then(lifted, back, 1.0).first(fs(home, up, home, up, home), scene) is None  # back only 0.5 s at a time
    assert Then(lifted, back, 1.0).first(fs(home, up, home, home, home), scene) == 4


@pytest.mark.parametrize("task", BENCHMARK)
def test_every_signal_belongs_to_jobs_of_its_task_and_a_still_run_shows_none(task):
    for variant in ("exact", "borrow"):
        for level in ("easy", "hard"):
            inst = tasks.build(task, level, 0, variant=variant)
            J = importlib.import_module(inst.spec.impl).jobs(inst)
            names = {j.name for j in J.jobs} | {p.name for t in J.things for p in t.placing}
            sig = signals.of(inst)
            assert sig and signals.declared(inst)
            for s in sig:
                assert s.jobs and set(s.jobs) <= names, (task, variant, s.jobs)
            assert len({s.name for s in sig}) == len(sig)
            assert shown(inst, run(inst, {}, hold=20)) == []  # nothing moves: nothing shows


def test_con01_a_door_that_swings_shut_on_the_box():
    inst = tasks.build("Con-01", "easy", 0)
    box, de = inst.scene.obj("box").pose, inst.scene.obj("door_e").pose
    mat = next(r for r in inst.scene.regions if r.name == "mat").center
    assert InRegion("box", "cabinet_in")(run(inst, {})[0], inst.scene)
    open_e, open_w = turned(de, 85.0, (0, 0, 1)), turned(inst.scene.obj("door_w").pose, 85.0, (0, 0, -1))
    # tried and let go while the box is inside
    failed = shown(inst, run(inst, {}, {"door_e": open_e}, {"door_e": de}))
    assert failed == ["hold the east door open by its knob: it swung shut with the box still inside"]
    # the task done: both doors open, the box out onto the mat, the doors let shut -- nothing
    done = run(inst, {}, {"door_e": open_e, "door_w": open_w}, {"door_e": open_e, "door_w": open_w,
                                                                 "box": moved(box, mat)}, {"box": moved(box, mat)})
    assert shown(inst, done) == []


def test_con04_a_valve_let_open_while_air_came_in_and_a_pedal_let_up():
    from mrbench.tasks.impl import con04_pressure_test as M

    inst = tasks.build("Con-04", "easy", 0)
    shut = {v: M._turned(inst, v, M.VALVE_HOLD) for v in ("valve_w", "valve_e")}
    air = {"pedal": M._turned(inst, "pedal", M.PEDAL_HOLD)}
    # both valves shut, then the west one let go as its hand goes to the pedal and presses it down
    failed = score(inst, run(inst, {}, shut, {"valve_e": shut["valve_e"]}, {"valve_e": shut["valve_e"], **air}))
    assert failed.signals_hit == ["shut the west bleed valve: hold its lever down by its grip: once shut, it opened "
                                  "again while air came in, before the pipe held its pressure"]
    assert failed.signals_first_s
    # the pedal pressed, then let up as both valves are shut
    assert shown(inst, run(inst, {}, air, shut, shut)) == [
        "let air in: hold the foot valve's pedal down by its toe bar: once down, it rose and shut while both bleed "
        "valves were shut, before the pipe held its pressure"]
    # the pressure held for the test, then the valves let go before the pedal: nothing
    assert shown(inst, run(inst, {}, {**shut, **air}, {**shut, **air}, air, {})) == []


def test_con04_borrow_a_weight_in_the_pocket_that_leaves_the_pedal_up():
    from mrbench.tasks.impl import con04_pressure_test as M

    inst = tasks.build("Con-04", "easy", 0, variant="borrow")
    shut = {v: M._turned(inst, v, M.VALVE_HOLD) for v in ("valve_w", "valve_e")}
    b = inst.metadata["borrow"]
    decoy, prop = b["decoys"][0], b["props"][0]

    def set_in(name, deg=0.0):
        h = inst.scene.obj(name).part("body").size[1]
        return (*M.pocket(inst, deg, h / 2), *M._turned(inst, "pedal", deg)[3:])

    assert shown(inst, run(inst, {decoy: set_in(decoy)}, {decoy: set_in(decoy), **shut})) == [
        "let air in: hold the foot valve's pedal down by its toe bar: a weight set in its pocket left it up while both "
        "bleed valves were shut"]
    # the prop takes the pedal down with it, the valves shut: the pressure held, nothing
    down = {"pedal": M._turned(inst, "pedal", M.REST_DEG), prop: set_in(prop, M.REST_DEG)}
    assert shown(inst, run(inst, {prop: set_in(prop)}, down, {**down, **shut}, {**down, **shut})) == []


def test_rea01_and_rea02_things_off_the_table_or_the_track():
    rea01 = tasks.build("Rea-01", "hard", 0)
    piece = rea01.metadata["fruit"][1]
    p = rea01.scene.obj(piece).pose
    assert shown(rea01, run(rea01, {piece: moved(p, (p[0], p[1], 0.05))})) == [
        f"carry {piece} from the start area into the crate: it fell off the table"]
    rea02 = tasks.build("Rea-02", "easy", 0)
    t = rea02.scene.obj("tomato").pose
    assert shown(rea02, run(rea02, {"tomato": moved(t, (t[0] + 0.3, t[1], 0.80))})) == [
        "laying the track: the tomato fell off the track"]


def test_the_failure_report_lists_the_signals_that_showed():
    inst = tasks.build("Con-01", "easy", 0)
    de = inst.scene.obj("door_e").pose
    sc = score(inst, run(inst, {}, {"door_e": turned(de, 60.0, (0, 0, 1))}, {"door_e": de}))
    case = SimpleNamespace(inst=inst)
    rep = failure.report(case, {"played": True, "success": False, **sc.to_json()})
    assert rep["job_failure_signals"] == sc.signals_hit and sc.signals_hit
    old = failure.report(case, {"played": True, "success": False, "milestones_hit": []})  # a score with no signals
    assert old["job_failure_signals"] == "none known for this task or this run"


def test_con02_a_leaf_that_drops_before_the_pin_is_through():
    inst = tasks.build("Con-02", "easy", 0)
    lw = inst.scene.obj("leaf_w").pose
    level, dropped = (*lw[:3], 1.0, 0.0, 0.0, 0.0), turned((*lw[:3], 1.0, 0.0, 0.0, 0.0), 20.0, (0, 1, 0))
    assert shown(inst, run(inst, {"leaf_w": level}, {"leaf_w": dropped})) == [
        "hold the blue leaf level by its bar: once level, it dropped before the pin was through"]


def test_con03_the_lid_shut_on_the_ball_and_the_ball_dropped():
    from mrbench.tasks.impl import con03_lid_ball as M

    inst = tasks.build("Con-03", "easy", 0)
    lid, ball = inst.scene.obj("lid").pose, inst.scene.obj("ball").pose
    assert shown(inst, run(inst, {}, {"lid": turned(lid, 40.0)}, {"lid": lid})) == [
        "hold the lid open by its tab: it fell shut with the ball still in the box"]
    up = moved(ball, (ball[0], ball[1], ball[2] + M.WALL_H + 0.05))
    on_table = moved(ball, (ball[0] + 0.8, ball[1], ball[2] - 0.05))
    assert shown(inst, run(inst, {"ball": up}, {"ball": on_table})) == [
        "lifting the ball (both sides): once lifted over the wall, the ball fell back outside the cradle"]


def test_rea03_a_run_that_stops_at_the_middle_stretch():
    from mrbench.engines.base import quat_rotate
    from mrbench.tasks.impl import rea03_domino_run as M

    for variant, near in (("exact", 10), ("borrow", 9)):
        inst = tasks.build("Rea-03", "easy", 0, variant=variant)
        spacing = inst.metadata["spacing_m"]

        def fallen(i):  # domino_i tipped 60 degrees east about its foot, on its place
            name = f"domino_{i:02d}"
            q = turned((0, 0, 0, *inst.scene.obj(name).pose[3:]), 60.0, (0, 1, 0))[3:]
            up = quat_rotate(q, (0.0, 0.0, inst.scene.obj(name).height / 2))
            foot = (M.place_x(i, spacing), M.LINE_Y, M.Z)
            return {name: (*(foot[k] + up[k] for k in range(3)), *q)}

        stopped = run(inst, {}, fallen(near), hold=12)  # 3 s and domino_14 still up
        assert shown(inst, stopped) == ["filling the middle stretch: the run reached the middle stretch and stopped there"]
        through = run(inst, {}, fallen(near), {**fallen(near), **fallen(14)}, hold=2)
        assert shown(inst, through) == []


def test_rea04_the_sponge_off_the_counter():
    inst = tasks.build("Rea-04", "easy", 0)
    s = inst.scene.obj("sponge").pose
    assert shown(inst, run(inst, {"sponge": moved(s, (s[0], s[1] - 0.6, s[2] - 0.3))})) == [
        "passing the sponge on: the sponge fell off the counter"]


def test_left_at_counts_only_a_rest_that_lasts_to_the_end():
    from mrbench.scoring.predicates import LeftAt

    scene = SceneSpec(arms=(), objects=(ObjectSpec("a", "box", (0.04,) * 3, (0, 0, 0.02)),),
                      regions=(RegionSpec("pad", (0.5, 0.0, 0.02), (0.05, 0.05, 0.05)),))
    home, pad, pad2, away = (0, 0, 0.02), (0.5, 0, 0.02), (0.53, 0, 0.02), (1.0, 0, 0.02)

    def fs(*ps):
        return [Frame(t=i * 0.5, objects={"a": pose7(p)}) for i, p in enumerate(ps)]

    left = LeftAt("a", InRegion("a", "pad"), 2.0)
    assert left.first(fs(home, pad, pad, pad, pad, pad), scene) == 5  # on the pad from 0.5 s to the end (2.5 s)
    assert left.first(fs(home, pad, pad, pad), scene) is None  # only 1 s there
    assert left.first(fs(home, pad, pad, pad, pad, pad, away), scene) is None  # taken on at the end
    assert left.first(fs(home, pad, pad, pad2, pad2, pad2, pad2, pad2), scene) == 7  # moved on the pad: the last rest
    assert left.first(fs(home, home, home, home, home), scene) is None  # never on the pad


def test_con02_the_pin_pushed_in_while_a_leaf_hangs():
    inst = tasks.build("Con-02", "easy", 0)
    M = importlib.import_module(inst.spec.impl)
    level = {"leaf_w": M.level_pose(inst, "leaf_w"), "leaf_e": M.level_pose(inst, "leaf_e")}
    pin_home = M.keyframes(inst)[-1]["pin"]
    # the orange leaf held level, the blue one hanging (nothing holds it, or a wedge too low): the pin in the orange
    # leaf's cheeks, stopped at the blue one's slot
    hung = shown(inst, run(inst, {"leaf_e": level["leaf_e"]}, {"leaf_e": level["leaf_e"], "pin": pin_home}))
    assert hung == ["hold the blue leaf level by its bar: the pin was pushed in while it hung out of line"]
    assert shown(inst, run(inst, level, {**level, "pin": pin_home})) == []  # both level, the pin home: nothing


def test_rea01_a_piece_left_on_a_pad_or_in_the_lane():
    from mrbench.tasks.impl import rea01_relay as M

    inst = tasks.build("Rea-01", "easy", 0)
    piece = inst.metadata["fruit"][0]
    p = inst.scene.obj(piece).pose
    pad_a = inst.scene.region("pad_a").center
    left = f"carry {piece} from the start area into the crate: it was left on its way, on a hand-over pad or in the " \
           "lane through the covered channel"
    assert shown(inst, run(inst, {}, {piece: moved(p, pad_a)}, hold=16)) == [left]
    in_channel = (0.0, M.LANE_Y, M.Z + M.FRUIT_Z)
    assert shown(inst, run(inst, {}, {piece: moved(p, in_channel)}, hold=16)) == [left]
    # short of the channel's west mouth (x -0.195), where a board too short leaves it
    short = (-0.248, M.LANE_Y, M.Z + M.FRUIT_Z)
    assert shown(inst, run(inst, {}, {piece: moved(p, short)}, hold=16)) == [left]
    crate = inst.scene.region("bin_in").center
    assert shown(inst, run(inst, {piece: moved(p, pad_a)}, {piece: moved(p, crate)}, hold=16)) == []  # taken on


def test_rea04_the_sponge_left_on_a_pad():
    inst = tasks.build("Rea-04", "easy", 0)
    s = inst.scene.obj("sponge").pose
    pad_a, sink = inst.scene.region("pad_a").center, inst.scene.region("sink_in").center
    assert shown(inst, run(inst, {}, {"sponge": moved(s, pad_a)}, hold=16)) == [
        "passing the sponge on: it was left on a hand-over pad, never taken on"]
    assert shown(inst, run(inst, {"sponge": moved(s, pad_a)}, {"sponge": moved(s, sink)}, hold=16)) == []
