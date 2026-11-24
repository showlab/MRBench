import math

import pytest

from mrbench.engines.base import Frame, ObjectSpec, RegionSpec, SceneSpec, pose7
from mrbench.scoring.predicates import (Always, Before, Eventually, HoldsAtEnd, HoldsFor, InRegion, Lifted, OnTop,
                                        Upright, tilt_deg)

SCENE = SceneSpec(arms=(), objects=(ObjectSpec("a", "box", (0.04,) * 3, (0, 0, 0.02)),
                                    ObjectSpec("b", "box", (0.04,) * 3, (0.5, 0, 0.02))),
                  regions=(RegionSpec("goal", (0.5, 0.0, 0.05), (0.05, 0.05, 0.05)),))


def frames(*poses, dt=0.5):
    return [Frame(t=i * dt, objects={"a": pose7(p), "b": pose7((0.5, 0, 0.02))}) for i, p in enumerate(poses)]


def test_region_lifted_ontop():
    f = frames((0.5, 0.0, 0.06))[0]
    assert InRegion("a", "goal")(f, SCENE)
    assert Lifted("a", 0.03)(f, SCENE) and not Lifted("a", 0.05)(f, SCENE)
    assert OnTop("a", "b")(f, SCENE)
    assert not OnTop("b", "a")(f, SCENE)


def test_holds_for_needs_an_uninterrupted_window():
    on, off = (0.5, 0, 0.02), (0, 0, 0.02)
    p = InRegion("a", "goal")
    assert HoldsFor(p, 1.0).first(frames(on, on, on), SCENE) == 2
    assert HoldsFor(p, 1.0).first(frames(on, on, off, on, on), SCENE) is None


def test_holds_at_end_counts_only_the_stretch_that_reaches_the_end():
    """HoldsAtEnd: there when the run ends, the last ``seconds`` at least; satisfied ``seconds`` into that stretch."""
    on, off = (0.5, 0, 0.02), (0, 0, 0.02)
    p = InRegion("a", "goal")
    end, held = HoldsAtEnd(p, 1.0), HoldsFor(p, 1.0)
    assert end.first(frames(off, on, on, on, on), SCENE) == 3  # there from 0.5 s to the end: 1 s of it at 1.5 s
    assert end.first(frames(off, on, on, on, off), SCENE) is None  # held there 1 s, then taken away
    assert held.first(frames(off, on, on, on, off), SCENE) == 3  # ... which HoldsFor passes
    assert end.first(frames(on, on, on, off, on, on), SCENE) is None  # back at the end, 0.5 s only
    assert end.first(frames(on, on, on, off, on, on, on), SCENE) == 6  # left and come back: the last stretch counts
    assert end.first(frames(on, on), SCENE) is None and end.first([], SCENE) is None  # a run too short; no run
    assert HoldsAtEnd(p).first(frames(off, off, on), SCENE) == 2  # no time asked: there in the last frame
    # it implies HoldsFor and is never satisfied before it; at the same frame where the thing is never left again
    for run in (frames(off, on, on, on, on), frames(on, on, on, off, on, on, on), frames(on, on, on, on)):
        assert held.first(run, SCENE) <= end.first(run, SCENE)
    assert held.first(frames(on, on, on, on), SCENE) == end.first(frames(on, on, on, on), SCENE) == 2


def test_con03_success_needs_the_ball_in_the_cradle_when_the_run_ends():
    """The ball held in the cradle a second and then carried off, or let go and rolled out, reaches every
    milestone and does not succeed; left there, it succeeds at the moment ``ball_settled`` is reached."""
    from mrbench.scoring.scorer import score
    from mrbench.tasks import build
    from mrbench.tasks.impl import con03_lid_ball as m

    for variant in ("exact", "borrow"):
        inst = build("Con-03", "easy", 0, variant=variant)
        sc = inst.scene
        ball = sc.obj("ball").pose
        opened = m._opened(sc, m.OPEN_DEG)
        over_box = (ball[0], ball[1], ball[2] + m.WALL_H + 0.05, *ball[3:])
        cradle = (*sc.region("cradle_in").center, *ball[3:])
        beside = (cradle[0] + 0.30, cradle[1], sc.table_top_z + m.BALL_R, *ball[3:])  # on the table east of the cradle

        def run(*states, dt=0.1, hold=15):  # each state (body -> pose) for 1.5 s
            return [Frame(t=(k * hold + i) * dt, objects=st) for k, st in enumerate(states) for i in range(hold)]

        out = [{"lid": opened}, {"lid": opened, "ball": over_box}, {"lid": opened, "ball": cradle}]
        every = ["lid_open", "ball_out", "ball_in_cradle", "ball_settled"]
        left = score(inst, run(*out, {"ball": cradle}))
        assert left.success and left.milestones_hit == every
        assert left.first_hit_s["success"] == left.first_hit_s["ball_settled"] == 4.0  # in the cradle from 3.0 s
        for after in ({"ball": over_box}, {"ball": beside}):  # carried off again; rolled out onto the table
            gone = score(inst, run(*out, after))
            assert not gone.success and "success" not in gone.first_hit_s
            assert gone.milestones_hit == every and gone.first_hit_s["ball_settled"] == 4.0
        back = score(inst, run(*out, {"ball": over_box}, {"ball": cradle}))  # taken out and set back: the last stay
        assert back.success and back.first_hit_s["success"] == 7.0 and back.first_hit_s["ball_settled"] == 4.0
        short = score(inst, run(*out, {"ball": over_box}, {"ball": cradle}, hold=15)[:-8])  # back 0.7 s before the end
        assert not short.success


def test_before_and_always():
    fs = frames((0, 0, 0.02), (0, 0, 0.2), (0.5, 0, 0.02))
    lifted, placed = Eventually(Lifted("a", 0.1)), Eventually(InRegion("a", "goal"))
    assert Before(lifted, placed)(fs, SCENE)
    assert not Before(placed, lifted)(fs, SCENE)
    assert Always(Upright("a", 1.0))(fs, SCENE)


def test_tilt():
    assert tilt_deg(pose7((0, 0, 0), 90)[3:]) == pytest.approx(0.0, abs=1e-6)
    assert tilt_deg((0.7071068, 0.7071068, 0.0, 0.0)) == pytest.approx(90.0, abs=1e-4)


def test_a_cord_over_a_peg_and_its_ties():
    """ClothOver: a cord on a peg's top, hanging down both sides; a tie holds a column of it, or one point."""
    from mrbench.engines.base import ClothSpec, ClothTie, Part
    from mrbench.scoring.predicates import ClothOver

    peg = ObjectSpec("hook", "compound", (0.012, 0.08, 0.012), (0.0, 0.0, 0.3), dynamic=False,
                     parts=(Part("peg", "box", (0.012, 0.08, 0.012), (0.0, 0.0, 0.0)),))
    cord = ClothSpec("cord", (1.0, 0.02), (0.0, 0.0, 0.005), ties=(ClothTie("a", 0.0), ClothTie("b", 0.5, -0.01)))
    assert cord.tie_nodes(cord.ties[0]) == (0, 1, 2) and len(cord.tie_nodes(cord.ties[1])) == 1
    scene = SceneSpec(arms=(), objects=(peg, *SCENE.objects), cloths=(cord,))

    def over(points):
        return ClothOver("cord", "hook", "peg")(Frame(t=0.0, objects={}, cloths={"cord": [points[i] for i in cord.marks]}),
                                                scene)

    top = 0.306 + cord.radius
    hung = [(u, v, top - 0.5 * abs(u)) for u, v in cord.nodes]  # down both sides of the peg at x = 0
    assert over(hung)
    assert not over([(u + 0.3, v, z) for u, v, z in hung])  # its top beside the peg
    assert not over([(u, v, top - 0.5 * abs(u) - 0.1) for u, v in cord.nodes])  # all of it below the peg
    assert not over([(u, v, top - 0.5 * max(u, 0.0)) for u, v in cord.nodes])  # lying on it, west half level


def test_a_domino_line_whole_when_set_off():
    """ChainStands: standing dominoes each within falling reach of the next, from the first place to the last; a
    place left empty breaks it, a tall plank across it does not.  AtOnset: it held just before the first fell."""
    from mrbench.engines.base import axis_quat
    from mrbench.scoring.predicates import AtOnset, ChainStands, DownOnLine

    dom = (0.02, 0.06, 0.12)
    objs = [ObjectSpec(f"d{i}", "box", dom, (i * 0.08, 0.0, 0.06)) for i in range(6)]
    objs.append(ObjectSpec("plank", "box", (0.02, 0.06, 0.30), (9.0, 0.0, 0.15)))
    sc = SceneSpec(arms=(), objects=tuple(objs))
    names = tuple(o.name for o in objs)
    up = lambda x, h=0.06: pose7((x, 0.0, h))  # noqa: E731
    fallen = (0.0, 0.0, 0.03, *axis_quat((0.0, 1.0, 0.0), 30.0))
    whole = ChainStands(names, 0.0, 0.0, 0.40)
    line = {f"d{i}": up(i * 0.08) for i in range(6)}
    assert whole(Frame(t=0.0, objects={**line, "plank": up(9.0, 0.15)}), sc)
    gap = {**line, "d2": up(5.0)}  # one place left empty
    assert not whole(Frame(t=0.0, objects={**gap, "plank": up(9.0, 0.15)}), sc)
    bridged = {**gap, "d1": up(5.2), "plank": up(0.08, 0.15)}  # a tall plank where d1 stood falls across
    assert whole(Frame(t=0.0, objects=bridged), sc)
    # the first one leaning 14 degrees as it is pushed, its foot slid 1.2 cm: its middle 2.65 cm on from its place (more
    # than end_tol), its foot 1.2 cm (as when Rea-03's borrow reference pushes it in Isaac)
    def leaning(foot_x):
        q = axis_quat((0.0, 1.0, 0.0), 14.0)
        s, c = math.sin(math.radians(14.0)), math.cos(math.radians(14.0))
        return (foot_x + 0.06 * s, 0.0, 0.06 * c, *q)
    assert whole(Frame(t=0.0, objects={**line, "d0": leaning(0.012), "plank": up(9.0, 0.15)}), sc)
    assert not whole(Frame(t=0.0, objects={**line, "d0": leaning(0.03), "plank": up(9.0, 0.15)}), sc)  # slid off
    down = DownOnLine(names, 0.0, 0.0, 0.40)
    frames = [Frame(t=0.0, objects={**line, "plank": up(9.0, 0.15)}),
              Frame(t=0.5, objects={**{k: (v[0], *fallen[1:]) for k, v in line.items()}, "plank": up(9.0, 0.15)})]
    assert down(frames[1], sc) and not down(frames[0], sc)
    assert AtOnset(whole, ~Upright("d0", 15.0)).first(frames, sc) == 1
    frames[0] = Frame(t=0.0, objects={**gap, "plank": up(9.0, 0.15)})
    assert AtOnset(whole, ~Upright("d0", 15.0)).first(frames, sc) is None


def test_a_domino_line_is_set_off_by_the_domino_on_its_first_place():
    """FallenAt: a line's first domino is whichever stands on its first place.  Rea-03: domino_00 knocked where it
    waits, off the line, has not set the line off; stood in place and pushed with the line whole, it has; with the
    middle left empty, or the line set off from its east end, the run fails."""
    from mrbench.engines.base import axis_quat
    from mrbench.scoring import score
    from mrbench.scoring.predicates import FallenAt
    from mrbench.tasks import build
    from mrbench.tasks.impl import rea03_domino_run as R

    sc = SceneSpec(arms=(), objects=(ObjectSpec("d", "box", R.DOMINO, R.standing(0.0)[:3]),))
    at_first = FallenAt(("d",), (0.0, R.LINE_Y, R.Z), (0.04, 0.03, 0.02))
    held = R.fallen(0.0, 20.0)
    held = (*held[:2], held[2] + 0.10, *held[3:])

    def fell(pose):
        return at_first(Frame(t=0.0, objects={"d": pose}), sc)

    assert not fell(R.standing(0.0)) and not fell(R.fallen(0.0, 10.0))
    assert fell(R.fallen(0.0, 20.0)) and fell(R.fallen(0.0, 90.0))  # tipped east about its foot's east edge
    assert not fell(R.fallen(0.5, 20.0))  # fallen elsewhere
    assert not fell(held)  # held tilted over the place

    inst = build("Rea-03", "hard", 3)
    s, n = inst.metadata["spacing_m"], R.N
    start = {o.name: o.pose for o in inst.scene.dynamic_objects}
    w = start["domino_00"]
    knocked = {**start, "domino_00": (*w[:3], *axis_quat((1.0, 0.0, 0.0), 18.0))}
    line = {f"domino_{i:02d}": R.standing(R.place_x(i, s)) for i in range(n)}
    pushed = {**line, "domino_00": R.fallen(R.place_x(0, s), 10.0)}
    first = {**line, "domino_00": R.fallen(R.place_x(0, s), 30.0)}
    down = {f"domino_{i:02d}": R.fallen(R.place_x(i, s), 30.0 if i < n - 1 else 90.0) for i in range(n)}

    def run(*steps):
        return score(inst, [Frame(t=t, objects=o) for t, o in steps])

    good = run((0.0, start), (4.6, knocked), (5.0, knocked), (40.0, line), (45.0, pushed), (45.1, first),
               (48.0, down), (49.5, down))
    assert good.success and good.first_hit_s["first_down"] == 45.1
    assert "first_down" not in run((0.0, start), (4.6, knocked), (5.0, knocked)).milestones_hit
    middle = {f"domino_{i:02d}": start[f"domino_{i:02d}"] for i in R.GAPS["south"]}  # left where they wait
    assert not run((0.0, start), (40.0, {**line, **middle}), (45.0, {**pushed, **middle}),
                   (45.1, {**first, **middle}), (48.0, down), (49.5, down)).success
    from_east = {**line, "domino_24": R.fallen(R.place_x(n - 1, s), 90.0)}
    assert not run((0.0, start), (40.0, line), (45.0, from_east), (48.0, {**down, "domino_00": line["domino_00"]}),
                   (48.1, down), (49.5, down)).success
    borrow = build("Rea-03", "hard", 3, variant="borrow")
    knocked_b = {**{o.name: o.pose for o in borrow.scene.dynamic_objects}, "domino_00": knocked["domino_00"]}
    assert "first_down" not in score(borrow, [Frame(t=0.0, objects=knocked_b)]).milestones_hit


def test_hinge_axis():
    """A hinge's axis defaults to vertical; z_to takes the joint's z onto any axis."""
    from mrbench.engines.base import HingeSpec, quat_rotate, z_to

    assert HingeSpec("h", "door", (0.0, 0.0, 0.0), 1).axis == (0.0, 0.0, 1.0)
    for axis in [(0, 0, 1), (0, 1, 0), (1, 0, 0), (0, 0, -1), (0.6, 0, 0.8), (-0.3, 0.4, -0.866)]:
        n = sum(a * a for a in axis) ** 0.5
        got = quat_rotate(z_to(axis), (0.0, 0.0, 1.0))
        assert all(abs(g - a / n) < 1e-9 for g, a in zip(got, axis))


def test_level_and_through_slots():
    """Con-02's predicates on its own poses: level leaves with the pin home (through the tongue and both cheeks) pass; a
    leaf at its stop, the pin short of the far cheek, or the pin off to one side, fail."""
    from mrbench.scoring.predicates import Level, ThroughSlots
    from mrbench.tasks import build
    from mrbench.tasks.impl import con02_hinged_leaves as P

    inst = build("Con-02", "easy", 0)
    sc = inst.scene
    lw, le = P.level_pose(inst, "leaf_w"), P.level_pose(inst, "leaf_e")
    through = ThroughSlots("pin", P.KEY[1], (P.KEY[0], P.KEY[2]), P.slots(inst.metadata["slot_clearance_m"]))

    def frame(pin_y, pin_x=0.0):
        x, _, z = P.JOINT
        return Frame(0.0, {"leaf_w": lw, "leaf_e": le, "pin": (x + pin_x, pin_y, z, 1.0, 0.0, 0.0, 0.0)})

    home = frame(P.PIN_HOME_Y)
    assert through(home, sc) and Level("leaf_w", 3.0)(home, sc) and Level("leaf_e", 3.0)(home, sc)
    assert not Level("leaf_e", 3.0)(Frame(0.0, {"leaf_e": sc.obj("leaf_e").pose}), sc)  # at its stop, 30 degrees up
    assert not through(frame(P.JOINT[1] - 0.15), sc)  # short of the north cheek
    assert not through(frame(P.PIN_HOME_Y, pin_x=0.01), sc)  # 1 cm across: out of a 46 mm slot round a 34 mm key


def test_rea03_bridge_needs_the_plank_to_strike_domino_14_below_its_top():
    """The borrow configuration's middle stretch counts as bridged only by a plank whose end strikes domino_14's face
    below its top: the 30 cm prop does; the 36 cm distractor, whose side would land on domino_14's top corner, does
    not -- though both reach across."""
    from mrbench.engines.base import Frame
    from mrbench.tasks import build
    from mrbench.tasks.impl import rea03_domino_run as m

    for seed in range(5):
        inst = build("Rea-03", "easy", seed, variant="borrow")
        b = inst.metadata["borrow"]
        if inst.scene.obj(b["decoys"][0]).size[2] < m.PLANK[2]:
            continue
        stood = (m.plank_x(inst.metadata["spacing_m"]), m.LINE_Y)
        bridged = {}
        for which in ("props", "decoys"):
            body = inst.scene.obj(b[which][0])
            frame = Frame(0.0, {body.name: (*stood, m.Z + body.size[2] / 2, 1.0, 0.0, 0.0, 0.0)})
            ms = {x.name: x.formula for x in inst.milestones}
            bridged[which] = ms["middle_gap_filled"].first([frame], inst.scene) is not None
        assert bridged == {"props": True, "decoys": False}
        return
    raise AssertionError("no seed 0-4 with the taller distractor")


def test_rea02_long_channel_counts_laid_only_by_its_own_two_ends():
    """Rea-02's borrow configuration: the long channel let down in its place counts as laid; the short one, let go where
    the long one's middle goes (a --decoy run), does not."""
    from mrbench.engines.base import Frame
    from mrbench.tasks import build
    from mrbench.tasks.impl import rea02_tomato_flume as m

    inst = build("Rea-02", "easy", 0, variant="borrow")
    b = inst.metadata["borrow"]
    place = m.laid_at(m.track()[0], m.LONG)
    lowest = {"channel_3": m.laid(2)}
    for name, laid in ((b["props"][0], True), (b["decoys"][0], False)):
        frame = Frame(0.0, {**lowest, name: place})
        ms = {x.name: x.formula for x in inst.milestones}
        assert (ms["upper_laid"].first([frame], inst.scene) is not None) == laid


@pytest.mark.parametrize("tid", ["Con-01", "Con-03", "Rea-01", "Rea-04"])
def test_a_task_whose_hands_set_its_thing_down_asks_it_there_at_the_end(tid):
    """Con-01, Con-03, Rea-01 and Rea-04: the thing a hand sets down -- the box, the ball, the fruit, the sponge --
    must still be where it goes when the run ends, not there for a second at any time."""
    from mrbench.tasks import build

    assert "HoldsAtEnd" in repr(build(tid, "easy", 0).success)


