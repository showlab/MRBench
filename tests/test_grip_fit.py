"""A privileged grip takes only a part that fits between the fingers (clearance.too_wide, robots.collision.grip_span)."""

from mrbench.engines.base import ObjectSpec, Part
from mrbench.robots import collision as col


def _ball(r=0.12):
    return ObjectSpec("ball", "sphere", (r,), (0.0, 0.0, 0.0), 0.0, (1.0, 0.5, 0.0), 0.6, True)


def test_sphere_span_is_its_diameter():
    ball = _ball()
    w = col.grip_span(ball, ball.pose, (0.13, 0.0, 0.0), (1.0, 0.0, 0.0), 0.02)
    assert abs(w - 0.24) < 1e-9
    assert col.grip_span(ball, ball.pose, (0.30, 0.0, 0.0), (1.0, 0.0, 0.0), 0.02) is None  # too far: nothing


def test_box_span_along_the_fingers():
    box = ObjectSpec("crate", "box", (0.30, 0.05, 0.10), (0.0, 0.0, 0.0), 0.0, (1.0, 1.0, 1.0), 0.2, True)
    assert abs(col.grip_span(box, box.pose, (0.0, 0.03, 0.0), (0.0, 1.0, 0.0), 0.02) - 0.05) < 1e-9  # across its 5 cm
    assert abs(col.grip_span(box, box.pose, (0.0, 0.03, 0.0), (1.0, 0.0, 0.0), 0.02) - 0.30) < 1e-9  # along it


def test_a_body_too_wide_for_one_hand_is_taken_between_two():
    """Another hand on its far side (120 degrees or more round from this one) makes a squeeze."""
    c = (0.0, 0.0, 0.0)
    assert col.squeezed(c, (-0.12, 0.0, 0.0), [(0.12, 0.0, 0.0)])  # opposite sides
    assert col.squeezed(c, (-0.12, 0.0, 0.0), [(0.077, 0.092, 0.0)])  # 130 degrees round
    assert not col.squeezed(c, (-0.12, 0.0, 0.0), [(0.0, 0.12, 0.0)])  # a quarter round: it would slip out
    assert not col.squeezed(c, (-0.12, 0.0, 0.0), [])  # one hand alone


def test_a_compound_is_taken_by_its_nearest_narrow_part():
    parts = (Part("body", "box", (0.30, 0.20, 0.20)), Part("handle", "box", (0.10, 0.012, 0.012), (0.0, 0.0, 0.13)))
    toolbox = ObjectSpec("toolbox", "compound", (0.30, 0.20, 0.26), (0.0, 0.0, 0.0), 0.0, (1.0, 0.0, 0.0), 1.0, True,
                         parts=parts)
    # at the handle (2.5 cm above the body's top): only the handle is near, 1.2 cm across the fingers
    w = col.grip_span(toolbox, toolbox.pose, (0.0, 0.0, 0.13), (0.0, 1.0, 0.0), 0.02)
    assert abs(w - 0.012) < 1e-9
