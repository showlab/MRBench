"""The clearance check's model of a propped hinged body (``HingeSpec.propped``): let go, it swings back only as far
as the bodies in its way let it; a held body coming into it pushes it open."""
import math

import pytest

from mrbench.policies import clearance as cl
from mrbench.tasks.impl._common import box, hinge

#: a door 30 cm long turning about a vertical axis at the origin, 10 cm up; closed it lies along +x
DOOR = box("door", (0.15, 0.0), (0.30, 0.02, 0.20), z=0.10)
HINGE = hinge("door_hinge", DOOR, (0.0, 0.0, 0.10), 1, open_max_deg=90.0, propped=True)


def _boxes(*objs):
    from mrbench.robots import collision as col

    return {o.name: col.boxes_of(o, o.pose) for o in objs}


def test_a_door_turned_and_its_angle_read_back():
    pose = cl.turned(HINGE, DOOR.pose, 90.0)
    assert pose[0] == pytest.approx(0.0, abs=1e-9) and pose[1] == pytest.approx(0.15)
    assert cl.opened(HINGE, DOOR.pose, cl.turned(HINGE, DOOR.pose, 37.0)) == pytest.approx(37.0)


def test_let_go_it_swings_back_till_a_body_in_its_way():
    block = box("block", (0.15 * math.cos(math.radians(45)), 0.15 * math.sin(math.radians(45))), (0.04, 0.04, 0.08),
                z=0.10)
    deg, by = cl.settle(HINGE, DOOR, 90.0, _boxes(block))
    assert by == {"block"} and 45.0 < deg < 70.0
    assert cl.settle(HINGE, DOOR, 90.0, {}) == (0.0, set())  # nothing in its way: back at its stop


def test_a_held_body_coming_into_it_pushes_it_open():
    block = box("block", (0.20, 0.015), (0.04, 0.04, 0.08), z=0.10)  # 1.5 cm into the closed door
    deg, by = cl.settle(HINGE, DOOR, 0.0, _boxes(block))
    assert by == {"block"} and 0.0 < deg < 20.0


def test_pressed_past_its_stop_it_is_on_its_stop_held_by_what_presses_it():
    block = box("block", (0.005, 0.15), (0.04, 0.04, 0.08), z=0.10)  # where the door lies on its stop (90 degrees)
    assert cl.settle(HINGE, DOOR, 80.0, _boxes(block)) == (90.0, {"block"})
