"""Arm kinematics and collision geometry: checked against what Isaac measured, and against themselves."""

import math

import pytest

np = pytest.importorskip("numpy")

from mrbench.robots import collision as col  # noqa: E402
from mrbench.robots import get_model  # noqa: E402
from mrbench.robots import kinematics as kin  # noqa: E402

#: ee link at home, base frame, as Isaac reported it (panda_hand, gripper_base, gripper_frame_link); the
#: SO-101 is measured with every joint at 0 (its home rolls the wrist a quarter turn)
MEASURED_HOME = {"panda": (0.389, 0.0, 0.458), "piper": (0.102, 0.0, 0.341), "so101": (0.391, 0.0, 0.226)}
MEASURED_AT = {"so101": (0.0, 0.0, 0.0, 0.0, 0.0)}


def _fetched(name):
    try:
        kin.chain(name)
        return True
    except Exception:
        return False


@pytest.mark.parametrize("name", sorted(MEASURED_HOME))
def test_home_matches_isaac(name):
    if not _fetched(name):
        pytest.skip(f"{name} URDF not fetched")
    pos = kin.fk(name, MEASURED_AT.get(name, get_model(name).home))[:3, 3]
    assert pos == pytest.approx(MEASURED_HOME[name], abs=2e-3)


@pytest.mark.parametrize("name", sorted(MEASURED_HOME))
def test_jacobian_matches_finite_differences(name):
    if not _fetched(name):
        pytest.skip(f"{name} URDF not fetched")
    q = np.array(get_model(name).home) + 0.1
    t, jac = kin.tcp_and_jacobian(name, q)
    num = np.stack([kin.pose_error(t, kin.tcp(name, q + np.eye(len(q))[k] * 1e-6)) / 1e-6 for k in range(len(q))], 1)
    assert np.abs(jac - num).max() < 1e-4


def test_ik_recovers_a_reachable_pose():
    m = get_model("panda")
    q0 = np.array(m.home) + np.array([0.3, 0.2, -0.2, 0.3, 0.2, -0.3, 0.2])
    target = kin.tcp("panda", q0)
    q = kin.ik("panda", target, m.home)
    assert q is not None
    assert np.linalg.norm(kin.pose_error(kin.tcp("panda", q), target)) < 2e-3


def test_capsule_and_box_gaps():
    a = col.Capsule("a", np.array([0.0, 0.0, 0.0]), np.array([1.0, 0.0, 0.0]), 0.1)
    b = col.Capsule("b", np.array([0.5, 0.5, 0.0]), np.array([0.5, 0.5, 1.0]), 0.1)
    assert col.capsule_capsule(a, b) == pytest.approx(0.3, abs=1e-6)
    box = col.Box("box", np.array([0.0, 0.0, -0.5]), np.eye(3), np.array([1.0, 1.0, 0.4]))
    assert col.capsule_box(a, box) == pytest.approx(0.0, abs=1e-6)  # its underside touches the box's top
    other = col.Box("other", np.array([0.0, 0.0, 0.3]), np.eye(3), np.array([0.1, 0.1, 0.1]))
    assert col.box_box(box, other) == pytest.approx(0.3, abs=1e-6)  # its top at -0.1, the other's bottom at 0.2
    assert math.isclose(float(col.point_box(np.array([[0.0, 0.0, -0.5]]), box)[0]), -0.4, abs_tol=1e-9)
