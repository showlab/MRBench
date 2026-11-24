import xml.etree.ElementTree as ET

import pytest

from mrbench.robots import MODELS, get_model
from mrbench.robots import fetch as fetch_mod


def test_three_models_with_expected_dof():
    assert {n: m.arm_dof for n, m in MODELS.items()} == {"panda": 7, "piper": 6, "so101": 5}


@pytest.mark.parametrize("name", sorted(MODELS))
def test_home_inside_limits_and_gripper_mapping(name):
    m = get_model(name)
    assert len(m.home) == len(m.arm_limits) == m.arm_dof
    assert m.clip_arm(m.home) == pytest.approx(m.home)
    assert m.gripper_targets(1.0) == pytest.approx(m.gripper_open)
    assert m.gripper_targets(0.0) == pytest.approx(m.gripper_closed)
    assert m.gripper_targets(7.0) == pytest.approx(m.gripper_open)  # clamped
    assert len(m.gripper_open) == len(m.gripper_closed) == len(m.gripper_joints)


@pytest.mark.parametrize("name", [n for n, m in MODELS.items() if m.source == "urdf"])
def test_model_matches_fetched_urdf(name):
    """Joint names, limits and the EE link must agree with the pinned upstream URDF."""
    if not fetch_mod.status(name)["ok"]:
        pytest.skip(f"{name} not fetched (`mrbench assets fetch`)")
    root = ET.parse(fetch_mod.urdf_path(name)).getroot()
    joints = {j.get("name"): j for j in root.findall("joint")}
    links = {link.get("name") for link in root.findall("link")}
    m = get_model(name)
    assert m.ee_link in links
    for jn, (lo, hi) in zip(m.arm_joints, m.arm_limits):
        lim = joints[jn].find("limit")
        assert (float(lim.get("lower")), float(lim.get("upper"))) == pytest.approx((lo, hi), abs=1e-3), jn
    for jn, o, c in zip(m.gripper_joints, m.gripper_open, m.gripper_closed):
        lim = joints[jn].find("limit")
        lo, hi = float(lim.get("lower")), float(lim.get("upper"))
        assert lo - 1e-6 <= min(o, c) and max(o, c) <= hi + 1e-6, jn
