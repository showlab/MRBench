"""Convert fetched URDFs to USD once (needs the Isaac app running for the URDF importer)."""

from __future__ import annotations

from pathlib import Path

from ...robots import MODELS
from ...robots import fetch as fetch_mod
from .robots import GAINS, usd_path


def urdf_robots() -> list[str]:
    return [n for n, m in MODELS.items() if m.source == "urdf"]


def convert(name: str, force: bool = False) -> Path:
    from isaaclab.sim.converters import UrdfConverter, UrdfConverterCfg

    urdf = fetch_mod.require(name)
    out = usd_path(name)
    g = GAINS[name]
    cfg = UrdfConverterCfg(
        asset_path=str(urdf),
        usd_dir=str(out.parent),
        usd_file_name=out.name,
        fix_base=True,
        # keep fixed links: the end-effector frames (gripper_base, gripper_frame_link) are fixed children
        merge_fixed_joints=False,
        # not instanceable: an instanceable robot's links move in the physics but are drawn where they
        # started
        make_instanceable=False,
        force_usd_conversion=force,
        joint_drive=UrdfConverterCfg.JointDriveCfg(
            gains=UrdfConverterCfg.JointDriveCfg.PDGainsCfg(stiffness=g.arm_kp, damping=g.arm_kd)
        ),
    )
    UrdfConverter(cfg)
    return out
