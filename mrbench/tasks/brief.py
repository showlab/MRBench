"""What an agent is told about a task instance, at each information level (``spec.INFO_LEVELS``).

The instruction gives only the goal: no roles, no steps, no tool to use, no minimal support set.  Beside it the
agent gets the scene's facts -- the card's ``facts`` and every body and marked place on the table -- and the arms
on the table: at ``given`` with their capabilities, at ``query`` only by name and place, their capabilities
answered one arm at a time by ``capabilities``.  The capabilities are the two the benchmark asks about, what an
arm reaches and what its hand can take; payload and force are left out.
"""

from __future__ import annotations

from ..core.errors import ConfigError
from ..robots import get_model
from .instance import TaskInstance
from .spec import INFO_LEVELS


def _r(v, n: int = 3):
    return [round(x, n) for x in v]


def capabilities(inst: TaskInstance, slot: str) -> dict:
    """One arm's capabilities: what its robot reaches and what its hand can take."""
    m = get_model(inst.scene.arm(slot).model)
    return {"arm": slot, "robot": m.display_name, "dof": m.arm_dof, "reach_m": m.reach_m,
            "joint_limits_rad": [_r(lim) for lim in m.arm_limits],
            "gripper": "parallel" if len(m.gripper_joints) == 2 else "single jaw", "grip_open_m": m.grip_open_m}


def brief(inst: TaskInstance, info: str = "given") -> dict:
    """The goal, the scene's facts and the arms, as an agent gets them at ``info``."""
    if info not in INFO_LEVELS:
        raise ConfigError(f"info must be one of {INFO_LEVELS}, got {info!r}")
    sc = inst.scene
    arms = [{"arm": a.slot, "base_xyz": _r(a.pos), "facing_deg": a.yaw_deg} for a in sc.arms]
    if info == "given":
        arms = [{**a, **capabilities(inst, a["arm"])} for a in arms]
    return {
        "task": inst.spec.id, "level": inst.level, "seed": inst.seed, "info": info,
        "goal": inst.instruction,
        "facts": list(inst.spec.facts),
        "table": {"size_m": _r(sc.table_size), "top_z": sc.table_top_z},
        "arms": arms,
        "bodies": [{"name": o.name, "shape": o.shape, "size_m": _r(o.size), "xyz": _r(o.pos), "yaw_deg": o.yaw_deg,
                    "mass_kg": o.mass_kg if o.dynamic else None, "movable": o.dynamic} for o in sc.objects],
        "cloths": [{"name": c.name, "size_m": _r(c.size), "xyz": _r(c.pos)} for c in sc.cloths],
        "places": [{"name": r.name, "centre": _r(r.center), "half_extents": _r(r.half)}
                   for r in sc.regions],
    }
