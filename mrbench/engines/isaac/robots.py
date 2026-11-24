"""Isaac Lab ``ArticulationCfg`` for each robot model.  Call only after ``app.launch()``.

* Panda comes from ``isaaclab_assets`` (USD on the Isaac Nucleus).
* Piper and SO-101 come from the URDFs fetched by ``mrbench assets fetch`` and
  converted once to USD by ``mrbench assets convert`` (see convert.py).

The joint drives' gains are below, per robot model.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from ...core.errors import AssetMissing
from ...core.paths import third_party_dir
from ...robots import get_model


@dataclass(frozen=True)
class Gains:
    arm_kp: float
    arm_kd: float
    arm_effort: float
    grip_kp: float
    grip_kd: float
    grip_effort: float


GAINS = {
    # Piper URDF: effort 100 on arm joints, 10 on the prismatic fingers.
    "piper": Gains(arm_kp=400.0, arm_kd=40.0, arm_effort=100.0, grip_kp=1e3, grip_kd=1e2, grip_effort=10.0),
    # SO-101: hobby servos (STS3215, ~3 N*m stall); URDF says effort 10.
    "so101": Gains(arm_kp=17.8, arm_kd=0.6, arm_effort=10.0, grip_kp=17.8, grip_kd=0.6, grip_effort=10.0),
}


def usd_path(name: str) -> Path:
    return third_party_dir() / "usd" / name / f"{name}.usd"


def articulation_cfg(model_name: str, prim_path: str):
    import isaaclab.sim as sim_utils
    from isaaclab.actuators import ImplicitActuatorCfg
    from isaaclab.assets import ArticulationCfg

    model = get_model(model_name)
    if model.source == "isaaclab_assets":
        # High-PD variant: gravity compensation off-loaded, stiff joints -- suited to joint-position targets.
        # Spawned from Isaac Sim's own Franka USD rather than Isaac Lab's instanceable one: an instanceable
        # robot's links move in the physics but are drawn where they started (the cameras see a frozen arm).
        from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
        from isaaclab_assets.robots.franka import FRANKA_PANDA_HIGH_PD_CFG

        cfg = FRANKA_PANDA_HIGH_PD_CFG.replace(prim_path=prim_path)
        return cfg.replace(spawn=cfg.spawn.replace(usd_path=f"{ISAAC_NUCLEUS_DIR}/Robots/Franka/franka.usd"))

    usd = usd_path(model_name)
    if not usd.exists():
        raise AssetMissing(f"{usd} not found; run `mrbench assets fetch` then `mrbench assets convert`")
    g = GAINS[model_name]
    joint_pos = dict(zip(model.arm_joints, model.home)) | dict(zip(model.gripper_joints, model.gripper_open))
    return ArticulationCfg(
        prim_path=prim_path,
        spawn=sim_utils.UsdFileCfg(
            usd_path=str(usd),
            activate_contact_sensors=False,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(disable_gravity=False, max_depenetration_velocity=5.0),
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                enabled_self_collisions=False, solver_position_iteration_count=8, solver_velocity_iteration_count=0
            ),
        ),
        init_state=ArticulationCfg.InitialStateCfg(joint_pos=joint_pos),
        actuators={
            "arm": ImplicitActuatorCfg(joint_names_expr=list(model.arm_joints), effort_limit_sim=g.arm_effort,
                                       stiffness=g.arm_kp, damping=g.arm_kd),
            "gripper": ImplicitActuatorCfg(joint_names_expr=list(model.gripper_joints), effort_limit_sim=g.grip_effort,
                                           stiffness=g.grip_kp, damping=g.grip_kd),
        },
        soft_joint_pos_limit_factor=1.0,
    )
