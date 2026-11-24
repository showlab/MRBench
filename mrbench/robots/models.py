"""Engine-independent robot models.

These are the facts every layer needs (joint names, limits, home pose, gripper
mapping, approximate reach) without touching Isaac Sim.  The Isaac half --
USD paths, actuator gains -- lives in ``mrbench.engines.isaac.robots``.

Joint names and limits are copied from the upstream descriptions:

* ``panda``  -- Isaac Lab's ``FRANKA_PANDA_CFG`` (USD on the Isaac Nucleus).
* ``piper``  -- agilexrobotics/piper_ros ``piper_description.urdf`` (see sources.yaml).
* ``so101``  -- TheRobotStudio/SO-ARM100 ``so101_new_calib.urdf`` (see sources.yaml).

``tests/test_robots.py`` re-checks the URDF-backed ones against the fetched files.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

from ..core.errors import ConfigError

Limits = tuple[tuple[float, float], ...]
Vec3 = tuple[float, float, float]


@dataclass(frozen=True)
class WristCamera:
    """A camera on the hand: a RealSense D405 on a bracket, 8.5 cm above the tool point, looking along
    the fingers tilted 20 degrees toward them, so the open fingers are in the picture's bottom corners
    and what they hold at its bottom middle; the D405's lens (its middle square, about 60 degrees:
    18 mm on the 20.955 mm sensor) -- on the hand's front: the side away from the arm's base when the
    hand points down ahead of it.  The planner keeps it there when it can.

    Everything is in the tool frame (``RobotModel.tcp_offset_m``).  The camera's own frame is the
    optical one: it looks along its +z, the picture's x to the right and y down.  The housing (and
    the bracket) are solid boxes on the hand: the other arms and the scene can run into them."""

    pos: Vec3
    quat: tuple[float, float, float, float]
    #: (centre, half extents) of the camera's housing, a box along the tool frame's axes
    housing: tuple[Vec3, Vec3]
    #: the same, of the bracket between the hand and the housing
    bracket: Optional[tuple[Vec3, Vec3]] = None
    focal_mm: float = 18.0
    #: the near clipping plane: the housing reaches 2.4 cm along what it sees
    near_m: float = 0.026


#: the D405 looking along the fingers tilted 20 degrees toward them, from the hand's -x side and, turned half a turn about the tool axis, from its +x side
_LOOK_FROM_MINUS_X = (0.696364, -0.122788, 0.122788, -0.696364)
_LOOK_FROM_PLUS_X = (0.696364, -0.122788, -0.122788, 0.696364)
#: the D405's housing: 23 x 42 x 42 mm, thin along the line from the hand
_D405 = (0.012, 0.021, 0.021)


@dataclass(frozen=True)
class RobotModel:
    name: str
    display_name: str
    vendor: str
    arm_joints: tuple[str, ...]
    arm_limits: Limits
    #: Joint-space home pose for the arm joints (rad), inside the limits.
    home: tuple[float, ...]
    gripper_joints: tuple[str, ...]
    gripper_open: tuple[float, ...]
    gripper_closed: tuple[float, ...]
    #: Link used as the end-effector frame (must survive URDF import, so the
    #: converter keeps fixed joints -- see engines/isaac/convert.py).
    ee_link: str
    #: Approximate horizontal reach from the base axis (m); used only by the
    #: layout design rule, not by control.
    reach_m: float
    payload_kg: float
    #: "isaaclab_assets" (shipped with Isaac Lab) or "urdf" (fetched, see sources.yaml).
    source: str
    notes: str = ""
    #: The tool point -- between the fingertips -- this far along ``ee_link``'s z.  The
    #: tool frame is ``ee_link``'s: z points out through the fingers, y is the way they open.
    tcp_offset_m: float = 0.0
    wrist_camera: Optional[WristCamera] = None
    #: how far apart the fingers open (m), for a parallel gripper; None where it is not known
    grip_open_m: Optional[float] = None

    @property
    def arm_dof(self) -> int:
        return len(self.arm_joints)

    def clip_arm(self, q: Sequence[float]) -> tuple[float, ...]:
        if len(q) != self.arm_dof:
            raise ConfigError(f"{self.name}: expected {self.arm_dof} arm joint targets, got {len(q)}")
        return tuple(min(max(float(v), lo), hi) for v, (lo, hi) in zip(q, self.arm_limits))

    def gripper_targets(self, opening: float) -> tuple[float, ...]:
        """Map a normalised opening (0 = closed, 1 = open) to gripper joint targets."""
        a = min(max(float(opening), 0.0), 1.0)
        return tuple(c + a * (o - c) for c, o in zip(self.gripper_closed, self.gripper_open))


PANDA = RobotModel(
    name="panda",
    display_name="Franka Emika Panda",
    vendor="Franka Robotics",
    arm_joints=tuple(f"panda_joint{i}" for i in range(1, 8)),
    arm_limits=(
        (-2.8973, 2.8973), (-1.7628, 1.7628), (-2.8973, 2.8973), (-3.0718, -0.0698),
        (-2.8973, 2.8973), (-0.0175, 3.7525), (-2.8973, 2.8973),
    ),
    home=(0.0, -0.569, 0.0, -2.810, 0.0, 3.037, 0.741),  # Isaac Lab FRANKA_PANDA_CFG
    gripper_joints=("panda_finger_joint1", "panda_finger_joint2"),
    gripper_open=(0.04, 0.04),
    gripper_closed=(0.0, 0.0),
    ee_link="panda_hand",
    reach_m=0.855,
    payload_kg=3.0,
    source="isaaclab_assets",
    notes="7-DoF + parallel gripper",
    tcp_offset_m=0.1034,  # Isaac Lab's Franka tasks put the end-effector frame here
    grip_open_m=0.08,  # each finger travels 4 cm
    # on the hand's front, its +x side: 7.5 cm out from the hand's middle
    wrist_camera=WristCamera((0.075, 0.0, -0.085), _LOOK_FROM_PLUS_X, ((0.075, 0.0, -0.085), _D405),
                             bracket=((0.0475, 0.0, -0.085), (0.0155, 0.015, 0.003))),
)

PIPER = RobotModel(
    name="piper",
    display_name="AgileX PiPER",
    vendor="AgileX Robotics",
    arm_joints=tuple(f"joint{i}" for i in range(1, 7)),
    arm_limits=(
        (-2.618, 2.618), (0.0, 3.14), (-2.967, 0.0),
        (-1.745, 1.745), (-1.22, 1.22), (-2.0944, 2.0944),
    ),
    home=(0.0, 0.5, -0.6, 0.0, 0.4, 0.0),
    gripper_joints=("joint7", "joint8"),
    gripper_open=(0.035, -0.035),
    gripper_closed=(0.0, 0.0),
    ee_link="gripper_base",
    reach_m=0.626,
    payload_kg=1.5,
    source="urdf",
    notes="6-DoF + parallel gripper (two prismatic fingers)",
    tcp_offset_m=0.125,  # the finger meshes end 0.136 m out along gripper_base z
    grip_open_m=0.07,  # each finger travels 3.5 cm
    # on the gripper body's -x face (it runs -4.0 to +3.05 cm), its front: pointing down ahead of the base,
    # gripper_base's +x points back at it.  Half way up the body, 8.5 cm above the tool point
    wrist_camera=WristCamera((-0.06, 0.0, -0.085), _LOOK_FROM_MINUS_X, ((-0.06, 0.0, -0.085), _D405),
                             bracket=((-0.044, 0.0, -0.085), (0.004, 0.015, 0.003))),
)

SO101 = RobotModel(
    name="so101",
    display_name="SO-101",
    vendor="TheRobotStudio / Hugging Face LeRobot",
    arm_joints=("shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll"),
    arm_limits=(
        (-1.91986, 1.91986), (-1.74533, 1.74533), (-1.69, 1.69),
        (-1.65806, 1.65806), (-2.74385, 2.84121),
    ),
    # stretched out ahead, the wrist rolled a quarter turn: the jaw then opens sideways (at 0 it opens up and down)
    home=(0.0, 0.0, 0.0, 0.0, 1.5708),
    gripper_joints=("gripper",),
    # The jaw is a single revolute joint (URDF range -0.17..1.75 rad).
    gripper_open=(1.0,),
    gripper_closed=(0.0,),
    ee_link="gripper_frame_link",
    # Joint offsets in the URDF add up to ~0.41 m from the shoulder; 0.38 m is a
    # conservative usable reach.  Payload is not published; ~0.2 kg is an estimate.
    reach_m=0.38,
    payload_kg=0.2,
    source="urdf",
    notes="5-DoF + single revolute jaw",
    # its tool frame opens along x (the moving jaw on the -x side) and its jaw turns about y, which at home
    # points down: the camera sits on top of the gripper (its -y face, at -2.8 cm), half way up it, looking
    # along the jaws tilted 20 degrees toward them, so that its picture is upright at home -- and pointing
    # down ahead of the base, that face is the front
    wrist_camera=WristCamera((0.0, -0.043, -0.06), (0.984808, -0.173648, 0.0, 0.0),
                             ((0.0, -0.043, -0.06), (0.021, 0.012, 0.021)),
                             bracket=((0.0, -0.0295, -0.06), (0.015, 0.0015, 0.003))),
)

MODELS: dict[str, RobotModel] = {m.name: m for m in (PANDA, PIPER, SO101)}


def get_model(name: str) -> RobotModel:
    try:
        return MODELS[name]
    except KeyError:
        raise ConfigError(f"unknown robot model {name!r}; available: {', '.join(sorted(MODELS))}") from None
