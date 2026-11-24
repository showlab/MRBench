"""The Isaac Sim engine: builds a SceneSpec as an Isaac Lab ``InteractiveScene``.

One stage per process: the first ``load`` builds the scene; later loads of a
scene with the same composition (same arms and objects, e.g. another seed)
only reset poses.  A different task needs a new process (``mrbench run --isolate``).

Runs on Isaac Sim 4.5 / Isaac Lab 2.1.
"""

from __future__ import annotations

import math
import os
from typing import Mapping, Optional

from ...core.errors import ConfigError
from ...robots import get_model
from ..base import ArmCommand, Engine, Frame, ObjectSpec, Pose7, RegionSpec, SceneSpec, Vec3, yaw_quat
from . import app, environment

#: RTX settings (Isaac Lab ``RenderCfg`` fields).  FXAA, not the default DLSS:
#: DLSS rebuilds a frame from earlier ones and leaves ghosts after a teleport.
#: Ambient occlusion makes objects sit on the table instead of floating above it.
RENDER_SETTINGS = {"antialiasing_mode": "FXAA", "enable_ambient_occlusion": True}
#: RTX carries lighting from frame to frame; after a reset or a teleport it needs
#: this many passes before the picture stops changing.
RENDER_PASSES = 30
#: how opaque a glass body is drawn (ObjectSpec.glass): enough to see it is there, little enough to see through it
GLASS_OPACITY = 0.1
#: Rigid bodies never sleep: PhysX puts a body at rest to sleep, and a pose written
#: through the GPU tensor API does not wake it -- a body lifted that way would hang
#: in mid-air.  The scenes are small; staying awake costs little.
NEVER_SLEEP = 0.0
#: the room, the light, the table and the floor are the scene's look (``environment.py``)


def linear_rgb(rgb: tuple[float, float, float]) -> tuple[float, float, float]:
    """sRGB (colours here and in the task scenes are sRGB, as a person picks them) -> linear: USD's
    PreviewSurface takes linear albedo, so an sRGB orange passed straight through renders as pale amber
    and red as pink."""
    return tuple(c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4 for c in rgb)  # type: ignore[return-value]


def _scene_key(scene: SceneSpec) -> tuple:
    return (tuple((a.slot, a.model, a.pos, a.yaw_deg) for a in scene.arms),
            tuple((o.name, o.shape, o.size, o.dynamic, o.parts) for o in scene.objects),
            tuple(r.name for r in scene.regions), scene.hinges, tuple((c.name, c.size, c.cell) for c in scene.cloths),
            scene.fits, scene.look, tuple((p.name, p.kind) for p in scene.props))


class IsaacEngine(Engine):
    name = "isaac"

    def __init__(self, headless: bool = True, device: str = "cuda:0", physics_dt: float = 1 / 120,
                 decimation: int = 12, enable_cameras: bool = False,
                 cameras: Optional[Mapping[str, tuple[Vec3, Vec3, float]]] = None,
                 camera_size: tuple[int, int] = (800, 600), wrist_cameras: bool = False,
                 wrist_size: tuple[int, int] = (320, 240), **_ignored):
        """``cameras`` maps a view name to (eye, target, focal length in mm) in the
        world frame; each becomes a fixed RGB camera that :meth:`render` reads.  With
        ``wrist_cameras`` every arm's wrist camera (``RobotModel.wrist_camera``) takes
        pictures too, ``wrist_<arm>``.  The cameras' housings are on the hands either way."""
        self.headless, self.device = headless, device
        self.physics_dt, self.decimation = physics_dt, decimation
        self.control_dt = physics_dt * decimation  # 0.1 s by default
        self.cameras = dict(cameras or {})
        self.camera_size = camera_size
        self.wrist_cameras, self.wrist_size = wrist_cameras, wrist_size
        self.enable_cameras = enable_cameras or bool(self.cameras) or wrist_cameras
        self._wrist: dict = {}
        self._fits: dict = {}
        self._key = None
        self._t = 0.0
        self._cloths = None

    # ----------------------------------------------------------------- build

    def _shape_cfg(self, o: ObjectSpec | RegionSpec, collide: bool, dynamic: bool, mass: float = 0.0,
                   gravity: bool = True):
        import isaaclab.sim as sim_utils

        if getattr(o, "glass", False):  # see-through: OmniGlass draws black here; a faint, smooth surface instead
            glass = sim_utils.PreviewSurfaceCfg(diffuse_color=linear_rgb(o.color), opacity=GLASS_OPACITY, roughness=0.1)
            kw = {"visual_material": glass}
        else:
            kw = {"visual_material": sim_utils.PreviewSurfaceCfg(diffuse_color=linear_rgb(o.color))}
        if collide:
            kw["collision_props"] = sim_utils.CollisionPropertiesCfg()
        if dynamic:
            kw["rigid_props"] = sim_utils.RigidBodyPropertiesCfg(sleep_threshold=NEVER_SLEEP, disable_gravity=not gravity)
            kw["mass_props"] = sim_utils.MassPropertiesCfg(mass=mass)
        if collide and getattr(o, "friction", None) is not None:
            kw["physics_material"] = sim_utils.RigidBodyMaterialCfg(
                static_friction=o.friction, dynamic_friction=o.friction, friction_combine_mode=o.friction_combine)
        if isinstance(o, RegionSpec):
            return sim_utils.CuboidCfg(size=(2 * o.half[0], 2 * o.half[1], 0.002), **kw)
        if o.shape == "box":
            return sim_utils.CuboidCfg(size=tuple(o.size), **kw)
        if o.shape == "cylinder":
            return sim_utils.CylinderCfg(radius=o.size[0], height=o.size[1], axis="Z", **kw)
        return sim_utils.SphereCfg(radius=o.size[0], **kw)

    def _build(self, scene: SceneSpec) -> None:
        import isaaclab.sim as sim_utils
        from isaaclab.assets import AssetBaseCfg, RigidObjectCfg
        from isaaclab.scene import InteractiveScene, InteractiveSceneCfg

        from .robots import articulation_cfg

        # PhysX's stabilization pass stays on, Isaac Lab 2.1's default, and is said here so that it is chosen: a slow
        # body touching a static one with other shapes near it gets about 1/8 of its weight.  Rea-02's tomato stays on
        # its stop and Rea-01's borrowed orange on its collar by it; without it they do not, and it does not shut
        # Con-03's end walls either.
        # MRBENCH_STABILIZATION=0 turns it off, to compare.
        stabilize = os.environ.get("MRBENCH_STABILIZATION", "1") != "0"
        self.sim = sim_utils.SimulationContext(sim_utils.SimulationCfg(
            dt=self.physics_dt, device=self.device, render=sim_utils.RenderCfg(**RENDER_SETTINGS),
            physx=sim_utils.PhysxCfg(enable_stabilization=stabilize)))
        environment.apply_rtx_settings()
        self.sim.set_camera_view(eye=(0.0, -2.2, 2.0), target=(0.0, 0.0, scene.table_top_z))

        cfg = InteractiveSceneCfg(num_envs=1, env_spacing=4.0, replicate_physics=False)
        # the room, its light, the floor and the table, as the scene's look has them
        for name, entity in environment.scene_cfgs(scene).items():
            setattr(cfg, name, entity)
        for a in scene.arms:
            robot = articulation_cfg(a.model, f"{{ENV_REGEX_NS}}/{a.slot}")
            setattr(cfg, a.slot, robot.replace(init_state=robot.init_state.replace(pos=a.pos, rot=yaw_quat(a.yaw_deg))))
        hanging = {t.body for c in scene.cloths for t in c.ties if t.hang}  # placed by Cloths.hang every step
        for o in scene.objects:
            path = f"{{ENV_REGEX_NS}}/{o.name}"
            if o.parts:  # a compound body, written once as a small USD file
                from . import compound

                spawn = sim_utils.UsdFileCfg(usd_path=str(compound.write(o)))
            else:
                spawn = self._shape_cfg(o, o.name not in hanging, o.dynamic, o.mass_kg, gravity=o.name not in hanging)
            rot = tuple(o.pose[3:])
            if o.dynamic:
                setattr(cfg, o.name, RigidObjectCfg(
                    prim_path=path, spawn=spawn, init_state=RigidObjectCfg.InitialStateCfg(pos=o.pos, rot=rot)))
            else:
                setattr(cfg, o.name, AssetBaseCfg(
                    prim_path=path, spawn=spawn, init_state=AssetBaseCfg.InitialStateCfg(pos=o.pos, rot=rot)))
        for p in scene.props:
            setattr(cfg, p.name, environment.prop_cfg(p, f"{{ENV_REGEX_NS}}/{p.name}"))
        for r in scene.regions:
            if r.visible:  # flat, non-colliding marker on the table
                z = scene.table_top_z + 0.001
                setattr(cfg, f"marker_{r.name}", AssetBaseCfg(
                    prim_path=f"{{ENV_REGEX_NS}}/marker_{r.name}", spawn=self._shape_cfg(r, False, False),
                    init_state=AssetBaseCfg.InitialStateCfg(pos=(r.center[0], r.center[1], z))))

        if self.cameras:
            from isaaclab.sensors import CameraCfg

            w, h = self.camera_size
            for name, (_, _, focal) in self.cameras.items():
                setattr(cfg, f"cam_{name}", CameraCfg(
                    prim_path=f"{{ENV_REGEX_NS}}/cam_{name}", width=w, height=h, data_types=["rgb"],
                    spawn=sim_utils.PinholeCameraCfg(focal_length=focal, focus_distance=400.0,
                                                     horizontal_aperture=20.955, clipping_range=(0.05, 20.0))))

        self.scene = InteractiveScene(cfg)
        import isaacsim.core.utils.stage as stage_utils

        environment.dress(stage_utils.get_current_stage(), scene)
        self._uninstance_arms(scene)
        origin = tuple(self.scene.env_origins[0].tolist())
        self._add_hinges(scene, origin)
        self._add_fits(scene)
        self._add_camera_housings(scene)
        self._wrist = self._add_wrist_cameras(scene) if self.wrist_cameras else {}
        if scene.cloths:
            from . import cloth
            from .puppeteer import spawn_holds

            for c in scene.cloths:
                cloth.spawn(c, origin)
                cloth.tie(c, scene)
            spawn_holds(scene)  # what a hand attaches cloth to (puppeteer.py)
        self.sim.reset()
        if self.enable_cameras:  # textures and materials loaded before the first picture
            environment.wait_for_assets(self.sim)
        self._cloths = None
        if scene.cloths:
            self._cloths = cloth.Cloths(self.sim, scene.cloths, origin, self.device, scene)
        if self.cameras:
            import torch

            origin = self.scene.env_origins[0]
            for name, (eye, target, _) in self.cameras.items():
                self.scene[f"cam_{name}"].set_world_poses_from_view(
                    torch.tensor([eye], device=self.device) + origin,
                    torch.tensor([target], device=self.device) + origin)
        self._joint_ids = {}
        for a in scene.arms:
            m, robot = get_model(a.model), self.scene[a.slot]
            arm_ids, _ = robot.find_joints(list(m.arm_joints), preserve_order=True)
            grip_ids, _ = robot.find_joints(list(m.gripper_joints), preserve_order=True)
            self._joint_ids[a.slot] = (arm_ids, grip_ids)

    def _add_hinges(self, scene: SceneSpec, origin) -> None:
        """Each hinge as a PhysX revolute joint from the world to its body, the spring an
        angular drive toward ``preload_deg`` past closed.  Before ``sim.reset()``."""
        import isaacsim.core.utils.stage as stage_utils
        from pxr import Gf, Sdf, UsdPhysics

        from ..base import quat_mul, quat_rotate, z_to

        stage = stage_utils.get_current_stage()
        for h in scene.hinges:
            body = scene.obj(h.body)
            q = body.pose[3:]
            back = (q[0], -q[1], -q[2], -q[3])
            joint = UsdPhysics.RevoluteJoint.Define(stage, f"/World/envs/env_0/{h.name}")
            joint.CreateBody1Rel().SetTargets([Sdf.Path(f"/World/envs/env_0/{h.body}")])  # body0 unset: the world
            # the joint's z along the hinge's axis: the world frame turned by r
            r = z_to(h.axis)
            joint.CreateAxisAttr("Z")
            joint.CreateLocalPos0Attr(Gf.Vec3f(*[float(p + o) for p, o in zip(h.pivot, origin)]))
            joint.CreateLocalRot0Attr(Gf.Quatf(float(r[0]), Gf.Vec3f(*[float(v) for v in r[1:]])))
            # the same point and axes seen from the body as it starts: the joint angle is 0 there
            local = quat_rotate(back, [p - c for p, c in zip(h.pivot, body.pos)])
            rel = quat_mul(back, r)
            joint.CreateLocalPos1Attr(Gf.Vec3f(*[float(v) for v in local]))
            joint.CreateLocalRot1Attr(Gf.Quatf(float(rel[0]), Gf.Vec3f(*[float(v) for v in rel[1:]])))
            lo, hi = sorted((0.0, h.sign * h.open_max_deg))
            joint.CreateLowerLimitAttr(float(lo))
            joint.CreateUpperLimitAttr(float(hi))
            drive = UsdPhysics.DriveAPI.Apply(joint.GetPrim(), "angular")
            drive.CreateTypeAttr("force")
            # USD angular drive gains are per degree
            drive.CreateStiffnessAttr(float(h.stiffness * math.pi / 180))
            drive.CreateDampingAttr(float(h.damping * math.pi / 180))
            drive.CreateTargetPositionAttr(float(-h.sign * h.preload_deg))

    def _add_fits(self, scene: SceneSpec) -> None:
        """Each tight fit as a PhysX prismatic joint along the base's z, from where the body starts, its
        linear drive the spring that holds it in; ``_release_fits`` takes the joint away once the body has
        come out of it.  Before ``sim.reset()``."""
        import isaacsim.core.utils.stage as stage_utils
        from pxr import Gf, Sdf, UsdPhysics

        from ..base import quat_mul, quat_rotate

        stage = stage_utils.get_current_stage()
        self._fits = {}
        for f in scene.fits:
            base, body = scene.obj(f.base).pose, scene.obj(f.body).pose
            back = (base[3], -base[4], -base[5], -base[6])
            local = quat_rotate(back, [b - a for a, b in zip(base[:3], body[:3])])
            rel = quat_mul(back, body[3:])
            path = f"/World/envs/env_0/{f.name}"
            joint = UsdPhysics.PrismaticJoint.Define(stage, path)
            joint.CreateBody0Rel().SetTargets([Sdf.Path(f"/World/envs/env_0/{f.base}")])
            joint.CreateBody1Rel().SetTargets([Sdf.Path(f"/World/envs/env_0/{f.body}")])
            joint.CreateAxisAttr("Z")
            joint.CreateLocalPos0Attr(Gf.Vec3f(*[float(v) for v in local]))
            joint.CreateLocalRot0Attr(Gf.Quatf(float(rel[0]), Gf.Vec3f(*[float(v) for v in rel[1:]])))
            joint.CreateLocalPos1Attr(Gf.Vec3f(0.0, 0.0, 0.0))
            joint.CreateLocalRot1Attr(Gf.Quatf(1.0, Gf.Vec3f(0.0, 0.0, 0.0)))
            joint.CreateLowerLimitAttr(0.0)  # it does not go further in
            joint.CreateUpperLimitAttr(float(3 * f.travel_m))
            drive = UsdPhysics.DriveAPI.Apply(joint.GetPrim(), "linear")
            drive.CreateTypeAttr("force")
            stiffness = f.break_force_n / f.travel_m
            drive.CreateStiffnessAttr(float(stiffness))
            drive.CreateDampingAttr(float(2.0 * (stiffness * scene.obj(f.body).mass_kg) ** 0.5))
            drive.CreateTargetPositionAttr(0.0)
            self._fits[f.name] = (f, local, path)

    def _hang(self) -> None:
        """Bodies hanging from a cloth (``ClothTie.hang``) to under their points -- after every physics step."""
        if self._cloths is not None and self._cloths.hanging:
            self._cloths.hang(self.scene, self._scene.table_top_z)

    def _release_fits(self) -> None:
        """Take away the joint of every fit whose body has come out further than its travel."""
        import isaacsim.core.utils.stage as stage_utils

        from ..base import quat_rotate

        origin = self.scene.env_origins[0].tolist()
        for name, (f, local, path) in list(self._fits.items()):
            base, body = self.scene[f.base].data, self.scene[f.body].data
            bp = [float(v) - o for v, o in zip(base.root_pos_w[0], origin)]
            q = [float(v) for v in base.root_quat_w[0]]
            out = quat_rotate((q[0], -q[1], -q[2], -q[3]), [float(v) - o - b for v, o, b in
                                                          zip(body.root_pos_w[0], origin, bp)])
            if out[2] - local[2] > f.travel_m:
                stage_utils.get_current_stage().RemovePrim(path)
                del self._fits[name]

    def _uninstance_arms(self, scene: SceneSpec) -> None:
        """Give every arm its own meshes.  A converted robot's visuals are instanceable: with two copies of the
        same robot in a stage (Rea-01's two Pipers) they share one prototype, and the second is drawn standing at
        home while its links move -- what it carries seems to fly on its own.  Before ``sim.reset()``."""
        import isaacsim.core.utils.stage as stage_utils
        from pxr import Usd

        stage = stage_utils.get_current_stage()
        for a in scene.arms:
            for p in Usd.PrimRange(stage.GetPrimAtPath(f"/World/envs/env_0/{a.slot}")):
                if p.IsInstanceable():
                    p.SetInstanceable(False)

    def _add_camera_housings(self, scene: SceneSpec) -> None:
        """A dark box where each wrist camera is, and its bracket, part of the hand's rigid body (the
        camera is a solid thing on the real robot; without one another arm passes through it)."""
        import isaacsim.core.utils.stage as stage_utils
        from pxr import Gf, Sdf, UsdGeom, UsdPhysics, UsdShade

        from .puppeteer import ee_link_path

        stage = stage_utils.get_current_stage()
        dark = Gf.Vec3f(0.08, 0.08, 0.09)
        for a in scene.arms:
            m = get_model(a.model)
            cam = m.wrist_camera
            if cam is None:
                continue
            link = ee_link_path(a.slot, m.ee_link)
            boxes = [("camera_housing", cam.housing)] + ([("camera_bracket", cam.bracket)] if cam.bracket else [])
            for name, ((x, y, z), half) in boxes:
                path = f"{link}/{name}"
                cube = UsdGeom.Cube.Define(stage, path)
                cube.CreateSizeAttr(2.0)
                xform = UsdGeom.Xformable(cube)
                xform.AddTranslateOp().Set(Gf.Vec3d(x, y, z + m.tcp_offset_m))
                xform.AddScaleOp().Set(Gf.Vec3f(*half))
                cube.CreateDisplayColorAttr([dark])
                UsdPhysics.CollisionAPI.Apply(cube.GetPrim())
                material = UsdShade.Material.Define(stage, path + "_material")
                shader = UsdShade.Shader.Define(stage, path + "_material/Shader")
                shader.CreateIdAttr("UsdPreviewSurface")
                shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(dark)
                material.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")
                UsdShade.MaterialBindingAPI.Apply(cube.GetPrim()).Bind(material)

    def _add_wrist_cameras(self, scene: SceneSpec) -> dict:
        """One camera on each arm's hand, where its model mounts it, moving with the hand.  Before ``sim.reset()``."""
        import isaaclab.sim as sim_utils
        from isaaclab.sensors import Camera, CameraCfg

        from .puppeteer import ee_link_path

        out = {}
        w, h = self.wrist_size
        for a in scene.arms:
            m = get_model(a.model)
            cam = m.wrist_camera
            if cam is None:
                continue
            x, y, z = cam.pos
            out[a.slot] = Camera(CameraCfg(
                prim_path=f"{ee_link_path(a.slot, m.ee_link)}/wrist_camera", width=w, height=h, data_types=["rgb"],
                offset=CameraCfg.OffsetCfg(pos=(x, y, z + m.tcp_offset_m), rot=cam.quat, convention="ros"),
                spawn=sim_utils.PinholeCameraCfg(focal_length=cam.focal_mm, focus_distance=400.0,
                                                 horizontal_aperture=20.955, clipping_range=(cam.near_m, 20.0))))
        return out

    # ------------------------------------------------------------------ API

    def load(self, scene: SceneSpec, seed: int) -> Frame:
        key = _scene_key(scene)
        if self._key is None:
            app.launch(headless=self.headless, enable_cameras=self.enable_cameras, device=self.device)
            self._build(scene)
            self._key = key
        elif key != self._key:
            raise ConfigError("the Isaac engine builds one stage per process; run each task in its own "
                              "process (`mrbench run --isolate`)")
        self._scene = scene
        self._reset(scene)
        return self._frame()

    def _reset(self, scene: SceneSpec) -> None:
        import torch

        origin = self.scene.env_origins[0]
        for o in scene.dynamic_objects:
            pose = torch.tensor([o.pose], device=self.device)
            pose[:, :3] += origin
            self.scene[o.name].write_root_pose_to_sim(pose)
            self.scene[o.name].write_root_velocity_to_sim(torch.zeros(1, 6, device=self.device))
        for p in scene.props:
            pose = torch.tensor([environment.prop_root_pose(p)], device=self.device)
            pose[:, :3] += origin
            self.scene[p.name].write_root_pose_to_sim(pose)
            self.scene[p.name].write_root_velocity_to_sim(torch.zeros(1, 6, device=self.device))
        if self._cloths is not None:
            self._cloths.reset()
        for a in scene.arms:
            robot = self.scene[a.slot]
            robot.write_joint_state_to_sim(robot.data.default_joint_pos, robot.data.default_joint_vel)
            robot.set_joint_position_target(robot.data.default_joint_pos)
        self.scene.write_data_to_sim()
        self.sim.step(render=not self.headless)
        self.scene.update(self.physics_dt)
        self._t = 0.0

    def step(self, commands: Mapping[str, ArmCommand]) -> Frame:
        import torch

        for slot, cmd in commands.items():
            m = get_model(self._scene.arm(slot).model)
            arm_ids, grip_ids = self._joint_ids[slot]
            robot = self.scene[slot]
            robot.set_joint_position_target(torch.tensor([m.clip_arm(cmd.q)], device=self.device), joint_ids=arm_ids)
            robot.set_joint_position_target(torch.tensor([m.gripper_targets(cmd.gripper)], device=self.device),
                                            joint_ids=grip_ids)
        for _ in range(self.decimation):
            self.scene.write_data_to_sim()
            self.sim.step(render=not self.headless)
            self.scene.update(self.physics_dt)
            if self._fits:
                self._release_fits()
            self._hang()
        self._t += self.control_dt
        return self._frame()

    def set_object_pose(self, name: str, pose: Pose7) -> Frame:
        import torch

        p = torch.tensor([pose], device=self.device)
        p[:, :3] += self.scene.env_origins[0]
        self.scene[name].write_root_pose_to_sim(p)
        self.scene[name].write_root_velocity_to_sim(torch.zeros(1, 6, device=self.device))
        return self._frame()

    def set_cloth_points(self, name: str, points) -> Frame:
        if self._cloths is None or name not in self._cloths.specs:
            raise ConfigError(f"no cloth {name!r} in this scene")
        self._cloths.write(name, points)
        return self._frame()

    def render(self, passes: int = RENDER_PASSES) -> dict:
        """RGB pictures (H x W x 3 uint8 numpy arrays) from every camera, of the
        state as it is now.  Physics does not advance, so poses written with
        :meth:`set_object_pose` (or :meth:`set_cloth_points`) are drawn exactly where they were put."""
        if not self.cameras and not self._wrist:
            raise ConfigError("this IsaacEngine was built without cameras")
        for _ in range(passes):
            self.sim.render()
        return self.pictures()

    def pictures(self) -> dict:
        """What every camera sees as last rendered: the fixed ones by name, the wrist ones as ``wrist_<arm>``."""
        out = {}
        cams = [(name, self.scene[f"cam_{name}"]) for name in self.cameras]
        cams += [(f"wrist_{slot}", cam) for slot, cam in self._wrist.items()]
        for name, cam in cams:
            cam.update(0.0, force_recompute=True)
            out[name] = cam.data.output["rgb"][0, ..., :3].cpu().numpy()
        return out

    def _frame(self) -> Frame:
        origin = self.scene.env_origins[0]
        objects = {}
        for o in self._scene.dynamic_objects:
            s = self.scene[o.name].data.root_state_w[0]
            objects[o.name] = tuple((s[:3] - origin).tolist() + s[3:7].tolist())
        arms = {}
        for a in self._scene.arms:
            arm_ids, grip_ids = self._joint_ids[a.slot]
            q = self.scene[a.slot].data.joint_pos[0]
            m = get_model(a.model)
            g = q[grip_ids].tolist()
            span = [o - c for o, c in zip(m.gripper_open, m.gripper_closed)]
            opening = sum((v - c) / s for v, c, s in zip(g, m.gripper_closed, span) if s) / max(len(g), 1)
            arms[a.slot] = {"q": q[arm_ids].tolist(), "gripper": round(opening, 4)}
        cloths = {} if self._cloths is None else {n: self._cloths.marks(n) for n in self._cloths.specs}
        props = {}
        for p in self._scene.props:
            s = self.scene[p.name].data.root_state_w[0]
            props[p.name] = environment.prop_pose(p, (s[:3] - origin).tolist() + s[3:7].tolist())
        return Frame(t=self._t, objects=objects, arms=arms, cloths=cloths, props=props)

    def close(self) -> None:
        app.close()
