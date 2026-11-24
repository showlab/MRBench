"""Plays a scripted reference (``policies/script.py``) on the Isaac engine.

* **Arms:** the script is planned first, into joint trajectories (``policies/plan.py``:
  inverse kinematics along every path, checked before anything moves); every physics
  step each arm's joint targets are read off its trajectory.
* **Grips are privileged:** closing on a body fixes it to the hand with a joint (unless
  the grip attaches nothing: a hand pressing a body down only closes its fingers); so
  does a held tool's clip pressed round a body (``script.Clip``: a sponge mop's), if it fits;
  closing on a cloth attaches the particles under the fingertips to a weightless
  *hold* body that a joint fixes at the tool point (a cloth cannot be attached to
  an articulation link directly: that pins the particles to the
  world).  Releasing removes the joint or the attachment.  Everything else -- the
  door springs, the toolbox's weight, the towel -- is physics.
* **Output:** the frames scoring needs (one per control step) and, if asked, a
  video from one camera with the phase's note as a caption.

Call only after ``app.launch()`` and ``engine.load()``.
"""

from __future__ import annotations

import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Optional

from ...robots import get_model
from ..base import Frame, Pose7, quat_mul, quat_rotate
from .cloth import ENV, cloth_path

#: a closing hand takes the cloth particles this close to the point between its fingertips (FINGERTIPS past
#: the tool point, along the fingers): the patch it pinches, its fingertips down on the cloth.  At 4 cm from
#: the tool point, a hand still well above a towel would take it up without touching it.
CLOTH_GRIP_RADIUS, FINGERTIPS = 0.015, 0.01
#: a hand closing on a body takes it only if its tool point is this close to the body: a privileged grip
#: attaches whatever is named, and one 14 cm from the hand would be carried along at that offset.  Nor if the part it
#: closes on is wider across the fingers than they open (``clearance.too_wide``): a 24 cm ball
GRIP_REACH = 0.02
#: a distractor run stops once its task's decoy_stop has held this long (``demo.decoy_protocol``)
DECOY_STOP_HOLD = 1.0
#: a hold: a body with no shape whose only job is to carry attached cloth
HOLD_SIZE, HOLD_MASS, HOLD_INERTIA = 0.01, 0.05, 1.0e-5


def hold_path(slot: str) -> str:
    return f"{ENV}/{slot}_hold"


def _conj(q):
    return (q[0], -q[1], -q[2], -q[3])


def _relative(a: Pose7, b: Pose7) -> Pose7:
    """``b`` in ``a``'s frame."""
    back = _conj(a[3:])
    return (*quat_rotate(back, [y - x for x, y in zip(a[:3], b[:3])]), *quat_mul(back, b[3:]))  # type: ignore[return-value]


def ee_link_path(slot: str, link: str) -> str:
    """The USD path of an arm's link, found by name under the arm's prim."""
    import isaacsim.core.utils.stage as stage_utils
    from pxr import Usd

    root = stage_utils.get_current_stage().GetPrimAtPath(f"{ENV}/{slot}")
    for prim in Usd.PrimRange(root):
        if prim.GetName() == link:
            return str(prim.GetPath())
    raise KeyError(f"{slot} has no link {link!r}")


def spawn_holds(scene) -> None:
    """One hold per arm, fixed at its tool point -- before the simulation starts (only
    scenes with cloth need them)."""
    import isaacsim.core.utils.stage as stage_utils
    from pxr import Gf, PhysxSchema, Sdf, UsdGeom, UsdPhysics

    stage = stage_utils.get_current_stage()
    for a in scene.arms:
        m = get_model(a.model)
        path = hold_path(a.slot)
        cube = UsdGeom.Cube.Define(stage, path)
        cube.CreateSizeAttr(HOLD_SIZE)
        UsdGeom.Xformable(cube).AddTranslateOp().Set(Gf.Vec3d(*[float(v) for v in a.pos]))
        UsdGeom.Imageable(cube).MakeInvisible()
        UsdPhysics.RigidBodyAPI.Apply(cube.GetPrim())
        PhysxSchema.PhysxRigidBodyAPI.Apply(cube.GetPrim()).CreateDisableGravityAttr(True)
        mass = UsdPhysics.MassAPI.Apply(cube.GetPrim())
        mass.CreateMassAttr(HOLD_MASS)
        mass.CreateDiagonalInertiaAttr(Gf.Vec3f(HOLD_INERTIA, HOLD_INERTIA, HOLD_INERTIA))
        joint = UsdPhysics.FixedJoint.Define(stage, path + "_joint")
        joint.CreateBody0Rel().SetTargets([Sdf.Path(ee_link_path(a.slot, m.ee_link))])
        joint.CreateBody1Rel().SetTargets([Sdf.Path(path)])
        joint.CreateLocalPos0Attr(Gf.Vec3f(0.0, 0.0, float(m.tcp_offset_m)))
        joint.CreateLocalRot0Attr(Gf.Quatf(1.0, Gf.Vec3f(0.0, 0.0, 0.0)))
        joint.CreateLocalPos1Attr(Gf.Vec3f(0.0, 0.0, 0.0))
        joint.CreateLocalRot1Attr(Gf.Quatf(1.0, Gf.Vec3f(0.0, 0.0, 0.0)))
        joint.CreateExcludeFromArticulationAttr(True)


class Puppeteer:
    def __init__(self, engine, inst, camera: Optional[str] = None, fps: float = 20.0, size=(960, 720)):
        self.e, self.inst = engine, inst
        self.dt = engine.physics_dt
        self.t = 0.0
        self.frames: list[Frame] = [engine._frame()]
        self.camera = camera
        self.record_every = max(1, round(1.0 / (fps * self.dt)))
        self.fps = 1.0 / (self.record_every * self.dt)
        self._shots: Optional[Path] = Path(tempfile.mkdtemp(prefix="mrbench-video-")) if camera else None
        self._n_shots = 0
        self.note = ""
        #: per phase: how far each arm that moved ended from its target (m, deg)
        self.trace: list[dict] = []
        #: grips that closed on nothing: the body was out of the hand's reach (``GRIP_REACH``)
        self.missed: list[dict] = []
        self.arms = {}
        for a in inst.scene.arms:
            m = get_model(a.model)
            robot = engine.scene[a.slot]
            body = robot.find_bodies(m.ee_link)[0][0]
            arm_ids, grip_ids = engine._joint_ids[a.slot]
            span = sum(abs(o - c) for o, c in zip(m.gripper_open, m.gripper_closed))
            self.arms[a.slot] = {"model": m, "robot": robot, "body": body, "arm_ids": arm_ids, "grip_ids": grip_ids, "opening": 1.0,
                                 "span": span, "held": None, "held_target": None, "link": ee_link_path(a.slot, m.ee_link)}
        #: a tool's clip -> the joint that fixes what it holds to it
        self._clipped: dict[str, str] = {}
        self._holds = {}
        if inst.scene.cloths:
            self._holds = {s: engine.sim.physics_sim_view.create_rigid_body_view(hold_path(s)) for s in self.arms}
            self._place_holds()

    # ------------------------------------------------------------------ reading
    def _origin(self):
        return self.e.scene.env_origins[0]

    def body_pose(self, slot: str) -> Pose7:
        arm = self.arms[slot]
        st = arm["robot"].data.body_state_w[0, arm["body"]]
        return tuple((st[:3] - self._origin()).tolist() + st[3:7].tolist())  # type: ignore[return-value]

    def tcp_pose(self, slot: str) -> Pose7:
        b = self.body_pose(slot)
        off = quat_rotate(b[3:], (0.0, 0.0, self.arms[slot]["model"].tcp_offset_m))
        return (*(p + o for p, o in zip(b[:3], off)), *b[3:])  # type: ignore[return-value]

    def _place_holds(self) -> None:
        """Each hold at its tool point, at rest (they start wherever they were spawned)."""
        import torch

        ids = torch.tensor([0], device=self.e.device, dtype=torch.int32)
        o = self._origin().tolist()
        for slot, view in self._holds.items():
            p = self.tcp_pose(slot)
            w, x, y, z = p[3:]
            view.set_transforms(torch.tensor([[p[0] + o[0], p[1] + o[1], p[2] + o[2], x, y, z, w]],
                                             device=self.e.device), ids)
            view.set_velocities(torch.zeros((1, 6), device=self.e.device), ids)

    # ------------------------------------------------------------------ grips
    def _grip(self, slot: str, target: str, width: float) -> None:
        import isaacsim.core.utils.stage as stage_utils
        from pxr import Gf, PhysxSchema, Sdf, UsdPhysics, Vt

        arm = self.arms[slot]
        arm["opening"] = min(1.0, width / arm["span"]) if arm["span"] else 0.0
        stage = stage_utils.get_current_stage()
        path = f"{ENV}/grip_{slot}"
        tcp = self.tcp_pose(slot)
        cloths = {c.name: c for c in self.inst.scene.cloths}
        if target in cloths:
            import numpy as np

            spec = cloths[target]
            pts = self.e._cloths._all(target)
            tips = np.asarray(tcp[:3]) + FINGERTIPS * np.asarray(quat_rotate(tcp[3:], (0.0, 0.0, 1.0)))
            patch = [int(i) for i in np.where(np.linalg.norm(pts - tips, axis=1) <= CLOTH_GRIP_RADIUS)[0]]
            if not patch:  # the fingers close on nothing: reported, and the run goes on to fail on its own
                self.missed.append({"t": round(self.t, 2), "arm": slot, "target": target,
                                    "gap_m": round(float(np.linalg.norm(pts - tips, axis=1).min()), 3)})
                return
            back = _conj(tcp[3:])
            att = PhysxSchema.PhysxPhysicsAttachment.Define(stage, path)
            att.GetActor0Rel().SetTargets([Sdf.Path(cloth_path(target))])
            att.GetActor1Rel().SetTargets([Sdf.Path(hold_path(slot))])
            att.CreatePoints0Attr().Set(Vt.Vec3fArray([Gf.Vec3f(float(spec.nodes[i][0]), float(spec.nodes[i][1]), 0.0)
                                                       for i in patch]))
            att.CreatePoints1Attr().Set(Vt.Vec3fArray([Gf.Vec3f(*[float(v) for v in quat_rotate(
                back, [a - b for a, b in zip(pts[i], tcp[:3])])]) for i in patch]))
            att.CreateCollisionFilterIndices0Attr().Set(patch)
            att.CreateFilterType0Attr().Set("Vertices")
        else:
            obj = self.e._frame().objects[target]
            if self._gap(target, obj, tcp) > GRIP_REACH:  # the fingers close on nothing
                self.missed.append({"t": round(self.t, 2), "arm": slot, "target": target,
                                    "gap_m": round(self._gap(target, obj, tcp), 3)})
                return
            from ...policies.clearance import FIT_TOL
            from ...robots import collision as col

            # the fingers close across the tool's y: a part wider than they open is not taken (a 24 cm ball) -- unless
            # another hand is on it, on its far side: then the two hands hold it between them, and the grip attaches
            # it (to this hand), until either lets go
            w = col.grip_span(self.inst.scene.obj(target), obj, tcp[:3], quat_rotate(tcp[3:], (0.0, 1.0, 0.0)),
                              GRIP_REACH)
            widest = arm["model"].grip_open_m
            partner = None
            if w is not None and widest and w > widest + FIT_TOL:
                on_it = {o: self.tcp_pose(o)[:3] for o in self.arms
                         if o != slot and self._gap(target, obj, self.tcp_pose(o)) <= GRIP_REACH}
                partner = next((o for o, p in on_it.items() if col.squeezed(obj[:3], tcp[:3], [p])), None)
                if partner is None:
                    self.missed.append({"t": round(self.t, 2), "arm": slot, "target": target,
                                        "too_wide_m": round(w, 3), "fingers_open_m": widest})
                    return
            arm["squeezed_with"] = partner
            rel = _relative(self.body_pose(slot), obj)
            joint = UsdPhysics.FixedJoint.Define(stage, path)
            joint.CreateBody0Rel().SetTargets([Sdf.Path(arm["link"])])
            joint.CreateBody1Rel().SetTargets([Sdf.Path(f"{ENV}/{target}")])
            joint.CreateLocalPos0Attr(Gf.Vec3f(*[float(v) for v in rel[:3]]))
            joint.CreateLocalRot0Attr(Gf.Quatf(float(rel[3]), Gf.Vec3f(*[float(v) for v in rel[4:]])))
            joint.CreateLocalPos1Attr(Gf.Vec3f(0.0, 0.0, 0.0))
            joint.CreateLocalRot1Attr(Gf.Quatf(1.0, Gf.Vec3f(0.0, 0.0, 0.0)))
            joint.CreateExcludeFromArticulationAttr(True)
        arm["held"] = path
        arm["held_target"] = target

    def _clip(self, tool: str, target: str) -> None:
        """A tool's clip takes ``target`` -- fixed to the tool with a joint, as a grip fixes a body to a hand -- if it
        is within a grip's reach of the point between the jaws and fits between them (``collision.clip_takes``)."""
        import isaacsim.core.utils.stage as stage_utils
        from pxr import Gf, Sdf, UsdPhysics

        from ...policies.clearance import FIT_TOL
        from ...robots import collision as col

        if not any(arm.get("held_target") == tool for arm in self.arms.values()):
            self.missed.append({"t": round(self.t, 2), "tool": tool, "target": target,
                                "why": "the tool must be held before operating its clip"})
            return
        objs = self.e._frame().objects
        sc = self.inst.scene
        ok, why = col.clip_takes(sc.obj(tool), objs[tool], sc.obj(target), objs[target], GRIP_REACH, FIT_TOL)
        if not ok:  # the clip closes on nothing: reported, and the run goes on to fail on its own
            self.missed.append({"t": round(self.t, 2), "tool": tool, "target": target, "why": why})
            return
        path = f"{ENV}/clip_{tool}"
        rel = _relative(objs[tool], objs[target])
        joint = UsdPhysics.FixedJoint.Define(stage_utils.get_current_stage(), path)
        joint.CreateBody0Rel().SetTargets([Sdf.Path(f"{ENV}/{tool}")])
        joint.CreateBody1Rel().SetTargets([Sdf.Path(f"{ENV}/{target}")])
        joint.CreateLocalPos0Attr(Gf.Vec3f(*[float(v) for v in rel[:3]]))
        joint.CreateLocalRot0Attr(Gf.Quatf(float(rel[3]), Gf.Vec3f(*[float(v) for v in rel[4:]])))
        joint.CreateLocalPos1Attr(Gf.Vec3f(0.0, 0.0, 0.0))
        joint.CreateLocalRot1Attr(Gf.Quatf(1.0, Gf.Vec3f(0.0, 0.0, 0.0)))
        self._clipped[tool] = path

    def _unclip(self, tool: str) -> None:
        import isaacsim.core.utils.stage as stage_utils

        if not any(arm.get("held_target") == tool for arm in self.arms.values()):
            self.missed.append({"t": round(self.t, 2), "tool": tool,
                                "why": "the tool must be held before releasing its clip"})
            return
        path = self._clipped.pop(tool, None)
        if path:
            stage_utils.get_current_stage().RemovePrim(path)

    def _gap(self, name: str, pose: Pose7, tcp: Pose7) -> float:
        """How far the tool point is from the body's surface (0 inside it)."""
        from ...robots import collision as col
        from ...robots import kinematics as kin

        p = kin.np.array(tcp[:3])
        point = col.Capsule("tcp", p, p, 0.0)
        return max(0.0, min(col.capsule_box(point, b) for b in col.boxes_of(self.inst.scene.obj(name), pose)))

    def _release(self, slot: str) -> None:
        import isaacsim.core.utils.stage as stage_utils

        arm = self.arms[slot]
        if arm["held"]:
            stage_utils.get_current_stage().RemovePrim(arm["held"])
            arm["held"] = None
        arm["held_target"] = None
        arm["opening"] = 1.0
        # a body held between two hands drops when either lets go
        for other in self.arms.values():
            if other is not arm and other.get("squeezed_with") == slot and other["held"]:
                stage_utils.get_current_stage().RemovePrim(other["held"])
                other["held"], other["squeezed_with"] = None, None
                other["held_target"] = None

    # ------------------------------------------------------------------ playing
    def _step(self, plan, t_next: float) -> None:
        import torch

        for slot, arm in self.arms.items():
            arm["robot"].set_joint_position_target(torch.tensor([plan.arms[slot].at(t_next)], device=self.e.device),
                                                   joint_ids=arm["arm_ids"])
            arm["robot"].set_joint_position_target(
                torch.tensor([arm["model"].gripper_targets(arm["opening"])], device=self.e.device),
                joint_ids=arm["grip_ids"])
        n = round((self.t + self.dt) / self.dt)
        shoot = self._shots is not None and n % self.record_every == 0
        self.e.scene.write_data_to_sim()
        # a frame is rendered inside the physics step that ends at it: rendered afterwards on its own
        # (``sim.render()``), the arms would be drawn where they started while everything else moved
        self.e.sim.step(render=shoot)
        self.e.scene.update(self.dt)
        if self.e._fits:  # a tight fit pulled out of its seat lets go
            self.e._release_fits()
        self.e._hang()  # lanterns on a cord hang under it
        self.t += self.dt
        if n % self.e.decimation == 0:
            self.e._t = self.t
            self.frames.append(self.e._frame())
        if shoot:
            self._shoot()

    def _off_plan(self, planned: dict) -> dict:
        """How far each arm's tool point is from where the plan has it (m, deg)."""
        import math

        out = {}
        for slot, goal in planned.items():
            now = self.tcp_pose(slot)
            dot = min(1.0, abs(sum(a * b for a, b in zip(now[3:], goal[3:]))))
            out[slot] = (round(math.dist(now[:3], goal[:3]), 4), round(math.degrees(2 * math.acos(dot)), 1))
        return out

    def play(self, script) -> list[Frame]:
        """Plan the script, then play it: joint targets from the plan, grips at their times."""
        from ...policies.plan import plan as make_plan
        from ...policies.script import Clip, Grip, Open, Unclip

        from ...policies.clearance import check

        plan = make_plan(self.inst, script)
        clashes = check(self.inst, plan)
        self.clashes = clashes
        # strict: a plan that brings anything too close is not played -- except a distractor run (the borrow version
        # with its props and distractors swapped, ``metadata["decoy"]``), whose distractor is meant to meet what it
        # cannot do: its clashes are reported (demo.decoy_protocol) and judged there, not refused here
        if clashes and not self.inst.metadata.get("decoy"):
            from ...policies.plan import PlanError

            raise PlanError("the plan does not keep clear:\n  " + "\n  ".join(map(str, clashes)))
        events = sorted(plan.events, key=lambda e: e[0])
        steps = max(1, round(plan.duration / self.dt))
        ei, pi = 0, 0
        # a distractor run may stop safely once its declared failure shows (``demo.decoy_protocol``): the task's
        # ``decoy_stop(inst)`` predicate, armed once it has been false, holding DECOY_STOP_HOLD seconds -- a script
        # written for the prop would otherwise go on as if it had worked
        stop, armed, since, self.safe_stop = None, False, None, None
        if self.inst.metadata.get("decoy"):
            import importlib

            impl = importlib.import_module(self.inst.spec.impl)
            stop = impl.decoy_stop(self.inst) if hasattr(impl, "decoy_stop") else None
        self.note = plan.phases[0][2] if plan.phases else ""
        for k in range(steps):
            now = k * self.dt
            while pi + 1 < len(plan.phases) and plan.phases[pi + 1][0] <= now + 1e-9:
                self.trace.append({"t": round(self.t, 2), "note": plan.phases[pi][2],
                                   "off_plan": self._off_plan(plan.tcp_at_phase_end[pi]),
                                   "tcp": {s: [round(v, 3) for v in self.tcp_pose(s)[:3]] for s in self.arms},
                                   "planned": {s: [round(v, 3) for v in p[:3]]
                                               for s, p in plan.tcp_at_phase_end[pi].items()}})
                pi += 1
                self.note = plan.phases[pi][2]
            while ei < len(events) and events[ei][0] <= now + 1e-9:
                act = events[ei][1]
                if isinstance(act, Grip) and act.attach:
                    self._grip(act.slot, act.target, act.width)
                elif isinstance(act, Grip):  # a hand that presses or pushes: the fingers only close
                    arm = self.arms[act.slot]
                    arm["opening"] = min(1.0, act.width / arm["span"]) if arm["span"] else 0.0
                elif isinstance(act, Open):
                    arm = self.arms[act.slot]
                    arm["opening"] = min(1.0, act.width / arm["span"]) if arm["span"] else 1.0
                elif isinstance(act, Clip):
                    self._clip(act.tool, act.target)
                elif isinstance(act, Unclip):
                    self._unclip(act.tool)
                else:
                    self._release(act.slot)
                ei += 1
            self._step(plan, now + self.dt)
            if stop is not None and self.frames:
                f = self.frames[-1]
                if not stop(f, self.inst.scene):
                    armed, since = True, None
                elif armed:
                    since = f.t if since is None else since
                    if f.t - since >= DECOY_STOP_HOLD:
                        self.safe_stop = round(f.t, 2)
                        break
        self.trace.append({"t": round(self.t, 2), "note": plan.phases[pi][2],
                           "off_plan": self._off_plan(plan.tcp_at_phase_end[pi])})
        return self.frames

    # ------------------------------------------------------------------ video
    def _shoot(self) -> None:
        from PIL import Image, ImageDraw, ImageFont

        pics = self.e.pictures()
        img = Image.fromarray(pics[self.camera])
        draw = ImageDraw.Draw(img)
        try:
            font = ImageFont.truetype("DejaVuSans.ttf", 22)
            small = ImageFont.truetype("DejaVuSans.ttf", 16)
        except OSError:
            font = small = ImageFont.load_default()
        text = f"{self.inst.spec.id} {self.inst.level}  t={self.t:4.1f}s  {self.note}"
        x0, y0, x1, y1 = draw.textbbox((10, 8), text, font=font)
        draw.rectangle((0, 0, x1 + 10, y1 + 8), fill=(30, 30, 30))
        draw.text((10, 8), text, fill=(255, 255, 255), font=font)
        wrists = [(s, pics[f"wrist_{s}"]) for s in self.arms if f"wrist_{s}" in pics]
        if wrists:  # a row under the main view: each arm's wrist camera, west to east
            wrists.sort(key=lambda sp: self.inst.scene.arm(sp[0]).pos[0])
            w = img.width // len(wrists)
            h = round(w * wrists[0][1].shape[0] / wrists[0][1].shape[1])
            sheet = Image.new("RGB", (img.width, img.height + h), (30, 30, 30))
            sheet.paste(img, (0, 0))
            for i, (slot, pic) in enumerate(wrists):
                tile = Image.fromarray(pic).resize((w, h))
                ImageDraw.Draw(tile).text((6, 4), f"{slot} wrist", fill=(255, 255, 255), font=small)
                sheet.paste(tile, (i * w, img.height))
            img = sheet
        img.save(self._shots / f"{self._n_shots:05d}.png")
        self._n_shots += 1

    def save_video(self, path: Path) -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        # ffmpeg reads the terminal for keys unless told not to: run in the background (under timeout, nohup, &) it is
        # stopped there by SIGTTIN, and the demo with it, for good
        subprocess.run(["ffmpeg", "-nostdin", "-y", "-loglevel", "error", "-framerate", f"{self.fps:.3f}",
                        "-i", str(self._shots / "%05d.png"), "-c:v", "libx264", "-pix_fmt", "yuv420p",
                        "-crf", "23", "-movflags", "+faststart", str(path)], stdin=subprocess.DEVNULL, check=True)
        shutil.rmtree(self._shots, ignore_errors=True)
        return path

