"""Cloth on Isaac: a PhysX particle cloth.

A ``ClothSpec`` becomes a triangle mesh of its grid points, each a particle
``radius`` thick, with springs along the triangles' edges; it collides with the
table, the arms and itself.  Particle cloth is solved on the GPU only, which is
where this engine runs anyway.  Positions are read and written through the
tensor view; the USD mesh's own points never change.

Friction between two fingers does not hold a particle cloth: a closing hand
attaches the patch under it to a body fixed at the hand (``puppeteer.py``).

Call only after ``app.launch()``.
"""

from __future__ import annotations

from typing import Sequence

from ..base import Vec3
from ..cloth import ClothSpec

ENV = "/World/envs/env_0"
#: how far beyond a particle's radius a contact starts (the solver's passes per step are the cloth's own,
#: ``ClothSpec.iterations``)
CONTACT_MARGIN = 1.5


def cloth_path(name: str) -> str:
    return f"{ENV}/cloth_{name}"


def spawn(cloth: ClothSpec, origin: Vec3) -> None:
    """Put one cloth on the stage, flat where it starts -- before the simulation starts."""
    import isaacsim.core.utils.stage as stage_utils
    from omni.physx.scripts import particleUtils, physicsUtils
    from pxr import Gf, PhysxSchema, Sdf, UsdGeom, UsdPhysics, UsdShade, Vt

    from ..base import yaw_quat
    from .engine import linear_rgb

    stage = stage_utils.get_current_stage()
    PhysxSchema.PhysxSceneAPI.Apply(stage.GetPrimAtPath("/physicsScene")) \
        .CreateEnableExternalForcesEveryIterationAttr().Set(True)
    path = cloth_path(cloth.name)
    mesh = UsdGeom.Mesh.Define(stage, path)
    mesh.CreatePointsAttr().Set(Vt.Vec3fArray([Gf.Vec3f(float(u), float(v), 0.0) for u, v in cloth.nodes]))
    mesh.CreateFaceVertexCountsAttr().Set([3] * len(cloth.triangles))
    mesh.CreateFaceVertexIndicesAttr().Set([int(i) for t in cloth.triangles for i in t])
    mesh.CreateSubdivisionSchemeAttr().Set("none")
    mesh.CreateDoubleSidedAttr().Set(True)
    physicsUtils.setup_transform_as_scale_orient_translate(mesh)
    physicsUtils.set_or_add_translate_op(mesh, Gf.Vec3f(*[float(p + o) for p, o in zip(cloth.pos, origin)]))
    w, x, y, z = yaw_quat(cloth.yaw_deg)
    physicsUtils.set_or_add_orient_op(mesh, Gf.Quatf(w, Gf.Vec3f(x, y, z)))

    radius = cloth.radius
    system = path + "_particles"
    particleUtils.add_physx_particle_system(
        stage=stage, particle_system_path=Sdf.Path(system), simulation_owner=Sdf.Path("/physicsScene"),
        contact_offset=CONTACT_MARGIN * radius, rest_offset=radius,
        particle_contact_offset=CONTACT_MARGIN * radius, solid_rest_offset=radius,
        fluid_rest_offset=0.0, solver_position_iterations=cloth.iterations)
    material = path + "_material"
    particleUtils.add_pbd_particle_material(stage, Sdf.Path(material), friction=cloth.friction)
    physicsUtils.add_physics_material_to_prim(stage, stage.GetPrimAtPath(system), Sdf.Path(material))
    particleUtils.add_physx_particle_cloth(
        stage=stage, path=Sdf.Path(path), dynamic_mesh_path=None, particle_system_path=Sdf.Path(system),
        spring_stretch_stiffness=cloth.stretch_stiffness, spring_bend_stiffness=cloth.bend_stiffness,
        spring_shear_stiffness=cloth.shear_stiffness, spring_damping=cloth.spring_damping,
        self_collision=True, self_collision_filter=True)
    UsdPhysics.MassAPI.Apply(mesh.GetPrim()).CreateMassAttr().Set(float(cloth.mass_kg))

    look = UsdShade.Material.Define(stage, path + "_look")
    shader = UsdShade.Shader.Define(stage, path + "_look/shader")
    shader.CreateIdAttr("UsdPreviewSurface")
    shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*linear_rgb(cloth.color)))
    shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.9)
    look.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")
    UsdShade.MaterialBindingAPI.Apply(mesh.GetPrim()).Bind(look)


def tie(cloth: ClothSpec, scene) -> None:
    """Hold the cloth's tied points fixed in their bodies' frames, where they are at the start -- after the bodies
    and the cloth are on the stage, before the simulation starts.  (As a hand's grip on a cloth, puppeteer.py: a
    PhysX attachment between the particle cloth and a rigid body.)  A tied body does not collide with its cloth."""
    import isaacsim.core.utils.stage as stage_utils
    from pxr import Gf, PhysxSchema, Sdf, Vt

    from ..base import pose7, quat_rotate

    stage = stage_utils.get_current_stage()
    for i, t in enumerate(cloth.ties):
        if t.hang:  # moved under its point every step (Cloths.hang)
            continue
        body = scene.obj(t.body)
        pose = pose7(body.pos, body.yaw_deg, body.quat)
        back = (pose[3], -pose[4], -pose[5], -pose[6])
        nodes = cloth.tie_nodes(t)
        att = PhysxSchema.PhysxPhysicsAttachment.Define(stage, f"{cloth_path(cloth.name)}_tie_{i}")
        att.GetActor0Rel().SetTargets([Sdf.Path(cloth_path(cloth.name))])
        att.GetActor1Rel().SetTargets([Sdf.Path(f"{ENV}/{t.body}")])
        att.CreatePoints0Attr().Set(Vt.Vec3fArray([Gf.Vec3f(float(cloth.nodes[n][0]), float(cloth.nodes[n][1]), 0.0)
                                                   for n in nodes]))
        att.CreatePoints1Attr().Set(Vt.Vec3fArray([Gf.Vec3f(*[float(v) for v in quat_rotate(
            back, [a - b for a, b in zip(cloth.at(cloth.nodes[n]), pose[:3])])]) for n in nodes]))
        # the body collides with none of the cloth (brushed by the particles next to its own, a light body is flung off)
        att.CreateCollisionFilterIndices0Attr().Set(list(range(len(cloth.nodes))))
        att.CreateFilterType0Attr().Set("Vertices")


class Cloths:
    """The cloths of a built scene, read and moved through PhysX's tensor views."""

    def __init__(self, sim, cloths: Sequence[ClothSpec], origin: Vec3, device: str, scene=None):
        self.specs = {c.name: c for c in cloths}
        self.origin, self.device = tuple(float(v) for v in origin), device
        self._views = {c.name: sim.physics_sim_view.create_particle_cloth_view(cloth_path(c.name)) for c in cloths}
        # the hanging bodies: (cloth, node, body, how far from the node, which way it lay from it, its half height)
        self.hanging = []
        for c in cloths:
            for t in c.ties:
                if t.hang and scene is not None:
                    o = scene.obj(t.body)
                    node = c.tie_nodes(t)[0]
                    p = c.at(c.nodes[node])
                    d = [b - a for a, b in zip(p, o.pos)]
                    h = (d[0] ** 2 + d[1] ** 2) ** 0.5
                    side = (d[0] / h, d[1] / h) if h > 1e-6 else (0.0, -1.0)
                    self.hanging.append((c.name, node, o.name, sum(v * v for v in d) ** 0.5, side, o.height / 2))

    def hang(self, entities, table_top: float) -> None:
        """Put every hanging body under its point: as far below it as it lay from it, or, while that would be in
        the table, on the table beside it, that far from it, the way it lay."""
        import torch

        if not self.hanging:
            return
        pts = {name: self._all(name) for name in {h[0] for h in self.hanging}}
        for cloth, node, body, dist, (sx, sy), half in self.hanging:
            x, y, z = (float(v) for v in pts[cloth][node])
            rest = table_top + half
            if z - dist >= rest:
                pos = (x, y, z - dist)
            else:
                dz = min(dist, z - rest) if z > rest else 0.0
                h = max(dist * dist - dz * dz, 0.0) ** 0.5
                pos = (x + sx * h, y + sy * h, max(rest, z - dz))
            pose = torch.tensor([[pos[0] + self.origin[0], pos[1] + self.origin[1], pos[2] + self.origin[2],
                                  1.0, 0.0, 0.0, 0.0]], device=self.device)
            entities[body].write_root_pose_to_sim(pose)
            entities[body].write_root_velocity_to_sim(torch.zeros(1, 6, device=self.device))

    def _all(self, name: str):
        """Every particle of a cloth now, (n, 3), in the env's frame, on the host."""
        raw = self._views[name].get_positions().detach().cpu().numpy().reshape(-1, 3)
        return raw[: len(self.specs[name].nodes)] - self.origin

    def marks(self, name: str) -> list[Vec3]:
        pts = self._all(name)
        return [tuple(float(v) for v in pts[i]) for i in self.specs[name].marks]  # type: ignore[misc]

    def write(self, name: str, points: Sequence[Vec3]) -> None:
        """Every particle somewhere (``ClothSpec.nodes`` order), at rest."""
        import torch

        view = self._views[name]
        ids = torch.tensor([0], device=self.device, dtype=torch.int32)
        flat = torch.tensor([[float(p[k]) + self.origin[k] for p in points for k in range(3)]],
                            device=self.device, dtype=torch.float32)
        width = int(view.get_positions().reshape(1, -1).shape[-1])
        if flat.shape[-1] < width:  # the view's buffer may be longer than the cloth
            flat = torch.nn.functional.pad(flat, (0, width - flat.shape[-1]))
        view.set_positions(flat, ids)
        view.set_velocities(torch.zeros_like(flat), ids)

    def reset(self) -> None:
        for name, spec in self.specs.items():
            self.write(name, spec.flat_points())
