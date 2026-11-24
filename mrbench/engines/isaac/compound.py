"""Compound rigid bodies (a door with its handle, a coat hanger) as small USD files.

Isaac Lab spawns a body from one shape config or from a USD file.  A body made
of several shapes is therefore written here as a USD layer -- one Xform with
the rigid-body and mass APIs, one collider per part, each with its own colour
(and friction, if it has one) -- and spawned with ``UsdFileCfg``.  Files are named by a hash of the spec and
``FORMAT``, so a scene rebuilt with the same bodies reuses them.  Call only after ``app.launch()``.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

from ...core.paths import runs_dir
from ..base import ObjectSpec, yaw_quat


#: bump when what ``write`` puts in a file changes: files are reused by name
FORMAT = 3


def usd_dir() -> Path:
    return runs_dir() / "usd"


def write(o: ObjectSpec, directory: Path | None = None) -> Path:
    from pxr import Gf, PhysxSchema, Sdf, Usd, UsdGeom, UsdPhysics, UsdShade

    from .engine import NEVER_SLEEP, linear_rgb

    directory = directory or usd_dir()
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{o.name}-{hashlib.sha1(repr((FORMAT, o)).encode()).hexdigest()[:12]}.usda"
    if path.exists():
        return path
    stage = Usd.Stage.CreateNew(str(path))
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    root = UsdGeom.Xform.Define(stage, "/body")
    stage.SetDefaultPrim(root.GetPrim())
    if o.dynamic:
        UsdPhysics.RigidBodyAPI.Apply(root.GetPrim())
        UsdPhysics.MassAPI.Apply(root.GetPrim()).CreateMassAttr(float(o.mass_kg))
        PhysxSchema.PhysxRigidBodyAPI.Apply(root.GetPrim()).CreateSleepThresholdAttr(float(NEVER_SLEEP))
    for p in o.parts:
        prim = f"/body/{p.name}"
        if p.shape == "box":
            geom, scale = UsdGeom.Cube.Define(stage, prim), p.size
            geom.CreateSizeAttr(1.0)
        elif p.shape == "cylinder":
            geom, scale = UsdGeom.Cylinder.Define(stage, prim), None
            geom.CreateRadiusAttr(float(p.size[0]))
            geom.CreateHeightAttr(float(p.size[1]))
            geom.CreateAxisAttr(UsdGeom.Tokens.z)
        else:
            geom, scale = UsdGeom.Sphere.Define(stage, prim), None
            geom.CreateRadiusAttr(float(p.size[0]))
        xf = UsdGeom.Xformable(geom)
        xf.AddTranslateOp().Set(Gf.Vec3d(*[float(v) for v in p.pos]))
        w, x, y, z = yaw_quat(p.yaw_deg)
        xf.AddOrientOp().Set(Gf.Quatf(w, Gf.Vec3f(x, y, z)))
        if scale:
            xf.AddScaleOp().Set(Gf.Vec3f(*[float(v) for v in scale]))
        UsdPhysics.CollisionAPI.Apply(geom.GetPrim())

        look = UsdShade.Material.Define(stage, f"/body/looks/{p.name}")
        shader = UsdShade.Shader.Define(stage, f"/body/looks/{p.name}/shader")
        shader.CreateIdAttr("UsdPreviewSurface")
        shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(
            Gf.Vec3f(*linear_rgb(p.color or o.color)))
        shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.5)
        look.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")
        binding = UsdShade.MaterialBindingAPI.Apply(geom.GetPrim())
        binding.Bind(look)
        if p.friction is not None:  # its own physics material, bound for physics only
            phys = UsdShade.Material.Define(stage, f"/body/physics/{p.name}")
            api = UsdPhysics.MaterialAPI.Apply(phys.GetPrim())
            api.CreateStaticFrictionAttr(float(p.friction))
            api.CreateDynamicFrictionAttr(float(p.friction))
            PhysxSchema.PhysxMaterialAPI.Apply(phys.GetPrim()).CreateFrictionCombineModeAttr(p.friction_combine)
            binding.Bind(phys, UsdShade.Tokens.weakerThanDescendants, "physics")
    stage.GetRootLayer().Save()
    return path
