"""What surrounds the task on Isaac: the room, the light, the table, the floor and the props.

``SceneSpec.look`` names them (``scenery`` says what each name is); this module
builds them:

* **room** -- Isaac's ``Simple_Room`` at full size round the table (its own table,
  3.19 x 1.63 m, is nearly the benchmark's; it is hidden, and so is its dome light): walls, a
  window to the north and a ceiling light.  ``studio``: no walls, the photo
  studio's backdrop seen all round.  ``plain``: a grey floor,
  even sky and one distant light;
* **light** -- a photo studio's HDRI (Poly Haven ``brown_photostudio_02``) on the
  dome, its strength, colour temperature and turn from ``scenery.LIGHTS``;
* **table** -- a top ``top_m`` thick on four cast-aluminium legs (NVIDIA's
  material), or a block to the floor; its finish an OmniPBR material, photographs
  projected in world space so that no part's scale stretches the grain;
* **floor** -- a slab over the room's floor with a finish of its own, or the room's;
* **backdrop** -- a furnished corner behind the table (``scenery.backdrop``): shapes and YCB scans, drawn only (no
  collider, no body), under ``/World/backdrop``;
* **props** -- each a rigid body: a YCB scan from the asset server, made physical
  here (Isaac ships most of them as pictures only: no body, no collider), or a
  compound of shapes (``compound.write``).

RTX loads every material and texture before it draws a frame (``RTX_SETTINGS``),
and :func:`wait_for_assets` renders until nothing is loading.  Call only after
``app.launch()``.
"""

from __future__ import annotations

from ...scenery import FLOORS, LIGHTS, PROPS, TABLES, Finish
from ...scenery import backdrop, patterns
from ...scenery.downloads import hdri, texture
from ...scenery.props import UP_QUAT
from ..base import ObjectSpec, PropSpec, SceneSpec, quat_mul, quat_rotate, yaw_quat

ROOM_USD = "Environments/Simple_Room/simple_room.usd"
ROOM_PRIM = "/World/room"
BACKDROP_PRIM = "/World/backdrop"
#: the room's floor, below its origin
ROOM_FLOOR = -0.77
#: its own table, dome light and ground plane, taken out
ROOM_HIDDEN = ("table_low_327", "DomeLight", "GroundPlane")
#: its inside, along x and y (m), where a floor of our own covers its floor
ROOM_INSIDE = ((-4.4, 4.4), (-3.3, 4.85))
#: the dome's strength with the studio light at 1 (the HDRI's own units), in the room and in the studio
HDRI_INTENSITY = {"room": 1000.0, "studio": 1000.0}
#: the plain look: a dim dome for fill and a distant light from above and a little south-west, for shadows
PLAIN_DOME, PLAIN_SUN = 300.0, 600.0
PLAIN_SUN_ROT = (0.9659, 0.1830, -0.1830, 0.0)
LEG_MATERIAL = "Materials/Base/Metals/Aluminum_Cast.mdl"
LEG_RADIUS, LEG_INSET = 0.025, 0.08
#: the "frame" form: 80 mm square profiles for legs; 40 x 80 mm rails under the top and 15 cm above the floor;
#: anodised aluminium (satin: metallic, it mirrors the dim room and renders near black)
FRAME_LEG, FRAME_RAIL, FRAME_LOW = 0.08, (0.04, 0.08), 0.15
FRAME_FINISH = Finish(colour=(0.74, 0.75, 0.77), roughness=0.45)
#: a floor slab's thickness; how far its top stands over the room's floor
FLOOR_T, FLOOR_LIFT = 0.02, 0.002
#: Raw RTX settings, written with carb once the simulation context exists (Isaac Lab 2.1's ``RenderCfg``
#: refuses keys it does not know).  No automatic exposure: a frame's brightness must not depend on the
#: frames before it.  No flat ambient fill: the HDRI lights the scene.  Materials and textures load before a
#: frame is drawn, never stream in over later ones (a part could otherwise be drawn as noise for a whole video).
RTX_SETTINGS = {
    "/rtx/post/histogram/enabled": False,
    "/rtx/sceneDb/ambientLightIntensity": 0.0,
    "/rtx/materialDb/syncLoads": True,
    # a surface's opacity below 1 drawn as partly see-through (a glass guard), not as a cut-out
    "/rtx/raytracing/fractionalCutoutOpacity": True,
    "/rtx/translucency/enabled": True,
    "/rtx/hydra/materialSyncLoads": True,
    "/rtx-transient/resourcemanager/texturestreaming/async": False,
}


def apply_rtx_settings() -> None:
    import carb

    settings = carb.settings.get_settings()
    for key, value in RTX_SETTINGS.items():
        settings.set(key, value)


def wait_for_assets(sim, limit: int = 600) -> int:
    """Render until every material and texture has loaded; how many frames that took."""
    try:
        from isaacsim.core.simulation_manager import SimulationManager
    except ImportError:
        return 0
    n = 0
    while SimulationManager.assets_loading() and n < limit:
        sim.render()
        n += 1
    return n


def _linear(rgb):
    from .engine import linear_rgb

    return linear_rgb(rgb)


# ------------------------------------------------------------------ the scene's entities

def scene_cfgs(scene: SceneSpec) -> dict:
    """The room, its light, the floor and the table, as Isaac Lab scene entities by name."""
    import isaaclab.sim as sim_utils
    from isaaclab.assets import AssetBaseCfg
    from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR, NVIDIA_NUCLEUS_DIR

    look = scene.look
    light = LIGHTS[look.light]
    turn = yaw_quat(light.turn_deg)
    out = {}
    if look.room == "plain":
        out["light"] = AssetBaseCfg(prim_path="/World/light",
                                    spawn=sim_utils.DomeLightCfg(intensity=PLAIN_DOME * light.scale))
        out["sun"] = AssetBaseCfg(prim_path="/World/sun",
                                  spawn=sim_utils.DistantLightCfg(intensity=PLAIN_SUN * light.scale, angle=2.0),
                                  init_state=AssetBaseCfg.InitialStateCfg(rot=quat_mul(turn, PLAIN_SUN_ROT)))
    else:
        out["light"] = AssetBaseCfg(
            prim_path="/World/light",
            spawn=sim_utils.DomeLightCfg(intensity=HDRI_INTENSITY[look.room] * light.scale, texture_file=str(hdri()),
                                         texture_format="latlong", visible_in_primary_ray=True),
            init_state=AssetBaseCfg.InitialStateCfg(rot=turn))
    if look.room == "room":
        out["room"] = AssetBaseCfg(
            prim_path=ROOM_PRIM, spawn=sim_utils.UsdFileCfg(usd_path=f"{ISAAC_NUCLEUS_DIR}/{ROOM_USD}"),
            init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, -ROOM_FLOOR)))

    # the floor: what the scene stands on, drawn with its finish unless it is the room's own
    if look.room == "room":
        (x0, x1), (y0, y1) = ROOM_INSIDE
        size, centre, top = (x1 - x0, y1 - y0), ((x0 + x1) / 2, (y0 + y1) / 2), FLOOR_LIFT
    else:
        size, centre, top = ((8.0, 8.0) if look.room == "plain" else (24.0, 24.0)), (0.0, 0.0), 0.0
    out["ground"] = AssetBaseCfg(
        prim_path="/World/ground",
        spawn=sim_utils.CuboidCfg(size=(*size, FLOOR_T), collision_props=sim_utils.CollisionPropertiesCfg(),
                                  visible=look.floor != "room"),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(*centre, top - FLOOR_T / 2)))

    tx, ty, tz = scene.table_size
    top_m = tz if look.table_form == "block" else look.top_m
    out["table"] = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/table",
        spawn=sim_utils.CuboidCfg(size=(tx, ty, top_m), collision_props=sim_utils.CollisionPropertiesCfg()),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, scene.table_top_z - top_m / 2)))
    if look.table_form == "legs":
        height = scene.table_top_z - top_m
        metal = sim_utils.MdlFileCfg(mdl_path=f"{NVIDIA_NUCLEUS_DIR}/{LEG_MATERIAL}")
        for i, (sx, sy) in enumerate(((1, 1), (1, -1), (-1, 1), (-1, -1))):
            out[f"table_leg{i}"] = AssetBaseCfg(
                prim_path=f"{{ENV_REGEX_NS}}/table_leg{i}",
                spawn=sim_utils.CylinderCfg(radius=LEG_RADIUS, height=height, visual_material=metal),
                init_state=AssetBaseCfg.InitialStateCfg(
                    pos=(sx * (tx / 2 - LEG_INSET), sy * (ty / 2 - LEG_INSET), height / 2)))
    elif look.table_form == "frame":  # square profiles for legs, rails under the top's edges and low along its sides
        under = scene.table_top_z - top_m
        lx, ly = tx / 2 - LEG_INSET, ty / 2 - LEG_INSET
        parts = [(f"table_leg{i}", (sx * lx, sy * ly, under / 2), (FRAME_LEG, FRAME_LEG, under))
                 for i, (sx, sy) in enumerate(((1, 1), (1, -1), (-1, 1), (-1, -1)))]
        parts += [(f"table_rail{i}", (0.0, s * ly, under - FRAME_RAIL[1] / 2), (2 * lx, *FRAME_RAIL))
                  for i, s in enumerate((1, -1))]
        parts += [(f"table_rail{2 + i}", (s * lx, 0.0, under - FRAME_RAIL[1] / 2),
                   (FRAME_RAIL[0], 2 * ly, FRAME_RAIL[1])) for i, s in enumerate((1, -1))]
        parts += [(f"table_rail{4 + i}", (0.0, s * ly, FRAME_LOW), (2 * lx, *FRAME_RAIL))
                  for i, s in enumerate((1, -1))]
        for name, pos, size in parts:
            out[name] = AssetBaseCfg(prim_path=f"{{ENV_REGEX_NS}}/{name}", spawn=sim_utils.CuboidCfg(size=size),
                                     init_state=AssetBaseCfg.InitialStateCfg(pos=pos))
    return out


def dress(stage, scene: SceneSpec, env: str = "/World/envs/env_0") -> None:
    """After the scene is built, before the simulation starts: take out what of the room is not the benchmark's, give the
    table and the floor their finishes, and set the light's colour temperature."""
    from pxr import UsdLux, UsdShade

    look = scene.look
    for name in ROOM_HIDDEN if look.room == "room" else ():
        # deactivated, not just hidden: the room's own table, hidden, still collides -- 3 cm above the benchmark's, it would
        # hold up whatever stood in the middle of the table
        prim = stage.GetPrimAtPath(f"{ROOM_PRIM}/{name}")
        if prim.IsValid():
            prim.SetActive(False)
    bind = UsdShade.MaterialBindingAPI
    table = stage.GetPrimAtPath(f"{env}/table")
    bind.Apply(table).Bind(material(stage, "/World/Looks/table_top", TABLES[look.table]),
                           UsdShade.Tokens.strongerThanDescendants)
    if look.table_form == "frame":
        frame = material(stage, "/World/Looks/table_frame", FRAME_FINISH)
        for prim in stage.GetPrimAtPath(env).GetChildren():
            if prim.GetName().startswith(("table_leg", "table_rail")):
                bind.Apply(prim).Bind(frame, UsdShade.Tokens.strongerThanDescendants)
    if look.floor != "room":
        ground = stage.GetPrimAtPath("/World/ground")
        bind.Apply(ground).Bind(material(stage, "/World/Looks/floor", FLOORS[look.floor]),
                                UsdShade.Tokens.strongerThanDescendants)
    build_backdrop(stage, scene)
    kelvin = LIGHTS[look.light].kelvin
    for path in ("/World/light", "/World/sun"):
        prim = stage.GetPrimAtPath(path)
        if prim.IsValid():
            light = UsdLux.LightAPI(prim)
            light.CreateEnableColorTemperatureAttr(True)
            light.CreateColorTemperatureAttr(float(kelvin))


def build_backdrop(stage, scene: SceneSpec) -> None:
    """The look's backdrop (``scenery.backdrop``) under ``BACKDROP_PRIM``: its shapes, each in its finish, and its
    scans, referenced from the asset server; nothing of it collides or moves."""
    from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
    from pxr import Gf, Usd, UsdGeom, UsdLux, UsdPhysics, UsdShade

    pieces, ornaments = backdrop.build(scene.look.backdrop, scene.table_size)
    if not pieces and not ornaments:
        return
    UsdGeom.Xform.Define(stage, BACKDROP_PRIM)
    finishes, materials = backdrop.finishes(), {}
    for piece in pieces:
        path = f"{BACKDROP_PRIM}/{piece.name}"
        scale = (1.0, 1.0, 1.0)
        if piece.shape == "box":
            geom = UsdGeom.Cube.Define(stage, path)
            geom.CreateSizeAttr(1.0)
            scale = piece.size
        elif piece.shape == "cylinder":
            geom = UsdGeom.Cylinder.Define(stage, path)
            geom.CreateRadiusAttr(piece.size[0])
            geom.CreateHeightAttr(piece.size[1])
            geom.CreateAxisAttr(UsdGeom.Tokens.z)
        else:
            geom = UsdGeom.Sphere.Define(stage, path)
            geom.CreateRadiusAttr(piece.size[0])
        xf = UsdGeom.Xformable(geom)
        xf.AddTranslateOp().Set(Gf.Vec3d(*piece.pos))
        xf.AddRotateZOp().Set(float(piece.yaw_deg))
        xf.AddScaleOp().Set(Gf.Vec3f(*scale))
        if piece.finish not in materials:
            materials[piece.finish] = material(stage, f"/World/Looks/backdrop_{piece.finish}", finishes[piece.finish])
        UsdShade.MaterialBindingAPI.Apply(geom.GetPrim()).Bind(materials[piece.finish],
                                                               UsdShade.Tokens.strongerThanDescendants)
    for orn in ornaments:
        prop = PROPS[orn.kind]
        pose = prop_root_pose(PropSpec(orn.name, orn.kind, orn.pos, orn.yaw_deg, prop.size, 0.0))
        holder = UsdGeom.Xform.Define(stage, f"{BACKDROP_PRIM}/{orn.name}")
        xf = UsdGeom.Xformable(holder)
        xf.AddTranslateOp().Set(Gf.Vec3d(*pose[:3]))
        xf.AddOrientOp().Set(Gf.Quatf(float(pose[3]), Gf.Vec3f(*map(float, pose[4:]))))
        # the file's root has its own transform: referenced one level down, under a holder Xform
        stage.DefinePrim(f"{BACKDROP_PRIM}/{orn.name}/scan").GetReferences().AddReference(
            f"{ISAAC_NUCLEUS_DIR}/{prop.usd}")
    for lamp in backdrop.lamps(scene.look.backdrop, scene.table_size):
        light = UsdLux.RectLight.Define(stage, f"{BACKDROP_PRIM}/{lamp.name}")
        light.CreateWidthAttr(lamp.size[0])
        light.CreateHeightAttr(lamp.size[1])
        light.CreateIntensityAttr(lamp.intensity * LIGHTS[scene.look.light].scale)
        light.CreateEnableColorTemperatureAttr(True)
        light.CreateColorTemperatureAttr(float(LIGHTS[scene.look.light].kelvin))
        UsdGeom.Xformable(light).AddTranslateOp().Set(Gf.Vec3d(*lamp.pos))
        if lamp.cone_deg < 90.0:
            shaping = UsdLux.ShapingAPI.Apply(light.GetPrim())
            shaping.CreateShapingConeAngleAttr(float(lamp.cone_deg))
            shaping.CreateShapingConeSoftnessAttr(0.3)
    # drawn only, whatever a file brings
    root = stage.GetPrimAtPath(BACKDROP_PRIM)
    for q in Usd.PrimRange(root):
        if q.IsInstanceable():
            q.SetInstanceable(False)
    for q in Usd.PrimRange(root):
        for api in (UsdPhysics.RigidBodyAPI, UsdPhysics.CollisionAPI, UsdPhysics.MassAPI):
            if q.HasAPI(api):
                q.RemoveAPI(api)


def material(stage, path: str, finish: Finish):
    """An OmniPBR material for ``finish`` at ``path``: photographs or drawn maps projected in world space, or a
    colour."""
    from pxr import Gf, Sdf, UsdShade

    mat = UsdShade.Material.Define(stage, path)
    shader = UsdShade.Shader.Define(stage, f"{path}/Shader")
    shader.SetSourceAsset("OmniPBR.mdl", "mdl")
    shader.SetSourceAssetSubIdentifier("OmniPBR", "mdl")
    shader.CreateIdAttr("OmniPBR")

    def put(name, type_, value):
        shader.CreateInput(name, type_).Set(value)

    if finish.maps:
        for name, part, space in (("diffuse_texture", "diff", "sRGB"), ("normalmap_texture", "nor_gl", "raw"),
                                  ("reflectionroughness_texture", "rough", "raw")):
            inp = shader.CreateInput(name, Sdf.ValueTypeNames.Asset)
            kind, _, name = finish.maps.rpartition(":")
            inp.Set(Sdf.AssetPath(str(patterns.texture(name, part) if kind == "pattern" else texture(name, part))))
            inp.GetAttr().SetColorSpace(space)
        put("reflection_roughness_texture_influence", Sdf.ValueTypeNames.Float, 1.0)
        put("diffuse_tint", Sdf.ValueTypeNames.Color3f, Gf.Vec3f(*finish.tint))
        put("project_uvw", Sdf.ValueTypeNames.Bool, True)
        put("world_or_object", Sdf.ValueTypeNames.Bool, True)
        put("texture_scale", Sdf.ValueTypeNames.Float2, Gf.Vec2f(finish.repeat, finish.repeat))
        if finish.metallic:
            put("metallic_constant", Sdf.ValueTypeNames.Float, 1.0)
    else:
        put("diffuse_color_constant", Sdf.ValueTypeNames.Color3f, Gf.Vec3f(*_linear(finish.colour)))
        put("reflection_roughness_constant", Sdf.ValueTypeNames.Float, float(finish.roughness))
        if finish.metallic:
            put("metallic_constant", Sdf.ValueTypeNames.Float, 1.0)
    out = shader.CreateOutput("out", Sdf.ValueTypeNames.Token)
    for output in (mat.CreateSurfaceOutput("mdl"), mat.CreateVolumeOutput("mdl"), mat.CreateDisplacementOutput("mdl")):
        output.ConnectToSource(out)
    return mat


# ------------------------------------------------------------------ props

def prop_root_pose(p: PropSpec) -> tuple:
    """Where a prop's body frame is (x y z qw qx qy qz): a scan's file frame turned upright, its bounding box's
    centre at ``p.pos``; a built prop's frame is its box's centre."""
    prop = PROPS[p.kind]
    q = quat_mul(yaw_quat(p.yaw_deg), UP_QUAT[prop.up]) if prop.usd else yaw_quat(p.yaw_deg)
    c = quat_rotate(q, prop.centre)
    return (*(a - b for a, b in zip(p.pos, c)), *q)


def prop_pose(p: PropSpec, root) -> tuple:
    """From a prop's body frame where it is now (``root``: x y z qw qx qy qz) to where its upright bounding box
    is -- ``PropSpec.pos`` and its turn, as the mock engine reports a prop."""
    prop = PROPS[p.kind]
    q = tuple(root[3:])
    c = quat_rotate(q, prop.centre)
    if prop.usd:
        w, x, y, z = UP_QUAT[prop.up]
        q = quat_mul(q, (w, -x, -y, -z))
    return (*(a + b for a, b in zip(root[:3], c)), *q)


def prop_cfg(p: PropSpec, prim_path: str):
    """A prop as a rigid object."""
    import isaaclab.sim as sim_utils
    from isaaclab.assets import RigidObjectCfg
    from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR

    from .engine import NEVER_SLEEP

    prop = PROPS[p.kind]
    rigid = sim_utils.RigidBodyPropertiesCfg(sleep_threshold=NEVER_SLEEP)
    if prop.usd:
        spawn = scan_cfg_class()(usd_path=f"{ISAAC_NUCLEUS_DIR}/{prop.usd}", rigid_props=rigid,
                                 mass_props=sim_utils.MassPropertiesCfg(mass=p.mass_kg))
    else:
        from . import compound

        body = ObjectSpec(p.kind.replace(":", "_"), "compound", prop.size, (0.0, 0.0, 0.0), 0.0, prop.colour,
                          p.mass_kg, True, prop.parts)
        spawn = sim_utils.UsdFileCfg(usd_path=str(compound.write(body)))
    pose = prop_root_pose(p)
    return RigidObjectCfg(prim_path=prim_path, spawn=spawn,
                          init_state=RigidObjectCfg.InitialStateCfg(pos=tuple(pose[:3]), rot=tuple(pose[3:])))


def spawn_scan(prim_path, cfg, translation=None, orientation=None, **kwargs):
    """A scan from its USD file, made a rigid body with a convex-hull collider on each mesh: Isaac's YCB props
    come as meshes alone, and Isaac Lab's USD spawner only changes physics a file already has."""
    import isaaclab.sim as sim_utils
    import isaacsim.core.utils.stage as stage_utils
    from isaaclab.sim import schemas
    from isaaclab.sim.spawners.from_files.from_files import spawn_from_usd
    from pxr import Usd, UsdGeom, UsdPhysics

    from ...core.errors import ConfigError

    prim = spawn_from_usd(prim_path, cfg, translation=translation, orientation=orientation)
    stage = stage_utils.get_current_stage()
    for path in sim_utils.find_matching_prim_paths(prim_path):  # ``prim_path`` may name every env's copy
        root = stage.GetPrimAtPath(path)
        for q in Usd.PrimRange(root):
            if q.IsInstanceable():
                q.SetInstanceable(False)
        meshes = [q for q in Usd.PrimRange(root) if q.IsA(UsdGeom.Mesh)]
        if not meshes:
            raise ConfigError(f"{cfg.usd_path} put no mesh at {path}: is the asset server reachable?")
        if not any(q.HasAPI(UsdPhysics.RigidBodyAPI) for q in Usd.PrimRange(root)):
            schemas.define_rigid_body_properties(path, cfg.rigid_props)
            schemas.define_mass_properties(path, cfg.mass_props)
        for q in meshes:
            if not q.HasAPI(UsdPhysics.CollisionAPI):
                UsdPhysics.CollisionAPI.Apply(q)
                UsdPhysics.MeshCollisionAPI.Apply(q).CreateApproximationAttr().Set("convexHull")
    return prim


_SCAN_CFG = None


def scan_cfg_class():
    """``UsdFileCfg`` spawned by :func:`spawn_scan` (a configclass can be defined only once Isaac runs)."""
    global _SCAN_CFG
    if _SCAN_CFG is None:
        import isaaclab.sim as sim_utils
        from isaaclab.utils import configclass

        @configclass
        class ScanCfg(sim_utils.UsdFileCfg):
            func: object = spawn_scan

        _SCAN_CFG = ScanCfg
    return _SCAN_CFG
