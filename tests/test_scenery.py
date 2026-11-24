import dataclasses
import math

import pytest

from mrbench.core.errors import ConfigError
from mrbench.engines.base import Look
from mrbench.engines.mock import MockEngine
from mrbench.robots import get_model
from mrbench.scenery import (CLUTTER, FLOORS, LIGHTS, LOOKS, PROPS, TABLES, check_look, describe, dress, parse_look,
                             random_look, repaint_text)
from mrbench.scenery.downloads import TEXTURES
from mrbench.scenery.patterns import PATTERNS, PITCH, PROFILE, TILE_M, height, maps
from mrbench.scenery.props import EDGE, GROUPS, REACH_GAP, TALL_M, footprint, overlap
from mrbench.tasks import all_specs, build
from mrbench.tasks.validate import check_props

SPECS = all_specs()


def test_every_named_and_drawn_look_is_valid():
    for look in LOOKS.values():
        assert check_look(look) is look
        assert describe(look)
    drawn = [random_look(s) for s in range(30)]
    assert len({d.table for d in drawn}) > 5 and len({d.light for d in drawn}) > 2
    assert all(d.clutter == "none" for d in drawn) and all(lk.clutter == "none" for lk in LOOKS.values())
    assert random_look(7) == random_look(7)


def test_every_finish_is_a_pinned_texture_or_a_colour():
    from mrbench.scenery.backdrop import finishes

    for finish in list(TABLES.values()) + list(FLOORS.values()) + list(finishes().values()):
        kind, _, name = finish.maps.rpartition(":")
        assert not finish.maps or (name in PATTERNS if kind == "pattern" else finish.maps in TEXTURES)
    assert set(LIGHTS) >= {"studio", "warm", "dim"}


def test_parse_look():
    look = parse_look("default,table=cloth:blue,floor=parquet,light=warm,clutter=desk:4,form=block,top=0.06")
    assert (look.table, look.floor, look.light, look.clutter, look.clutter_n, look.table_form, look.top_m) == \
        ("cloth:blue", "parquet", "warm", "desk", 4, "block", 0.06)
    assert parse_look("kitchen,clutter=none").clutter_n == 0
    assert parse_look("table=tile:stone").table == "tile:stone"  # changes alone start from the default
    assert parse_look("default,clutter=workshop").clutter_n > 0
    assert parse_look("default,room=studio").floor != "room"  # a studio has no floor of its own
    assert parse_look("random", seed=3) == random_look(3)
    assert parse_look("default,backdrop=kitchen").backdrop == "kitchen" and parse_look("kitchen,backdrop=none")
    for bad in ("nosuch", "default,table=glass", "default,colour=red", "default,clutter=attic:3", "default,top=0.5",
                "default,backdrop=attic"):
        with pytest.raises(ConfigError):
            parse_look(bad)


def test_every_prop_has_a_shape():
    for kind, prop in PROPS.items():
        assert all(v > 0 for v in prop.size), kind
        assert bool(prop.usd) != bool(prop.parts), kind
        if prop.usd:
            assert prop.usd.startswith("Props/YCB/") and prop.up in ("-y", "+z")
    for groups in CLUTTER.values():
        assert set(groups) <= set(GROUPS)


def test_a_groups_props_stand_apart():
    """Within a group, 1.5 cm at least between any two props' footprints."""
    for g in GROUPS.values():
        assert all(k in PROPS for k, *_ in g.items)
        boxes = [footprint(k, x, y, yaw) for k, x, y, yaw in g.items]
        assert not any(overlap(boxes[i], boxes[j], 0.015) for i in range(len(boxes)) for j in range(i)), g.name


@pytest.mark.parametrize("tid", sorted(t for t, s in SPECS.items() if s.status != "planned"))
def test_props_stand_where_no_arm_works(tid):
    """Every prop on the table, square to it, beyond every arm's reach, clear of the task's things; no two
    overlapping; set out in groups along the north and south edges."""
    inst = build(tid, "hard", 0, "default")
    assert check_props(inst) == []
    dressed = dress(inst, "default,clutter=mixed:4")
    props = dressed.scene.props
    assert props, "no room for a single group"
    tx, ty, _ = inst.scene.table_size
    boxes = [footprint(p.kind, p.pos[0], p.pos[1], p.yaw_deg) for p in props]
    for p, (x0, y0, x1, y1) in zip(props, boxes):
        assert -tx / 2 + EDGE <= x0 and x1 <= tx / 2 - EDGE and -ty / 2 + EDGE <= y0 and y1 <= ty / 2 - EDGE
        assert p.yaw_deg % 90 == 0 or p.kind == "ycb:025_mug"  # square to the table (a mug's handle aside)
        assert math.isclose(p.pos[2] - p.size[2] / 2, inst.scene.table_top_z, abs_tol=1e-3)
        for a in inst.scene.arms:
            gap = math.hypot(max(x0 - a.pos[0], 0, a.pos[0] - x1), max(y0 - a.pos[1], 0, a.pos[1] - y1))
            assert gap >= get_model(a.model).reach_m + REACH_GAP - 1e-9
        if p.size[2] > TALL_M:
            assert p.pos[1] >= 0.0
        assert min(abs(y0 + ty / 2), abs(ty / 2 - y1)) < 0.25  # near the edge its group stands against
    for i in range(len(boxes)):
        for j in range(i):
            assert not overlap(boxes[i], boxes[j])


def test_a_group_is_set_out_square_and_kept_apart_from_the_next():
    dressed = dress(build("Rea-01", "hard", 0, "default"), "default,clutter=home:4")
    kinds = [p.kind for p in dressed.scene.props]
    assert "book_row" in kinds or "books" in kinds
    books = next(p for p in dressed.scene.props if p.kind in ("books", "book_row"))
    assert books.yaw_deg in (0.0, 180.0)


def test_dressing_is_deterministic_and_leaves_the_task_alone():
    inst = build("Con-04", "hard", 0)
    with_props = "kitchen,clutter=kitchen:2"
    a, b = dress(inst, with_props), dress(inst, with_props)
    assert a.scene == b.scene and a.scene.props
    assert dress(inst, with_props, seed=1).scene.props != a.scene.props
    assert a.scene.objects == inst.scene.objects and a.scene.cloths == inst.scene.cloths
    assert dataclasses.replace(a, scene=inst.scene) == inst
    assert build("Con-04", "hard", 0, with_props).scene == a.scene


BENCHMARK = ("Con-01", "Con-02", "Con-03", "Con-04", "Rea-01", "Rea-02", "Rea-03", "Rea-04")


def test_the_eight_share_one_look():
    assert {SPECS[t].look for t in BENCHMARK} == {"workshop"}


@pytest.mark.parametrize("backdrop", ("kitchen", "workshop", "home"))
@pytest.mark.parametrize("tid", BENCHMARK)
def test_the_backdrop_stands_out_of_reach_and_out_of_every_cameras_way(tid, backdrop):
    from mrbench.agent.snapshot import VIEWS
    from mrbench.render import default_views
    from mrbench.scenery import backdrop as B

    sc = build(tid, "easy", 0).scene
    assert LOOKS[backdrop].backdrop == backdrop and B.lamps(backdrop, sc.table_size)
    pieces, ornaments = B.build(backdrop, sc.table_size)
    assert pieces and ornaments and all(PROPS[o.kind].usd for o in ornaments)
    assert {p.finish for p in pieces} <= set(B.finishes())
    assert len({p.name for p in pieces} | {o.name for o in ornaments}) == len(pieces) + len(ornaments)  # USD paths
    boxes = [(*B.footprint(p), p.pos[2] - p.half()[2], p.pos[2] + p.half()[2]) for p in pieces]
    boxes += [(*B.footprint(o, PROPS[o.kind].size), o.pos[2] - PROPS[o.kind].size[2] / 2,
               o.pos[2] + PROPS[o.kind].size[2] / 2) for o in ornaments]
    for a in sc.arms:
        reach = get_model(a.model).reach_m + B.REACH_GAP
        for x0, x1, y0, y1, _, _ in boxes:
            assert math.hypot(max(x0 - a.pos[0], 0, a.pos[0] - x1), max(y0 - a.pos[1], 0, a.pos[1] - y1)) > reach
    walls = {p.name: p for p in pieces if p.name.startswith("wall_")}
    for eye, _, _ in list(default_views().values()) + list(VIEWS.values()):
        assert eye[1] < walls["wall_n"].pos[1] and eye[0] > walls["wall_w"].pos[0]  # on the room's side
        for x0, x1, y0, y1, z0, z1 in boxes:
            g = B.CAMERA_GAP
            assert not (x0 - g < eye[0] < x1 + g and y0 - g < eye[1] < y1 + g and z0 - g < eye[2] < z1 + g)


def test_colour_words_change_all_at_once():
    words = {"blue": "orange", "orange": "red"}
    assert repaint_text("the blue and the orange leaf", words) == "the orange and the red leaf"
    assert repaint_text("a blueish orangery", words) == "a blueish orangery"  # whole words only
    assert repaint_text("the blue leaf", None) == "the blue leaf"


@pytest.mark.parametrize("tid", BENCHMARK)
def test_random_colours_paint_what_the_impl_names_and_follow_in_the_words(tid):
    card, painted = build(tid, "easy", 0, "kitchen"), build(tid, "easy", 0, "kitchen,colors=random")
    assert painted.scene.look.colors == "random" and card.metadata.get("repaint") is None
    named = card.metadata.get("paint") or {}
    words = painted.metadata.get("repaint") or {}
    assert set(words) == set(named.values())
    assert len(set(words.values())) == len(words)  # no two things that were apart share a colour
    for w in named.values():  # every word the impl names is in what the team is told, or names a thing it sees
        assert w in card.instruction or tid in ("Con-03", "Rea-01", "Rea-04")
    assert painted.instruction == repaint_text(card.instruction, words)
    assert [o.name for o in painted.scene.objects] == [o.name for o in card.scene.objects]
    assert [(o.pos, o.size, o.mass_kg) for o in painted.scene.objects] == \
        [(o.pos, o.size, o.mass_kg) for o in card.scene.objects]  # only colours change
    assert painted.success == card.success
    again = build(tid, "easy", 0, "kitchen,colors=random")
    assert again.scene == painted.scene and again.instruction == painted.instruction
    assert any(build(tid, "easy", s, "kitchen,colors=random").metadata.get("repaint") != words
               for s in range(1, 6)) or not words


def test_the_mock_engine_reports_props_where_they_stand():
    inst = build("Rea-01", "easy", 0, "office,clutter=desk:2")
    frame = MockEngine().load(inst.scene, 0)
    assert set(frame.props) == {p.name for p in inst.scene.props}
    assert all(frame.props[p.name] == p.pose for p in inst.scene.props)
    assert "props" in frame.to_json()


def test_cli_lists_looks_and_shows_props(capsys):
    from mrbench.cli import main

    assert main(["looks"]) == 0
    out = capsys.readouterr().out
    assert "kitchen" in out and "cloth:blue" in out and "evening" in out
    assert main(["show", "Rea-01", "--look", "office,clutter=desk:2"]) == 0
    out = capsys.readouterr().out
    assert "look: laminate:white table" in out and "prop_0" in out
    assert main(["show", "Rea-01", "--look", "nosuch"]) == 2


def test_drawn_tops_have_their_holes_and_slots_where_they_should():
    """A breadboard tile: 8 x 8 holes on the 25 mm grid; an extrusion tile: a slot down each 40 mm profile."""
    np = pytest.importorskip("numpy")
    px = 400
    h, _ = height("breadboard_black", px)
    holes = h <= -0.999
    per_m = px / TILE_M
    centres = [int((i + 0.5) * PITCH * per_m) for i in range(round(TILE_M / PITCH))]
    assert all(holes[r, c] for r in centres for c in centres)
    assert not holes[int(PITCH * per_m), int(PITCH * per_m)]  # between four holes
    assert 0.03 < holes.mean() < 0.06  # 6 mm holes on a 25 mm grid: 4.5 % of the top
    h, _ = height("extrusion", px)
    slot_rows = [int((i + 0.5) * PROFILE * per_m) for i in range(round(TILE_M / PROFILE))]
    assert all((h[r] <= -0.999).all() for r in slot_rows)  # slots run along x
    h, scratch = height("workbench_steel", px)
    assert not (h <= -0.999).any() and 0 < scratch.mean() < 0.02  # the workbench: no holes, a few fine scratches
    for name in PATTERNS:
        d, n, r = maps(name, 64)
        assert d.shape == n.shape == (64, 64, 3) and r.shape == (64, 64) and d.dtype == np.uint8


def test_every_benchmark_task_has_its_sweeps_from_its_impl_as_it_is():
    """The cells each built version's reference sweeps (``scenery.sweep``), planned from the impl's present code: after
    a change to an impl, plan them again (``python -m mrbench.scenery.sweep TASK``)."""
    import json

    from mrbench.scenery import sweep

    for tid in BENCHMARK:
        for variant in SPECS[tid].variants:
            for level in ("easy", "hard"):
                p = sweep.path(tid, variant, level)
                assert p.exists(), f"no sweep {p.name}"
                assert json.loads(p.read_text())["impl_sha"] == sweep.impl_sha(tid), f"{p.name} is stale"


@pytest.mark.parametrize("tid", BENCHMARK)
def test_distractors_stand_within_reach_clear_of_the_work_and_change_nothing_else(tid):
    from mrbench.scenery import distractors as D
    from mrbench.scenery import sweep

    card, inst = build(tid, "easy", 1), build(tid, "easy", 1, "kitchen,distract=3")
    shown = inst.metadata["distractors"]
    assert len(shown) == 3 and set(shown) == {o.name for o in inst.scene.objects} - {o.name for o in card.scene.objects}
    assert inst.scene.objects[:len(card.scene.objects)] == card.scene.objects and inst.success == card.success
    swept = sweep.cells(tid)
    for o in inst.scene.objects[len(card.scene.objects):]:
        kind = D.KINDS[o.name.split("_")[0]]
        assert o.name.split("_")[0] in D.allowed(tid) and o.dynamic and o.mass_kg == kind.mass_kg
        assert any(math.dist(o.pos[:2], a.pos[:2]) <= D.REACH * get_model(a.model).reach_m for a in inst.scene.arms)
        assert sweep.cell_of(*o.pos[:2]) not in swept
    assert build(tid, "easy", 1, "kitchen,distract=3").scene == inst.scene  # from the seed
    assert build(tid, "easy", 2, "kitchen,distract=3").scene != inst.scene


def test_no_distractor_can_hold_a_door_or_a_leaf():
    from mrbench.scenery import distractors as D

    assert all(D.KINDS[k].height < 0.05 for k in D.allowed("Con-01"))  # under the lever's bar, 5 cm up
    assert sum(D.KINDS[k].height for k in D.allowed("Con-02")) < 0.10  # all stacked, far under the leaf's 20 cm
    assert max(max(k.size) for k in D.KINDS.values()) <= 0.22 and max(k.mass_kg for k in D.KINDS.values()) <= 0.45


def test_no_distractor_part_shares_a_name_with_a_distractor():
    from mrbench.scenery import distractors as D

    for kind in D.KINDS.values():
        assert not {p.name for p in kind.parts} & set(D.KINDS)
