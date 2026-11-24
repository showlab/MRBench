import math

import pytest

from mrbench.core.errors import ConfigError
from mrbench.robots import MODELS
from mrbench.tasks import all_specs, build, swap_decoys, tag
from mrbench.tasks.combos import combinations, mount_fits
from mrbench.tasks.layouts import MOUNTS
from mrbench.tasks.loader import _parse_task
from mrbench.tasks.spec import BENCHMARK_SUITES, INFO_LEVELS, LEVELS, MIN_ARMS_PER_TASK, SUITES
from mrbench.tasks.validate import check_card, check_instance

SPECS = all_specs()
#: the cards with an implementation (a planned card is a card only)
BUILT = {t: s for t, s in SPECS.items() if s.status != "planned"}
#: the tasks whose borrow configuration is built
BORROW = sorted(t for t, s in BUILT.items() if "borrow" in s.variants)


def test_every_task_built_on_the_three_fixed_mounts():
    assert len([t for t in SPECS if t.startswith(("Con-", "Rea-"))]) == 8 and sorted(set(SPECS) - set(BUILT)) == []
    for s in SPECS.values():
        assert sorted(a.mount for a in s.arms) == sorted(MOUNTS)
        assert {a.model for a in s.arms} <= set(MODELS)


@pytest.mark.parametrize("tid", sorted(BUILT))
def test_the_cards_robots_are_one_combination_that_does(tid):
    """Every task can be done by some robots (the card's), and the card's still need all three arms."""
    fits = mount_fits(tid)
    for a in SPECS[tid].arms:
        assert fits[a.slot][a.model].can_do
    combos = {tuple(sorted(c.items())): (can, need) for c, can, need in combinations(tid)}
    assert combos[tuple(sorted((a.slot, a.model) for a in SPECS[tid].arms))] == (True, True)


def test_combinations_follow_the_reach_rules():
    # the relay: two Pipers at the ends cannot meet, two Frankas could hand over directly
    combos = {tuple(sorted(c.items())): (can, need) for c, can, need in combinations("Rea-01")}
    assert combos[(("east", "piper"), ("south", "panda"), ("west", "piper"))] == (True, True)
    assert combos[(("east", "panda"), ("south", "panda"), ("west", "panda"))] == (True, False)
    assert not mount_fits("Rea-01")["west"]["so101"].can_do


@pytest.mark.parametrize("tid", sorted(BUILT))
def test_card_rules(tid):
    assert len(SPECS[tid].arms) >= MIN_ARMS_PER_TASK
    assert check_card(SPECS[tid]) == []


@pytest.mark.parametrize("level", LEVELS)
@pytest.mark.parametrize("tid", sorted(BUILT))
def test_instance_rules(tid, level):
    """Reach / out-of-reach, do-nothing fails, oracle keyframes succeed with every milestone."""
    for seed in (0, 1):
        assert check_instance(build(tid, level, seed)) == []


@pytest.mark.parametrize("tid", sorted(BUILT))
def test_build_is_deterministic_per_seed(tid):
    a, b, c = build(tid, "easy", 3), build(tid, "easy", 3), build(tid, "easy", 4)
    assert a.scene == b.scene
    assert a.scene != c.scene


def test_hard_level_changes_the_knob():
    for tid in BUILT:
        e, h = build(tid, "easy", 0), build(tid, "hard", 0)
        assert e.knob_value != h.knob_value


def test_loader_rejects_unknown_fields():
    good = {"id": "X1", "name": "x", "summary": "s", "horizon_s": 1, "success": "s",
            "milestones": [], "status": "scaffold", "impl": "m",
            "arms": [{"mount": "west", "model": "panda", "role": "r"}],
            "knob": {"field": "f", "description": "d", "levels": {"easy": 1, "hard": 2}, "unit": "u"}}
    _parse_task("reach", good, "t")
    with pytest.raises(ConfigError):
        _parse_task("reach", {**good, "typo_field": 1}, "t")
    with pytest.raises(ConfigError):
        _parse_task("reach", {**good, "knob": {**good["knob"], "levels": {"easy": 1, "medium": 2}}}, "t")
    with pytest.raises(ConfigError):  # suites are fixed: why one arm is not enough
        _parse_task("handover", good, "t")
    with pytest.raises(ConfigError):
        _parse_task("reach", {**good, "variants": ["borrow"]}, "t")
    with pytest.raises(ConfigError):
        _parse_task("reach", {**good, "borrow": {"prop": "p", "distractor": "d"}}, "t")


def test_two_benchmark_suites_of_four_tasks():
    bench = [s for s in SPECS.values() if s.id.startswith(("Con-", "Rea-"))]
    by_suite = {}
    for s in bench:
        by_suite.setdefault(s.suite, []).append(s.id)
    assert by_suite == {"concurrency": ["Con-01", "Con-02", "Con-03", "Con-04"],
                        "reach": ["Rea-01", "Rea-02", "Rea-03", "Rea-04"]}
    for s in bench:  # every benchmark task has its borrow configuration built
        assert s.borrow is not None and set(s.variants) == {"exact", "borrow"}


def test_the_borrow_card_says_which_arm_it_takes_away():
    good = {"id": "X1", "name": "x", "summary": "s", "horizon_s": 1, "success": "s",
            "milestones": [], "status": "scaffold", "impl": "m",
            "arms": [{"mount": "west", "model": "panda", "role": "r"}, {"mount": "south", "model": "panda", "role": "r"}],
            "knob": {"field": "f", "description": "d", "levels": {"easy": 1, "hard": 2}, "unit": "u"},
            "borrow": {"replaces": "r", "prop": "p", "distractor": "d"}}
    assert _parse_task("reach", good, "t").borrow.remove is None  # a design only: no arm named yet
    with pytest.raises(ConfigError):  # built, it must name the arm taken away
        _parse_task("reach", {**good, "variants": ["exact", "borrow"]}, "t")
    built = {**good, "variants": ["exact", "borrow"], "borrow": {**good["borrow"], "remove": "south"}}
    assert _parse_task("reach", built, "t").borrow.remove == "south"
    with pytest.raises(ConfigError):  # one of its own arms
        _parse_task("reach", {**built, "borrow": {**built["borrow"], "remove": "east"}}, "t")
    for t in ("Con-01", "Con-02", "Con-03", "Con-04", "Rea-01", "Rea-02", "Rea-03", "Rea-04"):
        assert SPECS[t].borrow.remove in {a.mount for a in SPECS[t].arms}


def test_a_decoy_is_only_for_a_borrow_configuration():
    assert tag("Con-01") == "Con-01" and tag("Con-01", "borrow", True) == "Con-01-borrow-decoy"
    with pytest.raises(ConfigError):
        build("Rea-03", "easy", 0, decoy=True)


def test_swapping_decoys_swaps_the_bodies_names_and_what_is_tied_to_them():
    import dataclasses

    from mrbench.engines.base import HingeSpec

    inst = build("Rea-03", "easy", 0)  # an exact instance dressed as a borrow one: two dominoes as prop and decoy
    a, b = inst.scene.obj("domino_05"), inst.scene.obj("domino_06")
    hinge = HingeSpec("h", "domino_05", (0.0, 0.0, 0.0), 1)
    fake = dataclasses.replace(inst, scene=dataclasses.replace(inst.scene, hinges=(hinge,)), variant="borrow",
                               metadata={**inst.metadata, "borrow": {"props": ["domino_05"], "decoys": ["domino_06"]}})
    swapped = swap_decoys(fake)
    assert swapped.scene.obj("domino_05").pos == b.pos and swapped.scene.obj("domino_06").pos == a.pos
    assert swapped.scene.hinges[0].body == "domino_06"  # the hinge stays with its body
    assert swapped.metadata["decoy"] and swapped.success == inst.success
    with pytest.raises(ConfigError):
        swap_decoys(dataclasses.replace(fake, metadata={**inst.metadata, "borrow": {"props": ["x"], "decoys": ["y"]}}))


def test_the_commands_take_a_configuration():
    from mrbench.cli import build_parser

    p = build_parser()
    for argv in (["validate", "--ik", "--configuration", "borrow", "Con-01"], ["check", "--configuration", "borrow", "Con-01"],
                 ["render", "--configuration", "borrow", "Con-01"], ["demo", "--variant", "borrow", "--decoy", "Con-01"]):
        assert p.parse_args(argv).variant == "borrow"
    assert p.parse_args(["demo", "Con-01"]).variant == "exact" and not p.parse_args(["demo", "Con-01"]).decoy


@pytest.mark.parametrize("tid", BORROW)
@pytest.mark.parametrize("level", LEVELS)
def test_borrow_version(tid, level):
    """Built, the borrow configuration has one arm fewer than the exact one, the prop and the distractor, the same
    instruction, and its oracle keyframes meet the success formula on the mock engine (``check_instance``)."""
    exact, inst = build(tid, level, 0), build(tid, level, 0, variant="borrow")
    gone = SPECS[tid].borrow.remove
    assert [a.slot for a in inst.scene.arms] == [a.slot for a in exact.scene.arms if a.slot != gone]
    b = inst.metadata["borrow"]
    names = {o.name for o in inst.scene.objects}
    assert b["props"] and len(b["props"]) == len(b["decoys"]) and set(b["props"] + b["decoys"]) <= names
    assert inst.instruction == exact.instruction
    assert check_instance(inst) == []
    assert build(tid, level, 0, variant="borrow", decoy=True).metadata["decoy"]


@pytest.mark.parametrize("tid", sorted(BUILT))
def test_instructions_give_only_the_goal(tid):
    """No arm, robot or role is named: who does what is the agent's to work out."""
    from mrbench.tasks.validate import ROLE_WORDS

    for level in LEVELS:
        text = build(tid, level, 0).instruction
        assert not ROLE_WORDS.search(text), (tid, text)
    assert ROLE_WORDS.search("The west arm takes it") and ROLE_WORDS.search("one Piper") and \
        not ROLE_WORDS.search("the far west end of the table")


def test_brief_tells_capabilities_only_when_given():
    from mrbench.tasks.brief import brief, capabilities

    inst = build("Rea-01", "easy", 0)
    given, query = (brief(inst, info) for info in INFO_LEVELS)
    assert given["goal"] == query["goal"] == inst.instruction
    assert [a["arm"] for a in given["arms"]] == [a["arm"] for a in query["arms"]] == ["west", "south", "east"]
    assert given["arms"][0]["reach_m"] == 0.626 and given["arms"][0]["grip_open_m"] == 0.07
    assert all("reach_m" not in a and "robot" not in a for a in query["arms"])
    assert capabilities(inst, "south")["robot"] == "Franka Emika Panda"
    assert "role" not in str(given)
    with pytest.raises(ConfigError):
        brief(inst, "probe")


def test_bodies_starting_inside_each_other_are_refused():
    import dataclasses

    inst = build("Con-01", "easy", 0)
    box = inst.scene.obj("box")
    twin = dataclasses.replace(box, name="box_twin", pos=(box.pos[0] + 0.02, *box.pos[1:]))
    inst = dataclasses.replace(inst, scene=dataclasses.replace(inst.scene, objects=inst.scene.objects + (twin,)))
    assert any("box and box_twin start" in q and "inside each other" in q for q in check_instance(inst))


def test_a_spring_wound_past_its_limit_is_refused():
    """Isaac drives a hinge toward its spring's target the shorter way round, so a spring wound near or past half a
    turn pulls the wrong way: Con-04's pedal on a soft spring, wound 168 degrees at rest and 182 at its stop, would
    stay pressed down when let go.  Every built hinge stays under the limit."""
    import dataclasses

    from mrbench.tasks.impl import con04_pressure_test as M
    from mrbench.tasks.validate import HINGE_WOUND_MAX_DEG

    inst = build("Con-04", "easy", 0)
    pedal = next(h for h in inst.scene.hinges if h.body == "pedal")
    assert pedal.preload_deg + pedal.open_max_deg < HINGE_WOUND_MAX_DEG
    soft = dataclasses.replace(pedal, stiffness=1.5, preload_deg=math.degrees(M.PEDAL_TORQUE / 1.5))
    inst = dataclasses.replace(inst, scene=dataclasses.replace(
        inst.scene, hinges=tuple(soft if h.body == "pedal" else h for h in inst.scene.hinges)))
    assert any("pedal_hinge's spring is wound 182 degrees" in q for q in check_instance(inst))


def test_a_version_may_have_its_own_time_budget():
    good = {"id": "X1", "name": "x", "summary": "s", "horizon_s": 10, "success": "s",
            "milestones": [], "status": "scaffold", "impl": "m",
            "arms": [{"mount": "west", "model": "panda", "role": "r"}],
            "knob": {"field": "f", "description": "d", "levels": {"easy": 1, "hard": 2}, "unit": "u"}}
    s = _parse_task("reach", {**good, "horizons": {"borrow": 25}}, "t")
    assert (s.horizon("exact"), s.horizon("borrow"), s.horizon("surplus")) == (10.0, 25.0, 10.0)
    for bad in ({"exact": 12}, {"spare": 12}, {"borrow": 0}, [25]):  # exact's is horizon_s; known configurations; > 0
        with pytest.raises(ConfigError):
            _parse_task("reach", {**good, "horizons": bad}, "t")
    # Con-01's borrow reference plays 96.5 s: its configuration gets 120 s, the exact one keeps the card's 70
    assert (SPECS["Con-01"].horizon(), SPECS["Con-01"].horizon("borrow")) == (70.0, 120.0)


def test_con02_borrow_draws_a_larger_distractor_in_half_its_scenes():
    """Con-02's borrow configuration: the wide distractor (32 cm, wider than the corridor under the blue leaf's bar) in
    half the scenes, the low one (as wide as the prop, lower) in the rest."""
    from mrbench.tasks.impl import con02_hinged_leaves as P

    kinds = [P.decoy_kind(lv, s) for lv in ("easy", "hard") for s in range(5)]
    assert kinds.count("wide") == 5
    assert P.WEDGE_WIDE > P.CORRIDOR + 0.05 and P.WEDGE_D + 0.169 > P.CORRIDOR  # turned, no narrower than the corridor
    for lv, s in (("easy", 0), ("easy", 1), ("hard", 1)):
        inst = build("Con-02", lv, s, variant="borrow")
        prop, decoy = inst.metadata["borrow"]["props"][0], inst.metadata["borrow"]["decoys"][0]
        widths = {n: inst.scene.obj(n).part("block").size[2] for n in (prop, decoy)}
        tops = {n: inst.scene.obj(n).part("block").size[0] for n in (prop, decoy)}
        if inst.metadata["decoy_kind"] == "wide":
            assert widths[decoy] == P.WEDGE_WIDE and tops[decoy] == tops[prop] and widths[prop] == P.WEDGE_W
        else:
            assert widths[decoy] == widths[prop] and tops[decoy] < tops[prop]


def test_con02_wide_scenes_put_the_prop_in_either_place():
    """Con-02 borrow, wide distractor: the prop stands west of the post's plate in some scenes, in the corridor's line in
    others (no place says which is the prop); where the wide one is in its way, the reference first moves it east of
    the pin, then slides the prop into the corridor's line."""
    from mrbench.tasks.impl import con02_hinged_leaves as P

    seen = set()
    for lv in ("easy", "hard"):
        for s in range(5):
            if P.decoy_kind(lv, s) != "wide":
                continue
            inst = build("Con-02", lv, s, variant="borrow")
            prop = inst.scene.obj(inst.metadata["borrow"]["props"][0])
            west = prop.pos[0] < P.WEDGE_XS[0] - 0.1
            seen.add(west)
            assert P._slide(inst, prop) == (P.WEDGE_XS[0] - prop.pos[0] if west else 0.0)
            notes = " ".join(ph.note for ph in P.script(inst))
            assert ("carries it out of the way" in notes) == west
    assert seen == {True, False}
