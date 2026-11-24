"""The comparison framework end to end without a model or the simulator (``agent/fakes.py``): every row runs in
both settings, the budgets hold across sessions, the per-arm view is enforced, and grading follows the rules."""

import json
import shutil
import subprocess

import pytest

from mrbench.agent import fakes
from mrbench.agent.cases import make_case
from mrbench.agent.env import Env, EnvConfig
from mrbench.agent.episode import Episode, run_episode
from mrbench.agent.grade import grade_dir
from mrbench.agent.ledger import LIMITS, Ledger
from mrbench.agent.methods import get, registry


def test_methods_command_lists_every_row_without_requiring_class_docstrings(capsys):
    from mrbench.agent.bench import main

    assert main(["methods"]) == 0
    rows = [line.split()[0] for line in capsys.readouterr().out.splitlines()]
    assert rows == list(registry())


def make_snapshot(path, case_id):
    """A stand-in for an Isaac snapshot: arms at home, nothing moved, a fixed and a wrist camera per arm."""
    from mrbench.agent.cases import split_case
    from mrbench.robots import get_model

    case = make_case(*split_case(case_id))
    arms = {a.slot: {"q": list(get_model(a.model).home), "gripper": 1.0, "torque": [0.0] * get_model(a.model).arm_dof,
                     "torque_limit": None} for a in case.inst.scene.arms}
    views = ["main", "top"] + [f"wrist_{a}" for a in arms]
    path.mkdir(parents=True, exist_ok=True)
    (path / "state.json").write_text(json.dumps({"case": case_id, "arms": arms, "objects": {}, "views": views}))
    for v in views:
        (path / f"{v}.jpg").write_bytes(b"\xff\xd8\xff")
    return path


def episode(tmp_path, method, case_id, setting):
    snap = make_snapshot(tmp_path / "snap" / case_id, case_id)
    return Episode(method, case_id, setting, 1, "test", fakes.backend(), root=tmp_path / "bench" / method / case_id,
                   snapshot=snap)


@pytest.mark.parametrize("case", ["Con-01-exact", "Rea-02-shortfall"])
@pytest.mark.parametrize("setting", ["one-shot", "equal-budget"])
@pytest.mark.parametrize("method", sorted(n for n, m in registry().items() if m.__module__.startswith("mrbench.")))
def test_every_row_runs_end_to_end(tmp_path, method, setting, case):
    ep = episode(tmp_path, method, case, setting)
    out = run_episode(get(method), ep)
    g = grade_dir(ep.root)
    if setting == "one-shot" and not get(method).one_shot:
        assert g["answered"] is None
        return
    assert g["answered"], out
    led = json.loads((ep.root / "ledger.json").read_text())
    assert led["used"]["runs"] <= led["limits"]["runs"]
    if setting == "one-shot":
        assert led["used"]["runs"] == 0 and not out["success"]
    gold_rows = {"oracle-script", "oracle-team", "oracle-team-skill", "llm-solver", "smart-llm", "roco", "hmas2"}
    feasible = case.endswith("exact")
    if method in gold_rows:  # the fake model answers with the gold team
        assert g["mss_right"], g
        if setting == "equal-budget" and feasible:
            assert g["success"] and g["mss_sr"], g
        if not feasible:
            assert led["used"]["runs"] == 0  # an infeasible answer ends the episode before anything is tried


def test_the_loop_goes_back_to_the_first_step_after_a_failed_run(tmp_path):
    from mrbench.agent.methods.base import Method

    class Stubborn(Method):
        name = "stub"
        calls = 0

        def plan(self, ep, feedback):
            Stubborn.calls += 1
            if feedback is not None:  # the report reaches the first step
                assert feedback["report"]["success"] is False and feedback["plan"]["arms_used"] == ["south"]
            team = ["south"] if feedback is None else ["east", "south", "west"]
            plan = {"feasible": True, "arms_used": team, "idle": [{"arm": a, "reason": "-"} for a in ep.arms
                                                                  if a not in team]}
            ep.record_plan(plan, "stub")
            return plan

    ep = episode(tmp_path, "stub", "Con-01-exact", "equal-budget")
    out = run_episode(Stubborn(), ep)
    assert Stubborn.calls == 2 and out["success"] and out["rounds"] == 2
    assert grade_dir(ep.root)["team"] == ["east", "south", "west"]


def env(tmp_path, case_id="Con-01-exact", **kw):
    snap = make_snapshot(tmp_path / "snap", case_id)
    led = Ledger.create(tmp_path / "ledger.json", "equal-budget")
    cfg = EnvConfig(case=case_id, snapshot=str(snap), log=str(tmp_path / "log"), ledger=str(led.path), **kw)
    return Env(cfg, sim=fakes.sim), led


def test_an_arm_sees_only_its_own_hand(tmp_path):
    e, _ = env(tmp_path, tools=["briefing", "look", "read_arm", "can_reach", "respond"], view="west")
    assert "wrist_east" not in e.cameras and "wrist_west" in e.cameras and "main" in e.cameras
    assert e.logged_call("read_arm", {"arm": "west"})[1] is False
    for name, args in (("read_arm", {"arm": "east"}), ("look", {"camera": "wrist_east"}),
                       ("can_reach", {"arm": "south", "x": 0, "y": -0.2, "z": 0.8})):
        content, err = e.logged_call(name, args)
        assert err, (name, content)
    assert "wrist_east" not in json.dumps(e.tools())


def test_runs_are_counted_for_the_case_across_sessions(tmp_path):
    code = "PLAN = [phase(1.0, 'open', open_fingers('west', 0.03))]"
    e1, led = env(tmp_path, tools=["check_program", "try_in_sim", "submit_program"], session="a")
    e1.call("check_program", {"code": code})
    n = LIMITS["equal-budget"]["runs"]
    for _ in range(n):
        assert "runs_left" in e1.call("try_in_sim", {"code": code})[0]["text"]
    e2 = Env(EnvConfig(**{**json.loads(e1.cfg.to_json()), "log": str(tmp_path / "log2"), "session": "b"}),
             sim=fakes.sim)
    e2.call("check_program", {"code": code})
    assert "No simulator runs left" in e2.call("try_in_sim", {"code": code})[0]["text"]
    assert led.read()["used"]["runs"] == n and len(led.records("runs")) == n


def test_clip_program_checks_report_invalid_bodies_without_crashing(tmp_path):
    e, _ = env(tmp_path, tools=["check_program"])
    code = "PLAN=[phase(.25,'',open_fingers('west',.03),clip('missing','payload'),unclip('missing'))]"
    content, error = e.logged_call("check_program", {"code": code})
    assert not error, content
    report = json.loads((e.log / "check_1.json").read_text())
    assert not report['ok']
    assert report['movers'] == ['west']
    assert any('no dynamic rigid body' in p for p in report['problems'])


def test_fake_execution_accepts_tool_events_without_an_arm_slot(tmp_path):
    from mrbench.agent.program import run_sandbox

    code = "PLAN=[phase(.25,'',open_fingers('west',.03),open_fingers('east',.03),open_fingers('south',.03),clip('tool','payload'),unclip('tool'))]"
    data, error = run_sandbox(code, {slot: {} for slot in ('west', 'east', 'south')}, {})
    assert data is not None, error
    path = tmp_path / 'plan.json'
    path.write_text(json.dumps(data))
    report = fakes.sim('Con-01-exact', path, tmp_path / 'simulation', '0', None)
    assert report['success']


def test_one_shot_sessions_cannot_try(tmp_path):
    ep = episode(tmp_path, "llm-solver", "Con-01-exact", "one-shot")
    ep.start()
    cfg = ep.config("x", ["briefing", "probe", "check_program", "try_in_sim", "submit_team_plan"])
    assert "try_in_sim" not in cfg.tools and "probe" not in cfg.tools and "check_program" in cfg.tools


def test_the_gold_team_of_a_borrow_version_names_its_prop(tmp_path):
    # oracle-team hands the executor the gold team with its prop, so its answer is graded right
    ep = episode(tmp_path, "oracle-team", "Con-01-borrow", "one-shot")
    run_episode(get("oracle-team"), ep)
    g = grade_dir(ep.root)
    assert g["mss_right"] and set(ep.case.expect.things) <= set(g["things_used"]), g


@pytest.mark.skipif(shutil.which("bwrap") is None, reason="bubblewrap is not installed")
def test_the_harness_sandbox_hides_the_repository(tmp_path):
    from mrbench.core.paths import REPO_ROOT

    box = tmp_path / "box"
    box.mkdir()
    cmd = ["bwrap", "--die-with-parent", "--unshare-all", "--ro-bind", "/usr", "/usr", "--symlink", "usr/bin", "/bin",
           "--symlink", "usr/lib", "/lib", "--symlink", "usr/lib64", "/lib64", "--ro-bind", "/etc", "/etc",
           "--proc", "/proc", "--dev", "/dev", "--bind", str(box), "/work", "--chdir", "/work",
           "/bin/sh", "-c", f"ls {REPO_ROOT} 2>&1; ls /work"]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0 and "No such file" not in r.stdout:
        pytest.skip(f"bubblewrap cannot run here: {r.stderr.strip()[:200]}")
    assert "No such file" in r.stdout and "mrbench" not in r.stdout


def test_rounds_without_a_run_end_the_episode(tmp_path):
    from mrbench.agent.episode import MAX_DRY_ROUNDS, ExecResult
    from mrbench.agent.methods.base import Method

    class NeverRuns(Method):
        name = "dry"

        def plan(self, ep, feedback):
            plan = {"feasible": True, "arms_used": list(ep.arms), "idle": []}
            ep.record_plan(plan, "dry")
            return plan

        def execute(self, ep, plan, feedback):
            return ExecResult(False, {"success": False, "played": False}, [], "no program passed")

    ep = episode(tmp_path, "dry", "Con-01-exact", "equal-budget")
    out = run_episode(NeverRuns(), ep)
    assert out["rounds"] == MAX_DRY_ROUNDS and "without a run" in out["ended"]


def test_a_smaller_team_that_succeeds_beats_the_gold(tmp_path):
    ep = episode(tmp_path, "oracle-team", "Con-01-exact", "equal-budget")
    run_episode(get("oracle-team"), ep)
    out = json.loads((ep.root / "episode.json").read_text())
    out["final_plan"] = {**out["final_plan"], "arms_used": ["west", "east"],
                         "idle": [{"arm": "south", "reason": "-"}]}
    (ep.root / "episode.json").write_text(json.dumps(out))
    led = json.loads((ep.root / "ledger.json").read_text())
    led["runs"][-1]["movers"] = ["west", "east"]
    (ep.root / "ledger.json").write_text(json.dumps(led))
    g = grade_dir(ep.root)
    assert g["success"] and g["beats_gold"] and g["mss_right"] and g["team"] == ["east", "west"]


def test_the_agent_is_told_of_every_body_in_the_eight():
    """A body left out of a case's setup is in the pictures but not in the briefing, locate or measure: the agent
    would plan round a wall it was never told of (such as Rea-01's low partition)."""
    from mrbench.agent.cases import SETUPS

    for task in (t for t in SETUPS if t.startswith(("Con-", "Rea-"))):
        for variant in ("exact", "borrow"):  # (a borrow version's prop and distractor, described alike)
            if variant not in SETUPS[task].expect:
                continue
            c = make_case(task, variant)
            told = set(c.setup.bodies) | {p for _, parts in c.setup.groups.values() for p in parts}
            assert not [o.name for o in c.inst.scene.objects if o.name not in told], (task, variant)
            assert not [cl.name for cl in c.inst.scene.cloths if cl.name not in told], (task, variant)
            b = c.inst.metadata.get("borrow")
            if b:
                assert c.setup.bodies[b["props"][0]] == c.setup.bodies[b["decoys"][0]], task


def test_the_evidence_rule_wants_a_reason_for_every_arm(tmp_path):
    e, _ = env(tmp_path, tools=["submit_team_plan"], evidence=True)
    plan = {"feasible": True, "arms_used": ["west", "east", "south"], "idle": [], "missing": "",
            "without_each_arm": [{"arm": "west", "tried": "x", "failed_because": "y", "evidence": "can_reach: ..."}]}
    out = e.call("submit_team_plan", plan)[0]["text"]
    assert "Not accepted" in out and "east is used" in out and "south is used" in out and "west is used" not in out
    plan["without_each_arm"] += [{"arm": "east", "tried": "x", "failed_because": "check_hold: lift_off holds false"},
                                 {"arm": "south", "tried": "x", "failed_because": "it would not work"}]
    out = e.call("submit_team_plan", plan)[0]["text"]
    assert "east is used" not in out and "south is used" in out  # a reason naming a tool result counts


def test_respond_takes_its_fields_beside_the_message_and_refuses_none(tmp_path):
    from mrbench.agent.methods.smart_llm import SUBTASKS

    e, _ = env(tmp_path, tools=["respond"], respond_schema=SUBTASKS)
    out = e.call("respond", {"message": "m"})[0]["text"]
    assert "Not recorded" in out and "subtasks" in out and not e.over
    out = e.call("respond", {"message": "m", "subtasks": '[{"id": "s1", "what": "x", "skills": []}]'})[0]["text"]
    assert "Recorded" in out and e.over
    rec = json.loads((tmp_path / "log" / "output.json").read_text())["respond"]
    assert rec["data"]["subtasks"][0]["id"] == "s1"


def test_respond_reads_data_as_text_with_a_stray_brace_and_refuses_text(tmp_path):
    from mrbench.agent.methods.roco import TURN

    e, _ = env(tmp_path, tools=["respond"], respond_schema=TURN)
    out = e.call("respond", {"message": "m", "data": "I agree"})[0]["text"]
    assert "Not recorded" in out and not e.over
    out = e.call("respond", {"message": "m", "data": '{"taking_part": true, "proposal": {"feasible": true}}}'})
    assert "Recorded" in out[0]["text"] and e.over
    rec = json.loads((tmp_path / "log" / "output.json").read_text())["respond"]
    assert rec["data"]["proposal"] == {"feasible": True}


def test_every_sense_answers_without_an_error(tmp_path):
    from mrbench.agent.env import SENSES

    e, _ = env(tmp_path, tools=list(SENSES))
    calls = {"briefing": {}, "look": {"camera": "main"}, "read_arm": {"arm": "south"}, "locate": {"name": "box"},
             "can_reach": {"arm": "south", "x": 0.0, "y": -0.3, "z": 0.9}, "measure": {"name": "box"},
             "heft": {"arm": "south", "thing": "box"}, "note": {"fact": "the box is light", "evidence": "heft"}}
    assert set(calls) == set(SENSES)
    for name, args in calls.items():
        content, err = e.logged_call(name, args)
        assert not err, (name, content)
    assert "mass_kg" in e.logged_call("heft", calls["heft"])[0][0]["text"]


def test_can_reach_over_the_base_says_no_without_crashing(tmp_path):
    e, _ = env(tmp_path, tools=["can_reach"])
    a = e.case.inst.scene.arm("south")
    r = e.obs.reach_once("south", (a.pos[0], a.pos[1], 1.0), "level")
    assert r[0] is False and "over the base" in r[1]


def test_check_hold_knows_every_hinge_by_its_own_axis(tmp_path):
    """check_hold knows every hinge by its own axis: Con-01's doors (vertical), Con-02's leaves and Con-03's lid, and a
    hand on Con-01's knob, 0.21 m out on its lever, holds the door."""
    from mrbench.agent import physics
    from mrbench.agent.cases import split_case
    from mrbench.agent.observe import Observer

    def held(case_id, body, part, **hold):
        case = make_case(*split_case(case_id))
        obs = Observer(case, make_snapshot(tmp_path / case_id, case_id))
        at = hold.pop("at", None) or obs.locate(f"{body.partition(':')[0]}:{part}")["centre"]
        return physics.check_hold(case, obs, body, None, [{"kind": "hold", "at": at, **hold}])["holds"][0]

    knob = held("Con-01-exact", "door_w", "knob", by="west", open_deg=80)
    assert knob["on_part"] == "knob" and knob["radius_from_hinge_m"] > 0.18 and knob["with_rough_masses"]["holds"]
    lid = held("Con-03-exact", "lid:tab", "tab", by="west", open_deg=70)  # a part names its body
    assert lid["with_rough_masses"]["holds"] and lid["with_rough_masses"]["closing_nm"] > 0
    # the blue leaf has no spring: held level, its own weight is what the hand holds up
    leaf = held("Con-02-exact", "leaf_w", "bar", by="west", open_deg=15)
    assert leaf["with_rough_masses"]["closing_nm"] > 0.3
    off = held("Con-03-exact", "lid", "tab", by="west", at=[0.0, 0.5, 1.6])
    assert "error" in off
    # a leaf on a hinge is no stop for the other leaf: each falls unless held
    for kind in ("stop", "weight"):
        with pytest.raises(KeyError, match="turns on a hinge"):
            held("Con-02-exact", "leaf_e", "bar", kind=kind, by="leaf_w", open_deg=15)


def test_a_bleed_valve_let_go_is_said_to_open_as_the_briefing_says(tmp_path):
    """The hinge tools count a hinge's turn from its 0 deg stop and call the way back "shut".  Con-04's bleed valve is
    open at 0 deg (its lever up), so ``check_hold`` with no holds must say that let go it opens, as the briefing does
    ("let go, its lever lifts and the valve opens"), not that its spring turns it shut."""
    e, _ = env(tmp_path, case_id="Con-04-exact", tools=["briefing", "check_hold", "scene_graph"])

    def said(tool, **args):
        return e.logged_call(tool, args)[0][0]["text"]

    assert "its lever lifts and the valve opens" in said("briefing")
    for valve in ("valve_w", "valve_e"):
        unheld = said("check_hold", body=valve, holds=[])
        assert "shut" not in unheld and "back to its 0 deg stop, where its lever is up and the valve is open" in unheld
    assert "where it is up and the foot valve is shut" in said("check_hold", body="pedal", holds=[])
    held = said("check_hold", body="valve_w", holds=[{"kind": "hold", "by": "west", "open_deg": 8.5,
                                                      "at": e.obs.locate("valve_w:grip")["centre"]}])
    assert '"stops"' in held and "turning it shut" not in held and "with_margin" in held
    graph = said("scene_graph")
    assert graph.count('"at_0_deg"') == 3 and "turning_it_shut" not in graph
    # a door is shut at 0 deg: its replies are as they were, to the letter
    door, _ = env(tmp_path / "door", tools=["check_hold", "scene_graph"])
    assert "its spring and its weight turn it shut (0.10 N m at 0 deg open)" in \
        door.logged_call("check_hold", {"body": "door_w", "holds": []})[0][0]["text"]
    graph = door.logged_call("scene_graph", {})[0][0]["text"]
    assert '"turning_it_shut"' in graph and "at_0_deg" not in graph


def test_a_weighed_weight_on_a_hinged_body_takes_the_small_margin(tmp_path):
    """A reply's heading says a weighed mass is taken at 0.95 with margin, as for a stop and for a weight on a body on
    the table: a weight on a hinged body (Con-04 borrow's prop on the pedal) takes the same small margin once it has
    been weighed, and 0.7 before."""
    e, _ = env(tmp_path, case_id="Con-04-borrow", tools=["check_hold", "heft"])
    pocket = e.obs.locate("pedal:pocket")["centre"]

    def resisting():
        content, err = e.logged_call("check_hold", {"body": "pedal", "holds": [
            {"kind": "weight", "by": "weight_1", "at": pocket, "open_deg": 14}]})
        h = json.loads(content[0]["text"].partition("\n[")[0])["holds"][0]
        margin = next(v for k, v in h.items() if k.startswith("with_margin"))
        assert "weighed ones x0.95" in next(k for k in h if k.startswith("with_margin"))
        return h["with_rough_masses"]["resisting_nm"], margin["resisting_nm"]

    rough, margin = resisting()
    assert margin == pytest.approx(0.7 * rough, abs=0.02)  # not weighed: the rough mass's margin
    assert "mass_kg" in e.logged_call("heft", {"arm": "west", "thing": "weight_1"})[0][0]["text"]
    rough, margin = resisting()
    assert margin == pytest.approx(0.95 * rough, abs=0.02)  # weighed: the small margin


def test_the_task_level_needs_every_version_right_in_the_same_repeat():
    from mrbench.agent.gold import variants
    from mrbench.agent.grade import by_version, table, task_level

    def row(case, repeat, right, method="m"):
        v = case.partition("@")[0].split("-", 2)[2]
        return {"method": method, "setting": "one-shot", "case": case, "variant": v, "repeat": repeat,
                "answered": True, "mss_right": right, "success": False}

    vs = variants("Con-01")
    assert vs == ("exact", "surplus", "shortfall", "swapped", "borrow")
    rows = [row(f"Con-01-{v}@easy.s0.varied", 0, True) for v in vs]  # all right
    rows += [row(f"Con-01-{v}@easy.s1.varied", 0, v != "borrow") for v in vs]  # one wrong
    rows += [row(f"Con-01-{v}@hard.s0.varied", 0, True) for v in vs if v != "swapped"]  # one not run: left out
    out = task_level(rows)
    assert "| m | one-shot | 2 | 0.500 |" in out and out.rstrip().endswith("| 1 |")
    assert "| m | one-shot | 3/3 | 3/3 | 3/3 | 2/2 | 2/3 |" in by_version(rows)
    # two repeats of one scene: one right, one not -> the scene counts half
    rows = [row(f"Con-01-{v}@easy.s0.varied", k, k == 0 or v != "exact") for v in vs for k in (0, 1)]
    assert "| m | one-shot | 1 | 0.500 |" in task_level(rows)
    assert "task level" in table(rows)


