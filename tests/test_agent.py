"""The agent side's pieces that run without Isaac or a model: cases, what the agent is told, the program
sandbox and the server's checks on a submission."""

import json

import pytest

from mrbench.agent import program
from mrbench.agent.cases import SETUPS, cases, make_case
from mrbench.agent.observe import Observer


def test_every_case_builds_with_its_team():
    for task, variant in cases():
        c = make_case(task, variant)
        slots = {a.slot for a in c.inst.scene.arms}
        if variant in ("shortfall", "borrow"):
            assert SETUPS[task].remove not in slots
        if variant == "surplus":
            assert "north" in slots
        # the steel blocks where a task has no borrow version of its own (``TaskSetup.blocks``)
        assert bool(c.blocks) == (variant in ("borrow", "spare") and bool(SETUPS[task].blocks))
        e = c.expect
        assert e.feasible == bool(e.minimal) and all(s <= slots for s in e.minimal + e.workable)


def test_the_surplus_arm_is_clear_of_the_scene_at_home():
    from mrbench.policies import clearance
    from mrbench.policies.plan import plan
    from mrbench.policies.script import phase

    for task, variant in cases():
        if variant == "surplus" and task.startswith(("Con-", "Rea-")):
            c = make_case(task, variant)
            clashes = clearance.check(c.inst, plan(c.inst, [phase(1.0, "idle")]))
            assert not [k for k in clashes if c.added in (k.a.split(".")[0], k.b.split(".")[0])], (task, clashes[:2])


def test_a_swapped_scene_keeps_every_rule_but_its_reach_claims_and_its_arms_start_clear():
    """The swapped configuration replaces one place's robot: the reach claims made for the card's robot may break (that
    is the point), every other design rule holds, and every arm at home is clear of the scene and of the others."""
    from mrbench.policies import clearance
    from mrbench.policies.plan import plan
    from mrbench.policies.script import phase
    from mrbench.robots import get_model
    from mrbench.tasks.validate import check_instance

    for task, variant in cases():
        if variant != "swapped":
            continue
        c = make_case(task, variant)
        slot, model = c.swapped
        assert c.inst.scene.arm(slot).model == model != make_case(task, "exact").inst.scene.arm(slot).model
        claims = (" must reach ", " must NOT reach ", " must not meet ")
        assert not [p for p in check_instance(c.inst) if not any(w in p for w in claims)], task
        assert not clearance.check(c.inst, plan(c.inst, [phase(1.0, "idle")])), task
    from mrbench.robots.collision import grip_span_m

    assert (grip_span_m(get_model("panda")), grip_span_m(get_model("piper"))) == (0.08, 0.07)
    assert 0.03 < grip_span_m(get_model("so101")) < 0.08  # a jaw's swing, not its angle


def test_the_sandbox_runs_only_the_api():
    arms = {"west": {"base": [-0.7, 0.0, 0.75], "yaw": 0.0}}
    plan, _ = program.run_sandbox("p = tool_pose('west', -0.3, 0.0, 0.9)\nPLAN = [phase(1, 'go', move('west', p))]",
                                  arms, {})
    assert plan[0]["actions"][0]["type"] == "move" and len(program.to_script(plan)) == 1
    for bad in ("import os\nPLAN = []", "open('/etc/passwd')\nPLAN = []", "PLAN = [phase(1, 'x', move('east', [0]*7))]"):
        plan, err = program.run_sandbox(bad, arms, {})
        assert plan is None and err


def test_a_case_in_random_colours_tells_the_team_the_colours_it_sees():
    c = make_case("Con-02", "exact", "easy", 2, look="kitchen,colors=random")
    words = c.inst.metadata["repaint"]
    assert words == {"blue": "red", "orange": "blue"}
    assert c.inst.instruction.startswith("Pin the red and the blue leaf")
    assert c.setup.bodies["leaf_w"].startswith("red leaf") and c.setup.bodies["leaf_e"].startswith("blue leaf")
    assert "the red leaf hangs down and the blue leaf springs up" in " ".join(c.setup.notes)
    assert make_case("Con-02", "exact").setup.bodies["leaf_w"].startswith("blue leaf")  # the card's colours


def test_a_case_id_may_name_a_scene():
    from mrbench.agent.cases import SETTINGS, case_from_id, parse_case, scene_ids, split_case

    assert parse_case("Con-01-exact") == ("Con-01", "exact", "easy", 0, "card")
    assert parse_case("Con-01-exact@hard.s2.varied") == ("Con-01", "exact", "hard", 2, "varied")
    assert split_case("Con-02-borrow@easy.s1.card") == ("Con-02", "borrow")
    with pytest.raises(ValueError):
        parse_case("Con-01-exact@hard.2.varied")
    assert scene_ids(["Con-01-exact"], seeds=(0, 1))[1] == "Con-01-exact@easy.s1.varied"
    c = case_from_id("Con-02-shortfall@hard.s1.varied")
    assert (c.level, c.seed, c.inst.scene.look.colors) == ("hard", 1, "random") and "varied" in SETTINGS
    # the room by the seed: kitchen, workshop, home (a cloth on the table), and again
    from mrbench.agent.cases import setting_look

    assert [setting_look("varied", s).split(",")[0] for s in (0, 1, 2, 3)] == ["kitchen", "workshop", "home", "kitchen"]
    assert setting_look("card", 1) is None
    assert (c.inst.scene.look.table, c.inst.scene.look.floor) == ("metal:workbench", "concrete")  # seed 1: the workshop
    shown = c.inst.metadata["distractors"]
    assert len(shown) == 3 and set(shown) <= set(c.setup.bodies)  # the team is told what they are
    assert c.expect == case_from_id("Con-02-shortfall").expect  # the scene does not change the answer


def _look(n):
    return {"role": "user", "content": [{"type": "text", "text": "views"},
                                        *({"type": "image_url", "image_url": {"url": f"data:{k}"}} for k in range(n))]}


def _lines(messages):
    """What stands where pictures were, in order."""
    return [p["text"] for m in messages if isinstance(m["content"], list) for p in m["content"]
            if p["type"] == "text" and p["text"] != "views"]


def test_old_pictures_are_thinned_in_what_is_sent_and_all_can_be_taken_out_of_the_history():
    from mrbench.agent import loop

    messages = [{"role": "system", "content": "s"}, _look(2), _look(3)]
    assert _lines(loop._view(messages)) == [loop.OLD_PICTURE] * 2  # only the last KEEP_IMAGES are sent
    assert _lines(messages) == []  # the kept history is not touched by the view
    assert loop._withdraw(messages) == loop.KEEP_IMAGES  # the ones in view say why, the older ones keep their text
    assert _lines(messages) == [loop.OLD_PICTURE] * 2 + [loop.WITHDRAWN] * 3
    assert loop._withdraw(messages) == 0  # nothing left to take out
    messages.append(_look(2))  # the session looks again
    assert loop._withdraw(messages) == 2


class _Server:
    """A stand-in for the environment server: a camera and the submission."""

    done = None

    def __init__(self, *args):
        pass

    def tools(self):
        return [{"name": n, "description": n, "inputSchema": {"type": "object"}} for n in ("look", "submit")]

    def call(self, name, args):
        if name == "submit":
            self.done.write_text("")
            return {"content": [{"type": "text", "text": "taken"}]}
        return {"content": [{"type": "text", "text": "main"}, {"type": "image", "mimeType": "image/jpeg", "data": "AA=="}]}

    def close(self):
        pass


def _session(tmp_path, monkeypatch, script):
    """One session of the loop against ``_Server``; the model's turns come from ``script`` (an exception is raised)."""
    from mrbench.agent import loop

    sent = []

    def chat(base_url, body):
        sent.append(body["messages"])
        step = script(len(sent))
        if isinstance(step, Exception):
            raise step
        call = {"id": "c", "function": {"name": step, "arguments": "{}"}}
        return {"choices": [{"message": {"content": "", "tool_calls": [call]}, "finish_reason": "tool_calls"}], "usage": {}}

    _Server.done = tmp_path / "done"
    monkeypatch.setattr(loop, "MCPClient", _Server)
    monkeypatch.setattr(loop, "_chat", chat)
    monkeypatch.setattr(loop.time, "sleep", lambda s: None)
    stats = loop.run("http://model", "m", "system", "first", [], {}, ".", tmp_path, True, "submit", _Server.done)
    return stats, sent, [json.loads(line) for line in (tmp_path / "transcript.jsonl").read_text().splitlines()]


def test_an_answer_the_service_cut_is_asked_again_without_the_pictures(tmp_path, monkeypatch):
    from mrbench.agent import loop

    cut = RuntimeError('HTTP 400: {"code": "data_inspection_failed"}')
    stats, sent, log = _session(tmp_path, monkeypatch, lambda n: {1: "look", 2: cut, 3: "submit"}[n])
    assert (stats["turns"], stats["inspections"], stats["withdrawn"], stats["errors"]) == (2, 1, 1, 0)
    assert "gave_up" not in stats
    pictures = [sum(p["type"] == "image_url" for m in ms if isinstance(m["content"], list) for p in m["content"])
                for ms in sent]
    assert pictures == [0, 1, 0]  # the picture was in view when the answer was cut, and is not sent again
    assert _lines(sent[2]) == ["The picture(s) you asked for:", loop.WITHDRAWN]
    assert [e.get("withdrawn") for e in log if "error" in e] == [1]


def test_a_session_gives_up_when_the_service_cuts_too_many_answers_in_a_row(tmp_path, monkeypatch):
    from mrbench.agent import loop

    cut = RuntimeError("data_inspection_failed")
    stats, sent, log = _session(tmp_path, monkeypatch, lambda n: cut)
    assert stats["gave_up"] and stats["turns"] == 0 and stats["errors"] == 0  # not counted as server hiccups
    assert stats["inspections"] == len(sent) == loop.INSPECTIONS[0] + 1
    assert log[-1]["type"] == "result" and log[-1]["gave_up"]


def test_the_program_check_stops_at_its_time_bound_and_otherwise_answers_as_without_one():
    import importlib

    from mrbench.agent import env
    from mrbench.agent.cases import case_from_id

    inst = case_from_id("Con-01-exact").inst
    script = importlib.import_module(inst.spec.impl).script(inst)
    free = program.check(inst, script)  # no bound: as the scripted references and `mrbench validate` check
    bounded = program.check(inst, script, timeout_s=env.CHECK_TIMEOUT_S)
    assert free["ok"] and (bounded["ok"], bounded["problems"]) == (free["ok"], free["problems"])
    late = program.check(inst, script, timeout_s=0.0)  # the bound passed before the search starts
    assert not late["ok"] and late["plan"] is None and len(late["problems"]) == 1
    assert late["problems"][0].startswith("inverse kinematics: phase ")
    assert "ran out of time (0 s)" in late["problems"][0] and "not a verdict" in late["problems"][0]
