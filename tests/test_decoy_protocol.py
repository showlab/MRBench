"""The distractor-run protocol (``mrbench demo --decoy``): which clashes count against a negative, and the safe stops."""
import importlib

from mrbench.demo import decoy_protocol
from mrbench.engines.base import Frame
from mrbench.policies.clearance import Clash
from mrbench.tasks import build


class _Result:
    def __init__(self, success):
        self.success = success


def test_a_clash_into_something_other_than_the_distractor_spoils_the_negative():
    inst = build("Con-02", "easy", 0, variant="borrow", decoy=True)
    d = inst.metadata["borrow"]["props"][0]  # the distractor, under the prop's name
    defect = Clash(30.0, "push", f"{d}.body", "shield_w", -0.02)
    near = Clash(31.0, "", "south.hand", "table", 0.008)
    into = Clash(32.0, "", "south.hand", "table", -0.004)
    ok = decoy_protocol(inst, [defect, near], _Result(False), [])
    assert ok["valid_negative"] and ok["clashes_with_distractor"] and not ok["other_clashes_into"]
    bad = decoy_protocol(inst, [defect, into], _Result(False), [])
    assert not bad["valid_negative"] and bad["other_clashes_into"]
    assert not decoy_protocol(inst, [defect], _Result(True), [])["valid_negative"]  # it did the task: no negative
    # stopped safely at 31.5 s: the plan's clash at 32 s was never played
    stopped = decoy_protocol(inst, [defect, into], _Result(False), [], safe_stop=31.5)
    assert stopped["valid_negative"] and not stopped["other_clashes_into"] and stopped["safe_stop_s"] == 31.5


def test_the_safe_stops_hold_at_the_start_so_they_must_arm_first():
    for task in ("Con-01",):
        inst = build(task, "easy", 0, variant="borrow", decoy=True)
        stop = importlib.import_module(inst.spec.impl).decoy_stop(inst)
        shut = Frame(t=0.0, objects={o.name: o.pose for o in inst.scene.dynamic_objects})
        assert stop(shut, inst.scene)  # door or lid shut at the start: the stop arms only once it has been open
