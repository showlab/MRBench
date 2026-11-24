"""The skill-level executor (``agent/skill_exec.py``, docs/BASELINES.md) without the simulator: which script a
team plan gets, and that every gold team of the 26 feasible versions has one."""

import pytest

from mrbench.agent.cases import case_from_id
from mrbench.agent.skill_exec import REFERENCE, choose, gold_plan_of, library

FEASIBLE = [f"{t}-{v}" for v in ("exact", "surplus", "borrow") for t in
            ("Con-01", "Con-02", "Con-03", "Con-04", "Rea-01", "Rea-02", "Rea-03", "Rea-04")] + \
           ["Rea-02-swapped", "Rea-04-swapped"]


def plan(team, things=(), feasible=True):
    return {"feasible": feasible, "arms_used": list(team), "things_used": [{"thing": t, "job": "-"} for t in things]}


@pytest.mark.parametrize("case_id", FEASIBLE)
def test_every_gold_team_of_a_feasible_version_has_a_skill_with_that_team(case_id):
    case = case_from_id(case_id)
    teams = {s.team for s in library(case)}
    for gold in case.expect.minimal:
        assert gold in teams, (case_id, sorted(gold))
    skill, why = choose(case, gold_plan_of(case))
    assert skill is not None and skill.team in case.expect.minimal, why


def test_an_exact_version_gets_its_reference():
    skill, _ = choose(case_from_id("Con-01-exact"), plan(["east", "south", "west"]))
    assert skill.name == REFERENCE and skill.team == {"east", "south", "west"}


def test_too_few_arms_or_a_prop_not_named_get_no_skill_and_say_why():
    skill, why = choose(case_from_id("Con-01-exact"), plan(["south", "west"]))
    assert skill is None and "arms not in the team: east" in why
    case = case_from_id("Con-01-borrow")
    skill, why = choose(case, plan(["south", "west"]))
    assert skill is None and "things not named: block_2" in why
    assert choose(case, plan(["south", "west"], ["block_2"]))[0].name == REFERENCE


def test_a_plan_that_says_infeasible_gets_no_skill():
    skill, why = choose(case_from_id("Con-01-exact"), plan([], feasible=False))
    assert skill is None and "cannot be done" in why


def test_a_larger_team_gets_the_fewest_armed_skill_inside_it():
    case = case_from_id("Rea-01-surplus")  # gold north + west, a route; the reference moves east, south and west
    skill, _ = choose(case, plan(["east", "north", "south", "west"]))
    assert skill.name == "Rea-01-surplus-nw" and skill.team == {"north", "west"}
    assert choose(case, plan(["east", "south", "west"]))[0].name == REFERENCE


def test_a_route_that_moves_an_idle_arm_aside_says_so():
    s = {s.name: s for s in library(case_from_id("Rea-02-swapped"))}["Rea-02-swapped"]
    assert s.team == {"east", "west"} and s.parked == {"south"}
