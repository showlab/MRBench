"""The job model and its solver (``tasks/jobs.py``): on a toy, and against the gold of the benchmark's eight."""

from types import SimpleNamespace

import pytest

from mrbench.tasks.jobs import ANOTHER, AnyOf, Capability, Job, Jobs, Thing, solve


class Toy(Capability):
    """Who reaches what, given: {(arm, site)} reached and {(arm, arm)} that meet."""

    def __init__(self, reached, meets=()):
        self.reached, self.meets = set(reached), {frozenset(m) for m in meets}

    def reach(self, slot, site, slack=0.0):
        sites = () if site is None else site.points if isinstance(site, AnyOf) else (site,)
        return any((slot, s) in self.reached for s in sites)

    def meet(self, a, b, slack):
        return frozenset((a, b)) in self.meets


def team(*arms):
    return SimpleNamespace(scene=SimpleNamespace(arms=[SimpleNamespace(slot=a) for a in arms]))


def sets(*names):
    return tuple(frozenset(n.split("+")) for n in names)


HOLD = Jobs((Job("hold w", sites=("kw",), group="g"), Job("hold e", sites=("ke",), group="g"),
             Job("draw", sites=("h",), group="g")))


def test_three_jobs_at_once_take_three_doers_even_if_one_can_do_two():
    cap = Toy({("w", "kw"), ("e", "ke"), ("s", "h"), ("s", "ke"), ("s", "kw")})
    a = solve(team("w", "e", "s"), decl=HOLD, cap=cap)
    assert a.feasible and a.minimal == sets("e+s+w")
    a = solve(team("w", "s"), decl=HOLD, cap=cap)
    # every job has a doer, but not all at once: one more hand for either job only the south arm can do
    assert not a.feasible and set(a.missing) == {"hold e", "draw"}
    a = solve(team("w", "s"), decl=HOLD, cap=Toy({("w", "kw"), ("s", "h")}))
    assert not a.feasible and a.missing == ("hold e",)  # no doer at all: named alone


def test_surplus_lists_every_fewest_team():
    cap = Toy({("w", "kw"), ("e", "ke"), ("s", "h"), ("n", "kw")})
    a = solve(team("w", "e", "s", "n"), decl=HOLD, cap=cap)
    assert a.feasible and set(a.minimal) == set(sets("w+e+s", "n+e+s"))


def test_a_carry_goes_by_one_doer_or_a_chain_of_doers_that_meet():
    relay = Jobs((Job("carry", "carry", sites=("start", "crate")),))
    cap = Toy({("w", "start"), ("e", "crate"), ("n", "crate")}, meets={("w", "s"), ("s", "e"), ("w", "n")})
    assert solve(team("w", "s", "e"), decl=relay, cap=cap).minimal == sets("w+s+e")
    assert solve(team("w", "s", "e", "n"), decl=relay, cap=cap).minimal == sets("w+n")
    a = solve(team("w", "e"), decl=relay, cap=cap)
    assert not a.feasible and a.missing == ("carry",)
    one = Toy({("w", "start"), ("w", "crate")})
    assert solve(team("w", "e"), decl=relay, cap=one).minimal == sets("w")


def test_a_thing_does_a_job_once_an_arm_has_placed_it_and_a_distractor_does_none():
    decl = Jobs(HOLD.jobs, (Thing("prop", does=("hold e",), placing=(Job("set prop", sites=("p",)),)),
                            Thing("decoy", why="too low")))
    cap = Toy({("w", "kw"), ("s", "h"), ("s", "p")})
    a = solve(team("w", "s"), decl=decl, cap=cap)
    assert a.feasible and a.minimal == sets("w+s") and a.things_used == ("prop",) and a.things_rejected == ("decoy",)
    a = solve(team("w", "s"), decl=decl, cap=Toy({("w", "kw"), ("s", "h")}))  # no one can set it
    assert not a.feasible and a.missing == ("hold e",)
    # with every arm there the prop is not needed: the fewest arms use none
    a = solve(team("w", "e", "s"), decl=decl, cap=Toy({("w", "kw"), ("e", "ke"), ("s", "h"), ("s", "p")}))
    assert a.minimal == sets("w+s") and a.things_used == ("prop",)
    assert solve(team("w", "e", "s"), decl=Jobs(HOLD.jobs, decl.things[1:]),
                 cap=Toy({("w", "kw"), ("e", "ke"), ("s", "h")})).things_used == ()


def test_a_site_may_be_any_of_several_points_and_none_is_nothing_to_take():
    decl = Jobs((Job("hold", sites=(AnyOf(("a", "b")),)), Job("fill", "carry", sites=(None, "x"))),
                (Thing("plank", does=("fill",), placing=(Job("stand plank", sites=("q",)),)),))
    a = solve(team("w"), decl=decl, cap=Toy({("w", "b"), ("w", "x"), ("w", "q")}))
    assert a.feasible and a.minimal == sets("w") and a.things_used == ("plank",)
    a = solve(team("w"), decl=Jobs(decl.jobs), cap=Toy({("w", "b"), ("w", "x")}))
    assert not a.feasible and a.missing == ("fill",) and ANOTHER not in a.missing


def test_jobs_are_checked_when_declared():
    with pytest.raises(ValueError):
        Job("x", "carry", sites=("a",))
    with pytest.raises(ValueError):
        Jobs((Job("x"),), (Thing("t", does=("y",)),))


# ---------------------------------------------------------------------------------------------- the eight's gold
BENCHMARK = ("Con-01", "Con-02", "Con-03", "Con-04", "Rea-01", "Rea-02", "Rea-03", "Rea-04")


@pytest.mark.parametrize("task", BENCHMARK)
def test_the_solver_gives_every_gold_answer_of_the_eight(task):
    """Every version of the task, at both levels and seeds 0-5, solved with the IK results kept in tasks/facts (a
    missing one fails: ``python -m mrbench.tasks.jobs facts TASK``): one answer per version, the one in gold.json,
    and the one its agent case grades with."""
    import json

    from mrbench.agent import gold
    from mrbench.agent.cases import GOLD, make_case
    from mrbench.tasks.jobs import Facts

    solved, differ = gold.compute([task], Facts(offline=True))
    assert not differ, differ
    assert solved[task] == json.loads(GOLD.read_text())[task]
    assert set(solved[task]) == set(gold.variants(task)) >= {"exact", "surplus", "shortfall", "swapped"}
    for v, g in solved[task].items():
        for level in ("easy", "hard"):
            e = make_case(task, v, level).expect
            assert (e.feasible, sorted(sorted(s) for s in e.minimal)) == (g["feasible"], g["minimal"]), (v, level)
            assert e.missing == "; ".join(g["missing"][level]) and bool(e.missing) == (not e.feasible)
            if v == "borrow":  # the prop used, the distractor left
                b = make_case(task, v, level).inst.metadata["borrow"]
                assert (list(e.things), list(e.rejected)) == (b["props"], b["decoys"])


@pytest.mark.parametrize("task", BENCHMARK)
def test_the_versions_change_the_answer(task):
    """The exact team is a minimal support set and the only one; the shortfall cannot be done; the swapped team,
    every place still staffed, cannot do it or does it with fewer arms; the borrow version takes the prop."""
    from mrbench.agent.cases import make_case

    exact = make_case(task, "exact").expect
    slots = {a.slot for a in make_case(task, "exact").inst.scene.arms}
    assert exact.feasible and exact.minimal == (frozenset(slots),)
    assert not make_case(task, "shortfall").expect.feasible
    c = make_case(task, "swapped")
    assert c.swapped and {a.slot for a in c.inst.scene.arms} == slots
    assert c.inst.scene.arm(c.swapped[0]).model == c.swapped[1] != make_case(task, "exact").inst.scene.arm(
        c.swapped[0]).model
    e = c.expect
    assert not e.feasible or len(e.minimal[0]) < len(slots)
    surplus = make_case(task, "surplus").expect
    assert surplus.feasible and all(len(s) <= len(slots) for s in surplus.minimal)
