"""Which way round a move's hand may go: ``Move.turn`` and ``Move.allow_flip`` are one switch."""

import pytest

from mrbench.policies.plan import _Step, _ways
from mrbench.policies.script import Move

POSE = (0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0)


def _step(move: Move) -> _Step:
    return _Step(t0=0.0, seconds=1.0, phase=0, note="", move=move, ss=[1.0])


@pytest.mark.parametrize("flip", [False, True])
def test_allow_flip_false_is_turn_false(flip):
    assert _ways(flip, _step(Move("west", POSE))) == (flip, not flip)  # either way, the present one first
    assert _ways(flip, _step(Move("west", POSE, allow_flip=False))) == (False,)
    assert _ways(flip, _step(Move("west", POSE, turn=False))) == (False,)
    assert _ways(flip, _step(Move("west", POSE, turn=True))) == (True,)


def test_the_two_names_cannot_ask_for_both_ways():
    with pytest.raises(ValueError):
        Move("west", POSE, allow_flip=False, turn=True)
