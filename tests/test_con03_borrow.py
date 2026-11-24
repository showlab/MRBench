"""Regression guards for the hooked-stay borrow scene and its two-arm policy."""
import pytest

from mrbench.tasks import build
from mrbench.tasks.impl import con03_lid_ball as task


@pytest.mark.parametrize('level', ['easy', 'hard'])
@pytest.mark.parametrize('seed', range(5))
def test_borrow_retains_original_bodies_and_uses_only_present_arms(level, seed):
    exact = build('Con-03', level, seed, look='default')
    borrow = build('Con-03', level, seed, look='default', variant='borrow')
    assert {a.slot for a in borrow.scene.arms} == {'west', 'east'}
    for original in exact.scene.objects:
        assert borrow.scene.obj(original.name) == original
    script = task.script(borrow)
    assert all(action.slot in {'west', 'east'} for ph in script for action in ph.actions)
    assert sum(ph.seconds for ph in script) + .5 <= borrow.spec.horizon_s


def _height(obj):
    from mrbench.robots.collision import boxes_of

    boxes = boxes_of(obj, obj.pose)
    return max(float((b.center + abs(b.rot) @ b.half)[2]) for b in boxes) - min(
        float((b.center - abs(b.rot) @ b.half)[2]) for b in boxes)


def test_the_decoy_is_longer_or_shorter_half_the_time_without_a_mass_cue():
    longer = []
    for level in ('easy', 'hard'):
        for seed in range(5):
            inst = build('Con-03', level, seed, look='default', variant='borrow')
            prop = inst.scene.obj(inst.metadata['borrow']['props'][0])
            decoy = inst.scene.obj(inst.metadata['borrow']['decoys'][0])
            assert prop.mass_kg == decoy.mass_kg
            assert [p.name for p in prop.parts] == [p.name for p in decoy.parts]
            d = _height(decoy) - _height(prop)
            assert abs(d) > .10  # one attribute: the bar's length, 12 cm either way
            longer.append(d > 0)
    assert sum(longer) == len(longer) // 2  # the prop is the bigger stay in exactly half the scenes


@pytest.mark.parametrize('dl', [task.DECOY_DL, -task.DECOY_DL])
def test_neither_distractor_length_can_ever_catch_the_lid_edge(dl):
    import math

    fy, fz = task.stay_pivot()
    edge = [math.hypot(y - fy, z - fz) for d in range(0, 1101) for y, z in task.lid_corners(d / 10)]
    h, lean = task._hook(), math.radians(task.stay_lean())
    du, dv = dl * math.sin(lean), dl * math.cos(lean)
    tip_low = math.hypot(h['tip'] + du, h['under'] - task.HOOK_TIP + dv)
    bight = max(math.hypot(h['top'][0] + task.STAY_BAR / 2 + du, h['under'] + dv), math.hypot(h['tip'] + du, h['under'] + dv))
    if dl > 0:
        assert tip_low > max(edge) + .04  # the hook passes beyond the edge at every opening
    else:
        assert bight < min(edge) - .04  # the edge never gets under the bight
