"""Rea-04's borrow configuration: the mops' geometry, the riser, and the exact scene it keeps."""
from dataclasses import replace
import pytest
from mrbench.tasks.impl import rea04_wipe_spills as task

@pytest.mark.parametrize('level', ['easy', 'hard'])
@pytest.mark.parametrize('seed', range(5))
def test_borrow_keeps_goal_and_the_raised_exact_scene(level, seed):
    exact = task.build(level, seed)
    inst = replace(task.build(level, seed, variant='borrow'), variant='borrow')
    assert inst.instruction == exact.instruction
    assert inst.success == exact.success
    assert inst.milestones == exact.milestones
    assert {a.slot for a in inst.scene.arms} == {'west', 'east'}
    assert 'borrow' in inst.spec.variants
    from mrbench.tasks import build
    registered = build('Rea-04', level, seed, variant='borrow')
    assert registered.metadata['borrow'] == inst.metadata['borrow']
    assert {a.slot for a in registered.scene.arms} == {'west', 'east'}
    for obj in exact.scene.objects:
        assert inst.scene.obj(obj.name) == obj
    riser, spill = exact.scene.obj('riser'), exact.scene.obj('spill_south')
    assert not riser.dynamic and riser.size[2] == pytest.approx(task.RISER_H)
    assert riser.pos[:2] == spill.pos[:2]
    # the spill lies on the riser's top, and the middle wipe counts there
    assert spill.pos[2] - spill.size[1] / 2 == pytest.approx(riser.pos[2] + riser.size[2] / 2)
    wipe = next(r for r in exact.scene.regions if r.name == 'wipe_south')
    assert wipe.center[2] == pytest.approx(task.Z + task.RISER_H + task.SPILL_T + task.SPONGE[2] / 2)
    # no rim round the spill; the spill fits the top; the sponge's middle stays over the top as it scrubs; and a sponge held
    # by its end from an end arm's nearest reach (12 cm, IK) has its middle off the top
    assert not [o for o in exact.scene.objects if o.name.startswith('tray_')]
    assert spill.size[0] <= task.RISER_HALF
    assert task.WIPE_R + task.SCRUB[0] < task.RISER_HALF
    assert 0.12 - (task.SPONGE[0] / 2 + 0.02) > task.RISER_HALF + 0.01

@pytest.mark.parametrize('level', ['easy', 'hard'])
@pytest.mark.parametrize('seed', range(5))
def test_visible_clip_gap_distinguishes_the_equal_mass_mops(level, seed):
    inst = task.build(level, seed, variant='borrow')
    prop = inst.scene.obj(inst.metadata['borrow']['props'][0])
    decoy = inst.scene.obj(inst.metadata['borrow']['decoys'][0])
    def gap(obj):
        a, b = obj.part('jaw_1'), obj.part('jaw_2')
        return abs(a.pos[1] - b.pos[1]) - (a.size[1] + b.size[1]) / 2
    assert gap(prop) > task.SPONGE[1]
    kind = inst.metadata['borrow']['decoy_kind']
    assert kind == task.decoy_kind(level, seed)
    if kind == 'narrow':
        assert gap(decoy) < min(task.SPONGE)
    else:  # wider than the sponge any way round (corner to corner) by more than a clip's slack
        from mrbench.robots.collision import CLIP_LOOSE
        assert gap(decoy) > task.math.hypot(*task.SPONGE[:2]) + CLIP_LOOSE
    # the prop takes the sponge lying flat (it is never narrower across the jaws than its width)
    from mrbench.robots.collision import CLIP_LOOSE as loose
    assert 0 < gap(prop) - task.SPONGE[1] < loose
    assert prop.mass_kg == decoy.mass_kg
    for name in ('jaw_1', 'jaw_2'):
        a, b = prop.part(name), decoy.part(name)
        assert a.size == b.size
        assert (a.pos[0], a.pos[2]) == (b.pos[0], b.pos[2])
    for name in ('neck', 'handle'):
        assert prop.part(name) == decoy.part(name)
    a, b = prop.part('plate'), decoy.part('plate')
    assert a.pos == b.pos
    assert (a.size[0], a.size[2]) == (b.size[0], b.size[2])
    assert a.size[1] - b.size[1] == pytest.approx(gap(prop) - gap(decoy))

def test_decoy_direction_is_balanced_over_seeds():
    for level in ('easy', 'hard'):
        kinds = [task.decoy_kind(level, s) for s in range(5)]
        assert set(kinds) == {'narrow', 'wide'}
    assert task.decoy_kind('easy', 0) != task.decoy_kind('hard', 0)


