"""Tool attachment semantics and payload clearance, without Isaac or agent evaluation."""
from dataclasses import replace
import sys
import types

import pytest

from mrbench.engines.base import SceneSpec, axis_quat, transform
from mrbench.policies.script import Clip, Grip, Move, Unclip, home_pose, offset, phase
from mrbench.robots import collision as col
from mrbench.tasks import build
from mrbench.tasks.impl._common import box, compound, part


def make_tool(opening, centre=(0., 0., 1.)):
    y = opening / 2 + .005
    parts = [part('handle', 'box', (.03, .03, .03)),
             part('post', 'box', (.012, .012, .20), (0., 0., -.10)),
             part('neck', 'box', (.22, .012, .012), (.11, 0., -.20))]
    for k, sign in [(1, -1), (2, 1)]:
        parts += [part(f'jaw_{k}', 'box', (.10, .010, .04), (.22, sign * y, 0.)),
                  part(f'link_{k}', 'box', (.012, .010, .20), (.22, sign * y, -.10))]
    return compound('tool', centre, parts, 'grey', mass=.2)


@pytest.mark.parametrize('opening, fits', [(.058, True), (.036, False), (.110, False), (.060, True), (.061, False)])
def test_clip_uses_geometric_opening_and_target_distance(opening, fits):
    tool = make_tool(opening)
    target = box('payload', (.22, 0.), (.08, .05, .045), z=1.)
    assert col.clip_takes(tool, tool.pose, target, target.pose, .02, .003)[0] is fits
    far = replace(target, pos=(.22, .30, 1.))
    assert not col.clip_takes(tool, tool.pose, far, far.pose, .02, .003)[0]


def test_clip_geometry_is_invariant_under_rigid_transform():
    tool = make_tool(.058)
    pose = (.4, -.2, 1.3, *axis_quat((0., 0., 1.), 42.))
    target = box('payload', (.22, 0.), (.08, .05, .045), z=1.)
    target_pose = (*transform(pose, (.22, 0., 0.)), *pose[3:])
    assert col.clip_takes(tool, pose, target, target_pose, .02, .003)[0]
    assert col.clip_jaws(tool, pose)[2] == pytest.approx(.058)


@pytest.mark.parametrize('same_centres', [False, True])
def test_invalid_jaws_do_not_form_a_clip(same_centres):
    tool = make_tool(0.)
    if same_centres:
        tool = replace(tool, parts=tuple(replace(p, pos=(.22, 0., 0.)) if p.name.startswith('jaw_') else p
                                        for p in tool.parts))
    with col.kin.np.errstate(divide='raise', invalid='raise'):
        assert col.clip_jaws(tool, tool.pose) is None
    assert not col.clip_takes(tool, tool.pose, tool, tool.pose, .02)[0]


def test_program_api_preserves_clip_events_and_reports_unknown_bodies():
    from mrbench.agent.program import run_sandbox, to_script, check

    data, error = run_sandbox("PLAN=[phase(.25,'',clip('tool','payload')),phase(.25,'',unclip('tool'))]", {}, {})
    assert data is not None, error
    script = to_script(data)
    assert script[0].actions == (Clip('tool', 'payload'),)
    assert script[1].actions == (Unclip('tool'),)
    inst = build('Con-02', 'easy', 0, look='default')
    result = check(inst, script)
    assert not result['ok'] and any('no dynamic rigid body' in p for p in result['problems'])


@pytest.mark.parametrize('action', [Clip('obstacle', 'payload'), Clip('tool', 'obstacle'), Unclip('obstacle')])
def test_program_rejects_static_clip_bodies_before_planning(action):
    from mrbench.agent.program import check

    inst, _ = fixture()
    result = check(inst, [phase(.25, 'invalid static attachment', action)])
    assert result['plan'] is None
    assert result['problems'] == ["phase 0: no dynamic rigid body 'obstacle' for the clip action"]


def fixture(opening=.058, payload_size=(.08, .05, .045)):
    base = build('Con-02', 'easy', 0, look='default')
    home = home_pose(base, 'west')
    tool = make_tool(opening, home[:3])
    centre = (home[0] + .22, home[1], home[2])
    target = box('payload', centre[:2], payload_size, z=centre[2])
    obstacle = box('obstacle', (centre[0] + .15, centre[1]), (.006, .006, .006), z=centre[2], dynamic=False)
    scene = SceneSpec((base.scene.arm('west'),), (tool, target, obstacle))
    inst = replace(base, scene=scene, reach=(), out_of_reach=(), apart=())
    script = [phase(.25, 'take the tool', Grip('west', 'tool', .03)),
              phase(.25, 'close its clip', Clip('tool', 'payload')),
              phase(1., 'carry the tool', Move('west', offset(home, dx=.15), allow_flip=False)),
              phase(.5, 'hold before releasing'),
              phase(.25, 'release the payload', Unclip('tool'))]
    return inst, script


@pytest.mark.parametrize('opening, carried', [(.058, True), (.036, False)])
def test_clearance_tracks_only_a_payload_that_fits_the_held_clip(opening, carried):
    from mrbench.policies.plan import plan
    from mrbench.policies.clearance import check

    inst, script = fixture(opening)
    planned = plan(inst, script)
    assert [type(a) for _, a in planned.events][-1] is Unclip
    clashes = check(inst, planned)
    payload_hits = [c for c in clashes if {c.a, c.b} == {'payload', 'obstacle'}]
    assert bool(payload_hits) is carried


def test_swept_cells_include_the_clipped_payload_overhang(monkeypatch):
    from mrbench.scenery.sweep import sweep, cell_of

    inst, script = fixture(payload_size=(.24, .05, .18))
    policy = types.ModuleType('_tool_clip_test_policy')
    policy.script = lambda current: script
    monkeypatch.setitem(sys.modules, policy.__name__, policy)
    inst = replace(inst, spec=replace(inst.spec, impl=policy.__name__))
    cells = sweep(inst)
    tool = inst.scene.obj('tool')
    assert cell_of(tool.pos[0] + .22 + .15 + .10, tool.pos[1]) in cells


def test_a_remote_clip_cannot_attach_a_payload_for_a_later_pickup():
    from mrbench.policies.plan import plan
    from mrbench.policies.clearance import check

    inst, script = fixture()
    script = [script[1], script[0], *script[2:]]
    clashes = check(inst, plan(inst, script))
    assert not any({c.a, c.b} == {'payload', 'obstacle'} for c in clashes)


def test_a_remote_unclip_does_not_release_the_attachment():
    from mrbench.policies.plan import plan
    from mrbench.policies.clearance import check
    from mrbench.policies.script import Release

    inst, script = fixture()
    script = [*script[:2], phase(.25, 'put the tool down', Release('west')),
              phase(.25, 'cannot operate it remotely', Unclip('tool')),
              phase(.25, 'take it again', Grip('west', 'tool', .03)), *script[2:]]
    clashes = check(inst, plan(inst, script))
    assert any({c.a, c.b} == {'payload', 'obstacle'} for c in clashes)


def test_clearance_takes_no_static_body_into_a_clip():
    """A static target (no pose in the walk: it never moves) is not taken, and the walk does not fail on it -- the
    path a task's keyframes or `mrbench check` take, which the agent's check (refusing it earlier) does not."""
    from mrbench.policies.plan import plan
    from mrbench.policies.clearance import check

    inst, script = fixture()
    script = [script[0], phase(.25, 'close its clip on the obstacle', Clip('tool', 'obstacle')), *script[2:]]
    clashes = check(inst, plan(inst, script))  # no KeyError
    assert not any({c.a, c.b} == {'payload', 'obstacle'} for c in clashes)
