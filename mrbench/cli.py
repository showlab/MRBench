"""`mrbench` command line."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

from .core.errors import MRBenchError
from .core.paths import runs_dir
from .tasks.spec import VARIANTS


def _seeds(text: str) -> list[int]:
    out: list[int] = []
    for part in text.split(","):
        a, _, b = part.partition("-")
        out += list(range(int(a), int(b) + 1)) if b else [int(a)]
    return out


def cmd_doctor(args) -> int:
    from .doctor import FAIL, run

    checks = run()
    for c in checks:
        print(f"[{c.status:>4}] {c.name:<18} {c.detail}")
    return 1 if any(c.status == FAIL for c in checks) else 0


def cmd_robots(args) -> int:
    from .robots import MODELS

    print(f"{'name':<7} {'model':<20} {'arm DoF':>7} {'gripper':<42} {'reach':>6} {'payload':>8}  source")
    for m in MODELS.values():
        print(f"{m.name:<7} {m.display_name:<20} {m.arm_dof:>7} {', '.join(m.gripper_joints):<42} "
              f"{m.reach_m:>5.2f}m {m.payload_kg:>7.1f}kg  {m.source}")
    return 0


def cmd_tasks(args) -> int:
    from .tasks import all_specs

    specs = all_specs().values()
    if args.markdown:
        print("| id | suite | name | arms (mount: model) | knob easy / hard | versions built | status |")
        print("|---|---|---|---|---|---|---|")
        for s in specs:
            arms = ", ".join(f"{a.mount}: {a.model}" for a in s.arms)
            k = s.knob
            print(f"| {s.id} | {s.suite} | {s.name} | {arms} | {k.field} {k.levels['easy']} / "
                  f"{k.levels['hard']} {k.unit} | {', '.join(s.variants)} | {s.status} |")
        return 0
    for s in specs:
        print(f"{s.id:<4} {s.suite:<12} {s.name:<26} {'+'.join(s.models):<20} {','.join(s.variants):<8} {s.status}")
    return 0


def cmd_show(args) -> int:
    from .tasks import build, get

    from .tasks.spec import VARIANTS

    s = get(args.task)
    print(f"{s.id}  {s.name}  [{s.suite}, {s.status}]\n\n{s.summary}\n")
    if s.facts:
        print("facts (told to the agent):\n" + "\n".join(f"  - {f}" for f in s.facts) + "\n")
    print("exact team, with the reference roles (never told to the agent):")
    for a in s.arms:
        print(f"  {a.mount:<6} {a.model:<6} {a.role}")
    print(f"\nversions built: {', '.join(s.variants)}; planned: {', '.join(v for v in VARIANTS if v not in s.variants)}")
    if s.borrow:
        b = s.borrow
        print(f"borrow: a prop instead of the arm {b.replaces}" + (f" (the {b.remove} arm taken away)" if b.remove else "")
              + f"\n  prop: {b.prop}\n  distractor: {b.distractor}")
    print(f"\nknob: {s.knob.field} = {dict(s.knob.levels)} {s.knob.unit}   horizon: {s.horizon_s:g} s"
          + "".join(f", {v} {h:g} s" for v, h in s.horizons))
    print(f"success: {s.success}\nmilestones: {', '.join(s.milestones)}")
    if s.todo:
        print("todo:\n" + "\n".join(f"  - {t}" for t in s.todo))
    print("\nwho could stand where (reach rules; ok = can do the role, alone = could then do without another arm):")
    print(_fits_text(s.id))
    inst = build(s.id, args.level, args.seed, args.look)
    print(f"\n[{args.level}, seed {args.seed}] goal: {inst.instruction}")
    for o in inst.scene.dynamic_objects:
        print(f"  {o.name:<10} {o.shape:<8} size={o.size} pos=({o.pos[0]:+.3f}, {o.pos[1]:+.3f}, {o.pos[2]:.3f})")
    if args.look or s.look != "default":
        from .scenery import describe

        print(f"\nlook: {describe(inst.scene.look)}")
        for p in inst.scene.props:
            print(f"  {p.name:<8} {p.kind:<26} pos=({p.pos[0]:+.3f}, {p.pos[1]:+.3f})")
    return 0


def _fits_text(task: str) -> str:
    from .robots import MODELS
    from .tasks.combos import combinations, mount_fits

    lines = []
    for slot, fits in mount_fits(task).items():
        cells = [f"{m} {'ok' if f.can_do else '--'}{'' if f.needs_others or not f.can_do else ' alone'}"
                 for m, f in fits.items()]
        lines.append(f"  {slot:<6} needs {next(iter(fits.values())).farthest:.2f} m   " + "   ".join(cells))
    combos = combinations(task)
    can = [c for c, ok, _ in combos if ok]
    needed = [c for c, ok, need in combos if ok and need]
    lines.append(f"  {len(can)} of {len(combos)} combinations of {', '.join(MODELS)} can do it; "
                 f"{len(needed)} of them still need all three arms")
    return "\n".join(lines)


def cmd_brief(args) -> int:
    from .tasks import build
    from .tasks.brief import brief

    inst = build(args.task, args.level, args.seed, args.look)
    print(json.dumps(brief(inst, args.info), indent=1 if args.pretty else None, ensure_ascii=False))
    return 0


def cmd_combos(args) -> int:
    from .tasks import all_specs

    for tid in args.tasks or list(all_specs()):
        print(f"{tid}\n{_fits_text(tid)}")
    return 0


def cmd_looks(args) -> int:
    from .scenery import BACKDROPS, CLUTTER, FLOORS, FORMS, LIGHTS, LOOKS, ROOMS, TABLES, describe

    from .tasks import all_specs

    print("named looks (--look NAME; --look random draws one from the seed):")
    for name, look in LOOKS.items():
        print(f"  {name:<9} {describe(look)}")
    print("\neach task's own (its card's `look`; --look overrides it):")
    print("  " + "  ".join(f"{t} {s.look}" for t, s in all_specs().items()))
    print("\nchanges (--look NAME,key=value,...):")
    print(f"  table=   {', '.join(TABLES)}")
    print(f"  form=    {', '.join(FORMS)}    top=  the top's thickness (m), with legs")
    print(f"  floor=   {', '.join(FLOORS)}, room (the room's own)")
    print(f"  room=    {', '.join(ROOMS)}")
    print(f"  backdrop= {', '.join(BACKDROPS)} (a furnished corner behind the table, drawn only)")
    print(f"  light=   {', '.join(LIGHTS)}")
    print("  clutter= SET or SET:N (N of its groups, each one or two props set out together) or none")
    for name, groups in CLUTTER.items():
        print(f"    {name:<8} {', '.join(groups) if name != 'mixed' else 'every group'}")
    print("  colors=  card (the task's own colours) or random (the things its impl names as painted, drawn from the "
          "seed; the instruction's colour words follow)")
    print("  distract= N everyday things within the arms' reach that the task does not need (scenery.distractors)")
    return 0


def cmd_validate(args) -> int:
    from .tasks import get
    from .tasks.validate import validate

    for task in args.tasks:
        get(task)  # An unknown task must not silently pass an empty validation.
    problems = validate(args.tasks or None, args.variant)
    for p in problems:
        print(f"- {p}")
    print("all design rules hold"
          if not problems else f"{len(problems)} problem(s)")
    if getattr(args, "ik", False):  # the reach claims checked with the arms' kinematics (slow: a minute a task)
        from .tasks import build, ids
        from .tasks.reachability import check

        found = []
        from .tasks import get

        for t in args.tasks or ids():
            if get(t).status == "planned" or args.variant not in get(t).variants:
                continue
            try:
                found += check(build(t, "easy", 0, variant=args.variant), slack=args.slack)
            except Exception as e:  # noqa: BLE001 -- a task that does not build is validate's to report
                found.append(f"{t}: not checked ({e})")
        for f in found:
            print(f"- reach (IK): {f}")
        print("reach claims hold by IK" if not found else f"{len(found)} reach claim(s) broken by IK")
        if args.strict and found:
            return 1
    return 1 if problems else 0


def cmd_assets(args) -> int:
    from .robots import fetch as fetch_mod
    from .robots import MODELS

    names = args.robots or [n for n, m in MODELS.items() if m.source == "urdf"]
    if args.action == "status":
        for n in names:
            print(f"{n:<7} {fetch_mod.status(n)['detail']}")
    elif args.action == "fetch":
        for n in names:
            print(f"{n:<7} -> {fetch_mod.fetch(n, force=args.force)}")
    elif args.action == "looks":  # the photographs the looks are made of
        from .scenery.downloads import fetch_all

        files = fetch_all()
        print(f"{len(files)} files, checked, in {files[0].parents[1]}")
    elif args.action == "convert":
        from .engines.isaac import app
        from .engines.isaac.convert import convert

        app.launch(headless=True)
        try:
            for n in names:
                print(f"{n:<7} -> {convert(n, force=args.force)}")
        finally:
            app.close()
    return 0


def cmd_run(args) -> int:
    from .engines import make_engine
    from .policies import make_policy
    from .runner import log as runlog
    from .runner.episode import replay_keyframes, run_episode
    from .tasks import build

    out = Path(args.output) if args.output else runs_dir() / f"{time.strftime('%Y%m%d-%H%M%S')}.jsonl"
    jobs = [(t, lv, s) for t in args.tasks for lv in args.levels for s in _seeds(args.seeds)]

    if args.isolate and len(jobs) > 1:
        rc = 0
        for t, lv, s in jobs:
            cmd = [sys.executable, "-m", "mrbench", "run", t, "--levels", lv, "--seeds", str(s), "--engine",
                   args.engine, "--policy", args.policy, "-o", str(out)] + (["--gui"] if args.gui else []) \
                + (["--look", args.look] if args.look else [])
            rc |= subprocess.call(cmd)
        return rc

    engine = make_engine(args.engine, headless=not args.gui)
    try:
        for t, lv, s in jobs:
            inst = build(t, lv, s, args.look)
            if args.policy == "oracle":
                res = replay_keyframes(inst, engine)
            else:
                res = run_episode(inst, engine, make_policy(args.policy), args.horizon, policy_name=args.policy)
            runlog.append(out, res.to_json(keep_frames=args.keep_frames))
            print(f"{t}/{lv}/s{s}: success={res.success} progress={res.progress:.2f} "
                  f"milestones={res.milestones_hit} sim={res.sim_time_s}s wall={res.wall_time_s}s")
    finally:
        engine.close()
    print(f"log -> {out}")
    return 0


def _look_dir(look: str) -> Path:
    """Where a run in another look than the default writes, unless told: never over the docs' pictures."""
    return runs_dir() / "looks" / "".join(c if c.isalnum() or c in "-_." else "_" for c in look)


def cmd_render(args) -> int:
    from . import render
    from .core.paths import REPO_ROOT
    from .tasks import all_specs

    raw = Path(args.raw) if args.raw else _look_dir(args.look) / "raw" if args.look else runs_dir() / "render"
    if args.child:  # one task-level in this process (the parent spawned it), then leave without Kit's teardown
        import traceback

        code = 0
        try:
            print(json.dumps(render.render_level(args.tasks[0], args.levels[0], args.seed, raw, look=args.look,
                                                 variant=args.variant), indent=1))
        except Exception:
            traceback.print_exc()
            code = 1
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(code)
    tasks = args.tasks or list(all_specs())
    out = Path(args.output) if args.output else _look_dir(args.look) if args.look else \
        REPO_ROOT / "docs" / "media" / "tasks"
    if args.compose_only:
        from .tasks import tag

        for t in tasks:
            for view in render.VIEW_NAMES:
                print(f"{t}: {render.compose(tag(t, args.variant), raw, out, view)}")
        return 0
    return render.render_all(tasks, args.seed, raw, out, args.gpus.split(","), levels=args.levels, look=args.look,
                             variant=args.variant)


def cmd_demo(args) -> int:
    from . import demo
    from .core.paths import REPO_ROOT
    from .tasks import all_specs

    out = Path(args.output) if args.output else _look_dir(args.look) if args.look else \
        REPO_ROOT / "docs" / "media" / "videos"
    if args.child:  # one task-level in this process, then leave without Kit's teardown
        import traceback

        code = 0
        try:
            print(json.dumps(demo.demo_level(args.tasks[0], args.levels[0], args.seed, out, look=args.look,
                                             variant=args.variant, decoy=args.decoy), indent=1))
        except Exception:
            traceback.print_exc()
            code = 1
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(code)
    tasks = args.tasks or [t for t in all_specs() if demo.has_script(t)]
    missing = [t for t in tasks if not demo.has_script(t)]
    if missing:
        print(f"error: no scripted reference for {', '.join(missing)}", file=sys.stderr)
        return 2
    return demo.demo_all(tasks, args.levels, args.seed, out, args.gpus.split(","), look=args.look,
                         variant=args.variant, decoy=args.decoy)


def cmd_check(args) -> int:
    """Plan each task's scripted reference (IK along every path) and walk it for clearance, listing every clash --
    what ``mrbench demo`` refuses a plan for, without Isaac."""
    import importlib

    from .agent import program
    from .tasks import build

    program.MAX_CLASHES = 1000
    bad = 0
    for t in args.tasks:
        for lv in args.levels:
            inst = build(t, lv, args.seed, variant=args.variant)
            mod = importlib.import_module(inst.spec.impl)
            if not hasattr(mod, "script"):
                print(f"{t}/{lv}: no scripted reference")
                continue
            r = program.check(inst, mod.script(inst))
            print(f"{t}/{lv}: {'clear' if r['ok'] else 'NOT clear'}")
            for p in r["problems"]:
                print(f"  - {p}")
            for w in r.get("warnings") or []:
                print(f"  (warning) {w}")
            bad += not r["ok"]
    return 1 if bad else 0


def cmd_batch(args) -> int:
    from . import batch

    out = Path(args.output) if args.output else runs_dir() / "batch" / time.strftime("%Y%m%d-%H%M%S")
    if args.summary:
        print(batch.summarize(out))
        return 0
    todo = batch.scenes(args.tasks or list(batch.BENCHMARK), args.variants, args.levels, _seeds(args.seeds))
    if args.child:  # one scene in this process (the parent spawned it), then leave without Kit's teardown
        import traceback

        code = 0
        try:
            r = batch.play_scene(todo[0], out, args.robots, args.masses, args.look, video=not args.no_video,
                                 sizes=args.sizes)
            print(json.dumps({k: r.get(k) for k in ("scene", "success", "milestones_hit", "missed_grips")}))
        except Exception:
            traceback.print_exc()
            code = 1
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(code)
    print(f"{len(todo)} scene(s) -> {out}")
    return 1 if batch.run(todo, out, args.gpus.split(","), args.robots, args.masses, args.look, args.dry, args.force,
                          args.workers, video=not args.no_video, timeout_s=args.timeout or None,
                          sizes=args.sizes) else 0


def cmd_summary(args) -> int:
    from collections import defaultdict

    from .runner import log as runlog

    groups: dict = defaultdict(list)
    for r in runlog.read(Path(args.log)):
        groups[(r["task"], r["level"], r["engine"], r["policy"])].append(r)
    print(f"{'task':<5} {'level':<5} {'engine':<6} {'policy':<16} {'n':>3} {'success':>8} {'progress':>9}")
    for (t, lv, e, p), rs in sorted(groups.items()):
        sr = sum(r["success"] for r in rs) / len(rs)
        pr = sum(r["progress"] for r in rs) / len(rs)
        print(f"{t:<5} {lv:<5} {e:<6} {p:<16} {len(rs):>3} {sr:>8.2f} {pr:>9.2f}")
    return 0


LOOK_HELP = ("how the scene looks (default: the task card's): a named look, random, and key=value changes "
             "(see `mrbench looks`)")
VARIANT_HELP = "the configuration: exact, surplus, shortfall, swapped or borrow"


def cmd_snapshot(args) -> int:
    from .agent import bench, snapshot

    ids = bench.scenes(args.scenes)
    argv = ["snapshot", "all", "--gpus", args.gpus, "--cases", ",".join(ids)] + (["--force"] if args.force else [])
    return snapshot.main(argv)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="mrbench", description="MRBench: heterogeneous multi-arm tasks in Isaac Sim")
    sub = p.add_subparsers(dest="cmd", required=True)

    sub.add_parser("doctor", help="check Python, GPU, Isaac Sim, assets").set_defaults(fn=cmd_doctor)
    sub.add_parser("robots", help="list robot models").set_defaults(fn=cmd_robots)

    t = sub.add_parser("tasks", help="list tasks")
    t.add_argument("--markdown", action="store_true")
    t.set_defaults(fn=cmd_tasks)

    s = sub.add_parser("show", help="show one task card and a built instance")
    s.add_argument("task")
    s.add_argument("--level", default="easy", choices=("easy", "hard"))
    s.add_argument("--seed", type=int, default=0)
    s.add_argument("--look", default=None, help=LOOK_HELP)
    s.set_defaults(fn=cmd_show)

    b = sub.add_parser("brief", help="what an agent is told about one task instance (JSON)")
    b.add_argument("task")
    b.add_argument("--level", default="easy", choices=("easy", "hard"))
    b.add_argument("--seed", type=int, default=0)
    b.add_argument("--info", default="given", choices=("given", "query"),
                   help="given: the arms' capabilities too; query: the arms only by name and place")
    b.add_argument("--look", default=None, help=LOOK_HELP)
    b.add_argument("--pretty", action="store_true")
    b.set_defaults(fn=cmd_brief)

    sub.add_parser("looks", help="the tables, floors, rooms, lights and props a scene can have").set_defaults(
        fn=cmd_looks)

    c = sub.add_parser("combos", help="which robots could stand on each mount, task by task")
    c.add_argument("tasks", nargs="*")
    c.set_defaults(fn=cmd_combos)

    v = sub.add_parser("validate", help="check every design rule (no GPU needed)")
    v.add_argument("--ik", action="store_true", help="also check the reach claims with the arms' inverse kinematics")
    v.add_argument("--strict", action="store_true", help="with --ik: a broken reach claim fails")
    v.add_argument("--slack", type=float, default=0.0,
                   help="with --ik: also count an arm that reaches a body's surface within a grip's reach (2 cm), "
                        "or a place this far short of its middle (m)")
    v.add_argument("--configuration", "--variant", dest="variant", default="exact", choices=VARIANTS, help=VARIANT_HELP)
    v.add_argument("tasks", nargs="*")
    v.set_defaults(fn=cmd_validate)

    a = sub.add_parser("assets", help="fetch robots and looks / convert robots to USD")
    a.add_argument("action", choices=ASSET_ACTIONS)
    a.add_argument("robots", nargs="*")
    a.add_argument("--force", action="store_true")
    a.set_defaults(fn=cmd_assets)

    r = sub.add_parser("run", help="run episodes")
    r.add_argument("tasks", nargs="+")
    r.add_argument("--engine", default="mock", choices=("mock", "isaac"))
    r.add_argument("--policy", default="do_nothing", help="do_nothing | random | oracle (state teleport, mock only)")
    r.add_argument("--levels", nargs="+", default=["easy"], choices=("easy", "hard"))
    r.add_argument("--seeds", default="0", help="e.g. 0-4 or 0,3,7")
    r.add_argument("--horizon", type=float, default=None, help="override the task horizon (s)")
    r.add_argument("--isolate", action="store_true", help="one process per episode (needed for several Isaac tasks)")
    r.add_argument("--gui", action="store_true", help="Isaac: open the viewer instead of running headless")
    r.add_argument("--keep-frames", action="store_true", help="store every frame in the log")
    r.add_argument("--look", default=None, help=LOOK_HELP)
    r.add_argument("-o", "--output", default=None, help="JSONL log (default runs/<timestamp>.jsonl)")
    r.set_defaults(fn=cmd_run)

    d = sub.add_parser("render", help="Isaac: pictures of each task's start and goal, both levels")
    d.add_argument("tasks", nargs="*", help="default: every task")
    d.add_argument("--levels", nargs="+", default=["easy", "hard"], choices=("easy", "hard"))
    d.add_argument("--seed", type=int, default=0)
    d.add_argument("--gpus", default=os.environ.get("MRBENCH_ISAAC_GPU", "0"),
                   help="comma-separated GPU indices; one Isaac process per GPU at a time")
    d.add_argument("--raw", default=None,
                   help="per-view PNGs and settle reports (default runs/render; with --look, runs/looks/<look>/raw)")
    d.add_argument("-o", "--output", default=None,
                   help="composed strips (default docs/media/tasks; with --look, runs/looks/<look>)")
    d.add_argument("--compose-only", action="store_true", help="only lay out what --raw already holds")
    d.add_argument("--look", default=None, help=LOOK_HELP)
    d.add_argument("--configuration", "--variant", dest="variant", default="exact", choices=VARIANTS, help=VARIANT_HELP)
    d.add_argument("--child", action="store_true", help=argparse.SUPPRESS)
    d.set_defaults(fn=cmd_render)

    c = sub.add_parser("check", help="plan a task's scripted reference and check its clearance (no Isaac)")
    c.add_argument("tasks", nargs="+")
    c.add_argument("--levels", nargs="+", default=["easy", "hard"], choices=("easy", "hard"))
    c.add_argument("--seed", type=int, default=0)
    c.add_argument("--configuration", "--variant", dest="variant", default="exact", choices=VARIANTS, help=VARIANT_HELP)
    c.set_defaults(fn=cmd_check)

    v = sub.add_parser("demo", help="Isaac: a video of the scripted reference doing a task (physics on)")
    v.add_argument("tasks", nargs="*", help="default: every task with a scripted reference")
    v.add_argument("--levels", nargs="+", default=["easy", "hard"], choices=("easy", "hard"))
    v.add_argument("--seed", type=int, default=0)
    v.add_argument("--gpus", default=os.environ.get("MRBENCH_ISAAC_GPU", "0"),
                   help="comma-separated GPU indices; one Isaac process per GPU at a time")
    v.add_argument("-o", "--output", default=None,
                   help="videos and scores (default docs/media/videos; with --look, runs/looks/<look>)")
    v.add_argument("--look", default=None, help=LOOK_HELP)
    v.add_argument("--configuration", "--variant", dest="variant", default="exact", choices=VARIANTS, help=VARIANT_HELP)
    v.add_argument("--decoy", action="store_true",
                   help="borrow version: the prop and the distractor swap names, so the reference takes the "
                        "distractor (it should fail)")
    v.add_argument("--child", action="store_true", help=argparse.SUPPRESS)
    v.set_defaults(fn=cmd_demo)

    b = sub.add_parser("batch", help="Isaac: many seeded scenes, each played with its scripted reference; a success "
                                     "table and the failures")
    b.add_argument("tasks", nargs="*", help="default: the benchmark's eight")
    b.add_argument("--configurations", "--variants", dest="variants", nargs="+", default=["exact"], choices=VARIANTS,
                   help="the versions to play, each where a task has built it")
    b.add_argument("--levels", nargs="+", default=["easy", "hard"], choices=("easy", "hard"))
    b.add_argument("--seeds", default="0-2", help="e.g. 0-29 or 0,3,7")
    b.add_argument("--robots", default="card", choices=("card", "random"),
                   help="random: per seed, a robot combination that can do the task and still needs every arm")
    b.add_argument("--masses", default="card", choices=("card", "random"),
                   help="random: per seed, every movable body's mass times a factor in batch.MASS_RANGE")
    b.add_argument("--sizes", default="card", choices=("card", "random"),
                   help="random: per seed, a task's free object's size (batch.SIZES) times a factor drawn")
    b.add_argument("--look", default=None, help=LOOK_HELP)
    b.add_argument("--gpus", default=os.environ.get("MRBENCH_ISAAC_GPU", "0"),
                   help="comma-separated GPU indices; one Isaac process per GPU at a time")
    b.add_argument("--dry", action="store_true", help="no Isaac: plan each reference and check its clearance")
    b.add_argument("--workers", type=int, default=4, help="with --dry: how many scenes at once")
    b.add_argument("--force", action="store_true", help="play scenes again that -o already has")
    b.add_argument("--no-video", action="store_true", help="no cameras: only the score (faster)")
    b.add_argument("--timeout", type=float, default=1800.0,
                   help="seconds after which a scene's Isaac process is killed (default 1800; 0 for none)")
    b.add_argument("--summary", action="store_true", help="only write the summary of the reports -o holds")
    b.add_argument("-o", "--output", default=None, help="the reports, videos and summary (default runs/batch/<time>)")
    b.add_argument("--child", action="store_true", help=argparse.SUPPRESS)
    b.set_defaults(fn=cmd_batch)

    m = sub.add_parser("summary", help="aggregate a JSONL log")
    m.add_argument("log")
    m.set_defaults(fn=cmd_summary)
    from .core import plugins

    plugins.load()
    from .agent import bench

    bench.add_parsers(sub)
    n = sub.add_parser("snapshot", help="Isaac: every scene's camera pictures and arm state, which the methods read")
    n.add_argument("--scenes", default="paper", help="paper (the 480 varied scenes), default (the 40 default scenes), "
                                                      "all, or a comma list of ids")
    n.add_argument("--gpus", default=os.environ.get("MRBENCH_ISAAC_GPU", "0"), help="GPUs to spread the scenes over")
    n.add_argument("--force", action="store_true", help="take them again even if they are there")
    n.set_defaults(fn=cmd_snapshot)
    for build in COMMANDS:
        build(sub)
    for cmd, builds in ARGUMENTS.items():
        for build in builds:
            build(sub.choices[cmd])
    return p


#: what a plugin adds to the command line (``core/plugins.py``): subcommands (``add_command``: each a function given
#: the subparsers, which adds its parser and sets its ``fn``), arguments of an existing command (``add_arguments``),
#: handlers run before a command -- one that returns an exit code answers in its place -- or after it
#: (``add_handler``), checks of every command's arguments (``add_guard``: one raises ``MRBenchError`` to refuse them),
#: and the actions of ``assets`` (``ASSET_ACTIONS``, which its handler answers)
COMMANDS: list = []
ARGUMENTS: dict = {}
BEFORE: dict = {}
AFTER: dict = {}
GUARDS: list = []
ASSET_ACTIONS = ["status", "fetch", "convert", "looks"]


def add_command(build) -> None:
    COMMANDS.append(build)


def add_arguments(cmd: str, build) -> None:
    ARGUMENTS.setdefault(cmd, []).append(build)


def add_handler(cmd: str, handle, after: bool = False) -> None:
    (AFTER if after else BEFORE).setdefault(cmd, []).append(handle)


def add_guard(check) -> None:
    GUARDS.append(check)


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    try:
        for check in GUARDS:
            check(args)
        for handle in BEFORE.get(args.cmd, []):
            code = handle(args)
            if code is not None:
                return code
        code = args.fn(args)
        for handle in AFTER.get(args.cmd, []):
            handle(args)
        return code
    except MRBenchError as e:
        print(f"error: {e}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
