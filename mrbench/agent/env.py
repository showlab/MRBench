"""The environment every compared method talks to (docs/BASELINES.md, "What every row shares"): one case, its
senses, its arms' kinematics, the program API, the simulator, the budgets and the output -- and nothing else.

A session (one LLM conversation, or one rule's turn) gets an ``Env`` built from an ``EnvConfig``: which tools it
has, and, for the per-arm rows (RoCo- and HMAS-2-style), whose view -- the fixed cameras and only its own arm's wrist
camera, ``read_arm``, ``can_reach`` and ``heft``.  The budgets that belong to the case (simulator runs, probes) and
every record the episode is graded on live in the episode's ``Ledger``, shared by its sessions; a session's own
caps (sensing calls, program checks) live here.  ``mcp.py`` serves an ``Env`` over MCP; ``llm.FakeBackend`` calls it
in-process.

Tool groups:

* senses, shared by every row: ``briefing``, ``look``, ``read_arm``, ``locate``, ``can_reach`` (with margins),
  ``measure``, ``heft``, ``note``;
* control, shared: ``check_program``, ``try_in_sim``, ``submit_program``; ``probe`` (a small motion, read back);
* output: ``submit_team_plan``; ``respond`` (a stage's or an agent's own output, for methods with stages or
  several agents);
* reasoning: ``scene_graph``, ``check_hold``, ``who_can``, ``meet`` (every row's); ``solve_team`` (LLM + solver's
  own); a plugin adds tools of its own rows (``add_tool``).
"""

from __future__ import annotations

import base64
import contextlib
import fcntl
import json
import os
import subprocess
import sys
import tempfile
import time
import traceback
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Callable, Optional

from ..core.paths import REPO_ROOT
from . import failure, program
from .cases import case_from_id
from .ledger import Ledger
from .observe import APPROACHES, Observer

#: a text result longer than this is cut
MAX_TEXT = 6000
#: the interpreter that has Isaac Sim / Isaac Lab (docs/GETTING_STARTED.md)
ISAAC_PYTHON = os.environ.get("MRBENCH_ISAAC_PYTHON", sys.executable)
#: wall-clock seconds an Isaac run may take; on a loaded machine raise it with MRBENCH_SIM_TIMEOUT_S
SIM_TIMEOUT_S = int(os.environ.get("MRBENCH_SIM_TIMEOUT_S", 900))
#: wall-clock seconds the planner may search in a session's program check (``check_program``, a probe's check); past
#: it the check fails, saying so (a check that searches this long is failing on inverse kinematics).
#: MRBENCH_CHECK_TIMEOUT_S
CHECK_TIMEOUT_S = float(os.environ.get("MRBENCH_CHECK_TIMEOUT_S", 600))
#: a probe plays at most this long
PROBE_MAX_S = 8.0

SENSES = ("briefing", "look", "read_arm", "locate", "can_reach", "measure", "heft", "note")
CONTROL = ("check_program", "try_in_sim", "submit_program")
OWN = ("scene_graph", "check_hold", "solve_team", "who_can", "meet")
#: the reasoning tools every row is given: a row differs from another in how it is organised, not in what it may
#: measure.  ``solve_team`` (the LLM + solver row's solver) stays with the row that is made of it
SHARED = ("who_can", "meet", "scene_graph", "check_hold", "probe")
OUTPUT = ("submit_team_plan", "respond")
ALL_TOOLS = SENSES + CONTROL + ("probe",) + OWN + OUTPUT
#: tools that never count against a session's call budget
FREE = ("note", "submit_team_plan", "submit_program", "respond")


def _text(obj, indent=None) -> str:
    s = obj if isinstance(obj, str) else json.dumps(obj, indent=indent)
    return s if len(s) <= MAX_TEXT else s[:MAX_TEXT] + f"\n... ({len(s) - MAX_TEXT} more characters cut)"


def _t(text: str) -> dict:
    return {"type": "text", "text": text}


#: tools whose result an evidence citation names
CITABLE = ["check_hold", "can_reach", "who_can", "meet", "measure", "locate", "heft", "probe",
           "scene_graph", "read_arm", "solve_team", "try_in_sim"]
#: tools a plugin adds (``add_tool``): name -> (handler, schema)
EXTRA_TOOLS: dict[str, tuple[Callable, Callable]] = {}
#: the constraint monitor a plugin adds (``add_monitor``), for ``EnvConfig.monitor``
_MONITOR: list[Callable] = []


def add_tool(name: str, handler: Callable, schema: Callable, citable: bool = False) -> None:
    """A tool of a plugin's rows (``core/plugins.py``): ``handler(env, args)`` answers a call with content blocks,
    ``schema(env, parts)`` gives its (description, input schema) -- ``parts`` are ``schema_parts(env)``; ``citable``:
    an evidence citation may name its result."""
    if name in ALL_TOOLS:
        raise ValueError(f"a tool {name!r} exists already")
    EXTRA_TOOLS[name] = (handler, schema)
    if citable:
        CITABLE.insert(0, name)


def add_monitor(factory: Callable) -> None:
    """The constraint monitor a session with ``EnvConfig.monitor`` reviews team checks with (``core/plugins.py``)."""
    _MONITOR[:] = [factory]


def all_tools() -> tuple[str, ...]:
    from ..core import plugins

    plugins.load()
    return ALL_TOOLS + tuple(EXTRA_TOOLS)


def _cites(evidence, *reasons) -> bool:
    """Evidence given: its own field filled in, or a reason that names the tool result it rests on."""
    if (evidence or "").strip():
        return True
    return any(t in (r or "") for r in reasons for t in CITABLE)


def _parsed(v):
    """A list or object a model passed as JSON text, parsed; anything else as it is.  Closing brackets left over
    after the value are dropped: a model may write an object argument as text with one brace too many."""
    if isinstance(v, str) and v.strip()[:1] in ("[", "{"):
        text = v.strip()
        try:
            out, end = json.JSONDecoder().raw_decode(text)
        except json.JSONDecodeError:
            return v
        return out if not text[end:].strip(" \n]}") else v
    return v


@dataclass
class EnvConfig:
    case: str
    snapshot: str
    log: str
    ledger: str
    tools: list = field(default_factory=lambda: list(SENSES) + ["submit_team_plan"])
    #: an arm: this session sees as that arm (the fixed cameras and only its own wrist camera, read_arm, can_reach)
    view: Optional[str] = None
    #: the team plan a program must keep to (an idle arm must not act); None: the latest plan in the ledger
    team_plan: Optional[dict] = None
    #: tools after which the session is over (further calls are refused)
    final: list = field(default_factory=lambda: ["submit_team_plan", "submit_program", "respond"])
    calls: int = 60
    checks: int = 10
    #: a constraint monitor (``add_monitor``) on a plugin's team check
    monitor: bool = False
    #: the evidence rule: a feasible team plan names, for each arm it uses, the team without it that was tried and
    #: the tool result that shows it fails
    evidence: bool = False
    gpu: str = "0"
    #: what respond's ``data`` must hold: a JSON schema, and what to say about it
    respond_schema: Optional[dict] = None
    respond_desc: str = "Give your output for this stage; this ends your turn."
    #: a name for the records (the session's path in the episode)
    session: str = ""

    def to_json(self) -> str:
        return json.dumps(asdict(self))

    @classmethod
    def from_json(cls, s: str) -> "EnvConfig":
        return cls(**json.loads(s))


#: plays a program in the simulator: (case id, plan json, out dir, gpu, probe seconds or None) -> report dict
SimRunner = Callable[[str, Path, Path, str, Optional[float]], dict]


@contextlib.contextmanager
def _sim_slot(gpu: str):
    """With MRBENCH_SIM_SLOTS=N, at most N Isaac runs at a time on a GPU, across every process on this machine that
    sets it (a lock file a slot, in MRBENCH_SIM_SLOT_DIR or the temporary folder); the wait is not run time."""
    n = int(os.environ.get("MRBENCH_SIM_SLOTS", 0))
    if n <= 0:
        yield
        return
    d = Path(os.environ.get("MRBENCH_SIM_SLOT_DIR") or tempfile.gettempdir())
    d.mkdir(parents=True, exist_ok=True)
    while True:
        for i in range(n):
            f = (d / f"mrbench-sim-gpu{gpu}-{i}.lock").open("w")
            try:
                fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except OSError:
                f.close()
                continue
            try:
                yield
            finally:
                f.close()
            return
        time.sleep(5)


def isaac_runner(case_id: str, plan_json: Path, out: Path, gpu: str, probe_s: Optional[float]) -> dict:
    from ..core import plugins

    env = {**os.environ, "CUDA_VISIBLE_DEVICES": gpu, **plugins.child_env(), "OMNI_KIT_ACCEPT_EULA": "YES"}
    cmd = [ISAAC_PYTHON, "-m", "mrbench.agent.execute", case_id, str(plan_json), str(out)]
    if probe_s is not None:
        cmd += ["--probe", str(probe_s)]
    out.mkdir(parents=True, exist_ok=True)
    with _sim_slot(gpu), (out.parent / f"{out.name}.log").open("w") as f:
        try:
            subprocess.run(cmd, env=env, stdout=f, stderr=f, stdin=subprocess.DEVNULL, timeout=SIM_TIMEOUT_S,
                           cwd=REPO_ROOT)
        except subprocess.TimeoutExpired:
            pass
    rep_f = out / "report.json"
    return json.loads(rep_f.read_text()) if rep_f.exists() else {"played": False, "error": "did not start or timed out"}


class Env:
    def __init__(self, cfg: EnvConfig, sim: Optional[SimRunner] = None):
        unknown = set(cfg.tools) - set(all_tools())
        if unknown:
            raise ValueError(f"no tools {sorted(unknown)}")
        self.cfg = cfg
        self.case = case_from_id(cfg.case)
        self.obs = Observer(self.case, Path(cfg.snapshot))
        self.log = Path(cfg.log)
        self.log.mkdir(parents=True, exist_ok=True)
        self.ledger = Ledger(Path(cfg.ledger))
        self.sim = sim or isaac_runner
        self.arms = [a.slot for a in self.case.inst.scene.arms]
        if cfg.view is not None and cfg.view not in self.arms:
            raise ValueError(f"no arm {cfg.view!r} to see as")
        self.used = 0
        self.checks_left = cfg.checks
        self.k = 0
        self.notes: list[str] = []
        self.finished: list[str] = []
        self.passed: dict[str, int] = {}  # program text -> the check that passed it
        self.reach_log: list[dict] = []
        self.reach_cache: dict = {}
        self.monitor = None
        if cfg.monitor:
            if not _MONITOR:
                raise ValueError("a constraint monitor was asked for, and no plugin adds one (add_monitor)")
            self.monitor = _MONITOR[0]()
        self._things = None

    # ------------------------------------------------------------------ what this session may do
    @property
    def own_arms(self) -> list[str]:
        return [self.cfg.view] if self.cfg.view else self.arms

    @property
    def cameras(self) -> list[str]:
        views = self.obs.state["views"]
        if self.cfg.view is None:
            return list(views)
        return [v for v in views if not v.startswith("wrist_") or v == f"wrist_{self.cfg.view}"]

    @property
    def over(self) -> bool:
        return any(t in self.finished for t in self.cfg.final)

    def tools(self) -> list[dict]:
        specs = _schemas(self)
        return [specs[n] for n in self.cfg.tools]

    # ------------------------------------------------------------------ calls
    def call(self, name: str, args: dict) -> list[dict]:
        if name not in self.cfg.tools:
            raise KeyError(f"no tool {name!r} in this session")
        if self.over:
            return [_t("Your output is in; this session is over. Stop now and reply DONE.")]
        if name not in FREE:
            if self.used >= self.cfg.calls:
                return [_t("No calls left. Give your output now (" + ", ".join(
                    t for t in self.cfg.final if t in self.cfg.tools) + ").")]
            self.used += 1
        left = f"\n[{self.cfg.calls - self.used} calls left in this session]"
        out = getattr(self, f"_{name}")(args) if name not in EXTRA_TOOLS else EXTRA_TOOLS[name][0](self, args)
        if out and out[0]["type"] == "text" and name not in FREE:
            out[0] = _t(out[0]["text"] + left)
        return out

    # ------------------------------------------------------------------ senses
    def _arm(self, args) -> str:
        arm = args.get("arm")
        if arm not in self.own_arms:
            raise KeyError(f"{arm!r} is not an arm you can query; yours: {', '.join(self.own_arms)}")
        return arm

    def _briefing(self, args):
        b = self.obs.briefing()
        b["cameras"] = {k: v for k, v in b["cameras"].items() if k in self.cameras}
        if self.cfg.view:
            b["you"] = f"You are the {self.cfg.view} arm: you see the fixed cameras and your own hand's camera, and " \
                       f"you can query only your own joints and kinematics."
        return [_t(_text(b, indent=1))]

    def _look(self, args):
        cam = args["camera"]
        if cam not in self.cameras:
            raise KeyError(f"no camera {cam!r} for you; cameras: {', '.join(self.cameras)}")
        data = base64.b64encode(self.obs.image(cam).read_bytes()).decode()
        caption = self.obs.views().get(cam, cam)
        return [_t(f"camera {cam}: {caption}"), {"type": "image", "data": data, "mimeType": "image/jpeg"}]

    def _read_arm(self, args):
        return [_t(_text(self.obs.proprio(self._arm(args))))]

    def _locate(self, args):
        return [_t(_text(self.obs.locate(args["name"])))]

    def _measure(self, args):
        return [_t(_text(self.obs.measure(args["name"])))]

    def _can_reach(self, args):
        r = self.obs.can_reach(self._arm(args), (args["x"], args["y"], args["z"]), args.get("point", "any"),
                               float(args.get("holding") or 0.0))
        self.reach_log.append(r)
        return [_t(_text(r))]

    def _heft(self, args):
        from . import physics

        return [_t(_text(physics.heft(self.case, self.obs, self._arm(args), args["thing"])))]

    def _note(self, args):
        self.notes.append(args["fact"])
        (self.log / "notes.json").write_text(json.dumps(self.notes, indent=1))
        return [_t("Notebook:\n" + "\n".join(f"{i + 1}. {n}" for i, n in enumerate(self.notes)))]

    # ------------------------------------------------------------------ the reasoning tools
    def _scene_graph(self, args):
        from . import graph

        return [_t(_text(graph.scene_graph(self.case, self.obs, self.reach_log)))]

    def _check_hold(self, args):
        from . import physics

        return [_t(_text(physics.check_hold(self.case, self.obs, args["body"], args.get("load"),
                                            args.get("holds", []))))]

    def _who_can(self, args):
        """Every arm of this session asked at once whether it reaches a point (with margins): no arm left out."""
        p, point, hold = (args["x"], args["y"], args["z"]), args.get("point", "any"), float(args.get("holding") or 0.0)
        out = {}
        for arm in self.own_arms:
            r = self.obs.can_reach(arm, p, point, hold)
            self.reach_log.append(r)
            ok = [k for k, v in r["reachable"].items() if v.startswith("yes")]
            out[arm] = {"reaches": bool(ok), "pointing": ok, "margin": r["margin"]}
        return [_t(_text({"point": list(p), "arms": out}))]

    def _meet(self, args):
        """Places where two arms can pass a thing from hand to hand: each tool point from its own side, ``gap`` apart
        across the thing, clear of the fixed bodies -- searched over the table between their bases and up to
        0.55 m over it (over a partition too).  A few places, nearest the middle first."""
        from . import meet

        return [_t(_text(meet.meeting_places(self.obs, args["arm_a"], args["arm_b"],
                                             float(args.get("gap") or meet.GAP), int(args.get("max") or 3))))]

    def _solve_team(self, args):
        from . import graph

        return [_t(_text(graph.solve_team(self.case, self.obs, args.get("jobs", []), args.get("transfers", []),
                                          self.reach_cache)))]

    # ------------------------------------------------------------------ programs and the simulator
    def things_for_program(self) -> dict:
        if self._things is None:
            names = [t["name"] for t in self.obs.things()]
            names += [f"{t['name']}:{p}" for t in self.obs.things() for p in t.get("parts", [])]
            self._things = {n: self.obs.locate(n) for n in names}
        return self._things

    def team_plan(self) -> Optional[dict]:
        if self.cfg.team_plan is not None:
            return self.cfg.team_plan
        plans = self.ledger.records("plans")
        return plans[-1]["plan"] if plans else None

    def compile(self, code: str):
        arm_info = {a.slot: {"base": list(a.pos), "yaw": a.yaw_deg} for a in self.case.inst.scene.arms}
        plan_json, printed = program.run_sandbox(code, arm_info, self.things_for_program())
        if plan_json is None:
            return None, None, printed
        return plan_json, program.to_script(plan_json), printed

    def _check_program(self, args):
        code = args["code"]
        if self.checks_left <= 0:
            return [_t("No checks left in this session: submit your program with submit_program.")]
        self.checks_left -= 1
        self.k += 1
        k = self.k
        (self.log / f"program_{k}.py").write_text(code)
        plan_json, script, printed = self.compile(code)
        lines = [f"Check {k} ({self.checks_left} checks left)"]
        if plan_json is None:
            rec = {"ok": False, "stage": "sandbox", "problems": [printed]}
            lines += ["sandbox: the program failed", printed, "Result: FAIL"]
        else:
            (self.log / f"plan_{k}.json").write_text(json.dumps(plan_json))
            if printed.strip():
                lines.append("printed:\n" + printed.strip()[-1200:])
            movers = sorted(program.acting_arms(script))
            team = self.team_plan()
            idle = [d["arm"] for d in (team or {}).get("idle", [])]
            dur = sum(ph.seconds for ph in script)
            lines.append(f"phases: {len(script)}, {dur:.1f} s; arms that act: {', '.join(movers) or 'none'}")
            problems = [f"team plan: {a} is idle in the team plan but acts here" for a in movers if a in idle]
            r = program.check(self.case.inst, script, timeout_s=CHECK_TIMEOUT_S)
            problems += r["problems"]
            if r.get("warnings"):
                lines.append("warnings: " + "; ".join(r["warnings"][:4]))
            lines += ["problems:\n- " + "\n- ".join(problems)] if problems else ["inverse kinematics and clearance: ok"]
            ok = not problems
            if ok:
                self.passed[code] = k
            lines.append("Result: PASS -- ready for try_in_sim or submit_program" if ok else "Result: FAIL")
            rec = {"ok": ok, "problems": problems, "movers": movers, "seconds": round(dur, 1)}
        (self.log / f"check_{k}.json").write_text(json.dumps(rec, indent=1))
        self.ledger.add("programs", {"session": self.cfg.session, "k": k, "ok": rec["ok"],
                                     "plan": str(self.log / f"plan_{k}.json"), "movers": rec.get("movers", []),
                                     "problems": len(rec["problems"])})
        return [_t(_text("\n".join(lines)))]

    def play(self, code_or_k, probe_s: Optional[float] = None) -> tuple[Optional[dict], Path]:
        """Play a checked program (its text, or its check number): the run's report and its folder."""
        k = code_or_k if isinstance(code_or_k, int) else self.passed[code_or_k]
        out = self.log / (f"probe_{k}" if probe_s is not None else f"sim_{k}")
        rep = self.sim(self.case.id, self.log / f"plan_{k}.json", out, self.cfg.gpu, probe_s)
        return rep, out

    def _keyframes(self, rep: dict, out: Path) -> list[dict]:
        content = []
        for key in rep.get("keyframes", []):
            f = out / key["file"]
            if f.exists():
                content.append(_t(f"t = {key['t']} s"))
                content.append({"type": "image", "data": base64.b64encode(f.read_bytes()).decode(),
                                "mimeType": "image/jpeg"})
        return content

    def _try_in_sim(self, args):
        code = args["code"]
        if code not in self.passed:
            return [_t("Run check_program on this exact program first; only a program that passes it is played.")]
        left = self.ledger.take("runs")
        if left is None:
            return [_t("No simulator runs left for this case: submit what you have.")]
        k = self.passed[code]
        rep, out = self.play(code)
        check = json.loads((self.log / f"check_{k}.json").read_text())
        fr = failure.report(self.case, rep, check)
        self.ledger.add("runs", {"session": self.cfg.session, "k": k, "success": bool(rep.get("success")),
                                 "played": bool(rep.get("played")), "report": str(out / "report.json"),
                                 "plan": str(self.log / f"plan_{k}.json"), "movers": check.get("movers", []),
                                 "milestones_hit": rep.get("milestones_hit")})
        if not rep.get("played"):
            return [_t(_text({"played": False, "error": rep.get("error", "")[:1500], "runs_left": left}))]
        summary = {"success": rep["success"], "runs_left": int(left), **({} if rep["success"] else
                                                                       {"failure_report": fr})}
        return [_t(_text(summary))] + self._keyframes(rep, out)

    def _probe(self, args):
        code, secs = args["code"], float(args.get("seconds") or PROBE_MAX_S)
        if secs > PROBE_MAX_S:
            return [_t(f"A probe plays at most {PROBE_MAX_S:.0f} s.")]
        plan_json, script, printed = self.compile(code)
        if plan_json is None:
            return [_t("sandbox: the program failed\n" + printed)]
        dur = sum(ph.seconds for ph in script)
        if dur > PROBE_MAX_S:
            return [_t(f"This program plays {dur:.1f} s; a probe plays at most {PROBE_MAX_S:.0f} s.")]
        r = program.check(self.case.inst, script, timeout_s=CHECK_TIMEOUT_S)
        if not r["ok"]:
            return [_t("The probe did not pass the check (nothing was spent):\n- " + "\n- ".join(r["problems"]))]
        left = self.ledger.take("probes")
        if left is None:
            return [_t("No probes left for this case.")]
        self.k += 1
        (self.log / f"plan_{self.k}.json").write_text(json.dumps(plan_json))
        rep, out = self.play(self.k, probe_s=dur)
        self.ledger.add("probes", {"session": self.cfg.session, "k": self.k, "report": str(out / "report.json")})
        if not rep.get("played"):
            return [_t(_text({"played": False, "error": rep.get("error", "")[:1500], "probes_left": left}))]
        seen = {"played_s": round(dur, 1), "how_far_each_thing_moved_and_turned": rep.get("worst"),
                "where_things_ended_m_from_start": rep.get("moved_m"),
                "grips_that_closed_on_nothing": rep.get("missed_grips") or [], "probes_left": int(left),
                "note": "a probe reads the world back; it does not say whether the task would succeed"}
        return [_t(_text(seen))] + self._keyframes(rep, out)

    def _submit_program(self, args):
        code = args["code"]
        rec = {"code": code, "notes": args.get("notes", ""), "passed_check": code in self.passed,
               "check": self.passed.get(code), "checks_used": self.k}
        (self.log / "program_final.py").write_text(code)
        (self.log / "program_final.json").write_text(json.dumps(rec, indent=1))
        self._finish("submit_program", rec)
        return [_t("Program recorded. Stop now and reply DONE." if self.over else "Program recorded.")]

    # ------------------------------------------------------------------ output
    def _submit_team_plan(self, plan: dict):
        plan = normalise_plan(plan)
        problems = validate_plan(plan, self.arms, {t["name"] for t in self.obs.things()})
        if self.cfg.evidence and plan.get("feasible"):
            cited = {d.get("arm") for d in plan.get("without_each_arm") or []
                     if _cites(d.get("evidence"), d.get("failed_because"), d.get("tried"))}
            problems += [f"evidence: {a} is used, but without_each_arm has no entry for it that names the tool result "
                         f"showing the team without it fails (put it in evidence, e.g. 'check_hold: ...')"
                         for a in plan.get("arms_used", []) if a not in cited]
            problems += [f"evidence: {t.get('thing')} is rejected without the tool result that shows it would not do "
                         f"(put it in evidence, e.g. 'check_hold: ...')"
                         for t in plan.get("things_rejected") or [] if not _cites(t.get("evidence"), t.get("why"))]
        if problems:
            return [_t("Not accepted, fix and submit again:\n- " + "\n- ".join(problems))]
        rec = {**plan, "calls_used": self.used, "notes": self.notes}
        (self.log / "team_plan.json").write_text(json.dumps(rec, indent=1))
        self.ledger.add("plans", {"session": self.cfg.session, "plan": rec})
        self._finish("submit_team_plan", rec)
        return [_t("Team plan recorded. Stop now and reply DONE." if self.over else "Team plan recorded.")]

    def _respond(self, args):
        data = args.get("data")
        if data is None:  # some models put the fields beside the message instead of in data
            data = {k: v for k, v in args.items() if k not in ("message", "data")} or None
        data = _parsed(data)
        if isinstance(data, dict):
            data = {k: _parsed(v) for k, v in data.items()}
        wanted = (self.cfg.respond_schema or {}).get("required", [])
        lacking = [k for k in wanted if not isinstance(data, dict) or data.get(k) is None]
        if self.cfg.respond_schema and (not isinstance(data, dict) or lacking):
            return [_t("Not recorded: give your output as the `data` argument, an object with "
                       + ", ".join(f"`{k}`" for k in (lacking or wanted or ["its fields"]))
                       + " (lists and objects, not text). Call respond again.")]
        self._finish("respond", {"message": args.get("message", ""), "data": data})
        return [_t("Recorded. Stop now and reply DONE.")]

    def _finish(self, tool: str, data) -> None:
        self.finished.append(tool)
        outs = json.loads((self.log / "output.json").read_text()) if (self.log / "output.json").exists() else {}
        outs[tool] = data
        (self.log / "output.json").write_text(json.dumps(outs, indent=1))

    # ------------------------------------------------------------------ logging
    def logged_call(self, name: str, args: dict) -> tuple[list[dict], bool]:
        t0 = time.time()
        try:
            content, err = self.call(name, args or {}), False
        except Exception as e:  # noqa: BLE001 -- a bad call is the caller's to fix
            content, err = [_t(f"Error: {type(e).__name__}: {e}")], True
            with (self.log / "server_errors.log").open("a") as f:
                f.write(traceback.format_exc() + "\n")
        with (self.log / "calls.jsonl").open("a") as f:
            f.write(json.dumps({"t": round(t0, 2), "s": round(time.time() - t0, 2), "tool": name, "args": args,
                                "error": err, "reply": [c["text"] if c["type"] == "text" else "<image>"
                                                        for c in content]}) + "\n")
        return content, err


# ---------------------------------------------------------------------- the team plan


def normalise_plan(plan: dict) -> dict:
    """A team plan with its fields filled in: ``objects_used`` with ``object`` / ``use`` is read as ``things_used``
    with ``thing`` / ``job``, and a field left out gets its empty value."""
    p = dict(plan)
    if "objects_used" in p and "things_used" not in p:
        p["things_used"] = [{"thing": o.get("object", ""), "job": o.get("use", ""),
                             **({"instead_of": o["instead_of"]} if o.get("instead_of") else {})}
                            for o in p.pop("objects_used") or []]
    for k, v in (("things_used", []), ("things_rejected", []), ("idle", []), ("arms_used", []), ("missing", ""),
                 ("steps", []), ("key_constraints", [])):
        p.setdefault(k, v)
    return p


def validate_plan(plan: dict, arms: list[str], things: set[str]) -> list[str]:
    used = list(plan.get("arms_used", []))
    idle = [d.get("arm") for d in plan.get("idle", [])]
    problems = []
    for a in arms:
        n = used.count(a) + idle.count(a)
        if n != 1:
            problems.append(f"arm {a} appears {n} times in arms_used and idle (must be exactly once)")
    problems += [f"no arm {a!r} on the table" for a in used + idle if a not in arms]
    named = [t.get("thing", "") for t in plan.get("things_used", [])] + \
            [t.get("thing", "") for t in plan.get("things_rejected", [])]
    problems += [f"no thing {t!r} on the table (things: {', '.join(sorted(things))})" for t in named
                 if t.partition(":")[0] not in things]
    if plan.get("feasible") and not used:
        problems.append("feasible but no arm used")
    if not plan.get("feasible") and not plan.get("missing"):
        problems.append("infeasible: say what is missing (which job has no doer, and what would do it)")
    if not plan.get("feasible") and used:
        problems.append("infeasible: arms_used must be empty (put every arm in idle)")
    return problems


# ---------------------------------------------------------------------- tool descriptions
def schema_parts(env: Env) -> dict:
    """The pieces the tools' input schemas are made of (an arm of this session, any arm, a point, code, an object)."""
    return {"arm": {"type": "string", "enum": env.own_arms},
            "point": {"type": "array", "items": {"type": "number"}, "description": "[x, y, z] in the world frame (m)"},
            "code": {"type": "string", "description": "Python that sets PLAN (see the arm API in your instructions)"},
            "obj": lambda props, req=(): {"type": "object", "properties": props, "required": list(req)},
            "all_arms": {"type": "string", "enum": env.arms}}


def _schemas(env: Env) -> dict[str, dict]:
    parts = schema_parts(env)
    arm, point, code, obj, all_arms = (parts[k] for k in ("arm", "point", "code", "obj", "all_arms"))
    s = {
        "briefing": ("The goal, the coordinate frame, the arms (robot, where each is mounted, which way it faces), the "
                     "things on the table (what each is, a rough mass, parts the cameras can pick out), facts that do "
                     "not show, and the cameras.", obj({})),
        "look": ("A picture from one camera (fixed cameras round the table, and the arms' hand cameras). The room "
                 "round the table -- its walls, cabinets, shelves and what stands on them -- is out of every arm's "
                 "reach and not part of the task.",
                 obj({"camera": {"type": "string", "enum": env.cameras}}, ["camera"])),
        "read_arm": ("One arm's proprioception now: joint angles and limits, where its tool point (between the "
                     "fingertips) is and which way the hand points, how far apart its fingers are, its joint torques.",
                     obj({"arm": arm}, ["arm"])),
        "locate": ("A thing's bounding box from the cameras (centre, min and max corners in the world frame, top "
                   "height; for a single box also its size and turn). Name a thing from the briefing, or one of its "
                   "parts as 'thing:part'. Accurate to a few millimetres.", obj({"name": {"type": "string"}}, ["name"])),
        "measure": ("A thing's (or part's) sizes: each piece's size along its own axes and its centre, a few mm off -- "
                    "lengths, heights, openings.", obj({"name": {"type": "string"}}, ["name"])),
        "can_reach": ("Ask an arm's own kinematics whether it can put its tool point (between the fingertips) at a "
                      "point (x, y, z in the world frame, metres), the hand pointing 'down', 'steep' (60 deg below "
                      "level), 'tilted' (40 deg) or 'level' -- the last three away from the arm's base, toward the "
                      "point -- or 'any'. With 'holding' (m): the tip of a straight tool that long, held along the "
                      "way the hand points. Answers with a margin: how much farther out it would still reach, or how "
                      "far short it falls. The arm keeps 1 cm clear of the table and out of fixed things; the path "
                      "there, other arms and payload are not checked.",
                      obj({"arm": arm, "x": {"type": "number"}, "y": {"type": "number"}, "z": {"type": "number"},
                           "point": {"type": "string", "enum": ["any", *APPROACHES]},
                           "holding": {"type": "number", "description": "a held tool's length (m), if any"}},
                          ["arm", "x", "y", "z"])),
        "heft": ("An arm that reaches a movable thing lifts it 2 cm and puts it back, reading its joint torques: its "
                 "mass to about 3 %, or 'too heavy'. Computed, not played: it spends no probe.",
                 obj({"arm": arm, "thing": {"type": "string"}}, ["arm", "thing"])),
        "note": ("Write a fact you have established (with its evidence) into your notebook; the notebook is returned "
                 "in full.", obj({"fact": {"type": "string"}}, ["fact"])),
        "scene_graph": ("The physical scene graph: each arm's payload and how hard it presses and grips; each thing's "
                        "rough mass, whether it is fixed and what it rests on; joints (hinges: their axis, "
                        "springs and what turns them shut; tight fits with the force that pulls them out); a friction estimate; and the reach edges "
                        "you have established so far.", obj({})),
        "check_hold": ("Physics: does a body stay put under a load, held the ways you list (press, grip, hold a "
                       "handle, a weight set on it, a stop or prop set in a hinged body's way)? With the rough masses "
                       "and with a margin, naming the weakest edge. A hinged body (a door, a lid, a leaf): each hold on "
                       "its own, against its spring and its own weight; `at` is the point of the body held, where it "
                       "is when held -- or give `open_deg`, how far open from its closed stop it is held.",
                       obj({"body": {"type": "string"},
                            "load": obj({"at": point, "force": point}),
                            "holds": {"type": "array", "items": obj({
                                "kind": {"type": "string", "enum": ["press", "grip", "hold", "weight", "stop"]},
                                "by": {"type": "string", "description": "an arm, or a movable thing"}, "at": point,
                                "open_deg": {"type": "number", "description": "a hinged body: degrees open"}},
                                ["kind", "by"])}}, ["body", "holds"])),
        "who_can": ("Every arm asked at once whether its tool point (or a held tool's tip) reaches a point, pointing a "
                    "given way, each with its margin -- so that no arm is left out of a job it could do.",
                    obj({"x": {"type": "number"}, "y": {"type": "number"}, "z": {"type": "number"},
                         "point": {"type": "string", "enum": ["any", *APPROACHES]},
                         "holding": {"type": "number", "description": "a held tool's length (m), if any"}},
                        ["x", "y", "z"])),
        "meet": ("Where two arms can hand a thing over directly: places each reaches from its own side, their tool "
                 "points 'gap' apart across the thing (default 6 cm), clear of fixed bodies -- searched over the table "
                 "between their bases and up to 0.55 m above it, over a partition too. Lists a few, with how each arm "
                 "points; or says it found none.",
                 obj({"arm_a": all_arms, "arm_b": all_arms, "gap": {"type": "number"}, "max": {"type": "integer"}},
                     ["arm_a", "arm_b"])),
        "solve_team": ("Give the job graph; get the fewest arms. Jobs: what must be done, where ('at'), by which "
                       "doers (arms and/or things you have shown can do it, or ['auto']: every arm whose kinematics "
                       "reach 'at'), and 'with': the jobs it must happen at the same time as. Transfers: a thing "
                       "carried from one spot to another through hand-off spots ('via'). Returns every smallest team "
                       "with who does what, or what blocks it.",
                       obj({"jobs": {"type": "array", "items": obj({
                           "id": {"type": "string"}, "what": {"type": "string"}, "at": point,
                           "doers": {"type": "array", "items": {"type": "string"}},
                           "with": {"type": "array", "items": {"type": "string"}}}, ["id", "what", "doers"])},
                            "transfers": {"type": "array", "items": obj({
                                "id": {"type": "string"}, "thing": {"type": "string"}, "from": point, "to": point,
                                "via": {"type": "array", "items": point}}, ["id", "from", "to"])}}, ["jobs"])),
        "check_program": ("Run your program in the sandbox and check it before anything moves: inverse kinematics "
                          "along every path, clearance between arms and with the scene, and the team plan (an idle "
                          "arm must not move). Physics is not checked.", obj({"code": code}, ["code"])),
        "try_in_sim": ("Play a program that has passed check_program in the simulator, with physics: success or, if "
                       "not, the failure report (milestones reached and missed, the jobs' failure signals, grips that "
                       "closed on nothing, arms behind their plans, things that moved), and keyframe pictures. Takes "
                       "minutes; the runs are counted for the whole case.", obj({"code": code}, ["code"])),
        "probe": (f"Play a small motion (a program of at most {PROBE_MAX_S:.0f} s, checked first) in the simulator "
                  "and read back how each thing moved and turned, with keyframes -- to settle one doubt (does the "
                  "door swing back? does the turntable turn?). It does not say whether the task would succeed. "
                  "Counted for the whole case.", obj({"code": code, "seconds": {"type": "number"}}, ["code"])),
        "submit_program": ("Submit your final program.", obj({"code": code, "notes": {"type": "string"}}, ["code"])),
        "submit_team_plan": (
            "Submit the team plan. Every arm must appear exactly once, in arms_used or in idle. If the task cannot "
            "be done with what is on the table, feasible is false, arms_used empty, and missing says which job has "
            "no doer and what would do it.",
            obj({"feasible": {"type": "boolean", "description": "whether these arms, with what is on the table, can "
                                                                "do the whole task"},
                 "arms_used": {"type": "array", "items": all_arms, "description": "the fewest arms that do it"},
                 "idle": {"type": "array", "items": obj({"arm": all_arms, "reason": {"type": "string"}},
                                                        ["arm", "reason"])},
                 "things_used": {"type": "array", "items": obj({
                     "thing": {"type": "string"}, "job": {"type": "string"},
                     "instead_of": {"type": "string", "description": "the arm's job it takes over, if any"}},
                     ["thing", "job"])},
                 "things_rejected": {"type": "array", "items": obj({
                     "thing": {"type": "string"}, "why": {"type": "string"},
                     "evidence": {"type": "string", "description": "the tool result it rests on"}}, ["thing", "why"])},
                 "missing": {"type": "string", "description": "if infeasible: the job without a doer, and what kind "
                                                              "of arm or thing would do it; else empty"},
                 "steps": {"type": "array", "items": obj({
                     "step": {"type": "integer"}, "doer": {"type": "string", "description": "an arm or a thing"},
                     "action": {"type": "string"}, "at_same_time_as": {"type": "array", "items": {"type": "integer"}}},
                     ["step", "doer", "action"])},
                 "key_constraints": {"type": "array", "items": {"type": "string"},
                                     "description": "what forces the number of arms, with numbers"},
                 "without_each_arm": {"type": "array", "items": obj({
                     "arm": all_arms, "tried": {"type": "string"}, "failed_because": {"type": "string"},
                     "evidence": {"type": "string"}}, ["arm", "tried", "failed_because"])},
                 "confidence": {"type": "number"}},
                ["feasible", "arms_used", "idle", "missing"])),
        "respond": (env.cfg.respond_desc,
                    obj({"message": {"type": "string", "description": "what you say (to the team, or for the record)"},
                         "data": env.cfg.respond_schema or {"type": "object"}}, ["message"])),
    }
    s.update({n: schema(env, parts) for n, (_, schema) in EXTRA_TOOLS.items()})
    return {n: {"name": n, "description": d, "inputSchema": schema} for n, (d, schema) in s.items()}
