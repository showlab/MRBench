"""Runs one session -- one LLM conversation with one ``Env`` -- on a backend, the same way for every method:

* ``ClaudeCLI``: a headless Claude Code session (``claude -p``) with every built-in tool removed, no settings
  sources, our system prompt, and only the environment server as an MCP server, in a sandbox directory outside the
  repository;
* ``ClaudeHarness``: Claude Code keeping its own tools (Bash, files) but run inside a
  bubblewrap sandbox where the repository does not exist; the environment is served from outside on a Unix socket
  (``mcp.serve_socket``) that a stdlib bridge inside joins to Claude Code's stdio;
* ``ClaudeAPI``: the Messages API with our own loop -- no harness, so the model sees our system prompt, the first
  message and the environment's tools and replies, nothing else (``claude -p`` adds an Agent SDK line to the system
  prompt, the account's e-mail and an environment block with the working folder and the date);
* ``OpenAICompat``: any model behind an OpenAI-compatible chat API (a local vLLM), the context kept by ``loop.py``;
* ``FakeBackend``: no model -- a Python function plays the session against an in-process ``Env`` (tests, and the
  rule-based rows).

A session without an environment (``spec.env`` None) is a plain completion: no tools, the final text is the output.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Optional

from ..core.paths import REPO_ROOT
from .env import Env, EnvConfig

#: where sessions run: outside the repository, so they see no CLAUDE.md, no cards, no references
SANDBOX = Path(os.environ.get("MRBENCH_AGENT_SANDBOX", Path.home() / ".cache" / "mrbench" / "agent_sandbox"))
TIMEOUT_S = 45 * 60
#: the environment server's interpreter: the repository's own virtualenv
PYTHON = str(REPO_ROOT / ".venv" / "bin" / "python") if (REPO_ROOT / ".venv").exists() else sys.executable
#: the environment server's variables: one BLAS thread -- by default OpenBLAS starts 64 per server, for no speed-up
#: on the reach checks (six times the CPU), and sessions run dozens of servers side by side
def server_env() -> dict[str, str]:
    """The environment of a session's tool server: this process's plugins from the same places, one BLAS thread."""
    from ..core import plugins

    return {**plugins.child_env(), "OPENBLAS_NUM_THREADS": "1", "OMP_NUM_THREADS": "1"}


@dataclass
class SessionSpec:
    #: the session's path in the episode ("round1/stage1", "round2/roco/turn3-west")
    name: str
    system: str
    first: str
    env: Optional[EnvConfig] = None
    timeout_s: float = TIMEOUT_S


@dataclass
class SessionResult:
    name: str
    log: Path
    #: what the session's final tools recorded (``output.json``): {tool: data}
    output: dict = field(default_factory=dict)
    #: the model's last text
    text: str = ""
    turns: Optional[int] = None
    cost_usd: float = 0.0
    seconds: float = 0.0
    exit: Any = 0
    tool_calls: int = 0


def _read_output(log: Path) -> dict:
    f = log / "output.json"
    return json.loads(f.read_text()) if f.exists() else {}


def _read_result(transcript: Path, r: "SessionResult") -> "SessionResult":
    """The final text, turns and cost of a ``claude -p`` session, from its stream-json transcript's result line."""
    for line in transcript.read_text().splitlines()[::-1]:
        try:
            d = json.loads(line)
        except json.JSONDecodeError:
            continue
        if d.get("type") == "result":
            r.text, r.turns, r.cost_usd = d.get("result") or "", d.get("num_turns"), d.get("total_cost_usd") or 0.0
            break
    return r


def _count_calls(log: Path) -> int:
    f = log / "calls.jsonl"
    return sum(1 for _ in f.open()) if f.exists() else 0


class Backend:
    label = "backend"

    def run(self, spec: SessionSpec, log: Path, box: Path) -> SessionResult:
        raise NotImplementedError


class ClaudeCLI(Backend):
    def __init__(self, model: str = "opus", effort: str = "high"):
        self.model, self.effort = model, effort
        self.label = f"claude:{model}:{effort}"

    def command(self, spec: SessionSpec, box: Path) -> list[str]:
        cmd = ["claude", "-p", spec.first, "--model", self.model, "--effort", self.effort, "--tools", "",
               "--strict-mcp-config", "--setting-sources", "", "--system-prompt", spec.system,
               "--output-format", "stream-json", "--verbose", "--no-session-persistence"]
        if spec.env is not None:
            (box / "session.json").write_text(spec.env.to_json())
            mcp = {"mcpServers": {"env": {"command": PYTHON, "args": ["-m", "mrbench.agent.mcp", "--config",
                                                                      str(box / "session.json")],
                                          "env": server_env()}}}
            (box / "mcp.json").write_text(json.dumps(mcp, indent=1))
            cmd += ["--mcp-config", str(box / "mcp.json"), "--allowedTools", "mcp__env"]
        return cmd

    def run(self, spec: SessionSpec, log: Path, box: Path) -> SessionResult:
        log.mkdir(parents=True, exist_ok=True)
        box.mkdir(parents=True, exist_ok=True)
        cmd = self.command(spec, box)
        t0 = time.time()
        with (log / "transcript.jsonl").open("w") as out, (log / "claude_stderr.log").open("w") as err:
            try:
                code = subprocess.run(cmd, cwd=box, stdin=subprocess.DEVNULL, stdout=out, stderr=err,
                                      timeout=spec.timeout_s, env=self.environ()).returncode
            except subprocess.TimeoutExpired:
                code = "timeout"
        r = SessionResult(spec.name, log, _read_output(log), exit=code, seconds=round(time.time() - t0, 1),
                          tool_calls=_count_calls(log))
        return _read_result(log / "transcript.jsonl", r)

    def environ(self) -> dict:
        # a child of another Claude Code session must not join it
        return {k: v for k, v in os.environ.items() if not k.startswith(("CLAUDECODE", "CLAUDE_CODE_"))}


#: Claude Code's own tools that the harness keeps (computing only: inside the sandbox there is nothing else)
HARNESS_TOOLS = "Bash,Read,Write,Edit,Glob,Grep"


class ClaudeHarness(ClaudeCLI):
    """Claude Code with its own tools, in a bubblewrap sandbox: the sandbox folder (as /work), the system's /usr and
    /etc, Claude Code's own install, its login, and a socket to the environment -- not the repository, not $HOME."""

    def __init__(self, model: str = "opus", effort: str = "high"):
        super().__init__(model, effort)
        self.label = f"harness:{model}:{effort}"

    def run(self, spec: SessionSpec, log: Path, box: Path) -> SessionResult:
        if spec.env is None:
            raise ValueError("the harness needs an environment")
        log.mkdir(parents=True, exist_ok=True)
        box.mkdir(parents=True, exist_ok=True)
        (box / "session.json").write_text(spec.env.to_json())
        (box / "bridge.py").write_text(__import__("mrbench.agent.mcp", fromlist=["BRIDGE"]).BRIDGE)
        sock = box / "env.sock"
        # bound by a relative path from inside the folder: a Unix socket's path may not be longer than 108 bytes
        server = subprocess.Popen([PYTHON, "-m", "mrbench.agent.mcp", "--config", str(box / "session.json"),
                                   "--socket", "env.sock"], env={**os.environ, **server_env()}, cwd=box,
                                  stdout=subprocess.DEVNULL, stderr=(log / "env_server.log").open("w"))
        for _ in range(100):
            if sock.exists():
                break
            time.sleep(0.1)
        (box / "mcp.json").write_text(json.dumps({"mcpServers": {"env": {
            "command": "/usr/bin/python3", "args": ["/work/bridge.py", "/work/env.sock"]}}}, indent=1))
        home = box / "home"
        (home / ".claude").mkdir(parents=True, exist_ok=True)
        (home / ".claude.json").write_text(json.dumps({"hasCompletedOnboarding": True}))
        real = Path.home()
        claude = Path(shutil.which("claude") or real / ".local/bin/claude").resolve()
        cmd = ["bwrap", "--die-with-parent", "--unshare-all", "--share-net", "--new-session",
               "--ro-bind", "/usr", "/usr", "--symlink", "usr/bin", "/bin", "--symlink", "usr/lib", "/lib",
               "--symlink", "usr/lib64", "/lib64", "--symlink", "usr/sbin", "/sbin", "--ro-bind", "/etc", "/etc",
               *(["--ro-bind", "/run/systemd/resolve", "/run/systemd/resolve"]  # /etc/resolv.conf points there
                 if Path("/run/systemd/resolve").exists() else []),
               "--proc", "/proc", "--dev", "/dev", "--tmpfs", "/tmp",
               "--bind", str(box), "/work", "--bind", str(home), "/home/agent",
               "--bind", str(real / ".claude" / ".credentials.json"), "/home/agent/.claude/.credentials.json",
               "--ro-bind", str(claude), "/opt/claude/claude",
               "--chdir", "/work", "--clearenv", "--setenv", "HOME", "/home/agent",
               "--setenv", "PATH", "/usr/local/bin:/usr/bin:/bin", "--setenv", "LANG", "C.UTF-8",
               "/opt/claude/claude", "-p", spec.first, "--model", self.model, "--effort", self.effort,
               "--tools", HARNESS_TOOLS, "--allowedTools", HARNESS_TOOLS + ",mcp__env",
               "--strict-mcp-config", "--mcp-config", "/work/mcp.json", "--setting-sources", "",
               "--system-prompt", spec.system, "--output-format", "stream-json", "--verbose",
               "--no-session-persistence"]
        t0 = time.time()
        try:
            with (log / "transcript.jsonl").open("w") as out, (log / "claude_stderr.log").open("w") as err:
                try:
                    code = subprocess.run(cmd, stdin=subprocess.DEVNULL, stdout=out, stderr=err,
                                          timeout=spec.timeout_s).returncode
                except subprocess.TimeoutExpired:
                    code = "timeout"
        finally:
            server.terminate()
            try:
                server.wait(timeout=10)
            except subprocess.TimeoutExpired:
                server.kill()
        # the environment logged into the session's log folder (spec.env.log); the transcript is read as for ClaudeCLI
        r = SessionResult(spec.name, log, _read_output(Path(spec.env.log)), exit=code,
                          seconds=round(time.time() - t0, 1), tool_calls=_count_calls(Path(spec.env.log)))
        return _read_result(log / "transcript.jsonl", r)


class OpenAICompat(Backend):
    def __init__(self, model: str, base_url: str = "http://127.0.0.1:8000/v1", think: bool = True):
        self.model, self.base_url, self.think = model, base_url, think
        self.label = f"openai:{model}:{'think' if think else 'nothink'}"

    def run(self, spec: SessionSpec, log: Path, box: Path) -> SessionResult:
        from . import loop

        log.mkdir(parents=True, exist_ok=True)
        box.mkdir(parents=True, exist_ok=True)
        t0 = time.time()
        if spec.env is None:
            body = {"model": self.model, "messages": [{"role": "system", "content": spec.system},
                                                      {"role": "user", "content": spec.first}],
                    "chat_template_kwargs": {"enable_thinking": self.think}, **loop.SAMPLING[self.think]}
            resp = loop._chat(self.base_url, body)
            text = resp["choices"][0]["message"].get("content") or ""
            (log / "transcript.jsonl").write_text(json.dumps({"response": text}) + "\n")
            return SessionResult(spec.name, log, {}, text, 1, 0.0, round(time.time() - t0, 1))
        (box / "session.json").write_text(spec.env.to_json())
        cmd = [PYTHON, "-m", "mrbench.agent.mcp", "--config", str(box / "session.json")]
        stats = loop.run(self.base_url, self.model, spec.system, spec.first, cmd,
                         {**os.environ, **server_env()}, str(REPO_ROOT), log, self.think,
                         ", ".join(spec.env.final), log / "output.json")
        return SessionResult(spec.name, log, _read_output(log), "", stats.get("turns"), 0.0,
                             round(time.time() - t0, 1), tool_calls=stats.get("tool_calls", 0))


#: the API's ids for the model names the runs use, and list prices in $ per million tokens (input, output, cache
#: read; a five-minute cache write is 1.25 x input)
MODELS = {"opus": "claude-opus-5-5", "sonnet": "claude-sonnet-5-5", "haiku": "claude-haiku-5-5"}
PRICES = {"claude-opus-5-5": (4.00, 20.00, 0.20), "claude-sonnet-5-5": (2.00, 10.00, 0.20),
          "claude-haiku-5-5": (0.10, 0.50, 0.01)}
MAX_TOKENS = 64000


def _api_block(part: dict) -> dict:
    """An environment reply part (MCP) as a Messages API content block."""
    if part["type"] == "image":
        return {"type": "image", "source": {"type": "base64", "media_type": part["mimeType"], "data": part["data"]}}
    return {"type": "text", "text": part["text"]}


class ClaudeAPI(Backend):
    """The Messages API, with the environment server (stdio MCP, as for ``claude -p``) and a loop of our own: adaptive
    thinking at ``effort``, the whole history sent back each turn (thinking blocks as returned), cached; the session
    ends when the model stops calling tools or a final tool is in.  Credentials: ``ANTHROPIC_API_KEY``, and
    ``ANTHROPIC_WORKSPACE_ID`` for a key that is not scoped to one workspace."""

    def __init__(self, model: str = "opus", effort: str = "high"):
        self.model, self.effort = model, effort
        self.label = f"api:{model}:{effort}"

    def _client(self):
        import anthropic

        ws = os.environ.get("ANTHROPIC_WORKSPACE_ID")
        return anthropic.Anthropic(max_retries=8, default_headers={"anthropic-workspace-id": ws} if ws else None)

    def _cost(self, u) -> float:
        i, o, read = PRICES.get(MODELS.get(self.model, self.model), (0.0, 0.0, 0.0))
        return (u.input_tokens * i + (u.cache_creation_input_tokens or 0) * i * 1.25
                + (u.cache_read_input_tokens or 0) * read + u.output_tokens * o) / 1e6

    def run(self, spec: SessionSpec, log: Path, box: Path) -> SessionResult:
        from .loop import MCPClient

        log.mkdir(parents=True, exist_ok=True)
        box.mkdir(parents=True, exist_ok=True)
        client, t0 = self._client(), time.time()
        mcp, tools, done = None, [], None
        if spec.env is not None:
            (box / "session.json").write_text(spec.env.to_json())
            mcp = MCPClient([PYTHON, "-m", "mrbench.agent.mcp", "--config", str(box / "session.json")],
                            {**os.environ, **server_env()}, str(box))
            tools = [{"name": t["name"], "description": t["description"], "input_schema": t["inputSchema"]}
                     for t in mcp.tools()]
            done = Path(spec.env.log) / "output.json"
        messages = [{"role": "user", "content": spec.first}]
        r = SessionResult(spec.name, log, turns=0)
        try:
            with (log / "transcript.jsonl").open("w") as tr:
                while not (done and done.exists()):
                    if time.time() - t0 > spec.timeout_s:
                        r.exit = "timeout"
                        break
                    try:
                        with client.messages.stream(
                                model=MODELS.get(self.model, self.model), max_tokens=MAX_TOKENS, system=spec.system,
                                messages=messages, thinking={"type": "adaptive"},
                                output_config={"effort": self.effort}, cache_control={"type": "ephemeral"},
                                **({"tools": tools} if tools else {})) as s:
                            m = s.get_final_message()
                    except Exception as e:  # noqa: BLE001 -- the SDK has retried already; the session ends here
                        tr.write(json.dumps({"error": f"{type(e).__name__}: {e}"}) + "\n")
                        r.exit = "error"
                        break
                    r.turns += 1
                    r.cost_usd += self._cost(m.usage)
                    tr.write(json.dumps({"turn": r.turns, "stop": m.stop_reason, "id": m.id,
                                         "usage": m.usage.model_dump(exclude_none=True),
                                         "content": [b.model_dump(exclude_none=True) for b in m.content]}) + "\n")
                    messages.append({"role": "assistant", "content": m.content})
                    r.text = "".join(b.text for b in m.content if b.type == "text") or r.text
                    calls = [b for b in m.content if b.type == "tool_use"]
                    # a call is run whatever the stop reason says (a model may end a turn on a call with
                    # "end_turn"), unless its input may be cut short or was refused
                    if not calls or mcp is None or m.stop_reason in ("max_tokens", "refusal"):
                        break
                    results = []
                    for c in calls:
                        try:
                            reply = mcp.call(c.name, c.input)
                        except RuntimeError as e:  # the server's own error, not the tool's
                            reply = {"content": [{"type": "text", "text": f"Error: {e}"}], "isError": True}
                        results.append({"type": "tool_result", "tool_use_id": c.id,
                                        "content": [_api_block(p) for p in reply.get("content", [])],
                                        **({"is_error": True} if reply.get("isError") else {})})
                    messages.append({"role": "user", "content": results})
        finally:
            if mcp is not None:
                mcp.close()
        r.cost_usd = round(r.cost_usd, 4)
        r.seconds = round(time.time() - t0, 1)
        if spec.env is not None:
            r.output, r.tool_calls = _read_output(Path(spec.env.log)), _count_calls(Path(spec.env.log))
        return r


#: a fake session: (spec, env or None) -> the final text; it calls ``env.logged_call`` for its tools
Player = Callable[[SessionSpec, Optional[Env]], str]


class FakeBackend(Backend):
    """No model: ``player`` plays every session against an in-process ``Env`` (whose simulator is ``sim``)."""

    def __init__(self, player: Player, sim=None, label: str = "fake"):
        self.player, self.sim, self.label = player, sim, label

    def run(self, spec: SessionSpec, log: Path, box: Path) -> SessionResult:
        log.mkdir(parents=True, exist_ok=True)
        t0 = time.time()
        env = Env(spec.env, sim=self.sim) if spec.env is not None else None
        text = self.player(spec, env) or ""
        (log / "transcript.jsonl").write_text(json.dumps({"fake": True, "text": text}) + "\n")
        return SessionResult(spec.name, log, _read_output(log) if env else {}, text, 1, 0.0,
                             round(time.time() - t0, 1), tool_calls=_count_calls(log))
