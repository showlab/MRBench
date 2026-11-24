"""Serves one session's ``Env`` over MCP (JSON-RPC 2.0, one message per line): on stdio, as Claude Code starts an
MCP server, or on a Unix socket, for a session that runs in a sandbox where the repository is not there to start the
server from (``llm.ClaudeHarness``).  Inside such
a sandbox ``BRIDGE`` -- the standard library only, copied in as a file -- joins Claude Code's stdio to the socket.

    python -m mrbench.agent.mcp --config SESSION.json [--socket PATH]
"""

from __future__ import annotations

import argparse
import json
import os
import socket
import sys
from pathlib import Path

from .env import Env, EnvConfig

PROTOCOL = "2024-11-05"

#: stdio <-> a Unix socket, for a sandbox: python3 bridge.py SOCKET
BRIDGE = r'''import socket, sys, threading
s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
s.connect(sys.argv[1])
def up():
    for line in sys.stdin.buffer:
        s.sendall(line)
    s.shutdown(socket.SHUT_WR)
threading.Thread(target=up, daemon=True).start()
f = s.makefile("rb")
for line in f:
    sys.stdout.buffer.write(line)
    sys.stdout.buffer.flush()
'''


def handle(env: Env, msg: dict):
    method = msg.get("method")
    if method == "initialize":
        return {"protocolVersion": msg.get("params", {}).get("protocolVersion", PROTOCOL),
                "capabilities": {"tools": {}}, "serverInfo": {"name": "mrbench-env", "version": "0.2"}}
    if method == "tools/list":
        return {"tools": env.tools()}
    if method == "tools/call":
        p = msg["params"]
        content, err = env.logged_call(p["name"], p.get("arguments") or {})
        return {"content": content, "isError": err}
    if method == "ping":
        return {}
    if msg.get("id") is None:  # a notification
        return None
    raise LookupError(method)


def serve(env: Env, rfile, wfile) -> None:
    for line in rfile:
        line = line.strip()
        if not line:
            continue
        msg = json.loads(line)
        try:
            result = handle(env, msg)
            if msg.get("id") is None:
                continue
            out = {"jsonrpc": "2.0", "id": msg["id"], "result": result}
        except LookupError as e:
            out = {"jsonrpc": "2.0", "id": msg.get("id"), "error": {"code": -32601, "message": f"no method {e}"}}
        wfile.write(json.dumps(out) + "\n")
        wfile.flush()


def serve_socket(env: Env, path: Path) -> None:
    """Serve one connection after another on a Unix socket, until the process is stopped."""
    path = Path(path)
    if path.exists():
        path.unlink()
    srv = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    srv.bind(str(path))
    os.chmod(path, 0o600)
    srv.listen(1)
    while True:
        conn, _ = srv.accept()
        with conn, conn.makefile("r") as r, conn.makefile("w") as w:
            serve(env, r, w)


def main(argv=None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True, type=Path)
    p.add_argument("--socket", type=Path)
    a = p.parse_args(argv)
    env = Env(EnvConfig.from_json(a.config.read_text()))
    if a.socket:
        serve_socket(env, a.socket)
    else:
        serve(env, sys.stdin, sys.stdout)
    return 0


if __name__ == "__main__":
    sys.exit(main())
