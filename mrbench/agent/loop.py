"""The agent loop for any model behind an OpenAI-compatible chat API (a local vLLM serving Qwen, ...), talking to the
same environment servers as the Claude Code runs (stdio MCP).  Standard library only.

Here the context is ours to keep, and is kept so:

* the system prompt and the first message stay; every assistant turn keeps its tool calls and its text, never its
  reasoning (Qwen's own advice: past thinking is not sent back);
* tool results stay as text; a picture comes back as a user message after the tool's reply, and only the latest
  ``KEEP_IMAGES`` pictures are sent -- older ones become a line saying which camera it was (the agent can look again);
* a tool result more than ``KEEP_FULL`` calls old is cut to ``OLD_CHARS`` characters, and only the latest notebook
  is kept whole (each ``note`` reply holds the whole notebook);
* the server's budget ends the stage; a model that stops calling tools is reminded, at most ``NUDGES`` times.

Everything sent and received is logged (pictures as ``<image>``) to ``transcript.jsonl``.
"""

from __future__ import annotations

import copy
import json
import os
import subprocess
import time
import urllib.request
from pathlib import Path

#: the environment variable that holds a cloud endpoint's key (set by whoever starts the run; never logged)
KEY_ENV = "MRBENCH_OPENAI_API_KEY"
KEEP_IMAGES = 3
KEEP_FULL = 12
OLD_CHARS = 400
NUDGES = 3
MAX_TURNS = 120
#: a cloud endpoint's output filter may cut an answer short (``data_inspection_failed``): that turn is asked again,
#: at most this many times in a row and this many times a session, apart from the 3 other request errors
INSPECTIONS = (8, 60)
#: what an old picture becomes, and what a picture becomes that was in view when the filter cut an answer (the cuts
#: come with pictures in the context: the turn is asked again without them, see ``_withdraw``)
OLD_PICTURE = "(this picture is no longer shown; look again if needed)"
WITHDRAWN = ("(this picture was taken out: the service cut the answer short while it was shown; you can look again, "
             "but the same may happen)")
#: sampling, as Qwen recommends for each mode
SAMPLING = {True: {"temperature": 0.6, "top_p": 0.95, "top_k": 20, "max_tokens": 16384},
            False: {"temperature": 0.7, "top_p": 0.8, "top_k": 20, "max_tokens": 4096}}
NUDGE = ("You have not called a tool. Keep working with the tools; when you are done, call the submission tool "
         "({submit}).")


class MCPClient:
    """A stdio MCP server in a child process."""

    def __init__(self, cmd: list[str], env: dict, cwd: str):
        self.p = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
                                  text=True, env=env, cwd=cwd)
        self.n = 0
        self.request("initialize", {"protocolVersion": "2024-11-05", "capabilities": {},
                                    "clientInfo": {"name": "mrbench-loop", "version": "0.1"}})
        self.p.stdin.write(json.dumps({"jsonrpc": "2.0", "method": "notifications/initialized"}) + "\n")
        self.p.stdin.flush()

    def request(self, method: str, params: dict) -> dict:
        self.n += 1
        self.p.stdin.write(json.dumps({"jsonrpc": "2.0", "id": self.n, "method": method, "params": params}) + "\n")
        self.p.stdin.flush()
        while True:
            line = self.p.stdout.readline()
            if not line:
                raise RuntimeError("the environment server stopped")
            msg = json.loads(line)
            if msg.get("id") == self.n:
                if "error" in msg:
                    raise RuntimeError(msg["error"])
                return msg["result"]

    def tools(self) -> list[dict]:
        return self.request("tools/list", {})["tools"]

    def call(self, name: str, args: dict) -> dict:
        return self.request("tools/call", {"name": name, "arguments": args})

    def close(self) -> None:
        self.p.stdin.close()
        self.p.wait(timeout=10)


def _chat(base_url: str, body: dict, timeout: float = 1800) -> dict:
    headers = {"Content-Type": "application/json"}
    if base_url.startswith("https://"):  # a cloud endpoint: its key, thinking in the request, streamed
        think = (body.get("chat_template_kwargs") or {}).get("enable_thinking", True)
        body = {k: v for k, v in body.items() if k not in ("chat_template_kwargs", "top_k")}
        body.update(enable_thinking=bool(think), stream=True, stream_options={"include_usage": True})
        if os.environ.get(KEY_ENV):
            headers["Authorization"] = "Bearer " + os.environ[KEY_ENV]
    req = urllib.request.Request(base_url.rstrip("/") + "/chat/completions", data=json.dumps(body).encode(),
                                 headers=headers)
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return _streamed(r) if body.get("stream") else json.loads(r.read())


def _streamed(resp) -> dict:
    """A streamed answer put back together as the one body an answer not streamed is: the words, the thinking, the
    tool calls (their parts joined by index), the usage (the last chunk's)."""
    msg = {"role": "assistant", "content": "", "reasoning_content": ""}
    calls: dict = {}
    usage, finish = None, None
    for raw in resp:
        line = raw.decode("utf-8", "replace").strip()
        if not line.startswith("data:"):
            continue
        data = line[5:].strip()
        if data == "[DONE]":
            break
        chunk = json.loads(data)
        if chunk.get("error"):
            raise RuntimeError(chunk["error"])
        usage = chunk.get("usage") or usage
        for choice in chunk.get("choices") or ():
            delta = choice.get("delta") or {}
            msg["content"] += delta.get("content") or ""
            msg["reasoning_content"] += delta.get("reasoning_content") or delta.get("reasoning") or ""
            for tc in delta.get("tool_calls") or ():
                c = calls.setdefault(tc.get("index", len(calls)), {"id": "", "type": "function",
                                                                   "function": {"name": "", "arguments": ""}})
                c["id"] = tc.get("id") or c["id"]
                fn = tc.get("function") or {}
                c["function"]["name"] += fn.get("name") or ""
                c["function"]["arguments"] += fn.get("arguments") or ""
            finish = choice.get("finish_reason") or finish
    if calls:
        msg["tool_calls"] = [calls[k] for k in sorted(calls)]
    return {"choices": [{"index": 0, "message": msg, "finish_reason": finish}], "usage": usage or {}}


def _view(messages: list[dict]) -> list[dict]:
    """What is sent: the kept history with old pictures and old tool results thinned."""
    out = copy.deepcopy(messages)
    images = [(i, j) for i, m in enumerate(out) if isinstance(m.get("content"), list)
              for j, part in enumerate(m["content"]) if part.get("type") == "image_url"]
    for i, j in images[:-KEEP_IMAGES]:
        out[i]["content"][j] = {"type": "text", "text": OLD_PICTURE}
    tools = [i for i, m in enumerate(out) if m["role"] == "tool"]
    notes = [i for i in tools if out[i].get("_tool") == "note"]
    for i in tools[:-KEEP_FULL]:
        if out[i].get("_tool") == "note" and i != (notes[-1] if notes else None):
            out[i]["content"] = "(an older copy of the notebook; the latest is further down)"
        elif len(out[i]["content"]) > OLD_CHARS:
            out[i]["content"] = out[i]["content"][:OLD_CHARS] + " ... (cut: an old result)"
    for m in out:
        m.pop("_tool", None)
    return out


def _withdraw(messages: list[dict]) -> int:
    """Take every picture out of the kept history, in place: the ones in view say why.  Returns how many were in view."""
    images = [(i, j) for i, m in enumerate(messages) if isinstance(m.get("content"), list)
              for j, part in enumerate(m["content"]) if part.get("type") == "image_url"]
    for n, (i, j) in enumerate(images):
        shown = n >= len(images) - KEEP_IMAGES
        messages[i]["content"][j] = {"type": "text", "text": WITHDRAWN if shown else OLD_PICTURE}
    return min(len(images), KEEP_IMAGES)


def _strip_images(obj):
    if isinstance(obj, dict):
        if obj.get("type") == "image_url":
            return {"type": "image_url", "image_url": "<image>"}
        return {k: _strip_images(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_strip_images(v) for v in obj]
    return obj


def run(base_url: str, model: str, system: str, first: str, server_cmd: list[str], env: dict, cwd: str, log: Path,
        think: bool, submit_tool: str, done_file: Path) -> dict:
    """One session.  Returns usage: turns, prompt and completion tokens, reasoning characters, nudges, seconds."""
    mcp = MCPClient(server_cmd, env, cwd)
    tools = [{"type": "function", "function": {"name": t["name"], "description": t["description"],
                                               "parameters": t["inputSchema"]}} for t in mcp.tools()]
    messages = [{"role": "system", "content": system}, {"role": "user", "content": first}]
    stats = {"turns": 0, "prompt_tokens": 0, "completion_tokens": 0, "max_prompt_tokens": 0, "reasoning_chars": 0,
             "nudges": 0, "tool_calls": 0, "errors": 0, "inspections": 0, "withdrawn": 0}
    t0, cut = time.time(), 0  # cut: answers in a row the endpoint's filter cut short
    with (log / "transcript.jsonl").open("w") as tr:
        while stats["turns"] < MAX_TURNS and not done_file.exists():
            body = {"model": model, "messages": _view(messages), "tools": tools, "tool_choice": "auto",
                    "chat_template_kwargs": {"enable_thinking": think}, **SAMPLING[think]}
            try:
                resp = _chat(base_url, body)
            except Exception as e:  # noqa: BLE001 -- a server hiccup: log it and try once more
                gone = _withdraw(messages) if "data_inspection_failed" in str(e) else 0
                tr.write(json.dumps({"error": str(e), **({"withdrawn": gone} if gone else {})}) + "\n")
                tr.flush()
                if "data_inspection_failed" in str(e):
                    stats["inspections"] += 1
                    stats["withdrawn"] += gone
                    cut += 1
                else:
                    stats["errors"] += 1
                if stats["errors"] > 3 or cut > INSPECTIONS[0] or stats["inspections"] > INSPECTIONS[1]:
                    stats["gave_up"] = True  # the session ends on request errors, not on what the model did
                    break
                time.sleep(5)
                continue
            cut = 0
            stats["turns"] += 1
            u = resp.get("usage") or {}
            stats["prompt_tokens"] += u.get("prompt_tokens", 0)
            stats["completion_tokens"] += u.get("completion_tokens", 0)
            stats["max_prompt_tokens"] = max(stats["max_prompt_tokens"], u.get("prompt_tokens", 0))
            msg = resp["choices"][0]["message"]
            reasoning = msg.get("reasoning_content") or msg.get("reasoning") or ""
            stats["reasoning_chars"] += len(reasoning)
            tr.write(json.dumps({"turn": stats["turns"], "usage": u, "finish": resp["choices"][0].get("finish_reason"),
                                 "reasoning": reasoning, "content": msg.get("content"),
                                 "tool_calls": msg.get("tool_calls")}) + "\n")
            calls = msg.get("tool_calls") or []
            messages.append({"role": "assistant", "content": msg.get("content") or "",
                             **({"tool_calls": calls} if calls else {})})
            if not calls:
                if stats["nudges"] >= NUDGES:
                    break
                stats["nudges"] += 1
                messages.append({"role": "user", "content": NUDGE.format(submit=submit_tool)})
                continue
            pictures = []
            for c in calls:
                stats["tool_calls"] += 1
                name = c["function"]["name"]
                try:
                    args = json.loads(c["function"].get("arguments") or "{}")
                    result = mcp.call(name, args)
                    parts = result.get("content", [])
                except Exception as e:  # noqa: BLE001 -- bad arguments are the model's to fix
                    parts = [{"type": "text", "text": f"Error: could not call {name}: {e}"}]
                text = "\n".join(p["text"] for p in parts if p["type"] == "text")
                imgs = [p for p in parts if p["type"] == "image"]
                messages.append({"role": "tool", "tool_call_id": c.get("id", ""), "content": text, "_tool": name})
                for p in imgs:
                    pictures.append({"type": "image_url",
                                     "image_url": {"url": f"data:{p['mimeType']};base64,{p['data']}"}})
                tr.write(json.dumps({"tool": name, "args": c["function"].get("arguments"), "reply": text[:2000],
                                     "images": len(imgs)}) + "\n")
            if pictures:
                messages.append({"role": "user", "content": [{"type": "text", "text": "The picture(s) you asked for:"},
                                                             *pictures]})
        stats["seconds"] = round(time.time() - t0)
        tr.write(json.dumps({"type": "result", **stats}) + "\n")
    mcp.close()
    return stats
