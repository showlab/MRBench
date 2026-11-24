"""Extensions from outside the package: task cards, case setups, method rows, tools and commands.

A method of your own joins the comparison framework this way.  A plugin is a module that a distribution names in
the entry-point group ``mrbench.plugins``, or that the environment variable ``MRBENCH_PLUGINS`` names
(comma-separated ``module`` or ``module:function``, for an interpreter where the plugin is importable but not
installed, such as Isaac Sim's).  Each is imported once, the first time one of the registries is read, and its
``register()`` (or the named function) is called.  It adds what it brings through the registries' own functions:

* ``mrbench.tasks.loader.add_cards(directory)`` -- task cards (``*.yaml``) read after the package's own;
* ``mrbench.agent.cases.add_setups({task: TaskSetup})`` -- the agent cases of those tasks;
* ``mrbench.agent.methods.add_rows(*classes, main=(), compared=(), references=())`` -- method rows;
* ``mrbench.agent.env.add_tool(name, handler, schema, citable=False)`` and ``add_monitor(factory)`` -- tools a
  row's sessions may be given;
* ``mrbench.cli.add_command(name, build)`` -- subcommands of ``mrbench``.

``MRBENCH_NO_PLUGINS=1`` loads none (the benchmark and the baselines alone).
"""

from __future__ import annotations

import importlib
import os
import threading
from importlib import metadata

GROUP = "mrbench.plugins"

_LOCK = threading.RLock()
_loaded: list[str] | None = None
_loading = False


def names() -> list[str]:
    """The plugins to load, each once: entry points, then ``MRBENCH_PLUGINS``."""
    if os.environ.get("MRBENCH_NO_PLUGINS"):
        return []
    found = [ep.value for ep in metadata.entry_points(group=GROUP)]
    found += [n.strip() for n in os.environ.get("MRBENCH_PLUGINS", "").split(",") if n.strip()]
    return list(dict.fromkeys(found))


def load() -> list[str]:
    """Import every plugin and call its register function, once; the names loaded.  Called by each registry before
    it is read; a call made while the plugins are being loaded (a plugin reading a registry) returns at once."""
    global _loaded, _loading
    with _LOCK:
        if _loaded is not None or _loading:
            return list(_loaded or [])
        _loading = True
        try:
            done = []
            for name in names():
                module, _, fn = name.partition(":")
                getattr(importlib.import_module(module), fn or "register")()
                done.append(name)
            _loaded = done
        finally:
            _loading = False
        return list(_loaded)


def child_env() -> dict[str, str]:
    """What a child process of ``mrbench`` (an Isaac run, a session's tool server) needs to load the plugins this
    process has, from the same places: ``PYTHONPATH`` (the repository root, then each plugin's directory) and
    ``MRBENCH_PLUGINS`` (or ``MRBENCH_NO_PLUGINS``)."""
    from pathlib import Path

    from .paths import REPO_ROOT

    loaded = load()
    paths = [str(REPO_ROOT)]
    for name in loaded:
        top = importlib.import_module(name.partition(":")[0].split(".")[0])
        where = str(Path(top.__file__).resolve().parents[1])
        if where not in paths:
            paths.append(where)
    env = {"PYTHONPATH": os.pathsep.join(paths)}
    if os.environ.get("MRBENCH_NO_PLUGINS"):
        env["MRBENCH_NO_PLUGINS"] = "1"
    elif loaded:
        env["MRBENCH_PLUGINS"] = ",".join(loaded)
    return env
