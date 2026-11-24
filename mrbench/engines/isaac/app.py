"""Isaac Sim application lifecycle.

``SimulationApp`` must exist before any ``omni.*``, ``pxr`` or ``isaaclab``
import, and only once per process.  Every Isaac import in this package is
therefore inside a function that runs after :func:`launch`.
"""

from __future__ import annotations

import importlib.util
import os
import sys
from typing import Optional

from ...core.errors import EngineNotAvailable

_LAUNCHER = None

INSTALL_HINT = "Isaac Sim / Isaac Lab are not importable in this interpreter; see docs/GETTING_STARTED.md."

EULA_HINT = """Isaac Sim will not start until NVIDIA's licence terms are accepted.
Read them, and if you accept, `export OMNI_KIT_ACCEPT_EULA=YES`.
MultiRobotBench never sets this for you."""


def is_available() -> bool:
    return all(importlib.util.find_spec(m) is not None for m in ("isaacsim", "isaaclab"))


#: Isaac Lab builds every Nucleus USD path (the Panda's included) from this
#: setting when ``isaaclab`` is first imported.
CLOUD_ROOT_SETTING = "/persistent/isaac/asset_root/cloud"


def launch(headless: bool = True, enable_cameras: bool = False, device: Optional[str] = None):
    """Start the app once per process and return the ``SimulationApp``.

    ``MRBENCH_ISAAC_GPU=<i>`` pins the process to one GPU (as ``cuda:0``); left
    alone, Kit opens every visible GPU and compiles shaders on each.
    ``MRBENCH_KIT_ARGS`` goes to Kit as its own command line, e.g. fewer threads for
    many processes on one shared machine: ``--/plugins/carb.tasking.plugin/threadCount=16
    --/persistent/physics/numThreads=4`` (each process otherwise starts ~560 threads).
    """
    global _LAUNCHER
    if _LAUNCHER is not None:
        return _LAUNCHER.app
    if os.environ.get("MRBENCH_ISAAC_GPU"):
        os.environ.setdefault("CUDA_VISIBLE_DEVICES", os.environ["MRBENCH_ISAAC_GPU"])
    if not is_available():
        raise EngineNotAvailable(INSTALL_HINT)
    if os.environ.get("OMNI_KIT_ACCEPT_EULA", "").upper() not in ("YES", "Y", "1") and not sys.stdin.isatty():
        raise EngineNotAvailable(EULA_HINT)

    from isaaclab.app import AppLauncher

    kwargs = {"headless": headless, "enable_cameras": enable_cameras}
    if device:
        kwargs["device"] = device
    if os.environ.get("MRBENCH_KIT_ARGS"):
        kwargs["kit_args"] = os.environ["MRBENCH_KIT_ARGS"]
    _LAUNCHER = AppLauncher(**kwargs)
    _fix_asset_root()
    return _LAUNCHER.app


def _fix_asset_root() -> None:
    """A pip install of Isaac Sim 4.5 leaves the cloud asset root empty, so Isaac
    Lab's Nucleus paths start with the string ``None`` and the Panda never loads.
    Copy the resolved root across before anything imports ``isaaclab_assets``."""
    import carb

    settings = carb.settings.get_settings()
    if settings.get(CLOUD_ROOT_SETTING):
        return
    root = os.environ.get("MRBENCH_ISAAC_NUCLEUS")
    if not root:
        try:
            from isaacsim.storage.native import get_assets_root_path

            root = get_assets_root_path()
        except Exception:
            root = None
    root = root or settings.get("/persistent/isaac/asset_root/default")
    if root:
        settings.set(CLOUD_ROOT_SETTING, root)


def close() -> None:
    global _LAUNCHER
    if _LAUNCHER is not None:
        _LAUNCHER.app.close()
        _LAUNCHER = None
