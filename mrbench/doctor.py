"""`mrbench doctor`: what is installed, what is missing, what to do next.  Never launches Isaac."""

from __future__ import annotations

import importlib.metadata as md
import importlib.util
import os
import platform
import shutil
import subprocess
import sys
from dataclasses import dataclass

from .robots import MODELS
from .robots import fetch as fetch_mod

OK, WARN, FAIL = "ok", "warn", "FAIL"


@dataclass
class Check:
    name: str
    status: str
    detail: str


def _python() -> Check:
    v = sys.version_info
    ok = (v.major, v.minor) in ((3, 10), (3, 11))
    return Check("python", OK if ok else WARN,
                 f"{platform.python_version()} ({'matches' if ok else 'Isaac Sim 4.5 ships 3.10;'} Isaac Sim 4.5)")


def _glibc() -> Check:
    lib, ver = platform.libc_ver()
    if lib != "glibc":
        return Check("glibc", WARN, f"{lib or 'unknown'} {ver}")
    major, minor = (int(x) for x in ver.split(".")[:2])
    if (major, minor) >= (2, 34):
        return Check("glibc", OK, f"{ver}: pip wheels of isaacsim 4.5 (manylinux_2_34) install")
    return Check("glibc", WARN, f"{ver} < 2.34: `pip install isaacsim` is refused; use the standalone "
                                "Isaac Sim 4.5 package or the container (docs/GETTING_STARTED.md)")


def _gpu() -> Check:
    if not shutil.which("nvidia-smi"):
        return Check("gpu", FAIL, "nvidia-smi not found")
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=index,name,driver_version,memory.used,memory.total",
                              "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=20).stdout
    except (OSError, subprocess.SubprocessError) as e:
        return Check("gpu", FAIL, str(e))
    rows = [r.split(", ") for r in out.strip().splitlines()]
    if not rows:
        return Check("gpu", FAIL, "no GPU listed")
    free = [(r[0], int(r[4]) - int(r[3])) for r in rows]
    best = max(free, key=lambda x: x[1])
    status = OK if best[1] >= 12000 else WARN
    return Check("gpu", status, f"{len(rows)} x {rows[0][1]}, driver {rows[0][2]}; most free: GPU {best[0]} "
                                f"with {best[1] / 1024:.1f} GiB (Isaac Sim wants >= 12 GiB)")


def _module(name: str) -> Check:
    if importlib.util.find_spec(name) is None:
        return Check(name, WARN, "not importable here (needed only for --engine isaac)")
    try:
        return Check(name, OK, md.version(name))
    except md.PackageNotFoundError:
        return Check(name, OK, "importable")


def _eula() -> Check:
    ok = os.environ.get("OMNI_KIT_ACCEPT_EULA", "").upper() in ("YES", "Y", "1")
    return Check("eula", OK if ok else WARN, "OMNI_KIT_ACCEPT_EULA set" if ok else
                 "OMNI_KIT_ACCEPT_EULA not set (needed for headless Isaac runs; your decision)")


def _assets() -> list[Check]:
    from .engines.isaac.robots import usd_path

    out = []
    for name, m in MODELS.items():
        if m.source != "urdf":
            out.append(Check(f"robot:{name}", OK, f"from {m.source}"))
            continue
        st = fetch_mod.status(name)
        usd = usd_path(name)
        detail = st["detail"] + ("; USD converted" if usd.exists() else "; USD not converted (`mrbench assets convert`)")
        out.append(Check(f"robot:{name}", OK if st["ok"] and usd.exists() else WARN, detail))
    return out


def _plugins() -> Check:
    from .core import plugins

    loaded = plugins.load()
    return Check("plugins", OK, ", ".join(loaded) if loaded else "none (the package's own tasks and rows only)")


def run() -> list[Check]:
    return [_python(), _glibc(), _gpu(), _module("isaacsim"), _module("isaaclab"), _module("isaaclab_assets"),
            _eula(), _plugins(), *_assets()]
