"""Download pinned robot descriptions (URDF + meshes) from GitHub.

Only the URDF and the meshes it references are fetched -- not the whole
upstream repository.  ``package://`` URIs are rewritten to paths relative to
the URDF so Isaac Sim's importer can resolve them without ROS.
"""

from __future__ import annotations

import hashlib
import json
import posixpath
import re
import urllib.request
from importlib import resources
from pathlib import Path

import yaml

from ..core.errors import AssetMissing, ConfigError
from ..core.paths import third_party_dir

RAW_URL = "https://raw.githubusercontent.com/{repo}/{commit}/{path}"
_MESH_RE = re.compile(r'filename="([^"]+)"')


def load_sources() -> dict:
    text = resources.files("mrbench.robots").joinpath("sources.yaml").read_text()
    return yaml.safe_load(text)


def robot_dir(name: str) -> Path:
    return third_party_dir() / "robots" / name


def urdf_path(name: str) -> Path:
    return robot_dir(name) / f"{name}.urdf"


def _safe_rel(path: str) -> str:
    """Normalise a relative POSIX path and refuse anything escaping its root."""
    norm = posixpath.normpath(path)
    if norm.startswith(("/", "..")) or norm == ".":
        raise ConfigError(f"refusing unsafe mesh path {path!r}")
    return norm


def _resolve_mesh(uri: str, src: dict) -> tuple[str, str]:
    """Map a mesh URI in the URDF to (path in upstream repo, path relative to local URDF)."""
    if uri.startswith("package://"):
        pkg, _, rest = uri[len("package://"):].partition("/")
        if pkg not in src.get("package_map", {}):
            raise ConfigError(f"no package_map entry for {pkg!r} (mesh {uri})")
        rest = _safe_rel(rest)
        return f"{src['package_map'][pkg]}/{rest}", rest
    local = _safe_rel(uri)
    return posixpath.normpath(posixpath.join(posixpath.dirname(src["urdf"]), local)), local


def _download(url: str, timeout: float = 60.0) -> bytes:
    with urllib.request.urlopen(url, timeout=timeout) as resp:  # noqa: S310 (fixed https host)
        return resp.read()


def fetch(name: str, force: bool = False) -> Path:
    """Fetch robot ``name``; returns the local URDF path.  Idempotent unless ``force``."""
    sources = load_sources()
    if name not in sources:
        raise ConfigError(f"{name!r} has no entry in robots/sources.yaml")
    src = sources[name]
    out = robot_dir(name)
    if not force and status(name)["ok"]:
        return urdf_path(name)

    raw = lambda p: RAW_URL.format(repo=src["repo"], commit=src["commit"], path=p)  # noqa: E731
    urdf_text = _download(raw(src["urdf"])).decode()
    files: dict[str, str] = {}
    for uri in sorted(set(_MESH_RE.findall(urdf_text))):
        repo_path, local = _resolve_mesh(uri, src)
        data = _download(raw(repo_path))
        dest = out / local
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(data)
        files[local] = hashlib.sha256(data).hexdigest()
        urdf_text = urdf_text.replace(f'filename="{uri}"', f'filename="{local}"')

    out.mkdir(parents=True, exist_ok=True)
    urdf_path(name).write_text(urdf_text)
    files[urdf_path(name).name] = hashlib.sha256(urdf_text.encode()).hexdigest()
    stamp = {"repo": src["repo"], "commit": src["commit"], "license": src.get("license"), "files": files}
    (out / "stamp.json").write_text(json.dumps(stamp, indent=2) + "\n")
    return urdf_path(name)


def status(name: str) -> dict:
    """Whether ``name`` is fetched at the pinned commit with unmodified files."""
    stamp_file = robot_dir(name) / "stamp.json"
    if not stamp_file.exists():
        return {"ok": False, "detail": "not fetched"}
    stamp = json.loads(stamp_file.read_text())
    pinned = load_sources()[name]["commit"]
    if stamp.get("commit") != pinned:
        return {"ok": False, "detail": f"fetched {stamp.get('commit', '?')[:8]}, pinned {pinned[:8]}"}
    for rel, digest in stamp["files"].items():
        f = robot_dir(name) / rel
        if not f.exists() or hashlib.sha256(f.read_bytes()).hexdigest() != digest:
            return {"ok": False, "detail": f"{rel} missing or modified"}
    return {"ok": True, "detail": f"{len(stamp['files'])} files @ {pinned[:8]}"}


def require(name: str) -> Path:
    st = status(name)
    if not st["ok"]:
        raise AssetMissing(f"robot {name!r}: {st['detail']}; run `mrbench assets fetch`")
    return urdf_path(name)
