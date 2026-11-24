import json
import subprocess
import sys

from mrbench.cli import main


def test_validate_command_passes():
    assert main(["validate"]) == 0


def test_importing_everything_never_imports_isaac():
    """Every module, including mrbench.engines.isaac.*, must import without Isaac Sim."""
    code = (
        "import importlib, pkgutil, sys, mrbench\n"
        "for m in pkgutil.walk_packages(mrbench.__path__, 'mrbench.'):\n"
        "    if m.name != 'mrbench.__main__': importlib.import_module(m.name)\n"
        "bad = [n for n in sys.modules if n.split('.')[0] in ('isaacsim', 'isaaclab', 'isaaclab_assets', 'omni', 'pxr')]\n"
        "print(bad); sys.exit(1 if bad else 0)\n"
    )
    r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert r.returncode == 0, r.stdout + r.stderr
