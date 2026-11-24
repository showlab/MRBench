"""Filesystem locations.  Large or third-party files never live in the package."""

from __future__ import annotations

import os
from pathlib import Path

#: Repository root when running from a source checkout (the supported mode).
REPO_ROOT = Path(__file__).resolve().parents[2]


def third_party_dir() -> Path:
    """Fetched robot descriptions and converted USDs (gitignored).

    Override with ``MRBENCH_THIRD_PARTY`` to keep them on another disk.
    """
    return Path(os.environ.get("MRBENCH_THIRD_PARTY", REPO_ROOT / "third_party"))


def runs_dir() -> Path:
    return Path(os.environ.get("MRBENCH_RUNS", REPO_ROOT / "runs"))
