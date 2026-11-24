"""Episode logs are JSONL: one EpisodeResult per line, appended as episodes finish."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterator


def append(path: Path, record: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as f:
        f.write(json.dumps(record) + "\n")


def read(path: Path) -> Iterator[dict]:
    with path.open() as f:
        for line in f:
            if line.strip():
                yield json.loads(line)
