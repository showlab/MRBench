"""What one episode -- one method on one case, one repeat -- has spent and done, shared by every session in it.

Every session of an episode runs its own environment server (a separate process), and a method may run several at
once (one per arm), so the budgets that belong to the case -- the simulator runs and probes every compared method
gets the same of (docs/PROTOCOL.md) -- and the LLM spending are kept in one JSON file, read and written under a
lock.  It also records every team plan submitted, every program checked and every run, in order: what the episode is
graded on.
"""

from __future__ import annotations

import fcntl
import json
import time
from pathlib import Path
from typing import Any, Callable, Optional

#: the protocol's two settings (docs/PROTOCOL.md): one-shot (the team is submitted before any run) and the equal
#: budget (probes and runs, and a failed run's report goes back to the method); and ``exec``: the team plan carried
#: out in one round -- the shared executor's session and its runs, no probe, nothing back to the method
#: (``mrbench evaluate --plans-from``: the plan is the method's one-shot answer, so what is measured is whether that
#: answer can be carried out)
SETTINGS = ("one-shot", "equal-budget", "exec")
LIMITS = {
    "one-shot": {"probes": 0, "runs": 0, "sessions": 40, "cost_usd": 15.0},
    "equal-budget": {"probes": 2, "runs": 3, "sessions": 80, "cost_usd": 30.0},
    "exec": {"probes": 0, "runs": 3, "sessions": 40, "cost_usd": 15.0},
}
KINDS = ("probes", "runs", "sessions", "cost_usd")


class Ledger:
    def __init__(self, path: Path):
        self.path = Path(path)

    @classmethod
    def create(cls, path: Path, setting: str, limits: Optional[dict] = None, **info) -> "Ledger":
        if setting not in SETTINGS:
            raise ValueError(f"setting must be one of {SETTINGS}, got {setting!r}")
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        d = {"setting": setting, "limits": {**LIMITS[setting], **(limits or {})}, "used": {k: 0 for k in KINDS},
             "started": time.time(), **info, "plans": [], "programs": [], "runs": [], "probes": [], "sessions": []}
        path.write_text(json.dumps(d, indent=1))
        return cls(path)

    # ------------------------------------------------------------------ locked access
    def _update(self, fn: Callable[[dict], Any]) -> Any:
        with self.path.open("r+") as f:
            fcntl.flock(f, fcntl.LOCK_EX)
            d = json.load(f)
            out = fn(d)
            f.seek(0)
            f.truncate()
            json.dump(d, f, indent=1)
            return out

    def read(self) -> dict:
        with self.path.open() as f:
            fcntl.flock(f, fcntl.LOCK_SH)
            return json.load(f)

    # ------------------------------------------------------------------ budgets
    def left(self, kind: str) -> float:
        d = self.read()
        return d["limits"][kind] - d["used"][kind]

    def take(self, kind: str, amount: float = 1) -> Optional[float]:
        """Spend ``amount`` of ``kind`` if that much is left: what is left after, or None (nothing spent)."""
        def fn(d):
            if d["used"][kind] + amount > d["limits"][kind] + 1e-9:
                return None
            d["used"][kind] += amount
            return d["limits"][kind] - d["used"][kind]
        return self._update(fn)

    def spend(self, kind: str, amount: float) -> None:
        """Count what was spent whatever the limit (an LLM session's cost is known only once it is over)."""
        def fn(d):
            d["used"][kind] = round(d["used"][kind] + amount, 6)
        self._update(fn)

    def exhausted(self) -> Optional[str]:
        """The first LLM budget used up (sessions, dollars), if any."""
        d = self.read()
        for k in ("sessions", "cost_usd"):
            if d["used"][k] >= d["limits"][k]:
                return k
        return None

    # ------------------------------------------------------------------ records
    def add(self, kind: str, record: dict) -> int:
        """Append a record (to "plans", "programs", "runs", "probes" or "sessions"); returns its index."""
        def fn(d):
            d[kind].append({"t": round(time.time(), 1), **record})
            return len(d[kind]) - 1
        return self._update(fn)

    def records(self, kind: str) -> list[dict]:
        return self.read()[kind]
