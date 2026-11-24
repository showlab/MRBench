"""The compared methods (docs/BASELINES.md), by row name.  A plugin adds rows of its own
(``add_rows``, ``core/plugins.py``)."""

from __future__ import annotations

from .base import Method

#: the compared rows (docs/BASELINES.md): three earlier methods, adapted, and LLM + solver; a plugin's rows join
#: them through ``add_rows``
MAIN = ("smart-llm", "roco", "hmas2", "llm-solver")
#: no LLM: what the executor and the scoring can reach at best
REFERENCES = ("oracle-script", "oracle-team")

_ROWS: list[type[Method]] = []
_MAIN: list[str] = []
_COMPARED: list[str] = []
_REFERENCES: list[str] = []


def add_rows(*rows: type[Method], main=(), compared=(), references=()) -> None:
    """A plugin's rows; ``main``, ``compared`` and ``references`` name those of them that join the main table, that
    every other row is compared with (``grade.paired``), and that are reference rows."""
    _ROWS.extend(rows)
    _MAIN.extend(main)
    _COMPARED.extend(compared)
    _REFERENCES.extend(references)


def registry() -> dict[str, type[Method]]:
    from ...core import plugins
    from .hmas2 import HMAS2
    from .llm_solver import LLMSolver
    from .reference import OracleScript, OracleTeam, OracleTeamSkill
    from .roco import RoCo
    from .smart_llm import SmartLLM

    plugins.load()
    rows = (OracleScript, OracleTeam, OracleTeamSkill, LLMSolver, SmartLLM, RoCo, HMAS2, *_ROWS)
    return {m.name: m for m in rows}


def main_rows() -> tuple[str, ...]:
    """The main table's rows, a plugin's among them."""
    registry()
    return MAIN + tuple(_MAIN)


def compared_rows() -> tuple[str, ...]:
    """The rows every other row is compared with, paired by scene (``grade.paired``); the package names none."""
    registry()
    return tuple(_COMPARED)


def reference_rows() -> tuple[str, ...]:
    registry()
    return REFERENCES + tuple(_REFERENCES)


def get(name: str) -> Method:
    rows = registry()
    if name not in rows:
        raise KeyError(f"no method {name!r}; methods: {', '.join(rows)}")
    return rows[name]()


__all__ = ["Method", "registry", "get", "add_rows", "main_rows", "compared_rows", "reference_rows", "MAIN",
           "REFERENCES"]
