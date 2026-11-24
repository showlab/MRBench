"""LLM + solver: the model writes the task as a job graph, a solver searches the arm sets (``graph.solve_team``)."""

from __future__ import annotations

from typing import Optional

from ..env import SENSES, SHARED
from ..episode import Episode, feedback_text
from .base import FIRST, Method, prompt


class LLMSolver(Method):
    """The model translates the task into a job graph (doers: arms and things it has shown can do a job), a
    solver searches the arm sets.  No solution means infeasible, with the jobs left uncovered."""

    name = "llm-solver"
    tools = list(SENSES) + list(SHARED) + ["solve_team", "submit_team_plan"]

    def plan(self, ep: Episode, feedback: Optional[dict]) -> Optional[dict]:
        r = ep.session("stage1", prompt("solver"), FIRST + feedback_text(feedback), self.tools, calls=60)
        return r.output.get("submit_team_plan")
