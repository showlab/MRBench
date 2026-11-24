"""SMART-LLM-style (Kannan et al., arXiv 2309.10062): a fixed, centralised procedure in stages -- task decomposition,
coalition formation, task allocation -- then execution.

Adapted only in what it takes in and gives out (docs/BASELINES.md): coalition formation queries the arms
(``can_reach``, ``measure``) instead of reading skill lists; things on the table are passive candidates for a
subtask; a subtask with no coalition means infeasible; the allocation asks for the fewest robots (SMART-LLM's own
"minimum number of robots").  After a failed run, the last plan and the report go into the decomposition prompt and
the stages run again.
"""

from __future__ import annotations

import json
from typing import Optional

from ..env import SHARED
from ..episode import Episode, feedback_text
from .base import Method, prompt

POINT = {"type": "array", "items": {"type": "number"}}
SUBTASKS = {"type": "object", "properties": {"subtasks": {"type": "array", "items": {"type": "object", "properties": {
    "id": {"type": "string"}, "what": {"type": "string"}, "where": POINT,
    "skills": {"type": "array", "items": {"type": "string"}},
    "at_same_time_as": {"type": "array", "items": {"type": "string"}},
    "after": {"type": "array", "items": {"type": "string"}}}, "required": ["id", "what", "skills"]}}},
    "required": ["subtasks"]}
COALITIONS = {"type": "object", "properties": {"coalitions": {"type": "array", "items": {"type": "object", "properties": {
    "subtask": {"type": "string"}, "arms": {"type": "array", "items": {"type": "string"}},
    "things": {"type": "array", "items": {"type": "string"},
               "description": "things on the table that could do this subtask instead of an arm (a weight, a stop, a "
                              "support, a tool, a carrier) -- not the things the subtask acts on"},
    "evidence": {"type": "string"}},
    "required": ["subtask", "arms", "things"]}}}, "required": ["coalitions"]}


class SmartLLM(Method):
    name = "smart-llm"

    def plan(self, ep: Episode, feedback: Optional[dict]) -> Optional[dict]:
        r1 = ep.session("decompose", prompt("smart_decompose"),
                        "Decompose the task on this table. Start with `briefing`." + feedback_text(feedback),
                        ["briefing", "look", "locate", "measure", "note", "scene_graph", "respond"], respond_schema=SUBTASKS,
                        respond_desc="Give the subtasks (in data); this ends the stage.", calls=30)
        subtasks = (r1.output.get("respond") or {}).get("data")
        if not subtasks:
            return None
        r2 = ep.session("coalition", prompt("smart_coalition"),
                        "The subtasks:\n" + json.dumps(subtasks, indent=1) + "\n\nForm a coalition for each. Start "
                        "with `briefing`.", ["briefing", "read_arm", "locate", "measure", "can_reach", "heft", "note",
                                             *SHARED, "respond"],
                        respond_schema=COALITIONS, respond_desc="Give the coalitions (in data); this ends the stage.",
                        calls=60)
        coalitions = (r2.output.get("respond") or {}).get("data")
        if not coalitions:
            return None
        r3 = ep.session("allocate", prompt("smart_allocate"),
                        "The subtasks:\n" + json.dumps(subtasks, indent=1) + "\n\nThe coalitions:\n"
                        + json.dumps(coalitions, indent=1) + "\n\nAllocate, and submit the team plan.",
                        ["briefing", "submit_team_plan"], calls=5)
        return r3.output.get("submit_team_plan")
