"""RoCo-style (Mandi et al., arXiv 2307.04738): one LLM per arm, in dialogue; arms volunteer for parts; the dialogue
ends when every arm agrees on one plan, and that plan is carried out.

Adapted only in what it takes in and gives out (docs/BASELINES.md): each arm sees the fixed cameras and only its own
wrist camera, and queries only its own joints and kinematics (``read_arm``, ``can_reach``); an arm may choose not to
take part; the team may say "infeasible, and what is missing"; things on the table are resources.  The agreed plan
becomes the team plan; the shared executor writes and runs the program.  After a failed run the report goes to every
arm as a message from the environment and a new dialogue starts.
"""

from __future__ import annotations

import json
from typing import Optional

from ..env import SHARED
from ..episode import Episode, feedback_text
from .base import PROPOSAL, Method, prompt, record_proposal

TURN = {"type": "object", "properties": {
    "taking_part": {"type": "boolean", "description": "whether you take part in the plan you back"},
    "proposal": {**PROPOSAL, "description": "a whole team plan, if you propose one (the fields of submit_team_plan)"},
    "agree": {"type": "boolean", "description": "you agree with the latest proposal as it stands"}}}
ROUNDS = 3
#: each arm's tools, the shared reasoning tools among them (its view keeps ``who_can`` and ``meet`` to its own arm)
ARM_TOOLS = ["briefing", "look", "read_arm", "locate", "can_reach", "measure", "note", *SHARED, "respond"]


class RoCo(Method):
    name = "roco"

    def plan(self, ep: Episode, feedback: Optional[dict]) -> Optional[dict]:
        order = sorted(ep.arms, key=lambda a: ("west", "south", "east", "north").index(a)
                       if a in ("west", "south", "east", "north") else 9)
        dialogue: list[str] = []
        if feedback:
            dialogue.append("[environment] " + feedback_text(feedback).strip())
        latest, latest_by, agreed = None, None, set()
        for k in range(1, ROUNDS + 1):
            for arm in order:
                r = ep.session(f"turn{k}-{arm}", prompt("roco_arm", ARM=arm, ARMS=", ".join(order)),
                               "The dialogue so far:\n" + ("\n".join(dialogue) or "(you speak first)")
                               + (f"\n\nThe latest proposal ({latest_by}'s):\n{json.dumps(latest, indent=1)}"
                                  if latest else "") + "\n\nYour turn.", ARM_TOOLS, view=arm, respond_schema=TURN,
                               respond_desc="Say your message to the team (message) and, in data, whether you take "
                                            "part, a proposal if you make one, and whether you agree with the latest "
                                            "proposal; this ends your turn.", calls=25)
                out = r.output.get("respond") or {}
                data = out.get("data") or {}
                dialogue.append(f"[{arm}] {out.get('message', '(said nothing)')}")
                if data.get("proposal"):
                    latest, latest_by, agreed = data["proposal"], arm, {arm}
                    dialogue.append(f"[{arm} proposes] {json.dumps(data['proposal'])}")
                elif latest is not None and data.get("agree"):
                    agreed.add(arm)
                if latest is not None and agreed >= set(order):
                    break
            if latest is not None and agreed >= set(order):
                break
        (ep.root / f"round{ep.round}" / "dialogue.txt").write_text("\n".join(dialogue))
        if latest is None:
            return None
        return record_proposal(ep, latest, "summariser")
