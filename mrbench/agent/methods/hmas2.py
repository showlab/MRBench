"""HMAS-2-style (Chen et al., arXiv 2309.15943, the hybrid of a central planner and local feedback): a central LLM
proposes the whole plan; each arm's own LLM checks its part and accepts or vetoes it; the centre revises until all
accept (or the rounds run out), then the plan is carried out.

Adapted only in what it takes in and gives out (docs/BASELINES.md): the proposal names the team, the idle arms, the
things and whether the task can be done; each arm, with the same view as in RoCo-style (the fixed cameras, only its
own wrist camera, ``read_arm`` and ``can_reach``), vetoes with its own kinematics.  The centre does not query the
arms' kinematics itself.  The proposals and the answers are kept on a plan board.  After a failed run the report goes to the centre and every arm; a new round of proposal and
checks.
"""

from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor
from typing import Optional

from ..env import SHARED
from ..episode import Episode, feedback_text
from .base import PROPOSAL, Method, prompt, record_proposal

CHECK = {"type": "object", "properties": {"accept": {"type": "boolean"}, "reason": {"type": "string"},
                                          "evidence": {"type": "string"}}, "required": ["accept", "reason"]}
ROUNDS = 3
#: the shared reasoning tools for both; the centre does not query the arms' kinematics (``who_can``, ``meet``): that
#: is the arms' part in HMAS-2
CENTRE_TOOLS = ["briefing", "look", "locate", "measure", "note", *[t for t in SHARED if t not in ("who_can", "meet")],
                "respond"]
ARM_TOOLS = ["briefing", "look", "read_arm", "locate", "can_reach", "measure", *SHARED, "respond"]


class HMAS2(Method):
    name = "hmas2"

    def plan(self, ep: Episode, feedback: Optional[dict]) -> Optional[dict]:
        board: list[dict] = []  # {"id", "proposal", "responses": {arm: {accept, reason, evidence}}, "status"}
        for k in range(1, ROUNDS + 1):
            past = [{"id": p["id"], "status": p["status"], "responses": p["responses"]} for p in board]
            r = ep.session(f"centre-{k}", prompt("hmas_centre"),
                           ("The plan board (your earlier proposals and the arms' answers):\n"
                            + json.dumps(past, indent=1) + "\n\n" if past else "")
                           + feedback_text(feedback) + "\n\nPropose the plan. Start with `briefing`.",
                           CENTRE_TOOLS, respond_schema=PROPOSAL,
                           respond_desc="Propose the team plan (data) and say why (message); this ends your turn.",
                           calls=40)
            out = r.output.get("respond") or {}
            if not out.get("data"):
                break
            entry = {"id": f"P{k}", "proposal": out["data"], "note": out.get("message", ""), "responses": {},
                     "status": "pending"}
            board.append(entry)

            def check(arm):
                rr = ep.session(f"check{k}-{arm}", prompt("hmas_arm", ARM=arm),
                                f"The centre's proposal {entry['id']}:\n{json.dumps(entry['proposal'], indent=1)}\n\n"
                                f"The centre says: {entry['note']}" + feedback_text(feedback)
                                + "\n\nCheck your part (or that you are left idle) with your own tools.",
                                ARM_TOOLS, view=arm, respond_schema=CHECK,
                                respond_desc="Accept or veto (data, with the reason and your evidence); this ends "
                                             "your turn.", calls=20)
                return arm, (rr.output.get("respond") or {}).get("data") or {"accept": False, "reason": "no answer"}
            with ThreadPoolExecutor(len(ep.arms)) as pool:
                entry["responses"] = dict(pool.map(check, ep.arms))
            entry["status"] = "agreed" if all(v.get("accept") for v in entry["responses"].values()) else "rejected"
            if entry["status"] == "agreed":
                break
        (ep.root / f"round{ep.round}" / "board.json").write_text(json.dumps(board, indent=1))
        if not board:
            return None
        return record_proposal(ep, board[-1]["proposal"], "centre")
