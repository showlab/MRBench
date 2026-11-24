"""The agent and the methods it is compared with, on one environment, one loop and one scoring
(docs/PROTOCOL.md): a method judges the minimal support set from what a robot team could sense.

A session sees no task card, scripted reference or repository: what it gets is what the team could sense --
pictures from fixed and wrist cameras, each arm's joint angles, hand pose and finger opening, a list of what is on the
table -- and the tools its row is given.

* ``cases.py`` -- the test cases: a task's scene with an arm added, an arm taken away, things put down;
* ``snapshot.py`` -- renders a case in Isaac and records the arms' state (Isaac process only);
* ``observe.py`` -- what a session is told and what its queries answer; ``physics.py``, ``graph.py``, ``meet.py`` --
  the reasoning tools;
* ``env.py`` -- the environment a session talks to: the shared tools, a per-arm view, the session's caps;
* ``ledger.py`` -- an episode's shared budgets and records; ``failure.py`` -- the failure report after a run;
* ``mcp.py`` -- serves an ``Env`` over MCP (stdio, or a Unix socket for a sandbox);
* ``llm.py`` -- runs a session on a backend (``claude -p``, the Messages API with our own loop, Claude Code in a
  sandbox, an OpenAI-compatible model through ``loop.py``, a fake);
* ``episode.py`` -- one method on one case: plan, carry out, back to the first step after a failed run;
* ``methods/`` and ``prompts/bench/`` -- the rows; ``grade.py`` -- scoring; ``bench.py`` -- the command line;
  ``fakes.py`` -- a stand-in model and simulator for dry runs and tests.

"""
