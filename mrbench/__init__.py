"""MultiRobotBench: heterogeneous multi-arm manipulation tasks in Isaac Sim.

Importing this package never imports Isaac Sim.  Everything under
``mrbench.engines.isaac`` imports ``isaacsim`` / ``isaaclab`` lazily, after the
app has been launched, so task cards, scoring and the mock engine work on a
machine without a GPU.
"""

__version__ = "0.0.1"
