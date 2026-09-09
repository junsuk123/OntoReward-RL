"""Isaac Sim bindings.

Importing anything below this package (other than :mod:`simlab.sim.app` and
:mod:`simlab.sim.assets`) pulls in ``omni``/``isaacsim`` at module level, which
only works once a ``SimulationApp`` is alive. This ``__init__`` deliberately
re-exports nothing so that ``import simlab.sim.app`` stays cheap.
"""
