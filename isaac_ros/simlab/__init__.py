"""Isaac Sim scene toolkit: UGV + simulated people.

Import order matters. ``simlab.config``, ``simlab.algorithms`` and
``simlab.utils`` are plain Python and can be imported at any time. Everything
under ``simlab.sim`` touches ``omni``/``isaacsim`` and may only be imported
*after* a ``SimulationApp`` exists -- see :mod:`simlab.cli`.
"""

__version__ = "0.1.0"
