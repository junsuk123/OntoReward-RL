"""Axis-generic landing-task contract shared by the 2D and 3D routes.

The 3D route began as a copy of the 2D one, so every concern they share drifted
independently: four terminal-reward tables, two 9x12 graph builders, two
curriculum ramps. This package holds the single definition of each, with the
number of horizontal axes as a parameter -- 2D is one, 3D is two.

Nothing here reads simulator truth, actions, outcomes or labels.
"""
