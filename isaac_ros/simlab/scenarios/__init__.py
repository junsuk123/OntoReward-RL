"""Scenario definitions: flight environments, event timelines, occlusion truth.

Every module here is plain Python -- no Isaac Sim, no ROS 2 -- because both
sides of the system need the same answers. The simulator builds the stage and
gates the cameras from these tables; the dataset collector, running in a
different interpreter, reconstructs the identical geometry and event schedule
from the same episode YAML so its labels agree with what the camera saw.

Submodules are imported explicitly (``from simlab.scenarios.timeline import
...``) rather than re-exported here: ``simlab.config.schema`` imports
:mod:`simlab.scenarios.environments` for validation, and an eager re-export
would make that a cycle.
"""
