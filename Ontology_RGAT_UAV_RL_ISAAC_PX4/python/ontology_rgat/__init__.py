"""Ontology-guided R-GAT reward shaping for UAV landing on a moving deck.

The learner: everything that used to live in MATLAB. Isaac Sim owns the
physics, PX4 owns the estimator and the attitude loop, the ROS 2 gateway owns
the wire format, and this package owns the experiment -- the episode, the
ontology, the R-GAT potential, PPO, the evaluation and the figures.

Entry points are the scripts one directory up: ``run_pipeline.py``,
``run_episode.py``, ``run_hardware_policy.py`` and ``run_benchmark.py``.
"""
from __future__ import annotations

__all__ = ["__version__"]

__version__ = "2.0.0"
