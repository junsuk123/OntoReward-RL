"""ROS 2 side of the pipeline.

These modules run under the *system* ROS 2 Python (Humble / 3.10), not Isaac
Sim's interpreter. They import :mod:`simlab.algorithms` and
:mod:`simlab.config`, both of which are plain Python, and never touch
``simlab.sim``.
"""
