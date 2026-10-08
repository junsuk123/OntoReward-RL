"""Thin ROS 2 wrappers around ``ontology_rgat.minimal``.

Every node here only converts messages and calls the pure functions the local
training uses, so training and deployment cannot drift apart.
"""
