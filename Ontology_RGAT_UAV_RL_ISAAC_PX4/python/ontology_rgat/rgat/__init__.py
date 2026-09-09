"""The relational graph attention potential.

See ``layers.py`` for what this takes from Busbridge et al. 2019 and the
reference release at https://github.com/babylonhealth/rgat, and NOTICE for the
attribution.
"""
from __future__ import annotations

from .layers import RGAT, RelationalGraphAttention
from .model import RGATPotential, build_potential, load_potential, save_potential
from .topology import Topology

__all__ = ["RGAT", "RelationalGraphAttention", "RGATPotential", "Topology",
           "build_potential", "load_potential", "save_potential"]
