"""Learned temporal relative-state estimation."""

# Export the leaf auxiliary module before importing the legacy wrapper.  The
# wrapper depends on ppo.temporal_backbone, while ppo.recurrent consumes this
# auxiliary head; this order keeps direct ``ontology_rgat.estimation`` imports
# free of a package-initialisation cycle.
from .relative_state_aux import RelativeStateAuxiliaryHead
from .lstm_relative_state import EstimatorOutput, LSTMRelativeStateEstimator

__all__ = ["EstimatorOutput", "LSTMRelativeStateEstimator",
           "RelativeStateAuxiliaryHead"]
