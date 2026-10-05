import importlib.util
from pathlib import Path

import numpy as np

from ontology_rgat.spatial.core import Measurement


def test_fault_hides_optical_input_only_after_declared_time():
    spec=importlib.util.spec_from_file_location('spatial_fault_probe',
        Path(__file__).resolve().parents[1]/'tools/check_spatial_landing.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    m=Measurement(1.,np.zeros(3),np.ones(3),np.array([1.,0.,0.,0.]),
                  np.zeros(3),np.ones(3),.9,10,.9,np.zeros(3),np.array([1.,0.,0.,0.]))
    assert module.diagnostic_optical_fault(m,1.,None) is m
    assert module.diagnostic_optical_fault(m,1.,2.) is m
    hidden=module.diagnostic_optical_fault(m,2.,2.)
    assert hidden.optical_position is None and hidden.confidence==0.
    assert m.optical_position is not None and m.confidence==.9
    assert hidden.own_velocity is m.own_velocity and hidden.own_position is m.own_position
    assert hidden.quaternion is m.quaternion and hidden.sample_id==m.sample_id
