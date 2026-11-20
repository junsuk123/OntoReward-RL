"""Display names of the self-navigation estimators are separate from their internal ids.

``wmon`` is the author's own estimator from separate research; documents and figures show
it as the FFT-assisted kinematic WLS estimator, while configs, results, checkpoints and
hashes keep the internal id.  The mapping must never enter a manifest.
"""
from pathlib import Path

import pytest

from ontology_rgat.current_semantics import navigation as nav
from ontology_rgat.current_semantics.contracts import contract_manifest, load_config

PROJECT = Path(__file__).resolve().parents[2]


def test_every_estimator_has_a_display_name_and_wmon_is_not_a_kalman():
    for kind in nav.ESTIMATOR_KINDS:
        assert nav.estimator_display_name(kind) and nav.estimator_display_name(kind, short=True)
    assert nav.estimator_display_name("wmon") == \
        "FFT-Assisted Kinematic Weighted Least Squares (WLS) Estimator"
    assert nav.estimator_display_name("wmon", short=True) == "Kinematic WLS"
    assert "Kalman" not in nav.estimator_display_name("wmon")
    assert nav.estimator_display_name("kalman") == "Constant-Acceleration Kalman Filter (CA-KF)"
    assert "separate research" in nav.ESTIMATOR_PROVENANCE["wmon"]
    with pytest.raises(ValueError):
        nav.estimator_display_name("ekf")


def test_internal_ids_and_manifests_are_untouched():
    config = load_config(PROJECT / "config" / "current_semantics"
                         / "spatial_view_contact_v1_nav4_wmon.yaml")
    part = config.navigation.manifest_part()
    assert part["estimator"]["kind"] == "wmon"
    text = str(contract_manifest(config))
    for name in nav.ESTIMATOR_DISPLAY_NAMES.values():
        assert name not in text
