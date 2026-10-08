import math

import numpy as np
import pytest

from ontology_rgat.direct_policy.graph import planar_graph, shuffled_relations, spatial_graph
from ontology_rgat.direct_policy.observation import (
    MarkerFrame, Navigation2D, PlanarCalibration, PlanarEstimate,
    PlanarObservationAssembler, SpatialEstimate,
    planar_vector, spatial_to_planar_named, spatial_vector,
)


def test_planar_registered_order_and_source_scaling():
    state = PlanarEstimate((7, 0), (1.5, 0), (5, 4.6), (0.5, -0.3),
                           0.1, 0.2, True, 0.2, True, 0.05,
                           pad_offset_z_m=0.6)
    value = planar_vector(state)
    expected = np.array([
        2/5, 4/12, 1/11, 1.5/11.5, -0.3/1.8,
        math.sin(0.1), math.cos(0.1), 0.2/(0.2+math.pi/2),
        1, 0.2/3.2, 1, 0.05/3.05,
    ])
    np.testing.assert_allclose(value, expected, atol=1e-12, rtol=1e-12)


def test_uninitialised_estimator_zeros_only_pad_dependent_fields():
    state = PlanarEstimate((99, 99), (99, 99), (5, 4.6), (0.5, -0.3),
                           0.1, 0.2, False, math.inf, True, 0,
                           estimate_initialized=False)
    value = planar_vector(state)
    np.testing.assert_array_equal(value[:4], 0)
    assert value[4] != 0 and value[9] == 1 and value[10] == 1


def test_planar_graph_has_pinned_nodes_relations_edges_and_column_flattening():
    graph = planar_graph(np.linspace(-0.5, 0.5, 12))
    assert graph.features.shape == (7, 6)
    assert len(graph.schema.nodes) == 7
    assert len(graph.schema.relations) == 4
    assert len(graph.schema.edges) == 16       # nine semantic + seven self
    np.testing.assert_array_equal(graph.matlab_flatten(), graph.features.reshape(-1))
    assert tuple(graph.schema.readout_groups) == tuple((i,) for i in range(7))


def test_spatial_named_projection_is_the_planar_limit():
    planar = PlanarEstimate((7, 0), (1.5, 0), (5, 4.6), (0.5, -0.3),
                            0.1, 0.2, True, 0.2, True, 0.05, pad_offset_z_m=0.6)
    spatial = SpatialEstimate((7, 0, 0.6), (1.5, 0, 0), (5, 0, 4.6),
                              (0.5, 0, -0.3), (0, 0.1, 0), (0, 0.2, 0),
                              True, 0.2, True, 0.05)
    np.testing.assert_allclose(spatial_to_planar_named(spatial_vector(spatial)),
                               planar_vector(planar), atol=1e-12)
    assert spatial_graph(spatial_vector(spatial)).features.shape == (7, 8)


def test_relation_shuffle_preserves_counts_and_self_but_is_not_global_rename():
    graph = planar_graph(np.zeros(12))
    shuffled = shuffled_relations(graph.schema, 7)
    self_id = graph.schema.relations.index("self")
    np.testing.assert_array_equal(np.sort(shuffled), np.sort(graph.relation))
    np.testing.assert_array_equal(shuffled[graph.relation == self_id], self_id)
    assert not np.array_equal(shuffled, graph.relation)


def test_assembler_rejects_future_navigation_and_episode_leakage():
    assembler = PlanarObservationAssembler()
    assembler.reset("episode-a")
    nav = Navigation2D(2.0, (0, 2), (0, 0), 0, 0)
    with pytest.raises(ValueError, match="newer"):
        assembler.assemble(None, nav, nav, 1.0, episode_id="episode-a")
    with pytest.raises(ValueError, match="episode"):
        assembler.assemble(None, nav, nav, 2.0, episode_id="episode-b")


def test_pnp_kf_path_uses_registered_corners_and_capture_navigation():
    cv2 = pytest.importorskip("cv2")
    calibration = PlanarCalibration.source_default()
    rotation = np.diag([1.0, -1.0, -1.0])
    rvec, _ = cv2.Rodrigues(rotation)
    translation = np.array([[0.2], [0.1], [3.0]])
    corners = {}
    for marker_id, (cx, cy) in calibration.marker_centers_xy.items():
        half = calibration.marker_sides_m[marker_id]/2
        objects = np.array([(cx-half, cy+half, 0), (cx+half, cy+half, 0),
                            (cx+half, cy-half, 0), (cx-half, cy-half, 0)], float)
        pixels, _ = cv2.projectPoints(objects, rvec, translation,
                                      calibration.camera_matrix, calibration.distortion)
        corners[marker_id] = pixels.reshape(4, 2)
    frame = MarkerFrame(1.0, 1.05, (512, 320), corners)
    nav = Navigation2D(1.0, (0, 4), (1, 0), 0, 0)
    assembler = PlanarObservationAssembler(calibration); assembler.reset("e")
    vector, provenance = assembler.assemble(frame, nav, nav, 1.05, episode_id="e")
    assert provenance["frame_used"] and provenance["estimate_initialized"]
    assert np.isfinite(vector).all() and vector[8] == 1
