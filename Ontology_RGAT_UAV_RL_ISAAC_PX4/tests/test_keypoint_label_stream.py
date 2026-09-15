"""The training-only pose stream that labels rendered keypoint frames.

Isaac publishes the rendered frame and its pad-relative truth pose from the
same tick with the same stamp, so a frame is labelled with *its own* pose or
not at all. A neighbouring tick would be a silent label error: at 2 m/s one
30 Hz frame is 6.6 cm of motion, several pixels near touchdown.
"""
from __future__ import annotations

import threading
import time
from types import SimpleNamespace

import numpy as np
import pytest

from ontology_rgat.perception.ros_camera import LatestPadRelativeTruthPose


def _pose_message(stamp_ns: int, position=(1.0, -2.0, 3.0),
                  orientation=(1.0, 0.0, 0.0, 0.0)):
    return SimpleNamespace(
        header=SimpleNamespace(stamp=SimpleNamespace(
            sec=stamp_ns // 1_000_000_000, nanosec=stamp_ns % 1_000_000_000)),
        pose=SimpleNamespace(
            position=SimpleNamespace(
                x=position[0], y=position[1], z=position[2]),
            orientation=SimpleNamespace(
                w=orientation[0], x=orientation[1],
                y=orientation[2], z=orientation[3])))


def test_a_frame_is_matched_to_its_own_stamp_only():
    poses = LatestPadRelativeTruthPose()
    for index in range(3):
        poses.callback(_pose_message(
            1_000_000_000 + index * 33_000_000, position=(index, 0.0, 5.0)))

    matched = poses.at(1_000_000_000 + 33_000_000)
    assert matched is not None
    np.testing.assert_allclose(matched[:3], [1.0, 0.0, 5.0])
    np.testing.assert_allclose(matched[3:], [1.0, 0.0, 0.0, 0.0])

    # One frame period away is a different frame, not a near-enough match.
    assert poses.at(1_000_000_000 + 33_000_000 + 1) is None
    assert poses.at(1_000_000_000 + 16_000_000) is None


def test_a_late_pose_is_waited_for_rather_than_approximated():
    poses = LatestPadRelativeTruthPose()
    poses.callback(_pose_message(1_000_000_000, position=(9.0, 9.0, 9.0)))
    stamp = 1_033_000_000

    def publish_late():
        time.sleep(0.05)
        poses.callback(_pose_message(stamp, position=(4.0, 5.0, 6.0)))

    thread = threading.Thread(target=publish_late)
    thread.start()
    try:
        matched = poses.wait_for(stamp, timeout_s=2.0)
    finally:
        thread.join()

    assert matched is not None
    np.testing.assert_allclose(matched[:3], [4.0, 5.0, 6.0])


def test_a_pose_that_never_arrives_drops_the_frame():
    poses = LatestPadRelativeTruthPose()
    poses.callback(_pose_message(1_000_000_000))

    started = time.monotonic()
    assert poses.wait_for(2_000_000_000, timeout_s=0.05) is None
    assert time.monotonic() - started < 1.0


def test_a_non_finite_pose_is_discarded_instead_of_labelling_a_frame():
    poses = LatestPadRelativeTruthPose()
    poses.callback(_pose_message(1_000_000_000, position=(float("nan"), 0.0, 0.0)))

    assert poses.at(1_000_000_000) is None


def test_the_camera_source_refuses_labels_without_a_truth_pose_topic():
    from ontology_rgat.perception.ros_camera import RosGrayscaleSource

    source = RosGrayscaleSource.__new__(RosGrayscaleSource)
    source.truth_poses = None
    with pytest.raises(RuntimeError, match="geometric keypoint labels"):
        source.labelled()
