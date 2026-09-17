"""Vision encoders for benchmark actor observations."""

from .image_encoder import grayscale_image_tensor
from .keypoint_encoder import KeypointEncoderOutput, ShinKeypointEncoder
from .keypoint_pretrain import (EMPIRICAL_CALIBRATION_FORMAT, PRETRAIN_FORMAT,
                                calibrate_keypoint_encoder,
                                calibration_viewpoints,
                                empirical_keypoint_dataset,
                                needs_empirical_calibration,
                                prepare_keypoint_encoder,
                                synthetic_keypoint_dataset)
from .pad_geometry import (KEYPOINT_LAYOUT_ID, LANDING_PAD_VISUAL_VERSION,
                          PAD_LANDMARK_COUNT, PAD_LANDMARK_RADIUS_M,
                          CameraModel, PadKeypointProjection,
                          geometric_pad_center_in_fov, landing_pad_texture,
                          pad_landmarks, project_landing_pad)
from .ros_camera import (LatestGrayscaleFrame, LatestPadRelativeTruthPose,
                         RosGrayscaleSource)
from .semantic_observation import (
    FORBIDDEN_SEMANTIC_FIELDS, POINT_CONFIDENCE_THRESHOLD,
    SEMANTIC_FEATURE_NAMES,
    SEMANTIC_GRAPH_INPUT_DIM, SEMANTIC_GRAPH_VERSION, SEMANTIC_NODE_NAMES,
    SEMANTIC_RELATION_NAMES, SemanticObservation,
    assert_semantic_payload_safe, semantic_graph, semantic_observation,
    semantic_observation_from_payload)

__all__ = ["grayscale_image_tensor", "KeypointEncoderOutput", "ShinKeypointEncoder",
           "PRETRAIN_FORMAT", "EMPIRICAL_CALIBRATION_FORMAT",
           "prepare_keypoint_encoder", "calibration_viewpoints",
           "needs_empirical_calibration",
           "calibrate_keypoint_encoder", "empirical_keypoint_dataset",
           "synthetic_keypoint_dataset", "LatestGrayscaleFrame",
           "LatestPadRelativeTruthPose", "RosGrayscaleSource",
           "KEYPOINT_LAYOUT_ID", "LANDING_PAD_VISUAL_VERSION",
           "PAD_LANDMARK_COUNT", "PAD_LANDMARK_RADIUS_M", "CameraModel",
           "PadKeypointProjection", "geometric_pad_center_in_fov",
           "landing_pad_texture", "pad_landmarks", "project_landing_pad",
           "FORBIDDEN_SEMANTIC_FIELDS", "POINT_CONFIDENCE_THRESHOLD",
           "SEMANTIC_FEATURE_NAMES", "SEMANTIC_GRAPH_INPUT_DIM",
           "SEMANTIC_GRAPH_VERSION", "SEMANTIC_NODE_NAMES",
           "SEMANTIC_RELATION_NAMES", "SemanticObservation",
           "assert_semantic_payload_safe", "semantic_graph",
           "semantic_observation", "semantic_observation_from_payload"]
