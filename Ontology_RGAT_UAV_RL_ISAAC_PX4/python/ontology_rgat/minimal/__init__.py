"""Minimal-observation landing contract: pad position + own EKF state.

Design: ``docs/MINIMAL_OBSERVATION_ROS_PIPELINE_KO.md``. Every module here is
pure Python (no rclpy), so local training and the ROS nodes in
``ros2_ws/src/ontology_rgat_landing`` call the same functions.

``constants``    TBox constants and pad-loss thresholds
``observation``  o_t (13) and the assembler that builds it from detections + EKF
``pad_loss``     PadMemory and the ONE terminal-vs-lost classifier
``ontology``     the 12-node, 6-relation schema and the graph builder
``supervisor``   safety laws S1-S6 on the minimal observation only
``graph_policy`` R-GAT actor/critic over the graph
"""
OBSERVATION_SCHEMA_ID = "minimal-landing-obs/2"
ONTOLOGY_SCHEMA_ID = "minimal-landing-ontology/6"
