"""LandingObservation -> SemanticGraph, plus the GetGraphSchema service."""
from __future__ import annotations

from .common import (add_repo_paths, add_reset_service, msg_to_observation,
                     reliable_latest_qos, run, topic)


def main() -> None:
    add_repo_paths()
    from rclpy.node import Node
    from ontology_rgat_interfaces.msg import LandingObservation as ObservationMsg, SemanticGraph
    from ontology_rgat_interfaces.srv import GetGraphSchema
    from ontology_rgat.minimal import OBSERVATION_SCHEMA_ID, ONTOLOGY_SCHEMA_ID
    from ontology_rgat.minimal import ontology as onto

    class OntologyNode(Node):
        def __init__(self):
            super().__init__("ontology_node")
            ns = self.declare_parameter("namespace", "/landing_uav0").value
            self.ontology = onto.MinimalOntology()
            self.pub = self.create_publisher(
                SemanticGraph, topic(ns, "ontology/graph"), reliable_latest_qos())
            self.create_subscription(ObservationMsg, topic(ns, "observation"),
                                     self.on_observation, reliable_latest_qos())
            self.create_service(GetGraphSchema, topic(ns, "ontology/get_schema"),
                                self.on_schema)
            add_reset_service(self, ns, self.ontology.reset)
            self.get_logger().info(f"{ONTOLOGY_SCHEMA_ID} hash {self.ontology.hash[:12]}")

        def on_observation(self, msg):
            if msg.schema_id != OBSERVATION_SCHEMA_ID:
                self.get_logger().error(
                    f"observation schema {msg.schema_id!r} != {OBSERVATION_SCHEMA_ID!r}; dropped",
                    throttle_duration_sec=5.0)
                return
            graph = self.ontology.build(msg_to_observation(msg))
            out = SemanticGraph()
            out.header = msg.header
            out.schema_id, out.schema_hash = ONTOLOGY_SCHEMA_ID, graph.schema_hash
            out.num_nodes, out.feature_dim = graph.features.shape
            out.node_features = graph.features.reshape(-1).tolist()
            out.node_class = list(onto.NODE_CLASS_OF)
            out.edge_source = onto.EDGE_SOURCE.tolist()
            out.edge_target = onto.EDGE_TARGET.tolist()
            out.edge_relation = onto.EDGE_RELATION.tolist()
            out.edge_weight = graph.edge_weight.tolist()
            self.pub.publish(out)

        def on_schema(self, _request, response):
            response.schema_id = ONTOLOGY_SCHEMA_ID
            response.schema_hash = self.ontology.hash
            response.observation_schema_id = OBSERVATION_SCHEMA_ID
            response.node_names = list(onto.NODES)
            response.node_classes = [onto.NODE_CLASSES[i] for i in onto.NODE_CLASS_OF]
            response.relation_names = list(onto.RELATIONS)
            response.feature_channels = list(onto.FEATURE_CHANNELS)
            response.edge_source = onto.EDGE_SOURCE.tolist()
            response.edge_target = onto.EDGE_TARGET.tolist()
            response.edge_relation = onto.EDGE_RELATION.tolist()
            return response

    run(OntologyNode)


if __name__ == "__main__":
    main()
