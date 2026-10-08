"""SemanticGraph -> R-GAT -> requested ENU acceleration.

At startup the schema is fetched once from the ontology node and checked
against the policy's own hash; every graph message is checked again. On any
mismatch no command is published. ``relational_health`` is published with
every command so an inactive relational path is visible during a flight.
"""
from __future__ import annotations

import numpy as np

from .common import add_repo_paths, reliable_latest_qos, run, topic


def main() -> None:
    add_repo_paths()
    from rclpy.node import Node
    from geometry_msgs.msg import Vector3Stamped
    from std_msgs.msg import Float32
    from ontology_rgat_interfaces.msg import SemanticGraph
    from ontology_rgat_interfaces.srv import GetGraphSchema
    from ontology_rgat.minimal import ontology as onto
    from ontology_rgat.minimal.graph_policy import MinimalGraphPolicy

    def load_graph_policy(path):
        """A bare policy checkpoint, or a ppo_ontology_rgat arm checkpoint."""
        import torch
        payload = torch.load(path, map_location="cpu", weights_only=False)
        if "arm" not in payload:
            return MinimalGraphPolicy.load(path)
        from ontology_rgat.minimal.arms import load_arm
        name, arm, _ = load_arm(path)
        if name != "ppo_ontology_rgat":
            raise ValueError(f"{path} is a {name} checkpoint, not the R-GAT arm")
        return arm.policy

    class RgatPolicyNode(Node):
        def __init__(self):
            super().__init__("rgat_policy_node")
            ns = self.declare_parameter("namespace", "/landing_uav0").value
            checkpoint = self.declare_parameter("checkpoint", "").value
            self.deterministic = bool(self.declare_parameter("deterministic", True).value)
            if checkpoint:
                self.policy = load_graph_policy(checkpoint)
                self.get_logger().info(f"loaded {checkpoint}")
            else:
                self.policy = MinimalGraphPolicy()
                self.get_logger().warn("no checkpoint: UNTRAINED policy (pipeline test only)")
            self.policy.eval()
            self.schema_ok = False
            self.cmd_pub = self.create_publisher(
                Vector3Stamped, topic(ns, "policy/acceleration_command"), reliable_latest_qos())
            self.health_pub = self.create_publisher(
                Float32, topic(ns, "policy/relational_health"), reliable_latest_qos())
            self.create_subscription(SemanticGraph, topic(ns, "ontology/graph"),
                                     self.on_graph, reliable_latest_qos())
            self.client = self.create_client(GetGraphSchema, topic(ns, "ontology/get_schema"))
            self.create_timer(1.0, self.request_schema)

        def request_schema(self):
            if self.schema_ok or self.client.service_is_ready() is False:
                return
            self.client.call_async(GetGraphSchema.Request()).add_done_callback(self.on_schema)

        def on_schema(self, future):
            schema = future.result()
            matches = (schema.schema_hash == self.policy.schema_hash
                       and list(schema.edge_source) == onto.EDGE_SOURCE.tolist()
                       and list(schema.edge_target) == onto.EDGE_TARGET.tolist()
                       and list(schema.edge_relation) == onto.EDGE_RELATION.tolist())
            if matches and not self.schema_ok:
                self.get_logger().info(f"schema {schema.schema_id} verified")
            elif not matches:
                self.get_logger().error(
                    f"schema {schema.schema_hash[:12]} != policy {self.policy.schema_hash[:12]}")
            self.schema_ok = matches

        def on_graph(self, msg):
            if not self.schema_ok or msg.schema_hash != self.policy.schema_hash:
                return
            features = np.asarray(msg.node_features, dtype=np.float32).reshape(
                msg.num_nodes, msg.feature_dim)
            weights = np.asarray(msg.edge_weight, dtype=np.float32)
            action = self.policy.act(features, weights, deterministic=self.deterministic)
            cmd = Vector3Stamped()
            cmd.header = msg.header          # decision time of the source observation
            cmd.header.frame_id = "map"
            cmd.vector.x, cmd.vector.y, cmd.vector.z = map(float, action)
            self.cmd_pub.publish(cmd)
            self.health_pub.publish(Float32(
                data=self.policy.relational_activity(features, weights)))

    run(RgatPolicyNode)


if __name__ == "__main__":
    main()
