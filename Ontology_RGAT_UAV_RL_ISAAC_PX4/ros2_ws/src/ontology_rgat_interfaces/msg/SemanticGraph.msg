# One ABox instance of the landing ontology, built from one LandingObservation.
#
# header.stamp copies the source observation's decision time, so a graph can
# always be traced to the o_t it was built from. Names of nodes, classes,
# relations and feature channels are NOT repeated per message: fetch them once
# with GetGraphSchema and check schema_hash on every message.
std_msgs/Header header
string schema_id                    # "minimal-landing-ontology/6"
string schema_hash                  # sha256 of the schema; a subscriber rejects a mismatch

uint16 num_nodes
uint16 feature_dim
float32[] node_features             # row-major, num_nodes x feature_dim, already normalized
uint16[] node_class                 # class id per node (index into schema node_classes)

# Typed directed edges, self loops included. The edge SET is the schema's static
# superset; a conditional relation is expressed by edge_weight in [0, 1]
# (e.g. Camera -observes-> Pad is weighted by detection freshness), never by
# adding or removing edges.
uint16[] edge_source
uint16[] edge_target
uint8[] edge_relation               # relation id (index into schema relation_names)
float32[] edge_weight
