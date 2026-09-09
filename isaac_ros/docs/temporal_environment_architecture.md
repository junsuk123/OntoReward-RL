# Accelerated temporal environment architecture

`EnvironmentTimeManager` owns virtual days, months, years, and normalized time. It
never changes Isaac Sim's physics timestep. `TemporalEventScheduler` supplies a
seeded, replayable event history. `EnvironmentStateFactory` combines virtual time,
events, season, weather, and optional semantic-case overrides into explicit component
change scores. The global score is the configured weighted sum of those components.
Vegetation, building, road, season, weather, and lighting changes are implemented as
independent manager classes, allowing one dimension to be varied without mutating the
others.

The fast OpenCV backend mirrors the same stable semantic objects as the USD adapter:
terrain, a road and markings, persistent/developing buildings, and individually
identified trees. It is used for high-throughput experiments. The Isaac Sim 5.1
adapter authors those objects below `/World/Environment`, updates only mutable prim
attributes between epochs, captures the fixed nadir-camera trajectory through
Replicator, and exports each epoch's USD layer.

Year0 images remain immutable. Every later query is globally retrieved against the
whole Year0 trajectory using ORB and verified by homography RANSAC. Ground-truth
environment changes and camera poses are evaluation metadata, never retrieval input.

The default vehicle is a procedural fixed-wing USD aircraft carrying a nadir camera.
It follows a tangent-aligned Archimedean spiral over a 300 m square map. Radius,
turns, samples, direction, altitude, resolution, and video frame rate are controlled
by YAML. The same metric poses and headings are replayed at every epoch.
