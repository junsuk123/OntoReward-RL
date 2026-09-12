# AGILEX RANGER MINI 3.0 model

[Active project README](../../README.md) ·
[Primary S5 route audit](../../docs/images/metasejong_gwanggaeto_ugv_route.png)

The files under `source/` are the `ranger_mini_v3` package copied from
AGILEX Robotics' public [`ugv_gazebo_sim`](https://github.com/agilexrobotics/ugv_gazebo_sim)
repository at commit `27633a956c845903ee630538afeb17fe70afdd84`.
The upstream `package.xml` declares the package license as BSD. The files are
kept unchanged; `ranger_base.zip` is the upstream archive of
`ranger_base.dae`.

Generate the Isaac Sim USD from these pinned sources with:

```bash
./scripts/import_ranger_mini_v3.sh
```

The simulator references `ranger_mini_v3.usd` as a visual-only child of the
kinematic landing platform. If that generated asset is absent, it falls back
to audited primitives with the same external dimensions and URDF wheel
locations, so a missing visual asset cannot break the experiment.

In the primary `config/shin2026-system.yaml` profile, this visual carries a
1.5×1.5 m landing deck on the Meta-Sejong S5/Gwanggaeto 37-point road loop.
Trajectory dynamics remain deterministic code in `isaac_sim/pad_motion.py`;
the imported URDF/USD is appearance and geometry provenance, not a learned or
wheel-physics controller. The configured carrier dimensions are
0.720×0.500×0.345 m, mass 75 kg, payload rating 120 kg, and speed limit
1.0 m/s.
