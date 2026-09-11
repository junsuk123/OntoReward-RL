# AGILEX RANGER MINI 3.0 model

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
