"""Optional Meta-Sejong campus USD scene adapter.

The competition assets are proprietary and are deliberately not vendored by
this repository.  ``scripts/import_metasejong_map.sh`` extracts a user's
already-installed Docker image into the ignored external asset directory.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any


SCENARIOS = {
    "demo": {
        "usd": "playground/S1/SejongUniv_S1.usd",
        "eye": (-61.53516, 139.77144, 25.0),
        "target": (-65.0475, 130.98898, 16.24493),
        # The official S1 start is enclosed by decorative trees.  They are
        # useful for the competition scene, but hide both robots in a landing
        # experiment and account for hundreds of invalid collision meshes.
        "vegetation": ("S1/_3_Prop/Tree",),
    },
    "dongcheon": {
        "usd": "playground/S3/SejongUniv_S3.usd",
        "eye": (-34.49608, 25.24217, 40.0),
        "target": (-43.07861, 0.26965, 15.93874),
    },
    "jiphyeon": {
        "usd": "playground/S4/SejongUniv_S4.usd",
        "eye": (6.78023, -225.27218, 30.0287378),
        "target": (16.00401, -219.77422, 21.10599),
    },
    "gwanggaeto": {
        "usd": "playground/S5/SejongUniv_S5.usd",
        "eye": (-21.52438, -99.7126, 25.0),
        "target": (-42.10069, -76.20936, 21.7482),
    },
}

ALIASES = {"jipyhyeon": "jiphyeon", "jipyeon": "jiphyeon"}


@dataclass(frozen=True)
class MetaSejongConfig:
    enabled: bool
    scenario: str
    asset_root: Path
    scale: float
    prim_path: str
    hide_vegetation: bool

    @classmethod
    def from_mapping(cls, data: dict[str, Any], workspace: Path) -> "MetaSejongConfig":
        raw = data.get("metasejong", {}) or {}
        scenario = str(raw.get("scenario", "demo")).strip().lower()
        scenario = ALIASES.get(scenario, scenario)
        if scenario not in SCENARIOS:
            choices = ", ".join(SCENARIOS)
            raise ValueError(f"metasejong.scenario must be one of: {choices}")

        default_root = str(
            raw.get("asset_root", "external/metasejong/resources/models")
        )
        configured_root = os.environ.get("METASEJONG_ASSET_ROOT", default_root)
        asset_root = Path(configured_root).expanduser()
        if not asset_root.is_absolute():
            asset_root = (workspace / asset_root).resolve()
        scale = float(raw.get("scale", 0.01))
        if scale <= 0.0:
            raise ValueError("metasejong.scale must be positive")
        prim_path = str(raw.get("prim_path", "/World/metasejong"))
        if not prim_path.startswith("/"):
            raise ValueError("metasejong.prim_path must be an absolute USD prim path")
        return cls(
            bool(raw.get("enabled", False)), scenario, asset_root, scale, prim_path,
            bool(raw.get("hide_vegetation", False)),
        )

    @property
    def scenario_data(self) -> dict[str, object]:
        return SCENARIOS[self.scenario]

    @property
    def usd_path(self) -> Path:
        return self.asset_root / str(self.scenario_data["usd"])

    def validate_assets(self) -> None:
        if self.enabled and not self.usd_path.is_file():
            raise FileNotFoundError(
                f"Meta-Sejong {self.scenario!r} map is missing: {self.usd_path}. "
                "Run scripts/import_metasejong_map.sh or set METASEJONG_ASSET_ROOT."
            )


class MetaSejongScene:
    """Reference a competition map without importing Isaac during unit tests."""

    def __init__(self, config: MetaSejongConfig):
        self.config = config

    def spawn(self, world) -> None:
        if not self.config.enabled:
            return
        self.config.validate_assets()

        from isaacsim.core.utils.viewports import set_camera_view
        from pxr import Gf, UsdGeom

        stage = world.stage
        root = UsdGeom.Xform.Define(stage, self.config.prim_path)
        root.GetPrim().GetReferences().AddReference(str(self.config.usd_path))
        UsdGeom.XformCommonAPI(root).SetScale(Gf.Vec3f(*(self.config.scale,) * 3))
        scenario = self.config.scenario_data
        hidden = 0
        if self.config.hide_vegetation:
            # SetActive authors an override in this transient stage; it never
            # edits the licensed source USD.  Deactivation removes rendering,
            # shadows and collision APIs together, unlike visibility alone.
            for relative_path in scenario.get("vegetation", ()):
                path = f"{self.config.prim_path}/{relative_path}"
                prim = stage.GetPrimAtPath(path)
                if prim.IsValid():
                    prim.SetActive(False)
                    hidden += 1
        set_camera_view(eye=scenario["eye"], target=scenario["target"])
        print(
            f"[metasejong] loaded {self.config.scenario}: {self.config.usd_path} "
            f"(scale={self.config.scale:g}, vegetation_groups_hidden={hidden})",
            flush=True,
        )
