#!/usr/bin/env python3
"""Convert the pinned official Ranger Mini V3 URDF/DAE package to USD.

Run through Isaac Sim's ``python.sh``; normal Python does not ship the URDF
importer.  Mesh URLs are expanded to absolute temporary paths before import so
the result is independent of ROS_PACKAGE_PATH.
"""

from __future__ import annotations

import argparse
import shutil
import tempfile
import xml.etree.ElementTree as ET
import zipfile
from pathlib import Path

from isaacsim import SimulationApp


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


ARGS = arguments()
APP = SimulationApp({"headless": True})

import omni.kit.commands  # noqa: E402
from pxr import Usd, UsdGeom  # noqa: E402


def prepared_urdf(source: Path, temporary: Path) -> Path:
    package = temporary / "ranger_mini_v3"
    shutil.copytree(source, package)
    archive = package / "meshes" / "ranger_base.zip"
    with zipfile.ZipFile(archive) as bundle:
        bundle.extractall(package / "meshes")

    tree = ET.parse(package / "urdf" / "ranger_mini.xacro")
    prefix = "package://ranger_mini_v3/"
    for mesh in tree.getroot().iter("mesh"):
        filename = mesh.get("filename", "")
        if filename.startswith(prefix):
            mesh.set("filename", str((package / filename[len(prefix):]).resolve()))
    destination = package / "urdf" / "ranger_mini.urdf"
    tree.write(destination, encoding="utf-8", xml_declaration=True)
    return destination


def main() -> None:
    source = ARGS.source.expanduser().resolve()
    output = ARGS.output.expanduser().resolve()
    required = source / "urdf" / "ranger_mini.xacro"
    if not required.is_file():
        raise FileNotFoundError(f"official Ranger Mini source is incomplete: {required}")
    output.parent.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(prefix="ranger-mini-v3-") as temporary:
        urdf = prepared_urdf(source, Path(temporary))
        status, config = omni.kit.commands.execute("URDFCreateImportConfig")
        if not status:
            raise RuntimeError("Isaac URDF importer did not create its configuration")
        config.merge_fixed_joints = False
        config.convex_decomp = False
        config.import_inertia_tensor = True
        config.fix_base = True
        config.distance_scale = 1.0
        status, prim_path = omni.kit.commands.execute(
            "URDFParseAndImportFile",
            urdf_path=str(urdf),
            import_config=config,
            dest_path=str(output),
            get_articulation_root=True,
        )
        if not status or not output.is_file():
            raise RuntimeError(f"URDF import failed (status={status}, prim={prim_path!r})")

    stage = Usd.Stage.Open(str(output))
    if stage is None:
        raise RuntimeError(f"generated USD cannot be reopened: {output}")
    if not stage.GetDefaultPrim():
        children = list(stage.GetPseudoRoot().GetChildren())
        if not children:
            raise RuntimeError("generated USD has no root prim")
        stage.SetDefaultPrim(children[0])
        stage.GetRootLayer().Save()
    bounds = UsdGeom.BBoxCache(
        Usd.TimeCode.Default(), [UsdGeom.Tokens.default_, UsdGeom.Tokens.render]
    ).ComputeWorldBound(stage.GetDefaultPrim()).ComputeAlignedRange()
    size = tuple(float(value) for value in bounds.GetSize())
    # Catch the common millimetre/metre importer error before a 720 m vehicle
    # is committed or loaded into the campus scene.
    if max(size) > 2.0 or min(size) < 0.05:
        raise RuntimeError(f"implausible imported Ranger bounds (metres): {size}")
    print(f"Generated official RANGER MINI 3.0 USD: {output}")
    print("Visual bounds (m): " + " x ".join(f"{value:.3f}" for value in size))


try:
    main()
finally:
    APP.close()
