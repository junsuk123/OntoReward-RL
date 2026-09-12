#!/usr/bin/env python3
"""Audit a configured Meta-Sejong waypoint route against the USD road mesh.

This is intentionally an offline tool: it reads USD geometry directly and
does not start Isaac Sim, PX4, ROS, or the competition Docker container.
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import matplotlib.tri as mtri
import numpy as np
from scipy.ndimage import distance_transform_edt


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "isaac_sim"))

from config_loader import load_config  # noqa: E402
from metasejong_scene import MetaSejongConfig  # noqa: E402
from pad_motion import PadMotionConfig  # noqa: E402


def _triangles(face_counts, face_indices) -> np.ndarray:
    """Triangulate USD polygon faces using a fan (roads are convex quads/tris)."""
    triangles: list[tuple[int, int, int]] = []
    offset = 0
    for count in face_counts:
        face = face_indices[offset:offset + count]
        triangles.extend((face[0], face[i], face[i + 1])
                         for i in range(1, count - 1))
        offset += count
    return np.asarray(triangles, dtype=np.int32)


def read_road_meshes(usd_path: Path, scale: float):
    try:
        from pxr import Gf, Usd, UsdGeom
    except ImportError as exc:  # pragma: no cover - installation-dependent
        raise RuntimeError("USD Python bindings are required (pip install usd-core)") from exc

    stage = Usd.Stage.Open(str(usd_path))
    if stage is None:
        raise RuntimeError(f"could not open USD stage: {usd_path}")
    cache = UsdGeom.XformCache()
    meshes = []
    for prim in stage.Traverse():
        if not prim.IsA(UsdGeom.Mesh) or "/CarRoad/" not in str(prim.GetPath()):
            continue
        mesh = UsdGeom.Mesh(prim)
        matrix = cache.GetLocalToWorldTransform(prim)
        points = np.asarray([
            tuple(matrix.Transform(Gf.Vec3d(*point)))
            for point in mesh.GetPointsAttr().Get()
        ], dtype=float) * float(scale)
        faces = _triangles(list(mesh.GetFaceVertexCountsAttr().Get()),
                           list(mesh.GetFaceVertexIndicesAttr().Get()))
        triangulation = mtri.Triangulation(points[:, 0], points[:, 1], faces)
        meshes.append((str(prim.GetPath()), points, faces, triangulation))
    if not meshes:
        raise RuntimeError(f"no /CarRoad/ meshes found in {usd_path}")
    return meshes


def sample_polyline(points: np.ndarray, spacing: float) -> np.ndarray:
    result = [points[0]]
    for start, end in zip(points[:-1], points[1:]):
        length = float(np.linalg.norm(end - start))
        count = max(1, int(math.ceil(length / spacing)))
        result.extend(start + (end - start) * (index / count)
                      for index in range(1, count + 1))
    return np.asarray(result)


def audit(config_path: Path, resolution: float, plot_path: Path | None) -> int:
    config = load_config(config_path)
    scene = MetaSejongConfig.from_mapping(config, ROOT)
    pad = PadMotionConfig.from_mapping(config)
    scene.validate_assets()
    if pad.mode != "waypoints":
        raise ValueError("route audit requires pad.motion: waypoints")

    meshes = read_road_meshes(scene.usd_path, scene.scale)
    route = np.asarray(pad.route_waypoints_enu_m, dtype=float)
    samples = sample_polyline(route, min(0.05, 0.5 * resolution))

    all_points = np.concatenate([item[1] for item in meshes], axis=0)
    padding = 2.0
    xmin, ymin = np.min(all_points[:, :2], axis=0) - padding
    xmax, ymax = np.max(all_points[:, :2], axis=0) + padding
    xs = np.arange(xmin, xmax + resolution, resolution)
    ys = np.arange(ymin, ymax + resolution, resolution)
    xx, yy = np.meshgrid(xs, ys)
    inside = np.zeros(xx.shape, dtype=bool)
    finders = []
    interpolators = []
    for _, points, _, triangulation in meshes:
        finder = triangulation.get_trifinder()
        inside |= finder(xx, yy) >= 0
        finders.append(finder)
        interpolators.append(mtri.LinearTriInterpolator(triangulation, points[:, 2]))

    route_inside = np.zeros(len(samples), dtype=bool)
    for finder in finders:
        route_inside |= finder(samples[:, 0], samples[:, 1]) >= 0

    clearance = distance_transform_edt(inside, sampling=resolution)
    col = np.clip(np.rint((samples[:, 0] - xmin) / resolution).astype(int),
                  0, len(xs) - 1)
    row = np.clip(np.rint((samples[:, 1] - ymin) / resolution).astype(int),
                  0, len(ys) - 1)
    sampled_clearance = clearance[row, col]
    sampled_clearance[~route_inside] = 0.0

    elevation_errors = []
    for waypoint in route:
        candidates = []
        for interpolator in interpolators:
            value = interpolator(waypoint[0], waypoint[1])
            if not np.ma.is_masked(value):
                candidates.append(float(value))
        elevation_errors.append(
            min((abs(value - waypoint[2]) for value in candidates), default=math.inf)
        )

    route_length = float(np.linalg.norm(np.diff(route, axis=0), axis=1).sum())
    min_clearance = float(np.min(sampled_clearance))
    swept_radius = 0.5 * math.hypot(*pad.deck_size_m)
    margin = min_clearance - swept_radius
    max_z_error = float(max(elevation_errors))
    triangle_count = sum(len(item[2]) for item in meshes)

    print(f"map: {scene.usd_path}")
    print(f"road meshes / triangles: {len(meshes)} / {triangle_count}")
    print(f"route waypoints / length: {len(route)} / {route_length:.2f} m")
    print(f"minimum pavement-edge clearance: {min_clearance:.2f} m")
    print(f"UGV swept half-diagonal: {swept_radius:.2f} m")
    print(f"minimum conservative clearance margin: {margin:.2f} m")
    print(f"maximum waypoint elevation error: {max_z_error:.3f} m")

    passed = bool(np.all(route_inside) and margin > 0.0 and max_z_error <= 0.08)
    print("result: PASS" if passed else "result: FAIL")

    if plot_path is not None:
        import matplotlib.pyplot as plt

        plot_path.parent.mkdir(parents=True, exist_ok=True)
        figure, axis = plt.subplots(figsize=(8.0, 8.0), constrained_layout=True)
        axis.imshow(inside, origin="lower", extent=(xmin, xmax, ymin, ymax),
                    cmap="Greys", alpha=0.65)
        axis.plot(route[:, 0], route[:, 1], color="#e63946", linewidth=2.2,
                  label="UGV waypoint route")
        axis.scatter(*route[0, :2], color="#2a9d8f", s=65, zorder=3,
                     label="start / finish" if pad.waypoint_loop else "start")
        if not pad.waypoint_loop:
            axis.scatter(*route[-1, :2], color="#457b9d", s=65, zorder=3,
                         label="reverse point")
        scenario = str(config.get("metasejong", {}).get("scenario", "campus"))
        scenario_label = {"demo": "S1", "dongcheon": "S3",
                          "jiphyeon": "S4", "gwanggaeto": "S5"}.get(
                              scenario, scenario)
        axis.set(title=(f"Meta-Sejong {scenario_label} ({scenario}) road and "
                        "audited UGV route"),
                 xlabel="world X (m)", ylabel="world Y (m)", aspect="equal")
        axis.legend(loc="best")
        figure.savefig(plot_path, dpi=180)
        plt.close(figure)
        print(f"plot: {plot_path}")
    return 0 if passed else 1


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path,
                        default=ROOT / "config" / "metasejong-demo.yaml")
    parser.add_argument("--resolution", type=float, default=0.10,
                        help="road-clearance raster cell size in metres")
    parser.add_argument("--plot", type=Path,
                        help="optional output PNG showing the road and route")
    args = parser.parse_args()
    if args.resolution <= 0.0:
        parser.error("--resolution must be positive")
    return audit(args.config.resolve(), args.resolution,
                 args.plot.resolve() if args.plot else None)


if __name__ == "__main__":
    raise SystemExit(main())
