#!/usr/bin/env python3
"""Cache a real city block from OpenStreetMap, for isaac_sim/osm_city.py.

    scripts/fetch_city.py --name seoul-jongno \
        --latitude 37.5720 --longitude 126.9794 --radius 320

Writes ``assets/city/<name>.json``: every building footprint within RADIUS of
the origin, as a lat/lon ring plus whatever height the mapper recorded. The
simulator reads that file, never the network, so a run is reproducible from the
extract rather than from whatever Overpass returned that morning -- and the
extract is small enough to commit alongside the results it produced.

Pick a spot with a real street canyon: the point of the urban GNSS model is
buildings tall enough and close enough to take satellites away, so a low-rise
suburb makes a control condition rather than an experiment.

OpenStreetMap data is © OpenStreetMap contributors, licensed ODbL 1.0. Keep the
attribution the extract carries if you publish anything rendered from it.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

WORKSPACE = Path(__file__).resolve().parents[1]
ENDPOINTS = ("https://overpass-api.de/api/interpreter",
             "https://overpass.kumi.systems/api/interpreter")
ATTRIBUTION = "© OpenStreetMap contributors, ODbL 1.0"

QUERY = """
[out:json][timeout:{timeout}];
(
  way["building"](around:{radius},{lat},{lon});
  relation["building"]["type"="multipolygon"](around:{radius},{lat},{lon});
);
out body geom;
"""


def _parse_height(tags: dict) -> tuple[float | None, float | None]:
    """Metres and storeys, from the several tags mappers actually use."""
    height = None
    raw = tags.get("height") or tags.get("building:height")
    if raw:
        # "18", "18 m", "18.5m" -- and occasionally feet, which is not worth
        # guessing at, so anything unparseable falls through to the levels.
        text = str(raw).lower().replace("m", "").strip()
        try:
            value = float(text)
            height = value if value > 0.0 else None
        except ValueError:
            height = None
    levels = None
    raw_levels = tags.get("building:levels") or tags.get("levels")
    if raw_levels:
        try:
            value = float(str(raw_levels).split(";")[0])
            levels = value if value > 0.0 else None
        except ValueError:
            levels = None
    return height, levels


def fetch(latitude: float, longitude: float, radius: float,
          timeout: int = 90) -> dict:
    body = QUERY.format(radius=int(radius), lat=latitude, lon=longitude,
                        timeout=timeout)
    payload = urllib.parse.urlencode({"data": body}).encode()
    last: Exception | None = None
    for endpoint in ENDPOINTS:
        for attempt in range(3):
            try:
                request = urllib.request.Request(
                    endpoint, data=payload,
                    headers={"User-Agent": "ontology-rgat-uav/1.0 (research)"})
                with urllib.request.urlopen(request, timeout=timeout + 30) as response:
                    return json.loads(response.read().decode())
            except (urllib.error.URLError, TimeoutError, ValueError) as exc:
                last = exc
                # Overpass rate-limits by IP; backing off is expected use, not
                # an error worth failing the whole fetch over.
                time.sleep(4.0 * (attempt + 1))
        print(f"  {endpoint} did not answer; trying the next mirror.", file=sys.stderr)
    raise SystemExit(f"Could not reach any Overpass mirror: {last}")


def to_extract(answer: dict, name: str, latitude: float, longitude: float,
               altitude: float, radius: float) -> dict:
    buildings = []
    for element in answer.get("elements", []):
        geometry = element.get("geometry")
        if element.get("type") == "relation":
            # Take each outer way of a multipolygon as its own block: the
            # occlusion model is boxes, so an outline with holes buys nothing.
            members = [m for m in element.get("members", [])
                       if m.get("role") == "outer" and m.get("geometry")]
            rings = [m["geometry"] for m in members]
        else:
            rings = [geometry] if geometry else []
        height, levels = _parse_height(element.get("tags") or {})
        for ring in rings:
            points = [[float(node["lat"]), float(node["lon"])] for node in ring]
            if len(points) < 3:
                continue
            entry: dict = {"ring": points}
            if height is not None:
                entry["height_m"] = height
            if levels is not None:
                entry["levels"] = levels
            buildings.append(entry)
    return {
        "name": name,
        "attribution": ATTRIBUTION,
        "fetched_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "origin": {"latitude": latitude, "longitude": longitude,
                   "altitude": altitude},
        "radius_m": radius,
        "buildings": buildings,
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--name", required=True,
                        help="extract name; becomes assets/city/<name>.json")
    parser.add_argument("--latitude", type=float, required=True)
    parser.add_argument("--longitude", type=float, required=True)
    parser.add_argument("--altitude", type=float, default=0.0,
                        help="ground height above the WGS84 ellipsoid, metres")
    parser.add_argument("--radius", type=float, default=320.0,
                        help="metres around the origin to fetch")
    parser.add_argument("--out", default=None, help="override the output path")
    args = parser.parse_args()

    print(f"Fetching buildings within {args.radius:.0f} m of "
          f"{args.latitude:.5f}, {args.longitude:.5f} ...")
    answer = fetch(args.latitude, args.longitude, args.radius)
    extract = to_extract(answer, args.name, args.latitude, args.longitude,
                         args.altitude, args.radius)
    path = Path(args.out) if args.out else WORKSPACE / "assets" / "city" / f"{args.name}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as stream:
        json.dump(extract, stream)
    tagged = sum(1 for b in extract["buildings"] if "height_m" in b or "levels" in b)
    print(f"{len(extract['buildings'])} footprints ({tagged} with a mapped height) "
          f"-> {path.relative_to(WORKSPACE)}")
    if not extract["buildings"]:
        print("Nothing here. Check the coordinates, or widen --radius.",
              file=sys.stderr)
        return 1
    print("Point config/system.yaml at it:\n"
          f"  urban.source: osm\n  urban.extract: {args.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
