#!/usr/bin/env bash
set -euo pipefail

workspace_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
image=${METASEJONG_IMAGE:-metasejong:metacom-2025-with-playground-r06}
destination=${METASEJONG_IMPORT_DIR:-$workspace_root/external/metasejong/resources/models/playground}

if [[ ${1:-} == "--help" ]]; then
  printf '%s\n' \
    "Usage: $0" \
    "Extracts the licensed Meta-Sejong playground from an installed Docker image." \
    "Overrides: METASEJONG_IMAGE=$image METASEJONG_IMPORT_DIR=$destination"
  exit 0
fi
if ! command -v docker >/dev/null 2>&1 || ! docker info >/dev/null 2>&1; then
  printf 'Docker is unavailable or its daemon is not running.\n' >&2
  exit 1
fi
if ! docker image inspect "$image" >/dev/null 2>&1; then
  printf 'Meta-Sejong image is not installed: %s\n' "$image" >&2
  exit 1
fi

mkdir -p "$destination"
container=$(docker create --entrypoint /bin/true "$image")
cleanup() { docker rm -f "$container" >/dev/null 2>&1 || true; }
trap cleanup EXIT
docker cp "$container:/metacom2025/resources/models/playground/." "$destination"

for map in \
  S1/SejongUniv_S1.usd \
  S3/SejongUniv_S3.usd \
  S4/SejongUniv_S4.usd \
  S5/SejongUniv_S5.usd; do
  if [[ ! -f "$destination/$map" ]]; then
    printf 'Extracted map is incomplete; missing %s\n' "$map" >&2
    exit 1
  fi
done
printf 'Meta-Sejong playground extracted to %s\n' "$destination"
