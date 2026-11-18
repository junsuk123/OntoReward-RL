"""Paired Isaac comparison of the three navigation profiles on their common cases.

    python tools/isaac_navigation_compare.py [results/current_semantics_3d]

Reads the per-episode records of /3 (isaac-development/2), /4 Kalman (isaac-development/2)
and /4 WMON (isaac-development/1; same cases, same post-handover-fix stack) and prints success
and unsafe counts per arm and condition, plus paired discordance with an exact binomial p.
"""
import json, sys, math
from collections import defaultdict
RD = sys.argv[1] if len(sys.argv) > 1 else "results/current_semantics_3d"
SRC = {
    "/3 truth (dev/2)": f"{RD}/spatial_current_semantics_v1/isaac_development_2/episodes.jsonl",
    "/4 Kalman (dev/2)": f"{RD}/spatial_current_semantics_v1_nav4_489abc2d/isaac_development_2/episodes.jsonl",
    "/4 WMON (dev/1)": f"{RD}/spatial_current_semantics_v1_nav4_5aa4091b/isaac_development/episodes.jsonl",
}
UNSAFE = {"UNSAFE_CONTACT", "SAFETY_ENVELOPE_VIOLATION"}
data = {}
for name, path in SRC.items():
    rec = {}
    for line in open(path):
        r = json.loads(line)
        rec[(r["condition"], r["seed"], r["model"])] = r["status"]
    data[name] = rec
    print(name, len(rec))
common = set.intersection(*(set(v) for v in data.values()))
print("common cases", len(common))
arms = sorted({k[2].split("/")[1] for k in common})
conds = ["composition_id", "composition_ood", "combined_id"]
print("\nsuccess on common cases (count / n)")
hdr = f"{'arm':18s}" + "".join(f"{n:>22s}" for n in SRC)
print(hdr)
for arm in arms + ["ALL"]:
    keys = [k for k in common if arm == "ALL" or k[2].endswith("/" + arm)]
    row = f"{arm:18s}"
    for n in SRC:
        s = sum(data[n][k] == "SUCCESS" for k in keys)
        u = sum(data[n][k] in UNSAFE for k in keys)
        row += f"{s:>8d}/{len(keys):<4d}u{u:<3d}     "
    print(row)
print("\nby condition (ALL arms)")
for c in conds:
    keys = [k for k in common if k[0] == c]
    print(f"{c:18s}" + "".join(f"{sum(data[n][k]=='SUCCESS' for k in keys):>8d}/{len(keys):<10d}" for n in SRC))
print("\nby condition x arm")
for c in conds:
    for arm in arms:
        keys = [k for k in common if k[0] == c and k[2].endswith("/" + arm)]
        print(f"{c:16s}{arm:18s}" + "".join(f"{sum(data[n][k]=='SUCCESS' for k in keys):>5d}/{len(keys):<5d}" for n in SRC))
# paired /4 Kalman vs /3 and /4 WMON vs /4 Kalman (McNemar discordant counts)
names = list(SRC)
def disc(a, b, keys):
    ab = sum(data[a][k] == "SUCCESS" and data[b][k] != "SUCCESS" for k in keys)
    ba = sum(data[a][k] != "SUCCESS" and data[b][k] == "SUCCESS" for k in keys)
    n = ab + ba
    # exact two-sided binomial p
    p = sum(math.comb(n, i) for i in range(0, min(ab, ba) + 1)) / 2 ** n * 2 if n else 1.0
    return ab, ba, min(p, 1.0)
print("\npaired discordance (a only / b only, exact p)")
for a, b in [(names[1], names[0]), (names[2], names[0]), (names[1], names[2])]:
    print(f"{a} vs {b}: ALL {disc(a, b, common)}")
    for arm in arms:
        keys = [k for k in common if k[2].endswith('/' + arm)]
        print(f"   {arm:18s}{disc(a, b, keys)}")
# per training seed success for rgat
print("\nrgat per seed")
for sd in range(10, 15):
    keys = [k for k in common if k[2] == f"{sd}/ontology_rgat_ppo"]
    print(sd, [sum(data[n][k] == 'SUCCESS' for k in keys) for n in SRC], len(keys))
