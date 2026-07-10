#!/usr/bin/env python3
"""Annotate retrievals.csv with the source city/hex file for each H3 cell.

Scans GeoAI/data/hex/hex/*_hexagons_res{7,9}.geojson, finds every h3_id mentioned,
then maps the H3 cells in leon-clip/outputs/retrieval_examples/retrievals.csv to the
city file they live in. Writes two outputs:

- h3_to_city.csv      (h3, resolution, city, hex_file)
- retrievals_with_city.csv  (retrievals.csv + columns query_city, retrieved_city,
                             query_ancestor_city)
"""
from __future__ import annotations

import csv
import json
import re
from collections import defaultdict
from pathlib import Path

HEX_DIR = Path("/home/szymon/repo/geo/GeoAI/data/hex/hex")
RETRIEVALS_CSV = Path("/home/szymon/repo/geo/leon-clip/outputs/retrieval_examples/retrievals.csv")
OUT_DIR = RETRIEVALS_CSV.parent
H3_TO_CITY = OUT_DIR / "h3_to_city.csv"
RETRIEVALS_OUT = OUT_DIR / "retrievals_with_city.csv"

CITY_FILE_RE = re.compile(r"^(?P<city>.+)_hexagons_res(?P<res>\d+)\.geojson$")
H3_ID_RE = re.compile(r'"h3_id"\s*:\s*"([0-9a-fA-F]+)"')


def collect_target_ids() -> tuple[set[str], set[str]]:
    """Return (res9_ids, res7_ids) — ids we need to locate."""
    res9: set[str] = set()
    res7: set[str] = set()
    with RETRIEVALS_CSV.open("r", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            for column in ("query_h3", "retrieved_h3"):
                value = row.get(column, "").strip()
                if value:
                    res9.add(value)
            ancestor = row.get("query_ancestor_res7", "").strip()
            if ancestor:
                res7.add(ancestor)
    return res9, res7


def scan_geojson(path: Path, wanted: set[str]) -> set[str]:
    """Return the subset of `wanted` whose id appears in this geojson file."""
    found: set[str] = set()
    if not wanted:
        return found
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            match = H3_ID_RE.search(line)
            if match is None:
                continue
            h3_id = match.group(1)
            if h3_id in wanted:
                found.add(h3_id)
    return found


def main() -> int:
    res9_ids, res7_ids = collect_target_ids()
    print(f"[targets] {len(res9_ids)} unique res-9 ids, {len(res7_ids)} unique res-7 ancestors")

    files = sorted(HEX_DIR.iterdir())
    h3_to_city: dict[str, tuple[str, str, int]] = {}  # h3 -> (city, filename, resolution)
    per_city_counts: dict[str, int] = defaultdict(int)

    for path in files:
        m = CITY_FILE_RE.match(path.name)
        if m is None:
            continue
        resolution = int(m.group("res"))
        if resolution not in (7, 9):
            continue
        city = m.group("city")
        wanted = res9_ids if resolution == 9 else res7_ids
        # Skip cells we already located in a previous file.
        wanted = wanted - {h for h, (_, _, r) in h3_to_city.items() if r == resolution}
        if not wanted:
            continue
        found = scan_geojson(path, wanted)
        if not found:
            continue
        for h3_id in found:
            h3_to_city[h3_id] = (city, path.name, resolution)
            per_city_counts[city] += 1
        print(f"  [{path.name}] +{len(found)} matches (city total {per_city_counts[city]})")

    missing_res9 = res9_ids - {h for h, (_, _, r) in h3_to_city.items() if r == 9}
    missing_res7 = res7_ids - {h for h, (_, _, r) in h3_to_city.items() if r == 7}
    print(f"[done] mapped {len(h3_to_city)} ids; missing res9={len(missing_res9)} res7={len(missing_res7)}")

    with H3_TO_CITY.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["h3", "resolution", "city", "hex_file"])
        for h3_id, (city, filename, resolution) in sorted(h3_to_city.items()):
            writer.writerow([h3_id, resolution, city, filename])
    print(f"[write] {H3_TO_CITY}")

    with RETRIEVALS_CSV.open("r", encoding="utf-8") as src, \
            RETRIEVALS_OUT.open("w", newline="", encoding="utf-8") as dst:
        reader = csv.DictReader(src)
        fieldnames = list(reader.fieldnames or []) + [
            "query_city", "retrieved_city", "query_ancestor_city", "same_city",
        ]
        writer = csv.DictWriter(dst, fieldnames=fieldnames)
        writer.writeheader()
        for row in reader:
            qc = h3_to_city.get(row.get("query_h3", ""))
            rc = h3_to_city.get(row.get("retrieved_h3", ""))
            ac = h3_to_city.get(row.get("query_ancestor_res7", ""))
            row["query_city"] = qc[0] if qc else ""
            row["retrieved_city"] = rc[0] if rc else ""
            row["query_ancestor_city"] = ac[0] if ac else ""
            row["same_city"] = "1" if (qc and rc and qc[0] == rc[0]) else "0"
            writer.writerow(row)
    print(f"[write] {RETRIEVALS_OUT}")

    if missing_res9:
        print(f"[warn] {len(missing_res9)} res-9 ids not found (sample: {sorted(missing_res9)[:5]})")
    if missing_res7:
        print(f"[warn] {len(missing_res7)} res-7 ids not found (sample: {sorted(missing_res7)[:5]})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
