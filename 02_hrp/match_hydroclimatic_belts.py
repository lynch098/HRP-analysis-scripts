
"""Assign each basin to its largest-area hydroclimatic-belt intersection in EPSG:6933."""

import json as _path_json
from pathlib import Path as _ConfigPath


_DATA_PATHS = _path_json.loads(
    (_ConfigPath(__file__).resolve().parents[1] / "paths.json").read_text(encoding="utf-8")
)
from pathlib import Path
import argparse
import json

import geopandas as gpd
import pandas as pd
import shapely

RUN_DIR = Path(_DATA_PATHS["HRP_RUN"])
INPUT_CSV = RUN_DIR / "main/static_hrp.csv"
BELT_SHP = Path(_DATA_PATHS["BELT_SHP"])
OUT_DIR = RUN_DIR / "hydroclimatic_belts"
AREA_CRS = "EPSG:6933"
EXPECTED_COUNTS = {"major": 80, "sub": 295}

BELTS = {0: "ICE", 1: "BOR", 2: "NML", 3: "NDR", 4: "NST",
         5: "EQT", 6: "SST", 7: "SDR", 8: "SML"}

MIN_COVERAGE = 0.95
CLOSE_MARGIN = 0.05


def polygon_only(g):
    if g is None or g.is_empty:
        return None
    g = shapely.make_valid(g) if not g.is_valid else g
    if g.geom_type in ("Polygon", "MultiPolygon"):
        return g
    if g.geom_type == "GeometryCollection":
        parts = [polygon_only(p) for p in g.geoms]
        parts = [p for p in parts if p is not None and not p.is_empty]
        return shapely.union_all(parts) if parts else None
    return None


def clean_project(gdf):
    if gdf.crs is None:
        raise ValueError("Missing CRS; provide a verified CRS before matching.")
    repaired = int((~gdf.geometry.is_valid & gdf.geometry.notna()).sum())
    gdf = gdf.copy()
    gdf.geometry = gdf.geometry.map(polygon_only)
    gdf = gdf[gdf.geometry.notna() & ~gdf.geometry.is_empty].copy()
    if gdf.empty:
        raise ValueError("No polygonal geometry after repair.")
    gdf = gdf.to_crs(AREA_CRS)
    repaired += int((~gdf.geometry.is_valid).sum())
    gdf.geometry = gdf.geometry.map(polygon_only)
    if gdf.geometry.isna().any() or gdf.geometry.is_empty.any():
        raise ValueError("Invalid polygon after projection.")
    return gdf, repaired


def match_polygon(g, belt_geoms):
    area = float(g.area)
    if not (0 < area < float("inf")):
        raise ValueError("Non-positive or non-finite basin area.")
    intersections = {code: g.intersection(bg) for code, bg in belt_geoms.items()
                     if g.intersects(bg)}
    areas = {code: float(part.area) for code, part in intersections.items() if part.area > 0}
    if not areas:
        return dict(belt=None, hydrobelt_code=None, basin_area_km2=area / 1e6,
                    coverage_pct=0.0, status="REVIEW", flags="NO_OVERLAP"), []
    ranked = sorted(areas, key=lambda code: (-areas[code], code))
    first = ranked[0]
    second = ranked[1] if len(ranked) > 1 else None
    covered = float(shapely.union_all(list(intersections.values())).area)
    coverage = covered / area
    margin = (areas[first] - (areas[second] if second is not None else 0)) / area
    flags = []
    if coverage < MIN_COVERAGE:
        flags.append("LOW_COVERAGE")
    if areas[first] / area < 0.5:
        flags.append("NO_ABSOLUTE_MAJORITY")
    if second is not None and margin < CLOSE_MARGIN:
        flags.append("CLOSE_TOP_TWO")
    if sum(areas.values()) - covered > area * 1e-6:
        flags.append("BELT_POLYGONS_OVERLAP")
    tied = second is not None and abs(areas[first] - areas[second]) <= area * 1e-9
    if tied:
        flags.append("TIED_MAXIMUM")
    row = dict(belt=None if tied else BELTS[first], hydrobelt_code=None if tied else first,
               basin_area_km2=area / 1e6, coverage_pct=100 * coverage,
               dominant_area_pct=100 * areas[first] / area,
               second_belt=BELTS[second] if second is not None else None,
               second_area_pct=100 * areas[second] / area if second is not None else 0.0,
               top_two_margin_pp=100 * margin, n_intersecting_belts=len(areas),
               status="REVIEW" if flags else "OK", flags=";".join(flags))
    details = [dict(belt=BELTS[code], hydrobelt_code=code,
                    overlap_area_km2=areas[code] / 1e6, basin_area_pct=100 * areas[code] / area)
               for code in ranked]
    return row, details


def main():
    units = pd.read_csv(INPUT_CSV, dtype={"unit_key": str})
    required = {"unit_key", "level", "basin_shp", "HRP_type"}
    if not required.issubset(units.columns):
        raise ValueError(f"Missing columns: {required - set(units.columns)}")
    if units.unit_key.isna().any() or units.unit_key.duplicated().any():
        raise ValueError("Missing or duplicate unit_key.")
    if units.level.value_counts().to_dict() != EXPECTED_COUNTS:
        raise ValueError(f"Expected {EXPECTED_COUNTS}; got {units.level.value_counts().to_dict()}")
    for fp in units.basin_shp:
        if not Path(fp).is_file():
            raise FileNotFoundError(fp)
    belts = gpd.read_file(BELT_SHP)
    if "hydrobelt" not in belts or belts.hydrobelt.isna().any():
        raise ValueError("Missing hydrobelt codes.")
    if not belts.hydrobelt.isin(BELTS).all():
        raise ValueError(f"Unknown hydrobelt codes: {belts.hydrobelt.unique()}")
    belts, belt_repairs = clean_project(belts)
    belt_geoms = {int(code): polygon_only(shapely.union_all(group.geometry.tolist()))
                  for code, group in belts.groupby("hydrobelt")}
    rows, overlaps = [], []
    for i, r in enumerate(units.itertuples(index=False), 1):
        print(f"[{i}/{len(units)}] {r.unit_key}", flush=True)
        basin, repairs = clean_project(gpd.read_file(r.basin_shp))
        geom = polygon_only(shapely.union_all(basin.geometry.tolist()))
        row, detail = match_polygon(geom, belt_geoms)
        row.update(unit_key=r.unit_key, level=r.level, HRP_type=r.HRP_type,
                   basin_shp=r.basin_shp, geometry_repairs=repairs)
        rows.append(row)
        overlaps.extend(dict(unit_key=r.unit_key, level=r.level, **d) for d in detail)
    mapping = pd.DataFrame(rows)
    parents = mapping.loc[mapping.level == "major"].set_index("unit_key").belt
    mapping["parent_major"] = mapping.unit_key.str.split("__").str[0]
    mapping["parent_belt"] = mapping.parent_major.map(parents)
    mapping["differs_from_parent"] = (
        (mapping.level == "sub") & mapping.belt.notna() & mapping.parent_belt.notna()
        & mapping.belt.ne(mapping.parent_belt))
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    mapping.to_csv(OUT_DIR / "basin_belts.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(overlaps).to_csv(OUT_DIR / "belt_overlap_areas.csv", index=False, encoding="utf-8-sig")
    mapping.loc[mapping.status == "REVIEW"].to_csv(OUT_DIR / "belt_review.csv", index=False, encoding="utf-8-sig")

    composition = (mapping.dropna(subset=["belt"])
                   .groupby(["level", "belt", "HRP_type"]).size().unstack(fill_value=0)
                   .reindex(columns=["Near-river", "Dispersion", "Far-river"], fill_value=0))
    composition["n"] = composition.sum(axis=1)
    for kind in ["Near-river", "Dispersion", "Far-river"]:
        composition[kind + "_pct"] = 100 * composition[kind] / composition["n"]
    composition.reset_index().to_csv(OUT_DIR / "fig1a_belt_composition.csv", index=False, encoding="utf-8-sig")
    config = dict(input_csv=str(INPUT_CSV), belt_shp=str(BELT_SHP), area_crs=AREA_CRS,
                  assignment="Largest intersection area per basin; exact ties left unassigned",
                  share_denominator="Complete basin polygon area", codes=BELTS,
                  expected_counts=EXPECTED_COUNTS, review_min_coverage=MIN_COVERAGE,
                  review_top_two_margin=CLOSE_MARGIN, belt_geometry_repairs=belt_repairs,
                  unassigned=int(mapping.belt.isna().sum()), review=int((mapping.status == "REVIEW").sum()))
    (OUT_DIR / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    print(f"Output: {OUT_DIR}\nAssigned: {mapping.belt.notna().sum()}/{len(mapping)}; review: {config['review']}")
    if mapping.belt.isna().any():
        raise RuntimeError("Unassigned basins: inspect belt_review.csv before downstream analysis.")


def self_test():
    from shapely.geometry import box, Polygon
    row, detail = match_polygon(box(0, 0, 10, 10), {1: box(0, 0, 7, 10), 2: box(7, 0, 10, 10)})
    assert row["belt"] == "BOR" and row["dominant_area_pct"] == 70
    assert row["coverage_pct"] == 100 and len(detail) == 2
    row, _ = match_polygon(box(0, 0, 10, 10), {1: box(0, 0, 5, 10), 2: box(5, 0, 10, 10)})
    assert row["belt"] is None and "TIED_MAXIMUM" in row["flags"]
    row, _ = match_polygon(box(0, 0, 10, 10), {1: box(0, 0, 4, 10)})
    assert "LOW_COVERAGE" in row["flags"]
    row, _ = match_polygon(box(0, 0, 10, 10), {1: box(20, 20, 30, 30)})
    assert row["flags"] == "NO_OVERLAP"
    fixed = polygon_only(Polygon([(0, 0), (2, 2), (0, 2), (2, 0), (0, 0)]))
    assert fixed.is_valid and fixed.area > 0
    print("Self-test passed; no study data processed.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    self_test() if args.self_test else main()
