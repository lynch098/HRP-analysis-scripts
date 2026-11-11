
"""Select tributary units using HydroBASINS Pfafstetter codes and spatial overlap.
Final acceptance is determined by the subsequent outlet-connectivity checks."""

import json as _path_json
from pathlib import Path as _ConfigPath


_DATA_PATHS = _path_json.loads(
    (_ConfigPath(__file__).resolve().parents[1] / "paths.json").read_text(encoding="utf-8")
)
from pathlib import Path
from functools import lru_cache
from datetime import datetime
import argparse
import json
import logging
import re
import shutil
import sys

import geopandas as gpd
import numpy as np
import pandas as pd
import pyogrio
import shapely
from pyproj import CRS
from shapely.geometry import box, LineString, MultiLineString, Polygon


HYDRO_DIR = Path(_DATA_PATHS["HYDROBASINS_DIR"])
MAJOR_MANIFEST = Path(
    _DATA_PATHS["MAJOR_MANIFEST"]
)
OUTPUT_BASE = Path(_DATA_PATHS["SELECTION_OUTPUT_ROOT"])
EXPECTED_MAJORS = 80


ROOT_MIN_COVERAGE = 0.95
ROOT_MIN_PURITY = 0.95
FALLBACK_NEST_MIN = 0.95
FALLBACK_MIN_IOU = 0.50

CHILD_MIN_IN_MAJOR = 0.95
AREA_REL_TOL = 0.05
OVERLAP_REL_TOL = 1e-6

RIVER_BOUNDARY_KM = 3.0
RIVER_BOUNDARY_FRAC = 0.80
RIVER_MIN_PIECE_KM = 1.0
RIVER_SNAP_KM = 0.1
RIVER_MIN_SPAN_FRAC = 0.50
RIVER_OUTLET_KM = 5.0

WRITE_SHAPEFILES = True
AREA_CRS = "EPSG:6933"
REGIONS = ("af", "ar", "as", "au", "eu", "na", "sa", "si")
EVEN_DIGITS = {2, 4, 6, 8}
ATTRS = ["HYBAS_ID", "NEXT_DOWN", "NEXT_SINK", "MAIN_BAS", "PFAF_ID",
         "SUB_AREA", "UP_AREA", "ENDO", "COAST", "ORDER"]
INTEGER_ATTRS = [c for c in ATTRS if c not in {"SUB_AREA", "UP_AREA"}]
LOG = logging.getLogger("selection")
DOC_URL = "https://data.hydrosheds.org/file/technical-documentation/HydroBASINS_TechDoc_v1c.pdf"
WGS84 = CRS.from_epsg(4326)


def clean_polygon(g):
    if g is None or g.is_empty:
        raise ValueError("Empty geometry")
    g = shapely.make_valid(g) if not g.is_valid else g
    if g.geom_type in {"Polygon", "MultiPolygon"}:
        return g
    parts = [clean_polygon(p) for p in g.geoms
             if p.geom_type in {"Polygon", "MultiPolygon", "GeometryCollection"}
             and not p.is_empty]
    if not parts:
        raise ValueError("No polygon after geometry repair")
    return shapely.union_all(parts)


def line_parts(g):

    if g is None or g.is_empty:
        return []
    g = shapely.force_2d(g)
    if g.geom_type == "LineString":
        return [g] if g.length > 0 else []
    if hasattr(g, "geoms"):
        out = []
        for p in g.geoms:
            out.extend(line_parts(p))
        return out
    return []


def overlap_metrics(a, b):
    inter = a.intersection(b).area
    return dict(coverage=inter / a.area, purity=inter / b.area,
                iou=inter / (a.area + b.area - inter))


def local_aeqd(geom_wgs84):
    p = geom_wgs84.representative_point()
    return CRS.from_proj4(f"+proj=aeqd +lat_0={p.y} +lon_0={p.x} "
                          "+datum=WGS84 +units=m +no_defs")


def long_side_km(geom):

    if geom is None or geom.is_empty:
        return 0.0
    rect = shapely.minimum_rotated_rectangle(geom)
    if rect.geom_type == "Polygon":
        xy = np.asarray(rect.exterior.coords)
        return float(np.hypot(*np.diff(xy, axis=0).T).max()) / 1000
    return rect.length / 1000


def discover(root):

    catalog, inventory = {}, []
    pattern = re.compile(r"hybas_([a-z]{2})_lev(\d{2})_v1c\.shp", re.I)
    for f in sorted(Path(root).rglob("*.shp")):
        m = pattern.fullmatch(f.name)
        if not m or m[1].lower() not in REGIONS:
            continue
        key = (m[1].lower(), int(m[2]))
        if key in catalog:
            raise ValueError(f"Duplicate region/level: {catalog[key]} and {f}")
        for suffix in (".dbf", ".shx", ".prj"):
            if not f.with_suffix(suffix).exists():
                raise FileNotFoundError(f.with_suffix(suffix))
        info = pyogrio.read_info(f)
        if not info["crs"]:
            raise ValueError(f"Missing CRS: {f}")
        try:
            is_wgs84 = CRS.from_user_input(info["crs"]).equals(WGS84, ignore_axis_order=True)
        except Exception:
            is_wgs84 = False
        if not is_wgs84:
            raise ValueError(f"Expected WGS84 (EPSG:4326) HydroBASINS input: {f}")
        missing = set(ATTRS) - set(info["fields"])
        if missing:
            raise ValueError(f"Missing fields {missing}: {f}")
        if {"LAKE", "SIDE"} & set(info["fields"]):
            raise ValueError(f"Use the standard format without inserted lakes: {f}")
        catalog[key] = f
        inventory.append(dict(region=key[0], level=key[1], path=str(f),
                              features=info["features"], crs=str(info["crs"]),
                              shp_bytes=f.stat().st_size,
                              shp_mtime_ns=f.stat().st_mtime_ns))
    missing = [(r, l) for r in REGIONS for l in range(1, 13) if (r, l) not in catalog]
    if missing:
        raise FileNotFoundError(f"Missing standard region/level files: {missing}")
    return catalog, pd.DataFrame(inventory)


def region_bounds(catalog):
    out = {}
    for r in REGIONS:
        info = pyogrio.read_info(catalog[r, 1])
        b = info.get("total_bounds")
        if b is None or not np.all(np.isfinite(b)):
            b = gpd.read_file(catalog[r, 1], engine="pyogrio").total_bounds
        out[r] = tuple(float(x) for x in b)
    return out


@lru_cache(maxsize=None)
def attributes(path):
    t = pyogrio.read_dataframe(path, read_geometry=False, columns=ATTRS)
    for c in INTEGER_ATTRS:
        t[c] = pd.to_numeric(t[c], errors="raise").astype("int64")
    if t.HYBAS_ID.duplicated().any():
        raise ValueError(f"Duplicate HYBAS_ID in {path}")
    n_dup = int(t.PFAF_ID.duplicated(keep=False).sum())
    if n_dup:
        LOG.warning("WARNING: %s: %d units share a PFAF_ID; splits containing them go to REVIEW",
                    Path(path).name, n_dup)


    virtual = (t.ENDO == 2) & (t.NEXT_DOWN > 0)
    physical_incoming = t.loc[~virtual & (t.NEXT_DOWN > 0), "NEXT_DOWN"].value_counts()
    virtual_incoming = t.loc[virtual, "NEXT_DOWN"].value_counts()
    t["incoming_surface_units"] = t.HYBAS_ID.map(physical_incoming).fillna(0).astype(int)
    t["incoming_virtual_units"] = t.HYBAS_ID.map(virtual_incoming).fillna(0).astype(int)
    t["up_sub_area_rel_diff"] = (t.UP_AREA - t.SUB_AREA).abs() / t.SUB_AREA
    return t


def geometry_rows(path, ids, bbox=None):
    ids = sorted({int(x) for x in ids})
    if not ids:
        return gpd.GeoDataFrame(columns=ATTRS + ["geometry"], geometry="geometry", crs=4326)
    where = "HYBAS_ID IN (" + ",".join(map(str, ids)) + ")"
    g = pyogrio.read_dataframe(path, where=where, bbox=bbox, columns=ATTRS)
    if not g.crs:
        raise ValueError(f"Missing CRS: {path}")
    g = g.to_crs(4326)
    if not g.empty:
        g.geometry = g.geometry.map(clean_polygon)
    return g


def complete_unit(row):
    return (int(row.COAST) == 0 and int(row.incoming_surface_units) == 0
            and float(row.up_sub_area_rel_diff) <= AREA_REL_TOL)


def load_majors(path, selected_names):
    t = pd.read_csv(path)
    for c in ("basin_name", "basin_shp", "river_shp"):
        if c not in t:
            raise ValueError(f"Manifest requires column {c}: {path}")
        if t[c].isna().any():
            raise ValueError(f"Missing {c} in manifest")
    if len(t) != EXPECTED_MAJORS or t.basin_name.duplicated().any():
        raise ValueError(f"Expected {EXPECTED_MAJORS} unique majors, got {len(t)}")
    if selected_names:
        absent = set(selected_names) - set(t.basin_name)
        if absent:
            raise ValueError(f"Unknown basin names: {absent}")
        t = t[t.basin_name.isin(selected_names)].copy()
    rows = []
    for row in t.itertuples(index=False):
        p = Path(row.basin_shp)
        if not p.is_absolute():
            p = path.parent / p
        rp = Path(row.river_shp)
        if not rp.is_absolute():
            rp = path.parent / rp
        g = gpd.read_file(p, engine="pyogrio")
        if not g.crs:
            raise ValueError(f"Missing CRS: {p}")
        parts = [clean_polygon(x) for x in g.to_crs(4326).geometry
                 if x is not None and not x.is_empty]
        geom = clean_polygon(shapely.union_all(parts))
        ea = gpd.GeoSeries([geom], crs=4326).to_crs(AREA_CRS).iloc[0]
        if ea.area <= 0:
            raise ValueError(f"Non-positive major area: {p}")
        rows.append(dict(major=row.basin_name, major_source=str(p), river_shp=str(rp),
                         geometry=geom, ea=ea, area_km2=ea.area / 1e6))
    return rows, t


def choose_fallback(complete_candidates):

    nested = [c for c in complete_candidates
              if max(c["coverage"], c["purity"]) >= FALLBACK_NEST_MIN
              and c["iou"] >= FALLBACK_MIN_IOU]
    if not nested:
        return None
    best = max(nested, key=lambda c: (round(c["iou"], 6), -c["level"]))
    relation = "CONTAINS_MAJOR" if best["coverage"] >= FALLBACK_NEST_MIN else "WITHIN_MAJOR"
    return {**best, "root_tier": "FALLBACK", "root_relation": relation}


def match_root(major, catalog, bounds, candidate_log):

    a, ma = major["ea"], major["area_km2"]
    regions = [r for r in REGIONS if box(*bounds[r]).intersects(major["geometry"])]
    complete = []
    for level in range(3, 13):
        passing = []
        for region in regions:
            path = catalog[region, level]
            t = attributes(str(path))
            plausible = t[(t.SUB_AREA >= ma * 0.5) & (t.SUB_AREA <= ma * 2)]
            if plausible.empty:
                continue
            g = geometry_rows(path, plausible.HYBAS_ID, bbox=major["geometry"].bounds)
            if g.empty:
                continue
            projected = g.to_crs(AREA_CRS)
            by_id = t.set_index("HYBAS_ID")
            for idx, row in g.iterrows():
                geom_ea = clean_polygon(projected.loc[idx, "geometry"])
                scores = overlap_metrics(a, geom_ea)
                if scores["iou"] < 0.10:
                    continue
                meta = by_id.loc[int(row.HYBAS_ID)]
                good_shape = (scores["coverage"] >= ROOT_MIN_COVERAGE
                              and scores["purity"] >= ROOT_MIN_PURITY)
                good_topology = complete_unit(meta)
                record = dict(major=major["major"], region=region, level=level,
                              HYBAS_ID=int(row.HYBAS_ID), PFAF_ID=int(row.PFAF_ID),
                              root_area_km2=geom_ea.area / 1e6, **scores,
                              incoming_surface_units=int(meta.incoming_surface_units),
                              incoming_virtual_units=int(meta.incoming_virtual_units),
                              up_sub_area_rel_diff=float(meta.up_sub_area_rel_diff),
                              COAST=int(meta.COAST), ENDO=int(meta.ENDO),
                              complete_unit=good_topology,
                              passes=good_shape and good_topology, source=str(path))
                candidate_log.append(record)
                if good_topology:
                    item = {**record, "geometry": row.geometry, "ea": geom_ea}
                    complete.append(item)
                    if good_shape:
                        passing.append(item)
        if len(passing) == 1:
            return {**passing[0], "root_tier": "MATCHED",
                    "root_relation": "EQUIVALENT"}, "MATCHED"
        if len(passing) > 1:
            return None, "REVIEW_AMBIGUOUS_ROOT"
    fallback = choose_fallback(complete)
    if fallback is not None:
        return fallback, "MATCHED_FALLBACK"
    return None, "NO_ROOT"


def first_subdivision(root, catalog):

    prefix, skipped = int(root["PFAF_ID"]), []
    for level in range(root["level"] + 1, 13):
        path = catalog[root["region"], level]
        t = attributes(str(path))
        children = t[t.PFAF_ID // 10 == prefix].copy()
        if len(children) == 1 and int(children.iloc[0].PFAF_ID) == prefix * 10:
            skipped.append(level)
            prefix *= 10
            continue
        return children, level, prefix, skipped
    return pd.DataFrame(), 12, prefix, skipped


def child_topology_reasons(row, siblings):
    reasons = []
    if not complete_unit(row):
        reasons.append("NOT_COMPLETE_CATCHMENT")
    if int(row.ENDO) == 2 and int(row.NEXT_DOWN) > 0:
        reasons.append("VIRTUAL_OUTLET")
    downstream = siblings[siblings.HYBAS_ID == int(row.NEXT_DOWN)]
    if len(downstream) != 1:
        reasons.append("OUTLET_NOT_IN_PARENT_CHILDREN")
    elif int(downstream.iloc[0].PFAF_ID) % 10 not in {1, 3, 5, 7, 9}:
        reasons.append("OUTLET_NOT_ON_ODD_CODE_MAINSTEM")
    return reasons


def child_decision(digit, in_major, anomalies):

    if digit not in EVEN_DIGITS:
        return "EXCLUDE", ["MAINSTEM_INTERBASIN_DIGIT"]
    if in_major < CHILD_MIN_IN_MAJOR:
        return "EXCLUDE", ["NOT_WITHIN_MAJOR"]
    if anomalies:
        return "REVIEW", anomalies
    return "ACCEPT", []


def select_children(major, root, catalog, child_audit):
    siblings, level, prefix, skipped = first_subdivision(root, catalog)
    info = dict(split_level=level, split_parent_pfaf=prefix,
                skipped_levels=";".join(map(str, skipped)), n_children=len(siblings))
    if siblings.empty:
        return [], {**info, "selection_status": "NO_SUBDIVISION"}
    path = catalog[root["region"], level]
    geo = geometry_rows(path, siblings.HYBAS_ID)
    if len(geo) != len(siblings):
        raise ValueError(f"Incomplete child geometry read: {major['major']}")
    geo_ea = geo.to_crs(AREA_CRS)
    child_union = clean_polygon(shapely.union_all(list(geo_ea.geometry)))
    partition = overlap_metrics(root["ea"], child_union)
    info["partition_coverage"] = partition["coverage"]
    info["partition_purity"] = partition["purity"]
    good_partition = min(partition["coverage"], partition["purity"]) >= 0.999
    dup_pfaf = set(siblings.loc[siblings.PFAF_ID.duplicated(keep=False), "PFAF_ID"])
    meta = siblings.set_index("HYBAS_ID")
    candidates = []
    for idx, g in geo.iterrows():
        hid = int(g.HYBAS_ID)
        row = meta.loc[hid]
        digit = int(row.PFAF_ID) % 10
        gea = clean_polygon(geo_ea.loc[idx, "geometry"])
        in_major = gea.intersection(major["ea"]).area / gea.area
        in_root = gea.intersection(root["ea"]).area / gea.area
        anomalies = []
        if digit in EVEN_DIGITS:
            anomalies = child_topology_reasons(row, siblings)
            if not good_partition or in_root < 0.999:
                anomalies.append("PARENT_PARTITION_MISMATCH")
            if int(row.PFAF_ID) in dup_pfaf:
                anomalies.append("DUPLICATE_PFAF")
        decision, reasons = child_decision(digit, in_major, anomalies)
        rec = dict(major=major["major"], HYBAS_ID=hid, PFAF_ID=int(row.PFAF_ID),
                   pfaf_digit=digit, level=level, region=root["region"],
                   NEXT_DOWN=int(row.NEXT_DOWN), NEXT_SINK=int(row.NEXT_SINK),
                   MAIN_BAS=int(row.MAIN_BAS), hydro_order=int(row.ORDER),
                   ENDO=int(row.ENDO), COAST=int(row.COAST),
                   source_sub_area_km2=float(row.SUB_AREA),
                   source_up_area_km2=float(row.UP_AREA),
                   area_km2=gea.area / 1e6, frac_in_major=in_major, frac_in_root=in_root,
                   incoming_surface_units=int(row.incoming_surface_units),
                   incoming_virtual_units=int(row.incoming_virtual_units),
                   up_sub_area_rel_diff=float(row.up_sub_area_rel_diff),
                   root_HYBAS_ID=root["HYBAS_ID"], root_PFAF_ID=root["PFAF_ID"],
                   root_level=root["level"], root_tier=root["root_tier"],
                   root_relation=root["root_relation"], source=str(path),
                   decision=decision, reason=";".join(reasons))
        child_audit.append(rec)
        if digit in EVEN_DIGITS:
            candidates.append({**rec, "geometry": g.geometry, "ea": gea})
    info["selection_status"] = selection_status(candidates)
    return candidates, info


def selection_status(candidates):
    acc = sum(c["decision"] == "ACCEPT" for c in candidates)
    if any(c["decision"] == "REVIEW" for c in candidates):
        return "REVIEW_CHILDREN"
    if acc == 4:
        return "PASS"
    if acc > 0:
        return "PASS_FEWER_THAN_FOUR"
    return "NO_TRIBUTARY_IN_MAJOR"


def pair_audit(records):

    cols = ["major_a", "id_a", "major_b", "id_b", "same_parent", "relation",
            "fraction_a", "fraction_b", "intersection_km2"]
    rows = []
    if records:
        tree = shapely.STRtree([r["ea"] for r in records])
        for i, a in enumerate(records):
            for j in tree.query(a["ea"], predicate="intersects"):
                if int(j) <= i:
                    continue
                b = records[int(j)]
                inter = a["ea"].intersection(b["ea"]).area
                fa, fb = inter / a["ea"].area, inter / b["ea"].area
                if max(fa, fb) <= OVERLAP_REL_TOL:
                    continue
                relation = "DUPLICATE" if min(fa, fb) > .999 else (
                    "NESTED" if max(fa, fb) > .999 else "PARTIAL_OVERLAP")
                rows.append(dict(major_a=a["major"], id_a=a["HYBAS_ID"],
                                 major_b=b["major"], id_b=b["HYBAS_ID"],
                                 same_parent=a["major"] == b["major"], relation=relation,
                                 fraction_a=fa, fraction_b=fb, intersection_km2=inter / 1e6))
    return pd.DataFrame(rows, columns=cols)


def read_major_rivers(path):
    g = gpd.read_file(path, engine="pyogrio")
    if not g.crs:
        raise ValueError(f"Missing CRS: {path}")
    g = g.to_crs(4326)
    lines = []
    for geom in g.geometry:
        lines.extend(line_parts(geom))
    if not lines:
        raise ValueError(f"No line geometry in river file: {path}")
    return gpd.GeoSeries(lines, crs=4326)


def clip_rivers(sub_wgs84, rivers_wgs84):

    crs = local_aeqd(sub_wgs84)

    poly = clean_polygon(gpd.GeoSeries([shapely.segmentize(sub_wgs84, 0.05)],
                                       crs=4326).to_crs(crs).iloc[0])
    basin_len = long_side_km(poly)
    m = dict(basin_length_km=basin_len, river_clipped_km=0.0, river_kept_km=0.0,
             river_boundary_dropped_km=0.0, river_short_dropped_km=0.0,
             river_n_networks=0, river_main_km=0.0, river_main_span_km=0.0,
             river_span_ratio=0.0, river_outlet_dist_km=np.nan,
             river_density_km_per_1000km2=0.0, river_ok=False, river_reason="NO_RIVER_IN_SUBBASIN")
    near = rivers_wgs84[rivers_wgs84.intersects(box(*sub_wgs84.bounds))]
    if near.empty:
        return [], m
    lines = gpd.GeoSeries(shapely.segmentize(np.asarray(near.values), 0.05),
                          crs=4326).to_crs(crs)
    pieces = []
    for geom in shapely.intersection(np.asarray(lines.values), poly):
        pieces.extend(line_parts(geom))
    if not pieces:
        return [], m

    merged = line_parts(shapely.line_merge(shapely.union_all(pieces)))
    m["river_clipped_km"] = sum(p.length for p in merged) / 1000

    band = poly.boundary.simplify(250).buffer(RIVER_BOUNDARY_KM * 1000)
    kept = []
    for p in merged:
        if p.length < RIVER_MIN_PIECE_KM * 1000:
            m["river_short_dropped_km"] += p.length / 1000
        elif p.intersection(band).length / p.length >= RIVER_BOUNDARY_FRAC:
            m["river_boundary_dropped_km"] += p.length / 1000
        else:
            kept.append(p)
    m["river_kept_km"] = sum(p.length for p in kept) / 1000
    area_km2 = poly.area / 1e6
    m["river_density_km_per_1000km2"] = m["river_kept_km"] / area_km2 * 1000
    if not kept:
        m["river_reason"] = "ONLY_BOUNDARY_OR_SHORT_PIECES"
        return [], m
    blobs = list(shapely.get_parts(shapely.union_all(
        [p.buffer(RIVER_SNAP_KM * 1000) for p in kept])))
    networks = []
    for b in blobs:
        members = [p for p in kept if p.intersects(b)]
        if members:
            networks.append(members)
    m["river_n_networks"] = len(networks)
    main = max(networks, key=lambda ms: sum(p.length for p in ms))
    main_geom = MultiLineString(main)
    m["river_main_km"] = main_geom.length / 1000
    m["river_main_span_km"] = long_side_km(main_geom)
    m["river_span_ratio"] = m["river_main_span_km"] / basin_len if basin_len > 0 else 0.0
    m["river_outlet_dist_km"] = main_geom.distance(poly.boundary) / 1000
    span_ok = m["river_span_ratio"] >= RIVER_MIN_SPAN_FRAC
    outlet_ok = m["river_outlet_dist_km"] <= RIVER_OUTLET_KM
    m["river_ok"] = bool(span_ok and outlet_ok)
    m["river_reason"] = "" if m["river_ok"] else ";".join(
        r for r, bad in [("MAIN_CHANNEL_TOO_SHORT", not span_ok),
                         ("MAIN_CHANNEL_NOT_AT_OUTLET", not outlet_ok)] if bad)
    kept_wgs = list(gpd.GeoSeries(kept, crs=crs).to_crs(4326).values)
    return kept_wgs, m


def tabular(records):
    skip = {"geometry", "ea", "river_geoms"}
    return pd.DataFrame([{k: v for k, v in x.items() if k not in skip} for x in records])


def save_csv(df, out, name):
    if len(df.columns) == 0:
        df = pd.DataFrame(columns=["major", "status"])
    df.to_csv(out / name, index=False, encoding="utf-8-sig")


def write_layer(gdf, gpkg, layer):
    if gdf is not None and not gdf.empty:
        gdf.to_file(gpkg, layer=layer, driver="GPKG", engine="pyogrio")


def write_outputs(records, roots, out):
    gpkg = out / "hydrobasins_tributaries.gpkg"
    if roots:
        write_layer(gpd.GeoDataFrame(tabular(roots), geometry=[r["geometry"] for r in roots],
                                     crs=4326), gpkg, "matched_major_roots")
    if records:
        g = gpd.GeoDataFrame(tabular(records), geometry=[r["geometry"] for r in records],
                             crs=4326)
        write_layer(g[g.decision == "ACCEPT"], gpkg, "selected_tributaries")
        write_layer(g[g.decision == "REVIEW"], gpkg, "review_tributaries")
        write_layer(g[g.excluded_by_river_check.astype(bool)], gpkg, "excluded_by_river_check")
    river_rows = [dict(unit_key=r["unit_key"], major=r["major"], HYBAS_ID=r["HYBAS_ID"],
                       geometry=MultiLineString(r["river_geoms"]))
                  for r in records if r["decision"] == "ACCEPT" and r.get("river_geoms")]
    if river_rows:
        write_layer(gpd.GeoDataFrame(river_rows, geometry="geometry", crs=4326),
                    gpkg, "selected_rivers")
    if not WRITE_SHAPEFILES:
        return
    bdir, rdir = out / "shapefiles" / "basins", out / "shapefiles" / "rivers"
    bdir.mkdir(parents=True)
    rdir.mkdir(parents=True)
    for r in records:
        if r["decision"] != "ACCEPT":
            continue

        stem = r["unit_key"]
        one = gpd.GeoDataFrame([dict(HYBAS_ID=r["HYBAS_ID"], PFAF_ID=r["PFAF_ID"],
                                     PARENT=r["major"], LEVEL=r["level"],
                                     AREA_KM2=r["area_km2"], NEXT_DOWN=r["NEXT_DOWN"])],
                               geometry=[r["geometry"]], crs=4326)
        one.to_file(bdir / f"{stem}.shp", driver="ESRI Shapefile", encoding="UTF-8",
                    engine="pyogrio")
        riv = gpd.GeoDataFrame(dict(HYBAS_ID=[r["HYBAS_ID"]] * len(r["river_geoms"]),
                                    PARENT=[r["major"]] * len(r["river_geoms"])),
                               geometry=r["river_geoms"], crs=4326)
        riv.to_file(rdir / f"{stem}.shp", driver="ESRI Shapefile", encoding="UTF-8",
                    engine="pyogrio")
        r["basin_shp"] = str(bdir / f"{stem}.shp")
        r["river_shp"] = str(rdir / f"{stem}.shp")


def self_test():

    t = pd.DataFrame([
        dict(HYBAS_ID=1, PFAF_ID=6221, NEXT_DOWN=0, COAST=0, ENDO=0,
             incoming_surface_units=2, up_sub_area_rel_diff=5.0),
        dict(HYBAS_ID=2, PFAF_ID=6222, NEXT_DOWN=1, COAST=0, ENDO=0,
             incoming_surface_units=0, up_sub_area_rel_diff=0.001),
        dict(HYBAS_ID=3, PFAF_ID=6224, NEXT_DOWN=1, COAST=0, ENDO=2,
             incoming_surface_units=0, up_sub_area_rel_diff=0.0)])
    assert not complete_unit(t.iloc[0])
    assert child_topology_reasons(t.iloc[1], t) == []
    assert "VIRTUAL_OUTLET" in child_topology_reasons(t.iloc[2], t)

    assert child_decision(3, 1.0, []) == ("EXCLUDE", ["MAINSTEM_INTERBASIN_DIGIT"])
    assert child_decision(2, 0.10, ["VIRTUAL_OUTLET"]) == ("EXCLUDE", ["NOT_WITHIN_MAJOR"])
    assert child_decision(2, 0.99, ["VIRTUAL_OUTLET"])[0] == "REVIEW"
    assert child_decision(8, 0.99, []) == ("ACCEPT", [])
    m = overlap_metrics(box(0, 0, 2, 2), box(0, 0, 1, 1))
    assert m["coverage"] == .25 and m["purity"] == 1
    records = [dict(major="A", HYBAS_ID=1, ea=box(0, 0, 2, 2)),
               dict(major="B", HYBAS_ID=2, ea=box(0, 0, 1, 1))]
    assert pair_audit(records).iloc[0].relation == "NESTED"

    cands = [dict(level=4, coverage=1.00, purity=0.55, iou=0.55),
             dict(level=5, coverage=0.90, purity=0.99, iou=0.89),
             dict(level=6, coverage=0.30, purity=1.00, iou=0.30)]
    fb = choose_fallback(cands)
    assert fb["level"] == 5 and fb["root_relation"] == "WITHIN_MAJOR"
    assert choose_fallback(cands[:1])["root_relation"] == "CONTAINS_MAJOR"
    assert choose_fallback(cands[2:]) is None

    from unittest.mock import patch
    root = dict(PFAF_ID=622, level=3, region="sa")
    tables = {4: pd.DataFrame({"PFAF_ID": [6220]}),
              5: pd.DataFrame({"PFAF_ID": [62201, 62202, 62203, 62204]}),
              6: pd.DataFrame({"PFAF_ID": [622011, 622012]})}
    with patch(__name__ + ".attributes", side_effect=lambda p: tables[int(p)]):
        children, level, prefix, skipped = first_subdivision(root, {("sa", i): i for i in tables})
        assert level == 5 and prefix == 6220 and skipped == [4]
        assert len(children) == 4

    sub = box(10.0, 45.0, 11.0, 45.5)
    main = LineString([(10.02, 45.25), (10.5, 45.26), (11.05, 45.25)])
    edge = LineString([(10.1, 45.001), (10.9, 45.001)])
    stub = LineString([(10.5, 45.40), (10.5, 45.41)])
    kept, mm = clip_rivers(sub, gpd.GeoSeries([main, edge, stub], crs=4326))
    assert mm["river_ok"], mm
    assert mm["river_boundary_dropped_km"] > 50, mm
    _, mm2 = clip_rivers(sub, gpd.GeoSeries([edge], crs=4326))
    assert not mm2["river_ok"] and mm2["river_reason"] == "ONLY_BOUNDARY_OR_SHORT_PIECES", mm2
    _, mm3 = clip_rivers(sub, gpd.GeoSeries([LineString([(10.45, 45.2), (10.55, 45.3)])], crs=4326))
    assert not mm3["river_ok"], mm3
    _, mm4 = clip_rivers(sub, gpd.GeoSeries([LineString([(20, 20), (21, 21)])], crs=4326))
    assert mm4["river_reason"] == "NO_RIVER_IN_SUBBASIN"
    print("Self-test passed: completeness, decisions, fallback root, 0-skip, "
          "first split, nesting and river checks.")


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--hydro-dir", type=Path, default=HYDRO_DIR)
    parser.add_argument("--major-manifest", type=Path, default=MAJOR_MANIFEST)
    parser.add_argument("--output-base", type=Path, default=OUTPUT_BASE)
    parser.add_argument("--basins", nargs="+", help="Optional subset of exact manifest names")
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        self_test()
        return

    out = args.output_base / ("run_" + datetime.now().strftime("%Y%m%d_%H%M%S_%f"))
    out.mkdir(parents=True, exist_ok=False)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s",
                        handlers=[logging.StreamHandler(sys.stdout),
                                  logging.FileHandler(out / "run.log", encoding="utf-8")])
    logging.getLogger("pyogrio").setLevel(logging.WARNING)
    shutil.copy2(Path(__file__), out / Path(__file__).name)
    config = dict(
        hydro_dir=str(args.hydro_dir), major_manifest=str(args.major_manifest),
        requested_basins=args.basins,
        root_min_coverage=ROOT_MIN_COVERAGE, root_min_purity=ROOT_MIN_PURITY,
        fallback_nest_min=FALLBACK_NEST_MIN, fallback_min_iou=FALLBACK_MIN_IOU,
        child_min_in_major=CHILD_MIN_IN_MAJOR, upstream_area_rel_tolerance=AREA_REL_TOL,
        overlap_rel_tolerance=OVERLAP_REL_TOL,
        river_boundary_km=RIVER_BOUNDARY_KM, river_boundary_frac=RIVER_BOUNDARY_FRAC,
        river_min_piece_km=RIVER_MIN_PIECE_KM, river_snap_km=RIVER_SNAP_KM,
        river_min_span_frac=RIVER_MIN_SPAN_FRAC, river_outlet_km=RIVER_OUTLET_KM,
        geometry_policy="full original HydroBASINS polygon; repair only; no clipping",
        selection_rule=("coarsest complete root with reciprocal 95% match, else highest-IoU "
                        "nested complete unit (IoU>=0.5); first genuine split; digits 2,4,6,8; "
                        ">=95% inside the major"),
        river_rule=("major's own river network clipped to the sub-basin; boundary fragments "
                    "removed; a connected main channel must reach the boundary and span "
                    ">=50% of the sub-basin length"),
        area_crs=AREA_CRS, source_documentation=DOC_URL,
        geopandas=gpd.__version__, shapely=shapely.__version__, pyogrio=pyogrio.__version__)
    (out / "config.json").write_text(json.dumps(config, ensure_ascii=False, indent=2),
                                     encoding="utf-8")

    catalog, inventory = discover(args.hydro_dir)
    save_csv(inventory, out, "00_input_inventory.csv")
    majors, manifest = load_majors(args.major_manifest, args.basins)
    save_csv(manifest, out, "00_major_manifest_used.csv")
    bounds = region_bounds(catalog)

    root_candidates, roots, child_audit, records, summaries = [], [], [], [], []
    for i, major in enumerate(majors, 1):
        LOG.info("[%02d/%02d] %s (%.0f km2)", i, len(majors), major["major"], major["area_km2"])
        root, root_status = match_root(major, catalog, bounds, root_candidates)
        summary = dict(major=major["major"], major_area_km2=major["area_km2"],
                       major_source=major["major_source"], river_source=major["river_shp"],
                       root_status=root_status, selection_status="", n_selected=0,
                       n_review=0)
        if root is not None:
            roots.append(root)
            for key in ("root_tier", "root_relation", "region", "level", "HYBAS_ID",
                        "PFAF_ID", "coverage", "purity", "iou"):
                summary["root_" + key.replace("root_", "")] = root[key]
            children, info = select_children(major, root, catalog, child_audit)
            records.extend(children)
            summary.update(info)
        else:
            mine = [x for x in root_candidates if x["major"] == major["major"]]
            if mine:
                best = max(mine, key=lambda x: x["iou"])
                summary.update(best_candidate_id=best["HYBAS_ID"], best_iou=best["iou"],
                               best_coverage=best["coverage"], best_purity=best["purity"],
                               best_complete=best["complete_unit"])
        summaries.append(summary)
        save_csv(pd.DataFrame(summaries), out, "01_major_selection_summary.csv")
        save_csv(pd.DataFrame(root_candidates), out, "02_root_match_candidates.csv")
        save_csv(pd.DataFrame(child_audit), out, "03_first_split_audit.csv")
        LOG.info("  %s / %s", root_status, summary.get("selection_status", ""))


    pairs = pair_audit([r for r in records if r["decision"] == "ACCEPT"])
    bad = set(pairs.loc[pairs.same_parent.astype(bool), "major_a"]) if not pairs.empty else set()
    for r in records:
        if r["major"] in bad and r["decision"] == "ACCEPT":
            r["decision"], r["reason"] = "REVIEW", "WITHIN_PARENT_OVERLAP"
    save_csv(pairs, out, "05_selected_pairwise_overlap.csv")


    for r in records:
        r["excluded_by_river_check"] = False
    by_major = {m["major"]: m for m in majors}
    river_audit = []
    for major_name in sorted({r["major"] for r in records if r["decision"] == "ACCEPT"}):
        rivers = read_major_rivers(by_major[major_name]["river_shp"])
        for r in records:
            if r["major"] != major_name or r["decision"] != "ACCEPT":
                continue
            kept, metrics = clip_rivers(r["geometry"], rivers)
            r.update(metrics)
            r["river_geoms"] = kept
            if not metrics["river_ok"]:
                r["decision"], r["reason"] = "EXCLUDE", metrics["river_reason"]
                r["excluded_by_river_check"] = True
            river_audit.append(dict(major=r["major"], HYBAS_ID=r["HYBAS_ID"],
                                    PFAF_ID=r["PFAF_ID"], area_km2=r["area_km2"],
                                    **metrics))
        LOG.info("  rivers clipped for %s", major_name)
    save_csv(pd.DataFrame(river_audit), out, "07_river_check.csv")


    final = {(r["major"], r["HYBAS_ID"]): (r["decision"], r["reason"]) for r in records}
    for row in child_audit:
        key = (row["major"], row["HYBAS_ID"])
        if key in final:
            row["decision"], row["reason"] = final[key]
    for s in summaries:
        mine = [r for r in records if r["major"] == s["major"]]
        s["n_selected"] = sum(r["decision"] == "ACCEPT" for r in mine)
        s["n_review"] = sum(r["decision"] == "REVIEW" for r in mine)
        s["n_excluded_by_river_check"] = sum(bool(r["excluded_by_river_check"]) for r in mine)
        if mine:
            s["selection_status"] = selection_status(mine)

    for r in records:
        r["unit_key"] = f"{r['major']}__HYBAS_{r['HYBAS_ID']}"
        r["subbasin_name"] = f"HYBAS_{r['HYBAS_ID']}"
        r["parent_mainstem"] = r["major"]
        r["river_name"] = ""
        r["source_dataset"] = "HydroBASINS_v1c_standard"
    write_outputs(records, roots, out)
    selected = [r for r in records if r["decision"] == "ACCEPT"]
    save_csv(tabular(selected), out, "04_selected_subbasin_master.csv")
    save_csv(tabular([r for r in records if r["decision"] == "REVIEW"]), out,
             "04_review_subbasins.csv")
    save_csv(pd.DataFrame(summaries), out, "01_major_selection_summary.csv")
    save_csv(pd.DataFrame(child_audit), out, "03_first_split_audit.csv")
    unique = tabular(selected)
    if not unique.empty:
        unique = unique.drop_duplicates("HYBAS_ID")
    save_csv(unique, out, "06_unique_hybas_units.csv")

    excluded_river = sum(s.get("n_excluded_by_river_check", 0) for s in summaries)
    passed_geometry = len(river_audit)
    result = dict(
        major_count=len(majors), matched_root_count=len(roots),
        root_status_counts=pd.Series([s["root_status"] for s in summaries]).value_counts().to_dict(),
        selection_status_counts=pd.Series([s["selection_status"] or "NONE" for s in summaries])
        .value_counts().to_dict(),
        selected_subbasins=len(selected),
        unique_hybas_ids=len({r["HYBAS_ID"] for r in selected}),
        review_rows=sum(r["decision"] == "REVIEW" for r in records),
        excluded_by_river_check=excluded_river,
        excluded_by_river_check_share=(excluded_river / passed_geometry) if passed_geometry else 0.0,
        majors_with_four=sum(s["n_selected"] == 4 for s in summaries),
        majors_with_none=sum(s["n_selected"] == 0 for s in summaries),
        overlapping_selected_pairs=len(pairs), completed=True, output_directory=str(out))
    (out / "summary.json").write_text(json.dumps(result, ensure_ascii=False, indent=2),
                                      encoding="utf-8")
    LOG.info("Completed: %s", json.dumps(result, ensure_ascii=False))
    if passed_geometry and excluded_river / passed_geometry > 0.10:
        LOG.warning("WARNING: more than 10% of geometric selections failed the river check; "
                    "decide how to handle this before any HRP computation.")


if __name__ == "__main__":
    main()
