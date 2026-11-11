
"""Extract mainstems and tributaries from HydroRIVERS; validate outlet connectivity.
Configure file locations in paths.json and network thresholds below."""

import json as _path_json
from pathlib import Path as _ConfigPath


_DATA_PATHS = _path_json.loads(
    (_ConfigPath(__file__).resolve().parents[1] / "paths.json").read_text(encoding="utf-8")
)
from pathlib import Path
from datetime import datetime
import argparse
import json
import logging
import shutil
import sys
import time

import numpy as np
import pandas as pd
import geopandas as gpd
import pyogrio
import shapely
from pyproj import CRS
from shapely.geometry import LineString, MultiLineString


HYDRORIVERS = Path(_DATA_PATHS["HYDRORIVERS"])
HYDROBASINS_DIR = Path(_DATA_PATHS["HYDROBASINS_DIR"])

SELECTION_RUN = Path(_DATA_PATHS["SELECTION_RUN"])
OUTPUT_BASE = Path(_DATA_PATHS["NETWORK_OUTPUT_ROOT"])

THETA = 0.10
SENSITIVITY_THETAS = (0.05, 0.20)
WRITE_MAJOR_NETWORKS = True
A_MIN = 0.001
MAX_CANDIDATES_PER_MAJOR = 300
MATCH_BUFFER_KM = 3.0
SKIP_CONFLUENCE_KM = 6.0
MIN_EVAL_KM = 2.0
PRESENT_FRAC = 0.5
MAINSTEM_MIN_FRAC = 0.5
INCLUDE_VIRTUAL_OUTLET_REVIEW = False
AREA_RATIO_TOL = 0.10
WRITE_SHAPEFILES = True

REGIONS = {1: "af", 2: "eu", 3: "si", 4: "as", 5: "au", 6: "sa", 7: "na", 8: "ar"}
RIVER_FIELDS = ["HYRIV_ID", "NEXT_DOWN", "LENGTH_KM", "UPLAND_SKM", "CATCH_SKM", "ORD_STRA",
                "ORD_CLAS", "HYBAS_L12"]
CAL_COLUMNS = ["major", "HYRIV_ID", "upland_km2", "rel_area", "path_km", "evaluated_km",
               "match_frac", "included", "evaluable", "used"]
RIVER_COLUMNS = ["unit_key", "HYRIV_ID", "UPLAND", "LENGTH_KM", "ORD_STRA", "ORD_CLAS",
                 "ROLE", "geometry"]
LOG = logging.getLogger("rivers")
DOC_URL = "https://data.hydrosheds.org/file/technical-documentation/HydroRIVERS_TechDoc_v10.pdf"


def local_aeqd(geom_wgs84):
    p = geom_wgs84.representative_point()
    return CRS.from_proj4(f"+proj=aeqd +lat_0={p.y} +lon_0={p.x} "
                          "+datum=WGS84 +units=m +no_defs")


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


class RiverNet:


    def __init__(self, ids, next_down, length_km, upland_km2, pfaf12, fid=None,
                 strahler=None, classical=None, catch_km2=None):
        self.id = np.asarray(ids, dtype=np.int64)
        n = len(self.id)
        self.n = n
        self.length = np.asarray(length_km, dtype=np.float64)
        self.upl = np.asarray(upland_km2, dtype=np.float64)
        self.catch = (np.zeros(n, np.float64) if catch_km2 is None
                      else np.asarray(catch_km2, dtype=np.float64))
        self.fid = np.arange(n, dtype=np.int64) if fid is None else np.asarray(fid, np.int64)
        self.strahler = (np.zeros(n, np.int16) if strahler is None
                         else np.asarray(strahler, np.int16))
        self.classical = (np.zeros(n, np.int16) if classical is None
                          else np.asarray(classical, np.int16))
        order = np.argsort(self.id)
        ids_sorted = self.id[order]
        nd = np.asarray(next_down, dtype=np.int64)
        pos = np.clip(np.searchsorted(ids_sorted, nd), 0, n - 1)
        ok = (nd > 0) & (ids_sorted[pos] == nd)
        self.down = np.where(ok, order[pos], -1).astype(np.int64)
        src = np.nonzero(self.down >= 0)[0]
        dst = self.down[src]
        o = np.argsort(dst, kind="stable")
        self.up_src = src[o].astype(np.int64)
        self.up_off = np.concatenate([[0], np.cumsum(np.bincount(dst, minlength=n))])
        self.pfaf12 = np.asarray(pfaf12, dtype=np.int64)
        valid = np.nonzero(self.pfaf12 >= 0)[0]
        self.p_order = valid[np.argsort(self.pfaf12[valid], kind="stable")]
        self.p_sorted = self.pfaf12[self.p_order]
        self.mask = np.zeros(n, dtype=bool)

    def upstream(self, i, mask=None):
        ups = self.up_src[self.up_off[i]:self.up_off[i + 1]]
        m = self.mask if mask is None else mask
        return ups[m[ups]]

    def unit_reaches(self, pfaf, level):

        scale = 10 ** (12 - int(level))
        lo, hi = int(pfaf) * scale, (int(pfaf) + 1) * scale
        a = np.searchsorted(self.p_sorted, lo, "left")
        b = np.searchsorted(self.p_sorted, hi, "left")
        return self.p_order[a:b]

    def activate(self, reaches):
        self.mask[:] = False
        self.mask[reaches] = True

    def outlet(self, reaches):
        return int(reaches[np.argmax(self.upl[reaches])]) if len(reaches) else -1

    def trace_up(self, i):

        path = [int(i)]
        while True:
            ups = self.upstream(i)
            if len(ups) == 0:
                return path
            i = int(ups[np.argmax(self.upl[ups])])
            path.append(i)

    def rule_network(self, outlet, thr):

        keep, stack, terminals = [], [int(outlet)], []
        while stack:
            i = stack.pop()
            keep.append(i)
            ups = self.upstream(i)
            big = ups[self.upl[ups] >= thr]
            if len(big) == 0:
                terminals.append(i)
            stack.extend(int(u) for u in big)
        for t in terminals:
            keep.extend(self.trace_up(t)[1:])
        return np.array(keep, dtype=np.int64), len(terminals)

    def systems_network(self, outlet, others, thr):

        keep, n = self.rule_network(outlet, thr)
        main = set(self.trace_up(outlet))
        parts, n_sys = [], 1
        for e, size in others:
            if size >= thr:
                k2, n2 = self.rule_network(int(e), thr)
                parts.append(k2)
                n += n2
                n_sys += 1
        sep = np.concatenate(parts) if parts else np.array([], np.int64)
        return np.concatenate([keep, sep]), n, main, set(int(r) for r in sep), n_sys

    def tributaries(self, outlet, thr):

        out, stack = [], [int(outlet)]
        while stack:
            i = stack.pop()
            ups = self.upstream(i)
            if len(ups) == 0:
                continue
            main_child = int(ups[np.argmax(self.upl[ups])])
            for u in ups:
                u = int(u)
                if self.upl[u] >= thr:
                    stack.append(u)
                    if u != main_child:
                        out.append(u)
        return out


def resolve_hydrorivers(path):
    p = Path(path)
    if p.is_file() and p.suffix.lower() == ".shp":
        return p
    if p.is_dir():
        hits = sorted(p.rglob("HydroRIVERS_v10*.shp"))
        exact = [h for h in hits if h.stem.lower() == "hydrorivers_v10"]
        if exact:
            return exact[0]
        if len(hits) == 1:
            return hits[0]
        if hits:
            raise ValueError("Several HydroRIVERS shapefiles found; point HYDRORIVERS to the "
                             f"global HydroRIVERS_v10.shp: {[h.name for h in hits]}")
    if p.with_suffix(".shp").is_file():
        return p.with_suffix(".shp")
    raise FileNotFoundError(f"HydroRIVERS shapefile not found at {p}")


def hydrobasins_file(root, region, level):
    hits = sorted(Path(root).rglob(f"hybas_{region}_lev{int(level):02d}_v1c.shp"))
    if len(hits) != 1:
        raise FileNotFoundError(f"Expected one hybas_{region}_lev{int(level):02d}_v1c.shp "
                                f"under {root}, found {len(hits)}")
    return hits[0]


def pfaf12_lookup(hb_dir):
    frames = []
    for region in REGIONS.values():
        f = hydrobasins_file(hb_dir, region, 12)
        frames.append(pyogrio.read_dataframe(f, read_geometry=False,
                                             columns=["HYBAS_ID", "PFAF_ID"]))
    t = pd.concat(frames, ignore_index=True)
    h = t.HYBAS_ID.to_numpy(np.int64)
    p = t.PFAF_ID.to_numpy(np.int64)
    o = np.argsort(h)
    h, p = h[o], p[o]

    def lookup(x):
        x = np.asarray(x, dtype=np.int64)
        pos = np.clip(np.searchsorted(h, x), 0, len(h) - 1)
        return np.where(h[pos] == x, p[pos], -1)
    return lookup


def load_rivernet(shp, lookup):
    LOG.info("Reading HydroRIVERS attributes (about 8.5 million reaches) ...")
    kw = dict(read_geometry=False, columns=RIVER_FIELDS, fid_as_index=True)
    try:
        t = pyogrio.read_dataframe(shp, use_arrow=True, **kw)
    except Exception:
        t = pyogrio.read_dataframe(shp, **kw)
    missing = set(RIVER_FIELDS) - set(t.columns)
    if missing:
        raise ValueError(f"HydroRIVERS fields missing: {missing}")
    num = lambda c: pd.to_numeric(t[c], errors="coerce").fillna(0)
    pf = lookup(num("HYBAS_L12").to_numpy(np.int64))
    LOG.info("  %d reaches; %.2f%% without a level-12 HydroBASINS match",
             len(t), 100 * float(np.mean(pf < 0)))
    net = RiverNet(num("HYRIV_ID"), num("NEXT_DOWN"), num("LENGTH_KM"), num("UPLAND_SKM"),
                   pf, fid=t.index.to_numpy(np.int64), strahler=num("ORD_STRA"),
                   classical=num("ORD_CLAS"), catch_km2=num("CATCH_SKM"))
    del t
    return net


def read_geometries(shp, net, reach_idx):

    idx = np.unique(np.asarray(reach_idx, dtype=np.int64))
    if len(idx) == 0:
        return {}
    fids = net.fid[idx]
    g = pyogrio.read_dataframe(shp, fids=fids, columns=["HYRIV_ID"], fid_as_index=True)
    if g.crs is None:
        g = g.set_crs(4326)
    elif not CRS.from_user_input(g.crs).equals(CRS.from_epsg(4326), ignore_axis_order=True):
        g = g.to_crs(4326)
    by_fid = dict(zip(g.index.to_numpy(np.int64), g.geometry.values))
    return {int(i): by_fid.get(int(f)) for i, f in zip(idx, fids)}


def agreement(a, y, theta):

    a = np.asarray(a, dtype=float)
    y = np.asarray(y, dtype=bool)
    pred = a >= theta
    tp = int(np.sum(pred & y))
    fp = int(np.sum(pred & ~y))
    fn = int(np.sum(~pred & y))
    tn = int(np.sum(~pred & ~y))
    den = np.sqrt(float(tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    return dict(tp=tp, fp=fp, fn=fn, tn=tn,
                tpr=tp / (tp + fn) if tp + fn else np.nan,
                precision=tp / (tp + fp) if tp + fp else np.nan,
                f1=2 * tp / (2 * tp + fp + fn) if tp + fp + fn else np.nan,
                mcc=(tp * tn - fp * fn) / den if den > 0 else np.nan)


def length_match_theta(rel, path, base, target):

    rel = np.asarray(rel, dtype=float)
    path = np.asarray(path, dtype=float)
    o = np.argsort(-rel, kind="stable")
    r = rel[o]
    cum = base + np.cumsum(path[o])
    k = int(np.searchsorted(cum, target * (1 + 1e-9), side="right"))
    return float(r[k]) if k < len(r) else A_MIN


def match_fraction(path, net, geoms_ea, buffer_ea, skip_km):

    lengths = net.length[path]
    start = np.cumsum(lengths) - lengths
    keep = [r for r, s in zip(path, start) if s >= skip_km and geoms_ea.get(r) is not None]
    parts = []
    for r in keep:
        parts.extend(line_parts(geoms_ea[r]))
    if not parts:
        return np.nan, 0.0
    g = MultiLineString(parts)
    if g.length / 1000 < MIN_EVAL_KM:
        return np.nan, g.length / 1000
    return g.intersection(buffer_ea).length / g.length, g.length / 1000


def read_lines(path):
    g = gpd.read_file(path, engine="pyogrio")
    if g.crs is None:
        g = g.set_crs(4326)
    g = g.to_crs(4326)
    parts = []
    for geom in g.geometry:
        parts.extend(line_parts(geom))
    return gpd.GeoSeries(parts, crs=4326)


def unit_candidates(net, reaches, outlet, thr):

    cands = net.tributaries(outlet, thr)
    flowing, sinks = unit_exits(net, reaches)
    for e in np.r_[flowing, sinks]:
        if int(e) != outlet and net.upl[e] >= thr:
            cands.append(int(e))
            cands.extend(net.tributaries(int(e), thr))
    return cands


def calibrate(net, shp, majors, manifest):

    plans = []
    for row in majors.itertuples(index=False):
        reaches = net.unit_reaches(int(row.root_PFAF_ID), int(row.root_level))
        if len(reaches) == 0:
            LOG.warning("WARNING: no HydroRIVERS reach in %s; left out", row.major)
            continue
        net.activate(reaches)
        outlet = net.outlet(reaches)
        A = net.upl[outlet]
        cands = unit_candidates(net, reaches, outlet, A_MIN * A)
        cands = sorted(cands, key=lambda c: -net.upl[c])[:MAX_CANDIDATES_PER_MAJOR]
        paths = {c: net.trace_up(c) for c in cands}
        main = net.trace_up(outlet)
        plans.append(dict(major=row.major, reaches=reaches, outlet=outlet, A=A,
                          paths=paths, main=main))
    LOG.info("Comparison: %d majors, %d rivers", len(plans),
             sum(len(p["paths"]) for p in plans))

    rows, majors_out = [], []
    river_src = dict(zip(manifest.basin_name, manifest.river_shp))
    for p in plans:
        lines = read_lines(river_src[p["major"]])
        if lines.empty:
            LOG.warning("WARNING: empty network file for %s", p["major"])
            continue
        crs = local_aeqd(shapely.box(*lines.total_bounds))
        lines_ea = lines.to_crs(crs)
        buffer_ea = shapely.union_all(list(lines_ea.buffer(MATCH_BUFFER_KM * 1000).values))
        ids = [r for path in list(p["paths"].values()) + [p["main"]] for r in path]
        ids = list(dict.fromkeys(ids))
        geoms = read_geometries(shp, net, ids)
        gs = gpd.GeoSeries([geoms.get(r) for r in ids], crs=4326).to_crs(crs)
        geoms_ea = dict(zip(ids, gs.values))
        main_frac, _ = match_fraction(p["main"], net, geoms_ea, buffer_ea, 0.0)
        actual_km = float(lines_ea.length.sum() / 1000)
        info = dict(major=p["major"], outlet_upland_km2=p["A"],
                    main_stem_km=float(net.length[p["main"]].sum()),
                    main_stem_match=main_frac, network_km=actual_km,
                    network_density=actual_km / p["A"] * 1000,
                    n_candidates=len(p["paths"]))
        usable = bool(np.isfinite(main_frac) and main_frac >= MAINSTEM_MIN_FRAC)
        info["used_in_calibration"] = usable
        n0 = len(rows)
        for c, path in p["paths"].items():
            frac, eval_km = match_fraction(path, net, geoms_ea, buffer_ea, SKIP_CONFLUENCE_KM)
            rows.append(dict(major=p["major"], HYRIV_ID=int(net.id[c]),
                             upland_km2=float(net.upl[c]), rel_area=float(net.upl[c] / p["A"]),
                             path_km=float(net.length[path].sum()), evaluated_km=eval_km,
                             match_frac=frac,
                             included=bool(np.isfinite(frac) and frac >= PRESENT_FRAC),
                             evaluable=bool(np.isfinite(frac)), used=usable))
        mine = rows[n0:]
        info["n_evaluated"] = sum(r["evaluable"] for r in mine)
        info["n_included"] = sum(r["included"] for r in mine)
        info["oracle_km"] = info["main_stem_km"] + sum(r["path_km"] for r in mine
                                                       if r["included"])
        info["geometry_factor"] = info["oracle_km"] / actual_km if actual_km > 0 else np.nan
        info["theta_length_match"] = length_match_theta(
            [r["rel_area"] for r in mine], [r["path_km"] for r in mine],
            info["main_stem_km"], info["oracle_km"])
        majors_out.append(info)
        LOG.info("  %-18s main stem match %.2f; %d rivers; %d included",
                 p["major"], main_frac if np.isfinite(main_frac) else -1, len(p["paths"]),
                 info["n_included"])
    return pd.DataFrame(rows, columns=CAL_COLUMNS), pd.DataFrame(majors_out)


def curve_lengths(cal_used, used, theta):

    sel = cal_used[cal_used.rel_area >= theta].groupby("major").path_km
    length = used.main_stem_km + sel.sum().reindex(used.index, fill_value=0.0)
    return length, sel.size().reindex(used.index, fill_value=0)


def theta_floor(cal, bym):

    floor = A_MIN
    for m in bym.loc[bym.n_candidates >= MAX_CANDIDATES_PER_MAJOR, "major"]:
        v = cal.loc[cal.major == m, "rel_area"].min()
        if np.isfinite(v):
            floor = max(floor, float(v))
    return floor


def theta_curve(cal, bym, thetas):
    used = bym[bym.used_in_calibration].set_index("major")
    c = cal[cal.major.isin(used.index)]
    ev = c[c.evaluable]
    rows = []
    for th in thetas:
        length, ntrib = curve_lengths(c, used, th)
        raw = length / used.network_km
        cor = length / used.oracle_km
        rows.append(dict(theta=float(th),
                         tributaries_median=float(ntrib.median()),
                         ratio_raw_median=float(raw.median()),
                         ratio_corrected_median=float(cor.median()),
                         ratio_corrected_q25=float(cor.quantile(0.25)),
                         ratio_corrected_q75=float(cor.quantile(0.75)),
                         share_within_1p5=float(((cor >= 1 / 1.5) & (cor <= 1.5)).mean()),
                         share_within_2=float(((cor >= 0.5) & (cor <= 2.0)).mean()),
                         ratio_raw_pooled=float(length.sum() / used.network_km.sum()),
                         **agreement(ev.rel_area, ev.included, th)))
    return pd.DataFrame(rows)


def curve_summary(curve, cal, bym, theta):
    used = bym[bym.used_in_calibration]
    row = curve.loc[np.isclose(curve.theta, theta)].iloc[0]
    below = curve[curve.ratio_corrected_median <= 1.0]
    return dict(
        majors_in_calibration=int(len(used)),
        majors_left_out=sorted(bym.loc[~bym.used_in_calibration, "major"].tolist()),
        calibration_rivers=int((cal.used & cal.evaluable).sum()),
        rivers_in_existing_networks=int((cal.used & cal.included).sum()),
        geometry_factor_median=float(used.geometry_factor.median()),
        tributaries_median_existing=float(used.n_included.median()),
        at_theta=dict(tributaries_median_rule=float(row.tributaries_median),
                      length_ratio_raw_median=float(row.ratio_raw_median),
                      length_ratio_corrected_median=float(row.ratio_corrected_median),
                      length_ratio_corrected_iqr=[float(row.ratio_corrected_q25),
                                                  float(row.ratio_corrected_q75)],
                      share_within_factor_1p5=float(row.share_within_1p5),
                      share_within_factor_2=float(row.share_within_2),
                      f1=float(row.f1), mcc=float(row.mcc), tpr=float(row.tpr),
                      precision=float(row.precision)),
        theta_max_mcc=float(curve.theta[curve.mcc.idxmax()]),
        theta_max_f1=float(curve.theta[curve.f1.idxmax()]),
        theta_corrected_length_match=float(below.theta.min()) if len(below) else None,
        theta_length_match_major_median=float(used.theta_length_match.median()))


def load_selection(run_dir):
    run_dir = Path(run_dir)
    if not (run_dir / "07_river_check.csv").is_file():
        runs = sorted(d for d in run_dir.glob("run_*") if (d / "07_river_check.csv").is_file())
        if not runs:
            raise FileNotFoundError(f"No selection run with 07_river_check.csv under {run_dir}")
        run_dir = runs[-1]
    subs = pd.read_csv(run_dir / "07_river_check.csv")[["major", "HYBAS_ID", "PFAF_ID",
                                                          "area_km2"]]
    subs["selection_note"] = "geometric selection"
    review = run_dir / "04_review_subbasins.csv"
    if INCLUDE_VIRTUAL_OUTLET_REVIEW and review.is_file():
        r = pd.read_csv(review)
        if not r.empty and "reason" in r:
            r = r[r.reason.fillna("") == "VIRTUAL_OUTLET"][["major", "HYBAS_ID", "PFAF_ID",
                                                           "area_km2"]]
            r["selection_note"] = "REVIEW accepted: VIRTUAL_OUTLET only"
            subs = pd.concat([subs, r], ignore_index=True)
    summary = pd.read_csv(run_dir / "01_major_selection_summary.csv")
    manifest = pd.read_csv(run_dir / "00_major_manifest_used.csv")
    for c in ("HYBAS_ID", "PFAF_ID"):
        subs[c] = subs[c].astype("int64")
    return run_dir, subs.drop_duplicates(["major", "HYBAS_ID"]), summary, manifest


def unit_exits(net, reaches):

    d = net.down[reaches]
    leaves = np.ones(len(reaches), dtype=bool)
    ok = d >= 0
    leaves[ok] = ~net.mask[d[ok]]
    exits = reaches[leaves]
    return exits[net.down[exits] >= 0], exits[net.down[exits] < 0]


def in_unit(pfaf12, pfaf, level):
    scale = 10 ** (12 - int(level))
    return int(pfaf) * scale <= int(pfaf12) < (int(pfaf) + 1) * scale


def extract_subbasins(net, subs, theta, roots=None):

    rows, nets = [], {}
    roots = roots or {}
    for s in subs.itertuples(index=False):
        level = len(str(int(s.PFAF_ID)))
        key = f"{s.major}__HYBAS_{int(s.HYBAS_ID)}"
        reaches = net.unit_reaches(int(s.PFAF_ID), level)
        rec = dict(unit_key=key, major=s.major, HYBAS_ID=int(s.HYBAS_ID),
                   PFAF_ID=int(s.PFAF_ID), level=level,
                   region=REGIONS.get(int(str(int(s.HYBAS_ID))[0]), "?"),
                   area_km2=float(s.area_km2), selection_note=s.selection_note,
                   n_unit_reaches=len(reaches), theta=theta)
        if len(reaches) == 0:
            rec.update(flags="NO_REACHES", decision="EXCLUDE")
            rows.append(rec)
            continue
        net.activate(reaches)
        flowing, sinks = unit_exits(net, reaches)
        flags = []
        if len(flowing):
            outlet = int(flowing[np.argmax(net.upl[flowing])])
        else:
            outlet = net.outlet(reaches)
            flags.append("OUTLET_IS_SINK")
        if net.upl[outlet] < net.upl[net.outlet(reaches)]:
            flags.append("OUTLET_NOT_LARGEST")
        A = float(net.upl[outlet])
        sink_km2 = float(net.upl[sinks[sinks != outlet]].sum())
        other_km2 = float(net.upl[flowing[flowing != outlet]].sum())
        d = int(net.down[outlet])
        to_parent = np.nan
        if d >= 0 and s.major in roots:
            to_parent = in_unit(net.pfaf12[d], *roots[s.major])
            if not to_parent:
                flags.append("OUTLET_NOT_TO_PARENT")
        ratio = A / s.area_km2 if s.area_km2 else np.nan
        if not np.isfinite(ratio) or abs(ratio - 1) > AREA_RATIO_TOL:
            flags.append("OUTLET_AREA_MISMATCH")
        serious = {"OUTLET_IS_SINK", "OUTLET_NOT_TO_PARENT"} & set(flags)
        if serious == {"OUTLET_IS_SINK"} and str(s.selection_note).startswith("REVIEW accepted"):
            serious = set()
        others = [(int(e), float(net.upl[e])) for e in np.r_[flowing, sinks]
                  if int(e) != outlet]
        keep, n_rivers, main, sep, n_sys = net.systems_network(outlet, others, theta * A)
        network_km = float(net.length[keep].sum())
        rec.update(outlet_HYRIV_ID=int(net.id[outlet]), outlet_upland_km2=A,
                   area_ratio=ratio, sink_area_km2=sink_km2, other_exit_area_km2=other_km2,
                   explained_ratio=(A + sink_km2 + other_km2) / s.area_km2
                   if s.area_km2 else np.nan,
                   outlet_next_HYRIV_ID=int(net.id[d]) if d >= 0 else 0,
                   outlet_to_parent=to_parent,
                   n_reaches=len(keep), n_rivers=n_rivers, n_systems=n_sys,
                   n_tributaries=n_rivers - 1, network_km=network_km,
                   main_stem_km=float(net.length[sorted(main)].sum()),
                   density_km_per_1000km2=network_km / A * 1000,
                   flags=";".join(flags), decision="REVIEW" if serious else "ACCEPT")
        rows.append(rec)
        nets[key] = (keep, main, sep)
    return pd.DataFrame(rows), nets


def root_rule_lengths(net, majors, theta):

    rows = []
    for row in majors.itertuples(index=False):
        reaches = net.unit_reaches(int(row.root_PFAF_ID), int(row.root_level))
        if len(reaches) == 0:
            continue
        net.activate(reaches)
        o = net.outlet(reaches)
        flowing, sinks = unit_exits(net, reaches)
        others = [(int(e), float(net.upl[e])) for e in np.r_[flowing, sinks] if int(e) != o]
        keep, n_rivers, _, _, n_sys = net.systems_network(o, others, theta * net.upl[o])
        rows.append(dict(major=row.major, rule_network_km=float(net.length[keep].sum()),
                         rule_n_tributaries=n_rivers - 1, rule_n_systems=n_sys))
    return pd.DataFrame(rows, columns=["major", "rule_network_km", "rule_n_tributaries",
                                       "rule_n_systems"])


def polygonal(geom):
    if geom is None or geom.is_empty:
        return None
    if not geom.is_valid:
        geom = shapely.make_valid(geom)
    parts = [p for p in shapely.get_parts(geom) if p.geom_type in ("Polygon", "MultiPolygon")]
    return shapely.union_all(parts) if parts else None


def read_outline(path):
    g = gpd.read_file(path, engine="pyogrio")
    g = g.set_crs(4326) if g.crs is None else g.to_crs(4326)
    parts = [x for x in g.geometry if x is not None and not x.is_empty]
    return polygonal(shapely.union_all(parts)) if parts else None


def boxes_overlap(a, b):
    return not (a[2] < b[0] or a[0] > b[2] or a[3] < b[1] or a[1] > b[3])


def level12_points(hb_dir, boxes, chunk=50000):

    frames = []
    for region in REGIONS.values():
        f = hydrobasins_file(hb_dir, region, 12)
        try:
            tb = pyogrio.read_info(f).get("total_bounds")
        except Exception:
            tb = None
        if tb is not None and not any(boxes_overlap(b, tb) for b in boxes):
            continue
        start = 0
        while True:
            t = pyogrio.read_dataframe(f, columns=["PFAF_ID"], skip_features=start,
                                       max_features=chunk)
            if len(t) == 0:
                break
            pts = shapely.point_on_surface(np.asarray(t.geometry.values, dtype=object))
            if t.crs is not None and not CRS.from_user_input(t.crs).equals(
                    CRS.from_epsg(4326), ignore_axis_order=True):
                pts = np.asarray(gpd.GeoSeries(pts, crs=t.crs).to_crs(4326).values, dtype=object)
            frames.append(pd.DataFrame(dict(pfaf=t.PFAF_ID.to_numpy(np.int64),
                                            x=shapely.get_x(pts), y=shapely.get_y(pts))))
            if len(t) < chunk:
                break
            start += chunk
        LOG.info("  level-12 units located: %s", region)
    if not frames:
        return pd.DataFrame(dict(pfaf=np.array([], np.int64), x=[], y=[]))
    return pd.concat(frames, ignore_index=True)


def outline_reaches(net, px, py, ppf, outline):

    xmin, ymin, xmax, ymax = outline.bounds
    m = (px >= xmin) & (px <= xmax) & (py >= ymin) & (py <= ymax)
    shapely.prepare(outline)
    inside = shapely.contains_xy(outline, px[m], py[m])
    pf = np.unique(ppf[m][inside])
    lo = np.searchsorted(net.p_sorted, pf, "left")
    hi = np.searchsorted(net.p_sorted, pf, "right")
    parts = [net.p_order[a:b] for a, b in zip(lo, hi) if b > a]
    return (np.concatenate(parts) if parts else np.array([], np.int64)), len(pf)


def drain_to_exits(net, reaches):

    reaches = np.asarray(reaches, dtype=np.int64)
    k = len(reaches)
    if getattr(net, "_loc", None) is None:
        net._loc = np.full(net.n, -1, dtype=np.int64)
    loc = net._loc
    loc[reaches] = np.arange(k)
    try:
        d = net.down[reaches]
        nd = np.full(k, -1, dtype=np.int64)
        ok = d >= 0
        nd[ok] = loc[d[ok]]
    finally:
        loc[reaches] = -1
    parent = np.where(nd >= 0, nd, np.arange(k))
    for _ in range(64):
        nxt = parent[parent]
        if np.array_equal(nxt, parent):
            break
        parent = nxt
    area = np.bincount(parent, weights=net.catch[reaches], minlength=k)
    return parent, area


def major_outlines(net, hb_dir, summary):

    outlines = {}
    for row in summary.drop_duplicates("major").itertuples(index=False):
        try:
            geom = read_outline(row.major_source)
        except Exception as exc:
            LOG.warning("WARNING: outline of %s not readable (%s)", row.major, exc)
            geom = None
        outlines[row.major] = dict(geom=geom, area=float(row.major_area_km2),
                                   root_status=row.root_status)
    boxes = [o["geom"].bounds for o in outlines.values() if o["geom"] is not None]
    LOG.info("Locating level-12 units for %d major outlines ...", len(boxes))
    pts = level12_points(hb_dir, boxes)
    px, py, ppf = pts.x.to_numpy(float), pts.y.to_numpy(float), pts.pfaf.to_numpy(np.int64)
    for m, o in outlines.items():
        o.update(reaches=np.array([], np.int64), n_units=0)
        if o["geom"] is None:
            continue
        reaches, n_units = outline_reaches(net, px, py, ppf, o["geom"])
        o.update(reaches=reaches, n_units=n_units)
        if len(reaches):
            o.update(outline_system(net, reaches))
    return outlines


def outline_system(net, reaches):

    parent, area = drain_to_exits(net, reaches)
    exits = np.unique(parent)
    k = exits[np.argmax(area[exits])]
    rest = exits[exits != k]
    others = area[rest]
    return dict(outlet=int(reaches[k]), A=float(area[k]), other_km2=float(others.sum()),
                largest_other_km2=float(others.max()) if len(others) else 0.0,
                n_outlets=len(exits),
                others=[(int(reaches[i]), float(area[i])) for i in rest])


def major_outline_networks(net, outlines, theta):

    rows, nets = [], {}
    for m, o in outlines.items():
        rec = dict(major=m, root_status=o["root_status"], theta=theta,
                   major_area_km2=o["area"], n_level12_units=o["n_units"],
                   n_unit_reaches=len(o["reaches"]))
        if "outlet" not in o:
            rec["flags"] = "NO_OUTLINE" if o["geom"] is None else "NO_REACHES"
            rows.append(rec)
            continue
        net.activate(o["reaches"])
        out, A = o["outlet"], o["A"]
        keep, n_rivers, main, sep, n_sys = net.systems_network(out, o["others"], theta * A)
        d = int(net.down[out])
        ratio = A / o["area"] if o["area"] > 0 else np.nan
        flags = []
        if not np.isfinite(ratio) or abs(ratio - 1) > AREA_RATIO_TOL:
            flags.append("OUTLET_AREA_MISMATCH")
        if net.upl[out] > A * (1 + AREA_RATIO_TOL):
            flags.append("INFLOW_FROM_OUTSIDE")
        if n_sys > 1:
            flags.append("SECOND_SYSTEM")
        rec.update(outlet_HYRIV_ID=int(net.id[out]),
                   outlet_next_HYRIV_ID=int(net.id[d]) if d >= 0 else 0,
                   outlet_upland_km2=float(net.upl[out]), drainage_in_outline_km2=A,
                   area_ratio=ratio, other_outlets_km2=o["other_km2"],
                   largest_other_outlet_km2=o["largest_other_km2"], n_outlets=o["n_outlets"],
                   explained_ratio=(A + o["other_km2"]) / o["area"] if o["area"] > 0 else np.nan,
                   rule_n_tributaries=n_rivers - 1, rule_n_systems=n_sys,
                   rule_network_km=float(net.length[keep].sum()),
                   main_stem_km=float(net.length[sorted(main)].sum()), flags=";".join(flags))
        rows.append(rec)
        nets[m] = (keep, main, sep)
    return pd.DataFrame(rows), nets


def original_network_lengths(summary):
    out = {}
    for row in summary.drop_duplicates("major").itertuples(index=False):
        try:
            lines = read_lines(row.river_source)
            if not lines.empty:
                crs = local_aeqd(shapely.box(*lines.total_bounds))
                out[row.major] = float(lines.to_crs(crs).length.sum() / 1000)
        except Exception as exc:
            LOG.warning("WARNING: original network of %s not readable (%s)", row.major, exc)
    return out


def read_subbasin_polygons(hb_dir, subs):
    out = {}
    for (region, level), g in subs.groupby(["region", "level"]):
        f = hydrobasins_file(hb_dir, region, level)
        ids = sorted(set(int(x) for x in g.HYBAS_ID))
        t = pyogrio.read_dataframe(f, where="HYBAS_ID IN (" + ",".join(map(str, ids)) + ")",
                                   columns=["HYBAS_ID"])
        t = t.to_crs(4326) if t.crs is not None else t.set_crs(4326)
        for hid, geom in zip(t.HYBAS_ID.astype("int64"), t.geometry):
            out[int(hid)] = shapely.make_valid(geom) if not geom.is_valid else geom
    return out


def river_frame(net, geoms, keep, main, key, sep=()):
    def role(r):
        if r in main:
            return "main stem"
        return "separate system" if r in sep else "tributary"
    recs = [dict(unit_key=key, HYRIV_ID=int(net.id[r]), UPLAND=float(net.upl[r]),
                 LENGTH_KM=float(net.length[r]), ORD_STRA=int(net.strahler[r]),
                 ORD_CLAS=int(net.classical[r]), ROLE=role(int(r)), geometry=geoms.get(int(r)))
            for r in keep if geoms.get(int(r)) is not None]
    return gpd.GeoDataFrame(pd.DataFrame(recs, columns=RIVER_COLUMNS), geometry="geometry",
                            crs=4326)


def write_layer(frames, gpkg, layer):
    frames = [f for f in frames if not f.empty]
    if frames:
        gpd.GeoDataFrame(pd.concat(frames, ignore_index=True), geometry="geometry",
                         crs=4326).to_file(gpkg, layer=layer, driver="GPKG", engine="pyogrio")


def write_outputs(out, net, geoms, table, nets, polygons, basin_paths=None):

    gpkg = out / "subbasin_rivers.gpkg"
    accepted = table[table.decision == "ACCEPT"]
    write_basins = basin_paths is None
    river_rows, basin_rows = [], []
    if WRITE_SHAPEFILES:
        (out / "shapefiles" / "rivers").mkdir(parents=True, exist_ok=True)
        if write_basins:
            (out / "shapefiles" / "basins").mkdir(parents=True, exist_ok=True)
    basin_shp, river_shp = {}, {}
    for s in accepted.itertuples(index=False):
        keep, main, sep = nets[s.unit_key]
        rv = river_frame(net, geoms, keep, main, s.unit_key, sep)
        river_rows.append(rv)
        if write_basins:
            poly = polygons.get(int(s.HYBAS_ID))
            bv = gpd.GeoDataFrame([dict(unit_key=s.unit_key, HYBAS_ID=int(s.HYBAS_ID),
                                        PFAF_ID=int(s.PFAF_ID), PARENT=s.major,
                                        AREA_KM2=float(s.area_km2))],
                                  geometry=[poly], crs=4326)
            basin_rows.append(bv)
        if WRITE_SHAPEFILES:
            if rv.empty:
                LOG.warning("WARNING: no river geometry for %s", s.unit_key)
            else:
                rp = out / "shapefiles" / "rivers" / f"{s.unit_key}.shp"
                rv.drop(columns="unit_key").to_file(rp, engine="pyogrio", encoding="UTF-8")
                river_shp[s.unit_key] = str(rp)
            if write_basins:
                bp = out / "shapefiles" / "basins" / f"{s.unit_key}.shp"
                bv.drop(columns="unit_key").to_file(bp, engine="pyogrio", encoding="UTF-8")
                basin_shp[s.unit_key] = str(bp)
    if not write_basins:
        basin_shp = dict(basin_paths)
    write_layer(basin_rows, gpkg, "subbasins")
    write_layer(river_rows, gpkg, "subbasin_rivers")
    table = table.copy()
    table["basin_shp"] = table.unit_key.map(basin_shp)
    table["river_shp"] = table.unit_key.map(river_shp)
    return table


def write_major_networks(out, net, geoms, mtable, mnets, summary, bym, orig_km):

    folder = out / "shapefiles" / "major_rivers"
    if WRITE_SHAPEFILES:
        folder.mkdir(parents=True, exist_ok=True)
    frames, paths = [], {}
    for m in mtable.major:
        if m not in mnets:
            continue
        keep, main, sep = mnets[m]
        rv = river_frame(net, geoms, keep, main, m, sep)
        frames.append(rv)
        if WRITE_SHAPEFILES and not rv.empty:
            p = folder / f"{m}.shp"
            rv.drop(columns="unit_key").to_file(p, engine="pyogrio", encoding="UTF-8")
            paths[m] = str(p)
    write_layer(frames, out / "subbasin_rivers.gpkg", "major_rivers")
    src = summary.drop_duplicates("major").set_index("major")
    t = mtable.copy()
    t["basin_shp"] = t.major.map(src.major_source)
    t["river_shp"] = t.major.map(paths)
    t["river_shp_original"] = t.major.map(src.river_source)
    if "rule_network_km" not in t:
        return t
    t["network_km_original"] = t.major.map(orig_km)
    t["length_ratio_raw"] = t.rule_network_km / t.network_km_original
    if not bym.empty:
        b = bym.set_index("major")
        t["used_in_calibration"] = t.major.map(b.used_in_calibration)
        t["main_stem_match"] = t.major.map(b.main_stem_match)
        t["n_tributaries_original"] = t.major.map(b.n_included)
        t["length_ratio_corrected"] = t.rule_network_km / t.major.map(b.oracle_km)
    return t


def subbasin_master(table):
    master = table[table.decision == "ACCEPT"].copy()
    master["subbasin_name"] = "HYBAS_" + master.HYBAS_ID.astype(str)
    master["parent_mainstem"] = master.major
    master["source_dataset"] = "HydroBASINS v1c standard; HydroRIVERS v10"
    return master


def theta_tag(t):
    return f"theta_{t:g}".replace(".", "p")


def self_test():


    ids = [1, 2, 3, 4, 5, 6, 7, 8]
    down = [0, 1, 2, 3, 2, 5, 3, 5]
    upl = [100, 90, 50, 40, 30, 15, 3, 12]
    length = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]
    pf = [10 ** 11 * 1 + k for k in range(8)]
    net = RiverNet(ids, down, length, upl, pf)
    reaches = net.unit_reaches(1, 1)
    assert len(reaches) == 8
    net.activate(reaches)
    o = net.outlet(reaches)
    assert net.id[o] == 1
    assert [int(net.id[i]) for i in net.trace_up(o)] == [1, 2, 3, 4]
    keep, nr = net.rule_network(o, 0.25 * 100)
    got = sorted(int(net.id[i]) for i in keep)
    assert got == [1, 2, 3, 4, 5, 6], got
    assert nr == 2
    keep, nr = net.rule_network(o, 0.10 * 100)
    assert sorted(int(net.id[i]) for i in keep) == [1, 2, 3, 4, 5, 6, 8] and nr == 3
    tribs = sorted(int(net.id[i]) for i in net.tributaries(o, 2))
    assert tribs == [5, 7, 8], tribs

    A = net.upl[o]
    cands = net.tributaries(o, A_MIN * A)
    rows = [dict(major="M", rel_area=net.upl[c] / A, path_km=net.length[net.trace_up(c)].sum())
            for c in cands]
    cal = pd.DataFrame(rows)
    used = pd.DataFrame(dict(main_stem_km=[net.length[net.trace_up(o)].sum()]), index=["M"])
    for th in (0.25, 0.1, 0.02):
        approx, ntrib = curve_lengths(cal, used, th)
        keep, nr = net.rule_network(o, th * A)
        assert abs(approx["M"] - net.length[keep].sum()) < 1e-9, th
        assert ntrib["M"] == nr - 1, th

    net2 = RiverNet(ids, down, length, upl, [622 * 10 ** 9 + 1] * 4 + [623 * 10 ** 9 + 1] * 4)
    assert sorted(int(net2.id[i]) for i in net2.unit_reaches(622, 3)) == [1, 2, 3, 4]

    ag = agreement([0.5, 0.2, 0.1, 0.05], [1, 1, 0, 0], 0.15)
    assert (ag["tp"], ag["fp"], ag["fn"], ag["tn"]) == (2, 0, 0, 2) and ag["mcc"] == 1.0
    ag = agreement([0.5, 0.2, 0.1, 0.05], [1, 0, 1, 0], 0.15)
    assert ag["f1"] == 0.5 and ag["mcc"] == 0.0
    t = length_match_theta([0.5, 0.2, 0.1, 0.05], [10, 10, 10, 10], 10, 30)
    assert t == 0.1, t
    assert length_match_theta([0.5, 0.2], [10, 10], 10, 10) == 0.5
    assert length_match_theta([0.5, 0.2], [10, 10], 10, 100) == A_MIN

    net3 = RiverNet([1, 2, 3], [0, 1, 2], [5.0, 5.0, 5.0], [3, 2, 1], [1, 1, 1])
    geoms = {0: LineString([(0, 0), (5000, 0)]), 1: LineString([(5000, 0), (10000, 0)]),
             2: LineString([(10000, 0), (15000, 0)])}
    buf = LineString([(9000, 0), (20000, 0)]).buffer(3000)
    frac, km = match_fraction([0, 1, 2], net3, geoms, buf, skip_km=6.0)
    assert abs(frac - 1.0) < 1e-9 and abs(km - 5.0) < 1e-9, (frac, km)
    frac, km = match_fraction([0, 1, 2], net3, geoms, buf, skip_km=0.0)
    assert abs(frac - 9000 / 15000) < 1e-9, frac
    assert theta_tag(0.05) == "theta_0p05" and theta_tag(0.2) == "theta_0p2"


    ids = [100, 1, 2, 3, 4, 5, 6, 7]
    down = [0, 100, 1, 1, 0, 4, 0, 6]
    upl = [1000, 50, 30, 15, 60, 20, 40, 10]
    pf = [2 * 10 ** 11 + 1] + [10 ** 11 + k for k in range(5)] + [3 * 10 ** 11 + 1] * 2
    net4 = RiverNet(ids, down, [1.0] * 8, upl, pf)
    subs = pd.DataFrame(dict(major=["M", "M"], HYBAS_ID=[1000000001, 1000000003],
                             PFAF_ID=[1, 3], area_km2=[110.0, 40.0],
                             selection_note=["geometric selection"] * 2))
    tab, nets4 = extract_subbasins(net4, subs, 0.1, {"M": (2, 1)})
    r = tab.iloc[0]
    assert r.outlet_HYRIV_ID == 1 and r.outlet_next_HYRIV_ID == 100 and r.outlet_to_parent
    assert r.sink_area_km2 == 60 and abs(r.explained_ratio - 1) < 1e-12, r
    assert r["flags"] == "OUTLET_NOT_LARGEST;OUTLET_AREA_MISMATCH" and r.decision == "ACCEPT"

    assert sorted(int(net4.id[i]) for i in nets4[r.unit_key][0]) == [1, 2, 3, 4, 5]
    assert r.n_tributaries == 2 and r.n_systems == 2 and r.main_stem_km == 2.0
    line = {i: LineString([(0, 0), (1, 1)]) for i in range(net4.n)}
    rv = river_frame(net4, line, *nets4[r.unit_key][:2], r.unit_key, nets4[r.unit_key][2])
    roles = dict(zip(rv.HYRIV_ID, rv.ROLE))
    assert roles == {1: "main stem", 2: "main stem", 3: "tributary",
                     4: "separate system", 5: "separate system"}, roles
    r = tab.iloc[1]
    assert r["flags"] == "OUTLET_IS_SINK" and r.decision == "REVIEW", r
    tab, _ = extract_subbasins(net4, subs, 0.1, {"M": (3, 1)})
    assert "OUTLET_NOT_TO_PARENT" in tab.iloc[0]["flags"] and tab.iloc[0].decision == "REVIEW"
    subs.loc[1, "selection_note"] = "REVIEW accepted: VIRTUAL_OUTLET only"
    tab, _ = extract_subbasins(net4, subs, 0.1, {"M": (2, 1)})
    assert tab.iloc[1].decision == "ACCEPT"


    ids = [10, 11, 12, 13, 14, 15]
    down = [0, 10, 10, 12, 0, 0]
    catch = [5.0, 600.0, 200.0, 100.0, 20.0, 30.0]
    upl = [905.0, 600.0, 300.0, 100.0, 20.0, 30.0]
    pf = [101, 102, 103, 104, 105, 106]
    net5 = RiverNet(ids, down, [1.0] * 6, upl, pf, catch_km2=catch)
    px = np.array([0.5, 5.0, 0.2, 0.9, 0.6, 0.3])
    py = np.array([0.5, 5.0, 0.8, 0.1, 0.6, 0.3])
    reaches, n_units = outline_reaches(net5, px, py, np.array(pf), shapely.box(0, 0, 1, 1))
    assert sorted(int(net5.id[i]) for i in reaches) == [10, 12, 13, 14, 15] and n_units == 5
    parent, area = drain_to_exits(net5, reaches)
    got = {int(net5.id[reaches[i]]): area[i] for i in np.unique(parent)}
    assert got == {10: 305.0, 14: 20.0, 15: 30.0}, got
    o = dict(geom=shapely.box(0, 0, 1, 1), area=355.0, root_status="NO_ROOT",
             reaches=reaches, n_units=n_units, **outline_system(net5, reaches))
    assert net5.id[o["outlet"]] == 10 and o["A"] == 305.0 and o["other_km2"] == 50.0
    tab, nets5 = major_outline_networks(net5, {"X": o}, 0.1)
    r = tab.iloc[0]
    assert r["flags"] == "OUTLET_AREA_MISMATCH;INFLOW_FROM_OUTSIDE", r["flags"]
    assert sorted(int(net5.id[i]) for i in nets5["X"][0]) == [10, 12, 13]
    assert r.rule_n_tributaries == 0 and abs(r.explained_ratio - 1) < 1e-12

    net4.activate(net4.unit_reaches(1, 1))
    reach1 = net4.unit_reaches(1, 1)
    out1 = int(np.nonzero(net4.id == 1)[0][0])
    c4 = unit_candidates(net4, reach1, out1, A_MIN * 50)
    cal4 = pd.DataFrame([dict(major="U", rel_area=net4.upl[c] / 50,
                              path_km=net4.length[net4.trace_up(c)].sum()) for c in c4])
    used4 = pd.DataFrame(dict(main_stem_km=[net4.length[net4.trace_up(out1)].sum()]),
                         index=["U"])
    fl, sk = unit_exits(net4, reach1)
    oth = [(int(e), float(net4.upl[e])) for e in np.r_[fl, sk] if int(e) != out1]
    for th in (1.5, 0.5, 0.25, 0.1):
        approx, ntrib = curve_lengths(cal4, used4, th)
        keep, nr, *_ = net4.systems_network(out1, oth, th * 50)
        assert abs(approx["U"] - net4.length[keep].sum()) < 1e-9 and ntrib["U"] == nr - 1, th
    tab, nets5 = major_outline_networks(net5, {"X": o}, 0.05)
    assert "SECOND_SYSTEM" in tab.iloc[0]["flags"] and tab.iloc[0].rule_n_systems == 3
    assert sorted(int(net5.id[i]) for i in nets5["X"][0]) == [10, 12, 13, 14, 15]
    print("Self-test passed: topology, unit ranges, main stem, rule network, tributaries, "
          "rule length from candidates, agreement, length matching and river matching.")


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--hydrorivers", type=Path, default=HYDRORIVERS)
    parser.add_argument("--hydrobasins-dir", type=Path, default=HYDROBASINS_DIR)
    parser.add_argument("--selection-run", type=Path, default=SELECTION_RUN)
    parser.add_argument("--output-base", type=Path, default=OUTPUT_BASE)
    parser.add_argument("--theta", type=float, default=None,
                        help=f"river threshold (default THETA = {THETA})")
    parser.add_argument("--sensitivity", type=float, nargs="*", default=None,
                        help=f"extra thetas for the robustness check (default "
                             f"{list(SENSITIVITY_THETAS)}); give the flag alone to skip")
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        self_test()
        return

    theta = THETA if args.theta is None else float(args.theta)
    source = "THETA in script" if args.theta is None else "--theta on command line"
    sens_in = SENSITIVITY_THETAS if args.sensitivity is None else args.sensitivity
    sens = sorted({float(t) for t in sens_in if abs(float(t) - theta) > 1e-12})
    for t in [theta] + sens:
        if not 0 < t < 1:
            raise ValueError(f"theta must lie between 0 and 1, got {t}")
    thetas = [theta] + sens

    out = args.output_base / ("rivers_" + datetime.now().strftime("%Y%m%d_%H%M%S_%f"))
    out.mkdir(parents=True, exist_ok=False)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s",
                        handlers=[logging.StreamHandler(sys.stdout),
                                  logging.FileHandler(out / "run.log", encoding="utf-8")])
    logging.getLogger("pyogrio").setLevel(logging.WARNING)
    shutil.copy2(Path(__file__), out / Path(__file__).name)
    t_start = time.time()

    shp = resolve_hydrorivers(args.hydrorivers)
    run_dir, subs, summary, manifest = load_selection(args.selection_run)
    LOG.info("HydroRIVERS: %s", shp)
    LOG.info("Selection run: %s (%d sub-basins)", run_dir, len(subs))
    LOG.info("theta = %g (%s); sensitivity: %s", theta, source, sens or "none")
    config = dict(hydrorivers=str(shp), hydrobasins_dir=str(args.hydrobasins_dir),
                  selection_run=str(run_dir), theta=theta, theta_source=source,
                  sensitivity_thetas=sens, write_major_networks=WRITE_MAJOR_NETWORKS,
                  a_min=A_MIN, max_candidates_per_major=MAX_CANDIDATES_PER_MAJOR,
                  match_buffer_km=MATCH_BUFFER_KM, skip_confluence_km=SKIP_CONFLUENCE_KM,
                  min_eval_km=MIN_EVAL_KM, present_frac=PRESENT_FRAC,
                  mainstem_min_frac=MAINSTEM_MIN_FRAC,
                  include_virtual_outlet_review=INCLUDE_VIRTUAL_OUTLET_REVIEW,
                  area_ratio_tol=AREA_RATIO_TOL, rule=(
                      "rivers whose drainage area where they join a larger river or end (sea, "
                      "closed depression, unit edge) >= theta x drainage area at the unit "
                      "outlet, each traced to its source along the largest-drainage branch; "
                      "main stem always included"),
                  major_networks=("drawn inside each major outline; reaches whose level-12 "
                                  "unit has its inside point in the outline; outlet = largest "
                                  "drainage inside the outline"),
                  source_documentation=DOC_URL, geopandas=gpd.__version__,
                  shapely=shapely.__version__, pyogrio=pyogrio.__version__)
    (out / "config.json").write_text(json.dumps(config, ensure_ascii=False, indent=2),
                                     encoding="utf-8")

    lookup = pfaf12_lookup(args.hydrobasins_dir)
    net = load_rivernet(shp, lookup)

    matched = summary[summary.root_status == "MATCHED"].copy()
    matched = matched.dropna(subset=["root_PFAF_ID", "root_level"])
    cal, bym = calibrate(net, shp, matched, manifest)
    cal.to_csv(out / "01_calibration_rivers.csv", index=False, encoding="utf-8-sig")


    mt0 = root_rule_lengths(net, matched, theta)
    if not bym.empty and len(mt0):
        bym = bym.merge(mt0, on="major", how="left")
        bym["rule_density"] = bym.rule_network_km / bym.outlet_upland_km2 * 1000
        bym["length_ratio_raw"] = bym.rule_network_km / bym.network_km
        bym["length_ratio_corrected"] = bym.rule_network_km / bym.oracle_km
    bym.to_csv(out / "02_calibration_by_major.csv", index=False, encoding="utf-8-sig")

    calib = None
    if not bym.empty and bym.used_in_calibration.any():
        floor = theta_floor(cal, bym)
        grid = np.r_[np.geomspace(2 * A_MIN, 0.5, 161),
                     [0.005, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1,
                      0.12, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5], thetas]
        grid = np.unique(grid[grid > floor])
        curve = theta_curve(cal, bym, grid)
        curve.to_csv(out / "05_theta_curve.csv", index=False, encoding="utf-8-sig")
        if theta > floor:
            calib = curve_summary(curve, cal, bym, theta)
            used = bym[bym.used_in_calibration].set_index("major")
            approx, _ = curve_lengths(cal[cal.major.isin(used.index)], used, theta)
            gap = (approx / used.rule_network_km - 1).abs().max()
            if not gap < 1e-6:
                LOG.warning("WARNING: theta curve and network lengths differ by up to %.2g", gap)
        else:
            LOG.warning("WARNING: theta %.4g lies below the complete-candidate floor %.4g; "
                        "raise MAX_CANDIDATES_PER_MAJOR", theta, floor)
    else:
        LOG.warning("WARNING: no major usable for the comparison; theta curve skipped")


    rooted = summary.dropna(subset=["root_PFAF_ID", "root_level"])
    roots = {r.major: (int(r.root_PFAF_ID), int(r.root_level))
             for r in rooted.itertuples(index=False)}
    subnets = {t: extract_subbasins(net, subs, t, roots) for t in thetas}
    need = [r for t in thetas for keep, *_ in subnets[t][1].values() for r in keep]
    majors, orig_km = {}, {}
    if WRITE_MAJOR_NETWORKS:
        outlines = major_outlines(net, args.hydrobasins_dir, summary)
        majors = {t: major_outline_networks(net, outlines, t) for t in thetas}
        orig_km = original_network_lengths(summary)
        need += [r for t in thetas for keep, *_ in majors[t][1].values() for r in keep]
        m0 = majors[theta][0]
        LOG.info("Major outlines: %d of %d with a network; flags %s", len(majors[theta][1]),
                 len(m0), m0["flags"].fillna("").str.split(";").explode()
                 .loc[lambda x: x != ""].value_counts().to_dict() or "none")
    LOG.info("Reading %d river geometries ...", len(set(need)))
    geoms = read_geometries(shp, net, need)
    polygons = read_subbasin_polygons(args.hydrobasins_dir, subnets[theta][0])

    results = {}
    for t in thetas:
        folder = out if t == theta else out / "sensitivity" / theta_tag(t)
        folder.mkdir(parents=True, exist_ok=True)
        basin_paths = None if t == theta else dict(zip(results[theta]["table"].unit_key,
                                                       results[theta]["table"].basin_shp))
        table = write_outputs(folder, net, geoms, subnets[t][0], subnets[t][1], polygons,
                              basin_paths)
        table.to_csv(folder / "03_subbasin_networks.csv", index=False, encoding="utf-8-sig")
        subbasin_master(table).to_csv(folder / "04_subbasin_master.csv", index=False,
                                      encoding="utf-8-sig")
        mtable = None
        if WRITE_MAJOR_NETWORKS:
            mtable = write_major_networks(folder, net, geoms, majors[t][0], majors[t][1],
                                          summary, bym, orig_km)
            mtable.to_csv(folder / "06_major_rule_networks.csv", index=False,
                          encoding="utf-8-sig")
        results[t] = dict(table=table, mtable=mtable)
        acc = table[table.decision == "ACCEPT"]
        LOG.info("theta %g: %d sub-basins written, median %.0f tributaries; %d major networks",
                 t, len(acc), acc.n_tributaries.median() if len(acc) else np.nan,
                 0 if mtable is None else int(mtable.river_shp.notna().sum()))

    table = results[theta]["table"]
    flag_counts = (table["flags"].fillna("").str.split(";").explode()
                   .loc[lambda x: x != ""].value_counts().to_dict())
    LOG.info("Sub-basin flags: %s; decisions: %s", flag_counts or "none",
             table.decision.value_counts().to_dict())
    sens_summary = {}
    for t in thetas:
        tb, mtb = results[t]["table"], results[t]["mtable"]
        nt = tb.loc[tb.decision == "ACCEPT"].get("n_tributaries", pd.Series(dtype=float))
        sens_summary[f"{t:g}"] = dict(
            folder=str(out if t == theta else out / "sensitivity" / theta_tag(t)),
            subbasin_tributaries_median=float(nt.median()) if len(nt) else None,
            subbasins_mainstem_only=int((nt == 0).sum()),
            major_networks=0 if mtb is None else int(mtb.river_shp.notna().sum()))
    result = dict(
        theta=theta, theta_source=source, sensitivity_thetas=sens,
        comparison_with_major_networks=calib,
        subbasins=int(len(table)),
        subbasins_accepted=int((table.decision == "ACCEPT").sum()),
        subbasins_flagged=int((table["flags"].fillna("") != "").sum()),
        subbasin_flag_counts={k: int(v) for k, v in flag_counts.items()},
        subbasins_review=table.loc[table.decision == "REVIEW", "unit_key"].tolist(),
        major_flags=({} if results[theta]["mtable"] is None else
                     {f: sorted(results[theta]["mtable"].loc[
                         results[theta]["mtable"]["flags"].fillna("").str.contains(f),
                         "major"].tolist())
                      for f in ("NO_OUTLINE", "NO_REACHES", "OUTLET_AREA_MISMATCH",
                                "INFLOW_FROM_OUTSIDE", "SECOND_SYSTEM")}),
        networks_by_theta=sens_summary,
        minutes=round((time.time() - t_start) / 60, 1),
        output_directory=str(out))
    (out / "theta.json").write_text(json.dumps(result, ensure_ascii=False, indent=2,
                                               default=float), encoding="utf-8")
    LOG.info("Completed: %s", json.dumps(result, ensure_ascii=False, default=float))


if __name__ == "__main__":
    main()
