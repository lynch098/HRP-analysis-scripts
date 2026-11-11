

"""Calculate historical and projected flood exposure: change_fe = (E2/P2)/(E1/P1).
Zero baseline exposure yields undefined ratios. Run --check before calculation."""

import json as _path_json
from pathlib import Path as _ConfigPath


_DATA_PATHS = _path_json.loads(
    (_ConfigPath(__file__).resolve().parents[1] / "paths.json").read_text(encoding="utf-8")
)
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass
import argparse
import hashlib
import json
import multiprocessing as mp
import os
import re
import shutil
import time
import traceback
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import pandas as pd
from scipy import stats
import geopandas as gpd
import shapely
import rasterio
from rasterio.crs import CRS
from rasterio.enums import Resampling
from rasterio.features import geometry_mask
from rasterio.merge import merge as rio_merge
from rasterio.transform import Affine, array_bounds
from rasterio.warp import reproject, transform_bounds
from rasterio.windows import Window, from_bounds


HRP_RUN = Path(_DATA_PATHS["HRP_RUN"])
REGION_CSV = HRP_RUN / "downstream_20261019_165337/regions.csv"
SOUTH_SHP = Path(_DATA_PATHS["SOUTH_SHP"])
OUTPUT_ROOT = Path(_DATA_PATHS["FLOOD_OUTPUT_ROOT"])


GLOBPOP_DIR = Path(_DATA_PATHS["POP_DIR"])
GLOBPOP_PATTERN = "GlobPOP_Count_30arc_{year}_I32.tiff"
HIST_YEARS = list(range(1990, 2021))


SSP_POP_DIR = Path(_DATA_PATHS["SSP_POP_DIR"])
SSP_POP_PATTERN = "SSP{ssp}/SPP{ssp}/SSP{ssp}_{year}.tif"
FUTURE_BASE_YEAR = 2020
FUTURE_YEARS = [2030, 2050, 2080]


AQUEDUCT_DIR = Path(_DATA_PATHS["AQUEDUCT_DIR"])
HAZARD_MANIFEST = None
AQUEDUCT_URL = "http://wri-projects.s3.amazonaws.com/AqueductFloodTool/download/v2/"
RETURN_PERIOD = 100
GCMS = ["GFDL-ESM2M", "HadGEM2-ES", "IPSL-CM5A-LR", "MIROC-ESM-CHEM", "NorESM1-M"]

DEPTH_THRESHOLD_M = 2.0
DEPTH_STRICT = True


WATER_MODE = "jrc_yearly"
WATER_JRC_DIR = Path(_DATA_PATHS["WATER_JRC_DIR"])
WATER_JRC_PATTERN = "global_sw{year}.tif"
WATER_JRC_YEARS = list(range(1990, 2021))
WATER_JRC_MIN_SHARE = 0.5
WATER_RASTER = Path(_DATA_PATHS["WATER_RASTER"])
WATER_RULE = ("==", 1)
WATER_EXCLUDE_FRACTION = 0.5

SCENARIOS = {"SSP2-4.5": (2, "rcp4p5"), "SSP2-8.5": (2, "rcp8p5"), "SSP3-8.5": (3, "rcp8p5")}


FIG3A_INTERVAL = (1990, 2020)
HIST_INTERVALS = [(1990, 1995), (1995, 2000), (2000, 2005), (2005, 2010), (2010, 2015), (2015, 2020)]
HIST_CUMULATIVE_YEARS = [1995, 2000, 2005, 2010, 2015, 2020]
FUTURE_INTERVALS = [(2020, 2030), (2030, 2050), (2050, 2080)]
MIN_BASE_EXPOSED = 0.0
LOW_BASE_FLAG = 1000.0


GDP_MODE = "total_raster"
GDP_RASTER = Path(_DATA_PATHS["GDP_RASTER"])
GDP_BAND = {1990: 1, 2020: 7}
GDP_TABLE = None
GDP_YEARS = (1990, 2020)
GDP_THRESHOLD = 10000.0
GDP_SENSITIVITY = [5000.0, 7500.0, 10000.0, 12500.0, 15000.0]

ALL_TOUCHED = False
ALIGN_TOL_PX = 0.01
WORKERS = 1
METHOD_VERSION = "flood-exposure-v3.1-2026-10-19"
GDP_METHOD_VERSION = "native-grid-gdp-v3.2-2026-10-20"
TYPES = ["Near-river", "Dispersion", "Far-river"]
OLD_VALUES = {
    "south_Q2_pct": 80.47, "north_Q1_pct": 59.02, "Q2_dispersion_n": 61, "Q2_near_n": 39,
    "near_first_interval_south": 1.38, "near_first_interval_north": 1.05,
}
HIST_KEY = "historical|WATCH|0"
VALID_HAZ_KEY = "valid|hazard|0"
VALID_WATER_KEY = "valid|water|0"
WATER_FRAC_KEY = "water|fraction|0"
AQ_RE = re.compile(r"^inunriver_(historical|rcp\dp\d)_([0-9A-Za-z\-]+)_(\d{4})_rp(\d+)\.tiff?$", re.I)
AQ_FIELD = {"GFDL-ESM2M": "0000GFDL-ESM2M", "HadGEM2-ES": "0000HadGEM2-ES", "IPSL-CM5A-LR": "00IPSL-CM5A-LR",
            "MIROC-ESM-CHEM": "MIROC-ESM-CHEM", "NorESM1-M": "00000NorESM1-M"}
MAX_SRC_PIXELS = 32_000_000
BAND_ROWS = 256


def ts():
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def safe_name(s):
    return "".join(ch if ch.isalnum() or ch in "-_." else "_" for ch in str(s))


def fingerprint(path):
    p = Path(path)
    h = hashlib.sha1()
    if p.is_dir():
        for f in sorted(list(p.rglob("*.tif")) + list(p.rglob("*.tiff")) + list(p.rglob("*.vrt"))):
            st = f.stat()
            h.update(f"{f.relative_to(p)}|{st.st_size}|{st.st_mtime_ns}".encode())
    else:
        st = p.stat()
        h.update(f"{p.name}|{st.st_size}|{st.st_mtime_ns}".encode())
    return h.hexdigest()[:16]


def shp_fingerprint(shp):
    shp = Path(shp)
    h = hashlib.sha1()
    for f in sorted(shp.parent.glob(shp.stem + ".*")):
        st = f.stat()
        h.update(f"{f.name}|{st.st_size}|{st.st_mtime_ns}".encode())
    return h.hexdigest()[:16]


def sha(obj):
    return hashlib.sha1(json.dumps(obj, sort_keys=True, default=str).encode()).hexdigest()[:16]


def as_crs(c):
    return c if isinstance(c, CRS) else CRS.from_user_input(c)


def same_crs(a, b):
    a, b = as_crs(a), as_crs(b)
    if a == b:
        return True
    if a.is_geographic and b.is_geographic:
        return (a.to_epsg() or 4326) == (b.to_epsg() or 4326)
    return False


@dataclass
class Grid:
    crs: str
    transform: tuple
    width: int
    height: int

    @property
    def tr(self):
        return Affine(*self.transform[:6])

    @property
    def sig(self):
        return sha([as_crs(self.crs).to_wkt(), [round(v, 12) for v in self.transform[:6]], self.width, self.height])

    def as_dict(self):
        return dict(crs=self.crs, transform=tuple(self.transform), width=self.width, height=self.height)


def grid_of(path):
    with rasterio.open(path) as s:
        crs = s.crs if s.crs else CRS.from_epsg(4326)
        return Grid(crs.to_wkt(), tuple(s.transform)[:6], s.width, s.height)


def pixel_offset(src_tr, dst_tr):
    return (dst_tr.c - src_tr.c) / src_tr.a, (dst_tr.f - src_tr.f) / src_tr.e


def offset_note(h, g):

    if not same_crs(h.crs, g.crs):
        return "different CRS"
    if abs(h.tr.a - g.tr.a) > 1e-9 * abs(h.tr.a) or abs(h.tr.e - g.tr.e) > 1e-9 * abs(h.tr.e):
        return f"different cell size ({g.tr.a:.6g} vs {h.tr.a:.6g})"
    dx, dy = pixel_offset(h.tr, g.tr)
    return f"origin offset {dx - round(dx):+.3f} / {dy - round(dy):+.3f} px"


def aligned_offset(src_tr, src_crs, src_size, dst_tr, dst_crs, tol=None):

    tol = ALIGN_TOL_PX if tol is None else tol
    if not same_crs(src_crs, dst_crs):
        return None
    if max(abs(src_tr.b), abs(src_tr.d), abs(dst_tr.b), abs(dst_tr.d)) > 1e-12:
        return None
    n = max(src_size)
    if abs(src_tr.a - dst_tr.a) * n > tol * abs(dst_tr.a) or abs(src_tr.e - dst_tr.e) * n > tol * abs(dst_tr.e):
        return None
    dx, dy = pixel_offset(src_tr, dst_tr)
    if abs(dx - round(dx)) > tol or abs(dy - round(dy)) > tol:
        return None
    return int(round(dx)), int(round(dy))


def read_aligned(src, off, shape, band=1, fill=np.nan):

    c0, r0 = off
    h, w = shape
    out = np.full((h, w), fill, np.float32)
    rs, re_ = max(r0, 0), min(r0 + h, src.height)
    cs, ce = max(c0, 0), min(c0 + w, src.width)
    if rs < re_ and cs < ce:
        a = src.read(band, window=Window(cs, rs, ce - cs, re_ - rs), masked=True)
        out[rs - r0:re_ - r0, cs - c0:ce - c0] = np.ma.filled(a.astype(np.float32), fill)
    return out


class Source:


    def __init__(self, path, band=1):
        self.path, self.band = Path(path), band
        if self.path.is_dir():
            files = sorted(list(self.path.rglob("*.tif")) + list(self.path.rglob("*.tiff")))
            if not files:
                raise FileNotFoundError(f"No .tif under {self.path}")
            self.index = []
            for f in files:
                with rasterio.open(f) as s:
                    self.index.append((str(f), tuple(s.bounds)))
                    if len(self.index) == 1:
                        self.crs, self.res = (s.crs or CRS.from_epsg(4326)), s.res
            bb = [b for _, b in self.index]
            for i in range(len(bb)):
                for j in range(i + 1, len(bb)):
                    a, b = bb[i], bb[j]
                    w = min(a[2], b[2]) - max(a[0], b[0])
                    h = min(a[3], b[3]) - max(a[1], b[1])
                    small = min((a[2] - a[0]) * (a[3] - a[1]), (b[2] - b[0]) * (b[3] - b[1]))
                    if w > 0 and h > 0 and w * h > 0.01 * small:
                        raise ValueError(f"Overlapping rasters in {self.path}; possible time series cannot be mosaicked as spatial tiles: "
                                         f"{self.index[i][0]} / {self.index[j][0]}")
            self.single = None
        else:
            self.single = rasterio.open(self.path)
            self.crs, self.res = (self.single.crs or CRS.from_epsg(4326)), self.single.res

    def close(self):
        if self.single is not None:
            self.single.close()

    def read_bounds(self, bounds, pad=2):
        l, b, r, t = bounds
        if self.single is not None:
            s = self.single
            win = from_bounds(l, b, r, t, transform=s.transform)
            c0 = max(int(np.floor(win.col_off)) - pad, 0)
            r0 = max(int(np.floor(win.row_off)) - pad, 0)
            c1 = min(int(np.ceil(win.col_off + win.width)) + pad, s.width)
            r1 = min(int(np.ceil(win.row_off + win.height)) + pad, s.height)
            if c0 >= c1 or r0 >= r1:
                return None, None
            w = Window(c0, r0, c1 - c0, r1 - r0)
            a = s.read(self.band, window=w, masked=True)
            return np.ma.filled(a.astype(np.float32), np.nan), s.window_transform(w)
        px, py = self.res
        bb = (l - pad * px, b - pad * py, r + pad * px, t + pad * py)
        hits = [f for f, fb in self.index if fb[0] < bb[2] and fb[2] > bb[0] and fb[1] < bb[3] and fb[3] > bb[1]]
        if not hits:
            return None, None
        dss = [rasterio.open(f) for f in hits]
        try:
            arr, tr = rio_merge(dss, bounds=bb, res=self.res, nodata=np.nan, dtype="float32", indexes=[self.band])
        finally:
            for d in dss:
                d.close()
        return arr[0], tr

    def ratio_to(self, grid):

        l, b, r, t = array_bounds(grid.height, grid.width, grid.tr)
        if not same_crs(self.crs, grid.crs):
            l, b, r, t = transform_bounds(grid.crs, self.crs, l, b, r, t, densify_pts=21)
        return max((r - l) / self.res[0] / grid.width, (t - b) / self.res[1] / grid.height)


def fraction_on_tile(source, fn, t_tr, t_shape, t_crs):

    h, w = t_shape
    if source.single is not None:
        s = source.single
        off = aligned_offset(s.transform, source.crs, (s.width, s.height), t_tr, t_crs)
        if off is not None:
            return fn(read_aligned(s, off, t_shape, band=source.band, fill=np.nan))
    l, b, r, t = array_bounds(h, w, t_tr)
    if not same_crs(source.crs, t_crs):
        l, b, r, t = transform_bounds(t_crs, source.crs, l, b, r, t, densify_pts=21)
    arr, s_tr = source.read_bounds((l, b, r, t))
    out = np.zeros((h, w), np.float32)
    if arr is None:
        return out
    reproject(fn(arr).astype(np.float32), out, src_transform=s_tr, src_crs=source.crs,
              dst_transform=t_tr, dst_crs=as_crs(t_crs), resampling=Resampling.average)
    return np.clip(out, 0, 1)


def make_fn(kind, params):
    if kind == "depth":
        threshold, strict = params

        def fn(a):
            with np.errstate(invalid="ignore"):
                hit = (a > threshold) if strict else (a >= threshold)
            return (hit & np.isfinite(a)).astype(np.float32)
        return fn
    if kind == "water":
        op, val = params

        def fn(a):
            ok = np.isfinite(a)
            with np.errstate(invalid="ignore"):
                if op == "nonzero":
                    hit = a != 0
                elif op == ">=":
                    hit = a >= val
                elif op == ">":
                    hit = a > val
                elif op == "==":
                    hit = a == val
                elif op == "between":
                    hit = (a >= val[0]) & (a <= val[1])
                elif op == "in":
                    hit = np.isin(a, list(val))
                else:
                    raise ValueError(f"Unknown water rule {params}")
            return (hit & ok).astype(np.float32)
        return fn
    if kind == "valid":
        return lambda a: np.isfinite(a).astype(np.float32)
    raise ValueError(kind)


def mann_whitney(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    a, b = a[np.isfinite(a)], b[np.isfinite(b)]
    if len(a) < 2 or len(b) < 2:
        return np.nan
    return float(stats.mannwhitneyu(a, b, alternative="two-sided").pvalue)


def wilcoxon_paired(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    ok = np.isfinite(a) & np.isfinite(b)
    if ok.sum() < 5 or np.allclose(a[ok], b[ok]):
        return np.nan
    return float(stats.wilcoxon(a[ok], b[ok]).pvalue)


def bh(p):
    p = np.asarray(p, float)
    q = np.full(p.shape, np.nan)
    ok = np.isfinite(p)
    if ok.any():
        pv = p[ok]
        o = np.argsort(pv)
        r = pv[o] * len(pv) / np.arange(1, len(pv) + 1)
        r = np.minimum.accumulate(r[::-1])[::-1]
        qq = np.empty_like(pv)
        qq[o] = np.minimum(r, 1)
        q[ok] = qq
    return q


def stars(p):
    if not np.isfinite(p):
        return ""
    return "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""


def polygons(g):
    if g is None or g.is_empty:
        return []
    if g.geom_type == "Polygon":
        return [g]
    if hasattr(g, "geoms"):
        return [p for sub in g.geoms for p in polygons(sub)]
    return []


def load_polygon(shp, crs):
    gdf = gpd.read_file(shp)
    if gdf.crs is None:
        raise RuntimeError(f"Shapefile has no CRS: {shp}")
    gdf = gdf[gdf.geometry.notna() & ~gdf.geometry.is_empty]
    if not same_crs(gdf.crs.to_wkt(), crs):
        gdf = gdf.to_crs(as_crs(crs).to_wkt())
    parts = [p for g in gdf.geometry for p in polygons(g if g.is_valid else shapely.make_valid(g))]
    if not parts:
        raise RuntimeError(f"No polygon geometry: {shp}")
    m = shapely.union_all(parts)
    return m if m.is_valid else shapely.make_valid(m)


def assign_regions(units):

    south = gpd.read_file(SOUTH_SHP)
    if south.crs is None:
        b = south.total_bounds
        if not (-180.5 <= b[0] <= b[2] <= 180.5 and -90.5 <= b[1] <= b[3] <= 90.5):
            raise RuntimeError(f"South.shp has no CRS and bounds {b} are not lon/lat")
        south = south.set_crs("EPSG:4326")
    sp = shapely.union_all([p for g in south.to_crs("EPSG:6933").geometry
                            for p in polygons(g if g.is_valid else shapely.make_valid(g))])
    rows = []
    for r in units.itertuples(index=False):
        g = load_polygon(r.basin_shp, "EPSG:6933")
        ratio = float(np.clip(g.intersection(sp).area / g.area, 0, 1))
        rows.append(dict(unit_key=r.unit_key, south_area_ratio=ratio, region="South" if ratio > 0.5 else "North"))
    return pd.DataFrame(rows)


def load_units(hrp_run, region_csv):
    st = pd.read_csv(Path(hrp_run) / "main" / "static_hrp.csv")
    need = {"unit_key", "level", "HRP_type", "basin_shp"}
    if not need <= set(st.columns):
        raise ValueError(f"static_hrp.csv lacks {need - set(st.columns)}")
    st["unit_key"] = st.unit_key.astype(str)
    if st.unit_key.duplicated().any():
        raise ValueError("Duplicated unit_key in static_hrp.csv")
    missing = [p for p in st.basin_shp if not Path(p).is_file()]
    if missing:
        raise FileNotFoundError(f"{len(missing)} basin shapefiles missing, e.g. {missing[:3]}")
    st["major"] = st.unit_key.str.split("__").str[0]
    if region_csv is None:
        cands = sorted(Path(hrp_run).glob("downstream_*/regions.csv"))
        region_csv = cands[-1] if cands else None
    if region_csv is not None:
        reg = pd.read_csv(region_csv)
        reg["unit_key"] = reg.unit_key.astype(str)
        src = str(region_csv)
    else:
        print("No regions.csv found; assigning North/South with South.shp", flush=True)
        reg = assign_regions(st[["unit_key", "basin_shp"]])
        src = str(SOUTH_SHP)
    keep = ["unit_key", "region"] + (["south_area_ratio"] if "south_area_ratio" in reg.columns else [])
    st = st.merge(reg[keep], on="unit_key", how="left")
    if st.region.isna().any():
        raise ValueError(f"No region for {st.unit_key[st.region.isna()].tolist()[:10]}")
    if "south_area_ratio" not in st.columns:
        st["south_area_ratio"] = np.nan
    return st[["unit_key", "level", "major", "HRP_type", "region", "south_area_ratio", "basin_shp"]], src


def find_hazards():
    found, dups = {}, []
    if HAZARD_MANIFEST:
        m = pd.read_csv(HAZARD_MANIFEST)
        for r in m.itertuples(index=False):
            sc = str(r.scenario).lower()
            key = HIST_KEY if sc == "historical" else f"{sc}|{r.gcm}|{int(r.year)}"
            found[key] = str(r.path)
        return found, dups
    for p in sorted(Path(AQUEDUCT_DIR).rglob("inunriver_*")):
        m = AQ_RE.match(p.name)
        if not m or int(m.group(4)) != RETURN_PERIOD:
            continue
        sc = m.group(1).lower()
        gcm = m.group(2).lstrip("0")
        key = HIST_KEY if sc == "historical" else f"{sc}|{gcm}|{int(m.group(3))}"
        if key in found:
            dups.append((key, found[key], str(p)))
            continue
        found[key] = str(p)
    return found, dups


def required_hazards():
    rcps = sorted({rcp for _, rcp in SCENARIOS.values()})
    return [HIST_KEY] + [f"{rcp}|{g}|{y}" for rcp in rcps for g in GCMS for y in FUTURE_YEARS]


def match_gcm(found):
    lut = {g.lower(): g for g in GCMS}
    out = {}
    for k, v in found.items():
        if k == HIST_KEY:
            out[k] = v
            continue
        sc, g, y = k.split("|")
        out[f"{sc}|{lut.get(g.lower(), g)}|{y}"] = v
    return out


def gdp_paths():
    if GDP_MODE not in ("total_raster", "percap_raster") or GDP_RASTER is None:
        return None
    s = str(GDP_RASTER)
    return ({str(y): (s.format(year=y) if "{year}" in s else s) for y in GDP_YEARS},
            {str(y): (1 if "{year}" in s else int(GDP_BAND[y])) for y in GDP_YEARS})


def inventory():
    inv = dict(globpop={}, ssp={}, hazards={}, missing_hazards=[], problems=[], notes=[])
    for y in HIST_YEARS:
        p = Path(GLOBPOP_DIR) / GLOBPOP_PATTERN.format(year=y)
        if p.is_file():
            inv["globpop"][y] = str(p)
        else:
            inv["problems"].append(f"GlobPOP missing: {p}")
    for s in sorted({s for s, _ in SCENARIOS.values()}):
        for y in [FUTURE_BASE_YEAR] + FUTURE_YEARS:
            p = Path(SSP_POP_DIR) / SSP_POP_PATTERN.format(ssp=s, year=y)
            if p.is_file():
                inv["ssp"][(s, y)] = str(p)
            else:
                inv["problems"].append(f"SSP population missing: {p}")
    found, dups = find_hazards()
    found = match_gcm(found)
    for k, a, b in dups:
        inv["problems"].append(f"Duplicate hazard file for {k}: {a} | {b}")
    for k in required_hazards():
        if k in found:
            inv["hazards"][k] = found[k]
        else:
            inv["missing_hazards"].append(k)
            inv["problems"].append(f"Hazard missing: {k}")
    if WATER_MODE == "jrc_yearly":
        miss = [y for y in WATER_JRC_YEARS if not (Path(WATER_JRC_DIR) / WATER_JRC_PATTERN.format(year=y)).is_file()]
        if miss:
            inv["problems"].append(f"JRC yearly water missing for {miss[:10]} in {WATER_JRC_DIR}")
    elif WATER_MODE == "raster":
        if not Path(WATER_RASTER).exists():
            inv["problems"].append(f"Water raster missing: {WATER_RASTER}")
    else:
        inv["notes"].append("WATER_MODE is None: permanent water is not excluded")
    gp = gdp_paths()
    if GDP_MODE == "table":
        if not (GDP_TABLE and Path(GDP_TABLE).is_file()):
            inv["problems"].append(f"GDP table missing: {GDP_TABLE}")
    elif GDP_MODE and gp is None:
        inv["notes"].append("GDP_RASTER not set: Fig. 3a quadrants are skipped")
    elif gp:
        for y, p in gp[0].items():
            if not Path(p).is_file():
                inv["problems"].append(f"GDP raster missing: {p}")
                continue
            with rasterio.open(p) as s:
                if gp[1][y] > s.count:
                    inv["problems"].append(f"GDP band {gp[1][y]} for {y} exceeds {s.count} bands in {p}")
    return inv


def describe_raster(path, kind):
    with rasterio.open(path) as s:
        desc = [d for d in s.descriptions if d]
        return dict(kind=kind, path=str(path), crs=(s.crs.to_string() if s.crs else "missing (EPSG:4326 assumed)"),
                    width=s.width, height=s.height, res_x=s.res[0], res_y=s.res[1],
                    left=s.bounds.left, top=s.bounds.top, dtype=s.dtypes[0], nodata=s.nodata,
                    bands=s.count, band_names=";".join(desc[:8]), block=str(s.block_shapes[0]),
                    tiled=bool(s.block_shapes[0][1] < s.width), size_mb=round(Path(path).stat().st_size / 2**20, 1))


def check_grids(inv):
    grids = {}
    for name, files in [("GlobPOP", list(inv["globpop"].values())), ("SSP", list(inv["ssp"].values())),
                        ("hazard", list(inv["hazards"].values()))]:
        if not files:
            continue
        g0 = grid_of(files[0])
        bad = [f for f in files[1:] if grid_of(f) != g0]
        if bad:
            raise ValueError(f"{name} files do not share one grid, e.g. {bad[:2]}")
        grids[name] = g0
    return grids


def missing_urls(inv):
    urls = []
    for k in inv["missing_hazards"]:
        if k == HIST_KEY:
            name = f"inunriver_historical_000000000WATCH_1980_rp{RETURN_PERIOD:05d}.tif"
        else:
            sc, g, y = k.split("|")
            name = f"inunriver_{sc}_{AQ_FIELD.get(g, g.rjust(14, '0'))}_{y}_rp{RETURN_PERIOD:05d}.tif"
        urls.append(AQUEDUCT_URL + name)
    return urls


def download_missing(inv, ref_grid=None, retries=3):

    import urllib.request
    dest = Path(AQUEDUCT_DIR) / "downloaded_v2"
    dest.mkdir(parents=True, exist_ok=True)
    done, failed = [], []
    for url in missing_urls(inv):
        name = url.rsplit("/", 1)[1]
        out, part = dest / name, dest / (name + ".part")
        if out.is_file():
            done.append(name)
            continue
        for attempt in range(1, retries + 1):
            try:
                t0 = time.time()
                with urllib.request.urlopen(url, timeout=120) as r, open(part, "wb") as f:
                    total = int(r.headers.get("Content-Length") or 0)
                    got = 0
                    while True:
                        chunk = r.read(1 << 20)
                        if not chunk:
                            break
                        f.write(chunk)
                        got += len(chunk)
                if total and got != total:
                    raise IOError(f"incomplete download ({got} of {total} bytes)")
                g = grid_of(part)
                if ref_grid is not None and g != ref_grid:
                    raise IOError("grid differs from the local Aqueduct rasters")
                os.replace(part, out)
                done.append(name)
                print(f"  {name}: {got / 2**20:.0f} MB in {time.time() - t0:.0f}s", flush=True)
                break
            except Exception as e:
                part.unlink(missing_ok=True)
                status = getattr(e, "code", None)
                print(f"  {name}: attempt {attempt} failed ({e})", flush=True)
                if status == 404 or attempt == retries:
                    failed.append((name, str(e)))
                    break
                time.sleep(5 * attempt)
    return dest, done, failed


def build_jrc_permanent(unit_bounds):
    files = [Path(WATER_JRC_DIR) / WATER_JRC_PATTERN.format(year=y) for y in WATER_JRC_YEARS]
    g0 = grid_of(files[0])
    bad = [str(f) for f in files[1:] if grid_of(f) != g0]
    if bad:
        raise ValueError(f"JRC yearly files do not share one grid: {bad[:3]}")
    b = [transform_bounds("EPSG:4326", g0.crs, *x, densify_pts=21) if not same_crs("EPSG:4326", g0.crs) else x
         for x in unit_bounds.values()]
    crop = crop_grid(g0, b)
    out = OUTPUT_ROOT / "masks" / f"jrc_permanent_{crop.sig}.tif"
    tags = dict(method=METHOD_VERSION, kind="jrc_permanent", files=sha([fingerprint(f) for f in files]),
                years=f"{WATER_JRC_YEARS[0]}-{WATER_JRC_YEARS[-1]}", min_share=WATER_JRC_MIN_SHARE)
    if cached_ok(out, tags, crop):
        return out
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_name(out.stem + ".tmp.tif")
    c0 = int(round((crop.tr.c - g0.tr.c) / g0.tr.a))
    r0 = int(round((crop.tr.f - g0.tr.f) / g0.tr.e))
    t0 = time.time()
    srcs = [rasterio.open(f) for f in files]
    prof = dict(driver="GTiff", width=crop.width, height=crop.height, count=1, dtype="uint8", nodata=255,
                crs=as_crs(crop.crs), transform=crop.tr, tiled=True, blockxsize=256, blockysize=256,
                compress="deflate", BIGTIFF="IF_SAFER")
    try:
        with rasterio.open(tmp, "w", **prof) as dst:
            for r in range(0, crop.height, BAND_ROWS):
                n = min(BAND_ROWS, crop.height - r)
                n3 = np.zeros((n, crop.width), np.uint8)
                nobs = np.zeros((n, crop.width), np.uint8)
                for s in srcs:
                    a = s.read(1, window=Window(c0, r0 + r, crop.width, n))
                    nobs += ((a >= 1) & (a <= 3)).astype(np.uint8)
                    n3 += (a == 3).astype(np.uint8)
                perm = (n3.astype(np.float32) >= WATER_JRC_MIN_SHARE * nobs.astype(np.float32)).astype(np.uint8)
                perm[nobs == 0] = 255
                dst.write(perm, 1, window=Window(0, r, crop.width, n))
            dst.update_tags(**{k: str(v) for k, v in tags.items()})
    finally:
        for s in srcs:
            s.close()
    os.replace(tmp, out)
    print(f"  JRC permanent water {WATER_JRC_YEARS[0]}-{WATER_JRC_YEARS[-1]} built in {time.time() - t0:.0f}s", flush=True)
    return out


def lattice_for(pop_grid, haz_grid):
    off = aligned_offset(haz_grid.tr, haz_grid.crs, (haz_grid.width, haz_grid.height), pop_grid.tr, pop_grid.crs)
    return haz_grid if off is not None else pop_grid


def crop_grid(lattice, bounds_list, pad=2):
    l = min(b[0] for b in bounds_list)
    b_ = min(b[1] for b in bounds_list)
    r = max(b[2] for b in bounds_list)
    t = max(b[3] for b in bounds_list)
    win = from_bounds(l, b_, r, t, transform=lattice.tr)
    c0 = max(int(np.floor(win.col_off)) - pad, 0)
    r0 = max(int(np.floor(win.row_off)) - pad, 0)
    c1 = min(int(np.ceil(win.col_off + win.width)) + pad, lattice.width)
    r1 = min(int(np.ceil(win.row_off + win.height)) + pad, lattice.height)
    if c0 >= c1 or r0 >= r1:
        raise RuntimeError("Units do not overlap the raster grid")
    tr = rasterio.windows.transform(Window(c0, r0, c1 - c0, r1 - r0), lattice.tr)
    return Grid(lattice.crs, tuple(tr)[:6], c1 - c0, r1 - r0)


def boxes_px(grid, bounds_list):
    out = []
    for (l, b, r, t) in bounds_list:
        w = from_bounds(l, b, r, t, transform=grid.tr)
        out.append((int(np.floor(w.col_off)) - 1, int(np.floor(w.row_off)) - 1,
                    int(np.ceil(w.col_off + w.width)) + 1, int(np.ceil(w.row_off + w.height)) + 1))
    return sorted(set(out))


def cached_ok(path, tags, grid):
    if not Path(path).is_file():
        return False
    try:
        with rasterio.open(path) as s:
            t = s.tags()
            same = all(t.get(k) == str(v) for k, v in tags.items())
            return same and Grid(s.crs.to_wkt(), tuple(s.transform)[:6], s.width, s.height).sig == grid.sig
    except Exception:
        return False


def float_profile(grid):
    return dict(driver="GTiff", width=grid.width, height=grid.height, count=1, dtype="float32",
                crs=as_crs(grid.crs), transform=grid.tr, tiled=True, blockxsize=256, blockysize=256,
                compress="deflate", predictor=3, BIGTIFF="IF_SAFER")


def build_fraction_raster(out, grid, source, fn, boxes, tags, exclude=None, label=""):

    if cached_ok(out, tags, grid):
        return "cached"
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_name(out.stem + f".tmp{os.getpid()}.tif")
    ratio = source.ratio_to(grid) if source is not None else 1.0
    chunk = grid.width if ratio <= 1.0 else int(np.clip(MAX_SRC_PIXELS / (BAND_ROWS * ratio * ratio), 16, grid.width))
    t0 = time.time()
    ex_src = rasterio.open(exclude[0]) if exclude else None
    try:
        with rasterio.open(tmp, "w", **float_profile(grid)) as dst:
            for r0 in range(0, grid.height, BAND_ROWS):
                r1 = min(r0 + BAND_ROWS, grid.height)
                band = np.zeros((r1 - r0, grid.width), np.float32)
                if source is not None and any(bx[1] < r1 and bx[3] > r0 for bx in boxes):
                    for c0 in range(0, grid.width, chunk):
                        c1 = min(c0 + chunk, grid.width)
                        if not any(bx[0] < c1 and bx[2] > c0 and bx[1] < r1 and bx[3] > r0 for bx in boxes):
                            continue
                        tr = rasterio.windows.transform(Window(c0, r0, c1 - c0, r1 - r0), grid.tr)
                        band[:, c0:c1] = fraction_on_tile(source, fn, tr, (r1 - r0, c1 - c0), grid.crs)
                    if ex_src is not None:
                        wv = ex_src.read(1, window=Window(0, r0, grid.width, r1 - r0))
                        band[wv >= exclude[1] - 1e-6] = 0
                dst.write(band, 1, window=Window(0, r0, grid.width, r1 - r0))
            dst.update_tags(**{k: str(v) for k, v in tags.items()})
    finally:
        if ex_src is not None:
            ex_src.close()
    os.replace(tmp, out)
    print(f"    {label} built in {time.time() - t0:.0f}s", flush=True)
    return "built"


def mask_task(t):
    src = Source(t["source"]) if t["source"] else None
    try:
        return build_fraction_raster(t["out"], Grid(**t["grid"]), src, make_fn(t["kind"], t["params"]),
                                     t["boxes"], t["tags"], exclude=t["exclude"], label=t["label"])
    finally:
        if src is not None:
            src.close()


def run_tasks(tasks, workers):
    if workers <= 1 or len(tasks) <= 1:
        return [mask_task(t) for t in tasks]
    out = []
    with ProcessPoolExecutor(max_workers=workers, mp_context=mp.get_context("spawn")) as ex:
        for fu in as_completed([ex.submit(mask_task, t) for t in tasks]):
            out.append(fu.result())
    return out


def prepare_masks(inv, grids, unit_bounds, workers):

    if WATER_MODE == "jrc_yearly":
        water_src, water_rule = str(build_jrc_permanent(unit_bounds)), ("==", 1)
    elif WATER_MODE == "raster":
        water_src, water_rule = str(WATER_RASTER), WATER_RULE
    else:
        water_src, water_rule = None, None
    wfp = fingerprint(water_src) if water_src else "none"
    out, todo = {}, {}
    for name in ["GlobPOP", "SSP"]:
        if name not in grids:
            continue
        lat = lattice_for(grids[name], grids["hazard"])
        b_lat = [transform_bounds("EPSG:4326", lat.crs, *b, densify_pts=21) if not same_crs("EPSG:4326", lat.crs) else b
                 for b in unit_bounds.values()]
        mg = crop_grid(lat, b_lat)
        todo.setdefault(mg.sig, (mg, b_lat))
        out[name] = mg.sig
    plans = {}
    for sig, (mg, b_lat) in todo.items():
        boxes = boxes_px(mg, b_lat)
        base = dict(method=METHOD_VERSION, boxes=sha(boxes))
        mdir = OUTPUT_ROOT / "masks" / f"grid_{mg.sig}"
        print(f"Mask grid {mg.sig}: {mg.width} x {mg.height} cells -> {mdir}", flush=True)
        wpath, wvalid = mdir / "water_fraction.tif", mdir / "water_valid_fraction.tif"
        stage1 = []
        if water_src:
            wt = dict(base, kind="water", source=water_src, source_fp=wfp, rule=json.dumps(water_rule))
            stage1 = [dict(out=str(wpath), grid=mg.as_dict(), source=water_src, kind="water", params=water_rule,
                           boxes=boxes, tags=wt, exclude=None, label="permanent water"),
                      dict(out=str(wvalid), grid=mg.as_dict(), source=water_src, kind="valid", params=None,
                           boxes=boxes, tags=dict(wt, kind="water_valid"), exclude=None, label="water coverage")]
        run_tasks(stage1, workers)
        excl = (str(wpath), WATER_EXCLUDE_FRACTION) if water_src else None
        paths, stage2 = {}, []
        for key, hp in inv["hazards"].items():
            ht = dict(base, kind="flood", source=Path(hp).name, source_fp=fingerprint(hp), depth=DEPTH_THRESHOLD_M,
                      strict=DEPTH_STRICT, water_fp=wfp, water_rule=json.dumps(water_rule),
                      water_exclude=WATER_EXCLUDE_FRACTION)
            mp_ = mdir / f"{Path(hp).stem}__{'gt' if DEPTH_STRICT else 'ge'}{DEPTH_THRESHOLD_M:g}m.tif"
            stage2.append(dict(out=str(mp_), grid=mg.as_dict(), source=hp, kind="depth",
                               params=(DEPTH_THRESHOLD_M, DEPTH_STRICT), boxes=boxes, tags=ht, exclude=excl,
                               label=Path(hp).name))
            paths[key] = str(mp_)
            if key == HIST_KEY:
                vp = mdir / f"{Path(hp).stem}__valid.tif"
                stage2.append(dict(out=str(vp), grid=mg.as_dict(), source=hp, kind="valid", params=None,
                                   boxes=boxes, tags=dict(base, kind="hazard_valid", source_fp=ht["source_fp"]),
                                   exclude=None, label="hazard coverage"))
                paths[VALID_HAZ_KEY] = str(vp)
        if water_src:
            paths[VALID_WATER_KEY] = str(wvalid)
            paths[WATER_FRAC_KEY] = str(wpath)
        states = run_tasks(stage2, workers)
        print(f"  {sum(s == 'cached' for s in states)} of {len(states)} masks reused from cache", flush=True)
        plans[sig] = (mg, paths)
    return {name: plans[sig] for name, sig in out.items()}, water_src


def make_ctx(geoms, grid):

    tr = grid.tr
    sub = crop_grid(grid, [g.bounds for g in geoms.values()], pad=1)
    c0 = int(round((sub.tr.c - tr.c) / tr.a))
    r0 = int(round((sub.tr.f - tr.f) / tr.e))
    h, w = sub.height, sub.width
    masks = {k: geometry_mask([g], out_shape=(h, w), transform=sub.tr, invert=True, all_touched=ALL_TOUCHED)
             for k, g in geoms.items()}
    union = np.zeros((h, w), bool)
    for m in masks.values():
        union |= m
    cells = np.flatnonzero(union)
    del union
    if cells.size == 0:
        raise RuntimeError(f"No population cell centre inside {list(geoms)}")
    pos = {}
    for k in list(masks):
        p = np.searchsorted(cells, np.flatnonzero(masks.pop(k))).astype(np.int32)
        if len(p) == 0:
            raise RuntimeError(f"No population cell centre inside {k}")
        pos[k] = None if len(p) == len(cells) else p
    return dict(window=Window(c0, r0, w, h), grid=sub, cells=cells, pos=pos)


def sub_sum(v, p):
    return v.sum(dtype=np.float64) if p is None else v[p].sum(dtype=np.float64)


def read_pop(path, ctx):
    with rasterio.open(path) as s:
        a = s.read(1, window=ctx["window"], masked=True)
    v = np.ma.filled(a, 0).ravel()[ctx["cells"]].astype(np.float64)
    v[~np.isfinite(v) | (v < 0)] = 0.0
    return v


def read_mask(path, ctx):
    g = ctx["grid"]
    with rasterio.open(path) as s:
        off = aligned_offset(s.transform, s.crs, (s.width, s.height), g.tr, g.crs)
        if off is None:
            raise RuntimeError(f"Mask {path} is not aligned with the population grid")
        a = read_aligned(s, off, (g.height, g.width), fill=0)
    return a.ravel()[ctx["cells"]].astype(np.float64)


def values_on_ctx(path, band, ctx, require_aligned=False):

    g = ctx["grid"]
    with rasterio.open(path) as s:
        s_crs = s.crs or CRS.from_epsg(4326)
        off = aligned_offset(s.transform, s_crs, (s.width, s.height), g.tr, g.crs)
        if off is not None:
            a = read_aligned(s, off, (g.height, g.width), band=band, fill=np.nan)
        elif require_aligned:
            raise RuntimeError("GDP_MODE='total_raster' needs the GDP raster on the GlobPOP grid")
        else:
            src = Source(path, band=band)
            try:
                l, b, r, t = array_bounds(g.height, g.width, g.tr)
                if not same_crs(s_crs, g.crs):
                    l, b, r, t = transform_bounds(g.crs, s_crs, l, b, r, t, densify_pts=21)
                arr, s_tr = src.read_bounds((l, b, r, t))
            finally:
                src.close()
            a = np.full((g.height, g.width), -1e30, np.float32)
            if arr is not None:
                reproject(np.where(np.isfinite(arr), arr, -1e30).astype(np.float32), a, src_transform=s_tr,
                          src_crs=s_crs, src_nodata=-1e30, dst_transform=g.tr, dst_crs=as_crs(g.crs),
                          dst_nodata=-1e30, resampling=Resampling.nearest)
            a = np.where(a <= -1e29, np.nan, a)
    v = a.ravel()[ctx["cells"]].astype(np.float64)
    v[v < 0] = np.nan
    return v


def contexts(job):
    geoms_by_crs, ctx_by_grid = {}, {}

    def get(grid):
        ck = as_crs(grid.crs).to_wkt()
        if ck not in geoms_by_crs:
            geoms_by_crs[ck] = {k: load_polygon(shp, grid.crs) for k, shp in job["units"]}
        if grid.sig not in ctx_by_grid:
            ctx_by_grid[grid.sig] = make_ctx(geoms_by_crs[ck], grid)
        return ctx_by_grid[grid.sig]
    return get


def process_group(job):
    t0 = time.time()
    rows = []
    get = contexts(job)
    for prod in job["products"]:
        ctx = get(Grid(**prod["grid"]))
        cache = {}

        def pop(y):
            if y in cache:
                return cache[y]
            v = read_pop(prod["files"][str(y)], ctx)
            if len(prod["files"]) <= 10:
                cache[y] = v
            return v

        for hkey, years in prod["combos"]:
            f = read_mask(prod["masks"][hkey], ctx)
            sc, gcm, hy = hkey.split("|")
            fsum = {k: float(sub_sum(f, p)) for k, p in ctx["pos"].items()}
            for y in years:
                pv = pop(y)
                pf = pv * f
                for k, p in ctx["pos"].items():
                    P = float(sub_sum(pv, p))
                    E = float(sub_sum(pf, p))
                    if E > P * (1 + 1e-9) + 1e-6:
                        raise RuntimeError(f"E > P for {k}")
                    n = len(ctx["cells"]) if p is None else len(p)
                    rows.append((k, prod["name"], prod["ssp"], y, sc, gcm, int(hy), P, E, fsum[k], n))
                del pf
    cols = ["unit_key", "pop_product", "ssp", "pop_year", "hazard", "gcm", "hazard_year", "P", "E",
            "flood_cells", "n_cells"]
    return dict(group=job["group"], table=pd.DataFrame(rows, columns=cols), seconds=time.time() - t0)


def read_values(path, band, ctx):

    with rasterio.open(path) as s:
        a = s.read(band, window=ctx["window"], masked=True)
    v = np.ma.filled(a.astype(np.float64), np.nan).ravel()[ctx["cells"]]
    v[~np.isfinite(v) | (v < 0)] = np.nan
    return v


def process_gdp(job):
    t0 = time.time()
    get = contexts(job)
    ctx = get(Grid(**job["grid"]))
    rows = []
    for y in job["years"]:
        pv = read_pop(job["pop"][str(y)], ctx)
        path, band = job["path"][str(y)], job["band"][str(y)]
        if job["mode"] == "total_raster":
            gctx = get(grid_of(path))
            gv = values_on_ctx(path, band, gctx, require_aligned=True)
            valid = np.isfinite(gv)
            g0 = np.where(valid, gv, 0.0)

            src = Source(path, band=band)
            try:
                pg = ctx["grid"]
                cf = fraction_on_tile(src, lambda a: (np.isfinite(a) & (a >= 0)).astype(np.float32),
                                      pg.tr, (pg.height, pg.width), pg.crs)
                pcov = cf.ravel()[ctx["cells"]].astype(np.float64)
            finally:
                src.close()
            for k, p in ctx["pos"].items():
                gp = gctx["pos"][k]
                pop_total = float(sub_sum(pv, p))
                gdp_total = float(sub_sum(g0, gp)) if sub_sum(valid, gp) > 0 else np.nan
                coverage = float(sub_sum(pv * pcov, p)) / pop_total if pop_total > 0 else np.nan
                val = gdp_total / pop_total if pop_total > 0 else np.nan
                rows.append((k, y, val, coverage, gdp_total, pop_total))
            continue
        gv = values_on_ctx(path, band, ctx)
        ok = np.isfinite(gv)
        pv_ok = np.where(ok, pv, 0.0)
        g0 = np.where(ok, gv, 0.0)
        num = g0 * pv_ok if job["mode"] == "percap_raster" else g0
        for k, p in ctx["pos"].items():
            tot, cov, s = float(sub_sum(pv, p)), float(sub_sum(pv_ok, p)), float(sub_sum(num, p))
            if job["mode"] == "percap_raster":
                val = s / cov if cov > 0 else np.nan
            else:
                val = s / tot if tot > 0 else np.nan
            rows.append((k, y, val, cov / tot if tot > 0 else np.nan, np.nan, tot))
    return dict(group=job["group"], table=pd.DataFrame(rows, columns=[
        "unit_key", "year", "gdp_pc", "gdp_pop_coverage", "gdp_total_native", "population_total"]),
                seconds=time.time() - t0)


def build_jobs(units, inv, grids, masks):
    ssps = sorted({s for s, _ in SCENARIOS.values()})
    rcps_of = {s: sorted({r for ss, r in SCENARIOS.values() if ss == s}) for s in ssps}
    prods = []
    if inv["globpop"]:
        mg, mpaths = masks["GlobPOP"]
        combos = [(HIST_KEY, HIST_YEARS)] + [(k, [HIST_YEARS[-1]]) for k in (VALID_HAZ_KEY, VALID_WATER_KEY) if k in mpaths]
        prods.append(dict(name="GlobPOP", ssp=0, grid=grids["GlobPOP"].as_dict(),
                          files={str(y): p for y, p in inv["globpop"].items()}, combos=combos,
                          masks={k: mpaths[k] for k, _ in combos}))
    if inv["ssp"] and ssps:
        mg, mpaths = masks["SSP"]
        for s in ssps:
            yrs = [FUTURE_BASE_YEAR] + FUTURE_YEARS
            combos = [(HIST_KEY, yrs)] + [(k, [FUTURE_BASE_YEAR]) for k in (VALID_HAZ_KEY, VALID_WATER_KEY) if k in mpaths]
            for rcp in rcps_of[s]:
                for y in FUTURE_YEARS:
                    for g in GCMS:
                        combos.append((f"{rcp}|{g}|{y}", [FUTURE_BASE_YEAR, y]))
            prods.append(dict(name=f"SSP{s}", ssp=s, grid=grids["SSP"].as_dict(),
                              files={str(y): inv["ssp"][(s, y)] for y in yrs}, combos=combos,
                              masks={k: mpaths[k] for k, _ in combos}))
    jobs = []
    for grp, g in units.groupby("major", sort=True):
        ulist = [(r.unit_key, r.basin_shp) for r in g.itertuples(index=False)]
        sig = sha(dict(v=METHOD_VERSION, units=[(k, shp_fingerprint(s)) for k, s in ulist], all_touched=ALL_TOUCHED,
                       prods=[dict(p, files={y: fingerprint(f) for y, f in p["files"].items()},
                                   masks={h: fingerprint(m) for h, m in p["masks"].items()}) for p in prods]))
        jobs.append(dict(group=grp, units=ulist, products=prods, sig=sig))
    return jobs


def build_gdp_jobs(units, inv, grids):
    gp = gdp_paths()
    if gp is None or "GlobPOP" not in grids:
        return []
    jobs = []
    for grp, g in units.groupby("major", sort=True):
        ulist = [(r.unit_key, r.basin_shp) for r in g.itertuples(index=False)]
        job = dict(group=grp, units=ulist, grid=grids["GlobPOP"].as_dict(), mode=GDP_MODE, years=list(GDP_YEARS),
                   pop={str(y): inv["globpop"][y] for y in GDP_YEARS}, path=gp[0], band=gp[1])
        job["sig"] = sha(dict(v=GDP_METHOD_VERSION, units=[(k, shp_fingerprint(s)) for k, s in ulist], mode=GDP_MODE,
                              band=gp[1], files={y: fingerprint(p) for y, p in gp[0].items()},
                              pop={y: fingerprint(p) for y, p in job["pop"].items()}, all_touched=ALL_TOUCHED))
        jobs.append(job)
    return jobs


def run_jobs(jobs, workers, func, tag):
    cdir = OUTPUT_ROOT / "cache"
    cdir.mkdir(parents=True, exist_ok=True)
    results, todo = {}, []
    dt = {"unit_key": str, "pop_product": str, "hazard": str, "gcm": str}
    for j in jobs:
        f = cdir / f"{safe_name(j['group'])}__{tag}__{j['sig']}.csv.gz"
        if f.is_file():
            results[j["group"]] = pd.read_csv(f, dtype=dt)
        else:
            todo.append((j, f))
    print(f"{tag}: {len(jobs)} groups ({len(jobs) - len(todo)} from cache, {len(todo)} to compute)", flush=True)
    errors = []

    def keep(res, f):
        res["table"].to_csv(f, index=False, compression="gzip")
        results[res["group"]] = res["table"]

    if workers <= 1:
        for i, (j, f) in enumerate(todo, 1):
            try:
                res = func(j)
                keep(res, f)
                print(f"  [{i}/{len(todo)}] {j['group']}: {len(j['units'])} units, {res['seconds']:.0f}s", flush=True)
            except Exception:
                errors.append((tag, j["group"], traceback.format_exc()))
                print(f"  [{i}/{len(todo)}] {j['group']}: FAILED", flush=True)
    else:
        with ProcessPoolExecutor(max_workers=workers, mp_context=mp.get_context("spawn")) as ex:
            futs = {ex.submit(func, j): (j, f) for j, f in todo}
            for i, fu in enumerate(as_completed(futs), 1):
                j, f = futs[fu]
                try:
                    res = fu.result()
                    keep(res, f)
                    print(f"  [{i}/{len(todo)}] {j['group']}: {len(j['units'])} units, {res['seconds']:.0f}s", flush=True)
                except Exception:
                    errors.append((tag, j["group"], traceback.format_exc()))
                    print(f"  [{i}/{len(todo)}] {j['group']}: FAILED", flush=True)
    return results, errors


def write_clip(path, arr, grid, dtype="float32"):
    with rasterio.open(path, "w", driver="GTiff", width=grid.width, height=grid.height, count=1, dtype=dtype,
                       crs=as_crs(grid.crs), transform=grid.tr, compress="deflate") as d:
        d.write(arr.astype(dtype), 1)


def inspect_units(keys, all_units, inv, grids, masks):

    unknown = sorted(set(keys) - set(all_units.unit_key))
    if unknown:
        raise ValueError(f"Unknown inspect units: {unknown}")
    out = OUTPUT_ROOT / f"inspect_{ts()}"
    rows = []
    ssps = sorted({s for s, _ in SCENARIOS.values()})
    for key in keys:
        u = all_units[all_units.unit_key == key]
        if u.empty:
            print(f"  unknown unit: {key}", flush=True)
            continue
        d = out / safe_name(key)
        d.mkdir(parents=True, exist_ok=True)
        shp = u.basin_shp.iloc[0]
        for prod in ["GlobPOP", "SSP"]:
            if prod not in grids or prod not in masks:
                continue
            grid = grids[prod]
            ctx = make_ctx({key: load_polygon(shp, grid.crs)}, grid)
            g, win = ctx["grid"], ctx["window"]
            inside = np.zeros(g.height * g.width, bool)
            inside[ctx["cells"]] = True
            inside = inside.reshape(g.height, g.width)
            write_clip(d / f"{prod}_unit_mask.tif", inside, g, "uint8")
            _, mpaths = masks[prod]
            lay = {}
            for k, name in [(HIST_KEY, "flood_fraction_hist"), (VALID_HAZ_KEY, "hazard_valid_fraction"),
                            (WATER_FRAC_KEY, "water_fraction"), (VALID_WATER_KEY, "water_valid_fraction")]:
                if k in mpaths:
                    with rasterio.open(mpaths[k]) as s:
                        off = aligned_offset(s.transform, s.crs, (s.width, s.height), g.tr, g.crs)
                        lay[name] = read_aligned(s, off, (g.height, g.width), fill=0).astype(np.float64)
                    write_clip(d / f"{prod}_{name}.tif", lay[name], g)
            raw = None
            if "hazard" in grids and HIST_KEY in inv["hazards"]:
                h = grids["hazard"]
                off = aligned_offset(h.tr, h.crs, (h.width, h.height), g.tr, g.crs)
                if off is not None:
                    with rasterio.open(inv["hazards"][HIST_KEY]) as s:
                        raw = read_aligned(s, off, (g.height, g.width), fill=np.nan).astype(np.float64)
                    write_clip(d / f"{prod}_depth_hist_m.tif", raw, g)
            before_water = None
            if HIST_KEY in inv["hazards"]:
                src = Source(inv["hazards"][HIST_KEY])
                try:
                    before_water = fraction_on_tile(src, make_fn("depth", (DEPTH_THRESHOLD_M, DEPTH_STRICT)),
                                                    g.tr, (g.height, g.width), g.crs)
                finally:
                    src.close()
                write_clip(d / f"{prod}_flood_fraction_hist_before_water.tif", before_water, g)
            pops = ({str(y): inv["globpop"][y] for y in (HIST_YEARS[0], HIST_YEARS[-1])} if prod == "GlobPOP"
                    else {f"SSP{s}_{FUTURE_BASE_YEAR}": inv["ssp"][(s, FUTURE_BASE_YEAR)] for s in ssps})
            for label, path in pops.items():
                with rasterio.open(path) as s:
                    a = s.read(1, window=win, masked=True)
                pop = np.ma.filled(a.astype(np.float64), 0)
                pop[~np.isfinite(pop) | (pop < 0)] = 0
                write_clip(d / f"{prod}_pop_{label}.tif", pop, g)
                pi = pop[inside]
                P = float(pi.sum())

                def share(arr):
                    return 100 * float((pi * arr[inside]).sum()) / P if P > 0 and arr is not None else np.nan

                f = lay.get("flood_fraction_hist")
                wf = lay.get("water_fraction")
                rec = dict(unit_key=key, grid=prod, population=label, cells=int(inside.sum()), P=P,
                           E_hist=float((pi * f[inside]).sum()) if f is not None else np.nan,
                           flood_fraction_sum=float(f[inside].sum()) if f is not None else np.nan,
                           pop_on_hazard_nodata_pct=100 - share(lay.get("hazard_valid_fraction")),
                           pop_on_water_noobs_pct=(100 - share(lay.get("water_valid_fraction"))
                                                   if "water_valid_fraction" in lay else np.nan),
                           pop_in_excluded_water_cells_pct=(share((wf >= WATER_EXCLUDE_FRACTION - 1e-6).astype(float))
                                                            if wf is not None else np.nan))
                rec["E_hist_before_water_exclusion"] = (
                    float((pi * before_water[inside]).sum()) if before_water is not None else np.nan)
                if raw is not None:
                    with np.errstate(invalid="ignore"):
                        deep = ((raw > DEPTH_THRESHOLD_M) if DEPTH_STRICT else (raw >= DEPTH_THRESHOLD_M)) & np.isfinite(raw)
                    rec.update(deep_cells_raw=int(deep[inside].sum()),
                               wet_cells_raw=int(((raw > 0) & np.isfinite(raw))[inside].sum()))
                rows.append(rec)
        print(f"  {key}: clips written to {d}", flush=True)
    tab = pd.DataFrame(rows)
    out.mkdir(parents=True, exist_ok=True)
    tab.to_csv(out / "inspect_summary.csv", index=False, encoding="utf-8-sig")
    print(tab.to_string(index=False))
    print(f"Inspect output: {out}")


def metrics(E1, P1, E2, P2, y1, y2):
    E1, P1, E2, P2 = (np.asarray(x, float) for x in (E1, P1, E2, P2))
    with np.errstate(divide="ignore", invalid="ignore"):
        e1 = np.where(P1 > 0, E1 / P1, np.nan)
        e2 = np.where(P2 > 0, E2 / P2, np.nan)
        ok = (E1 > MIN_BASE_EXPOSED) & (P1 > 0) & (P2 > 0)
        fe = np.where(ok, e2 / e1, np.nan)
        ann = 100 * (np.power(fe, 1.0 / (y2 - y1)) - 1)
        RE = np.where(E1 > 0, E2 / E1, np.nan)
        RP = np.where(P1 > 0, P2 / P1, np.nan)
        arch = np.where(P1 - E1 != 0, (E2 - P2) / (E1 - P1), np.nan)
    zero = E1 <= MIN_BASE_EXPOSED
    status = np.where(~zero, "ok", np.where(E2 > 0, "new_exposure", "zero_at_both_endpoints"))
    return dict(E1=E1, P1=P1, E2=E2, P2=P2, share1_pct=100 * e1, share2_pct=100 * e2,
                change_fe=fe, change_fe_annual_pct=ann, dE=E2 - E1, dshare_pp=100 * (e2 - e1),
                R_E=RE, R_P=RP, archived_index=arch, zero_base=zero, zero_status=status,
                low_base=E1 < LOW_BASE_FLAG)


def build_indices(lv):

    out, by_gcm = [], []
    h = lv[(lv.pop_product == "GlobPOP") & (lv.hazard == "historical")]
    E = h.pivot(index="unit_key", columns="pop_year", values="E")
    P = h.pivot(index="unit_key", columns="pop_year", values="P")
    y0 = HIST_YEARS[0]
    hist_iv = HIST_INTERVALS + [(y0, y) for y in HIST_CUMULATIVE_YEARS] + [FIG3A_INTERVAL, (y0, HIST_YEARS[-1])]
    for y1, y2 in dict.fromkeys(hist_iv):
        if y1 not in E.columns or y2 not in E.columns:
            continue
        d = pd.DataFrame(metrics(E[y1], P[y1], E[y2], P[y2], y1, y2), index=E.index).reset_index()
        d.insert(1, "scenario", "historical")
        d.insert(2, "y1", y1)
        d.insert(3, "y2", y2)
        d["interval_type"] = "hist_chain" if (y1, y2) in HIST_INTERVALS else "hist_span"
        d["chain"] = (y1, y2) in HIST_INTERVALS
        d["from_base"] = y1 == y0
        d["n_gcm"] = 0
        d["n_gcm_valid"] = 0
        out.append(d)
    annual = (h.assign(share_pct=100 * h.E / h.P.where(h.P > 0))[["unit_key", "pop_year", "P", "E", "share_pct"]]
              .rename(columns={"pop_year": "year"}))
    b0 = annual[annual.year == HIST_YEARS[0]].set_index("unit_key")
    base = b0.share_pct.where(b0.E > MIN_BASE_EXPOSED)
    annual["share_rel_first"] = annual.share_pct / annual.unit_key.map(base)
    annual["rel_status"] = np.where(annual.unit_key.map(base).notna(), "ok", "zero_baseline")
    fut_levels, decomp = [], []
    fut_iv = list(dict.fromkeys(FUTURE_INTERVALS + [(FUTURE_BASE_YEAR, y) for y in FUTURE_YEARS]))
    for sname, (ssp, rcp) in SCENARIOS.items():
        L = lv[lv.pop_product == f"SSP{ssp}"]
        if L.empty:
            continue
        hb = L[L.hazard == "historical"]
        Eh = hb.pivot(index="unit_key", columns="pop_year", values="E")
        Ph = hb.pivot(index="unit_key", columns="pop_year", values="P")
        cp = L[(L.hazard == rcp) & (L.pop_year == L.hazard_year)]
        cc = L[(L.hazard == rcp) & (L.pop_year == FUTURE_BASE_YEAR)]
        Eg = cp.pivot_table(index="unit_key", columns=["hazard_year", "gcm"], values="E")
        Ecc = cc.pivot_table(index="unit_key", columns=["hazard_year", "gcm"], values="E")
        Ey = {FUTURE_BASE_YEAR: Eh[FUTURE_BASE_YEAR]}
        for y in FUTURE_YEARS:
            gs = sorted(Eg[y].columns)
            if gs != sorted(GCMS):
                raise RuntimeError(f"{sname} {y}: GCMs {gs} differ from GCMS {sorted(GCMS)}")
            Ey[y] = Eg[y][gs].mean(axis=1)
            idx = Ey[y].index
            fut_levels.append(pd.DataFrame(dict(unit_key=idx, scenario=sname, year=y, E=Ey[y].values,
                                                P=Ph[y].reindex(idx).values, n_gcm=len(gs))))
            decomp.append(pd.DataFrame(dict(
                basin_name=idx, scenario=sname, period=y, P0=Eh[FUTURE_BASE_YEAR].reindex(idx).values,
                P_cc=Ecc[y][gs].mean(axis=1).reindex(idx).values, P_pg=Eh[y].reindex(idx).values,
                P_coupled=Ey[y].values, P_total_base=Ph[FUTURE_BASE_YEAR].reindex(idx).values,
                P_total_future=Ph[y].reindex(idx).values, n_gcm=len(gs))))
        fut_levels.append(pd.DataFrame(dict(unit_key=Eh.index, scenario=sname, year=FUTURE_BASE_YEAR,
                                            E=Eh[FUTURE_BASE_YEAR].values, P=Ph[FUTURE_BASE_YEAR].values, n_gcm=0)))
        for y1, y2 in fut_iv:
            idx = Ey[y2].index
            m = metrics(Ey[y1].reindex(idx), Ph[y1].reindex(idx), Ey[y2], Ph[y2].reindex(idx), y1, y2)
            d = pd.DataFrame(m, index=idx).reset_index()
            d.insert(1, "scenario", sname)
            d.insert(2, "y1", y1)
            d.insert(3, "y2", y2)
            d["interval_type"] = "future_chain" if (y1, y2) in FUTURE_INTERVALS else "from_base"
            d["chain"] = (y1, y2) in FUTURE_INTERVALS
            d["from_base"] = y1 == FUTURE_BASE_YEAR
            d["n_gcm"] = len(GCMS)
            gcm_names = sorted(GCMS)
            gv = np.vstack([metrics((Eh[FUTURE_BASE_YEAR] if y1 == FUTURE_BASE_YEAR else Eg[y1][g]).reindex(idx),
                                    Ph[y1].reindex(idx), Eg[y2][g], Ph[y2].reindex(idx), y1, y2)["change_fe"]
                            for g in gcm_names])
            valid = np.isfinite(gv)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                d["change_fe_gcm_min"] = np.nanmin(gv, axis=0)
                d["change_fe_gcm_max"] = np.nanmax(gv, axis=0)
                d["change_fe_gcm_median"] = np.nanmedian(gv, axis=0)
                d["change_fe_gcm_sd"] = np.nanstd(gv, axis=0, ddof=1) if gv.shape[0] > 1 else np.nan

            d["n_gcm_valid"] = valid.sum(axis=0)
            d["gcm_range_complete"] = d.n_gcm_valid == len(gcm_names)
            d["gcm_above_1"] = (gv > 1).sum(axis=0)
            out.append(d)
            for gi, g in enumerate(gcm_names):
                by_gcm.append(pd.DataFrame(dict(unit_key=idx, scenario=sname, y1=y1, y2=y2, gcm=g,
                                                chain=(y1, y2) in FUTURE_INTERVALS, from_base=y1 == FUTURE_BASE_YEAR,
                                                change_fe=gv[gi])))
    idx = pd.concat(out, ignore_index=True)
    idx["interval"] = idx.y1.astype(str) + "-" + idx.y2.astype(str)
    gcm_tab = pd.concat(by_gcm, ignore_index=True) if by_gcm else pd.DataFrame()
    if len(gcm_tab):
        gcm_tab["interval"] = gcm_tab.y1.astype(str) + "-" + gcm_tab.y2.astype(str)
    return idx, annual, (pd.concat(fut_levels, ignore_index=True) if fut_levels else pd.DataFrame()), \
        (pd.concat(decomp, ignore_index=True) if decomp else pd.DataFrame()), gcm_tab


def expand(df):

    a = pd.concat([df, df.assign(level="major+sub")], ignore_index=True)
    return pd.concat([a, a.assign(HRP_type="All")], ignore_index=True)


def by_type(df):
    return pd.concat([df, df.assign(HRP_type="All")], ignore_index=True)


def describe(df, value, keys, geo=False):

    g = df.groupby(keys, dropna=False)[value]
    out = g.agg(n="count", mean="mean", sd="std", median="median").reset_index()
    out["q25"] = g.quantile(0.25).values
    out["q75"] = g.quantile(0.75).values
    out["se"] = out.sd / np.sqrt(out.n)
    if geo:
        d = df[keys].copy()
        d["_lg"] = np.log(df[value].where(df[value] > 0))
        lg = d.groupby(keys, dropna=False)["_lg"].agg(n_pos="count", lmean="mean", lsd="std").reset_index()
        half = stats.t.ppf(0.975, (lg.n_pos - 1).clip(lower=1)) * lg.lsd / np.sqrt(lg.n_pos)
        lg["geo_mean"] = np.exp(lg.lmean)
        lg["geo_ci_low"] = np.exp(lg.lmean - half)
        lg["geo_ci_high"] = np.exp(lg.lmean + half)
        out = out.merge(lg[keys + ["n_pos", "geo_mean", "geo_ci_low", "geo_ci_high"]], on=keys, how="left")
        out["n_nonpositive"] = out.n - out.n_pos
        if "level" in out:
            out.loc[out.level != "major", ["geo_ci_low", "geo_ci_high"]] = np.nan
    return out


def pooled(df, keys):
    """Pooled ratio of regional exposed shares; exclude combined nested levels to avoid double counting."""
    d = df[df.level.isin(["major", "sub"])]
    g = d.groupby(keys, dropna=False)[["E1", "P1", "E2", "P2"]].sum().reset_index()
    ok = (g.E1 > 0) & (g.P1 > 0) & (g.P2 > 0)
    g["pooled_share1_pct"] = 100 * g.E1 / g.P1.where(g.P1 > 0)
    g["pooled_share2_pct"] = 100 * g.E2 / g.P2.where(g.P2 > 0)
    g["pooled_change_fe"] = (g.pooled_share2_pct / g.pooled_share1_pct).where(ok)
    counts = d.assign(_zero=d.E1 <= MIN_BASE_EXPOSED).groupby(keys, dropna=False).agg(
        n_pooled=("unit_key", "size"), n_pooled_zero_baseline=("_zero", "sum")).reset_index()
    return g[keys + ["pooled_share1_pct", "pooled_share2_pct", "pooled_change_fe"]].merge(counts, on=keys)


def group_stats(rows, keys):

    st = describe(expand(rows), "change_fe", keys, geo=True)
    return st.merge(pooled(expand(rows), keys), on=keys, how="left")


def ns_tests(df, value, keys):

    d = by_type(df[df.level == "major"])
    rows = []
    for key, g in d.groupby(keys, dropna=False):
        a = g.loc[g.region == "North", value].dropna()
        b = g.loc[g.region == "South", value].dropna()
        rows.append({**dict(zip(keys, key if isinstance(key, tuple) else (key,))), "level": "major",
                     "n_North": len(a), "n_South": len(b), "mean_North": a.mean(), "mean_South": b.mean(),
                     "median_North": a.median(), "median_South": b.median(), "MW_p": mann_whitney(a, b)})
    t = pd.DataFrame(rows)
    if len(t):
        t["BH_q"] = bh(t.MW_p.to_numpy())
        t["stars_p"] = t.MW_p.map(stars)
        t["stars_q"] = t.BH_q.map(stars)
    return t


def quadrant(fe, gdp, thr):
    if not (np.isfinite(fe) and np.isfinite(gdp)):
        return np.nan
    if fe > 1:
        return "Q1" if gdp > thr else "Q2"
    return "Q4" if gdp > thr else "Q3"


def load_gdp_table(units):
    t = pd.read_csv(GDP_TABLE)
    t["unit_key"] = t.unit_key.astype(str)
    if {"gdp_pc_1990", "gdp_pc_2020"} <= set(t.columns):
        t["gdp_pc_mean"] = (t.gdp_pc_1990 + t.gdp_pc_2020) / 2
    elif "gdp_pc" in t.columns:
        t["gdp_pc_mean"] = t.gdp_pc
    else:
        raise ValueError("GDP table needs gdp_pc_1990 and gdp_pc_2020, or gdp_pc")
    miss = set(units.unit_key) - set(t.unit_key)
    if miss:
        raise ValueError(f"GDP table lacks {len(miss)} units, e.g. {sorted(miss)[:5]}")
    return t[["unit_key", "gdp_pc_mean"] + [c for c in ["gdp_pc_1990", "gdp_pc_2020"] if c in t.columns]]


def decomposition(decomp, U):

    d = decomp.merge(U.rename(columns={"unit_key": "basin_name", "HRP_type": "hrp_type"}), on="basin_name", how="left")
    base_ok = (d.P0 > 0).to_numpy()
    all_pos = (base_ok & (d.P_cc > 0) & (d.P_pg > 0) & (d.P_coupled > 0)
               & (d.P_total_base > 0) & (d.P_total_future > 0)).to_numpy()
    d["count_status"] = np.where(base_ok, "ok", "zero_baseline")
    d["index_status"] = np.where(all_pos, "ok", np.where(base_ok, "zero_counterfactual", "zero_baseline"))
    P0 = d.P0.where(base_ok)
    d["delta_cc"] = d.P_cc / P0 - 1
    d["delta_pg"] = d.P_pg / P0 - 1
    d["delta_total"] = d.P_coupled / P0 - 1
    d["delta_interaction"] = d.delta_total - d.delta_cc - d.delta_pg
    den = d.delta_cc.abs() + d.delta_pg.abs()
    den_i = den + d.delta_interaction.abs()
    d["C_climate"] = 100 * d.delta_cc.abs() / den.where(den > 0)
    d["C_population"] = 100 * d.delta_pg.abs() / den.where(den > 0)
    d["interaction_share_pct"] = 100 * d.delta_interaction.abs() / den_i.where(den_i > 0)
    e0 = P0 / d.P_total_base.where(d.P_total_base > 0)
    d["fe_climate"] = d.P_cc / P0
    d["fe_population"] = (d.P_pg / d.P_total_future.where(d.P_total_future > 0)) / e0
    d["fe_total"] = (d.P_coupled / d.P_total_future.where(d.P_total_future > 0)) / e0
    d["fe_interaction"] = (d.fe_total / (d.fe_climate * d.fe_population)).where(all_pos)
    lc, lp, li = (np.log(d[c].where(all_pos)).abs()
                  for c in ["fe_climate", "fe_population", "fe_interaction"])
    tot = (lc + lp + li).where(lambda s: s > 0)
    d["S_climate_pct"] = 100 * lc / tot
    d["S_population_pct"] = 100 * lp / tot
    d["S_interaction_pct"] = 100 * li / tot
    num = d.select_dtypes("number").columns
    d[num] = d[num].replace([np.inf, -np.inf], np.nan)
    rows = []
    for (lev, sc, per), g in d.groupby(["level", "scenario", "period"]):
        major = lev == "major"
        ok_i = g.index_status.eq("ok")
        rows.append(dict(level=lev, scenario=sc, period=per, n_units=len(g),
                         n_count=int(g.C_climate.notna().sum()), n_index=int(ok_i.sum()),
                         n_zero_baseline=int(g.count_status.eq("zero_baseline").sum()),
                         n_zero_counterfactual=int(g.index_status.eq("zero_counterfactual").sum()),
                         count_climate_median=g.C_climate.median(), count_population_median=g.C_population.median(),
                         count_interaction_median=g.interaction_share_pct.median(),
                         count_wilcoxon_p=wilcoxon_paired(g.C_population, g.C_climate) if major else np.nan,
                         index_climate_median=g.S_climate_pct.median(), index_population_median=g.S_population_pct.median(),
                         index_interaction_median=g.S_interaction_pct.median(),
                         index_wilcoxon_p=wilcoxon_paired(np.log(g.fe_population[ok_i]).abs(),
                                                          np.log(g.fe_climate[ok_i]).abs()) if major else np.nan))
    return d, pd.DataFrame(rows)


def ns_diff(st):

    piv = st.pivot_table(index=["scenario", "interval", "y2", "level", "HRP_type"], columns="region",
                         values=["n", "median", "geo_mean", "pooled_change_fe"]).reset_index()
    piv.columns = ["_".join(str(c) for c in col if str(c)) for col in piv.columns]
    for v in ["median", "geo_mean", "pooled_change_fe"]:
        if f"{v}_South" in piv and f"{v}_North" in piv:
            piv[f"{v}_South_minus_North"] = piv[f"{v}_South"] - piv[f"{v}_North"]
    return piv


def summaries(units, idx, gdp, decomp, gcm_tab):
    U = units.drop(columns=["basin_shp"])
    ix = idx.merge(U, on="unit_key", how="left")
    S = {}
    keys = ["scenario", "interval", "y1", "y2", "level", "HRP_type", "region"]
    hist = ix.scenario == "historical"
    fig3a = hist & (ix.y1 == FIG3A_INTERVAL[0]) & (ix.y2 == FIG3A_INTERVAL[1])

    chain = ix[ix.chain | fig3a].drop_duplicates(["unit_key", "scenario", "interval"])
    cum = ix[ix.from_base]
    S["fig3b_cumulative_stats"] = group_stats(cum, keys)
    S["fig3b_chain_stats"] = group_stats(chain, keys)
    S["fig3b_chain_annualized"] = describe(expand(chain), "change_fe_annual_pct", keys)
    S["fig3b_lowbase_excluded"] = pd.concat(
        [group_stats(cum[~cum.low_base], keys).assign(family="cumulative"),
         group_stats(chain[~chain.low_base], keys).assign(family="chain")], ignore_index=True)
    tk = ["scenario", "interval", "y1", "y2", "HRP_type"]
    S["fig3b_tests"] = pd.concat([ns_tests(cum, "change_fe", tk).assign(family="cumulative"),
                                  ns_tests(chain, "change_fe", tk).assign(family="chain")], ignore_index=True)
    one = expand(ix[ix.chain | ix.from_base].drop_duplicates(["unit_key", "scenario", "interval"]))
    one = one[one.HRP_type == "All"]
    S["zero_status"] = (one.groupby(["scenario", "interval", "level", "region", "zero_status"]).size()
                        .unstack(fill_value=0).reset_index())

    if len(gcm_tab):
        gt = gcm_tab.merge(U, on="unit_key", how="left")
        per = describe(expand(gt), "change_fe", keys + ["gcm"])
        S["fig3b_gcm_medians"] = per
        S["fig3b_gcm_spread"] = (per.groupby(keys, dropna=False)
                                 .agg(gcm_median_min=("median", "min"), gcm_median_max=("median", "max"),
                                      n_gcm=("median", "count"), n_units_min=("n", "min"), n_units_max=("n", "max"))
                                 .reset_index())

    trend = []
    fcs = S["fig3b_cumulative_stats"]
    for (sc, lev, ht, reg), g in fcs[fcs.scenario != "historical"].groupby(["scenario", "level", "HRP_type", "region"]):
        g = g.sort_values("y2")
        x = np.r_[FUTURE_BASE_YEAR, g.y2.to_numpy(float)]
        y = np.r_[1.0, g["median"].to_numpy(float)]
        if len(x) >= 3 and np.isfinite(y).all():
            trend.append(dict(scenario=sc, level=lev, HRP_type=ht, region=reg,
                              theil_sen_per_decade=10 * stats.theilslopes(y, x)[0], last_median=y[-1],
                              last_pooled=g.pooled_change_fe.iloc[-1],
                              note="slope of the median cumulative index (2020 = 1); descriptive"))
    S["future_trends"] = pd.DataFrame(trend)

    y1, y2 = HIST_YEARS[0], HIST_YEARS[-1]
    sh = ix[(ix.scenario == "historical") & (ix.y1 == y1) & (ix.y2 == y2)].copy()
    S["s12_units"] = sh[["unit_key", "level", "major", "region", "HRP_type", "share1_pct", "share2_pct",
                         "dshare_pp", "R_E", "R_P", "change_fe", "archived_index", "E1", "E2", "P1", "P2"]]
    s12, t12 = [], []
    for v in ["share1_pct", "share2_pct", "dshare_pp", "R_E", "change_fe"]:
        d = describe(expand(sh), v, ["level", "HRP_type", "region"])
        d.insert(0, "variable", v)
        s12.append(d)
        t = ns_tests(sh, v, ["HRP_type"])
        t.insert(0, "variable", v)
        t12.append(t)
    S["s12_stats"] = pd.concat(s12, ignore_index=True)
    S["s12_tests"] = pd.concat(t12, ignore_index=True)

    qa_units = ix[(ix.scenario == "historical") & (ix.y1 == FIG3A_INTERVAL[0]) & (ix.y2 == FIG3A_INTERVAL[1])].copy()
    qa_units["fe_group"] = np.where(qa_units.change_fe > 1, "Intensification",
                                    np.where(qa_units.change_fe <= 1, "Stabilization", "undefined"))
    if gdp is not None:
        qa_units = qa_units.merge(gdp, on="unit_key", how="left")
        qa_units["quadrant"] = [quadrant(a, b, GDP_THRESHOLD) for a, b in zip(qa_units.change_fe, qa_units.gdp_pc_mean)]
        qd = qa_units.dropna(subset=["quadrant"])
        rows = []
        for lev, g in pd.concat([qd, qd.assign(level="major+sub")]).groupby("level"):
            for (reg, ht), gg in by_type(g).groupby(["region", "HRP_type"]):
                for q in ["Q1", "Q2", "Q3", "Q4"]:
                    n = int((gg.quadrant == q).sum())
                    rows.append(dict(level=lev, region=reg, HRP_type=ht, quadrant=q, n=n, n_group=len(gg),
                                     pct=100 * n / len(gg) if len(gg) else np.nan))
        S["fig3a_counts"] = pd.DataFrame(rows)
        sens = []
        for thr in GDP_SENSITIVITY:
            d = qa_units.assign(q=[quadrant(a, b, thr) for a, b in zip(qa_units.change_fe, qa_units.gdp_pc_mean)])
            d = d.dropna(subset=["q"])
            for lev, g in pd.concat([d, d.assign(level="major+sub")]).groupby("level"):
                for reg, gg in g.groupby("region"):
                    for q in ["Q1", "Q2", "Q3", "Q4"]:
                        sens.append(dict(threshold=thr, level=lev, region=reg, quadrant=q, pct=100 * (gg.q == q).mean()))
        S["gdp_sensitivity"] = pd.DataFrame(sens)
    S["fig3a_units"] = qa_units
    S["fig3a_valid_n"] = (pd.concat([qa_units, qa_units.assign(level="major+sub")])
                          .groupby(["level", "region", "zero_status"]).size().unstack(fill_value=0).reset_index())
    valid = pd.concat([qa_units, qa_units.assign(level="major+sub")]).copy()
    valid["_index_valid"] = np.isfinite(valid.change_fe)
    valid["_gdp_valid"] = np.isfinite(valid.gdp_pc_mean) if "gdp_pc_mean" in valid else False
    valid["_quadrant_valid"] = valid.quadrant.notna() if "quadrant" in valid else False
    counts = valid.groupby(["level", "region"]).agg(
        n_total=("unit_key", "size"), n_index_valid=("_index_valid", "sum"),
        n_gdp_valid=("_gdp_valid", "sum"), n_quadrant_valid=("_quadrant_valid", "sum")).reset_index()
    counts["n_unassigned"] = counts.n_total - counts.n_quadrant_valid
    S["fig3a_valid_n"] = S["fig3a_valid_n"].merge(counts, on=["level", "region"])
    above = qa_units.assign(above=(qa_units.change_fe > 1).astype(float).where(qa_units.change_fe.notna()))
    S["fe_above1"] = describe(expand(above), "above", ["level", "HRP_type", "region"]).rename(columns={"mean": "share_above_1"})

    ov = []
    for fam, sel in [("cumulative", ix.from_base & ~hist), ("chain", ix.chain & ~hist)]:
        fut = ix[sel]
        if not fut.empty and {"SSP2-4.5", "SSP2-8.5"} <= set(fut.scenario):
            w = fut.pivot_table(index=["unit_key", "level", "interval", "y2"], columns="scenario", values="change_fe").reset_index()
            w["abs_gap"] = (w["SSP2-4.5"] - w["SSP2-8.5"]).abs()
            ov.append(describe(pd.concat([w, w.assign(level="major+sub")]), "abs_gap", ["level", "interval", "y2"])
                      .assign(family=fam))
    if ov:
        S["scenario_overlap"] = pd.concat(ov, ignore_index=True)
    S["ns_difference"] = pd.concat([ns_diff(S["fig3b_cumulative_stats"]).assign(family="cumulative"),
                                    ns_diff(S["fig3b_chain_stats"]).assign(family="chain")], ignore_index=True)
    if not decomp.empty:
        S["decomp"], S["decomp_summary"] = decomposition(decomp, U)
    return S


def manuscript_numbers(S, units):
    rows = []

    def add(item, where, old, new, basis=""):
        rows.append(dict(item=item, location=where, manuscript_value=old, new_value=new, basis=basis))

    mj = units[units.level == "major"]
    add("Majors North / South", "SI Table 1; all regional tests", "28 / 52",
        f"{(mj.region == 'North').sum()} / {(mj.region == 'South').sum()}", "regions.csv in use")
    s12 = S["s12_stats"]

    def med(var, lev, reg, ht="All"):
        r = s12[(s12.variable == var) & (s12.level == lev) & (s12.region == reg) & (s12.HRP_type == ht)]
        return (float(r["median"].iloc[0]), int(r.n.iloc[0])) if len(r) else (np.nan, 0)

    def tp(var, ht="All"):
        t = S["s12_tests"]
        r = t[(t.variable == var) & (t.HRP_type == ht)]
        return float(r.MW_p.iloc[0]) if len(r) else np.nan

    for lev in ["major", "major+sub"]:
        for var, label, where, old in [
                ("dshare_pp", f"Median change in exposed share {HIST_YEARS[0]}-{HIST_YEARS[-1]} (pp), South [X] / North [Y]",
                 "Results, flood paragraph 1", "[X] / [Y]"),
                ("share2_pct", f"Median exposed share {HIST_YEARS[-1]} (%), South / North", "Abstract; Results; Fig. S12",
                 "South higher"),
                ("R_E", f"Median growth of exposed population E{HIST_YEARS[-1]}/E{HIST_YEARS[0]}, South / North",
                 "Abstract; Results; Fig. S12", "South greater"),
                ("change_fe", f"Median change_fe {HIST_YEARS[0]}-{HIST_YEARS[-1]}, South / North", "Results; Fig. 3a",
                 "")]:
            (a, na), (b, nb) = med(var, lev, "South"), med(var, lev, "North")
            add(f"{label}, {lev}", where, old, f"{a:.3f} (n={na}) / {b:.3f} (n={nb})", f"MW p (majors) = {tp(var):.3g}")
    vn = S["fig3a_valid_n"]
    for lev in ["major", "sub", "major+sub"]:
        recs = vn[vn.level == lev].to_dict("records")
        add(f"Units with a defined {FIG3A_INTERVAL[0]}-{FIG3A_INTERVAL[1]} index, {lev}", "Fig. 3a and S9 denominators", "",
            "; ".join(f"{r['region']}: {int(r.get('ok', 0))} defined, {int(r.get('new_exposure', 0))} new exposure, "
                      f"{int(r.get('zero_at_both_endpoints', 0))} zero exposure at both endpoints" for r in recs),
            "units with zero baseline exposure have no index and are reported separately")
    if "fig3a_counts" in S:
        c = S["fig3a_counts"]

        def pq(lev, reg, q, ht="All"):
            r = c[(c.level == lev) & (c.region == reg) & (c.quadrant == q) & (c.HRP_type == ht)]
            return (float(r.pct.iloc[0]), int(r.n.iloc[0]), int(r.n_group.iloc[0])) if len(r) else (np.nan, 0, 0)

        p, n, N = pq("major+sub", "South", "Q2")
        add("South basins in Quadrant II (major+sub, %)", "Results; Fig. 3a", OLD_VALUES["south_Q2_pct"],
            f"{p:.2f} ({n}/{N})", f"interval {FIG3A_INTERVAL}, GDP threshold {GDP_THRESHOLD:g}; N = units with a defined index and valid GDP")
        p, n, N = pq("major+sub", "North", "Q1")
        add("North basins in Quadrant I (major+sub, %)", "Results; Fig. 3a", OLD_VALUES["north_Q1_pct"],
            f"{p:.2f} ({n}/{N})", f"interval {FIG3A_INTERVAL}; N = units with a defined index and valid GDP")
        q2 = c[(c.level == "major+sub") & (c.quadrant == "Q2") & (c.HRP_type != "All")].groupby("HRP_type").n.sum()
        add("Quadrant II counts by HRP type (major+sub, both regions)", "Results; Fig. 3a",
            f"Dispersion {OLD_VALUES['Q2_dispersion_n']}, Near-river {OLD_VALUES['Q2_near_n']}",
            ", ".join(f"{k} {int(v)}" for k, v in q2.sort_values(ascending=False).items()))
        for lev in ["major", "sub", "major+sub"]:
            add(f"Quadrant composition, {lev}", "Fig. S9", "",
                "; ".join(f"{reg} (N={pq(lev, reg, 'Q1')[2]}): " + ", ".join(f"{q} {pq(lev, reg, q)[0]:.1f}%"
                                                                            for q in ["Q1", "Q2", "Q3", "Q4"])
                          for reg in ["North", "South"]))
    else:
        add("Quadrant numbers (Fig. 3a, S9)", "Results; Fig. 3a", "", "not computed", "set GDP_RASTER")
    st, tests = S["fig3b_chain_stats"], S["fig3b_tests"]
    first = f"{HIST_INTERVALS[0][0]}-{HIST_INTERVALS[0][1]}"
    for lev in ["major", "major+sub"]:
        r = st[(st.scenario == "historical") & (st.interval == first) & (st.level == lev) & (st.HRP_type == "Near-river")]

        def fmt(reg):
            x = r[r.region == reg]
            if not len(x):
                return "nan"
            x = x.iloc[0]
            return f"{x['median']:.3f} [{x.q25:.3f}, {x.q75:.3f}] (n={int(x.n)})"

        t = tests[(tests.family == "chain") & (tests.scenario == "historical") & (tests.interval == first)
                  & (tests.HRP_type == "Near-river")]
        add(f"Near-river median index {first} [IQR], South vs North, {lev}", "Results; Fig. 3b",
            f"{OLD_VALUES['near_first_interval_south']} vs {OLD_VALUES['near_first_interval_north']} (means)",
            f"{fmt('South')} vs {fmt('North')}", f"MW p (majors) = {float(t.MW_p.iloc[0]) if len(t) else np.nan:.3g}")
    nd = S["ns_difference"]
    ndc = nd[nd.family == "cumulative"]
    col = "median_South_minus_North"
    if col in ndc:
        for lev in ["major", "major+sub"]:
            d = ndc[(ndc.level == lev) & (ndc.HRP_type == "Dispersion")]
            add(f"Dispersion: intervals with South median > North median, {lev}", "Results; Fig. 3b", "all intervals",
                f"{int((d[col] > 0).sum())} of {int(d[col].notna().sum())}",
                "cumulative intervals (historical from 1990, future from 2020)")
        late = ndc[(ndc.scenario != "historical") & (ndc.y2 > 2030) & (ndc.level == "major+sub") & (ndc.HRP_type != "All")]
        if len(late):
            g = late.assign(a=late[col].abs()).groupby("scenario").a.mean()
            add("Mean |South - North| of medians after 2030 by scenario (major+sub, three types)", "Results; Fig. S11",
                "largest under SSP3-8.5", ", ".join(f"{k}: {v:.4f}" for k, v in g.items()), "cumulative from 2020")
        s3 = ndc[(ndc.scenario == "SSP3-8.5") & (ndc.level == "major+sub") & (ndc.HRP_type != "All")]
        if len(s3) and "median_South" in s3:
            add("SSP3-8.5: South median > 1 by year and type (major+sub)", "Results; Fig. S11",
                "South elevated through 2080",
                ", ".join(f"{k}: {int((g.median_South > 1).sum())}/{len(g)}" for k, g in s3.groupby("HRP_type")),
                "cumulative from 2020")
    if "scenario_overlap" in S:
        o = S["scenario_overlap"]
        o = o[(o.level == "major+sub") & (o.family == "cumulative")]
        add("Median |SSP2-4.5 - SSP2-8.5| by year (major+sub)", "Results; Fig. S10", "similar",
            ", ".join(f"{r.interval}: {r['median']:.4f}" for _, r in o.iterrows()), "cumulative from 2020")
    ft = S.get("future_trends")
    if ft is not None and len(ft):
        f3 = ft[(ft.scenario == "SSP3-8.5") & (ft.HRP_type == "All")]
        add("SSP3-8.5 slope of median cumulative index per decade, North / South", "SI Fig. S11 (opposing trends)",
            "opposing",
            "; ".join(f"{lev}: " + " / ".join(f"{float(g.loc[g.region == r, 'theil_sen_per_decade'].iloc[0]):+.4f}"
                                              if (g.region == r).any() else "nan" for r in ["North", "South"])
                      for lev, g in f3.groupby("level")), "2020 = 1; descriptive")
    if "decomp_summary" in S:
        ds = S["decomp_summary"]
        for r in ds[ds.level == "major"].itertuples():
            add(f"Decomposition {r.scenario} {r.period}: climate / population / interaction (majors, medians)",
                "Discussion; SI decomposition", "population > climate",
                f"counts {r.count_climate_median:.1f}% / {r.count_population_median:.1f}% (n={r.n_count}, "
                f"p={r.count_wilcoxon_p:.3g}); index {r.index_climate_median:.1f}% / {r.index_population_median:.1f}% / "
                f"{r.index_interaction_median:.1f}% (n={r.n_index}, p={r.index_wilcoxon_p:.3g})",
                "counts = Jiang et al. on exposed population; index = change_fe")
    return pd.DataFrame(rows)


def qa_table(lv, idx, units, inv, masks, grids, water_src, groups=""):
    rows = []

    def add(check, value, note=""):
        rows.append(dict(check=check, value=value, note=note))

    if groups:
        add("PARTIAL RUN", groups, "only these groups; summaries are not manuscript values")
    if not SCENARIOS:
        add("HISTORICAL ONLY", True, "future scenarios and decomposition not computed")
    mj = units[units.level == "major"]
    add("units", len(units), f"majors {len(mj)} (North {(mj.region == 'North').sum()}, South {(mj.region == 'South').sum()}), "
                             f"subs {(units.level == 'sub').sum()}")
    bl = mj[mj.south_area_ratio.between(0.4, 0.6)]
    add("majors with South area share 40-60%", len(bl),
        ", ".join(f"{r.unit_key} {100 * r.south_area_ratio:.1f}% -> {r.region}" for r in bl.itertuples()))
    flood = lv[lv.hazard != "valid"]
    add("max E/P", float((flood.E / flood.P.where(flood.P > 0)).max()), "must be <= 1")
    hist = lv[(lv.pop_product == "GlobPOP") & (lv.hazard == "historical")]
    nf = hist.groupby("unit_key").flood_cells.max()
    add("units with no flood cell (historical hazard)", int((nf <= 0).sum()), ", ".join(nf[nf <= 0].index[:20]))
    for sc, g in idx.groupby("scenario"):
        z, lo = g[g.zero_base], g[g.low_base]
        add(f"zero-base unit-intervals ({sc})", len(z), f"{z.unit_key.nunique()} units; change_fe undefined")
        add(f"low-base unit-intervals, E1 < {LOW_BASE_FLAG:g} ({sc})", len(lo), f"{lo.unit_key.nunique()} units; flagged")
    cov = lv[lv.hazard == "valid"].copy()
    if len(cov):
        cov["nodata_pct"] = 100 * (1 - cov.E / cov.P.where(cov.P > 0))
        for (prod, layer), g in cov.groupby(["pop_product", "gcm"]):
            g = g.sort_values("nodata_pct", ascending=False)
            add(f"population on {'hazard NoData' if layer == 'hazard' else 'water no-observation'} cells, {prod} "
                f"{int(g.pop_year.iloc[0])} (%): median / max", f"{g.nodata_pct.median():.3f} / {g.nodata_pct.max():.3f}",
                f"units > 1%: {int((g.nodata_pct > 1).sum())}; " + ", ".join(f"{r.unit_key} {r.nodata_pct:.1f}%"
                                                                      for r in g.head(5).itertuples()))
    if "SSP" in grids and "GlobPOP" in grids:
        g0 = hist[hist.pop_year == FUTURE_BASE_YEAR].set_index("unit_key")
        for s in sorted({s for s, _ in SCENARIOS.values()}):
            h2 = lv[(lv.pop_product == f"SSP{s}") & (lv.hazard == "historical") & (lv.pop_year == FUTURE_BASE_YEAR)]
            h2 = h2.set_index("unit_key")
            if h2.empty or g0.empty:
                continue
            rp = h2.P / g0.P.reindex(h2.index)
            s_ssp, s_gp = h2.E / h2.P, (g0.E / g0.P).reindex(h2.index)
            both = np.isfinite(s_ssp) & (s_gp > 0) & np.isfinite(s_gp)
            re_ = (s_ssp / s_gp)[both]
            add(f"SSP{s} {FUTURE_BASE_YEAR} / GlobPOP {FUTURE_BASE_YEAR}: total population, median [min, max]",
                f"{rp.median():.3f} [{rp.min():.3f}, {rp.max():.3f}]", "product difference at the base year")
            add(f"SSP{s} {FUTURE_BASE_YEAR} / GlobPOP {FUTURE_BASE_YEAR}: exposed share, median [min, max]",
                f"{re_.median():.3f} [{re_.min():.3f}, {re_.max():.3f}]",
                f"defined for {int(both.sum())} units; undefined (zero/invalid denominator or invalid numerator) for "
                f"{int((~both).sum())}; future index uses SSP for both years")
    fut = idx[idx.scenario != "historical"]
    if len(fut) and "gcm_range_complete" in fut:
        for fam, g in [("cumulative from 2020", fut[fut.from_base]), ("chain between future years", fut[~fut.from_base])]:
            g = g[~g.zero_base]
            add(f"future unit-intervals with an incomplete GCM index range ({fam})", int(g.gcm_range_complete.eq(False).sum()),
                f"of {len(g)} with a defined ensemble index; the GCM range covers n_gcm_valid models only")
    add("GCMs", ", ".join(GCMS) if SCENARIOS else "none", f"{len(inv['hazards'])} hazard files")
    add("depth rule", f"{'>' if DEPTH_STRICT else '>='} {DEPTH_THRESHOLD_M} m, RP{RETURN_PERIOD}")
    jrc = (f"JRC years {WATER_JRC_YEARS[0]}-{WATER_JRC_YEARS[-1]}, permanent share >= {WATER_JRC_MIN_SHARE}; "
           if WATER_MODE == "jrc_yearly" else "")
    add("permanent water", f"{WATER_MODE}: {water_src}", jrc + f"cell excluded if overlap >= {WATER_EXCLUDE_FRACTION}")
    for name, (mg, _) in masks.items():
        add(f"mask grid for {name}", f"{mg.width} x {mg.height}", mg.sig)
    for name in ["GlobPOP", "SSP"]:
        if name in grids and "hazard" in grids:
            h, g = grids["hazard"], grids[name]
            al = aligned_offset(h.tr, h.crs, (h.width, h.height), g.tr, g.crs) is not None
            add(f"hazard grid aligned with {name}", al,
                offset_note(h, g) + ("; flood cells used directly" if al else "; area-weighted flood fractions"))
    if GDP_MODE in ("total_raster", "percap_raster") and GDP_RASTER is not None:
        add("GDP", f"{GDP_MODE}: {GDP_RASTER}",
            f"bands {GDP_BAND}; per capita = sum of GDP cells (GDP grid, cell centres) / sum of GlobPOP cells"
            if GDP_MODE == "total_raster" else f"bands {GDP_BAND}; population-weighted mean of per-capita values")
    return pd.DataFrame(rows)


def write_excel(path, sheets):
    with pd.ExcelWriter(path, engine="openpyxl") as xw:
        for name, df in sheets:
            if df is None or len(df) == 0:
                continue
            df.to_excel(xw, sheet_name=name[:31], index=False)
            ws = xw.sheets[name[:31]]
            ws.freeze_panes = "B2"
            ws.auto_filter.ref = ws.dimensions


def main():
    global SCENARIOS
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--missing-urls", action="store_true", help="Write WRI URLs for missing hazard maps")
    ap.add_argument("--download-missing", action="store_true", help="Download missing hazard maps into AQUEDUCT_DIR/downloaded_v2")
    ap.add_argument("--masks-only", action="store_true")
    ap.add_argument("--groups", default="")
    ap.add_argument("--workers", type=int, default=WORKERS)
    ap.add_argument("--hrp-run", default=None)
    ap.add_argument("--historical-only", action="store_true", help="Calculate historical exposure for 1990-2020 only")
    ap.add_argument("--inspect", default="", help="Export clipped rasters for GIS inspection; comma-separated unit_key values")
    a = ap.parse_args()
    t0 = time.time()
    if a.historical_only:
        SCENARIOS = {}
    hrp_run = Path(a.hrp_run) if a.hrp_run else HRP_RUN
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    all_units, region_src = load_units(hrp_run, REGION_CSV)
    units = all_units
    if a.groups:
        keep = [g.strip() for g in a.groups.split(",") if g.strip()]
        units = all_units[all_units.major.isin(keep)]
        if units.empty:
            raise ValueError(f"No units for groups {keep}")
    maj = units.level == "major"
    print(f"Units: {len(units)} (majors {maj.sum()}: North {(units.region[maj] == 'North').sum()}, "
          f"South {(units.region[maj] == 'South').sum()}); regions from {region_src}", flush=True)
    inv = inventory()
    if a.missing_urls:
        urls = missing_urls(inv)
        out = OUTPUT_ROOT / "missing_hazard_urls.txt"
        out.write_text("\n".join(urls), encoding="utf-8")
        print("\n".join(urls) + f"\n{len(urls)} URLs written to {out}\n"
              f"Verify filenames at {AQUEDUCT_URL}index.html and place downloaded maps in AQUEDUCT_DIR")
        return
    if a.download_missing:
        ref = grid_of(inv["hazards"][HIST_KEY]) if HIST_KEY in inv["hazards"] else None
        dest, done, failed = download_missing(inv, ref)
        print(f"{len(done)} file(s) ready in {dest}; {len(failed)} failed")
        for n, e in failed:
            print(f"  FAILED {n}: {e}")
        if failed:
            print("HTTP 404 indicates an unavailable WRI file; review the GCM inventory. Retry other errors to resume, "
                  "or download the --missing-urls entries in a browser into AQUEDUCT_DIR. ")
        print("Run --check after completing downloads.")
        return
    desc = []
    for label, files in [("GlobPOP", list(inv["globpop"].values())[:1]), ("SSP", list(inv["ssp"].values())[:1]),
                         ("hazard", list(inv["hazards"].values())[:1])]:
        for f in files:
            desc.append(describe_raster(f, label))
    if WATER_MODE == "jrc_yearly":
        f = Path(WATER_JRC_DIR) / WATER_JRC_PATTERN.format(year=WATER_JRC_YEARS[-1])
        if f.is_file():
            desc.append(describe_raster(f, "JRC yearly"))
    elif WATER_MODE == "raster" and Path(WATER_RASTER).is_file():
        desc.append(describe_raster(WATER_RASTER, "water"))
    gp = gdp_paths()
    if gp and Path(list(gp[0].values())[0]).is_file():
        desc.append(describe_raster(list(gp[0].values())[0], "GDP"))
    if desc:
        print(pd.DataFrame(desc).drop(columns=["path"]).to_string(index=False), flush=True)
    found = {}
    for k in inv["hazards"]:
        if k != HIST_KEY:
            sc, g, y = k.split("|")
            found.setdefault(f"{sc}|{y}", []).append(g)
    print(f"GlobPOP years: {len(inv['globpop'])}/{len(HIST_YEARS)}; SSP files: {len(inv['ssp'])}; "
          f"historical hazard: {HIST_KEY in inv['hazards']}; GCMS required: {', '.join(GCMS) if SCENARIOS else 'none'}",
          flush=True)
    for k, v in sorted(found.items()):
        print(f"  {k}: {len(v)} GCMs ({', '.join(sorted(v))})")
    for n in inv["notes"]:
        print("NOTE:", n)
    if inv["problems"]:
        print("\n".join(["PROBLEMS:"] + inv["problems"][:40]), flush=True)
        if inv["missing_hazards"]:
            print(f"Missing {len(inv['missing_hazards'])} hazard maps: run --missing-urls; "
                  f"use a complete GCM inventory or run --historical-only.", flush=True)
    grids = check_grids(inv)
    for name in ["GlobPOP", "SSP"]:
        if name in grids and "hazard" in grids:
            h, g = grids["hazard"], grids[name]
            al = aligned_offset(h.tr, h.crs, (h.width, h.height), g.tr, g.crs) is not None
            print(f"{name} grid aligned with hazard grid: {al} ({offset_note(h, g)}"
                  f"{'' if al else '; flood fractions will be area-weighted'})", flush=True)
    if gp and "GlobPOP" in grids and Path(list(gp[0].values())[0]).is_file():
        gg = grid_of(list(gp[0].values())[0])
        g = grids["GlobPOP"]
        al = aligned_offset(gg.tr, gg.crs, (gg.width, gg.height), g.tr, g.crs) is not None
        print(f"GDP raster aligned with GlobPOP: {al} ({offset_note(g, gg)}); "
              + ("GDP totals are summed on the GDP grid itself" if GDP_MODE == "total_raster"
                 else "per-capita values taken by nearest cell"), flush=True)
    if a.check:
        out = OUTPUT_ROOT / f"check_{ts()}"
        out.mkdir(parents=True)
        pd.DataFrame(desc).to_csv(out / "rasters.csv", index=False, encoding="utf-8-sig")
        pd.DataFrame({"problem": inv["problems"] + ["NOTE " + n for n in inv["notes"]]}).to_csv(
            out / "problems.csv", index=False, encoding="utf-8-sig")
        print(f"Check written to {out}")
        return
    if inv["problems"]:
        raise SystemExit("Fix the problems above (or run with --check / --historical-only)")

    print("Preparing masks (all units; reused by later runs):", flush=True)
    bounds = {r.unit_key: load_polygon(r.basin_shp, "EPSG:4326").bounds for r in all_units.itertuples(index=False)}
    masks, water_src = prepare_masks(inv, grids, bounds, a.workers)
    if a.inspect:
        inspect_units([k.strip() for k in a.inspect.split(",") if k.strip()], all_units, inv, grids, masks)
        return
    if a.masks_only:
        print("Masks ready")
        return
    results, errors = run_jobs(build_jobs(units, inv, grids, masks), a.workers, process_group, "exposure")
    gdp_results, gdp_errors = run_jobs(build_gdp_jobs(units, inv, grids), a.workers, process_gdp, "gdp")
    out = OUTPUT_ROOT / f"run_{ts()}"
    out.mkdir(parents=True)
    if errors or gdp_errors:
        pd.DataFrame(errors + gdp_errors, columns=["stage", "group", "traceback"]).to_csv(
            out / "errors.csv", index=False, encoding="utf-8-sig")
        raise SystemExit(f"{len(errors) + len(gdp_errors)} groups failed; see {out / 'errors.csv'} (finished groups are cached)")
    lv = pd.concat(results.values(), ignore_index=True)
    lv["unit_key"] = lv.unit_key.astype(str)
    lv = lv[lv.unit_key.isin(units.unit_key)]
    gdp_df = None
    gr = None
    if GDP_MODE == "table":
        gdp_df = load_gdp_table(units)
    elif gdp_results:
        gr = pd.concat(gdp_results.values(), ignore_index=True)
        gr["unit_key"] = gr.unit_key.astype(str)
        w = gr.pivot(index="unit_key", columns="year", values="gdp_pc")
        cov = gr.pivot(index="unit_key", columns="year", values="gdp_pop_coverage")
        gdp_df = pd.DataFrame({"unit_key": w.index, f"gdp_pc_{GDP_YEARS[0]}": w[GDP_YEARS[0]].values,
                               f"gdp_pc_{GDP_YEARS[1]}": w[GDP_YEARS[1]].values,
                               "gdp_pc_mean": w[list(GDP_YEARS)].mean(axis=1, skipna=False).values,
                               "gdp_pop_coverage_min": cov.min(axis=1, skipna=False).values})
    idx, annual, fut_levels, decomp, gcm_tab = build_indices(lv)
    S = summaries(units, idx, gdp_df, decomp, gcm_tab)
    mn = manuscript_numbers(S, units)
    qa = qa_table(lv, idx, units, inv, masks, grids, water_src, a.groups)
    U = units.drop(columns=["basin_shp"])
    idx_out = idx.merge(U, on="unit_key", how="left")
    lead = ["unit_key", "level", "major", "region", "HRP_type", "scenario", "interval", "interval_type", "y1", "y2"]
    idx_out = idx_out[lead + [c for c in idx_out.columns if c not in lead]]
    cov = lv[lv.hazard == "valid"].assign(nodata_pct=lambda d: 100 * (1 - d.E / d.P.where(d.P > 0)))
    cov = cov.rename(columns={"gcm": "layer", "pop_year": "year"})[["unit_key", "pop_product", "year", "layer", "nodata_pct"]]
    files = dict(change_fe=idx_out, levels=lv[lv.hazard != "valid"].merge(U, on="unit_key", how="left"),
                 annual_share=annual.merge(U, on="unit_key", how="left"),
                 future_levels=fut_levels.merge(U, on="unit_key", how="left") if len(fut_levels) else fut_levels,
                 coverage=cov.merge(U, on="unit_key", how="left"), manuscript_numbers=mn, qa=qa)
    if gr is not None:
        files["gdp_by_year"] = gr
    if len(gcm_tab):
        files["change_fe_by_gcm"] = gcm_tab.merge(U, on="unit_key", how="left")
    for k in ["fig3a_units", "fig3a_counts", "fig3a_valid_n", "fig3b_cumulative_stats", "fig3b_chain_stats",
              "fig3b_chain_annualized", "fig3b_lowbase_excluded", "fig3b_tests", "fig3b_gcm_spread", "fig3b_gcm_medians",
              "zero_status", "future_trends", "s12_units", "s12_stats", "s12_tests", "scenario_overlap", "ns_difference",
              "gdp_sensitivity", "decomp", "decomp_summary", "fe_above1"]:
        if k in S:
            files[k] = S[k]
    for k, df in files.items():
        df.to_csv(out / f"{k}.csv", index=False, encoding="utf-8-sig")
    if "decomp" in S:
        cols = ["basin_name", "scenario", "period", "P0", "P_cc", "P_pg", "region", "hrp_type", "P_coupled", "level", "n_gcm"]
        d = S["decomp"][cols]
        d[d.level == "major"].to_csv(out / "flood_driver_counterfactuals.csv", index=False, encoding="utf-8-sig")
        d[d.level == "sub"].to_csv(out / "flood_driver_counterfactuals_subbasins.csv", index=False, encoding="utf-8-sig")
    write_excel(out / "flood_exposure_v3.xlsx", [
        ("Definitions", pd.DataFrame({"Definitions": __doc__.splitlines()})), ("Manuscript numbers", mn), ("QA", qa),
        ("Fig3a units", S["fig3a_units"]), ("Fig3a valid N", S["fig3a_valid_n"]),
        ("Fig3a quadrants", S.get("fig3a_counts")), ("Index above 1", S["fe_above1"]),
        ("Fig3b cumulative", S["fig3b_cumulative_stats"]), ("Fig3b chain", S["fig3b_chain_stats"]),
        ("Fig3b chain annualized", S["fig3b_chain_annualized"]), ("Fig3b low-base excluded", S["fig3b_lowbase_excluded"]),
        ("Fig3b tests (majors)", S["fig3b_tests"]), ("Fig3b GCM spread", S.get("fig3b_gcm_spread")),
        ("Zero-base status", S["zero_status"]),
        ("Future trends", S["future_trends"]), ("S12 units", S["s12_units"]), ("S12 stats", S["s12_stats"]),
        ("S12 tests (majors)", S["s12_tests"]), ("Scenario overlap", S.get("scenario_overlap")),
        ("South-North difference", S["ns_difference"]), ("GDP sensitivity", S.get("gdp_sensitivity")),
        ("Decomposition", S.get("decomp")), ("Decomposition summary", S.get("decomp_summary")),
        ("Coverage", files["coverage"]), ("change_fe all", idx_out), ("Annual share", files["annual_share"]),
    ])
    cfg = {k: (str(v) if isinstance(v, Path) else v) for k, v in globals().items()
           if k.isupper() and isinstance(v, (str, int, float, bool, list, tuple, dict, Path, type(None)))}
    cfg.update(hrp_run=str(hrp_run), regions=region_src, groups=a.groups, water_source=water_src, inputs=dict(
        globpop=inv["globpop"], ssp={f"{k[0]}_{k[1]}": v for k, v in inv["ssp"].items()}, hazards=inv["hazards"]))
    (out / "config.json").write_text(json.dumps(cfg, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    shutil.copy2(__file__, out / Path(__file__).name)
    print("\n" + mn.to_string(index=False))
    print(f"\nOutput: {out}  ({(time.time() - t0) / 60:.1f} min)")


if __name__ == "__main__":
    main()
