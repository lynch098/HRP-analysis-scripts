
"""Sub-basin population-weighted pixel-centre distance trends, 1990-2020.
D(t) = sum(P_i(t)*d_i)/sum(P_i(t)); distances are fixed across years."""

import json as _path_json
from pathlib import Path as _ConfigPath


_DATA_PATHS = _path_json.loads(
    (_ConfigPath(__file__).resolve().parents[1] / "paths.json").read_text(encoding="utf-8")
)
from pathlib import Path
from datetime import datetime
import json
import shutil
import traceback
import hashlib
import os
import re
import sys
import numpy as np
import pandas as pd
from scipy.stats import theilslopes, norm

NETWORK_RUN = Path(_DATA_PATHS["NETWORK_RUN"])
MASTER = NETWORK_RUN / "04_subbasin_master.csv"
HRP_RUN = Path(_DATA_PATHS["HRP_RUN"])
POP_DIR = Path(_DATA_PATHS["POP_DIR"])
OUTPUT_ROOT = Path(_DATA_PATHS["SUBBASIN_TREND_OUTPUT_ROOT"])
YEARS = list(range(1990, 2021))
PERIODS = [(1990, 2020), (2000, 2020)]
ALPHA = 0.05
FDR_ALPHA = 0.05
THEILSEN_CI = 0.95


def load_master():
    m = pd.read_csv(MASTER, dtype={"HYBAS_ID": str, "PFAF_ID": str})
    required = {"unit_key", "major", "decision", "basin_shp", "river_shp"}
    if not required.issubset(m):
        raise ValueError(f"Master missing columns: {required - set(m.columns)}")
    m = m.loc[m.decision.eq("ACCEPT")].copy()
    if m.empty or m.unit_key.isna().any() or m.unit_key.duplicated().any():
        raise ValueError("Empty or invalid roster")
    for col in ["basin_shp", "river_shp"]:
        for fp in m[col]:
            if pd.isna(fp) or not Path(fp).is_file():
                raise FileNotFoundError(str(fp))
    config = json.loads((HRP_RUN / "config.json").read_text(encoding="utf-8"))
    if Path(config["network_run"]).resolve() != NETWORK_RUN.resolve():
        raise ValueError("HRP profiles and selected network run do not match")
    h = pd.read_csv(HRP_RUN / "main" / "static_hrp.csv")
    h = h.loc[h["set"].eq("sub")].copy()
    if h.unit_key.duplicated().any() or set(h.unit_key) != set(m.unit_key):
        raise ValueError("HRP and master sub-basin IDs differ; no partial join allowed")
    hm = h.set_index("unit_key")
    for r in m.itertuples(index=False):
        for col in ["basin_shp", "river_shp"]:
            if Path(getattr(r, col)).resolve() != Path(hm.loc[r.unit_key, col]).resolve():
                raise ValueError(f"Geometry path mismatch: {r.unit_key}, {col}")
    m["HRP_type"] = m.unit_key.map(hm.HRP_type)
    m["parent_mainstem"] = m.major
    return m


def start_run(kind, master):
    out = OUTPUT_ROOT / (kind + "_" + datetime.now().strftime("%Y%m%d_%H%M%S_%f"))
    out.mkdir(parents=True, exist_ok=False)
    shutil.copy2(__file__, out / Path(__file__).name)
    master.to_csv(out / "00_input_master.csv", index=False, encoding="utf-8-sig")
    (out / "config.json").write_text(json.dumps(dict(
        source_master=str(MASTER), source_profiles=str(HRP_RUN), population_dir=str(POP_DIR),
        n_units=len(master), years=YEARS, periods=PERIODS,
        definitions=__doc__, stats="Classical two-sided MK with ties; Theil-Sen 95% CI; BH FDR within Sub and each period. No temporal autocorrelation correction."
    ), ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Units: {len(master)}; output: {out}", flush=True)
    return out


def summarize_trends(annual, value_col, directional=False):
    records = []
    for (start, end) in PERIODS:
        for unit, g in annual.groupby("unit_key", sort=False):
            g = g.loc[g.year.between(start, end)].sort_values("year")
            expected = np.arange(start, end + 1)
            if not np.array_equal(g.year.to_numpy(), expected):
                raise ValueError(f"Missing/duplicate years: {unit}, {start}-{end}")
            y = g[value_col].to_numpy(float)
            if not np.isfinite(y).all():
                raise ValueError(f"Non-finite {value_col}: {unit}")
            fit = theilslopes(y, expected, alpha=THEILSEN_CI)
            mk = mann_kendall_test(y)
            rec = dict(unit_key=unit, period=f"{start}-{end}", n_years=len(y),
                       first_value=y[0], last_value=y[-1], mean_value=float(y.mean()),
                       slope_per_year=float(fit.slope), slope_per_decade=float(fit.slope*10),
                       slope_low95_per_decade=float(fit.low_slope*10),
                       slope_high95_per_decade=float(fit.high_slope*10),
                       MK_p=mk["p"], MK_z=mk["z"], MK_tau=mk["tau"],
                       direction=("Riverward" if fit.slope < 0 else "Away" if fit.slope > 0 else "Stable")
                       if directional else ("Decreasing" if fit.slope < 0 else "Increasing" if fit.slope > 0 else "Stable"))
            records.append(rec)
    result = pd.DataFrame(records)
    result["MK_q_FDR"] = np.nan
    for _, g in result.groupby("period"):
        result.loc[g.index, "MK_q_FDR"] = bh_fdr(g.MK_p.to_numpy())
    result["significant_nominal"] = result.MK_p < ALPHA
    result["significant_FDR"] = result.MK_q_FDR < FDR_ALPHA
    return result


def export(out, annual, trends, master, filename):
    attrs = master[["unit_key", "parent_mainstem", "HRP_type"]]
    annual = annual.merge(attrs, on="unit_key", how="left", validate="many_to_one")
    trends = trends.merge(attrs, on="unit_key", how="left", validate="many_to_one")
    for label, frame in [("annual", annual), ("trends", trends)]:
        frame.to_csv(out / f"{label}.csv", index=False, encoding="utf-8-sig")
    with pd.ExcelWriter(out / filename, engine="openpyxl") as writer:
        for sheet, frame in [("Annual", annual), ("Trends", trends), ("Input master", master)]:
            frame.to_excel(writer, sheet_name=sheet, index=False)
            writer.sheets[sheet].freeze_panes = "C2"
            writer.sheets[sheet].auto_filter.ref = writer.sheets[sheet].dimensions
        pd.DataFrame({"Definition": __doc__.splitlines()}).to_excel(writer, sheet_name="Definitions", index=False)
    print(f"Finished: {out / filename}", flush=True)

import geopandas as gpd
import rasterio
from rasterio.features import geometry_mask, geometry_window
from rasterio.windows import Window
from pyproj import CRS, Transformer
import shapely
from shapely import points
from shapely.strtree import STRtree

POINT_CHUNK_SIZE = 100_000
ANALYSIS_VERSION = "pixel_subbasin_new_network_20261019_v1"
FORCE_RECALCULATE_DISTANCE = False
EXPECTED_N_YEARS = len(YEARS)
START_YEAR, END_YEAR = YEARS[0], YEARS[-1]
ALIASES = {}
CACHE_DIR = OUTPUT_ROOT / "riverward_distance_cache"
YEAR_CACHE_DIR = OUTPUT_ROOT / "riverward_year_cache"


def clean_name(x):

    if pd.isna(x):
        return np.nan

    key = str(x).strip().upper()

    for ch in [" ", "-", "/", "."]:
        key = key.replace(ch, "_")

    key = re.sub(r"_+", "_", key).strip("_")
    compact = key.replace("_", "")

    return ALIASES.get(key, ALIASES.get(compact, key))

def safe_filename(x):
    x = clean_name(x)
    return re.sub(r'[\\/:*?"<>|]+', "_", str(x))

def vector_dataset_fingerprint(shp_path):

    shp_path = Path(shp_path)
    members = sorted(shp_path.parent.glob(shp_path.stem + ".*"))

    h = hashlib.sha1()
    for fp in members:
        try:
            st = fp.stat()
            token = f"{fp.name}|{st.st_size}|{st.st_mtime_ns}"
            h.update(token.encode("utf-8"))
        except OSError:
            pass

    return h.hexdigest()

def affine_equal(a, b, atol=1e-12):
    return np.allclose(
        np.asarray(tuple(a), dtype=float),
        np.asarray(tuple(b), dtype=float),
        rtol=0,
        atol=atol,
    )

def basin_run_signature(basin_name, basin_fp, river_fp):
    text = "|".join(
        [
            ANALYSIS_VERSION,
            clean_name(basin_name),
            GRID_SIGNATURE,
            vector_dataset_fingerprint(basin_fp),
            vector_dataset_fingerprint(river_fp),
        ]
    )
    return hashlib.sha1(text.encode("utf-8")).hexdigest()[:16]

def _make_valid(geom):
    if geom is None or geom.is_empty:
        return geom
    if geom.is_valid:
        return geom
    try:
        return shapely.make_valid(geom)
    except Exception:
        return geom.buffer(0)

def polygon_parts(geom):

    if geom is None or geom.is_empty:
        return []

    gt = geom.geom_type
    if gt == "Polygon":
        return [geom]
    if gt == "MultiPolygon":
        return list(geom.geoms)
    if gt == "GeometryCollection":
        out = []
        for g in geom.geoms:
            out.extend(polygon_parts(g))
        return out
    return []

def line_parts(geom):

    if geom is None or geom.is_empty:
        return []

    gt = geom.geom_type
    if gt in {"LineString", "LinearRing"}:
        return [geom]
    if gt == "MultiLineString":
        return list(geom.geoms)
    if gt == "GeometryCollection":
        out = []
        for g in geom.geoms:
            out.extend(line_parts(g))
        return out
    return []

def load_basin_geometry(basin_fp):
    gdf = gpd.read_file(basin_fp)
    if gdf.crs is None:
        raise RuntimeError(f"Basin shapefile has no CRS: {basin_fp}")

    gdf = gdf[gdf.geometry.notna() & ~gdf.geometry.is_empty].copy()
    if len(gdf) == 0:
        raise RuntimeError(f"Basin shapefile has no valid geometry: {basin_fp}")

    if gdf.crs != REF_CRS:
        gdf = gdf.to_crs(REF_CRS)

    geoms = [_make_valid(g) for g in gdf.geometry]
    geoms = [g for g in geoms if g is not None and not g.is_empty]

    try:
        merged = shapely.union_all(np.asarray(geoms, dtype=object))
    except Exception:
        merged = gpd.GeoSeries(geoms, crs=REF_CRS).unary_union

    merged = _make_valid(merged)
    polys = polygon_parts(merged)
    if len(polys) == 0:
        raise RuntimeError(f"No polygon geometry after merge: {basin_fp}")

    try:
        merged_poly = shapely.union_all(np.asarray(polys, dtype=object))
    except Exception:
        merged_poly = gpd.GeoSeries(polys, crs=REF_CRS).unary_union

    return _make_valid(merged_poly)

def load_and_clip_river_lines(river_fp, basin_geom):

    gdf = gpd.read_file(river_fp)
    if gdf.crs is None:
        raise RuntimeError(f"River shapefile has no CRS: {river_fp}")

    gdf = gdf[gdf.geometry.notna() & ~gdf.geometry.is_empty].copy()
    if len(gdf) == 0:
        raise RuntimeError(f"River shapefile has no valid geometry: {river_fp}")

    if gdf.crs != REF_CRS:
        gdf = gdf.to_crs(REF_CRS)

    out = []
    for geom in gdf.geometry:
        geom = _make_valid(geom)
        if geom is None or geom.is_empty:
            continue
        try:
            clipped = geom.intersection(basin_geom)
        except Exception:
            clipped = _make_valid(geom).intersection(_make_valid(basin_geom))
        out.extend(line_parts(clipped))

    out = [g for g in out if g is not None and not g.is_empty]
    if len(out) == 0:
        raise RuntimeError(
            f"No river line remains after clipping to basin: {river_fp}"
        )

    return gpd.GeoDataFrame(geometry=out, crs=REF_CRS)

def get_basin_window(src, basin_geom):
    try:
        win = geometry_window(src, [basin_geom], pad_x=0, pad_y=0)
    except Exception:
        return None

    col0 = max(0, int(np.floor(win.col_off)))
    row0 = max(0, int(np.floor(win.row_off)))
    col1 = min(src.width, int(np.ceil(win.col_off + win.width)))
    row1 = min(src.height, int(np.ceil(win.row_off + win.height)))

    width = col1 - col0
    height = row1 - row0
    if width <= 0 or height <= 0:
        return None

    return Window(col0, row0, width, height)

def make_local_aeqd(basin_geom):

    basin_wgs84 = gpd.GeoSeries([basin_geom], crs=REF_CRS).to_crs("EPSG:4326").iloc[0]
    p = basin_wgs84.representative_point()

    return CRS.from_proj4(
        f"+proj=aeqd +lat_0={p.y} +lon_0={p.x} "
        f"+datum=WGS84 +units=m +no_defs"
    )

def find_ever_populated_pixels(basin_geom, window):
    height = int(window.height)
    width = int(window.width)
    win_transform = rasterio.windows.transform(window, REF_TRANSFORM)


    basin_mask = geometry_mask(
        [basin_geom],
        out_shape=(height, width),
        transform=win_transform,
        invert=True,
        all_touched=False,
    )

    ever_positive = np.zeros((height, width), dtype=bool)

    for year in YEARS:
        with rasterio.open(POP_FILES[year]) as src:
            arr = src.read(1, window=window, masked=True)

        data = np.asarray(arr.data)
        arr_mask = np.asarray(arr.mask)

        if arr_mask.ndim == 0:
            valid_nodata = np.ones((height, width), dtype=bool)
        else:
            valid_nodata = ~arr_mask

        valid = (
            basin_mask
            & valid_nodata
            & np.isfinite(data)
            & (data > 0)
        )
        ever_positive |= valid

    rows, cols = np.where(ever_positive)

    return (
        rows.astype(np.int32),
        cols.astype(np.int32),
        win_transform,
    )

def calculate_pixel_distances(rows, cols, win_transform, basin_geom, river_fp):


    aeqd = make_local_aeqd(basin_geom)
    transformer = Transformer.from_crs(REF_CRS, aeqd, always_xy=True)

    river_gdf = load_and_clip_river_lines(river_fp, basin_geom)
    river_metric = river_gdf.to_crs(aeqd)

    river_geoms = [
        g for g in river_metric.geometry
        if g is not None and not g.is_empty
    ]
    if len(river_geoms) == 0:
        raise RuntimeError("No valid projected river geometry.")

    river_arr = np.asarray(river_geoms, dtype=object)
    tree = STRtree(river_arr)

    n = len(rows)
    distances_m = np.empty(n, dtype=np.float32)

    for start in range(0, n, POINT_CHUNK_SIZE):
        end = min(start + POINT_CHUNK_SIZE, n)

        rr = rows[start:end].astype(float)
        cc = cols[start:end].astype(float)


        x = (
            win_transform.c
            + (cc + 0.5) * win_transform.a
            + (rr + 0.5) * win_transform.b
        )
        y = (
            win_transform.f
            + (cc + 0.5) * win_transform.d
            + (rr + 0.5) * win_transform.e
        )

        xm, ym = transformer.transform(x, y)
        pts = points(xm, ym)


        nearest_idx = np.asarray(tree.nearest(pts), dtype=np.int64)
        nearest_geoms = river_arr[nearest_idx]

        d = shapely.distance(pts, nearest_geoms)
        d = np.asarray(d, dtype=np.float64)

        if not np.all(np.isfinite(d)):
            raise RuntimeError("Non-finite nearest-river distance encountered.")

        distances_m[start:end] = d.astype(np.float32)

        if n >= POINT_CHUNK_SIZE:
            print(
                f"    distance pixels: {end:,}/{n:,} "
                f"({end / n * 100:.1f}%)"
            )

    return (distances_m / 1000.0).astype(np.float32)

def load_distance_cache(cache_fp, expected_signature):
    try:
        z = np.load(cache_fp, allow_pickle=False)

        version = str(z["analysis_version"].item())
        signature = str(z["run_signature"].item())

        if version != ANALYSIS_VERSION:
            return None
        if signature != expected_signature:
            return None

        return {
            "row_off": int(z["row_off"]),
            "col_off": int(z["col_off"]),
            "height": int(z["height"]),
            "width": int(z["width"]),
            "rows": z["rows"],
            "cols": z["cols"],
            "distance_km": z["distance_km"],
        }
    except Exception:
        return None

def get_distance_template(basin_name, basin_fp, river_fp, basin_geom):
    signature = basin_run_signature(basin_name, basin_fp, river_fp)
    slug = safe_filename(basin_name)
    cache_fp = CACHE_DIR / f"{slug}__{signature}.npz"

    if cache_fp.exists() and not FORCE_RECALCULATE_DISTANCE:
        cached = load_distance_cache(cache_fp, signature)
        if cached is not None:
            print("  ✓ valid distance cache reused")
            return cached, signature

    print("  Building pixel-level distance template...")

    with rasterio.open(REF_FILE) as src:
        window = get_basin_window(src, basin_geom)

    if window is None:
        raise RuntimeError("Basin does not overlap the GlobPOP raster.")

    rows, cols, win_transform = find_ever_populated_pixels(
        basin_geom,
        window,
    )

    print(f"  Ever-populated pixels: {len(rows):,}")
    if len(rows) == 0:
        raise RuntimeError("No positive population pixel in 1990–2020.")

    distance_km = calculate_pixel_distances(
        rows,
        cols,
        win_transform,
        basin_geom,
        river_fp,
    )

    if np.nanmin(distance_km) < -1e-9:
        raise RuntimeError("Negative distance encountered.")

    temp_fp = cache_fp.with_suffix(".tmp.npz")

    np.savez_compressed(
        temp_fp,
        analysis_version=np.array(ANALYSIS_VERSION),
        run_signature=np.array(signature),
        row_off=np.int32(window.row_off),
        col_off=np.int32(window.col_off),
        height=np.int32(window.height),
        width=np.int32(window.width),
        rows=rows,
        cols=cols,
        distance_km=distance_km,
    )

    os.replace(temp_fp, cache_fp)
    print("  ✓ distance cache saved")

    return (
        {
            "row_off": int(window.row_off),
            "col_off": int(window.col_off),
            "height": int(window.height),
            "width": int(window.width),
            "rows": rows,
            "cols": cols,
            "distance_km": distance_km,
        },
        signature,
    )

def calculate_basin_year_series(basin_name, signature, template):
    rows = template["rows"]
    cols = template["cols"]
    distance_km = template["distance_km"].astype(np.float64, copy=False)

    window = Window(
        template["col_off"],
        template["row_off"],
        template["width"],
        template["height"],
    )

    output = []

    for year in YEARS:
        with rasterio.open(POP_FILES[year]) as src:
            arr = src.read(1, window=window, masked=True)

        pop = np.asarray(arr.data[rows, cols], dtype=np.float64)
        arr_mask = np.asarray(arr.mask)

        if arr_mask.ndim == 0:
            good_mask = np.ones(len(rows), dtype=bool)
        else:
            good_mask = ~arr_mask[rows, cols]

        valid = (
            good_mask
            & np.isfinite(pop)
            & (pop > 0)
            & np.isfinite(distance_km)
        )

        p = pop[valid]
        d = distance_km[valid]

        total_population = np.sum(p, dtype=np.float64)

        if total_population <= 0:
            weighted_mean_km = np.nan
        else:
            weighted_mean_km = (
                np.sum(p * d, dtype=np.float64) / total_population
            )

        output.append(
            {
                "analysis_version": ANALYSIS_VERSION,
                "run_signature": signature,
                "basin_name": basin_name,
                "year": year,
                "total_population": total_population,
                "n_populated_pixels": int(valid.sum()),
                "pw_mean_distance_km": weighted_mean_km,
            }
        )

    return pd.DataFrame(output)

def valid_existing_yearly(df, basin_name, signature):
    required = {
        "analysis_version",
        "run_signature",
        "basin_name",
        "year",
        "pw_mean_distance_km",
    }

    if not required.issubset(df.columns):
        return False
    if len(df) != EXPECTED_N_YEARS:
        return False
    if set(pd.to_numeric(df["year"], errors="coerce").dropna().astype(int)) != set(YEARS):
        return False
    if not (df["analysis_version"].astype(str) == ANALYSIS_VERSION).all():
        return False
    if not (df["run_signature"].astype(str) == signature).all():
        return False
    if not (df["basin_name"].astype(str).map(clean_name) == clean_name(basin_name)).all():
        return False

    return True

def mann_kendall_test(values):

    x = np.asarray(values, dtype=float)
    x = x[np.isfinite(x)]
    n = len(x)

    if n < 2:
        return {
            "S": np.nan,
            "var_S": np.nan,
            "z": np.nan,
            "p": np.nan,
            "tau": np.nan,
            "trend": "insufficient",
        }

    S = 0
    for k in range(n - 1):
        S += np.sign(x[k + 1:] - x[k]).sum()

    _, counts = np.unique(x, return_counts=True)
    tie_term = np.sum(counts * (counts - 1) * (2 * counts + 5))
    var_S = (n * (n - 1) * (2 * n + 5) - tie_term) / 18.0

    if var_S <= 0:
        z = 0.0
        p = 1.0
    elif S > 0:
        z = (S - 1) / np.sqrt(var_S)
        p = 2 * norm.sf(abs(z))
    elif S < 0:
        z = (S + 1) / np.sqrt(var_S)
        p = 2 * norm.sf(abs(z))
    else:
        z = 0.0
        p = 1.0

    denom = 0.5 * n * (n - 1)
    tau = S / denom if denom > 0 else np.nan

    if p < ALPHA and S > 0:
        trend = "increasing"
    elif p < ALPHA and S < 0:
        trend = "decreasing"
    else:
        trend = "no trend"

    return {
        "S": float(S),
        "var_S": float(var_S),
        "z": float(z),
        "p": float(p),
        "tau": float(tau),
        "trend": trend,
    }

def bh_fdr(pvalues):
    p = np.asarray(pvalues, dtype=float)
    q = np.full(len(p), np.nan)

    valid = np.isfinite(p)
    pv = p[valid]
    if len(pv) == 0:
        return q

    order = np.argsort(pv)
    ranked = pv[order]
    n = len(ranked)

    adjusted = ranked * n / np.arange(1, n + 1)
    adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]
    adjusted = np.clip(adjusted, 0, 1)

    back = np.empty(n)
    back[order] = adjusted
    q[valid] = back

    return q


def initialize_grid(out):
    global POP_FILES, REF_FILE, REF_CRS, REF_TRANSFORM, REF_WIDTH, REF_HEIGHT, GRID_SIGNATURE
    POP_FILES = {y: POP_DIR / f"GlobPOP_Count_30arc_{y}_I32.tiff" for y in YEARS}
    missing = [str(p) for p in POP_FILES.values() if not p.is_file()]
    if missing:
        raise FileNotFoundError("\n".join(missing))
    REF_FILE = POP_FILES[START_YEAR]
    with rasterio.open(REF_FILE) as src:
        REF_CRS, REF_TRANSFORM, REF_WIDTH, REF_HEIGHT = src.crs, src.transform, src.width, src.height
    if REF_CRS is None:
        raise ValueError("Reference raster has no CRS")
    signature_parts = [str(REF_CRS), repr(tuple(REF_TRANSFORM)), str(REF_WIDTH), str(REF_HEIGHT)]
    check = []
    for year, path in POP_FILES.items():
        with rasterio.open(path) as src:
            match = src.crs == REF_CRS and affine_equal(src.transform, REF_TRANSFORM) and src.width == REF_WIDTH and src.height == REF_HEIGHT
        st = path.stat()
        signature_parts.append(f"{path.resolve()}|{st.st_size}|{st.st_mtime_ns}")
        check.append(dict(year=year, path=str(path), grid_matches=match))
    pd.DataFrame(check).to_csv(out / "00_raster_grid_check.csv", index=False, encoding="utf-8-sig")
    if not all(r["grid_matches"] for r in check):
        raise ValueError("Annual population grids differ; inspect 00_raster_grid_check.csv")
    GRID_SIGNATURE = hashlib.sha1("|".join(signature_parts).encode("utf-8")).hexdigest()
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    YEAR_CACHE_DIR.mkdir(parents=True, exist_ok=True)


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(errors="replace")
    master = load_master()
    out = start_run("riverward", master)
    initialize_grid(out)
    all_years, failures = [], []
    for i, row in enumerate(master.itertuples(index=False), 1):
        print(f"[{i}/{len(master)}] {row.unit_key}", flush=True)
        try:
            signature = basin_run_signature(row.unit_key, row.basin_shp, row.river_shp)
            yearly_path = YEAR_CACHE_DIR / f"{safe_filename(row.unit_key)}__{signature}.csv"
            annual = None
            if yearly_path.is_file():
                try:
                    saved = pd.read_csv(yearly_path)
                    if valid_existing_yearly(saved, row.unit_key, signature) and np.isfinite(saved.pw_mean_distance_km).all() and (saved.total_population > 0).all():
                        annual = saved
                        print("  Annual cache reused", flush=True)
                except (ValueError, KeyError, pd.errors.ParserError):
                    pass
            if annual is None:
                basin = load_basin_geometry(row.basin_shp)
                template, signature = get_distance_template(row.unit_key, row.basin_shp, row.river_shp, basin)
                annual = calculate_basin_year_series(row.unit_key, signature, template)
                if not np.isfinite(annual.pw_mean_distance_km).all() or (annual.total_population <= 0).any():
                    raise ValueError("Incomplete valid annual distance series")
                temporary = yearly_path.with_suffix(".tmp.csv")
                annual.to_csv(temporary, index=False, encoding="utf-8-sig")
                os.replace(temporary, yearly_path)
                del template

            annual = annual.sort_values("year").copy()
            annual["unit_key"] = row.unit_key
            all_years.append(annual)
            print("  Complete", flush=True)
        except Exception as exc:
            failures.append(dict(unit_key=row.unit_key, error=str(exc), traceback=traceback.format_exc()))
            pd.DataFrame(failures).to_csv(out / "errors.csv", index=False, encoding="utf-8-sig")
            print(f"  FAILED: {exc}", flush=True)
    if failures:
        raise RuntimeError(f"{len(failures)} failed; no final summary. Completed units cached. See {out / 'errors.csv'}")
    annual = pd.concat(all_years, ignore_index=True)
    trends = summarize_trends(annual, "pw_mean_distance_km", directional=True)
    trends["theilsen_slope_km_year"] = trends.slope_per_year
    trends["theilsen_slope_km_decade"] = trends.slope_per_decade
    trends["slope_percent_mean_per_decade"] = np.where(trends.mean_value > 0,
        100 * trends.slope_per_decade / trends.mean_value, np.nan)
    export(out, annual, trends, master, "subbasin_Riverward_results.xlsx")


if __name__ == "__main__":
    main()
