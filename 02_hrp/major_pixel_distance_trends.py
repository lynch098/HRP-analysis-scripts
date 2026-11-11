
"""Major-basin population-weighted pixel-centre distance trends, 1990-2020.
D(t) = sum(P_i(t)*d_i)/sum(P_i(t)); distances are fixed across years."""

import json as _path_json
from pathlib import Path as _ConfigPath


_DATA_PATHS = _path_json.loads(
    (_ConfigPath(__file__).resolve().parents[1] / "paths.json").read_text(encoding="utf-8")
)

from pathlib import Path
import os
import re
import time
import hashlib
import warnings

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import geopandas as gpd

import rasterio
from rasterio.features import geometry_mask, geometry_window
from rasterio.windows import Window

from pyproj import CRS, Transformer

import shapely
from shapely import points
from shapely.strtree import STRtree

from scipy.stats import (
    theilslopes,
    norm,
    wilcoxon,
    binomtest,
)


BASE = Path(_DATA_PATHS["BASE"])


MASTER_FILE = BASE / "dbf" / "cata-90" / "cata-90.xlsx"
MASTER_SHEET = "Sheet4"
MASTER_BASIN_COL = "basin"


BASIN_MAIN = BASE / "basin-selected-renamed"
RIVER_MAIN = BASE / "river-selected-renamed"


BASIN_FIX2 = BASE / "basin-selected-renamed2"
RIVER_FIX2 = BASE / "river-selected-renamed2"

BASIN_FIX_H = BASE / "basin-selected-renamed-h"
RIVER_FIX_H = BASE / "river-selected-renamed-h"


GLOBPOP_DIR = BASE / "GlobPOP"
POP_PATTERN = re.compile(
    r"^GlobPOP_Count_30arc_(\d{4})_I32\.tiff$",
    re.IGNORECASE,
)

START_YEAR = 1990
END_YEAR = 2020
YEARS = list(range(START_YEAR, END_YEAR + 1))
EXPECTED_N_YEARS = len(YEARS)
EXPECTED_N_BASINS = 80


OUT_DIR = BASE / "R1D_pixel_riverward_final"
CACHE_DIR = OUT_DIR / "distance_cache"
BASIN_YEAR_DIR = OUT_DIR / "basin_year"

for _d in [OUT_DIR, CACHE_DIR, BASIN_YEAR_DIR]:
    _d.mkdir(parents=True, exist_ok=True)


POINT_CHUNK_SIZE = 100_000


ANALYSIS_VERSION = "pixel_absolute_distance_v3_20260824"


FORCE_RECALCULATE_DISTANCE = False
FORCE_RECALCULATE_YEARLY = False


REQUIRE_ALL_80_FOR_GLOBAL = True


ALPHA = 0.05
FDR_ALPHA = 0.05
THEILSEN_CI = 0.95


ALIASES = {
    "HAI": "HAI_HE",
    "HUAI": "HUAI_HE",
    "LIAO": "LIAO_HE",
    "MOBILE": "MOBILE_RIVER",
    "TARIM": "TARIM_HE",
    "YELLOW": "YELLOW_RIVER",
}


def clean_name(x):

    key = str(x).strip().upper()
    key = key.replace(" ", "_").replace("-", "_")
    key = re.sub(r"_+", "_", key).strip("_")
    return ALIASES.get(key, key)


def safe_filename(x):
    x = clean_name(x)
    return re.sub(r'[\\/:*?"<>|]+', "_", x)


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


def discover_population_files():
    if not GLOBPOP_DIR.exists():
        raise FileNotFoundError(f"GlobPOP directory not found: {GLOBPOP_DIR}")

    files = {}
    for fp in GLOBPOP_DIR.iterdir():
        if not fp.is_file():
            continue
        m = POP_PATTERN.match(fp.name)
        if m is None:
            continue
        year = int(m.group(1))
        if START_YEAR <= year <= END_YEAR:
            files[year] = fp

    files = dict(sorted(files.items()))
    missing = sorted(set(YEARS) - set(files))

    if missing:
        raise RuntimeError(f"Missing GlobPOP years: {missing}")

    if len(files) != EXPECTED_N_YEARS:
        raise RuntimeError(
            f"Expected {EXPECTED_N_YEARS} GlobPOP rasters, found {len(files)}."
        )

    return files


POP_FILES = discover_population_files()
REF_FILE = POP_FILES[START_YEAR]


def affine_equal(a, b, atol=1e-12):
    return np.allclose(
        np.asarray(tuple(a), dtype=float),
        np.asarray(tuple(b), dtype=float),
        rtol=0,
        atol=atol,
    )


print("=" * 90)
print("R1D PIXEL-LEVEL RIVERWARD ANALYSIS")
print("=" * 90)
print(f"Analysis version: {ANALYSIS_VERSION}")
print(f"Output: {OUT_DIR}")

print("\nSTEP 1 — Checking GlobPOP rasters")

with rasterio.open(REF_FILE) as ref:
    REF_CRS = ref.crs
    REF_TRANSFORM = ref.transform
    REF_WIDTH = ref.width
    REF_HEIGHT = ref.height
    REF_BOUNDS = ref.bounds

if REF_CRS is None:
    raise RuntimeError("Reference GlobPOP raster has no CRS.")

raster_check_rows = []

for year, fp in POP_FILES.items():
    with rasterio.open(fp) as src:
        ok_crs = src.crs == REF_CRS
        ok_shape = (src.width == REF_WIDTH) and (src.height == REF_HEIGHT)
        ok_transform = affine_equal(src.transform, REF_TRANSFORM)
        ok = ok_crs and ok_shape and ok_transform

        raster_check_rows.append(
            {
                "year": year,
                "file": str(fp),
                "crs_ok": ok_crs,
                "shape_ok": ok_shape,
                "transform_ok": ok_transform,
                "all_ok": ok,
            }
        )

        if not ok:
            raise RuntimeError(
                f"GlobPOP grid mismatch in {year}: {fp}\n"
                "All 31 rasters must share CRS, width, height and affine transform."
            )

pd.DataFrame(raster_check_rows).to_csv(
    OUT_DIR / "00_raster_grid_check.csv",
    index=False,
    encoding="utf-8-sig",
)

print(f"✓ {EXPECTED_N_YEARS} annual rasters found and perfectly aligned")
print(f"  CRS: {REF_CRS}")
print(f"  Shape: {REF_HEIGHT} × {REF_WIDTH}")
print(f"  Resolution: {abs(REF_TRANSFORM.a)}, {abs(REF_TRANSFORM.e)}")


print("\nSTEP 2 — Reading official 80-basin master list")

if not MASTER_FILE.exists():
    raise FileNotFoundError(f"Master file not found: {MASTER_FILE}")

master_raw = pd.read_excel(MASTER_FILE, sheet_name=MASTER_SHEET)

if MASTER_BASIN_COL not in master_raw.columns:
    raise ValueError(
        f"Column '{MASTER_BASIN_COL}' not found in {MASTER_FILE}.\n"
        f"Columns: {master_raw.columns.tolist()}"
    )

master = (
    master_raw[[MASTER_BASIN_COL]]
    .dropna()
    .copy()
)
master["basin_key"] = master[MASTER_BASIN_COL].map(clean_name)

if master["basin_key"].duplicated().any():
    dup = master.loc[master["basin_key"].duplicated(False), "basin_key"].tolist()
    raise RuntimeError(f"Duplicate basin names in master list: {dup}")

MASTER_BASINS = master["basin_key"].tolist()

if len(MASTER_BASINS) != EXPECTED_N_BASINS:
    raise RuntimeError(
        f"Master list must contain exactly {EXPECTED_N_BASINS} unique basins; "
        f"found {len(MASTER_BASINS)}."
    )

print(f"✓ Master list contains exactly {len(MASTER_BASINS)} unique major basins")
print("  COLORADO and RIO_COLORADO are treated as separate basins.")


def read_shp_folder(folder):
    folder = Path(folder)
    if not folder.exists():
        return {}
    return {
        clean_name(fp.stem): fp
        for fp in folder.glob("*.shp")
    }


def build_latest_map(main_folder, fix2_folder, fixh_folder):

    resolved = {}
    source = {}

    for label, folder in [
        ("main", main_folder),
        ("renamed2", fix2_folder),
        ("renamed-h", fixh_folder),
    ]:
        current = read_shp_folder(folder)
        for key, fp in current.items():
            resolved[key] = fp
            source[key] = label

    return resolved, source


BASIN_MAP, BASIN_SOURCE = build_latest_map(
    BASIN_MAIN,
    BASIN_FIX2,
    BASIN_FIX_H,
)

RIVER_MAP, RIVER_SOURCE = build_latest_map(
    RIVER_MAIN,
    RIVER_FIX2,
    RIVER_FIX_H,
)

missing_basin = [b for b in MASTER_BASINS if b not in BASIN_MAP]
missing_river = [b for b in MASTER_BASINS if b not in RIVER_MAP]

if missing_basin or missing_river:
    raise RuntimeError(
        "Master-list shapefile resolution failed.\n"
        f"Missing basin shapefiles: {missing_basin}\n"
        f"Missing river shapefiles: {missing_river}"
    )

resolution_rows = []
for basin in MASTER_BASINS:
    resolution_rows.append(
        {
            "basin_name": basin,
            "basin_shp": str(BASIN_MAP[basin]),
            "basin_version": BASIN_SOURCE[basin],
            "river_shp": str(RIVER_MAP[basin]),
            "river_version": RIVER_SOURCE[basin],
        }
    )

resolution_df = pd.DataFrame(resolution_rows)
resolution_df.to_csv(
    OUT_DIR / "00_master_input_resolution.csv",
    index=False,
    encoding="utf-8-sig",
)

print("\nSTEP 3 — Resolving latest basin/river shapefiles")
print(f"✓ Resolved basin shapefiles: {len(resolution_df)} / 80")
print(f"✓ Resolved river shapefiles: {len(resolution_df)} / 80")

fix_used = resolution_df[
    (resolution_df["basin_version"] != "main")
    | (resolution_df["river_version"] != "main")
]
if len(fix_used):
    print("\nLatest-version overrides used by the official 80-basin sample:")
    print(
        fix_used[
            ["basin_name", "basin_version", "river_version"]
        ].to_string(index=False)
    )


ref_stat = REF_FILE.stat()
GRID_SIGNATURE_TEXT = "|".join(
    [
        str(REF_CRS),
        repr(tuple(REF_TRANSFORM)),
        str(REF_WIDTH),
        str(REF_HEIGHT),
        str(ref_stat.st_size),
        str(ref_stat.st_mtime_ns),
    ]
)
GRID_SIGNATURE = hashlib.sha1(GRID_SIGNATURE_TEXT.encode("utf-8")).hexdigest()


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


print("\n" + "=" * 90)
print("STEP 4 — Pixel-level calculation for 80 major basins")
print("=" * 90)

all_basin_year = []
failed_basins = []

for i, basin_name in enumerate(MASTER_BASINS, start=1):
    basin_fp = BASIN_MAP[basin_name]
    river_fp = RIVER_MAP[basin_name]
    signature = basin_run_signature(basin_name, basin_fp, river_fp)

    slug = safe_filename(basin_name)
    basin_csv = BASIN_YEAR_DIR / f"{slug}__{signature}.csv"

    print("\n" + "-" * 90)
    print(f"[{i:02d}/80] {basin_name}")
    print(f"  basin: {basin_fp}")
    print(f"  river: {river_fp}")

    t0 = time.time()

    try:
        yearly = None

        if basin_csv.exists() and not FORCE_RECALCULATE_YEARLY:
            try:
                existing = pd.read_csv(basin_csv)
                if valid_existing_yearly(existing, basin_name, signature):
                    yearly = existing
                    print("  ✓ valid basin-year result reused")
            except Exception:
                yearly = None

        if yearly is None:
            basin_geom = load_basin_geometry(basin_fp)

            template, signature2 = get_distance_template(
                basin_name,
                basin_fp,
                river_fp,
                basin_geom,
            )

            if signature2 != signature:
                raise RuntimeError("Internal signature mismatch.")

            yearly = calculate_basin_year_series(
                basin_name,
                signature,
                template,
            )

            tmp_csv = basin_csv.with_suffix(".tmp.csv")
            yearly.to_csv(tmp_csv, index=False, encoding="utf-8-sig")
            os.replace(tmp_csv, basin_csv)

        all_basin_year.append(yearly)
        print(f"  ✓ finished in {time.time() - t0:.1f} s")

    except Exception as exc:
        print(f"  ✗ FAILED: {exc}")
        failed_basins.append(
            {
                "basin_name": basin_name,
                "basin_shp": str(basin_fp),
                "river_shp": str(river_fp),
                "error": repr(exc),
            }
        )


FAILED_FILE = OUT_DIR / "00_failed_basins.csv"

if failed_basins:
    pd.DataFrame(failed_basins).to_csv(
        FAILED_FILE,
        index=False,
        encoding="utf-8-sig",
    )
else:
    if FAILED_FILE.exists():
        FAILED_FILE.unlink()

if len(all_basin_year) == 0:
    raise RuntimeError("All 80 basin calculations failed.")

basin_year = pd.concat(all_basin_year, ignore_index=True)
basin_year["basin_name"] = basin_year["basin_name"].map(clean_name)
basin_year = basin_year.sort_values(["basin_name", "year"]).reset_index(drop=True)

BASIN_YEAR_OUTPUT = OUT_DIR / "01_basin_year_pixel_weighted_distance.csv"
basin_year.to_csv(BASIN_YEAR_OUTPUT, index=False, encoding="utf-8-sig")

print("\n" + "=" * 90)
print("PIXEL CALCULATION FINISHED")
print("=" * 90)
print(f"Successful basins: {basin_year['basin_name'].nunique()}")
print(f"Failed basins: {len(failed_basins)}")


def finite_count(series):
    return int(np.isfinite(pd.to_numeric(series, errors="coerce")).sum())


diagnostic = (
    basin_year.groupby("basin_name", dropna=False)
    .agg(
        n_rows=("year", "size"),
        n_years=("year", "nunique"),
        first_year=("year", "min"),
        last_year=("year", "max"),
        n_valid_distance_years=("pw_mean_distance_km", finite_count),
        min_distance_km=("pw_mean_distance_km", "min"),
        max_distance_km=("pw_mean_distance_km", "max"),
        mean_distance_km=("pw_mean_distance_km", "mean"),
    )
    .reset_index()
)

diagnostic["complete_31_years"] = (
    (diagnostic["n_rows"] == EXPECTED_N_YEARS)
    & (diagnostic["n_years"] == EXPECTED_N_YEARS)
    & (diagnostic["first_year"] == START_YEAR)
    & (diagnostic["last_year"] == END_YEAR)
    & (diagnostic["n_valid_distance_years"] == EXPECTED_N_YEARS)
)


missing_successful = sorted(set(MASTER_BASINS) - set(diagnostic["basin_name"]))
if missing_successful:
    diagnostic = pd.concat(
        [
            diagnostic,
            pd.DataFrame(
                {
                    "basin_name": missing_successful,
                    "n_rows": 0,
                    "n_years": 0,
                    "first_year": np.nan,
                    "last_year": np.nan,
                    "n_valid_distance_years": 0,
                    "min_distance_km": np.nan,
                    "max_distance_km": np.nan,
                    "mean_distance_km": np.nan,
                    "complete_31_years": False,
                }
            ),
        ],
        ignore_index=True,
    )


order_map = {name: i for i, name in enumerate(MASTER_BASINS)}
diagnostic["_order"] = diagnostic["basin_name"].map(order_map)
diagnostic = diagnostic.sort_values("_order").drop(columns="_order")

DIAGNOSTIC_FILE = OUT_DIR / "00_data_diagnostics.csv"
diagnostic.to_csv(DIAGNOSTIC_FILE, index=False, encoding="utf-8-sig")


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


print("\nSTEP 5 — Basin-level Theil–Sen + Mann–Kendall")

trend_rows = []
insufficient_years = []

for basin_name in MASTER_BASINS:
    g = basin_year[basin_year["basin_name"] == basin_name].copy()
    g = g.sort_values("year")


    if g["year"].duplicated().any():
        insufficient_years.append(
            {
                "basin_name": basin_name,
                "reason": "duplicate_years",
                "n_rows": len(g),
                "n_valid_years": g["year"].nunique(),
                "missing_years": "",
            }
        )
        continue

    g["pw_mean_distance_km"] = pd.to_numeric(
        g["pw_mean_distance_km"], errors="coerce"
    )

    valid_g = g[np.isfinite(g["pw_mean_distance_km"])].copy()
    actual_years = set(pd.to_numeric(valid_g["year"], errors="coerce").dropna().astype(int))
    missing_years = sorted(set(YEARS) - actual_years)

    if len(valid_g) != EXPECTED_N_YEARS or missing_years:
        insufficient_years.append(
            {
                "basin_name": basin_name,
                "reason": "incomplete_31_year_series",
                "n_rows": len(g),
                "n_valid_years": len(valid_g),
                "missing_years": ",".join(map(str, missing_years)),
            }
        )
        continue

    x = valid_g["year"].to_numpy(dtype=float)
    y = valid_g["pw_mean_distance_km"].to_numpy(dtype=float)

    slope, intercept, ci_low, ci_high = theilslopes(
        y,
        x,
        alpha=THEILSEN_CI,
    )

    mk = mann_kendall_test(y)

    slope_decade = slope * 10.0
    ci_low_decade = ci_low * 10.0
    ci_high_decade = ci_high * 10.0

    if slope < 0:
        direction = "Riverward"
    elif slope > 0:
        direction = "Away from rivers"
    else:
        direction = "Stable"

    trend_rows.append(
        {
            "analysis_version": ANALYSIS_VERSION,
            "basin_name": basin_name,
            "n_years": EXPECTED_N_YEARS,
            "distance_1990_km": y[0],
            "distance_2020_km": y[-1],
            "absolute_change_1990_2020_km": y[-1] - y[0],
            "median_distance_km": float(np.median(y)),
            "theilsen_intercept_km": intercept,
            "theilsen_slope_km_year": slope,
            "theilsen_slope_km_decade": slope_decade,
            "CI_low_km_decade": ci_low_decade,
            "CI_high_km_decade": ci_high_decade,
            "direction": direction,
            "MK_trend": mk["trend"],
            "MK_S": mk["S"],
            "MK_var_S": mk["var_S"],
            "MK_z": mk["z"],
            "MK_tau": mk["tau"],
            "MK_p": mk["p"],
        }
    )


INSUFFICIENT_FILE = OUT_DIR / "00_insufficient_years.csv"
if insufficient_years:
    pd.DataFrame(insufficient_years).to_csv(
        INSUFFICIENT_FILE,
        index=False,
        encoding="utf-8-sig",
    )
else:
    if INSUFFICIENT_FILE.exists():
        INSUFFICIENT_FILE.unlink()

trend_df = pd.DataFrame(trend_rows)
if len(trend_df) == 0:
    raise RuntimeError("No basin has a complete 31-year trend series.")

trend_df["MK_q_FDR"] = bh_fdr(trend_df["MK_p"].to_numpy())
trend_df["significant_FDR"] = trend_df["MK_q_FDR"] < FDR_ALPHA

trend_df["FDR_result"] = np.select(
    [
        trend_df["significant_FDR"]
        & (trend_df["theilsen_slope_km_decade"] < 0),
        trend_df["significant_FDR"]
        & (trend_df["theilsen_slope_km_decade"] > 0),
    ],
    [
        "Significant riverward",
        "Significant away",
    ],
    default="No significant trend",
)


trend_df["_order"] = trend_df["basin_name"].map(order_map)
trend_df = trend_df.sort_values("_order").drop(columns="_order")

TREND_OUTPUT = OUT_DIR / "02_basin_pixel_TheilSen_MK_FDR.csv"
trend_df.to_csv(TREND_OUTPUT, index=False, encoding="utf-8-sig")


n_complete = trend_df["basin_name"].nunique()

print("\n" + "=" * 90)
print("TREND COMPLETENESS")
print("=" * 90)
print(f"Complete basins: {n_complete} / 80")

if n_complete != EXPECTED_N_BASINS:
    print("Global inference NOT performed.")
    print(f"See: {FAILED_FILE}")
    print(f"See: {INSUFFICIENT_FILE}")
    print(f"See: {DIAGNOSTIC_FILE}")

    if REQUIRE_ALL_80_FOR_GLOBAL:
        raise RuntimeError(
            f"Only {n_complete}/80 basins have complete trends. "
            "Resolve missing basins before global inference."
        )


print("\nSTEP 6 — Global directional inference")

slopes = trend_df["theilsen_slope_km_decade"].to_numpy(dtype=float)
slopes = slopes[np.isfinite(slopes)]

n_total = len(slopes)
n_riverward = int(np.sum(slopes < 0))
n_away = int(np.sum(slopes > 0))
n_stable = int(np.sum(slopes == 0))
n_nonzero = n_riverward + n_away


try:
    w = wilcoxon(
        slopes,
        alternative="less",
        zero_method="wilcox",
    )
    wilcoxon_W = float(w.statistic)
    wilcoxon_p = float(w.pvalue)
except Exception:
    wilcoxon_W = np.nan
    wilcoxon_p = np.nan


if n_nonzero > 0:
    sign_result = binomtest(
        n_riverward,
        n=n_nonzero,
        p=0.5,
        alternative="greater",
    )
    sign_p = float(sign_result.pvalue)
else:
    sign_p = np.nan


if n_nonzero > 0:
    sign_two_p = float(
        binomtest(
            n_riverward,
            n=n_nonzero,
            p=0.5,
            alternative="two-sided",
        ).pvalue
    )
else:
    sign_two_p = np.nan


global_result = pd.DataFrame(
    [
        {
            "analysis_version": ANALYSIS_VERSION,
            "n_basins": n_total,
            "riverward_n": n_riverward,
            "riverward_pct": n_riverward / n_total * 100,
            "away_n": n_away,
            "away_pct": n_away / n_total * 100,
            "stable_n": n_stable,
            "median_slope_km_decade": float(np.median(slopes)),
            "mean_slope_km_decade": float(np.mean(slopes)),
            "Q1_slope_km_decade": float(np.percentile(slopes, 25)),
            "Q3_slope_km_decade": float(np.percentile(slopes, 75)),
            "Wilcoxon_W": wilcoxon_W,
            "Wilcoxon_p_one_sided_riverward": wilcoxon_p,
            "SignTest_p_one_sided_majority_riverward": sign_p,
            "SignTest_p_two_sided": sign_two_p,
            "FDR_significant_riverward_n": int(
                (
                    trend_df["significant_FDR"]
                    & (trend_df["theilsen_slope_km_decade"] < 0)
                ).sum()
            ),
            "FDR_significant_away_n": int(
                (
                    trend_df["significant_FDR"]
                    & (trend_df["theilsen_slope_km_decade"] > 0)
                ).sum()
            ),
        }
    ]
)

GLOBAL_OUTPUT = OUT_DIR / "03_global_pixel_riverward_test.csv"
global_result.to_csv(GLOBAL_OUTPUT, index=False, encoding="utf-8-sig")


g = global_result.iloc[0]

summary_lines = [
    "Pixel-level population-weighted river-distance trend analysis",
    "=" * 65,
    "",
    f"Analysis version: {ANALYSIS_VERSION}",
    f"Official major basins: {int(g['n_basins'])}",
    "",
    "Metric:",
    "D_bt = sum_i(P_bti * d_bi) / sum_i(P_bti)",
    "where d_bi is the pixel-centre distance to the nearest river (km).",
    "",
    f"Riverward basins: {int(g['riverward_n'])} ({g['riverward_pct']:.1f}%)",
    f"Away-from-river basins: {int(g['away_n'])} ({g['away_pct']:.1f}%)",
    f"Stable basins: {int(g['stable_n'])}",
    "",
    f"Median Theil–Sen slope: {g['median_slope_km_decade']:.6f} km decade^-1",
    f"Q1–Q3: {g['Q1_slope_km_decade']:.6f} to {g['Q3_slope_km_decade']:.6f} km decade^-1",
    "",
    f"One-sided Wilcoxon P (riverward): {g['Wilcoxon_p_one_sided_riverward']:.6g}",
    f"One-sided sign-test P (>50% riverward): {g['SignTest_p_one_sided_majority_riverward']:.6g}",
    f"Two-sided sign-test P: {g['SignTest_p_two_sided']:.6g}",
    "",
    f"FDR-significant riverward basins: {int(g['FDR_significant_riverward_n'])}",
    f"FDR-significant away basins: {int(g['FDR_significant_away_n'])}",
    "",
    "North/South inference was intentionally not run in this script;",
    "resolve the complete 80-basin region classification first.",
]

SUMMARY_FILE = OUT_DIR / "04_manuscript_ready_summary.txt"
SUMMARY_FILE.write_text("\n".join(summary_lines), encoding="utf-8")


print("\n" + "=" * 90)
print("GLOBAL PIXEL-LEVEL RESULT")
print("=" * 90)
print(global_result.to_string(index=False))

print("\n" + "=" * 90)
print("OUTPUT FILES")
print("=" * 90)
print(f"00 input resolution : {OUT_DIR / '00_master_input_resolution.csv'}")
print(f"00 raster check     : {OUT_DIR / '00_raster_grid_check.csv'}")
print(f"00 diagnostics      : {DIAGNOSTIC_FILE}")
if FAILED_FILE.exists():
    print(f"00 failed basins    : {FAILED_FILE}")
if INSUFFICIENT_FILE.exists():
    print(f"00 incomplete years : {INSUFFICIENT_FILE}")
print(f"01 basin-year       : {BASIN_YEAR_OUTPUT}")
print(f"02 basin trends     : {TREND_OUTPUT}")
print(f"03 global test      : {GLOBAL_OUTPUT}")
print(f"04 summary          : {SUMMARY_FILE}")
print(f"distance cache      : {CACHE_DIR}")
print("\n✓ ALL DONE")
