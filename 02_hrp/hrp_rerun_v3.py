
"""Calculate HRP profiles, effective distances, thresholds and annual types.
Stratified within-cell sampling assumes uniform population within each cell.
Use --self-check for synthetic checks or --convergence N for sampling sensitivity."""

import json as _path_json
from pathlib import Path as _ConfigPath


_DATA_PATHS = _path_json.loads(
    (_ConfigPath(__file__).resolve().parents[1] / "paths.json").read_text(encoding="utf-8")
)
from pathlib import Path
from datetime import datetime
import argparse
import itertools
import json
import shutil
import tempfile
import time
import traceback

import numpy as np
import pandas as pd
import geopandas as gpd
import shapely
import rasterio
from rasterio.features import geometry_window, rasterize
from rasterio.windows import Window
from pyproj import CRS, Transformer
from scipy import ndimage
from scipy.spatial import cKDTree


NETWORK_RUN = Path(_DATA_PATHS["NETWORK_RUN"])
POP_DIR = Path(_DATA_PATHS["POP_DIR"])
POP_PATTERN = "GlobPOP_Count_30arc_{year}_I32.tiff"
OUTPUT_ROOT = Path(_DATA_PATHS["HRP_OUTPUT_ROOT"])


UNIT_SETS = {
    "sub": dict(table=NETWORK_RUN / "04_subbasin_master.csv", id="unit_key",
                basin="basin_shp", river="river_shp", level="sub"),
    "major": dict(table=NETWORK_RUN / "06_major_rule_networks.csv", id="major",
                  basin="basin_shp", river="river_shp_original", level="major"),
    "major_rule": dict(table=NETWORK_RUN / "06_major_rule_networks.csv", id="major",
                       basin="basin_shp", river="river_shp", level="major"),
}
SCENARIOS = {
    "main": ["major", "sub"],
    "rule_networks": ["major_rule", "sub"],
}


LEGACY_TYPES = {}


YEARS = list(range(1990, 2021))
REFERENCE_YEAR = 2020
COVERAGE = 0.90
N_BANDS = 100
MIN_RUN = 5
THRESHOLD_POLICY = "global"
THRESHOLD_SAMPLE = "annual"
HIGH_IF_EQUAL = True
MARGIN_SCAN_PP = 1.0
FRAGILE_MARGIN_PP = 0.05
CELLS_PER_INTERVAL = 3.0
MIN_SAMPLES_SIDE = 2
MAX_SAMPLES_SIDE = 160
CHUNK_SAMPLES = 1_000_000
NODE_STEP = 4
KD_NEIGHBOURS = 8
R90_BINS = 200_000
RANDOM_SEED = 20261018
WORKERS = -1
METHOD_VERSION = "hrp-v3-exact-distance-2026-10-18"


def band_shares(pop):

    pop = np.asarray(pop, dtype=float)
    total = pop.sum(axis=-1, keepdims=True)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(total > 0, pop / total * 100.0, np.nan)


def is_high(s, threshold):
    s = np.asarray(s, dtype=float)
    return np.isfinite(s) & ((s >= threshold) if HIGH_IF_EQUAL else (s > threshold))


def retained_runs(high, min_len=MIN_RUN):
    """Remove runs shorter than min_len, then merge adjacent equal states; end indices are exclusive."""
    runs, i, n = [], 0, len(high)
    while i < n:
        j = i
        while j < n and high[j] == high[i]:
            j += 1
        if j - i >= min_len:
            if runs and runs[-1][0] == bool(high[i]):
                runs[-1][2] = j
            else:
                runs.append([bool(high[i]), i, j])
        i = j
    return runs


def strict_run_mask(high, min_len=MIN_RUN):

    high = np.asarray(high, dtype=bool)
    out = np.zeros(len(high), dtype=bool)
    edges = np.diff(np.r_[False, high, False].astype(int))
    for a, b in zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)):
        if b - a >= min_len:
            out[a:b] = True
    return out


def hrp_type(high):
    runs = retained_runs(high)
    n_high = sum(1 for r in runs if r[0])
    if not runs:
        return "REVIEW", runs, n_high
    if n_high == 1:
        return ("Near-river" if runs[0][0] else "Far-river"), runs, n_high
    return "Dispersion", runs, n_high


def type_margin(s, high, threshold, hrp):

    gap = np.abs(s - threshold)
    for k in np.argsort(gap):
        if not gap[k] < MARGIN_SCAN_PP:
            break
        flipped = high.copy()
        flipped[k] = not flipped[k]
        if hrp_type(flipped)[0] != hrp:
            return float(gap[k])
    return np.nan


def classify(pop, threshold, full_pop=np.nan):
    pop = np.asarray(pop, dtype=float)
    s = band_shares(pop)
    high = is_high(s, threshold)
    hrp, runs, n_high = hrp_type(high)
    agglo = np.zeros(len(pop), dtype=bool)
    for state, a, b in runs:
        if state:
            agglo[a:b] = True
    strict = strict_run_mask(high)
    total = float(pop.sum())
    above = float(pop[high].sum()) if high.sum() >= MIN_RUN else np.nan
    first = next((r for r in runs if r[0]), None)
    pct = (lambda x: 100 * x / total) if total > 0 else (lambda x: np.nan)
    return dict(
        HRP_type=hrp,
        classification="-".join("high" if r[0] else "low" for r in runs),
        n_agglomerations=n_high,
        first_agglomeration_band=(first[1] + 1) if first else np.nan,
        dispersion_kind=("none" if n_high == 0 else "multiple") if hrp == "Dispersion" else "",
        type_margin_pp=type_margin(s, high, threshold, hrp),
        threshold=threshold,
        profile_population=total,
        full_basin_population=float(full_pop),
        captured_population_pct=100 * total / full_pop if full_pop > 0 else np.nan,
        agglomeration_share_pct=pct(float(pop[agglo].sum())),
        strict_run_share_pct=pct(float(pop[strict].sum())),
        above_threshold_share_pct=pct(above),
        above_threshold_share_of_full_basin_pct=100 * above / full_pop if full_pop > 0 else np.nan,
        high_band_count=int(high.sum()),
        agglomeration_band_count=int(agglo.sum()),
        high_bands=",".join(map(str, np.flatnonzero(high) + 1)),
        agglomeration_bands=";".join(f"{a + 1}-{b}" for st, a, b in runs if st),
    )


def weighted_percentile(v, w, q):
    o = np.argsort(v, kind="stable")
    v, w = v[o], w[o]
    cw = np.cumsum(w) - 0.5 * w
    return np.interp(q * w.sum(), cw, v)


def iqr_threshold(values, weights=None):
    v = np.asarray(values, dtype=float)
    w = np.ones(len(v)) if weights is None else np.asarray(weights, dtype=float)
    ok = np.isfinite(v)
    v, w = v[ok], w[ok]
    if weights is None:
        q1, q3 = np.percentile(v, [25, 75])
    else:
        q1, q3 = weighted_percentile(v, w, 0.25), weighted_percentile(v, w, 0.75)
    return dict(Q1=q1, Q3=q3, IQR=q3 - q1, T=q3 + 1.5 * (q3 - q1), n_values=len(v))


def read_geometry(path, polygon):
    g = gpd.read_file(path)
    if g.crs is None:
        raise ValueError(f"No CRS: {path}")
    g = g.to_crs(4326)
    parts = []

    def collect(geom):
        if geom is None or geom.is_empty:
            return
        if polygon and not geom.is_valid:
            geom = shapely.make_valid(geom)
        wanted = ("Polygon",) if polygon else ("LineString", "LinearRing")
        if geom.geom_type in wanted:
            parts.append(geom)
        elif hasattr(geom, "geoms"):
            for p in geom.geoms:
                collect(p)

    for geom in g.geometry:
        collect(geom)
    if not parts:
        raise ValueError(f"No usable geometry: {path}")
    if polygon:
        return shapely.union_all(parts)
    return shapely.MultiLineString([shapely.LineString(p.coords) for p in parts])


def reproject(geom, transformer):
    return shapely.transform(
        geom, lambda xy: np.column_stack(transformer.transform(xy[:, 0], xy[:, 1])))


def population(src, window):
    a = src.read(1, window=window, masked=True)
    valid = ~np.ma.getmaskarray(a) & np.isfinite(a.data) & (a.data >= 0)
    return np.where(valid, a.data, 0).astype("float64"), valid


def int_window(win, src):
    col0, row0 = int(np.floor(win.col_off)), int(np.floor(win.row_off))
    col1 = int(np.ceil(win.col_off + win.width))
    row1 = int(np.ceil(win.row_off + win.height))
    w = Window(col0, row0, col1 - col0, row1 - row0)
    return w.intersection(Window(0, 0, src.width, src.height))


def lower_quantile(d, w, q):

    o = np.argsort(d, kind="stable")
    cw = np.cumsum(w[o])
    return float(d[o][min(int(np.searchsorted(cw, q * cw[-1])), len(cw) - 1)])


class PixelGeometry:


    def __init__(self, wt, H, W, src_crs, metric, step=NODE_STEP):
        tr = Transformer.from_crs(src_crs, metric, always_xy=True)
        self.cols = np.unique(np.r_[np.arange(0, W + 1, step), W]).astype(float)
        self.rows = np.unique(np.r_[np.arange(0, H + 1, step), H]).astype(float)
        C, R = np.meshgrid(self.cols, self.rows)
        self.X, self.Y = (np.asarray(a, dtype=float) for a in tr.transform(*(wt * (C, R))))
        c0, r0 = np.minimum(C, W - 1), np.minimum(R, H - 1)
        P = {k: np.array(tr.transform(*(wt * (c0 + dc, r0 + dr))))
             for k, (dc, dr) in dict(a=(0, 0), b=(1, 0), c=(1, 1), d=(0, 1),
                                     o=(0.5, 0.5)).items()}
        half = max(float(np.hypot(*(P[k] - P["o"])).max()) for k in "abcd")
        self.half_diag = 1.01 * half + 2.0
        self.side = max(float(np.hypot(*(P["b"] - P["a"])).max()),
                        float(np.hypot(*(P["d"] - P["a"])).max()))
        if CRS.from_user_input(src_crs).is_geographic:
            top = wt.f + np.arange(H) * wt.e
            self.row_area = (6371.0072 ** 2) * np.radians(abs(wt.a)) * np.abs(
                np.sin(np.radians(top)) - np.sin(np.radians(top + wt.e)))
        else:
            self.row_area = np.full(H, abs(wt.a * wt.e) / 1e6)
        self.pixel_m = float(np.sqrt(self.row_area[H // 2]) * 1000)

    @staticmethod
    def _locate(v, nodes):
        i0 = np.clip(np.searchsorted(nodes, v, side="right") - 1, 0, len(nodes) - 2)
        return i0, (v - nodes[i0]) / (nodes[i0 + 1] - nodes[i0])

    def __call__(self, fc, fr):
        c0, tc = self._locate(fc, self.cols)
        r0, tr = self._locate(fr, self.rows)
        out = []
        for A in (self.X, self.Y):
            top = A[r0, c0] * (1 - tc) + A[r0, c0 + 1] * tc
            bot = A[r0 + 1, c0] * (1 - tc) + A[r0 + 1, c0 + 1] * tc
            out.append(top * (1 - tr) + bot * tr)
        return out[0], out[1]


class RiverDistance:
    """Exact point-to-segment river distances using a vertex KD-tree and radius search when nearest-vertex candidates are insufficient."""

    def __init__(self, river_m, spacing, k=KD_NEIGHBOURS):
        segs = []
        for part in shapely.get_parts(river_m):
            xy = np.asarray(part.coords, dtype=float)[:, :2]
            if len(xy) >= 2:
                segs.append(np.c_[xy[:-1], xy[1:]])
        if not segs:
            raise ValueError("River network has no segments")
        segs = np.concatenate(segs)
        L = np.hypot(segs[:, 2] - segs[:, 0], segs[:, 3] - segs[:, 1])
        segs, L = segs[L > 0], L[L > 0]
        if not len(segs):
            raise ValueError("River network has only zero-length segments")
        n = np.ceil(L / spacing).astype(np.int64) + 1
        sid = np.repeat(np.arange(len(segs)), n)
        f = (np.arange(n.sum()) - np.repeat(np.cumsum(n) - n, n)) / np.repeat(n - 1, n)
        self.x1, self.y1 = segs[:, 0].copy(), segs[:, 1].copy()
        self.dx, self.dy = segs[:, 2] - segs[:, 0], segs[:, 3] - segs[:, 1]
        self.l2 = self.dx ** 2 + self.dy ** 2
        self.sid = sid
        self.half2 = (float((L / (n - 1)).max()) / 2) ** 2
        self.tree = cKDTree(np.c_[self.x1[sid] + f * self.dx[sid], self.y1[sid] + f * self.dy[sid]])
        self.k = int(min(k, len(sid)))
        self.fallback = 0

    def _seg(self, x, y, c):
        x1, y1, dx, dy = self.x1[c], self.y1[c], self.dx[c], self.dy[c]
        t = np.clip(((x - x1) * dx + (y - y1) * dy) / self.l2[c], 0.0, 1.0)
        return np.hypot(x - x1 - t * dx, y - y1 - t * dy)

    def __call__(self, x, y):
        out = np.empty(len(x))
        for a in range(0, len(x), CHUNK_SAMPLES):
            out[a:a + CHUNK_SAMPLES] = self._query(x[a:a + CHUNK_SAMPLES], y[a:a + CHUNK_SAMPLES])
        return out

    def _query(self, x, y):
        xy = np.column_stack([x, y])
        dd, ii = self.tree.query(xy, k=self.k, workers=WORKERS)
        if dd.ndim == 1:
            dd, ii = dd[:, None], ii[:, None]
        best = np.full(len(x), np.inf)
        for j in range(ii.shape[1]):
            best = np.minimum(best, self._seg(x, y, self.sid[ii[:, j]]))
        rad = np.sqrt(best * best + self.half2)
        bad = np.flatnonzero(dd[:, -1] < rad)
        if len(bad):
            self.fallback += len(bad)
            lists = self.tree.query_ball_point(xy[bad], rad[bad] * (1 + 1e-12) + 1e-9,
                                               workers=WORKERS)
            n = np.fromiter((len(v) for v in lists), dtype=np.int64, count=len(lists))
            idx = np.fromiter(itertools.chain.from_iterable(lists), dtype=np.int64,
                              count=int(n.sum()))
            pt = np.repeat(bad, n)
            dseg = self._seg(x[pt], y[pt], self.sid[idx])
            best[bad] = np.minimum(best[bad], np.minimum.reduceat(dseg, np.r_[0, np.cumsum(n)[:-1]]))
        return best


_U = np.uint64
_C1, _C2, _C3 = _U(0x9E3779B97F4A7C15), _U(0xBF58476D1CE4E5B9), _U(0x94D049BB133111EB)


def _mix(z):

    z = z + _C1
    z = (z ^ (z >> _U(30))) * _C2
    z = (z ^ (z >> _U(27))) * _C3
    return z ^ (z >> _U(31))


class Sampler:


    def __init__(self, W, row0, col0, src_width, seed):
        self.W, self.row0, self.col0, self.sw = W, row0, col0, src_width
        self.key = _mix(np.array([seed], dtype=np.uint64))[0]

    def points(self, pixels, m):
        if m * m >= 2 ** 15:
            raise ValueError("Too many samples per pixel")
        k = m * m
        r, c = np.divmod(np.asarray(pixels, dtype=np.int64), self.W)
        g = (r + self.row0).astype(np.uint64) * _U(self.sw) + (c + self.col0).astype(np.uint64)
        z = ((np.repeat(g, k) << _U(16)) | np.tile(np.arange(k, dtype=np.uint64) << _U(1), len(g))) \
            ^ self.key
        u = (_mix(z) >> _U(11)).astype(np.float64) * (1.0 / 9007199254740992.0)
        v = (_mix(z ^ _U(1)) >> _U(11)).astype(np.float64) * (1.0 / 9007199254740992.0)
        s = np.tile(np.arange(k), len(g))
        return np.repeat(c, k) + (s // m + u) / m, np.repeat(r, k) + (s % m + v) / m


def unit_profile(basin_ll, river_ll, pop_paths, years, ref_year, density=1.0, seed=RANDOM_SEED):

    t_start = time.time()
    if basin_ll.bounds[2] - basin_ll.bounds[0] > 180:
        raise ValueError("Antimeridian geometry requires explicit longitude handling")
    p = basin_ll.representative_point()
    metric = CRS.from_proj4(f"+proj=aeqd +lat_0={p.y:.6f} +lon_0={p.x:.6f} "
                            "+datum=WGS84 +units=m +no_defs")
    river_m = reproject(river_ll, Transformer.from_crs(4326, metric, always_xy=True))
    with rasterio.open(pop_paths[ref_year]) as src:
        grid = (src.crs, src.transform, src.width, src.height)
        basin_src = reproject(basin_ll, Transformer.from_crs(4326, src.crs, always_xy=True))
        win = int_window(geometry_window(src, [basin_src], pad_x=2, pad_y=2), src)
        wt = src.window_transform(win)
        H, W = int(win.height), int(win.width)
        ref, _ = population(src, win)
        src_crs = src.crs
    ref = ref.ravel()
    geo = PixelGeometry(wt, H, W, src_crs, metric)
    hd = geo.half_diag
    sampler = Sampler(W, int(win.row_off), int(win.col_off), grid[2], seed)
    shapely.prepare(basin_src)
    dist = RiverDistance(river_m, max(geo.pixel_m / 4, 5.0))


    def burn(g):
        return rasterize([(g, 1)], out_shape=(H, W), transform=wt, all_touched=True,
                         dtype="uint8").astype(bool)

    ring = np.ones((3, 3), dtype=bool)
    edge = ndimage.binary_dilation(burn(basin_src.boundary), ring)
    cand = np.flatnonzero((ndimage.binary_dilation(burn(basin_src), ring) | edge).ravel())
    is_edge = edge.ravel()[cand]
    centre_in = shapely.contains_xy(basin_src, *(wt * (cand % W + 0.5, cand // W + 0.5)))
    keep = is_edge | centre_in
    cand, is_edge, centre_in = cand[keep], is_edge[keep], centre_in[keep]
    pos_edge = np.flatnonzero(is_edge)
    refc = ref[cand]


    d_c = dist(*geo(cand % W + 0.5, cand // W + 0.5))

    def inside_fraction(m):
        f = np.ones(len(cand))
        n = max(1, CHUNK_SAMPLES // (m * m))
        for a in range(0, len(pos_edge), n):
            part = pos_edge[a:a + n]
            fc, fr = sampler.points(cand[part], m)
            ins = shapely.contains_xy(basin_src, *(wt * (fc, fr)))
            f[part] = ins.reshape(len(part), m * m).mean(axis=1)
        return f

    def inside_samples(pos, m):

        n = max(1, CHUNK_SAMPLES // (m * m))
        for test in (False, True):
            sub = pos[is_edge[pos] == test]
            for a in range(0, len(sub), n):
                part = sub[a:a + n]
                fc, fr = sampler.points(cand[part], m)
                own = np.repeat(part, m * m)
                if test:
                    ins = shapely.contains_xy(basin_src, *(wt * (fc, fr)))
                    fc, fr, own = fc[ins], fr[ins], own[ins]
                X, Y = geo(fc, fr)
                yield own, X, Y

    def sides_needed(width):
        return int(np.ceil(geo.side * CELLS_PER_INTERVAL * density / width))


    r0 = lower_quantile(d_c, refc * centre_in, COVERAGE)
    m = sides_needed(max(r0 - hd, 0.25 * r0, 1.0) / N_BANDS)
    m = min(max(m, MIN_SAMPLES_SIDE), MAX_SAMPLES_SIDE)
    attempt = 0
    while True:
        attempt += 1
        fb = inside_fraction(m)
        mass = refc * fb
        full_ref = float(mass.sum())
        if full_ref <= 0:
            raise ValueError("Reference-year basin population is zero")
        rc = lower_quantile(d_c, mass, COVERAGE)
        lo, hi = rc - hd, rc + hd
        below = float(mass[d_c + hd < lo].sum())
        hist = np.zeros(R90_BINS)
        for own, X, Y in inside_samples(np.flatnonzero((d_c + hd >= lo) & (d_c - hd <= hi)), m):
            d = dist(X, Y)
            wgt = refc[own] / (m * m)
            below += float(wgt[d < lo].sum())
            s = (d >= lo) & (d <= hi)
            b = np.minimum(((d[s] - lo) / (hi - lo) * R90_BINS).astype(np.int64), R90_BINS - 1)
            hist += np.bincount(b, weights=wgt[s], minlength=R90_BINS)
        target = COVERAGE * full_ref
        cum = below + np.cumsum(hist)
        k = min(int(np.searchsorted(cum, target)), R90_BINS - 1)
        prev = below if k == 0 else cum[k - 1]
        within = min(max((target - prev) / hist[k], 0.0), 1.0) if hist[k] > 0 else 0.0
        r90 = max(lo + (k + within) / R90_BINS * (hi - lo), 1e-3)
        need = sides_needed(r90 / N_BANDS)
        if need <= m or m >= MAX_SAMPLES_SIDE or attempt >= 3:
            break
        m = min(need, MAX_SAMPLES_SIDE)


    width = r90 / N_BANDS

    def band(d):
        return np.clip(np.ceil(d / width), 1, N_BANDS).astype(np.int64)

    reach = d_c - hd <= r90
    whole = reach & (d_c + hd <= r90) & (band(d_c - hd) == band(d_c + hd))
    pos_w = np.flatnonzero(whole)
    pp, pb, pf = [pos_w], [band(d_c[pos_w])], [fb[pos_w]]
    n_samples = 0
    for own, X, Y in inside_samples(np.flatnonzero(reach & ~whole), m):
        d = dist(X, Y)
        n_samples += len(d)
        sel = d <= r90
        uk, cnt = np.unique(own[sel] * (N_BANDS + 1) + band(d[sel]), return_counts=True)
        pp.append(uk // (N_BANDS + 1))
        pb.append(uk % (N_BANDS + 1))
        pf.append(cnt / (m * m))
    pair_pix = cand[np.concatenate(pp)]
    pair_b = np.concatenate(pb)
    frac = np.concatenate(pf)
    band_area = np.bincount(pair_b, weights=frac * geo.row_area[pair_pix // W],
                            minlength=N_BANDS + 1)[1:]
    sums, full = [], []
    for year in years:
        with rasterio.open(pop_paths[year]) as src:
            if (src.crs, src.transform, src.width, src.height) != grid:
                raise ValueError(f"Raster grid differs in {year}; do not silently resample")
            pop, _ = population(src, win)
        pop = pop.ravel()
        sums.append(np.bincount(pair_b, weights=pop[pair_pix] * frac, minlength=N_BANDS + 1)[1:])
        full.append(float((pop[cand] * fb).sum()))
    sums, full = np.asarray(sums), np.asarray(full)
    if (full <= 0).any() or (sums.sum(axis=1) <= 0).any():
        raise ValueError("Zero population in some year")
    iref = list(years).index(ref_year)
    spacing = geo.side / m
    ratio = width / spacing
    ok = bool(ratio >= CELLS_PER_INTERVAL * density * (1 - 1e-9))
    flags = ([] if ok else ["BELOW_TARGET_DENSITY"]) + \
        (["INTERVAL_FINER_THAN_PIXEL"] if width < geo.pixel_m else [])
    audit = dict(metric_crs=f"AEQD {p.y:.4f},{p.x:.4f}", R90_km=r90 / 1000,
                 interval_m=width, pixel_m=geo.pixel_m, interval_to_pixel=width / geo.pixel_m,
                 samples_per_pixel_side=m, sample_spacing_m=spacing, interval_to_spacing=ratio,
                 reference_population_coverage=float(sums[iref].sum() / full[iref]),
                 basin_pixels=int((fb > 0).sum()), edge_pixels=int(len(pos_edge)),
                 whole_pixels=int(len(pos_w)), sampled_pixels=int((reach & ~whole).sum()),
                 band_samples=int(n_samples), distance_fallback_points=int(dist.fallback),
                 attempts=attempt, empty_bands=int((band_area == 0).sum()),
                 precision_ok=ok, flags=";".join(flags),
                 seconds=round(time.time() - t_start, 1))
    return sums, full, band_area, audit


def load_units(names):
    frames = []
    for name in names:
        cfg = UNIT_SETS[name]
        t = pd.read_csv(cfg["table"])
        if "decision" in t:
            t = t[t.decision.eq("ACCEPT")]
        missing = {cfg["id"], cfg["basin"], cfg["river"]} - set(t.columns)
        if missing:
            raise ValueError(f"{name}: columns missing in {cfg['table']}: {missing}")
        u = pd.DataFrame(dict(unit_key=t[cfg["id"]].astype(str).to_numpy(),
                              basin_shp=t[cfg["basin"]].to_numpy(),
                              river_shp=t[cfg["river"]].to_numpy()))
        u["set"], u["level"] = name, cfg["level"]
        bad = u.basin_shp.isna() | u.river_shp.isna()
        if bad.any():
            raise ValueError(f"{name}: missing basin/river path for {u.unit_key[bad].tolist()}")
        if u.unit_key.duplicated().any():
            raise ValueError(f"{name}: duplicate ids")
        frames.append(u)
    return pd.concat(frames, ignore_index=True)


def file_stamp(path):
    st = Path(path).stat()
    return [str(path), st.st_size, int(st.st_mtime)]


def shp_stamp(path):
    p = Path(path)
    files = sorted(f for f in p.parent.glob(p.stem + ".*")
                   if f.suffix.lower() in (".shp", ".shx", ".dbf", ".prj", ".cpg"))
    return [file_stamp(f) for f in files] or [file_stamp(p)]


def cache_meta(row, pop_paths):
    return dict(version=METHOD_VERSION, basin=shp_stamp(row.basin_shp),
                river=shp_stamp(row.river_shp), years=list(YEARS), ref=REFERENCE_YEAR,
                coverage=COVERAGE, bands=N_BANDS, cells_per_interval=CELLS_PER_INTERVAL,
                min_side=MIN_SAMPLES_SIDE, max_side=MAX_SAMPLES_SIDE, node_step=NODE_STEP,
                seed=RANDOM_SEED, population=[file_stamp(pop_paths[y]) for y in YEARS])


def safe_name(s):
    return "".join(ch if ch.isalnum() or ch in "-_." else "_" for ch in str(s))


def compute_profile(row, pop_paths, density=1.0, seed=RANDOM_SEED):
    basin = read_geometry(row.basin_shp, True)
    river = read_geometry(row.river_shp, False)
    return unit_profile(basin, river, pop_paths, YEARS, REFERENCE_YEAR, density, seed)


def get_profile(row, pop_paths, cache_dir):
    f = cache_dir / row.set / f"{safe_name(row.unit_key)}.npz"
    meta = cache_meta(row, pop_paths)
    if f.is_file():
        with np.load(f, allow_pickle=False) as z:
            if json.loads(str(z["meta"])) == meta:
                return (np.array(z["sums"]), np.array(z["full"]), np.array(z["band_area"]),
                        json.loads(str(z["audit"])), True)
    sums, full, band_area, audit = compute_profile(row, pop_paths)
    f.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(f, sums=sums, full=full, band_area=band_area,
                        meta=json.dumps(meta), audit=json.dumps(audit))
    return sums, full, band_area, audit, False


def write_profile_csv(path, unit, sums, band_area, r90_km):
    ny = len(YEARS)
    share = band_shares(sums)
    k = np.arange(1, N_BANDS + 1)
    pd.DataFrame(dict(
        unit_key=unit, year=np.repeat(YEARS, N_BANDS), gradient=np.tile(k, ny),
        outer_distance_km=np.tile(r90_km * k / N_BANDS, ny), SUM=sums.ravel(),
        percentage=share.ravel(), cumulative_percentage=np.cumsum(share, axis=1).ravel(),
        derivative=share.ravel(), band_area_km2=np.tile(band_area, ny),
    )).to_csv(path, index=False, encoding="utf-8-sig")


def threshold_table(profiles, units):

    rows = []
    for sample in ("annual", "static"):
        vals = {}
        for u in units.itertuples(index=False):
            sums = profiles[u.unit_key]["sums"]
            v = band_shares(sums).ravel() if sample == "annual" else band_shares(sums.mean(axis=0))
            vals.setdefault(u.level, []).append(v)
        vals = {k: np.concatenate(v) for k, v in vals.items()}
        allv = np.concatenate(list(vals.values()))
        rows.append(dict(sample=sample, policy="global", level="all", **iqr_threshold(allv)))
        w = np.concatenate([np.full(len(v), 1.0 / len(v)) for v in vals.values()])
        rows.append(dict(sample=sample, policy="balanced", level="all", **iqr_threshold(allv, w)))
        for level, v in vals.items():
            rows.append(dict(sample=sample, policy="by_level", level=level, **iqr_threshold(v)))
    return pd.DataFrame(rows)


def pick_threshold(table, policy, sample, level):
    t = table[(table["sample"] == sample) & (table.policy == policy)]
    t = t[t.level == (level if policy == "by_level" else "all")]
    return float(t["T"].iloc[0])


def classify_units(profiles, units, table, policy, sample):
    static, annual = [], []
    for u in units.itertuples(index=False):
        T = pick_threshold(table, policy, sample, u.level)
        sums, full = profiles[u.unit_key]["sums"], profiles[u.unit_key]["full"]
        static.append(dict(unit_key=u.unit_key, set=u.set, level=u.level,
                           **classify(sums.mean(axis=0), T, full.mean())))
        for j, year in enumerate(YEARS):
            annual.append(dict(unit_key=u.unit_key, set=u.set, level=u.level, year=year,
                               **classify(sums[j], T, full[j])))
    return pd.DataFrame(static), pd.DataFrame(annual)


def transitions(annual):
    wide = annual.pivot(index="unit_key", columns="year", values="HRP_type")
    known = wide.ne("REVIEW").all(axis=1)
    n = wide.ne(wide.shift(axis=1)).iloc[:, 1:].sum(axis=1).where(known)
    return wide, n


def annual_counts(annual):
    t = annual.groupby(["level", "year"]).HRP_type.value_counts().unstack(fill_value=0)
    out = t.reset_index()
    changes = []
    for level, g in annual.groupby("level"):
        w = g.pivot(index="unit_key", columns="year", values="HRP_type")
        ch = w.ne(w.shift(axis=1)).sum(axis=0).astype(float)
        ch.iloc[0] = np.nan
        changes.append(pd.DataFrame(dict(level=level, year=ch.index,
                                         changed_from_previous=ch.values)))
    return out.merge(pd.concat(changes), on=["level", "year"], how="left")


def legacy_comparison(static):
    rows = []
    for name, cfg in LEGACY_TYPES.items():
        p = Path(cfg["path"])
        old = (pd.read_excel(p, sheet_name=cfg.get("sheet", 0))
               if p.suffix.lower() in (".xlsx", ".xls") else pd.read_csv(p))
        old = old.set_index(old[cfg["id"]].astype(str))[cfg["type"]]
        s = static[static.set == name].copy()
        s["legacy_type"] = s.unit_key.map(old)
        rows.append(s[["unit_key", "set", "HRP_type", "legacy_type"]])
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


DEFINITIONS = [
    ("Concentration index S", "Population share (%) of each 1% distance interval; forward "
                              "difference of the cumulative profile from P(0)=0. Same for "
                              "static and annual profiles."),
    ("Threshold", "T = Q3 + 1.5*IQR, computed once from the current units and fixed across years."),
    ("High state", ">= T" if HIGH_IF_EQUAL else "> T"),
    ("Runs", f"Runs shorter than {MIN_RUN} intervals removed, neighbouring runs of the same "
             "state merged (original classify_series). Near-river: one agglomeration and the "
             "first retained run is high. Far-river: one agglomeration, not first. Dispersion: "
             "none or several (dispersion_kind)."),
    ("Static profile", "Mean of the annual interval populations over all years, normalised."),
    ("Distance", "Exact distance from each sample point to the nearest river polyline segment, "
                 "in an azimuthal equidistant projection centred on the unit."),
    ("Population allocation", "Each 30-arc-second pixel represented by m x m stratified random "
                              "points (deterministic hash seeds); pixel population assumed "
                              "uniform within the pixel. m gives at least "
                              f"{CELLS_PER_INTERVAL:g} sample spacings per interval width. A "
                              "numerical allocation, not finer population information."),
    ("Reff", "Population-weighted 90th percentile of the sample distances in the reference "
             "year, fixed for all years."),
    ("agglomeration_share_pct", "Population in the retained high runs used for the "
                                "classification (after merging) / population within Reff."),
    ("strict_run_share_pct", f"Population in uninterrupted high runs of >= {MIN_RUN} "
                             "intervals / population within Reff."),
    ("above_threshold_share_pct", f"Population in all intervals with S >= T / population "
                                  f"within Reff; NaN when fewer than {MIN_RUN} such intervals. "
                                  "This is the original 'concentrated zone' definition."),
    ("Full-basin population", "Valid population of the basin under the same allocation."),
    ("type_margin_pp", "Smallest |S - T| (percentage points) among intervals whose high/low "
                       "state, flipped alone, would change the HRP type. Small values mean "
                       "the type hinges on an interval close to the threshold. NaN when no "
                       f"such interval lies within {MARGIN_SCAN_PP:g} pp of T."),
]


def write_scenario(out, name, units, profiles, audits, extra):
    out.mkdir(parents=True, exist_ok=True)
    table = threshold_table(profiles, units)
    static, annual = classify_units(profiles, units, table, THRESHOLD_POLICY, THRESHOLD_SAMPLE)
    wide, n_trans = transitions(annual)
    static["annual_transition_count"] = static.unit_key.map(n_trans)
    static["annual_type_unchanged"] = static.annual_transition_count.map(
        lambda x: None if pd.isna(x) else bool(x == 0))
    mode = annual.groupby("unit_key").HRP_type.agg(lambda v: v.value_counts().index[0])
    static["annual_modal_type"] = static.unit_key.map(mode)
    static["precision_ok"] = static.unit_key.map(lambda k: audits[k].get("precision_ok"))
    static = static.merge(extra, on=["unit_key", "set"], how="left")

    choices = {}
    for policy in ("global", "balanced", "by_level"):
        for sample in ("annual", "static"):
            choices[f"{policy}|{sample}"] = [
                classify(profiles[u.unit_key]["sums"].mean(axis=0),
                         pick_threshold(table, policy, sample, u.level),
                         profiles[u.unit_key]["full"].mean())["HRP_type"]
                for u in units.itertuples(index=False)]
    ch = pd.DataFrame(choices, index=units.unit_key).reset_index()
    ch = ch.merge(units[["unit_key", "level"]], on="unit_key")
    ref_col = f"{THRESHOLD_POLICY}|{THRESHOLD_SAMPLE}"
    summary = []
    for col in choices:
        for level, g in ch.groupby("level"):
            counts = g[col].value_counts()
            summary.append(dict(choice=col, level=level, n=len(g),
                                agreement_with_used=float((g[col] == g[ref_col]).mean()),
                                **{t: int(counts.get(t, 0)) for t in
                                   ("Near-river", "Dispersion", "Far-river", "REVIEW")}))
    summary = pd.DataFrame(summary)
    counts = annual_counts(annual)
    legacy = legacy_comparison(static)
    aud = pd.DataFrame([dict(unit_key=k, **audits[k]) for k in units.unit_key])
    notes = pd.DataFrame(DEFINITIONS + [
        ("Threshold used", f"policy={THRESHOLD_POLICY}, sample={THRESHOLD_SAMPLE}"),
        ("Scenario", f"{name}: " + ", ".join(SCENARIOS[name]))], columns=["item", "definition"])
    frames = [("Static HRP", static), ("Annual HRP", annual), ("Annual types", wide.reset_index()),
              ("Thresholds", table), ("Threshold choices", summary), ("Annual counts", counts),
              ("Profile audit", aud), ("Definitions", notes)]
    if not legacy.empty:
        frames.append(("Legacy comparison", legacy))
    for sheet, frame in frames:
        frame.to_csv(out / f"{sheet.lower().replace(' ', '_')}.csv", index=False,
                     encoding="utf-8-sig")
    ch.to_csv(out / "threshold_choice_types.csv", index=False, encoding="utf-8-sig")
    with pd.ExcelWriter(out / f"HRP_results_{name}.xlsx", engine="openpyxl") as writer:
        for sheet, frame in frames:
            frame.to_excel(writer, sheet_name=sheet, index=False)
            ws = writer.sheets[sheet]
            ws.freeze_panes = "B2"
            ws.auto_filter.ref = ws.dimensions
    used = table[(table["sample"] == THRESHOLD_SAMPLE) & (table.policy == THRESHOLD_POLICY)]
    print(f"[{name}] thresholds used: " + ", ".join(f"{r.level}={r.T:.4f}" for r in used.itertuples()))
    for level, g in static.groupby("level"):
        a = annual[annual.level == level]
        print(f"[{name}] {level}: " + ", ".join(f"{k} {v}" for k, v in g.HRP_type.value_counts().items())
              + f"; precision ok {int(g.precision_ok.sum())}/{len(g)}; type within "
              f"{FRAGILE_MARGIN_PP:g} pp of T: static {int((g.type_margin_pp < FRAGILE_MARGIN_PP).sum())}, "
              f"annual {int((a.type_margin_pp < FRAGILE_MARGIN_PP).sum())}/{len(a)}")
    return static, table


def convergence(out, units, profiles, pop_paths, table, n):

    aud = pd.DataFrame([dict(unit_key=r.unit_key, **profiles[r.unit_key]["audit"])
                        for r in units.itertuples(index=False)])
    n = min(n, len(aud))
    hard = aud.sort_values("interval_to_pixel").unit_key.head(max(1, (2 * n) // 3)).tolist()
    rest = aud[~aud.unit_key.isin(hard)].unit_key
    pick = hard + rest.sample(min(len(rest), n - len(hard)), random_state=0).tolist()
    rows = []
    for key in pick:
        row = units[units.unit_key == key].iloc[0]
        base = profiles[key]
        T = pick_threshold(table, THRESHOLD_POLICY, THRESHOLD_SAMPLE, row.level)
        print(f"  convergence: {row.set}:{key}", flush=True)
        sums, full, _, audit = compute_profile(row, pop_paths, density=2.0, seed=RANDOM_SEED + 1)
        a = classify(base["sums"].mean(axis=0), T, base["full"].mean())
        b = classify(sums.mean(axis=0), T, full.mean())
        s0, s1 = band_shares(base["sums"].mean(axis=0)), band_shares(sums.mean(axis=0))
        ya = [classify(base["sums"][j], T, base["full"][j]) for j in range(len(YEARS))]
        yb = [classify(sums[j], T, full[j])["HRP_type"] for j in range(len(YEARS))]
        flips = [x["type_margin_pp"] for x, y in zip(ya, yb) if x["HRP_type"] != y]
        rows.append(dict(
            unit_key=key, set=row.set, level=row.level, threshold=T,
            interval_to_pixel=base["audit"]["interval_to_pixel"],
            samples_per_pixel_side=base["audit"]["samples_per_pixel_side"],
            samples_per_pixel_side_dense=audit["samples_per_pixel_side"],
            R90_km=base["audit"]["R90_km"], R90_km_dense=audit["R90_km"],
            R90_rel_diff=audit["R90_km"] / base["audit"]["R90_km"] - 1,
            S_max_abs_diff_pp=float(np.nanmax(np.abs(s1 - s0))),
            S_annual_max_abs_diff_pp=float(np.nanmax(np.abs(band_shares(sums)
                                                            - band_shares(base["sums"])))),
            S_band1=s0[0], S_band1_dense=s1[0],
            agglomeration_share=a["agglomeration_share_pct"],
            agglomeration_share_dense=b["agglomeration_share_pct"],
            strict_run_share=a["strict_run_share_pct"],
            strict_run_share_dense=b["strict_run_share_pct"],
            above_threshold_share=a["above_threshold_share_pct"],
            above_threshold_share_dense=b["above_threshold_share_pct"],
            type=a["HRP_type"], type_dense=b["HRP_type"], type_same=a["HRP_type"] == b["HRP_type"],
            type_margin_pp=a["type_margin_pp"],
            annual_types_same=len(YEARS) - len(flips),
            annual_flip_max_margin_pp=max(flips) if flips else np.nan,
            seconds_dense=audit["seconds"]))
    res = pd.DataFrame(rows)
    res.to_csv(out / "convergence.csv", index=False, encoding="utf-8-sig")
    print(f"Convergence ({len(res)} units, density x2, new seed): static type unchanged "
          f"{res.type_same.mean():.0%}, annual types unchanged "
          f"{res.annual_types_same.sum()}/{len(res) * len(YEARS)} (changed ones had margins "
          f"<= {res.annual_flip_max_margin_pp.max():.3f} pp), "
          f"max |dReff| {res.R90_rel_diff.abs().max():.3%}, "
          f"max |dS| {res.S_annual_max_abs_diff_pp.max():.3f} pp")


def main(scenarios=None, convergence_n=0):
    names = scenarios or list(SCENARIOS)
    sets = sorted({s for n in names for s in SCENARIOS[n]})
    units = load_units(sets)
    pop_paths = {y: POP_DIR / POP_PATTERN.format(year=y) for y in YEARS}
    missing = [str(p) for p in pop_paths.values() if not p.is_file()]
    missing += [str(p) for p in pd.concat([units.basin_shp, units.river_shp])
                if not Path(p).is_file()]
    if missing:
        raise FileNotFoundError("\n".join(sorted(set(missing))))
    out = OUTPUT_ROOT / ("run_" + datetime.now().strftime("%Y%m%d_%H%M%S"))
    out.mkdir(parents=True, exist_ok=False)
    cache = OUTPUT_ROOT / "profile_cache"
    shutil.copy2(__file__, out / Path(__file__).name)
    config = dict(network_run=str(NETWORK_RUN), population=str(POP_DIR / POP_PATTERN),
                  scenarios={n: SCENARIOS[n] for n in names},
                  unit_sets={k: {kk: str(vv) for kk, vv in v.items()}
                             for k, v in UNIT_SETS.items() if k in sets},
                  years=YEARS, reference_year=REFERENCE_YEAR, coverage=COVERAGE, bands=N_BANDS,
                  min_run=MIN_RUN, threshold_policy=THRESHOLD_POLICY,
                  threshold_sample=THRESHOLD_SAMPLE, high_if_equal=HIGH_IF_EQUAL,
                  cells_per_interval=CELLS_PER_INTERVAL, max_samples_side=MAX_SAMPLES_SIDE,
                  node_step=NODE_STEP, seed=RANDOM_SEED, method_version=METHOD_VERSION)
    (out / "config.json").write_text(json.dumps(config, ensure_ascii=False, indent=2),
                                     encoding="utf-8")
    print(f"Units: {len(units)} ("
          + ", ".join(f"{k} {v}" for k, v in units.set.value_counts().items()) + ")")
    print(f"Output: {out}", flush=True)
    profiles, audits, failures = {}, {}, []
    for n, row in enumerate(units.itertuples(index=False), 1):
        key = (row.set, row.unit_key)
        try:
            sums, full, band_area, audit, cached = get_profile(row, pop_paths, cache)
            profiles[key] = dict(sums=sums, full=full, band_area=band_area, audit=audit)
            audits[key] = audit
            pdir = out / "profiles" / row.set
            pdir.mkdir(parents=True, exist_ok=True)
            write_profile_csv(pdir / f"{safe_name(row.unit_key)}_profiles.csv", row.unit_key,
                              sums, band_area, audit["R90_km"])
            print(f"[{n}/{len(units)}] {row.set}:{row.unit_key}  Reff {audit['R90_km']:.1f} km  "
                  f"{audit['samples_per_pixel_side']}x{audit['samples_per_pixel_side']}/pixel  "
                  f"{'cache' if cached else str(audit['seconds']) + ' s'}"
                  f"{'' if audit['precision_ok'] else '  PRECISION BELOW TARGET'}", flush=True)
        except Exception as exc:
            failures.append(dict(set=row.set, unit_key=row.unit_key, error=str(exc),
                                 traceback=traceback.format_exc()))
            pd.DataFrame(failures).to_csv(out / "errors.csv", index=False, encoding="utf-8-sig")
            print(f"  FAILED {row.set}:{row.unit_key}: {exc}", flush=True)
    pd.DataFrame([dict(set=k[0], unit_key=k[1], **v) for k, v in audits.items()]).to_csv(
        out / "profile_audit.csv", index=False, encoding="utf-8-sig")
    if failures:
        raise RuntimeError(f"{len(failures)} units failed; classification not run. "
                           f"See {out / 'errors.csv'}. Successful profiles are cached.")
    low = [f"{k[0]}:{k[1]}" for k, v in audits.items() if not v["precision_ok"]]
    if low:
        print(f"WARNING {len(low)} units below the target sampling density (marked "
              f"precision_ok = False; treat as provisional): {', '.join(low[:10])}")
    results = {}
    for name in names:
        sel = units[units.set.isin(SCENARIOS[name])].reset_index(drop=True)
        if sel.unit_key.duplicated().any():
            raise ValueError(f"Scenario {name}: the same id appears in two sets")
        prof = {r.unit_key: profiles[(r.set, r.unit_key)] for r in sel.itertuples()}
        aud = {r.unit_key: audits[(r.set, r.unit_key)] for r in sel.itertuples()}
        extra = sel[["unit_key", "set", "basin_shp", "river_shp"]]
        results[name] = (sel, prof) + write_scenario(out / name, name, sel, prof, aud, extra)
    if {"main", "rule_networks"} <= set(results):


        _, _, st_m, tab_m = results["main"]
        sel_r, prof_r, st_r, _ = results["rule_networks"]
        a = st_m.query("level == 'major'").set_index("unit_key").HRP_type
        b = st_r.query("level == 'major'").set_index("unit_key").HRP_type.reindex(a.index)
        T_main = pick_threshold(tab_m, THRESHOLD_POLICY, THRESHOLD_SAMPLE, "major")
        c = pd.Series({r.unit_key: classify(prof_r[r.unit_key]["sums"].mean(axis=0), T_main,
                                            prof_r[r.unit_key]["full"].mean())["HRP_type"]
                       for r in sel_r[sel_r.level == "major"].itertuples()}).reindex(a.index)
        comp = pd.DataFrame(dict(type_original_network=a, type_rule_network=b,
                                 type_rule_network_main_threshold=c))
        comp["agree_own_thresholds"] = comp.type_original_network.eq(comp.type_rule_network)
        comp["agree_same_threshold"] = comp.type_original_network.eq(
            comp.type_rule_network_main_threshold)
        comp.rename_axis("major").reset_index().to_csv(
            out / "robustness_major_types.csv", index=False, encoding="utf-8-sig")
        print(f"Major types, original vs rule networks: {comp.agree_same_threshold.mean():.1%} agree "
              f"with the same threshold, {comp.agree_own_thresholds.mean():.1%} with each "
              f"scenario's own threshold")
    if convergence_n and "main" in results:
        sel_m, prof_m, _, tab_m = results["main"]
        convergence(out, sel_m, prof_m, pop_paths, tab_m, convergence_n)
    print(f"Finished: {out}")


def self_check():
    from rasterio.transform import from_origin
    T = 2.0
    pop = np.r_[[3.0] * 5, [0.5] * 95]
    r = classify(pop, T)
    assert r["HRP_type"] == "Near-river" and r["agglomeration_bands"] == "1-5"
    assert classify(np.r_[[0.5] * 5, [3.0] * 5, [0.5] * 90], T)["HRP_type"] == "Far-river"
    assert classify(np.r_[[3.0] * 4, [0.5] * 96], T)["HRP_type"] == "Dispersion"
    r = classify(np.r_[[3.0] * 5, [0.5] * 20, [3.0] * 5, [0.5] * 70], T)
    assert r["HRP_type"] == "Dispersion" and r["dispersion_kind"] == "multiple"
    r = classify(np.r_[[3.0] * 6, [0.5] * 2, [3.0] * 6, [0.5] * 86], T)
    assert r["HRP_type"] == "Near-river" and r["agglomeration_bands"] == "1-14"
    assert r["strict_run_share_pct"] < r["agglomeration_share_pct"]
    assert classify(np.tile([5.0, 0.1], 50), 1.0)["HRP_type"] == "REVIEW"
    assert abs(band_shares(pop)[0] - 3.0 / pop.sum() * 100) < 1e-12
    pop2 = np.r_[[2.05] * 5, [0.5] * 95]
    r = classify(pop2, 3.5)
    assert r["HRP_type"] == "Near-river" and abs(r["type_margin_pp"] - (band_shares(pop2)[0] - 3.5)) < 1e-12
    assert np.isnan(classify(pop, T)["type_margin_pp"])
    v = np.r_[np.zeros(300), np.ones(100)]
    assert abs(iqr_threshold(v, np.r_[np.full(300, 1 / 300), np.full(100, 1 / 100)])["Q3"] - 1) < 1e-9

    tmp = Path(tempfile.mkdtemp(prefix="hrp_check_"))
    res = 30 / 3600
    n = 60
    tr = from_origin(30.0, 0.5, res, res)
    yy = 0.5 - (np.arange(n) + 0.5) * res
    river_lat = 0.25 + 7.5 / 3600
    river = shapely.LineString([(30.05, river_lat), (30.45, river_lat)])
    basin = shapely.box(30.05, 0.20, 30.45, 0.30)
    dist_km = np.abs(yy - river_lat)[:, None] * 110.57 * np.ones((1, n))

    def make(name, arr):
        out = {}
        for k, year in enumerate((2019, 2020)):
            f = tmp / f"{name}_{year}.tif"
            with rasterio.open(f, "w", driver="GTiff", height=n, width=n, count=1,
                               dtype="int32", crs="EPSG:4326", transform=tr, nodata=-1) as dst:
                dst.write(((1 + 0.01 * k) * arr).astype("int32"), 1)
            out[year] = f
        return out


    flat = make("flat", np.full((n, n), 500.0))
    sums, full, band_area, audit = unit_profile(basin, river, flat, [2019, 2020], 2020)
    s = band_shares(sums[-1])
    rel = s / s.mean()
    expected = 0.9 * (0.30 - 0.20) * 110.57 / 2
    assert int((s <= 0).sum()) == 0, "empty intervals"
    assert abs(rel[0] - 1) < 0.02, f"first interval biased ({rel[0]:.4f})"
    assert np.abs(rel - 1).max() < 0.03, f"profile not flat (max {np.abs(rel - 1).max():.4f})"
    assert np.abs(band_area / band_area.mean() - 1).max() < 0.03
    assert abs(audit["reference_population_coverage"] - 0.90) < 1e-4, audit
    assert abs(audit["R90_km"] / expected - 1) < 0.005, audit
    assert audit["precision_ok"], audit

    sums2, _, _, audit2 = unit_profile(basin, river, flat, [2019, 2020], 2020, density=2.0, seed=7)
    d_r90 = audit2["R90_km"] / audit["R90_km"] - 1
    d_s = np.abs(band_shares(sums2[-1]) - s).max()
    assert abs(d_r90) < 0.002 and d_s < 0.03, (d_r90, d_s)

    decay = make("decay", 2000 * np.exp(-dist_km / 1.5))
    sums, full, _, _ = unit_profile(basin, river, decay, [2019, 2020], 2020)
    s3 = band_shares(sums[-1])
    assert s3[0] > s3[60] and classify(sums[-1], 1.3, full[-1])["HRP_type"] == "Near-river"

    nb_ = 3 * 120 + 8
    tr = from_origin(30.0 - 4 * res, 1.5 + 4 * res, res, res)
    big = {}
    for k, year in enumerate((2019, 2020)):
        f = tmp / f"big_{year}.tif"
        with rasterio.open(f, "w", driver="GTiff", height=nb_, width=nb_, count=1, dtype="int32",
                           crs="EPSG:4326", transform=tr, nodata=-1) as dst:
            dst.write(np.full((nb_, nb_), 500 + 5 * k, dtype="int32"), 1)
        big[year] = f
    river_big = shapely.LineString([(30.0, 7.5 / 3600), (33.0, 7.5 / 3600)])
    sums, _, _, audit4 = unit_profile(shapely.box(30.0, -1.5, 33.0, 1.5), river_big, big,
                                      [2019, 2020], 2020)
    rel4 = band_shares(sums[-1]) / band_shares(sums[-1]).mean()
    expected4 = 0.45 * 3.0 * 110.574
    assert audit4["whole_pixels"] > 0, audit4
    assert abs(audit4["R90_km"] / expected4 - 1) < 0.005, audit4
    assert np.abs(rel4 - 1).max() < 0.03, f"large flat profile (max {np.abs(rel4 - 1).max():.4f})"
    shutil.rmtree(tmp, ignore_errors=True)
    print(f"Self-check passed. Uniform case: Reff {audit['R90_km']:.3f} km (expected "
          f"{expected:.3f}), first interval {rel[0]:.4f} of the mean, max deviation "
          f"{np.abs(rel - 1).max():.4f}, {audit['samples_per_pixel_side']}x"
          f"{audit['samples_per_pixel_side']} samples/pixel. Doubled density and new seed: "
          f"dReff {d_r90:+.4%}, max |dS| {d_s:.4f} pp. Large uniform case: Reff "
          f"{audit4['R90_km']:.1f} km (expected {expected4:.1f}), max deviation "
          f"{np.abs(rel4 - 1).max():.4f}, {audit4['whole_pixels']} whole pixels.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--self-check", action="store_true")
    ap.add_argument("--scenario", action="append", choices=list(SCENARIOS))
    ap.add_argument("--convergence", type=int, default=0,
                    help="recompute N units of the main scenario at doubled sampling density")
    a = ap.parse_args()
    if a.self_check:
        self_check()
    else:
        main(a.scenario, a.convergence)
