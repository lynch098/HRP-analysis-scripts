
"""Regional assignment, HRP transitions and distance-ordered Gini summaries.
For retained fixed-zone population shares, use fig2b_zone_shares.py."""

import json as _path_json
from pathlib import Path as _ConfigPath


_DATA_PATHS = _path_json.loads(
    (_ConfigPath(__file__).resolve().parents[1] / "paths.json").read_text(encoding="utf-8")
)
from pathlib import Path
from datetime import datetime
import argparse
import json
import time
import warnings

import numpy as np
import pandas as pd
import geopandas as gpd
import shapely
from scipy import stats


OUTPUT_ROOT = Path(_DATA_PATHS["HRP_OUTPUT_ROOT"])
RUN_DIR = Path(_DATA_PATHS["HRP_RUN"])
SOUTH_SHP = Path(_DATA_PATHS["SOUTH_SHP"])
REGION_TABLE = None
AREA_CRS = "EPSG:6933"
SOUTH_THRESHOLD = 0.50
BORDERLINE = (0.45, 0.55)
SHARE_COL = "above_threshold_share_pct"
SHARE_CUTOFF = 30.0
PERSIST_YEARS = 3
P1, P2 = (1990, 2005), (2006, 2020)
ALPHA = 0.05
N_BANDS = 100
# Provisional endorheic roster: verify against basin attributes before interpreting this summary.
ENDORHEIC_MAJORS = {"ABHE_BAD", "CHUY", "CUVELAI", "OKAVANGO", "RHARSA", "TARIM_HE", "URAL",
                    "VOLGA", "KURA"}
BELT_CSV = RUN_DIR / "hydroclimatic_belts" / "basin_belts.csv"
TYPES = ["Near-river", "Dispersion", "Far-river"]
ABBR = {"Near-river": "N", "Dispersion": "D", "Far-river": "F", "REVIEW": "R"}


def find_run_dir(arg):
    if arg:
        return Path(arg)
    if RUN_DIR is not None:
        return Path(RUN_DIR)
    runs = sorted(p for p in OUTPUT_ROOT.glob("run_*") if (p / "main" / "static_hrp.csv").is_file())
    if not runs:
        raise FileNotFoundError(f"No run_* folder with main\\static_hrp.csv under {OUTPUT_ROOT}")
    return runs[-1]


def safe_name(s):
    return "".join(ch if ch.isalnum() or ch in "-_." else "_" for ch in str(s))


def retained_runs(seq, min_len):

    runs, i, n = [], 0, len(seq)
    while i < n:
        j = i
        while j < n and seq[j] == seq[i]:
            j += 1
        if j - i >= min_len:
            if runs and runs[-1][0] == seq[i]:
                runs[-1][2] = j
            else:
                runs.append([seq[i], i, j])
        i = j
    return runs


def theil_mk(x, y):

    x, y = np.asarray(x, float), np.asarray(y, float)
    ok = np.isfinite(x) & np.isfinite(y)
    if ok.sum() < 4 or np.ptp(y[ok]) == 0:
        return np.nan, np.nan
    slope = stats.theilslopes(y[ok], x[ok])[0]
    p = stats.kendalltau(x[ok], y[ok])[1]
    return float(slope), float(p)


def wilcoxon_p(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    ok = np.isfinite(a) & np.isfinite(b)
    if ok.sum() < 3 or np.allclose(a[ok], b[ok]):
        return np.nan
    try:
        return float(stats.wilcoxon(a[ok], b[ok]).pvalue)
    except ValueError:
        return np.nan


def fisher_p(a_yes, a_n, b_yes, b_n):
    return float(stats.fisher_exact([[a_yes, a_n - a_yes], [b_yes, b_n - b_yes]])[1])


def pct(x, n):
    return 100.0 * x / n if n else np.nan


def select(g, hrp_type="All", region="All"):
    m = np.ones(len(g), dtype=bool)
    if hrp_type != "All":
        m &= (g.HRP_type == hrp_type).to_numpy()
    if region != "All":
        m &= (g.region == region).to_numpy()
    return g[m]


def polygons(g):
    if g is None or g.is_empty:
        return []
    if g.geom_type == "Polygon":
        return [g]
    if hasattr(g, "geoms"):
        return [p for sub in g.geoms for p in polygons(sub)]
    return []


def polygon_union(gdf, crs):
    if gdf.crs is None:
        raise RuntimeError("Shapefile has no CRS")
    gdf = gdf[gdf.geometry.notna() & ~gdf.geometry.is_empty].to_crs(crs)
    parts = [p for g in gdf.geometry for p in polygons(g if g.is_valid else shapely.make_valid(g))]
    if not parts:
        raise RuntimeError("No polygon geometry")
    merged = shapely.union_all(parts)
    return merged if merged.is_valid else shapely.make_valid(merged)


def load_south():
    south = gpd.read_file(SOUTH_SHP)
    if south.crs is None:
        b = south.total_bounds
        if not (np.isfinite(b).all() and -180.5 <= b[0] <= b[2] <= 180.5 and -90.5 <= b[1] <= b[3] <= 90.5):
            raise RuntimeError(f"South.shp has no CRS and bounds {b} are not longitude/latitude")
        south = south.set_crs("EPSG:4326")
        print("South.shp has no .prj; coordinates look like lon/lat, EPSG:4326 assigned")
    return polygon_union(south, AREA_CRS)


def assign_regions(units):
    """Assign each basin independently using its South.shp area fraction in EPSG:6933 (>0.5 is South)."""
    south = load_south()
    shapely.prepare(south)
    rows = []
    for k, r in enumerate(units.itertuples(index=False), 1):
        g = polygon_union(gpd.read_file(r.basin_shp), AREA_CRS)
        area = float(g.area)
        inter = float(g.intersection(south).area) if south.intersects(g) else 0.0
        ratio = float(np.clip(inter / area, 0, 1))
        rows.append(dict(unit_key=r.unit_key, set=r.set, level=r.level, south_area_ratio=ratio,
                         region="South" if ratio > SOUTH_THRESHOLD else "North",
                         IsSouth=int(ratio > SOUTH_THRESHOLD),
                         crosses_boundary=bool(1e-6 < ratio < 1 - 1e-6),
                         borderline_45_55=bool(BORDERLINE[0] <= ratio <= BORDERLINE[1]),
                         representative_point_in_south=bool(south.covers(g.representative_point())),
                         area_km2=area / 1e6))
        if k % 25 == 0 or k == len(units):
            print(f"  regions {k}/{len(units)}", flush=True)
    return pd.DataFrame(rows)


def gini_series(sums):
    """Distance-ordered Gini for annual band populations: abs(2*m - 1 + q1/n)."""
    q = sums / sums.sum(axis=1, keepdims=True)
    n = q.shape[1]
    m = (q * ((np.arange(1, n + 1) - 0.5) / n)).sum(axis=1)
    return np.abs(2 * m - 1 + q[:, 0] / n)


def gini_trapezoid(sums):

    q = sums / sums.sum(axis=1, keepdims=True)
    n = q.shape[1]
    X = np.arange(1, n + 1) / n
    Y = np.cumsum(q, axis=1)
    return np.abs(1 - ((X[1:] - X[:-1]) * (Y[:, 1:] + Y[:, :-1])).sum(axis=1))


def load_gini(run_dir, units, years):
    out = {}
    for r in units.itertuples(index=False):
        f = run_dir / "profiles" / r.set / f"{safe_name(r.unit_key)}_profiles.csv"
        d = pd.read_csv(f, usecols=["year", "gradient", "SUM"])
        sums = d.pivot(index="year", columns="gradient", values="SUM").reindex(years).to_numpy(float)
        g = gini_series(sums)
        if not np.allclose(g, gini_trapezoid(sums)):
            raise ValueError(f"Gini formulas disagree for {r.unit_key}")
        out[r.unit_key] = g
    return out


def main(run_arg=None):
    t0 = time.time()
    run_dir = find_run_dir(run_arg)
    out = run_dir / ("downstream_" + datetime.now().strftime("%Y%m%d_%H%M%S"))
    out.mkdir(parents=True, exist_ok=False)
    print(f"Run folder: {run_dir}\nOutput: {out}", flush=True)
    st = pd.read_csv(run_dir / "main" / "static_hrp.csv")
    an = pd.read_csv(run_dir / "main" / "annual_hrp.csv")
    rule = run_dir / "rule_networks" / "static_hrp.csv"
    rule = pd.read_csv(rule) if rule.is_file() else None
    st["unit_key"] = st.unit_key.astype(str)
    an["unit_key"] = an.unit_key.astype(str)
    years = sorted(an.year.unique())
    print(f"Units: {len(st)} ({(st.level == 'major').sum()} majors); years {years[0]}-{years[-1]}")


    if REGION_TABLE:
        reg = pd.read_csv(REGION_TABLE)
        reg["unit_key"] = reg.unit_key.astype(str)
        print(f"Regions read from {REGION_TABLE}")
    else:
        print("Assigning North/South with South.shp:", flush=True)
        reg = assign_regions(st[["unit_key", "set", "level", "basin_shp"]])
    parent_reg = reg[reg.level == "major"].set_index("unit_key").region
    reg["parent_region"] = np.where(reg.level == "sub",
                                    reg.unit_key.str.split("__").str[0].map(parent_reg), reg.region)
    reg["differs_from_parent"] = reg.region != reg.parent_region
    reg.to_csv(out / "regions.csv", index=False, encoding="utf-8-sig")
    st = st.merge(reg[["unit_key", "region", "south_area_ratio"]], on="unit_key", how="left")
    if st.region.isna().any():
        raise ValueError(f"No region for: {st.unit_key[st.region.isna()].tolist()[:10]}")
    an = an.merge(reg[["unit_key", "region"]], on="unit_key", how="left")
    maj = st.level == "major"
    print(f"Majors: North {(st.region[maj] == 'North').sum()}, South {(st.region[maj] == 'South').sum()} "
          f"(SI: 28 / 52); sub-basins: North {(st.region[~maj] == 'North').sum()}, "
          f"South {(st.region[~maj] == 'South').sum()}")
    cross = reg[(reg.level == "major") & reg.crosses_boundary].sort_values("south_area_ratio")
    print("Major basins crossing the boundary (compare with SI Fig. S14): "
          + ", ".join(f"{r.unit_key} {100 * r.south_area_ratio:.0f}% South -> {r.region}" for r in cross.itertuples()))
    differ = reg[reg.differs_from_parent]
    print(f"Sub-basins whose region differs from their major basin: {len(differ)}"
          + (": " + ", ".join(f"{r.unit_key} ({r.region})" for r in differ.head(20).itertuples()) if len(differ) else ""))


    comp = []
    for level, g in list(st.groupby("level")) + [("major+sub", st)]:
        for region in ("All", "North", "South"):
            gg = g if region == "All" else g[g.region == region]
            c = gg.HRP_type.value_counts()
            comp.append(dict(level=level, region=region, n=len(gg),
                             **{t: int(c.get(t, 0)) for t in TYPES},
                             **{f"pct_{t}": pct(c.get(t, 0), len(gg)) for t in TYPES}))
    comp = pd.DataFrame(comp)
    mj = st[maj]
    nN, nS = (mj.region == "North").sum(), (mj.region == "South").sum()
    tests = []
    for t in TYPES:
        a, b = ((mj.region == "North") & (mj.HRP_type == t)).sum(), ((mj.region == "South") & (mj.HRP_type == t)).sum()
        tests.append(dict(test=f"{t} share, North vs South (majors, Fisher)", North=f"{a}/{nN}", South=f"{b}/{nS}",
                          p=fisher_p(a, nN, b, nS)))


    ann_comp = []
    for (level, year), g in an.groupby(["level", "year"]):
        for region in ("All", "North", "South"):
            gg = g if region == "All" else g[g.region == region]
            c = gg.HRP_type.value_counts()
            ann_comp.append(dict(level=level, region=region, year=year, n=len(gg),
                                 **{f"pct_{t}": pct(c.get(t, 0), len(gg)) for t in TYPES}))
    ann_comp = pd.DataFrame(ann_comp)
    ann_mean = ann_comp.groupby(["level", "region"])[[f"pct_{t}" for t in TYPES]].agg(["mean", "std"])
    ann_mean.columns = [f"{a}_{b}" for a, b in ann_mean.columns]
    ann_mean = ann_mean.reset_index()


    seqs = an.sort_values("year").groupby("unit_key").HRP_type.apply(list)
    trans = []
    for key, seq in seqs.items():
        raw_n = sum(1 for a, b in zip(seq[:-1], seq[1:]) if a != b)
        runs = retained_runs(seq, PERSIST_YEARS)
        path = " > ".join(f"{ABBR.get(v, v)}({years[a]}-{years[b - 1]})" for v, a, b in runs)
        trans.append(dict(unit_key=key, raw_transitions=raw_n, raw_changed=raw_n > 0,
                          persistent_status="unstable" if not runs else ("changed" if len(runs) > 1 else "stable"),
                          persistent_transitions=max(len(runs) - 1, 0),
                          persistent_changed=len(runs) > 1,
                          first_persistent_change=years[runs[1][1]] if len(runs) > 1 else np.nan,
                          persistent_path=path, annual_sequence="".join(ABBR.get(v, "?") for v in seq)))
    trans = pd.DataFrame(trans)
    st = st.merge(trans, on="unit_key", how="left")
    tsum = []
    for level, g in st.groupby("level"):
        for region in ("All", "North", "South"):
            gg = g if region == "All" else g[g.region == region]
            n = len(gg)
            tsum.append(dict(level=level, region=region, n=n,
                             raw_changed=int(gg.raw_changed.sum()), raw_changed_pct=pct(gg.raw_changed.sum(), n),
                             persistent_changed=int(gg.persistent_changed.sum()),
                             persistent_changed_pct=pct(gg.persistent_changed.sum(), n),
                             unstable=int((gg.persistent_status == "unstable").sum()),
                             unchanged_raw_pct=pct((~gg.raw_changed).sum(), n),
                             unchanged_persistent_pct=pct((~gg.persistent_changed).sum(), n)))
    tsum = pd.DataFrame(tsum)
    mj = st[maj]
    for col in ("raw_changed", "persistent_changed"):
        a, b = mj[(mj.region == "North")][col].sum(), mj[(mj.region == "South")][col].sum()
        tests.append(dict(test=f"{col}, North vs South (majors, Fisher)", North=f"{a}/{nN}", South=f"{b}/{nS}",
                          p=fisher_p(a, nN, b, nS)))
    piv = {level: g.pivot(index="unit_key", columns="year", values="HRP_type") for level, g in an.groupby("level")}
    year_rows = []
    for prev, y in zip(years[:-1], years[1:]):
        row = dict(year=y)
        for level, w in piv.items():
            row[f"{level}_raw_changes"] = int((w[y] != w[prev]).sum())
            row[f"{level}_persistent_changes"] = int((st[st.level == level].first_persistent_change == y).sum())
        year_rows.append(row)
    trans_years = pd.DataFrame(year_rows)


    st["share"] = st[SHARE_COL].fillna(0.0)
    sh = []
    for level, g in st.groupby("level"):
        for t in TYPES:
            for region in ("All", "North", "South"):
                gg = select(g, t, region)
                sh.append(dict(level=level, type=t, region=region, n=len(gg),
                               n_at_least_cutoff=int((gg.share >= SHARE_CUTOFF).sum()),
                               pct_at_least_cutoff=pct((gg.share >= SHARE_CUTOFF).sum(), len(gg)),
                               median_share=float(gg.share.median()) if len(gg) else np.nan,
                               mean_share=float(gg.share.mean()) if len(gg) else np.nan))
    shares = pd.DataFrame(sh)
    an["share"] = an[SHARE_COL].fillna(0.0)
    an["period"] = np.where(an.year.between(*P1), "P1", np.where(an.year.between(*P2), "P2", ""))
    an["conc_pop"] = an.share / 100 * an.profile_population
    anp = an[an.period != ""]
    wide = anp.groupby(["unit_key", "period"]).share.mean().unstack()
    pooled = anp.groupby(["unit_key", "period"])[["conc_pop", "profile_population"]].sum().reset_index()
    st = st.merge(wide.rename(columns={"P1": "share_P1", "P2": "share_P2"}), left_on="unit_key",
                  right_index=True, how="left")
    sp = []
    for level, g in st.groupby("level"):
        for t in ["All"] + TYPES:
            for region in ("All", "North", "South"):
                gg = select(g, t, region)
                pp = pooled[pooled.unit_key.isin(gg.unit_key)].groupby("period")[["conc_pop", "profile_population"]].sum()
                sp.append(dict(level=level, type=t, region=region, n=len(gg),
                               basin_mean_P1=float(gg.share_P1.mean()), basin_mean_P2=float(gg.share_P2.mean()),
                               wilcoxon_p=wilcoxon_p(gg.share_P1, gg.share_P2),
                               pooled_P1=100 * pp.loc["P1", "conc_pop"] / pp.loc["P1", "profile_population"] if "P1" in pp.index else np.nan,
                               pooled_P2=100 * pp.loc["P2", "conc_pop"] / pp.loc["P2", "profile_population"] if "P2" in pp.index else np.nan))
    share_periods = pd.DataFrame(sp)


    print("Gini coefficients from the profiles:", flush=True)
    G = load_gini(run_dir, st[["unit_key", "set"]], years)
    yr = np.array(years, float)
    grow = []
    for key, g in G.items():
        s, p = theil_mk(yr, g)
        s1, _ = theil_mk(yr[(yr >= P1[0]) & (yr <= P1[1])], g[(yr >= P1[0]) & (yr <= P1[1])])
        s2, _ = theil_mk(yr[(yr >= P2[0]) & (yr <= P2[1])], g[(yr >= P2[0]) & (yr <= P2[1])])
        d1 = np.diff(g[(yr >= P1[0]) & (yr <= P1[1])]).mean()
        d2 = np.diff(g[(yr >= P2[0]) & (yr <= P2[1])]).mean()
        trend = "n.s." if not (p < ALPHA) else ("increase" if s > 0 else "decrease")
        grow.append(dict(unit_key=key, gini_first=g[0], gini_last=g[-1], gini_mean=g.mean(),
                         gini_slope_per_yr=s, gini_mk_p=p, gini_trend=trend,
                         slope_P1=s1, slope_P2=s2, mean_annual_change_P1=d1, mean_annual_change_P2=d2))
    gtab = pd.DataFrame(grow)
    st = st.merge(gtab, on="unit_key", how="left")
    gsum = []
    for level, g in st.groupby("level"):
        for t in TYPES:
            for region in ("All", "North", "South"):
                gg = select(g, t, region)
                n = len(gg)
                gsum.append(dict(level=level, type=t, region=region, n=n,
                                 sig_increase_pct=pct((gg.gini_trend == "increase").sum(), n),
                                 sig_decrease_pct=pct((gg.gini_trend == "decrease").sum(), n),
                                 mean_gini_first_year=float(gg.gini_first.mean()) if n else np.nan))
    gsum = pd.DataFrame(gsum)
    gser = []
    mj = st[maj]
    for t in ["All"] + TYPES:
        for region in ("All", "North", "South"):
            gg = select(mj, t, region)
            if gg.empty:
                continue
            series = np.mean([G[k] for k in gg.unit_key], axis=0)
            s, p = theil_mk(yr, series)
            gser.append(dict(type=t, region=region, n=len(gg), slope_per_yr=s, slope_x100_per_yr=100 * s,
                             mk_p=p, mean_first=series[0], mean_last=series[-1]))
    gser = pd.DataFrame(gser)
    gper = []
    for region in ("All", "North", "South"):
        gg = mj if region == "All" else mj[mj.region == region]
        for col in ("slope", "mean_annual_change"):
            a, b = gg[f"{col}_P1"], gg[f"{col}_P2"]
            gper.append(dict(region=region, measure=col, n=len(gg), mean_P1=float(a.mean()), mean_P2=float(b.mean()),
                             median_P1=float(a.median()), median_P2=float(b.median()),
                             P2_greater_pct=pct((b > a).sum(), len(gg)), wilcoxon_p=wilcoxon_p(a, b)))
    gper = pd.DataFrame(gper)


    rule_comp = pd.DataFrame()
    if rule is not None:
        rm = rule[rule.level == "major"].copy()
        rm["unit_key"] = rm.unit_key.astype(str)
        rm = rm.merge(reg[reg.level == "major"][["unit_key", "region"]], on="unit_key", how="left")
        rows = []
        for region in ("All", "North", "South"):
            gg = rm if region == "All" else rm[rm.region == region]
            c = gg.HRP_type.value_counts()
            rows.append(dict(region=region, n=len(gg), **{t: int(c.get(t, 0)) for t in TYPES}))
        rule_comp = pd.DataFrame(rows)

    # Provisional endorheic roster: verify against basin attributes before interpreting this summary.
    mj = st[maj].copy()
    mj["endorheic"] = mj.unit_key.isin(ENDORHEIC_MAJORS)
    endo = pd.DataFrame([dict(group="endorheic" if e else "exorheic", n=len(g),
                              far=int((g.HRP_type == "Far-river").sum()),
                              far_pct=pct((g.HRP_type == "Far-river").sum(), len(g)))
                         for e, g in mj.groupby("endorheic")])
    belts = pd.DataFrame()
    if BELT_CSV:
        b = pd.read_csv(BELT_CSV)
        key = "unit_key" if "unit_key" in b else "major"
        b[key] = b[key].astype(str)
        bb = st.merge(b[[key, "belt"]], left_on="unit_key", right_on=key, how="left")
        belts = bb.groupby(["level", "belt"]).HRP_type.value_counts().unstack(fill_value=0).reset_index()


    def c(level, region, t):
        r = comp[(comp.level == level) & (comp.region == region)].iloc[0]
        return int(r[t]), r[f"pct_{t}"], int(r.n)

    am = ann_mean[(ann_mean.level == "major") & (ann_mean.region == "All")].iloc[0]
    tm = tsum[(tsum.level == "major")].set_index("region")
    shm = shares[(shares.level == "major") & (shares.region == "All")].set_index("type")
    em = endo.set_index("group")
    gs = gsum[(gsum.level == "major") & (gsum.region == "All")].set_index("type")
    gsr = gser.set_index(["type", "region"])
    spr = share_periods[(share_periods.level == "major") & (share_periods.type == "All")].set_index("region")
    gp = gper[gper.measure == "slope"].set_index("region")

    def fmt_trend(t, region):
        if (t, region) not in gsr.index:
            return "n/a"
        r = gsr.loc[(t, region)]
        return f"{r.slope_x100_per_yr:+.3f} (x100 per yr), MK p={r.mk_p:.3g}"

    numbers = [
        ("Abstract", "Near-river share of major basins, annual mean +- SD", "42.7 +- 2.3%",
         f"{am['pct_Near-river_mean']:.1f} +- {am['pct_Near-river_std']:.1f}%"),
        ("Abstract", "Dispersion share of major basins, annual mean +- SD", "46.3 +- 2.3%",
         f"{am['pct_Dispersion_mean']:.1f} +- {am['pct_Dispersion_std']:.1f}%"),
        ("Results", "Major basins, static Near / Dispersion / Far", "37 / 34 / 9",
         " / ".join(str(c("major", "All", t)[0]) for t in TYPES)),
        ("Results", "Global North majors, Near / Dispersion / Far", "20 / 6 / 2",
         " / ".join(str(c("major", "North", t)[0]) for t in TYPES) + f" (n = {c('major', 'North', TYPES[0])[2]})"),
        ("Results", "Global South majors, Near / Dispersion / Far", "17 / 28 / 7",
         " / ".join(str(c("major", "South", t)[0]) for t in TYPES) + f" (n = {c('major', 'South', TYPES[0])[2]})"),
        ("Results", "Near-river share, North vs South majors", "",
         f"{c('major', 'North', 'Near-river')[1]:.1f}% vs {c('major', 'South', 'Near-river')[1]:.1f}%, "
         f"Fisher p={tests[0]['p']:.3g}"),
        ("Results", "Dispersion share, North vs South majors", "",
         f"{c('major', 'North', 'Dispersion')[1]:.1f}% vs {c('major', 'South', 'Dispersion')[1]:.1f}%, "
         f"Fisher p={tests[1]['p']:.3g}"),
        ("Results", "Near-river / Dispersion / Far-river, majors + sub-basins", "82 / 85 / 22 (80 + 109)",
         " / ".join(str(c("major+sub", "All", t)[0]) for t in TYPES) + f" ({len(st)} units)"),
        ("Results", "Near-river majors with >= 30% in concentrated zones", "> 62%",
         f"{shm.loc['Near-river', 'n_at_least_cutoff']}/{shm.loc['Near-river', 'n']} "
         f"({shm.loc['Near-river', 'pct_at_least_cutoff']:.1f}%)"),
        ("Results", "Far-river majors with >= 30% in concentrated zones", "two-thirds",
         f"{shm.loc['Far-river', 'n_at_least_cutoff']}/{shm.loc['Far-river', 'n']} "
         f"({shm.loc['Far-river', 'pct_at_least_cutoff']:.1f}%)"),
        ("Results", "Far-river share, endorheic vs exorheic majors (check ENDORHEIC_MAJORS)", "33.3% vs 8.5%",
         f"{em.loc['endorheic', 'far_pct']:.1f}% ({em.loc['endorheic', 'far']}/{em.loc['endorheic', 'n']}) vs "
         f"{em.loc['exorheic', 'far_pct']:.1f}% ({em.loc['exorheic', 'far']}/{em.loc['exorheic', 'n']})"),
        ("Results", f"Majors keeping one type 1990-2020 (persistent, >= {PERSIST_YEARS} yr)", "75.0%",
         f"{tm.loc['All', 'unchanged_persistent_pct']:.1f}% (raw: {tm.loc['All', 'unchanged_raw_pct']:.1f}%)"),
        ("Results", "Majors changing type at least once, South vs North (persistent)", "28.8% vs 17.9%",
         f"{tm.loc['South', 'persistent_changed_pct']:.1f}% vs {tm.loc['North', 'persistent_changed_pct']:.1f}% "
         f"(raw: {tm.loc['South', 'raw_changed_pct']:.1f}% vs {tm.loc['North', 'raw_changed_pct']:.1f}%)"),
        ("Results", "Gini rose significantly in Near-river majors", "51.35%",
         f"{gs.loc['Near-river', 'sig_increase_pct']:.1f}%"),
        ("Results", "Initial mean Gini of Near-river majors", "0.425",
         f"{gs.loc['Near-river', 'mean_gini_first_year']:.3f}"),
        ("Results", "Gini rose significantly in Far-river majors", "55.56%",
         f"{gs.loc['Far-river', 'sig_increase_pct']:.1f}%"),
        ("Results", "Dispersion majors, significant increase / decrease in Gini", "41.18% / 38.24%",
         f"{gs.loc['Dispersion', 'sig_increase_pct']:.1f}% / {gs.loc['Dispersion', 'sig_decrease_pct']:.1f}%"),
        ("Results", "South Near-river majors, trend of mean Gini", "+0.03% per yr (p < 0.05)",
         fmt_trend("Near-river", "South")),
        ("Results", "South Dispersion majors, trend of mean Gini", "-0.03% per yr (p < 0.001)",
         fmt_trend("Dispersion", "South")),
        ("Results", "Concentrated-zone share, South majors, P1 -> P2 (pooled; check Fig. 2b definition)",
         "41.06% -> 45.36%",
         f"{spr.loc['South', 'pooled_P1']:.2f}% -> {spr.loc['South', 'pooled_P2']:.2f}% "
         f"(basin mean {spr.loc['South', 'basin_mean_P1']:.2f}% -> {spr.loc['South', 'basin_mean_P2']:.2f}%)"),
        ("Results", "Concentrated-zone share, North majors, P1 -> P2 (pooled; check Fig. 2b definition)",
         "36.58% -> 36.63%",
         f"{spr.loc['North', 'pooled_P1']:.2f}% -> {spr.loc['North', 'pooled_P2']:.2f}% "
         f"(basin mean {spr.loc['North', 'basin_mean_P1']:.2f}% -> {spr.loc['North', 'basin_mean_P2']:.2f}%)"),
        ("Results", "Mean annual Gini change larger in P2 than P1 (Theil-Sen per basin)", "both regions",
         "; ".join(f"{rg}: {gp.loc[rg, 'mean_P1']:+.5f} -> {gp.loc[rg, 'mean_P2']:+.5f}, Wilcoxon p={gp.loc[rg, 'wilcoxon_p']:.3g}"
                   for rg in ("North", "South"))),
    ]
    numbers = pd.DataFrame(numbers, columns=["section", "statement", "manuscript_value", "new_value"])

    not_covered = pd.DataFrame([
        ("Fig. 1c Pareto fits", "Fitting procedure not specified in Methods; needs the original script."),
        ("Fig. 1d settlement distance vs basin area", "Definition of agglomeration distance needs the original script."),
        ("Fig. 2b", "Concentrated-zone share computed here two ways (pooled and basin mean); check against the original definition."),
        ("Fig. 1a hydroclimatic belts", "Provide BELT_CSV to compute belt composition."),
        ("Flood exposure (Fig. 3, SI Figs. S9-S12)", "Exposure indices unchanged; regroup by the new HRP types and regions."),
        ("Riverward distance trends (Fig. 1e, R1D)", "Separate pixel analysis; only the type labels and regions change."),
    ], columns=["item", "note"])

    unit_cols = ["unit_key", "set", "level", "region", "south_area_ratio", "HRP_type", "type_margin_pp",
                 "share", "share_P1", "share_P2", "raw_transitions", "persistent_status", "persistent_transitions",
                 "first_persistent_change", "persistent_path", "annual_sequence", "gini_first", "gini_last",
                 "gini_slope_per_yr", "gini_mk_p", "gini_trend", "slope_P1", "slope_P2"]
    units = st[[c for c in unit_cols if c in st]].rename(columns={"share": SHARE_COL})
    sheets = [("Manuscript numbers", numbers), ("Tests", pd.DataFrame(tests)), ("Regions", reg),
              ("Units", units), ("Composition", comp), ("Annual composition", ann_mean),
              ("Annual composition by year", ann_comp), ("Transitions", tsum),
              ("Transitions by year", trans_years), ("Shares", shares), ("Shares by period", share_periods),
              ("Gini summary", gsum), ("Gini type-region trends", gser), ("Gini period change", gper),
              ("Rule networks", rule_comp), ("Endorheic", endo), ("Belts", belts), ("Not covered", not_covered)]
    for name, frame in sheets:
        frame.to_csv(out / f"{name.lower().replace(' ', '_').replace('-', '_')}.csv", index=False,
                     encoding="utf-8-sig")
    with pd.ExcelWriter(out / "HRP_downstream_v3.xlsx", engine="openpyxl") as writer:
        for name, frame in sheets:
            frame.to_excel(writer, sheet_name=name[:31], index=False)
            ws = writer.sheets[name[:31]]
            ws.freeze_panes = "B2"
            if len(frame):
                ws.auto_filter.ref = ws.dimensions
    (out / "config.json").write_text(json.dumps(dict(
        run_dir=str(run_dir), south_shp=str(SOUTH_SHP), area_crs=AREA_CRS, south_threshold=SOUTH_THRESHOLD,
        share_col=SHARE_COL, share_cutoff=SHARE_CUTOFF, persist_years=PERSIST_YEARS, P1=P1, P2=P2,
        alpha=ALPHA, endorheic=sorted(ENDORHEIC_MAJORS), belt_csv=str(BELT_CSV)), indent=2), encoding="utf-8")

    print("\nManuscript numbers:")
    for r in numbers.itertuples():
        print(f"  {r.statement}\n      manuscript: {r.manuscript_value or '-'} | new: {r.new_value}")
    print(f"\nFinished in {time.time() - t0:.0f} s: {out}")


if __name__ == "__main__":
    warnings.filterwarnings("ignore", category=RuntimeWarning, message="Mean of empty slice")
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", default=None, help="hrp_rerun_v3 run folder (run_<time>)")
    main(ap.parse_args().run_dir)
