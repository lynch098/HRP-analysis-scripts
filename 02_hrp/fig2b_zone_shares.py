
"""Compare population shares in retained concentration zones.
Hold P1 zone boundaries fixed for P2 to separate within-zone and boundary changes."""

import json as _path_json
from pathlib import Path as _ConfigPath


_DATA_PATHS = _path_json.loads(
    (_ConfigPath(__file__).resolve().parents[1] / "paths.json").read_text(encoding="utf-8")
)
from datetime import datetime
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from scipy import stats


RUN_DIR = Path(_DATA_PATHS["HRP_RUN"])
STATIC_CSV = RUN_DIR / "main/static_hrp.csv"
REGION_CSV = RUN_DIR / "downstream_20261019_165337/regions.csv"
ANNUAL_CSV = RUN_DIR / "main/annual_hrp.csv"
OUT_DIR = Path(__file__).resolve().parent / ("results_" + datetime.now().strftime("%Y%m%d_%H%M%S"))


PERIODS = {"P1": (1990, 2005), "P2": (2006, 2020)}
LEVEL = "major"
N_BANDS = 100
MIN_RUN = 5
HIGH_IF_EQUAL = True
N_CLASSES = 6
REGIONS = ("North", "South")
TYPE_LABEL = {"Near-river": "River-adjacent", "Dispersion": "Diffuse", "Far-river": "River-detached",
              "REVIEW": "REVIEW"}
TYPE_ROWS = ["All", "River-adjacent", "Diffuse", "River-detached"]


def _p(x):
    s = str(x).strip() if x is not None else ""
    return Path(s) if s not in ("", ".") else None


def safe_name(s):

    return "".join(ch if ch.isalnum() or ch in "-_." else "_" for ch in str(s))


def band_shares(pop):
    pop = np.asarray(pop, dtype=float)
    total = pop.sum()
    return pop / total * 100.0 if total > 0 else np.full(len(pop), np.nan)


def is_high(s, threshold):
    s = np.asarray(s, dtype=float)
    return np.isfinite(s) & ((s >= threshold) if HIGH_IF_EQUAL else (s > threshold))


def retained_runs(high, min_len=MIN_RUN):

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


def zone(pop, threshold):

    runs = retained_runs(is_high(band_shares(pop), threshold))
    mask = np.zeros(len(pop), dtype=bool)
    for state, a, b in runs:
        if state:
            mask[a:b] = True
    n_high = sum(1 for r in runs if r[0])
    if not runs:
        hrp = "REVIEW"
    elif n_high == 1:
        hrp = "Near-river" if runs[0][0] else "Far-river"
    else:
        hrp = "Dispersion"
    return mask, TYPE_LABEL[hrp]


def band_ranges(mask):

    idx = np.flatnonzero(mask)
    if not len(idx):
        return ""
    out, start = [], idx[0]
    for a, b in zip(idx[:-1], idx[1:]):
        if b != a + 1:
            out.append((start, a))
            start = b
    out.append((start, idx[-1]))
    return "; ".join(f"{a + 1}-{b + 1}" if a != b else f"{a + 1}" for a, b in out)


def wilcoxon_p(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    ok = np.isfinite(a) & np.isfinite(b)
    if ((a[ok] - b[ok]) != 0).sum() < 5:
        return np.nan
    return float(stats.wilcoxon(a[ok], b[ok]).pvalue)


def subset(df, region, t):
    q = df[df.region.eq(region)]
    return q if t == "All" else q[q.static_type.eq(t)]


def load_inputs():
    run, sp, rp = _p(RUN_DIR), _p(STATIC_CSV), _p(REGION_CSV)
    for name, p in [("RUN_DIR", run), ("STATIC_CSV", sp), ("REGION_CSV", rp)]:
        if p is None or not p.exists():
            sys.exit(f"Set {name} to a valid input (current: {p}).")
    static = pd.read_csv(sp, encoding="utf-8-sig", dtype={"unit_key": str})
    static = static[static.level.eq(LEVEL)].copy()
    T = static.threshold.dropna().unique()
    if len(T) != 1:
        sys.exit(f"Non-unique major-basin threshold in static_hrp.csv: {T}")
    T = float(T[0])
    reg = pd.read_csv(rp, encoding="utf-8-sig", dtype={"unit_key": str})
    static = static.drop(columns=["region"], errors="ignore")
    static = static.merge(reg[["unit_key", "region"]], on="unit_key", how="left", validate="one_to_one")
    if static.region.isna().any():
        sys.exit("Missing units in regions.csv: " + ", ".join(static.unit_key[static.region.isna()].head(10)))
    static["static_type"] = static.HRP_type.map(TYPE_LABEL).fillna(static.HRP_type)
    need = [y for a, b in PERIODS.values() for y in range(a, b + 1)]
    prof = {}
    for r in static.itertuples(index=False):
        f = run / "profiles" / r.set / f"{safe_name(r.unit_key)}_profiles.csv"
        if not f.is_file():
            sys.exit(f"Profile file not found: {f}")
        w = pd.read_csv(f, encoding="utf-8-sig").pivot(index="year", columns="gradient", values="SUM").sort_index()
        if w.shape[1] != N_BANDS:
            sys.exit(f"{f.name}: found {w.shape[1]} distance bands; expected {N_BANDS}")
        miss = sorted(set(need) - set(w.index))
        if miss:
            sys.exit(f"{f.name}: missing years {miss[:5]}")
        prof[r.unit_key] = w
    return run, static, T, prof


def period_sum(w, period):
    a, b = PERIODS[period]
    return w.loc[a:b].to_numpy(float).sum(axis=0)


def regional_profiles(static, prof):
    return {(region, per): sum(period_sum(prof[k], per) for k in static.unit_key[static.region.eq(region)])
            for region in REGIONS for per in PERIODS}


def sextile_reproduction(R):
    S = {k: band_shares(v) for k, v in R.items()}
    allv = np.concatenate(list(S.values()))
    cls = pd.qcut(allv, N_CLASSES, labels=False)
    cut = float(allv[cls == N_CLASSES - 1].min())
    rows = []
    for region in REGIONS:
        m1 = S[(region, "P1")] >= cut
        for per in PERIODS:
            m = S[(region, per)] >= cut
            rows.append(dict(region=region, period=per, cutoff_pct_per_band=cut, n_bands=int(m.sum()),
                             bands=band_ranges(m), share_pct=float(S[(region, per)][m].sum()),
                             share_in_P1_bands_pct=float(S[(region, per)][m1].sum())))
    return pd.DataFrame(rows)


def basin_periods(static, prof, T):
    rows = []
    for r in static.itertuples(index=False):
        p1, p2 = period_sum(prof[r.unit_key], "P1"), period_sum(prof[r.unit_key], "P2")
        m1, t1 = zone(p1, T)
        m2, t2 = zone(p2, T)
        rows.append(dict(unit_key=r.unit_key, region=r.region, static_type=r.static_type, type_P1=t1, type_P2=t2,
                         zone_P1=band_ranges(m1), zone_P2=band_ranges(m2),
                         width_P1=int(m1.sum()), width_P2=int(m2.sum()),
                         pop_P1=float(p1.sum()), pop_P2=float(p2.sum()),
                         zone_pop_P1=float(p1[m1].sum()), zone_pop_P2_fixed=float(p2[m1].sum()),
                         zone_pop_P2_own=float(p2[m2].sum()),
                         share_P1_pct=100 * float(p1[m1].sum()) / p1.sum(),
                         share_P2_fixed_pct=100 * float(p2[m1].sum()) / p2.sum(),
                         share_P2_own_pct=100 * float(p2[m2].sum()) / p2.sum()))
    return pd.DataFrame(rows)


def summarize_periods(B):
    rows = []
    for region in REGIONS:
        for t in TYPE_ROWS:
            a = subset(B, region, t)
            if a.empty:
                continue
            b = a[a.width_P1 > 0]
            pooled = (lambda col, popcol: 100 * b[col].sum() / b[popcol].sum() if len(b) else np.nan)
            p1, p2f, p2o = pooled("zone_pop_P1", "pop_P1"), pooled("zone_pop_P2_fixed", "pop_P2"), \
                pooled("zone_pop_P2_own", "pop_P2")
            rows.append(dict(
                region=region, static_type=t, n_basins=len(a), n_with_zone_P1=len(b), n_without_zone_P1=len(a) - len(b),
                pooled_P1_pct=p1, pooled_P2_fixed_pct=p2f, pooled_P2_own_pct=p2o,
                within_zone_change_pp=p2f - p1, boundary_change_pp=p2o - p2f, total_change_pp=p2o - p1,
                basin_mean_P1_pct=float(b.share_P1_pct.mean()), basin_mean_P2_fixed_pct=float(b.share_P2_fixed_pct.mean()),
                n_increase_fixed=int((b.share_P2_fixed_pct > b.share_P1_pct).sum()),
                wilcoxon_p_fixed=wilcoxon_p(b.share_P1_pct, b.share_P2_fixed_pct),
                median_width_P1=float(b.width_P1.median()) if len(b) else np.nan,
                median_width_P2=float(b.width_P2.median()) if len(b) else np.nan,
                n_wider_P2=int((b.width_P2 > b.width_P1).sum()), n_narrower_P2=int((b.width_P2 < b.width_P1).sum()),
                n_zone_lost_P2=int((b.width_P2 == 0).sum())))
    return pd.DataFrame(rows)


def basin_annual(static, prof, T):
    rows = []
    for r in static.itertuples(index=False):
        w = prof[r.unit_key]
        for per, (y0, y1) in PERIODS.items():
            for y in range(y0, y1 + 1):
                pop = w.loc[y].to_numpy(float)
                m, hrp = zone(pop, T)
                rows.append(dict(unit_key=r.unit_key, region=r.region, static_type=r.static_type, year=y, period=per,
                                 annual_type=hrp, has_zone=bool(m.any()), zone_pop=float(pop[m].sum()),
                                 profile_pop=float(pop.sum()), zone_share_pct=100 * float(pop[m].sum()) / pop.sum()))
    return pd.DataFrame(rows)


def summarize_annual(A):
    per_basin = A.groupby(["unit_key", "region", "static_type", "period"]).zone_share_pct.mean().unstack("period")
    per_basin = per_basin.reset_index()
    rows = []
    for region in REGIONS:
        for t in TYPE_ROWS:
            a, b = subset(A, region, t), subset(per_basin, region, t)
            if a.empty:
                continue
            g = a.groupby("period")[["zone_pop", "profile_pop"]].sum()
            rows.append(dict(region=region, static_type=t, n_basins=len(b),
                             pooled_P1_pct=100 * g.loc["P1", "zone_pop"] / g.loc["P1", "profile_pop"],
                             pooled_P2_pct=100 * g.loc["P2", "zone_pop"] / g.loc["P2", "profile_pop"],
                             basin_mean_P1_pct=float(b.P1.mean()), basin_mean_P2_pct=float(b.P2.mean()),
                             wilcoxon_p=wilcoxon_p(b.P1, b.P2),
                             basin_years_without_zone_P1=int((~a[a.period.eq("P1")].has_zone).sum()),
                             basin_years_without_zone_P2=int((~a[a.period.eq("P2")].has_zone).sum())))
    return pd.DataFrame(rows), per_basin


def check_annual(A):
    ap = _p(ANNUAL_CSV)
    if ap is None:
        return "ANNUAL_CSV not configured; annual zones were not cross-checked."
    if not ap.is_file():
        return f"ANNUAL_CSV not found: {ap}"
    an = pd.read_csv(ap, encoding="utf-8-sig", dtype={"unit_key": str})
    m = A.merge(an[["unit_key", "year", "agglomeration_share_pct", "HRP_type"]], on=["unit_key", "year"], how="inner")
    if m.empty:
        return "No matching basin-years in annual_hrp.csv; cross-check skipped."
    diff = float((m.zone_share_pct - m.agglomeration_share_pct.fillna(0)).abs().max())
    tdiff = int((m.annual_type != m.HRP_type.map(TYPE_LABEL)).sum())
    ok = diff < 1e-6 and tdiff == 0
    return (f"Checked {len(m)} basin-years: maximum zone-share difference {diff:.1e} percentage points; {tdiff} type mismatches"
            + ("; consistent." if ok else "; inconsistent. Verify that profiles and annual_hrp.csv belong to the same run."))


def pooled_zones(R, T):
    rows = []
    for region in REGIONS:
        m1, _ = zone(R[(region, "P1")], T)
        for per in PERIODS:
            s = band_shares(R[(region, per)])
            m, hrp = zone(R[(region, per)], T)
            rows.append(dict(region=region, period=per, profile_type=hrp, zone_bands=band_ranges(m),
                             zone_share_pct=float(s[m].sum()), share_in_P1_zone_pct=float(s[m1].sum()),
                             n_bands_at_or_above_T=int(is_high(s, T).sum()), max_band_share_pct=float(np.nanmax(s))))
    return pd.DataFrame(rows)


def main():
    run, static, T, prof = load_inputs()
    out = _p(OUT_DIR) or run / ("fig2b_zones_" + datetime.now().strftime("%Y%m%d_%H%M%S"))
    out.mkdir(parents=True, exist_ok=True)
    n = static.region.value_counts()
    print(f"Major basins: {len(static)} (North {n.get('North', 0)}, South {n.get('South', 0)}); T = {T:.4f}%", flush=True)

    R = regional_profiles(static, prof)
    sext = sextile_reproduction(R)
    B = basin_periods(static, prof, T)
    Bs = summarize_periods(B)
    A = basin_annual(static, prof, T)
    As, Ab = summarize_annual(A)
    pz = pooled_zones(R, T)
    note = check_annual(A)
    curves = pd.DataFrame([dict(region=k[0], period=k[1], band=j + 1, band_share_pct=s, cumulative_share_pct=c)
                           for k, v in R.items()
                           for j, (s, c) in enumerate(zip(band_shares(v), np.cumsum(band_shares(v))))])

    p1, p2 = (f"{a}–{b}" for a, b in PERIODS.values())
    readme = pd.DataFrame({"item": [
        "Periods", "Threshold T", "Zone rule", "Current Fig. 2B (check)", "Main result (within-zone change)",
        "Decomposition", "Annual zones", "Pooled-profile zones (reference only)", "Annual check", "Regions"],
        "definition": [
        f"P1 = {p1}, P2 = {p2}",
        f"{T:.4f}% per 1% distance interval, read from static_hrp.csv",
        f"S >= T; runs shorter than {MIN_RUN} intervals removed, neighbouring runs of the same state merged (as in Fig. 1)",
        "Band populations summed over the major basins of a region and the years of a period; band shares of the "
        "four pooled profiles split into six quantile classes together; top class summed",
        "Zone identified on each basin's P1 profile (P1 years summed); P2 share in the same bands. Pooled = sum of "
        "zone population / sum of population within Reff. Basins without a P1 zone excluded and counted",
        "Total change = within-zone change (fixed P1 bands) + boundary change (zone re-identified on the P2 profile)",
        "Zone identified for every basin and year, as agglomeration_share_pct in annual_hrp; years without a zone "
        "count as 0, so this also reflects zones appearing or disappearing",
        "Fig. 1 rule applied to the pooled regional profiles. Zones of different basins blur when profiles are "
        "pooled, so this interval is not a basin concentration zone; not for the figure",
        note, f"regions.csv: {REGION_CSV}"]})
    with pd.ExcelWriter(out / "fig2b_zone_shares.xlsx", engine="openpyxl") as xw:
        for name, df in [("README", readme), ("main within-zone", Bs), ("basins by period", B),
                         ("annual zones", As), ("annual basin means", Ab), ("current B check", sext),
                         ("pooled-profile zones", pz), ("pooled curves", curves)]:
            df.to_excel(xw, sheet_name=name, index=False)

    L = [f"T = {T:.4f}%; P1 = {p1}, P2 = {p2}; major basins: North {n.get('North', 0)}, South {n.get('South', 0)}", note, "",
         "1. Quantile-based regional profile summary:"]
    for region in REGIONS:
        q = sext[sext.region.eq(region)].set_index("period")
        L.append(f"   {region}: {q.share_pct.P1:.2f}% ({q.n_bands.P1} bands) -> {q.share_pct.P2:.2f}% ({q.n_bands.P2} bands); "
                 f"fixed P1 bands: {q.share_in_P1_bands_pct.P1:.2f}% -> {q.share_in_P1_bands_pct.P2:.2f}%")
    L += ["", "2. Fixed P1 concentration zones: population-weighted shares (basin means in parentheses):"]
    for region in REGIONS:
        q = Bs[Bs.region.eq(region) & Bs.static_type.eq("All")].iloc[0]
        L.append(f"   {region} (n = {q.n_with_zone_P1}; {q.n_without_zone_P1} additional basins have no P1 zone): "
                 f"{q.pooled_P1_pct:.2f}% → {q.pooled_P2_fixed_pct:.2f}%（{q.basin_mean_P1_pct:.2f}% → "
                 f"{q.basin_mean_P2_fixed_pct:.2f}%); {q.n_increase_fixed}/{q.n_with_zone_P1} increased; "
                 f"paired Wilcoxon P = {q.wilcoxon_p_fixed:.3g}")
    L += ["", "3. Population-weighted decomposition (percentage points): total = within-zone + boundary change"]
    for region in REGIONS:
        q = Bs[Bs.region.eq(region) & Bs.static_type.eq("All")].iloc[0]
        L.append(f"   {region}: {q.total_change_pp:+.2f} = {q.within_zone_change_pp:+.2f} {q.boundary_change_pp:+.2f}；"
                 f"median zone width {q.median_width_P1:g} -> {q.median_width_P2:g} bands; wider {q.n_wider_P2}, narrower "
                 f"{q.n_narrower_P2}, zone lost in P2 {q.n_zone_lost_P2}")
    L += ["", "   By static type (population-weighted within-zone change):"]
    for region in REGIONS:
        for t in TYPE_ROWS[1:]:
            q = Bs[Bs.region.eq(region) & Bs.static_type.eq(t)]
            if not len(q):
                continue
            q = q.iloc[0]
            if q.n_with_zone_P1 == 0:
                L.append(f"   {region} {t} (n = 0/{q.n_basins}): no P1 concentration zones")
                continue
            p = f"P = {q.wilcoxon_p_fixed:.3g}" if np.isfinite(q.wilcoxon_p_fixed) else "Too few basins for testing"
            L.append(f"   {region} {t} (n = {q.n_with_zone_P1}/{q.n_basins}): {q.pooled_P1_pct:.2f}% → "
                     f"{q.pooled_P2_fixed_pct:.2f}%; {q.n_increase_fixed} increased; {p}")
    L += ["", "4. Annual zones: agglomeration_share_pct; years without a zone receive zero:"]
    for region in REGIONS:
        q = As[As.region.eq(region) & As.static_type.eq("All")].iloc[0]
        L.append(f"   {region} (n = {q.n_basins}): {q.pooled_P1_pct:.2f}% -> {q.pooled_P2_pct:.2f}% (basin means "
                 f"{q.basin_mean_P1_pct:.2f}% -> {q.basin_mean_P2_pct:.2f}%, P = {q.wilcoxon_p:.3g}); basin-years without zones "
                 f"{q.basin_years_without_zone_P1} → {q.basin_years_without_zone_P2}")
    L += ["", "5. Reference: classification of pooled profiles:"]
    for region in REGIONS:
        q = pz[pz.region.eq(region)].set_index("period")
        L.append(f"   {region}: P1 {q.profile_type.P1}, zone {q.zone_bands.P1 or 'none'}; P2 {q.profile_type.P2}, "
                 f"zone {q.zone_bands.P2 or 'none'}; maximum pooled band share {q.max_band_share_pct.max():.2f}%")
    (out / "fig2b_numbers.txt").write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n".join(L))
    print(f"\nOutput: {out}")


if __name__ == "__main__":
    main()
