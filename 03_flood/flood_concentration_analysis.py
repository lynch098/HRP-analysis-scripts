"""Calculate exposure concentration and regional decompositions.
e = E/P; s = flooded area / basin area; concentration ratio A = e/s."""

import json as _path_json
from pathlib import Path as _ConfigPath


_DATA_PATHS = _path_json.loads(
    (_ConfigPath(__file__).resolve().parents[1] / "paths.json").read_text(encoding="utf-8")
)
import copy
import sys
import warnings
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats


LEVELS_CSV = Path(_DATA_PATHS["LEVELS_CSV"])
AREA_CSV = Path(_DATA_PATHS["AREA_CSV"])
STATIC_CSV = Path(_DATA_PATHS["STATIC_CSV"])
REGION_CSV = Path(_DATA_PATHS["REGION_CSV"])
OUT_DIR = Path(__file__).resolve().parent / "concentration_results"
FIG_NAME = "fig3_concentration"
YEAR, BASE_YEAR = 2020, 1990
SCENARIOS = {"SSP2-4.5": (2, "rcp4p5"), "SSP2-8.5": (2, "rcp8p5"), "SSP3-8.5": (3, "rcp8p5")}
FUT_YEARS = (2030, 2050, 2080)
POP_FIXED_YEAR = 2020
POP_MODE = "fixed"
POP_SENSITIVITY = True
POP_LABEL = {"fixed": f"fixed at {POP_FIXED_YEAR}", "same-year": "same-year SSP"}
CURVE_SCENARIO = "SSP2-8.5"
N_BOOT = 2000
N_PERM = 5000
SEED = 20261028
MIN_N_CI = 3
MAP_MODE = "change"
MAP_BREAKS = {"change": (0.95, 1.05), "2020": (1.0, 2.0)}
TYPES = ["Near-river", "Dispersion", "Far-river"]
TYPE_LABEL = {"Near-river": "River-adjacent", "Dispersion": "Diffuse", "Far-river": "River-detached"}
TYPE_ALIASES = {"Near-river": "Near-river", "River-adjacent": "Near-river", "Dispersion": "Dispersion",
                "Diffuse": "Dispersion", "Far-river": "Far-river", "River-detached": "Far-river"}
COMP = [("e", "Exposed share", "D", 3.4), ("A", "Concentration ratio", "o", 3.6), ("s", "Flood-zone area", "s", 3.2)]
KEYS = ["unit_key", "grid", "hazard", "gcm", "hazard_year"]

def _p(x):
    s = str(x).strip() if x is not None else ""
    return Path(s) if s not in ("", ".") else None

def glabel(t):
    return "All" if t is None else TYPE_LABEL[t]

def load():
    p = _p(LEVELS_CSV)
    if p is None or not p.is_file():
        sys.exit(f"Set LEVELS_CSV to an exposure levels.csv file (current: {p}).")
    lv = pd.read_csv(p, dtype={"unit_key": str, "pop_product": str, "hazard": str, "gcm": str}, low_memory=False)
    need = {"unit_key", "pop_product", "pop_year", "hazard", "gcm", "hazard_year", "P", "E", "flood_cells", "n_cells"}
    if need - set(lv.columns):
        sys.exit(f"Missing columns in levels.csv: {sorted(need - set(lv.columns))}")
    for col, path, name in [("HRP_type", STATIC_CSV, "STATIC_CSV"), ("region", REGION_CSV, "REGION_CSV")]:
        if col not in lv.columns or lv[col].isna().all():
            q = _p(path)
            if q is None or not q.is_file():
                sys.exit(f"levels.csv lacks {col}; configure {name}.")
            add = pd.read_csv(q, dtype={"unit_key": str})[["unit_key", col]].drop_duplicates("unit_key")
            lv = lv.drop(columns=[col], errors="ignore").merge(add, on="unit_key", how="left")
    lv = lv[lv.level.astype(str).eq("major")] if "level" in lv.columns else lv[~lv.unit_key.str.contains("__")]
    lv = lv.copy()
    lv["type"] = lv.HRP_type.astype(str).str.strip().map(TYPE_ALIASES)
    lv["reg"] = np.where(lv.region.astype(str).str.lower().str.contains("south"), "South", "North")
    for c in ["P", "E", "flood_cells", "n_cells"]:
        lv[c] = pd.to_numeric(lv[c], errors="coerce")
    for c in ["pop_year", "hazard_year"]:
        lv[c] = pd.to_numeric(lv[c], errors="coerce").round().astype("Int64")
    lv["grid"] = np.where(lv.pop_product.eq("GlobPOP"), "GlobPOP", "SSP")
    bad = lv.drop_duplicates("unit_key").type.isna().sum()
    if bad:
        print(f"Warning: {bad} major basins have unrecognized HRP types and are excluded from grouped summaries.")
    return use_area(lv), p

def use_area(lv):

    q = _p(AREA_CSV)
    lv.attrs["area"] = "pixel counts"
    if q is None:
        return lv
    if not q.is_file():
        sys.exit(f"AREA_CSV not found: {q}")
    ar = pd.read_csv(q, dtype={"unit_key": str, "grid": str, "hazard": str, "gcm": str})
    ar["hazard_year"] = pd.to_numeric(ar.hazard_year, errors="coerce").round().astype("Int64")
    ar = ar.drop_duplicates(KEYS)[KEYS + ["flood_cells", "area_km2", "flood_area_km2"]] \
        .rename(columns={"flood_cells": "flood_cells_check"})
    lv = lv.merge(ar, on=KEYS, how="left")
    hz = lv.hazard.astype(str)
    used = (lv.grid.eq("GlobPOP") & hz.eq("historical")) | (lv.grid.eq("SSP") & hz.str.startswith("rcp"))
    miss = lv[used & lv.area_km2.isna()]
    if len(miss):
        ex = miss.drop_duplicates(KEYS)[KEYS].head(5).to_string(index=False)
        sys.exit(f"AREA_CSV lacks {miss.drop_duplicates(KEYS).shape[0]} unit-hazard combinations; examples:\n{ex}")
    rel = (lv.flood_cells - lv.flood_cells_check).abs() / lv.flood_cells.abs().clip(lower=1e-9)
    if (rel[used] > 1e-6).any():
        print(f"Warning: sum(f) differs from levels.csv in {int((rel[used] > 1e-6).sum())} rows; verify identical hazard masks.")
    lv["flood_cells_px"], lv["n_cells_px"] = lv.flood_cells, lv.n_cells
    lv.loc[used, "flood_cells"] = lv.loc[used, "flood_area_km2"]
    lv.loc[used, "n_cells"] = lv.loc[used, "area_km2"]
    lv.attrs["area"] = "pixel-area weighted"
    return lv

def basin_table(lv):

    h = lv[(lv.pop_product == "GlobPOP") & (lv.hazard == "historical")]

    def year(y):
        d = h[h.pop_year == y].drop_duplicates("unit_key").set_index("unit_key")
        if d.empty:
            sys.exit(f"No GlobPOP {y} records with historical hazard in levels.csv.")
        return d

    a, b = year(YEAR), year(BASE_YEAR)
    u = a[["type", "reg", "P", "E", "flood_cells", "n_cells"]].copy()
    with np.errstate(divide="ignore", invalid="ignore"):
        u["e_pct"] = 100 * u.E / u.P
        u["s_pct"] = 100 * u.flood_cells / u.n_cells
        u["A"] = np.where(u.flood_cells > 0, u.e_pct / u.s_pct, np.nan)
        u["P0"], u["E0"] = b.P.reindex(u.index), b.E.reindex(u.index)
        u["e0_pct"] = 100 * u.E0 / u.P0
        u["A0"] = np.where(u.flood_cells > 0, u.e0_pct / u.s_pct, np.nan)
        u["A_change"] = np.where(u.A0 > 0, u.A / u.A0, np.nan)
    return u.reset_index()

def future_table(lv, sname, pop=None):

    pop = pop or POP_MODE
    if pop not in POP_LABEL:
        sys.exit(f"POP_MODE must be one of {list(POP_LABEL)} (current: {pop!r}).")
    ssp, rcp = SCENARIOS[sname]
    f = lv[(lv.pop_product == f"SSP{ssp}") & (lv.hazard == rcp) & lv.hazard_year.isin(FUT_YEARS)]
    keep = (f.pop_year == POP_FIXED_YEAR) if pop == "fixed" else (f.pop_year == f.hazard_year)
    f = f[keep.fillna(False).astype(bool)]
    if f.empty or set(f.hazard_year.dropna().astype(int)) != set(FUT_YEARS):
        return None, None
    piv = f.pivot_table(index="unit_key", columns=["hazard_year", "gcm"], values=["E", "flood_cells"], aggfunc="first")
    per = {(v, y): piv[c][y].sort_index(axis=1) for v, c in [("E", "E"), ("F", "flood_cells")] for y in FUT_YEARS}
    meta = f.drop_duplicates("unit_key").set_index("unit_key")[["type", "reg", "P", "n_cells"]]
    cols = {f"{v}{y}": per[(v, y)].mean(axis=1).reindex(meta.index) for v, y in per}
    if pop == "same-year":
        Py = f.assign(hy=f.hazard_year.astype(int)).groupby(["unit_key", "hy"]).P.first().unstack()
        cols.update({f"P{y}": Py[y].reindex(meta.index) for y in FUT_YEARS})
    d = meta.assign(**cols, n_gcm=per[("E", FUT_YEARS[0])].notna().sum(axis=1).reindex(meta.index))
    return d.dropna(subset=list(cols)), per

def pop_arrays(g, y0, y1):

    if f"P{y0}" in g.columns:
        return g[f"P{y0}"].to_numpy(float), g[f"P{y1}"].to_numpy(float)
    return None, None

def basin_future(d):

    y0, y1 = FUT_YEARS[0], FUT_YEARS[-1]
    P0, P1 = (d[f"P{y0}"], d[f"P{y1}"]) if f"P{y0}" in d.columns else (1.0, 1.0)
    with np.errstate(divide="ignore", invalid="ignore"):
        re_ = (d[f"E{y1}"] / P1) / (d[f"E{y0}"] / P0)
        rs_ = d[f"F{y1}"] / d[f"F{y0}"]
    ok = (d[f"E{y0}"] > 0) & (d[f"F{y0}"] > 0) & (d[f"F{y1}"] > 0)
    m = pd.DataFrame(dict(e_change=re_.where(ok), s_change=rs_.where(ok)), index=d.index)
    m["A_change"] = m.e_change / m.s_change
    return m

def map_table(u, fut):

    if MAP_MODE not in MAP_BREAKS:
        sys.exit(f"MAP_MODE must be one of {list(MAP_BREAKS)} (current: {MAP_MODE!r}).")
    val = u.set_index("unit_key").A if MAP_MODE == "2020" else fut.A_change
    mv = u[["unit_key", "type", "reg"]].copy()
    mv["map_value"] = mv.unit_key.map(val).astype(float)
    ok = np.isfinite(mv.map_value) & mv.type.notna()
    mv["map_class"] = np.where(ok, np.digitize(mv.map_value.fillna(0), MAP_BREAKS[MAP_MODE]), np.nan)
    return mv

def pooled(d, idx=None):

    E, P, F, N = (d[c].to_numpy(float) for c in ["E", "P", "flood_cells", "n_cells"])
    if idx is None:
        return E.sum() / P.sum(), F.sum() / N.sum()
    return E[idx].sum(1) / P[idx].sum(1), F[idx].sum(1) / N[idx].sum(1)

def ci(v):
    v = np.asarray(v, float)
    v = v[np.isfinite(v)]
    return tuple(np.percentile(v, [2.5, 97.5])) if len(v) else (np.nan, np.nan)

def select(d, t, reg):
    m = d.reg.eq(reg)
    return d[m] if t is None else d[m & d.type.eq(t)]

def shown_types(u):

    return [t for t in TYPES if min(len(select(u, t, "North")), len(select(u, t, "South"))) >= MIN_N_CI]

def south_north(u, rng):

    rows, boots = [], {}
    for t in [None] + TYPES:
        S, N = select(u, t, "South"), select(u, t, "North")
        if S.empty or N.empty:
            continue
        (eS, sS), (eN, sN) = pooled(S), pooled(N)
        R = dict(e=eS / eN, s=sS / sN)
        R["A"] = R["e"] / R["s"]
        rec = dict(group=glabel(t), n_North=len(N), n_South=len(S), e_North_pct=100 * eN, e_South_pct=100 * eS,
                   s_North_pct=100 * sN, s_South_pct=100 * sS, A_North=eN / sN, A_South=eS / sS,
                   **{f"R_{k}": v for k, v in R.items()})
        if min(len(S), len(N)) >= MIN_N_CI:
            iS, iN = rng.integers(0, len(S), (N_BOOT, len(S))), rng.integers(0, len(N), (N_BOOT, len(N)))
            (beS, bsS), (beN, bsN) = pooled(S, iS), pooled(N, iN)
            with np.errstate(divide="ignore", invalid="ignore"):
                b = dict(e=beS / beN, s=bsS / bsN)
                b["A"] = b["e"] / b["s"]
            boots[glabel(t)] = b
            for k in b:
                rec[f"R_{k}_lo"], rec[f"R_{k}_hi"] = ci(b[k])
        if R["e"] > 0 and abs(np.log(R["e"])) > 0.05:
            rec["share_of_log_gap_from_A_pct"] = 100 * np.log(R["A"]) / np.log(R["e"])
        rows.append(rec)
    return pd.DataFrame(rows), boots

def change(E0, E1, F0, F1, idx=None, P0=None, P1=None):

    with np.errstate(divide="ignore", invalid="ignore"):
        if idx is None:
            e, s = E1.sum() / E0.sum(), F1.sum() / F0.sum()
            if P0 is not None:
                e = e * (P0.sum() / P1.sum())
        else:
            e, s = E1[idx].sum(1) / E0[idx].sum(1), F1[idx].sum(1) / F0[idx].sum(1)
            if P0 is not None:
                e = e * (P0[idx].sum(1) / P1[idx].sum(1))
    return dict(e=e, s=s, A=e / s)

def future_curves(d, per, rng):

    y0, rows = FUT_YEARS[0], []
    for reg in ["North", "South"]:
        for t in [None] + TYPES:
            g = select(d, t, reg)
            if g.empty:
                continue
            idx = rng.integers(0, len(g), (N_BOOT, len(g))) if len(g) >= MIN_N_CI else None
            E0, F0 = g[f"E{y0}"].to_numpy(float), g[f"F{y0}"].to_numpy(float)
            for k, *_ in COMP:
                rows.append(dict(region=f"Global {reg}", group=glabel(t), n=len(g), year=y0, component=k,
                                 value=1.0, lo=1.0, hi=1.0, gcm_min=1.0, gcm_max=1.0, gcm_n_above1=np.nan,
                                 n_gcm=np.nan))
            for y in FUT_YEARS[1:]:
                E1, F1 = g[f"E{y}"].to_numpy(float), g[f"F{y}"].to_numpy(float)
                P0, P1 = pop_arrays(g, y0, y)
                R = change(E0, E1, F0, F1, P0=P0, P1=P1)
                B = change(E0, E1, F0, F1, idx, P0, P1) if idx is not None else None
                ks = g.index
                with np.errstate(divide="ignore", invalid="ignore"):
                    eg = per[("E", y)].loc[ks].sum() / per[("E", y0)].loc[ks].sum()
                    if P0 is not None:
                        eg = eg * (P0.sum() / P1.sum())
                    sg = per[("F", y)].loc[ks].sum() / per[("F", y0)].loc[ks].sum()
                G = dict(e=eg, s=sg, A=eg / sg)
                for k, *_ in COMP:
                    lo, hi = ci(B[k]) if B is not None else (np.nan, np.nan)
                    rows.append(dict(region=f"Global {reg}", group=glabel(t), n=len(g), year=y, component=k,
                                     value=float(R[k]), lo=lo, hi=hi, gcm_min=float(np.nanmin(G[k])),
                                     gcm_max=float(np.nanmax(G[k])), gcm_n_above1=int((G[k] > 1).sum()),
                                     n_gcm=int(np.isfinite(G[k]).sum())))
    return pd.DataFrame(rows)

def future_tests(d, rng):

    y0, y1 = FUT_YEARS[0], FUT_YEARS[-1]
    rows = []
    for t in [None] + TYPES:
        g = d if t is None else d[d.type.eq(t)]
        south = g.reg.eq("South").to_numpy()
        nS, nN = int(south.sum()), int((~south).sum())
        if min(nS, nN) < MIN_N_CI:
            continue
        E0, E1, F0, F1 = (g[c].to_numpy(float) for c in [f"E{y0}", f"E{y1}", f"F{y0}", f"F{y1}"])
        P0, P1 = pop_arrays(g, y0, y1)

        def log_gap(lab):

            L = lab.astype(float)
            with np.errstate(divide="ignore", invalid="ignore"):
                e = np.log((L @ E1) / (L @ E0)) - np.log(((1 - L) @ E1) / ((1 - L) @ E0))
                if P0 is not None:
                    e = e + np.log((L @ P0) / (L @ P1)) - np.log(((1 - L) @ P0) / ((1 - L) @ P1))
                s = np.log((L @ F1) / (L @ F0)) - np.log(((1 - L) @ F1) / ((1 - L) @ F0))
            return dict(e=e, s=s, A=e - s)

        obs = {k: float(v[0]) for k, v in log_gap(south[None, :]).items()}
        perm = np.array([rng.permutation(south) for _ in range(N_PERM)])
        null = log_gap(perm)
        iS = rng.integers(0, nS, (N_BOOT, nS))
        iN = rng.integers(0, nN, (N_BOOT, nN))
        S_, N_ = g[south], g[~south]
        bS = change(*(S_[c].to_numpy(float) for c in [f"E{y0}", f"E{y1}", f"F{y0}", f"F{y1}"]), iS,
                    *pop_arrays(S_, y0, y1))
        bN = change(*(N_[c].to_numpy(float) for c in [f"E{y0}", f"E{y1}", f"F{y0}", f"F{y1}"]), iN,
                    *pop_arrays(N_, y0, y1))
        for k, lab, *_ in COMP:
            nv = null[k][np.isfinite(null[k])]
            p = (1 + np.sum(np.abs(nv) >= abs(obs[k]) - 1e-12)) / (1 + len(nv))
            lo, hi = ci(bS[k] / bN[k])
            rows.append(dict(group=glabel(t), component=k, quantity=lab, n_North=nN, n_South=nS,
                             south_over_north=float(np.exp(obs[k])), lo=lo, hi=hi, P_permutation=float(p)))
    return pd.DataFrame(rows)

def within_between(d):
    """Factor pooled concentration change into within-basin and between-basin components using exposure- and flooded-area-weighted growth."""
    y0, y1 = FUT_YEARS[0], FUT_YEARS[-1]
    rows = []
    for reg in ["North", "South"]:
        for t in [None] + TYPES:
            g = select(d, t, reg)
            if g.empty:
                continue
            E0, E1, F0, F1 = (g[c].to_numpy(float) for c in [f"E{y0}", f"E{y1}", f"F{y0}", f"F{y1}"])
            P0, P1 = pop_arrays(g, y0, y1)
            Re, Rs = E1.sum() / E0.sum(), F1.sum() / F0.sum()
            if P0 is not None:
                Re = Re * (P0.sum() / P1.sum())
            rs_b = np.divide(F1, F0, out=np.zeros_like(F1), where=F0 > 0)
            mE = float(np.sum(E0 / E0.sum() * rs_b))
            ok = (E0 > 0) & (F0 > 0) & (F1 > 0)
            rA = (E1[ok] / E0[ok]) / (F1[ok] / F0[ok])
            if P0 is not None:
                rA = rA * (P0[ok] / P1[ok])
            rows.append(dict(region=f"Global {reg}", group=glabel(t), n=len(g), R_A=Re / Rs, within=Re / mE,
                             between=mE / Rs, n_basins_valid=int(ok.sum()), n_above1=int((rA > 1).sum()),
                             n_below1=int((rA < 1).sum()), median_basin=float(np.median(rA)) if ok.any() else np.nan))
    return pd.DataFrame(rows)

def mw(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    a, b = a[np.isfinite(a)], b[np.isfinite(b)]
    if min(len(a), len(b)) < MIN_N_CI:
        return np.nan
    return float(stats.mannwhitneyu(a, b, alternative="two-sided").pvalue)

def tests(u, col):
    rows = []
    for t in [None] + TYPES:
        a, b = select(u, t, "North")[col].dropna(), select(u, t, "South")[col].dropna()
        rows.append(dict(variable=col, test="Global North vs Global South", group=glabel(t), n_1=len(a), n_2=len(b),
                         median_1=a.median(), median_2=b.median(), P=mw(a, b)))
    for reg in ["North", "South"]:
        r = u[u.reg.eq(reg)]
        a, b = r[r.type.eq("Near-river")][col].dropna(), r[r.type.ne("Near-river") & r.type.notna()][col].dropna()
        rows.append(dict(variable=col, test="River-adjacent vs other types", group=f"Global {reg}", n_1=len(a),
                         n_2=len(b), median_1=a.median(), median_2=b.median(), P=mw(a, b)))
    return pd.DataFrame(rows)

def hist_change(u):
    rows = []
    for reg in ["North", "South"]:
        for t in [None] + TYPES:
            g = select(u, t, reg)
            if g.empty:
                continue
            pooled_ratio = (g.E.sum() / g.P.sum()) / (g.E0.sum() / g.P0.sum())
            rows.append(dict(region=f"Global {reg}", group=glabel(t), n=len(g), pooled_ratio=pooled_ratio,
                             median_ratio=g.A_change.median(), n_increased=int((g.A_change > 1).sum()),
                             n_valid=int(g.A_change.notna().sum())))
    return pd.DataFrame(rows)

def geo_decomp(u):

    rows = []
    ok = (u.E > 0) & (u.flood_cells > 0)
    for t in [None] + TYPES:
        S, N = select(u[ok], t, "South"), select(u[ok], t, "North")
        if min(len(S), len(N)) == 0:
            continue
        m = {k: np.exp(np.log(S[c]).mean() - np.log(N[c]).mean()) for k, c in [("e", "e_pct"), ("A", "A"), ("s", "s_pct")]}
        excl = int((~ok & (u.type.eq(t) if t is not None else True)).sum())
        rows.append(dict(group=glabel(t), n_North=len(N), n_South=len(S), excluded_zero=excl,
                         **{f"R_{k}": v for k, v in m.items()}))
    return pd.DataFrame(rows)

def main():
    lv, path = load()
    out = _p(OUT_DIR) or path.parent / "fig3_concentration"
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(SEED)
    u = basin_table(lv)
    Te, Ta = tests(u, "e_pct"), tests(u, "A")
    C, boots = south_north(u, rng)
    curves, ftests, wb, dmain = {}, {}, {}, None
    states, dfut = {}, {}
    for sname in SCENARIOS:
        d, per = future_table(lv, sname)
        if d is not None:
            states[sname], dfut[sname] = copy.deepcopy(rng.bit_generator.state), d
            curves[sname] = future_curves(d, per, rng).assign(scenario=sname, population=POP_LABEL[POP_MODE])
            ftests[sname] = future_tests(d, rng).assign(scenario=sname, population=POP_LABEL[POP_MODE])
            wb[sname] = within_between(d).assign(scenario=sname)
            if sname == CURVE_SCENARIO:
                dmain = d
    if dmain is None:
        sys.exit(f"No {CURVE_SCENARIO} GCM hazard records with {POP_LABEL[POP_MODE]} population in levels.csv.")


    alt = "same-year" if POP_MODE == "fixed" else "fixed"
    pcurves, ptests, pnote = {}, {}, []
    for sname in (states if POP_SENSITIVITY else []):
        d2, per2 = future_table(lv, sname, alt)
        if d2 is None:
            pnote.append(f"{sname}: no {POP_LABEL[alt]} population records; population sensitivity skipped.")
            continue
        d0 = dfut[sname]
        if set(d2.index) == set(d0.index):
            d2 = d2.loc[d0.index]
        else:
            pnote.append(f"{sname}: population variants contain different basin sets ({len(d0)} and {len(d2)}); resamples are not paired.")
        r2 = np.random.default_rng()
        r2.bit_generator.state = copy.deepcopy(states[sname])
        pcurves[sname] = future_curves(d2, per2, r2).assign(scenario=sname, population=POP_LABEL[alt])
        ptests[sname] = future_tests(d2, r2).assign(scenario=sname, population=POP_LABEL[alt])
    fut = basin_future(dmain)
    mv = map_table(u, fut)
    Dc, Dt = curves.get(CURVE_SCENARIO), ftests.get(CURVE_SCENARIO, pd.DataFrame())
    Hc, G = hist_change(u), geo_decomp(u)


    inter = None
    ra, df = TYPE_LABEL["Near-river"], TYPE_LABEL["Dispersion"]
    if ra in boots and df in boots:
        r = C.set_index("group")
        delta = np.log(r.loc[ra, "R_A"]) - np.log(r.loc[df, "R_A"])
        with np.errstate(divide="ignore", invalid="ignore"):
            bd = np.log(boots[ra]["A"]) - np.log(boots[df]["A"])
        inter = (delta, *ci(bd))

    tag = f"{FUT_YEARS[0]}_{FUT_YEARS[-1]}"
    basins = u.merge(fut.add_suffix(f"_{tag}").reset_index(), on="unit_key", how="left") \
        .merge(mv[["unit_key", "map_value", "map_class"]], on="unit_key", how="left")
    with pd.ExcelWriter(out / f"{FIG_NAME}_values.xlsx", engine="openpyxl") as xw:
        pd.DataFrame({"README": __doc__.splitlines()}).to_excel(xw, sheet_name="README", index=False)
        basins.assign(HRP_type=basins.type.map(TYPE_LABEL), region=basins.reg.map(lambda r: f"Global {r}")) \
            .drop(columns=["type", "reg"]).to_excel(xw, sheet_name="basins (map join)", index=False)
        pd.concat(curves.values(), ignore_index=True).to_excel(xw, sheet_name="C curves", index=False)
        pd.concat(ftests.values(), ignore_index=True).to_excel(xw, sheet_name="C South-North tests", index=False)
        if pcurves:
            pd.concat(pcurves.values(), ignore_index=True).to_excel(xw, sheet_name="SI pop curves", index=False)
            pd.concat(ptests.values(), ignore_index=True).to_excel(xw, sheet_name="SI pop tests", index=False)
        pd.concat(wb.values(), ignore_index=True).to_excel(xw, sheet_name="A within-between", index=False)
        pd.concat([Te, Ta], ignore_index=True).to_excel(xw, sheet_name="B and SI tests", index=False)
        C.to_excel(xw, sheet_name="SI South-North 2020", index=False)
        Hc.to_excel(xw, sheet_name="historical change", index=False)
        G.to_excel(xw, sheet_name="sensitivity geo-mean", index=False)

    print(f"Analysis values saved to: {out}")


if __name__ == "__main__":
    main()
