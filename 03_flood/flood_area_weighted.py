
"""Calculate flooded-area fraction as sum(f_i*a_i)/sum(a_i).
Use the same masks and basin membership as the bundled exposure module."""

import json as _path_json
from pathlib import Path as _ConfigPath


_DATA_PATHS = _path_json.loads(
    (_ConfigPath(__file__).resolve().parents[1] / "paths.json").read_text(encoding="utf-8")
)
import importlib
import sys
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd


FE_SCRIPT = Path(__file__).resolve().with_name("flood_exposure_v3_3_reviewed.py")
OUT_DIR = Path(_DATA_PATHS["FLOOD_AREA_OUTPUT_ROOT"])
WORKERS = 3
GROUPS = ""
R_EARTH_KM = 6371.0088
VERSION = "area-weighted-v1"


def _p(x):
    s = str(x).strip() if x is not None else ""
    return Path(s) if s not in ("", ".") else None


FE_PATH = _p(FE_SCRIPT) or Path(__file__).resolve().parent / "flood_exposure_v3.py"
if not FE_PATH.is_file():
    sys.exit(f"Exposure module not found: {FE_PATH}. Check FE_SCRIPT.")
sys.path.insert(0, str(FE_PATH.parent))
fe = importlib.import_module(FE_PATH.stem)


def cell_area_km2(ctx):

    g = ctx["grid"]
    tr = g.tr
    rows = ctx["cells"] // g.width
    crs = fe.as_crs(g.crs)
    if crs.is_geographic:
        top = tr.f + rows * tr.e
        bot = top + tr.e
        return R_EARTH_KM ** 2 * np.radians(abs(tr.a)) * np.abs(np.sin(np.radians(top)) - np.sin(np.radians(bot)))
    factor = crs.linear_units_factor[1] if crs.linear_units_factor else 1.0
    return np.full(len(rows), abs(tr.a * tr.e) * factor ** 2 / 1e6)


def area_group(job):

    t0 = time.time()
    get = fe.contexts(job)
    rows = []
    for prod in job["products"]:
        ctx = get(fe.Grid(**prod["grid"]))
        a = cell_area_km2(ctx)
        atot = {k: fe.sub_sum(a, p) for k, p in ctx["pos"].items()}
        for hkey in prod["keys"]:
            f = fe.read_mask(prod["masks"][hkey], ctx)
            fa = f * a
            sc, gcm, hy = hkey.split("|")
            for k, p in ctx["pos"].items():
                n = len(ctx["cells"]) if p is None else len(p)
                rows.append((k, prod["name"], sc, gcm, int(hy), n, float(fe.sub_sum(f, p)), float(atot[k]),
                             float(fe.sub_sum(fa, p))))
            del f, fa
    cols = ["unit_key", "grid", "hazard", "gcm", "hazard_year", "n_cells", "flood_cells", "area_km2", "flood_area_km2"]
    return dict(group=job["group"], table=pd.DataFrame(rows, columns=cols), seconds=time.time() - t0)


def build_jobs(units, grids, masks):
    rcps = {r for _, r in fe.SCENARIOS.values()}

    def wanted(k):
        if k == fe.HIST_KEY:
            return True
        part = k.split("|")
        return len(part) == 3 and part[0] in rcps and part[1] in fe.GCMS and part[2].isdigit() \
            and int(part[2]) in fe.FUTURE_YEARS

    prods = []
    for name in ["GlobPOP", "SSP"]:
        if name not in masks or name not in grids:
            continue
        _, paths = masks[name]
        keys = sorted((k for k in paths if wanted(k) and (name == "SSP" or k == fe.HIST_KEY)),
                      key=lambda k: (k != fe.HIST_KEY, k))
        prods.append(dict(name=name, grid=grids[name].as_dict(), masks={k: paths[k] for k in keys}, keys=keys))
    jobs = []
    for grp, g in units.groupby("major", sort=True):
        ulist = [(r.unit_key, r.basin_shp) for r in g.itertuples(index=False)]
        sig = fe.sha(dict(v=VERSION, units=[(k, fe.shp_fingerprint(s)) for k, s in ulist],
                          prods=[dict(name=p["name"], grid=p["grid"],
                                      masks={h: fe.fingerprint(m) for h, m in p["masks"].items()}) for p in prods]))
        jobs.append(dict(group=grp, units=ulist, products=prods, sig=sig))
    return jobs


def main():
    t0 = time.time()
    all_units, region_src = fe.load_units(fe.HRP_RUN, fe.REGION_CSV)
    units = all_units
    if GROUPS.strip():
        keep = [g.strip() for g in GROUPS.split(",") if g.strip()]
        units = all_units[all_units.major.isin(keep)]
        if units.empty:
            sys.exit(f"Unknown basin groups: {keep}")
    print(f"Units: {len(units)} (major basins: {(units.level == 'major').sum()})", flush=True)
    inv = fe.inventory()
    if inv["problems"]:
        print("\n".join(["Exposure input errors:"] + inv["problems"][:20]))
        sys.exit("Resolve these errors with the exposure module --check; use the settings that generated levels.csv.")
    grids = fe.check_grids(inv)
    print("Preparing flood masks using the exposure cache:", flush=True)
    bounds = {r.unit_key: fe.load_polygon(r.basin_shp, "EPSG:4326").bounds for r in all_units.itertuples(index=False)}
    masks, _ = fe.prepare_masks(inv, grids, bounds, WORKERS)

    out = _p(OUT_DIR) or Path(fe.OUTPUT_ROOT) / "area_weighted"
    cache = out / "cache"
    cache.mkdir(parents=True, exist_ok=True)
    jobs = build_jobs(units, grids, masks)
    n_maps = sum(len(p["keys"]) for p in jobs[0]["products"]) if jobs else 0
    results, todo, errors = {}, [], []
    for j in jobs:
        f = cache / f"{fe.safe_name(j['group'])}__{j['sig']}.csv.gz"
        if f.is_file():
            results[j["group"]] = pd.read_csv(f, dtype={"unit_key": str, "grid": str, "hazard": str, "gcm": str})
        else:
            todo.append((j, f))
    print(f"Groups: {len(jobs)} (cached {len(jobs) - len(todo)}, pending {len(todo)}); {n_maps} hazard maps per group", flush=True)

    def keep(res, f):
        res["table"].to_csv(f, index=False, compression="gzip")
        results[res["group"]] = res["table"]
        print(f"  {res['group']}: {res['seconds']:.0f} s", flush=True)

    if WORKERS <= 1:
        for j, f in todo:
            try:
                keep(area_group(j), f)
            except Exception:
                errors.append((j["group"], traceback.format_exc()))
    else:
        with ProcessPoolExecutor(max_workers=WORKERS) as ex:
            futs = {ex.submit(area_group, j): (j, f) for j, f in todo}
            for fu in as_completed(futs):
                j, f = futs[fu]
                try:
                    keep(fu.result(), f)
                except Exception:
                    errors.append((j["group"], traceback.format_exc()))
    if errors:
        pd.DataFrame(errors, columns=["group", "traceback"]).to_csv(out / "errors.csv", index=False, encoding="utf-8-sig")
        sys.exit(f"{len(errors)} groups failed; see {out / 'errors.csv'}. Completed groups are cached.")

    tab = pd.concat(results.values(), ignore_index=True)
    tab["unit_key"] = tab.unit_key.astype(str)
    tab = tab[tab.unit_key.isin(units.unit_key)].merge(units[["unit_key", "level"]], on="unit_key", how="left")
    lead = ["unit_key", "level", "grid", "hazard", "gcm", "hazard_year"]
    tab = tab[lead + [c for c in tab.columns if c not in lead]]
    tab.to_csv(out / "flood_area_weighted.csv", index=False, encoding="utf-8-sig")


    h = tab[(tab.level == "major") & (tab.grid == "GlobPOP") & (tab.hazard == "historical")].copy()
    if len(h):
        with np.errstate(divide="ignore", invalid="ignore"):
            h["s_pixel_pct"] = 100 * h.flood_cells / h.n_cells
            h["s_area_pct"] = 100 * h.flood_area_km2 / h.area_km2
            h["area_over_pixel"] = h.s_area_pct / h.s_pixel_pct
        h = h.dropna(subset=["area_over_pixel"]).sort_values("area_over_pixel")
        print("\nArea-weighted / pixel-count flood-zone fraction (major basins, historical hazard)")
        print(f"  Median {h.area_over_pixel.median():.3f}; range {h.area_over_pixel.min():.3f}-{h.area_over_pixel.max():.3f}")
        print("  Largest differences:")
        for r in pd.concat([h.head(5), h.tail(5)]).itertuples():
            print(f"    {r.unit_key}: pixel-based {r.s_pixel_pct:.2f}%, area-based {r.s_area_pct:.2f}% (x{r.area_over_pixel:.3f})")
    print(f"\nOutput: {out / 'flood_area_weighted.csv'} ({time.time() - t0:.0f} s)")
    print("Set AREA_CSV to this output before running flood_concentration_analysis.py.")


if __name__ == "__main__":
    main()
