
"""Calculate concentration-zone midpoint and boundary distances.
The midpoint is geometric, not population-weighted; Diffuse has no single midpoint."""

import json as _path_json
from pathlib import Path as _ConfigPath


_DATA_PATHS = _path_json.loads(
    (_ConfigPath(__file__).resolve().parents[1] / "paths.json").read_text(encoding="utf-8")
)
from pathlib import Path
from datetime import datetime
import argparse
import json
import re
import numpy as np
import pandas as pd

RUN_DIR = Path(_DATA_PATHS["HRP_RUN"])
OUT_ROOT = Path(__file__).resolve().parent
N_BANDS = 100
TYPE_CODES = {
    'Near-river': 'near-river', 'River-adjacent': 'near-river',
    'Dispersion': 'dispersion', 'Diffuse': 'dispersion',
    'Far-river': 'far-river', 'River-detached': 'far-river',
}
DISPLAY = {'near-river': 'River-adjacent', 'dispersion': 'Diffuse',
           'far-river': 'River-detached'}


def zone_boundaries(text, r90):
    if pd.isna(text) or not str(text).strip():
        return []
    zones = []
    for token in str(text).split(';'):
        match = re.fullmatch(r'\s*(\d+)\s*-\s*(\d+)\s*', token)
        if not match:
            raise ValueError(f'Invalid zone bands: {text!r}')
        a, b = map(int, match.groups())
        if not 1 <= a <= b <= N_BANDS or (zones and a <= zones[-1]['end_band']):
            raise ValueError(f'Invalid/overlapping zone bands: {text!r}')
        near, far = (a-1)*r90/N_BANDS, b*r90/N_BANDS
        zones.append(dict(start_band=a, end_band=b, near_boundary_km=near,
                          far_boundary_km=far, midpoint_km=(near+far)/2))
    return zones


def distances(code, zones):
    if code == 'dispersion':
        return np.nan, np.nan, 'DIFFUSE_NO_SINGLE_DISTANCE'
    if len(zones) != 1:
        raise ValueError(f'{DISPLAY[code]} must have exactly one concentration zone')
    z = zones[0]
    selected = z['far_boundary_km'] if code == 'near-river' else z['near_boundary_km']
    return z['midpoint_km'], selected, ''


def require_unique(df, keys, name):
    if df[keys].isna().any().any() or df.duplicated(keys).any():
        raise ValueError(f'Missing/duplicate keys in {name}')


def calculate(run, out_root):
    sources = {'static': run/'main/static_hrp.csv',
               'profile_audit': run/'profile_audit.csv',
               'area': run/'hydroclimatic_belts/basin_belts.csv',
               'annual': run/'main/annual_hrp.csv'}
    st = pd.read_csv(sources['static'])
    require_unique(st, ['unit_key'], 'static HRP')
    if st.level.value_counts().to_dict() != {'major': 80, 'sub': 295}:
        raise ValueError('Expected exactly 80 Major + 295 Sub units')
    st['type'] = st.HRP_type.map(TYPE_CODES)
    if st.type.isna().any():
        raise ValueError('Unknown HRP type')
    audit = pd.read_csv(sources['profile_audit'])
    require_unique(audit, ['unit_key', 'set'], 'profile audit')
    st = st.merge(audit[['unit_key', 'set', 'R90_km']], on=['unit_key', 'set'],
                  how='left', validate='one_to_one')
    area = pd.read_csv(sources['area'])
    require_unique(area, ['unit_key'], 'area table')
    if set(area.unit_key) != set(st.unit_key):
        raise ValueError('Area table IDs differ from the current HRP sample')
    check = st[['unit_key', 'level', 'basin_shp']].merge(
        area[['unit_key', 'level', 'basin_shp']], on='unit_key', suffixes=('_s', '_a'))
    if not (check.level_s.eq(check.level_a) & check.basin_shp_s.eq(check.basin_shp_a)).all():
        raise ValueError('Area table does not describe the same basin polygons')
    st = st.merge(area[['unit_key', 'basin_area_km2']], on='unit_key', validate='one_to_one')
    annual = pd.read_csv(sources['annual'])
    pop = annual[annual.year.eq(2020)][['unit_key', 'full_basin_population']]
    require_unique(pop, ['unit_key'], '2020 population')
    st = st.merge(pop.rename(columns={'full_basin_population': 'POP2020'}),
                  on='unit_key', how='left', validate='one_to_one')
    numeric = st[['R90_km', 'basin_area_km2', 'POP2020']].to_numpy(float)
    if not np.isfinite(numeric).all() or (numeric[:, :2] <= 0).any() or (numeric[:, 2] < 0).any():
        raise ValueError('Missing/invalid R90, polygon area or 2020 population')

    records, zone_records = [], []
    for row in st.itertuples(index=False):
        zones = zone_boundaries(row.agglomeration_bands, row.R90_km)
        if len(zones) != row.n_agglomerations:
            raise ValueError(f'Zone count mismatch: {row.unit_key}')
        midpoint, mixed, reason = distances(row.type, zones)
        for i, z in enumerate(zones, 1):
            zone_records.append(dict(unit_key=row.unit_key, level=row.level,
                                     HRP_type=DISPLAY[row.type], zone_number=i, **z))
        z = zones[0] if row.type != 'dispersion' else {}
        records.append(dict(unit_key=row.unit_key, level=row.level,
            HRP_type=DISPLAY[row.type], type=row.type, area=row.basin_area_km2,
            area_km2=row.basin_area_km2, POP2020=row.POP2020, R90_km=row.R90_km,
            n_agglomerations=len(zones), agglomeration_bands=row.agglomeration_bands,
            start_band=z.get('start_band', np.nan), end_band=z.get('end_band', np.nan),
            near_boundary_km=z.get('near_boundary_km', np.nan),
            far_boundary_km=z.get('far_boundary_km', np.nan),
            midpoint_distance_km=midpoint, mixed_boundary_distance_km=mixed,
            mixed_boundary_side={'near-river':'far','far-river':'near','dispersion':'undefined'}[row.type],
            exclusion_reason=reason))
    data = pd.DataFrame(records)
    data['_type'] = data.type.map({'near-river':0,'dispersion':1,'far-river':2})
    data['_level'] = data.level.map({'major':0,'sub':1})
    data = data.sort_values(['_level','_type','unit_key']).drop(columns=['_level','_type']).reset_index(drop=True)
    out = out_root / ('run_' + datetime.now().strftime('%Y%m%d_%H%M%S_%f'))
    out.mkdir(parents=True, exist_ok=False)
    data.to_csv(out/'distance_comparison_all375.csv', index=False, encoding='utf-8-sig')
    pd.DataFrame(zone_records).to_csv(out/'all_zone_boundaries.csv', index=False, encoding='utf-8-sig')
    counts = []
    for method, column in [('method1_midpoint','midpoint_distance_km'),
                           ('method2_mixed_boundary','mixed_boundary_distance_km')]:
        d = data.copy()
        d['distance'] = d[column]
        d['distance_km'] = d[column]
        d['log_eligible'] = np.isfinite(d.distance) & d.distance.gt(0) & d.area.gt(0)
        d.loc[d.distance.eq(0), 'exclusion_reason'] = 'ZERO_DISTANCE'
        d.to_csv(out/f'{method}_all375.csv', index=False, encoding='utf-8-sig')
        for level in ['major','sub']:
            d[d.level.eq(level)].to_csv(out/f'{method}_{level}.csv', index=False, encoding='utf-8-sig')
        for (level, typ), g in d.groupby(['level','HRP_type'], sort=False):
            counts.append(dict(method=method, level=level, HRP_type=typ, n_total=len(g),
                n_defined=int(g.distance.notna().sum()), n_zero=int(g.distance.eq(0).sum()),
                n_log_eligible=int(g.log_eligible.sum()),
                n_log_eligible_ge_1M_km2=int((g.log_eligible & g.area.ge(1_000_000)).sum())))
    pd.DataFrame(counts).to_csv(out/'sample_counts.csv', index=False, encoding='utf-8-sig')
    (out/'config.json').write_text(json.dumps({
        'inputs': {k:str(v) for k,v in sources.items()}, 'n_bands':N_BANDS,
        'method1':'(near_boundary_km + far_boundary_km) / 2',
        'method2':'River-adjacent: far_boundary_km; River-detached: near_boundary_km',
        'midpoint_definition':'Geometric midpoint of the distance interval, not a population-weighted centroid',
        'diffuse':'Undefined single distance; all individual zones recorded separately',
        'scope':'Static classification, 80 Major + 295 Sub; no new classification or regression',
        'type_column':'Legacy plotting code; HRP_type contains the current display names',
        'interpretation':'Method 2 measures the outer reach for River-adjacent and the inner offset for River-detached',
        'zero_handling':'Keep zero; exclude from log distance without adding an epsilon'
    }, ensure_ascii=False, indent=2), encoding='utf-8')
    print(pd.DataFrame(counts).to_string(index=False))
    print(f'Output: {out}')


def self_test():
    z = zone_boundaries('1-5', 100)
    assert distances('near-river', z)[:2] == (2.5, 5)
    z = zone_boundaries('21-30', 100)
    assert distances('far-river', z)[:2] == (25, 20)
    z = zone_boundaries('21-30', 200)
    assert distances('far-river', z)[:2] == (50, 40)
    z = zone_boundaries('1-5;21-30', 100)
    a, b, reason = distances('dispersion', z)
    assert np.isnan(a) and np.isnan(b) and reason
    try:
        zone_boundaries('1-5;5-10', 100)
    except ValueError:
        pass
    else:
        raise AssertionError('Overlapping bands must fail')
    print('PASS: boundary, midpoint, scaling and multi-zone checks; no study data processed.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', type=Path, default=RUN_DIR)
    parser.add_argument('--out-root', type=Path, default=OUT_ROOT)
    parser.add_argument('--self-test', action='store_true')
    args = parser.parse_args()
    self_test() if args.self_test else calculate(args.run_dir, args.out_root)
