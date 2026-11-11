
"""Fit Pareto CDFs to static population profiles.
R-squared describes the CDF fit, not a fit to PDF values."""

import json as _path_json
from pathlib import Path as _ConfigPath


_DATA_PATHS = _path_json.loads(
    (_ConfigPath(__file__).resolve().parents[1] / "paths.json").read_text(encoding="utf-8")
)
from pathlib import Path
from datetime import datetime
import argparse
import json
import numpy as np
import pandas as pd
from scipy.optimize import least_squares

RUN_DIR = Path(_DATA_PATHS["HRP_RUN"])
OUTPUT_ROOT = Path(__file__).resolve().parent
YEARS = list(range(1990, 2021))
X = np.arange(1, 101, dtype=float)

LOW = np.array([1e-4, 1e-4])
HIGH = np.array([1e4, 100.0])


def cdf(x, alpha, xm):
    x = np.asarray(x, dtype=float)
    y = np.zeros_like(x)
    mask = x >= xm
    y[mask] = -np.expm1(alpha * np.log(xm / x[mask]))
    return y


def r_squared(observed, predicted):
    ss = np.sum((observed - observed.mean()) ** 2)
    return float(1 - np.sum((observed - predicted) ** 2) / ss) if ss > 0 else np.nan


def fit_cdf(observed):

    xm_starts = np.unique(np.r_[0.05, 0.5, 2, 5, 10, 20, 40, 60, 80, 95,
                               np.interp([0.1, 0.3, 0.5], observed, X)])
    best = None
    n_success = 0
    errors = []
    for xm0 in xm_starts:
        for alpha0 in (0.3, 1.5, 8.0):
            initial = np.clip([alpha0, xm0], LOW * 1.01, HIGH * 0.999)
            try:
                result = least_squares(
                    lambda z: cdf(X, *np.exp(z)) - observed,
                    np.log(initial), bounds=(np.log(LOW), np.log(HIGH)),
                    max_nfev=2000, ftol=1e-10, xtol=1e-10, gtol=1e-10)
                if not result.success or not np.isfinite(result.fun).all():
                    continue
                n_success += 1
                sse = float(np.dot(result.fun, result.fun))
                if best is None or sse < best[0]:
                    best = (sse, result)
            except (ValueError, FloatingPointError) as exc:
                errors.append(str(exc))
    if best is None:
        raise RuntimeError('No converged fit. ' + '; '.join(errors[:2]))
    params = np.exp(best[1].x)
    at_bound = bool(np.any(params <= LOW * 1.001) or np.any(params >= HIGH * 0.999))
    return params, at_bound, n_success


def load_shares(row, run_dir):
    name = ''.join(c if c.isalnum() or c in '-_.' else '_' for c in row.unit_key)
    path = run_dir / 'profiles' / row.set / f'{name}_profiles.csv'
    d = pd.read_csv(path)
    if set(d.unit_key.astype(str)) != {row.unit_key}:
        raise ValueError(f'Profile identity mismatch: {path}')
    w = d.pivot(index='year', columns='gradient', values='SUM')
    if set(w.index) != set(YEARS) or set(w.columns) != set(range(1, 101)):
        raise ValueError(f'Expected 31 years and 100 bands: {path}')
    pop = w.reindex(index=YEARS, columns=range(1, 101)).to_numpy(float)
    if not np.isfinite(pop).all() or (pop < 0).any() or (pop.sum(axis=1) <= 0).any():
        raise ValueError(f'Invalid population: {path}')
    mean_pop = pop.mean(axis=0)
    return mean_pop / mean_pop.sum()


def main(run_dir):
    units = pd.read_csv(run_dir / 'main/static_hrp.csv')
    if units.unit_key.isna().any() or units.unit_key.duplicated().any():
        raise ValueError('Unit keys are missing or duplicated')
    if units.level.value_counts().to_dict() != {'major': 80, 'sub': 295}:
        raise ValueError('Expected exactly 80 Major and 295 Sub')
    if not units.HRP_type.isin(['Near-river', 'Dispersion', 'Far-river']).all():
        raise ValueError('Unexpected HRP types')
    out = OUTPUT_ROOT / ('run_' + datetime.now().strftime('%Y%m%d_%H%M%S_%f'))
    out.mkdir(parents=True, exist_ok=False)
    config = dict(run_dir=str(run_dir), years=YEARS, n_major=80, n_sub=295,
                  model='ordinary Pareto CDF; unweighted CDF least squares',
                  static_profile='normalize mean annual band populations',
                  x_unit='percent of fixed reference R90',
                  bounds={'alpha': [LOW[0], HIGH[0]], 'xm': [LOW[1], HIGH[1]]},
                  R2='R2_CDF; not PDF R2 or evidence of a power-law distribution',
                  profile_domain='0--100; ordinary Pareto extends beyond this domain',
                  source_classification=str(run_dir/'main/static_hrp.csv'))
    (out/'config.json').write_text(json.dumps(config, indent=2), encoding='utf-8')
    print(f'Output: {out}', flush=True)
    rows, curves = [], []
    for i, r in enumerate(units.sort_values(['level', 'unit_key']).itertuples(index=False), 1):
        record = dict(ZONE_CODE=r.unit_key, unit_key=r.unit_key, level=r.level,
                      type=r.HRP_type.lower(), R2=np.nan, alpha=np.nan, xm=np.nan,
                      R2_CDF=np.nan, R2_band=np.nan, modeled_tail_beyond_100=np.nan,
                      parameter_at_bound=False, n_converged_starts=0,
                      fit_status='FAILED', error='')
        try:
            shares = load_shares(r, run_dir)
            observed = np.cumsum(shares)
            (alpha, xm), at_bound, n_success = fit_cdf(observed)
            predicted = cdf(X, alpha, xm)
            band_predicted = np.diff(np.r_[0.0, predicted])
            r2 = r_squared(observed, predicted)
            record.update(R2=r2, R2_CDF=r2, alpha=float(alpha), xm=float(xm),
                          R2_band=r_squared(shares, band_predicted),
                          RMSE_CDF=float(np.sqrt(np.mean((observed-predicted)**2))),
                          max_CDF_error=float(np.max(np.abs(observed-predicted))),
                          band_L1_error=float(np.abs(shares-band_predicted).sum()),
                          modeled_tail_beyond_100=float(1-predicted[-1]),
                          parameter_at_bound=at_bound, n_converged_starts=n_success,
                          fit_status='REVIEW_BOUND' if at_bound else 'CONVERGED')
            curves.extend(dict(unit_key=r.unit_key, level=r.level, band=int(x),
                               observed_share=float(s), modeled_band_mass=float(b),
                               observed_CDF=float(y), modeled_CDF=float(f))
                          for x,s,b,y,f in zip(X,shares,band_predicted,observed,predicted))
        except (OSError, ValueError, RuntimeError, FloatingPointError) as exc:
            record['error'] = str(exc)
        rows.append(record)

        pd.DataFrame(rows).to_csv(out/'pareto_parameters.csv', index=False, encoding='utf-8-sig')
        print(f'[{i:03d}/375] {r.unit_key}: {record["fit_status"]}; '
              f'R2_CDF={record["R2"]:.4f}, alpha={record["alpha"]:.4f}, xm={record["xm"]:.4f}', flush=True)
    result = pd.DataFrame(rows)
    pd.DataFrame(curves).to_csv(out/'pareto_diagnostics.csv',index=False,encoding='utf-8-sig')
    summary = result.groupby(['level','type']).agg(
        n=('unit_key','size'), n_fitted=('alpha','count'),
        n_failed=('fit_status',lambda s: int(s.eq('FAILED').sum())),
        n_boundary=('fit_status',lambda s: int(s.eq('REVIEW_BOUND').sum())),
        alpha_mean=('alpha','mean'), xm_mean=('xm','mean'),
        CDF_R2_mean=('R2_CDF','mean'), CDF_R2_median=('R2_CDF','median'),
        band_R2_median=('R2_band','median'),
        modeled_tail_mean=('modeled_tail_beyond_100','mean'))
    summary.to_csv(out/'pareto_type_summary.csv',encoding='utf-8-sig')
    result[result.fit_status.ne('CONVERGED')].to_csv(out/'fit_review.csv',index=False,encoding='utf-8-sig')
    print('\nFinished. CONVERGED means optimizer convergence, not an adequate scientific fit.')
    print('Send pareto_parameters.csv, pareto_type_summary.csv and fit_review.csv for review.')
    print(out)


def self_test():
    from scipy.stats import pareto
    for alpha, xm in [(0.7,0.4), (1.3,12.4), (3.2,61.7)]:
        target = pareto.cdf(X, b=alpha, scale=xm)
        assert np.allclose(cdf(X,alpha,xm),target)
        params, bound, count = fit_cdf(target)
        assert np.max(np.abs(cdf(X,*params)-target)) < 1e-6
        assert np.allclose(params,[alpha,xm],rtol=1e-4) and not bound
    assert np.isnan(r_squared(np.ones(100),np.ones(100)))
    print('PASS: formula, parameter recovery (including xm > 1.5), and degenerate R2.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir',type=Path,default=RUN_DIR)
    parser.add_argument('--self-test',action='store_true')
    args = parser.parse_args()
    self_test() if args.self_test else main(args.run_dir)
