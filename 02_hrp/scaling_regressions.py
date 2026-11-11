"""Fit ln(distance) against ln(area) using concentration-zone midpoints.
OLS inference and pairs-bootstrap intervals do not correct for nested-basin dependence."""
from pathlib import Path
import argparse
import numpy as np
import pandas as pd
from scipy.stats import linregress
from seaborn.algorithms import bootstrap

SEED = 20261023

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, required=True,
                        help='method1_midpoint_all375.csv from calculate_fig1d_distances.py')
    parser.add_argument('--output-dir', type=Path, default=Path(__file__).parent / 'scaling_results')
    args = parser.parse_args()
    data = pd.read_csv(args.input)
    if data.unit_key.duplicated().any() or data.unit_key.isna().any():
        raise ValueError('Duplicate/missing unit IDs')
    if data.level.value_counts().to_dict() != {'sub': 295, 'major': 80}:
        raise ValueError('Expected 80 major basins and 295 sub-basins')
    if not np.allclose(data.distance, data.midpoint_distance_km, equal_nan=True):
        raise ValueError('Expected the midpoint-distance scheme')
    results, predictions = [], []
    for code, level in [('near-river', 'sub'), ('near-river', 'major'), ('far-river', 'combined')]:
        group = data[data.type.eq(code)]
        if level != 'combined':
            group = group[group.level.eq(level)]
        if len(group) < 3 or not np.isfinite(group[['area', 'distance']]).all().all() or not group[['area', 'distance']].gt(0).all().all():
            raise ValueError(f'Invalid regression inputs: {code}, {level}')
        x, y = np.log(group.area.to_numpy()), np.log(group.distance.to_numpy())
        fit = linregress(x, y)
        grid = np.linspace(x.min(), x.max(), 160)
        prediction = fit.intercept + fit.slope * grid
        low = high = np.full(len(grid), np.nan)
        if not (code == 'near-river' and level == 'major'):
            design = np.column_stack([np.ones(len(x)), x])
            betas = bootstrap(design, y, func=lambda a, b: np.linalg.pinv(a) @ b, n_boot=1000, seed=SEED)
            fitted = betas @ np.vstack([np.ones(len(grid)), grid])
            low, high = np.percentile(fitted, [2.5, 97.5], axis=0)
        results.append(dict(type=code, scope=level, n=len(group), exponent_k=fit.slope,
                            intercept_ln_c=fit.intercept, R2_log=fit.rvalue**2, nominal_OLS_p=fit.pvalue))
        predictions.append(pd.DataFrame(dict(type=code, scope=level, area_km2=np.exp(grid),
            distance_km=np.exp(prediction), lower95_km=np.exp(low), upper95_km=np.exp(high))))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(results).to_csv(args.output_dir / 'scaling_regressions.csv', index=False)
    pd.concat(predictions, ignore_index=True).to_csv(args.output_dir / 'scaling_predictions.csv', index=False)

if __name__ == '__main__':
    main()
