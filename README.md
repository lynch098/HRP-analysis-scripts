# Core analysis code

This compact archive contains 14 scripts supporting basin selection, HRP
classification and temporal trends, and flood exposure/concentration analysis.
Figure layout, supplementary figure production, manuscript formatting and
Dataset S1/S2 export scripts are excluded. Numerical outputs are still written
to CSV/XLSX where needed to pass results between analysis steps.

## Calculation order

1. `01_networks/select_hydrobasins_tributaries_v2.py`: HydroBASINS candidates.
2. `01_networks/extract_subbasin_rivers_hydrorivers.py`: network topology and
   final accepted sub-basins. Use its final roster, not the earlier selection
   script's legacy clipping-based acceptance list.
3. `02_hrp/hrp_rerun_v3.py`: distance profiles, effective extent, global
   threshold, static and annual HRP types, classification robustness outputs.
4. `02_hrp/match_hydroclimatic_belts.py`: spatial belt matching.
5. `02_hrp/hrp_downstream_v3.py`: regional assignment, Gini, temporal summaries.
6. `02_hrp/major_pixel_distance_trends.py` and
   `02_hrp/02_rerun_subbasin_riverward.py`: pixel-distance directional trends.
7. `02_hrp/fit_fig1c.py`: Pareto parameter estimation.
8. `02_hrp/calculate_fig1d_distances.py`: concentration-zone midpoint distances;
   pass its `method1_midpoint_all375.csv` to
   `02_hrp/scaling_regressions.py --input <file>`.
9. `02_hrp/fig2b_zone_shares.py`: retained baseline concentration zones and
   their fixed-zone population shares across periods.
10. `03_flood/flood_exposure_v3_3_reviewed.py`: historical and future exposure.
11. `03_flood/flood_area_weighted.py`: area-weighted flood-zone fractions.
12. `03_flood/flood_concentration_analysis.py`: concentration ratios,
    exposure-gap decomposition, future changes, bootstrap/permutation tests
    and population sensitivity.

## Inputs and configuration

Install dependencies from requirements.txt in an environment compatible with
the geospatial libraries. Versions are not pinned: this is a selected source
archive, not a locked execution environment. Raw rasters, geometries, manifests
and intermediate analysis outputs are external inputs and are not bundled.
Edit `paths.json` for data and output locations. `DATA_PATHS.md` lists each
configuration key, the script variables using it, and required file formats.
Paths stored inside input CSV manifests must also be valid on the execution
machine. Keep one consistent basin roster and run version throughout.
The area script imports the exposure module bundled beside it.

The downstream script includes a legacy above-threshold population-share
summary. For the revised concentration-zone comparison use fig2b_zone_shares.py,
not that legacy summary. Directional trends in absolute distance are produced
by the pixel-distance scripts, separately from Gini/profile summaries.

## Version status

The original analysis sources were retained. Packaging did not rerun the study.
The NEGRO boundary correction has a newer HRP run; downstream Gini, directional,
GDP and flood outputs must be checked for consistent use of that correction
before claiming a fully updated reproduction. Do not combine corrected HRP
tables with older raster-derived results without checking their provenance.
Scaling inference retains the original nominal OLS/pairs-bootstrap method;
dependence between nested basins is not corrected in that calculation.

Syntax, path equivalence and computation-structure checks were performed, not an end-to-end execution.
SCRIPT_MANIFEST.csv records script checksums. Documentation and runtime messages
are in English. Calculations, thresholds, tests and cache version identifiers
were retained during language and path-configuration cleanup.
