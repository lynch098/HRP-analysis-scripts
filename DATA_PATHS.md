# Data paths

Edit `paths.json` before running. It contains the existing local data locations; no data files are included in this archive. Use forward slashes. Run scripts from any working directory, keeping the package folder structure intact.

## Configuration entries

| Key | Current location | Exists locally |
|---|---|---|
| `HYDRORIVERS` | `D:/workshop/summer/HydroRIVERS_v10_shp/HydroRIVERS_v10_shp/HydroRIVERS_v10.shp` | Yes |
| `HYDROBASINS_DIR` | `D:/workshop/summer/hydrosheds` | Yes |
| `SELECTION_RUN` | `D:/Users/Dell/Documents/river/outputs/hydrobasins_selection_v2_20261017/run_20261017_161116_026461` | Yes |
| `NETWORK_OUTPUT_ROOT` | `D:/Users/Dell/Documents/river/outputs/hydrorivers_outlet_check_20261017` | Yes |
| `MAJOR_MANIFEST` | `D:/workshop/summer/R1D_pixel_riverward_final/00_master_input_resolution.csv` | Yes |
| `SELECTION_OUTPUT_ROOT` | `D:/Users/Dell/Documents/river/outputs/hydrobasins_selection_v2_20261017` | Yes |
| `NETWORK_RUN` | `D:/Users/Dell/Documents/river/outputs/hydrorivers_outlet_check_20261017/rivers_20261017_192038_701131` | Yes |
| `HRP_RUN` | `D:/Users/Dell/Documents/river/outputs/hrp_rerun_v3/run_20261019_143105` | Yes |
| `POP_DIR` | `D:/workshop/summer/GlobPOP` | Yes |
| `SUBBASIN_TREND_OUTPUT_ROOT` | `D:/Users/Dell/Documents/river/outputs/subbasin_gini_riverward_20261019` | Yes |
| `HRP_OUTPUT_ROOT` | `D:/Users/Dell/Documents/river/outputs/hrp_rerun_v3` | Yes |
| `SOUTH_SHP` | `D:/workshop/database/Global South/South.shp` | Yes |
| `BASE` | `D:/workshop/summer` | Yes |
| `BELT_SHP` | `D:/workshop/summer/hess-17-1093-2013-supplement/gis-files/meybeck_et_al_2013_hydrobelts_shp/meybeck_et_al_2013_hydrobelts.shp` | Yes |
| `FLOOD_AREA_OUTPUT_ROOT` | `D:/Users/Dell/Documents/river/outputs/flood_area_weighted_20261028/results` | Yes |
| `LEVELS_CSV` | `D:/Users/Dell/Documents/river/outputs/flood_exposure_v3/run_20261021_105943/levels.csv` | Yes |
| `AREA_CSV` | `D:/Users/Dell/Documents/river/outputs/flood_area_weighted_20261028/results/flood_area_weighted.csv` | Yes |
| `STATIC_CSV` | `D:/Users/Dell/Documents/river/outputs/hrp_rerun_v3/run_20261019_143105/main/static_hrp.csv` | Yes |
| `REGION_CSV` | `D:/Users/Dell/Documents/river/outputs/hrp_rerun_v3/run_20261019_143105/downstream_20261019_165337/regions.csv` | Yes |
| `FLOOD_OUTPUT_ROOT` | `D:/Users/Dell/Documents/river/outputs/flood_exposure_v3` | Yes |
| `SSP_POP_DIR` | `D:/workshop/summer/POP_SSP` | Yes |
| `AQUEDUCT_DIR` | `D:/workshop/summer/flood-data/Floods` | Yes |
| `WATER_JRC_DIR` | `D:/workshop/summer/JRC` | Yes |
| `WATER_RASTER` | `D:/workshop/summer/flood-data/JRC_2020.tif` | Yes |
| `GDP_RASTER` | `D:/workshop/summer/GDP/rast_gdpTot_1990_2020_30arcsec.tif` | Yes |

## Script variables

| Script | Variable | Configuration key |
|---|---|---|
| `01_networks/extract_subbasin_rivers_hydrorivers.py` | `HYDRORIVERS` | `HYDRORIVERS` |
| `01_networks/extract_subbasin_rivers_hydrorivers.py` | `HYDROBASINS_DIR` | `HYDROBASINS_DIR` |
| `01_networks/extract_subbasin_rivers_hydrorivers.py` | `SELECTION_RUN` | `SELECTION_RUN` |
| `01_networks/extract_subbasin_rivers_hydrorivers.py` | `OUTPUT_BASE` | `NETWORK_OUTPUT_ROOT` |
| `01_networks/select_hydrobasins_tributaries_v2.py` | `HYDRO_DIR` | `HYDROBASINS_DIR` |
| `01_networks/select_hydrobasins_tributaries_v2.py` | `MAJOR_MANIFEST` | `MAJOR_MANIFEST` |
| `01_networks/select_hydrobasins_tributaries_v2.py` | `OUTPUT_BASE` | `SELECTION_OUTPUT_ROOT` |
| `02_hrp/02_rerun_subbasin_riverward.py` | `NETWORK_RUN` | `NETWORK_RUN` |
| `02_hrp/02_rerun_subbasin_riverward.py` | `HRP_RUN` | `HRP_RUN` |
| `02_hrp/02_rerun_subbasin_riverward.py` | `POP_DIR` | `POP_DIR` |
| `02_hrp/02_rerun_subbasin_riverward.py` | `OUTPUT_ROOT` | `SUBBASIN_TREND_OUTPUT_ROOT` |
| `02_hrp/calculate_fig1d_distances.py` | `RUN_DIR` | `HRP_RUN` |
| `02_hrp/fig2b_zone_shares.py` | `RUN_DIR` | `HRP_RUN` |
| `02_hrp/fit_fig1c.py` | `RUN_DIR` | `HRP_RUN` |
| `02_hrp/hrp_downstream_v3.py` | `OUTPUT_ROOT` | `HRP_OUTPUT_ROOT` |
| `02_hrp/hrp_downstream_v3.py` | `RUN_DIR` | `HRP_RUN` |
| `02_hrp/hrp_downstream_v3.py` | `SOUTH_SHP` | `SOUTH_SHP` |
| `02_hrp/hrp_rerun_v3.py` | `NETWORK_RUN` | `NETWORK_RUN` |
| `02_hrp/hrp_rerun_v3.py` | `POP_DIR` | `POP_DIR` |
| `02_hrp/hrp_rerun_v3.py` | `OUTPUT_ROOT` | `HRP_OUTPUT_ROOT` |
| `02_hrp/major_pixel_distance_trends.py` | `BASE` | `BASE` |
| `02_hrp/match_hydroclimatic_belts.py` | `RUN_DIR` | `HRP_RUN` |
| `02_hrp/match_hydroclimatic_belts.py` | `BELT_SHP` | `BELT_SHP` |
| `03_flood/flood_area_weighted.py` | `OUT_DIR` | `FLOOD_AREA_OUTPUT_ROOT` |
| `03_flood/flood_concentration_analysis.py` | `LEVELS_CSV` | `LEVELS_CSV` |
| `03_flood/flood_concentration_analysis.py` | `AREA_CSV` | `AREA_CSV` |
| `03_flood/flood_concentration_analysis.py` | `STATIC_CSV` | `STATIC_CSV` |
| `03_flood/flood_concentration_analysis.py` | `REGION_CSV` | `REGION_CSV` |
| `03_flood/flood_exposure_v3_3_reviewed.py` | `HRP_RUN` | `HRP_RUN` |
| `03_flood/flood_exposure_v3_3_reviewed.py` | `SOUTH_SHP` | `SOUTH_SHP` |
| `03_flood/flood_exposure_v3_3_reviewed.py` | `OUTPUT_ROOT` | `FLOOD_OUTPUT_ROOT` |
| `03_flood/flood_exposure_v3_3_reviewed.py` | `GLOBPOP_DIR` | `POP_DIR` |
| `03_flood/flood_exposure_v3_3_reviewed.py` | `SSP_POP_DIR` | `SSP_POP_DIR` |
| `03_flood/flood_exposure_v3_3_reviewed.py` | `AQUEDUCT_DIR` | `AQUEDUCT_DIR` |
| `03_flood/flood_exposure_v3_3_reviewed.py` | `WATER_JRC_DIR` | `WATER_JRC_DIR` |
| `03_flood/flood_exposure_v3_3_reviewed.py` | `WATER_RASTER` | `WATER_RASTER` |
| `03_flood/flood_exposure_v3_3_reviewed.py` | `GDP_RASTER` | `GDP_RASTER` |

## Required files and derived paths

- HydroBASINS: standard v1c shapefiles, including .shp/.shx/.dbf/.prj; HydroRIVERS: global vector network with drainage and connectivity fields.
- Major manifest: `00_master_input_resolution.csv`; preserve geometry/network path columns. Shapefile paths inside CSV manifests must also be updated when transferring data to another computer.
- Network run: `04_subbasin_master.csv` (accepted sub-basins) and `06_major_rule_networks.csv` (original and alternative major networks). Use the final outlet-checked roster.
- HRP run: `main/static_hrp.csv`, `main/annual_hrp.csv`, and `profiles/<set>/<unit>_profiles.csv`. Optional network sensitivity uses `rule_networks/static_hrp.csv`.
- Region table: `regions.csv` from the corresponding downstream run. Major and sub-basin regions are assigned independently from polygon overlap with South.shp.
- GlobPOP: `GlobPOP_Count_30arc_{year}_I32.tiff`, 1990-2020. SSP population: files matching SSP_POP_PATTERN in the exposure script.
- Aqueduct: RP100 historical and GCM/scenario/year depth rasters; retain source filenames or configure HAZARD_MANIFEST. Run the exposure script with `--check` to inspect coverage.
- JRC: annual water-class rasters matched by WATER_JRC_PATTERN; classes 0/1/2/3 mean unobserved/non-water/seasonal/permanent.
- GDP: native-grid total GDP raster; configure GDP_BAND and GDP_MODE in the exposure script.
- Major pixel trends additionally read `dbf/cata-90/cata-90.xlsx` (Sheet4, basin column) under BASE. Basin/river folders under BASE use their configured replacement priority; verify corrected geometries before reusing caches.
- Flood concentration requires `levels.csv` and `flood_area_weighted.csv` from matching masks, units and grids. STATIC_CSV and REGION_CSV must belong to the same study version.
- Effective run folders are timestamped. After generating a new run, update dependent run/table entries together; configuration does not automatically select the newest run.
- Relative output folders based on `__file__` remain beside their scripts. Other output roots are listed above. CLI `--input`, `--run-dir` and `--output-dir`, where supported, override their script defaults.
