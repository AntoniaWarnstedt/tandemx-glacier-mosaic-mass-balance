# tandemx-glacier-mosaic-mass-balance

Glacier elevation and mass change from TanDEM-X, ICESat-2 and CryoSat-2.

Processing and plotting scripts for the paper

> **Elevation changes of Wordie Ice Shelf tributaries derived from TanDEM-X digital elevation models and satellite altimetry data (2014–2025)**
>
> Antonia Warnstedt¹\*, Christian Sommer¹, Thorsten Seehaus¹, José-Luis Bueso-Bello² & Matthias H. Braun¹
>
> ¹ Institut für Geographie, Friedrich-Alexander-Universität Erlangen-Nürnberg, Erlangen, 91058, Germany
> ² Microwaves and Radar Institute, German Aerospace Center (DLR), Wessling, 82234, Germany
> \* Corresponding author

The code covers four steps:

1. **ICESat-2 filtering.** Cleaning of repeat-track and crossover dh/dt points.
2. **ICESat-2 gridding.** Kriging of the repeat-track dh/dt with uncertainty, and dh/dt along the glacier centerlines.
3. **TanDEM-X.** DEM mosaicking, dh/dt, void filling and mass balance with uncertainty assessment.
4. **Validation.** TanDEM-X elevations and dh/dt compared with CryoSat-2 and ICESat-2.

## Repository structure

```
icesat2/                       ICESat-2 dh/dt (run from inside this folder or with python icesat2/<script>.py)
  config.py                    all settings and paths
  point_filters.py             range filter, local k-NN MAD outlier filter, output writer
  filter_repeat_track.py       cleans repeat-track dh/dt (captoolkit fitsec.py output)
  filter_crossovers.py         cleans crossover dh/dt (captoolkit xover.py output)
  plot_filter_qc.py            QC maps of kept and removed points
  kriging.py                   variogram, track-offset adjustment, kriging, cross-validation
  grid_dhdt.py                 dh/dt grid, its uncertainty, catchment mean, crossover comparison
  plot_grid_diagnostics.py     variogram / cross-validation and crossover figures
  centerline_dhdt.py           dh/dt grid sampled along the glacier centerlines
  plot_centerline_dhdt.py      along-flowline dh/dt profiles

tandemx/                       TanDEM-X package
  config.py                    all settings and paths
  utils.py                     grid, raster I/O, reference-DEM products, period helpers
  inventory.py                 glacier outlines and analysis domains per period
  mosaicking.py                strip mosaicking (deramping, feathering), multi-year merge
  elevation_change.py          dh and dh/dt between epochs
  void_filling.py              hypsometric void filling (local + elevation-bin means)
  stable_terrain.py            stable-terrain statistics
  variogram.py                 empirical variogram and decorrelation length
  uncertainty.py               spatially correlated uncertainty (N_eff, Eq. 1)
  penetration.py               SAR-penetration ramp and onset detection
  mass_balance.py              volume and mass change per period
  export.py                    summary tables
  state.py                     hand-over of results from processing to plotting
  plotting/                    summary maps and hypsometry, centerline profiles and heatmap
run_tandemx_processing.py      TanDEM-X processing chain
run_tandemx_plotting.py        TanDEM-X figures

validation/                    cross-sensor validation package
  config.py                    all settings and paths
  common.py                    geoid conversion, co-located gridding, statistics
  cryosat2.py                  CryoSat-2 vs TanDEM-X elevation
  icesat2.py                   ICESat-2 vs TanDEM-X elevation
  cryosat2_dhdt.py             CryoSat-2 vs TanDEM-X dh/dt
  cross_checks.py              ICESat-2 vs CryoSat-2 in the same cells, TanDEM-X vs altimeters on rock
  plot_combined.py             combined CryoSat-2 / ICESat-2 figures
run_validation_processing.py   validation statistics
run_validation_plotting.py     validation figures
```

## Configuration

Every script reads its settings from the `config.py` of its folder, and every setting is documented inline. The code is not tied to a particular region or set of years: all paths are placeholders (`path/to/data`, `path/to/output`), and the study setup is defined in the config files.

Before running, set in

- **`icesat2/config.py`**: input and output paths, the filter thresholds and the kriging settings.
- **`tandemx/config.py`**: `DATA_DIR`, `OUTPUT_DIR`, the acquisition years `TDX_YEARS`, optional `MERGED_EPOCHS` (years combined into one epoch, e.g. `[["2016", "2017"]]`), the file-name templates `DEM_DIR_TEMPLATE` and `OUTLINE_TEMPLATE` (`{year}` is replaced by each year), and the glacier inventory. dh/dt is computed for every pair of epochs.
- **`validation/config.py`**: the TanDEM-X acquisition dates `TDX_ACQUISITIONS`, the ICESat-2 epochs `IS2_YEARS` and the input file patterns. Epoch pairs and pooled epochs are derived automatically unless set explicitly.

Figure settings (axis limits, colours, labels) default to automatic values.

Each config file ends its setup sections with a commented-out **example block** holding the settings used in the paper. Uncomment it to reproduce the paper's setup.

## Installation

```bash
conda env create -f environment.yml && conda activate tandemx-glacier-mosaic-mass-balance
# or
pip install -r requirements.txt
```

`contextily` is optional. It is only needed for the satellite basemap in the overview panels.

## Workflow

```bash
# 1. ICESat-2
python icesat2/filter_repeat_track.py
python icesat2/filter_crossovers.py
python icesat2/plot_filter_qc.py          # optional QC figures
python icesat2/grid_dhdt.py               # kriged dh/dt grid and uncertainty
python icesat2/plot_grid_diagnostics.py
python icesat2/centerline_dhdt.py         # dh/dt along the centerlines
python icesat2/plot_centerline_dhdt.py

# 2. TanDEM-X
python run_tandemx_processing.py          # mosaics, dh/dt, uncertainty, mass balance, tables
python run_tandemx_plotting.py            # figures (see below)

# 3. Validation (uses the TanDEM-X mosaics and dh/dt rasters from step 2)
python run_validation_processing.py
python run_validation_plotting.py
```

The processing scripts write all rasters and tables. They also write a state file (`processing_state.pkl`, `validation_state.pkl` or `<prefix>_plot_state.pkl`, plus `.npz` arrays). The plotting scripts read only that state file, so figures can be redrawn without reprocessing.

### Figures

| Script | Figure |
|---|---|
| `run_tandemx_plotting.py` | `comprehensive_summary_maps_3x3.png`: DEMs, dh/dt and void-filled dh/dt of three epochs / periods |
| | `comprehensive_summary_hypsometry_1x3.png`: catchment and observed area per elevation bin with dh/dt |
| | `centerline_analysis/combined_9panel_dem_and_dhdt_profiles_satellite.png`: satellite map with centerlines, surface elevation and dh/dt profiles |
| | `centerline_analysis/glacier_comparison_heatmap.png`: mean centerline dh/dt per glacier and period |
| `run_validation_plotting.py` | `combined_scatter_landscape.png`, `combined_bias_landscape.png`, `combined_hypsometric_landscape.png`, `combined_elevation_slope_heatmap_5panel_landscape.png` |
| `icesat2/plot_*.py` | filter QC maps, variogram / cross-validation, crossover comparison, centerline dh/dt profiles |

Glaciers are numbered in the alphabetical order of their centerline file names.

## Input data

All data are expected in one projected CRS (`tandemx/config.py: CRS`, default EPSG:3031).

| Data | Format | Config variable |
|---|---|---|
| ICESat-2 repeat-track dh/dt ([captoolkit](https://github.com/fspaolo/captoolkit) `fitsec.py`; columns `p1`, `p1_err`) | shapefile | `icesat2/config.py: RT_INPUT` |
| ICESat-2 crossover dh/dt (captoolkit `xover.py`; columns `dhdt`, `dh_elv`, `dt_year`, `slp_1`, `slp_2`) | shapefile | `XO_INPUT` |
| TanDEM-X DEM strips, co-registered and referenced to EGM2008 (`DEM*.tif`, acquisition date in file name) | GeoTIFF, one folder per year | `tandemx/config.py: TDX_YEARS`, `DEM_DIR_TEMPLATE` |
| Glacier / catchment outlines per year | shapefile | `OUTLINE_TEMPLATE`, `GLACIER_INVENTORY_MAP` |
| Reference DEM (or derived products) | GeoTIFF | `REF_DEM_SOURCE`, `REF_PRODUCTS_DIR` |
| Stable terrain (rock outcrop) | shapefile | `ROCK_OUTCROP_PATH` |
| Centerlines (`Centerline_<Name>_NN.shp`) | shapefile | `tandemx/config.py: CENTERLINE_DIR`, `icesat2/config.py: CL_DIR` |
| CryoSat-2 CryoTEMPO-EOLIS points (`cs2_points_<YYYY>`) | GeoPackage / shapefile | `validation/config.py: CS2_SHP_DIR` |
| ICESat-2 ATL06 points with ellipsoidal height `h_elv` and decimal-year time `t_year` | shapefile | `IS2_BASE_PATH`, `IS2_FILE_GLOB` |
| Rock outcrop (cross-check on stable ground) | shapefile | `validation/config.py: ROCK_OUTCROP_PATH` |
| EGM2008 geoid grid | GeoTIFF | `GEOID_GRID_EGM2008` |

## Methods references

- Brenner, A. C., et al. (1983): Slope-induced errors in radar altimetry over continental ice sheets. *J. Geophys. Res.*, 88(C3), 1617–1623.
- McNabb, R., et al. (2019): Sensitivity of glacier volume change estimation to DEM void interpolation. *The Cryosphere*, 13, 895–910.
- Hugonnet, R., et al. (2022): Uncertainty analysis of digital elevation models by spatial inference from stable terrain. *IEEE J. Sel. Top. Appl. Earth Obs. Remote Sens.*, 15, 6456–6472.
- Seehaus, T., et al. (2023): Mass changes of the northern Antarctic Peninsula Ice Sheet derived from repeat bi-static synthetic aperture radar acquisitions for the period 2013–2017. *The Cryosphere*, 17, 4629–4644.
- Huss, M. (2013): Density assumptions for converting geodetic glacier volume change to mass change. *The Cryosphere*, 7, 877–887.
- Rolstad, C., Haug, T. and Denby, B. (2009): Spatially integrated geodetic glacier mass balance and its uncertainty based on geostatistical analysis: application to the western Svartisen ice cap, Norway. *J. Glaciol.*, 55(192), 666–680.
- Höhle, J. and Höhle, M. (2009): Accuracy assessment of digital elevation models by means of robust statistical methods. *ISPRS J. Photogramm. Remote Sens.*, 64(4), 398–406.

## License

MIT License, see `LICENSE`.
