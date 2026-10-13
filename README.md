# tandemx-glacier-mosaic-mass-balance

Glacier elevation and mass change from TanDEM-X, ICESat-2 and CryoSat-2.

Processing and plotting scripts for the paper

> **Elevation changes of Wordie Ice Shelf tributaries derived from TanDEM-X digital elevation models and satellite altimetry data (2014–2025)**
>
> Antonia Warnstedt¹\*, Christian Sommer¹, Thorsten Seehaus¹, Jose-Luis Bueso-Bello² & Matthias H. Braun¹
>
> ¹ Institute of Geography, Friedrich-Alexander-University Erlangen-Nürnberg
> ² Microwaves and Radar Institute, German Aerospace Center (DLR)
> \* Corresponding author

The code covers four steps:

1. **ICESat-2 filtering.** Cleaning of repeat-track and crossover dh/dt points.
2. **ICESat-2 interpolation.** Gridding of the repeat-track dh/dt.
3. **TanDEM-X.** DEM mosaicking, dh/dt, void filling and mass balance with uncertainty assessment.
4. **Validation.** TanDEM-X elevations and dh/dt compared with CryoSat-2 and ICESat-2.

## Repository structure

```
icesat2/                       ICESat-2 dh/dt (run from inside this folder or with python icesat2/<script>.py)
  config.py                    all settings and paths
  point_filters.py             range filter, local k-NN MAD outlier filter, output writer
  filter_repeat_track.py       cleans repeat-track dh/dt (captoolkit fitsec.py output)
  filter_crossovers.py         cleans crossover dh/dt (captoolkit xover.py output)
  interpolate_dhdt.py          IDW + Gaussian interpolation of the repeat-track dh/dt to a grid
  plot_filter_qc.py            QC maps of kept and removed points

tandemx/                       TanDEM-X package
  config.py                    all settings and paths
  utils.py                     grid, raster I/O, reference-DEM products, period helpers
  inventory.py                 glacier outlines and analysis domains per period
  mosaicking.py                strip mosaicking (deramping, feathering), multi-year merge
  elevation_change.py          dh and dh/dt between epochs
  void_filling.py              hypsometric void filling (elevation-bin means; optional spatial refinement, off by default)
  stable_terrain.py            stable-terrain statistics
  variogram.py                 empirical variogram and decorrelation length
  uncertainty.py               spatially correlated uncertainty (N_eff, Eq. 6 of the paper)
  penetration.py               SAR-penetration ramp and onset detection
  mass_balance.py              volume and mass change per period
  export.py                    summary tables
  state.py                     hand-over of results from processing to plotting
  plotting/                    maps, hypsometry, time series, centerline profiles
run_tandemx_processing.py      TanDEM-X processing chain
run_tandemx_plotting.py        TanDEM-X figures

validation/                    cross-sensor validation package
  config.py                    all settings and paths
  common.py                    geoid conversion, co-located gridding, statistics
  cryosat2.py                  CryoSat-2 vs TanDEM-X elevation
  icesat2.py                   ICESat-2 vs TanDEM-X elevation
  cryosat2_dhdt.py             CryoSat-2 vs TanDEM-X dh/dt
  plot_combined.py             combined CryoSat-2 / ICESat-2 figures
run_validation_processing.py   validation statistics
run_validation_plotting.py     validation figures
```

## Configuration

Every script reads its settings from the `config.py` of its folder, and every setting is documented inline. The code is not tied to a particular region or set of years: all paths are placeholders (`path/to/data`, `path/to/output`), and the study setup is defined in the config files.

Before running, set in

- **`icesat2/config.py`**: input and output paths and the filter thresholds.
- **`tandemx/config.py`**: `DATA_DIR`, `OUTPUT_DIR`, the acquisition years `TDX_YEARS`, optional `MERGED_EPOCHS` (years combined into one epoch, e.g. `[["2016", "2017"]]`), the file-name templates `DEM_DIR_TEMPLATE` and `OUTLINE_TEMPLATE` (`{year}` is replaced by each year), and the glacier inventory. dh/dt is computed for every pair of epochs.
- **`validation/config.py`**: the TanDEM-X acquisition dates `TDX_ACQUISITIONS`, the ICESat-2 epochs `IS2_YEARS` and the input file templates. Epoch pairs and pooled epochs are derived automatically unless set explicitly.

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
python icesat2/interpolate_dhdt.py

# 2. TanDEM-X
python run_tandemx_processing.py          # mosaics, dh/dt, uncertainty, mass balance, tables
python run_tandemx_plotting.py            # figures

# 3. Validation (uses the TanDEM-X mosaics and dh/dt rasters from step 2)
python run_validation_processing.py
python run_validation_plotting.py
```

The processing scripts write all rasters and tables. They also write a state file (`processing_state.pkl` or `validation_state.pkl`, plus `.npz` arrays). The plotting scripts read only that state file, so figures can be redrawn without reprocessing.

## Input data

All data are expected in one projected CRS (`tandemx/config.py: CRS`, default EPSG:3031).

| Data | Format | Config variable |
|---|---|---|
| ICESat-2 repeat-track dh/dt ([captoolkit](https://github.com/nasa-jpl/captoolkit) `fitsec.py`; columns `p1`, `p1_err`) | shapefile | `icesat2/config.py: RT_INPUT` |
| ICESat-2 crossover dh/dt ([captoolkit](https://github.com/nasa-jpl/captoolkit) `xover.py`; columns `dhdt`, `dh_elv`, `dt_year`, `slp_1`, `slp_2`) | shapefile | `XO_INPUT` |
| TanDEM-X DEM strips, co-registered and referenced to EGM2008 (`DEM*.tif`, acquisition date in file name) | GeoTIFF, one folder per year | `tandemx/config.py: TDX_YEARS`, `DEM_DIR_TEMPLATE` |
| Glacier / catchment outlines per year | shapefile | `OUTLINE_TEMPLATE`, `GLACIER_INVENTORY_MAP` |
| Reference DEM (or derived products) | GeoTIFF | `REF_DEM_SOURCE`, `REF_PRODUCTS_DIR` |
| Stable terrain (rock outcrop) | shapefile | `ROCK_OUTCROP_PATH` |
| Centerlines (`Centerline_<Name>_NN.shp`), bed topography, grounding line | shapefile / GeoTIFF | `CENTERLINE_DIR`, `BEDROCK_PATH`, `GROUNDING_LINE_PATH` |
| CryoSat-2 CryoTEMPO-EOLIS points (`cs2_points_<YYYY>`) | GeoPackage / shapefile | `validation/config.py: CS2_SHP_DIR` |
| ICESat-2 ATL06 points (column `h_egm96`), one file per year | shapefile | `IS2_FILE_TEMPLATE` |
| EGM96 / EGM2008 geoid grids | GTX / GeoTIFF | `GEOID_GRID_EGM96`, `GEOID_GRID_EGM2008` |

## Citation

If you use this code, please cite the paper and the archived code:

- Warnstedt, A., Sommer, C., Seehaus, T., Bueso-Bello, J.-L., and Braun, M. H.: Elevation changes of Wordie Ice Shelf tributaries derived from TanDEM-X digital elevation models and satellite altimetry data (2014–2025), *Earth Syst. Sci. Data*, in review.
- Code: https://doi.org/10.5281/zenodo.XXXXXXX
- Dataset: https://doi.org/10.5281/zenodo.20298843

## Methods references

- Braun, M. H., et al. (2019): Constraining glacier elevation and mass changes in South America. *Nat. Clim. Change*, 9, 130–136. https://doi.org/10.1038/s41558-018-0375-7
- Brenner, A. C., et al. (1983): Slope-induced errors in radar altimetry over continental ice sheets. *J. Geophys. Res.*, 88(C3), 1617–1623.
- Dowd, P. A. (1984): The variogram and kriging: robust and resistant estimators. In: *Geostatistics for Natural Resources Characterization*, Springer, 91–106. https://doi.org/10.1007/978-94-009-3699-7_6
- Hugonnet, R., et al. (2022): Uncertainty analysis of digital elevation models by spatial inference from stable terrain. *IEEE J. Sel. Top. Appl. Earth Obs. Remote Sens.*, 15, 6456–6472. https://doi.org/10.1109/JSTARS.2022.3188922
- Huss, M. (2013): Density assumptions for converting geodetic glacier volume change to mass change. *The Cryosphere*, 7, 877–887. https://doi.org/10.5194/tc-7-877-2013
- Malz, P., et al. (2018): Elevation and mass changes of the Southern Patagonia Icefield derived from TanDEM-X and SRTM data. *Remote Sens.*, 10, 188. https://doi.org/10.3390/rs10020188
- McNabb, R., et al. (2019): Sensitivity of glacier volume change estimation to DEM void interpolation. *The Cryosphere*, 13, 895–910. https://doi.org/10.5194/tc-13-895-2019
- Nuth, C. and Kääb, A. (2011): Co-registration and bias corrections of satellite elevation data sets for quantifying glacier thickness change. *The Cryosphere*, 5, 271–290. https://doi.org/10.5194/tc-5-271-2011
- Paolo, F., Nilsson, J., and Gardner, A. (2020): captoolkit: First release (0.1.0). https://doi.org/10.5281/zenodo.3665785
- Seehaus, T., et al. (2019): Changes of the tropical glaciers throughout Peru between 2000 and 2016 – mass balance and area fluctuations. *The Cryosphere*, 13, 2537–2556. https://doi.org/10.5194/tc-13-2537-2019
- Seehaus, T., et al. (2023): Mass changes of the northern Antarctic Peninsula Ice Sheet derived from repeat bi-static synthetic aperture radar acquisitions for the period 2013–2017. *The Cryosphere*, 17, 4629–4644. https://doi.org/10.5194/tc-17-4629-2023

## License

MIT License, see `LICENSE`.
