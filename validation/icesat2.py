"""ICESat-2 (ATL06) vs. TanDEM-X surface elevation per DEM epoch.
"""

from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio

from . import config as cfg
from .common import (
    colocated_aggregate,
    compute_hypsometry,
    convert_gdf_to_egm2008,
    decimal_year_to_timestamp,
    get_tdx_dates,
    is2_quality_columns,
    print_validity_breakdown,
    sample_raster_values,
    sample_slope_from_own_dem,
)


def calculate_statistics(gdf, year_str, is2_col):
    """Summary statistics for TDX-IS2 differences, incl. NMAD (Höhle & Höhle, 2009)."""
    diff = gdf['dh'].values
    nmad = 1.4826 * np.median(np.abs(diff - np.median(diff)))
    sd = {
        'year':        year_str,
        'n_points':    len(gdf),
        'mean_diff':   np.mean(diff),
        'median_diff': np.median(diff),
        'std_diff':    np.std(diff),
        'nmad_diff':   nmad,
        'rmse':        np.sqrt(np.mean(diff**2)),
        'mae':         np.mean(np.abs(diff)),
        'min_diff':    np.min(diff),
        'max_diff':    np.max(diff),
        'pct_5':       np.percentile(diff, 5),
        'pct_95':      np.percentile(diff, 95),
        'mean_tdx':    gdf['tdx_h'].mean(),
        'mean_is2':    gdf[is2_col].mean(),
    }
    print(f"\n  Stats [{year_str}]")
    print(f"    Mean diff (TDX−IS2) : {sd['mean_diff']:+.2f} m")
    print(f"    Median diff         : {sd['median_diff']:+.2f} m")
    print(f"    Std                 : {sd['std_diff']:.2f} m")
    print(f"    NMAD                : {sd['nmad_diff']:.2f} m")
    print(f"    RMSE                : {sd['rmse']:.2f} m")
    print(f"    MAE                 : {sd['mae']:.2f} m")
    print(f"    5th / 95th pct      : {sd['pct_5']:.2f} / {sd['pct_95']:.2f} m")
    return sd


def get_is2_files_for_year(year_str, base_path=None, pattern=None):
    """All ICESat-2 point files for one TanDEM-X period ('2024-2025' ->
    the 2024 AND the 2025 files), found with cfg.IS2_FILE_GLOB below
    cfg.IS2_BASE_PATH. Every match is printed so the run log shows exactly
    which files went in."""
    base_path = Path(cfg.IS2_BASE_PATH if base_path is None else base_path)
    pattern = cfg.IS2_FILE_GLOB if pattern is None else pattern
    sub_years = (year_str.split('-')
                 if isinstance(year_str, str) and '-' in year_str
                 else [str(year_str)])
    files = []
    for y in sub_years:
        hits = sorted(base_path.glob(pattern.format(year=y)))
        if not hits:
            print(f"  WARNING: no ICESat-2 file for {y} in {base_path} "
                  f"(pattern '{pattern.format(year=y)}')")
        for h in hits:
            print(f"  ICESat-2 file for {y}: {h}")
        files.extend(hits)
    # same file must not enter twice (e.g. a name containing both years)
    return list(dict.fromkeys(files))


# ATL06 'delta_time' = seconds since the ATLAS SDP epoch 2018-01-01 00:00:00 (GPS time).
ATLAS_SDP_EPOCH = pd.Timestamp('2018-01-01 00:00:00')


def _is2_timestamps(gdf):
    """Acquisition timestamps of ICESat-2 points. Uses cfg.IS2_TIME_COL (decimal
    year, 't_year') when present; otherwise a 'datetime'/'date'/'time'
    column, then 'dec_year', then ATL06 'delta_time'. Returns (Series of Timestamps, column name) or (None, None)."""
    cols = {c.lower(): c for c in gdf.columns}
    order = [cfg.IS2_TIME_COL.lower()] + [c for c in ('datetime', 'date', 'time', 't_year', 'dec_year')
                                      if c != cfg.IS2_TIME_COL.lower()]
    for name in order:
        if name not in cols:
            continue
        col = cols[name]
        if name in ('t_year', 'dec_year') or pd.api.types.is_numeric_dtype(gdf[col]):
            vals = pd.to_numeric(gdf[col], errors='coerce').values
            if np.nanmedian(vals) < 1900 or np.nanmedian(vals) > 2100:
                print(f"    time column '{col}' is numeric but not a decimal year -- skipped")
                continue
            return (pd.Series(decimal_year_to_timestamp(vals).values, index=gdf.index), col)
        return pd.to_datetime(gdf[col], errors='coerce'), col
    if 'delta_time' in cols:
        dt = pd.to_numeric(gdf[cols['delta_time']], errors='coerce')
        return ATLAS_SDP_EPOCH + pd.to_timedelta(dt, unit='s'), cols['delta_time']
    return None, None


def load_is2_points(year, target_crs, clip_gdf, tdx_dates=None,
                    window_days=None,
                    base_path=None, height_col=None):
    """
    Load the raw (ellipsoidal) ICESat-2 points of one TanDEM-X period,
    reproject them to `target_crs`, clip to `clip_gdf`, keep points within
    +-window_days of a TanDEM-X acquisition and convert the ellipsoidal
    height straight to EGM2008 (column 'h_egm2008'; the same single-step
    route as CryoSat-2's convert_cs2_to_geoid()).

    Shared by process_year() (glacier comparison) and run_rock_comparison()
    so both use identical points. Returns an empty GeoDataFrame if nothing
    is left. Without a usable time column all points are kept (with a warning).
    """
    height_col = cfg.IS2_HEIGHT_COL if height_col is None else height_col
    window_days = cfg.IS2_MAX_TEMPORAL_OFFSET if window_days is None else window_days
    paths = [p for p in get_is2_files_for_year(str(year), base_path=base_path) if p.exists()]
    if not paths:
        return gpd.GeoDataFrame()
    gdfs = [gpd.read_file(p).to_crs(target_crs) for p in paths]
    gdf = pd.concat(gdfs, ignore_index=True) if len(gdfs) > 1 else gdfs[0]
    gdf = gpd.GeoDataFrame(gdf, geometry='geometry', crs=target_crs)
    if height_col not in gdf.columns:
        raise ValueError(f"ICESat-2 files for {year} have no '{height_col}' column "
                         f"(columns: {list(gdf.columns)}). Set cfg.IS2_HEIGHT_COL.")
    if clip_gdf is not None:
        clip_gdf = clip_gdf.to_crs(target_crs) if clip_gdf.crs != target_crs else clip_gdf
        gdf = gpd.clip(gdf, clip_gdf).copy()
    print(f"  ICESat-2 {year}: {len(gdf):,} points inside the clip area")
    if gdf.empty:
        return gdf

    tdx_dates = get_tdx_dates(str(year)) if tdx_dates is None else tdx_dates
    ts, tcol = _is2_timestamps(gdf)
    if ts is None:
        # warn and keep all points
        print(f"  WARNING: no usable time column in the ICESat-2 files for {year} "
              f"(columns: {list(gdf.columns)}) -- proceeding WITHOUT the "
              f"+-{window_days}-day filter.")
        tdx_dates = []
    else:
        gdf['datetime_parsed'] = ts.values
        print(f"  ICESat-2 dates from '{tcol}': {pd.Series(ts).min()} to {pd.Series(ts).max()}")
    if tdx_dates:
        offs = pd.concat([(gdf['datetime_parsed'] - t).abs() for t in tdx_dates],
                         axis=1).min(axis=1)
        gdf['temporal_offset_days'] = offs.dt.days
        inside = gdf['temporal_offset_days'] <= window_days
        months = pd.Series(pd.DatetimeIndex(gdf['datetime_parsed']).month).value_counts().sort_index()
        print(f"  Points per month (from '{tcol}'): "
              + ', '.join(f"{int(m)}: {n:,}" for m, n in months.items()))
        print(f"  {int(inside.sum()):,} of {len(gdf):,} points lie within +-{window_days} days "
              f"of a TanDEM-X date (by '{tcol}')")
        if cfg.IS2_APPLY_TIME_FILTER:
            gdf = gdf[inside].copy()
            print(f"  Temporal filter applied: kept {len(gdf):,} points")
        else:
            print("  Temporal filter NOT applied (cfg.IS2_APPLY_TIME_FILTER = False; files pre-filtered)")
    if gdf.empty:
        return gdf

    # Ellipsoid (WGS84) -> EGM2008, single step (from_grid=None).
    gdf = convert_gdf_to_egm2008(gdf, height_col, from_grid=None,
                                 out_col='h_egm2008', to_grid=cfg.GEOID_GRID_EGM2008)
    off = gdf['h_egm2008_offset'].values
    print(f"  {height_col} (ellipsoidal) -> EGM2008: applied -N = "
          f"{np.nanmin(off):+.2f} to {np.nanmax(off):+.2f} m "
          f"(should be roughly minus the local geoid undulation; ~0 means the grid did not load)")
    if not np.isfinite(off).any() or np.nanmax(np.abs(off)) < 0.5:
        raise RuntimeError("EGM2008 conversion changed the heights by < 0.5 m: "
                           f"check cfg.GEOID_GRID_EGM2008 ({cfg.GEOID_GRID_EGM2008}).")
    return gdf


def process_year(year, roi_gdf, results, use_aggregation=True):
    """
    Comparison for one TanDEM-X epoch: load_is2_points() (ellipsoidal ATL06
    heights -> EGM2008, clipped, time window), TanDEM-X elevation and slope
    at every point, co-located aggregation to the shared grid, then
    dh = TanDEM-X - ICESat-2 per cell.
    """
    print(f"\n{'='*60}")
    print(f"Processing year: {year}")
    print(f"{'='*60}")

    dem_path       = Path(cfg.TDX_DIR) / f"dem_{year}.tif"
    out_folder     = Path(cfg.IS2_OUTPUT_DIR)

    if not dem_path.exists():
        print(f"  Warning: DEM not found at {dem_path}. Skipping.")
        return

    with rasterio.open(dem_path) as src:
        dem_crs = src.crs

    is2_clipped = load_is2_points(year, dem_crs, roi_gdf)
    if is2_clipped.empty:
        print(f"  No ICESat-2 points left for {year}.")
        return
    height_col = 'h_egm2008'

    # ── Sample DEM (+ slope of THIS epoch's DEM) at every raw ICESat-2
    #    point (points already in dem_crs) ──────────────────────────────────
    print(f"  Sampling DEM elevation at {len(is2_clipped):,} ICESat-2 points...")
    is2_clipped = is2_clipped.copy()
    is2_clipped['tdx_h'] = sample_raster_values(is2_clipped, dem_path,
                                                valid_min=-9999, valid_max=10000)
    slope_pts = sample_slope_from_own_dem(is2_clipped, dem_path)
    if slope_pts is not None:
        is2_clipped['tdx_slope'] = slope_pts
    print_validity_breakdown('IS2 (points)', is2_clipped['tdx_h'].values,
                             (is2_clipped['tdx_h'] - is2_clipped[height_col]).values,
                             diff_abs_max=cfg.ABS_DIFF_MAX_M)

    # ── Co-located grid aggregation (same grid, same shared function as the
    #    CS2 pipeline; see AGGREGATION_GRID_SIZE_M for the product-density
    #    rationale) ───────────────────────────────────────────────────────
    if use_aggregation:
        print(f"  Co-located aggregation to a {cfg.AGGREGATION_GRID_SIZE_M:.0f} m grid...")
        quality_cols = is2_quality_columns(is2_clipped.columns)
        extra_cols = ['temporal_offset_days', 'tdx_slope'] + quality_cols
        is2_clipped = colocated_aggregate(
            is2_clipped, sensor_col=height_col, raster_col='tdx_h', label='IS2',
            extra_cols=extra_cols, diff_abs_max=cfg.ABS_DIFF_MAX_M,
            min_points=cfg.AGGREGATION_MIN_POINTS_PER_CELL,
            grid_size=cfg.AGGREGATION_GRID_SIZE_M, origin=cfg.AGGREGATION_ORIGIN_M)
        if is2_clipped.empty:
            print("  No occupied grid cells after aggregation.")
            return

    # Sign convention: TanDEM-X - ICESat-2 (matches the CS2 pipeline's own
    # 'diff' = TanDEM-X - sensor convention, so both sensors are natively
    # comparable without any later sign-flipping). Bounds match the CS2
    # pipeline's own sentinel-value filter (-9999/10000) for consistency.
    is2_clipped['dh'] = is2_clipped['tdx_h'] - is2_clipped[height_col]
    print_validity_breakdown('IS2', is2_clipped['tdx_h'].values, is2_clipped['dh'].values,
                             diff_abs_max=cfg.ABS_DIFF_MAX_M)
    valid = (np.isfinite(is2_clipped['tdx_h']) &
             np.isfinite(is2_clipped['dh'])    &
             (is2_clipped['tdx_h'] > -9999)    &
             (is2_clipped['tdx_h'] < 10000)    &
             (is2_clipped['dh'].abs() < cfg.ABS_DIFF_MAX_M))
    is2_clipped = is2_clipped[valid].copy()
    if is2_clipped.empty:
        print("  No valid comparisons (all nodata/sentinel or |diff| >= 100 m).")
        return

    print(f"  Valid comparisons: {len(is2_clipped):,}")

    # ── Save shapefile ────────────────────────────────────────────────────────
    is2_out = is2_clipped.copy()
    for col in is2_out.columns:
        if pd.api.types.is_datetime64_any_dtype(is2_out[col]):
            is2_out[col] = is2_out[col].dt.strftime('%Y-%m-%dT%H:%M:%S')
    is2_out.to_file(out_folder / f"diff_{year}.shp")
    print(f"  Difference shapefile → diff_{year}.shp")

    # ── Plots ─────────────────────────────────────────────────────────────────

    # ── Hypsometric ───────────────────────────────────────────────────────────
    hyps = compute_hypsometry(
        is2_clipped, elev_col='tdx_h', diff_col='dh',
        bin_width=cfg.HYPS_BIN_WIDTH_M, min_samples=cfg.HYPS_MIN_SAMPLES,
    )
    if not hyps.empty:
        hyps.to_csv(out_folder / f'hypsometric_summary_{year}.csv',
                    index=False, float_format='%.3f')

    # ── Extra per-year diagnostics (temporal bias, quality, elevation boxplot) ─

    # ── Store for the multi-year summary plots + combined grids ──────────────
    stats = calculate_statistics(is2_clipped, str(year), height_col)
    results[str(year)] = {'gdf': is2_clipped, 'stats': stats, 'is2_col': height_col,
                           'dem_path': str(dem_path)}


def run_is2_pipeline(roi_path=cfg.IS2_ROI_PATH,
                      years=cfg.IS2_YEARS,
                      output_dir=cfg.IS2_OUTPUT_DIR,
                      use_aggregation=True):
    """
    Run the full ICESat-2 vs TanDEM-X comparison for every year in `years`.
    Populates the module-level RESULTS / HYPS_ALL dicts (mirroring
    self.comparisons / self.hyps_all in the CS2 pipeline) and returns
    out_folder for convenience.
    """
    # Reset the module-level result dicts before this run.
    results = {}

    if not Path(roi_path).exists():
        print(f"Error: ROI file not found at {roi_path}")
        return {}
    roi_gdf = gpd.read_file(roi_path)
    for year in years:
        try:
            process_year(year, roi_gdf, results, use_aggregation=use_aggregation)
        except Exception as e:
            print(f"  Error processing {year}: {e}")
            raise

    return results
