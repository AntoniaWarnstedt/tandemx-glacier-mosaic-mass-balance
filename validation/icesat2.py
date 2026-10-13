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


def get_is2_files_for_year(year_str):
    files = []
    sub_years = (year_str.split('-')
                 if isinstance(year_str, str) and '-' in year_str
                 else [str(year_str)])
    for y in sub_years:
        files.append(Path(cfg.IS2_FILE_TEMPLATE.format(year=y)))
    return files


def process_year(year, roi_gdf, results, use_aggregation=True):
    """Full comparison pipeline for one TanDEM-X mosaic year (ICESat-2 side)."""
    print(f"\n{'='*60}")
    print(f"Processing year: {year}")
    print(f"{'='*60}")

    dem_path       = Path(cfg.TDX_DIR) / f"dem_{year}.tif"
    is2_paths      = get_is2_files_for_year(str(year))
    out_folder     = Path(cfg.IS2_OUTPUT_DIR)

    if not dem_path.exists():
        print(f"  Warning: DEM not found at {dem_path}. Skipping.")
        return

    existing_is2 = [p for p in is2_paths if p.exists()]
    if not existing_is2:
        print(f"  Warning: No ICESat-2 files found for {year}. Skipping.")
        return

    with rasterio.open(dem_path) as src:
        dem_crs = src.crs

    print("  Loading ICESat-2 data...")
    gdfs = [gpd.read_file(p) for p in existing_is2]
    is2_gdf = pd.concat(gdfs, ignore_index=True) if len(gdfs) > 1 else gdfs[0]

    # ── Reproject to the TanDEM-X raster's CRS (BEFORE aggregation) ────────
    if is2_gdf.crs != dem_crs:
        print(f"  Reprojecting points to {dem_crs}...")
        is2_gdf = is2_gdf.to_crs(dem_crs)
    if roi_gdf.crs != dem_crs:
        roi_gdf = roi_gdf.to_crs(dem_crs)

    print("  Clipping points to ROI...")
    is2_clipped = gpd.clip(is2_gdf, roi_gdf).copy()
    if is2_clipped.empty:
        print(f"  No points found inside ROI for year {year}.")
        return

    # ── Temporal filtering ──────────────────────────────────────────────────
    valid_tdx_dates = get_tdx_dates(str(year))
    date_col = next(
        (c for c in is2_clipped.columns
         if c.lower() in ['datetime', 'date', 'time']), None
    )
    # Fallback: some IS2 shapefiles carry only a decimal-year column (e.g.
    # 't_year', mirroring the CS2 pipeline's own 'dec_year' fallback in
    # load_cs2_shapefiles) instead of a day-level timestamp.
    dec_year_col = None if date_col else next(
        (c for c in ['t_year', 'dec_year'] if c in is2_clipped.columns), None
    )

    if valid_tdx_dates and (date_col or dec_year_col):
        print(f"  Checking points against {len(valid_tdx_dates)} "
              f"TDX acquisition dates...")
        if date_col:
            is2_clipped['datetime_parsed'] = pd.to_datetime(
                is2_clipped[date_col], errors='coerce')
        else:
            is2_clipped['datetime_parsed'] = decimal_year_to_timestamp(
                is2_clipped[dec_year_col].values).values
            print(f"    NOTE: no day-level timestamp column; reconstructed "
                  f"from decimal-year column '{dec_year_col}'.")
        before = len(is2_clipped)
        diffs_list = [(is2_clipped['datetime_parsed'] - t).abs()
                      for t in valid_tdx_dates]
        min_diffs = pd.concat(diffs_list, axis=1).min(axis=1)
        is2_clipped['temporal_offset_days'] = min_diffs.dt.days
        is2_clipped = is2_clipped[
            is2_clipped['temporal_offset_days'] <= cfg.IS2_MAX_TEMPORAL_OFFSET
        ].copy()
        print(f"  Temporal filtering (±{cfg.IS2_MAX_TEMPORAL_OFFSET} days): "
              f"kept {len(is2_clipped):,} of {before:,} points")
    else:
        if not date_col and not dec_year_col:
            print("  WARNING: No 'datetime'/'date'/'time' or decimal-year "
                  "('t_year'/'dec_year') column found in IS2 shapefile. "
                  "Columns actually present:")
            print(f"    {list(is2_clipped.columns)}")
            print("  If one of these is actually the acquisition timestamp "
                  "(e.g. 'delta_time', 'utc_time', 'gps_time'), the ±30-day "
                  "TDX window filter above needs that exact name added to "
                  "the date_col candidate list -- otherwise it silently "
                  "falls through to using the WHOLE year's data unfiltered, "
                  "which also feeds more (and less well time-matched) "
                  "points into the grid aggregation below than intended.")
        else:
            print(f"  WARNING: No known TDX acquisition dates for {year}.")
        print("  Proceeding WITHOUT temporal filtering.")

    if is2_clipped.empty:
        print("  No points remain after temporal filtering.")
        return

    if 'h_egm96' not in is2_clipped.columns:
        raise ValueError(
            f"IS2 shapefile for year {year} has no 'h_egm96' column "
            f"(columns present: {list(is2_clipped.columns)}). Refusing to "
            f"silently fall back to an ellipsoidal height column, since "
            f"that would make this year's TDX-IS2 bias not comparable to "
            f"CS2's geoid-referenced bias. Fix the upstream IS2 "
            f"pre-processing so it always provides 'h_egm96' for this year."
        )

    # ------------------------------------------------------------------
    # The IS2 input files carry EGM96-referenced heights ('h_egm96').
    # ------------------------------------------------------------------
    print("  Converting EGM96 → EGM2008 geoid heights...")
    is2_clipped = convert_gdf_to_egm2008(
        is2_clipped, 'h_egm96', from_grid=cfg.GEOID_GRID_EGM96,
        out_col='h_egm2008', to_grid=cfg.GEOID_GRID_EGM2008)
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
