"""Elevation change (dh) and elevation-change rate (dh/dt) between two epochs.
"""

import os

import numpy as np
import rasterio

from . import config as cfg
from .utils import (
    apply_mask_and_validate,
    format_period_label,
    nan_median_filter,
    parse_period,
    reject_abs_rate,
    remove_seam_ring,
    save_raster,
)


def compute_elevation_change(resA, resB, yA, yB, grid_info, bad_mask, slope_mask=None):
    """
    Compute elevation change (dh), raw dh/dt, and filtered dh/dt between two
    years.
    """
    print("-" * 50)
    print(f"▶ Computing elevation change: {format_period_label(yA, yB)}")

    try:
        with rasterio.open(resA['dem_file']) as src:
            demA = src.read(1).astype(np.float64)
            nodataA = src.nodata
        with rasterio.open(resB['dem_file']) as src:
            demB = src.read(1).astype(np.float64)
            nodataB = src.nodata
        with rasterio.open(resA['date_file']) as src:
            dateA = src.read(1).astype(np.float64)
            nodataDateA = src.nodata
        with rasterio.open(resB['date_file']) as src:
            dateB = src.read(1).astype(np.float64)
            nodataDateB = src.nodata
    except Exception as e:
        print(f"   Error loading files: {e}")
        return None

    if nodataA is not None:
        demA[demA == nodataA] = np.nan
    if nodataB is not None:
        demB[demB == nodataB] = np.nan
    if nodataDateA is not None:
        dateA[dateA == nodataDateA] = np.nan
    if nodataDateB is not None:
        dateB[dateB == nodataDateB] = np.nan

    dh = demB - demA
    dt = dateB - dateA

    yA_parsed = parse_period(yA)
    yB_parsed = parse_period(yB)
    simple_dt = float(yB_parsed[1]) - float(yA_parsed[0])

    # fallback where date difference is missing or too small
    dt = np.where(~np.isfinite(dt) | (np.abs(dt) < 0.1), simple_dt, dt)

    # mask invalid dh locations in dt too
    invalid_input = ~np.isfinite(dh)
    dt[invalid_input] = np.nan

    # raw dh/dt
    with np.errstate(invalid='ignore', divide='ignore'):
        dhdt_raw_observed = np.where(np.abs(dt) > 1e-6, dh / dt, np.nan)

    # reject unrealistic rates (|dh/dt| > cfg.DHDT_MAX_ABS_RATE -> NaN, not clipped)
    data_voids = ~np.isfinite(dhdt_raw_observed)   # real voids, before rejection
    n_rej = int(np.sum(np.isfinite(dhdt_raw_observed)
                       & (np.abs(np.nan_to_num(dhdt_raw_observed)) > cfg.DHDT_MAX_ABS_RATE)))
    dhdt_raw_observed = reject_abs_rate(dhdt_raw_observed)
    print(f"      Rejected {n_rej:,} px with |dh/dt| > {cfg.DHDT_MAX_ABS_RATE:g} m a⁻¹ "
          f"({format_period_label(yA, yB)})")

    # 3x3 NaN-aware median filter (voids ignored, not zero-filled), then
    # remove the one-pixel seam ring bordering data voids / scene edges
    # (localized phase-unwrapping errors).
    valid_mask = np.isfinite(dhdt_raw_observed)
    if np.any(valid_mask):
        dhdt_filtered_3x3 = nan_median_filter(dhdt_raw_observed, size=3)
        dhdt_filtered_3x3 = remove_seam_ring(dhdt_filtered_3x3, cfg.SEAM_RING_REMOVE_PX,
                                             void_mask=data_voids)
    else:
        dhdt_filtered_3x3 = np.full_like(dhdt_raw_observed, np.nan)

    dhdt_filtered_3x3 = reject_abs_rate(dhdt_filtered_3x3)

    # Apply study mask
    dh = apply_mask_and_validate(dh, bad_mask, f"dh {format_period_label(yA, yB)}")
    dhdt_raw_observed = apply_mask_and_validate(dhdt_raw_observed, bad_mask, f"dhdt raw {format_period_label(yA, yB)}")
    dhdt_filtered_3x3 = apply_mask_and_validate(dhdt_filtered_3x3, bad_mask, f"dhdt filtered {format_period_label(yA, yB)}")
    dt = apply_mask_and_validate(dt, bad_mask, f"dt {format_period_label(yA, yB)}")

    # Optional slope exclusion (>45°) to match methods section globally
    if slope_mask is not None:
        slope_excluded = np.isfinite(slope_mask) & (slope_mask > 45.0)
        dh[slope_excluded] = np.nan
        dt[slope_excluded] = np.nan
        dhdt_raw_observed[slope_excluded] = np.nan
        dhdt_filtered_3x3[slope_excluded] = np.nan

    # save rasters with method-consistent names
    dh_out = os.path.join(cfg.OUT_DIR, f"dh_{yB}_minus_{yA}_raw.tif")
    dhdt_raw_out = os.path.join(cfg.OUT_DIR, f"dhdt_raw_observed_{yB}_minus_{yA}.tif")
    dhdt_filt_out = os.path.join(cfg.OUT_DIR, f"dhdt_filtered_3x3_{yB}_minus_{yA}.tif")
    dt_out = os.path.join(cfg.OUT_DIR, f"dt_{yB}_minus_{yA}.tif")

    save_raster(dh.astype(np.float32), dh_out, grid_info,
                units="m", description=f"Raw elevation difference, {yB} minus {yA}")
    save_raster(dhdt_raw_observed.astype(np.float32), dhdt_raw_out, grid_info,
                units="m a-1", description=f"Raw (unfiltered) elevation-change rate, {yB} minus {yA}")
    save_raster(dhdt_filtered_3x3.astype(np.float32), dhdt_filt_out, grid_info,
                units="m a-1", description=f"3x3-median-filtered elevation-change rate, {yB} minus {yA}")
    save_raster(dt.astype(np.float32), dt_out, grid_info,
                units="years", description=f"Per-pixel observation period (decimal years), {yB} minus {yA}")

    print(f"   ✓ Saved: {os.path.basename(dh_out)}")
    print(f"   ✓ Saved: {os.path.basename(dhdt_raw_out)}")
    print(f"   ✓ Saved: {os.path.basename(dhdt_filt_out)}")
    print(f"   ✓ Saved: {os.path.basename(dt_out)}")

    dh_valid = dh[np.isfinite(dh)]
    dhdt_raw_valid = dhdt_raw_observed[np.isfinite(dhdt_raw_observed)]
    dhdt_valid = dhdt_filtered_3x3[np.isfinite(dhdt_filtered_3x3)]

    if len(dh_valid) > 0:
        print("   Statistics for dh:")
        print(f"      Mean: {np.mean(dh_valid):.2f} m, Median: {np.median(dh_valid):.2f} m")

    if len(dhdt_raw_valid) > 0:
        print("   Statistics for raw dhdt:")
        print(f"      Mean: {np.mean(dhdt_raw_valid):.2f} m/yr, Median: {np.median(dhdt_raw_valid):.2f} m/yr")

    if len(dhdt_valid) > 0:
        print("   Statistics for filtered dhdt:")
        print(f"      Mean: {np.mean(dhdt_valid):.2f} m/yr, Median: {np.median(dhdt_valid):.2f} m/yr")

    return {
        'yA': yA,
        'yB': yB,
        'dh': dh,
        'dhdt': dhdt_filtered_3x3,
        'dhdt_filtered_3x3': dhdt_filtered_3x3,    # explicit method key
        'dhdt_raw': dhdt_raw_observed,
        'dhdt_raw_observed': dhdt_raw_observed,    # explicit method key
        'dt': dt,
        'demA': demA,
        'demB': demB,
        'dh_file': dh_out,
        'dhdt_file': dhdt_filt_out,
        'dhdt_raw_file': dhdt_raw_out,
        'dt_file': dt_out
    }


def calc_obs_period_r_style(dt_raster, glacier_mask):
    """
    Match R Stage 3:
    regional, glacier, stable observation-period summaries.
    """
    def summarize(arr):
        vals = arr[np.isfinite(arr)]
        if vals.size == 0:
            return {
                "N_cell": 0,
                "Dt_min": np.nan,
                "Dt_mean": np.nan,
                "Dt_median": np.nan,
                "Dt_max": np.nan
            }
        return {
            "N_cell": int(vals.size),
            "Dt_min": float(np.nanmin(vals)),
            "Dt_mean": float(np.nanmean(vals)),
            "Dt_median": float(np.nanmedian(vals)),
            "Dt_max": float(np.nanmax(vals))
        }

    reg = summarize(dt_raster)
    glc = summarize(np.where(glacier_mask == 1, dt_raster, np.nan))
    stb = summarize(np.where(glacier_mask == 0, dt_raster, np.nan))

    out = {}
    for prefix, d in [("reg", reg), ("glc", glc), ("stb", stb)]:
        for k, v in d.items():
            out[f"{k}_{prefix}"] = v
    return out
