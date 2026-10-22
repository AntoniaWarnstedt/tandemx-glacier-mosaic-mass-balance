"""Geodetic mass balance per period: on-ice statistics, void filling, stable
terrain, variogram, penetration and the conversion to volume and mass change.
"""

import os

import numpy as np
import pandas as pd

from . import config as cfg
from .elevation_change import calc_obs_period_r_style
from .inventory import (
    compute_area_perimeter_error_fraction_from_geom_r_style,
    compute_inventory_area_metrics_option_c,
    compute_observed_support_areas_km2,
)
from .penetration import compute_sar_penetration_r_style, summarize_penetration_test_coverage
from .stable_terrain import (
    build_paper_stable_terrain_domain,
    build_sigma_map_from_slope_bins,
    build_stable_dhdt_from_yearly_stable_dems,
    build_variogram_terrain_domain,
    compute_stable_slope_error_r_style,
    fit_stable_tilt,
    write_stable_by_strip_table,
)
from .uncertainty import compute_paper_eq1_uncertainty, compute_spatially_averaged_mean_uncertainty
from .utils import (
    canonical_period_key,
    create_coordinate_arrays,
    format_period_label,
    load_rock_mask,
    pixel_area_km2_from_grid,
    rasterize_geometry_to_grid_mask,
    save_raster,
)
from .variogram import estimate_decorrelation_length
from .void_filling import (
    apply_void_fill_and_smooth,
    compute_on_ice_bin_statistics_r_style,
    expand_bin_stats_for_void_fill,
    summarize_regional_on_ice_change_r_style,
)


def compute_mass_balance_for_area_definition(
        area_label,
        area_km2,
        dhdt_mean_myr,
        dhdt_unc_measurement_myr,
        area_fraction_uncertainty,
        density_kgm3,
        density_uncertainty_kgm3=cfg.DENSITY_UNCERTAINTY_KGM3,
        sar_penetration_uncertainty_myr=np.nan,
        observed_time_years=np.nan,
        dhdt_unc_measurement_no_pen_myr=None,
):
    """Compute volume/mass/specific mass balance for one area definition."""
    out = {}
    prefix = area_label

    if not np.isfinite(area_km2) or area_km2 <= 0 or not np.isfinite(dhdt_mean_myr):
        out[f"{prefix}_area_km2"] = area_km2
        out[f"{prefix}_dhdt_myr"] = dhdt_mean_myr
        out[f"{prefix}_dVdt_km3yr"] = np.nan
        out[f"{prefix}_dMdt_Gtyr"] = np.nan
        out[f"{prefix}_specific_mwe_yr"] = np.nan
        out[f"{prefix}_specific_mwe_yr_unc"] = np.nan
        out[f"{prefix}_dhdt_unc_measurement_myr"] = dhdt_unc_measurement_myr
        out[f"{prefix}_dhdt_unc_penetration_myr"] = np.nan
        out[f"{prefix}_dVdt_unc_measurement_km3yr"] = np.nan
        out[f"{prefix}_dVdt_unc_area_km3yr"] = np.nan
        out[f"{prefix}_dVdt_unc_combined_no_pen_km3yr"] = np.nan
        out[f"{prefix}_dVdt_unc_penetration_km3yr"] = np.nan
        out[f"{prefix}_dVdt_unc_combined_with_pen_km3yr"] = np.nan
        out[f"{prefix}_dMdt_unc_measurement_Gtyr"] = np.nan
        out[f"{prefix}_dMdt_unc_area_Gtyr"] = np.nan
        out[f"{prefix}_dMdt_unc_density_Gtyr"] = np.nan
        out[f"{prefix}_dMdt_unc_combined_no_pen_Gtyr"] = np.nan
        out[f"{prefix}_dMdt_unc_penetration_Gtyr"] = np.nan
        out[f"{prefix}_dMdt_unc_combined_with_pen_Gtyr"] = np.nan
        out[f"{prefix}_Total_mass_change_Gt"] = np.nan
        out[f"{prefix}_Total_mass_change_unc_no_pen_Gt"] = np.nan
        out[f"{prefix}_Total_mass_change_unc_with_pen_Gt"] = np.nan
        return out

    area_m2 = area_km2 * cfg.KM2_TO_M2  # km^2 -> m^2

    # dV/dt [km^3 a^-1] = dh/dt [m a^-1] x A [m^2], converted m^3 -> km^3
    dVdt_km3yr = (dhdt_mean_myr * area_m2) * cfg.M3_TO_KM3
    # dM/dt [Gt a^-1] = dV/dt [m^3 a^-1] x rho [kg m^-3], converted kg -> Gt
    # (area_m2 * density_kgm3 has units kg m^-1 a^-1 per m/a of dh/dt; the
    # m^3->kg->Gt chain is folded into a single KG_TO_GT factor here since
    # dhdt_mean_myr * area_m2 is already a volume rate in m^3 a^-1)
    dMdt_Gtyr = (dhdt_mean_myr * area_m2 * density_kgm3) * cfg.KG_TO_GT
    # Specific mass balance [m w.e. a^-1] = dM/dt [Gt a^-1] / A [km^2], with
    # the Gt/km^2 -> m w.e. conversion folded in (1 Gt/km^2 = 1000 m w.e.,
    # since RHO_WATER_KGM3 = 1000 kg/m^3 -- see Hock and others 2026, Section
    # 3.1.1). Algebraically this reduces to dh/dt x (rho/rho_w).
    specific_mwe_yr = dMdt_Gtyr / area_km2 * cfg.RHO_WATER_KGM3

    # Specific Mass Balance Uncertainty
    smb_unc_measurement = abs(dhdt_unc_measurement_myr * (density_kgm3 / cfg.RHO_WATER_KGM3)) if np.isfinite(
        dhdt_unc_measurement_myr) else np.nan
    smb_unc_density = abs(dhdt_mean_myr * (density_uncertainty_kgm3 / cfg.RHO_WATER_KGM3)) if np.isfinite(dhdt_mean_myr) else np.nan

    specific_mwe_yr_unc = np.sqrt(
        (smb_unc_measurement if np.isfinite(smb_unc_measurement) else 0.0) ** 2 +
        (smb_unc_density if np.isfinite(smb_unc_density) else 0.0) ** 2
    )

    dVdt_unc_measurement_km3yr = abs((dhdt_unc_measurement_myr * area_m2) * cfg.M3_TO_KM3) if np.isfinite(
        dhdt_unc_measurement_myr) else np.nan
    dVdt_unc_area_km3yr = abs(dVdt_km3yr * area_fraction_uncertainty) if np.isfinite(
        area_fraction_uncertainty) else np.nan

    dVdt_unc_combined_km3yr = np.sqrt(
        (dVdt_unc_measurement_km3yr if np.isfinite(dVdt_unc_measurement_km3yr) else 0.0) ** 2 +
        (dVdt_unc_area_km3yr if np.isfinite(dVdt_unc_area_km3yr) else 0.0) ** 2
    )

    dMdt_unc_measurement_Gtyr = abs((dhdt_unc_measurement_myr * area_m2 * density_kgm3) * cfg.KG_TO_GT) if np.isfinite(
        dhdt_unc_measurement_myr) else np.nan
    dMdt_unc_area_Gtyr = abs(dMdt_Gtyr * area_fraction_uncertainty) if np.isfinite(
        area_fraction_uncertainty) else np.nan
    dMdt_unc_density_Gtyr = abs(dMdt_Gtyr * (density_uncertainty_kgm3 / density_kgm3)) if np.isfinite(
        density_kgm3) and density_kgm3 > 0 else np.nan

    dMdt_unc_combined_Gtyr = np.sqrt(
        (dMdt_unc_measurement_Gtyr if np.isfinite(dMdt_unc_measurement_Gtyr) else 0.0) ** 2 +
        (dMdt_unc_area_Gtyr if np.isfinite(dMdt_unc_area_Gtyr) else 0.0) ** 2 +
        (dMdt_unc_density_Gtyr if np.isfinite(dMdt_unc_density_Gtyr) else 0.0) ** 2
    )

    total_mass_change_Gt = dMdt_Gtyr * observed_time_years if np.isfinite(observed_time_years) else np.nan
    total_mass_change_unc_Gt = dMdt_unc_combined_Gtyr * observed_time_years if np.isfinite(
        observed_time_years) else np.nan

    if dhdt_unc_measurement_no_pen_myr is not None and np.isfinite(dhdt_unc_measurement_no_pen_myr):
        dVdt_unc_measurement_no_pen_km3yr = abs((dhdt_unc_measurement_no_pen_myr * area_m2) * cfg.M3_TO_KM3)
        dVdt_unc_combined_no_pen_km3yr = np.sqrt(
            dVdt_unc_measurement_no_pen_km3yr ** 2 +
            (dVdt_unc_area_km3yr if np.isfinite(dVdt_unc_area_km3yr) else 0.0) ** 2
        )
        dMdt_unc_measurement_no_pen_Gtyr = abs(
            (dhdt_unc_measurement_no_pen_myr * area_m2 * density_kgm3) * cfg.KG_TO_GT
        )
        dMdt_unc_combined_no_pen_Gtyr = np.sqrt(
            dMdt_unc_measurement_no_pen_Gtyr ** 2 +
            (dMdt_unc_area_Gtyr if np.isfinite(dMdt_unc_area_Gtyr) else 0.0) ** 2 +
            (dMdt_unc_density_Gtyr if np.isfinite(dMdt_unc_density_Gtyr) else 0.0) ** 2
        )
        total_mass_change_unc_no_pen_Gt = (
            dMdt_unc_combined_no_pen_Gtyr * observed_time_years
            if np.isfinite(observed_time_years) else np.nan
        )
    else:
        # no separate no-penetration input: reuse the combined uncertainty
        dVdt_unc_combined_no_pen_km3yr = dVdt_unc_combined_km3yr
        dMdt_unc_combined_no_pen_Gtyr = dMdt_unc_combined_Gtyr
        total_mass_change_unc_no_pen_Gt = total_mass_change_unc_Gt

    out[f"{prefix}_area_km2"] = float(area_km2)
    out[f"{prefix}_dhdt_myr"] = float(dhdt_mean_myr)
    out[f"{prefix}_dVdt_km3yr"] = float(dVdt_km3yr)
    out[f"{prefix}_dMdt_Gtyr"] = float(dMdt_Gtyr)
    out[f"{prefix}_specific_mwe_yr"] = float(specific_mwe_yr)
    out[f"{prefix}_specific_mwe_yr_unc"] = float(specific_mwe_yr_unc) if np.isfinite(specific_mwe_yr_unc) else np.nan

    out[f"{prefix}_dhdt_unc_measurement_myr"] = float(dhdt_unc_measurement_myr) if np.isfinite(
        dhdt_unc_measurement_myr) else np.nan
    out[f"{prefix}_dhdt_unc_penetration_myr"] = np.nan

    out[f"{prefix}_dVdt_unc_measurement_km3yr"] = float(dVdt_unc_measurement_km3yr) if np.isfinite(
        dVdt_unc_measurement_km3yr) else np.nan
    out[f"{prefix}_dVdt_unc_area_km3yr"] = float(dVdt_unc_area_km3yr) if np.isfinite(dVdt_unc_area_km3yr) else np.nan
    out[f"{prefix}_dVdt_unc_combined_no_pen_km3yr"] = float(dVdt_unc_combined_no_pen_km3yr) if np.isfinite(
        dVdt_unc_combined_no_pen_km3yr) else np.nan
    out[f"{prefix}_dVdt_unc_penetration_km3yr"] = np.nan
    out[f"{prefix}_dVdt_unc_combined_with_pen_km3yr"] = float(dVdt_unc_combined_km3yr) if np.isfinite(
        dVdt_unc_combined_km3yr) else np.nan

    out[f"{prefix}_dMdt_unc_measurement_Gtyr"] = float(dMdt_unc_measurement_Gtyr) if np.isfinite(
        dMdt_unc_measurement_Gtyr) else np.nan
    out[f"{prefix}_dMdt_unc_area_Gtyr"] = float(dMdt_unc_area_Gtyr) if np.isfinite(dMdt_unc_area_Gtyr) else np.nan
    out[f"{prefix}_dMdt_unc_density_Gtyr"] = float(dMdt_unc_density_Gtyr) if np.isfinite(
        dMdt_unc_density_Gtyr) else np.nan
    out[f"{prefix}_dMdt_unc_combined_no_pen_Gtyr"] = float(dMdt_unc_combined_no_pen_Gtyr) if np.isfinite(
        dMdt_unc_combined_no_pen_Gtyr) else np.nan
    out[f"{prefix}_dMdt_unc_penetration_Gtyr"] = np.nan
    out[f"{prefix}_dMdt_unc_combined_with_pen_Gtyr"] = float(dMdt_unc_combined_Gtyr) if np.isfinite(
        dMdt_unc_combined_Gtyr) else np.nan

    out[f"{prefix}_Total_mass_change_Gt"] = float(total_mass_change_Gt) if np.isfinite(total_mass_change_Gt) else np.nan
    out[f"{prefix}_Total_mass_change_unc_no_pen_Gt"] = float(total_mass_change_unc_no_pen_Gt) if np.isfinite(
        total_mass_change_unc_no_pen_Gt) else np.nan
    out[f"{prefix}_Total_mass_change_unc_with_pen_Gt"] = float(total_mass_change_unc_Gt) if np.isfinite(
        total_mass_change_unc_Gt) else np.nan

    return out


def calculate_mass_balance_following_r_logic(
    change_dict,
    year_results,
    grid_info,
    preloaded_elev_bins,
    preloaded_ref_dem,
    preloaded_slope,
    preloaded_slope_bins,
    GLACIER_MASK_DIR,
    GLACIER_INVENTORY_MAP,
    DEFAULT_GLACIER_SHP,
    STABLE_TERRAIN_DIR,
    sar_pen_type="ramp",
    sar_pen_bias=2.0,
    density_A=850.0,
    density_B=900.0,
    region_label=None
):
    """See compute_sar_penetration_r_style() and compute_paper_eq1_uncertainty()."""
    yA = change_dict["yA"]
    yB = change_dict["yB"]

    if region_label is None:
        region_label = cfg.REGION_LABEL

    # All mass-balance products are based on the 3x3-median-filtered dh/dt.
    dhdt_input = change_dict["dhdt"].copy()
    dt_raster = change_dict["dt"].copy()

    pixel_area_m2 = grid_info["resolution"] ** 2

    # Load the rock outcrop mask
    rock_mask = load_rock_mask(cfg.ROCK_OUTCROP_PATH, grid_info)

    # ------------------------------------------------------------------
    # Paper-style exclusion of slopes > 45°
    # ------------------------------------------------------------------
    steep_mask = np.isfinite(preloaded_slope) & (preloaded_slope > 45.0)

    # ------------------------------------------------------------------
    # Inventory geometry bookkeeping
    # ------------------------------------------------------------------
    inventory_area_stats = compute_inventory_area_metrics_option_c(
        yA, yB,
        GLACIER_MASK_DIR,
        GLACIER_INVENTORY_MAP,
        DEFAULT_GLACIER_SHP,
        target_metric_crs=cfg.CRS,
        area_mask_option=cfg.AREA_MASK_OPTION,
    )

    start_gdf = inventory_area_stats["start_gdf"]
    end_gdf = inventory_area_stats["end_gdf"]
    start_geom = inventory_area_stats["start_geom"]
    end_geom = inventory_area_stats["end_geom"]
    # The TRUE start/end/mean/union areas are still exported unchanged via
    # inventory_area_stats / the "Inventory_area_*_km2" output fields
    # regardless of this option -- only the analysis domain itself changes.
    union_geom = inventory_area_stats["support_geom"]
    intersection_geom = inventory_area_stats["intersection_geom"]

    obs_area_stats = compute_observed_support_areas_km2(
        dhdt_input=dhdt_input,
        grid_info=grid_info,
        start_geom=start_geom,
        end_geom=end_geom,
        union_geom=union_geom,
        intersection_geom=intersection_geom
    )

    obs_union_mask = obs_area_stats["obs_union_mask"]
    union_inventory_mask = rasterize_geometry_to_grid_mask(union_geom, grid_info, invert=True)

    # This can only shrink union_inventory_mask/obs_union_mask further, never
    # grow it, so it's safe to stack on top of the geometry-based restriction
    # regardless of whatever is or isn't going on with the shapefiles.
    if cfg.AREA_MASK_OPTION == "B":
        demB_arr = change_dict.get("demB")
        if demB_arr is not None:
            demB_valid = np.isfinite(demB_arr)
            n_before = int(np.sum(union_inventory_mask))
            union_inventory_mask = union_inventory_mask & demB_valid
            obs_union_mask = obs_union_mask & demB_valid
            n_after = int(np.sum(union_inventory_mask))
            print(f"      Option B [{yA}-{yB}]: demB-validity AND -- "
                  f"union_inventory_mask {n_before:,} -> {n_after:,} px "
                  f"({n_before - n_after:,} dropped for no finite yB DEM there)")
        else:
            print(f"      ⚠️  Option B [{yA}-{yB}]: change_dict has no 'demB' array; "
                  f"skipping the demB-validity safety net.")

    # Always use the true, untrimmed inventory union here so the void-filled
    # raster keeps covering the same full catchment as boundary_geom,
    # regardless of AREA_MASK_OPTION.
    voi_fill_geom = inventory_area_stats["union_geom"]
    voi_fill_mask = rasterize_geometry_to_grid_mask(voi_fill_geom, grid_info, invert=True)

    # ------------------------------------------------------------------
    # Observation time over observed support
    # ------------------------------------------------------------------
    obs_stats = calc_obs_period_r_style(dt_raster, obs_union_mask.astype(np.uint8))
    obsp_mean_glc = obs_stats["Dt_mean_glc"]
    obsp_medi_glc = obs_stats["Dt_median_glc"]

    # ------------------------------------------------------------------
    # On-ice observed hypsometry from the RAW (uncorrected) dh/dt.
    # ------------------------------------------------------------------
    dhdt_on_ice_obs_raw = np.where(obs_union_mask, dhdt_input, np.nan)

    onice_bins_raw = compute_on_ice_bin_statistics_r_style(
        dhdt_on_ice=dhdt_on_ice_obs_raw,
        elev_bins=preloaded_elev_bins,
        glacier_mask=obs_union_mask.astype(np.uint8)
    )
    onice_summary_raw = summarize_regional_on_ice_change_r_style(onice_bins_raw, pixel_area_m2)

    if onice_summary_raw is None:
        return None

    # Same raster (3x3-filtered dh/dt, before any bias correction), same pixels
    # (obs_union_mask), same elevation bins -- only the minimum pixel count
    # differs.
    onice_bins_detect = compute_on_ice_bin_statistics_r_style(
        dhdt_on_ice=dhdt_on_ice_obs_raw,
        elev_bins=preloaded_elev_bins,
        glacier_mask=obs_union_mask.astype(np.uint8),
        min_pixels=cfg.SAR_PEN_DETECT_MIN_BIN_PIXELS,
    )

    # coverage of the catchment per elevation bin. The break test can only see
    # elevations that were actually observed in THIS period, so report how much
    # of the catchment above SAR_PEN_START_ELEV that is.
    pen_coverage = summarize_penetration_test_coverage(
        elev_bins=preloaded_elev_bins,
        catchment_mask=union_inventory_mask,
        observed_mask=obs_union_mask & np.isfinite(dhdt_input),
        detect_bin_stats=onice_bins_detect,
        pixel_area_km2=pixel_area_m2 * 1e-6,
        band_start_m=cfg.SAR_PEN_START_ELEV,
        min_bin_pixels=cfg.SAR_PEN_DETECT_MIN_BIN_PIXELS,
    )
    try:
        _cov_dir = os.path.join(cfg.OUT_DIR, "sar_penetration")
        os.makedirs(_cov_dir, exist_ok=True)
        _slug = canonical_period_key(format_period_label(yA, yB)).replace('/', '-')
        pen_coverage["table"].to_csv(
            os.path.join(_cov_dir, f"sar_pen_coverage_{_slug}.csv"), index=False)
    except Exception as e:
        print(f"      [sar-pen coverage table not written: {e}]")
    _cs = pen_coverage["summary"]
    print(f"      Break-test coverage [{format_period_label(yA, yB)}]: catchment max bin "
          f"{_cs['catchment_max_bin_m']:.0f} m, highest TESTED bin "
          f"{_cs['max_tested_bin_m']:.0f} m; catchment area >= {cfg.SAR_PEN_START_ELEV} m: "
          f"{_cs['catchment_area_above_start_km2']:.1f} km2, observed there: "
          f"{_cs['observed_area_above_start_km2']:.1f} km2 "
          f"({_cs['observed_pct_above_start']:.1f}%); catchment area ABOVE the highest "
          f"tested bin (never tested): {_cs['catchment_area_above_tested_km2']:.1f} km2")

    glac_mean_raw_obs_uncorrected = onice_summary_raw["mean_m_raw"]

    # Auto-detection of the affected elevation band uses the RAW (uncorrected)
    # on-ice bin stats -- see detect_penetration_break_elevation() and the
    # settings block (SAR_PEN_START_ELEV / SAR_PEN_END_ELEV / SAR_PEN_BIAS).
    sar_out = compute_sar_penetration_r_style(
        ref_dem=preloaded_ref_dem,
        dt_raster=dt_raster,
        glacier_mask=union_inventory_mask.astype(np.uint8),
        pixel_area_m2=pixel_area_m2,
        sar_pen_type=sar_pen_type,
        sar_pen_bias=sar_pen_bias,
        pen_start_elev_m=cfg.SAR_PEN_START_ELEV,
        pen_end_elev_m=cfg.SAR_PEN_END_ELEV,
        onice_bin_stats=onice_bins_detect,  # detection-threshold bins
        pen_detect_min_bin_pixels=cfg.SAR_PEN_DETECT_MIN_BIN_PIXELS,
        apply_bias_correction=cfg.SAR_PEN_APPLY_BIAS_CORRECTION,
        residual_unc_fraction=cfg.SAR_PEN_RESIDUAL_UNC_FRACTION,
        dhdt_on_ice_raw=dhdt_on_ice_obs_raw,
        slope_raster=preloaded_slope,  # pixel-level slope (degrees)
        elev_bins_raster=preloaded_elev_bins,  # matches onice_bins_raw's "cl" categories
        stratified_glacier_mask=obs_union_mask.astype(np.uint8),  # same mask as onice_bins_raw
        require_break=cfg.SAR_PEN_REQUIRE_BREAK,
        break_ref_min_elev_m=cfg.SAR_PEN_BREAK_REF_MIN_ELEV,
        break_sigma_thresh=cfg.SAR_PEN_BREAK_SIGMA_THRESH,
        break_min_consecutive=cfg.SAR_PEN_BREAK_MIN_CONSECUTIVE,
    )
    # Reported as-is for transparency (matches the "SARpen_myr"/"Mean_sar_pen"
    # CSV columns). sar_pen_uncertainty_myr -- the (possibly zero) UNCERTAINTY
    # of that correction, fed into compute_paper_eq1_uncertainty further down.
    sar_out["meta"].update({f"pen_cov_{k}": v for k, v in pen_coverage["summary"].items()})
    sar_pen_bias_myr = sar_out["meta"]["Mean_sar_pen"]
    sar_pen_uncertainty_myr = sar_out["meta"]["Mean_sar_pen_uncertainty_myr"]
    sar_pen_rate_raster = sar_out["sar_pen_rate"]
    print(f"      SAR penetration: source={sar_out['meta']['pen_break_source']}  "
          f"start_elev={sar_out['meta']['pen_start_elev_m']}  "
          f"end_elev={sar_out['meta']['pen_end_elev_m']}  "
          f"mean_pen(bias)={sar_pen_bias_myr:.4f} m/yr  "
          f"mean_pen(uncertainty)={sar_pen_uncertainty_myr:.4f} m/yr  "
          f"correction_applied={sar_out['meta']['pen_correction_applied']}  "
          f"subtract_from_dhdt={cfg.SAR_PEN_APPLY_BIAS_CORRECTION}  "
          f"-- check the diagnostic plot.")


    if cfg.SAR_PEN_APPLY_BIAS_CORRECTION:
        # Subtract the elevation-dependent penetration rate from the dh/dt
        # raster wherever it is defined (zero outside the affected band, or
        # everywhere if no band was set/detected -- in which case this is a no-
        # op and dhdt_input_corr == dhdt_input).
        dhdt_input_corr = dhdt_input.copy()
        pen_valid = np.isfinite(sar_pen_rate_raster)
        dhdt_input_corr[pen_valid] = dhdt_input[pen_valid] - sar_pen_rate_raster[pen_valid]
    else:
        dhdt_input_corr = dhdt_input.copy()

    # ------------------------------------------------------------------
    # This is what feeds the reported central estimates (glac_mean_raw_obs, and
    # via onice_bins, the void-filled glacier-wide raster below).
    # ------------------------------------------------------------------
    dhdt_on_ice_obs = np.where(obs_union_mask, dhdt_input_corr, np.nan)

    onice_bins = compute_on_ice_bin_statistics_r_style(
        dhdt_on_ice=dhdt_on_ice_obs,
        elev_bins=preloaded_elev_bins,
        glacier_mask=obs_union_mask.astype(np.uint8)
    )
    onice_summary = summarize_regional_on_ice_change_r_style(onice_bins, pixel_area_m2)

    if onice_summary is None:
        return None

    glac_mean_raw_obs = onice_summary["mean_m_raw"]
    glac_mean_voi_obs = onice_summary["mean_m_voi"]

    # ------------------------------------------------------------------
    # Area bookkeeping
    # ------------------------------------------------------------------
    observed_union_area_km2 = obs_area_stats["Obs_area_union_km2"]
    inventory_mean_area_km2 = inventory_area_stats["Area_mean_km2"]
    inventory_union_area_km2 = inventory_area_stats["Area_union_km2"]

    coverage_vs_union_pct = (
        100.0 * observed_union_area_km2 / inventory_union_area_km2
        if np.isfinite(inventory_union_area_km2) and inventory_union_area_km2 > 0
        else np.nan
    )

    coverage_vs_mean_pct = (
        100.0 * observed_union_area_km2 / inventory_mean_area_km2
        if np.isfinite(inventory_mean_area_km2) and inventory_mean_area_km2 > 0
        else np.nan
    )

    coverage_fraction_union = (
        observed_union_area_km2 / inventory_union_area_km2
        if np.isfinite(observed_union_area_km2) and np.isfinite(inventory_union_area_km2) and inventory_union_area_km2 > 0
        else np.nan
    )

    # ------------------------------------------------------------------
    # Stable-terrain uncertainty field
    # ------------------------------------------------------------------
    stable_dhdt = build_stable_dhdt_from_yearly_stable_dems(
        yA=yA,
        yB=yB,
        year_results=year_results,
        grid_info=grid_info,
        STABLE_TERRAIN_DIR=STABLE_TERRAIN_DIR
    )

    stable_dhdt_source = ("stable-ground strip mosaics (STABLE_TERRAIN_SOURCE='stb_mosaic')"
                          if cfg.STABLE_TERRAIN_SOURCE == "stb_mosaic"
                          else "yearly stable-ground DEMs (STABLE_TERRAIN_DIR)")
    if stable_dhdt is None and cfg.STABLE_TERRAIN_SOURCE == "stb_mosaic":
        stable_dhdt_source = ("FALLBACK: rock-outcrop pixels of the dh/dt mosaic "
                              "(no stable-ground strip mosaics)")
        print("      ⚠️  Stable-ground strip mosaics missing -- using rock pixels of the dh/dt mosaic.")
        stable_dhdt = np.where(~union_inventory_mask, dhdt_input, np.nan)
    if stable_dhdt is None and cfg.STABLE_TERRAIN_SOURCE == "dem_rock":
        stable_dhdt_source = ("rock-outcrop pixels of the glacier dh/dt mosaic "
                              "(STABLE_TERRAIN_SOURCE='dem_rock')")
        stable_dhdt = np.where(rock_mask, dhdt_input, np.nan).astype(np.float32)
        print(f"      Stable dh/dt from the DEM mosaics' own rock pixels: "
              f"{int(np.isfinite(stable_dhdt).sum()):,} px")
    if stable_dhdt is None:
        if cfg.STABLE_TERRAIN_SOURCE == "mosaic":
            stable_dhdt_source = "rock-outcrop pixels of the dh/dt mosaic (STABLE_TERRAIN_SOURCE='mosaic')"
        else:
            stable_dhdt_source = "FALLBACK: off-glacier pixels of the dh/dt mosaic"
            print(f"      ⚠️  No stable-ground DEMs found for {yA}/{yB} in "
                  f"STABLE_TERRAIN_DIR -- falling back to off-glacier dh/dt.")
        stable_dhdt = np.where(~union_inventory_mask, dhdt_input, np.nan)

    # Exclude slopes > 45° also from stable-terrain uncertainty field
    stable_dhdt[steep_mask] = np.nan

    if cfg.STABLE_TERRAIN_SOURCE == "stb_mosaic":
        try:
            write_stable_by_strip_table(yA, yB, year_results, stable_dhdt, dhdt_input,
                                        rock_mask, cfg.OUT_DIR)
        except Exception as e:
            print(f"      [per-strip stable table skipped: {e}]")

    # Save the stable-terrain dh/dt field (all stable pixels, and the
    # rock-outcrop-only subset) for inspection, e.g. near swath edges.
    try:
        save_raster(np.asarray(stable_dhdt, dtype=np.float32),
                    os.path.join(cfg.OUT_DIR, f"stable_dhdt_{yB}_minus_{yA}.tif"),
                    grid_info, units="m a-1",
                    description=(f"Stable-terrain dh/dt (slopes > 45 deg removed), "
                                 f"{yB} minus {yA}; source: {stable_dhdt_source}"))
        save_raster(np.where(rock_mask, stable_dhdt, np.nan).astype(np.float32),
                    os.path.join(cfg.OUT_DIR, f"stable_dhdt_rock_{yB}_minus_{yA}.tif"),
                    grid_info, units="m a-1",
                    description=(f"Stable-terrain dh/dt on rock outcrop only "
                                 f"(slopes > 45 deg removed), {yB} minus {yA}; "
                                 f"source: {stable_dhdt_source}"))
        print(f"   ✓ Saved: stable_dhdt_{yB}_minus_{yA}.tif, "
              f"stable_dhdt_rock_{yB}_minus_{yA}.tif ({stable_dhdt_source})")
    except Exception as e:
        print(f"      [stable dh/dt raster export skipped: {e}]")

    # ------------------------------------------------------------------
    # Stable-terrain tilt (de-ramping): fit a plane on rock outside the
    # glacier; sigma_stable and the variogram use the detrended residuals,
    # the plane's glacier mean becomes the systematic uncertainty term.
    # ------------------------------------------------------------------
    stable_tilt = None
    if cfg.STABLE_DETREND:
        _glac = union_inventory_mask.astype(bool)
        # dem_rock: the rock mask alone defines stable terrain (nunataks under
        # the catchment outline are kept); otherwise exclude the glacier mask.
        _excl = (np.zeros_like(_glac) if cfg.STABLE_TERRAIN_SOURCE == "dem_rock" else _glac)
        _tilt_fit_mask = rock_mask & ~_excl & np.isfinite(stable_dhdt)
        _tilt_slope_max = np.nan
        if cfg.STABLE_TILT_SLOPE_MAX is not None and preloaded_slope is not None:
            if isinstance(cfg.STABLE_TILT_SLOPE_MAX, str) and cfg.STABLE_TILT_SLOPE_MAX == "glacier_p95":
                _gs = preloaded_slope[_glac & np.isfinite(preloaded_slope)]
                _tilt_slope_max = float(np.percentile(_gs, 95)) if _gs.size else np.nan
            else:
                _tilt_slope_max = float(cfg.STABLE_TILT_SLOPE_MAX)
            if np.isfinite(_tilt_slope_max):
                _tilt_fit_mask &= np.isfinite(preloaded_slope) & (preloaded_slope <= _tilt_slope_max)
                print(f"      Stable tilt fitted on rock with slope <= {_tilt_slope_max:.1f} deg "
                      f"({cfg.STABLE_TILT_SLOPE_MAX}); {int(_tilt_fit_mask.sum()):,} px")
        stable_tilt = fit_stable_tilt(
            stable_dhdt, _tilt_fit_mask, _glac, grid_info,
            n_boot=cfg.STABLE_TILT_BOOT_N, block_m=cfg.STABLE_TILT_BLOCK_M)
        # Detrending of the residuals used for sigma_stable and the variogram
        # uses the plane fitted on ALL rock (the same pixels the variogram is
        # computed on), so no large-scale trend is left in them; the glacier-
        # representative plane above only sets the systematic term.
        _detrend = fit_stable_tilt(
            stable_dhdt, rock_mask & ~_excl & np.isfinite(stable_dhdt), _glac, grid_info,
            n_boot=cfg.STABLE_TILT_BOOT_N, block_m=cfg.STABLE_TILT_BLOCK_M)
        if stable_tilt is not None:
            stable_tilt["slope_max_deg"] = _tilt_slope_max
            _plane = (_detrend if _detrend is not None else stable_tilt)["plane"]
            stable_dhdt = (stable_dhdt - _plane).astype(np.float32)
            if _detrend is not None:
                print(f"      Detrending plane (all rock): E {_detrend['b_per_km'] * 100:+.3f}, "
                      f"N {_detrend['c_per_km'] * 100:+.3f} m/yr per 100 km")
            print(f"      Stable tilt: E {stable_tilt['b_per_km'] * 100:+.3f}, "
                  f"N {stable_tilt['c_per_km'] * 100:+.3f} m/yr per 100 km; "
                  f"glacier mean {stable_tilt['glacier_mean_myr']:+.3f} ± "
                  f"{stable_tilt['glacier_se_myr']:.3f} m/yr "
                  f"({stable_tilt['n_px']:,} px, {stable_tilt['n_blocks']} blocks); "
                  f"NMAD {stable_tilt['nmad_raw']:.2f} -> {stable_tilt['nmad_detrended']:.2f} m/yr")
            try:
                save_raster(np.where(rock_mask, stable_dhdt, np.nan).astype(np.float32),
                            os.path.join(cfg.OUT_DIR, f"stable_dhdt_rock_detrended_{yB}_minus_{yA}.tif"),
                            grid_info, units="m a-1",
                            description=f"Stable-terrain dh/dt on rock, planar tilt removed, {yB} minus {yA}")
            except Exception as e:
                print(f"      [detrended stable raster export skipped: {e}]")

    # ------------------------------------------------------------------
    # TWO stable terrain domains
    # ------------------------------------------------------------------

    # NARROW: catchment (+ STABLE_SIGMA_BUFFER_M) ∩ rock  →  slope-bin σ_stable only
    stable_domain_mask = build_paper_stable_terrain_domain(
        glacier_mask_bool=union_inventory_mask.astype(bool),
        rock_mask_bool=rock_mask,
        grid_info=grid_info,
        buffer_distance_m=cfg.STABLE_SIGMA_BUFFER_M,
        water_mask_bool=None
    )
    dhdt_stable_for_sigma = np.where(stable_domain_mask, stable_dhdt, np.nan)

    # WIDE: all rock outcrop  →  variogram estimation only
    variogram_domain_mask = build_variogram_terrain_domain(
        rock_mask_bool=rock_mask,
        grid_info=grid_info,
        glacier_mask_bool=(None if cfg.STABLE_TERRAIN_SOURCE == "dem_rock"
                           else union_inventory_mask.astype(bool)),
        steep_mask=steep_mask,
    )
    dhdt_stable_for_variogram = np.where(
        variogram_domain_mask & np.isfinite(stable_dhdt),
        stable_dhdt,
        np.nan,
    )

    n_narrow = int(np.sum(np.isfinite(dhdt_stable_for_sigma)))
    n_wide = int(np.sum(np.isfinite(dhdt_stable_for_variogram)))
    print(f"      Stable terrain — σ domain: {n_narrow:,} px | "
          f"variogram domain: {n_wide:,} px")

    # ------------------------------------------------------------------
    # Slope-bin σ_stable  (narrow domain only)
    # ------------------------------------------------------------------
    stable_bins, stable_summary = compute_stable_slope_error_r_style(
        stable_dhdt=dhdt_stable_for_sigma,
        slope_bins=preloaded_slope_bins,
        glacier_mask=union_inventory_mask.astype(np.uint8),
        stable_domain_mask=stable_domain_mask,
        pixel_area_m2=pixel_area_m2,
    )

    if stable_summary is None:
        return None

    # Applied here as a uniform (order-0) shift -- the ramp fit during co-
    # registration already handles spatially-varying tilt; this corrects
    # whatever constant offset is left over on top of that.
    stable_bias_myr = stable_summary.get("mean_fil_all_gw", np.nan)
    if not np.isfinite(stable_bias_myr):
        stable_bias_myr = stable_summary.get("mean_fil_all", np.nan)

    stable_bias_detrended_myr = stable_bias_myr
    stable_tilt_term_myr = np.nan
    if stable_tilt is not None:
        stable_tilt_term_myr = float(np.hypot(stable_tilt["glacier_mean_myr"],
                                              np.nan_to_num(stable_tilt["glacier_se_myr"])))
        print(f"      Stable tilt at glacier (sensitivity): {stable_tilt['glacier_mean_myr']:+.4f} ± "
              f"{stable_tilt['glacier_se_myr']:.4f} m/yr -> {stable_tilt_term_myr:.4f} m/yr "
              f"({'IN budget' if cfg.STABLE_SYSTEMATIC_IN_BUDGET else 'not in budget'})")
    if cfg.STABLE_SYSTEMATIC_IN_BUDGET:
        stable_bias_for_unc = (stable_tilt_term_myr if stable_tilt is not None else stable_bias_myr)
    else:
        stable_bias_for_unc = np.nan          # Eq. 6-10: no bias/tilt term

    apply_stable_bias = bool(cfg.STABLE_BIAS_APPLY_CORRECTION) and np.isfinite(stable_bias_myr)

    print(f"      Stable-terrain residual bias: {stable_bias_myr:.4f} m/yr  "
          f"applied_as_bias_correction={apply_stable_bias}  "
          f"-- ideally ~0; a large residual usually means leftover "
          f"co-registration tilt/offset, not real off-glacier change."
          if np.isfinite(stable_bias_myr) else
          "      Stable-terrain residual bias: n/a (no valid stable-terrain pixels)")

    if apply_stable_bias:
        # Raw observed-support mean (computed earlier, pre-void-fill).
        glac_mean_raw_obs = glac_mean_raw_obs - stable_bias_myr
        glac_mean_voi_obs = glac_mean_voi_obs - stable_bias_myr
        finite_corr = np.isfinite(dhdt_input_corr)
        dhdt_input_corr[finite_corr] = dhdt_input_corr[finite_corr] - stable_bias_myr

        # shift the bin means used for void filling as well
        if onice_bins is not None and not onice_bins.empty:
            for col in ("mean_ch_fil", "medi_ch_fil", "mean_ch_voi", "medi_ch_voi"):
                if col in onice_bins.columns:
                    onice_bins[col] = onice_bins[col] - stable_bias_myr

    # Same naming convention as dhdt_filtered_3x3_{yB}_minus_{yA}.tif, so it
    # can be dropped into the altimetry validation (DHDT_PAIRS). The per-pixel
    # penetration rate that was subtracted is written alongside.
    try:
        _corr_desc = []
        if cfg.SAR_PEN_APPLY_BIAS_CORRECTION and sar_out["meta"].get("pen_correction_applied"):
            _corr_desc.append(
                f"SAR-penetration ramp {sar_out['meta']['pen_start_elev_m']:.0f}-"
                f"{sar_out['meta']['pen_end_elev_m']:.0f} m, max {sar_pen_bias:.1f} m")
        else:
            _corr_desc.append(f"no SAR-penetration correction "
                              f"({sar_out['meta']['pen_break_source']})")
        if apply_stable_bias:
            _corr_desc.append(f"stable-terrain bias {stable_bias_myr:+.4f} m/yr removed")
        else:
            _corr_desc.append("no stable-terrain bias correction")
        dhdt_corr_out = os.path.join(cfg.OUT_DIR, f"dhdt_bias_corrected_{yB}_minus_{yA}.tif")
        save_raster(dhdt_input_corr.astype(np.float32), dhdt_corr_out, grid_info,
                    units="m a-1",
                    description=(f"3x3-median-filtered, bias-corrected elevation-change "
                                 f"rate (not void-filled), {yB} minus {yA}: "
                                 + "; ".join(_corr_desc)))
        print(f"   ✓ Saved: {os.path.basename(dhdt_corr_out)}  ({'; '.join(_corr_desc)})")
        pen_rate_out = os.path.join(cfg.OUT_DIR, f"sar_pen_rate_{yB}_minus_{yA}.tif")
        save_raster(np.asarray(sar_pen_rate_raster, dtype=np.float32), pen_rate_out, grid_info,
                    units="m a-1",
                    description=(f"SAR-penetration correction rate subtracted from dh/dt, "
                                 f"{yB} minus {yA} (all zero if no break was detected)"))
        print(f"   ✓ Saved: {os.path.basename(pen_rate_out)}")
    except Exception as e:
        print(f"      ⚠️  Could not write bias-corrected dh/dt raster: {e}")

    # ------------------------------------------------------------------
    # Glacier bounding box → physical cap for variogram max_lag
    # ------------------------------------------------------------------
    if union_geom is not None and not union_geom.is_empty:
        gb = union_geom.bounds  # (minx, miny, maxx, maxy)
        glc_width_m = float(gb[2] - gb[0])
        glc_height_m = float(gb[3] - gb[1])
        glc_shortest_m = min(glc_width_m, glc_height_m)
        print(f"      Glacier bbox: {glc_width_m / 1000:.0f} × "
              f"{glc_height_m / 1000:.0f} km  →  "
              f"shortest dim = {glc_shortest_m / 1000:.0f} km  →  "
              f"max_lag cap = {glc_shortest_m / 2000:.0f} km")
    else:
        glc_shortest_m = None

    # ------------------------------------------------------------------
    # Variogram estimation (wide domain, adaptive max_lag)
    # ------------------------------------------------------------------
    x_coords, y_coords = create_coordinate_arrays(
        grid_info["shape"], grid_info["transform"]
    )

    if cfg.STANDARDIZE_VARIOGRAM and stable_bins is not None and not stable_bins.empty:
        # Step 1 — build per-pixel σ_dh map from the slope-bin model
        sigma_map = build_sigma_map_from_slope_bins(
            slope_bins_raster=preloaded_slope_bins,
            stable_bin_stats=stable_bins,
        )

        # Restrict sigma_map to the variogram domain only
        sigma_for_variogram = np.where(
            variogram_domain_mask & np.isfinite(sigma_map) & (sigma_map > 0),
            sigma_map,
            np.nan,
        )

        # Step 2 — standardise: z_dh = dh / σ_dh
        # Result is dimensionless with variance ≈ 1 on stable terrain.
        # Clip to ±5 σ to prevent residual outliers dominating the sill.
        z_dh_for_variogram = dhdt_stable_for_variogram / sigma_for_variogram
        z_dh_for_variogram = np.where(
            np.abs(z_dh_for_variogram) <= 5.0,
            z_dh_for_variogram,
            np.nan,
        )

        n_std = int(np.sum(np.isfinite(z_dh_for_variogram)))
        sigma_min = float(np.nanmin(sigma_for_variogram)) if n_std > 0 else np.nan
        sigma_max = float(np.nanmax(sigma_for_variogram)) if n_std > 0 else np.nan
        print(f"      Variogram domain: {n_wide:,} stable-terrain pixels "
              f"({n_wide * pixel_area_km2_from_grid(grid_info):.1f} km²) "
              f"— full rock-outcrop extent")
        print(f"      Standardising z_dh: σ_map range "
              f"{sigma_min:.3f}–{sigma_max:.3f} m/yr  "
              f"({n_std:,} valid z_dh pixels)")

        vgm_input = z_dh_for_variogram

    else:
        # Fallback: raw dh/dt (original behaviour)
        n_wide_px = int(np.sum(np.isfinite(dhdt_stable_for_variogram)))
        print(f"      Variogram domain: {n_wide_px:,} stable-terrain pixels "
              f"({n_wide_px * pixel_area_km2_from_grid(grid_info):.1f} km²) "
              f"— full rock-outcrop extent")
        vgm_input = dhdt_stable_for_variogram

    # Step 3 — run variogram on standardised (or raw) input
    vgm_result = estimate_decorrelation_length(
        vgm_input,  # z_dh or raw dh
        x_coords,
        y_coords,
        sample_size=10_000,
        n_pairs=1_000_000,
        n_lag_bins=35,
        random_seed=42,
        glacier_max_dimension_m=glc_shortest_m,
    )

    dcor_l = vgm_result["decorrelation_length_m"]
    variogram_params = vgm_result["variogram_params"]
    corr_func = vgm_result["correlation_func"]
    dcor_detail = vgm_result["empirical_variogram"]
    ref_dcor_fac = vgm_result["rolstad_equivalent_factor"]
    boundary_hit = vgm_result["boundary_hit"]
    max_lag_used = vgm_result["max_lag_used_m"]

    # Diagnostic: empirical variogram (lag, semivariance, pair count) and
    # the fitted model at the same lags, so the fit can be judged directly.
    try:
        if dcor_detail is not None and len(dcor_detail):
            _v = dcor_detail.copy()
            try:
                _rho = np.asarray([corr_func(float(h)) for h in _v["lags"]], dtype=float)
                _v["model_gamma"] = float(np.sum(variogram_params["psill"].values)) * (1.0 - _rho)
            except Exception:
                pass
            if "long_level_gamma" in vgm_result:
                _v["long_level_gamma"] = vgm_result["long_level_gamma"]
            _v.to_csv(os.path.join(cfg.OUT_DIR, f"variogram_{yB}_minus_{yA}.csv"), index=False)
            print(f"      ✓ Saved: variogram_{yB}_minus_{yA}.csv")
    except Exception as e:
        print(f"      [variogram export skipped: {e}]")

    if boundary_hit:
        print(f"      ⚠️ Variogram boundary hit at {max_lag_used / 1000:.0f} km — "
              f"uncertainty may be underestimated. "
              f"Check for long-wavelength DEM processing artefacts.")

    if cfg.VOID_FILL_SEARCH_PX_MODE == "adaptive" and np.isfinite(dcor_l) and dcor_l > 0:
        spatial_search_px_dynamic = int(np.clip(
            round(dcor_l / grid_info["resolution"]),
            cfg.VOID_FILL_SEARCH_PX_MIN,
            cfg.VOID_FILL_SEARCH_PX_MAX,
        ))
        print(f"      Void-fill search radius: {spatial_search_px_dynamic} px "
              f"({spatial_search_px_dynamic * grid_info['resolution'] / 1000:.2f} km) "
              f"— adaptive, from measured decorrelation length "
              f"{dcor_l / 1000:.1f} km")
    else:
        spatial_search_px_dynamic = cfg.VOID_FILL_SEARCH_PX_FIXED
        print(f"      Void-fill search radius: {spatial_search_px_dynamic} px "
              f"({spatial_search_px_dynamic * grid_info['resolution'] / 1000:.2f} km) "
              f"— fixed fallback (mode={cfg.VOID_FILL_SEARCH_PX_MODE}, "
              f"dcor_l finite={np.isfinite(dcor_l)})")

    # onice_bins itself only has rows for elevation bins present in
    # obs_union_mask (the observed-support footprint); widen it to
    # union_inventory_mask's full elevation range here, void-fill-lookup only,
    # so bins never touched by any strip this period still get a cross-
    # elevation-interpolated fallback instead of leaving a permanent NaN hole.
    if onice_bins is not None and not onice_bins.empty:
        onice_bins_for_void_fill = expand_bin_stats_for_void_fill(
            onice_bins, preloaded_elev_bins, voi_fill_mask
        )
        dhdt_voi_raster = apply_void_fill_and_smooth(
            dhdt_obs          = np.where(voi_fill_mask, dhdt_input_corr, np.nan),
            elev_bins         = preloaded_elev_bins,
            glacier_mask      = voi_fill_mask.astype(int),
            onice_bin_stats   = onice_bins_for_void_fill,
            spatial_search_px = spatial_search_px_dynamic,
            boundary_kernel   = 3,
        )
    else:
        dhdt_voi_raster = np.where(voi_fill_mask, dhdt_input_corr, np.nan).astype(float)

    dhdt_voi_raster[~voi_fill_mask] = np.nan

    # Reported mean over the analysis domain (union_inventory_mask); the
    # void-filled raster itself covers the full catchment.
    glac_mean_voi_union = (
        np.nanmean(dhdt_voi_raster[union_inventory_mask])
        if np.any(np.isfinite(dhdt_voi_raster[union_inventory_mask]))
        else np.nan
    )

    # Cross-check: raster-based void-filled mean (full inventory domain) vs.
    # the R-style hypsometric bin-weighted void-filled mean (observed -support
    # domain only, glac_mean_voi_obs). These use different methods and slightly
    # different domains, so they won't match exactly, but a large gap is worth
    # investigating (bin rejection, sparse coverage, etc).
    if np.isfinite(glac_mean_voi_union) and np.isfinite(glac_mean_voi_obs):
        print(f"      Void-filled mean cross-check: raster/union="
              f"{glac_mean_voi_union:.4f} m/yr  vs  hypsometric/obs-union="
              f"{glac_mean_voi_obs:.4f} m/yr  "
              f"(diff={glac_mean_voi_union - glac_mean_voi_obs:+.4f} m/yr)")

    if (cfg.SAR_PEN_APPLY_BIAS_CORRECTION or apply_stable_bias) and onice_bins_raw is not None and not onice_bins_raw.empty:
        onice_bins_raw_for_void_fill = expand_bin_stats_for_void_fill(
            onice_bins_raw, preloaded_elev_bins, voi_fill_mask
        )
        dhdt_voi_raster_uncorrected = apply_void_fill_and_smooth(
            dhdt_obs          = np.where(voi_fill_mask, dhdt_input, np.nan),
            elev_bins         = preloaded_elev_bins,
            glacier_mask      = voi_fill_mask.astype(int),
            onice_bin_stats   = onice_bins_raw_for_void_fill,
            spatial_search_px = spatial_search_px_dynamic,
            boundary_kernel   = 3,
        )
        dhdt_voi_raster_uncorrected[~voi_fill_mask] = np.nan
        # Same note as glac_mean_voi_union above -- accounting mean stays
        # scoped to union_inventory_mask even though the raster is wider.
        glac_mean_voi_union_uncorrected = (
            np.nanmean(dhdt_voi_raster_uncorrected[union_inventory_mask])
            if np.any(np.isfinite(dhdt_voi_raster_uncorrected[union_inventory_mask]))
            else np.nan
        )
    else:
        glac_mean_voi_union_uncorrected = glac_mean_voi_union

    # SAR penetration: sar_out / sar_pen_bias_myr were already computed earlier
    # (from the RAW, uncorrected bin stats) and, if
    # SAR_PEN_APPLY_BIAS_CORRECTION is True, the BIAS (sar_pen_bias_myr) was
    # already subtracted from the central dh/dt estimates above.

    # ------------------------------------------------------------------
    # PRIMARY uncertainty: Hugonnet Eq. 1
    # ------------------------------------------------------------------
    paper_unc = compute_paper_eq1_uncertainty(
        stable_summary=stable_summary,
        observed_area_km2=observed_union_area_km2,
        variogram_params=variogram_params,
        corr_func=corr_func,
        sar_penetration_uncertainty_myr=sar_pen_uncertainty_myr,
        coverage_fraction=coverage_fraction_union,
        union_geom=union_geom,
        grid_info=grid_info,
        combine_mode=cfg.SAR_PEN_COMBINE_MODE,
        # Uncorrected stable-terrain residual enters the uncertainty;
        # if it was subtracted from dh/dt, it is not counted again.
        stable_bias_myr=(np.nan if apply_stable_bias else stable_bias_for_unc),
    )

    dhdt_unc_measurement_primary_myr = paper_unc["total_no_void_with_pen_myr"]
    dhdt_unc_measurement_primary_with_void_myr = paper_unc["total_with_void_with_pen_myr"]

    print(f"      N_eff ({paper_unc['neff_method']}): {paper_unc['N_eff']:.1f}  "
          f"rolstad-equiv: {paper_unc['rolstad_factor']:.4f}  "
          f"stable-bias unc. term: {paper_unc['stable_bias_term_myr']:.4f} m/yr  "
          f"boundary_hit: {boundary_hit}  "
          f"dcor_l: {dcor_l / 1000:.1f} km")

    # SECONDARY / diagnostic: spatially-averaged mean uncertainty NOTE: this
    # path always uses raw dhdt_stable_for_variogram (not z_dh) because
    # compute_spatially_averaged_mean_uncertainty directly estimates σ_mean
    # from the raster residuals rather than via N_eff.
    spatial_unc = compute_spatially_averaged_mean_uncertainty(
        residual_raster=dhdt_stable_for_variogram,
        support_mask=union_inventory_mask,
        grid_info=grid_info,
        variogram_params=variogram_params,
        corr_func=corr_func,
        robust=True,
    )

    dhdt_unc_measurement_spatialavg_myr = spatial_unc["sigma_mean_myr"]

    if np.isfinite(coverage_fraction_union):
        spatial_void_factor = 1.0 + (1.0 - coverage_fraction_union)  # Braun et al. (2019)
        dhdt_unc_measurement_spatialavg_with_void_myr = (
            dhdt_unc_measurement_spatialavg_myr * spatial_void_factor
            if np.isfinite(dhdt_unc_measurement_spatialavg_myr) else np.nan
        )
    else:
        dhdt_unc_measurement_spatialavg_with_void_myr = \
            dhdt_unc_measurement_spatialavg_myr

    # ------------------------------------------------------------------

    # ------------------------------------------------------------------
    # Area uncertainty
    # ------------------------------------------------------------------
    glac_area_error_frac, ref_glc_area, ref_glc_peri = compute_area_perimeter_error_fraction_from_geom_r_style(
        union_geom
    )

    # ------------------------------------------------------------------
    # Products for multiple area definitions
    # Use PAPER Eq.1 uncertainty as primary exported uncertainty
    # ------------------------------------------------------------------
    density_uncertainty = 60.0

    area_products_A = {}
    area_products_B = {}

    inventory_support_area_km2 = inventory_area_stats["support_area_km2"]
    inventory_support_label = f"inventory_{inventory_area_stats['support_label']}"

    for area_label, area_km2, dhdt_mean in [
        ("obs_union", observed_union_area_km2, glac_mean_raw_obs),
        ("inventory_mean", inventory_mean_area_km2, glac_mean_voi_union),
        (inventory_support_label, inventory_support_area_km2, glac_mean_voi_union),
    ]:
        area_products_A.update(
            compute_mass_balance_for_area_definition(
                area_label=area_label,
                area_km2=area_km2,
                dhdt_mean_myr=dhdt_mean,
                dhdt_unc_measurement_myr=dhdt_unc_measurement_primary_with_void_myr,
                area_fraction_uncertainty=glac_area_error_frac,
                density_kgm3=density_A,
                density_uncertainty_kgm3=density_uncertainty,
                sar_penetration_uncertainty_myr=np.nan,
                observed_time_years=obsp_mean_glc,
                dhdt_unc_measurement_no_pen_myr=paper_unc["measurement_with_void_no_pen_myr"],
            )
        )

        tmp_B = compute_mass_balance_for_area_definition(
            area_label=area_label,
            area_km2=area_km2,
            dhdt_mean_myr=dhdt_mean,
            dhdt_unc_measurement_myr=dhdt_unc_measurement_primary_with_void_myr,
            area_fraction_uncertainty=glac_area_error_frac,
            density_kgm3=density_B,
            density_uncertainty_kgm3=density_uncertainty,
            sar_penetration_uncertainty_myr=np.nan,
            observed_time_years=obsp_mean_glc,
            dhdt_unc_measurement_no_pen_myr=paper_unc["measurement_with_void_no_pen_myr"],
        )

        for k, v in tmp_B.items():
            area_products_B[f"{k}_density900"] = v

    raw_rates = {
        "Region": region_label,
        "Inventory_Period": yA,
        "Obs_mean": obsp_mean_glc,
        "Obs_median": obsp_medi_glc,
        "MB_area_used_km2": observed_union_area_km2,
        "MB_area_coverage_pct": coverage_vs_union_pct,
        "dhdt_ma": glac_mean_raw_obs,
        "delt_dhdt_measurement": dhdt_unc_measurement_primary_myr,
        "delt_dhdt_measurement_with_void": dhdt_unc_measurement_primary_with_void_myr,
        "delt_dhdt_measurement_spatialavg": dhdt_unc_measurement_spatialavg_myr,
        "delt_dhdt_measurement_spatialavg_with_void": dhdt_unc_measurement_spatialavg_with_void_myr,
        "delt_dhdt_penetration": sar_pen_uncertainty_myr,
        "dVdt_km3a": area_products_A.get("obs_union_dVdt_km3yr"),
        "dMdt_A_gta": area_products_A.get("obs_union_dMdt_Gtyr"),
        "dMdt_B_gta": area_products_B.get("obs_union_dMdt_Gtyr_density900"),
        "sMdt_A_mwea": area_products_A.get("obs_union_specific_mwe_yr"),
        "sMdt_B_mwea": area_products_B.get("obs_union_specific_mwe_yr_density900"),
    }

    void_rates = {
        "Region": region_label,
        "Inventory_Period": yA,
        "Obs_mean": obsp_mean_glc,
        "Obs_median": obsp_medi_glc,
        "MB_area_used_km2": inventory_mean_area_km2,
        "MB_area_coverage_pct": coverage_vs_mean_pct,
        "dhdt_ma": glac_mean_voi_union,
        "delt_dhdt_measurement": dhdt_unc_measurement_primary_myr,
        "delt_dhdt_measurement_with_void": dhdt_unc_measurement_primary_with_void_myr,
        "delt_dhdt_measurement_spatialavg": dhdt_unc_measurement_spatialavg_myr,
        "delt_dhdt_measurement_spatialavg_with_void": dhdt_unc_measurement_spatialavg_with_void_myr,
        "delt_dhdt_penetration": sar_pen_uncertainty_myr,
        "dVdt_km3a": area_products_A.get("inventory_mean_dVdt_km3yr"),
        "dMdt_A_gta": area_products_A.get("inventory_mean_dMdt_Gtyr"),
        "dMdt_B_gta": area_products_B.get("inventory_mean_dMdt_Gtyr_density900"),
        "sMdt_A_mwea": area_products_A.get("inventory_mean_specific_mwe_yr"),
        "sMdt_B_mwea": area_products_B.get("inventory_mean_specific_mwe_yr_density900"),
    }

    void_total = {
        "Region": region_label,
        "Inventory_Period": yA,
        "Obs_mean": obsp_mean_glc,
        "MB_area_used_km2": inventory_mean_area_km2,
        "MB_area_coverage_pct": coverage_vs_mean_pct,
        "dM_A_gta": area_products_A.get("inventory_mean_Total_mass_change_Gt"),
        "delt_dM_A_no_pen": area_products_A.get("inventory_mean_Total_mass_change_unc_no_pen_Gt"),
        "delt_dM_A_with_pen": area_products_A.get("inventory_mean_Total_mass_change_unc_with_pen_Gt"),
        "dM_B_gta": area_products_B.get("inventory_mean_Total_mass_change_Gt_density900"),
        "delt_dM_B_no_pen": area_products_B.get("inventory_mean_Total_mass_change_unc_no_pen_Gt_density900"),
        "delt_dM_B_with_pen": area_products_B.get("inventory_mean_Total_mass_change_unc_with_pen_Gt_density900"),
    }


    # ------------------------------------------------------------------
    # Reference hypsometry over the FULL catchment (for area-per-bin plots)
    # ------------------------------------------------------------------
    _ref_mask = np.asarray(voi_fill_mask, dtype=bool) & np.isfinite(preloaded_elev_bins)
    _ref_cl, _ref_n = np.unique(preloaded_elev_bins[_ref_mask], return_counts=True)
    _ref_hyps_bins = pd.DataFrame({
        "cl": _ref_cl.astype(float),
        "n_px": _ref_n.astype(int),
        "area_km2": _ref_n.astype(float) * float(pixel_area_m2) * cfg.M2_TO_KM2,
    })
    return {
        "Period": format_period_label(yA, yB),
        "yA": yA,
        "yB": yB,
        "Region": region_label,

        "inventory_area_stats": {
            "Inventory_area_start_km2": inventory_area_stats["Area_start_km2"],
            "Inventory_area_end_km2": inventory_area_stats["Area_end_km2"],
            "Inventory_area_mean_km2": inventory_area_stats["Area_mean_km2"],
            "Inventory_area_union_km2": inventory_area_stats["Area_union_km2"],
            "Inventory_area_intersection_km2": inventory_area_stats["Area_intersection_km2"],
        },
        "obs_area_stats": {
            "Observed_inventory_area_start_km2": obs_area_stats["Obs_area_start_km2"],
            "Observed_inventory_area_end_km2": obs_area_stats["Obs_area_end_km2"],
            "Observed_inventory_area_union_km2": obs_area_stats["Obs_area_union_km2"],
            "Observed_inventory_area_intersection_km2": obs_area_stats["Obs_area_intersection_km2"],
        },

        "coverage_stats": {
            "coverage_vs_inventory_union_pct": coverage_vs_union_pct,
            "coverage_vs_inventory_mean_pct": coverage_vs_mean_pct,
            "coverage_fraction_union": coverage_fraction_union,
            "void_fraction_union": paper_unc.get("void_fraction"),
            "void_factor_union": paper_unc.get("void_factor"),
        },

        "obs_stats": obs_stats,
        "onice_bin_stats": onice_bins,
        # True reference hypsometry: every catchment pixel (full, untrimmed
        # inventory union = voi_fill_mask) per elevation bin, independent of
        # dh/dt coverage. onice_bins["n_ref"] is NOT this -- it is counted
        # inside obs_union_mask, i.e. it is the OBSERVED footprint.
        "ref_hyps_bins": _ref_hyps_bins,
        "stable_slope_bin_stats": stable_bins,

        "paper_uncertainty_stats": paper_unc,
        "spatial_uncertainty_stats": spatial_unc,
        "sar_pen_meta": sar_out["meta"],
        "sar_pen_detect_bins": onice_bins_detect,
        "sar_pen_coverage_table": pen_coverage["table"],
        "sar_pen_settings": {
            "sar_pen_bias": sar_pen_bias, "sar_pen_type": sar_pen_type,
            "core_max_elev": cfg.SAR_PEN_START_ELEV,
            "core_min_elev": cfg.SAR_PEN_BREAK_REF_MIN_ELEV if cfg.SAR_PEN_START_ELEV is not None else None,
        },

        "dcor_length_m": dcor_l,
        "dcor_factor": ref_dcor_fac,
        "dcor_detail": dcor_detail,
        "variogram_params": variogram_params,
        "variogram_boundary_hit": boundary_hit,
        "variogram_max_lag_m": max_lag_used,
        "variogram_fit_success": vgm_result.get("fit_success"),
        "variogram_n_stable_pts": vgm_result.get("n_stable_pts"),
        "variogram_mode": vgm_result.get("variogram_mode"),
        "stable_tilt": ({k: v for k, v in stable_tilt.items() if k != "plane"}
                        if stable_tilt is not None else None),
        "stable_bias_detrended_myr": stable_bias_detrended_myr,
        "stable_tilt_term_myr": stable_tilt_term_myr,

        "variogram_standardized": cfg.STANDARDIZE_VARIOGRAM,

        "glacier_area_error_fraction": glac_area_error_frac,
        "dhdt_voi_raster": dhdt_voi_raster,

        # filtered dh/dt with the applied corrections, not void-filled
        "dhdt_bias_corrected_raster": dhdt_input_corr,

        # obs_union_mask: analysis domain with valid dh/dt (basis of the
        # reported means); union_inventory_mask: analysis domain only
        "obs_union_mask": obs_union_mask.astype(np.uint8),
        "union_inventory_mask": union_inventory_mask.astype(np.uint8),
        "area_mask_option": inventory_area_stats.get("area_mask_option", cfg.AREA_MASK_OPTION),
        "area_mask_support_label": inventory_area_stats.get("support_label"),

        "dhdt_products": {
            "dhdt_observed_union_myr": glac_mean_raw_obs,
            "dhdt_voidfilled_union_myr": glac_mean_voi_union,

            # cross-check: bin-weighted hypsometric mean over the observed domain
            "dhdt_voidfilled_obsunion_hypsometric_myr": glac_mean_voi_obs,

            "dhdt_observed_union_uncorrected_myr": glac_mean_raw_obs_uncorrected,
            "dhdt_voidfilled_union_uncorrected_myr": glac_mean_voi_union_uncorrected,
            "dhdt_penetration_bias_correction_applied": bool(cfg.SAR_PEN_APPLY_BIAS_CORRECTION),
            "dhdt_penetration_break_source": sar_out["meta"]["pen_break_source"],
            "dhdt_penetration_start_elev_m": sar_out["meta"]["pen_start_elev_m"],
            "dhdt_penetration_end_elev_m": sar_out["meta"]["pen_end_elev_m"],

            "stable_terrain_bias_myr": stable_bias_myr,
            "stable_terrain_bias_correction_applied": bool(apply_stable_bias),
            "stable_terrain_bias_uncertainty_term_myr": paper_unc["stable_bias_term_myr"],
            "dhdt_unc_random_with_void_myr": paper_unc["random_with_void_myr"],

            "void_fill_search_radius_px": spatial_search_px_dynamic,
            "void_fill_search_radius_km": spatial_search_px_dynamic * grid_info["resolution"] * cfg.M_TO_KM,
            "void_fill_search_radius_mode": cfg.VOID_FILL_SEARCH_PX_MODE,

            "dhdt_unc_measurement_myr": dhdt_unc_measurement_primary_myr,
            "dhdt_unc_measurement_with_void_myr": dhdt_unc_measurement_primary_with_void_myr,

            "dhdt_unc_measurement_primary_myr": dhdt_unc_measurement_primary_myr,
            "dhdt_unc_measurement_primary_with_void_myr": dhdt_unc_measurement_primary_with_void_myr,

            # Diagnostics
            "dhdt_unc_measurement_spatialavg_myr": dhdt_unc_measurement_spatialavg_myr,
            "dhdt_unc_measurement_spatialavg_with_void_myr": dhdt_unc_measurement_spatialavg_with_void_myr,

            "dhdt_unc_penetration_myr": sar_pen_uncertainty_myr,
        },

        "mass_balance_density850": area_products_A,
        "mass_balance_density900": area_products_B,

        "raw_rates": raw_rates,
        "void_rates": void_rates,
        "void_total": void_total
    }
