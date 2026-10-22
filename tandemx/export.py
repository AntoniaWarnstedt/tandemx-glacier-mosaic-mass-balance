"""Summary tables of the mass-balance results.
"""

import os

import numpy as np
import pandas as pd


def _extract_vgm_components(variogram_params):
    """
    Flatten a variogram_params DataFrame (model, range, psill) into a plain
    dict. Returns NaN for missing components so every row has the same keys
    whether the two-component fit or the single-spherical fallback was used.
    """
    out = dict(
        vgm_gaussian_range_m=np.nan,
        vgm_gaussian_psill=np.nan,
        vgm_spherical_range_m=np.nan,
        vgm_spherical_psill=np.nan,
        vgm_fullcorr_psill=np.nan,
        vgm_total_sill=np.nan,
        vgm_n_components=0,
    )
    if variogram_params is None or len(variogram_params) == 0:
        return out

    out["vgm_total_sill"] = float(variogram_params["psill"].sum())
    out["vgm_n_components"] = int(len(variogram_params))

    for _, row in variogram_params.iterrows():
        model = str(row["model"]).lower()
        if model == "gaussian":
            out["vgm_gaussian_range_m"] = float(row["range"])
            out["vgm_gaussian_psill"] = float(row["psill"])
        elif model == "spherical":
            out["vgm_spherical_range_m"] = float(row["range"])
            out["vgm_spherical_psill"] = float(row["psill"])
        elif model == "fullcorr":
            out["vgm_fullcorr_psill"] = float(row["psill"])
    return out


def _vgm_meta(r):
    """
    Extract all variogram diagnostics from one mb_results entry r.
    Returns a flat dict ready to merge into a CSV row.
    """
    vp = r.get("variogram_params")
    comp = _extract_vgm_components(vp)

    paper_unc = r.get("paper_uncertainty_stats", {})

    # Read the fields that ARE stored (variogram_fit_success /
    # variogram_n_stable_pts) directly instead.
    fit_success = r.get("variogram_fit_success")
    n_stable = r.get("variogram_n_stable_pts")

    return {
        **comp,
        "decorrelation_length_dominant_m": r.get("dcor_length_m"),
        "vgm_fit_success": fit_success,
        "vgm_boundary_hit": r.get("variogram_boundary_hit"),
        "vgm_max_lag_used_m": r.get("variogram_max_lag_m"),
        "vgm_n_stable_pts": n_stable,
        "N_eff": paper_unc.get("N_eff"),
        "neff_method": paper_unc.get("neff_method"),
        "variogram_mode": r.get("variogram_mode"),
        "StableTilt_E_myr_per_100km": (r.get("stable_tilt") or {}).get("b_per_km", np.nan) * 100.0,
        "StableTilt_N_myr_per_100km": (r.get("stable_tilt") or {}).get("c_per_km", np.nan) * 100.0,
        "StableTilt_glacier_mean_myr": (r.get("stable_tilt") or {}).get("glacier_mean_myr", np.nan),
        "StableTilt_glacier_se_myr": (r.get("stable_tilt") or {}).get("glacier_se_myr", np.nan),
        "StableTilt_n_px": (r.get("stable_tilt") or {}).get("n_px", np.nan),
        "StableTilt_slope_max_deg": (r.get("stable_tilt") or {}).get("slope_max_deg", np.nan),
        "StableResidual_detrended_mean_myr": r.get("stable_bias_detrended_myr"),
        "StableTilt_term_myr": r.get("stable_tilt_term_myr"),
        "Sensitivity_sigma_incl_tilt_myr": (
            float(np.hypot((r.get("paper_uncertainty_stats") or {}).get("total_with_void_with_pen_myr", np.nan),
                           r.get("stable_tilt_term_myr", np.nan)))
            if r.get("stable_tilt_term_myr") is not None else np.nan),
    }


def save_regional_mass_balance_summary(mb_results, out_dir):
    """Save expanded regional mass-balance summary with consistent method labels."""
    rows = []

    for r in mb_results:
        obs = r.get("obs_stats", {})
        inv = r.get("inventory_area_stats", {})
        obs_area = r.get("obs_area_stats", {})
        cov = r.get("coverage_stats", {})
        spatial_unc = r.get("spatial_uncertainty_stats", {})
        sar = r.get("sar_pen_meta", {})
        dhdt_products = r.get("dhdt_products", {})
        paper_unc = r.get("paper_uncertainty_stats", {})
        mb850 = r.get("mass_balance_density850", {})
        mb900 = r.get("mass_balance_density900", {})

        row = {
            "Region": r.get("Region"),
            "Period": r.get("Period"),
            "yA": r.get("yA"),
            "yB": r.get("yB"),
            "AreaMaskOption": r.get("area_mask_option"),
            "AreaMaskSupportLabel": r.get("area_mask_support_label"),

            "Inventory_area_start_km2": inv.get("Inventory_area_start_km2"),
            "Inventory_area_end_km2": inv.get("Inventory_area_end_km2"),
            "Inventory_area_mean_km2": inv.get("Inventory_area_mean_km2"),
            "Inventory_area_catchment_km2": inv.get("Inventory_area_union_km2"),
            "Inventory_area_intersection_km2": inv.get("Inventory_area_intersection_km2"),

            "Observed_area_start_km2": obs_area.get("Observed_inventory_area_start_km2"),
            "Observed_area_end_km2": obs_area.get("Observed_inventory_area_end_km2"),
            "Observed_area_support_raw_km2": obs_area.get("Observed_inventory_area_union_km2"),
            "Observed_area_intersection_km2": obs_area.get("Observed_inventory_area_intersection_km2"),

            "Obs_time_mean_years": obs.get("Dt_mean_glc"),
            "Obs_time_median_years": obs.get("Dt_median_glc"),

            "coverage_vs_inventory_catchment_pct": cov.get("coverage_vs_inventory_union_pct"),
            "coverage_vs_inventory_mean_pct": cov.get("coverage_vs_inventory_mean_pct"),
            "coverage_fraction_union": cov.get("coverage_fraction_union"),
            "void_fraction_union": cov.get("void_fraction_union"),
            "void_factor_union": cov.get("void_factor_union"),

            "dhdt_raw_observed_support_myr": dhdt_products.get("dhdt_observed_union_myr"),
            "dhdtfilled_glacierwide_myr": dhdt_products.get("dhdt_voidfilled_union_myr"),

            "sigma_dhdt_eqX_no_void_myr": dhdt_products.get("dhdt_unc_measurement_primary_myr"),
            "sigma_dhdt_eqX_total_myr": dhdt_products.get("dhdt_unc_measurement_primary_with_void_myr"),
            "sigma_dhdt_spatialavg_no_void_myr": dhdt_products.get("dhdt_unc_measurement_spatialavg_myr"),
            "sigma_dhdt_spatialavg_with_void_myr": dhdt_products.get("dhdt_unc_measurement_spatialavg_with_void_myr"),
            "sigma_dhdt_penetration_myr": dhdt_products.get("dhdt_unc_penetration_myr"),

            "stable_sigma_myr": paper_unc.get("ht_awstable_myr"),
            "Scor_km2": paper_unc.get("Scor_km2"),
            "Sglc_km2": paper_unc.get("Sglc_km2"),
            "rolstad_factor": paper_unc.get("rolstad_factor"),
            "Areafrac": r.get("glacier_area_error_fraction"),

            "stable_sigma_residual_myr": spatial_unc.get("sigma_residual_myr"),
            "spatial_support_area_m2": spatial_unc.get("area_support_m2"),
            "spatial_correlation_area_m2": spatial_unc.get("area_correlation_m2"),
            "spatial_n_eff": spatial_unc.get("n_eff"),

            "decorrelation_length_L_m": r.get("dcor_length_m"),
            "decorrelation_factor_weighted": r.get("dcor_factor"),

            "SARpen_mean_myr": sar.get("Mean_sar_pen"),
            "SARpen_volume_m3yr": sar.get("Vol_sar_pen"),
            "SARpen_break_source": sar.get("pen_break_source"),
            "SARpen_start_elev_m": sar.get("pen_start_elev_m"),
            "SARpen_gatekeeper_inputs_ok": sar.get("pen_gatekeeper_inputs_ok"),
            "SARpen_slope_stratified_verdict": sar.get("pen_slope_stratified_verdict"),
            "SARpen_slope_stratified_detail": sar.get("pen_slope_stratified_detail"),
            "SARpen_correction_applied": sar.get("pen_correction_applied"),
            "SARpen_break_max_z_tested": sar.get("pen_break_max_z_tested"),
            "SARpen_max_tested_bin_m": sar.get("pen_cov_max_tested_bin_m"),
            "SARpen_catchment_max_bin_m": sar.get("pen_cov_catchment_max_bin_m"),
            "SARpen_catchment_area_above_start_km2": sar.get("pen_cov_catchment_area_above_start_km2"),
            "SARpen_observed_pct_above_start": sar.get("pen_cov_observed_pct_above_start"),
            "SARpen_catchment_area_above_tested_km2": sar.get("pen_cov_catchment_area_above_tested_km2"),
            "StableTerrainBias_myr": dhdt_products.get("stable_terrain_bias_myr"),
            "StableTerrainBias_correction_applied": dhdt_products.get("stable_terrain_bias_correction_applied"),
            "VoidFillSearchRadius_px": dhdt_products.get("void_fill_search_radius_px"),
            "VoidFillSearchRadius_km": dhdt_products.get("void_fill_search_radius_km"),
        }

        # Hugonnet variogram components (Gaussian + Spherical) and N_eff
        row.update(_vgm_meta(r))

        row.update(mb850)
        row.update(mb900)

        rows.append(row)

    out_csv = os.path.join(out_dir, "regional_mass_balance_summary.csv")
    pd.DataFrame(rows).to_csv(out_csv, index=False)
    print(f"   ✓ Saved: {out_csv}")


def save_regional_mass_balance_summary_main_results(mb_results, out_dir):
    """
    Main manuscript-ready results table with clear, method-consistent labels.
    Removes 'Primary' naming convention.
    """
    rows = []

    for r in mb_results:
        cov = r.get("coverage_stats", {})
        dhdt = r.get("dhdt_products", {})
        mb850 = r.get("mass_balance_density850", {})
        mb900 = r.get("mass_balance_density900", {})
        inv = r.get("inventory_area_stats", {})
        obs_area = r.get("obs_area_stats", {})
        paper_unc = r.get("paper_uncertainty_stats", {})
        sar = r.get("sar_pen_meta", {})

        # Catchment results -- Sommer et al. (2020)/Elsberg et al. (2001) MEAN
        # start/end area convention (NOT the per-pair inventory union).
        catchment_area_km2 = inv.get("Inventory_area_mean_km2")
        catchment_dMdt_850 = mb850.get("inventory_mean_dMdt_Gtyr")
        catchment_dMdt_unc_850 = mb850.get("inventory_mean_dMdt_unc_combined_with_pen_Gtyr")
        catchment_smb_850 = mb850.get("inventory_mean_specific_mwe_yr")
        catchment_smb_unc_850 = mb850.get("inventory_mean_specific_mwe_yr_unc")

        catchment_dMdt_900 = mb900.get("inventory_mean_dMdt_Gtyr_density900")
        catchment_dMdt_unc_900 = mb900.get("inventory_mean_dMdt_unc_combined_with_pen_Gtyr_density900")
        catchment_smb_900 = mb900.get("inventory_mean_specific_mwe_yr_density900")
        catchment_smb_unc_900 = mb900.get("inventory_mean_specific_mwe_yr_unc_density900")

        # Clear uncertainty naming
        sigma_dhdt_stable_spatial_myr = dhdt.get("dhdt_unc_measurement_primary_myr")
        sigma_dhdt_final_myr = dhdt.get("dhdt_unc_measurement_primary_with_void_myr")
        sigma_dhdt_penetration_myr = dhdt.get("dhdt_unc_penetration_myr")

        row = {
            "Region": r.get("Region"),
            "Period": r.get("Period"),
            "yA": r.get("yA"),
            "yB": r.get("yB"),
            # "C" (start∪end union, original default) or "B" (end-of-period
            # outline only) -- see AREA_MASK_OPTION near the top of the file.
            "AreaMaskOption": r.get("area_mask_option"),

            # Areas
            "area_observed_support_raw_km2": obs_area.get("Observed_inventory_area_union_km2"),
            "area_mean_start_end_km2": inv.get("Inventory_area_mean_km2"),
            "area_inventory_catchment_km2": catchment_area_km2,
            "coverage_vs_inventory_catchment_pct": cov.get("coverage_vs_inventory_union_pct"),

            # Elevation-change products
            "dhdt_raw_observed_support_myr": dhdt.get("dhdt_observed_union_myr"),
            "dhdt_filled_glacierwide_myr": dhdt.get("dhdt_voidfilled_union_myr"),

            # Uncertainty components
            "sigma_dhdt_stable_spatial_myr": sigma_dhdt_stable_spatial_myr,
            "sigma_dhdt_final_stable_spatial_void_SARpen_myr": sigma_dhdt_final_myr,
            "sigma_dhdt_penetration_myr": sigma_dhdt_penetration_myr,

            "stable_sigma_myr": paper_unc.get("ht_awstable_myr"),
            "decorrelation_length_L_m": r.get("dcor_length_m"),
            "Scor_km2": paper_unc.get("Scor_km2"),
            "Sglc_km2": paper_unc.get("Sglc_km2"),
            "rolstad_factor": paper_unc.get("rolstad_factor"),
            "void_fraction": paper_unc.get("void_fraction"),
            "void_factor": paper_unc.get("void_factor"),
            "Areafrac": r.get("glacier_area_error_fraction"),
            "SARpen_mean_myr": sar.get("Mean_sar_pen"),
            "SARpen_break_source": sar.get("pen_break_source"),
            "SARpen_start_elev_m": sar.get("pen_start_elev_m"),
            "SARpen_gatekeeper_inputs_ok": sar.get("pen_gatekeeper_inputs_ok"),
            "SARpen_slope_stratified_verdict": sar.get("pen_slope_stratified_verdict"),
            "SARpen_slope_stratified_detail": sar.get("pen_slope_stratified_detail"),
            "SARpen_correction_applied": sar.get("pen_correction_applied"),
            "SARpen_break_max_z_tested": sar.get("pen_break_max_z_tested"),
            "SARpen_max_tested_bin_m": sar.get("pen_cov_max_tested_bin_m"),
            "SARpen_catchment_max_bin_m": sar.get("pen_cov_catchment_max_bin_m"),
            "SARpen_catchment_area_above_start_km2": sar.get("pen_cov_catchment_area_above_start_km2"),
            "SARpen_observed_pct_above_start": sar.get("pen_cov_observed_pct_above_start"),
            "SARpen_catchment_area_above_tested_km2": sar.get("pen_cov_catchment_area_above_tested_km2"),
            "StableTerrainBias_myr": dhdt.get("stable_terrain_bias_myr"),
            "StableTerrainBias_correction_applied": dhdt.get("stable_terrain_bias_correction_applied"),
            "VoidFillSearchRadius_px": dhdt.get("void_fill_search_radius_px"),
            "VoidFillSearchRadius_km": dhdt.get("void_fill_search_radius_km"),

            # Catchment mass-balance result
            "Catchment_area_definition": "inventory_mean_area_voidfilled_sommer_elsberg",
            "Catchment_dMdt_Gtyr_850": catchment_dMdt_850,
            "Catchment_dMdt_unc_Gtyr_850": catchment_dMdt_unc_850,
            "Catchment_specific_mass_balance_mwe_yr_850": catchment_smb_850,
            "Catchment_specific_mass_balance_unc_mwe_yr_850": catchment_smb_unc_850,
            "Catchment_dMdt_Gtyr_900": catchment_dMdt_900,
            "Catchment_dMdt_unc_Gtyr_900": catchment_dMdt_unc_900,
            "Catchment_specific_mass_balance_mwe_yr_900": catchment_smb_900,
            "Catchment_specific_mass_balance_unc_mwe_yr_900": catchment_smb_unc_900,

            # Optional uncertainty on integrated products (mean-area convention)
            "sigma_dVdt_inventory_catchment_km3yr_850": mb850.get("inventory_mean_dVdt_unc_combined_with_pen_km3yr"),
            "sigma_dMdt_inventory_catchment_Gtyr_850": mb850.get("inventory_mean_dMdt_unc_combined_with_pen_Gtyr"),
            "sigma_dVdt_inventory_catchment_km3yr_900": mb900.get("inventory_mean_dVdt_unc_combined_with_pen_km3yr_density900"),
            "sigma_dMdt_inventory_catchment_Gtyr_900": mb900.get("inventory_mean_dMdt_unc_combined_with_pen_Gtyr_density900"),
        }

        # Hugonnet variogram components (Gaussian + Spherical) and N_eff
        row.update(_vgm_meta(r))

        rows.append(row)

    out_csv = os.path.join(out_dir, "regional_mass_balance_main_results.csv")
    pd.DataFrame(rows).to_csv(out_csv, index=False)
    print(f"   ✓ Saved main-results summary: {out_csv}")


def save_regional_mass_balance_summary_paper(mb_results, out_dir):
    """
    Save manuscript-ready summary table with method-consistent labels.
    """
    rows = []

    for r in mb_results:
        inv = r.get("inventory_area_stats", {})
        obs_area = r.get("obs_area_stats", {})
        cov = r.get("coverage_stats", {})
        obs = r.get("obs_stats", {})
        dhdt = r.get("dhdt_products", {})
        mb850 = r.get("mass_balance_density850", {})
        mb900 = r.get("mass_balance_density900", {})
        paper_unc = r.get("paper_uncertainty_stats", {})
        sar = r.get("sar_pen_meta", {})

        row = {
            "Region": r.get("Region"),
            "Period": r.get("Period"),
            "yA": r.get("yA"),
            "yB": r.get("yB"),
            # "C" (start∪end union, original default) or "B" (end-of-period
            # outline only) -- see AREA_MASK_OPTION near the top of the file.
            "AreaMaskOption": r.get("area_mask_option"),

            "obs_time_mean_years": obs.get("Dt_mean_glc"),
            "obs_time_median_years": obs.get("Dt_median_glc"),

            "area_observed_support_raw_km2": obs_area.get("Observed_inventory_area_union_km2"),
            "area_mean_start_end_km2": inv.get("Inventory_area_mean_km2"),
            "area_inventory_catchment_km2": inv.get("Inventory_area_mean_km2"),
            "coverage_vs_inventory_catchment_pct": cov.get("coverage_vs_inventory_union_pct"),

            "dhdt_raw_observed_support_myr": dhdt.get("dhdt_observed_union_myr"),
            "dhdt_filled_glacierwide_myr": dhdt.get("dhdt_voidfilled_union_myr"),

            "stable_sigma_myr": paper_unc.get("ht_awstable_myr"),
            "decorrelation_length_L_m": r.get("dcor_length_m"),
            "Scor_km2": paper_unc.get("Scor_km2"),
            "Sglc_km2": paper_unc.get("Sglc_km2"),
            "rolstad_factor": paper_unc.get("rolstad_factor"),
            "void_fraction": paper_unc.get("void_fraction"),
            "void_factor": paper_unc.get("void_factor"),
            "SARpen_myr": sar.get("Mean_sar_pen"),
            "SARpen_break_source": sar.get("pen_break_source"),
            "SARpen_start_elev_m": sar.get("pen_start_elev_m"),
            "SARpen_end_elev_m": sar.get("pen_end_elev_m"),
            "SARpen_gatekeeper_inputs_ok": sar.get("pen_gatekeeper_inputs_ok"),
            "SARpen_slope_stratified_verdict": sar.get("pen_slope_stratified_verdict"),
            "SARpen_slope_stratified_detail": sar.get("pen_slope_stratified_detail"),
            "SARpen_correction_applied": sar.get("pen_correction_applied"),
            "SARpen_break_max_z_tested": sar.get("pen_break_max_z_tested"),
            "SARpen_max_tested_bin_m": sar.get("pen_cov_max_tested_bin_m"),
            "SARpen_catchment_max_bin_m": sar.get("pen_cov_catchment_max_bin_m"),
            "SARpen_catchment_area_above_start_km2": sar.get("pen_cov_catchment_area_above_start_km2"),
            "SARpen_observed_pct_above_start": sar.get("pen_cov_observed_pct_above_start"),
            "SARpen_catchment_area_above_tested_km2": sar.get("pen_cov_catchment_area_above_tested_km2"),
            "StableTerrainBias_myr": dhdt.get("stable_terrain_bias_myr"),
            "StableTerrainBias_correction_applied": dhdt.get("stable_terrain_bias_correction_applied"),
            "VoidFillSearchRadius_px": dhdt.get("void_fill_search_radius_px"),
            "VoidFillSearchRadius_km": dhdt.get("void_fill_search_radius_km"),
            "Areafrac": r.get("glacier_area_error_fraction"),

            "sigma_dhdt_stable_spatial_myr": dhdt.get("dhdt_unc_measurement_primary_myr"),
            "sigma_dhdt_final_stable_spatial_void_SARpen_myr": dhdt.get("dhdt_unc_measurement_primary_with_void_myr"),

            "Catchment_area_definition": "inventory_mean_area_voidfilled_sommer_elsberg",
            "Catchment_dMdt_Gtyr_850": mb850.get("inventory_mean_dMdt_Gtyr"),
            "Catchment_dMdt_unc_Gtyr_850": mb850.get("inventory_mean_dMdt_unc_combined_with_pen_Gtyr"),
            "Catchment_specific_mass_balance_mwe_yr_850": mb850.get("inventory_mean_specific_mwe_yr"),
            "Catchment_specific_mass_balance_unc_mwe_yr_850": mb850.get("inventory_mean_specific_mwe_yr_unc"),
            "Catchment_total_mass_change_Gt_850": mb850.get("inventory_mean_Total_mass_change_Gt"),
            "Catchment_total_mass_change_unc_Gt_850": mb850.get("inventory_mean_Total_mass_change_unc_with_pen_Gt"),

            "Catchment_dMdt_Gtyr_900": mb900.get("inventory_mean_dMdt_Gtyr_density900"),
            "Catchment_dMdt_unc_Gtyr_900": mb900.get("inventory_mean_dMdt_unc_combined_with_pen_Gtyr_density900"),
            "Catchment_specific_mass_balance_mwe_yr_900": mb900.get("inventory_mean_specific_mwe_yr_density900"),
            "Catchment_specific_mass_balance_unc_mwe_yr_900": mb900.get("inventory_mean_specific_mwe_yr_unc_density900"),
            "Catchment_total_mass_change_Gt_900": mb900.get("inventory_mean_Total_mass_change_Gt_density900"),
            "Catchment_total_mass_change_unc_Gt_900": mb900.get("inventory_mean_Total_mass_change_unc_with_pen_Gt_density900"),
        }

        # Hugonnet variogram components (Gaussian + Spherical) and N_eff
        row.update(_vgm_meta(r))

        rows.append(row)

    out_csv = os.path.join(out_dir, "regional_mass_balance_summary_paper.csv")
    pd.DataFrame(rows).to_csv(out_csv, index=False)
    print(f"   ✓ Saved paper summary: {out_csv}")
