"""Uncertainty of the glacier-wide mean dh/dt (Eq. 1, Hugonnet et al. 2022 with
void factor after Braun et al. 2019).
"""

import numpy as np
import pandas as pd
from scipy import stats

from . import config as cfg
from .variogram import neff_circular_numerical, neff_hugonnet_polygon


def compute_paper_eq1_uncertainty(
        stable_summary: dict,
        observed_area_km2: float,
        variogram_params: pd.DataFrame,
        corr_func,
        sar_penetration_uncertainty_myr: float,
        coverage_fraction: float,
        union_geom=None,
        grid_info: dict = None,
        polygon_neff_subsample: int = 2000,
        combine_mode: str = "quadrature",
        stable_bias_myr: float = np.nan,
) -> dict:
    """Eq. 1 measurement uncertainty — Hugonnet et al. (2022)."""
    # Extract σ_stable (glacier-area-weighted, filtered)
    ht_awstable = stable_summary.get("sdev_fil_all_gw", np.nan)

    # Total sill from variogram
    total_sill = (
        float(np.sum(variogram_params["psill"].values))
        if variogram_params is not None and len(variogram_params) > 0
        else 1.0
    )

    def _f(v):
        """Safe float conversion (NaN if not finite)"""
        return float(v) if (v is not None and np.isfinite(v)) else np.nan

    # ── N_eff computation ─────────────────────────────────────────────
    if not np.isfinite(observed_area_km2) or observed_area_km2 <= 0:
        neff, neff_method = np.nan, "none"
    elif union_geom is not None and grid_info is not None:
        # Polygon-based: more accurate for irregular glaciers
        neff = neff_hugonnet_polygon(
            union_geom, grid_info, corr_func,
            subsample=polygon_neff_subsample,
        )
        neff_method = "hugonnet_polygon"
    else:
        # Circular: faster, default for simple shapes
        neff = neff_circular_numerical(observed_area_km2 * cfg.KM2_TO_M2, variogram_params)
        neff_method = "hugonnet_circular"

    sglc_km2 = float(observed_area_km2)

    if np.isfinite(neff) and neff > 0:
        rolstad_factor = 1.0 / np.sqrt(neff)
        # Equivalent correlation area: 5*S_glc / (π*L²) = N_eff
        L_equiv_m2 = 5.0 * sglc_km2 * cfg.KM2_TO_M2 / (np.pi * neff)
        scor_km2 = (np.pi * L_equiv_m2) * cfg.M2_TO_KM2
    else:
        rolstad_factor = np.nan
        scor_km2 = np.nan

    # ── Measurement uncertainty (no void, no penetration) ──────────────
    if np.isfinite(ht_awstable) and np.isfinite(neff) and neff > 0:
        measurement_no_void_no_pen = ht_awstable / np.sqrt(neff)
    else:
        measurement_no_void_no_pen = np.nan

    # ── Void factor ───────────────────────────────────────────────────
    # Void penalty after Braun et al. (2019, Nat. Clim. Change, Methods)
    if np.isfinite(coverage_fraction):
        coverage_fraction = float(np.clip(coverage_fraction, 0.0, 1.0))
        void_fraction = 1.0 - coverage_fraction
        # Braun et al. (2019): uncertainty doubled for the unmeasured
        # fraction -> sigma*[(1-f) + 2f] = sigma*(1 + f)
        void_factor = 1.0 + void_fraction
    else:
        void_fraction = np.nan
        void_factor = np.nan

    measurement_with_void_no_pen = (
        measurement_no_void_no_pen * void_factor
        if np.isfinite(measurement_no_void_no_pen) and np.isfinite(void_factor)
        else np.nan
    )

    # ── Systematic stable-terrain residual (uncorrected bias) ─────────
    # Added in quadrature AFTER the √N_eff reduction and void factor: a
    # systematic offset does not average out over the glacier area.
    random_no_void = measurement_no_void_no_pen
    random_with_void = measurement_with_void_no_pen
    bias_term = abs(float(stable_bias_myr)) if (stable_bias_myr is not None
                                               and np.isfinite(stable_bias_myr)) else 0.0
    if np.isfinite(measurement_no_void_no_pen):
        measurement_no_void_no_pen = float(np.sqrt(measurement_no_void_no_pen ** 2 + bias_term ** 2))
    if np.isfinite(measurement_with_void_no_pen):
        measurement_with_void_no_pen = float(np.sqrt(measurement_with_void_no_pen ** 2 + bias_term ** 2))

    # penetration term combined in quadrature (independent error sources)
    def _combine_pen(measurement_term, pen_term, mode):
        if not (np.isfinite(measurement_term) and np.isfinite(pen_term)):
            return np.nan
        if mode == "linear":
            return measurement_term + pen_term
        # default: quadrature / root-sum-square
        return float(np.sqrt(measurement_term ** 2 + pen_term ** 2))

    total_no_void_with_pen = _combine_pen(
        measurement_no_void_no_pen, sar_penetration_uncertainty_myr, combine_mode
    )

    total_with_void_with_pen = _combine_pen(
        measurement_with_void_no_pen, sar_penetration_uncertainty_myr, combine_mode
    )

    return {
        "ht_awstable_myr": _f(ht_awstable),
        "Scor_km2": _f(scor_km2),
        "Sglc_km2": _f(sglc_km2),
        "rolstad_factor": _f(rolstad_factor),

        # Hugonnet-specific
        "N_eff": _f(neff),
        "neff_method": neff_method,
        "total_sill": _f(total_sill),

        # Uncertainty products (main results)
        "random_no_void_myr": _f(random_no_void),
        "random_with_void_myr": _f(random_with_void),
        "stable_bias_term_myr": _f(bias_term),
        "measurement_no_void_no_pen_myr": _f(measurement_no_void_no_pen),
        "measurement_with_void_no_pen_myr": _f(measurement_with_void_no_pen),
        "total_no_void_with_pen_myr": _f(total_no_void_with_pen),
        "total_with_void_with_pen_myr": _f(total_with_void_with_pen),
        "combine_mode": combine_mode,

        # Void factors
        "void_fraction": _f(void_fraction),
        "void_factor": _f(void_factor),
    }


def compute_spatially_averaged_mean_uncertainty(
    residual_raster: np.ndarray,
    support_mask: np.ndarray,
    grid_info: dict,
    variogram_params: pd.DataFrame = None,
    corr_func=None,
    robust: bool = True,
    decorrelation_length_m: float = None,
) -> dict:
    """Spatially-averaged mean uncertainty — Hugonnet et al. (2022) edition."""
    pixel_area_m2   = float(grid_info["resolution"] ** 2)
    support_area_m2 = float(np.sum(support_mask) * pixel_area_m2)

    vals = residual_raster[np.isfinite(residual_raster)]

    if vals.size < 10:
        return {
            "sigma_residual_myr":  np.nan,
            "area_support_m2":     support_area_m2,
            "area_correlation_m2": np.nan,
            "n_eff":               np.nan,
            "sigma_mean_myr":      np.nan,
        }

    # ── Sigma from stable terrain ─────────────────────────────────────
    if robust:
        mad   = stats.median_abs_deviation(vals, nan_policy="omit")
        sigma = float(1.4826 * mad) if np.isfinite(mad) else np.nan
    else:
        sigma = float(np.nanstd(vals, ddof=1))

    # ── N_eff ─────────────────────────────────────────────────────────
    if variogram_params is not None and len(variogram_params) > 0:
        neff          = neff_circular_numerical(support_area_m2, variogram_params)
        dom_range     = float(variogram_params["range"].max())
        area_corr_m2  = float(np.pi * dom_range ** 2)
    else:
        L = (float(decorrelation_length_m)
             if decorrelation_length_m is not None
             and np.isfinite(decorrelation_length_m)
             and decorrelation_length_m > 0
             else 1000.0)
        area_corr_m2 = float(np.pi * L ** 2)
        neff = max(support_area_m2 / area_corr_m2, 1.0) if support_area_m2 > 0 else 1.0

    sigma_mean = (float(sigma / np.sqrt(neff))
                  if np.isfinite(sigma) and neff > 0 else np.nan)

    return {
        "sigma_residual_myr":  float(sigma)       if np.isfinite(sigma) else np.nan,
        "area_support_m2":     support_area_m2,
        "area_correlation_m2": area_corr_m2,
        "n_eff":               float(neff)         if np.isfinite(neff)  else np.nan,
        "sigma_mean_myr":      sigma_mean,
    }
