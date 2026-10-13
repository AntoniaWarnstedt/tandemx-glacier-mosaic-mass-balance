"""SAR penetration: onset elevation from inter-period consistency, break-in-slope
test and the elevation-dependent penetration ramp.
"""

import os

import numpy as np
import pandas as pd

from . import config as cfg
from .utils import canonical_period_key


def detect_penetration_break_elevation(
    bin_stats,
    elev_col="cl",
    dhdt_col="mean_ch_fil",
    n_col="n_fil",
    min_bin_pixels=None,
    core_quantile=0.75,
    residual_sigma_thresh=2.0,
    min_consecutive_bins=3,
    core_max_elev=None,
    core_min_elev=None,
):
    """
    Heuristic detection of a high-elevation SAR-penetration anomaly in the on-
    ice dh/dt hypsometry.
    """
    diag = {
        "status": None,
        "n_bins_total": 0,
        "n_bins_core": 0,
        "n_bins_upper": 0,
        "core_cut_elev_m": np.nan,
        "core_min_elev_m": np.nan,
        # reference fit, also used by the diagnostic plot
        "fit_slope": np.nan,
        "fit_intercept": np.nan,
        "sigma_core": np.nan,
        "max_z_upper": np.nan,     # largest residual z among TESTED bins only
        "n_upper_anomalous": 0,    # tested bins with z > residual_sigma_thresh
    }

    if bin_stats is None or bin_stats.empty:
        diag["status"] = "no_bin_stats"
        return None, diag
    if min_bin_pixels is None:
        min_bin_pixels = cfg.MIN_BIN_PIXELS

    df = bin_stats[[elev_col, dhdt_col, n_col]].dropna()
    df = df[df[n_col] >= min_bin_pixels]
    diag["n_bins_total"] = int(len(df))
    if len(df) < 8:
        diag["status"] = "insufficient_total_bins"
        return None, diag  # not enough bins to say anything meaningful

    df = df.sort_values(elev_col)
    elev = df[elev_col].values.astype(float)
    dhdt = df[dhdt_col].values.astype(float)

    if core_max_elev is not None:
        core_cut = float(core_max_elev)
        core_mask = elev < core_cut
        upper_mask = elev >= core_cut
    else:
        core_cut = np.quantile(elev, core_quantile)
        core_mask = elev <= core_cut
        upper_mask = elev > core_cut
    if core_min_elev is not None:
        core_mask &= elev >= float(core_min_elev)
        diag["core_min_elev_m"] = float(core_min_elev)
    else:
        diag["core_min_elev_m"] = float(np.nanmin(elev))
    diag["n_bins_core"] = int(core_mask.sum())
    diag["core_cut_elev_m"] = float(core_cut)
    if core_mask.sum() < 5:
        diag["status"] = "insufficient_core_bins"
        return None, diag

    # Robust linear fit on the core (lower/mid) elevation range
    slope, intercept = np.polyfit(elev[core_mask], dhdt[core_mask], 1)
    resid_core = dhdt[core_mask] - (slope * elev[core_mask] + intercept)
    sigma_core = 1.4826 * np.median(np.abs(resid_core - np.median(resid_core)))
    if not np.isfinite(sigma_core) or sigma_core == 0:
        sigma_core = np.nanstd(resid_core)
    if not np.isfinite(sigma_core) or sigma_core == 0:
        diag["status"] = "degenerate_core_sigma"
        return None, diag

    resid_all = dhdt - (slope * elev + intercept)
    z = resid_all / sigma_core
    diag["fit_slope"] = float(slope)
    diag["fit_intercept"] = float(intercept)
    diag["sigma_core"] = float(sigma_core)

    upper_idx = np.where(upper_mask)[0]
    diag["n_bins_upper"] = int(len(upper_idx))
    if len(upper_idx) > 0:
        diag["max_z_upper"] = float(np.nanmax(z[upper_idx]))
        diag["n_upper_anomalous"] = int(np.sum(z[upper_idx] > residual_sigma_thresh))

    # too few tested bins: untestable, not a negative result
    if len(upper_idx) < min_consecutive_bins:
        diag["status"] = "insufficient_upper_bins"
        return None, diag

    # Consecutive = adjacent in elevation among the bins that survived the
    # pixel-count filter (bins are sorted by elevation above).
    run_start = None
    for i in range(len(upper_idx) - min_consecutive_bins + 1):
        window = z[upper_idx[i: i + min_consecutive_bins]]
        if np.all(window > residual_sigma_thresh):
            run_start = upper_idx[i]
            break

    if run_start is None:
        diag["status"] = "no_anomaly_found"
        return None, diag

    diag["status"] = "break_detected"
    return float(elev[run_start]), diag


def detect_penetration_break_elevation_slope_stratified(
    dhdt_on_ice, elev_bins, slope_raster, glacier_mask,
    slope_bin_width_deg=cfg.SLOPE_BIN_WIDTH_DEG,
    slope_bin_max_deg=cfg.SLOPE_BIN_MAX_DEG,
    min_bin_pixels=None, core_quantile=0.75,
    residual_sigma_thresh=2.0, min_consecutive_bins=3,
    min_classes_for_verdict=2,
    core_max_elev=None, core_min_elev=None,
):
    """Slope-stratified wrapper around detect_penetration_break_elevation()."""
    if min_bin_pixels is None:
        min_bin_pixels = cfg.SAR_PEN_DETECT_MIN_BIN_PIXELS

    df = pd.DataFrame({
        "z": dhdt_on_ice.ravel(), "elev": elev_bins.ravel(),
        "slope": slope_raster.ravel(), "glc": glacier_mask.ravel(),
    })
    df = df[(df["glc"] == 1) & np.isfinite(df["z"])
            & np.isfinite(df["elev"]) & np.isfinite(df["slope"])].copy()

    results = {}
    if df.empty:
        return {"classes": results, "verdict": "no_data"}

    lo = 0.0
    while lo < slope_bin_max_deg:
        hi = min(lo + slope_bin_width_deg, slope_bin_max_deg)
        while True:
            sub = df[(df["slope"] >= lo) & (df["slope"] < hi)]
            bin_stats = (sub.groupby("elev")["z"]
                            .agg(mean_ch_fil="mean", n_fil="count")
                            .reset_index().rename(columns={"elev": "cl"}))
            h_break, diag = detect_penetration_break_elevation(
                bin_stats, min_bin_pixels=min_bin_pixels,
                core_quantile=core_quantile,
                residual_sigma_thresh=residual_sigma_thresh,
                min_consecutive_bins=min_consecutive_bins,
                core_max_elev=core_max_elev, core_min_elev=core_min_elev,
            )
            testable = diag["status"] in ("no_anomaly_found", "break_detected")
            if testable or hi >= slope_bin_max_deg:
                break
            hi = min(hi + slope_bin_width_deg, slope_bin_max_deg)  # widen and retry

        results[f"{lo:.0f}-{hi:.0f}"] = {
            "h_break": h_break, "status": diag["status"],
            "n_pixels": int(len(sub)), "widened": hi - lo > slope_bin_width_deg,
        }
        lo = hi

    tested = {k: v for k, v in results.items() if v["status"] in ("no_anomaly_found", "break_detected")}
    breaks = {k: v for k, v in tested.items() if v["status"] == "break_detected"}

    if len(tested) < min_classes_for_verdict:
        verdict = "inconclusive"
    elif not breaks:
        verdict = "no_anomaly_any_class"
    elif len(breaks) == len(tested):
        verdict = "elevation_driven"
    elif all(int(k.split("-")[0]) >= cfg.SLOPE_STEEP_THRESHOLD_DEG for k in breaks):
        verdict = "slope_driven"
    else:
        verdict = "mixed"

    return {"classes": results, "verdict": verdict}


def compute_sar_penetration(
    ref_dem,
    dt_raster,
    glacier_mask,
    pixel_area_m2,
    sar_pen_type="ramp",
    sar_pen_bias=2.0,
    pen_start_elev_m=None,
    pen_end_elev_m=None,
    onice_bin_stats=None,
    pen_detect_min_bin_pixels=None,
    apply_bias_correction=True,
    residual_unc_fraction=0.0,
    dhdt_on_ice_raw=None,
    slope_raster=None,  # pixel-level slope raster (degrees), e.g. your existing preloaded_slope.
    elev_bins_raster=None,
    stratified_glacier_mask=None,  # mask to use for the stratified test.
    require_break=True,
    break_ref_min_elev_m=None,  # lower edge of the reference fit (see SAR_PEN_BREAK_REF_MIN_ELEV).
    break_sigma_thresh=2.0,  # see SAR_PEN_BREAK_SIGMA_THRESH
    break_min_consecutive=3,  # see SAR_PEN_BREAK_MIN_CONSECUTIVE
):
    """
    Use `pen_start_elev_m` if given explicitly. 2. Else, try to auto-detect a
    break-in-slope anomaly from `onice_bin_stats` (see
    detect_penetration_break_elevation).
    """
    gmask = (glacier_mask == 1) & np.isfinite(ref_dem)

    pen_abs = np.full_like(ref_dem, np.nan, dtype=np.float32)
    pen_rate = np.full_like(ref_dem, np.nan, dtype=np.float32)
    affected = np.zeros_like(gmask, dtype=bool)

    empty_meta = {
        "N_sar_pen": 0,
        "N_glc_reg": 0,
        "elv_min": np.nan,
        "pen_min": np.nan,
        "elv_max": np.nan,
        "pen_max": np.nan,
        "cor_slp": np.nan,
        "Obs_mean_pen_cells": np.nan,
        "Mean_sar_pen": np.nan,
        "Mean_sar_pen_uncertainty_myr": np.nan,
        "Vol_sar_pen": np.nan,
        "pen_start_elev_m": np.nan,
        "pen_end_elev_m": np.nan,
        "pen_break_source": "none",
        "pen_break_diag": None,
        "pen_gatekeeper_inputs_ok": False,
        "pen_slope_stratified_verdict": "no_glacier_cells",
        "pen_slope_stratified_detail": "",
        "pen_correction_applied": False,
        "pen_band_tested_start_m": np.nan,
        "pen_band_tested_end_m": np.nan,
        "pen_break_max_z_tested": np.nan,
    }

    if not np.any(gmask):
        return {"sar_pen_rate": pen_rate, "sar_pen_abs": pen_abs, "meta": empty_meta}

    glc_elv = ref_dem[gmask]
    h_pen_max_glacier = np.nanmax(glc_elv) if np.any(np.isfinite(glc_elv)) else np.nan

    # ------------------------------------------------------------------
    # resolve whether -- and where -- a penetration correction is applied.
    # ------------------------------------------------------------------
    break_diag = None
    strat_result = None
    gatekeeper_inputs_ok = (dhdt_on_ice_raw is not None and slope_raster is not None
                            and elev_bins_raster is not None)
    explicit_band = pen_start_elev_m is not None
    detect_kwargs = dict(
        min_bin_pixels=pen_detect_min_bin_pixels,
        residual_sigma_thresh=break_sigma_thresh,
        min_consecutive_bins=break_min_consecutive,
        core_max_elev=pen_start_elev_m if explicit_band else None,
        core_min_elev=break_ref_min_elev_m if explicit_band else None,
    )

    if explicit_band and not require_break:
        h_pen_min = pen_start_elev_m
        pen_break_source = "explicit_forced"
    else:
        h_detect = None
        if onice_bin_stats is not None:
            h_detect, break_diag = detect_penetration_break_elevation(
                onice_bin_stats, **detect_kwargs
            )

        if h_detect is not None:
            # In auto mode the ramp starts at the break.
            h_pen_min = pen_start_elev_m if explicit_band else h_detect
            pen_break_source = ("explicit_band_break_confirmed" if explicit_band
                                else "auto_detected")
        else:
            h_pen_min = None
            if break_diag is None:
                pen_break_source = "untested:no_bin_stats"
            elif break_diag["status"] == "no_anomaly_found":
                # Test ran on enough bins and found nothing -- a genuine
                # non-detection, consistent with Rott et al. (2018).
                pen_break_source = "none_detected"
            else:
                # Not enough data to test (e.g. too few bins above the band
                # start). Not evidence of absence, but also no basis for a
                # correction -> none applied.
                pen_break_source = f"untested:{break_diag['status']}"

        # Slope-stratified gatekeeper: a detected break is only trusted if
        # it is not confined to steep terrain (layover/foreshortening can
        # mimic the same signature). Uses the same reference/test split.
        if (h_pen_min is not None and gatekeeper_inputs_ok):
            strat_mask = (stratified_glacier_mask if stratified_glacier_mask is not None
                          else glacier_mask)
            strat_result = detect_penetration_break_elevation_slope_stratified(
                dhdt_on_ice_raw, elev_bins_raster, slope_raster, strat_mask,
                min_bin_pixels=pen_detect_min_bin_pixels,
                residual_sigma_thresh=break_sigma_thresh,
                min_consecutive_bins=break_min_consecutive,
                core_max_elev=detect_kwargs["core_max_elev"],
                core_min_elev=detect_kwargs["core_min_elev"],
            )
            if strat_result["verdict"] == "slope_driven":
                print(f"      Penetration break at/above {h_pen_min:.0f} m looks slope-driven "
                      f"(steep classes only) -- treating as a geometry artefact, NOT "
                      f"applying a penetration correction. "
                      f"Per-class detail: {strat_result['classes']}")
                h_pen_min = None
                pen_break_source = "slope_artifact_excluded"
            elif strat_result["verdict"] == "mixed":
                print(f"      Penetration break at/above {h_pen_min:.0f} m: MIXED slope signal "
                      f"-- correction applied, but inspect the diagnostic plot. "
                      f"Per-class detail: {strat_result['classes']}")
            elif strat_result["verdict"] == "elevation_driven":
                print(f"      Penetration break at/above {h_pen_min:.0f} m confirmed across "
                      f"slope classes -- consistent with an elevation/snow-facies signal.")
            else:
                print(f"      Slope-stratified check for the break at/above {h_pen_min:.0f} m "
                      f"was {strat_result['verdict']} -- keeping the elevation-only result.")

        if break_diag is not None:
            print(f"      Penetration break-in-slope test "
                  f"(ref fit {break_diag['core_min_elev_m']:.0f}-{break_diag['core_cut_elev_m']:.0f} m, "
                  f"tested >= {break_diag['core_cut_elev_m']:.0f} m): "
                  f"status={break_diag['status']}  bins_ref={break_diag['n_bins_core']}  "
                  f"bins_tested={break_diag['n_bins_upper']}  "
                  f"anomalous={break_diag['n_upper_anomalous']}  "
                  f"max_z_tested={break_diag['max_z_upper']:.2f}  "
                  f"-> correction {'APPLIED' if h_pen_min is not None else 'NOT applied'} "
                  f"({pen_break_source})")
        else:
            print("      Penetration break-in-slope test: no bin stats -> correction NOT applied")

    # CSV-exportable label of what the slope-stratified gatekeeper did.
    if explicit_band and not require_break:
        pen_slope_stratified_verdict = "not_run_explicit_forced"
    elif strat_result is not None:
        pen_slope_stratified_verdict = strat_result["verdict"]
    elif h_pen_min is None:
        pen_slope_stratified_verdict = "not_run_no_break_detected"
    else:
        pen_slope_stratified_verdict = "not_run_inputs_missing"
    pen_slope_stratified_detail = (
        "|".join(f"{k}:{v['status']}" for k, v in strat_result["classes"].items())
        if strat_result else ""
    )

    if h_pen_min is None:
        # No explicit threshold, no detected anomaly -> assume zero
        # penetration bias rather than guessing at a threshold.
        pen_abs[gmask] = 0.0
        pen_rate[gmask] = 0.0
        meta = dict(empty_meta)
        meta.update({
            "N_glc_reg": int(np.sum(gmask)),
            "elv_max": float(h_pen_max_glacier) if np.isfinite(h_pen_max_glacier) else np.nan,
            "Mean_sar_pen": 0.0,
            "Mean_sar_pen_uncertainty_myr": 0.0,
            "Vol_sar_pen": 0.0,
            "pen_break_source": pen_break_source,
            "pen_break_diag": break_diag,
            "pen_gatekeeper_inputs_ok": gatekeeper_inputs_ok,
            "pen_slope_stratified_verdict": pen_slope_stratified_verdict,
            "pen_slope_stratified_detail": pen_slope_stratified_detail,
            "pen_correction_applied": False,
            "pen_band_tested_start_m": float(pen_start_elev_m) if explicit_band else np.nan,
            "pen_band_tested_end_m": float(pen_end_elev_m) if (explicit_band and pen_end_elev_m is not None) else np.nan,
            "pen_break_max_z_tested": (break_diag or {}).get("max_z_upper", np.nan),
        })
        return {"sar_pen_rate": pen_rate, "sar_pen_abs": pen_abs, "meta": meta}

    if np.isfinite(h_pen_min) and h_pen_min < 0:
        h_pen_min = 0.0

    h_pen_max = pen_end_elev_m if pen_end_elev_m is not None else h_pen_max_glacier

    # Base glacier cells to 0 absolute penetration before thresholding
    pen_abs[gmask] = 0.0

    if sar_pen_type == "constant":
        d_pen_min = sar_pen_bias
        d_pen_max = sar_pen_bias
        d_pen_slp = np.nan

        if np.isfinite(h_pen_min):
            affected = gmask & (ref_dem >= h_pen_min) & (ref_dem <= h_pen_max)
            pen_abs[affected] = sar_pen_bias

    elif sar_pen_type == "ramp":
        d_pen_min = 0.0
        d_pen_max = sar_pen_bias

        if np.isfinite(h_pen_min) and np.isfinite(h_pen_max) and h_pen_max > h_pen_min:
            d_pen_slp = d_pen_max / (h_pen_max - h_pen_min)
            # linear ramp between h_pen_min and h_pen_max, capped at d_pen_max above
            affected = gmask & (ref_dem >= h_pen_min)
            pen_abs[affected] = np.clip(
                (ref_dem[affected] - h_pen_min) * d_pen_slp, 0.0, d_pen_max
            )
        else:
            d_pen_slp = 0.0
            affected = gmask & np.isfinite(ref_dem)
            pen_abs[affected] = 0.0
    else:
        raise ValueError("sar_pen_type must be 'ramp' or 'constant'")

    dt_aff = dt_raster[affected & np.isfinite(dt_raster)]
    obs_per = np.nanmean(dt_aff) if dt_aff.size > 0 else np.nan

    valid_dt = gmask & np.isfinite(dt_raster) & (dt_raster > 0)
    pen_rate[valid_dt] = pen_abs[valid_dt] / dt_raster[valid_dt]
    no_dt = gmask & ~valid_dt
    if np.any(no_dt):
        pen_rate[no_dt] = np.nan

    mean_pen = np.nanmean(pen_rate[gmask]) if np.any(np.isfinite(pen_rate[gmask])) else np.nan
    area_glc_m2 = np.sum(gmask) * pixel_area_m2
    dvol_pen = mean_pen * area_glc_m2 if np.isfinite(mean_pen) else np.nan

    if np.isfinite(mean_pen):
        if apply_bias_correction:
            sar_pen_uncertainty = abs(mean_pen) * float(residual_unc_fraction)
        else:
            sar_pen_uncertainty = abs(mean_pen)
    else:
        sar_pen_uncertainty = np.nan

    return {
        "sar_pen_rate": pen_rate,
        "sar_pen_abs": pen_abs,
        "meta": {
            "N_sar_pen": int(np.sum(affected)),
            "N_glc_reg": int(np.sum(gmask)),
            "elv_min": float(h_pen_min) if np.isfinite(h_pen_min) else np.nan,
            "pen_min": float(d_pen_min) if np.isfinite(d_pen_min) else np.nan,
            "elv_max": float(h_pen_max) if np.isfinite(h_pen_max) else np.nan,
            "pen_max": float(d_pen_max) if np.isfinite(d_pen_max) else np.nan,
            "cor_slp": float(d_pen_slp) if np.isfinite(d_pen_slp) else np.nan,
            "Obs_mean_pen_cells": float(obs_per) if np.isfinite(obs_per) else np.nan,
            "Mean_sar_pen": float(mean_pen) if np.isfinite(mean_pen) else np.nan,
            "Mean_sar_pen_uncertainty_myr": float(sar_pen_uncertainty) if np.isfinite(sar_pen_uncertainty) else np.nan,
            "Vol_sar_pen": float(dvol_pen) if np.isfinite(dvol_pen) else np.nan,
            "pen_start_elev_m": float(h_pen_min) if np.isfinite(h_pen_min) else np.nan,
            "pen_end_elev_m": float(h_pen_max) if np.isfinite(h_pen_max) else np.nan,
            "pen_break_source": pen_break_source,
            "pen_break_diag": break_diag,
            "pen_gatekeeper_inputs_ok": gatekeeper_inputs_ok,
            "pen_slope_stratified_verdict": pen_slope_stratified_verdict,
            "pen_slope_stratified_detail": pen_slope_stratified_detail,
            "pen_correction_applied": bool(apply_bias_correction),
            "pen_ramp_defined": True,
            "pen_band_tested_start_m": float(pen_start_elev_m) if explicit_band else np.nan,
            "pen_band_tested_end_m": float(h_pen_max) if np.isfinite(h_pen_max) else np.nan,
            "pen_break_max_z_tested": (break_diag or {}).get("max_z_upper", np.nan),
        }
    }


def summarize_penetration_test_coverage(elev_bins, catchment_mask, observed_mask,
                                        detect_bin_stats, pixel_area_km2,
                                        band_start_m=None, min_bin_pixels=30):
    """
    per elevation bin, how much of the catchment exists and how much of it was
    observed (and is therefore visible to the break-in-slope test). The test
    can neither confirm nor exclude a penetration break in elevation bins that
    were never observed -- those values come from the hypsometric void filling
    (extrapolated from the observed bins). Returns {"table": DataFrame per bin,
    "summary": dict}.
    """
    eb = np.asarray(elev_bins, dtype=float)
    cm = np.asarray(catchment_mask, dtype=bool) & np.isfinite(eb)
    om = np.asarray(observed_mask, dtype=bool) & cm
    bins = np.sort(np.unique(eb[cm]))
    cat_n = np.array([np.sum(eb[cm] == b) for b in bins])
    obs_n = np.array([np.sum(eb[om] == b) for b in bins])
    tab = pd.DataFrame({"cl": bins,
                        "catchment_area_km2": cat_n * pixel_area_km2,
                        "observed_area_km2": obs_n * pixel_area_km2})
    tab["observed_pct"] = np.where(cat_n > 0, 100.0 * obs_n / np.maximum(cat_n, 1), np.nan)
    tested_bins = set()
    if detect_bin_stats is not None and not detect_bin_stats.empty:
        ok = (detect_bin_stats["n_fil"] >= min_bin_pixels) & np.isfinite(detect_bin_stats["mean_ch_fil"])
        tested_bins = set(detect_bin_stats.loc[ok, "cl"].values.tolist())
    tab["tested"] = tab["cl"].isin(tested_bins)
    max_tested = float(max(tested_bins)) if tested_bins else np.nan
    start = band_start_m if band_start_m is not None else -np.inf
    above = tab["cl"] >= start
    summ = {
        "catchment_max_bin_m": float(bins.max()) if bins.size else np.nan,
        "max_tested_bin_m": max_tested,
        "catchment_area_above_start_km2": float(tab.loc[above, "catchment_area_km2"].sum()),
        "observed_area_above_start_km2": float(tab.loc[above, "observed_area_km2"].sum()),
        "catchment_area_above_tested_km2": float(tab.loc[tab["cl"] > max_tested, "catchment_area_km2"].sum())
                                            if np.isfinite(max_tested) else np.nan,
        "catchment_area_total_km2": float(tab["catchment_area_km2"].sum()),
    }
    c = summ["catchment_area_above_start_km2"]
    summ["observed_pct_above_start"] = 100.0 * summ["observed_area_above_start_km2"] / c if c > 0 else np.nan
    return {"table": tab, "summary": summ}


def estimate_penetration_onset_from_period_consistency(all_changes, elev_bins,
                                                        out_dir=None):
    """
    Data-driven onset elevation z0 of the penetration-affected zone, from the
    internal consistency of three DEMs A -> B -> C (e.g. 2014, 2016/17,
    2024/25), replacing the ELA used by Braun et al. (2019).
    """
    from scipy import stats as _st

    def _key(y):
        return canonical_period_key(str(y))

    by_pair = {(_key(c["yA"]), _key(c["yB"])): c for c in all_changes}
    triplet = None
    for (a, b), cab in by_pair.items():
        for (b2, cc), cbc in by_pair.items():
            if b2 == b and (a, cc) in by_pair:
                triplet = (a, b, cc, cab, cbc)
                break
        if triplet:
            break
    out = {"z0": None, "status": "no_period_triplet", "table": None}
    if triplet is None:
        print("   ⚠️  Penetration onset: no A->B, B->C, A->C period triplet found.")
        return out
    a, b, cc, cab, cbc = triplet

    r1, t1 = np.asarray(cab["dhdt"], float), np.asarray(cab["dt"], float)
    r2, t2 = np.asarray(cbc["dhdt"], float), np.asarray(cbc["dt"], float)
    ok = (np.isfinite(r1) & np.isfinite(r2) & np.isfinite(t1) & np.isfinite(t2)
          & (t1 > 0) & (t2 > 0) & np.isfinite(elev_bins))
    with np.errstate(invalid="ignore", divide="ignore"):
        e = np.where(ok, t1 * t2 * (r1 - r2) / (t1 + t2), np.nan)

    df = pd.DataFrame({"cl": elev_bins[ok], "e": e[ok]})
    tab = (df.groupby("cl")["e"]
             .agg(n="size", e_median="median", e_mean="mean",
                  e_nmad=lambda v: 1.4826 * np.median(np.abs(v - np.median(v))))
             .reset_index().sort_values("cl"))
    out["table"] = tab
    out["triplet"] = f"{a} -> {b} -> {cc}"

    fit = tab[(tab["n"] >= cfg.SAR_PEN_DETECT_MIN_BIN_PIXELS)
              & (tab["cl"] >= cfg.SAR_PEN_ONSET_FIT_MIN_ELEV)]
    z, y = fit["cl"].values.astype(float), fit["e_median"].values.astype(float)
    w = fit["n"].values.astype(float)   # observed area per bin
    w = w / w.sum()
    sw = np.sqrt(w)
    if len(z) < 8:
        out["status"] = "too_few_bins"
        return out

    # area-weighted least squares (sparse top bins carry little weight)
    ybar = float(np.sum(w * y))
    sse0 = float(np.sum(w * (y - ybar) ** 2))
    best = None
    for z0 in z[1:-3]:
        X = np.column_stack([np.ones_like(z), np.maximum(0.0, z - z0)])
        coef, *_ = np.linalg.lstsq(X * sw[:, None], y * sw, rcond=None)
        sse = float(np.sum(w * (y - X @ coef) ** 2))
        if coef[1] > 0 and (best is None or sse < best[0]):
            best = (sse, z0, coef)
    if best is None:
        out["status"] = "no_positive_hinge"
        return out
    sse1, z0, coef = best
    n = len(z)
    # hinge adds 2 parameters (slope, z0) relative to the constant model
    F = ((sse0 - sse1) / 2.0) / (sse1 / max(n - 3, 1)) if sse1 > 0 else np.inf
    p = float(_st.f.sf(F, 2, max(n - 3, 1))) if np.isfinite(F) else 0.0
    amp = float(coef[1] * (z.max() - z0))
    out.update({"c": float(coef[0]), "slope_m_per_m": float(coef[1]),
                "amplitude_m": amp, "F": float(F), "p": p,
                "z0_candidate": float(z0)})
    if p < 0.01:
        out.update({"z0": float(z0), "status": "onset_detected"})
    else:
        out["status"] = "not_significant"

    print(f"   Penetration onset (inter-period consistency, {out['triplet']}): "
          f"status={out['status']}  z0={z0:.0f} m  offset c={coef[0]:+.2f} m  "
          f"slope={coef[1]*100:.2f} m/100 m  amplitude to {z.max():.0f} m={amp:.2f} m  "
          f"F={F:.1f}  p={p:.2g}")

    out.update({"labels": (a, b, cc), "fit_z_min": float(z.min()), "fit_z_max": float(z.max())})
    if out_dir is not None:
        os.makedirs(out_dir, exist_ok=True)
        tab.to_csv(os.path.join(out_dir, "penetration_onset_interperiod_consistency.csv"),
                   index=False)
    return out


def resolve_penetration_ramp(all_changes, elev_bins):
    """
    Set the onset (cfg.SAR_PEN_START_ELEV) and end (cfg.SAR_PEN_END_ELEV) of
    the penetration ramp used for all periods and return the onset estimate.
    """
    if isinstance(cfg.SAR_PEN_ONSET_MODE, str) and cfg.SAR_PEN_ONSET_MODE == "auto":
        onset = estimate_penetration_onset_from_period_consistency(
            all_changes, elev_bins, out_dir=os.path.join(cfg.OUT_DIR, "sar_penetration"))
        z0 = onset["z0"] if onset["z0"] is not None else cfg.SAR_PEN_ONSET_FALLBACK_M
        if onset["z0"] is None:
            print(f"   Penetration onset not detected ({onset['status']}) -> "
                  f"fallback z0 = {z0:.0f} m (Seehaus et al. 2023)")
    else:
        onset = {"z0": None, "slope_m_per_m": None}
        z0 = float(cfg.SAR_PEN_ONSET_MODE)

    if cfg.SAR_PEN_RAMP_WIDTH_M == "auto":
        slp = onset.get("slope_m_per_m") if onset.get("z0") is not None else None
        width = (cfg.SAR_PEN_BIAS / slp) if (slp is not None and slp > 0) else cfg.SAR_PEN_RAMP_WIDTH_FALLBACK_M
        width = float(np.clip(width, 300.0, 2000.0))
    else:
        width = float(cfg.SAR_PEN_RAMP_WIDTH_M)

    cfg.SAR_PEN_START_ELEV = z0
    cfg.SAR_PEN_END_ELEV = z0 + width
    print(f"   Penetration ramp for all periods: 0 m at {cfg.SAR_PEN_START_ELEV:.0f} m "
          f"-> {cfg.SAR_PEN_BIAS:.1f} m at {cfg.SAR_PEN_END_ELEV:.0f} m (constant above); "
          f"subtracted from dh/dt: {cfg.SAR_PEN_APPLY_BIAS_CORRECTION}")
    return onset
