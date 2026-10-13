"""Elevation-bin statistics of on-ice dh/dt and hypsometric void filling
(McNabb et al. 2019).
"""

import numpy as np
import pandas as pd

from . import config as cfg
from .utils import nan_median_filter


def filter_glacier_bin_r_style(values):
    """
    Per-bin outlier filter: keep values within the 2nd-98th percentile of the
    bin, applied to ALL bins regardless of pixel count.
    """
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]

    if values.size == 0:
        return values, np.nan, np.nan, "quan"

    lo = np.quantile(values, 0.02)
    hi = np.quantile(values, 0.98)

    filtered = values[(values >= lo) & (values <= hi)]
    return filtered, lo, hi, "quan"


def apply_r_style_tail_bin_cleanup(bin_stats, min_valid_n_raw=100):
    """
    Match the R logic that removes trailing highest bins with too few
    measurements before void filling.
    """
    if bin_stats is None or bin_stats.empty:
        return bin_stats

    out = bin_stats.copy()

    target_cols = [
        "mean_ch_raw", "medi_ch_raw", "sdev_ch_raw",
        "mean_ch_fil", "medi_ch_fil", "sdev_ch_fil"
    ]

    while True:
        valid_idx = np.where(np.isfinite(out["mean_ch_raw"].values))[0]
        if len(valid_idx) == 0:
            break

        last_idx = valid_idx[-1]
        n_raw_last = out.loc[last_idx, "n_raw"]

        if pd.isna(n_raw_last) or n_raw_last >= min_valid_n_raw:
            break

        out.loc[last_idx, target_cols] = np.nan

    return out


def fill_void_bins_mcnabb_style(bin_centers, values, poly_degree=3,
                                 min_coverage_frac=2.0 / 3.0):
    """McNabb et al. (2019) hypsometric void filling."""
    x = np.asarray(bin_centers, dtype=float)
    y = np.asarray(values,      dtype=float).copy()

    valid = np.isfinite(y)
    if not np.any(valid):
        return np.zeros_like(y, dtype=float)

    out = y.copy()
    if np.all(valid):
        return out  # nothing to fill

    full_range  = np.nanmax(x) - np.nanmin(x)
    valid_range = np.nanmax(x[valid]) - np.nanmin(x[valid])
    coverage_frac = (valid_range / full_range) if full_range > 0 else 0.0
    n_valid = int(np.sum(valid))

    # McNabb et al. (2019)'s own criterion: only trust a cubic fit if valid
    # bins cover at least two-thirds of the elevation range; need at least
    # poly_degree+1 points to fit a poly_degree-order polynomial at all.
    if coverage_frac >= min_coverage_frac and n_valid > poly_degree:
        # Centre elevations before fitting -- purely a numerical-conditioning
        # step (raw elevations in the thousands of metres make np.polyfit's
        # Vandermonde matrix ill-conditioned for cubic terms); does not change
        # the fitted curve itself.
        x0 = np.nanmean(x[valid])
        coeffs = np.polyfit(x[valid] - x0, y[valid], poly_degree)
        out[~valid] = np.polyval(coeffs, x[~valid] - x0)
    else:
        # Fallback -- see FALLBACK note above; not part of McNabb et al. (2019).
        out[~valid] = np.interp(x[~valid], x[valid], y[valid])

    return out


def compute_on_ice_bin_statistics_r_style(dhdt_on_ice, elev_bins, glacier_mask, min_pixels=cfg.MIN_BIN_PIXELS):
    """
    Hypsometric bin statistics following McNabb et al. (2019) / Sommer et al.
    (2022).
    """
    df = pd.DataFrame({
        "z":   dhdt_on_ice.ravel(),
        "bin": elev_bins.ravel(),
        "glc": glacier_mask.ravel()
    })

    df = df[(df["glc"] == 1) & np.isfinite(df["bin"])].copy()
    if df.empty:
        return pd.DataFrame()

    bin_ids = np.sort(df["bin"].dropna().unique())
    rows = []

    for b in bin_ids:
        sub = df[df["bin"] == b]
        n_ref = len(sub)

        raw_vals       = sub["z"].values
        raw_vals_valid = raw_vals[np.isfinite(raw_vals)]
        n_raw          = len(raw_vals_valid)

        # Require a minimum number of valid pixels for a robust bin mean
        if n_raw >= min_pixels:
            mean_raw = np.mean(raw_vals_valid)
            medi_raw = np.median(raw_vals_valid)
            sdev_raw = np.std(raw_vals_valid, ddof=1) if n_raw > 1 else np.nan

            fil_vals, lo, hi, fil_id = filter_glacier_bin_r_style(raw_vals_valid)
            n_fil = len(fil_vals)

            if n_fil >= min_pixels:
                mean_fil = np.mean(fil_vals)
                medi_fil = np.median(fil_vals)
                sdev_fil = np.std(fil_vals, ddof=1) if n_fil > 1 else np.nan
            else:
                # Reject if filtering destroyed the sample size
                n_fil, mean_fil, medi_fil, sdev_fil = 0, np.nan, np.nan, np.nan
                lo, hi, fil_id = np.nan, np.nan, "quan"
        else:
            # Reject bin as too noisy/sparse for regional projection
            mean_raw, medi_raw, sdev_raw = np.nan, np.nan, np.nan
            n_fil, mean_fil, medi_fil, sdev_fil = 0, np.nan, np.nan, np.nan
            lo, hi, fil_id = np.nan, np.nan, "quan"

        rows.append({
            "cl": b, "n_ref": n_ref, "n_raw": n_raw,
            "mean_ch_raw": mean_raw, "medi_ch_raw": medi_raw, "sdev_ch_raw": sdev_raw,
            "n_fil": n_fil, "mean_ch_fil": mean_fil, "medi_ch_fil": medi_fil,
            "sdev_ch_fil": sdev_fil,
            "fil_lo": lo, "fil_hi": hi, "fil_id": fil_id
        })

    bin_stats = pd.DataFrame(rows).sort_values("cl").reset_index(drop=True)

    # Clean extreme trailing edges (belt-and-suspenders alongside the
    # general min_pixels rejection above)
    bin_stats = apply_r_style_tail_bin_cleanup(bin_stats, min_valid_n_raw=min_pixels)

    # McNabb et al. (2019)-style void-bin fill: cubic polynomial fit when
    # valid-bin elevation coverage allows (their stated two-thirds criterion),
    # linear interpolation fallback otherwise; no post-fit smoothing or taper
    # in either case (see fill_void_bins_mcnabb_style).
    bin_stats["mean_ch_voi"] = fill_void_bins_mcnabb_style(
        bin_stats["cl"].values,
        bin_stats["mean_ch_fil"].values
    )
    bin_stats["medi_ch_voi"] = fill_void_bins_mcnabb_style(
        bin_stats["cl"].values,
        bin_stats["medi_ch_fil"].values
    )

    sdev_vals = bin_stats["sdev_ch_fil"].values.astype(float)
    mean_sdev = np.nanmean(sdev_vals) if np.any(np.isfinite(sdev_vals)) else np.nan
    sdev_voi  = sdev_vals.copy()
    if np.isfinite(mean_sdev):
        sdev_voi[~np.isfinite(sdev_voi)] = mean_sdev * 2.0
    bin_stats["sdev_ch_voi"] = sdev_voi

    return bin_stats


def summarize_regional_on_ice_change_r_style(bin_stats, pixel_area_m2):
    """
    Match calc_regional_change() from R.
    """
    if bin_stats is None or bin_stats.empty:
        return None

    n_ref_sum = np.nansum(bin_stats["n_ref"].values)

    n_raw_sum = np.nansum(bin_stats["n_raw"].values)
    mean_rg_raw = np.nansum(bin_stats["mean_ch_raw"] * bin_stats["n_raw"]) / n_raw_sum if n_raw_sum > 0 else np.nan
    medi_rg_raw = np.nansum(bin_stats["medi_ch_raw"] * bin_stats["n_raw"]) / n_raw_sum if n_raw_sum > 0 else np.nan
    sdev_rg_raw = np.nansum(bin_stats["sdev_ch_raw"] * bin_stats["n_raw"]) / n_raw_sum if n_raw_sum > 0 else np.nan
    scov_rg_raw = (n_raw_sum / n_ref_sum * 100.0) if n_ref_sum > 0 else np.nan

    n_fil_sum = np.nansum(bin_stats["n_fil"].values)
    mean_rg_fil = np.nansum(bin_stats["mean_ch_fil"] * bin_stats["n_fil"]) / n_fil_sum if n_fil_sum > 0 else np.nan
    medi_rg_fil = np.nansum(bin_stats["medi_ch_fil"] * bin_stats["n_fil"]) / n_fil_sum if n_fil_sum > 0 else np.nan
    sdev_rg_fil = np.nansum(bin_stats["sdev_ch_fil"] * bin_stats["n_fil"]) / n_fil_sum if n_fil_sum > 0 else np.nan
    scov_rg_fil = (n_fil_sum / n_ref_sum * 100.0) if n_ref_sum > 0 else np.nan

    n_voi_sum = n_ref_sum
    mean_rg_voi = np.nansum(bin_stats["mean_ch_voi"] * bin_stats["n_ref"]) / n_voi_sum if n_voi_sum > 0 else np.nan
    medi_rg_voi = np.nansum(bin_stats["medi_ch_voi"] * bin_stats["n_ref"]) / n_voi_sum if n_voi_sum > 0 else np.nan
    sdev_rg_voi = np.nansum(bin_stats["sdev_ch_voi"] * bin_stats["n_ref"]) / n_voi_sum if n_voi_sum > 0 else np.nan
    scov_rg_voi = 100.0 if n_voi_sum > 0 else np.nan

    return {
        "n_raw": n_raw_sum,
        "s_sqm_raw": n_raw_sum * pixel_area_m2,
        "s_per_raw": scov_rg_raw,
        "mean_m_raw": mean_rg_raw,
        "median_m_raw": medi_rg_raw,
        "sdev_m_raw": sdev_rg_raw,

        "n_fil": n_fil_sum,
        "s_sqm_fil": n_fil_sum * pixel_area_m2,
        "s_per_fil": scov_rg_fil,
        "mean_m_fil": mean_rg_fil,
        "median_m_fil": medi_rg_fil,
        "sdev_m_fil": sdev_rg_fil,

        "n_voi": n_voi_sum,
        "s_sqm_voi": n_voi_sum * pixel_area_m2,
        "s_per_voi": scov_rg_voi,
        "mean_m_voi": mean_rg_voi,
        "median_m_voi": medi_rg_voi,
        "sdev_m_voi": sdev_rg_voi
    }


def expand_bin_stats_for_void_fill(bin_stats, elev_bins, wide_mask):
    """
    compute_on_ice_bin_statistics_r_style's `bin_ids` (the set of elevation
    bins that get a row at all) comes from wherever `glacier_mask==1`, which at
    both call sites in calculate_mass_balance_following_r_logic is
    `obs_union_mask` -- the OBSERVED-support footprint (union of pixels
    actually covered by data this period), not the full glacier/catchment
    extent.
    """
    if bin_stats is None or bin_stats.empty:
        return bin_stats

    wide = np.asarray(wide_mask, dtype=bool)
    full_bin_ids = np.unique(elev_bins[wide & np.isfinite(elev_bins)])
    full_bin_ids = np.sort(full_bin_ids)
    if full_bin_ids.size == 0:
        return bin_stats

    base = bin_stats.set_index("cl")
    expanded = base.reindex(full_bin_ids)
    expanded.index.name = "cl"
    expanded = expanded.reset_index()

    expanded["mean_ch_voi"] = fill_void_bins_mcnabb_style(
        expanded["cl"].values,
        expanded["mean_ch_fil"].values if "mean_ch_fil" in expanded else expanded["cl"].values * np.nan,
    )
    expanded["medi_ch_voi"] = fill_void_bins_mcnabb_style(
        expanded["cl"].values,
        expanded["medi_ch_fil"].values if "medi_ch_fil" in expanded else expanded["cl"].values * np.nan,
    )

    n_new = int(expanded["cl"].isin(bin_stats["cl"]).eq(False).sum())
    if n_new > 0:
        print(f"      expand_bin_stats_for_void_fill: added {n_new} elevation bin(s) "
              f"absent from the observed-support footprint (present in the wider "
              f"mask instead) -- filled via cross-elevation interpolation for the "
              f"void-fill fallback only; regional summary stats are unaffected.")

    return expanded


def spatial_bilinear_void_fill(dhdt_current, glacier_mask, dhdt_raw_obs,
                                max_search_px=200, taper_frac=0.0,
                                fallback_idw_k=12, fallback_idw_power=2.0,
                                bin_mean_raster=None, random_seed=42):
    """
    Fill voids inside the glacier mask using spatial interpolation of the
    surrounding *observed* dhdt pixels, blended into the hypsometric fallback
    with a distance-based linear taper.
    """
    from scipy.interpolate import griddata
    from scipy.ndimage import distance_transform_edt

    out = dhdt_current.copy()
    glac = np.asarray(glacier_mask, dtype=bool)

    obs_mask  = glac & np.isfinite(dhdt_raw_obs)   # TRUE observed glacier pixels
    void_mask = glac & ~np.isfinite(dhdt_raw_obs)  # TRUE void glacier pixels

    if not np.any(obs_mask) or not np.any(void_mask):
        return out

    # Distance (in pixels) from each true void pixel to the nearest true
    # observed pixel. distance_transform_edt returns 0 where obs_mask is True.
    dist_to_obs = distance_transform_edt(~obs_mask)

    # Only attempt spatial fill / taper for true voids within the radius
    fill_candidates = void_mask & (dist_to_obs <= max_search_px)
    if not np.any(fill_candidates):
        return out

    obs_rows, obs_cols = np.where(obs_mask)
    obs_vals_raw       = dhdt_raw_obs[obs_rows, obs_cols]

    # ---- ANCHORING: switch to interpolating the anomaly-from-bin-mean ----
    anchored = bin_mean_raster is not None
    if anchored:
        obs_bin_mean = bin_mean_raster[obs_rows, obs_cols]
        valid_anom = np.isfinite(obs_bin_mean)
        # Only source points with a defined bin mean can be expressed as an
        # anomaly; the (normally rare) rest fall back to being excluded from
        # the anchored source cloud rather than poisoning it with NaNs.
        obs_rows_src = obs_rows[valid_anom]
        obs_cols_src = obs_cols[valid_anom]
        obs_src_vals = obs_vals_raw[valid_anom] - obs_bin_mean[valid_anom]  # anomaly
        anchored = anchored and len(obs_rows_src) > 0
    if not anchored:
        obs_rows_src = obs_rows
        obs_cols_src = obs_cols
        obs_src_vals = obs_vals_raw  # raw-value fallback path (no bin_mean_raster supplied)

    max_obs_pts = 200_000
    if len(obs_rows_src) > max_obs_pts:
        rng = np.random.default_rng(random_seed)
        idx = rng.choice(len(obs_rows_src), size=max_obs_pts, replace=False)
        obs_rows_src = obs_rows_src[idx]
        obs_cols_src = obs_cols_src[idx]
        obs_src_vals = obs_src_vals[idx]

    # Query points (true void pixels within the taper radius)
    voi_rows, voi_cols = np.where(fill_candidates)
    obs_pts = np.column_stack([obs_rows_src, obs_cols_src])
    query_pts = np.column_stack([voi_rows, voi_cols])

    # Linear interpolation where defined (inside the observed convex hull)
    interp_vals = griddata(
        points=obs_pts, values=obs_src_vals, xi=query_pts,
        method='linear', fill_value=np.nan,
    )

    # k-NN inverse-distance fallback outside the convex hull of the observations
    needs_fallback = ~np.isfinite(interp_vals)
    if np.any(needs_fallback):
        from scipy.spatial import cKDTree

        tree = cKDTree(obs_pts)
        fallback_query = query_pts[needs_fallback]
        k = min(fallback_idw_k, len(obs_pts))
        nn_dist, nn_idx = tree.query(fallback_query, k=k)
        if k == 1:
            nn_dist = nn_dist[:, None]
            nn_idx = nn_idx[:, None]

        # Guard against div-by-zero when a query pixel coincides exactly
        # with an observed pixel's location (shouldn't happen here since
        # query points are void pixels, but keep it safe/generic).
        nn_dist = np.maximum(nn_dist, 1e-6)
        w = 1.0 / nn_dist ** fallback_idw_power
        w /= w.sum(axis=1, keepdims=True)

        neighbour_vals = obs_src_vals[nn_idx]                 # (n_fallback, k)
        interp_vals[needs_fallback] = np.sum(w * neighbour_vals, axis=1)

    # ---- distance-based linear taper ----
    dist_at_void = dist_to_obs[voi_rows, voi_cols]
    taper_start_px = float(np.clip(taper_frac, 0.0, 1.0)) * max_search_px
    taper_span = max(max_search_px - taper_start_px, 1e-6)
    weight = np.where(
        dist_at_void <= taper_start_px,
        1.0,
        np.clip((max_search_px - dist_at_void) / taper_span, 0.0, 1.0),
    )

    have_interp = np.isfinite(interp_vals)

    if anchored:
        # blended = bin_mean + weight * interp_anomaly (dist=0: full local
        # anomaly; dist=max_search_px: bin mean)
        void_bin_mean = bin_mean_raster[voi_rows, voi_cols]
        can_anchor = have_interp & np.isfinite(void_bin_mean)
        hypso_vals = out[voi_rows, voi_cols]
        blended = np.where(
            can_anchor,
            void_bin_mean + weight * np.where(have_interp, interp_vals, 0.0),
            hypso_vals,  # this void pixel's own bin mean is undefined; leave as Step-1 left it
        )
    else:
        hypso_vals = out[voi_rows, voi_cols]
        blended = np.where(
            have_interp,
            weight * np.where(have_interp, interp_vals, 0.0) + (1.0 - weight) * hypso_vals,
            hypso_vals,
        )

    out[voi_rows, voi_cols] = blended

    return out


def boundary_median_smooth(dhdt_voi, dhdt_obs, glacier_mask, kernel=3):
    """
    Apply a `kernel`×`kernel` median filter exclusively to the narrow ring of
    pixels at the transition between observed and void-filled glacier cells.
    """
    from scipy.ndimage import binary_dilation

    glac     = np.asarray(glacier_mask, dtype=bool)
    obs_mask = glac & np.isfinite(dhdt_obs)
    voi_mask = glac & ~np.isfinite(dhdt_obs) & np.isfinite(dhdt_voi)

    if not np.any(obs_mask) or not np.any(voi_mask):
        return dhdt_voi.copy()

    # Boundary ring = observed pixels adjacent to at least one void pixel
    #               + void pixels adjacent to at least one observed pixel
    struct = np.ones((kernel, kernel), dtype=bool)
    obs_adjacent_to_void = obs_mask & binary_dilation(voi_mask, structure=struct)
    voi_adjacent_to_obs  = voi_mask & binary_dilation(obs_mask, structure=struct)
    boundary_ring = obs_adjacent_to_void | voi_adjacent_to_obs

    if not np.any(boundary_ring):
        return dhdt_voi.copy()

    # Run median filter over the full array (cheap; only write back on boundary)
    # NaN-aware: pixels outside the glacier / still-void are ignored rather
    # than entering the median as 0 (which biased edge pixels toward zero).
    smoothed = nan_median_filter(dhdt_voi, size=kernel)

    out = dhdt_voi.copy()
    out[boundary_ring] = smoothed[boundary_ring]
    return out


def apply_void_fill_and_smooth(dhdt_obs, elev_bins, glacier_mask,
                                onice_bin_stats,
                                spatial_search_px=200, boundary_kernel=3,
                                taper_frac=0.0,
                                fallback_idw_k=12, fallback_idw_power=2.0):
    """
    Void-filling pipeline. Only Step 1 (hypsometric fill, McNabb et al. 2019)
    runs by default; Steps 2 and 3 are switched off via
    VOID_FILL_SPATIAL_REFINEMENT / VOID_FILL_BOUNDARY_SMOOTH because they are
    not established methods in the cited literature.
    """
    glac = np.asarray(glacier_mask == 1, dtype=bool)

    # Bin-mean raster covering EVERY glacier pixel (observed and void alike),
    # not just void ones -- needed so Step 2 can express observed pixels as
    # anomalies relative to their own bin's mean (see ANCHORING in
    # spatial_bilinear_void_fill).
    bin_mean_raster = np.full(dhdt_obs.shape, np.nan, dtype=float)
    if onice_bin_stats is not None and not onice_bin_stats.empty:
        lut = dict(zip(onice_bin_stats["cl"].values,
                       onice_bin_stats["mean_ch_voi"].values))
        for b, fill_val in lut.items():
            mask_b = glac & np.isfinite(elev_bins) & (elev_bins == b)
            bin_mean_raster[mask_b] = fill_val

    # --- Step 1: hypsometric seed ---
    out = np.array(dhdt_obs, dtype=float, copy=True)
    void_overall = glac & ~np.isfinite(out)
    fillable = void_overall & np.isfinite(bin_mean_raster)
    out[fillable] = bin_mean_raster[fillable]

    out[~glac] = np.nan

    # --- Step 2: spatial refinement, anchored + distance-tapered ---
    # dhdt_obs (the ORIGINAL, pre-seed raster) is passed separately so this
    # step can tell true voids/observed pixels apart from what Step 1 wrote.
    if cfg.VOID_FILL_SPATIAL_REFINEMENT:
        out = spatial_bilinear_void_fill(out, glac, dhdt_obs,
                                          max_search_px=spatial_search_px,
                                          taper_frac=taper_frac,
                                          fallback_idw_k=fallback_idw_k,
                                          fallback_idw_power=fallback_idw_power,
                                          bin_mean_raster=bin_mean_raster)

    # --- Step 3: boundary median smoothing ---
    if cfg.VOID_FILL_BOUNDARY_SMOOTH:
        out = boundary_median_smooth(out, dhdt_obs, glac, kernel=boundary_kernel)

    return out
