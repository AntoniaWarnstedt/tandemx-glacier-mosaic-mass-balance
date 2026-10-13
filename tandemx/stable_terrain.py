"""Stable-terrain dh/dt, slope-binned stable-terrain error and planar tilt.
"""

import glob
import os
import warnings

import numpy as np
import pandas as pd
import rasterio
from scipy.ndimage import binary_dilation

from . import config as cfg
from .utils import (
    canonical_period_key,
    nan_median_filter,
    parse_period,
    remove_seam_ring,
    resample_to_grid,
)


def build_paper_stable_terrain_domain(
    glacier_mask_bool,
    rock_mask_bool,
    grid_info,
    buffer_distance_m=10000,
    water_mask_bool=None
):
    pixel_size = float(grid_info["resolution"])
    n_iter = int(np.ceil(buffer_distance_m / pixel_size))

    # Glacier + 10 km buffer search domain
    search_domain = binary_dilation(glacier_mask_bool.astype(bool), iterations=n_iter)

    # Stable terrain = within the 10km buffer AND is actually rock outcrop
    stable_domain = search_domain & rock_mask_bool

    if water_mask_bool is not None:
        stable_domain = stable_domain & (~water_mask_bool.astype(bool))

    return stable_domain


def build_variogram_terrain_domain(
    rock_mask_bool: np.ndarray,
    grid_info: dict,
    glacier_mask_bool: np.ndarray = None,
    steep_mask: np.ndarray = None,
    water_mask_bool: np.ndarray = None,
    slope_threshold: float = 45.0,
    preloaded_slope: np.ndarray = None,
) -> np.ndarray:
    """Full off-glacier stable terrain domain for variogram estimation ONLY."""
    domain = rock_mask_bool.astype(bool).copy()

    # Remove glacierized terrain
    if glacier_mask_bool is not None:
        domain &= ~glacier_mask_bool.astype(bool)

    # Remove water bodies
    if water_mask_bool is not None:
        domain &= ~water_mask_bool.astype(bool)

    # Remove steep slopes (>45°) — same exclusion used on the dh/dt rasters
    if steep_mask is not None:
        domain &= ~steep_mask.astype(bool)
    elif preloaded_slope is not None:
        domain &= ~(np.isfinite(preloaded_slope) & (preloaded_slope > slope_threshold))

    n_px = int(np.sum(domain))
    res  = float(grid_info["resolution"])
    print(f"      Variogram domain: {n_px:,} stable-terrain pixels "
          f"({n_px * res**2 * cfg.M2_TO_KM2:.1f} km²) — full rock-outcrop extent")
    return domain


def load_stable_dem_for_year(year_label, grid_info, STABLE_TERRAIN_DIR):
    """
    Load the stable-ground DEM (elevations on stable terrain, other pixels
    nodata) for one epoch from STABLE_TERRAIN_DIR.
    """
    key = canonical_period_key(year_label)
    candidates = sorted(glob.glob(os.path.join(STABLE_TERRAIN_DIR, "*.tif")))

    def _load(k):
        matches = [f for f in candidates if k in os.path.basename(f)]
        if not matches:
            return None, None
        arr = resample_to_grid(matches[0], grid_info, is_categorical=False)
        if arr is None:
            return None, matches[0]
        arr = np.asarray(arr, dtype=np.float64)
        arr[(arr == 0) | (arr < -50.0) | (arr > 3200.0)] = np.nan
        return arr, matches[0]

    arr, src = _load(key)
    if arr is not None:
        print(f"      Stable-ground DEM {key}: {os.path.basename(src)} "
              f"({int(np.isfinite(arr).sum()):,} valid px)")
        return arr

    parts = [p for p in key.split("-") if p]
    if len(parts) < 2:
        return None

    loaded = []
    for part in parts:
        a, src = _load(part)
        if a is not None:
            loaded.append(a)
            print(f"      Stable-ground DEM {key}: using {os.path.basename(src)} "
                  f"({int(np.isfinite(a).sum()):,} valid px)")
    if not loaded:
        return None
    if len(loaded) == 1:
        return loaded[0]
    stack = np.stack(loaded)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        merged = np.nanmean(stack, axis=0)
    print(f"      Stable-ground DEM {key}: merged {len(loaded)} yearly rasters "
          f"({int(np.isfinite(merged).sum()):,} valid px)")
    return merged


def build_stable_dhdt_from_yearly_stable_dems(
    yA, yB, year_results, grid_info, STABLE_TERRAIN_DIR
):
    """Build stable-terrain dh/dt from yearly stable-ground DEM rasters."""
    if cfg.STABLE_TERRAIN_SOURCE == "mosaic":
        return None
    if cfg.STABLE_TERRAIN_SOURCE == "stb_mosaic":
        return build_stable_dhdt_from_stb_mosaics(yA, yB, year_results)
    stableA = load_stable_dem_for_year(yA, grid_info, STABLE_TERRAIN_DIR)
    stableB = load_stable_dem_for_year(yB, grid_info, STABLE_TERRAIN_DIR)

    if stableA is None or stableB is None:
        return None

    if yA not in year_results or yB not in year_results:
        return None

    with rasterio.open(year_results[yA]["date_file"]) as src:
        dateA = src.read(1).astype(np.float32)
        nodataA = src.nodata
        if nodataA is not None:
            dateA[dateA == nodataA] = np.nan

    with rasterio.open(year_results[yB]["date_file"]) as src:
        dateB = src.read(1).astype(np.float32)
        nodataB = src.nodata
        if nodataB is not None:
            dateB[dateB == nodataB] = np.nan

    dh = stableB - stableA
    dt = dateB - dateA

    yA_parsed = parse_period(yA)
    yB_parsed = parse_period(yB)
    simple_dt = float(yB_parsed[1]) - float(yA_parsed[0])

    dt = np.where(~np.isfinite(dt) | (np.abs(dt) < 0.1), simple_dt, dt)

    with np.errstate(invalid='ignore', divide='ignore'):
        dhdt = np.where(np.abs(dt) > 1e-6, dh / dt, np.nan)

    return dhdt


def build_stable_dhdt_from_stb_mosaics(yA, yB, year_results):
    """
    Stable-terrain dh/dt from the stable-ground mosaics built alongside the DEM
    mosaics (mosaic_year_v2 / merge_multiyear_dems): same strips, same
    footprints, same de-ramp corrections, same merge. dt from their own per-
    pixel acquisition dates. Same dh/dt filtering as the glacier field (+-30 m
    a-1 clip, 3x3 NaN-aware median, seam-ring removal). Returns None if either
    mosaic is missing (caller falls back).
    """
    rA, rB = year_results.get(yA, {}), year_results.get(yB, {})
    if not (rA.get('stb_dem_file') and rB.get('stb_dem_file')):
        missing = yA if not rA.get('stb_dem_file') else yB
        print(f"      ⚠️  No stable-ground mosaic for {missing}.")
        return None

    def _rd(path):
        with rasterio.open(path) as src:
            a = src.read(1).astype(np.float64)
            if src.nodata is not None:
                a[a == src.nodata] = np.nan
        return a

    stA, stB = _rd(rA['stb_dem_file']), _rd(rB['stb_dem_file'])
    dA, dB = _rd(rA['stb_date_file']), _rd(rB['stb_date_file'])
    dh = stB - stA
    dt = dB - dA
    simple_dt = float(parse_period(yB)[1]) - float(parse_period(yA)[0])
    dt = np.where(~np.isfinite(dt) | (np.abs(dt) < 0.1), simple_dt, dt)
    with np.errstate(invalid='ignore', divide='ignore'):
        dhdt = np.where(np.isfinite(dh), dh / dt, np.nan)
    dhdt = np.clip(dhdt, -30, 30)
    dhdt = nan_median_filter(dhdt, size=3)
    dhdt = remove_seam_ring(dhdt, cfg.SEAM_RING_REMOVE_PX)
    dhdt = np.clip(dhdt, -30, 30)
    print(f"      Stable-ground mosaics {yA} / {yB}: {int(np.isfinite(dhdt).sum()):,} common px")
    return dhdt.astype(np.float32)


def write_stable_by_strip_table(yA, yB, year_results, stable_dhdt, dhdt_dem,
                                rock_mask, out_dir):
    """
    Diagnostic: stable dh/dt grouped by the strip pair (epoch A strip, epoch B
    strip) that dominates each stable pixel. Also reports the dh/dt of the
    glacier DEM mosaics on the same pixels where they have data, to show
    whether the stable-ground mosaics reproduce the DEM mosaics there. Writes
    stable_by_strip_<yB>_minus_<yA>.csv.
    """
    rA, rB = year_results.get(yA, {}), year_results.get(yB, {})
    if not (rA.get('stb_id_file') and rB.get('stb_id_file')):
        return

    def _rd(path):
        with rasterio.open(path) as src:
            a = src.read(1).astype(np.float64)
            if src.nodata is not None:
                a[a == src.nodata] = np.nan
        return a

    idA, idB = _rd(rA['stb_id_file']), _rd(rB['stb_id_file'])
    ok = np.isfinite(stable_dhdt) & np.isfinite(idA) & np.isfinite(idB)
    if not ok.any():
        return
    df = pd.DataFrame({"idA": idA[ok].astype(int), "idB": idB[ok].astype(int),
                       "stb": stable_dhdt[ok].astype(np.float64),
                       "dem": np.asarray(dhdt_dem, dtype=np.float64)[ok],
                       "rock": np.asarray(rock_mask, dtype=bool)[ok]})

    def _nmad(v):
        v = np.asarray(v, float)
        v = v[np.isfinite(v)]
        return float(1.4826 * np.median(np.abs(v - np.median(v)))) if v.size else np.nan

    rows = []
    for (a, b), g in df.groupby(["idA", "idB"]):
        both = g[np.isfinite(g.dem)]
        rows.append({"idA": a, "idB": b, "n_stable": len(g), "n_on_rock": int(g.rock.sum()),
                     "stable_median_myr": float(g.stb.median()), "stable_nmad_myr": _nmad(g.stb),
                     "n_with_dem": len(both),
                     "stable_median_on_dem_px_myr": float(both.stb.median()) if len(both) else np.nan,
                     "dem_mosaic_median_same_px_myr": float(both.dem.median()) if len(both) else np.nan})
    tab = pd.DataFrame(rows)
    names = {}
    for r in (rA, rB):
        t = r.get('stb_table')
        if t is not None:
            names.update(dict(zip(t["id"], t["strip"])))
    tab.insert(1, "stripA", tab.idA.map(names))
    tab.insert(3, "stripB", tab.idB.map(names))
    tab = tab.sort_values("n_stable", ascending=False)
    out = os.path.join(out_dir, f"stable_by_strip_{yB}_minus_{yA}.csv")
    tab.to_csv(out, index=False)
    both = df[np.isfinite(df.dem)]
    print(f"   ✓ Saved: {os.path.basename(out)} ({len(tab)} strip pairs)")
    if len(both):
        print(f"      On {len(both):,} stable px also covered by the DEM mosaics: median dh/dt "
              f"stable-ground mosaic {both.stb.median():+.2f} vs DEM mosaic {both.dem.median():+.2f} m a-1")


def filter_stable_bin_r_style(values):
    """
    Match the paper method more closely for stable terrain:
    - hard filter abs(z) <= 50
    - quantile filter 1%-99%
    """
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    values = values[(values >= -50.0) & (values <= 50.0)]

    if values.size == 0:
        return values, np.nan, np.nan

    lo = np.quantile(values, 0.01)
    hi = np.quantile(values, 0.99)
    filtered = values[(values >= lo) & (values <= hi)]
    return filtered, lo, hi


def compute_stable_slope_error_r_style(
    stable_dhdt,
    slope_bins,
    glacier_mask,
    stable_domain_mask,
    pixel_area_m2
):
    """Compute paper-style slope-binned stable-terrain uncertainty."""
    df = pd.DataFrame({
        "z": stable_dhdt.ravel(),
        "bin": slope_bins.ravel(),
        "glc": glacier_mask.ravel(),
        "stable_ok": stable_domain_mask.ravel().astype(bool)
    })

    df = df[np.isfinite(df["bin"])].copy()
    if df.empty:
        return pd.DataFrame(), None

    bin_ids = np.sort(df["bin"].dropna().unique())
    rows = []

    for b in bin_ids:
        sub = df[df["bin"] == b].copy()

        # glacier-pixel count in this slope bin for weighting
        b_gAre = np.sum(sub["glc"] == 1)

        # stable terrain only inside the paper-style domain
        sub_stable = sub[(sub["glc"] == 0) & (sub["stable_ok"])].copy()

        raw_vals = sub_stable["z"].values
        raw_vals = raw_vals[np.isfinite(raw_vals)]

        n_raw = len(raw_vals)
        mean_raw = np.mean(raw_vals) if n_raw > 0 else np.nan
        medi_raw = np.median(raw_vals) if n_raw > 0 else np.nan
        sdev_raw = np.std(raw_vals, ddof=1) if n_raw > 1 else np.nan

        fil_vals, lo, hi = filter_stable_bin_r_style(raw_vals)

        n_fil = len(fil_vals)
        mean_fil = np.mean(fil_vals) if n_fil > 0 else np.nan
        medi_fil = np.median(fil_vals) if n_fil > 0 else np.nan
        sdev_fil = np.std(fil_vals, ddof=1) if n_fil > 1 else np.nan

        rows.append({
            "cl": b,
            "n_glc": b_gAre,
            "n_raw": n_raw,
            "mean_raw": mean_raw,
            "medi_raw": medi_raw,
            "sdev_raw": sdev_raw,
            "n_fil": n_fil,
            "mean_fil": mean_fil,
            "medi_fil": medi_fil,
            "sdev_fil": sdev_fil,
            "fil_lo": lo,
            "fil_hi": hi
        })

    bin_stats = pd.DataFrame(rows).sort_values("cl").reset_index(drop=True)
    reg_error = summarize_stable_error_r_style(bin_stats, pixel_area_m2)
    return bin_stats, reg_error


def summarize_stable_error_r_style(bin_stats, pixel_area_m2):
    """
    Match calc_regional_error() from R.
    """
    if bin_stats is None or bin_stats.empty:
        return None

    n_raw_all_sum = np.nansum(bin_stats["n_raw"])
    mean_all_raw = np.nansum(bin_stats["mean_raw"] * bin_stats["n_raw"]) / n_raw_all_sum if n_raw_all_sum > 0 else np.nan
    medi_all_raw = np.nansum(bin_stats["medi_raw"] * bin_stats["n_raw"]) / n_raw_all_sum if n_raw_all_sum > 0 else np.nan
    sdev_all_raw = np.nansum(bin_stats["sdev_raw"] * bin_stats["n_raw"]) / n_raw_all_sum if n_raw_all_sum > 0 else np.nan

    n_fil_all_sum = np.nansum(bin_stats["n_fil"])
    mean_all_fil = np.nansum(bin_stats["mean_fil"] * bin_stats["n_fil"]) / n_fil_all_sum if n_fil_all_sum > 0 else np.nan
    medi_all_fil = np.nansum(bin_stats["medi_fil"] * bin_stats["n_fil"]) / n_fil_all_sum if n_fil_all_sum > 0 else np.nan
    sdev_all_fil = np.nansum(bin_stats["sdev_fil"] * bin_stats["n_fil"]) / n_fil_all_sum if n_fil_all_sum > 0 else np.nan

    n_glc_all_sum = np.nansum(bin_stats["n_glc"])
    mean_all_raw_gw = np.nansum(bin_stats["mean_raw"] * bin_stats["n_glc"]) / n_glc_all_sum if n_glc_all_sum > 0 else np.nan
    medi_all_raw_gw = np.nansum(bin_stats["medi_raw"] * bin_stats["n_glc"]) / n_glc_all_sum if n_glc_all_sum > 0 else np.nan
    sdev_all_raw_gw = np.nansum(bin_stats["sdev_raw"] * bin_stats["n_glc"]) / n_glc_all_sum if n_glc_all_sum > 0 else np.nan

    mean_all_fil_gw = np.nansum(bin_stats["mean_fil"] * bin_stats["n_glc"]) / n_glc_all_sum if n_glc_all_sum > 0 else np.nan
    medi_all_fil_gw = np.nansum(bin_stats["medi_fil"] * bin_stats["n_glc"]) / n_glc_all_sum if n_glc_all_sum > 0 else np.nan
    sdev_all_fil_gw = np.nansum(bin_stats["sdev_fil"] * bin_stats["n_glc"]) / n_glc_all_sum if n_glc_all_sum > 0 else np.nan

    return {
        "n_raw_all": n_raw_all_sum,
        "s_sqm_raw_all": n_raw_all_sum * pixel_area_m2,
        "mean_raw_all": mean_all_raw,
        "mean_raw_all_gw": mean_all_raw_gw,
        "median_raw_all": medi_all_raw,
        "median_raw_all_gw": medi_all_raw_gw,
        "sdev_raw_all": sdev_all_raw,
        "sdev_raw_all_gw": sdev_all_raw_gw,

        "n_fil_all": n_fil_all_sum,
        "s_sqm_fil_all": n_fil_all_sum * pixel_area_m2,
        "mean_fil_all": mean_all_fil,
        "mean_fil_all_gw": mean_all_fil_gw,
        "median_fil_all": medi_all_fil,
        "median_fil_all_gw": medi_all_fil_gw,
        "sdev_fil_all": sdev_all_fil,
        "sdev_fil_all_gw": sdev_all_fil_gw,
    }


def build_sigma_map_from_slope_bins(
        slope_bins_raster: np.ndarray,
        stable_bin_stats: pd.DataFrame,
) -> np.ndarray:
    """Project the per-slope-bin σ_dh back onto the raster grid."""
    sigma_map = np.full(slope_bins_raster.shape, np.nan, dtype=np.float64)

    if stable_bin_stats is None or stable_bin_stats.empty:
        return sigma_map

    bin_centres = stable_bin_stats['cl'].values.astype(float)
    sdev_vals = stable_bin_stats['sdev_fil'].values.astype(float)

    # Fill missing σ values by linear interpolation between known bins
    valid_mask = np.isfinite(sdev_vals)
    if not np.any(valid_mask):
        return sigma_map

    if not np.all(valid_mask):
        sdev_interp = np.interp(
            bin_centres,
            bin_centres[valid_mask],
            sdev_vals[valid_mask],
        )
    else:
        sdev_interp = sdev_vals.copy()

    # Assign per-pixel σ by matching slope-bin ID
    for cl, sv in zip(bin_centres, sdev_interp):
        if np.isfinite(sv) and sv > 0:
            pix_mask = (slope_bins_raster == cl)
            sigma_map[pix_mask] = sv

    # Pixels inside a slope-bin area but with no σ estimate → global mean
    global_mean = np.nanmean(sdev_interp[np.isfinite(sdev_interp)])
    if np.isfinite(global_mean) and global_mean > 0:
        no_sigma = np.isfinite(slope_bins_raster) & ~np.isfinite(sigma_map)
        sigma_map[no_sigma] = global_mean

    return sigma_map


def fit_stable_tilt(stable_dhdt, fit_mask, glacier_mask, grid_info,
                    n_boot=200, block_m=10_000.0, random_seed=42):
    """
    Robust plane dh/dt = a + b*x + c*y fitted to stable terrain (fit_mask),
    iteratively excluding |res - median| > 3 NMAD (5 passes). Coordinates in km
    relative to the glacier-pixel centroid, so `a` is the plane averaged over
    the glacier (the plane is linear). Its uncertainty comes from a block
    bootstrap (blocks of block_m, which respects spatial correlation).
    """
    tr = grid_info["transform"]
    rr, cc = np.nonzero(np.isfinite(stable_dhdt) & fit_mask)
    if rr.size < 500:
        print(f"      ⚠️ Stable tilt: only {rr.size} px -- not fitted.")
        return None
    gr, gc = np.nonzero(glacier_mask)
    if gr.size == 0:
        return None
    x0 = tr.c + (gc.mean() + 0.5) * tr.a
    y0 = tr.f + (gr.mean() + 0.5) * tr.e
    x = tr.c + (cc + 0.5) * tr.a
    y = tr.f + (rr + 0.5) * tr.e
    xk, yk = (x - x0) / 1e3, (y - y0) / 1e3
    v = stable_dhdt[rr, cc].astype(np.float64)
    keep = np.ones(v.size, bool)
    for _ in range(5):
        A = np.column_stack([np.ones(keep.sum()), xk[keep], yk[keep]])
        co, *_ = np.linalg.lstsq(A, v[keep], rcond=None)
        res = v - (co[0] + co[1] * xk + co[2] * yk)
        med = np.median(res[keep])
        nmad = 1.4826 * np.median(np.abs(res[keep] - med))
        keep = np.abs(res - med) <= 3.0 * nmad
    rng = np.random.default_rng(random_seed)
    bid = np.floor(x / block_m).astype(np.int64) * 1_000_003 + np.floor(y / block_m).astype(np.int64)
    kb = bid[keep]
    ki = np.nonzero(keep)[0]
    order = np.argsort(kb, kind="stable")
    ub, starts = np.unique(kb[order], return_index=True)
    groups = np.split(ki[order], starts[1:])
    boots = []
    for _ in range(int(n_boot)):
        pick = np.concatenate([groups[j] for j in rng.integers(0, len(groups), len(groups))])
        A = np.column_stack([np.ones(pick.size), xk[pick], yk[pick]])
        cb, *_ = np.linalg.lstsq(A, v[pick], rcond=None)
        boots.append(cb[0])
    se = float(np.std(boots)) if boots else np.nan
    ny, nx = grid_info["shape"]
    xs = (tr.c + (np.arange(nx) + 0.5) * tr.a - x0) / 1e3
    ys = (tr.f + (np.arange(ny) + 0.5) * tr.e - y0) / 1e3
    plane = (co[0] + co[1] * xs[None, :] + co[2] * ys[:, None]).astype(np.float32)
    return {"a": float(co[0]), "b_per_km": float(co[1]), "c_per_km": float(co[2]),
            "glacier_mean_myr": float(co[0]), "glacier_se_myr": se,
            "n_px": int(keep.sum()), "n_blocks": int(len(groups)),
            "nmad_raw": float(1.4826 * np.median(np.abs(v[keep] - np.median(v[keep])))),
            "nmad_detrended": float(nmad), "plane": plane}
