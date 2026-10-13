"""TanDEM-X strip mosaicking: edge erosion, iterative de-ramping to the growing
mosaic, feathered blending, merging of multi-year epochs and the matching
stable-ground mosaics.
"""

import glob
import os
import re

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from rasterio.features import rasterize
from scipy import stats

from . import config as cfg
from .utils import (
    apply_mask_and_validate,
    apply_year_specific_mask,
    get_decimal_date,
    prefilter_dem,
    resample_to_grid,
    save_raster,
)


def erode_strip_edges(data: np.ndarray, n_pixels: int) -> np.ndarray:
    """Remove artifact pixels at the valid/nodata boundary of a DEM strip."""
    if n_pixels <= 0:
        return data

    from scipy.ndimage import binary_erosion, generate_binary_structure

    data = data.copy()
    valid = np.isfinite(data)
    if not np.any(valid):
        return data

    struct = generate_binary_structure(2, 2)  # 8-connected
    eroded = binary_erosion(valid, structure=struct,
                            iterations=n_pixels, border_value=0)
    data[valid & ~eroded] = np.nan
    return data


def fit_ramp_to_difference(diff, x_coords, y_coords, order=2, sample_n=20000,
                           min_pts=500, mad_thresh=3.0, random_seed=42):
    """
    Fit a low-order polynomial ramp surface to the elevation difference between
    two overlapping DEM strips, then subtract it from the slave strip.
    """
    valid_mask = np.isfinite(diff)
    n_valid = np.sum(valid_mask)

    if n_valid < min_pts:
        return None

    valid_indices = np.where(valid_mask)
    if n_valid > sample_n:
        rng = np.random.default_rng(random_seed)
        sample_idx = rng.choice(n_valid, size=sample_n, replace=False)
    else:
        sample_idx = np.arange(n_valid)

    rows_samp = valid_indices[0][sample_idx]
    cols_samp = valid_indices[1][sample_idx]

    x_samp = x_coords[rows_samp, cols_samp]
    y_samp = y_coords[rows_samp, cols_samp]
    z_samp = diff[rows_samp, cols_samp]

    med_val = np.nanmedian(z_samp)
    mad_val = stats.median_abs_deviation(z_samp, nan_policy='omit')  # scale=1.0, no 1.4826 factor
    if mad_val > 0:
        keep = np.abs(z_samp - med_val) < mad_thresh * mad_val
        x_samp = x_samp[keep]
        y_samp = y_samp[keep]
        z_samp = z_samp[keep]

    if len(z_samp) < min_pts:
        return None

    x_center = np.mean(x_samp)
    y_center = np.mean(y_samp)
    x_scale  = np.std(x_samp) or 1.0
    y_scale  = np.std(y_samp) or 1.0

    xn = (x_samp - x_center) / x_scale
    yn = (y_samp - y_center) / y_scale

    if order == 1:
        X = np.column_stack([np.ones_like(xn), xn, yn])
    else:
        X = np.column_stack([np.ones_like(xn), xn, yn,
                             xn**2, yn**2, xn * yn])

    try:
        coefs, _, _, _ = np.linalg.lstsq(X, z_samp, rcond=None)
    except Exception:
        return None

    if np.any(np.isnan(coefs)):
        return None

    return {
        'coefs':    coefs,
        'x_center': x_center, 'y_center': y_center,
        'x_scale':  x_scale,  'y_scale':  y_scale,
        'order':    order,
        'n_pts':    len(z_samp),
        'residual_std': float(np.std(z_samp - X @ coefs)),
    }


def apply_ramp_correction(data, x_coords, y_coords, ramp_params):
    """Subtract the fitted ramp surface from a DEM strip."""
    if ramp_params is None:
        return data

    xn = (x_coords - ramp_params['x_center']) / ramp_params['x_scale']
    yn = (y_coords - ramp_params['y_center']) / ramp_params['y_scale']
    coefs = ramp_params['coefs']

    if ramp_params['order'] == 1:
        correction = coefs[0] + coefs[1]*xn + coefs[2]*yn
    else:
        correction = (coefs[0] + coefs[1]*xn + coefs[2]*yn
                      + coefs[3]*xn**2 + coefs[4]*yn**2 + coefs[5]*xn*yn)

    return data - correction


def feather_weights(data, width_px=cfg.FEATHER_WIDTH_PX):
    """
    Feathering weights for mosaicking: each valid pixel gets a weight that
    increases linearly with its distance to the nearest invalid pixel (strip
    edge or internal void), from 1/width_px at the edge to 1 at >= width_px.
    Invalid pixels get 0. Down-weights strip edges, where phase-unwrapping and
    geocoding errors concentrate, and blends seams smoothly.
    """
    from scipy.ndimage import distance_transform_edt
    valid = np.isfinite(data)
    if not np.any(valid):
        return np.zeros(data.shape, dtype=np.float64)
    if width_px is None or width_px <= 0:
        return valid.astype(np.float64)
    dist = distance_transform_edt(valid)          # 1 at edge pixels
    w = np.clip(dist / float(width_px), 0.0, 1.0)
    w[~valid] = 0.0
    return w


def deramp_strip_to_mosaic(strip, current_mosaic, x_coords, y_coords):
    """Vertically adjust `strip` to `current_mosaic` using their overlap."""
    strip = np.array(strip, dtype=np.float64, copy=True)
    overlap = np.isfinite(strip) & np.isfinite(current_mosaic)
    n_overlap = int(np.sum(overlap))
    steps = []

    def _median_shift(tag):
        shift = float(np.nanmedian((strip - current_mosaic)[overlap]))
        if np.isfinite(shift):
            steps.append(("shift", shift))
            return strip - shift, f"{tag} ({shift:+.2f} m, {n_overlap} overlap px)", steps
        return strip, f"no valid shift, adding as-is ({n_overlap} overlap px)", steps

    if n_overlap > cfg.DERAMP_MIN_PTS:
        n_iter, last_rms, last_params = 0, np.nan, None
        for it in range(cfg.DERAMP_MAX_ITER):
            diff = np.where(overlap, strip - current_mosaic, np.nan)
            params = fit_ramp_to_difference(
                diff, x_coords, y_coords,
                order=cfg.DERAMP_ORDER, sample_n=cfg.DERAMP_SAMPLE,
                min_pts=cfg.DERAMP_MIN_PTS, mad_thresh=cfg.MAD_THRESHOLD,
                random_seed=42 + it,
            )
            if params is None:
                break
            corrected = apply_ramp_correction(strip, x_coords, y_coords, params)
            increment = (strip - corrected)[overlap]
            strip = corrected
            steps.append(("ramp", params))
            n_iter, last_params = it + 1, params
            last_rms = float(np.sqrt(np.nanmean(increment ** 2)))
            if last_rms < cfg.DERAMP_CONVERGENCE_M:
                break
        if n_iter == 0:
            return _median_shift("ramp fit failed -> median shift")
        return strip, (f"quadratic de-ramp, {n_iter} iteration(s), last increment "
                       f"RMS={last_rms:.3f} m, residual std="
                       f"{last_params['residual_std']:.3f} m ({n_overlap} overlap px)"), steps
    if n_overlap > cfg.MEDIAN_SHIFT_MIN_PTS:
        return _median_shift("median vertical shift")
    return strip, f"added as-is ({n_overlap} overlap px)", steps


def apply_correction_steps(arr, x_coords, y_coords, steps):
    """Apply the de-ramp / shift steps returned by deramp_strip_to_mosaic."""
    if arr is None:
        return None
    out = np.array(arr, dtype=np.float64, copy=True)
    for kind, val in steps:
        if kind == "ramp":
            out = apply_ramp_correction(out, x_coords, y_coords, val)
        elif kind == "shift":
            out = out - val
    return out


def _date_time_key(path):
    m = re.search(r'(20[0-9]{2}[-_][0-9]{2}[-_][0-9]{2}_[0-9]{6})', os.path.basename(path))
    return m.group(1) if m else None


def find_stable_strip(dem_file, year_path):
    """Per-strip stable-ground raster: <year_path>/../<STB_SUBDIR>/DEM_STB_<date_time>*.tif"""
    key = _date_time_key(dem_file)
    if key is None:
        return None
    stb_dir = os.path.join(os.path.dirname(os.path.normpath(year_path)), cfg.STB_SUBDIR)
    cands = sorted(glob.glob(os.path.join(stb_dir, f"{cfg.STB_PREFIX}*{key}*.tif")))
    if not cands:
        return None
    if len(cands) > 1:
        # prefer the candidate sharing the longest name core with the DEM strip
        core = os.path.basename(dem_file).split(key, 1)[-1].replace(".tif", "")
        core = re.sub(r'_clip(_\d+)?$', '', core)
        cands.sort(key=lambda c: (core not in os.path.basename(c), c))
    return cands[0]


def load_strip_footprint(dem_file, year_label, grid_info, raw_strip=None):
    """
    Footprint of a strip as it went into the mosaic (boolean grid mask).
    1) STB_FOOTPRINT_DIR/<DEM basename>_footprint.shp, else any *<date_time>*footprint*.shp
    2) fallback: valid area of the raw DEM strip with interior holes filled.
    """
    from scipy.ndimage import binary_fill_holes
    fdir = cfg.STB_FOOTPRINT_DIR.format(year=year_label)
    base = os.path.splitext(os.path.basename(dem_file))[0]
    cand = os.path.join(fdir, base + "_footprint.shp")
    if not os.path.exists(cand):
        key = _date_time_key(dem_file)
        hits = sorted(glob.glob(os.path.join(fdir, f"*{key}*footprint*.shp"))) if key else []
        hits.sort(key=lambda h: (base not in os.path.basename(h), h))
        cand = hits[0] if hits else None
    if cand is not None and os.path.exists(cand):
        try:
            gdf = gpd.read_file(cand)
            if gdf.crs is not None:
                gdf = gdf.to_crs(rasterio.CRS.from_user_input(grid_info["crs"]).to_wkt())
            geoms = [g for g in gdf.geometry if g is not None and not g.is_empty]
            if geoms:
                fp = rasterize(geoms, out_shape=grid_info["shape"],
                               transform=grid_info["transform"], fill=0,
                               default_value=1, dtype=np.uint8).astype(bool)
                return fp, os.path.basename(cand)
        except Exception as e:
            print(f"      ⚠️  Footprint {os.path.basename(cand)} unreadable ({e}) -- using filled data area")
    if raw_strip is None:
        return None, None
    return binary_fill_holes(np.isfinite(raw_strip)), "filled DEM data area"


def mosaic_year_v2(year_path, year_label, grid_info, bad_mask, x_coords, y_coords):
    """
    Mosaic all DEM strips of one year by iterative vertical de-ramping with
    feathered (distance-to-edge weighted) overlap blending.
    """
    print("=" * 50)
    print(f"▶ Processing year {year_label}")

    dem_files = glob.glob(os.path.join(year_path, "DEM*.tif"))
    if not dem_files:
        print("   ! No DEMs found.")
        return None

    print(f"   Found {len(dem_files)} DEM files")

    strips_data = []
    strips_dates = []
    strips_files = []
    strips_stb = []       # per-strip stable-ground raster (clipped to footprint) or None
    strips_info = []      # per-strip diagnostics (-> stb_strips_<label>.csv)
    strips_fp = []        # per-strip footprint mask or None
    use_stb = (cfg.STABLE_TERRAIN_SOURCE == "stb_mosaic")

    for f in dem_files:
        try:
            data = resample_to_grid(f, grid_info)
        except Exception as e:
            print(f"   Skipped {os.path.basename(f)}: {e}")
            continue

        if data is None:
            print(f"   Skipped {os.path.basename(f)}: no overlap with study area grid")
            continue

        data = prefilter_dem(data)
        raw_for_fp = data
        data = erode_strip_edges(data, cfg.NODATA_EDGE_EROSION_PX)

        n_valid = np.sum(np.isfinite(data))
        if n_valid < 1000:
            print(f"   Skipped {os.path.basename(f)}: only {n_valid:,} valid pixels after erosion")
            continue

        file_date = get_decimal_date(f)
        strips_data.append(data)
        strips_dates.append(file_date)
        strips_files.append(f)
        print(f"   Loaded: {os.path.basename(f)} ({n_valid} valid pixels, date={file_date:.4f})")

        stb, fp = None, None
        if use_stb:
            stb_file = find_stable_strip(f, year_path)
            if stb_file is not None:
                stb = resample_to_grid(stb_file, grid_info)
                if stb is not None:
                    stb = prefilter_dem(stb)
                    stb[stb == 0] = np.nan
                    fp, fp_src = load_strip_footprint(f, year_label, grid_info, raw_for_fp)
                    if fp is not None:
                        stb = np.where(fp, stb, np.nan).astype(np.float32)
                    print(f"      + stable strip {os.path.basename(stb_file)} "
                          f"({int(np.isfinite(stb).sum()):,} px in footprint: {fp_src})")
            else:
                print(f"      (no stable-ground strip found for {os.path.basename(f)})")
        # Consistency of the stable-ground strip with its own DEM strip on
        # pixels where both have data (same acquisition -> should be ~0).
        info = {"strip": os.path.basename(f), "date": file_date,
                "stb_file": os.path.basename(stb_file) if (use_stb and stb is not None) else "",
                "footprint": fp_src if (use_stb and stb is not None) else "",
                "n_dem": int(n_valid),
                "n_stb": int(np.isfinite(stb).sum()) if stb is not None else 0}
        if stb is not None:
            both = np.isfinite(stb) & np.isfinite(data)
            info["n_common_stb_dem"] = int(both.sum())
            if both.sum() > 0:
                dd = (stb - data)[both].astype(np.float64)
                med = float(np.median(dd))
                info["stb_minus_dem_median_m"] = med
                info["stb_minus_dem_nmad_m"] = float(1.4826 * np.median(np.abs(dd - med)))
            if cfg.STB_ALIGN_TO_DEM:
                if both.sum() >= cfg.STB_ALIGN_MIN_PX:
                    stb = (stb - med).astype(np.float32)
                    info["stb_shift_applied_m"] = -med
                    print(f"      stable strip aligned to its DEM strip: shift {-med:+.2f} m "
                          f"(NMAD {info['stb_minus_dem_nmad_m']:.2f} m, {int(both.sum()):,} shared px)")
                else:
                    print(f"      ⚠️  stable strip left out: only {int(both.sum())} px shared "
                          f"with its DEM strip (< {cfg.STB_ALIGN_MIN_PX})")
                    info["stb_shift_applied_m"] = np.nan
                    info["n_stb"] = 0
                    stb, fp = None, None
        strips_stb.append(stb)
        strips_fp.append(fp)
        strips_info.append(info)

    if not strips_data:
        print("   ! No valid strips.")
        return None

    # Sort largest coverage first – largest strip becomes the co-registration reference
    valid_counts = [np.sum(np.isfinite(d)) for d in strips_data]
    sort_order = np.argsort(valid_counts)[::-1]
    strips_data  = [strips_data[i]  for i in sort_order]
    strips_dates = [strips_dates[i] for i in sort_order]
    strips_files = [strips_files[i] for i in sort_order]
    strips_stb   = [strips_stb[i]   for i in sort_order]
    strips_fp    = [strips_fp[i]    for i in sort_order]
    strips_info  = [strips_info[i]  for i in sort_order]

    print("\n   Building mosaic (largest strip first, iterative de-ramping, feathered overlaps)...")

    ref      = strips_data[0].astype(np.float64)
    w_ref    = feather_weights(ref)
    w_sum    = w_ref.copy()
    dem_wsum = np.where(w_ref > 0, ref * w_ref, 0.0)
    date_wsum = np.where(w_ref > 0, strips_dates[0] * w_ref, 0.0)

    # Stable-ground mosaic: same strips, same corrections, feather weights
    # from each strip's footprint (rock lies in the DEM's data holes).
    stb_wsum = np.zeros(ref.shape, dtype=np.float64)
    stb_dsum = np.zeros(ref.shape, dtype=np.float64)
    stb_w    = np.zeros(ref.shape, dtype=np.float64)
    stb_wmax = np.zeros(ref.shape, dtype=np.float32)
    stb_id   = np.zeros(ref.shape, dtype=np.int16)    # 1-based strip id, 0 = none

    def _add_stb(stb, fp, date, sid):
        if stb is None:
            return
        stb = np.asarray(stb, dtype=np.float64)
        base_valid = fp if fp is not None else np.isfinite(stb)
        w = feather_weights(np.where(base_valid, 1.0, np.nan))
        v = np.isfinite(stb) & (w > 0)
        stb_wsum[v] += stb[v] * w[v]
        stb_dsum[v] += date * w[v]
        stb_w[v]    += w[v]
        top = v & (w > stb_wmax)
        stb_id[top] = sid
        stb_wmax[top] = w[top]

    for k, inf in enumerate(strips_info):
        inf["id"] = k + 1
    strips_info[0]["correction"] = "reference"
    _add_stb(strips_stb[0], strips_fp[0], strips_dates[0], 1)

    print(f"   [1/{len(strips_data)}] Reference: {os.path.basename(strips_files[0])}")

    for i in range(1, len(strips_data)):
        strip      = strips_data[i].astype(np.float64)
        strip_date = strips_dates[i]
        strip_file = strips_files[i]

        with np.errstate(invalid='ignore', divide='ignore'):
            current_mosaic = np.where(w_sum > 0, dem_wsum / w_sum, np.nan)
        strip, msg, steps = deramp_strip_to_mosaic(strip, current_mosaic, x_coords, y_coords)
        print(f"   [{i+1}/{len(strips_data)}] {os.path.basename(strip_file)}: {msg}")
        _add_stb(apply_correction_steps(strips_stb[i], x_coords, y_coords, steps),
                 strips_fp[i], strip_date, i + 1)
        strips_info[i]["correction"] = msg

        # Feathered accumulation (distance-to-edge weighted mean in overlaps)
        w = feather_weights(strip)
        v = w > 0
        dem_wsum[v]  += strip[v] * w[v]
        date_wsum[v] += strip_date * w[v]
        w_sum[v]     += w[v]

    with np.errstate(invalid='ignore', divide='ignore'):
        dem_mos  = np.where(w_sum > 0, dem_wsum  / w_sum, np.nan)
        date_mos = np.where(w_sum > 0, date_wsum / w_sum, np.nan)

    # Apply study-area mask once on the finished composite
    dem_mos = apply_mask_and_validate(dem_mos, bad_mask, "Final mosaic")

    dem_mos  = dem_mos.astype(np.float32)
    date_mos = date_mos.astype(np.float32)

    dem_out  = os.path.join(cfg.OUT_DIR, f"dem_{year_label}.tif")
    date_out = os.path.join(cfg.OUT_DIR, f"date_{year_label}.tif")

    save_raster(dem_mos,  dem_out,  grid_info,
                units="m", description=f"TanDEM-X DEM mosaic, {year_label} (elevation a.s.l.)")
    save_raster(date_mos, date_out, grid_info,
                units="decimal year", description=f"Acquisition-date mosaic, {year_label}")

    n_final = int(np.sum(np.isfinite(dem_mos)))
    print(f"   ✓ Saved: {os.path.basename(dem_out)} ({n_final} valid pixels)")

    dem_bedrock = apply_year_specific_mask(dem_mos, year_label, grid_info)
    dem_bedrock_out = os.path.join(cfg.BEDROCK_DEM_DIR, f"dem_{year_label}_bedrock.tif")
    save_raster(dem_bedrock, dem_bedrock_out, grid_info,
                units="m", description=f"TanDEM-X DEM mosaic, {year_label}, bedrock/stable-terrain mask applied")
    print(f"   ✓ Saved bedrock DEM: {os.path.basename(dem_bedrock_out)}")

    result = {'dem_file': dem_out, 'date_file': date_out,
              'bedrock_dem_file': dem_bedrock_out}
    if use_stb and np.any(stb_w > 0):
        with np.errstate(invalid='ignore', divide='ignore'):
            stb_mos  = np.where(stb_w > 0, stb_wsum / stb_w, np.nan).astype(np.float32)
            stb_date = np.where(stb_w > 0, stb_dsum / stb_w, np.nan).astype(np.float32)
        stb_out  = os.path.join(cfg.OUT_DIR, f"stb_dem_{year_label}.tif")
        stbd_out = os.path.join(cfg.OUT_DIR, f"stb_date_{year_label}.tif")
        save_raster(stb_mos, stb_out, grid_info, units="m",
                    description=f"Stable-ground mosaic {year_label}: accepted strips, footprint-clipped, "
                                f"same de-ramp as the DEM mosaic")
        save_raster(stb_date, stbd_out, grid_info, units="decimal year",
                    description=f"Acquisition-date mosaic of the stable-ground mosaic, {year_label}")
        print(f"   ✓ Saved stable-ground mosaic: {os.path.basename(stb_out)} "
              f"({int(np.isfinite(stb_mos).sum()):,} px)")
        result.update({'stb_dem_file': stb_out, 'stb_date_file': stbd_out})
        id_out  = os.path.join(cfg.OUT_DIR, f"stb_stripid_{year_label}.tif")
        tab_out = os.path.join(cfg.OUT_DIR, f"stb_strips_{year_label}.csv")
        save_raster(np.where(stb_id > 0, stb_id, np.nan).astype(np.float32), id_out, grid_info,
                    description=f"Id of the strip dominating each stable-ground pixel, {year_label} "
                                f"(see stb_strips_{year_label}.csv)")
        tab = pd.DataFrame(strips_info)
        tab.insert(0, "year", year_label)
        tab.to_csv(tab_out, index=False)
        print(f"   ✓ Saved strip table: {os.path.basename(tab_out)}")
        col = tab["stb_minus_dem_median_m"] if "stb_minus_dem_median_m" in tab else pd.Series(np.nan, index=tab.index)
        bad = tab[np.abs(col) > 1.0]
        if len(bad):
            print(f"   ⚠️  {len(bad)} stable strip(s) differ from their own DEM strip by > 1 m (median)")
        result.update({'stb_id_file': id_out, 'stb_table': tab})
    return result


def merge_multiyear_dems(year_results, years_to_merge, grid_info, bad_mask, x_coords, y_coords):
    """
    Merge per-year mosaics into a combined epoch mosaic (e.g. 2016-2017): edge
    erosion, largest coverage first as reference, vertical-only correction of
    each further mosaic to the current mosaic via deramp_strip_to_mosaic()
    (iterative quadratic de-ramp / median shift / as-is).
    """
    merged_label = '-'.join(years_to_merge)
    print("=" * 50)
    print(f"▶ Merging multi-year DEM: {merged_label}")

    dems_to_merge   = []
    dates_to_merge  = []
    stbs_to_merge   = []
    stbd_to_merge   = []
    sids_to_merge   = []

    for year in years_to_merge:
        if year not in year_results:
            continue
        with rasterio.open(year_results[year]['dem_file']) as src:
            dem = src.read(1).astype(np.float64)
            nodata = src.nodata
        with rasterio.open(year_results[year]['date_file']) as src:
            date = src.read(1).astype(np.float64)

        if nodata is not None:
            dem[dem == nodata]   = np.nan
            date[date == nodata] = np.nan

        dem  = erode_strip_edges(dem, cfg.NODATA_EDGE_EROSION_PX)
        date = np.where(np.isfinite(dem), date, np.nan)

        stb_a, stbd_a, sid_a = None, None, None
        if year_results[year].get('stb_dem_file'):
            with rasterio.open(year_results[year]['stb_dem_file']) as src:
                stb_a = src.read(1).astype(np.float64)
                if src.nodata is not None:
                    stb_a[stb_a == src.nodata] = np.nan
            with rasterio.open(year_results[year]['stb_date_file']) as src:
                stbd_a = src.read(1).astype(np.float64)
                if src.nodata is not None:
                    stbd_a[stbd_a == src.nodata] = np.nan
            if year_results[year].get('stb_id_file'):
                with rasterio.open(year_results[year]['stb_id_file']) as src:
                    sid_a = src.read(1).astype(np.float64)
                    if src.nodata is not None:
                        sid_a[sid_a == src.nodata] = np.nan
                # make ids unique across the merged years: 1000 * (year index + 1) + id
                k = years_to_merge.index(year) + 1
                sid_a = np.where(np.isfinite(sid_a), sid_a + 1000 * k, 0).astype(np.int32)

        dems_to_merge.append(dem)
        dates_to_merge.append(date)
        stbs_to_merge.append(stb_a)
        stbd_to_merge.append(stbd_a)
        sids_to_merge.append(sid_a)
        print(f"   Loaded: {year} ({int(np.sum(np.isfinite(dem))):,} valid pixels)")

    if not dems_to_merge:
        print("   ! No valid DEMs to merge")
        return None

    # Sort largest coverage first
    valid_counts   = [int(np.sum(np.isfinite(d))) for d in dems_to_merge]
    sort_order     = np.argsort(valid_counts)[::-1]
    dems_to_merge  = [dems_to_merge[i]  for i in sort_order]
    dates_to_merge = [dates_to_merge[i] for i in sort_order]
    stbs_to_merge  = [stbs_to_merge[i]  for i in sort_order]
    stbd_to_merge  = [stbd_to_merge[i]  for i in sort_order]
    sids_to_merge  = [sids_to_merge[i]  for i in sort_order]
    years_sorted   = [years_to_merge[i] for i in sort_order]

    print("\n   Building merged mosaic (largest coverage first, iterative de-ramping, feathered overlaps)...")

    ref       = dems_to_merge[0].astype(np.float64)
    ref_date  = dates_to_merge[0]
    w_ref     = feather_weights(ref)
    w_sum     = w_ref.copy()
    dem_wsum  = np.where(w_ref > 0, ref * w_ref, 0.0)
    # date weights: same feather weights, only where a date exists
    wd_ref    = np.where(np.isfinite(ref_date), w_ref, 0.0)
    wd_sum    = wd_ref.copy()
    date_wsum = np.where(wd_ref > 0, ref_date * wd_ref, 0.0)

    # stable-ground merge: same corrections and merge rule as the DEMs
    s_wsum = np.zeros(ref.shape); s_dsum = np.zeros(ref.shape)
    s_w = np.zeros(ref.shape); s_wd = np.zeros(ref.shape)
    s_wmax = np.zeros(ref.shape, dtype=np.float32)
    s_id = np.zeros(ref.shape, dtype=np.int32)

    def _add_stb_m(stb, stbd, dem_cov, fill_only, sid=None):
        if stb is None:
            return
        cov = np.isfinite(stb) | np.isfinite(dem_cov)
        w = feather_weights(np.where(cov, 1.0, np.nan))
        if fill_only:
            w = np.where((w > 0) & (s_w == 0), 1.0, 0.0)
        v = np.isfinite(stb) & (w > 0)
        s_wsum[v] += stb[v] * w[v]
        s_w[v]    += w[v]
        if sid is not None:
            top = v & (w > s_wmax)
            s_id[top] = sid[top]
            s_wmax[top] = w[top]
        d = np.where(np.isfinite(stbd), stbd, np.nan)
        vd = v & np.isfinite(d)
        s_dsum[vd] += d[vd] * w[vd]
        s_wd[vd]   += w[vd]

    _add_stb_m(stbs_to_merge[0], stbd_to_merge[0], dems_to_merge[0], False, sids_to_merge[0])

    print(f"   [1/{len(dems_to_merge)}] Reference: {years_sorted[0]}")

    for i in range(1, len(dems_to_merge)):
        dem   = dems_to_merge[i].astype(np.float64)
        date  = dates_to_merge[i]
        year  = years_sorted[i]

        with np.errstate(invalid='ignore', divide='ignore'):
            current_mosaic = np.where(w_sum > 0, dem_wsum / w_sum, np.nan)
        dem, msg, steps = deramp_strip_to_mosaic(dem, current_mosaic, x_coords, y_coords)
        print(f"   [{i+1}/{len(dems_to_merge)}] {year}: {msg}")
        _add_stb_m(apply_correction_steps(stbs_to_merge[i], x_coords, y_coords, steps),
                   stbd_to_merge[i], dem, cfg.MERGE_MODE == "fill_gaps", sids_to_merge[i])

        w = feather_weights(dem)
        if cfg.MERGE_MODE == "fill_gaps":
            # only where no earlier (larger-coverage) year has data; full weight
            w = np.where((w > 0) & (w_sum == 0), 1.0, 0.0)
            print(f"      fill_gaps: {year} contributes {int(np.sum(w > 0)):,} px "
                  f"not covered by {', '.join(years_sorted[:i])}")
        v = w > 0
        dem_wsum[v] += dem[v] * w[v]
        w_sum[v]    += w[v]
        wd = np.where(np.isfinite(date), w, 0.0)
        vd = wd > 0
        date_wsum[vd] += date[vd] * wd[vd]
        wd_sum[vd]    += wd[vd]

    with np.errstate(invalid='ignore', divide='ignore'):
        dem_merged  = np.where(w_sum  > 0, dem_wsum  / w_sum,  np.nan)
        date_merged = np.where(wd_sum > 0, date_wsum / wd_sum, np.nan)

    dem_merged = apply_mask_and_validate(dem_merged, bad_mask, "Merged mosaic")

    dem_merged  = dem_merged.astype(np.float32)
    date_merged = date_merged.astype(np.float32)

    dem_out  = os.path.join(cfg.OUT_DIR, f"dem_{merged_label}.tif")
    date_out = os.path.join(cfg.OUT_DIR, f"date_{merged_label}.tif")

    save_raster(dem_merged,  dem_out,  grid_info,
                units="m", description=f"TanDEM-X DEM mosaic, merged {merged_label} (elevation a.s.l.)")
    save_raster(date_merged, date_out, grid_info,
                units="decimal year", description=f"Acquisition-date mosaic, merged {merged_label}")

    n_final = int(np.sum(np.isfinite(dem_merged)))
    print(f"   ✓ Saved: {os.path.basename(dem_out)} ({n_final} valid pixels)")

    dem_bedrock = apply_year_specific_mask(dem_merged, merged_label, grid_info)
    dem_bedrock_out = os.path.join(cfg.BEDROCK_DEM_DIR, f"dem_{merged_label}_bedrock.tif")
    save_raster(dem_bedrock, dem_bedrock_out, grid_info,
                units="m", description=f"TanDEM-X DEM mosaic, merged {merged_label}, bedrock/stable-terrain mask applied")
    print(f"   ✓ Saved bedrock DEM: {os.path.basename(dem_bedrock_out)}")

    result = {'dem_file': dem_out, 'date_file': date_out,
              'bedrock_dem_file': dem_bedrock_out}
    if np.any(s_w > 0):
        with np.errstate(invalid='ignore', divide='ignore'):
            stb_m  = np.where(s_w  > 0, s_wsum / s_w,  np.nan).astype(np.float32)
            stbd_m = np.where(s_wd > 0, s_dsum / s_wd, np.nan).astype(np.float32)
        stb_out  = os.path.join(cfg.OUT_DIR, f"stb_dem_{merged_label}.tif")
        stbd_out = os.path.join(cfg.OUT_DIR, f"stb_date_{merged_label}.tif")
        save_raster(stb_m, stb_out, grid_info, units="m",
                    description=f"Stable-ground mosaic, merged {merged_label} (same corrections as DEM merge)")
        save_raster(stbd_m, stbd_out, grid_info, units="decimal year",
                    description=f"Acquisition-date mosaic of the stable-ground mosaic, merged {merged_label}")
        print(f"   ✓ Saved stable-ground mosaic: {os.path.basename(stb_out)} "
              f"({int(np.isfinite(stb_m).sum()):,} px)")
        result.update({'stb_dem_file': stb_out, 'stb_date_file': stbd_out})
        tabs = []
        for k, year in enumerate(years_to_merge):
            t = year_results.get(year, {}).get('stb_table')
            if t is not None:
                t = t.copy()
                t["id"] = t["id"] + 1000 * (k + 1)
                tabs.append(t)
        if tabs and np.any(s_id > 0):
            id_out  = os.path.join(cfg.OUT_DIR, f"stb_stripid_{merged_label}.tif")
            tab_out = os.path.join(cfg.OUT_DIR, f"stb_strips_{merged_label}.csv")
            save_raster(np.where(s_id > 0, s_id, np.nan).astype(np.float32), id_out, grid_info,
                        description=f"Id of the strip dominating each stable-ground pixel, merged "
                                    f"{merged_label} (1000*year index + strip id)")
            tab = pd.concat(tabs, ignore_index=True)
            tab.to_csv(tab_out, index=False)
            result.update({'stb_id_file': id_out, 'stb_table': tab})
    return result
