"""
ICESat-2 repeat-track elevation-change grid (dh/dt_IS2) and its uncertainty
(sigma_IS2) from the cleaned captoolkit fitsec.py nodes, the catchment-mean
rate and the comparison with the crossover rates. Settings: config.py (section 4).

1. Track-to-track offsets: nodes are grouped into track segments (local line
   search for the track direction, then chains of nodes along it); one rate
   offset per segment is estimated where tracks cross (damped least squares)
   and removed before interpolation (TRACK_ADJUST). The remaining track error
   can be modelled in the kriging covariance (TRACK_ERROR_MODEL).
2. Ordinary kriging with sector-wise neighbour search (nodes from the tracks on
   both sides of every cell). The variogram is fitted to the nodes; nugget and
   track error are chosen by leave-track-out cross-validation (NUGGET_TUNING).
   Each node's noise variance is the nugget scaled by its fitsec error
   (p1_err / median p1_err)^2. The crossovers are not used in the
   interpolation and remain an independent check.
3. sigma_IS2 is the kriging prediction error, calibrated by leave-track-out
   cross-validation per distance-to-node class (APPLY_CV_CALIBRATION) so that
   +-1 sigma_IS2 covers 68 % of the observed errors.
4. Catchment mean: area-weighted mean of the grid with
   sigma_mean = RMS(sigma_IS2) / sqrt(N_eff), N_eff = A / A_corr (Rolstad et
   al., 2009; Hugonnet et al., 2022), plus an optional systematic term.
5. Crossovers vs. grid and vs. nearest node (N, mean, SD, median, NMAD).

Outputs (GRID_OUTPUT_DIR, file names start with GRID_PREFIX):
  <p>_dhdt.tif, <p>_sigma.tif, <p>_dist_to_node.tif     grids (m a-1, m a-1, m)
  <p>_track_offsets.csv, <p>_nodes_adjusted.gpkg          crossover adjustment
  <p>_nugget_tuning.csv, <p>_variogram.csv, <p>_variogram_fit.csv
  <p>_cv_nodes.csv, <p>_cv_by_distance.csv                cross-validation
  <p>_catchment_mean.csv                                  catchment mean and uncertainty
  <p>_xover_points.csv, <p>_xover_summary.csv             crossover comparison
  <p>_plot_state.pkl                                      input of plot_grid_diagnostics.py
"""
import os
import pickle

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from rasterio.features import geometry_mask
from rasterio.transform import from_origin
from scipy.spatial import cKDTree

import config as cfg
from kriging import (
    _gamma_model,
    block_cross_validation,
    correlation_area,
    empirical_variogram,
    fit_variogram,
    krige,
    make_folds,
    make_track_folds,
    nmad,
    summary,
    track_adjustment,
    track_directions,
    track_labels,
    _cv_score,
)


def main():
    """Variogram, track adjustment, nugget choice, cross-validation, kriging,
    catchment mean and crossover comparison."""
    rng = np.random.default_rng(cfg.RANDOM_SEED)
    os.makedirs(cfg.GRID_OUTPUT_DIR, exist_ok=True)
    out = lambda name: os.path.join(cfg.GRID_OUTPUT_DIR, f"{cfg.GRID_PREFIX}_{name}")

    print("Loading data...")
    mask_poly = gpd.read_file(cfg.GRID_MASK)
    pts = gpd.read_file(cfg.GRID_INPUT)
    crs = mask_poly.crs
    if pts.crs != crs:
        pts = pts.to_crs(crs)
    ok = np.isfinite(pts[cfg.P1_COL].values.astype(float)) & np.isfinite(pts[cfg.P1_ERR_COL].values.astype(float))
    pts = pts[ok].reset_index(drop=True)
    xy = np.c_[pts.geometry.x.values, pts.geometry.y.values]
    v = pts[cfg.P1_COL].values.astype(float)
    err = pts[cfg.P1_ERR_COL].values.astype(float)
    print(f"  nodes: {len(v):,}   p1 {v.min():.2f} to {v.max():.2f} m/a   "
          f"p1_err median {np.median(err):.3f} m/a")

    # ---------------- variogram ----------------
    print("Empirical variogram...")
    ev = empirical_variogram(xy, v, rng)
    vg = fit_variogram(ev)
    a_corr = correlation_area(vg['range_m'])
    print(f"  {cfg.VARIOGRAM_MODEL}: nugget {vg['nugget']:.4f}, partial sill {vg['psill']:.4f} (m/a)^2, "
          f"range {vg['range_m']:.0f} m  (A_corr {a_corr / 1e6:.1f} km^2)")
    ev['gamma_model'] = _gamma_model(ev['lag_mean'].values, vg['nugget'], vg['psill'], vg['range_m'])
    ev['used_in_fit'] = ev['lag_lo'] >= cfg.VARIOGRAM_MIN_LAG
    ev.to_csv(out('variogram.csv'), index=False, float_format='%.5f')

    if cfg.NOISE_WEIGHTING:
        ratio = np.clip((err / np.median(err)) ** 2, 1 / cfg.NOISE_RATIO_MAX, cfg.NOISE_RATIO_MAX)
    else:
        ratio = np.ones_like(err)

    # ---------------- nugget by cross-validation ----------------
    vg_fit = dict(vg)
    xo_tune = None
    if cfg.NUGGET_TUNING == 'crossovers':
        if cfg.GRID_XOVER and os.path.exists(cfg.GRID_XOVER):
            xo_tune = gpd.read_file(cfg.GRID_XOVER)
            xo_tune = xo_tune.to_crs(crs) if xo_tune.crs != crs else xo_tune
            xo_tune = gpd.clip(xo_tune, mask_poly)
            xo_xy = np.c_[xo_tune.geometry.x.values, xo_tune.geometry.y.values]
            xo_v = xo_tune[cfg.XO_COL].values.astype(float)
            okx = np.isfinite(xo_v)
            xo_xy, xo_v = xo_xy[okx], xo_v[okx]
        else:
            print("  NUGGET_TUNING = 'crossovers' but no crossover file -- fitted nugget kept")
    labels = None
    if cfg.TRACK_ERROR_MODEL:
        theta = track_directions(xy)
        labels = track_labels(xy, theta)
        sizes = np.bincount(labels)
        print(f"  track direction found for {np.isfinite(theta).mean():.0%} of the nodes; "
              f"{(sizes >= 10).sum():,} track segments with >= 10 nodes "
              f"(median {int(np.median(sizes[sizes >= 10])) if (sizes >= 10).any() else 0} nodes), "
              f"{(sizes == 1).sum():,} single nodes")
        if cfg.TRACK_ADJUST:
            off, otab = track_adjustment(xy, v, labels, theta)
            otab.to_csv(out('track_offsets.csv'), index=False, float_format='%.4f')
            gpd.GeoDataFrame(dict(p1=v, p1_adj=v - off, offset=off, p1_err=err, segment=labels),
                             geometry=gpd.points_from_xy(xy[:, 0], xy[:, 1]), crs=crs
                             ).to_file(out('nodes_adjusted.gpkg'), driver='GPKG')
            v = v - off
    track_cv = cfg.CV_SCHEME == 'track' and labels is not None
    fold = make_track_folds(labels, rng) if track_cv else make_folds(xy, rng)
    cv_theta = theta if track_cv else None
    if cfg.NUGGET_TUNING == 'track_cv' and not track_cv:
        print("  NUGGET_TUNING = 'track_cv' needs TRACK_ERROR_MODEL and CV_SCHEME = 'track' -- "
              "falling back to block CV")
    vg['track_var'] = 0.0
    if cfg.NUGGET_CANDIDATES and (cfg.NUGGET_TUNING in ('cv', 'track_cv') or xo_tune is not None):
        total = vg['nugget'] + vg['psill']
        sub = np.zeros(len(v), bool)
        sub[rng.choice(len(v), size=min(cfg.CV_TUNE_MAX_TEST, len(v)), replace=False)] = True
        n_c = sorted(set([vg['nugget'] / total] + list(cfg.NUGGET_CANDIDATES)))
        t_c = list(cfg.TRACK_VAR_CANDIDATES) if cfg.TRACK_ERROR_MODEL else [0.0]
        tune = []
        src = (f"{len(xo_v):,} crossovers" if xo_tune is not None
               else f"{'leave-track-out' if track_cv else 'block'} cross-validation, "
                    f"{sub.sum():,} held-out nodes; crossovers not used")
        print(f"Choosing nugget{' and track error' if cfg.TRACK_ERROR_MODEL else ''} "
              f"({cfg.CV_TUNE_SCORE} against {src})...")
        for fr in n_c:
            for tf in t_c:
                if fr + tf > 0.8:
                    continue
                vg_c = dict(vg, nugget=fr * total, track_var=tf * total,
                            psill=(1 - fr - tf) * total)
                if xo_tune is not None:
                    pc, _, dc = krige(xo_xy, xy, v, vg_c['nugget'] * ratio, vg_c, labels=labels)
                    res = np.where(dc <= cfg.MAX_DISTANCE, xo_v - pc, np.nan)
                else:
                    res = block_cross_validation(xy, v, vg_c['nugget'] * ratio, vg_c, fold, sub,
                                                 labels=labels, theta=cv_theta)['residual'].values
                res = res[np.isfinite(res)]
                sc = _cv_score(res)
                tune.append(dict(nugget_fraction=fr, track_fraction=tf, nugget=vg_c['nugget'],
                                 track_var=vg_c['track_var'], psill=vg_c['psill'],
                                 score=sc, mae=float(np.mean(np.abs(res))),
                                 rmse=float(np.sqrt(np.mean(res ** 2))), nmad=nmad(res),
                                 median=float(np.median(res)), n=len(res)))
                print(f"  nugget {fr:5.1%}, track {tf:5.1%} of sill: MAE {tune[-1]['mae']:.4f}, "
                      f"RMSE {tune[-1]['rmse']:.4f}, NMAD {tune[-1]['nmad']:.4f}")
        tune = pd.DataFrame(tune)
        ib = tune['score'].idxmin()
        best = tune.loc[ib]
        vg = dict(vg, nugget=float(best['nugget']), psill=float(best['psill']),
                  track_var=float(best['track_var']))
        tune['chosen'] = tune.index == ib
        tune.to_csv(out('nugget_tuning.csv'), index=False, float_format='%.5f')
        print(f"  -> nugget {vg['nugget']:.4f} ({best['nugget_fraction']:.1%}), track error "
              f"{vg['track_var']:.4f} ({best['track_fraction']:.1%}), partial sill {vg['psill']:.4f}")
    a_corr = correlation_area(vg['range_m'])
    pd.DataFrame([dict(model=cfg.VARIOGRAM_MODEL, fitted_nugget=vg_fit['nugget'],
                       fitted_psill=vg_fit['psill'], **vg, a_corr_km2=a_corr / 1e6)]).to_csv(
        out('variogram_fit.csv'), index=False, float_format='%.5f')
    noise = vg['nugget'] * ratio

    # ---------------- cross-validation ----------------
    print(f"{'Leave-track-out' if track_cv else 'Block'} cross-validation ({cfg.CV_FOLDS} folds)...")
    cv = block_cross_validation(xy, v, noise, vg, fold, labels=labels, theta=cv_theta)
    q68 = lambda z: float(np.nanpercentile(np.abs(z), 68.27)) if np.isfinite(z).any() else np.nan
    print(f"  residual: median {np.median(cv['residual']):+.3f}, NMAD {nmad(cv['residual']):.3f} m/a; "
          f"|z| 68th percentile {q68(cv['z']):.2f} (1 = sigma correct), z NMAD {nmad(cv['z']):.2f}, "
          f"z SD {np.nanstd(cv['z']):.2f}")
    rows = []
    for lo, hi in zip(np.asarray(cfg.CV_DIST_EDGES, float)[:-1], np.asarray(cfg.CV_DIST_EDGES, float)[1:]):
        s = (cv['dist_to_train'] >= lo) & (cv['dist_to_train'] < hi)
        if s.sum() < cfg.CV_MIN_PER_BIN:
            continue
        rows.append(dict(dist_lo_m=lo, dist_hi_m=hi, dist_mid_m=0.5 * (lo + hi),
                         **summary(cv.loc[s, 'residual'], 'res_'),
                         sigma_total_median=cv.loc[s, 'sigma_total'].median(),
                         z_abs_p68=q68(cv.loc[s, 'z']), z_nmad=nmad(cv.loc[s, 'z']),
                         z_std=cv.loc[s, 'z'].std()))
    cvd = pd.DataFrame(rows)
    # Calibration factor as a function of distance to the nearest node:
    # the 68th percentile of |z| per distance bin, so that +-1 sigma covers
    # 68 % of the cross-validation residuals at every distance. Beyond the
    # last bin the last factor is kept.
    if cfg.APPLY_CV_CALIBRATION and not cvd.empty:
        fd = np.maximum(cvd['z_abs_p68'].values, cfg.CV_CALIBRATION_MIN)
        factor_of = lambda d: np.interp(d, cvd['dist_mid_m'].values, fd)
    else:
        factor_of = lambda d: np.ones_like(np.asarray(d, float))
    cv['calibration_factor'] = factor_of(cv['dist_to_train'].values)
    cv['z_calibrated'] = cv['z'] / cv['calibration_factor']
    if not cvd.empty:
        cvd['calibration_factor'] = factor_of(cvd['dist_mid_m'].values)
        cvd['sigma_total_calibrated_median'] = cvd['sigma_total_median'] * cvd['calibration_factor']
    print(f"  after calibration: |z| 68th percentile {q68(cv['z_calibrated']):.2f}")
    cv.to_csv(out('cv_nodes.csv'), index=False, float_format='%.4f')
    cvd.to_csv(out('cv_by_distance.csv'), index=False, float_format='%.4f')

    # ---------------- grid (exact cell centres) ----------------
    minx, miny, maxx, maxy = mask_poly.total_bounds
    x0 = np.floor(minx / cfg.RESOLUTION) * cfg.RESOLUTION
    ytop = np.ceil(maxy / cfg.RESOLUTION) * cfg.RESOLUTION
    ncols = int(np.ceil((maxx - x0) / cfg.RESOLUTION))
    nrows = int(np.ceil((ytop - miny) / cfg.RESOLUTION))
    transform = from_origin(x0, ytop, cfg.RESOLUTION, cfg.RESOLUTION)
    inside = geometry_mask(mask_poly.geometry, out_shape=(nrows, ncols),
                           transform=transform, invert=True)
    rr, cc = np.nonzero(inside)
    cx = x0 + (cc + 0.5) * cfg.RESOLUTION
    cy = ytop - (rr + 0.5) * cfg.RESOLUTION
    print(f"Grid: {nrows} x {ncols} cells of {cfg.RESOLUTION:.0f} m, {inside.sum():,} inside the catchment")

    print("Kriging...")
    p, s2, dn = krige(np.c_[cx, cy], xy, v, noise, vg, labels=labels)
    far = dn > cfg.MAX_DISTANCE
    p[far] = np.nan; s2[far] = np.nan
    sig = factor_of(dn) * np.sqrt(s2)

    def to_grid(vals):
        g = np.full((nrows, ncols), np.nan, dtype='float32')
        g[rr, cc] = vals
        return g

    meta = dict(driver='GTiff', height=nrows, width=ncols, count=1, dtype='float32', crs=crs,
                transform=transform, nodata=np.nan, compress='lzw')
    for name, arr in (('dhdt.tif', to_grid(p)), ('sigma.tif', to_grid(sig)),
                      ('dist_to_node.tif', to_grid(np.where(np.isfinite(dn), dn, np.nan)))):
        with rasterio.open(out(name), 'w', **meta) as dst:
            dst.write(arr, 1)
        print(f"  wrote {out(name)}")

    # ---------------- catchment mean ----------------
    cell_area = cfg.RESOLUTION ** 2
    valid = np.isfinite(p)
    A_catch = inside.sum() * cell_area
    A_valid = valid.sum() * cell_area
    mean_grid = float(np.mean(p[valid]))
    rms_sig = float(np.sqrt(np.mean(sig[valid] ** 2)))
    n_eff = max(1.0, A_valid / a_corr)
    s_rand = rms_sig / np.sqrt(n_eff)
    s_tot = float(np.hypot(s_rand, cfg.SIGMA_SYS_MYR))
    # sigma_IS2 is calibrated to 68 % coverage. If the CV residuals are
    # heavy-tailed (SD of z larger than its 68th percentile), the RMS error
    # is larger than sigma_IS2; the conservative mean uncertainty scales the
    # random part by that ratio.
    tail = max(1.0, float(np.nanstd(cv['z_calibrated'])) /
               float(np.nanpercentile(np.abs(cv['z_calibrated']), 68.27)))
    s_tot_cons = float(np.hypot(s_rand * tail, cfg.SIGMA_SYS_MYR))
    near = valid & (dn <= cfg.NEAR_NODE_M)
    in_catch = mask_poly.union_all() if hasattr(mask_poly, 'union_all') else mask_poly.unary_union
    node_in = gpd.GeoSeries(gpd.points_from_xy(xy[:, 0], xy[:, 1]), crs=crs).within(in_catch).values
    vn = v[node_in]
    a_nodes = len(vn)
    row = dict(
        catchment_area_km2=A_catch / 1e6, interpolated_area_km2=A_valid / 1e6,
        coverage_pct=100 * A_valid / A_catch,
        mean_dhdt_myr=mean_grid, sigma_mean_myr=s_tot, sigma_random_myr=s_rand,
        sigma_mean_conservative_myr=s_tot_cons, cv_tail_ratio=tail,
        sigma_sys_myr=cfg.SIGMA_SYS_MYR, rms_sigma_cell_myr=rms_sig, n_eff=n_eff,
        a_corr_km2=a_corr / 1e6,
        cv_calibration_factor_min=float(cvd['calibration_factor'].min()) if 'calibration_factor' in cvd else 1.0,
        cv_calibration_factor_max=float(cvd['calibration_factor'].max()) if 'calibration_factor' in cvd else 1.0,
        mean_dhdt_within_1km_myr=float(np.mean(p[near])) if near.any() else np.nan,
        area_within_1km_km2=near.sum() * cell_area / 1e6,
        n_nodes_in_catchment=a_nodes,
        node_mean_myr=float(np.mean(vn)) if a_nodes else np.nan,
        node_median_myr=float(np.median(vn)) if a_nodes else np.nan,
        variogram_model=cfg.VARIOGRAM_MODEL, nugget=vg['nugget'], psill=vg['psill'],
        range_m=vg['range_m'])
    pd.DataFrame([row]).to_csv(out('catchment_mean.csv'), index=False, float_format='%.4f')
    print(f"\nCatchment mean dh/dt_IS2 = {mean_grid:+.2f} ± {s_tot:.2f} m/a "
          f"(conservative ± {s_tot_cons:.2f}; coverage {row['coverage_pct']:.1f} %, N_eff {n_eff:.0f}; "
          f"node mean {row['node_mean_myr']:+.2f}, node median {row['node_median_myr']:+.2f} m/a)")

    # ---------------- crossovers ----------------
    xover = None
    if cfg.GRID_XOVER and os.path.exists(cfg.GRID_XOVER):
        xover = compare_crossovers(out, crs, mask_poly, transform, to_grid(p), to_grid(sig),
                                   to_grid(dn), xy, v)
    else:
        print(f"Crossover file not found ({cfg.GRID_XOVER}) -- comparison skipped")

    # hand-over to plot_grid_diagnostics.py
    with open(out('plot_state.pkl'), 'wb') as f:
        pickle.dump(dict(ev=ev, vg_fit=vg_fit, cvd=cvd, vg=vg, xover=xover), f)
    print("Complete!")


def compare_crossovers(out, crs, mask_poly, transform, g_dhdt, g_sig, g_dist, xy, v):
    """Cleaned crossover rates vs. the grid cell containing each crossover and
    vs. the nearest node (independent check: crossovers are not used in the
    interpolation). Writes <p>_xover_points.csv and <p>_xover_summary.csv."""
    xo = gpd.read_file(cfg.GRID_XOVER)
    xo = xo.to_crs(crs) if xo.crs != crs else xo
    xo = gpd.clip(xo, mask_poly).reset_index(drop=True)
    if xo.empty:
        print("No crossovers inside the catchment")
        return
    x, y = xo.geometry.x.values, xo.geometry.y.values
    r, c = rasterio.transform.rowcol(transform, x, y, op=np.floor)
    r, c = np.asarray(r, dtype=np.int64), np.asarray(c, dtype=np.int64)
    okrc = (r >= 0) & (r < g_dhdt.shape[0]) & (c >= 0) & (c < g_dhdt.shape[1])
    take = lambda g: np.where(okrc, g[np.clip(r, 0, g.shape[0] - 1), np.clip(c, 0, g.shape[1] - 1)], np.nan)
    t = pd.DataFrame(dict(x=x, y=y, xo_dhdt=xo[cfg.XO_COL].values.astype(float),
                          grid_dhdt=take(g_dhdt), grid_sigma=take(g_sig), dist_to_node=take(g_dist)))
    if cfg.XO_DT_COL in xo.columns:
        t['dt_year'] = xo[cfg.XO_DT_COL].values.astype(float)
    d, i = cKDTree(xy).query(np.c_[x, y], k=1)
    t['node_dist'] = d
    t['node_dhdt'] = np.where(d <= cfg.XO_NODE_MATCH_M, v[i], np.nan)
    t['xo_minus_grid'] = t['xo_dhdt'] - t['grid_dhdt']
    t['xo_minus_node'] = t['xo_dhdt'] - t['node_dhdt']
    t['z_grid'] = t['xo_minus_grid'] / t['grid_sigma']
    t.to_csv(out('xover_points.csv'), index=False, float_format='%.4f')
    rows = [dict(comparison='crossover - grid, all', **summary(t['xo_minus_grid']),
                 z_nmad=nmad(t['z_grid'])),
            dict(comparison=f'crossover - nearest node (<= {cfg.XO_NODE_MATCH_M:.0f} m)',
                 **summary(t['xo_minus_node']))]
    for lo, hi in ((0, 300), (300, 1000), (1000, 3000), (3000, np.inf)):
        s = (t['dist_to_node'] >= lo) & (t['dist_to_node'] < hi)
        rows.append(dict(comparison=f'crossover - grid, node distance {lo:.0f}-{hi:.0f} m',
                         **summary(t.loc[s, 'xo_minus_grid']), z_nmad=nmad(t.loc[s, 'z_grid'])))
    sm = pd.DataFrame(rows)
    sm.to_csv(out('xover_summary.csv'), index=False, float_format='%.4f')
    s0 = rows[0]
    print(f"Crossover - grid: N = {s0.get('n', 0):,}, mean {s0.get('mean', np.nan):+.3f}, "
          f"SD {s0.get('std', np.nan):.3f}, median {s0.get('median', np.nan):+.3f}, "
          f"NMAD {s0.get('nmad', np.nan):.3f} m/a")

    return dict(table=t, summary=s0)


if __name__ == '__main__':
    main()
