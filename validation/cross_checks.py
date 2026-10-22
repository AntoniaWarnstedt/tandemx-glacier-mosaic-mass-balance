"""
Cross-checks of the TanDEM-X - altimeter offset (tables in OUTPUT_DIR/cross_checks):

- run_same_cell_comparison(): ICESat-2 vs CryoSat-2 in identical grid cells.
  The two altimeters sample different locations inside a cell, so TanDEM-X is
  used as transfer surface: IS2 - CS2 = (TDX - CS2) - (TDX - IS2); the raw
  difference of the cell means is written as well.
- run_rock_comparison(): TanDEM-X minus altimeter on ice-free rock, where
  neither penetration nor surface change exists; an offset there points at
  the vertical reference rather than at the glacier surface.
"""
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio

from . import config as cfg
from .common import (
    _sample_array_at_points,
    aggregate_points_to_grid,
    convert_cs2_to_geoid,
    filter_by_tdx_window,
    nmad,
    pooled_epochs,
    robust_summary,
    sample_raster_values,
    sample_slope_from_own_dem,
)
from .icesat2 import load_is2_points

_ROCK_RASTER_CACHE = {}


def _summary_row(label, d, **extra):
    r = dict(comparison=label, **extra)
    r.update(robust_summary(d))
    return r


SAME_CELL_SLOPE_CLASSES = [(0, 5), (5, 15), (15, 90)]   # slope classes (deg) of the same-cell table


def run_same_cell_comparison(cs2_comparator, is2_results, output_dir, periods=None):
    """
    ICESat-2 vs CryoSat-2 in the same aggregation cells (matched on their grid
    index). Writes the per-cell tables and same_cell_is2_cs2_summary.csv and
    returns the summary DataFrame.
    """
    if periods is None:
        periods = cfg.SAME_CELL_PERIODS
    if periods is None:
        cs2_keys = set(cs2_comparator.comparisons) if cs2_comparator else set()
        periods = sorted((cs2_keys & set(is2_results)) - {'All_Years'})
    output_dir = Path(output_dir); output_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    print("\n" + "=" * 60)
    print("Same-cell ICESat-2 vs CryoSat-2 comparison")
    print("=" * 60)
    for period in periods:
        c = (cs2_comparator.comparisons.get(period) if cs2_comparator else None)
        i = is2_results.get(str(period))
        if c is None or i is None:
            print(f"  {period}: CryoSat-2 or ICESat-2 comparison missing -- skipped")
            continue
        cs2, is2 = c['gdf'], i['gdf']
        if 'x_bin' not in cs2.columns or 'x_bin' not in is2.columns:
            print(f"  {period}: needs the aggregated (cell) comparisons -- skipped")
            continue
        cs2_col = 'elevation_mean' if 'elevation_mean' in cs2.columns else 'elevation'
        a = pd.DataFrame({'x_bin': cs2['x_bin'].values, 'y_bin': cs2['y_bin'].values,
                          'h_cs2': cs2[cs2_col].values, 'tdx_at_cs2': cs2['tdx_elev'].values,
                          'tdx_minus_cs2': cs2['diff'].values, 'n_cs2': cs2['n_points'].values})
        b = pd.DataFrame({'x_bin': is2['x_bin'].values, 'y_bin': is2['y_bin'].values,
                          'x': is2['x'].values, 'y': is2['y'].values,
                          'h_is2': is2[i['is2_col']].values, 'tdx_at_is2': is2['tdx_h'].values,
                          'tdx_minus_is2': is2['dh'].values, 'n_is2': is2['n_points'].values,
                          'slope': (is2['tdx_slope'].values if 'tdx_slope' in is2.columns
                                    else np.full(len(is2), np.nan))})
        m = b.merge(a, on=['x_bin', 'y_bin'], how='inner')
        if m.empty:
            print(f"  {period}: no common cells -- skipped")
            continue
        m['is2_minus_cs2'] = m['tdx_minus_cs2'] - m['tdx_minus_is2']     # TDX as transfer surface
        m['is2_minus_cs2_raw'] = m['h_is2'] - m['h_cs2']                 # plain cell means
        m['elev'] = m['tdx_at_is2']
        gm = gpd.GeoDataFrame(m, geometry=gpd.points_from_xy(m['x'], m['y']), crs=is2.crs)
        gm.to_file(output_dir / f'same_cell_is2_cs2_{period}.gpkg', driver='GPKG')
        m.to_csv(output_dir / f'same_cell_is2_cs2_{period}.csv',
                                  index=False, float_format='%.3f')

        extra = dict(period=period)
        rows += [
            _summary_row('TDX - ICESat-2, all ICESat-2 cells', is2['dh'].values, **extra),
            _summary_row('TDX - CryoSat-2, all CryoSat-2 cells', cs2['diff'].values, **extra),
            _summary_row('TDX - ICESat-2, common cells', m['tdx_minus_is2'], **extra),
            _summary_row('TDX - CryoSat-2, common cells', m['tdx_minus_cs2'], **extra),
            _summary_row('ICESat-2 - CryoSat-2, common cells (TDX transfer)', m['is2_minus_cs2'], **extra),
            _summary_row('ICESat-2 - CryoSat-2, common cells (raw cell means)', m['is2_minus_cs2_raw'], **extra),
        ]
        for lo, hi in SAME_CELL_SLOPE_CLASSES:
            sel = (m['slope'] >= lo) & (m['slope'] < hi)
            rows.append(_summary_row(f'ICESat-2 - CryoSat-2, common cells, slope {lo}-{hi} deg',
                                     m.loc[sel, 'is2_minus_cs2'], **extra))
        s_is2_cs2 = robust_summary(m['is2_minus_cs2'])
        print(f"  {period}: {len(m):,} common cells (of {len(is2):,} ICESat-2 / {len(cs2):,} CryoSat-2 cells)")
        print(f"    TDX - IS2 (common) : median {np.median(m['tdx_minus_is2']):+.2f} m, NMAD {nmad(m['tdx_minus_is2']):.2f} m")
        print(f"    TDX - CS2 (common) : median {np.median(m['tdx_minus_cs2']):+.2f} m, NMAD {nmad(m['tdx_minus_cs2']):.2f} m")
        print(f"    IS2 - CS2          : median {s_is2_cs2['median']:+.2f} m, NMAD {s_is2_cs2['nmad']:.2f} m, "
              f"mean {s_is2_cs2['mean']:+.2f} m")

    if not rows:
        return None
    summ = pd.DataFrame(rows)
    summ.to_csv(output_dir / 'same_cell_is2_cs2_summary.csv', index=False, float_format='%.3f')
    print(f"  Summary -> {output_dir / 'same_cell_is2_cs2_summary.csv'}")
    return summ


def _rock_raster_for_dem(dem_path, rock_gdf, domain_gdf):
    """Boolean rock mask on the grid of `dem_path`: rock polygons inside the
    domain, shrunk by cfg.ROCK_EDGE_BUFFER_M (binary erosion in pixels).
    Returns (mask, transform, crs). Cached per DEM grid."""
    from rasterio.features import rasterize
    from scipy.ndimage import binary_erosion
    with rasterio.open(dem_path) as src:
        key = (src.transform, src.width, src.height, str(src.crs))
        if key in _ROCK_RASTER_CACHE:
            return _ROCK_RASTER_CACHE[key]
        transform, crs, shape = src.transform, src.crs, (src.height, src.width)
    rock = rock_gdf.to_crs(crs) if rock_gdf.crs != crs else rock_gdf
    dom = domain_gdf.to_crs(crs) if domain_gdf.crs != crs else domain_gdf
    rmask = rasterize(((g, 1) for g in rock.geometry if g is not None and not g.is_empty),
                      out_shape=shape, transform=transform, fill=0, dtype='uint8').astype(bool)
    dmask = rasterize(((g, 1) for g in dom.geometry if g is not None and not g.is_empty),
                      out_shape=shape, transform=transform, fill=0, dtype='uint8').astype(bool)
    px = abs(transform.a)
    n_erode = int(np.ceil(cfg.ROCK_EDGE_BUFFER_M / px)) if cfg.ROCK_EDGE_BUFFER_M > 0 else 0
    if n_erode > 0:
        rmask = binary_erosion(rmask, iterations=n_erode)
    out = (rmask & dmask, transform, crs)
    _ROCK_RASTER_CACHE[key] = out
    print(f"    rock mask on {Path(dem_path).name}: {out[0].sum() * px * px / 1e6:.1f} km² "
          f"(eroded by {n_erode} px = {n_erode * px:.0f} m, inside ROI + {cfg.ROCK_DOMAIN_BUFFER_M / 1000:.0f} km)")
    return out


def _rock_points_diff(pts, height_col, dem_path, rock_mask, transform, crs, label):
    """Keep points on rock, sample DEM + slope, return DataFrame with diff."""
    if pts is None or pts.empty:
        return pd.DataFrame()
    pts = pts.to_crs(crs) if pts.crs != crs else pts
    on_rock = _sample_array_at_points(rock_mask.astype(float), transform, crs, pts)
    pts = pts[on_rock == 1].copy()
    if pts.empty:
        print(f"    {label}: no points on rock")
        return pd.DataFrame()
    pts['tdx_h'] = sample_raster_values(pts, dem_path, valid_min=-9999, valid_max=10000)
    sl = sample_slope_from_own_dem(pts, dem_path)
    pts['slope'] = sl if sl is not None else np.nan
    pts['diff'] = pts['tdx_h'] - pts[height_col]
    n0 = len(pts)
    pts = pts[np.isfinite(pts['diff']) & (pts['diff'].abs() < cfg.ROCK_ABS_DIFF_MAX_M)].copy()
    print(f"    {label}: {n0:,} points on rock, {len(pts):,} with a valid TanDEM-X value "
          f"and |diff| < {cfg.ROCK_ABS_DIFF_MAX_M:.0f} m")
    return pts


def run_rock_comparison(cs2_comparator, is2_results, output_dir):
    """
    TanDEM-X minus CryoSat-2 / ICESat-2 on ice-free rock for every period
    with a DEM. Point-level and 500 m cell-level statistics (all rock and
    rock with slope <= cfg.ROCK_LOW_SLOPE_DEG), next to the glacier statistics
    of the main comparison for the same period. Writes
    rock_comparison_summary.csv and per-point GeoPackages.
    """
    output_dir = Path(output_dir); output_dir.mkdir(parents=True, exist_ok=True)
    print("\n" + "=" * 60)
    print("Rock-outcrop comparison (TanDEM-X minus altimeter on stable ground)")
    print("=" * 60)
    if not Path(cfg.ROCK_OUTCROP_PATH).exists():
        print(f"  Rock mask not found: {cfg.ROCK_OUTCROP_PATH} -- skipped")
        return None
    rock = gpd.read_file(cfg.ROCK_OUTCROP_PATH)
    roi = gpd.read_file(cfg.IS2_ROI_PATH)

    rows = []
    pairs = cs2_comparator.find_file_pairs() if cs2_comparator else []
    pooled = pooled_epochs([p['year'] for p in pairs])
    for pair in pairs:
        period, dem_path = pair['year'], pair['tdx_file']
        if period not in pooled:
            continue
        with rasterio.open(dem_path) as src:
            dem_crs = src.crs
        roi_d = roi.to_crs(dem_crs)
        domain = gpd.GeoDataFrame(geometry=[roi_d.geometry.union_all().buffer(cfg.ROCK_DOMAIN_BUFFER_M)
                                            if hasattr(roi_d.geometry, 'union_all')
                                            else roi_d.unary_union.buffer(cfg.ROCK_DOMAIN_BUFFER_M)],
                                  crs=dem_crs)
        rmask, transform, crs = _rock_raster_for_dem(dem_path, rock, domain)
        print(f"  {period}:")

        sensors = []
        # CryoSat-2 (same loading, time window and geoid step as compare_year())
        sub_years = [int(y) for y in period.split('-') if y.isdigit()]
        cs2 = cs2_comparator.load_cs2_shapefiles(sub_years)
        if not cs2.empty:
            cs2 = filter_by_tdx_window(cs2, pair['tdx_dates'],
                                       window_days=cs2_comparator.max_temporal_offset)
        if not cs2.empty:
            cs2 = convert_cs2_to_geoid(cs2, cs2_comparator.geoid_grid)
            sensors.append(('CryoSat-2', _rock_points_diff(cs2, 'elevation_geoid', dem_path,
                                                          rmask, transform, crs, 'CryoSat-2'),
                            cs2_comparator.comparisons.get(period, {}).get('gdf'), 'diff'))
        # ICESat-2 (same loader as process_year())
        if str(period) in [str(y) for y in cfg.IS2_YEARS]:
            is2 = load_is2_points(period, dem_crs, domain)
            sensors.append(('ICESat-2', _rock_points_diff(is2, 'h_egm2008', dem_path,
                                                         rmask, transform, crs, 'ICESat-2'),
                            is2_results.get(str(period), {}).get('gdf'), 'dh'))

        for sensor, pts, glac, gcol in sensors:
            extra = dict(sensor=sensor, period=period)
            if glac is not None and len(glac):
                rows.append(_summary_row('glacier (main comparison, 500 m cells)',
                                         glac[gcol].values, **extra))
            if pts is None or pts.empty:
                continue
            gpd.GeoDataFrame(pts[['diff', 'tdx_h', 'slope', 'geometry']], geometry='geometry',
                             crs=crs).to_file(output_dir / f'rock_{sensor}_{period}.gpkg', driver='GPKG')
            low = pts['slope'] <= cfg.ROCK_LOW_SLOPE_DEG
            rows.append(_summary_row('rock, points', pts['diff'].values, **extra))
            rows.append(_summary_row(f'rock, points, slope <= {cfg.ROCK_LOW_SLOPE_DEG:.0f} deg',
                                     pts.loc[low, 'diff'].values, **extra))
            hcol = 'elevation_geoid' if sensor == 'CryoSat-2' else 'h_egm2008'
            cells = aggregate_points_to_grid(gpd.GeoDataFrame(pts, geometry='geometry', crs=crs),
                                             value_col=hcol, std_cols=['tdx_h'])
            if len(cells):
                rows.append(_summary_row('rock, 500 m cells',
                                         (cells['tdx_h'] - cells[hcol]).values, **extra))
            d = pts['diff'].values
            print(f"    {sensor} on rock: N = {len(d):,}, median {np.median(d):+.2f} m, "
                  f"NMAD {nmad(d):.2f} m; slope <= {cfg.ROCK_LOW_SLOPE_DEG:.0f} deg: N = {int(low.sum()):,}, "
                  f"median {np.median(pts.loc[low, 'diff']) if low.any() else np.nan:+.2f} m")

    if not rows:
        return None
    summ = pd.DataFrame(rows)
    summ.to_csv(output_dir / 'rock_comparison_summary.csv', index=False, float_format='%.3f')
    print(f"  Summary -> {output_dir / 'rock_comparison_summary.csv'}")
    return summ
