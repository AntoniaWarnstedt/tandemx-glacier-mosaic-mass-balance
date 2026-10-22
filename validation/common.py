"""Shared helpers of the validation: TanDEM-X acquisition dates, geoid
conversion, raster sampling, co-located grid aggregation and statistics.
"""

import os
import pickle
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from pyproj import Transformer

from . import config as cfg

_AGG_ARTIFACT_STEMS = ('h_egm96', 'h_egm2008', 'h_elv', 'tdx_h', 'tdx_elev',
                       'tdx_slope', 'elevation', 'elevation_mean')  # own gridding columns, not quality fields
_OWN_DEM_SLOPE_CACHE = {}  # DEM path -> (slope, transform, crs)


def get_tdx_dates(year_str) -> list:
    """Return pd.Timestamps for all TDX acquisitions matching year_str
    (CS2 pipeline's acquisition-date list)."""
    sub_years = year_str.split('-') if '-' in year_str else [str(year_str)]
    dates = []
    for y in sub_years:
        dates.extend(cfg.TDX_ACQUISITIONS.get(y, []))
    return [pd.to_datetime(d) for d in dates]


def epoch_years(epoch) -> list:
    """Single years of an epoch label ("2016-2017" -> ["2016", "2017"])."""
    return str(epoch).split('-')


def pooled_epochs(labels) -> list:
    """Epochs pooled into "All Years": cfg.ALL_YEARS_PERIOD_KEYS, or all labels
    with merged epochs replacing their single years."""
    if cfg.ALL_YEARS_PERIOD_KEYS:
        return list(cfg.ALL_YEARS_PERIOD_KEYS)
    labels = [str(x) for x in labels]
    covered = {y for x in labels if '-' in x for y in epoch_years(x)}
    return sorted((x for x in labels if x not in covered), key=lambda x: epoch_years(x))


def display_epoch(epoch) -> str:
    """Display form of an epoch label ("2016-2017" -> "2016/17")."""
    parts = epoch_years(epoch)
    return f"{parts[0]}/{parts[-1][-2:]}" if len(parts) == 2 else str(epoch)


def decimal_year_to_timestamp(dec_year) -> pd.Series:
    """
    Shared by both pipelines: the CS2 loader falls back to this when a
    shapefile has 'dec_year' but no 'datetime' column, and the ICESat-2 loader
    can use it when only a 't_year'-style column is available (no day-level
    timestamp at all) -- see process_year()'s date_col handling.
    """
    dec_year = np.asarray(dec_year, dtype=float)
    finite = np.isfinite(dec_year)
    dec_year_safe = np.where(finite, dec_year, 2000.0)  # placeholder, masked out below
    year = np.floor(dec_year_safe).astype(int)
    frac = dec_year_safe - year
    valid = finite & (year >= 1) & (year <= 9999)
    year_safe = np.where(valid, year, 2000)
    starts      = pd.to_datetime({'year': year_safe,     'month': 1, 'day': 1})
    next_starts = pd.to_datetime({'year': year_safe + 1, 'month': 1, 'day': 1})
    result = starts + frac * (next_starts - starts)
    return result.where(valid, pd.NaT)


def filter_by_tdx_window(gdf: gpd.GeoDataFrame,
                          tdx_dates: list,
                          window_days: int = 30) -> gpd.GeoDataFrame:
    """Keep rows within ±window_days of *any* TDX acquisition date.

    Adds 'temp_off' column = minimum absolute offset (days) to nearest date.
    """
    if gdf.empty or 'datetime' not in gdf.columns:
        return gdf

    before = len(gdf)
    diffs  = pd.concat(
        [(gdf['datetime'] - t).abs() for t in tdx_dates], axis=1
    ).min(axis=1)

    out = gdf.copy()
    out['temp_off'] = diffs.dt.days
    out = out[out['temp_off'] <= window_days]
    print(f"  TDX window filter (±{window_days} d, {len(tdx_dates)} date(s)): "
          f"{before:,} → {len(out):,} points")
    return out


def ellipsoidal_to_geoid(lon, lat, h_ellipsoidal, geoid_grid: str):
    """Convert WGS84 ellipsoidal heights to a geoid-referenced height,
    using any PROJ-readable undulation grid (.gtx or .tif) via
    +geoidgrids. Returns a plain numpy array."""
    lon = np.asarray(lon, dtype=float)
    lat = np.asarray(lat, dtype=float)
    h_ellipsoidal = np.asarray(h_ellipsoidal, dtype=float)
    proj_str = f"+proj=longlat +datum=WGS84 +geoidgrids={geoid_grid} +type=crs"
    transformer = Transformer.from_crs("EPSG:4979", proj_str, always_xy=True)
    _, _, h_geoid = transformer.transform(lon, lat, h_ellipsoidal)
    return h_geoid


def geoid_to_ellipsoidal(lon, lat, h_geoid, geoid_grid: str):
    """Inverse of ellipsoidal_to_geoid(): recover WGS84 ellipsoidal
    heights from a height already referenced to `geoid_grid`."""
    lon = np.asarray(lon, dtype=float)
    lat = np.asarray(lat, dtype=float)
    h_geoid = np.asarray(h_geoid, dtype=float)
    proj_str = f"+proj=longlat +datum=WGS84 +geoidgrids={geoid_grid} +type=crs"
    transformer = Transformer.from_crs(proj_str, "EPSG:4979", always_xy=True)
    _, _, h_ellipsoidal = transformer.transform(lon, lat, h_geoid)
    return h_ellipsoidal


def regeoid(lon, lat, h, from_grid, to_grid):
    """
    Re-reference a height array from one vertical datum to another, routing
    through the WGS84 ellipsoid as the common intermediate.
    """
    h_ellipsoidal = (geoid_to_ellipsoidal(lon, lat, h, from_grid)
                      if from_grid is not None else np.asarray(h, dtype=float))
    return (ellipsoidal_to_geoid(lon, lat, h_ellipsoidal, to_grid)
            if to_grid is not None else h_ellipsoidal)


def convert_gdf_to_egm2008(gdf: gpd.GeoDataFrame, height_col: str,
                            from_grid=None, out_col: str = 'elevation_egm2008',
                            to_grid: str = cfg.GEOID_GRID_EGM2008) -> gpd.GeoDataFrame:
    """
    Add `out_col` to a copy of `gdf`: the values in `gdf[height_col]` converted
    to EGM2008 (or whatever `to_grid` is) geoid heights.
    """
    gdf_ll = (gdf.to_crs(epsg=4326) if
              (gdf.crs is not None and gdf.crs.is_projected) else gdf)
    out = gdf.copy()
    out[out_col] = regeoid(
        gdf_ll.geometry.x.values, gdf_ll.geometry.y.values,
        gdf[height_col].values, from_grid, to_grid,
    )
    out[f'{out_col}_offset'] = out[out_col] - gdf[height_col].values
    return out


def convert_cs2_to_geoid(cs2_gdf: gpd.GeoDataFrame,
                          geoid_grid: str) -> gpd.GeoDataFrame:
    """
    Convert CryoSat-2 ellipsoidal heights (WGS84) to geoid heights referenced
    to `geoid_grid` (EGM2008, via GEOID_GRID_EGM2008 / GEOID_GRID_EGM2008
    elsewhere in this file).
    """
    return convert_gdf_to_egm2008(cs2_gdf, 'elevation', from_grid=None,
                                   out_col='elevation_geoid', to_grid=geoid_grid)


def sample_raster_values(points_gdf: gpd.GeoDataFrame, raster_path,
                          band: int = 1, valid_min: float = None,
                          valid_max: float = None) -> np.ndarray:
    """
    Nearest-pixel sample of `raster_path` at every point (points are
    reprojected to the raster CRS if needed). Nodata, non-finite values and
    anything outside (valid_min, valid_max) -- exclusive, if given -- come back
    as NaN, so the result can be averaged directly without a sentinel like
    -9999 leaking into a cell mean.
    """
    n = len(points_gdf)
    out = np.full(n, np.nan)
    if n == 0:
        return out
    with rasterio.open(raster_path) as src:
        pts = (points_gdf.to_crs(src.crs)
               if points_gdf.crs is not None and points_gdf.crs != src.crs else points_gdf)
        arr = src.read(band).astype(float)
        nodata, transform = src.nodata, src.transform
    if nodata is not None and np.isfinite(nodata):
        arr[arr == nodata] = np.nan
    rows, cols = rasterio.transform.rowcol(transform, pts.geometry.x.values,
                                           pts.geometry.y.values, op=np.floor)
    rows, cols = np.asarray(rows, dtype=np.int64), np.asarray(cols, dtype=np.int64)
    inside = (rows >= 0) & (rows < arr.shape[0]) & (cols >= 0) & (cols < arr.shape[1])
    out[inside] = arr[rows[inside], cols[inside]]
    out[~np.isfinite(out)] = np.nan
    if valid_min is not None:
        out[out <= valid_min] = np.nan
    if valid_max is not None:
        out[out >= valid_max] = np.nan
    return out


def _slope_deg_from_dem_array(elev: np.ndarray, transform) -> np.ndarray:
    """Simple central-difference (Horn-style) slope in degrees from an
    elevation array and its affine transform."""
    dx = transform.a
    dy = -transform.e  # transform.e is negative for north-up rasters
    dzdy, dzdx = np.gradient(elev, dy, dx)
    return np.degrees(np.arctan(np.sqrt(dzdx ** 2 + dzdy ** 2)))


def _slope_array_from_dem_file(dem_path):
    """Slope (degrees) + its transform/CRS for one DEM file, cached by
    path. Returns None (not raises) if the file is missing, same
    fail-soft policy as the other raster helpers in this script."""
    if not dem_path:
        return None
    dem_path = str(dem_path)
    if dem_path in _OWN_DEM_SLOPE_CACHE:
        return _OWN_DEM_SLOPE_CACHE[dem_path]
    if not Path(dem_path).exists():
        return None
    with rasterio.open(dem_path) as src:
        elev = src.read(1).astype(float)
        nodata = src.nodata
        transform, crs = src.transform, src.crs
    if nodata is not None:
        elev[elev == nodata] = np.nan
    result = (_slope_deg_from_dem_array(elev, transform), transform, crs)
    _OWN_DEM_SLOPE_CACHE[dem_path] = result
    return result


def _sample_array_at_points(arr, transform, crs, points_gdf):
    """Nearest-pixel sample of an in-memory array at each point in
    points_gdf (reprojected to the array's CRS first if needed)."""
    if len(points_gdf) == 0:
        return np.array([])
    pts = (points_gdf.to_crs(crs)
           if points_gdf.crs is not None and points_gdf.crs != crs else points_gdf)
    rows, cols = rasterio.transform.rowcol(transform, pts.geometry.x.values, pts.geometry.y.values)
    rows, cols = np.asarray(rows), np.asarray(cols)
    out = np.full(len(pts), np.nan)
    valid = (rows >= 0) & (rows < arr.shape[0]) & (cols >= 0) & (cols < arr.shape[1])
    out[valid] = arr[rows[valid], cols[valid]]
    return out


def sample_slope_from_own_dem(points_gdf, dem_path):
    """
    Slope (degrees) computed from the SPECIFIC TanDEM-X DEM file being
    validated in this panel/epoch -- the single-DEM-file equivalent of
    sample_slope_from_own_dem_multi(), used where every row in points_gdf
    shares one epoch (e.g. the ICESat-2 2024/25 panel).
    """
    cached = _slope_array_from_dem_file(dem_path)
    if cached is None:
        return None
    arr, transform, crs = cached
    return _sample_array_at_points(arr, transform, crs, points_gdf)


def sample_slope_from_own_dem_multi(gdf, path_col='tdx_file'):
    if path_col not in gdf.columns:
        return None
    out = np.full(len(gdf), np.nan)
    for dem_path, sub in gdf.groupby(path_col):
        cached = _slope_array_from_dem_file(dem_path)
        if cached is None:
            continue
        arr, transform, crs = cached
        out[gdf.index.get_indexer(sub.index)] = _sample_array_at_points(arr, transform, crs, sub)
    return out


def _require_metric_crs(gdf, fn_name):
    """Raise unless gdf is in a projected, metre-based CRS (see
    aggregate_points_to_grid() for why this is a hard error)."""
    if gdf.crs is None or not gdf.crs.is_projected:
        raise ValueError(
            f"{fn_name}() requires a projected, metre-based CRS -- got "
            f"{gdf.crs!r}. Reproject to the TanDEM-X raster's CRS before "
            f"aggregating, so that grid_size is unambiguously in metres.")
    unit_name = gdf.crs.axis_info[0].unit_name.lower() if gdf.crs.axis_info else ''
    if 'metre' not in unit_name and 'meter' not in unit_name:
        raise ValueError(
            f"{fn_name}() expects a metre-based CRS; the current CRS "
            f"({gdf.crs}) reports axis units of {unit_name!r}.")


def grid_cell_index(x, y, grid_size: float = None, origin: tuple = None):
    """Integer (column, row) index of the AGGREGATION grid cell containing
    each (x, y). Cell k spans [x0 + k*grid_size, x0 + (k+1)*grid_size)."""
    grid_size = cfg.AGGREGATION_GRID_SIZE_M if grid_size is None else grid_size
    ox, oy = cfg.AGGREGATION_ORIGIN_M if origin is None else origin
    xi = np.floor((np.asarray(x, dtype=float) - ox) / grid_size).astype(np.int64)
    yi = np.floor((np.asarray(y, dtype=float) - oy) / grid_size).astype(np.int64)
    return xi, yi


def aggregate_points_to_grid(gdf: gpd.GeoDataFrame,
                              value_col: str,
                              extra_cols: list = None,
                              std_cols: list = None,
                              grid_size: float = None,
                              origin: tuple = None,
                              min_points: int = 1) -> gpd.GeoDataFrame:
    """
    Aggregate point observations onto a regular grid, averaging `value_col`
    (and any of `extra_cols` / `std_cols` that are present) within each
    occupied cell. Shared by the CS2, ICESat-2 AND CS2-dhdt comparisons, so
    "aggregate to a grid" means exactly the same thing -- same grid size, same
    grid registration, same statistic -- everywhere (see
    AGGREGATION_GRID_SIZE_M for the grid-size reasoning).
    """
    if gdf.empty:
        return gpd.GeoDataFrame()
    _require_metric_crs(gdf, 'aggregate_points_to_grid')

    grid_size = cfg.AGGREGATION_GRID_SIZE_M if grid_size is None else grid_size
    ox, oy = cfg.AGGREGATION_ORIGIN_M if origin is None else origin

    present_std   = [c for c in (std_cols or []) if c in gdf.columns and c != value_col]
    present_extra = [c for c in (extra_cols or [])
                     if c in gdf.columns and c != value_col and c not in present_std]

    xi, yi = grid_cell_index(gdf.geometry.x.values, gdf.geometry.y.values, grid_size, (ox, oy))
    tmp = pd.DataFrame({'x_bin': xi, 'y_bin': yi})
    for c in [value_col] + present_std + present_extra:
        tmp[c] = pd.to_numeric(pd.Series(gdf[c].values), errors='coerce').values

    grp = tmp.groupby(['x_bin', 'y_bin'], sort=True)
    agg = grp[[value_col] + present_std + present_extra].mean()
    for c in [value_col] + present_std:
        agg[f'{c}_std'] = grp[c].std()          # ddof=1, NaN for n=1 (as before)
    agg['n_points'] = grp.size()
    agg = agg[agg['n_points'] >= max(int(min_points), 1)].reset_index()
    if agg.empty:
        return gpd.GeoDataFrame()

    agg['x'] = ox + (agg['x_bin'] + 0.5) * grid_size
    agg['y'] = oy + (agg['y_bin'] + 0.5) * grid_size
    lead = ['x', 'y', 'x_bin', 'y_bin', value_col, f'{value_col}_std', 'n_points']
    agg = agg[lead + [c for c in agg.columns if c not in lead]]
    geom = gpd.points_from_xy(agg['x'], agg['y'])
    return gpd.GeoDataFrame(agg, geometry=geom, crs=gdf.crs)


def colocated_aggregate(points_gdf: gpd.GeoDataFrame, sensor_col: str,
                         raster_col: str, label: str,
                         extra_cols: list = None, std_cols: list = None,
                         diff_abs_max: float = None,
                         min_points: int = 1,
                         grid_size: float = None,
                         origin: tuple = None) -> gpd.GeoDataFrame:
    """
    Co-located aggregation shared by the absolute-elevation (CS2, IS2) and dhdt
    comparisons. Expects `points_gdf` to already carry the raster value sampled
    at each point in `raster_col` (NaN = nodata).
    """
    n0 = len(points_gdf)
    rv = points_gdf[raster_col].values.astype(float)
    sv = points_gdf[sensor_col].values.astype(float)
    keep = np.isfinite(rv) & np.isfinite(sv)
    n_nodata = int((~keep).sum())
    n_outlier = 0
    if diff_abs_max is not None:
        big = keep & (np.abs(rv - sv) >= diff_abs_max)
        n_outlier = int(big.sum())
        keep &= ~big
    pts = points_gdf[keep]
    print(f"    {label} co-located points: {n0:,} -> {len(pts):,} "
          f"({n_nodata:,} without a valid TanDEM-X value"
          + (f", {n_outlier:,} with |diff| >= {diff_abs_max:.0f}" if diff_abs_max is not None else '')
          + ")")
    if pts.empty:
        return gpd.GeoDataFrame()

    grid_size = cfg.AGGREGATION_GRID_SIZE_M if grid_size is None else grid_size
    cells_all = aggregate_points_to_grid(
        pts, value_col=sensor_col, extra_cols=extra_cols,
        std_cols=[raster_col] + list(std_cols or []),
        grid_size=grid_size, origin=origin, min_points=1)
    print_aggregation_summary(label, len(pts), cells_all, grid_size)
    if cells_all.empty or min_points <= 1:
        return cells_all
    cells = cells_all[cells_all['n_points'] >= min_points].copy()
    print(f"    {label}: {len(cells):,} of {len(cells_all):,} cells have "
          f">= {min_points} co-located points")
    return cells


def write_cells_geotiff(cells: gpd.GeoDataFrame, value_col: str, path,
                         grid_size: float = None, origin: tuple = None):
    """
    Write per-cell values (output of aggregate_points_to_grid(), or any points
    in a metric CRS) to a GeoTIFF whose pixels ARE the aggregation cells --
    same size, same origin -- so the raster lines up exactly with every other
    product on that grid. If several rows fall in the same cell (e.g. the All-
    Years concat of three epochs) they are averaged.
    """
    if cells is None or len(cells) == 0 or value_col not in cells.columns:
        return
    grid_size = cfg.AGGREGATION_GRID_SIZE_M if grid_size is None else grid_size
    ox, oy = cfg.AGGREGATION_ORIGIN_M if origin is None else origin
    # Recomputed from geometry (cell centres fall inside their own cell),
    # so this also works for raw points or a different grid_size.
    xi, yi = grid_cell_index(cells.geometry.x.values, cells.geometry.y.values,
                             grid_size, (ox, oy))
    v = pd.to_numeric(cells[value_col], errors='coerce').values.astype(float)
    ok = np.isfinite(v)
    if not ok.any():
        return
    xi, yi, v = xi[ok], yi[ok], v[ok]
    x_min, x_max, y_min, y_max = xi.min(), xi.max(), yi.min(), yi.max()
    width, height = int(x_max - x_min + 1), int(y_max - y_min + 1)
    r = (y_max - yi).astype(np.int64)            # north-up: row 0 = top cell
    c = (xi - x_min).astype(np.int64)
    s = np.zeros((height, width)); n = np.zeros((height, width))
    np.add.at(s, (r, c), v); np.add.at(n, (r, c), 1)
    with np.errstate(invalid='ignore', divide='ignore'):
        arr = np.where(n > 0, s / np.maximum(n, 1), np.nan).astype('float32')
    tr = rasterio.transform.from_origin(ox + x_min * grid_size,
                                        oy + (y_max + 1) * grid_size,
                                        grid_size, grid_size)
    path = Path(path)
    with rasterio.open(path, 'w', driver='GTiff', height=height, width=width,
                       count=1, dtype='float32', crs=cells.crs, transform=tr,
                       compress='lzw', nodata=np.nan) as dst:
        dst.write(arr, 1)
    print(f"    {grid_size:.0f} m raster ({value_col}) -> {path}")


def is2_quality_columns(columns) -> list:
    """ICESat-2 quality/uncertainty columns (IS2_QUALITY_KEYWORDS match),
    excluding '<stem>_std' artifacts produced by aggregate_points_to_grid()."""
    artifacts = {f'{s}_std' for s in _AGG_ARTIFACT_STEMS}
    return [c for c in columns
            if c not in artifacts and any(k in c.lower() for k in cfg.IS2_QUALITY_KEYWORDS)]


def print_aggregation_summary(label: str, n_before: int,
                               agg_gdf: gpd.GeoDataFrame,
                               grid_size: float = None) -> None:
    """
    Print how many raw points went into aggregate_points_to_grid() versus how
    many occupied cells came out, plus the resulting points-per-cell density.
    """
    grid_size = cfg.AGGREGATION_GRID_SIZE_M if grid_size is None else grid_size
    n_cells = len(agg_gdf) if agg_gdf is not None else 0
    if n_cells == 0:
        print(f"    {label}: {n_before:,} raw points -> 0 occupied "
              f"{grid_size:.0f} m cells")
        return
    n_per_cell = agg_gdf['n_points'].values
    print(f"    {label}: {n_before:,} raw points -> {n_cells:,} occupied "
          f"{grid_size:.0f} m cells "
          f"(median {int(np.median(n_per_cell))}, "
          f"mean {n_per_cell.mean():.1f}, "
          f"max {int(n_per_cell.max())} points/cell)")


def print_validity_breakdown(label: str, tdx_vals, diff_vals,
                              tdx_lo: float = -9999, tdx_hi: float = 10000,
                              diff_abs_max: float = 100) -> None:
    """
    Break down WHY points fail the tdx_elev/diff validity filter, instead of
    only reporting the final pass-rate percentage.
    """
    tdx_vals  = np.asarray(tdx_vals,  dtype=float)
    diff_vals = np.asarray(diff_vals, dtype=float)
    n = len(tdx_vals)
    if n == 0:
        print(f"    {label} validity breakdown: 0 points to check")
        return

    tdx_nan       = ~np.isfinite(tdx_vals)
    tdx_out_range = np.isfinite(tdx_vals) & ((tdx_vals <= tdx_lo) | (tdx_vals >= tdx_hi))
    diff_nan      = np.isfinite(tdx_vals) & ~tdx_out_range & ~np.isfinite(diff_vals)
    diff_too_big  = (np.isfinite(tdx_vals) & ~tdx_out_range & np.isfinite(diff_vals)
                     & (np.abs(diff_vals) >= diff_abs_max))
    valid = np.isfinite(tdx_vals) & ~tdx_out_range & np.isfinite(diff_vals) & ~diff_too_big

    print(f"    {label} validity breakdown (of {n:,}): "
          f"{tdx_nan.sum():,} TDX NaN/nodata, "
          f"{tdx_out_range.sum():,} TDX outside [{tdx_lo:.0f}, {tdx_hi:.0f}], "
          f"{diff_nan.sum():,} diff NaN, "
          f"{diff_too_big.sum():,} |diff| >= {diff_abs_max:.0f} m "
          f"-> {valid.sum():,} valid ({100*valid.sum()/n:.1f}%)")


def nmad(x) -> float:
    """Normalised median absolute deviation, 1.4826 * median(|x - median(x)|)
    (Hoehle & Hoehle, 2009). NaNs are ignored; NaN for an empty input."""
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return np.nan
    return float(1.4826 * np.median(np.abs(x - np.median(x))))


def robust_summary(d, prefix: str = '') -> dict:
    """Mean/SD and median/NMAD (plus N, RMSE, 5th/95th percentile) of a
    difference array, as one dict."""
    d = np.asarray(d, dtype=float)
    d = d[np.isfinite(d)]
    keys = ['n', 'mean', 'std', 'median', 'nmad', 'rmse', 'p5', 'p95']
    if d.size == 0:
        return {f'{prefix}{k}': (0 if k == 'n' else np.nan) for k in keys}
    vals = [d.size, d.mean(), d.std(), np.median(d), nmad(d),
            np.sqrt(np.mean(d ** 2)), np.percentile(d, 5), np.percentile(d, 95)]
    return {f'{prefix}{k}': v for k, v in zip(keys, vals)}


def center_spread_cols():
    """(centre column, spread column, centre name, spread name) of the banded
    panels, following cfg.ROBUST_STATS."""
    if cfg.ROBUST_STATS:
        return 'median_diff', 'nmad_diff', 'Median', 'NMAD'
    return 'mean_diff', 'std_diff', 'Mean', 'std'


def compute_hypsometry(gdf, elev_col='tdx_elev', diff_col='diff',
                        bin_width=cfg.HYPS_BIN_WIDTH_M,
                        min_samples=cfg.HYPS_MIN_SAMPLES) -> pd.DataFrame:
    """Compute elevation-banded (hypsometric) difference statistics."""
    elevs = gdf[elev_col].values
    diffs = gdf[diff_col].values

    e_min = np.floor(np.nanmin(elevs) / bin_width) * bin_width
    e_max = np.ceil (np.nanmax(elevs) / bin_width) * bin_width
    edges = np.arange(e_min, e_max + bin_width, bin_width)

    rows = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        mask = (elevs >= lo) & (elevs < hi)
        d    = diffs[mask]
        d    = d[np.isfinite(d)]
        if len(d) < min_samples:
            continue
        rows.append({
            'elev_lo':     lo,
            'elev_hi':     hi,
            'elev_mid':    (lo + hi) / 2,
            'n':           len(d),
            'mean_diff':   np.mean(d),
            'median_diff': np.median(d),
            'std_diff':    np.std(d),
            'nmad_diff':   nmad(d),
            'rmse':        np.sqrt(np.mean(d**2)),
            'p5':          np.percentile(d, 5),
            'p95':         np.percentile(d, 95),
        })

    df = pd.DataFrame(rows)
    print(f"  Hypsometry: {len(df)} bands "
          f"({bin_width:.0f} m each, >={min_samples} samples/band)")
    return df


def save_state(results):
    """Store the comparison results (cs2 / is2 / dhdt) for the plotting run."""
    os.makedirs(os.path.dirname(cfg.STATE_FILE), exist_ok=True)
    with open(cfg.STATE_FILE, "wb") as f:
        pickle.dump(results, f)
    print(f"Saved validation state: {cfg.STATE_FILE}")


def load_state():
    if not os.path.exists(cfg.STATE_FILE):
        raise FileNotFoundError(f"{cfg.STATE_FILE} not found -- run run_validation_processing.py first.")
    with open(cfg.STATE_FILE, "rb") as f:
        return pickle.load(f)
