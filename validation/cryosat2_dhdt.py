"""CryoSat-2 vs. TanDEM-X elevation-change rates: CS2 points of two epochs are
paired (<= 30 m apart), converted to rates and compared with the TanDEM-X dh/dt
on the shared 500 m grid.
"""

import re
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from pyproj import Transformer
from scipy.spatial import cKDTree

from . import config as cfg
from .common import (
    epoch_years,
    colocated_aggregate,
    compute_hypsometry,
    convert_cs2_to_geoid,
    decimal_year_to_timestamp,
    filter_by_tdx_window,
    get_tdx_dates,
    sample_raster_values,
    sample_slope_from_own_dem,
    write_cells_geotiff,
)


def load_cs2_points_dhdt(years: list, shp_dir: str = cfg.CS2_SHP_DIR) -> gpd.GeoDataFrame:
    """
    Load and concatenate cs2_points_YYYY.gpkg/.shp for each year in `years`.
    Standalone analogue of CryoSatTanDEMXComparison.load_cs2_shapefiles()
    (identical logic) for use outside the class, since the dhdt pipeline needs
    to load several different year-groupings of the same files.
    """
    shp_dir = Path(shp_dir)
    gdfs = []
    for y in years:
        gpkg = shp_dir / f'cs2_points_{y}.gpkg'
        shp  = shp_dir / f'cs2_points_{y}.shp'
        path = gpkg if gpkg.exists() else shp if shp.exists() else None
        if path is None:
            print(f"    WARNING: No CS2 file for year {y} "
                  f"(tried {gpkg.name} and {shp.name})")
            continue
        print(f"    Loading {path.name}...")
        gdf = gpd.read_file(path)
        if 'datetime' in gdf.columns:
            gdf['datetime'] = pd.to_datetime(gdf['datetime'], errors='coerce')
        elif 'dec_year' in gdf.columns:
            gdf['datetime'] = decimal_year_to_timestamp(gdf['dec_year'].values).values
            print("      NOTE: 'datetime' absent; reconstructed from dec_year")
        if 'elev_unc' not in gdf.columns and 'uncertainty' in gdf.columns:
            gdf = gdf.rename(columns={'uncertainty': 'elev_unc'})
        gdfs.append(gdf)

    if not gdfs:
        return gpd.GeoDataFrame()

    combined = pd.concat(gdfs, ignore_index=True) if len(gdfs) > 1 else gdfs[0]
    combined = combined[combined['datetime'].notna()].copy()
    return combined


def prepare_dhdt_timestamp_points(label: str, years: list, target_crs,
                                   geoid_grid: str = cfg.GEOID_GRID_EGM2008) -> gpd.GeoDataFrame:
    """
    Load, ±30-day-window filter, geoid-convert, and reproject all CS2 points
    belonging to one named dhdt timestamp (e.g. '2016-2017'). Uses the SAME
    get_tdx_dates()/filter_by_tdx_window() as the absolute-elevation pipeline,
    so both comparisons apply the identical temporal-matching rule.
    """
    print(f"  Timestamp '{label}' (years {years}):")
    gdf = load_cs2_points_dhdt(years)
    if gdf.empty:
        print(f"    WARNING: no CS2 points found for years {years}")
        return gpd.GeoDataFrame()

    tdx_dates = get_tdx_dates(label)
    gdf = filter_by_tdx_window(gdf, tdx_dates, window_days=cfg.CS2_MAX_TEMPORAL_OFFSET)
    if gdf.empty:
        print(f"    WARNING: no points left within ±{cfg.CS2_MAX_TEMPORAL_OFFSET} days "
              f"of a TDX acquisition for '{label}'")
        return gpd.GeoDataFrame()

    if geoid_grid and Path(geoid_grid).exists():
        gdf = convert_cs2_to_geoid(gdf, geoid_grid)
        gdf['elevation'] = gdf['elevation_geoid']
    else:
        print("    WARNING: Geoid grid not found - using ellipsoidal heights.")

    gdf = gdf.to_crs(target_crs) if gdf.crs != target_crs else gdf
    gdf = gdf[np.isfinite(gdf['elevation'].astype(float)) & gdf['datetime'].notna()].copy()
    print(f"    {len(gdf):,} usable points")
    return gdf


def match_nearest_points(early: gpd.GeoDataFrame, late: gpd.GeoDataFrame,
                          max_dist: float = cfg.DHDT_MAX_PAIR_DISTANCE_M) -> pd.DataFrame:
    """
    For every point in `early`, find its nearest neighbour (Euclidean distance,
    both already in the same projected CRS) in `late`, keeping the pair only if
    the separation is <= max_dist.
    """
    if early.empty or late.empty:
        return pd.DataFrame()

    early_xy = np.column_stack([early.geometry.x.values, early.geometry.y.values])
    late_xy  = np.column_stack([late.geometry.x.values,  late.geometry.y.values])

    tree = cKDTree(late_xy)
    dist, idx = tree.query(early_xy, k=1)
    keep = dist <= max_dist
    if not np.any(keep):
        return pd.DataFrame()

    e = early[keep].reset_index(drop=True)
    l = late.iloc[idx[keep]].reset_index(drop=True)

    out = pd.DataFrame({
        'x_early':     e.geometry.x.values,
        'y_early':     e.geometry.y.values,
        'x_late':      l.geometry.x.values,
        'y_late':      l.geometry.y.values,
        'elev_early':  e['elevation'].values,
        'elev_late':   l['elevation'].values,
        'date_early':  pd.to_datetime(e['datetime'].values),
        'date_late':   pd.to_datetime(l['datetime'].values),
        'pair_dist_m': dist[keep],
    })
    out['x_mid'] = (out['x_early'] + out['x_late']) / 2.0
    out['y_mid'] = (out['y_early'] + out['y_late']) / 2.0
    out['dh'] = out['elev_late'] - out['elev_early']
    out['dt_years'] = (out['date_late'] - out['date_early']).dt.days / 365.25
    return out


def filter_dhdt_pairs(pairs: pd.DataFrame,
                       min_dt_years: float = cfg.DHDT_MIN_PAIR_DT_YEARS,
                       max_abs_rate: float = cfg.DHDT_MAX_ABS_RATE_M_PER_YR) -> pd.DataFrame:
    """Apply the dt / rate sanity filters and compute the final 'rate' column."""
    if pairs.empty:
        return pairs
    before = len(pairs)
    pairs = pairs[np.isfinite(pairs['dt_years']) & (pairs['dt_years'] >= min_dt_years)].copy()
    pairs['rate'] = pairs['dh'] / pairs['dt_years']
    pairs = pairs[np.isfinite(pairs['rate']) & (pairs['rate'].abs() <= max_abs_rate)].copy()
    print(f"    Pair filtering (dt >= {min_dt_years} yr, |rate| <= {max_abs_rate} m a⁻¹): "
          f"{before:,} -> {len(pairs):,} pairs")
    return pairs


def compute_dhdt_diff_stats(tdx_vals: np.ndarray, cs2_vals: np.ndarray, label: str,
                             n_pairs: np.ndarray = None) -> dict:
    """TanDEM-X-minus-CryoSat-2 dhdt difference statistics over all
    co-located grid cells (DHDT_GRID_SIZE_M). `n_pairs` = number of
    co-located CS2 point pairs per cell (optional, reported only)."""
    diff = tdx_vals - cs2_vals
    nmad = 1.4826 * np.median(np.abs(diff - np.median(diff)))
    r = np.corrcoef(tdx_vals, cs2_vals)[0, 1] if len(diff) > 1 else np.nan
    stats = {
        'comparison':    label,
        'n_cells':       len(diff),
        'grid_size_m':   cfg.DHDT_GRID_SIZE_M,
        'n_pairs_total': int(np.sum(n_pairs)) if n_pairs is not None else np.nan,
        'median_pairs_per_cell': float(np.median(n_pairs)) if n_pairs is not None else np.nan,
        'mean_diff':     np.mean(diff),
        'median_diff':   np.median(diff),
        'std_diff':      np.std(diff),
        'nmad_diff':     nmad,
        'rmse':          np.sqrt(np.mean(diff ** 2)),
        'mae':           np.mean(np.abs(diff)),
        'pct_5':         np.percentile(diff, 5),
        'pct_95':        np.percentile(diff, 95),
        'mean_tdx_rate': np.mean(tdx_vals),
        'mean_cs2_rate': np.mean(cs2_vals),
        'r_pearson':     r,
    }
    print(f"\n  dhdt stats [{label}]  (TanDEM-X minus CryoSat-2, m a⁻¹)")
    print(f"    N cells ({cfg.DHDT_GRID_SIZE_M:.0f} m)     : {stats['n_cells']:,}")
    print(f"    Mean diff          : {stats['mean_diff']:+.3f}")
    print(f"    Std                : {stats['std_diff']:.3f}")
    print(f"    RMSE               : {stats['rmse']:.3f}")
    print(f"    Pearson r          : {stats['r_pearson']:.3f}")
    return stats


def compute_dhdt_hypsometry(elev: np.ndarray, diff: np.ndarray,
                             bin_width: float = cfg.HYPS_BIN_WIDTH_M,
                             min_samples: int = cfg.HYPS_MIN_SAMPLES) -> pd.DataFrame:
    """
    Bin the per-cell (TanDEM-X minus CryoSat-2) dhdt difference by TanDEM-X
    surface elevation. `elev` is the co-located mean of
    dem_<later_timestamp>.tif at the CS2 pair midpoints of each cell, so no
    grid-matching between the DEM and the dhdt raster is required any more.
    Same binning as compute_hypsometry() for CS2/IS2.
    """
    df = pd.DataFrame({'e': np.asarray(elev, dtype=float),
                       'd': np.asarray(diff, dtype=float)}).dropna()
    if df.empty:
        print("    NOTE: no cells with a valid reference elevation; "
              "skipping hypsometric dhdt bias.")
        return pd.DataFrame()
    return compute_hypsometry(df, elev_col='e', diff_col='d',
                              bin_width=bin_width, min_samples=min_samples)


def run_dhdt_comparison(early_label: str, late_label: str, tdx_dhdt_filename: str,
                         output_dir: str = cfg.CS2_DHDT_OUTPUT_DIR,
                         geoid_grid: str = cfg.GEOID_GRID_EGM2008,
                         use_cache: bool = cfg.USE_DHDT_CACHE):
    """
    Full dhdt comparison for one timestamp pair, using CO-LOCATED aggregation
    on the shared DHDT_GRID_SIZE_M grid:
    """
    label = f'{late_label}_minus_{early_label}'
    print(f"\n{'=' * 60}")
    print(f"dhdt comparison: {label}")
    print(f"{'=' * 60}")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    tdx_path = Path(cfg.TDX_DIR) / tdx_dhdt_filename
    if not tdx_path.exists():
        print(f"  WARNING: TanDEM-X dhdt raster not found, skipping pair: {tdx_path}")
        return None

    with rasterio.open(tdx_path) as src:
        crs, res, shape = src.crs, src.res, src.shape
    print(f"  TanDEM-X dhdt raster: {shape[0]} x {shape[1]} px at "
          f"{res[0]:.0f} x {res[1]:.0f} m, CRS {crs} -> compared on a "
          f"{cfg.DHDT_GRID_SIZE_M:.0f} m co-located grid")

    # ── CS2 point-pair rates (cached as a CSV: the expensive step is
    #    loading + matching the point clouds, not the aggregation) ───────
    pair_csv = output_dir / f'cs2_point_pairs_{label}.csv'
    pairs = None
    if use_cache and pair_csv.exists():
        cached = pd.read_csv(pair_csv)
        if {'x_mid', 'y_mid', 'rate', 'dt_years'}.issubset(cached.columns):
            # Re-apply the current pair thresholds, so a cache written with
            # older (looser) settings can't leak pairs into this run.
            n_cached = len(cached)
            keep = (np.isfinite(cached['dt_years'])
                    & (cached['dt_years'] >= cfg.DHDT_MIN_PAIR_DT_YEARS)
                    & np.isfinite(cached['rate'])
                    & (cached['rate'].abs() <= cfg.DHDT_MAX_ABS_RATE_M_PER_YR))
            if 'pair_dist_m' in cached.columns:
                keep &= cached['pair_dist_m'] <= cfg.DHDT_MAX_PAIR_DISTANCE_M
            pairs = cached[keep].copy()
            print(f"  Using cached CS2 point pairs (skipping point loading/"
                  f"matching): {pair_csv}  [{n_cached:,} -> {len(pairs):,} pairs "
                  f"after current thresholds]")
        else:
            print("  NOTE: cached pair table lacks required columns; recomputing.")

    if pairs is None:
        early_pts = prepare_dhdt_timestamp_points(early_label, cfg.DHDT_TIMESTAMP_YEARS.get(early_label, epoch_years(early_label)),
                                                   crs, geoid_grid)
        late_pts = prepare_dhdt_timestamp_points(late_label, cfg.DHDT_TIMESTAMP_YEARS.get(late_label, epoch_years(late_label)),
                                                  crs, geoid_grid)
        if early_pts.empty or late_pts.empty:
            print("  WARNING: insufficient CS2 points for one or both timestamps, skipping pair.")
            return None

        print(f"  Matching nearest CS2 points ({early_label} -> {late_label}, "
              f"max {cfg.DHDT_MAX_PAIR_DISTANCE_M} m separation)...")
        pairs = match_nearest_points(early_pts, late_pts, cfg.DHDT_MAX_PAIR_DISTANCE_M)
        if pairs.empty:
            print("  WARNING: no point pairs within max distance, skipping pair.")
            return None
        print(f"    {len(pairs):,} candidate pairs within {cfg.DHDT_MAX_PAIR_DISTANCE_M} m")

        pairs = filter_dhdt_pairs(pairs)
        if pairs.empty:
            print("  WARNING: no pairs survived dt/rate filtering, skipping pair.")
            return None

    # ── Co-located TanDEM-X samples at every pair midpoint ──────────────
    pairs_gdf = gpd.GeoDataFrame(
        pairs.reset_index(drop=True),
        geometry=gpd.points_from_xy(pairs['x_mid'], pairs['y_mid']), crs=crs)
    print(f"  Sampling TanDEM-X dhdt at {len(pairs_gdf):,} pair midpoints...")
    pairs_gdf['tdx_rate'] = sample_raster_values(pairs_gdf, tdx_path)

    ref_dem = Path(cfg.TDX_DIR) / f'dem_{late_label}.tif'
    if ref_dem.exists():
        pairs_gdf['tdx_elev'] = sample_raster_values(pairs_gdf, ref_dem,
                                                     valid_min=-9999, valid_max=10000)
        slope_pts = sample_slope_from_own_dem(pairs_gdf, ref_dem)
        pairs_gdf['tdx_slope'] = slope_pts if slope_pts is not None else np.nan
    else:
        print(f"    NOTE: reference DEM not found ({ref_dem}); no elevation/"
              f"slope for the hypsometric and slope panels.")
        pairs_gdf['tdx_elev'] = np.nan
        pairs_gdf['tdx_slope'] = np.nan

    # Point-pair table incl. the co-located TanDEM-X value (drop geometry;
    # x_mid/y_mid already carry the location).
    pairs_gdf.drop(columns='geometry').to_csv(pair_csv, index=False, float_format='%.4f')
    print(f"    Point-pair table -> {pair_csv}")

    # ── Co-located aggregation onto the shared grid ─────────────────────
    cells = colocated_aggregate(
        pairs_gdf, sensor_col='rate', raster_col='tdx_rate', label='CS2 dhdt',
        extra_cols=['tdx_elev', 'tdx_slope', 'dt_years', 'pair_dist_m'],
        diff_abs_max=None, min_points=cfg.DHDT_MIN_PAIRS_PER_CELL,
        grid_size=cfg.DHDT_GRID_SIZE_M, origin=cfg.AGGREGATION_ORIGIN_M)
    if cells.empty:
        print(f"  WARNING: no cells with >= {cfg.DHDT_MIN_PAIRS_PER_CELL} co-located "
              f"pairs, skipping pair.")
        return None

    cells = cells.rename(columns={'rate': 'cs2_dhdt', 'rate_std': 'cs2_dhdt_std',
                                  'tdx_rate': 'tdx_dhdt', 'tdx_rate_std': 'tdx_dhdt_std',
                                  'n_points': 'n_pairs'})
    cells['diff'] = cells['tdx_dhdt'] - cells['cs2_dhdt']

    # Per-cell outputs: GeoPackage (all columns) + rasters on the grid.
    cells.to_file(output_dir / f'dhdt_cells_{label}.gpkg', driver='GPKG')
    print(f"    Cell table -> {output_dir / f'dhdt_cells_{label}.gpkg'}")
    for col, fname in (('cs2_dhdt', f'cs2_dhdt_{label}.tif'),
                       ('tdx_dhdt', f'tdx_dhdt_colocated_{label}.tif'),
                       ('diff',     f'diff_tdx_minus_cs2_dhdt_{label}.tif')):
        write_cells_geotiff(cells, col, output_dir / fname,
                            grid_size=cfg.DHDT_GRID_SIZE_M, origin=cfg.AGGREGATION_ORIGIN_M)

    tdx_v  = cells['tdx_dhdt'].values.astype(float)
    cs2_v  = cells['cs2_dhdt'].values.astype(float)
    diff_v = cells['diff'].values.astype(float)
    n_v    = cells['n_pairs'].values
    stats = compute_dhdt_diff_stats(tdx_v, cs2_v, label, n_pairs=n_v)

    # Cell-centre lon/lat for the map panels; elevation / slope are the
    # co-located means at the pair midpoints of each cell.
    to_ll = Transformer.from_crs(crs, "EPSG:4326", always_xy=True)
    lons_v, lats_v = to_ll.transform(cells['x'].values, cells['y'].values)
    elev_v  = cells['tdx_elev'].values.astype(float)
    slope_v = cells['tdx_slope'].values.astype(float)


    if cfg.ENABLE_HYPSOMETRIC_DHDT:
        hyps = compute_dhdt_hypsometry(elev_v, diff_v)
        if not hyps.empty:
            hyps.to_csv(output_dir / f'dhdt_hypsometric_{label}.csv',
                        index=False, float_format='%.3f')

    return {'stats': stats, 'tdx': tdx_v, 'cs2': cs2_v, 'diff': diff_v,
            'lons': np.asarray(lons_v), 'lats': np.asarray(lats_v),
            'elev': elev_v, 'slope': slope_v, 'n': n_v,
            'x_proj': cells['x'].values, 'y_proj': cells['y'].values, 'crs': crs}


def dhdt_pairs():
    """(later, earlier) epoch pairs: cfg.DHDT_PAIRS or all matching rasters in cfg.TDX_DIR."""
    if cfg.DHDT_PAIRS:
        return [tuple(p[:2]) for p in cfg.DHDT_PAIRS]
    pattern = re.escape(cfg.DHDT_FILE_TEMPLATE).replace(r"\{late\}", "(?P<late>[0-9-]+)") \
                                              .replace(r"\{early\}", "(?P<early>[0-9-]+)")
    pairs = []
    for f in sorted(Path(cfg.TDX_DIR).glob("*.tif")):
        m = re.fullmatch(pattern, f.name)
        if m:
            pairs.append((m.group("late"), m.group("early")))
    return sorted(pairs, key=lambda p: (epoch_years(p[0]), epoch_years(p[1])))


def run_cs2_dhdt_pipeline(output_dir: str = cfg.CS2_DHDT_OUTPUT_DIR, use_cache: bool = cfg.USE_DHDT_CACHE):
    """Run the dh/dt comparison for every epoch pair and write a summary CSV."""
    dhdt_results = {}
    for late_label, early_label in dhdt_pairs():
        tdx_file = cfg.DHDT_FILE_TEMPLATE.format(late=late_label, early=early_label)
        result = run_dhdt_comparison(early_label, late_label, tdx_file,
                                      output_dir=output_dir, use_cache=use_cache)
        if result:
            dhdt_results[f'{late_label}_minus_{early_label}'] = result

    if dhdt_results:
        df = pd.DataFrame([r['stats'] for r in dhdt_results.values()])
        out = Path(output_dir) / 'cs2_dhdt_validation_summary.csv'
        df.to_csv(out, index=False, float_format='%.4f')
        print(f"\n{'=' * 60}\nSummary -> {out}\n{'=' * 60}")
    else:
        print("\nNo dhdt comparisons produced any output -- check paths/config above.")
    return dhdt_results
