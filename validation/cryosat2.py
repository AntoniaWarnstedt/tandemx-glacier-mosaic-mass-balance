"""CryoSat-2 (CryoTEMPO-EOLIS) vs. TanDEM-X surface elevation per DEM epoch.
"""

from glob import glob
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from pyproj import Transformer

from . import config as cfg
from .common import (
    pooled_epochs,
    colocated_aggregate,
    compute_hypsometry,
    convert_cs2_to_geoid,
    decimal_year_to_timestamp,
    filter_by_tdx_window,
    get_tdx_dates,
    print_validity_breakdown,
    sample_raster_values,
    sample_slope_from_own_dem,
    write_cells_geotiff,
)


class CryoSatTanDEMXComparison:
    """Compare pre-processed CryoSat-2 point shapefiles with TanDEM-X DEMs."""

    def __init__(self,
                 cs2_shp_dir: str,
                 tdx_dir: str,
                 output_dir: str,
                 geoid_grid: str,
                 max_temporal_offset: int = 30,
                 hyps_bin_width: float = cfg.HYPS_BIN_WIDTH_M,
                 hyps_min_samples: int = cfg.HYPS_MIN_SAMPLES):
        """
        Parameters ---------- cs2_shp_dir : directory of
        cs2_points_YYYY.shp/.gpkg files. tdx_dir : directory of TanDEM-X DEM
        GeoTIFFs (dem_YYYY.tif). output_dir : all outputs written here.
        geoid_grid : path to the target geoid undulation grid (EGM2008 -- see
        GEOID_GRID_EGM2008). max_temporal_offset : ±days around each TDX
        acquisition (default 30). hyps_bin_width : elevation band width for
        hypsometric plot (m). hyps_min_samples : minimum points per elevation
        band.
        """
        self.cs2_shp_dir        = Path(cs2_shp_dir)
        self.tdx_dir            = Path(tdx_dir)
        self.output_dir         = Path(output_dir)
        self.geoid_grid         = geoid_grid
        self.max_temporal_offset = max_temporal_offset
        self.hyps_bin_width     = hyps_bin_width
        self.hyps_min_samples   = hyps_min_samples
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.comparisons        = {}   # year_str → {gdf, stats, aggregated}
        self.hyps_all           = {}   # year_str → hypsometric DataFrame (set by run_point_comparison)

    # -----------------------------------------------------------------------
    # Caption helpers
    # -----------------------------------------------------------------------


    # -----------------------------------------------------------------------
    # Shapefile loading
    # -----------------------------------------------------------------------

    def load_cs2_shapefiles(self, sub_years: list) -> gpd.GeoDataFrame:
        """Load and concatenate cs2_points_YYYY.gpkg/.shp for each year.

        Parses the 'datetime' string column to pd.Timestamp and normalises
        the uncertainty column name.
        """
        gdfs = []
        for y in sub_years:
            gpkg = self.cs2_shp_dir / f'cs2_points_{y}.gpkg'
            shp  = self.cs2_shp_dir / f'cs2_points_{y}.shp'
            path = gpkg if gpkg.exists() else shp if shp.exists() else None

            if path is None:
                print(f"  WARNING: No CS2 file for year {y} "
                      f"(tried {gpkg.name} and {shp.name})")
                continue

            print(f"  Loading {path.name}...")
            gdf = gpd.read_file(path)

            # Parse datetime string → pd.Timestamp
            if 'datetime' in gdf.columns:
                gdf['datetime'] = pd.to_datetime(gdf['datetime'], errors='coerce')
            elif 'dec_year' in gdf.columns:
                gdf['datetime'] = decimal_year_to_timestamp(gdf['dec_year'].values).values
                print("    NOTE: 'datetime' absent; reconstructed from dec_year")

            # Normalise uncertainty column name
            if 'elev_unc' not in gdf.columns and 'uncertainty' in gdf.columns:
                gdf = gdf.rename(columns={'uncertainty': 'elev_unc'})

            print(f"    {len(gdf):,} points, CRS: {gdf.crs}")
            gdfs.append(gdf)

        if not gdfs:
            return gpd.GeoDataFrame()

        combined = pd.concat(gdfs, ignore_index=True) if len(gdfs) > 1 else gdfs[0]
        combined = combined[combined['datetime'].notna()].copy()
        print(f"  Combined: {len(combined):,} points with valid datetime")
        return combined

    # -----------------------------------------------------------------------
    # File-pair discovery
    # -----------------------------------------------------------------------

    def find_file_pairs(self) -> list:
        """Match TanDEM-X DEM GeoTIFFs to year labels and acquisition dates."""
        tdx_files = sorted(glob(str(self.tdx_dir / "dem_*.tif")))
        print(f"Found {len(tdx_files)} TanDEM-X DEM file(s)")

        pairs = []
        for tdx_file in tdx_files:
            year_str  = Path(tdx_file).stem.replace('dem_', '')
            tdx_dates = get_tdx_dates(year_str)
            if tdx_dates:
                pairs.append({'year':      year_str,
                              'tdx_file':  str(tdx_file),
                              'tdx_dates': tdx_dates})
                print(f"  {year_str}: {len(tdx_dates)} acquisition date(s)")
            else:
                print(f"  WARNING: No TDX dates for '{year_str}' — skipping.")
        return pairs

    # -----------------------------------------------------------------------
    # Grid aggregation and raster sampling
    # -----------------------------------------------------------------------


    # -----------------------------------------------------------------------
    # Per-year comparison
    # -----------------------------------------------------------------------

    def compare_year(self, year_str: str, tdx_file: str, tdx_dates: list,
                     use_aggregation: bool = True) -> gpd.GeoDataFrame:
        """Full comparison pipeline for one TanDEM-X mosaic year."""
        print(f"\n{'='*60}")
        print(f"Processing year: {year_str}")
        print(f"{'='*60}")

        sub_years = [int(y) for y in year_str.split('-') if y.isdigit()]
        cs2_gdf   = self.load_cs2_shapefiles(sub_years)

        if cs2_gdf.empty:
            print(f"  No CS2 data for year(s) {sub_years}.")
            return gpd.GeoDataFrame()

        # ── ±30-day temporal filter ──────────────────────────────────────────
        cs2_gdf = filter_by_tdx_window(cs2_gdf, tdx_dates,
                                        window_days=self.max_temporal_offset)
        if cs2_gdf.empty:
            print(f"  No points within ±{self.max_temporal_offset} days.")
            return gpd.GeoDataFrame()

        # ── Geoid conversion ─────────────────────────────────────────────────
        if self.geoid_grid and Path(self.geoid_grid).exists():
            print("  Converting ellipsoidal → EGM2008 geoid heights...")
            cs2_gdf = convert_cs2_to_geoid(cs2_gdf, self.geoid_grid)
            cs2_gdf["elevation"] = cs2_gdf["elevation_geoid"]
        else:
            print("  WARNING: Geoid grid not found — using ellipsoidal heights.")

        # ── Reproject to TDX DEM CRS (BEFORE aggregation -- see docstring
        #    on aggregate_points_to_grid() for why the order matters) ───────
        with rasterio.open(tdx_file) as src:
            tdx_crs = src.crs
        cs2_gdf = cs2_gdf.to_crs(tdx_crs) if cs2_gdf.crs != tdx_crs else cs2_gdf

        # ── Sample TDX DEM (+ slope of THIS epoch's DEM) at every raw CS2
        #    point (points are already in tdx_crs) ───────────────────────
        print(f"  Sampling TDX DEM at {len(cs2_gdf):,} CS2 points: {Path(tdx_file).name}...")
        cs2_gdf = cs2_gdf.copy()
        cs2_gdf['tdx_elev'] = sample_raster_values(cs2_gdf, tdx_file,
                                                   valid_min=-9999, valid_max=10000)
        slope_pts = sample_slope_from_own_dem(cs2_gdf, tdx_file)
        if slope_pts is not None:
            cs2_gdf['tdx_slope'] = slope_pts
        print_validity_breakdown('CS2 (points)', cs2_gdf['tdx_elev'].values,
                                 (cs2_gdf['tdx_elev'] - cs2_gdf['elevation']).values,
                                 diff_abs_max=cfg.ABS_DIFF_MAX_M)

        # ── Co-located grid aggregation ──────────────────────────────────
        if use_aggregation:
            print(f"  Co-located aggregation to a {cfg.AGGREGATION_GRID_SIZE_M:.0f} m grid...")
            cs2_gdf = colocated_aggregate(
                cs2_gdf, sensor_col='elevation', raster_col='tdx_elev', label='CS2',
                extra_cols=['temp_off', 'elev_unc', 'tdx_slope'],
                diff_abs_max=cfg.ABS_DIFF_MAX_M,
                min_points=cfg.AGGREGATION_MIN_POINTS_PER_CELL,
                grid_size=cfg.AGGREGATION_GRID_SIZE_M, origin=cfg.AGGREGATION_ORIGIN_M)
            if cs2_gdf.empty:
                return gpd.GeoDataFrame()
            cs2_gdf = cs2_gdf.rename(columns={'elevation': 'elevation_mean',
                                               'temp_off': 'temp_off_mean',
                                               'elev_unc': 'mean_unc'})
            elevation_column = 'elevation_mean'
        else:
            elevation_column = 'elevation'

        cs2_gdf = cs2_gdf.copy()
        cs2_gdf['diff'] = cs2_gdf['tdx_elev'] - cs2_gdf[elevation_column]
        cs2_gdf['tdx_file'] = tdx_file

        # ── Validity filter (cell level; the point-level filter above
        #    already removed nodata and |diff| >= ABS_DIFF_MAX_M, so in
        #    aggregated mode this is a safety net) ─────────────────────────
        valid = (np.isfinite(cs2_gdf['tdx_elev']) &
                 np.isfinite(cs2_gdf['diff'])      &
                 (cs2_gdf['tdx_elev'] > -9999)     &
                 (cs2_gdf['tdx_elev'] < 10000)     &
                 (cs2_gdf['diff'].abs() < cfg.ABS_DIFF_MAX_M))
        print_validity_breakdown('CS2', cs2_gdf['tdx_elev'].values, cs2_gdf['diff'].values,
                                 diff_abs_max=cfg.ABS_DIFF_MAX_M)
        cs2_valid = cs2_gdf[valid].copy()
        print(f"  Valid comparisons: {len(cs2_valid):,} "
              f"({100*len(cs2_valid)/max(len(cs2_gdf), 1):.1f}%)")

        if cs2_valid.empty:
            return gpd.GeoDataFrame()

        stats = self.calculate_statistics(cs2_valid, year_str, elevation_column)
        self.comparisons[year_str] = {
            'gdf': cs2_valid, 'stats': stats, 'aggregated': use_aggregation,
            'tdx_file': tdx_file}
        return cs2_valid

    # -----------------------------------------------------------------------
    # Statistics
    # -----------------------------------------------------------------------

    def calculate_statistics(self, gdf: gpd.GeoDataFrame,
                               year_str: str,
                               elevation_col: str = 'elevation') -> dict:
        diff = gdf['diff'].values
        nmad = 1.4826 * np.median(np.abs(diff - np.median(diff)))
        sd = {
            'year':        year_str,
            'n_points':    len(gdf),
            'mean_diff':   np.mean(diff),
            'median_diff': np.median(diff),
            'std_diff':    np.std(diff),
            'nmad_diff':   nmad,
            'rmse':        np.sqrt(np.mean(diff**2)),
            'mae':         np.mean(np.abs(diff)),
            'min_diff':    np.min(diff),
            'max_diff':    np.max(diff),
            'pct_5':       np.percentile(diff, 5),
            'pct_95':      np.percentile(diff, 95),
            'mean_cs2':    gdf[elevation_col].mean(),
            'mean_tdx':    gdf['tdx_elev'].mean(),
        }
        print(f"\n  Stats [{year_str}]")
        print(f"    Mean diff (TDX−CS2) : {sd['mean_diff']:+.2f} m")
        print(f"    Median diff         : {sd['median_diff']:+.2f} m")
        print(f"    Std                 : {sd['std_diff']:.2f} m")
        print(f"    NMAD                : {sd['nmad_diff']:.2f} m")
        print(f"    RMSE                : {sd['rmse']:.2f} m")
        print(f"    MAE                 : {sd['mae']:.2f} m")
        print(f"    5th / 95th pct      : {sd['pct_5']:.2f} / {sd['pct_95']:.2f} m")
        return sd

    def save_statistics_summary(self):
        """Write all per-year overall statistics to a single CSV."""
        if not self.comparisons:
            return
        rows = [dict(d['stats'], aggregated=d['aggregated'])
                for d in self.comparisons.values()]
        df  = pd.DataFrame(rows)
        out = self.output_dir / 'comparison_statistics_summary.csv'
        df.to_csv(out, index=False, float_format='%.3f')
        print(f"\nSummary statistics → {out}")
        return df

    # -----------------------------------------------------------------------
    # Output: difference raster
    # -----------------------------------------------------------------------

    def create_difference_raster(self, gdf: gpd.GeoDataFrame,
                                   year_str: str,
                                   resolution: float = None):
        """
        Rasterise the TDX−CS2 difference field onto the SHARED aggregation grid
        (AGGREGATION_GRID_SIZE_M, AGGREGATION_ORIGIN_M), so each output pixel
        is exactly one comparison cell. Rows falling in the same cell (raw-
        point mode, or the All-Years concat) are averaged.
        """
        resolution = cfg.AGGREGATION_GRID_SIZE_M if resolution is None else resolution
        write_cells_geotiff(gdf, 'diff',
                            self.output_dir / f'diff_cs2_tdx_{year_str}.tif',
                            grid_size=resolution, origin=cfg.AGGREGATION_ORIGIN_M)

    # -----------------------------------------------------------------------
    # Output: plots
    # -----------------------------------------------------------------------


    # -----------------------------------------------------------------------
    # Extra diagnostic plots
    # -----------------------------------------------------------------------


    # -----------------------------------------------------------------------
    # Combined per-year grids (one figure per plot type, all years + All_Years)
    # -----------------------------------------------------------------------


    # -----------------------------------------------------------------------
    # Top-level runner
    # -----------------------------------------------------------------------

    def run_point_comparison(self, use_aggregation: bool = True):
        """Run the full TDX vs CryoSat-2 comparison for all matched DEM files."""
        pairs = self.find_file_pairs()

        for pair in pairs:
            yr  = pair['year']
            gdf = self.compare_year(yr, pair['tdx_file'], pair['tdx_dates'],
                                     use_aggregation=use_aggregation)
            if gdf.empty:
                continue

            self.create_difference_raster(gdf, yr)

            # Shapefile (stringify datetime columns)
            out_shp = self.output_dir / f"diff_{yr}.shp"
            gdf_out = gdf.copy()
            for col in gdf_out.columns:
                if pd.api.types.is_datetime64_any_dtype(gdf_out[col]):
                    gdf_out[col] = gdf_out[col].dt.strftime('%Y-%m-%dT%H:%M:%S')
            gdf_out.to_file(out_shp)
            print(f"  Difference shapefile → {out_shp}")

        # ── Combined all-years dataset ────────────────────────────────────────
        if self.comparisons:
            print("\n" + "="*60)
            print("Combined multi-year dataset (All Years)")
            print("="*60)
            # "All Years" = union of the pooled epochs (see pooled_epochs)
            pooled = pooled_epochs(self.comparisons.keys())
            present_periods = [yr for yr in pooled if yr in self.comparisons]
            missing_periods = [yr for yr in pooled if yr not in self.comparisons]
            other_years     = [yr for yr in self.comparisons if yr not in pooled]
            if missing_periods:
                print(f"  WARNING: expected period(s) missing from comparisons "
                      f"(no matching DEM file processed?) -- excluded from "
                      f"All-Years: {missing_periods}")
            if other_years:
                print(f"  NOTE: other year_str entries present but NOT part of "
                      f"ALL_YEARS_PERIOD_KEYS, so excluded from All-Years by "
                      f"design: {other_years}")
            if not present_periods:
                print("  ERROR: none of ALL_YEARS_PERIOD_KEYS were found in "
                      "comparisons -- skipping All-Years entirely.")
            else:
                all_gdf = pd.concat([self.comparisons[yr]['gdf'] for yr in present_periods],
                                     ignore_index=True)
                n_check = sum(len(self.comparisons[yr]['gdf']) for yr in present_periods)
                assert len(all_gdf) == n_check, (
                    "All-Years point count doesn't match the sum of its "
                    "constituent periods -- this should be impossible with a "
                    "plain concat and points to a bug.")
                print(f"  All-Years N = {len(all_gdf):,} "
                      f"(= sum of {present_periods}: "
                      f"{' + '.join(str(len(self.comparisons[yr]['gdf'])) for yr in present_periods)})")
                ec      = 'elevation_mean' if use_aggregation else 'elevation'
                s_all   = self.calculate_statistics(all_gdf, 'All_Years', ec)
                self.comparisons['All_Years'] = {
                    'gdf': all_gdf, 'stats': s_all, 'aggregated': use_aggregation}

                self.create_difference_raster(all_gdf, 'All_Years')

                out_all = self.output_dir / "diff_All_Years.shp"
                gdf_out = all_gdf.copy()
                for col in gdf_out.columns:
                    if pd.api.types.is_datetime64_any_dtype(gdf_out[col]):
                        gdf_out[col] = gdf_out[col].dt.strftime('%Y-%m-%dT%H:%M:%S')
                gdf_out.to_file(out_all)

                self.save_statistics_summary()

        # ── Pass 2: hypsometric plots with globally shared axis limits ────────
        self.hyps_all = {}
        if self.comparisons:
            hyps_all = {}
            for yr, comp in self.comparisons.items():
                h = compute_hypsometry(
                    comp['gdf'],
                    elev_col='tdx_elev',
                    diff_col='diff',
                    bin_width=self.hyps_bin_width,
                    min_samples=self.hyps_min_samples,
                )
                if not h.empty:
                    hyps_all[yr] = h
                    csv_path = self.output_dir / f'hypsometric_summary_{yr}.csv'
                    h.to_csv(csv_path, index=False, float_format='%.3f')

            self.hyps_all = hyps_all

        # ── Vertical-datum sanity check (console diagnostic, not a plot) ────
        self.diagnose_vertical_datum()


    # -----------------------------------------------------------------------
    # Vertical-datum diagnostic
    # -----------------------------------------------------------------------

    def diagnose_vertical_datum(self):
        """
        Check for a residual ellipsoidal-vs-geoid datum mismatch between TDX
        and CS2.
        """
        print("\n" + "="*60)
        print("Vertical-datum diagnostic")
        print("="*60)

        if not (self.geoid_grid and Path(self.geoid_grid).exists()):
            print("  Geoid grid not found at:", self.geoid_grid)
            print("  -> CS2 elevations are NOT being geoid-converted either "
                  "(compare_year() falls back to ellipsoidal heights with "
                  "only a printed WARNING). Check your run log for the "
                  "line 'WARNING: Geoid grid not found' to confirm which "
                  "branch actually executed for the results you're "
                  "reviewing — both sides being ellipsoidal is internally "
                  "consistent, just not geoid-corrected as the docstring "
                  "describes.")
            return None

        # Domain centre: prefer an already-loaded comparison, else skip.
        ref_gdf = None
        for comp in self.comparisons.values():
            if not comp['gdf'].empty:
                ref_gdf = comp['gdf']
                break
        if ref_gdf is None:
            print("  No loaded comparisons available to locate the domain "
                  "centre — run this after run_point_comparison().")
            return None

        gdf_ll = (ref_gdf.to_crs(epsg=4326) if ref_gdf.crs.is_projected
                  else ref_gdf)
        lon0 = float(gdf_ll.geometry.x.mean())
        lat0 = float(gdf_ll.geometry.y.mean())

        proj_str = (f"+proj=longlat +datum=WGS84 "
                    f"+geoidgrids={self.geoid_grid} +type=crs")
        transformer = Transformer.from_crs("EPSG:4979", proj_str,
                                            always_xy=True)
        # Feed ellipsoidal height = 0 through the same conversion used for
        # CS2; the result is -N (since H = h - N, with h = 0 => H = -N).
        _, _, h0 = transformer.transform([lon0], [lat0], [0.0])
        N = -h0[0]

        overall_bias = self.comparisons.get('All_Years', {}).get(
            'stats', {}).get('mean_diff')

        print(f"  Domain centre: lon={lon0:.3f}, lat={lat0:.3f}")
        print(f"  Local EGM2008 geoid undulation N ≈ {N:+.2f} m")
        if overall_bias is not None:
            print(f"  Observed overall mean TDX−CS2 bias: {overall_bias:+.2f} m")
            if abs(overall_bias) > 3 and abs(overall_bias - N) < abs(overall_bias) / 2:
                print("  -> The observed bias is a large fraction of the "
                      "local geoid undulation N. Since both sides are "
                      "supposed to be EGM2008, this looks like a residual "
                      "datum mismatch (e.g. the TDX mosaic isn't actually "
                      "EGM2008 as assumed, or the geoid conversion isn't "
                      "engaging at runtime -- check the run log for the "
                      "'WARNING: Geoid grid not found' line), not just "
                      "terrain/processing noise. Worth confirming rather "
                      "than assuming.")
            else:
                print("  -> Bias is not dominated by N; no obvious sign of "
                      "a residual whole-datum offset (small residual bias "
                      "is more likely terrain/processing noise than a "
                      "datum mismatch).")
        return N


def run_cs2_pipeline(cs2_shp_dir=cfg.CS2_SHP_DIR,
                      tdx_dir=cfg.TDX_DIR,
                      output_dir=cfg.CS2_OUTPUT_DIR,
                      geoid_grid=cfg.GEOID_GRID_EGM2008,
                      max_temporal_offset=cfg.CS2_MAX_TEMPORAL_OFFSET):
    """Run the full CryoSat-2 vs TanDEM-X comparison. Returns the
    CryoSatTanDEMXComparison instance so its results (self.comparisons,
    self.hyps_all, ...) can be used by the combined CS2+IS2 plots."""
    comparator = CryoSatTanDEMXComparison(
        cs2_shp_dir=cs2_shp_dir,
        tdx_dir=tdx_dir,
        output_dir=output_dir,
        geoid_grid=geoid_grid,
        max_temporal_offset=max_temporal_offset,
        hyps_bin_width=cfg.HYPS_BIN_WIDTH_M,
        hyps_min_samples=cfg.HYPS_MIN_SAMPLES,
    )
    comparator.run_point_comparison(use_aggregation=True)
    return comparator
