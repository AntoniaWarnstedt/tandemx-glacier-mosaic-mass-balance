"""Per-glacier centerline figures (dh, dh/dt, surface and bed profiles) and the
centerline data tables.
"""

import csv
import glob
import os
import re

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import rasterio

from .. import config as cfg
from .centerline_geometry import (
    _ice_fill_arrays_split_at_grounding_line,
    find_preferred_change,
    get_centerline_plot_segment,
    get_centerline_segment_by_year,
    get_grounding_line_distance_on_centerline,
    get_segment_distance_range_on_centerline,
    sample_raster_along_line_with_buffer,
    sample_raster_on_centerline_segment,
)
from .centerline_panels import (
    create_combined_bedrock_plot,
    create_combined_dem_and_dhdt_9panel,
    create_combined_dem_only_plot,
)
from .common import build_years_colormap, format_coord_labels, terminus_color, with_bedrock_source
from ..utils import (
    mask_year_for_label,
    change_sort_key,
    format_period_label,
    get_display_year,
    get_longest_linestring,
    should_skip_individual_year,
)


def plot_centerline_elevation_changes(all_changes, year_results, grid_info, centerline_dir, preloaded_ref_dem,
                                      mb_results=None,
                                      preferred_periods=cfg.CENTERLINE_PREFERRED_PERIODS):
    """
    Create centerline-based elevation change plots using one centerline folder
    and year-mask clipping.
    """
    print("\n" + "=" * 50)
    print("▶ Creating centerline elevation change plots...")

    if not os.path.exists(centerline_dir):
        print("   Warning: Centerline directory not found")
        return {}

    centerline_files = glob.glob(os.path.join(centerline_dir, "*.shp"))
    if not centerline_files:
        print("   Warning: No centerline shapefiles found")
        return {}

    print(f"   Found {len(centerline_files)} centerline files")

    # Load bedrock raster
    bedrock_raster_data = None
    bedrock_transform = None

    # Load bedrock
    if os.path.exists(cfg.BEDROCK_PATH):
        print(f"   Loading bedrock: {os.path.basename(cfg.BEDROCK_PATH)}")
        try:
            with rasterio.open(cfg.BEDROCK_PATH) as src:
                bedrock_raster_data = src.read(1).astype(np.float32)
                bedrock_transform = src.transform
                bedrock_nodata = src.nodata
                if bedrock_nodata is not None:
                    if np.isnan(bedrock_nodata):
                        bedrock_raster_data[~np.isfinite(bedrock_raster_data)] = np.nan
                    else:
                        bedrock_raster_data[bedrock_raster_data == bedrock_nodata] = np.nan
                bedrock_raster_data[
                    (bedrock_raster_data < -5000) | (bedrock_raster_data > 5000)
                ] = np.nan
                n_valid = np.sum(np.isfinite(bedrock_raster_data))
                print(f"   ✓ Loaded bedrock ({n_valid:,} valid pixels)")
        except Exception as e:
            print(f"   ⚠️  Could not load bedrock: {e}")
            bedrock_raster_data = None
    else:
        print(f"   ⚠️  bedrock file not found: {cfg.BEDROCK_PATH}")

    # Load the InSAR grounding line (loaded once here and reused for every
    # glacier below, same as the bedrock raster) -- reprojected once to
    # grid_info['crs'] so per-glacier intersection tests don't each pay for
    # their own reprojection.
    grounding_line_geom = None
    if os.path.exists(cfg.GROUNDING_LINE_PATH):
        print(f"   Loading grounding line: {os.path.basename(cfg.GROUNDING_LINE_PATH)}")
        try:
            gl_gdf = gpd.read_file(cfg.GROUNDING_LINE_PATH)
            gl_gdf = gl_gdf.to_crs(grid_info['crs'])
            grounding_line_geom = gl_gdf.unary_union
            print(f"   ✓ Loaded grounding line ({len(gl_gdf)} features)")
        except Exception as e:
            print(f"   ⚠️  Could not load grounding line: {e}")
            grounding_line_geom = None
    else:
        print(f"   ⚠️  Grounding line file not found: {cfg.GROUNDING_LINE_PATH}")

    all_changes_sorted = sorted(all_changes, key=change_sort_key)
    years_list = [f"{c['yA']}-{c['yB']}" for c in all_changes_sorted]
    years_list_display = [format_period_label(c['yA'], c['yB']) for c in all_changes_sorted]

    # Epochs present in the dh/dt periods and their outline years (same for all glaciers)
    epoch_labels = {y for c in all_changes_sorted for y in (c['yA'], c['yB'])}
    _global_surface_years_to_plot = {
        y for y in epoch_labels if not should_skip_individual_year(y, epoch_labels)
    }
    GLOBAL_RELEVANT_MASK_YEARS = {mask_year_for_label(y) for y in _global_surface_years_to_plot}

    mb_dict = {f"{r['yA']}-{r['yB']}": r for r in mb_results} if mb_results else {}

    def _get_dhdt_for_change(c):
        pk = f"{c['yA']}-{c['yB']}"
        mb_res = mb_dict.get(pk)
        if mb_res is not None and mb_res.get('dhdt_bias_corrected_raster') is not None:
            return mb_res['dhdt_bias_corrected_raster']
        print(f"      ⚠️ No bias-corrected dhdt available for {pk}; "
              f"falling back to filtered/uncorrected dhdt.")
        return c['dhdt']

    def _get_obs_union_mask_for_change(c):
        """
        Returns None (no masking) if unavailable, with a warning, so plotting
        still degrades gracefully rather than failing.
        """
        pk = f"{c['yA']}-{c['yB']}"
        mb_res = mb_dict.get(pk)
        if mb_res is not None and mb_res.get('obs_union_mask') is not None:
            mask_arr = mb_res['obs_union_mask'].astype(bool)
            print(f"      obs_union_mask[{pk}]: {int(mask_arr.sum()):,} px "
                  f"(area_mask_option={mb_res.get('area_mask_option', '?')}, "
                  f"support={mb_res.get('area_mask_support_label', '?')})")
            return mask_arr
        print(f"      ⚠️⚠️⚠️ UNMASKED FALLBACK for {pk}: no usable "
              f"obs_union_mask found (mb_dict has keys: {sorted(mb_dict.keys())}) -- "
              f"centerline dh/dhdt for this period will be sampled over the FULL "
              f"raster extent, ignoring AREA_MASK_OPTION entirely. If you expected "
              f"Option B/C clipping in this plot and see this line, that's why.")
        return None

    num_years = len(years_list)
    year_colors = build_years_colormap(num_years)

    map_change = find_preferred_change(all_changes_sorted, preferred_periods)
    if map_change is None:
        print("   Warning: No changes available for map background!")
        return {}

    map_label = format_period_label(map_change['yA'], map_change['yB'])
    print(f"   Using {map_label} for map backgrounds")

    all_glacier_data = {}
    all_glacier_bedrock_data = {}

    for glacier_idx, centerline_file in enumerate(centerline_files, start=1):
        centerline_name = os.path.splitext(os.path.basename(centerline_file))[0]
        glacier_name = centerline_name.replace('Centerline_', '')
        glacier_name = re.sub(r'_\d{2}$', '', glacier_name)
        glacier_name = glacier_name.replace('_', ' ')

        print(f"\n   Processing [{glacier_idx}]: {centerline_name}")

        try:
            centerline_gdf = gpd.read_file(centerline_file)
            if centerline_gdf.empty:
                continue

            centerline = get_longest_linestring(centerline_gdf.to_crs(grid_info['crs']).unary_union)
            if centerline is None:
                print("      ⚠️  No valid LineString geometry found")
                continue

            print(f"      Full centerline length: {centerline.length / 1000:.2f} km")

            year_segments = {}
            segment_ranges = {}
            terminus_positions_by_year = {}

            for mask_year in cfg.YEAR_MASKS.keys():
                seg = get_centerline_segment_by_year(centerline, mask_year, grid_info['crs'])
                year_segments[mask_year] = seg

                if seg is not None:
                    d0, d1 = get_segment_distance_range_on_centerline(centerline, seg)
                    segment_ranges[mask_year] = (d0, d1)
                    terminus_positions_by_year[mask_year] = d0
                    print(f"      ✓ {mask_year}: clipped segment {d0 / 1000:.2f} to {d1 / 1000:.2f} km")
                else:
                    print(f"      ⚠️  No valid centerline-mask intersection for {mask_year}")

            # only the terminus years selected in cfg.TERMINUS_LABEL_YEARS are marked
            if cfg.TERMINUS_LABEL_YEARS is not None:
                terminus_positions_by_year = {
                    yr: pos for yr, pos in terminus_positions_by_year.items()
                    if yr in cfg.TERMINUS_LABEL_YEARS
                }

            if not segment_ranges:
                print("      ⚠️  No valid clipped segments for any year")
                continue

            # Re-anchor this glacier's own distance origin to its own most
            # seaward mapped front (smallest d0 among
            # GLOBAL_RELEVANT_MASK_YEARS -- i.e. exactly the years that end up
            # plotted for this glacier, see the matching relevant_mask_years
            # filter used for xlim below) instead of leaving x=0 at wherever
            # this glacier's particular centerline shapefile happens to start.
            _relevant_ranges_for_offset = [
                segment_ranges[yr] for yr in GLOBAL_RELEVANT_MASK_YEARS if yr in segment_ranges
            ]
            if _relevant_ranges_for_offset:
                offset_m = min(d0 for d0, d1 in _relevant_ranges_for_offset)
                _span_max_dist = max(d1 for d0, d1 in _relevant_ranges_for_offset)
            else:
                offset_m = min(d0 for d0, d1 in segment_ranges.values())
                _span_max_dist = max(d1 for d0, d1 in segment_ranges.values())
            xmax_km = min(cfg.CENTERLINE_MAX_DISTANCE / 1000.0, (_span_max_dist - offset_m) / 1000.0)
            xmin_km = 0.0

            terminus_positions_by_year = {
                yr: pos - offset_m for yr, pos in terminus_positions_by_year.items()
            }

            # Grounding line crossing distance, re-anchored by the same
            # offset_m as terminus_positions_by_year above so it lines up
            # with everything else on this glacier's own shifted x-axis.
            grounding_line_distance_m = get_grounding_line_distance_on_centerline(
                centerline, grounding_line_geom
            )
            if grounding_line_distance_m is not None:
                grounding_line_distance_m = grounding_line_distance_m - offset_m
                print(f"      ✓ Grounding line crosses centerline at {grounding_line_distance_m / 1000:.2f} km")
            else:
                print("      ⚠️  No valid centerline-grounding line intersection")

            fig = plt.figure(figsize=(28, 9), layout='constrained')
            gs = fig.add_gridspec(1, 3, width_ratios=[1, 1, 1])
            ax_map = fig.add_subplot(gs[0])
            ax_dh = fig.add_subplot(gs[1])
            ax_dhdt = fig.add_subplot(gs[2])

            dhdt_data = _get_dhdt_for_change(map_change)

            minx, miny, maxx, maxy = centerline.buffer(cfg.CENTERLINE_MAP_BUFFER).bounds
            col_min, row_max = ~grid_info['transform'] * (minx, miny)
            col_max, row_min = ~grid_info['transform'] * (maxx, maxy)

            row_min = max(0, int(np.floor(row_min)))
            row_max = min(grid_info['shape'][0], int(np.ceil(row_max)))
            col_min = max(0, int(np.floor(col_min)))
            col_max = min(grid_info['shape'][1], int(np.ceil(col_max)))

            clipped_dhdt = dhdt_data[row_min:row_max, col_min:col_max]

            extent = [
                grid_info['transform'].c + col_min * grid_info['transform'].a,
                grid_info['transform'].c + col_max * grid_info['transform'].a,
                grid_info['transform'].f + row_max * grid_info['transform'].e,
                grid_info['transform'].f + row_min * grid_info['transform'].e
            ]

            vmax = np.nanpercentile(np.abs(clipped_dhdt), 98)
            if vmax == 0 or not np.isfinite(vmax):
                vmax = 1.0

            im = ax_map.imshow(
                clipped_dhdt, cmap='RdBu', vmin=-vmax, vmax=vmax,
                extent=extent, origin='upper', aspect='equal'
            )

            gpd.GeoSeries([get_centerline_plot_segment(centerline)], crs=grid_info['crs']).plot(
                ax=ax_map, color='black', linewidth=2.5, zorder=10
            )

            # Plot clipped year segments on the map
            for yr, seg in year_segments.items():
                if seg is not None:
                    gpd.GeoSeries([seg], crs=grid_info['crs']).plot(
                        ax=ax_map, color=terminus_color(yr, default='yellow'),
                        linewidth=4, alpha=0.85, zorder=11
                    )

            cbar = fig.colorbar(im, ax=ax_map, orientation='horizontal', pad=0.08, shrink=0.8)
            cbar.set_label(f'dhdt (m a⁻¹) [{map_label}]', fontsize=12)

            ax_map.set_title(f"Elevation Change Rate\n{map_label}", fontsize=14, fontweight='bold')
            ax_map.set_xlabel("Easting (m)", fontsize=12)
            ax_map.set_ylabel("Northing (m)", fontsize=12)
            ax_map.ticklabel_format(style='plain', useOffset=False)
            ax_map.tick_params(axis='x', labelrotation=45, labelsize=10)
            ax_map.tick_params(axis='y', labelsize=10)
            format_coord_labels(ax_map)

            all_dh_profiles = []
            all_dhdt_profiles = []
            all_dem_profiles = {}
            has_data = False
            glacier_profile_data = {}

            bedrock_distances = None
            bedrock_values = None

            # Bedrock sampled along the full centerline. max_distance is
            # CENTERLINE_MAX_DISTANCE + offset_m (NOT just
            # CENTERLINE_MAX_DISTANCE) because this raw array gets shifted by
            # offset_m downstream (see the NOTE below) before it's
            # plotted/compared against the shared, already-shifted xmax_km --
            # sampling only up to the raw CENTERLINE_MAX_DISTANCE here would
            # make the shifted curve fall short of xmax_km by exactly offset_m
            # for any glacier whose centerline doesn't start right at its own
            # mapped front (offset_m > 0).
            _bedrock_raw_max_distance = cfg.CENTERLINE_MAX_DISTANCE + offset_m
            if bedrock_raster_data is not None and bedrock_transform is not None:
                bedrock_distances, bedrock_values = sample_raster_along_line_with_buffer(
                    bedrock_raster_data, bedrock_transform, centerline,
                    cfg.CENTERLINE_SAMPLING_DISTANCE, buffer_width=100,
                    max_distance=_bedrock_raw_max_distance, stat='median'
                )
            # bedrock_distances is deliberately left RAW (unshifted)
            # here -- it gets passed straight into create_bedrock_plot below,
            # which does its own clipping AND shifting (by
            # distance_offset_m=offset_m) via clip_profile_to_global_span.
            # Shifting it here too would double-subtract offset_m inside that
            # function.

            surface_years_to_plot = set()
            surface_data_for_combined = [] # specifically for DEM-only

            for idx, change in enumerate(all_changes_sorted):
                dh_data = change['dh']
                dhdt_data = _get_dhdt_for_change(change)
                demA_data = change['demA']
                demB_data = change['demB']

                period_key = years_list[idx]
                period_display = years_list_display[idx]

                obs_mask = _get_obs_union_mask_for_change(change)
                if obs_mask is not None:
                    dh_data_masked = np.where(obs_mask, dh_data, np.nan)
                    dhdt_data_masked = np.where(obs_mask, dhdt_data, np.nan)
                else:
                    dh_data_masked = dh_data
                    dhdt_data_masked = dhdt_data

                # That mismatch is why the elevation lines truncated correctly
                # at each dashed terminus line but the dh/dt lines did not --
                # the two were never using the same restriction mechanism.
                # Restricting dh/dhdt to segB (the yB/later-date segment) here
                # makes them consistent with demB and with the dashed lines
                # themselves, independent of whatever AREA_MASK_OPTION does to
                # obs_union_mask upstream.
                segA = get_centerline_segment_by_year(centerline, change['yA'], grid_info['crs'])
                segB = get_centerline_segment_by_year(centerline, change['yB'], grid_info['crs'])

                # max_distance=CENTERLINE_MAX_DISTANCE + offset_m (not just
                # CENTERLINE_MAX_DISTANCE) for the same reason as the bedrock
                # sampling above: these distances get shifted by -offset_m
                # right below, and should still reach the full, already-shifted
                # xmax_km computed earlier in this function.
                _raw_max_distance = cfg.CENTERLINE_MAX_DISTANCE + offset_m

                distances_dh, values_dh = sample_raster_on_centerline_segment(
                    dh_data_masked, grid_info['transform'], centerline, segB,
                    cfg.CENTERLINE_SAMPLING_DISTANCE, buffer_width=100, stat='mean',
                    max_distance=_raw_max_distance
                )
                distances_dh = distances_dh - offset_m

                distances_dhdt, values_dhdt = sample_raster_on_centerline_segment(
                    dhdt_data_masked, grid_info['transform'], centerline, segB,
                    cfg.CENTERLINE_SAMPLING_DISTANCE, buffer_width=100, stat='mean',
                    max_distance=_raw_max_distance
                )
                distances_dhdt = distances_dhdt - offset_m

                distances_demA, values_demA = sample_raster_on_centerline_segment(
                    demA_data, grid_info['transform'], centerline, segA,
                    cfg.CENTERLINE_SAMPLING_DISTANCE, buffer_width=100, stat='mean',
                    max_distance=_raw_max_distance)
                distances_demA = distances_demA - offset_m

                distances_demB, values_demB = sample_raster_on_centerline_segment(
                    demB_data, grid_info['transform'], centerline, segB,
                    cfg.CENTERLINE_SAMPLING_DISTANCE, buffer_width=100, stat='mean',
                    max_distance=_raw_max_distance
                )
                distances_demB = distances_demB - offset_m

                if distances_demA.size > 0:
                    year_A = change['yA']
                    if not should_skip_individual_year(year_A, epoch_labels):
                        all_dem_profiles[year_A] = (distances_demA, values_demA)
                        if year_A not in surface_years_to_plot:
                            surface_years_to_plot.add(year_A)
                            surface_data_for_combined.append((year_A, distances_demA.copy(), values_demA.copy()))

                if distances_demB.size > 0:
                    year_B = change['yB']
                    if not should_skip_individual_year(year_B, epoch_labels):
                        all_dem_profiles[year_B] = (distances_demB, values_demB)
                        if year_B not in surface_years_to_plot:
                            surface_years_to_plot.add(year_B)
                            surface_data_for_combined.append((year_B, distances_demB.copy(), values_demB.copy()))

                if distances_dh.size > 0:
                    ax_dh.plot(
                        distances_dh / 1000.0, values_dh, color=year_colors[idx],
                        linewidth=2.2, label=period_display, alpha=0.8
                    )
                    all_dh_profiles.append((distances_dh, values_dh))
                    has_data = True
                else:
                    all_dh_profiles.append(None)

                if distances_dhdt.size > 0:
                    ax_dhdt.plot(
                        distances_dhdt / 1000.0, values_dhdt, color=year_colors[idx],
                        linewidth=2.2, label=period_display, alpha=0.8
                    )
                    all_dhdt_profiles.append((distances_dhdt, values_dhdt))

                    glacier_profile_data[period_key] = {
                        'distances_km': distances_dhdt / 1000.0,
                        'dh': values_dh if distances_dh.size > 0 else np.array([]),
                        'dhdt': values_dhdt,
                        'mean_dh': np.mean(values_dh) if distances_dh.size > 0 else np.nan,
                        'mean_dhdt': np.mean(values_dhdt),
                        'period_display': period_display
                    }
                else:
                    all_dhdt_profiles.append(None)

            all_glacier_data[glacier_name] = glacier_profile_data

            if has_data:
                # xmin_km/xmax_km already computed above (re-anchored so 0
                # = this glacier's own most seaward mapped front, matching
                # offset_m applied to distances_dh/distances_dhdt earlier).

                ax_dh.axhline(y=0, color='black', linestyle='-', alpha=0.5, linewidth=1.5)
                ax_dh.set_title("Elevation Change (dh)", fontsize=14, fontweight='bold')
                ax_dh.set_xlabel("Distance Along Centerline (km)", fontsize=12)
                ax_dh.set_ylabel("dh (m)", fontsize=12)
                ax_dh.grid(True, linestyle='--', alpha=0.55)
                ax_dh.legend(title="Period", loc='best', fontsize=9)
                if xmax_km > xmin_km:
                    ax_dh.set_xlim(xmin_km, xmax_km)
                else:
                    ax_dh.set_xlim(0, cfg.CENTERLINE_MAX_DISTANCE / 1000.0)

                ax_dhdt.axhline(y=0, color='black', linestyle='-', alpha=0.5, linewidth=1.5)
                ax_dhdt.set_title("Elevation Change Rate (dh)", fontsize=14, fontweight='bold')
                ax_dhdt.set_xlabel("Distance Along Centerline (km)", fontsize=12)
                ax_dhdt.set_ylabel("dhdt (m a⁻¹)", fontsize=12)
                ax_dhdt.grid(True, linestyle='--', alpha=0.55)
                ax_dhdt.legend(title="Period", loc='best', fontsize=9)
                if xmax_km > xmin_km:
                    ax_dhdt.set_xlim(xmin_km, xmax_km)
                else:
                    ax_dhdt.set_xlim(0, cfg.CENTERLINE_MAX_DISTANCE / 1000.0)

                fig.suptitle(f"[{glacier_idx}] Elevation Change Analysis: {glacier_name}",
                             fontsize=18, fontweight='bold')

                out_path = os.path.join(
                    cfg.CENTERLINE_PLOT_DIR,
                    f"{glacier_idx:02d}_{centerline_name}_elevation_analysis.png"
                )
                plt.savefig(out_path, dpi=cfg.DPI, bbox_inches='tight')
                print(f"      ✅ Saved: {os.path.basename(out_path)}")

                # Bedrock Plot
                bedrock_plot_data = create_bedrock_plot(
                    centerline_name, glacier_name, centerline,
                    year_results, surface_years_to_plot,
                    bedrock_distances, bedrock_values,
                    grid_info, terminus_positions_by_year,
                    glacier_number=glacier_idx, distance_offset_m=offset_m,
                    grounding_line_distance_m=grounding_line_distance_m
                )

                # DEM-only Plot
                create_dem_only_profile_plot(
                    centerline_name, glacier_name, centerline,
                    year_results, surface_years_to_plot,
                    grid_info, terminus_positions_by_year,
                    surface_data_for_combined,
                    glacier_number=glacier_idx, distance_offset_m=offset_m
                )

                if bedrock_plot_data is not None:
                    all_glacier_bedrock_data[glacier_name] = bedrock_plot_data
                    all_glacier_bedrock_data[glacier_name]['glacier_number'] = glacier_idx
                    # The bedrock dict has everything needed for the 9-panel dem plot as well

                csv_path = os.path.join(cfg.CENTERLINE_DATA_DIR, f"{glacier_idx:02d}_{centerline_name}_data.csv")
                with open(csv_path, 'w', newline='') as csvfile:
                    writer = csv.writer(csvfile)
                    header = ['distance_km',
                              'bedrock_shahateet_m']
                    unique_years = sorted(surface_years_to_plot)
                    for year in unique_years:
                        year_display = get_display_year(year)
                        header.append(f'{year_display}_surface_elevation_m')
                    for period_display in years_list_display:
                        header.extend([f'{period_display}_dh_m', f'{period_display}_dhdt_m_yr'])
                    writer.writerow(header)

                    max_len = max(
                        [len(bedrock_distances) if bedrock_distances is not None else 0] +
                        [len(prof[0]) if prof is not None else 0 for prof in all_dh_profiles + all_dhdt_profiles] +
                        [len(prof[0]) for prof in all_dem_profiles.values()]
                    )

                    for i in range(max_len):
                        row = []
                        dist_km = None

                        # bedrock_distances is left RAW (unshifted, see
                        # the note where it's sampled above), but
                        # all_dh_profiles is already shifted by offset_m --
                        # subtract it explicitly here too so this column means
                        # the same "distance from this glacier's own most
                        # seaward mapped front" in every row regardless of
                        # which branch fires.
                        if bedrock_distances is not None and i < len(bedrock_distances):
                            dist_km = (bedrock_distances[i] - offset_m) / 1000.0
                        elif all_dh_profiles and all_dh_profiles[0] is not None and i < len(all_dh_profiles[0][0]):
                            dist_km = all_dh_profiles[0][0][i] / 1000.0

                        row.append(f'{dist_km:.3f}' if dist_km is not None else '')

                        if bedrock_values is not None and i < len(bedrock_values):
                            row.append(f'{bedrock_values[i]:.2f}' if np.isfinite(bedrock_values[i]) else '')
                        else: row.append('')

                        for year in unique_years:
                            if year in all_dem_profiles:
                                _, values = all_dem_profiles[year]
                                if i < len(values):
                                    row.append(f'{values[i]:.2f}' if np.isfinite(values[i]) else '')
                                else: row.append('')
                            else: row.append('')

                        for dh_prof, dhdt_prof in zip(all_dh_profiles, all_dhdt_profiles):
                            dh_val = dh_prof[1][i] if dh_prof is not None and len(dh_prof[1]) > i else np.nan
                            dhdt_val = dhdt_prof[1][i] if dhdt_prof is not None and len(dhdt_prof[1]) > i else np.nan
                            row.append(f'{dh_val:.2f}' if np.isfinite(dh_val) else '')
                            row.append(f'{dhdt_val:.4f}' if np.isfinite(dhdt_val) else '')

                        writer.writerow(row)

                print(f"      ✅ Saved: {os.path.basename(csv_path)}")

            plt.close(fig)

        except Exception as e:
            print(f"      ❌ Error: {e}")
            import traceback
            traceback.print_exc()
            plt.close('all')

    print("\n   ✓ Centerline analysis complete")

    if all_glacier_data:
        create_glacier_comparison_plots(all_glacier_data, years_list, year_colors)

    if all_glacier_bedrock_data:
        create_combined_bedrock_plot(all_glacier_bedrock_data, grid_info)
        create_combined_dem_only_plot(all_glacier_bedrock_data, grid_info, preloaded_ref_dem)

        # ADD 'map_change' as the final argument here
        create_combined_dem_and_dhdt_9panel(all_glacier_bedrock_data, all_glacier_data, grid_info, preloaded_ref_dem,
                                            map_change)
    return all_glacier_data


def create_bedrock_plot(centerline_name, glacier_name, centerline,
                        year_results, surface_years_to_plot,
                        bedrock_distances, bedrock_values,
                        grid_info, terminus_positions_by_year,
                        glacier_number=None, distance_offset_m=None,
                        grounding_line_distance_m=None):
    """
    Create bedrock plot using one centerline and year-mask-clipped surface
    segments.
    """
    if not cfg.CREATE_BEDROCK_PLOTS:
        print("      Skipping bedrock plot (CREATE_BEDROCK_PLOTS=False)")
        return None

    print("      Creating bedrock plot...")

    fig, ax = plt.subplots(figsize=(14, 8))

    centerline_length = centerline.length
    print(f"         Full centerline length: {centerline_length / 1000:.2f} km")

    # Determine overall valid glacier span from all available clipped mask intersections
    valid_ranges = []
    valid_ranges_by_year = {}
    year_segments = {}
    for yr in cfg.YEAR_MASKS.keys():
        seg = get_centerline_segment_by_year(centerline, yr, grid_info['crs'])
        year_segments[yr] = seg
        if seg is not None:
            d0, d1 = get_segment_distance_range_on_centerline(centerline, seg)
            if d0 is not None and d1 is not None:
                valid_ranges.append((d0, d1))
                valid_ranges_by_year[yr] = (d0, d1)

    if not valid_ranges:
        print("      ⚠️  No valid mask-clipped centerline segments available")
        return None

    # Restrict the span used for the x-axis to only the mask-years actually
    # represented in surface_years_to_plot (via MASK_ASSIGNMENT). A raw year
    # folded into a merged period (e.g. "2016" once "2016-2017" exists) is
    # never actually sampled/plotted here, so letting its own d0 pull
    # global_min_dist/global_max_dist would drag the axis out to cover a front
    # position with no data drawn against it.
    relevant_mask_years = {mask_year_for_label(y) for y in surface_years_to_plot}
    relevant_ranges = [
        valid_ranges_by_year[yr] for yr in relevant_mask_years if yr in valid_ranges_by_year
    ]
    if relevant_ranges:
        global_min_dist = min(r[0] for r in relevant_ranges)
        global_max_dist = max(r[1] for r in relevant_ranges)
    else:
        global_min_dist = min(r[0] for r in valid_ranges)
        global_max_dist = max(r[1] for r in valid_ranges)

    # If the caller supplied its own authoritative offset (see docstring), use
    # it as the distance origin instead of the value just computed above, so
    # this plot's clipping/shift always agrees exactly with the caller's
    # already-shifted terminus_positions_by_year and its own dh/dhdt/surface
    # curves for this same glacier.
    if distance_offset_m is not None:
        global_min_dist = distance_offset_m

    print(f"         Overall clipped centerline span: {global_min_dist / 1000:.2f} to {global_max_dist / 1000:.2f} km")

    # Diagnostic: which YEAR_MASKS outline is actually the limiting (shortest)
    # one at the far/inland end for THIS glacier, and how much centerline it
    # leaves unused.
    if relevant_ranges:
        _limiting_yr, _limiting_d1 = min(
            ((yr, valid_ranges_by_year[yr][1]) for yr in relevant_mask_years if yr in valid_ranges_by_year),
            key=lambda t: t[1]
        )
        print(f"         Limiting (shortest) mask at inland end: {_limiting_yr} "
              f"-> reaches {_limiting_d1 / 1000:.2f} km of {centerline_length / 1000:.2f} km "
              f"full centerline ({(centerline_length - _limiting_d1) / 1000:.2f} km unused)")

    def clip_profile_to_global_span(distances, values):
        if distances is None or values is None or len(distances) == 0:
            return None, None
        mask = (distances >= global_min_dist) & (distances <= global_max_dist)
        d = distances[mask] - global_min_dist
        v = values[mask]
        if len(d) == 0:
            return None, None
        return d, v

    # --- Bedrock ---
    bedrock_dist_clipped = None
    bedrock_val_clipped = None
    if bedrock_distances is not None and bedrock_values is not None and len(bedrock_distances) > 0:
        bedrock_dist_clipped, bedrock_val_clipped = clip_profile_to_global_span(
            bedrock_distances, bedrock_values
        )

        if bedrock_dist_clipped is not None:
            ax.fill_between(
                bedrock_dist_clipped / 1000.0,
                bedrock_val_clipped,
                -800,
                alpha=0.4,
                color='dimgray',
                label=with_bedrock_source('Bedrock', ' '),
                zorder=1
            )
            ax.plot(
                bedrock_dist_clipped / 1000.0,
                bedrock_val_clipped,
                color='black',
                linewidth=2.5,
                linestyle='-',
                zorder=2
            )
            # Sea level reference (0 m): a darker blue "ocean" shade wherever
            # the bed dips below sea level, regardless of the grounding line
            # position. So this is intentionally back to the simple bed<0
            # condition; the ice-color fill just below is still split at the
            # grounding line (see that block) since solid-filling floating ice
            # down to the bed would be a different, more misleading kind of
            # error.
            ax.fill_between(
                bedrock_dist_clipped / 1000.0,
                bedrock_val_clipped,
                0,
                where=(bedrock_val_clipped < 0),
                interpolate=True,
                alpha=0.45,
                color='steelblue',
                zorder=1.3
            )
            ax.axhline(
                y=0,
                color='lightskyblue',
                linewidth=1.8,
                linestyle='-',
                alpha=0.95,
                zorder=5,
                label='Sea level (0 m)'
            )
            print(f"         ✓ Plotted bedrock from {bedrock_dist_clipped[0] / 1000:.2f} to {bedrock_dist_clipped[-1] / 1000:.2f} km")

    # --- Actual DEM surfaces, clipped by year mask intersection ---
    unique_years = sorted(surface_years_to_plot)
    dem_colormap = build_years_colormap(len(unique_years))
    surface_data_for_combined = []

    for idx, year in enumerate(unique_years):
        if year not in year_results:
            continue

        dem_file = year_results[year].get('dem_file')
        if dem_file is None or not os.path.exists(dem_file):
            print(f"         ⚠️  DEM not found for {year}")
            continue

        try:
            with rasterio.open(dem_file) as src:
                dem_data = src.read(1)
                nodata = src.nodata

            if nodata is not None:
                dem_data[dem_data == nodata] = np.nan

            seg = get_centerline_segment_by_year(centerline, year, grid_info['crs'])
            if seg is None:
                print(f"         ⚠️  No valid clipped centerline segment for {year}")
                continue

            # max_distance=CENTERLINE_MAX_DISTANCE + global_min_dist: this
            # profile is shifted by -global_min_dist right below, and should
            # still reach the full, already-shifted shared xmax_km -- see the
            # matching note on the bedrock sampling call in
            # plot_centerline_elevation_changes for the full rationale.
            distances_plot, values_plot = sample_raster_on_centerline_segment(
                dem_data, grid_info['transform'], centerline, seg,
                cfg.CENTERLINE_SAMPLING_DISTANCE, buffer_width=100, stat='mean',
                max_distance=cfg.CENTERLINE_MAX_DISTANCE + global_min_dist
            )
            distances_plot = distances_plot - global_min_dist

            if len(distances_plot) > 0:
                surface_data_for_combined.append((year, distances_plot.copy(), values_plot.copy()))

                ax.plot(
                    distances_plot / 1000.0,
                    values_plot,
                    color=dem_colormap[idx],
                    linewidth=2.5,
                    label=f'Surface {get_display_year(year)}',
                    alpha=0.9,
                    zorder=3
                )

                if bedrock_dist_clipped is not None and len(bedrock_dist_clipped) > 0:
                    try:
                        bedrock_interp = np.interp(
                            distances_plot,
                            bedrock_dist_clipped,
                            bedrock_val_clipped,
                            left=np.nan,
                            right=np.nan
                        )

                        # Landward of the grounding line: solid ice fill down
                        # to the bed. Seaward of it: freeboard-only fill down
                        # to sea level (below the waterline is unknown
                        # floating-ice-base depth, left to the ocean tint
                        # instead).
                        x_fill, lo_fill, hi_fill = _ice_fill_arrays_split_at_grounding_line(
                            distances_plot, values_plot, bedrock_interp,
                            bedrock_dist_clipped, bedrock_val_clipped,
                            grounding_line_distance_m
                        )
                        if x_fill is not None:
                            ax.fill_between(
                                x_fill, lo_fill, hi_fill,
                                alpha=0.2,
                                color=dem_colormap[idx],
                                zorder=0
                            )
                    except Exception:
                        pass

                print(f"         {get_display_year(year)}: plotted clipped surface from {distances_plot[0] / 1000:.2f} to {distances_plot[-1] / 1000:.2f} km")

        except Exception as e:
            print(f"         ⚠️  Error processing {year}: {e}")
            import traceback
            traceback.print_exc()

    if grounding_line_distance_m is not None and bedrock_dist_clipped is not None and len(bedrock_dist_clipped) > 0:
        try:
            bed_at_gl = np.interp(
                grounding_line_distance_m, bedrock_dist_clipped, bedrock_val_clipped,
                left=np.nan, right=np.nan
            )
            if np.isfinite(bed_at_gl):
                gl_km = grounding_line_distance_m / 1000.0
                ax.plot(
                    [gl_km, gl_km], [-800, bed_at_gl],
                    color='black', linewidth=2.5, linestyle='-',
                    solid_capstyle='butt', zorder=8,
                    label=cfg.GROUNDING_LINE_LABEL
                )
                ax.text(
                    gl_km, bed_at_gl, cfg.GROUNDING_LINE_SHORT_LABEL,
                    rotation=90, va='bottom', ha='center',
                    fontsize=10, fontweight='bold', color='black',
                    zorder=11,
                    bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.8, edgecolor='none')
                )
                print(f"         ✓ Plotted grounding line at {gl_km:.2f} km")
            else:
                print("         ⚠️  Grounding line position outside bedrock data range")
        except Exception as e:
            print(f"         ⚠️  Could not plot grounding line: {e}")

    # --- Marker lines showing clipped start positions for each year ---
    for mask_year in sorted(terminus_positions_by_year.keys()):
        position_km = terminus_positions_by_year[mask_year] / 1000.0

        ax.axvline(
            x=position_km,
            color=terminus_color(mask_year),
            linestyle='--',
            linewidth=2,
            alpha=0.7,
            zorder=10
        )

        ylim = ax.get_ylim()
        label_y = ylim[0] + 0.10 * (ylim[1] - ylim[0])

        ax.text(
            position_km,
            label_y,
            mask_year,
            rotation=90,
            verticalalignment='top',
            horizontalalignment='right',
            fontsize=10,
            fontweight='bold',
            color=terminus_color(mask_year),
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8, edgecolor='none')
        )

    ax.set_xlabel("Distance along flowline (km)", fontsize=13)
    ax.set_ylabel("Elevation (m)", fontsize=13)

    if glacier_number is not None:
        ax.set_title(f"[{glacier_number}] {glacier_name}", fontsize=15, fontweight='bold')
    else:
        ax.set_title(glacier_name, fontsize=15, fontweight='bold')

    ax.grid(True, linestyle='--', alpha=0.4)

    h1, l1 = ax.get_legend_handles_labels()
    legend = ax.legend(h1, l1, loc='lower right', fontsize=11)
    legend.set_zorder(20)
    legend.get_frame().set_facecolor('white')
    legend.get_frame().set_edgecolor('black')
    legend.get_frame().set_alpha(1.0)
    legend.get_frame().set_linewidth(1.5)

    _xmin_km = 0.0
    _xmax_km = min(cfg.CENTERLINE_MAX_DISTANCE / 1000.0, (global_max_dist - global_min_dist) / 1000.0)
    if _xmax_km > _xmin_km:
        ax.set_xlim(_xmin_km, _xmax_km)
    else:
        ax.set_xlim(0, cfg.CENTERLINE_MAX_DISTANCE / 1000.0)
    ax.autoscale(enable=True, axis='y')

    plt.tight_layout()

    if glacier_number is not None:
        out_path = os.path.join(cfg.CENTERLINE_PLOT_DIR, f"{glacier_number:02d}_{centerline_name}_bedrock_profile.png")
    else:
        out_path = os.path.join(cfg.CENTERLINE_PLOT_DIR, f"{centerline_name}_bedrock_profile.png")

    plt.savefig(out_path, dpi=cfg.DPI, bbox_inches='tight')
    plt.close()
    print("      ✅ Saved bedrock plot")

    return {
        'glacier_name': glacier_name,
        'bedrock_distances': bedrock_dist_clipped.copy() if bedrock_dist_clipped is not None else None,
        'bedrock_values': bedrock_val_clipped.copy() if bedrock_val_clipped is not None else None,
        'grounding_line_distance': grounding_line_distance_m,
        'surface_data': surface_data_for_combined,
        'terminus_positions': terminus_positions_by_year.copy(),
        'centerline_length': centerline_length,
        'dem_colormap': dem_colormap,
        # Both re-based to the shifted (offset-subtracted) coordinate system
        # used throughout this function -- 0.0 always means "this glacier's own
        # most seaward mapped front" now, not "wherever this glacier's
        # particular centerline shapefile happens to start".
        'global_min_dist': 0.0,
        'global_max_dist': global_max_dist - global_min_dist
    }


def create_dem_only_profile_plot(centerline_name, glacier_name, centerline,
                                 year_results, surface_years_to_plot,
                                 grid_info, terminus_positions_by_year,
                                 surface_data_for_combined,
                                 glacier_number=None, distance_offset_m=None):
    """Create a profile plot showing only the surface DEMs (no bedrock/thickness)."""
    print("      Creating DEM-only profile plot...")

    fig, ax = plt.subplots(figsize=(14, 8))


    # Determine overall valid glacier span
    valid_ranges = []
    valid_ranges_by_year = {}
    for yr in cfg.YEAR_MASKS.keys():
        seg = get_centerline_segment_by_year(centerline, yr, grid_info['crs'])
        if seg is not None:
            d0, d1 = get_segment_distance_range_on_centerline(centerline, seg)
            if d0 is not None and d1 is not None:
                valid_ranges.append((d0, d1))
                valid_ranges_by_year[yr] = (d0, d1)

    if not valid_ranges:
        return

    # Restrict the span used for the x-axis to only the mask-years actually
    # represented in surface_years_to_plot (via MASK_ASSIGNMENT) -- see the
    # matching comment in create_bedrock_plot for why.
    relevant_mask_years = {mask_year_for_label(y) for y in surface_years_to_plot}
    relevant_ranges = [
        valid_ranges_by_year[yr] for yr in relevant_mask_years if yr in valid_ranges_by_year
    ]
    if relevant_ranges:
        global_min_dist = min(r[0] for r in relevant_ranges)
        global_max_dist = max(r[1] for r in relevant_ranges)
    else:
        global_min_dist = min(r[0] for r in valid_ranges)
        global_max_dist = max(r[1] for r in valid_ranges)

    unique_years = sorted(surface_years_to_plot)
    dem_colormap = build_years_colormap(max(len(unique_years), 1))

    # Plot actual surface elevations
    for surf_idx, (year, distances_plot, values_plot) in enumerate(surface_data_for_combined):
        color_idx = min(surf_idx, len(dem_colormap) - 1)
        ax.plot(
            distances_plot / 1000.0,
            values_plot,
            color=dem_colormap[color_idx],
            linewidth=2.5,
            label=f'Surface {get_display_year(year)}',
            alpha=0.9,
            zorder=3
        )

    # Marker lines showing clipped start positions for each year
    for mask_year in sorted(terminus_positions_by_year.keys()):
        position_km = terminus_positions_by_year[mask_year] / 1000.0
        ax.axvline(x=position_km, color=terminus_color(mask_year, dem_panels=True),
                   linestyle='--', linewidth=2, alpha=0.7, zorder=10)

        label_y = -10 + 0.10 * (500 - (-10))
        ax.text(position_km, label_y, mask_year, rotation=90,
                verticalalignment='bottom', horizontalalignment='right',
                fontsize=10, fontweight='bold', color=terminus_color(mask_year, dem_panels=True),
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8, edgecolor='none'))

    ax.set_xlabel("Distance along flowline (km)", fontsize=13)
    ax.set_ylabel("Elevation [m]", fontsize=13)

    if glacier_number is not None:
        ax.set_title(f"[{glacier_number}] {glacier_name} (Surface Only)", fontsize=15, fontweight='bold')
    else:
        ax.set_title(f"{glacier_name} (Surface Only)", fontsize=15, fontweight='bold')

    ax.grid(True, linestyle='--', alpha=0.4)
    ax.legend(loc='upper left', fontsize=9).set_zorder(20)

    # The plotted curves/lines above are already shifted by the caller's offset
    # (surface_data_for_combined / terminus_positions_by_year), so the xlim
    # must use that SAME offset, not the raw global_min_dist just recomputed
    # above -- otherwise the axis window and the data it's supposed to frame
    # would disagree.
    _offset_for_xlim = distance_offset_m if distance_offset_m is not None else global_min_dist
    _xmin_km = 0.0
    _xmax_km = min(cfg.CENTERLINE_MAX_DISTANCE / 1000.0, (global_max_dist - _offset_for_xlim) / 1000.0)
    if _xmax_km > _xmin_km:
        ax.set_xlim(_xmin_km, _xmax_km)
    else:
        ax.set_xlim(0, cfg.CENTERLINE_MAX_DISTANCE / 1000.0)
    ax.set_ylim(-10, 500)

    if glacier_number is not None:
        out_path = os.path.join(cfg.CENTERLINE_DEM_PLOT_DIR, f"{glacier_number:02d}_{centerline_name}_dem_profile.png")
    else:
        out_path = os.path.join(cfg.CENTERLINE_DEM_PLOT_DIR, f"{centerline_name}_dem_profile.png")

    plt.savefig(out_path, dpi=cfg.DPI, bbox_inches='tight')
    plt.close()
    print("      ✅ Saved DEM-only plot")


def create_glacier_comparison_plots(all_glacier_data, years_list, year_colors):
    """Create comparison plots showing all glaciers together."""
    print("\n▶ Creating multi-glacier comparison plots...")

    if not all_glacier_data:
        return

    glacier_names = list(all_glacier_data.keys())
    periods = years_list  # Original format

    fig, ax = plt.subplots(figsize=(14, 8))

    for glacier_name in glacier_names:
        glacier_data = all_glacier_data[glacier_name]
        mean_dhdts = []
        valid_periods = []

        for period in periods:
            if period in glacier_data:
                mean_dhdt = glacier_data[period]['mean_dhdt']
                if np.isfinite(mean_dhdt):
                    mean_dhdts.append(mean_dhdt)
                    valid_periods.append(period)

        if valid_periods:
            mid_years = []
            for period in valid_periods:
                parts = period.split('-')
                if len(parts) >= 2:
                    yA = int(parts[0])
                    yB = int(parts[-1])
                    mid_year = (yA + yB) / 2
                    mid_years.append(mid_year)

            if mid_years:
                ax.plot(mid_years, mean_dhdts, 'o-', linewidth=2.5, markersize=8,
                        label=glacier_name, alpha=0.8)

    ax.axhline(y=0, color='black', linestyle='--', alpha=0.5, linewidth=1.5)
    ax.set_xlabel('Year', fontsize=14)
    ax.set_ylabel('Mean dhdt (m a⁻¹)', fontsize=14)
    ax.set_title('Glacier-wide Mean Elevation Change Rate Comparison', fontsize=16, fontweight='bold')
    ax.legend(loc='best', fontsize=11)
    ax.grid(True, alpha=0.3)

    out_path = os.path.join(cfg.CENTERLINE_PLOT_DIR, "glacier_comparison_mean_dhdt.png")
    plt.savefig(out_path, dpi=cfg.DPI, bbox_inches='tight')
    plt.close()
    print(f"   ✓ Saved: {os.path.basename(out_path)}")

    # Heatmap - format labels for display
    fig, ax = plt.subplots(figsize=(14, len(glacier_names) * 0.8 + 2))

    data_matrix = np.full((len(glacier_names), len(periods)), np.nan)

    for i, glacier_name in enumerate(glacier_names):
        glacier_data = all_glacier_data[glacier_name]
        for j, period in enumerate(periods):
            if period in glacier_data:
                data_matrix[i, j] = glacier_data[period]['mean_dhdt']

    vmax = np.nanmax(np.abs(data_matrix))
    if vmax == 0 or not np.isfinite(vmax):
        vmax = 1.0

    im = ax.imshow(data_matrix, cmap='RdBu', aspect='auto',
                   vmin=-vmax, vmax=vmax, interpolation='nearest')

    # Format period labels for display
    display_labels = []
    for period in periods:
        parts = period.split('-')

        if len(parts) == 2:
            # Simple case: "2014-2017"
            display_labels.append(format_period_label(parts[0], parts[1]))
        elif len(parts) == 3:
            # Merged end: "2014-2024-2025" → "2014-2024/25"
            display_labels.append(format_period_label(parts[0], f"{parts[1]}-{parts[2]}"))
        elif len(parts) == 4:
            # Both merged: "2016-2017-2024-2025" → "2016/17-2024/25"
            yA_formatted = get_display_year(f"{parts[0]}-{parts[1]}")  # "2016/17"
            yB_formatted = get_display_year(f"{parts[2]}-{parts[3]}")  # "2024/25"
            display_labels.append(f"{yA_formatted}-{yB_formatted}")
        else:
            display_labels.append(period)

    ax.set_xticks(np.arange(len(periods)))
    ax.set_yticks(np.arange(len(glacier_names)))
    ax.set_xticklabels(display_labels, rotation=45, ha='right')
    ax.set_yticklabels(glacier_names)

    ax.set_xlabel('Period', fontsize=12)
    ax.set_ylabel('Glacier', fontsize=12)
    ax.set_title('Mean Elevation Change Rate by Glacier and Period', fontsize=14, fontweight='bold')

    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Mean dhdt (m a⁻¹)', fontsize=12)

    for i in range(len(glacier_names)):
        for j in range(len(periods)):
            value = data_matrix[i, j]
            if np.isfinite(value):
                text_color = 'white' if abs(value) > vmax * 0.5 else 'black'
                ax.text(j, i, f'{value:.2f}', ha='center', va='center',
                        color=text_color, fontsize=9, fontweight='bold')

    plt.tight_layout()

    out_path = os.path.join(cfg.CENTERLINE_PLOT_DIR, "glacier_comparison_heatmap.png")
    plt.savefig(out_path, dpi=cfg.DPI, bbox_inches='tight')
    plt.close()
    print(f"   ✓ Saved: {os.path.basename(out_path)}")

    # Summary CSV - use formatted labels
    summary_csv_path = os.path.join(cfg.CENTERLINE_DATA_DIR, "glacier_comparison_summary.csv")
    with open(summary_csv_path, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        header = ['Glacier'] + [f'{label}_mean_dhdt' for label in display_labels]
        writer.writerow(header)

        for glacier_name in glacier_names:
            row = [glacier_name]
            glacier_data = all_glacier_data[glacier_name]
            for period in periods:
                if period in glacier_data:
                    mean_dhdt = glacier_data[period]['mean_dhdt']
                    row.append(f'{mean_dhdt:.4f}' if np.isfinite(mean_dhdt) else '')
                else:
                    row.append('')
            writer.writerow(row)

    print(f"   ✓ Saved summary: {os.path.basename(summary_csv_path)}")
