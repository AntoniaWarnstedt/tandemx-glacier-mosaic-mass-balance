"""Centerline profiles: surface elevation, dh and dh/dt sampled along each
glacier centerline, the profile tables and the glacier x period dh/dt heatmap.
"""

import csv
import os

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import rasterio

from .. import config as cfg
from .centerline_geometry import (
    get_centerline_segment_by_year,
    get_segment_distance_range_on_centerline,
    glacier_name_from_file,
    list_centerline_files,
    sample_raster_on_centerline_segment,
)
from ..utils import (
    change_sort_key,
    format_period_label,
    get_display_year,
    get_longest_linestring,
    mask_year_for_label,
    should_skip_individual_year,
)


def collect_centerline_profiles(all_changes, year_results, grid_info, mb_results=None):
    """
    Sample every centerline in cfg.CENTERLINE_DIR. Distances are measured from
    each glacier's most seaward mapped front (0 km); every profile is limited
    to the outline of its epoch (cfg.YEAR_MASKS via MASK_ASSIGNMENT).

    Returns
    -------
    all_glacier_data : {glacier: {period_key: {'distances_km', 'dh', 'dhdt',
                        'mean_dh', 'mean_dhdt', 'period_display'}}}
    glacier_profiles : {glacier: {'glacier_number', 'surface_data',
                        'terminus_positions', 'global_min_dist', 'global_max_dist'}}
    years_list       : period keys "yA-yB" in chronological order
    """
    print("\n▶ Sampling centerline profiles...")
    centerline_files = list_centerline_files()
    if not centerline_files or not all_changes:
        print("   ⚠️  No centerlines or no elevation-change periods")
        return {}, {}, []
    print(f"   Found {len(centerline_files)} centerline files")

    all_changes_sorted = sorted(all_changes, key=change_sort_key)
    years_list = [f"{c['yA']}-{c['yB']}" for c in all_changes_sorted]
    years_list_display = [format_period_label(c['yA'], c['yB']) for c in all_changes_sorted]

    # epochs of the dh/dt periods and their outline years (same for all glaciers)
    epoch_labels = {y for c in all_changes_sorted for y in (c['yA'], c['yB'])}
    surface_epochs = {y for y in epoch_labels if not should_skip_individual_year(y, epoch_labels)}
    relevant_mask_years = {mask_year_for_label(y) for y in surface_epochs}

    mb_dict = {f"{r['yA']}-{r['yB']}": r for r in mb_results} if mb_results else {}

    def dhdt_for_change(c):
        """dh/dt with the applied corrections (falls back to the filtered dh/dt)."""
        mb_res = mb_dict.get(f"{c['yA']}-{c['yB']}")
        if mb_res is not None and mb_res.get('dhdt_bias_corrected_raster') is not None:
            return mb_res['dhdt_bias_corrected_raster']
        print(f"      ⚠️ No corrected dhdt for {c['yA']}-{c['yB']}; using the filtered dhdt.")
        return c['dhdt']

    def obs_mask_for_change(c):
        """Analysis domain with valid dh/dt of the period (None = no masking)."""
        mb_res = mb_dict.get(f"{c['yA']}-{c['yB']}")
        if mb_res is not None and mb_res.get('obs_union_mask') is not None:
            return mb_res['obs_union_mask'].astype(bool)
        print(f"      ⚠️ No obs_union_mask for {c['yA']}-{c['yB']}; profile sampled unmasked.")
        return None

    def sample(raster, line, seg, max_distance):
        return sample_raster_on_centerline_segment(
            raster, grid_info['transform'], line, seg, cfg.CENTERLINE_SAMPLING_DISTANCE,
            buffer_width=100, stat='mean', max_distance=max_distance)

    os.makedirs(cfg.CENTERLINE_DATA_DIR, exist_ok=True)
    all_glacier_data, glacier_profiles = {}, {}

    for glacier_idx, centerline_file in enumerate(centerline_files, start=1):
        centerline_name = os.path.splitext(os.path.basename(centerline_file))[0]
        glacier_name = glacier_name_from_file(centerline_file)
        print(f"\n   [{glacier_idx}] {centerline_name}")

        try:
            centerline_gdf = gpd.read_file(centerline_file)
            if centerline_gdf.empty:
                continue
            centerline = get_longest_linestring(centerline_gdf.to_crs(grid_info['crs']).unary_union)
            if centerline is None:
                print("      ⚠️  No valid LineString geometry found")
                continue

            # centerline section inside each outline; its seaward end = terminus position
            segment_ranges, terminus_positions = {}, {}
            for mask_year in cfg.YEAR_MASKS:
                seg = get_centerline_segment_by_year(centerline, mask_year, grid_info['crs'])
                if seg is None:
                    continue
                d0, d1 = get_segment_distance_range_on_centerline(centerline, seg)
                if d0 is None or d1 is None:
                    continue
                segment_ranges[mask_year] = (d0, d1)
                terminus_positions[mask_year] = d0
            if cfg.TERMINUS_LABEL_YEARS is not None:
                terminus_positions = {yr: pos for yr, pos in terminus_positions.items()
                                      if yr in cfg.TERMINUS_LABEL_YEARS}
            if not segment_ranges:
                print("      ⚠️  No valid clipped segments for any year")
                continue

            # distance origin = most seaward front of the plotted epochs
            ranges = ([segment_ranges[yr] for yr in relevant_mask_years if yr in segment_ranges]
                      or list(segment_ranges.values()))
            offset_m = min(d0 for d0, _ in ranges)
            terminus_positions = {yr: pos - offset_m for yr, pos in terminus_positions.items()}
            raw_max_distance = cfg.CENTERLINE_MAX_DISTANCE + offset_m

            glacier_profile_data = {}
            dh_profiles = []            # (period_display, distances_m, values)
            surface_years = set()
            has_data = False

            for idx, change in enumerate(all_changes_sorted):
                dh_data, dhdt_data = change['dh'], dhdt_for_change(change)
                obs_mask = obs_mask_for_change(change)
                if obs_mask is not None:
                    dh_data = np.where(obs_mask, dh_data, np.nan)
                    dhdt_data = np.where(obs_mask, dhdt_data, np.nan)

                # dh, dh/dt and the later DEM on the yB outline, the earlier DEM on yA
                segA = get_centerline_segment_by_year(centerline, change['yA'], grid_info['crs'])
                segB = get_centerline_segment_by_year(centerline, change['yB'], grid_info['crs'])
                d_dh, v_dh = sample(dh_data, centerline, segB, raw_max_distance)
                d_dhdt, v_dhdt = sample(dhdt_data, centerline, segB, raw_max_distance)
                d_demA, _ = sample(change['demA'], centerline, segA, raw_max_distance)
                d_demB, _ = sample(change['demB'], centerline, segB, raw_max_distance)
                d_dh, d_dhdt = d_dh - offset_m, d_dhdt - offset_m

                for year, d_dem in ((change['yA'], d_demA), (change['yB'], d_demB)):
                    if d_dem.size > 0 and not should_skip_individual_year(year, epoch_labels):
                        surface_years.add(year)

                if d_dh.size > 0:
                    has_data = True
                    dh_profiles.append((years_list_display[idx], d_dh, v_dh))
                if d_dhdt.size > 0:
                    glacier_profile_data[years_list[idx]] = {
                        'distances_km': d_dhdt / 1000.0,
                        'dh': v_dh if d_dh.size > 0 else np.array([]),
                        'dhdt': v_dhdt,
                        'mean_dh': np.mean(v_dh) if d_dh.size > 0 else np.nan,
                        'mean_dhdt': np.mean(v_dhdt),
                        'period_display': years_list_display[idx],
                    }

            all_glacier_data[glacier_name] = glacier_profile_data
            if not has_data:
                continue

            profile = surface_profiles(centerline, year_results, surface_years, grid_info,
                                       segment_ranges, offset_m)
            if profile is None:
                continue
            profile['terminus_positions'] = terminus_positions
            profile['glacier_number'] = glacier_idx
            glacier_profiles[glacier_name] = profile

            write_profile_table(
                os.path.join(cfg.CENTERLINE_DATA_DIR, f"{glacier_idx:02d}_{centerline_name}_profiles.csv"),
                profile['surface_data'], dh_profiles, glacier_profile_data)

        except Exception as e:
            print(f"      ❌ Error: {e}")
            import traceback
            traceback.print_exc()

    write_glacier_summary_table(all_glacier_data, years_list)
    return all_glacier_data, glacier_profiles, years_list


def surface_profiles(centerline, year_results, surface_years, grid_info, segment_ranges, offset_m):
    """
    Surface elevation of each epoch's DEM mosaic along the centerline, limited
    to that epoch's outline and shifted by offset_m (0 km = most seaward front).
    """
    relevant = {mask_year_for_label(y) for y in surface_years}
    ranges = [segment_ranges[yr] for yr in relevant if yr in segment_ranges] or list(segment_ranges.values())
    global_max_dist = max(d1 for _, d1 in ranges)

    surface_data = []
    for year in sorted(surface_years):
        dem_file = year_results.get(year, {}).get('dem_file')
        if dem_file is None or not os.path.exists(dem_file):
            print(f"         ⚠️  DEM not found for {year}")
            continue
        with rasterio.open(dem_file) as src:
            dem_data = src.read(1)
            nodata = src.nodata
        if nodata is not None:
            dem_data[dem_data == nodata] = np.nan
        seg = get_centerline_segment_by_year(centerline, year, grid_info['crs'])
        if seg is None:
            print(f"         ⚠️  No valid clipped centerline segment for {year}")
            continue
        distances, values = sample_raster_on_centerline_segment(
            dem_data, grid_info['transform'], centerline, seg,
            cfg.CENTERLINE_SAMPLING_DISTANCE, buffer_width=100, stat='mean',
            max_distance=cfg.CENTERLINE_MAX_DISTANCE + offset_m)
        distances = distances - offset_m
        if len(distances) > 0:
            surface_data.append((year, distances.copy(), values.copy()))

    return {'surface_data': surface_data,
            'centerline_length': centerline.length,
            'global_min_dist': 0.0,
            'global_max_dist': global_max_dist - offset_m}


def write_profile_table(path, surface_data, dh_profiles, glacier_profile_data):
    """Profiles of one glacier in long format: quantity, epoch/period, distance, value."""
    with open(path, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['quantity', 'epoch_or_period', 'distance_km', 'value'])
        for year, d, v in surface_data:
            for di, vi in zip(d, v):
                w.writerow(['surface_elevation_m', get_display_year(year), f'{di / 1000:.3f}',
                            f'{vi:.2f}' if np.isfinite(vi) else ''])
        for label, d, v in dh_profiles:
            for di, vi in zip(d, v):
                w.writerow(['dh_m', label, f'{di / 1000:.3f}', f'{vi:.2f}' if np.isfinite(vi) else ''])
        for p in glacier_profile_data.values():
            for di, vi in zip(p['distances_km'], p['dhdt']):
                w.writerow(['dhdt_m_a', p['period_display'], f'{di:.3f}',
                            f'{vi:.4f}' if np.isfinite(vi) else ''])
    print(f"      ✅ Saved: {os.path.basename(path)}")


def _period_display_labels(periods):
    """'2016-2017-2024-2025' -> '2016/17-2024/25'."""
    labels = []
    for period in periods:
        parts = period.split('-')
        if len(parts) == 2:
            labels.append(format_period_label(parts[0], parts[1]))
        elif len(parts) == 3:
            labels.append(format_period_label(parts[0], f"{parts[1]}-{parts[2]}"))
        elif len(parts) == 4:
            labels.append(f"{get_display_year(f'{parts[0]}-{parts[1]}')}-"
                          f"{get_display_year(f'{parts[2]}-{parts[3]}')}")
        else:
            labels.append(period)
    return labels


def write_glacier_summary_table(all_glacier_data, periods):
    """Mean centerline dh/dt per glacier and period."""
    if not all_glacier_data:
        return
    path = os.path.join(cfg.CENTERLINE_DATA_DIR, "glacier_comparison_summary.csv")
    with open(path, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['Glacier'] + [f'{label}_mean_dhdt' for label in _period_display_labels(periods)])
        for glacier_name, glacier_data in all_glacier_data.items():
            row = [glacier_name]
            for period in periods:
                m = glacier_data.get(period, {}).get('mean_dhdt', np.nan)
                row.append(f'{m:.4f}' if np.isfinite(m) else '')
            w.writerow(row)
    print(f"   ✓ Saved summary: {os.path.basename(path)}")


def plot_glacier_period_heatmap(all_glacier_data, periods):
    """Heatmap of the mean centerline dh/dt per glacier and period."""
    if not all_glacier_data:
        return
    print("\n▶ Creating glacier x period dh/dt heatmap...")
    glacier_names = list(all_glacier_data.keys())

    fig, ax = plt.subplots(figsize=(14, len(glacier_names) * 0.8 + 2))
    data_matrix = np.full((len(glacier_names), len(periods)), np.nan)
    for i, glacier_name in enumerate(glacier_names):
        for j, period in enumerate(periods):
            if period in all_glacier_data[glacier_name]:
                data_matrix[i, j] = all_glacier_data[glacier_name][period]['mean_dhdt']

    vmax = np.nanmax(np.abs(data_matrix))
    if vmax == 0 or not np.isfinite(vmax):
        vmax = 1.0
    im = ax.imshow(data_matrix, cmap='RdBu', aspect='auto',
                   vmin=-vmax, vmax=vmax, interpolation='nearest')

    ax.set_xticks(np.arange(len(periods)))
    ax.set_yticks(np.arange(len(glacier_names)))
    ax.set_xticklabels(_period_display_labels(periods), rotation=45, ha='right')
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
                ax.text(j, i, f'{value:.2f}', ha='center', va='center',
                        color='white' if abs(value) > vmax * 0.5 else 'black',
                        fontsize=9, fontweight='bold')

    plt.tight_layout()
    out_path = os.path.join(cfg.CENTERLINE_PLOT_DIR, "glacier_comparison_heatmap.png")
    plt.savefig(out_path, dpi=cfg.DPI, bbox_inches='tight')
    plt.close()
    print(f"   ✓ Saved: {os.path.basename(out_path)}")
