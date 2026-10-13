"""Map figures: dh and dh/dt maps, void-filled and mass-balance maps, filtering
and stable-terrain maps, and the 3x3 summary figure.
"""

import glob
import os
import re

import geopandas as gpd
import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rasterio

from .. import config as cfg
from ..inventory import (
    build_union_outline_gdf,
    compute_inventory_area_metrics_option_c,
    get_period_inventory_exclude_mask,
    get_period_inventory_outline_gdf,
)
from .centerline_geometry import get_centerline_plot_segment
from .common import add_north_arrow, add_scale_bar, format_coord_labels, hyps_bin_areas_km2
from ..stable_terrain import (
    build_paper_stable_terrain_domain,
    build_stable_dhdt_from_yearly_stable_dems,
)
from ..utils import (
    canonical_period_key,
    mask_year_for_label,
    summary_epochs,
    format_period_label,
    get_display_year,
    get_longest_linestring,
    load_rock_mask,
    parse_period,
    rasterize_geometry_to_grid_mask,
    save_raster,
)


def plot_spatial_maps(all_changes, grid_info, bad_mask, mb_results=None, shp_path=None,
                      union_shp_paths=None):
    """
    Create spatial maps of dh and dhdt with '/' formatted labels and shapefile
    outlines.
    """
    print("\n" + "=" * 50)
    print("▶ Creating spatial maps...")

    SPATIAL_MAP_DIR = os.path.join(cfg.PLOT_DIR, "spatial_maps")
    os.makedirs(SPATIAL_MAP_DIR, exist_ok=True)
    all_changes = sorted(all_changes, key=lambda c: float(parse_period(c['yA'])[0]))
    DHDT_VMIN = -5.0
    DHDT_VMAX = 5.0

    mb_dict = {f"{r['yA']}-{r['yB']}": r for r in mb_results} if mb_results else {}

    # See get_period_inventory_outline_gdf's docstring for why this is the
    # union footprint, not the mean-area scalar itself.
    fallback_outline_gdf = None
    if union_shp_paths:
        fallback_outline_gdf = build_union_outline_gdf(union_shp_paths, grid_info['crs'])
        if fallback_outline_gdf is None:
            print("      ⚠️  Could not build union outline; falling back to shp_path")
    if fallback_outline_gdf is None and shp_path and os.path.exists(shp_path):
        try:
            fallback_outline_gdf = gpd.read_file(shp_path).to_crs(grid_info['crs'])
        except Exception as e:
            print(f"      ⚠️ Could not load shapefile outline: {e}")

    all_plot_data = []

    for change in all_changes:
        yA, yB = change['yA'], change['yB']
        period_key = f"{yA}-{yB}"
        dh = change['dh']
        outline_gdf = get_period_inventory_outline_gdf(yA, yB, grid_info)
        if outline_gdf is None:
            outline_gdf = fallback_outline_gdf

        mb_res = mb_dict.get(period_key)
        if mb_res is not None and mb_res.get('dhdt_bias_corrected_raster') is not None:
            dhdt = mb_res['dhdt_bias_corrected_raster']
        else:
            print(f"      ⚠️ No bias-corrected dhdt available for {period_key}; "
                  f"falling back to filtered/uncorrected dhdt.")
            dhdt = change['dhdt']
        display_label = format_period_label(yA, yB)

        dh_masked = np.copy(dh)
        dhdt_masked = np.copy(dhdt)
        dh_masked[bad_mask] = np.nan
        dhdt_masked[bad_mask] = np.nan

        # See get_period_inventory_exclude_mask's docstring.
        period_exclude_mask = get_period_inventory_exclude_mask(outline_gdf, grid_info)
        if period_exclude_mask is not None:
            dh_masked[period_exclude_mask] = np.nan
            dhdt_masked[period_exclude_mask] = np.nan

        dh_valid = dh_masked[np.isfinite(dh_masked)]
        dhdt_valid = dhdt_masked[np.isfinite(dhdt_masked)]

        all_plot_data.append({
            'yA': yA, 'yB': yB, 'display_label': display_label,
            'dhdt': dhdt_masked, 'dh': dh_masked,
            'outline_gdf': outline_gdf,
            'stats': {
                'n': len(dhdt_valid), 'mean': np.mean(dhdt_valid),
                'median': np.median(dhdt_valid), 'std': np.std(dhdt_valid)
            }
        })

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))
        fig.subplots_adjust(bottom=0.15, hspace=0.3, wspace=0.3)

        extent = [
            grid_info['transform'].c,
            grid_info['transform'].c + grid_info['width'] * grid_info['transform'].a,
            grid_info['transform'].f + grid_info['height'] * grid_info['transform'].e,
            grid_info['transform'].f
        ]

        dh_vmax = max(abs(np.nanpercentile(dh_valid, 2)), abs(np.nanpercentile(dh_valid, 98)))

        im1 = ax1.imshow(dh_masked, cmap='RdBu', vmin=-dh_vmax, vmax=dh_vmax, extent=extent, origin='upper')
        cbar1 = plt.colorbar(im1, ax=ax1, orientation='horizontal', pad=0.12, shrink=0.8)
        cbar1.set_label(f'Elevation Change (m) [{display_label}]', fontsize=12)
        ax1.set_title(f"Elevation Change: {display_label}", fontsize=14, fontweight='bold')
        format_coord_labels(ax1)

        im2 = ax2.imshow(dhdt_masked, cmap='RdBu', vmin=DHDT_VMIN, vmax=DHDT_VMAX, extent=extent, origin='upper')
        cbar2 = plt.colorbar(im2, ax=ax2, orientation='horizontal', pad=0.12, shrink=0.8)
        cbar2.set_label(f'Elevation Change Rate mosaics (m a⁻¹) [{display_label}]', fontsize=12)
        ax2.set_title(f"Elevation Change Rate (dhdt): {display_label}", fontsize=14, fontweight='bold')
        format_coord_labels(ax2)

        # --- Shapefile Outline Overlay ---
        if outline_gdf is not None:
            try:
                outline_gdf.boundary.plot(ax=ax1, color='black', linewidth=1, alpha=0.8)
                outline_gdf.boundary.plot(ax=ax2, color='black', linewidth=1, alpha=0.8)
            except Exception as e:
                print(f"      ⚠️ Could not plot shapefile overlay: {e}")

        # Stats text box
        stats_text = f"Mean = {np.mean(dhdt_valid):.2f} m a⁻¹\nMedian = {np.median(dhdt_valid):.2f} m a⁻¹\nStd = {np.std(dhdt_valid):.2f} m a⁻¹"
        ax2.text(0.02, 0.98, stats_text, transform=ax2.transAxes, fontsize=11, verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

        filename_label = display_label.replace('/', '-')
        out_path = os.path.join(SPATIAL_MAP_DIR, f"spatial_map_{filename_label}_filtered.png")
        plt.savefig(out_path, dpi=cfg.DPI, bbox_inches='tight')
        plt.close()

        print(f"   ✓ Saved: {os.path.basename(out_path)}")

    # Combined 6-panel plot
    if len(all_plot_data) > 0:
        print(f"\n   Creating combined 6-panel dhdt map...")

        n_plots = min(3, len(all_plot_data))
        n_cols = 3
        n_rows = 1

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(24, 8))
        axes = axes.flatten()
        fig.subplots_adjust(top=0.85, bottom=0.08, hspace=0.25, wspace=0.15)

        extent = [
            grid_info['transform'].c,
            grid_info['transform'].c + grid_info['width'] * grid_info['transform'].a,
            grid_info['transform'].f + grid_info['height'] * grid_info['transform'].e,
            grid_info['transform'].f
        ]

        for idx in range(n_plots):
            ax = axes[idx]
            data = all_plot_data[idx]

            im = ax.imshow(
                data['dhdt'], cmap='RdBu', vmin=DHDT_VMIN, vmax=DHDT_VMAX,
                extent=extent, origin='upper', interpolation='nearest'
            )

            cbar = plt.colorbar(im, ax=ax, orientation='horizontal', pad=0.15, shrink=0.9)
            cbar.set_label('dhdt (m a⁻¹)', fontsize=10)
            cbar.ax.tick_params(labelsize=9)

            ax.set_title(data['display_label'], fontsize=12, fontweight='bold')
            ax.set_xlabel("Easting (m)", fontsize=10, labelpad=8)
            ax.set_ylabel("Northing (m)", fontsize=10)
            ax.ticklabel_format(style='plain', useOffset=False)
            ax.tick_params(axis='x', labelrotation=45, labelsize=8)
            ax.tick_params(axis='y', labelsize=8)

            for label in ax.get_xticklabels():
                label.set_horizontalalignment('right')
            format_coord_labels(ax)

            # Shapefile Outline for 6-panel -- period-specific (see
            # get_period_inventory_outline_gdf), stored per-panel above.
            panel_outline_gdf = data.get('outline_gdf')
            if panel_outline_gdf is not None:
                try:
                    panel_outline_gdf.boundary.plot(ax=ax, color='black', linewidth=1, alpha=0.8)
                except Exception as e:
                    print(f"      ⚠️ Could not plot shapefile overlay: {e}")

            stats = data['stats']
            stats_text = f"Mean = {stats['mean']:.2f} m a⁻¹\nMedian = {stats['median']:.2f} m a⁻¹\nStd = {stats['std']:.2f} m a⁻¹"
            ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, fontsize=9, verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

        for idx in range(n_plots, n_cols):
            axes[idx].axis('off')

        fig.suptitle('Elevation Change Rate mosaics',
                     fontsize=16, fontweight='bold', y=0.95)

        out_path = os.path.join(SPATIAL_MAP_DIR, "dhdt_map_all_periods_6panel.png")
        plt.savefig(out_path, dpi=cfg.DPI, bbox_inches='tight')
        plt.close()

        print(f"   ✓ Saved combined 6-panel plot: {os.path.basename(out_path)}")

    print("   ✓ Spatial maps complete")


def plot_extended_spatial_maps(all_changes, mb_results, grid_info, bad_mask,
                               density_A=cfg.DENSITY_A, density_B=cfg.DENSITY_B, shp_path=None,
                               union_shp_paths=None):
    """
    Create individual and multi-panel spatial maps for: 1. Void-filled dh/dt (m
    a⁻¹) 2. Filter difference |raw - filtered| (m a⁻¹) 3.
    """
    print("\n" + "=" * 50)
    print("▶ Creating extended single and multi-panel spatial maps...")

    EXTENDED_MAP_DIR = os.path.join(cfg.PLOT_DIR, "extended_spatial_maps")
    os.makedirs(EXTENDED_MAP_DIR, exist_ok=True)

    mb_dict = {f"{r['yA']}-{r['yB']}": r for r in mb_results}

    extent = [
        grid_info['transform'].c,
        grid_info['transform'].c + grid_info['width'] * grid_info['transform'].a,
        grid_info['transform'].f + grid_info['height'] * grid_info['transform'].e,
        grid_info['transform'].f
    ]

    fallback_boundary_geom = None
    if union_shp_paths:
        fallback_boundary_geom = build_union_outline_gdf(union_shp_paths, grid_info['crs'])
        if fallback_boundary_geom is None:
            print("      ⚠️  Could not build union outline; falling back to shp_path")
    if fallback_boundary_geom is None and shp_path and os.path.exists(shp_path):
        try:
            fallback_boundary_geom = gpd.read_file(shp_path).to_crs(grid_info['crs'])
        except Exception as e:
            print(f"      ⚠️ Could not load shapefile: {e}")

    # Standardize limits
    dhdt_lim = 5.0
    mwe_lim_A = dhdt_lim * density_A / cfg.RHO_WATER_KGM3
    mwe_lim_B = dhdt_lim * density_B / cfg.RHO_WATER_KGM3

    density_scenarios = [
        {
            'key': 'dmwe_850', 'density': density_A, 'suffix': '',
            'mb_dict_key': 'mass_balance_density850',
            'dMdt_key': 'inventory_mean_dMdt_Gtyr',
            'dMdt_unc_key': 'inventory_mean_dMdt_unc_combined_with_pen_Gtyr',
            'title': f'Specific Mass Balance (\u03c1={density_A:.0f} kg m\u207b\u00b3)',
            'vmax': mwe_lim_A,
        },
        {
            'key': 'dmwe_900', 'density': density_B, 'suffix': '_density900',
            'mb_dict_key': 'mass_balance_density900',
            'dMdt_key': 'inventory_mean_dMdt_Gtyr_density900',
            'dMdt_unc_key': 'inventory_mean_dMdt_unc_combined_with_pen_Gtyr_density900',
            'title': f'Specific Mass Balance (\u03c1={density_B:.0f} kg m\u207b\u00b3)',
            'vmax': mwe_lim_B,
        },
    ]

    # Dictionaries to gather data for the multi-panel plots
    gathered_data = {
        'dhdt_voi': {'title': 'Void-Filled dhdt', 'cb_label': 'dhdt (m a⁻¹)', 'cmap': 'RdBu', 'vmin': -dhdt_lim,
                     'vmax': dhdt_lim, 'panels': []},
        'filter_diff': {'title': 'Filter Difference |raw - filtered| (pre-correction)', 'cb_label': 'Difference (m a⁻¹)', 'cmap': 'magma',
                        'vmin': 0, 'vmax': None, 'panels': []},  # vmax computed later
    }
    for sc in density_scenarios:
        gathered_data[sc['key']] = {
            'title': sc['title'],
            'cb_label': 'Specific mass balance (m w.e. a⁻¹)',
            'cmap': 'RdBu', 'vmin': -sc['vmax'], 'vmax': sc['vmax'], 'panels': []
        }

    # ---------------------------------------------------------
    # 1. Process data and generate SINGLE maps
    # ---------------------------------------------------------
    for change in all_changes:
        yA, yB = change['yA'], change['yB']
        period_key = f"{yA}-{yB}"
        display_label = format_period_label(yA, yB)

        if period_key not in mb_dict:
            continue

        mb_res = mb_dict[period_key]

        period_outline_gdf = get_period_inventory_outline_gdf(yA, yB, grid_info)
        if period_outline_gdf is None:
            period_outline_gdf = fallback_boundary_geom

        dhdt_raw = change.get('dhdt_raw_observed', change.get('dhdt_raw'))
        dhdt_filtered = change.get('dhdt_filtered_3x3', change.get('dhdt'))
        filter_diff = np.abs(dhdt_raw - dhdt_filtered)
        filter_diff[bad_mask] = np.nan
        filter_diff_exclude_mask = get_period_inventory_exclude_mask(period_outline_gdf, grid_info)
        if filter_diff_exclude_mask is not None:
            filter_diff[filter_diff_exclude_mask] = np.nan

        dhdt_voi = mb_res.get('dhdt_voi_raster', np.full(grid_info['shape'], np.nan))

        diff_vals = filter_diff[np.isfinite(filter_diff)]
        diff_vmax = np.nanpercentile(diff_vals, 98) if diff_vals.size > 0 else 1.0

        period_data = {
            'dhdt_voi': dhdt_voi,
            'filter_diff': filter_diff,
        }

        # Area-normalised specific mass-balance rate (m w.e. a⁻¹) per density scenario,
        # each annotated with its region-integrated total (Gt a⁻¹ ± uncertainty)
        region_totals = {}
        for sc in density_scenarios:
            period_data[sc['key']] = dhdt_voi * sc['density'] / cfg.RHO_WATER_KGM3#rho_w

            mb_sub = mb_res.get(sc['mb_dict_key'], {})
            dMdt_Gtyr = mb_sub.get(sc['dMdt_key'])
            dMdt_unc_Gtyr = mb_sub.get(sc['dMdt_unc_key'])
            region_totals[sc['key']] = (dMdt_Gtyr, dMdt_unc_Gtyr)

        # Save single maps & gather for multi-panel
        for var_key, data in period_data.items():
            valid_data = data[np.isfinite(data)]
            if len(valid_data) == 0:
                continue

            var_units = gathered_data[var_key]['cb_label'].split('(')[-1].rstrip(')')

            stats_text = (f"Mean = {np.mean(valid_data):.2f} {var_units}\n"
                          f"Median = {np.median(valid_data):.2f} {var_units}\n"
                          f"Std = {np.std(valid_data):.2f} {var_units}")

            if var_key in region_totals:
                dMdt_Gtyr, dMdt_unc_Gtyr = region_totals[var_key]
                if dMdt_Gtyr is not None and np.isfinite(dMdt_Gtyr):
                    unc_str = f" ± {dMdt_unc_Gtyr:.2f}" if dMdt_unc_Gtyr is not None and np.isfinite(dMdt_unc_Gtyr) else ""
                    stats_text += f"\nRegion total = {dMdt_Gtyr:.2f}{unc_str} Gt a⁻¹"

            # Store for multi-panel
            gathered_data[var_key]['panels'].append({
                'label': display_label,
                'data': data,
                'stats_text': stats_text,
                'diff_vmax': diff_vmax if var_key == 'filter_diff' else None,
                'outline_gdf': period_outline_gdf,
            })

            # Create single plot
            fig, ax = plt.subplots(figsize=(10, 8))
            vmin = gathered_data[var_key]['vmin']
            vmax = diff_vmax if var_key == 'filter_diff' else gathered_data[var_key]['vmax']
            cmap = gathered_data[var_key]['cmap']

            im = ax.imshow(data, cmap=cmap, vmin=vmin, vmax=vmax, extent=extent, origin='upper')
            cbar = plt.colorbar(im, ax=ax, orientation='horizontal', pad=0.12, shrink=0.8)
            cbar.set_label(gathered_data[var_key]['cb_label'], fontsize=12)

            if period_outline_gdf is not None:
                period_outline_gdf.boundary.plot(ax=ax, color='black', linewidth=1, alpha=0.8)

            ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, fontsize=11,
                    verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

            ax.set_title(f"{gathered_data[var_key]['title']}: {display_label}", fontsize=14, fontweight='bold')
            format_coord_labels(ax)

            filename_label = display_label.replace('/', '-')
            out_path = os.path.join(EXTENDED_MAP_DIR, f"single_{var_key}_{filename_label}.png")
            plt.savefig(out_path, dpi=cfg.DPI, bbox_inches='tight')
            plt.close(fig)

    # ---------------------------------------------------------
    # 2. Generate MULTI-PANEL (1xN) maps
    # ---------------------------------------------------------
    for var_key, var_info in gathered_data.items():
        panels = var_info['panels']
        n_plots = len(panels)
        if n_plots == 0:
            continue

        fig, axes = plt.subplots(1, n_plots, figsize=(8 * n_plots, 8))
        if n_plots == 1:
            axes = [axes]

        fig.subplots_adjust(top=0.85, bottom=0.15, wspace=0.2)

        # Calculate global vmax for filter_diff if needed
        if var_key == 'filter_diff':
            global_vmax = max(p['diff_vmax'] for p in panels)
            vmin, vmax = 0, global_vmax
        else:
            vmin, vmax = var_info['vmin'], var_info['vmax']

        for idx, panel in enumerate(panels):
            ax = axes[idx]
            im = ax.imshow(panel['data'], cmap=var_info['cmap'], vmin=vmin, vmax=vmax,
                           extent=extent, origin='upper', interpolation='nearest')

            cbar = plt.colorbar(im, ax=ax, orientation='horizontal', pad=0.12, shrink=0.9)
            cbar.set_label(var_info['cb_label'], fontsize=12)

            panel_outline_gdf = panel.get('outline_gdf')
            if panel_outline_gdf is not None:
                panel_outline_gdf.boundary.plot(ax=ax, color='black', linewidth=1, alpha=0.8)

            ax.text(0.02, 0.98, panel['stats_text'], transform=ax.transAxes, fontsize=11,
                    verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

            ax.set_title(panel['label'], fontsize=16, fontweight='bold')
            ax.set_xlabel("Longitude", fontsize=14)
            ax.set_ylabel("Latitude", fontsize=14)
            format_coord_labels(ax)

        fig.suptitle(f"{var_info['title']} - All Periods", fontsize=20, fontweight='bold', y=0.98)

        out_path = os.path.join(EXTENDED_MAP_DIR, f"multipanel_{var_key}_all_periods.png")
        plt.savefig(out_path, dpi=cfg.DPI, bbox_inches='tight')
        plt.close(fig)
        print(f"   ✓ Saved multi-panel map: {os.path.basename(out_path)}")


def build_period_uncertainty_maps(
    change_dict,
    year_results,
    grid_info,
    preloaded_slope,
    preloaded_slope_bins,
    GLACIER_MASK_DIR,
    GLACIER_INVENTORY_MAP,
    DEFAULT_GLACIER_SHP,
    STABLE_TERRAIN_DIR
):
    yA = change_dict["yA"]
    yB = change_dict["yB"]

    dhdt_raw = change_dict.get("dhdt_raw", change_dict["dhdt"]).copy()
    dhdt_filtered = change_dict["dhdt"].copy()

    steep_mask = np.isfinite(preloaded_slope) & (preloaded_slope > 45.0)

    # Load the rock outcrop mask
    rock_mask = load_rock_mask(cfg.ROCK_OUTCROP_PATH, grid_info)

    # Inventory geometry
    inventory_area_stats = compute_inventory_area_metrics_option_c(
        yA, yB,
        GLACIER_MASK_DIR,
        GLACIER_INVENTORY_MAP,
        DEFAULT_GLACIER_SHP,
        target_metric_crs=cfg.CRS,
        area_mask_option=cfg.AREA_MASK_OPTION,
    )

    # same analysis domain as in calculate_mass_balance_following_r_logic
    union_geom = inventory_area_stats["support_geom"]
    union_inventory_mask = rasterize_geometry_to_grid_mask(union_geom, grid_info, invert=True)

    if cfg.AREA_MASK_OPTION == "B":
        demB_arr = change_dict.get("demB")
        if demB_arr is not None:
            union_inventory_mask = union_inventory_mask & np.isfinite(demB_arr)

    # Stable dh/dt
    stable_dhdt = build_stable_dhdt_from_yearly_stable_dems(
        yA=yA,
        yB=yB,
        year_results=year_results,
        grid_info=grid_info,
        STABLE_TERRAIN_DIR=STABLE_TERRAIN_DIR
    )

    if stable_dhdt is None:
        stable_dhdt = np.where(~union_inventory_mask, dhdt_raw, np.nan)

    stable_dhdt[steep_mask] = np.nan

    # Candidate search domain utilizing the rock mask strictly
    stable_domain_mask = build_paper_stable_terrain_domain(
        glacier_mask_bool=union_inventory_mask.astype(bool),
        rock_mask_bool=rock_mask,
        grid_info=grid_info,
        buffer_distance_m=10000,
        water_mask_bool=None
    )

    # Actual usable stable terrain entering uncertainty workflow
    usable_stable_mask = stable_domain_mask.astype(bool) & np.isfinite(stable_dhdt)

    filter_difference = np.full_like(dhdt_raw, np.nan, dtype=float)
    both = np.isfinite(dhdt_raw) & np.isfinite(dhdt_filtered)
    filter_difference[both] = np.abs(dhdt_raw[both] - dhdt_filtered[both])

    # Diagnostic table: usable stable pixels per slope bin
    df = pd.DataFrame({
        "slope_bin": preloaded_slope_bins.ravel(),
        "stable_domain": stable_domain_mask.ravel().astype(bool),
        "usable_stable": usable_stable_mask.ravel().astype(bool)
    })
    df = df[np.isfinite(df["slope_bin"])].copy()

    stable_count_by_slope = (
        df.groupby("slope_bin", as_index=False)["usable_stable"]
        .sum()
        .rename(columns={"usable_stable": "n_stable_pixels"})
        .sort_values("slope_bin")
        .reset_index(drop=True)
    )

    return {
        "yA": yA,
        "yB": yB,
        "period_label": format_period_label(yA, yB),
        "dhdt_raw": dhdt_raw,
        "dhdt_filtered": dhdt_filtered,
        "filter_difference": filter_difference,
        "stable_domain_mask": stable_domain_mask.astype(np.uint8),
        "usable_stable_mask": usable_stable_mask.astype(np.uint8),
        "stable_dhdt": stable_dhdt,
        "stable_count_by_slope": stable_count_by_slope,
        "steep_mask": steep_mask.astype(np.uint8)
    }


def plot_uncertainty_and_filtering_maps(
    all_changes,
    year_results,
    grid_info,
    preloaded_slope,
    preloaded_slope_bins,
    GLACIER_MASK_DIR,
    GLACIER_INVENTORY_MAP,
    DEFAULT_GLACIER_SHP,
    STABLE_TERRAIN_DIR
):
    """
    Create manuscript-consistent map plots for each period showing: (1)
    dh/dt_raw (observed support) (2) dh/dt_filtered (3x3 median) (3) |dh/dt_raw
    - dh/dt_filtered| (4) stable-terrain support used for uncertainty (10 km
    buffer ∩ rock mask ∩ finite stable dh/dt)
    """
    out_dir = os.path.join(cfg.PLOT_DIR, "uncertainty_maps")
    os.makedirs(out_dir, exist_ok=True)

    print("\n▶ Creating uncertainty and filtering maps...")

    extent = [
        grid_info['transform'].c,
        grid_info['transform'].c + grid_info['width'] * grid_info['transform'].a,
        grid_info['transform'].f + grid_info['height'] * grid_info['transform'].e,
        grid_info['transform'].f
    ]

    for change in all_changes:
        layers = build_period_uncertainty_maps(
            change_dict=change,
            year_results=year_results,
            grid_info=grid_info,
            preloaded_slope=preloaded_slope,
            preloaded_slope_bins=preloaded_slope_bins,
            GLACIER_MASK_DIR=GLACIER_MASK_DIR,
            GLACIER_INVENTORY_MAP=GLACIER_INVENTORY_MAP,
            DEFAULT_GLACIER_SHP=DEFAULT_GLACIER_SHP,
            STABLE_TERRAIN_DIR=STABLE_TERRAIN_DIR
        )

        yA = layers["yA"]
        yB = layers["yB"]
        period_label = layers["period_label"]

        dhdt_raw = layers["dhdt_raw"]
        dhdt_filtered = layers["dhdt_filtered"]
        filter_difference = layers["filter_difference"]
        stable_domain_mask = layers["stable_domain_mask"]
        usable_stable_mask = layers["usable_stable_mask"]

        fig, axes = plt.subplots(2, 2, figsize=(16, 13))
        axes = axes.flatten()

        # Shared dhdt scale
        raw_vals = dhdt_raw[np.isfinite(dhdt_raw)]
        fil_vals = dhdt_filtered[np.isfinite(dhdt_filtered)]

        if raw_vals.size + fil_vals.size > 0:
            dhdt_stack = np.concatenate([raw_vals, fil_vals])
            vmax = np.nanpercentile(np.abs(dhdt_stack), 98)
        else:
            vmax = 1.0

        if not np.isfinite(vmax) or vmax == 0:
            vmax = 1.0

        # 1) dh/dt_raw observed
        im0 = axes[0].imshow(
            dhdt_raw, cmap="RdBu", vmin=-vmax, vmax=vmax,
            extent=extent, origin="upper"
        )
        axes[0].set_title(f"dhdt_raw (observed)\n{period_label}", fontsize=14, fontweight="bold")
        format_coord_labels(axes[0])
        cbar0 = plt.colorbar(im0, ax=axes[0], orientation='horizontal', pad=0.08, shrink=0.85)
        cbar0.set_label("m a⁻¹$^{-1}$")

        # 2) dh/dt filtered 3x3
        im1 = axes[1].imshow(
            dhdt_filtered, cmap="RdBu", vmin=-vmax, vmax=vmax,
            extent=extent, origin="upper"
        )
        axes[1].set_title(f"dhdt_filtered (3×3 median)\n{period_label}", fontsize=14, fontweight="bold")
        format_coord_labels(axes[1])
        cbar1 = plt.colorbar(im1, ax=axes[1], orientation='horizontal', pad=0.08, shrink=0.85)
        cbar1.set_label("m a⁻¹r$^{-1}$")

        # 3) Filtering effect
        diff_vals = filter_difference[np.isfinite(filter_difference)]
        diff_vmax = np.nanpercentile(diff_vals, 98) if diff_vals.size > 0 else 1.0
        if not np.isfinite(diff_vmax) or diff_vmax == 0:
            diff_vmax = 1.0

        im2 = axes[2].imshow(
            filter_difference, cmap="magma", vmin=0, vmax=diff_vmax,
            extent=extent, origin="upper"
        )
        axes[2].set_title(r"|dhdt$_{raw}$ − dhdt$_{filtered}$|", fontsize=14, fontweight="bold")
        format_coord_labels(axes[2])
        cbar2 = plt.colorbar(im2, ax=axes[2], orientation='horizontal', pad=0.08, shrink=0.85)
        cbar2.set_label("m a⁻¹$^{-1}$")

        # 4) Usable stable terrain
        stable_vis = np.full(usable_stable_mask.shape, np.nan, dtype=float)
        stable_vis[usable_stable_mask == 1] = 1.0

        im3 = axes[3].imshow(
            stable_vis, cmap="Greens", vmin=0, vmax=1,
            extent=extent, origin="upper"
        )
        axes[3].set_title(
            "Stable-terrain support for uncertainty\n(10 km glacier buffer ∩ rock mask ∩ finite stable dhdt)",
            fontsize=12, fontweight="bold"
        )
        format_coord_labels(axes[3])
        cbar3 = plt.colorbar(im3, ax=axes[3], orientation='horizontal', pad=0.08, shrink=0.85)
        cbar3.set_label("usable stable terrain (1=yes)")

        raw_n = int(np.sum(np.isfinite(dhdt_raw)))
        fil_n = int(np.sum(np.isfinite(dhdt_filtered)))
        cand_n = int(np.sum(stable_domain_mask == 1))
        usable_n = int(np.sum(usable_stable_mask == 1))
        mean_diff = np.nanmean(filter_difference) if diff_vals.size > 0 else np.nan

        fig.suptitle(
            f"Filtering and uncertainty setup: {period_label}\n"
            f"N(raw)={raw_n:,}, N(filtered)={fil_n:,}, "
            f"N(candidate stable)={cand_n:,}, N(usable stable)={usable_n:,}, "
            f"mean |raw-filtered|={mean_diff:.3f} m a⁻¹$^{{-1}}$",
            fontsize=15,
            fontweight="bold",
            y=0.98
        )

        plt.tight_layout(rect=[0, 0, 1, 0.95])

        period_slug = canonical_period_key(period_label).replace("/", "-")
        out_png = os.path.join(out_dir, f"uncertainty_filtering_maps_{period_slug}.png")
        plt.savefig(out_png, dpi=cfg.DPI, bbox_inches='tight')
        plt.close()

        save_raster(
            filter_difference.astype(np.float32),
            os.path.join(out_dir, f"filter_difference_abs_raw_minus_filtered_{period_slug}.tif"),
            grid_info,
            units="m a-1",
            description=f"Absolute difference |raw - 3x3-median-filtered| dh/dt, {period_label}"
        )

        save_raster(
            stable_domain_mask.astype(np.float32),
            os.path.join(out_dir, f"stable_domain_candidate_10km_rock_{period_slug}.tif"),
            grid_info,
            units="boolean (1=candidate stable terrain, 0=excluded)",
            description=f"Candidate stable-terrain domain (10 km glacier buffer ∩ rock mask), {period_label}"
        )

        save_raster(
            usable_stable_mask.astype(np.float32),
            os.path.join(out_dir, f"stable_domain_usable_for_uncertainty_{period_slug}.tif"),
            grid_info,
            units="boolean (1=usable stable terrain, 0=excluded)",
            description=f"Usable stable-terrain domain for sigma_stable estimation, {period_label}"
        )

        layers["stable_count_by_slope"].to_csv(
            os.path.join(out_dir, f"stable_pixels_by_5deg_slope_bins_{period_slug}.csv"),
            index=False
        )

        print(f"   ✓ Saved: {os.path.basename(out_png)}")


def plot_comprehensive_summary(year_results, all_changes, mb_results, grid_info, bad_mask, shp_path=None,
                               union_shp_paths=None):
    """Create two comprehensive summary figures:"""
    print("\n▶ Creating comprehensive summary plots (maps + hypsometry)...")

    target_years = summary_epochs(year_results.keys())
    if target_years is None:
        print("   ⚠️  Summary figures need three epochs, skipping")
        return
    e0, e1, e2 = target_years
    by_pair = {(c['yA'], c['yB']): c for c in all_changes}
    target_periods = [by_pair[p] for p in [(e0, e1), (e1, e2), (e0, e2)] if p in by_pair]

    extent = [
        grid_info['transform'].c,
        grid_info['transform'].c + grid_info['width'] * grid_info['transform'].a,
        grid_info['transform'].f + grid_info['height'] * grid_info['transform'].e,
        grid_info['transform'].f
    ]

    boundary_geom = None
    if union_shp_paths:
        boundary_geom = build_union_outline_gdf(union_shp_paths, grid_info['crs'])
        if boundary_geom is None:
            print("      ⚠️  Could not build union outline; falling back to shp_path")
    if boundary_geom is None and shp_path and os.path.exists(shp_path):
        try:
            boundary_geom = gpd.read_file(shp_path).to_crs(grid_info['crs'])
        except Exception as e:
            print(f"      ⚠️ Could not load shapefile for comprehensive plot: {e}")

    mb_dict = {f"{r['yA']}-{r['yB']}": r for r in mb_results}

    # Load centerlines once for the Row-2 dh/dt maps. Names/numbers are derived
    # exactly the way create_combined_dem_and_dhdt_9panel /
    # add_satellite_basemap_panel do it (glacier_name_to_number is built from
    # the same unsorted glob() order), so the numbers labelled here match the
    # glacier numbers used in the 9-panel figure's panel titles ("(b) Glacier
    # Name [N]") for the same script run.
    centerline_records = []  # list of (glacier_name, glacier_number, geom)
    if cfg.CENTERLINE_DIR and os.path.exists(cfg.CENTERLINE_DIR):
        centerline_files = glob.glob(os.path.join(cfg.CENTERLINE_DIR, "*.shp"))
        for glacier_idx, centerline_file in enumerate(centerline_files, start=1):
            centerline_name = os.path.splitext(os.path.basename(centerline_file))[0]
            glacier_name = centerline_name.replace('Centerline_', '')
            glacier_name = re.sub(r'_\d{2}$', '', glacier_name).replace('_', ' ')
            try:
                cl_gdf = gpd.read_file(centerline_file).to_crs(grid_info['crs'])
                cl_geom = get_longest_linestring(cl_gdf.unary_union)
            except Exception as e:
                print(f"      ⚠️ Could not load centerline {centerline_name}: {e}")
                continue
            if cl_geom is not None:
                centerline_records.append((glacier_name, glacier_idx, cl_geom))
        print(f"      Loaded {len(centerline_records)} centerlines for Row 2 dhdt maps")
    else:
        print("      ⚠️ Centerline directory not found; Row 2 dhdt maps will show no centerlines.")

    # Figure 1: Rows 1-3 (DEMs | filtered dhdt | void-filled dhdt) Each map
    # uses imshow's default equal-aspect behaviour, so if a gridspec cell's own
    # aspect ratio doesn't match the raster's true (height/width) aspect ratio,
    # matplotlib shrinks the axes box to fit inside the cell -- "letterboxing"
    # -- leaving blank margins above and below every panel.
    FIG_WIDTH_IN     = 20.0
    LEFT, RIGHT      = 0.05, 0.90
    WSPACE           = 0.05
    NCOLS, NROWS     = 3, 3
    TOP_MARGIN_IN    = 0.45   # room for row 1's own title, above the whole figure
    TITLE_GAP_IN     = 0.45   # room for each row 2/3 title, between rows
    BOTTOM_MARGIN_IN = 0.55   # room for row 3's "Longitude" label + tick labels

    data_aspect = grid_info['shape'][0] / grid_info['shape'][1]  # height / width

    col_width_frac = (RIGHT - LEFT) / (NCOLS + (NCOLS - 1) * WSPACE)
    col_width_in   = col_width_frac * FIG_WIDTH_IN
    row_content_in = col_width_in * data_aspect  # map height alone, no title

    FIG_HEIGHT_IN = (NROWS * row_content_in
                      + (NROWS - 1) * TITLE_GAP_IN
                      + TOP_MARGIN_IN + BOTTOM_MARGIN_IN)

    TOP    = 1 - TOP_MARGIN_IN / FIG_HEIGHT_IN
    BOTTOM = BOTTOM_MARGIN_IN / FIG_HEIGHT_IN
    HSPACE = TITLE_GAP_IN / row_content_in

    fig1 = plt.figure(figsize=(FIG_WIDTH_IN, FIG_HEIGHT_IN))
    # Right margin narrowed (0.95 -> 0.90) to make room for one shared,
    # vertical, row-height colorbar placed just right of each row's last map,
    # instead of one horizontal colorbar under every individual panel.
    gs1  = fig1.add_gridspec(NROWS, NCOLS, top=TOP, bottom=BOTTOM,
                              left=LEFT, right=RIGHT,
                              hspace=HSPACE, wspace=WSPACE)

    ROW_CBAR_GAP   = 0.014   # gap between the last map and the colorbar (figure fraction)
    ROW_CBAR_WIDTH = 0.018   # width of the colorbar axis (figure fraction)

    def add_row_colorbar(fig, im, ref_ax, label):
        """One vertical colorbar spanning ref_ax's full height, placed just
        right of the row's last (rightmost) map (all panels in a row share
        vmin/vmax/cmap, so any one panel's mappable represents the row)."""
        if im is None or ref_ax is None:
            return None
        pos = ref_ax.get_position()
        cax = fig.add_axes([pos.x1 + ROW_CBAR_GAP, pos.y0, ROW_CBAR_WIDTH, pos.height])
        cbar = fig.colorbar(im, cax=cax, orientation='vertical')
        cbar.set_label(label, fontsize=12)
        # Ticks/label default to the colorbar's right side, which is
        # correct here since it sits to the right of the last map.
        return cbar

    # Falls back to boundary_geom if a year-specific outline can't be built.
    row1_outline_cache = {}

    def get_row1_outline_gdf(year_label):
        mask_year = mask_year_for_label(year_label)
        if mask_year in row1_outline_cache:
            return row1_outline_cache[mask_year]
        mask_path = cfg.YEAR_MASKS.get(mask_year)
        gdf = None
        if mask_path and os.path.exists(mask_path):
            gdf = build_union_outline_gdf([mask_path], grid_info['crs'])
        else:
            print(f"      ⚠️  Row 1: no usable YEAR_MASKS path for year={year_label!r} "
                  f"(resolved mask_year={mask_year!r}, path={mask_path!r})")
        row1_outline_cache[mask_year] = gdf
        return gdf

    row1_im, row1_ax_last = None, None
    for col, year in enumerate(target_years):
        ax = fig1.add_subplot(gs1[0, col])
        letter = chr(97 + col)   # a, b, c
        if col == len(target_years) - 1:
            row1_ax_last = ax

        row1_boundary_geom = get_row1_outline_gdf(year)
        if row1_boundary_geom is None:
            row1_boundary_geom = boundary_geom

        if year in year_results:
            dem_file = year_results[year]['dem_file']
            with rasterio.open(dem_file) as src:
                dem_data = src.read(1)
                dem_data[dem_data == src.nodata] = np.nan

            dem_data[bad_mask] = np.nan

            # Restrict the DISPLAYED DEM to this panel's own year-specific
            # outline (not just draw the outline on top of the wider union-
            # clipped data) -- same "exclude outside this outline" treatment
            # rows 2/3 already apply to their dh/dt rasters via
            # get_period_inventory_exclude_mask, reused here with
            # row1_boundary_geom instead of a period outline.
            row1_exclude_mask = get_period_inventory_exclude_mask(row1_boundary_geom, grid_info)
            if row1_exclude_mask is not None:
                dem_data[row1_exclude_mask] = np.nan

            #vmin = np.nanpercentile(dem_data, 2)
            #vmax = np.nanpercentile(dem_data, 98)

            vmin = -30
            vmax = 2200

            im = ax.imshow(dem_data, cmap='bone', vmin=vmin, vmax=vmax,
                           extent=extent, origin='upper')
            row1_im = im

            if row1_boundary_geom is not None:
                row1_boundary_geom.boundary.plot(ax=ax, color='black', linewidth=1, alpha=0.8)

            ax.set_title(f"({letter}) DEM {get_display_year(year)}", fontsize=15, fontweight='bold', pad=4)
            format_coord_labels(ax)
            ax.set_xlabel("")
            ax.set_xticklabels([])
            if col > 0:
                ax.set_ylabel("")
                ax.set_yticklabels([])

            add_north_arrow(ax, grid_info, location='upper right',
                             frac_inset_x=0.025, frac_inset_y=0.07,
                             frac_length=0.06, fontsize=8,
                             color='black', outline_color='white', style='solid')
            add_scale_bar(ax, location='lower right',
                          color='black', outline_color='white',
                          fixed_length_km=25, frac_inset=0.035)
        else:
            ax.axis('off')
            ax.set_title(f"({letter}) DEM {get_display_year(year)} (Not Available)")

    add_row_colorbar(fig1, row1_im, row1_ax_last, 'Elevation (m a.s.l.)')

    # --- Row 2 & 3: Filtered dh/dt and Void-filled dh/dt ---
    row2_im, row2_ax_last = None, None
    row3_im, row3_ax_last = None, None
    for col, change in enumerate(target_periods):
        yA, yB = change['yA'], change['yB']
        period_key   = f"{yA}-{yB}"
        display_label = format_period_label(yA, yB)

        period_outline_gdf = get_period_inventory_outline_gdf(yA, yB, grid_info)
        if period_outline_gdf is None:
            period_outline_gdf = boundary_geom

        ax_fil = fig1.add_subplot(gs1[1, col])
        letter_fil = chr(97 + 3 + col)   # d, e, f
        if col == len(target_periods) - 1:
            row2_ax_last = ax_fil

        mb_res_row2 = mb_dict.get(period_key)
        if mb_res_row2 is not None and mb_res_row2.get('dhdt_bias_corrected_raster') is not None:
            dhdt_filtered = mb_res_row2['dhdt_bias_corrected_raster'].copy()
        else:
            print(f"      ⚠️ No bias-corrected dhdt available for {period_key} (Row 2); "
                  f"falling back to filtered/uncorrected dhdt.")
            dhdt_filtered = change.get('dhdt_filtered_3x3', change.get('dhdt')).copy()
        dhdt_filtered[bad_mask] = np.nan
        # Row 3 (void-filled, below) doesn't need this: it's already masked
        # with voi_fill_mask, the same geometry as period_outline_gdf.
        row2_exclude_mask = get_period_inventory_exclude_mask(period_outline_gdf, grid_info)
        if row2_exclude_mask is not None:
            dhdt_filtered[row2_exclude_mask] = np.nan

        im_fil = ax_fil.imshow(dhdt_filtered, cmap='RdBu', vmin=-5, vmax=5,
                               extent=extent, origin='upper')
        row2_im = im_fil

        if period_outline_gdf is not None:
            period_outline_gdf.boundary.plot(ax=ax_fil, color='black', linewidth=1, alpha=0.8)

        # Centerlines + numbered labels, drawn in black with a white halo
        # (matches the styling used for centerlines over the RdBu dh/dt
        # background elsewhere in the script, e.g. plot_centerline_
        # elevation_changes) so they stay legible against both the red and blue
        # ends of the colormap.
        label_offset = (extent[1] - extent[0]) * 0.04  # ~2% of map width
        for g_name, g_num, cl_geom in centerline_records:
            cl_geom_plot = get_centerline_plot_segment(cl_geom)
            ax_fil.plot(*cl_geom_plot.xy, color='maroon', linewidth=1.0, zorder=10, alpha=0.9,
                        path_effects=[pe.withStroke(linewidth=1.5, foreground='white')])
            end_point = cl_geom_plot.interpolate(1.0, normalized=True)
            label_x = end_point.x + label_offset
            label_y = end_point.y
            ax_fil.plot([end_point.x, label_x], [end_point.y, label_y],
                        color='black', linewidth=0.5, zorder=14, alpha=0.9,
                        path_effects=[pe.withStroke(linewidth=0.75, foreground='white')])
            ax_fil.text(label_x, label_y, str(g_num),
                        fontsize=11, fontweight='bold', color='black',
                        ha='left', va='center', zorder=15,
                        path_effects=[pe.withStroke(linewidth=2.0, foreground='white')],
                        bbox=dict(boxstyle='circle,pad=0.18', facecolor='none',
                                  edgecolor='black', linewidth=1.0))

        ax_fil.set_title(f"({letter_fil}) dhdt: {display_label}",
                          fontsize=13, fontweight='bold', pad=4)
        format_coord_labels(ax_fil)
        ax_fil.set_xlabel("")
        ax_fil.set_xticklabels([])
        if col > 0:
            ax_fil.set_ylabel("")
            ax_fil.set_yticklabels([])

        add_north_arrow(ax_fil, grid_info, location='upper right',
                         frac_inset_x=0.025, frac_inset_y=0.08,
                         frac_length=0.06, fontsize=8,
                         color='black', outline_color='white', style='solid')
        add_scale_bar(ax_fil, location='lower right',
                      color='black', outline_color='white',
                      fixed_length_km=25, frac_inset=0.035)

        # Row 3: Void-filled dh/dt
        ax_voi = fig1.add_subplot(gs1[2, col])
        letter_voi = chr(97 + 6 + col)   # g, h, i
        if col == len(target_periods) - 1:
            row3_ax_last = ax_voi

        if period_key in mb_dict:
            dhdt_voi = mb_dict[period_key].get('dhdt_voi_raster',
                                               np.full(grid_info['shape'], np.nan))
            im_voi = ax_voi.imshow(dhdt_voi, cmap='RdBu', vmin=-5, vmax=5,
                                   extent=extent, origin='upper')
            row3_im = im_voi

            if period_outline_gdf is not None:
                period_outline_gdf.boundary.plot(ax=ax_voi, color='black', linewidth=1, alpha=0.8)

            ax_voi.set_title(f"({letter_voi}) Void-Filled dhdt: {display_label}",
                             fontsize=13, fontweight='bold', pad=4)
            format_coord_labels(ax_voi)
            if col > 0:
                ax_voi.set_ylabel("")
                ax_voi.set_yticklabels([])

            add_north_arrow(ax_voi, grid_info, location='upper right',
                             frac_inset_x=0.025, frac_inset_y=0.08,
                             frac_length=0.06, fontsize=8,
                             color='black', outline_color='white', style='solid')
            add_scale_bar(ax_voi, location='lower right',
                          color='black', outline_color='white',
                          fixed_length_km=25, frac_inset=0.035)
        else:
            ax_voi.axis('off')

    add_row_colorbar(fig1, row2_im, row2_ax_last, 'dhdt (m a⁻¹)')
    add_row_colorbar(fig1, row3_im, row3_ax_last, 'Void-Filled dhdt (m a⁻¹)')

    out_path1 = os.path.join(cfg.PLOT_DIR, "comprehensive_summary_maps_3x3.png")
    plt.savefig(out_path1, dpi=300, bbox_inches='tight')
    plt.close(fig1)
    print(f"   ✓ Saved maps summary plot: {os.path.basename(out_path1)}")

    # =========================================================================
    # Figure 2: Row 4  (hypsometric mass-balance plots)
    # =========================================================================

    # Grey bars: true catchment hypsometry (reference DEM over the full
    # inventory union) -- identical for every period, so take it from any
    # result that carries it.
    global_x = None
    global_area_ref_km2 = None
    for _res in mb_dict.values():
        _ref = _res.get("ref_hyps_bins")
        if _ref is not None and len(_ref):
            _ref = _ref.sort_values("cl")
            global_x = _ref["cl"].values.astype(float)
            global_area_ref_km2 = _ref["area_km2"].values.astype(float)
            break
    if global_x is None:
        print("   WARNING: no 'ref_hyps_bins' in mb results -- re-run the "
              "mass-balance step; reference-area bars omitted.")

    max_area = np.nanmax(global_area_ref_km2) * 1.1 \
        if global_area_ref_km2 is not None else 1000

    fig2 = plt.figure(figsize=(20, 7))
    gs2  = fig2.add_gridspec(1, 3, top=0.88, bottom=0.18,
                              left=0.05, right=0.95, wspace=0.22)

    legend_handles = []
    legend_labels  = []

    for col, change in enumerate(target_periods):
        ax_hyp = fig2.add_subplot(gs2[0, col])
        letter_hyp = chr(97 + col)   # a, b, c (fresh lettering for stand-alone figure)

        yA, yB = change['yA'], change['yB']
        period_key = f"{yA}-{yB}"

        if period_key in mb_dict:
            res   = mb_dict[period_key]
            stats = res.get("onice_bin_stats", None)

            if stats is not None and not stats.empty:
                stats = stats.sort_values("cl").reset_index(drop=True)
                x = stats["cl"].values.astype(float)

                n_ref     = stats["n_ref"].fillna(0).astype(float).values
                n_ref_sum = np.nansum(n_ref)

                if n_ref_sum > 0:
                    # Blue bars: pixels with a finite dh/dt value this period.
                    area_obs_km2 = hyps_bin_areas_km2(res, stats)["obs"]

                    y_fil     = stats["mean_ch_fil"].values.astype(float)
                    y_fil_std = stats["sdev_ch_fil"].fillna(0).values.astype(float)

                    finite_fil = np.isfinite(y_fil)

                    bin_width = (x[1] - x[0]) * 0.88 if len(x) > 1 else 40.0

                    bar_ref = None
                    if global_x is not None and global_area_ref_km2 is not None:
                        bar_ref = ax_hyp.bar(global_x, global_area_ref_km2, width=bin_width,
                                              color="#e0e0e0", alpha=0.95,
                                              label="Catchment area per elevation bin (reference DEM)")

                    bar_fil = ax_hyp.bar(x, area_obs_km2, width=bin_width * 0.46,
                                          color="#4f79d8", alpha=0.95,
                                          label="Observed area (dh/dt available)")

                    ax_hyp.set_xlabel("Elevation (m a.s.l.)", fontsize=11)
                    ax_hyp.set_ylabel("Area (km²)", fontsize=11, color="#2f6fb6")
                    ax_hyp.tick_params(axis="y", labelcolor="#2f6fb6")
                    ax_hyp.grid(True, axis="y", linestyle="--", alpha=0.25)
                    ax_hyp.set_ylim(0, max_area)
                    if cfg.HYPS_SUMMARY_XLIM:
                        ax_hyp.set_xlim(*cfg.HYPS_SUMMARY_XLIM)

                    ax_hyp2 = ax_hyp.twinx()
                    ax_hyp2.axhline(0, color="0.45", lw=1.5, ls="--")

                    li_fil = None
                    if np.any(finite_fil):
                        li_fil = ax_hyp2.errorbar(
                            x[finite_fil], y_fil[finite_fil],
                            yerr=y_fil_std[finite_fil],
                            fmt="o-", color="crimson", ecolor="crimson",
                            ms=3.5, lw=1.2, elinewidth=1, capsize=2,
                            label="dhdt (bin mean) ±1σ", zorder=2, alpha=0.9)

                    ax_hyp2.set_ylabel("Elevation change rate (m a⁻¹)", fontsize=11,
                                       color="#8b0000")
                    ax_hyp2.tick_params(axis="y", labelcolor="#8b0000")
                    ax_hyp2.set_ylim(-5, 5)

                    ax_hyp.set_title(
                        f"({letter_hyp}) {format_period_label(yA, yB)}",
                        fontsize=13, fontweight='bold')

                    if col == 0:
                        legend_handles = [h for h in (bar_ref, bar_fil, li_fil) if h is not None]
                        legend_labels  = [l for h, l in zip(
                            (bar_ref, bar_fil, li_fil),
                            ("Catchment area per elevation bin (reference DEM)",
                             "Observed area (dh/dt available)",
                             "dhdt (bin mean) ±1σ")
                        ) if h is not None]
        else:
            ax_hyp.axis('off')

    if legend_handles:
        fig2.legend(legend_handles, legend_labels,
                    loc='lower center', ncol=6,
                    bbox_to_anchor=(0.5, 0.01),
                    frameon=False, fontsize=11)

    out_path2 = os.path.join(cfg.PLOT_DIR, "comprehensive_summary_hypsometry_1x3.png")
    plt.savefig(out_path2, dpi=300, bbox_inches='tight')
    plt.close(fig2)
    print(f"   ✓ Saved hypsometry summary plot: {os.path.basename(out_path2)}")
