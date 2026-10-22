"""Centerline figure: satellite map with the numbered centerlines and, for up
to eight glaciers, surface elevation and dh/dt profiles (3 x 3 panels)."""

import os

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

from .. import config as cfg
from .common import (
    _left_align_legend_title,
    _match_legend_panel_to_map,
    _measure_map_target_aspect,
    add_satellite_basemap_panel,
    terminus_color,
)
from ..utils import get_display_year

DEM_COLORS = ['royalblue', 'mediumorchid', 'hotpink', 'orchid']        # surface elevation per epoch
DHDT_COLORS = ['darkorange', 'crimson', 'peru', 'saddlebrown']         # dh/dt per period


def render_glacier_dem_dhdt_panel(ax_dem, ax_dhdt, glacier_name, glacier_number, letter,
                                  b_data, dhdt_info, dem_colors, dhdt_colors,
                                  elevation_ylabel="Elevation (m)",
                                  title_fontsize=16, label_fontsize=13,
                                  tick_labelsize=12,
                                  dhdt_ylim=None, elev_ylim=(-20.0, 500.0),
                                  shared_xlim=None):
    """One glacier: surface elevations (top) and dh/dt (bottom) along the centerline."""
    surface_data = b_data.get('surface_data', [])

    for s_idx, (year, distances, values) in enumerate(surface_data):
        color = dem_colors[s_idx % len(dem_colors)]
        ax_dem.plot(distances / 1000.0, values, color=color, linewidth=3.0,
                    label=f"Elev {get_display_year(year)}", alpha=0.9)

    label_y = elev_ylim[1] - 0.04 * (elev_ylim[1] - elev_ylim[0])

    terminus_positions = b_data.get('terminus_positions', {})
    for mask_year in sorted(terminus_positions.keys()):
        pos_km = terminus_positions[mask_year] / 1000.0
        t_col = terminus_color(mask_year)
        ax_dem.axvline(pos_km, color=t_col, linestyle='--', alpha=0.7, lw=1.5)
        ax_dhdt.axvline(pos_km, color=t_col, linestyle='--', alpha=0.7, lw=1.5)
        ax_dem.text(pos_km, label_y, mask_year, rotation=90, va='top', ha='right', fontsize=10, color=t_col,
                    bbox=dict(boxstyle='round,pad=0.1', facecolor='white', alpha=0.7, edgecolor='none'))

    ax_dem.set_ylabel(elevation_ylabel, fontsize=label_fontsize)
    ax_dem.set_title(f"({letter}) {glacier_name} [{glacier_number}]", fontsize=title_fontsize, fontweight='bold')
    ax_dem.grid(True, linestyle='--', alpha=0.4)
    ax_dem.set_ylim(*elev_ylim)
    ax_dem.tick_params(axis='both', labelsize=tick_labelsize)
    plt.setp(ax_dem.get_xticklabels(), visible=False)

    for p_idx, (period_key, p_data) in enumerate(dhdt_info.items()):
        dist_km = p_data['distances_km']
        dhdt_vals = p_data['dhdt']
        color = dhdt_colors[p_idx % len(dhdt_colors)]
        if len(dist_km) == len(dhdt_vals) and len(dist_km) > 0:
            ax_dhdt.plot(dist_km, dhdt_vals, color=color, linestyle='-', linewidth=2.5, label=f"dhdt {period_key}")

    ax_dhdt.axhline(0, color='black', linewidth=1, linestyle='-', alpha=0.5)
    ax_dhdt.set_ylabel("dhdt (m a⁻¹)", fontsize=label_fontsize)
    ax_dhdt.set_xlabel("Distance (km)", fontsize=label_fontsize)
    ax_dhdt.grid(True, linestyle='--', alpha=0.4)
    if shared_xlim is not None:
        _xmin_km, _xmax_km = shared_xlim
    else:
        _gmin = b_data.get('global_min_dist', 0.0)
        _gmax = b_data.get('global_max_dist', cfg.CENTERLINE_MAX_DISTANCE)
        _xmin_km = max(0.0, _gmin / 1000.0)
        _xmax_km = min(cfg.CENTERLINE_MAX_DISTANCE / 1000.0, _gmax / 1000.0)
    if _xmax_km > _xmin_km:
        ax_dhdt.set_xlim(_xmin_km, _xmax_km)
    else:
        ax_dhdt.set_xlim(0, cfg.CENTERLINE_MAX_DISTANCE / 1000.0)
    ax_dhdt.tick_params(axis='both', labelsize=tick_labelsize)
    if dhdt_ylim is not None:
        ax_dhdt.set_ylim(*dhdt_ylim)


def plot_centerline_profiles_figure(glacier_profiles, all_glacier_data, grid_info,
                                    dhdt_ylim_lower=-12, dhdt_ylim_upper=18,
                                    elev_ylim=(-20.0, 500.0)):
    """
    Panel (a): satellite basemap with the numbered centerlines and the legend;
    panels (b)-(i): surface elevation and dh/dt profiles of up to eight glaciers
    on shared axes. Writes combined_9panel_dem_and_dhdt_profiles_satellite.png.
    """
    if not glacier_profiles:
        print("      No data available for the centerline figure")
        return
    print("\n▶ Creating centerline profile figure...")

    sorted_glaciers = sorted(glacier_profiles.items(), key=lambda x: x[1].get('glacier_number', 999))
    grid_positions = [(0, 1), (0, 2), (1, 0), (1, 1), (1, 2), (2, 0), (2, 1), (2, 2)]

    # shared dh/dt range for all panels (explicit limits, else data percentiles)
    _all_dhdt_vals = []
    for _gname, _ in sorted_glaciers[:8]:
        for _p_data in all_glacier_data.get(_gname, {}).values():
            _vals = np.asarray(_p_data.get('dhdt', []))
            _vals = _vals[np.isfinite(_vals)]
            if _vals.size > 0:
                _all_dhdt_vals.append(_vals)
    if _all_dhdt_vals:
        _dhdt_concat = np.concatenate(_all_dhdt_vals)
        _dhdt_lo, _dhdt_hi = np.nanpercentile(_dhdt_concat, [0.5, 99.5])
        _dhdt_pad = 0.08 * (_dhdt_hi - _dhdt_lo) if _dhdt_hi > _dhdt_lo else 1.0
        _final_lo = dhdt_ylim_lower if dhdt_ylim_lower is not None else (_dhdt_lo - _dhdt_pad)
        _final_hi = dhdt_ylim_upper if dhdt_ylim_upper is not None else (_dhdt_hi + _dhdt_pad)
        shared_dhdt_ylim = (_final_lo, _final_hi)
    else:
        shared_dhdt_ylim = (dhdt_ylim_lower, dhdt_ylim_upper) if (dhdt_ylim_lower is not None and dhdt_ylim_upper is not None) else None

    # shared distance axis for all panels, so distances are comparable
    _all_min_dist = [
        b_data['global_min_dist'] for _, b_data in sorted_glaciers[:8]
        if b_data.get('global_min_dist') is not None
    ]
    _all_max_dist = [
        b_data['global_max_dist'] for _, b_data in sorted_glaciers[:8]
        if b_data.get('global_max_dist') is not None
    ]
    if _all_min_dist and _all_max_dist:
        shared_xmin_km = max(0.0, min(_all_min_dist) / 1000.0)
        shared_xmax_km = min(cfg.CENTERLINE_MAX_DISTANCE / 1000.0, max(_all_max_dist) / 1000.0)
        if shared_xmax_km <= shared_xmin_km:
            shared_xmin_km, shared_xmax_km = 0.0, cfg.CENTERLINE_MAX_DISTANCE / 1000.0
    else:
        shared_xmin_km, shared_xmax_km = 0.0, cfg.CENTERLINE_MAX_DISTANCE / 1000.0

    dem_colors, dhdt_colors = DEM_COLORS, DHDT_COLORS

    # legend entries from the first glacier: one per surface epoch / per plotted period
    first_name, first = sorted_glaciers[0]
    h_dem_global = [Line2D([], [], color=dem_colors[i % len(dem_colors)], linewidth=2.5, alpha=0.9)
                    for i in range(len(first.get('surface_data', [])))]
    l_dem_clean_global = [get_display_year(year) for year, _, _ in first.get('surface_data', [])]
    h_dhdt_global, l_dhdt_clean_global = [], []
    for p_idx, p_data in enumerate(all_glacier_data.get(first_name, {}).values()):
        if len(p_data['distances_km']) == len(p_data['dhdt']) and len(p_data['distances_km']) > 0:
            h_dhdt_global.append(Line2D([], [], color=dhdt_colors[p_idx % len(dhdt_colors)], linewidth=2))
            l_dhdt_clean_global.append(p_data['period_display'].replace('-', ' - '))

    fig = plt.figure(figsize=(26, 18), constrained_layout=True)
    gs_main = fig.add_gridspec(3, 3)

    # panel (a): map above a legend row
    map_h_ratio, leg_h_ratio = 4.4, 1.1
    gs_map_leg = gs_main[0, 0].subgridspec(2, 1, height_ratios=[map_h_ratio, leg_h_ratio],
                                              hspace=0.02)
    ax_map = fig.add_subplot(gs_map_leg[0, 0])
    ax_leg = fig.add_subplot(gs_map_leg[1, 0])

    # glacier panels first: the map extent is fitted to the final layout below
    for idx, (glacier_name, b_data) in enumerate(sorted_glaciers[:8]):
        pos = grid_positions[idx]
        gs_inner = gs_main[pos].subgridspec(2, 1, height_ratios=[2, 1])
        ax_dem = fig.add_subplot(gs_inner[0])
        ax_dhdt = fig.add_subplot(gs_inner[1], sharex=ax_dem)

        letter = chr(97 + idx + 1)
        render_glacier_dem_dhdt_panel(
            ax_dem, ax_dhdt, glacier_name, b_data.get('glacier_number', '?'), letter,
            b_data, all_glacier_data.get(glacier_name, {}),
            dem_colors, dhdt_colors,
            elevation_ylabel="Elevation (m.a.s.l.)",
            dhdt_ylim=shared_dhdt_ylim, elev_ylim=elev_ylim,
            shared_xlim=(shared_xmin_km, shared_xmax_km)
        )

    target_aspect = _measure_map_target_aspect(fig, ax_map, 26.0, 18.0)
    add_satellite_basemap_panel(ax_map, grid_info, glacier_profiles,
                                 target_aspect=target_aspect)

    # one legend frame: surface elevations (left) and dh/dt periods (right)
    ax_leg.set_xticks([])
    ax_leg.set_yticks([])
    for spine in ax_leg.spines.values():
        spine.set_visible(True)
        spine.set_edgecolor('black')
        spine.set_linewidth(1.1)
    ax_leg.set_facecolor('white')

    if h_dem_global:
        leg_e = ax_leg.legend(
            h_dem_global, l_dem_clean_global,
            loc='center left', bbox_to_anchor=(0.06, 0.5), fontsize=12,
            title="Surface elevation (m.a.s.l.)", title_fontsize=11,
            frameon=False, ncol=1,
            borderpad=0.6, labelspacing=0.6, handlelength=2.2, handletextpad=0.7
        )
        leg_e.get_title().set_fontweight('bold')
        leg_e._legend_box.align = "left"
        _left_align_legend_title(leg_e, ax_bound=ax_leg, max_frac=0.48)

    if h_dhdt_global:
        leg_d = ax_leg.legend(
            h_dhdt_global, l_dhdt_clean_global,
            loc='center left', bbox_to_anchor=(0.53, 0.5), fontsize=12,
            title="Elevation change rate (dhdt)", title_fontsize=11,
            frameon=False, ncol=1,
            borderpad=0.6, labelspacing=0.6, handlelength=2.2, handletextpad=0.7
        )
        leg_d.get_title().set_fontweight('bold')
        leg_d._legend_box.align = "left"
        _left_align_legend_title(leg_d, ax_bound=ax_leg, max_frac=0.98)

    # the second legend() call replaces the first one; add it back
    if h_dem_global:
        ax_leg.add_artist(leg_e)

    if h_dem_global and h_dhdt_global:
        ax_leg.plot([0.5, 0.5], [0.08, 0.92], transform=ax_leg.transAxes,
                     color='gray', linewidth=1, alpha=0.4)

    # legend frame as wide as the plotted map
    _match_legend_panel_to_map(fig, ax_map, ax_leg)

    out_path = os.path.join(cfg.CENTERLINE_PLOT_DIR,
                              "combined_9panel_dem_and_dhdt_profiles_satellite.png")
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"   ✓ Saved: {os.path.basename(out_path)}")

