"""Multi-panel centerline figures combining a map with the profiles of all glaciers.
"""

import glob
import os
import re

import geopandas as gpd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import rasterio

from .. import config as cfg
from .centerline_geometry import (
    _ice_fill_arrays_split_at_grounding_line,
    get_centerline_plot_segment,
)
from .common import (
    terminus_color,
    with_bedrock_source,
    _left_align_legend_title,
    _match_legend_panel_to_map,
    _measure_map_target_aspect,
    add_satellite_basemap_panel,
    build_years_colormap,
    format_coord_labels,
)
from ..utils import get_display_year, get_longest_linestring


def create_combined_bedrock_plot(all_glacier_bedrock_data, grid_info):
    """
    Create a single figure with all glacier bedrock profiles as subplots,
    including uncertainty profiles.
    """

    if not cfg.CREATE_BEDROCK_PLOTS:
        print(f"      Skipping combined bedrock plot (CREATE_BEDROCK_PLOTS=False)")
        return

    if not all_glacier_bedrock_data:
        print("      No bedrock data available for combined plot")
        return

    print(f"\n      Creating combined bedrock plot with {len(all_glacier_bedrock_data)} glaciers...")

    # Same ordering / grid positions as the satellite-basemap 9-panel
    # figure, so panel (b) is glacier [1] in the same cell, etc.
    sorted_glaciers = sorted(all_glacier_bedrock_data.items(), key=lambda x: x[1].get('glacier_number', 999))
    grid_positions = [(0, 1), (0, 2), (1, 0), (1, 1), (1, 2), (2, 0), (2, 1), (2, 2)]

    fig = plt.figure(figsize=(26, 18), constrained_layout=True)
    gs_main = fig.add_gridspec(3, 3)

    # Top-left cell: map on top, compact legend strip beneath it.
    map_h_ratio, leg_h_ratio = 8.0, 0.5
    gs_map_leg = gs_main[0, 0].subgridspec(2, 1, height_ratios=[map_h_ratio, leg_h_ratio], hspace=0.02)
    ax_map = fig.add_subplot(gs_map_leg[0, 0])
    ax_leg = fig.add_subplot(gs_map_leg[1, 0])

    # Reserve realistic margins for the map's eventual labels/title with
    # throwaway placeholders - exactly _measure_map_target_aspect's own pattern
    # (labels/ticks/title, nothing else - deliberately NO dummy image or
    # colorbar, see the docstring above for why), so the box measured below -
    # once all 8 profile panels also exist and the layout is fully resolved -
    # is what the map will actually get.
    ax_map.set_ylabel("Latitude", fontsize=10)
    ax_map.set_xlabel("Longitude", fontsize=10, labelpad=10)
    ax_map.set_yticks([0, 1, 2, 3, 4, 5])
    ax_map.set_yticklabels(["-69.28°", "-69.38°", "-69.47°", "-69.55°", "-69.64°", "-69.72°"])
    ax_map.set_xticks([0, 1, 2, 3])
    ax_map.set_xticklabels(["-67.28°", "-66.89°", "-66.49°", "-66.07°"])
    ax_map.tick_params(axis='x', labelrotation=45, labelsize=9)
    ax_map.tick_params(axis='y', labelsize=9)
    for label in ax_map.get_xticklabels():
        label.set_horizontalalignment('right')
    ax_map.set_title(f"(a) {with_bedrock_source('Bedrock Elevation Map')}\nwith Centerlines", fontsize=12, fontweight='bold')

    # Panels (b)-(i): glacier profiles, same grid_positions as the satellite-
    # basemap 9-panel figure so each glacier occupies the same cell in both
    # figures. Built before the map's real content (below) so the measurement
    # right after reflects the fully-resolved layout, not an intermediate one
    # missing 8 of the figure's 9 panels.
    _all_min_dist = [
        gd['global_min_dist'] for _, gd in sorted_glaciers[:8]
        if gd.get('global_min_dist') is not None
    ]
    _all_max_dist = [
        gd['global_max_dist'] for _, gd in sorted_glaciers[:8]
        if gd.get('global_max_dist') is not None
    ]
    if _all_min_dist and _all_max_dist:
        shared_xmin_km = max(0.0, min(_all_min_dist) / 1000.0)
        shared_xmax_km = min(cfg.CENTERLINE_MAX_DISTANCE / 1000.0, max(_all_max_dist) / 1000.0)
        if shared_xmax_km <= shared_xmin_km:
            shared_xmin_km, shared_xmax_km = 0.0, cfg.CENTERLINE_MAX_DISTANCE / 1000.0
    else:
        shared_xmin_km, shared_xmax_km = 0.0, cfg.CENTERLINE_MAX_DISTANCE / 1000.0

    # Handles captured from the first glacier panel, split into the two
    # legend groups shown in the fused legend panel beneath the map
    # (mirrors h_dem_global / h_dhdt_global in create_combined_dem_and_dhdt_9panel).
    h_surface_global, l_surface_global = [], []
    h_bedrock_global, l_bedrock_global = [], []

    for idx, (glacier_name, glacier_data) in enumerate(sorted_glaciers[:8]):

        pos = grid_positions[idx]
        ax = fig.add_subplot(gs_main[pos])

        bedrock_dist_clipped = glacier_data.get('bedrock_distances')
        bedrock_val_clipped = glacier_data.get('bedrock_values')
        grounding_line_distance_m = glacier_data.get('grounding_line_distance')

        surface_data = glacier_data.get('surface_data', [])
        terminus_positions = glacier_data.get('terminus_positions', {})
        centerline_length = glacier_data.get('centerline_length', np.nan)
        dem_colormap = glacier_data.get('dem_colormap', build_years_colormap(max(len(surface_data), 1)))
        glacier_number = glacier_data.get('glacier_number', idx + 1)

        xmin_km = shared_xmin_km
        xmax_km = shared_xmax_km

        # Bedrock
        if bedrock_dist_clipped is not None and len(bedrock_dist_clipped) > 0:
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
                linewidth=2,
                linestyle='-',
                zorder=2
            )
            # The ice-color fill just below is still split at the grounding
            # line.
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
                linewidth=1.5,
                linestyle='-',
                alpha=0.95,
                zorder=5,
                label='Sea level (0 m)'
            )

        # Plot actual surface elevations
        for surf_idx, (year, distances_plot, values_plot) in enumerate(surface_data):
            color_idx = min(surf_idx, len(dem_colormap) - 1)

            ax.plot(
                distances_plot / 1000.0,
                values_plot,
                color=dem_colormap[color_idx],
                linewidth=2,
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

                    # Landward of the grounding line: solid ice fill down to
                    # the bed. Seaward of it: freeboard-only fill down to sea
                    # level.
                    x_fill, lo_fill, hi_fill = _ice_fill_arrays_split_at_grounding_line(
                        distances_plot, values_plot, bedrock_interp,
                        bedrock_dist_clipped, bedrock_val_clipped,
                        grounding_line_distance_m
                    )
                    if x_fill is not None:
                        ax.fill_between(
                            x_fill, lo_fill, hi_fill,
                            alpha=0.2,
                            color=dem_colormap[color_idx],
                            zorder=0
                        )
                except Exception as e:
                    print(f"Interpolation failed: {e}")

        # Grounding line: solid black vertical segment restricted
        # between the bottom of the plot (-800, matching the bedrock
        # fill_between's own hardcoded lower bound above) and the bedrock
        # surface at this position. Only drawn if it falls within this panel's
        # shared displayed x-range, same gating as the terminus markers just
        # below.
        if grounding_line_distance_m is not None:
            gl_km = grounding_line_distance_m / 1000.0
            if (xmin_km <= gl_km <= xmax_km and bedrock_dist_clipped is not None
                    and len(bedrock_dist_clipped) > 0):
                try:
                    bed_at_gl = np.interp(
                        grounding_line_distance_m, bedrock_dist_clipped, bedrock_val_clipped,
                        left=np.nan, right=np.nan
                    )
                    if np.isfinite(bed_at_gl):
                        ax.plot(
                            [gl_km, gl_km], [-800, bed_at_gl],
                            color='black', linewidth=2, linestyle='-',
                            solid_capstyle='butt', zorder=8,
                            label=cfg.GROUNDING_LINE_LABEL
                        )
                        ax.text(
                            gl_km, bed_at_gl, cfg.GROUNDING_LINE_SHORT_LABEL,
                            rotation=90, va='bottom', ha='center',
                            fontsize=8, fontweight='bold', color='black',
                            zorder=11,
                            bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.8, edgecolor='none')
                        )
                except Exception as e:
                    print(f"Grounding line plot failed: {e}")

        # Plot year marker lines only if within valid displayed range
        for mask_year in sorted(terminus_positions.keys()):
            position_km = terminus_positions[mask_year] / 1000.0

            if not (xmin_km <= position_km <= xmax_km):
                continue

            ax.axvline(
                x=position_km,
                color=terminus_color(mask_year),
                linestyle='--',
                linewidth=1.5,
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
                va='top',
                ha='right',
                fontsize=8,
                fontweight='bold',
                color=terminus_color(mask_year),
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8, edgecolor='none')
            )

        # Panel lettering matches the satellite-basemap figure: (b)...(i),
        # same glacier -> same letter, since both use sorted_glaciers +
        # grid_positions in the same order.
        letter = chr(97 + idx + 1)
        ax.set_xlabel("Distance along flowline (km)", fontsize=11)
        ax.set_ylabel("Elevation (m)", fontsize=11)
        ax.set_title(f"({letter}) {glacier_name} [{glacier_number}]", fontsize=13, fontweight='bold')
        ax.grid(True, linestyle='--', alpha=0.4)

        if xmax_km > xmin_km:
            ax.set_xlim(xmin_km, xmax_km)
        else:
            ax.set_xlim(0, cfg.CENTERLINE_MAX_DISTANCE / 1000.0)

        ax.autoscale(enable=True, axis='y')

        # No per-panel legend anymore -- replaced by the single fused legend
        # panel beneath the map (same approach as the satellite- basemap
        # figure). Capture handles/labels from the first glacier panel only,
        # split into the two groups shown there.
        if idx == 0:
            h_all, l_all = ax.get_legend_handles_labels()
            for h, l in zip(h_all, l_all):
                if l.startswith('Surface '):
                    h_surface_global.append(h)
                    l_surface_global.append(l)
                else:
                    h_bedrock_global.append(h)
                    l_bedrock_global.append(l)

    # Panel (a): Bedrock map. That measurement is fundamentally a moving target
    # here: constrained_layout re-solves every axes' position on EVERY draw
    # (including the final savefig(bbox_inches='tight') pass), so whatever was
    # measured earlier -- before the colorbar existed, or even after it but
    # before the final save -- can go stale by the time the figure is actually
    # written out.
    try:
        with rasterio.open(cfg.BEDROCK_PATH) as src:
            bedrock_map_data = src.read(1).astype(np.float32)
            bedrock_map_transform = src.transform
            bedrock_nodata = src.nodata

            if bedrock_nodata is not None:
                if np.isnan(bedrock_nodata):
                    bedrock_map_data[~np.isfinite(bedrock_map_data)] = np.nan
                else:
                    bedrock_map_data[bedrock_map_data == bedrock_nodata] = np.nan

        mask_gdf = gpd.read_file(cfg.MASK_PATH)
        mask_gdf = mask_gdf.to_crs(grid_info['crs'])

        minx, miny, maxx, maxy = mask_gdf.total_bounds

        # Zoom in on the centerlines' own combined bounding box (+ padding)
        # instead of showing the whole drainage-basin mask - same idea as
        # add_satellite_basemap_panel's zoom_to_centerlines.
        ZOOM_TO_CENTERLINES = True
        ZOOM_PAD_FRAC = 0.18    # fraction of the longer side added as margin
        ZOOM_PAD_MIN_M = 300.0   # minimum margin in metres regardless of ZOOM_PAD_FRAC

        if ZOOM_TO_CENTERLINES and os.path.exists(cfg.CENTERLINE_DIR):
            cl_bounds_list = []
            for _cl_file in glob.glob(os.path.join(cfg.CENTERLINE_DIR, "*.shp")):
                try:
                    _cl_gdf = gpd.read_file(_cl_file).to_crs(grid_info['crs'])
                    _cl_geom = get_longest_linestring(_cl_gdf.unary_union)
                    if _cl_geom is not None:
                        cl_bounds_list.append(_cl_geom.bounds)
                except Exception:
                    pass
            if cl_bounds_list:
                _b = np.array(cl_bounds_list)
                cl_minx, cl_miny = _b[:, 0].min(), _b[:, 1].min()
                cl_maxx, cl_maxy = _b[:, 2].max(), _b[:, 3].max()
                span = max(cl_maxx - cl_minx, cl_maxy - cl_miny)
                pad = max(span * ZOOM_PAD_FRAC, ZOOM_PAD_MIN_M)
                minx, maxx = cl_minx - pad, cl_maxx + pad
                miny, maxy = cl_miny - pad, cl_maxy + pad

        col_min, row_max = ~bedrock_map_transform * (minx, miny)
        col_max, row_min = ~bedrock_map_transform * (maxx, maxy)

        # Clipped to the raster's own extent so the crop never reaches
        # beyond available data.
        row_min = max(0, int(np.floor(row_min)))
        row_max = min(bedrock_map_data.shape[0], int(np.ceil(row_max)))
        col_min = max(0, int(np.floor(col_min)))
        col_max = min(bedrock_map_data.shape[1], int(np.ceil(col_max)))

        bedrock_cropped = bedrock_map_data[row_min:row_max, col_min:col_max]

        extent = [
            bedrock_map_transform.c + col_min * bedrock_map_transform.a,
            bedrock_map_transform.c + col_max * bedrock_map_transform.a,
            bedrock_map_transform.f + row_max * bedrock_map_transform.e,
            bedrock_map_transform.f + row_min * bedrock_map_transform.e
        ]

        vmin = np.nanpercentile(bedrock_cropped, 2)
        vmax = np.nanpercentile(bedrock_cropped, 98)

        im = ax_map.imshow(
            bedrock_cropped,
            cmap='viridis',
            vmin=vmin,
            vmax=vmax,
            extent=extent,
            origin='upper',
            aspect='auto'
        )

        cbar = plt.colorbar(im, ax=ax_map, orientation='vertical', pad=0.03, shrink=0.97, fraction=0.05)
        cbar.set_label(with_bedrock_source('Bedrock Elevation (m)'), fontsize=10)

        # --------------------------------------------------------------
        # Lock the view to the cropped/zoomed extent computed above. That's why
        # changing ZOOM_PAD_FRAC/ZOOM_PAD_MIN_M alone had no visible effect.
        # --------------------------------------------------------------
        ax_map.set_xlim(extent[0], extent[1])
        ax_map.set_ylim(extent[2], extent[3])

        mask_gdf.boundary.plot(ax=ax_map, color='black', linewidth=1.5, zorder=5)

        if os.path.exists(cfg.CENTERLINE_DIR):
            centerline_files = glob.glob(os.path.join(cfg.CENTERLINE_DIR, "*.shp"))

            glacier_name_to_number = {}
            for glacier_name, data in all_glacier_bedrock_data.items():
                if 'glacier_number' in data:
                    glacier_name_to_number[glacier_name] = data['glacier_number']

            for centerline_file in centerline_files:
                try:
                    cl_gdf = gpd.read_file(centerline_file)
                    cl_gdf = cl_gdf.to_crs(grid_info['crs'])

                    centerline_name = os.path.splitext(os.path.basename(centerline_file))[0]
                    glacier_name = centerline_name.replace('Centerline_', '')
                    glacier_name = re.sub(r'_\d{2}$', '', glacier_name)
                    glacier_name = glacier_name.replace('_', ' ')

                    cl_geom = get_longest_linestring(cl_gdf.unary_union)
                    if cl_geom is None:
                        continue
                    cl_geom_plot = get_centerline_plot_segment(cl_geom)

                    ax_map.plot(*cl_geom_plot.xy, color='red', linewidth=2, zorder=10, alpha=0.8)

                    if glacier_name in glacier_name_to_number:
                        glacier_number = glacier_name_to_number[glacier_name]
                        midpoint = cl_geom_plot.interpolate(0.5, normalized=True)
                        ax_map.text(
                            midpoint.x, midpoint.y, str(glacier_number),
                            fontsize=11, fontweight='bold', color='white',
                            ha='center', va='center',
                            zorder=15,
                            bbox=dict(boxstyle='circle,pad=0.15', facecolor='black', edgecolor='white', alpha=0.8)
                        )
                except Exception as e:
                    print(f"Centerline load failed: {e}")

        # Re-asserting aspect='auto' here, after every plot() call that could
        # have touched it, makes sure it's the one that actually sticks for the
        # final render.
        ax_map.set_aspect('auto')

        ax_map.set_title(f"(a) {with_bedrock_source('Bedrock Elevation Map')}\nwith Centerlines", fontsize=12, fontweight='bold')
        ax_map.set_xlabel("Easting (m)", fontsize=10, labelpad=10)
        ax_map.set_ylabel("Northing (m)", fontsize=10)

        ax_map.xaxis.set_major_locator(plt.MaxNLocator(5))
        ax_map.yaxis.set_major_locator(plt.MaxNLocator(6))
        ax_map.xaxis.set_major_formatter(mticker.ScalarFormatter())
        ax_map.yaxis.set_major_formatter(mticker.ScalarFormatter())
        ax_map.ticklabel_format(style='plain', useOffset=False)
        ax_map.tick_params(axis='x', labelrotation=45, labelsize=9)
        ax_map.tick_params(axis='y', labelsize=9)

        for label in ax_map.get_xticklabels():
            label.set_horizontalalignment('right')

        format_coord_labels(ax_map)
        print(f"      ✓ Added bedrock map to panel (a)")

    except Exception as e:
        print(f"      ⚠️  Could not create bedrock map: {e}")
        import traceback
        traceback.print_exc()
        ax_map.text(
            0.5, 0.5, 'Bedrock map\nunavailable',
            ha='center', va='center', fontsize=12,
            transform=ax_map.transAxes
        )
        ax_map.axis('off')

    # Legend strip beneath the map: one flat legend with all entries (Surface
    # years + Bedrock/Sea level) in a single compact, 2-column block - not the
    # two-group divided legend used for the satellite-basemap figure's DEM/dhdt
    # split, which doesn't apply here.
    ax_leg.axis('off')
    ax_leg.set_facecolor('none')

    all_handles_global = h_surface_global + h_bedrock_global
    all_labels_global = l_surface_global + l_bedrock_global
    if all_handles_global:
        # loc='center' + bbox_to_anchor=(0.5, 0.5) (both in ax_leg's own axes-
        # fraction coordinates) explicitly centers the legend block within
        # ax_leg's box, rather than relying on loc='center''s default bbox (the
        # full 0-1 axes box, which gives the same result but only implicitly --
        # being explicit here also means this stays centered if ax_leg's own
        # bbox_to_anchor default ever changes upstream).
        leg_all = ax_leg.legend(
            all_handles_global, all_labels_global,
            loc='center', bbox_to_anchor=(0.5, 0.5), fontsize=10.5, ncol=2,
            frameon=True, edgecolor='black', facecolor='white', framealpha=1.0,
            borderpad=0.6, labelspacing=0.45, handlelength=2.0,
            handletextpad=0.6, columnspacing=1.4
        )

    # No suptitle -- each panel already carries its own (letter) title, so a
    # figure-level title only risks overlapping panel (b)'s title.


    out_path = os.path.join(cfg.CENTERLINE_PLOT_DIR, "combined_bedrock_profiles.png")
    plt.savefig(out_path, dpi=cfg.DPI, bbox_inches='tight')
    plt.close(fig)

    print(f"      ✅ Saved combined bedrock plot: {os.path.basename(out_path)}")


def render_glacier_dem_dhdt_panel(ax_dem, ax_dhdt, glacier_name, glacier_number, letter,
                                  b_data, dhdt_info, dem_colors, dhdt_colors,
                                  elevation_ylabel="Elevation (m)",
                                  title_fontsize=16, label_fontsize=13,
                                  tick_labelsize=12,
                                  dhdt_ylim=None, elev_ylim=(-20.0, 500.0),
                                  shared_xlim=None):
    """
    Render one glacier's stacked DEM-surface (top) + dh/dt (bottom) subplot
    pair into the given axes. Shared by the satellite-basemap figure variants
    (v3 landscape, v4 portrait) so their panel content stays identical to the
    original 9-panel / map-centred figures.
    """
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


def create_combined_dem_and_dhdt_9panel(all_glacier_bedrock_data, all_glacier_dhdt_data, grid_info, preloaded_ref_dem,
                                        map_change=None, dhdt_ylim_lower=-12, dhdt_ylim_upper=18,
                                        elev_ylim=(-20.0, 500.0)):
    """
    Creates a 9-panel plot: - Panel 1: Reference DEM map with numerated
    centerlines. - Panels 2-9: For each glacier, a stacked plot (top: DEM
    surfaces, bottom: dh/dt). - Legends are centralized between Panel 1 and
    Panel 2.
    """
    if not all_glacier_bedrock_data:
        print("      No data available for 9-panel DEM/dhdt combined plot")
        return

    print(f"\n      Creating 9-panel combined DEM & dh/dt plot...")

    fig = plt.figure(figsize=(26, 18))
    gs_main = fig.add_gridspec(3, 3, wspace=0.25, hspace=0.3)

    # ------------------------------------------------------------------
    # Panel 1: Reference DEM map
    # ------------------------------------------------------------------
    ax_map = fig.add_subplot(gs_main[0, 0])

    try:
        mask_gdf = gpd.read_file(cfg.MASK_PATH).to_crs(grid_info['crs'])
        bounds = mask_gdf.total_bounds
        minx, miny, maxx, maxy = bounds

        col_min, row_max = ~grid_info['transform'] * (minx, miny)
        col_max, row_min = ~grid_info['transform'] * (maxx, maxy)

        row_min = max(0, int(np.floor(row_min)))
        row_max = min(preloaded_ref_dem.shape[0], int(np.ceil(row_max)))
        col_min = max(0, int(np.floor(col_min)))
        col_max = min(preloaded_ref_dem.shape[1], int(np.ceil(col_max)))

        dem_cropped = preloaded_ref_dem[row_min:row_max, col_min:col_max]
        extent = [
            grid_info['transform'].c + col_min * grid_info['transform'].a,
            grid_info['transform'].c + col_max * grid_info['transform'].a,
            grid_info['transform'].f + row_max * grid_info['transform'].e,
            grid_info['transform'].f + row_min * grid_info['transform'].e
        ]

        vmin, vmax = np.nanpercentile(dem_cropped, 2), np.nanpercentile(dem_cropped, 98)
        im = ax_map.imshow(dem_cropped, cmap='terrain', vmin=vmin, vmax=vmax, extent=extent, origin='upper',
                           aspect='equal')

        # Adjust pad and shrink to avoid overlapping with the Longitude label
        cbar = plt.colorbar(im, ax=ax_map, orientation='horizontal', pad=0.15, shrink=0.6)
        cbar.set_label('Reference DEM Elevation (m)', fontsize=10)

        mask_gdf.boundary.plot(ax=ax_map, color='black', linewidth=1.5, zorder=5)

        if os.path.exists(cfg.CENTERLINE_DIR):
            glacier_name_to_number = {name: data.get('glacier_number', 0) for name, data in
                                      all_glacier_bedrock_data.items()}

            for centerline_file in glob.glob(os.path.join(cfg.CENTERLINE_DIR, "*.shp")):
                cl_gdf = gpd.read_file(centerline_file).to_crs(grid_info['crs'])
                centerline_name = os.path.splitext(os.path.basename(centerline_file))[0]
                g_name = centerline_name.replace('Centerline_', '')
                g_name = re.sub(r'_\d{2}$', '', g_name).replace('_', ' ')

                cl_geom = get_longest_linestring(cl_gdf.unary_union)
                if cl_geom:
                    cl_geom_plot = get_centerline_plot_segment(cl_geom)
                    ax_map.plot(*cl_geom_plot.xy, color='maroon', linewidth=2, zorder=10, alpha=0.8)
                    if g_name in glacier_name_to_number:
                        midpoint = cl_geom_plot.interpolate(0.5, normalized=True)
                        ax_map.text(midpoint.x, midpoint.y, str(glacier_name_to_number[g_name]),
                                    fontsize=9, fontweight='bold', color='white', ha='center', va='center', zorder=15,
                                    bbox=dict(boxstyle='circle,pad=0.15', facecolor='black', edgecolor='white',
                                              alpha=0.8))

        # Updated Panel Title to start with (a)
        ax_map.set_title('(a) Reference DEM & Centerlines', fontsize=12, fontweight='bold')
        ax_map.set_xlabel("Easting (m)", fontsize=10)
        ax_map.set_ylabel("Northing (m)", fontsize=10)
        format_coord_labels(ax_map)

    except Exception as e:
        print(f"      ⚠️  Could not create map panel: {e}")

    # ------------------------------------------------------------------
    # Panels 2-9: Glacier profiles
    # ------------------------------------------------------------------
    sorted_glaciers = sorted(all_glacier_bedrock_data.items(), key=lambda x: x[1].get('glacier_number', 999))
    grid_positions = [(0, 1), (0, 2), (1, 0), (1, 1), (1, 2), (2, 0), (2, 1), (2, 2)]

    # Shared dh/dt y-axis extent across all 8 glaciers (used by the satellite
    # -basemap figure variants below) so panels are directly comparable
    # instead of each auto-scaling to its own data range.
    _all_dhdt_vals = []
    for _gname, _ in sorted_glaciers[:8]:
        for _p_data in all_glacier_dhdt_data.get(_gname, {}).values():
            _vals = np.asarray(_p_data.get('dhdt', []))
            _vals = _vals[np.isfinite(_vals)]
            if _vals.size > 0:
                _all_dhdt_vals.append(_vals)
    if _all_dhdt_vals:
        _dhdt_concat = np.concatenate(_all_dhdt_vals)
        _dhdt_lo, _dhdt_hi = np.nanpercentile(_dhdt_concat, [0.5, 99.5])
        _dhdt_pad = 0.08 * (_dhdt_hi - _dhdt_lo) if _dhdt_hi > _dhdt_lo else 1.0
        # Each bound uses the explicit value if given, else falls back to
        # the data-derived (percentile + padding) value.
        _final_lo = dhdt_ylim_lower if dhdt_ylim_lower is not None else (_dhdt_lo - _dhdt_pad)
        _final_hi = dhdt_ylim_upper if dhdt_ylim_upper is not None else (_dhdt_hi + _dhdt_pad)
        shared_dhdt_ylim = (_final_lo, _final_hi)
    else:
        shared_dhdt_ylim = (dhdt_ylim_lower, dhdt_ylim_upper) if (dhdt_ylim_lower is not None and dhdt_ylim_upper is not None) else None

    # Shared distance (x-axis) range across all 8 glaciers, for the same
    # comparability reason as shared_dhdt_ylim above: each glacier's own valid-
    # data span (global_min_dist/global_max_dist, set in create_bedrock_plot)
    # differs, so letting every panel auto-fit to just its own span makes the
    # panels look tidy individually but means "5 km" sits at a different
    # physical position on every glacier's subplot -- destroying cross-glacier
    # comparability.
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

    dem_colors = ['royalblue', 'mediumorchid', 'hotpink', 'orchid']
    dhdt_colors = ['darkorange', 'crimson','peru', 'saddlebrown']

    # Placeholders for global legends
    h_dem_global, l_dem_global = [], []
    h_dhdt_global, l_dhdt_global = [], []
    # Clean-label versions for the satellite-basemap figure variants (v3/v4):
    # year-only DEM labels, and "yA - yB" spaced dhdt period labels.
    l_dem_clean_global, l_dhdt_clean_global = [], []

    for idx, (glacier_name, b_data) in enumerate(sorted_glaciers[:8]):
        pos = grid_positions[idx]

        gs_inner = gs_main[pos].subgridspec(2, 1, height_ratios=[2, 1], hspace=0.1)
        ax_dem = fig.add_subplot(gs_inner[0])
        ax_dhdt = fig.add_subplot(gs_inner[1], sharex=ax_dem)

        glacier_number = b_data.get('glacier_number', '?')
        surface_data = b_data.get('surface_data', [])

        # Plot DEMs
        for s_idx, (year, distances, values) in enumerate(surface_data):
            color = dem_colors[s_idx % len(dem_colors)]
            ax_dem.plot(distances / 1000.0, values, color=color, linewidth=2.5, label=f"Elev {get_display_year(year)}",
                        alpha=0.9)

        # Plot Terminus Positions
        terminus_positions = b_data.get('terminus_positions', {})
        _elev_label_y = elev_ylim[1] - 0.04 * (elev_ylim[1] - elev_ylim[0])
        for mask_year in sorted(terminus_positions.keys()):
            pos_km = terminus_positions[mask_year] / 1000.0
            t_col = terminus_color(mask_year)
            ax_dem.axvline(pos_km, color=t_col, linestyle='--', alpha=0.7, lw=1.5)
            ax_dhdt.axvline(pos_km, color=t_col, linestyle='--', alpha=0.7, lw=1.5)
            ax_dem.text(pos_km, _elev_label_y, mask_year, rotation=90, va='top', ha='right', fontsize=8, color=t_col,
                        bbox=dict(boxstyle='round,pad=0.1', facecolor='white', alpha=0.7, edgecolor='none'))

        ax_dem.set_ylabel("Elevation (m)", fontsize=10)

        # Sequentially calculate panel letters (b to i)
        letter = chr(97 + idx + 1)
        # Formatted smoothly as: (letter) Glacier Name [Number]
        ax_dem.set_title(f"({letter}) {glacier_name} [{glacier_number}]", fontsize=13, fontweight='bold')

        ax_dem.grid(True, linestyle='--', alpha=0.4)
        ax_dem.set_ylim(*elev_ylim)
        plt.setp(ax_dem.get_xticklabels(), visible=False)

        # Plot dh/dt
        dhdt_info = all_glacier_dhdt_data.get(glacier_name, {})
        dhdt_clean_labels_this_glacier = []
        for p_idx, (period_key, p_data) in enumerate(dhdt_info.items()):
            dist_km = p_data['distances_km']
            dhdt_vals = p_data['dhdt']
            color = dhdt_colors[p_idx % len(dhdt_colors)]

            if len(dist_km) == len(dhdt_vals) and len(dist_km) > 0:
                ax_dhdt.plot(dist_km, dhdt_vals, color=color, linestyle='-', linewidth=2, label=f"dhdt {period_key}")
                dhdt_clean_labels_this_glacier.append(
                    p_data.get('period_display', period_key).replace('-', ' - ')
                )

        ax_dhdt.axhline(0, color='black', linewidth=1, linestyle='-', alpha=0.5)
        ax_dhdt.set_ylabel("dhdt (m a⁻¹)", fontsize=10)
        ax_dhdt.set_xlabel("Distance (km)", fontsize=10)
        ax_dhdt.grid(True, linestyle='--', alpha=0.4)
        ax_dhdt.set_xlim(shared_xmin_km, shared_xmax_km)
        if shared_dhdt_ylim is not None:
            ax_dhdt.set_ylim(*shared_dhdt_ylim)

        # Capture handles for the global legends from the first plotted glacier
        if idx == 0:
            h_dem_global, l_dem_global = ax_dem.get_legend_handles_labels()
            h_dhdt_global, l_dhdt_global = ax_dhdt.get_legend_handles_labels()
            l_dem_clean_global = [get_display_year(year) for year, _, _ in surface_data]
            l_dhdt_clean_global = dhdt_clean_labels_this_glacier

    # ------------------------------------------------------------------
    # Add Global Legends in the space between Panel 1 and Panel 2
    # ------------------------------------------------------------------
    if h_dem_global and h_dhdt_global:
        # Force a rendering pass so get_position() is perfectly accurate
        fig.canvas.draw()

        # Explicitly shift Panel 1 (the map) to the left to make room
        pos = ax_map.get_position()
        ax_map.set_position([pos.x0 - 0.04, pos.y0, pos.width, pos.height])

        # Explicitly shift the colorbar too, so it stays centered under the map
        if 'cbar' in locals():
            cbar_pos = cbar.ax.get_position()
            cbar.ax.set_position([cbar_pos.x0 - 0.04, cbar_pos.y0, cbar_pos.width, cbar_pos.height])

        # Legend for Elevations (anchored closer to the map now)
        leg_elev = ax_map.legend(
            h_dem_global, l_dem_global,
            loc='upper left',
            bbox_to_anchor=(1.05, 1.0),
            fontsize=10,
            title="Surface Elevation",
            title_fontsize=11
        )
        leg_elev.get_title().set_fontweight('bold')

        # Legend for dh/dt (added as a separate artist)
        leg_dhdt = ax_map.legend(
            h_dhdt_global, l_dhdt_global,
            loc='upper left',
            bbox_to_anchor=(1.05, 0.65),
            fontsize=10,
            title="Elevation Change Rate",
            title_fontsize=11
        )
        leg_dhdt.get_title().set_fontweight('bold')

        # Must manually re-add the first legend to the axis since the second call removes it
        ax_map.add_artist(leg_elev)

    # --- (a) Save the original 3×3 combined figure ---
    out_path = os.path.join(cfg.CENTERLINE_PLOT_DIR, "combined_9panel_dem_and_dhdt_profiles.png")
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"      ✅ Saved 9-panel combined plot: {os.path.basename(out_path)}")

    # ------------------------------------------------------------------
    # (b) Save each glacier panel as an individual PNG
    # ------------------------------------------------------------------
    print(f"      Saving individual glacier panels...")
    panel_letters = [chr(97 + i + 1) for i in range(8)]   # b … i

    for idx, (glacier_name, b_data) in enumerate(sorted_glaciers[:8]):
        glacier_number = b_data.get('glacier_number', '?')
        surface_data   = b_data.get('surface_data', [])
        terminus_positions = b_data.get('terminus_positions', {})

        fig_ind, (ax_d, ax_h) = plt.subplots(
            2, 1, figsize=(8, 6),
            gridspec_kw={'height_ratios': [2, 1], 'hspace': 0.1}
        )

        # DEMs
        for s_idx, (year, distances, values) in enumerate(surface_data):
            color = dem_colors[s_idx % len(dem_colors)]
            ax_d.plot(distances / 1000.0, values, color=color, linewidth=2.5,
                      label=f"Elev {get_display_year(year)}", alpha=0.9)

        # Terminus lines
        for mask_year in sorted(terminus_positions.keys()):
            pos_km = terminus_positions[mask_year] / 1000.0
            t_col  = terminus_color(mask_year)
            ax_d.axvline(pos_km, color=t_col, linestyle='--', alpha=0.7, lw=1.5)
            ax_h.axvline(pos_km, color=t_col, linestyle='--', alpha=0.7, lw=1.5)
            ax_d.text(pos_km, elev_ylim[1] - 0.04 * (elev_ylim[1] - elev_ylim[0]), mask_year, rotation=90, va='top', ha='right',
                      fontsize=8, color=t_col,
                      bbox=dict(boxstyle='round,pad=0.1', facecolor='white',
                                alpha=0.7, edgecolor='none'))

        letter = panel_letters[idx]
        ax_d.set_title(f"({letter}) {glacier_name} [{glacier_number}]",
                       fontsize=13, fontweight='bold')
        ax_d.set_ylabel("Elevation (m)", fontsize=10)
        ax_d.set_ylim(*elev_ylim)
        ax_d.set_xlim(shared_xmin_km, shared_xmax_km)
        ax_d.grid(True, linestyle='--', alpha=0.4)
        plt.setp(ax_d.get_xticklabels(), visible=False)

        h_leg, l_leg = ax_d.get_legend_handles_labels()
        if h_leg:
            ax_d.legend(h_leg, l_leg, fontsize=8, loc='upper right',
                        title="Surface Elevation", title_fontsize=9)

        # dh/dt
        dhdt_info = all_glacier_dhdt_data.get(glacier_name, {})
        for p_idx, (period_key, p_data) in enumerate(dhdt_info.items()):
            dist_km   = p_data['distances_km']
            dhdt_vals = p_data['dhdt']
            color     = dhdt_colors[p_idx % len(dhdt_colors)]
            if len(dist_km) == len(dhdt_vals) and len(dist_km) > 0:
                ax_h.plot(dist_km, dhdt_vals, color=color, linestyle='-',
                          linewidth=2, label=f"dhdt {period_key}")

        ax_h.axhline(0, color='black', linewidth=1, linestyle='-', alpha=0.5)
        ax_h.set_ylabel("dhdt (m a⁻¹)", fontsize=10)
        ax_h.set_xlabel("Distance (km)", fontsize=10)
        ax_h.grid(True, linestyle='--', alpha=0.4)
        ax_h.set_xlim(shared_xmin_km, shared_xmax_km)
        if shared_dhdt_ylim is not None:
            ax_h.set_ylim(*shared_dhdt_ylim)

        h_dh, l_dh = ax_h.get_legend_handles_labels()
        if h_dh:
            ax_h.legend(h_dh, l_dh, fontsize=8, loc='lower left',
                        title="Elevation Change Rate", title_fontsize=9)

        ind_path = os.path.join(
            cfg.CENTERLINE_PLOT_DIR,
            f"{glacier_number:02d}_{glacier_name.replace(' ', '_')}_dem_dhdt_panel.png"
            if isinstance(glacier_number, int)
            else f"{glacier_name.replace(' ', '_')}_dem_dhdt_panel.png"
        )
        plt.savefig(ind_path, dpi=300, bbox_inches='tight')
        plt.close(fig_ind)
        print(f"         ✅ Saved individual panel: {os.path.basename(ind_path)}")

    # ------------------------------------------------------------------
    # (c) Map-centred layout: 4 glaciers | map + legends | 4 glaciers
    # ------------------------------------------------------------------
    print(f"      Creating map-centred layout...")

    # 3 columns: left glaciers | centre (map + legends) | right glaciers Centre
    # column is wider; each glacier column holds 4 rows of DEM+dhdt panels.
    # Inside the centre column: map takes ~70 % of the height, legends share
    # the remaining 30 % side by side.

    fig2 = plt.figure(figsize=(26, 20))

    gs2 = fig2.add_gridspec(
        4, 3,
        width_ratios=[2.8, 3.0, 2.8],
        height_ratios=[1, 1, 1, 1],
        wspace=0.22,
        hspace=0.38
    )

    # Centre column: map (rows 0-2) + legend row (row 3)
    gs_centre = gs2[:, 1].subgridspec(
        5, 1,
        height_ratios=[1, 1, 1, 0.05, 0.65],   # 3 map rows + tiny gap + legend row
        hspace=0.0
    )
    ax_map2 = fig2.add_subplot(gs_centre[0:3, 0])   # map spans first 3 sub-rows

    try:
        mask_gdf2 = gpd.read_file(cfg.MASK_PATH).to_crs(grid_info['crs'])
        bounds2   = mask_gdf2.total_bounds
        minx2, miny2, maxx2, maxy2 = bounds2

        col_min2, row_max2 = ~grid_info['transform'] * (minx2, miny2)
        col_max2, row_min2 = ~grid_info['transform'] * (maxx2, maxy2)
        row_min2 = max(0, int(np.floor(row_min2)))
        row_max2 = min(preloaded_ref_dem.shape[0], int(np.ceil(row_max2)))
        col_min2 = max(0, int(np.floor(col_min2)))
        col_max2 = min(preloaded_ref_dem.shape[1], int(np.ceil(col_max2)))

        dem_c2 = preloaded_ref_dem[row_min2:row_max2, col_min2:col_max2]
        ext2 = [
            grid_info['transform'].c + col_min2 * grid_info['transform'].a,
            grid_info['transform'].c + col_max2 * grid_info['transform'].a,
            grid_info['transform'].f + row_max2 * grid_info['transform'].e,
            grid_info['transform'].f + row_min2 * grid_info['transform'].e,
        ]

        vmin2, vmax2 = np.nanpercentile(dem_c2, 2), np.nanpercentile(dem_c2, 98)
        im2 = ax_map2.imshow(dem_c2, cmap='terrain', vmin=vmin2, vmax=vmax2,
                              extent=ext2, origin='upper', aspect='equal')
        cbar2 = plt.colorbar(im2, ax=ax_map2, orientation='horizontal',
                              pad=0.06, shrink=0.85)
        cbar2.set_label('Reference DEM Elevation (m)', fontsize=10)
        mask_gdf2.boundary.plot(ax=ax_map2, color='black', linewidth=1.5, zorder=5)

        if os.path.exists(cfg.CENTERLINE_DIR):
            g2n = {n: d.get('glacier_number', 0) for n, d in all_glacier_bedrock_data.items()}
            for cl_file in glob.glob(os.path.join(cfg.CENTERLINE_DIR, "*.shp")):
                cl2    = gpd.read_file(cl_file).to_crs(grid_info['crs'])
                cn2    = os.path.splitext(os.path.basename(cl_file))[0]
                gn2    = re.sub(r'_\d{2}$', '',
                                cn2.replace('Centerline_', '')).replace('_', ' ')
                cl_geom2 = get_longest_linestring(cl2.unary_union)
                if cl_geom2:
                    cl_geom2_plot = get_centerline_plot_segment(cl_geom2)
                    ax_map2.plot(*cl_geom2_plot.xy, color='maroon', linewidth=2,
                                 zorder=10, alpha=0.8)
                    if gn2 in g2n:
                        mp2 = cl_geom2_plot.interpolate(0.5, normalized=True)
                        ax_map2.text(mp2.x, mp2.y, str(g2n[gn2]),
                                     fontsize=10, fontweight='bold',
                                     color='white', ha='center', va='center',
                                     zorder=15,
                                     bbox=dict(boxstyle='circle,pad=0.15',
                                               facecolor='black',
                                               edgecolor='white', alpha=0.8))

        ax_map2.set_title('(a) Reference DEM & Centerlines',
                          fontsize=12, fontweight='bold')
        ax_map2.set_xlabel("Longitude", fontsize=10)
        ax_map2.set_ylabel("Latitude", fontsize=10)
        format_coord_labels(ax_map2)

    except Exception as e:
        print(f"      ⚠️  Map panel (centred layout) failed: {e}")

    # Legend area: split the bottom sub-row of the centre column in two
    gs_leg = gs_centre[4, 0].subgridspec(1, 2, wspace=0.05)
    ax_leg_l = fig2.add_subplot(gs_leg[0, 0])
    ax_leg_r = fig2.add_subplot(gs_leg[0, 1])
    for _ax in (ax_leg_l, ax_leg_r):
        _ax.set_xticks([])
        _ax.set_yticks([])
        for spine in _ax.spines.values():
            spine.set_visible(False)

    if h_dem_global:
        leg_e2 = ax_leg_l.legend(
            h_dem_global, l_dem_global,
            loc='center', fontsize=9,
            title="Surface Elevation", title_fontsize=10,
            frameon=True, framealpha=1.0, edgecolor='black',
            ncol=1
        )
        leg_e2.get_title().set_fontweight('bold')

    if h_dhdt_global:
        leg_d2 = ax_leg_r.legend(
            h_dhdt_global, l_dhdt_global,
            loc='center', fontsize=9,
            title="Elevation Change Rate", title_fontsize=10,
            frameon=True, framealpha=1.0, edgecolor='black',
            ncol=1
        )
        leg_d2.get_title().set_fontweight('bold')

    # --- Glacier panels: col 0 = glaciers 1-4, col 2 = glaciers 5-8 ---
    panel_letters2 = [chr(97 + i + 1) for i in range(8)]

    for idx, (glacier_name, b_data) in enumerate(sorted_glaciers[:8]):
        glacier_number2 = b_data.get('glacier_number', '?')
        surface_data2   = b_data.get('surface_data', [])
        terminus_pos2   = b_data.get('terminus_positions', {})

        col2 = 0 if idx < 4 else 2
        row2 = idx if idx < 4 else idx - 4

        gs_inner2 = gs2[row2, col2].subgridspec(2, 1, height_ratios=[2, 1], hspace=0.08)
        ax_d2 = fig2.add_subplot(gs_inner2[0])
        ax_h2 = fig2.add_subplot(gs_inner2[1], sharex=ax_d2)

        for s_idx, (year, distances, values) in enumerate(surface_data2):
            color = dem_colors[s_idx % len(dem_colors)]
            ax_d2.plot(distances / 1000.0, values, color=color, linewidth=2,
                       label=f"Elev {get_display_year(year)}", alpha=0.9)

        for mask_year in sorted(terminus_pos2.keys()):
            pos_km2 = terminus_pos2[mask_year] / 1000.0
            t_col2  = terminus_color(mask_year)
            ax_d2.axvline(pos_km2, color=t_col2, linestyle='--', alpha=0.7, lw=1.2)
            ax_h2.axvline(pos_km2, color=t_col2, linestyle='--', alpha=0.7, lw=1.2)
            ax_d2.text(pos_km2, elev_ylim[1] - 0.04 * (elev_ylim[1] - elev_ylim[0]), mask_year, rotation=90, va='top', ha='right',
                       fontsize=7, color=t_col2,
                       bbox=dict(boxstyle='round,pad=0.1', facecolor='white',
                                 alpha=0.7, edgecolor='none'))

        letter2 = panel_letters2[idx]
        ax_d2.set_title(f"({letter2}) {glacier_name} [{glacier_number2}]",
                        fontsize=10, fontweight='bold')
        ax_d2.set_ylabel("Elevation (m)", fontsize=9)
        ax_d2.set_ylim(*elev_ylim)
        ax_d2.grid(True, linestyle='--', alpha=0.4)
        plt.setp(ax_d2.get_xticklabels(), visible=False)

        dhdt_info2 = all_glacier_dhdt_data.get(glacier_name, {})
        for p_idx, (period_key, p_data) in enumerate(dhdt_info2.items()):
            dist_km2   = p_data['distances_km']
            dhdt_vals2 = p_data['dhdt']
            color      = dhdt_colors[p_idx % len(dhdt_colors)]
            if len(dist_km2) == len(dhdt_vals2) and len(dist_km2) > 0:
                ax_h2.plot(dist_km2, dhdt_vals2, color=color, linestyle='-',
                           linewidth=1.8, label=f"dhdt {period_key}")

        ax_h2.axhline(0, color='black', linewidth=1, linestyle='-', alpha=0.5)
        ax_h2.set_ylabel("dhdt (m a⁻¹)", fontsize=9)
        ax_h2.set_xlabel("Distance (km)", fontsize=9)
        ax_h2.grid(True, linestyle='--', alpha=0.4)
        ax_h2.set_xlim(shared_xmin_km, shared_xmax_km)
        if shared_dhdt_ylim is not None:
            ax_h2.set_ylim(*shared_dhdt_ylim)

    out_path2 = os.path.join(cfg.CENTERLINE_PLOT_DIR,
                              "combined_map_centred_dem_and_dhdt_profiles.png")
    plt.savefig(out_path2, dpi=300, bbox_inches='tight')
    plt.close(fig2)
    print(f"      ✅ Saved map-centred layout: {os.path.basename(out_path2)}")

    # (d) Satellite-basemap layout (landscape): same 3x3 arrangement as the
    # original, but panel (a) uses a satellite image instead of the reference
    # DEM, and the space freed up by dropping the DEM colorbar is used for the
    # two legends (relocated + relabelled).
    print(f"      Creating satellite-basemap layout (landscape)...")

    fig3 = plt.figure(figsize=(26, 18), constrained_layout=True)
    gs_main3 = fig3.add_gridspec(3, 3)

    # Legend row kept modest (1.1 of 5.5 units) deliberately: panel (d) FIP2
    # [3] sits directly below this whole map+legend cell in the next gridspec
    # row, so the fused legend panel must stay inside this row's allotted
    # height rather than growing into FIP2's space.
    map_h_ratio3, leg_h_ratio3 = 4.4, 1.1
    gs_map_leg3 = gs_main3[0, 0].subgridspec(2, 1, height_ratios=[map_h_ratio3, leg_h_ratio3],
                                              hspace=0.02)
    ax_map3 = fig3.add_subplot(gs_map_leg3[0, 0])
    ax_leg3 = fig3.add_subplot(gs_map_leg3[1, 0])

    # Build the 8 glacier panels BEFORE the map/legend content: with
    # constrained_layout, adding the rest of the grid can globally rebalance
    # column spacing, so measuring the map's available box (next) before this
    # content exists would understate how much width it will actually end up
    # with.
    for idx, (glacier_name, b_data) in enumerate(sorted_glaciers[:8]):
        pos = grid_positions[idx]
        gs_inner3 = gs_main3[pos].subgridspec(2, 1, height_ratios=[2, 1])
        ax_dem3 = fig3.add_subplot(gs_inner3[0])
        ax_dhdt3 = fig3.add_subplot(gs_inner3[1], sharex=ax_dem3)

        letter3 = chr(97 + idx + 1)
        render_glacier_dem_dhdt_panel(
            ax_dem3, ax_dhdt3, glacier_name, b_data.get('glacier_number', '?'), letter3,
            b_data, all_glacier_dhdt_data.get(glacier_name, {}),
            dem_colors, dhdt_colors,
            elevation_ylabel="Elevation (m.a.s.l.)",
            dhdt_ylim=shared_dhdt_ylim, elev_ylim=elev_ylim,
            shared_xlim=(shared_xmin_km, shared_xmax_km)
        )

    # Measured (not theoretical) physical aspect ratio of the box the map will
    # occupy - accounts for the real margin its tick labels/title reserve, so
    # the padding below doesn't overshoot and needlessly shrink the map (see
    # _measure_map_target_aspect / add_satellite_basemap_panel).
    target_aspect3 = _measure_map_target_aspect(fig3, ax_map3, 26.0, 18.0)
    add_satellite_basemap_panel(ax_map3, grid_info, all_glacier_bedrock_data,
                                 target_aspect=target_aspect3)

    # Single fused legend panel (one frame, full map width) instead of two
    # separate boxes - stays within the legend row's own height budget.
    ax_leg3.set_xticks([])
    ax_leg3.set_yticks([])
    for spine in ax_leg3.spines.values():
        spine.set_visible(True)
        spine.set_edgecolor('black')
        spine.set_linewidth(1.1)
    ax_leg3.set_facecolor('white')

    if h_dem_global:
        leg_e3 = ax_leg3.legend(
            h_dem_global, l_dem_clean_global,
            loc='center left', bbox_to_anchor=(0.06, 0.5), fontsize=12,
            title="Surface elevation (m.a.s.l.)", title_fontsize=11,
            frameon=False, ncol=1,
            borderpad=0.6, labelspacing=0.6, handlelength=2.2, handletextpad=0.7
        )
        leg_e3.get_title().set_fontweight('bold')
        leg_e3._legend_box.align = "left"
        # Bold title at this box width doesn't fit past the centre divider
        # (0.5) at fontsize>=12 regardless of anchor - title_fontsize=11
        # plus this cap keeps it from ever overflowing.
        _left_align_legend_title(leg_e3, ax_bound=ax_leg3, max_frac=0.48)

    if h_dhdt_global:
        # Left-aligned, starting right next to the centre divider line
        # rather than centred or hugging the far right edge.
        leg_d3 = ax_leg3.legend(
            h_dhdt_global, l_dhdt_clean_global,
            loc='center left', bbox_to_anchor=(0.53, 0.5), fontsize=12,
            title="Elevation change rate (dhdt)", title_fontsize=11,
            frameon=False, ncol=1,
            borderpad=0.6, labelspacing=0.6, handlelength=2.2, handletextpad=0.7
        )
        leg_d3.get_title().set_fontweight('bold')
        leg_d3._legend_box.align = "left"
        # Capped against the axes' own right edge for the same reason.
        _left_align_legend_title(leg_d3, ax_bound=ax_leg3, max_frac=0.98)

    # ax_leg3.legend() called a second time above detaches leg_e3 from the
    # axes (matplotlib only tracks one "primary" legend per axes) - re-add
    # it explicitly so both legends render together inside the one frame.
    if h_dem_global:
        ax_leg3.add_artist(leg_e3)

    # Faint divider so the two legend groups stay visually distinguishable
    # within their single shared frame.
    if h_dem_global and h_dhdt_global:
        ax_leg3.plot([0.5, 0.5], [0.08, 0.92], transform=ax_leg3.transAxes,
                     color='gray', linewidth=1, alpha=0.4)

    # Match the legend frame's width/x-position to the map's actual plotted
    # extent (not the wider gridspec column, which also includes the margin
    # reserved for the map's tick labels). Done last, once all 8 glacier panels
    # exist, so it reflects the fully-resolved layout rather than an
    # intermediate one.
    _match_legend_panel_to_map(fig3, ax_map3, ax_leg3)

    out_path3 = os.path.join(cfg.CENTERLINE_PLOT_DIR,
                              "combined_9panel_dem_and_dhdt_profiles_satellite.png")
    plt.savefig(out_path3, dpi=300, bbox_inches='tight')
    plt.close(fig3)
    print(f"      ✅ Saved satellite-basemap layout: {os.path.basename(out_path3)}")

    # ------------------------------------------------------------------
    # (e) Satellite-basemap layout (portrait): map + legends span the
    #     full width at the top, 8 glacier panels fill a 2x4 grid below.
    # ------------------------------------------------------------------
    print(f"      Creating satellite-basemap layout (portrait)...")

    fig4 = plt.figure(figsize=(18, 30), constrained_layout=True)
    gs_outer4 = fig4.add_gridspec(2, 1, height_ratios=[2.3, 4.0])

    # Top block: map + legend row, sized and spaced independently of the
    # glacier-panel grid below (avoids gaps scaling oddly with uneven ratios)
    map_h_ratio4, leg_h_ratio4 = 5.2, 1.1
    gs_top4 = gs_outer4[0, 0].subgridspec(2, 1, height_ratios=[map_h_ratio4, leg_h_ratio4],
                                           hspace=0.02)
    ax_map4 = fig4.add_subplot(gs_top4[0, 0])
    ax_leg4 = fig4.add_subplot(gs_top4[1, 0])

    # Glacier-panel grid: own spacing, independent of the top block. Built
    # BEFORE the map/legend content: with constrained_layout, adding the rest
    # of the grid can globally rebalance spacing, so measuring the map's
    # available box (next) before this content exists would understate how much
    # width it will actually end up with.
    gs_glaciers4 = gs_outer4[1, 0].subgridspec(4, 2)

    for idx, (glacier_name, b_data) in enumerate(sorted_glaciers[:8]):
        row4 = idx // 2
        col4 = idx % 2
        gs_inner4 = gs_glaciers4[row4, col4].subgridspec(2, 1, height_ratios=[2, 1])
        ax_dem4 = fig4.add_subplot(gs_inner4[0])
        ax_dhdt4 = fig4.add_subplot(gs_inner4[1], sharex=ax_dem4)

        letter4 = chr(97 + idx + 1)
        render_glacier_dem_dhdt_panel(
            ax_dem4, ax_dhdt4, glacier_name, b_data.get('glacier_number', '?'), letter4,
            b_data, all_glacier_dhdt_data.get(glacier_name, {}),
            dem_colors, dhdt_colors,
            elevation_ylabel="Elevation (m.a.s.l.)",
            dhdt_ylim=shared_dhdt_ylim, elev_ylim=elev_ylim,
            shared_xlim=(shared_xmin_km, shared_xmax_km)
        )

    # Measured (not theoretical) physical aspect ratio of the box the map will
    # occupy - accounts for the real margin its tick labels/title reserve, so
    # the padding below doesn't overshoot and needlessly shrink the map (see
    # _measure_map_target_aspect / add_satellite_basemap_panel).
    target_aspect4 = _measure_map_target_aspect(fig4, ax_map4, 18.0, 30.0)
    add_satellite_basemap_panel(ax_map4, grid_info, all_glacier_bedrock_data,
                                 target_aspect=target_aspect4)

    # Single fused legend panel (one frame, full map width) instead of two
    # separate boxes.
    ax_leg4.set_xticks([])
    ax_leg4.set_yticks([])
    for spine in ax_leg4.spines.values():
        spine.set_visible(True)
        spine.set_edgecolor('black')
        spine.set_linewidth(1.1)
    ax_leg4.set_facecolor('white')

    if h_dem_global:
        leg_e4 = ax_leg4.legend(
            h_dem_global, l_dem_clean_global,
            loc='center left', bbox_to_anchor=(0.06, 0.5), fontsize=13,
            title="Surface elevation (m.a.s.l.)", title_fontsize=14,
            frameon=False, ncol=1,
            borderpad=0.6, labelspacing=0.6, handlelength=2.4, handletextpad=0.8
        )
        leg_e4.get_title().set_fontweight('bold')
        leg_e4._legend_box.align = "left"
        _left_align_legend_title(leg_e4)

    if h_dhdt_global:
        # Left-aligned, starting right next to the centre divider line
        # rather than centred or hugging the far right edge.
        leg_d4 = ax_leg4.legend(
            h_dhdt_global, l_dhdt_clean_global,
            loc='center left', bbox_to_anchor=(0.53, 0.5), fontsize=13,
            title="Elevation change rate (dhdt)", title_fontsize=14,
            frameon=False, ncol=1,
            borderpad=0.6, labelspacing=0.6, handlelength=2.4, handletextpad=0.8
        )
        leg_d4.get_title().set_fontweight('bold')
        leg_d4._legend_box.align = "left"
        _left_align_legend_title(leg_d4)

    # ax_leg4.legend() called a second time above detaches leg_e4 from the
    # axes (matplotlib only tracks one "primary" legend per axes) - re-add
    # it explicitly so both legends render together inside the one frame.
    if h_dem_global:
        ax_leg4.add_artist(leg_e4)

    # Faint divider so the two legend groups stay visually distinguishable
    # within their single shared frame.
    if h_dem_global and h_dhdt_global:
        ax_leg4.plot([0.5, 0.5], [0.08, 0.92], transform=ax_leg4.transAxes,
                     color='gray', linewidth=1, alpha=0.4)

    # Match the legend frame's width/x-position to the map's actual plotted
    # extent (not the wider gridspec column, which also includes the margin
    # reserved for the map's tick labels). Done last, once all 8 glacier panels
    # exist, so it reflects the fully-resolved layout rather than an
    # intermediate one.
    _match_legend_panel_to_map(fig4, ax_map4, ax_leg4)

    out_path4 = os.path.join(cfg.CENTERLINE_PLOT_DIR,
                              "combined_9panel_dem_and_dhdt_profiles_satellite_portrait.png")
    plt.savefig(out_path4, dpi=300, bbox_inches='tight')
    plt.close(fig4)
    print(f"      ✅ Saved satellite-basemap layout (portrait): {os.path.basename(out_path4)}")


def create_combined_dem_only_plot(all_glacier_bedrock_data, grid_info, preloaded_ref_dem):
    """Create a 9-panel plot showing reference DEM and surface profiles for all glaciers."""
    if not all_glacier_bedrock_data:
        return

    print(f"\n      Creating combined DEM-only plot...")
    n_cols, n_rows = 3, 3
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, 18))
    axes = axes.flatten()

    # ------------------------------------------------------------------
    # Panel 1: reference DEM map
    # ------------------------------------------------------------------
    ax_map = axes[0]
    try:
        mask_gdf = gpd.read_file(cfg.MASK_PATH).to_crs(grid_info['crs'])
        bounds = mask_gdf.total_bounds

        # Calculate extents based on master grid info
        minx, miny, maxx, maxy = bounds
        col_min, row_max = ~grid_info['transform'] * (minx, miny)
        col_max, row_min = ~grid_info['transform'] * (maxx, maxy)

        row_min = max(0, int(np.floor(row_min)))
        row_max = min(preloaded_ref_dem.shape[0], int(np.ceil(row_max)))
        col_min = max(0, int(np.floor(col_min)))
        col_max = min(preloaded_ref_dem.shape[1], int(np.ceil(col_max)))

        dem_cropped = preloaded_ref_dem[row_min:row_max, col_min:col_max]
        extent = [
            grid_info['transform'].c + col_min * grid_info['transform'].a,
            grid_info['transform'].c + col_max * grid_info['transform'].a,
            grid_info['transform'].f + row_max * grid_info['transform'].e,
            grid_info['transform'].f + row_min * grid_info['transform'].e
        ]

        vmin = np.nanpercentile(dem_cropped, 2)
        vmax = np.nanpercentile(dem_cropped, 98)

        im = ax_map.imshow(dem_cropped, cmap='terrain', vmin=vmin, vmax=vmax,
                           extent=extent, origin='upper', aspect='equal')
        cbar = plt.colorbar(im, ax=ax_map, orientation='horizontal', pad=0.22, shrink=0.8)
        cbar.set_label('Elevation [m]', fontsize=10)

        mask_gdf.boundary.plot(ax=ax_map, color='black', linewidth=1.5, zorder=5)

        # Draw Centerlines
        if os.path.exists(cfg.CENTERLINE_DIR):
            glacier_name_to_number = {k: v.get('glacier_number') for k, v in all_glacier_bedrock_data.items()}
            for centerline_file in glob.glob(os.path.join(cfg.CENTERLINE_DIR, "*.shp")):
                try:
                    cl_gdf = gpd.read_file(centerline_file).to_crs(grid_info['crs'])
                    cl_geom = get_longest_linestring(cl_gdf.unary_union)
                    if cl_geom is None: continue
                    cl_geom_plot = get_centerline_plot_segment(cl_geom)

                    ax_map.plot(*cl_geom_plot.xy, color='maroon', linewidth=2, zorder=10, alpha=0.8)

                    cl_name = os.path.splitext(os.path.basename(centerline_file))[0]
                    g_name = re.sub(r'_\d{2}$', '', cl_name.replace('Centerline_', '')).replace('_', ' ')

                    if g_name in glacier_name_to_number:
                        g_num = glacier_name_to_number[g_name]
                        midpoint = cl_geom_plot.interpolate(0.5, normalized=True)
                        ax_map.text(midpoint.x, midpoint.y, str(g_num),
                                    fontsize=9, fontweight='bold', color='white', ha='center', va='center', zorder=15,
                                    bbox=dict(boxstyle='circle,pad=0.15', facecolor='black', edgecolor='white',
                                              alpha=0.8))
                except:
                    pass

        ax_map.set_title(f"{cfg.REFERENCE_DEM_LABEL} with centerlines", fontsize=12, fontweight='bold')
        ax_map.set_xlabel("Easting (m)", fontsize=10, labelpad=10)
        ax_map.set_ylabel("Northing (m)", fontsize=10)
        ax_map.ticklabel_format(style='plain', useOffset=False)
        format_coord_labels(ax_map)

    except Exception as e:
        print(f"      ⚠️  Could not create Ref DEM map: {e}")
        ax_map.axis('off')

    # ------------------------------------------------------------------
    # Panels 2-9: glacier DEM profiles
    # ------------------------------------------------------------------
    # Shared distance (x-axis) range across all plotted glaciers -- see the
    # matching block in create_combined_bedrock_plot / create_combined_dem_and_dhdt_9panel.
    _panel_glaciers = list(all_glacier_bedrock_data.items())[:len(axes) - 1]
    _all_min_dist = [
        gd['global_min_dist'] for _, gd in _panel_glaciers
        if gd.get('global_min_dist') is not None
    ]
    _all_max_dist = [
        gd['global_max_dist'] for _, gd in _panel_glaciers
        if gd.get('global_max_dist') is not None
    ]
    if _all_min_dist and _all_max_dist:
        shared_xmin_km = max(0.0, min(_all_min_dist) / 1000.0)
        shared_xmax_km = min(cfg.CENTERLINE_MAX_DISTANCE / 1000.0, max(_all_max_dist) / 1000.0)
        if shared_xmax_km <= shared_xmin_km:
            shared_xmin_km, shared_xmax_km = 0.0, cfg.CENTERLINE_MAX_DISTANCE / 1000.0
    else:
        shared_xmin_km, shared_xmax_km = 0.0, cfg.CENTERLINE_MAX_DISTANCE / 1000.0

    for idx, (glacier_name, glacier_data) in enumerate(all_glacier_bedrock_data.items()):
        if idx + 1 >= len(axes): break
        ax = axes[idx + 1]

        surface_data = glacier_data.get('surface_data', [])
        terminus_positions = glacier_data.get('terminus_positions', {})
        dem_colormap = glacier_data.get('dem_colormap', build_years_colormap(max(len(surface_data), 1)))
        glacier_number = glacier_data.get('glacier_number', idx + 1)

        xmin_km = shared_xmin_km
        xmax_km = shared_xmax_km

        for surf_idx, (year, distances_plot, values_plot) in enumerate(surface_data):
            color_idx = min(surf_idx, len(dem_colormap) - 1)
            ax.plot(distances_plot / 1000.0, values_plot, color=dem_colormap[color_idx],
                    linewidth=2, label=f'Surface {get_display_year(year)}', alpha=0.9, zorder=3)

        for mask_year in sorted(terminus_positions.keys()):
            position_km = terminus_positions[mask_year] / 1000.0
            if not (xmin_km <= position_km <= xmax_km): continue
            ax.axvline(x=position_km, color=terminus_color(mask_year, dem_panels=True),
                       linestyle='--', linewidth=1.5, alpha=0.7, zorder=10)

            label_y = -10 + 0.05 * (500 - (-10))
            ax.text(position_km, label_y, mask_year, rotation=90, va='bottom', ha='right',
                    fontsize=8, fontweight='bold', color=terminus_color(mask_year, dem_panels=True),
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8, edgecolor='none'))

        ax.set_xlabel("Distance along flowline (km)", fontsize=11)
        ax.set_ylabel("Elevation [m]", fontsize=11)
        ax.set_title(f"[{glacier_number}] {glacier_name}", fontsize=12, fontweight='bold')
        ax.grid(True, linestyle='--', alpha=0.4)

        ax.set_xlim(xmin_km if xmax_km > xmin_km else 0,
                    xmax_km if xmax_km > xmin_km else cfg.CENTERLINE_MAX_DISTANCE / 1000.0)
        ax.set_ylim(-10, 500)

        ax.legend(loc='upper left', fontsize=9).set_zorder(20)

    for idx in range(len(all_glacier_bedrock_data) + 1, 9):
        axes[idx].axis('off')

    plt.suptitle('Surface Elevation Profiles - All Glaciers', fontsize=16, fontweight='bold', y=0.995)
    plt.tight_layout()

    out_path = os.path.join(cfg.CENTERLINE_DEM_PLOT_DIR, "combined_dem_profiles.png")
    plt.savefig(out_path, dpi=cfg.DPI, bbox_inches='tight')
    plt.close()
    print(f"      ✅ Saved combined DEM plot: {os.path.basename(out_path)}")
