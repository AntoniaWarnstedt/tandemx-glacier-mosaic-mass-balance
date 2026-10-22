"""Summary figures: 3x3 maps (DEMs, dh/dt, void-filled dh/dt) and 1x3
hypsometry (catchment/observed area per elevation bin with dh/dt)."""

import os

import geopandas as gpd
import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
import numpy as np
import rasterio

from .. import config as cfg
from ..inventory import (
    build_union_outline_gdf,
    get_period_inventory_exclude_mask,
    get_period_inventory_outline_gdf,
)
from .centerline_geometry import get_centerline_plot_segment, glacier_name_from_file, list_centerline_files
from .common import add_north_arrow, add_scale_bar, format_coord_labels, hyps_bin_areas_km2
from ..utils import (
    mask_year_for_label,
    summary_epochs,
    format_period_label,
    get_display_year,
    get_longest_linestring,
)


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

    # Numbered centerlines for the dh/dt maps (same numbering as the centerline figure)
    centerline_records = []  # list of (glacier_name, glacier_number, geom)
    for glacier_idx, centerline_file in enumerate(list_centerline_files(), start=1):
        glacier_name = glacier_name_from_file(centerline_file)
        try:
            cl_gdf = gpd.read_file(centerline_file).to_crs(grid_info['crs'])
            cl_geom = get_longest_linestring(cl_gdf.unary_union)
        except Exception as e:
            print(f"      ⚠️ Could not load centerline {centerline_file}: {e}")
            continue
        if cl_geom is not None:
            centerline_records.append((glacier_name, glacier_idx, cl_geom))
    print(f"      Loaded {len(centerline_records)} centerlines for the dh/dt maps")

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
