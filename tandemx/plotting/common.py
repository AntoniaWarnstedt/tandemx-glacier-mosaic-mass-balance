"""Shared plotting helpers: coordinate labels, colour ramps, satellite basemap,
scale bar, north arrow and legend layout.
"""

import glob
import os
import re

import geopandas as gpd
import matplotlib.colors as mcolors
import matplotlib.offsetbox as moffsetbox
import matplotlib.patheffects as pe
import matplotlib.ticker as mticker
import numpy as np

from .. import config as cfg
from .centerline_geometry import get_centerline_plot_segment
from ..utils import get_longest_linestring

try:
    import contextily as cx
    HAS_CONTEXTILY = True
except ImportError:
    HAS_CONTEXTILY = False


def format_coord_labels(ax, target_crs=None):
    """Format axis labels to show degrees instead of projected coordinates."""
    from pyproj import Transformer

    target_crs = target_crs or cfg.CRS
    # Get current tick locations
    xticks = ax.get_xticks()
    yticks = ax.get_yticks()

    # Create transformer from projected CRS to WGS84 (lat/lon)
    transformer = Transformer.from_crs(target_crs, "EPSG:4326", always_xy=True)

    # Convert x-ticks (Easting to Longitude)
    xlabels = []
    for x in xticks:
        # Use middle of plot for y-coordinate during conversion
        y_mid = (ax.get_ylim()[0] + ax.get_ylim()[1]) / 2
        lon, lat = transformer.transform(x, y_mid)
        xlabels.append(f"{lon:.2f}°")

    # Convert y-ticks (Northing to Latitude)
    ylabels = []
    for y in yticks:
        # Use middle of plot for x-coordinate during conversion
        x_mid = (ax.get_xlim()[0] + ax.get_xlim()[1]) / 2
        lon, lat = transformer.transform(x_mid, y)
        ylabels.append(f"{lat:.2f}°")

    ax.set_xticklabels(xlabels)
    ax.set_yticklabels(ylabels)
    ax.set_xlabel("Longitude", fontsize=12)
    ax.set_ylabel("Latitude", fontsize=12)


def build_years_colormap(num_years):
    """Create a smooth colormap."""
    continuous = mcolors.LinearSegmentedColormap.from_list("blue_purple_pink", cfg.GRADIENT_COLORS, N=512)
    sample_points = np.linspace(0, 1, num_years)
    rgba = continuous(sample_points)
    return rgba


def hyps_bin_areas_km2(res, stats):
    """Per-elevation-bin areas (km^2) for the hypsometric area bars."""
    px_km2 = (float(cfg.RESOLUTION) ** 2) * cfg.M2_TO_KM2
    x = stats["cl"].values.astype(float)
    out = {
        "x": x,
        "obs": stats["n_raw"].fillna(0).astype(float).values * px_km2,
        "fil": stats["n_fil"].fillna(0).astype(float).values * px_km2,
        "x_ref": None,
        "ref": None,
    }
    ref = res.get("ref_hyps_bins") if res is not None else None
    if ref is not None and len(ref):
        ref = ref.sort_values("cl")
        out["x_ref"] = ref["cl"].values.astype(float)
        out["ref"] = ref["area_km2"].values.astype(float)
    else:
        print("      WARNING: result has no 'ref_hyps_bins' (re-run the mass-balance "
              "step); reference-area bars omitted.")
    return out


def add_satellite_basemap_panel(ax, grid_info, all_glacier_bedrock_data, mask_path=None,
                                 target_aspect=None, zoom_to_centerlines=True,
                                 centerline_zoom_pad_frac=0.18,
                                 centerline_zoom_pad_min_m=300.0):
    """
    Render a satellite-imagery basemap (Esri World Imagery via contextily,
    including the required service-layer attribution) into `ax`, with the
    glacier outline and numbered centerlines drawn on top.
    """
    mask_path = mask_path or cfg.MASK_PATH
    try:
        mask_gdf = gpd.read_file(mask_path).to_crs(grid_info['crs'])

        # --------------------------------------------------------------
        # Load centerlines once, up front.
        # --------------------------------------------------------------
        centerline_records = []  # list of (glacier_name, longest_linestring_geom)
        if cfg.CENTERLINE_DIR and os.path.exists(cfg.CENTERLINE_DIR):
            for centerline_file in glob.glob(os.path.join(cfg.CENTERLINE_DIR, "*.shp")):
                cl_gdf = gpd.read_file(centerline_file).to_crs(grid_info['crs'])
                centerline_name = os.path.splitext(os.path.basename(centerline_file))[0]
                g_name = centerline_name.replace('Centerline_', '')
                g_name = re.sub(r'_\d{2}$', '', g_name).replace('_', ' ')

                cl_geom = get_longest_linestring(cl_gdf.unary_union)
                if cl_geom:
                    centerline_records.append((g_name, cl_geom))

        if zoom_to_centerlines and centerline_records:
            bounds = np.array([geom.bounds for _, geom in centerline_records])
            minx, miny = bounds[:, 0].min(), bounds[:, 1].min()
            maxx, maxy = bounds[:, 2].max(), bounds[:, 3].max()
            span = max(maxx - minx, maxy - miny)
            pad = max(span * centerline_zoom_pad_frac, centerline_zoom_pad_min_m)
            minx, maxx = minx - pad, maxx + pad
            miny, maxy = miny - pad, maxy + pad
        else:
            minx, miny, maxx, maxy = mask_gdf.total_bounds

        if target_aspect and target_aspect > 0:
            data_w = maxx - minx
            data_h = maxy - miny
            data_aspect = data_w / data_h if data_h > 0 else target_aspect

            if data_aspect < target_aspect:
                # Box is relatively wider than the data: zoom out
                # horizontally (pad x symmetrically) to match.
                new_w = target_aspect * data_h
                pad = (new_w - data_w) / 2.0
                minx, maxx = minx - pad, maxx + pad
            else:
                # Box is relatively taller than the data: zoom out
                # vertically (pad y symmetrically) to match.
                new_h = data_w / target_aspect
                pad = (new_h - data_h) / 2.0
                miny, maxy = miny - pad, maxy + pad

        ax.set_xlim(minx, maxx)
        ax.set_ylim(miny, maxy)
        ax.xaxis.set_major_locator(mticker.MaxNLocator(5))
        ax.yaxis.set_major_locator(mticker.MaxNLocator(5))
        # True geographic proportions - no non-uniform stretching.
        ax.set_aspect('equal')

        if HAS_CONTEXTILY:
            try:
                cx.add_basemap(
                    ax, crs=grid_info['crs'], source=cx.providers.Esri.WorldImagery,
                    attribution_size=6, zoom='auto'
                )
            except Exception as e_tiles:
                print(f"      ⚠️  Could not fetch satellite basemap tiles: {e_tiles}")
        else:
            print("      ⚠️  contextily not installed — satellite basemap skipped "
                  "(pip install contextily). Panel will show glacier outline only.")

        # Black-on-white halo so the outline stays visible over both bright
        # ice and dark rock/ocean in the underlying imagery.
        outline_effect = [pe.withStroke(linewidth=3, foreground='white')]
        mask_gdf.boundary.plot(ax=ax, color='black', linewidth=1.5, zorder=5,
                               path_effects=outline_effect)

        glacier_name_to_number = {name: data.get('glacier_number', 0) for name, data in
                                  all_glacier_bedrock_data.items()}
        for g_name, cl_geom in centerline_records:
            cl_geom_plot = get_centerline_plot_segment(cl_geom)
            ax.plot(*cl_geom_plot.xy, color='maroon', linewidth=2.2, zorder=10, alpha=0.9,
                    path_effects=[pe.withStroke(linewidth=3.5, foreground='white')])
            if g_name in glacier_name_to_number:
                midpoint = cl_geom_plot.interpolate(0.5, normalized=True)
                ax.text(midpoint.x, midpoint.y, str(glacier_name_to_number[g_name]),
                        fontsize=11, fontweight='bold', color='white', ha='center', va='center', zorder=15,
                        bbox=dict(boxstyle='circle,pad=0.15', facecolor='black', edgecolor='white',
                                  alpha=0.85))

        ax.set_aspect('equal')

        ax.set_title('(a) Satellite Imagery & Centerlines', fontsize=14, fontweight='bold')
        ax.set_ylabel("Northing (m)", fontsize=12)
        format_coord_labels(ax)
        # format_coord_labels sets both axis labels/ticks - override just the
        # font sizes here to keep them consistent with the rest of the figure.
        ax.xaxis.label.set_fontsize(12)
        ax.yaxis.label.set_fontsize(12)
        ax.tick_params(axis='both', labelsize=11, pad=2)

        # North arrow + scale bar last, once xlim/ylim are final, so the scale
        # bar's length and both elements' positions are correct for whatever
        # extent ended up being displayed above. Both drawn in black (with a
        # white halo for legibility over dark ocean/rock).
        add_north_arrow(ax, grid_info, location='upper right', color='black',
                         outline_color='white', fontsize=13)
        add_scale_bar(ax, location='lower right', color='black',
                      outline_color='white', fontsize=10)

        return True

    except Exception as e:
        print(f"      ⚠️  Could not create satellite map panel: {e}")
        return False


def _nice_scale_bar_length_km(view_width_m, target_frac=0.22):
    candidates_km = [0.1, 0.2, 0.25, 0.5, 1, 2, 2.5, 5, 10, 20, 25, 50, 100, 200]
    if view_width_m <= 0 or not np.isfinite(view_width_m):
        return 1.0

    target_m = view_width_m * target_frac
    best = min(candidates_km, key=lambda c: abs(np.log(c * 1000.0) - np.log(max(target_m, 1.0))))

    # Guard rail: never let the bar exceed ~50% of the view width, even if
    # the nearest "nice" candidate above would.
    idx = candidates_km.index(best)
    while best * 1000.0 > view_width_m * 0.5 and idx > 0:
        idx -= 1
        best = candidates_km[idx]
    return best


def add_scale_bar(ax, location='lower left', frac_inset=0.06, target_frac=0.22,
                   color='white', outline_color='black', fontsize=8,
                   fixed_length_km=None):
    """
    Draw a simple, dependency-free scale bar (labelled in km) inside `ax`,
    sized from the axes' CURRENT xlim (assumed to be in metres, i.e. a
    projected CRS) so its length stays correct regardless of zoom level.
    """
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    view_w = xlim[1] - xlim[0]
    view_h = ylim[1] - ylim[0]

    bar_km = fixed_length_km if fixed_length_km is not None else \
        _nice_scale_bar_length_km(abs(view_w), target_frac=target_frac)
    bar_m = bar_km * 1000.0

    inset_x = frac_inset * abs(view_w)
    inset_y = frac_inset * abs(view_h)

    x_lo, x_hi = min(xlim), max(xlim)
    y_lo, y_hi = min(ylim), max(ylim)

    if 'left' in location:
        x0 = x_lo + inset_x
        x1 = x0 + bar_m
    else:
        x1 = x_hi - inset_x
        x0 = x1 - bar_m

    y0 = (y_lo + inset_y) if 'lower' in location else (y_hi - inset_y)

    outline = [pe.withStroke(linewidth=2.5, foreground=outline_color)]
    cap_h = inset_y * 0.35

    ax.plot([x0, x1], [y0, y0], color=color, linewidth=2.5, zorder=20,
            solid_capstyle='butt', path_effects=outline)
    for xc in (x0, x1):
        ax.plot([xc, xc], [y0 - cap_h, y0 + cap_h], color=color, linewidth=2.5,
                zorder=20, path_effects=outline)

    label = f"{bar_km:g} km"
    text_va = 'bottom' if 'lower' in location else 'top'
    text_y = y0 + inset_y * 0.35 if 'lower' in location else y0 - inset_y * 0.35
    ax.text((x0 + x1) / 2.0, text_y, label, color=color, fontsize=fontsize,
            fontweight='bold', ha='center', va=text_va, zorder=20,
            path_effects=[pe.withStroke(linewidth=2.5, foreground=outline_color)])


def compute_true_north_unit_vector(center_xy, crs, delta_lat_deg=0.01):
    """
    Compute the unit vector (dx, dy), expressed in the given projected `crs`'s
    own units, that points toward TRUE geographic north from `center_xy` (an
    (x, y) tuple in that same crs).
    """
    from shapely.geometry import Point

    cx, cy = center_xy
    pt_proj = gpd.GeoSeries([Point(cx, cy)], crs=crs)
    pt_geo = pt_proj.to_crs("EPSG:4326")
    lon0 = float(pt_geo.geometry.iloc[0].x)
    lat0 = float(pt_geo.geometry.iloc[0].y)

    pt2_geo = gpd.GeoSeries([Point(lon0, lat0 + delta_lat_deg)], crs="EPSG:4326")
    pt2_proj = pt2_geo.to_crs(crs)
    x2 = float(pt2_proj.geometry.iloc[0].x)
    y2 = float(pt2_proj.geometry.iloc[0].y)

    vec = np.array([x2 - cx, y2 - cy], dtype=float)
    norm = np.linalg.norm(vec)
    if not np.isfinite(norm) or norm == 0:
        return np.array([0.0, 1.0])  # fallback: assume grid-up
    return vec / norm


def add_north_arrow(ax, grid_info, location='upper right', frac_inset=0.08,
                     frac_inset_x=None, frac_inset_y=None,
                     frac_length=0.10, color='black', outline_color='white',
                     fontsize=11, style='line'):
    """
    Draw a north arrow pointing toward TRUE north (not just grid "up") in a
    corner of `ax`, sized and positioned from the axes' CURRENT xlim/ylim so
    both its length and its ANGLE render correctly on the page regardless of
    the map's zoom level.
    """
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    x_lo, x_hi = min(xlim), max(xlim)
    y_lo, y_hi = min(ylim), max(ylim)
    view_w = x_hi - x_lo
    view_h = y_hi - y_lo
    view_dim = min(view_w, view_h)

    inset_x = (frac_inset_x if frac_inset_x is not None else frac_inset) * view_w
    inset_y = (frac_inset_y if frac_inset_y is not None else frac_inset) * view_h
    anchor_x = (x_hi - inset_x) if 'right' in location else (x_lo + inset_x)
    anchor_y = (y_hi - inset_y) if 'upper' in location else (y_lo + inset_y)

    center_xy = ((x_lo + x_hi) / 2.0, (y_lo + y_hi) / 2.0)
    try:
        north_unit = compute_true_north_unit_vector(center_xy, grid_info['crs'])
    except Exception as e:
        print(f"      ⚠️  Could not compute true-north direction, falling back "
              f"to grid-up: {e}")
        north_unit = np.array([0.0, 1.0])

    arrow_len = frac_length * view_dim
    anchor = np.array([anchor_x, anchor_y])
    tip = anchor + north_unit * arrow_len

    if style == 'solid':
        # Filled compass-style arrowhead: a triangle with a shallow V-notch cut
        # into its base, built entirely in DATA coordinates (same trick as the
        # 'line' style below) so its angle stays correct regardless of the
        # map's zoom/aspect.
        from matplotlib.patches import Polygon
        perp = np.array([-north_unit[1], north_unit[0]])  # 90 deg rotation
        half_width = 0.34 * arrow_len
        notch = anchor + north_unit * (arrow_len * 0.30)
        pts = [tip, anchor + perp * half_width, notch, anchor - perp * half_width]
        arrow_patch = Polygon(pts, closed=True, facecolor=color,
                               edgecolor=color, linewidth=0.5, zorder=25)
        arrow_patch.set_path_effects([pe.withStroke(linewidth=3, foreground=outline_color)])
        ax.add_patch(arrow_patch)
    else:
        arrow_common = dict(xy=tuple(tip), xytext=tuple(anchor), xycoords='data',
                             textcoords='data', zorder=25)
        # Thicker "outline" arrow first, then the colored arrow on top - keeps
        # it visible over both bright ice and dark rock/ocean imagery.
        ax.annotate('', arrowprops=dict(arrowstyle='-|>', color=outline_color,
                                         lw=4.5, shrinkA=0, shrinkB=0), **arrow_common)
        ax.annotate('', arrowprops=dict(arrowstyle='-|>', color=color,
                                         lw=2.0, shrinkA=0, shrinkB=0), **arrow_common)

    label_xy = (anchor_x + north_unit[0] * arrow_len * 1.22,
                anchor_y + north_unit[1] * arrow_len * 1.22)
    ax.text(label_xy[0], label_xy[1], 'N', color=color, fontsize=fontsize,
            fontweight='bold', ha='center', va='center', zorder=25,
            path_effects=[pe.withStroke(linewidth=3, foreground=outline_color)])


def _measure_map_target_aspect(fig, ax_map, fig_w_in, fig_h_in):
    """
    Returns the physical (inches) width/height aspect ratio of the box `ax_map`
    will actually occupy, accounting for the margin constrained_layout reserves
    for its y-axis tick labels/"Latitude" label and title - NOT just a
    theoretical figsize/grid-ratio estimate, which ignores that margin and so
    systematically overestimates the available width (causing the subsequent
    aspect='equal' padding to overshoot and shrink the map well below the space
    it actually has).
    """
    ax_map.set_ylabel("Latitude", fontsize=12)
    ax_map.set_xlabel("Longitude", fontsize=12)
    ax_map.set_yticks([0, 1, 2, 3, 4])
    ax_map.set_yticklabels(["-69.25°", "-69.33°", "-69.40°", "-69.47°", "-69.55°"])
    ax_map.set_xticks([0, 1, 2, 3, 4])
    ax_map.set_xticklabels(["-66.75°", "-66.65°", "-66.55°", "-66.45°", "-66.35°"])
    ax_map.tick_params(axis='both', labelsize=11, pad=2)
    ax_map.set_title('(a) Satellite Imagery & Centerlines', fontsize=14, fontweight='bold')
    fig.canvas.draw()
    pos = ax_map.get_position()
    return (pos.width * fig_w_in) / (pos.height * fig_h_in)


def _left_align_legend_title(legend, ax_bound=None, max_frac=None):
    """
    Even with `legend._legend_box.align = "left"`, matplotlib aligns a legend's
    title with the left edge of the *whole* handle+label row - i.e. flush with
    the colour-swatch/handle, not with where the label TEXT actually starts.
    """
    fig = legend.axes.figure
    fig.canvas.draw()
    title_text = legend.get_title()
    labels = legend.get_texts()
    if not labels:
        return
    title_bbox = title_text.get_window_extent()
    ideal_shift_px = labels[0].get_window_extent().x0 - title_bbox.x0
    if ideal_shift_px <= 0:
        return
    shift_px = ideal_shift_px
    if ax_bound is not None and max_frac is not None:
        ax_bbox = ax_bound.get_window_extent()
        max_x1_px = ax_bbox.x0 + max_frac * ax_bbox.width
        available_px = max_x1_px - title_bbox.x1
        shift_px = max(0.0, min(ideal_shift_px, available_px))
    if shift_px <= 0:
        return
    shift_pts = shift_px * 72.0 / fig.dpi
    children = legend._legend_box.get_children()
    spacer = moffsetbox.DrawingArea(shift_pts, 0, 0, 0)
    children[0] = moffsetbox.HPacker(children=[spacer, children[0]], align="baseline", pad=0, sep=0)
    legend._legend_box.get_children()[:] = children
    legend._legend_box.stale = True
    fig.canvas.draw()


def _match_legend_panel_to_map(fig, ax_map, ax_leg):
    """
    Resize/reposition a companion legend axes so its frame spans exactly the
    map axes' actual plotted extent (the image itself), rather than the full
    gridspec column width - which also includes the margin reserved for the
    map's y-axis tick labels and "Latitude" label. Must be called after the map
    axes' labels/ticks/title are fully set, and after all other subplots in the
    figure have been added.
    """
    fig.canvas.draw()
    map_pos = ax_map.get_position()
    leg_pos = ax_leg.get_position()
    ax_leg.set_position([map_pos.x0, leg_pos.y0, map_pos.width, leg_pos.height])
    ax_map.set_position(map_pos)
    ax_map.set_in_layout(False)
    ax_leg.set_in_layout(False)


def with_bedrock_source(text, sep=" - "):
    """Append cfg.BEDROCK_SOURCE to a label, if set."""
    return f"{text}{sep}{cfg.BEDROCK_SOURCE}" if cfg.BEDROCK_SOURCE else text


def terminus_color(year, dem_panels=False, default="black"):
    """Colour of the terminus line/segment of an outline year.

    Uses cfg.TERMINUS_COLORS (or cfg.TERMINUS_COLORS_DEM_PANELS); years not
    listed get a grey ramp from light (first year) to black (last year).
    """
    colors = cfg.TERMINUS_COLORS
    if dem_panels and cfg.TERMINUS_COLORS_DEM_PANELS:
        colors = cfg.TERMINUS_COLORS_DEM_PANELS
    if year in colors:
        return colors[year]
    years = sorted(cfg.YEAR_MASKS, key=int)
    if year in years and len(years) > 1:
        return str(round(0.75 * (1 - years.index(year) / (len(years) - 1)), 3))
    return default
