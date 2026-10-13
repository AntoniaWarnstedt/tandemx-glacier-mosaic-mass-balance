"""Combined cross-sensor figures: one figure per plot type with the panels
CryoSat-2 (All Years + epochs), ICESat-2 and
CryoSat-2 dh/dt (per epoch pair).
"""

from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
from matplotlib.patches import Patch

from . import config as cfg
from .common import (
    display_epoch,
    epoch_years,
    pooled_epochs,
    compute_binned_stats,
    compute_hypsometry,
    grid_cell_index,
    is2_quality_columns,
    sample_slope_from_own_dem,
    sample_slope_from_own_dem_multi,
)

plt.rcParams.update(cfg.RC_PARAMS)
_MIXED_SLOPE_BIN_DEG = 5.0  # must match the 5 deg banding in _slot_slope_stats()


def _style_ax(ax):
    """Consistent science-paper axis styling."""
    ax.grid(True, **cfg.GRID_KW)
    ax.set_axisbelow(True)
    for sp in ax.spines.values():
        sp.set_color(cfg.SPINE_COLOR)
        sp.set_linewidth(1.0)
    ax.tick_params(**cfg.TICK_KW)


def _cs2_slot(cs2_results, key: str, label: str):
    """Build a unified 'slot' dict for one CS2 absolute-elevation panel."""
    comp = cs2_results.get(key)
    if comp is None or comp['gdf'].empty:
        return None
    gdf = comp['gdf']
    elev_col = 'elevation_mean' if 'elevation_mean' in gdf.columns else 'elevation'
    gdf_ll = gdf.to_crs(epsg=4326) if (gdf.crs is not None and gdf.crs.is_projected) else gdf
    temp_col = ('temp_off_mean' if 'temp_off_mean' in gdf.columns else
                'temp_off' if 'temp_off' in gdf.columns else None)
    qual_col = 'mean_unc' if 'mean_unc' in gdf.columns else ('elev_unc' if 'elev_unc' in gdf.columns else None)
    # Co-located mode: 'tdx_slope' = mean slope of this epoch's own DEM at
    # the altimeter points in each cell (same locations as tdx_elev).
    slope_vals = (gdf['tdx_slope'].values.astype(float) if 'tdx_slope' in gdf.columns
                  else sample_slope_from_own_dem_multi(gdf, path_col='tdx_file'))
    return dict(
        sensor='CryoSat-2', label=label, unit='m', key=key,
        x_proj=gdf.geometry.x.values, y_proj=gdf.geometry.y.values, crs=gdf.crs,
        cell_size=(cfg.AGGREGATION_GRID_SIZE_M if comp.get('aggregated') else None),
        x=gdf['tdx_elev'].values.astype(float), y=gdf[elev_col].values.astype(float),
        diff=gdf['diff'].values.astype(float),
        lons=gdf_ll.geometry.x.values, lats=gdf_ll.geometry.y.values,
        elev=gdf['tdx_elev'].values.astype(float),
        slope=(slope_vals if slope_vals is not None else None),
        temporal_offset=(gdf[temp_col].values.astype(float) if temp_col else None),
        quality=(gdf[qual_col].values.astype(float) if qual_col else None),
    )


def _is2_slot(is2_results, key: str, label: str):
    """Build a unified 'slot' dict for one ICESat-2 absolute-elevation panel."""
    comp = is2_results.get(key)
    if comp is None or comp['gdf'].empty:
        return None
    gdf = comp['gdf']
    is2_col = comp['is2_col']
    gdf_ll = gdf.to_crs(epsg=4326) if (gdf.crs is not None and gdf.crs.is_projected) else gdf
    qual_col = next(iter(is2_quality_columns(gdf.columns)), None)
    slope_vals = (gdf['tdx_slope'].values.astype(float) if 'tdx_slope' in gdf.columns
                  else sample_slope_from_own_dem(gdf, comp.get('dem_path')))
    return dict(
        sensor='ICESat-2', label=label, unit='m', key=key,
        x_proj=gdf.geometry.x.values, y_proj=gdf.geometry.y.values, crs=gdf.crs,
        cell_size=(cfg.AGGREGATION_GRID_SIZE_M if 'n_points' in gdf.columns else None),
        x=gdf['tdx_h'].values.astype(float), y=gdf[is2_col].values.astype(float),
        diff=gdf['dh'].values.astype(float),
        lons=gdf_ll.geometry.x.values, lats=gdf_ll.geometry.y.values,
        elev=gdf['tdx_h'].values.astype(float),
        slope=(slope_vals if slope_vals is not None else None),
        temporal_offset=(gdf['temporal_offset_days'].values.astype(float)
                          if 'temporal_offset_days' in gdf.columns else None),
        quality=(gdf[qual_col].values.astype(float) if qual_col else None),
    )


def _dhdt_slot(dhdt_results, key: str, label: str):
    """Build a unified 'slot' dict for one CryoSat-2 dhdt panel."""
    res = dhdt_results.get(key)
    if res is None:
        return None
    return dict(
        sensor='CryoSat-2 dhdt', label=label, unit='m a⁻¹', key=key,
        x_proj=res.get('x_proj'), y_proj=res.get('y_proj'), crs=res.get('crs'),
        cell_size=cfg.DHDT_GRID_SIZE_M,
        x=res['tdx'], y=res['cs2'], diff=res['diff'],
        lons=res.get('lons'), lats=res.get('lats'),
        elev=res.get('elev'), slope=res.get('slope'),
        temporal_offset=None, quality=None,
    )


def _collect_combined_slots(results):
    """
    Panels of the combined figures as (cs2_slots, dhdt_slots, is2_slots):
    CryoSat-2 "All Years" + each pooled epoch, CryoSat-2 dh/dt per epoch pair,
    ICESat-2 per cfg.COMBINED_IS2_EPOCHS. Entries are None without data.
    """
    cs2_keys = [k for k in results['cs2'] if k != 'All_Years']
    cs2_slots = [_cs2_slot(results['cs2'], 'All_Years', 'All Years')]
    cs2_slots += [_cs2_slot(results['cs2'], k, display_epoch(k)) for k in pooled_epochs(cs2_keys)]
    dhdt_slots = []
    for key in sorted(results['dhdt'], key=lambda k: [epoch_years(e) for e in k.split('_minus_')]):
        late, early = key.split('_minus_')
        dhdt_slots.append(_dhdt_slot(results['dhdt'], key,
                                     f"{display_epoch(early)} - {display_epoch(late)}"))
    is2_epochs = cfg.COMBINED_IS2_EPOCHS
    if is2_epochs is None:
        is2_epochs = [str(cfg.IS2_YEARS[-1])] if cfg.IS2_YEARS else []
    is2_slots = [_is2_slot(results['is2'], str(k), display_epoch(k)) for k in is2_epochs]
    return cs2_slots, dhdt_slots, is2_slots


def _slot_hypsometry(slot, bin_width=cfg.HYPS_BIN_WIDTH_M, min_samples=cfg.HYPS_MIN_SAMPLES):
    """Elevation-banded bias table for one slot, reusing compute_hypsometry()
    on a small wrapper DataFrame so CS2/IS2/dhdt slots all go through the
    identical binning logic."""
    elev, diff = slot.get('elev'), slot['diff']
    if elev is None:
        return pd.DataFrame()
    df = pd.DataFrame({'e': elev, 'd': diff}).replace([np.inf, -np.inf], np.nan).dropna()
    if df.empty:
        return pd.DataFrame()
    return compute_hypsometry(df, elev_col='e', diff_col='d', bin_width=bin_width, min_samples=min_samples)


def _slot_slope_stats(slot, min_samples=cfg.HYPS_MIN_SAMPLES):
    """5-degree slope-banded bias table for one slot, reusing
    compute_binned_stats() the same way _slot_hypsometry() reuses
    compute_hypsometry()."""
    slope, diff = slot.get('slope'), slot['diff']
    if slope is None:
        return pd.DataFrame()
    df = pd.DataFrame({'s': slope, 'd': diff}).replace([np.inf, -np.inf], np.nan).dropna()
    if df.empty:
        return pd.DataFrame()
    return compute_binned_stats(df.assign(slope_band=(df['s'] // 5) * 5),
                                 bin_col='slope_band', diff_col='d', min_samples=min_samples)


def _no_data(ax, msg='no data'):
    ax.axis('off')
    ax.text(0.5, 0.5, msg, transform=ax.transAxes, ha='center', va='center',
            fontsize=15, color='#888888')


def _cells_to_image(x, y, values, cell_size, origin=None):
    """
    Grid per-cell values onto their own cell_size grid (same registration as
    the aggregation, AGGREGATION_ORIGIN_M) and return (image, extent) for
    imshow, with extent in the projection's metres. Cells holding more than one
    value (e.g. the All-Years panel, where the three epochs share cell
    locations) show the mean of those values.
    """
    ok = np.isfinite(values) & np.isfinite(x) & np.isfinite(y)
    xi, yi = grid_cell_index(x[ok], y[ok], cell_size, origin)
    v = np.asarray(values, dtype=float)[ok]
    x0, x1, y0, y1 = xi.min(), xi.max(), yi.min(), yi.max()
    h, w = int(y1 - y0 + 1), int(x1 - x0 + 1)
    s = np.zeros((h, w)); n = np.zeros((h, w))
    np.add.at(s, (y1 - yi, xi - x0), v); np.add.at(n, (y1 - yi, xi - x0), 1)
    with np.errstate(invalid='ignore', divide='ignore'):
        img = np.where(n > 0, s / np.maximum(n, 1), np.nan)
    ox, oy = cfg.AGGREGATION_ORIGIN_M if origin is None else origin
    extent = (ox + x0 * cell_size, ox + (x1 + 1) * cell_size,
              oy + y0 * cell_size, oy + (y1 + 1) * cell_size)
    return img, extent


def _draw_type_map(ax, slot, include_title=True, map_extent=None):
    diff = np.asarray(slot['diff'], dtype=float)
    vmax = 15 if slot['unit'] == 'm' else max(np.nanpercentile(np.abs(diff), 98), 0.1)
    xp, yp, cell = slot.get('x_proj'), slot.get('y_proj'), slot.get('cell_size')
    if xp is not None and len(xp) > 0 and cell:
        # True-size cells: each grid cell is a cell x cell square in the
        # TanDEM-X projection (polar stereographic), axes in km.
        img, (l, r, b, t) = _cells_to_image(np.asarray(xp, float), np.asarray(yp, float),
                                           diff, cell)
        sc = ax.imshow(img, extent=(l / 1000, r / 1000, b / 1000, t / 1000), origin='upper',
                       cmap='RdBu', vmin=-vmax, vmax=vmax, interpolation='nearest',
                       rasterized=True)
        if map_extent is not None:
            ax.set_xlim(map_extent[0], map_extent[1])
            ax.set_ylim(map_extent[2], map_extent[3])
        ax.set_aspect('equal')
        ax.set(xlabel='Easting (km)', ylabel='Northing (km)')
        ax.xaxis.set_major_locator(mticker.MaxNLocator(nbins=4))
        ax.yaxis.set_major_locator(mticker.MaxNLocator(nbins=5))
    elif slot['lons'] is not None and len(slot['lons']) > 0:
        # Fallback (raw, non-aggregated points): plain dots in lon/lat.
        sc = ax.scatter(slot['lons'], slot['lats'], c=diff, s=7, alpha=0.75,
                        cmap='RdBu', vmin=-vmax, vmax=vmax, rasterized=True)
        ax.set(xlabel='Longitude (deg)', ylabel='Latitude (deg)')
    else:
        return _no_data(ax)
    cb = plt.colorbar(sc, ax=ax, fraction=0.046, pad=0.04)
    cb.ax.tick_params(labelsize=13)
    cb.set_label(f"TanDEM-X minus sensor ({slot['unit']})", fontsize=15)
    if include_title:
        ax.set_title(f"{slot['sensor']}: {slot['label']}", fontsize=17, fontweight='bold', pad=8, color='black')
    _style_ax(ax)


def _draw_type_histogram(ax, slot, include_title=True):
    color = cfg.SENSOR_COLORS.get(slot['sensor'], cfg.RED)
    diff = slot['diff'][np.isfinite(slot['diff'])]
    if len(diff) == 0:
        return _no_data(ax)
    lim = 25 if slot['unit'] == 'm' else 15
    bins = np.linspace(-lim, lim, 81)
    ax.hist(diff, bins=bins, color=color, alpha=0.7, edgecolor='white', linewidth=0.3)
    ax.axvline(0, color=cfg.NEUTRAL_LINE_COLOR, ls='--', lw=1.4)
    ax.axvline(diff.mean(), color='#1a1a1a', ls='-', lw=1.6, label=f'Mean {diff.mean():.2f}')
    ax.set(xlim=(-lim, lim), xlabel=f"TanDEM-X minus sensor ({slot['unit']})", ylabel='Count')
    if include_title:
        ax.set_title(f"{slot['sensor']}: {slot['label']}", fontsize=17, fontweight='bold', pad=8, color='black')
    ax.legend(fontsize=13, framealpha=0.9, edgecolor='#cccccc')
    _style_ax(ax)


def _draw_type_scatter(ax, slot, include_title=True, shared_elev_lim=None):
    color = cfg.SENSOR_COLORS.get(slot['sensor'], cfg.RED)
    x, y, diff = slot['x'], slot['y'], slot['diff']
    valid = np.isfinite(x) & np.isfinite(y)
    x, y = x[valid], y[valid]
    if len(x) == 0:
        return _no_data(ax)
    if slot['unit'] == 'm a⁻¹':
        lo, hi = cfg.DHDT_SCATTER_LIM
    elif shared_elev_lim is not None:
        lo, hi = shared_elev_lim
    else:
        lo = min(np.percentile(x, 2), np.percentile(y, 2))
        hi = max(np.percentile(x, 98), np.percentile(y, 98))
    d = diff[valid]
    ax.scatter(x, y, s=18, marker='+', lw=0.5, alpha=0.9, c=color, rasterized=True)
    ax.plot([lo, hi], [lo, hi], color=cfg.NEUTRAL_LINE_COLOR, ls='--', lw=1.4, zorder=3)
    info = (f"N = {len(x):,}\nMean = {d.mean():.2f} {slot['unit']}\n"
            f"Std = {d.std():.2f} {slot['unit']}\nRMSE = {np.sqrt((d ** 2).mean()):.2f} {slot['unit']}")
    from matplotlib.offsetbox import AnchoredText
    _at = AnchoredText(info, loc='upper left', frameon=True, pad=0.3, borderpad=0.25,
                       prop=dict(fontsize=14, family='monospace'))
    _at.patch.set_boxstyle('round,pad=0.0,rounding_size=0.3')
    _at.patch.set(facecolor='white', edgecolor='#cccccc', alpha=0.9)
    _at.set_zorder(5)
    ax.add_artist(_at)
    xlabel = f"TanDEM-X dhdt ({slot['unit']})" if slot['unit'] == 'm a⁻¹' else f"TanDEM-X ({slot['unit']})"
    ax.set(xlim=(lo, hi), ylim=(lo, hi), xlabel=xlabel,
           ylabel=f"{slot['sensor']} ({slot['unit']})")
    if include_title:
        ax.set_title(f"{slot['sensor']}: {slot['label']}", fontsize=17, fontweight='bold', pad=8, color='black')
    ax.set_aspect('equal', 'box')
    _style_ax(ax)


def _draw_type_bias(ax, slot, include_title=True, ylim=None):
    color = cfg.SENSOR_COLORS.get(slot['sensor'], cfg.RED)
    elev, diff = slot.get('elev'), slot['diff']
    if elev is None:
        return _no_data(ax)
    valid = np.isfinite(elev) & np.isfinite(diff)
    if valid.sum() == 0:
        return _no_data(ax)
    e, d = elev[valid], diff[valid]
    ax.scatter(e, d, s=3, alpha=0.3, c=color, rasterized=True)
    ax.axhline(0, color=cfg.NEUTRAL_LINE_COLOR, ls='--', lw=1.4, zorder=3)
    ax.axhline(d.mean(), color='#1a1a1a', lw=1.6, zorder=4, label=f'Mean {d.mean():.2f}')
    ax.set(xlabel='Reference elevation (m)', ylabel=f"TanDEM-X minus sensor ({slot['unit']})")
    if slot['unit'] == 'm a⁻¹':
        ax.set_ylim(-15, 15)
    elif ylim is not None:
        ax.set_ylim(ylim)
    if include_title:
        ax.set_title(f"{slot['sensor']}: {slot['label']}", fontsize=17, fontweight='bold', pad=8, color='black')
    ax.legend(fontsize=13, framealpha=0.9, edgecolor='#cccccc', loc='upper right')
    _style_ax(ax)


def _draw_type_statistics(ax, slot, include_title=True):
    color = cfg.SENSOR_COLORS.get(slot['sensor'], cfg.RED)
    ax.axis('off')
    diff = slot['diff'][np.isfinite(slot['diff'])]
    if len(diff) == 0:
        ax.text(0.5, 0.5, 'no data', transform=ax.transAxes, ha='center', va='center', fontsize=15)
        return
    u = slot['unit']
    txt = (f"{slot['sensor']} — {slot['label']}\n" + '-' * 32 + '\n'
           f"N          {len(diff):>10,}\n"
           f"Mean       {diff.mean():>+10.2f} {u}\n"
           f"Median     {np.median(diff):>+10.2f} {u}\n"
           f"Std        {diff.std():>10.2f} {u}\n"
           f"RMSE       {np.sqrt((diff ** 2).mean()):>10.2f} {u}\n"
           f"MAE        {np.abs(diff).mean():>10.2f} {u}\n"
           f"5th pct    {np.percentile(diff, 5):>+10.2f} {u}\n"
           f"95th pct   {np.percentile(diff, 95):>+10.2f} {u}")
    ax.text(0.06, 0.90, txt, transform=ax.transAxes, fontfamily='monospace', fontsize=14,
            va='top', linespacing=1.5,
            bbox=dict(boxstyle='round,pad=0.5', fc='#f9f9f9', ec=color, linewidth=1.4))
    if include_title:
        ax.set_title(f"{slot['sensor']}: {slot['label']}", fontsize=17, fontweight='bold', pad=8, color='black')


def _draw_type_hypsometric(ax, slot, include_title=True, xlim=None, ylim=None,
                            show_xlabel=True, show_ylabel=True, show_right_label=True,
                            n_ylim=None):
    color = cfg.SENSOR_COLORS.get(slot['sensor'], cfg.RED)
    hyps = _slot_hypsometry(slot)
    if hyps.empty:
        return _no_data(ax)
    ax2 = ax.twinx()
    bars = ax2.bar(hyps['elev_mid'], hyps['n'], width=cfg.HYPS_BIN_WIDTH_M * 0.82, color=cfg.GREY, alpha=0.30,
                    zorder=1, label='N per bin')
    if show_right_label:
        ax2.set_ylabel('N', fontsize=15, color=cfg.GREY)
    ax2.tick_params(axis='y', labelsize=13, colors=cfg.GREY)
    if n_ylim is not None:
        ax2.set_ylim(n_ylim)
    for sp in ['left', 'top', 'bottom']:
        ax2.spines[sp].set_visible(False)
    ax.errorbar(hyps['elev_mid'], hyps['mean_diff'], yerr=hyps['std_diff'], fmt='o', ms=4.5,
                color=color, ecolor=color, elinewidth=1.2, capsize=3, zorder=3,
                label='Mean bias (±1 std)')
    ax.axhline(0, color=cfg.NEUTRAL_LINE_COLOR, ls='--', lw=1.4, zorder=2)
    if show_xlabel:
        ax.set_xlabel('Reference elevation (m)')
    if show_ylabel:
        ax.set_ylabel(f"Mean diff ({slot['unit']})")
    if xlim is not None:
        ax.set_xlim(xlim)
    if ylim is not None:
        ax.set_ylim(ylim)
    handles1, labels1 = ax.get_legend_handles_labels()
    handles2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(handles1 + handles2, labels1 + labels2, loc='best', fontsize=12,
              framealpha=0.9, edgecolor='#cccccc')
    if include_title:
        ax.set_title(f"{slot['sensor']}: {slot['label']}", fontsize=17, fontweight='bold', pad=8, color='black')
    _style_ax(ax)


def _draw_type_temporal_bias(ax, slot, include_title=True):
    color = cfg.SENSOR_COLORS.get(slot['sensor'], cfg.RED)
    t, diff = slot.get('temporal_offset'), slot['diff']
    if t is None:
        return _no_data(ax, 'no temporal-offset data')
    valid = np.isfinite(t) & np.isfinite(diff)
    if valid.sum() == 0:
        return _no_data(ax)
    ax.scatter(t[valid], diff[valid], s=5, alpha=0.3, c=color, rasterized=True)
    ax.axhline(0, color=cfg.NEUTRAL_LINE_COLOR, ls='--', lw=1.4)
    ax.set(xlabel='Temporal offset from TanDEM-X acquisition (days)', ylabel=f"Diff ({slot['unit']})")
    if include_title:
        ax.set_title(f"{slot['sensor']}: {slot['label']}", fontsize=17, fontweight='bold', pad=8, color='black')
    _style_ax(ax)


def _draw_type_quality_bias(ax, slot, include_title=True):
    color = cfg.SENSOR_COLORS.get(slot['sensor'], cfg.RED)
    q, diff = slot.get('quality'), slot['diff']
    if q is None:
        return _no_data(ax, 'no quality metric available')
    valid = np.isfinite(q) & np.isfinite(diff)
    if valid.sum() == 0:
        return _no_data(ax)
    ax.scatter(q[valid], diff[valid], s=5, alpha=0.3, c=color, rasterized=True)
    ax.axhline(0, color='red', ls='--', lw=1.2)
    ax.set(xlabel='Data-quality metric', ylabel=f"Diff ({slot['unit']})")
    if include_title:
        ax.set_title(f"{slot['sensor']}: {slot['label']}", fontsize=17, fontweight='bold', pad=8, color='black')
    _style_ax(ax)


def _draw_type_elevation_boxplot(ax, slot, include_title=True):
    color = cfg.SENSOR_COLORS.get(slot['sensor'], cfg.RED)
    elev, diff = slot.get('elev'), slot['diff']
    if elev is None:
        return _no_data(ax)
    valid = np.isfinite(elev) & np.isfinite(diff)
    if valid.sum() < cfg.HYPS_MIN_SAMPLES:
        return _no_data(ax)
    e, d = elev[valid], diff[valid]
    edges = np.arange(np.floor(e.min() / cfg.HYPS_BIN_WIDTH_M) * cfg.HYPS_BIN_WIDTH_M,
                       e.max() + cfg.HYPS_BIN_WIDTH_M, cfg.HYPS_BIN_WIDTH_M)
    groups, labels = [], []
    for lo, hi in zip(edges[:-1], edges[1:]):
        sel = (e >= lo) & (e < hi)
        if sel.sum() >= cfg.HYPS_MIN_SAMPLES:
            groups.append(d[sel])
            labels.append(f'{int(lo)}')
    if not groups:
        return _no_data(ax)
    ax.boxplot(groups, positions=range(len(groups)), showfliers=False, widths=0.6,
               patch_artist=True, boxprops=dict(facecolor=color, alpha=0.35, edgecolor=color),
               medianprops=dict(color='#1a1a1a'))
    ax.set_xticks(range(len(groups)))
    ax.set_xticklabels(labels, rotation=90, fontsize=11)
    ax.axhline(0, color=cfg.NEUTRAL_LINE_COLOR, ls='--', lw=1.4)
    ax.set(xlabel='Elevation band (m)', ylabel=f"Diff ({slot['unit']})")
    if include_title:
        ax.set_title(f"{slot['sensor']}: {slot['label']}", fontsize=17, fontweight='bold', pad=8, color='black')
    _style_ax(ax)


def _draw_type_slope_bias(ax, slot, include_title=True):
    color = cfg.SENSOR_COLORS.get(slot['sensor'], cfg.RED)
    stats = _slot_slope_stats(slot)
    if stats.empty:
        return _no_data(ax, 'no slope data')
    ax.errorbar(stats['slope_band'], stats['mean_diff'], yerr=stats['std_diff'], fmt='o-',
                ms=4.5, color=color, ecolor=cfg.LGREY, elinewidth=1.2, capsize=3)
    ax.axhline(0, color=cfg.NEUTRAL_LINE_COLOR, ls='--', lw=1.4)
    ax.set(xlabel='Terrain slope (deg)', ylabel=f"Mean diff ({slot['unit']})")
    if include_title:
        ax.set_title(f"{slot['sensor']}: {slot['label']}", fontsize=17, fontweight='bold', pad=8, color='black')
    _style_ax(ax)


def _draw_type_elevation_slope_heatmap(ax, slot, include_title=True,
                                        show_xlabel=True, show_ylabel=True):
    """Draws one elevation x slope bias heatmap panel."""
    color = cfg.SENSOR_COLORS.get(slot['sensor'], cfg.RED)
    elev, slope, diff = slot.get('elev'), slot.get('slope'), slot['diff']
    if elev is None or slope is None:
        _no_data(ax, 'no elevation/slope data')
        return None
    valid = np.isfinite(elev) & np.isfinite(slope) & np.isfinite(diff)
    if valid.sum() < 20:
        _no_data(ax)
        return None
    e, s, d = elev[valid], slope[valid], diff[valid]
    e_edges = np.arange(np.floor(e.min() / 100) * 100, e.max() + 100, 100)
    s_edges = np.arange(0, cfg.HEATMAP_SLOPE_MAX_DEG + cfg.HEATMAP_SLOPE_BIN_WIDTH_DEG,
                         cfg.HEATMAP_SLOPE_BIN_WIDTH_DEG)
    if len(e_edges) < 2 or len(s_edges) < 2:
        _no_data(ax)
        return None
    df = pd.DataFrame({'e': e, 's': s, 'd': d})
    df['e_bin'] = pd.cut(df['e'], e_edges)
    df['s_bin'] = pd.cut(df['s'], s_edges)
    n_dropped = int(df['s_bin'].isna().sum())
    if n_dropped:
        print(f"    [{slot['sensor']}: {slot['label']}] elevation x slope heatmap: "
              f"{n_dropped:,}/{len(df):,} points exceed {cfg.HEATMAP_SLOPE_MAX_DEG:.0f}° "
              f"and are excluded from the fixed-range pivot.")
    pivot = df.groupby(['e_bin', 's_bin'], observed=False)['d'].mean().unstack()
    full_s_bins = pd.cut(pd.Series(dtype=float), s_edges).cat.categories
    pivot = pivot.reindex(columns=full_s_bins)
    if pivot.size == 0 or np.all(np.isnan(pivot.values)):
        _no_data(ax)
        return None
    im = ax.imshow(pivot.values, aspect='auto', cmap='RdBu', vmin=-10, vmax=10, origin='lower')
    # Real bin-edge tick labels (subsampled to ~8 per axis) instead of
    # hiding them entirely -- a heatmap a reader can't quantitatively
    # read the axes of isn't much more useful than a plain color swatch.
    n_e, n_s = pivot.shape
    e_step = max(1, n_e // 8)
    s_step = max(1, n_s // 8)
    e_idx = np.arange(0, n_e, e_step)
    s_idx = np.arange(0, n_s, s_step)
    ax.set_xticks(s_idx)
    ax.set_xticklabels([f'{int(pivot.columns[i].left)}' for i in s_idx], fontsize=19,
                        rotation=45, ha='right')
    if show_ylabel:
        ax.set_yticks(e_idx)
        ax.set_yticklabels([f'{int(pivot.index[i].left)}' for i in e_idx], fontsize=19)
        ax.set_ylabel('Elevation bin, lower edge (m)', fontsize=17)
    else:
        # Still show the tick marks (position reference for reading
        # across a row) but suppress the repeated numeric labels -- only
        # the left-column panel spells out the actual elevation values.
        ax.set_yticks(e_idx)
        ax.set_yticklabels([])
    if show_xlabel:
        ax.set_xlabel('Slope bin, lower edge (°)', fontsize=17)
    if include_title:
        ax.set_title(f"{slot['sensor']}: {slot['label']}", fontsize=17, fontweight='bold', pad=8, color='black')
    return im


def _grid_shape(n, with_dhdt, orientation):
    """(rows, cols): up to 4 columns (landscape) / 2 columns (portrait); one
    row (column) for the figures without dh/dt panels."""
    if with_dhdt:
        n_cols = min(4, n) if orientation == 'landscape' else min(2, n)
    else:
        n_cols = n if orientation == 'landscape' else 1
    return int(np.ceil(n / n_cols)), n_cols


def plot_combined_type(type_name: str, results, output_dir, orientation: str = 'landscape'):
    """
    Render one combined-type figure (e.g. 'hypsometric'): CryoSat-2 (All Years
    + epochs) -> ICESat-2 -> CryoSat-2 dh/dt, or CryoSat-2 + ICESat-2 only for
    the types without a dh/dt equivalent.
    """
    draw_fn, n_panels, desc = COMBINED_PLOT_TYPES[type_name]
    with_dhdt = n_panels == 8
    cs2_slots, dhdt_slots, is2_slots = _collect_combined_slots(results)
    slots = (cs2_slots + is2_slots + dhdt_slots) if with_dhdt else (cs2_slots + is2_slots)

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    n_rows, n_cols = _grid_shape(len(slots), with_dhdt, orientation)
    if with_dhdt:
        figsize = (6.5 * n_cols, 7 * n_rows) if orientation == 'landscape' else (7 * n_cols, 6 * n_rows)
    else:
        figsize = (5.6 * n_cols, 7 * n_rows) if orientation == 'landscape' else (7 * n_cols, 5.6 * n_rows)

    # constrained_layout is only enabled (via layout='compressed') for
    # 'scatter': its panels use ax.set_aspect('equal', 'box') (needed so the
    # 1:1 line reads as a true diagonal), which shrinks each axes box to a
    # square within its grid cell -- centered, with the leftover width/height
    # mismatch left as blank padding around it.
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize,
                              layout=('compressed' if type_name in ('scatter', 'map') else None))
    axes_flat = np.atleast_1d(axes).flatten()

    hyps_xlim_by_unit = {}
    if type_name == 'hypsometric':
        for unit, group in (('m', cs2_slots + is2_slots), ('m a⁻¹', dhdt_slots)):
            elevs = [s['elev'][np.isfinite(s['elev'])] for s in group
                     if s is not None and s.get('elev') is not None]
            elevs = [e for e in elevs if e.size > 0]
            if elevs:
                allelev = np.concatenate(elevs)
                hyps_xlim_by_unit[unit] = (float(allelev.min()), float(allelev.max()))

    # 'map': one shared extent (km) for every panel -- the union of all panels'
    # cells, padded by one cell -- so all eight aspect-equal map panels get
    # identical boxes, which (with the 'compressed' layout) removes the uneven
    # white space between rows.
    shared_map_extent = None
    if type_name == 'map':
        xs, ys, pads = [], [], []
        for s in slots:
            if s is not None and s.get('x_proj') is not None and len(s['x_proj']) and s.get('cell_size'):
                xs.append(np.asarray(s['x_proj'], float)); ys.append(np.asarray(s['y_proj'], float))
                pads.append(s['cell_size'])
        if xs:
            ax_, ay_, pad = np.concatenate(xs), np.concatenate(ys), max(pads)
            shared_map_extent = ((np.nanmin(ax_) - pad) / 1000, (np.nanmax(ax_) + pad) / 1000,
                                 (np.nanmin(ay_) - pad) / 1000, (np.nanmax(ay_) + pad) / 1000)

    shared_scatter_elev_lim = None
    if type_name == 'scatter':
        shared_scatter_elev_lim = cfg.SCATTER_ELEV_LIM

    # Likewise, the slope/elevation bin edges are identical across panels, so
    # only the outer edge of the grid (bottom row for the x-axis, left column
    # for the y-axis) needs its tick labels/axis titles drawn; inner panels
    # share those same labels implicitly. `last_heatmap_im` keeps whichever
    # panel's mappable is drawn last, for that shared colorbar to be built from
    # after the loop.
    last_heatmap_im = None

    for i, (ax, slot) in enumerate(zip(axes_flat, slots)):
        letter = chr(97 + i)  # 'a', 'b', 'c', ...
        row, col = i // n_cols, i % n_cols
        if slot is None:
            ax.axis('off')
        elif type_name == 'hypsometric':
            xlim = hyps_xlim_by_unit.get(str(slot['unit']))
            ylim = cfg.HYPS_BIAS_YLIM.get(str(slot['unit']))
            draw_fn(ax, slot, xlim=xlim, ylim=ylim,
                    show_xlabel=(row == n_rows - 1), show_ylabel=(col == 0),
                    show_right_label=(col == n_cols - 1),
                    n_ylim=(cfg.HYPS_N_YLIM_ALL_YEARS if slot.get('key') == 'All_Years'
                            else cfg.HYPS_N_YLIM_PANEL.get((slot['sensor'], slot.get('key')),
                                                       cfg.HYPS_N_YLIM.get(str(slot['unit'])))))
        elif type_name == 'bias':
            draw_fn(ax, slot, ylim=cfg.COMBINED_BIAS_YLIM.get(slot['sensor']))
        elif type_name == 'scatter':
            draw_fn(ax, slot, shared_elev_lim=shared_scatter_elev_lim)
        elif type_name == 'map':
            draw_fn(ax, slot, map_extent=shared_map_extent)
        elif type_name.startswith('elevation_slope_heatmap'):
            im = draw_fn(ax, slot, show_xlabel=(row == n_rows - 1), show_ylabel=(col == 0),
                         include_title=False)
            if im is not None:
                last_heatmap_im = im
        else:
            draw_fn(ax, slot)

        # Prepend the panel letter to whatever title the panel already has (or,
        # for an empty/no-data panel with no title, show just the letter) --
        # keeps panel lettering consistent across every combined-plot type
        # without touching each draw function. elevation_slope_heatmap panels
        # have include_title=False above (no title from draw_fn to prepend to),
        # so build theirs directly from the slot instead.
        if slot is not None and type_name.startswith('elevation_slope_heatmap'):
            new_title = f"({letter}) {slot['sensor']}: {slot['label']}"
        else:
            existing_title = ax.get_title()
            new_title = f'({letter}) {existing_title}' if existing_title else f'({letter})'
        ax.set_title(new_title, fontsize=17, fontweight='bold', color='black', pad=8)
    for ax in axes_flat[len(slots):]:
        ax.axis('off')

    # No fig.suptitle() here by design: these figures all land in the
    # combined_cs2_is2 output folder, and a repeated top-level heading
    # ("CryoSat-2 / ICESat-2 vs TanDEM-X — <description>") added little over
    # the already-descriptive per-panel titles while costing a full row of
    # vertical space -- removing it lets every panel type use that space for
    # the panels themselves.
    if type_name.startswith('elevation_slope_heatmap'):
        fig.subplots_adjust(right=0.90, wspace=0.05, hspace=0.22, top=0.97)
        if last_heatmap_im is not None:
            # Span the colorbar the full height of the panel grid -- from the
            # top of row 1's axes down to the bottom of the last row's axes --
            # rather than a hand-picked fraction of the figure, so it always
            # matches however many rows are on screen (2 for landscape, 4 for
            # portrait) even if the spacing above changes again later.
            top_row_axes = axes_flat[:n_cols]
            bottom_row_axes = axes_flat[n_cols * (n_rows - 1):n_cols * n_rows]
            y_top = max(a.get_position().y1 for a in top_row_axes)
            y_bottom = min(a.get_position().y0 for a in bottom_row_axes)
            cbar_ax = fig.add_axes([0.915, y_bottom, 0.015, y_top - y_bottom])
            cb = fig.colorbar(last_heatmap_im, cax=cbar_ax)
            cb.set_label('Mean TanDEM-X − sensor difference\n(m; m a⁻¹ for dhdt panels)'
                          if with_dhdt else 'Mean TanDEM-X − sensor difference (m)',
                          fontsize=17)
            cb.ax.tick_params(labelsize=15)
    elif type_name in ('scatter', 'map'):
        # The 'compressed' layout engine (enabled at figure creation above) is
        # a live layout engine, not a one-shot call like tight_layout -- it
        # keeps re-solving spacing as the figure is drawn/saved, and (unlike
        # plain constrained_layout) actually compresses away the dead space
        # left by the aspect-locked square panels rather than just
        # repositioning around it.
        fig.set_constrained_layout_pads(w_pad=0.02, h_pad=0.02,
                                         wspace=0.02, hspace=0.02)
    else:
        fig.tight_layout(h_pad=2.2)
    out = output_dir / f'combined_{type_name}_{orientation}.png'
    fig.savefig(out, dpi=cfg.PLOT_DPI, bbox_inches='tight')
    plt.close(fig)
    print(f"  combined_{type_name} ({orientation}, {len(slots)} panels) -> {out}")
    return out


def _draw_type_hyps_slope(ax, slot, xlim=None, ylim=None, n_ylim=None,
                           slope_xlim=None, show_xlabel=True, show_ylabel=True,
                           show_right_label=True, show_top_label=True):
    """Hypsometric bias (bottom x) + slope bias (top x) on one shared y-axis."""
    color = cfg.SENSOR_COLORS.get(slot['sensor'], cfg.RED)
    hyps = _slot_hypsometry(slot)
    sstats = _slot_slope_stats(slot)
    s_label = 'Bias vs. slope (±1 std band)'
    if hyps.empty and sstats.empty:
        _no_data(ax)
        return None

    # Right axis: N per elevation bin (exactly as in the hypsometric figure).
    ax2 = ax.twinx()
    handles = []
    if not hyps.empty:
        bars = ax2.bar(hyps['elev_mid'], hyps['n'], width=cfg.HYPS_BIN_WIDTH_M * 0.82,
                       color=cfg.GREY, alpha=0.30, zorder=1, label='N per elevation bin')
        handles.append(bars)
    if show_right_label:
        ax2.set_ylabel('N', fontsize=15, color=cfg.GREY)
    ax2.tick_params(axis='y', labelsize=13, colors=cfg.GREY)
    if n_ylim is not None:
        ax2.set_ylim(n_ylim)
    for sp in ['left', 'top', 'bottom']:
        ax2.spines[sp].set_visible(False)

    # Bottom axis: hypsometric bias.
    if not hyps.empty:
        h = ax.errorbar(hyps['elev_mid'], hyps['mean_diff'], yerr=hyps['std_diff'], fmt='o',
                        ms=4.5, color=color, ecolor=color, elinewidth=1.2, capsize=3,
                        zorder=3, label='Bias vs. elevation (±1 std)')
        handles.insert(0, h)
    ax.axhline(0, color=cfg.NEUTRAL_LINE_COLOR, ls='--', lw=1.4, zorder=2)
    if show_xlabel:
        ax.set_xlabel('Reference elevation (m)')
    if show_ylabel:
        ax.set_ylabel(f"Mean diff ({slot['unit']})")
    if xlim is not None:
        ax.set_xlim(xlim)
    if ylim is not None:
        ax.set_ylim(ylim)

    # Top axis: slope bias, same y.
    ax3 = ax.twiny()
    if not sstats.empty:
        mid = sstats['slope_band'].values + _MIXED_SLOPE_BIN_DEG / 2
        m = sstats['mean_diff'].values.astype(float)
        sd = sstats['std_diff'].fillna(0).values.astype(float)
        # ±1 std as a shaded band (not whiskers), so it doesn't collide
        # with the elevation series' whiskers drawn on the same y-axis.
        if len(mid) > 1:
            band = ax3.fill_between(mid, m - sd, m + sd, color=cfg.SLOPE_OVERLAY_COLOR,
                                    alpha=0.15, lw=0, zorder=2)
        else:
            ax3.errorbar(mid, m, yerr=sd, fmt='none', ecolor=cfg.SLOPE_OVERLAY_COLOR,
                         elinewidth=1.0, capsize=3, zorder=2)
            band = Patch(color=cfg.SLOPE_OVERLAY_COLOR, alpha=0.15, lw=0)
        line, = ax3.plot(mid, m, 's--', ms=6, mfc='white', mec=cfg.SLOPE_OVERLAY_COLOR,
                         mew=1.5, color=cfg.SLOPE_OVERLAY_COLOR, lw=1.2, zorder=4)
        s = (band, line)
        s_label = 'Bias vs. slope (±1 std band)'
        handles.insert(1, s)
    if slope_xlim is not None:
        ax3.set_xlim(slope_xlim)
    if show_top_label:
        ax3.set_xlabel('Terrain slope (°)', color=cfg.SLOPE_OVERLAY_COLOR)
    ax3.tick_params(axis='x', colors=cfg.SLOPE_OVERLAY_COLOR, labelsize=13)
    ax3.spines['top'].set_color(cfg.SLOPE_OVERLAY_COLOR)
    for sp in ['left', 'right', 'bottom']:
        ax3.spines[sp].set_visible(False)

    _style_ax(ax)
    ax.spines['top'].set_visible(False)
    # No per-panel legend (it covered data); the caller draws one shared
    # legend under the whole figure from these handles.
    labels = [s_label if type(hd) is tuple else hd.get_label() for hd in handles]
    return handles, labels


def plot_combined_hyps_slope(results, output_dir, orientation: str = 'landscape'):
    """
    combined_hypsometric with the slope bias overlaid in every panel:
    same 8 panels, same order, same axis limits as the hypsometric figure,
    plus a top slope axis (shared range across all panels).
    """
    cs2_slots, dhdt_slots, is2_slots = _collect_combined_slots(results)
    slots = cs2_slots + is2_slots + dhdt_slots

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    n_rows, n_cols = _grid_shape(len(slots), True, orientation)
    figsize = (6.5 * n_cols, 7.5 * n_rows) if orientation == 'landscape' else (7 * n_cols, 6.5 * n_rows)

    # Shared elevation x-range per unit group -- same rule as
    # plot_combined_type('hypsometric').
    hyps_xlim_by_unit = {}
    for unit, group in (('m', cs2_slots + is2_slots), ('m a⁻¹', dhdt_slots)):
        elevs = [s['elev'][np.isfinite(s['elev'])] for s in group
                 if s is not None and s.get('elev') is not None]
        elevs = [e for e in elevs if e.size > 0]
        if elevs:
            allelev = np.concatenate(elevs)
            hyps_xlim_by_unit[unit] = (float(allelev.min()), float(allelev.max()))

    # One slope range for every panel: 0 up to the highest populated bin.
    max_band = 0.0
    for s in slots:
        if s is not None:
            st = _slot_slope_stats(s)
            if not st.empty:
                max_band = max(max_band, float(st['slope_band'].max()))
    slope_xlim = (0.0, max_band + _MIXED_SLOPE_BIN_DEG)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    axes_flat = np.atleast_1d(axes).flatten()
    legend_items = None

    for i, (ax, slot) in enumerate(zip(axes_flat, slots)):
        letter = chr(97 + i)
        row, col = i // n_cols, i % n_cols
        if slot is None:
            ax.axis('off')
            ax.set_title(f'({letter})', fontsize=17, fontweight='bold', color='black', pad=8)
            continue
        ylim = cfg.HYPS_BIAS_YLIM.get(str(slot['unit']))
        n_ylim = (cfg.HYPS_N_YLIM_ALL_YEARS if slot.get('key') == 'All_Years'
                  else cfg.HYPS_N_YLIM_PANEL.get((slot['sensor'], slot.get('key')),
                                             cfg.HYPS_N_YLIM.get(str(slot['unit']))))
        _res = _draw_type_hyps_slope(ax, slot, xlim=hyps_xlim_by_unit.get(str(slot['unit'])),
                              ylim=ylim, n_ylim=n_ylim, slope_xlim=slope_xlim,
                              show_xlabel=(row == n_rows - 1), show_ylabel=(col == 0),
                              show_right_label=(col == n_cols - 1),
                              show_top_label=(row == 0))
        if _res is not None and (legend_items is None or len(_res[0]) > len(legend_items[0])):
            legend_items = _res
        ax.set_title(f"({letter}) {slot['sensor']}: {slot['label']}",
                     fontsize=17, fontweight='bold', color='black', pad=8)
    for ax in axes_flat[len(slots):]:
        ax.axis('off')

    fig.tight_layout(h_pad=2.2, rect=(0, 0.05 if orientation == 'landscape' else 0.03, 1, 1))
    if legend_items is not None:
        fig.legend(legend_items[0], legend_items[1], loc='lower center', ncol=3,
                   fontsize=16, frameon=False, bbox_to_anchor=(0.5, 0.0))
    out = output_dir / f'combined_hypsometric_slope_{orientation}.png'
    fig.savefig(out, dpi=cfg.PLOT_DPI, bbox_inches='tight')
    plt.close(fig)
    print(f"  combined_hypsometric_slope ({orientation}, {len(slots)} panels) -> {out}")
    return out


def plot_all_combined_types(results, output_dir):
    """Render every registered combined-plot type, in both landscape and
    portrait, directly into <output_dir> (no individual panel files)."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    for type_name in COMBINED_PLOT_TYPES:
        for orientation in ('landscape', 'portrait'):
            plot_combined_type(type_name, results, output_dir, orientation=orientation)
    # Additional mixed figure: hypsometric + slope bias stacked per panel.
    for orientation in ('landscape', 'portrait'):
        plot_combined_hyps_slope(results, output_dir, orientation=orientation)


# type name -> (draw function, number of panels, description)
COMBINED_PLOT_TYPES = {
    'map':                     (_draw_type_map,                  8, 'Spatial difference map'),
    'histogram':               (_draw_type_histogram,            8, 'Difference distribution'),
    'scatter':                 (_draw_type_scatter,               8, 'Elevation / dhdt scatter'),
    'bias':                    (_draw_type_bias,                 8, 'Vertical bias vs. reference elevation'),
    'statistics':              (_draw_type_statistics,           8, 'Summary statistics'),
    'hypsometric':             (_draw_type_hypsometric,          8, 'Hypsometric bias (bias vs. elevation band)'),
    'slope_bias':              (_draw_type_slope_bias,           8, 'Bias vs. terrain slope'),
    'elevation_slope_heatmap': (_draw_type_elevation_slope_heatmap, 8, 'Elevation x slope bias heatmap'),
    # Same heatmap without the CryoSat-2 dhdt panels (CryoSat-2 x4 + ICESat-2):
    # the dhdt panels only reach ~5-10 deg slope, sit on a +-10 m colour
    # scale meant for elevation differences (so they render white), and a
    # constant slope-dependent bias largely cancels in dh/dt anyway.
    'elevation_slope_heatmap_5panel': (_draw_type_elevation_slope_heatmap, 5, 'Elevation x slope bias heatmap (no dhdt)'),
    'temporal_bias':           (_draw_type_temporal_bias,        5, 'Bias vs. temporal offset from TanDEM-X'),
    'quality_bias':            (_draw_type_quality_bias,         5, 'Bias vs. data-quality metric'),
    'elevation_boxplot':       (_draw_type_elevation_boxplot,    5, 'Bias distribution by elevation band'),
}
