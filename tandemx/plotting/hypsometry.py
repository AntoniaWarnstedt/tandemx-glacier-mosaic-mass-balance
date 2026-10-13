"""Hypsometric figures: area-altitude distribution, dh/dt per elevation bin,
mass-balance profiles, time series and SAR-penetration diagnostics.
"""

import csv
import os

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .. import config as cfg
from ..penetration import detect_penetration_break_elevation
from .common import hyps_bin_areas_km2
from ..utils import (
    canonical_period_key,
    longest_change,
    format_period_label,
    get_master_grid_info,
    parse_period,
    pixel_area_km2_from_grid,
)


def plot_time_series(all_changes, year_results):
    """Create time series plots with / formatted labels."""
    print("\n▶ Creating time series plots...")

    consecutive_pairs = []
    for change in all_changes:
        yA_parsed = parse_period(change['yA'])
        yB_parsed = parse_period(change['yB'])
        year_diff = float(yB_parsed[1]) - float(yA_parsed[0])
        if year_diff <= 3:
            consecutive_pairs.append(change)

    if not consecutive_pairs:
        consecutive_pairs = all_changes

    consecutive_pairs.sort(key=lambda x: float(parse_period(x['yA'])[0]))

    fig, axes = plt.subplots(2, 1, figsize=(12, 10))
    mid_years = []
    mean_dh = []
    std_dh = []
    mean_dhdt = []
    std_dhdt = []
    period_labels = []

    for change in consecutive_pairs:
        yA_parsed = parse_period(change['yA'])
        yB_parsed = parse_period(change['yB'])
        mid_year = (float(yA_parsed[0]) + float(yB_parsed[1])) / 2
        mid_years.append(mid_year)
        period_labels.append(format_period_label(change['yA'], change['yB']))

        dh_valid = change['dh'][np.isfinite(change['dh'])]
        dhdt_valid = change['dhdt'][np.isfinite(change['dhdt'])]

        mean_dh.append(np.mean(dh_valid))
        std_dh.append(np.std(dh_valid))
        mean_dhdt.append(np.mean(dhdt_valid))
        std_dhdt.append(np.std(dhdt_valid))

    mid_years = np.array(mid_years)
    mean_dh = np.array(mean_dh)
    std_dh = np.array(std_dh)
    mean_dhdt = np.array(mean_dhdt)
    std_dhdt = np.array(std_dhdt)

    ax1 = axes[0]
    colors_dh = ['red' if v < 0 else 'blue' for v in mean_dh]
    ax1.errorbar(mid_years, mean_dh, yerr=std_dh, fmt='o-', capsize=5,
                 color='gray', linewidth=2, markersize=8, label='Mean ± Std')
    ax1.scatter(mid_years, mean_dh, c=colors_dh, s=100, zorder=2, edgecolors='black')
    ax1.axhline(y=0, color='black', linestyle='-', alpha=0.7, linewidth=1.5)
    ax1.set_xlabel('Year', fontsize=12)
    ax1.set_ylabel('Elevation Change dh (m)', fontsize=12)
    ax1.set_title('Elevation Change Over Time', fontsize=14, fontweight='bold')
    ax1.legend(loc='best')
    ax1.grid(True, alpha=0.3)

    for i, label in enumerate(period_labels):
        ax1.annotate(label, (mid_years[i], mean_dh[i]),
                     textcoords="offset points", xytext=(0, 10),
                     ha='center', fontsize=8, alpha=0.7)

    ax2 = axes[1]
    colors_dhdt = ['red' if v < 0 else 'blue' for v in mean_dhdt]
    ax2.errorbar(mid_years, mean_dhdt, yerr=std_dhdt, fmt='o-', capsize=5,
                 color='gray', linewidth=2, markersize=8, label='Mean ± Std')
    ax2.scatter(mid_years, mean_dhdt, c=colors_dhdt, s=100, zorder=2, edgecolors='black')
    ax2.axhline(y=0, color='black', linestyle='-', alpha=0.7, linewidth=1.5)
    ax2.set_xlabel('Year', fontsize=12)
    ax2.set_ylabel('Elevation Change Rate dhdt (m a⁻¹)', fontsize=12)
    ax2.set_title('Elevation Change Rate Over Time', fontsize=14, fontweight='bold')
    ax2.legend(loc='best')
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    out_path = os.path.join(cfg.PLOT_DIR, "time_series_dh_dhdt.png")
    plt.savefig(out_path, dpi=cfg.DPI, bbox_inches='tight')
    plt.close()
    print(f"   ✓ Saved: {out_path}")


def plot_hypsometric_analysis(all_changes, year_results, mb_results=None, bin_size=cfg.ELEV_BIN_SIZE):
    """Hypsometric analysis of cfg.HYPSOMETRY_PERIOD (default: longest period)."""
    if not cfg.CREATE_HYPSOMETRY_FILES:
        print("\n▶ Skipping hypsometric analysis (CREATE_HYPSOMETRY_FILES=False)")
        return

    print("\n▶ Creating hypsometric analysis...")

    if cfg.HYPSOMETRY_PERIOD:
        target_change = next((c for c in all_changes
                              if (c['yA'], c['yB']) == tuple(cfg.HYPSOMETRY_PERIOD)), None)
    else:
        target_change = longest_change(all_changes)

    if target_change is None:
        print(f"   ⚠️  Period {cfg.HYPSOMETRY_PERIOD} not found, skipping hypsometric analysis")
        return

    yA = target_change['yA']
    yB = target_change['yB']
    period_key = f"{yA}-{yB}"
    dh = target_change['dh']

    mb_dict = {f"{r['yA']}-{r['yB']}": r for r in mb_results} if mb_results else {}
    mb_res = mb_dict.get(period_key)
    if mb_res is not None and mb_res.get('dhdt_bias_corrected_raster') is not None:
        dhdt = mb_res['dhdt_bias_corrected_raster']
    else:
        print(f"      ⚠️ No bias-corrected dhdt available for {period_key}; "
              f"falling back to filtered/uncorrected dhdt.")
        dhdt = target_change['dhdt']
    demA = target_change['demA']

    print(f"   Processing: {format_period_label(yA, yB)}")

    grid_info = get_master_grid_info(cfg.MASK_PATH, cfg.RESOLUTION)
    pixel_area_km2 = pixel_area_km2_from_grid(grid_info)

    valid_mask = np.isfinite(demA) & np.isfinite(dh)
    elev_valid = demA[valid_mask]
    dh_valid = dh[valid_mask]
    dhdt_valid = dhdt[valid_mask]

    if len(elev_valid) == 0:
        print("   ⚠️  No valid data for hypsometric analysis")
        return

    elev_min = np.floor(np.nanmin(elev_valid) / bin_size) * bin_size
    elev_max = np.ceil(np.nanmax(elev_valid) / bin_size) * bin_size
    bins = np.arange(elev_min, elev_max + bin_size, bin_size)
    bin_centers = bins[:-1] + bin_size / 2

    area_km2 = []
    mean_dh = []
    mean_dhdt = []
    std_dh = []
    std_dhdt = []

    for i in range(len(bins) - 1):
        mask = (elev_valid >= bins[i]) & (elev_valid < bins[i + 1])
        n_pixels = np.sum(mask)
        area_km2.append(n_pixels * pixel_area_km2)

        if n_pixels > 0:
            mean_dh.append(np.mean(dh_valid[mask]))
            mean_dhdt.append(np.mean(dhdt_valid[mask]))
            std_dh.append(np.std(dh_valid[mask]))
            std_dhdt.append(np.std(dhdt_valid[mask]))
        else:
            mean_dh.append(np.nan)
            mean_dhdt.append(np.nan)
            std_dh.append(np.nan)
            std_dhdt.append(np.nan)

    area_km2 = np.array(area_km2)
    mean_dh = np.array(mean_dh)
    mean_dhdt = np.array(mean_dhdt)
    std_dh = np.array(std_dh)
    std_dhdt = np.array(std_dhdt)

    fig, ax = plt.subplots(figsize=(14, 6))

    vmax = np.nanmax(np.abs(mean_dhdt))
    if vmax == 0 or not np.isfinite(vmax):
        vmax = 1.0
    norm = mcolors.Normalize(vmin=-vmax, vmax=vmax)
    cmap = plt.cm.RdBu

    colors = cmap(norm(mean_dhdt))
    bars = ax.bar(bin_centers, area_km2, width=bin_size * 0.9,
                  color=colors, edgecolor='black', linewidth=0.5, alpha=0.8)

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, orientation='vertical', pad=0.02, shrink=0.8)
    cbar.set_label('Mean Elevation Change Rate (m a⁻¹)', fontsize=12)

    ax.set_xlabel('Elevation (m a.s.l.)', fontsize=13)
    ax.set_ylabel('Area (km²)', fontsize=13)
    ax.set_title(f'Area-Altitude Distribution with Elevation Change ({format_period_label(yA, yB)})',
                 fontsize=15, fontweight='bold')
    ax.grid(True, alpha=0.3, linestyle='--')

    plt.tight_layout()

    out_path = os.path.join(cfg.PLOT_DIR, "area_altitude_distribution.png")
    plt.savefig(out_path, dpi=cfg.DPI, bbox_inches='tight')
    plt.close()

    print(f"   ✓ Saved: {os.path.basename(out_path)}")

    csv_path = os.path.join(cfg.OUT_DIR, "hypsometry_statistics.csv")
    with open(csv_path, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['elevation_bin_center_m', 'area_km2', 'mean_dh_m', 'std_dh_m',
                        'mean_dhdt_m_yr', 'std_dhdt_m_yr'])
        for i in range(len(bin_centers)):
            writer.writerow([
                f'{bin_centers[i]:.1f}',
                f'{area_km2[i]:.4f}',
                f'{mean_dh[i]:.2f}' if np.isfinite(mean_dh[i]) else '',
                f'{std_dh[i]:.2f}' if np.isfinite(std_dh[i]) else '',
                f'{mean_dhdt[i]:.4f}' if np.isfinite(mean_dhdt[i]) else '',
                f'{std_dhdt[i]:.4f}' if np.isfinite(std_dhdt[i]) else ''
            ])

    print(f"   ✓ Saved statistics: {os.path.basename(csv_path)}")


def plot_hypsometric_analysis_all_periods(all_changes, year_results, mb_results=None, bin_size=cfg.ELEV_BIN_SIZE):
    """
    Plot hypsometric distribution for all periods with manuscript-consistent
    labels.
    """
    print("\n▶ Creating hypsometric analysis for all periods...")

    mb_dict = {f"{r['yA']}-{r['yB']}": r for r in mb_results} if mb_results else {}

    for change in all_changes:
        yA = change['yA']
        yB = change['yB']
        period_key = f"{yA}-{yB}"
        dh = change['dh']

        mb_res = mb_dict.get(period_key)
        if mb_res is not None and mb_res.get('dhdt_bias_corrected_raster') is not None:
            dhdt_filtered = mb_res['dhdt_bias_corrected_raster']
        else:
            print(f"      ⚠️ No bias-corrected dhdt available for {period_key}; "
                  f"falling back to filtered/uncorrected dhdt.")
            dhdt_filtered = change.get('dhdt_filtered_3x3', change['dhdt'])
        demA = change['demA']

        period_label = format_period_label(yA, yB)
        print(f"   Processing: {period_label}")

        grid_info = get_master_grid_info(cfg.MASK_PATH, cfg.RESOLUTION)
        pixel_area_km2 = pixel_area_km2_from_grid(grid_info)

        valid_mask = np.isfinite(demA) & np.isfinite(dh)
        elev_valid = demA[valid_mask]
        dhdt_valid_filtered = dhdt_filtered[valid_mask]

        elev_min = np.floor(np.nanmin(elev_valid) / bin_size) * bin_size
        elev_max = np.ceil(np.nanmax(elev_valid) / bin_size) * bin_size
        bins = np.arange(elev_min, elev_max + bin_size, bin_size)
        bin_centers = bins[:-1] + bin_size / 2

        area_km2 = np.zeros_like(bin_centers)
        mean_dhdt_filtered = np.full_like(bin_centers, np.nan, dtype=float)
        std_dhdt_filtered = np.full_like(bin_centers, np.nan, dtype=float)

        for i in range(len(bins) - 1):
            bin_mask = (elev_valid >= bins[i]) & (elev_valid < bins[i + 1])
            area_km2[i] = np.sum(bin_mask) * pixel_area_km2

            if np.sum(bin_mask):
                mean_dhdt_filtered[i] = np.nanmean(dhdt_valid_filtered[bin_mask])
                std_dhdt_filtered[i] = np.nanstd(dhdt_valid_filtered[bin_mask])

        min_area_show = 2.0
        show = area_km2 >= min_area_show

        fig, ax1 = plt.subplots(figsize=(12, 7))
        ax1.bar(
            bin_centers[show], area_km2[show], width=bin_size*0.9,
            color="cornflowerblue", alpha=0.75, label="Area by elevation bin [km²]", zorder=2
        )
        ax1.set_ylabel("Area (km²)", color="steelblue", fontsize=13)
        ax1.set_xlabel("Elevation (m a.s.l.)", fontsize=13)
        ax1.tick_params(axis="y", labelcolor="steelblue")
        ax1.legend(loc="upper left")
        ax1.set_zorder(1)
        ax1.patch.set_alpha(0)

        ax2 = ax1.twinx()
        ax2.errorbar(
            bin_centers[show], mean_dhdt_filtered[show], yerr=std_dhdt_filtered[show],
            fmt="o", color="crimson", markersize=7, elinewidth=2, label="dhdt ±1σ",
            capsize=2, zorder=3
        )
        ax2.set_ylabel("Elevation change rate (m a⁻¹)", color="maroon", fontsize=13)
        ax2.tick_params(axis="y", labelcolor="maroon")
        ax2.axhline(0, color='0.4', linestyle='--', linewidth=1.0)

        ax1.set_title(f"Hypsometric dhdt distribution: {period_label}", fontsize=16, fontweight="bold")
        ax2.legend(loc="upper right")

        plt.tight_layout()
        fig.subplots_adjust(top=0.88)

        out_path = os.path.join(cfg.PLOT_DIR, f"hypsometry_{canonical_period_key(yA)}_{canonical_period_key(yB)}.png")
        plt.savefig(out_path, dpi=300)
        plt.close()
        print(f"   ✓ Saved: {os.path.basename(out_path)}")


def plot_mass_balance_uncertainty(mb_results):
    """
    Clean hypsometric mass-balance plots with method-consistent terminology,
    without printed statistics on the plots.
    """
    out_dir = os.path.join(cfg.PLOT_DIR, "mass_balance")
    os.makedirs(out_dir, exist_ok=True)

    for res in mb_results:
        stats = res.get("onice_bin_stats", None)
        if stats is None or stats.empty:
            continue

        stats = stats.sort_values("cl").reset_index(drop=True)

        required_cols = {
            "cl", "n_ref", "n_raw", "n_fil",
            "mean_ch_voi", "mean_ch_fil",
            "sdev_ch_fil"
        }
        if not required_cols.issubset(set(stats.columns)):
            continue

        _areas = hyps_bin_areas_km2(res, stats)
        x = _areas["x"]
        x_ref, area_ref_km2 = _areas["x_ref"], _areas["ref"]
        area_obs_km2 = _areas["obs"]
        area_fil_km2 = _areas["fil"]

        y_fil = stats["mean_ch_fil"].values.astype(float)
        y_fil_std = stats["sdev_ch_fil"].fillna(0).values.astype(float)

        y_fill = stats["mean_ch_voi"].values.astype(float)

        finite_fil = np.isfinite(y_fil)
        finite_fill = np.isfinite(y_fill)

        fig = plt.figure(figsize=(14, 8), constrained_layout=True)
        gs = fig.add_gridspec(nrows=2, ncols=1, height_ratios=[0.85, 0.15])

        ax1 = fig.add_subplot(gs[0])
        ax_leg = fig.add_subplot(gs[1])
        ax_leg.axis("off")

        bin_width = (x[1] - x[0]) * 0.9 if len(x) > 1 else 40.0

        b1 = ax1.bar(
            x_ref if x_ref is not None else x,
            area_ref_km2 if area_ref_km2 is not None else np.zeros_like(x),
            width=bin_width,
            color="#c9d8f0", alpha=0.9, edgecolor="none",
            label="Catchment area per elevation bin (reference DEM) [km²]", zorder=1
        )
        b2 = ax1.bar(
            x, area_obs_km2, width=bin_width * 0.72,
            color="#5f7ddb", alpha=0.85, edgecolor="none",
            label="Observed area (dh/dt available) [km²]", zorder=2
        )
        b3 = ax1.bar(
            x, area_fil_km2, width=bin_width * 0.46,
            color="#37379b", alpha=0.7, edgecolor="none",
            label="Pixels used for bin mean (after outlier filter) [km²]", zorder=3
        )

        ax1.set_xlabel("Elevation (m a.s.l.)", fontsize=16)
        ax1.set_ylabel("Area (km²)", fontsize=16, color="#3572b0")
        ax1.tick_params(axis="y", labelcolor="#3572b0", labelsize=12)
        ax1.tick_params(axis="x", labelsize=12)
        ax1.grid(True, axis="y", linestyle="--", alpha=0.25)

        ax2 = ax1.twinx()
        ax2.axhline(0, color="0.45", lw=1.4, ls="--", zorder=0)

        line_fill = None
        err_fil = None

        if np.any(finite_fill):
            line_fill, = ax2.plot(
                x[finite_fill], y_fill[finite_fill],
                color="black", lw=2.8,
                label="dhdt_filled (void-filled)",
                zorder=6
            )

        if np.any(finite_fil):
            err_fil = ax2.errorbar(
                x[finite_fil], y_fil[finite_fil],
                yerr=y_fil_std[finite_fil],
                fmt="o-", color="crimson",
                markersize=4.5, lw=1.4,
                elinewidth=1.2, capsize=2.5,
                alpha=0.9,
                label="dhdt bin mean ±1σ",
                zorder=5
            )

        ax2.set_ylabel("Elevation change rate (m a⁻¹)", fontsize=16, color="#8b0000")
        ax2.tick_params(axis="y", labelcolor="#8b0000", labelsize=12)

        ax1.set_title(
            f"Hypsometric Profile: {res.get('Period', '')}",
            fontsize=20, fontweight="bold", pad=12
        )

        handles = [b1, b2, b3]
        labels = [
            "Catchment area per elevation bin (reference DEM) [km²]",
            "Observed area (dh/dt available) [km²]",
            "Pixels used for bin mean (after outlier filter) [km²]"
        ]
        if line_fill is not None:
            handles.append(line_fill); labels.append("dhdt_filled (void-filled)")
        if err_fil is not None:
            handles.append(err_fil); labels.append("dhdt bin mean ±1σ")

        ax_leg.legend(handles, labels, loc="center", ncol=2, fontsize=12, frameon=False)

        period_slug = canonical_period_key(res["Period"]).replace("/", "-")
        out_path = os.path.join(out_dir, f"mb_hypsometric_{period_slug}.png")
        plt.savefig(out_path, dpi=cfg.DPI, bbox_inches="tight")
        plt.close()


def plot_mass_balance_uncertainty_paper(mb_results, show_bin_errorbars=True):
    """
    Publication-style hypsometric mass-balance figure with manuscript-
    consistent labels, without printed statistics blocks.
    """
    out_dir = os.path.join(cfg.PLOT_DIR, "mass_balance_paper")
    os.makedirs(out_dir, exist_ok=True)

    for res in mb_results:
        stats = res.get("onice_bin_stats", None)
        if stats is None or stats.empty:
            continue

        stats = stats.sort_values("cl").reset_index(drop=True)

        required_cols = {
            "cl", "n_ref", "n_raw", "n_fil",
            "mean_ch_fil", "mean_ch_voi",
            "sdev_ch_fil"
        }
        if not required_cols.issubset(set(stats.columns)):
            continue

        inv_stats = res.get("inventory_area_stats", {})
        obs_stats = res.get("obs_area_stats", {})

        x = stats["cl"].values.astype(float)
        n_ref = stats["n_ref"].fillna(0).astype(float).values
        n_fil = stats["n_fil"].fillna(0).astype(float).values
        n_raw = stats["n_raw"].fillna(0).astype(float).values

        if np.nansum(n_ref) <= 0:
            continue
        _areas = hyps_bin_areas_km2(res, stats)
        x_ref, area_reference_km2 = _areas["x_ref"], _areas["ref"]
        area_observed_km2 = _areas["obs"]
        area_filtered_km2 = _areas["fil"]

        y_fil = stats["mean_ch_fil"].values.astype(float)
        y_fil_std = stats["sdev_ch_fil"].fillna(0).values.astype(float)
        y_voi = stats["mean_ch_voi"].values.astype(float)

        finite_fil = np.isfinite(y_fil)
        finite_voi = np.isfinite(y_voi)

        fig = plt.figure(figsize=(13.5, 8.5), constrained_layout=True)
        gs = fig.add_gridspec(nrows=2, ncols=1, height_ratios=[0.85, 0.15])

        ax = fig.add_subplot(gs[0])
        ax_leg = fig.add_subplot(gs[1])
        ax_leg.axis("off")

        ax2 = ax.twinx()
        bin_width = (x[1] - x[0]) * 0.88 if len(x) > 1 else 40.0

        bars_ref = ax.bar(x_ref if x_ref is not None else x,
                          area_reference_km2 if area_reference_km2 is not None else np.zeros_like(x),
                          width=bin_width, color="#d9d9d9", alpha=0.95, edgecolor="none",
                          label="Catchment area per elevation bin (reference DEM)", zorder=1)
        bars_obs = ax.bar(x, area_observed_km2, width=bin_width * 0.78, color="#8fb0e8", alpha=0.95, edgecolor="none",
                          label="Observed area (dh/dt available)", zorder=2)
        bars_fil = ax.bar(x, area_filtered_km2, width=bin_width * 0.46, color="#4f79d8", alpha=0.95, edgecolor="none",
                          label="Pixels used for bin mean (after outlier filter)", zorder=3)

        ax.set_xlabel("Elevation (m a.s.l.)", fontsize=18, labelpad=10)
        ax.set_ylabel("Area (km²)", fontsize=18, color="#2f6fb6", labelpad=12)
        ax.tick_params(axis="x", labelsize=13)
        ax.tick_params(axis="y", labelsize=13, labelcolor="#2f6fb6")
        ax.grid(True, axis="y", linestyle="--", alpha=0.25)

        ax2.axhline(0, color="0.45", lw=1.6, ls="--", zorder=0)

        handles = [bars_ref, bars_obs, bars_fil]
        labels = ["Catchment area per elevation bin (reference DEM)", "Observed area (dh/dt available)", "Pixels used for bin mean (after outlier filter)"]

        if np.any(finite_voi):
            line_voi, = ax2.plot(
                x[finite_voi], y_voi[finite_voi], color="black", lw=3.0, alpha=0.98,
                label="dhdt_filled (void-filled)", zorder=5
            )
            handles.append(line_voi); labels.append("dhdt_filled (void-filled)")

        if np.any(finite_fil):
            if show_bin_errorbars:
                fil_obj = ax2.errorbar(
                    x[finite_fil], y_fil[finite_fil], yerr=y_fil_std[finite_fil],
                    fmt="o-", ms=4.3, lw=1.2, color="crimson", ecolor="crimson",
                    elinewidth=1.0, capsize=2.5, alpha=0.9, label="dhdt bin mean ±1σ", zorder=6
                )
                handles.append(fil_obj); labels.append("dhdt bin mean ±1σ")
            else:
                fil_obj = ax2.scatter(
                    x[finite_fil], y_fil[finite_fil], s=30, color="crimson", alpha=0.95,
                    label="dhdt bin mean", zorder=6
                )
                handles.append(fil_obj); labels.append("dhdt bin mean")

        ax2.set_ylabel("Elevation change rate (m a⁻¹)", fontsize=18, color="#8b0000", labelpad=14)
        ax2.tick_params(axis="y", labelsize=13, labelcolor="#8b0000")

        ax.set_title(res.get("Period", ""), fontsize=25, fontweight="bold", pad=14)

        ax_leg.legend(handles, labels, loc="center", ncol=2, fontsize=13, frameon=False,
                      handlelength=2.8, columnspacing=2.0)

        period_slug = canonical_period_key(res["Period"]).replace("/", "-")
        suffix = "with_bin_errors" if show_bin_errorbars else "clean"
        out_path = os.path.join(out_dir, f"mb_hypsometric_paper_{period_slug}_{suffix}.png")
        plt.savefig(out_path, dpi=cfg.DPI, bbox_inches="tight")
        plt.close()


def plot_sar_penetration_diagnostic(
        elev, dhdt,
        n_px=None, area_km2=None, std_dhdt=None,
        catchment_elev=None, catchment_area_km2=None,
        pen_start_elev_m=None, pen_end_elev_m=None,
        sar_pen_bias=cfg.SAR_PEN_BIAS, sar_pen_type=cfg.SAR_PEN_TYPE,
        assumed_dt_years=None,
        min_bin_pixels=None, core_quantile=0.75,
        residual_sigma_thresh=2.0, min_consecutive_bins=3,
        period_label="", out_path=None,
        core_max_elev=None, core_min_elev=None,
        correction_applied=None,
):
    """
    Bottom: hypsometric area (or pixel count) per bin, sharing the same x-axis,
    so sparse high-elevation bins are visually explained rather than just
    looking noisy for no apparent reason.
    """
    elev = np.asarray(elev, dtype=float)
    dhdt = np.asarray(dhdt, dtype=float)

    if min_bin_pixels is None:
        min_bin_pixels = cfg.MIN_BIN_PIXELS

    pixel_area_km2 = (cfg.RESOLUTION ** 2) * cfg.M2_TO_KM2
    if n_px is None and area_km2 is not None:
        n_px = np.asarray(area_km2, dtype=float) / pixel_area_km2
    if area_km2 is None and n_px is not None:
        area_km2 = np.asarray(n_px, dtype=float) * pixel_area_km2
    if n_px is None:
        n_px = np.full_like(elev, np.nan)
    n_px = np.asarray(n_px, dtype=float)

    bin_stats_local = pd.DataFrame({"cl": elev, "mean_ch_fil": dhdt, "n_fil": n_px})
    h_break, break_diag = detect_penetration_break_elevation(
        bin_stats_local, elev_col="cl", dhdt_col="mean_ch_fil", n_col="n_fil",
        min_bin_pixels=min_bin_pixels, core_quantile=core_quantile,
        residual_sigma_thresh=residual_sigma_thresh,
        min_consecutive_bins=min_consecutive_bins,
        core_max_elev=core_max_elev, core_min_elev=core_min_elev,
    )

    # fit of the detector itself
    slope = break_diag.get("fit_slope", np.nan)
    intercept = break_diag.get("fit_intercept", np.nan)
    sigma_core = break_diag.get("sigma_core", np.nan)
    core_cut = break_diag.get("core_cut_elev_m", np.nan)
    core_lo = break_diag.get("core_min_elev_m", np.nan)
    fit_ok = np.isfinite(slope) and np.isfinite(sigma_core)
    if correction_applied is None:
        correction_applied = break_diag["status"] == "break_detected"

    fig, (ax_top, ax_bot) = plt.subplots(
        2, 1, figsize=(8, 7), sharex=True,
        gridspec_kw={"height_ratios": [3, 1], "hspace": 0.08}
    )

    if np.any(np.isfinite(n_px)):
        n_ref = np.nanmax(n_px)
        sizes = 15 + 60 * np.sqrt(np.clip(n_px, 0, None) / n_ref) if n_ref > 0 else np.full_like(elev, 20.0)
        sizes = np.where(np.isfinite(sizes), sizes, 20.0)
    else:
        sizes = np.full_like(elev, 20.0)

    below_min = n_px < min_bin_pixels
    ax_top.scatter(elev[~below_min], dhdt[~below_min], s=sizes[~below_min],
                   color="#1f4e79", zorder=4, label=f"observed mean dh/dt (n\u2265{min_bin_pixels} px)")
    if np.any(below_min):
        ax_top.scatter(elev[below_min], dhdt[below_min], s=20, marker="x",
                        color="#999999", zorder=4,
                        label=f"rejected bin (n<{min_bin_pixels} px)")

    if std_dhdt is not None:
        std_dhdt = np.asarray(std_dhdt, dtype=float)
        with np.errstate(invalid="ignore", divide="ignore"):
            sem = std_dhdt / np.sqrt(np.clip(n_px, 1, None))
        ax_top.errorbar(elev, dhdt, yerr=sem, fmt="none", ecolor="#1f4e79",
                         elinewidth=0.8, capsize=2, alpha=0.6, zorder=3)

    elev_grid = np.linspace(np.nanmin(elev), np.nanmax(elev), 200)
    if fit_ok:
        # Draw the reference trend only from the start of the reference
        # range upward (it says nothing about the glacier tongues below).
        fit_grid = np.linspace(core_lo if np.isfinite(core_lo) else np.nanmin(elev),
                               np.nanmax(elev), 200)
        fit_line = slope * fit_grid + intercept
        ax_top.plot(fit_grid, fit_line, "k--", lw=1.3, zorder=5,
                    label="reference fit (%.0f\u2013%.0f m), extrapolated" % (core_lo, core_cut))
        ax_top.fill_between(fit_grid, fit_line - residual_sigma_thresh * sigma_core,
                             fit_line + residual_sigma_thresh * sigma_core,
                             color="gray", alpha=0.15, zorder=1,
                             label=f"\u00b1{residual_sigma_thresh:.0f}\u03c3 (robust residual) band")
        ax_top.axvline(core_cut, color="gray", ls=":", lw=0.8, zorder=1)
        if np.isfinite(core_lo) and core_lo > np.nanmin(elev):
            ax_top.axvline(core_lo, color="gray", ls=":", lw=0.8, zorder=1)

    band_ok = (pen_start_elev_m is not None and pen_end_elev_m is not None
               and np.isfinite(pen_start_elev_m) and np.isfinite(pen_end_elev_m))
    if band_ok:
        band_state = "subtracted" if correction_applied else "uncertainty only, not subtracted"
        ax_top.axvspan(pen_start_elev_m, pen_end_elev_m, color="#e07b39",
                        alpha=0.15 if correction_applied else 0.07, zorder=0,
                        label=f"penetration ramp {pen_start_elev_m:.0f}\u2013{pen_end_elev_m:.0f} m ({band_state})")

        if correction_applied and assumed_dt_years is not None and assumed_dt_years > 0:
            if sar_pen_type == "ramp":
                slp = sar_pen_bias / (pen_end_elev_m - pen_start_elev_m)
                pen_depth = np.clip((elev_grid - pen_start_elev_m) * slp, 0.0, sar_pen_bias)
            else:
                pen_depth = np.where(
                    (elev_grid >= pen_start_elev_m) & (elev_grid <= pen_end_elev_m),
                    sar_pen_bias, 0.0
                )
            pen_rate_grid = pen_depth / assumed_dt_years
            # Show what the OBSERVED profile looks like after subtracting
            # the illustrative correction (only where a raw value exists).
            corr_at_bins = np.interp(elev, elev_grid, pen_rate_grid)
            corr_at_bins = np.where(elev >= pen_start_elev_m, corr_at_bins, 0.0)
            dhdt_corrected = dhdt - corr_at_bins
            in_band = elev >= pen_start_elev_m
            ax_top.plot(elev[in_band], dhdt_corrected[in_band], "o-", color="#c1272d",
                        ms=4, lw=1.2, zorder=4,
                        label=f"corrected profile (illustrative, \u0394t\u2248{assumed_dt_years:.1f} yr)")

    if break_diag["status"] == "break_detected":
        ax_top.axvline(h_break, color="#c1272d", lw=1.5, zorder=5,
                        label=f"detected break: {h_break:.0f} m")
    mz = break_diag.get("max_z_upper", np.nan)
    ax_top.text(0.98, 0.03,
                 f"break test (diagnostic only): {break_diag['status']}  "
                 f"(tested \u2265{core_cut:.0f} m: n={break_diag.get('n_bins_upper', 0)}, "
                 f"max z={mz:.2f}, threshold {residual_sigma_thresh:.0f}\u03c3 "
                 f"\u00d7{min_consecutive_bins} bins)\n"
                 f"penetration: {'subtracted' if correction_applied else 'uncertainty term only (Braun et al. 2019)'}",
                 transform=ax_top.transAxes, fontsize=8, color="#333333",
                 va="bottom", ha="right",
                 bbox=dict(boxstyle="round", fc="white", ec="#cccccc", alpha=0.9))

    ax_top.axhline(0, color="black", lw=0.6, alpha=0.5)
    ax_top.set_ylabel("mean dh/dt (m yr$^{-1}$)")
    title = "SAR-penetration diagnostic: on-ice dh/dt vs. elevation"
    if period_label:
        title += f" \u2014 {period_label}"
    ax_top.set_title(title, fontsize=11, fontweight="bold")
    ax_top.legend(fontsize=7.5, loc="upper left", framealpha=0.9)
    ax_top.grid(alpha=0.25)

    if area_km2 is not None:
        area_km2 = np.asarray(area_km2, dtype=float)
        width = (elev[1] - elev[0]) * 0.9 if len(elev) > 1 else 40.0
        # full catchment hypsometry behind the observed (tested) area, so
        # unobserved elevation ranges are visible at a glance.
        if catchment_elev is not None and catchment_area_km2 is not None:
            ax_bot.bar(np.asarray(catchment_elev, float), np.asarray(catchment_area_km2, float),
                       width=width, color="none", edgecolor="#555555", lw=0.6,
                       label="catchment (inventory)")
        tested = n_px >= min_bin_pixels
        ax_bot.bar(elev[tested], area_km2[tested], width=width, color="#8faed1",
                   edgecolor="none", label=f"observed, tested (n\u2265{min_bin_pixels} px)")
        if np.any(~tested & (n_px > 0)):
            ax_bot.bar(elev[~tested & (n_px > 0)], area_km2[~tested & (n_px > 0)], width=width,
                       color="#d9d9d9", edgecolor="none", label="observed, too few px")
        ax_bot.set_ylabel("area (km$^2$)")
        ax_bot.legend(fontsize=7, loc="upper right", framealpha=0.9)
    elif n_px is not None:
        width = (elev[1] - elev[0]) * 0.9 if len(elev) > 1 else 40.0
        ax_bot.bar(elev, n_px, width=width, color="#8faed1", edgecolor="none")
        ax_bot.set_ylabel("pixel count")
    ax_bot.set_xlabel("elevation (m a.s.l.)")
    ax_bot.grid(alpha=0.25)

    fig.tight_layout()
    if out_path is not None:
        fig.savefig(out_path, dpi=300, bbox_inches="tight")
        print(f"      Saved SAR-penetration diagnostic plot: {out_path}")

    return {"fig": fig, "break_diag": break_diag}


def plot_sar_penetration_diagnostics(mb_results):
    """Break-in-slope diagnostic figure (dh/dt vs. elevation) for every period."""
    out_dir = os.path.join(cfg.PLOT_DIR, "sar_penetration_diagnostics")
    os.makedirs(out_dir, exist_ok=True)
    for res in mb_results:
        bins = res.get("sar_pen_detect_bins")
        if bins is None or bins.empty:
            continue
        meta, st, cov = res["sar_pen_meta"], res["sar_pen_settings"], res["sar_pen_coverage_table"]
        applied = bool(meta.get("pen_correction_applied", False))
        label = format_period_label(res["yA"], res["yB"])
        plot_sar_penetration_diagnostic(
            elev=bins["cl"].values,
            dhdt=bins["mean_ch_fil"].values,
            std_dhdt=bins["sdev_ch_fil"].values if "sdev_ch_fil" in bins else None,
            n_px=bins["n_fil"].values,
            catchment_elev=cov["cl"].values,
            catchment_area_km2=cov["catchment_area_km2"].values,
            pen_start_elev_m=meta["pen_start_elev_m"] if applied else meta.get("pen_band_tested_start_m"),
            pen_end_elev_m=meta["pen_end_elev_m"] if applied else meta.get("pen_band_tested_end_m"),
            correction_applied=applied,
            core_max_elev=st["core_max_elev"],
            core_min_elev=st["core_min_elev"],
            residual_sigma_thresh=cfg.SAR_PEN_BREAK_SIGMA_THRESH,
            min_consecutive_bins=cfg.SAR_PEN_BREAK_MIN_CONSECUTIVE,
            sar_pen_bias=st["sar_pen_bias"],
            sar_pen_type=st["sar_pen_type"],
            assumed_dt_years=meta.get("Obs_mean_pen_cells"),
            min_bin_pixels=cfg.SAR_PEN_DETECT_MIN_BIN_PIXELS,
            period_label=label,
            out_path=os.path.join(out_dir, f"sar_pen_diagnostic_{canonical_period_key(label)}.png"),
        )
        plt.close("all")


def plot_penetration_onset(onset):
    """Inter-period consistency e(z) with the fitted hinge model (penetration onset)."""
    if not onset or onset.get("table") is None or "labels" not in onset:
        return
    out_dir = os.path.join(cfg.PLOT_DIR, "sar_penetration_diagnostics")
    os.makedirs(out_dir, exist_ok=True)
    tab = onset["table"]
    a, b, cc = onset["labels"]
    z0, c, slope = onset["z0_candidate"], onset["c"], onset["slope_m_per_m"]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.axhline(0, color="black", lw=0.6, alpha=0.5)
    ax.errorbar(tab["cl"], tab["e_median"],
                yerr=tab["e_nmad"] / np.sqrt(np.clip(tab["n"], 1, None)),
                fmt="o", color="#1f4e79", ms=4, label="bin median e (± NMAD/√n)")
    zz = np.linspace(onset["fit_z_min"], onset["fit_z_max"], 200)
    ax.plot(zz, c + slope * np.maximum(0, zz - z0), "--", color="#c1272d",
            label=f"hinge fit, z0 = {z0:.0f} m ({onset['status']}, p = {onset['p']:.1g})")
    ax.axvspan(tab["cl"].min(), cfg.SAR_PEN_ONSET_FIT_MIN_ELEV, color="gray", alpha=0.1,
               label="excluded from fit")
    ax.set_xlabel("elevation (m a.s.l.)")
    ax.set_ylabel(f"DEM {b} minus linear {a}–{cc} trend, e (m)")
    ax.set_title("Inter-period consistency: elevation-dependent offset of the middle DEM")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "penetration_onset_interperiod_consistency.png"), dpi=200)
    plt.close(fig)
