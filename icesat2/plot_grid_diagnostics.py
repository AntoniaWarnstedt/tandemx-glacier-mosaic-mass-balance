"""
Diagnostic figures of grid_dhdt.py (settings: config.py, section 4):
<p>_variogram.png (variogram fit and cross-validation by distance) and
<p>_xover.png (crossovers vs. the grid). Requires a previous run of grid_dhdt.py.
"""
import os
import pickle

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import config as cfg
from kriging import _gamma_model


def plot_crossovers(out, t, s0):
    """(a) crossover vs. grid dh/dt, (b) difference histograms, (c) difference
    vs. distance to the nearest node with +-2 sigma_IS2."""
    fig, axes = plt.subplots(1, 3, figsize=(20, 6.5))
    ok = np.isfinite(t['grid_dhdt'])
    ax = axes[0]
    ax.scatter(t.loc[ok, 'grid_dhdt'], t.loc[ok, 'xo_dhdt'], s=6, alpha=0.5, color='#2E6DA4')
    lim = np.nanpercentile(np.abs(np.r_[t['grid_dhdt'], t['xo_dhdt']]), 99.5) * 1.05
    ax.plot([-lim, lim], [-lim, lim], 'k--', lw=1)
    ax.set(xlim=(-lim, lim), ylim=(-lim, lim), xlabel='dh/dt$_{IS2}$ grid (m a$^{-1}$)',
           ylabel='Crossover dh/dt (m a$^{-1}$)', aspect='equal')
    ax.text(0.03, 0.97, f"N = {s0.get('n', 0):,}\nmedian {s0.get('median', np.nan):+.2f}\n"
            f"NMAD {s0.get('nmad', np.nan):.2f}\nmean {s0.get('mean', np.nan):+.2f}\n"
            f"SD {s0.get('std', np.nan):.2f} m a$^{{-1}}$", transform=ax.transAxes, va='top',
            family='monospace', bbox=dict(fc='white', ec='#cccccc'))
    ax.set_title('(a) Crossovers vs. grid', fontweight='bold')
    ax = axes[1]
    bins = np.linspace(-5, 5, 81)
    ax.hist(t['xo_minus_grid'].dropna(), bins=bins, color='#2E6DA4', alpha=0.6, label='crossover - grid')
    ax.hist(t['xo_minus_node'].dropna(), bins=bins, histtype='step', color='k', lw=1.5,
            label=f'crossover - nearest node (≤ {cfg.XO_NODE_MATCH_M:.0f} m)')
    ax.axvline(0, color='k', ls='--', lw=1)
    ax.set(xlabel='Difference (m a$^{-1}$)', ylabel='Count')
    ax.legend(fontsize=9)
    ax.set_title('(b) Differences', fontweight='bold')
    ax = axes[2]
    ax.scatter(t.loc[ok, 'dist_to_node'], t.loc[ok, 'xo_minus_grid'], s=5, alpha=0.4, color='#2E6DA4',
               label='crossover - grid')
    srt = np.argsort(t.loc[ok, 'dist_to_node'].values)
    ax.plot(t.loc[ok, 'dist_to_node'].values[srt], 2 * t.loc[ok, 'grid_sigma'].values[srt], 'r-', lw=1,
            label='± 2 σ$_{IS2}$ (grid only, crossover noise not included)')
    ax.plot(t.loc[ok, 'dist_to_node'].values[srt], -2 * t.loc[ok, 'grid_sigma'].values[srt], 'r-', lw=1)
    ax.axhline(0, color='k', ls='--', lw=1)
    ax.set(xlabel='Distance to nearest fitsec node (m)', ylabel='Crossover - grid (m a$^{-1}$)',
           ylim=(-5, 5))
    ax.legend(fontsize=9)
    ax.set_title('(c) vs. distance to data', fontweight='bold')
    for a_ in axes:
        a_.grid(True, color='#dddddd')
    fig.tight_layout()
    fig.savefig(out('xover.png'), dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"  wrote {out('xover.png')}")


def plot_variogram_cv(out, ev, vg, cvd, vg_used=None):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
    ax = axes[0]
    used = ev['used_in_fit']
    ax.plot(ev.loc[used, 'lag_mean'] / 1000, ev.loc[used, 'gamma_dowd'], 'o', color='#2E6DA4',
            label='empirical (Dowd), used in fit')
    ax.plot(ev.loc[~used, 'lag_mean'] / 1000, ev.loc[~used, 'gamma_dowd'], 'o', mfc='white',
            color='#2E6DA4', label=f'empirical, lag < {cfg.VARIOGRAM_MIN_LAG:.0f} m (shared data)')
    h = np.linspace(0, cfg.VARIOGRAM_MAX_LAG, 400)
    ax.plot(h / 1000, _gamma_model(h, vg['nugget'], vg['psill'], vg['range_m']), 'k-',
            label=f"{cfg.VARIOGRAM_MODEL}: nugget {vg['nugget']:.3f}, sill {vg['psill']:.3f}, "
                  f"range {vg['range_m'] / 1000:.1f} km (fitted)")
    if vg_used is not None and abs(vg_used['nugget'] - vg['nugget']) > 1e-12:
        ax.plot(h / 1000, _gamma_model(h, vg_used['nugget'], vg_used['psill'], vg_used['range_m']),
                'r--', label=f"used for kriging: nugget {vg_used['nugget']:.3f} (chosen by CV)")
    ax.set(xlabel='Lag (km)', ylabel='Semivariance ((m a$^{-1}$)$^2$)')
    ax.legend(fontsize=9)
    ax.set_title('(a) Variogram of fitsec dh/dt nodes', fontweight='bold')
    ax = axes[1]
    if not cvd.empty:
        mid = (cvd['dist_lo_m'] + cvd['dist_hi_m']) / 2 / 1000
        ax.plot(mid, cvd['res_nmad'], 'o-', color='#2E6DA4', label='CV residual NMAD')
        ax.plot(mid, cvd['res_rmse'], 's--', color='#2E6DA4', mfc='white', label='CV residual RMSE')
        ax.plot(mid, cvd['sigma_total_calibrated_median'], 'k-',
                label='predicted σ (median, after calibration)')
        ax.plot(mid, cvd['sigma_total_median'], 'k:', label='predicted σ (median, before calibration)')
    ax.set(xlabel='Distance to nearest remaining node (km)', ylabel='m a$^{-1}$')
    ax.legend(fontsize=9)
    ax.set_title('(b) Block cross-validation', fontweight='bold')
    for a_ in axes:
        a_.grid(True, color='#dddddd')
    fig.tight_layout()
    fig.savefig(out('variogram.png'), dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"  wrote {out('variogram.png')}")


def main():
    out = lambda name: os.path.join(cfg.GRID_OUTPUT_DIR, f"{cfg.GRID_PREFIX}_{name}")
    with open(out('plot_state.pkl'), 'rb') as f:
        st = pickle.load(f)
    if st['xover'] is not None:
        plot_crossovers(out, st['xover']['table'], st['xover']['summary'])
    plot_variogram_cv(out, st['ev'], st['vg_fit'], st['cvd'], st['vg'])


if __name__ == '__main__':
    main()
