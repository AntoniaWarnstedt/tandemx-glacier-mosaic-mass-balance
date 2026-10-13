"""
QC maps of the ICESat-2 filtering: kept points coloured by dh/dt, removed
points coloured by the filter that removed them.
Usage: python plot_filter_qc.py [repeat_track|crossover]  (default: both)
"""
import os
import sys

import geopandas as gpd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

import config as cfg  # noqa: E402
from point_filters import removed_path  # noqa: E402

DATASETS = {"repeat_track": (cfg.RT_OUTPUT, "p1"), "crossover": (cfg.XO_OUTPUT, "dhdt")}
REMOVED_STYLE = {"slope": ("orange", "slope filter"), "local": ("black", "local outlier filter")}


def plot_qc(name):
    path, col = DATASETS[name]
    kept = gpd.read_file(path)
    fig, ax = plt.subplots(figsize=(8, 8))
    kept.plot(ax=ax, column=col, cmap="RdBu", vmin=cfg.QC_DHDT_RANGE[0], vmax=cfg.QC_DHDT_RANGE[1],
              markersize=3, legend=True, legend_kwds={"label": "dh/dt (m a$^{-1}$), kept"})
    if os.path.exists(removed_path(path)):
        removed = gpd.read_file(removed_path(path))
        for key, (color, label) in REMOVED_STYLE.items():
            sub = removed[removed["removed_by"] == key]
            if len(sub):
                sub.plot(ax=ax, color=color, markersize=5, label=f"removed: {label} ({len(sub)})")
    ax.set_title(f"{name.replace('_', '-')}: kept vs. removed points "
                 f"(k={cfg.K_NEIGHBORS}, {cfg.LOCAL_N_SIGMA} x local MAD)")
    if ax.get_legend_handles_labels()[0]:
        ax.legend(loc="lower left")
    out = os.path.splitext(path)[0] + "_qc.png"
    fig.savefig(out, dpi=cfg.QC_DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out}")


if __name__ == "__main__":
    for n in (sys.argv[1:] or DATASETS):
        plot_qc(n)
