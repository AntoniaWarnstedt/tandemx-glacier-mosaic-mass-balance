"""
Along-flowline ICESat-2 dh/dt profiles of all glaciers in one figure
(<GRID_PREFIX>_dhdt_profiles.png). Requires a previous run of centerline_dhdt.py.
"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

import config as cfg


def main():
    prof = pd.read_csv(os.path.join(cfg.CL_OUTPUT_DIR, f"{cfg.GRID_PREFIX}_dhdt_profiles.csv"))
    plt.figure(figsize=(12, 7))
    for name, g in prof.groupby("glacier", sort=False):
        plt.plot(g["distance_m"].values / 1000.0, g["dhdt_m_a"].values, label=name, linewidth=2, alpha=0.8)
    plt.axhline(0, color='black', linestyle='--', linewidth=1.5)
    plt.xlabel('Distance from terminus (km)', fontsize=12)
    plt.ylabel('dh/dt (m/yr)', fontsize=12)
    period = f" ({cfg.PERIOD_LABEL})" if cfg.PERIOD_LABEL else ""
    plt.title(f'ICESat-2 Along-Flowline dh/dt Profiles{period}', fontsize=14, fontweight='bold')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=9)
    plt.grid(True, linestyle='--', alpha=0.4)
    plt.tight_layout()
    out = os.path.join(cfg.CL_OUTPUT_DIR, f"{cfg.GRID_PREFIX}_dhdt_profiles.png")
    plt.savefig(out, dpi=300)
    plt.close()
    print(f"✓ Saved: {out}")


if __name__ == "__main__":
    main()
