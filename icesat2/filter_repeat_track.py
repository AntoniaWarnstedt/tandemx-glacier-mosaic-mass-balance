"""
Clean the ICESat-2 repeat-track dh/dt points (captoolkit fitsec.py output).

Filters: absolute dh/dt range, fit uncertainty p1_err, local spatial outlier
filter. Settings: icesat2/config.py. QC figure: plot_filter_qc.py.
"""
import geopandas as gpd

import config as cfg
from point_filters import local_mad_filter, range_filter, save_results


def main():
    gdf = gpd.read_file(cfg.RT_INPUT)
    n0 = len(gdf)
    print(f"Loaded {n0} points from {cfg.RT_INPUT}")

    gdf, rm_range = range_filter(gdf, "p1", cfg.RT_MIN_DHDT, cfg.RT_MAX_DHDT, "dh/dt range")
    print(gdf["p1_err"].describe())
    gdf, rm_err = range_filter(gdf, "p1_err", -float("inf"), cfg.RT_MAX_P1_ERR, "p1_err")
    clean, rm_local = local_mad_filter(gdf, "p1", cfg.K_NEIGHBORS, cfg.LOCAL_N_SIGMA, cfg.MAD_FLOOR)

    print(f"Retained {len(clean)} of {n0} points ({100 * len(clean) / n0:.1f}%)")
    save_results(clean, {"range": rm_range, "p1_err": rm_err, "local": rm_local}, cfg.RT_OUTPUT)


if __name__ == "__main__":
    main()
