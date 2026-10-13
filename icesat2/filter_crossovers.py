"""
Clean the ICESat-2 crossover dh/dt points (captoolkit xover.py output).

Crossovers have no fit uncertainty, so the elevation difference (dh_elv), the
time separation (dt_year) and the local along-track slope (slp_1, slp_2) are
used as quality criteria, followed by the local spatial outlier filter.
Settings: icesat2/config.py. QC figure: plot_filter_qc.py.
"""
import geopandas as gpd

import config as cfg
from point_filters import local_mad_filter, range_filter, save_results


def main():
    gdf = gpd.read_file(cfg.XO_INPUT)
    n0 = len(gdf)
    print(f"Loaded {n0} points from {cfg.XO_INPUT}")

    gdf, rm_range = range_filter(gdf, "dhdt", cfg.XO_MIN_DHDT, cfg.XO_MAX_DHDT, "dh/dt range")

    gdf = gdf.assign(_abs_dh=gdf["dh_elv"].abs(), _abs_dt=gdf["dt_year"].abs(),
                     _slope=gdf[["slp_1", "slp_2"]].abs().max(axis=1))
    for q in (0.95, 0.99, 0.999):
        print(f"  {q * 100:.1f}th percentile |dh_elv| = {gdf['_abs_dh'].quantile(q):.2f} m, "
              f"slope = {gdf['_slope'].quantile(q):.2f} deg")

    gdf, rm_dh = range_filter(gdf, "_abs_dh", 0, cfg.XO_MAX_DH_ELV, "|dh_elv|")
    if cfg.XO_MIN_DT_YEAR > 0:
        gdf, rm_dt = range_filter(gdf, "_abs_dt", cfg.XO_MIN_DT_YEAR, float("inf"), "|dt_year|")
    else:
        rm_dt = gdf.iloc[:0]
    gdf, rm_slope = range_filter(gdf, "_slope", 0, cfg.XO_MAX_SLOPE_DEG, "slope")
    clean, rm_local = local_mad_filter(gdf, "dhdt", cfg.K_NEIGHBORS, cfg.LOCAL_N_SIGMA, cfg.MAD_FLOOR)

    drop = ["_abs_dh", "_abs_dt", "_slope"]
    print(f"Retained {len(clean)} of {n0} points ({100 * len(clean) / n0:.1f}%)")
    save_results(clean.drop(columns=drop),
                 {k: v.drop(columns=drop, errors="ignore") for k, v in
                  {"range": rm_range, "dh_elv": rm_dh, "dt_year": rm_dt,
                   "slope": rm_slope, "local": rm_local}.items()},
                 cfg.XO_OUTPUT)


if __name__ == "__main__":
    main()
