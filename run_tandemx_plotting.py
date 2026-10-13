"""
Figures of the TanDEM-X analysis (settings: tandemx/config.py).
Requires a previous run of run_tandemx_processing.py.
"""
import warnings

from tandemx import config as cfg
from tandemx.plotting.centerline_profiles import plot_centerline_elevation_changes
from tandemx.plotting.hypsometry import (
    plot_hypsometric_analysis,
    plot_hypsometric_analysis_all_periods,
    plot_mass_balance_uncertainty,
    plot_mass_balance_uncertainty_paper,
    plot_penetration_onset,
    plot_sar_penetration_diagnostics,
    plot_time_series,
)
from tandemx.plotting.maps import (
    plot_comprehensive_summary,
    plot_extended_spatial_maps,
    plot_spatial_maps,
    plot_uncertainty_and_filtering_maps,
)
from tandemx.state import load_state
from tandemx.utils import load_reference_products, setup_grid

warnings.filterwarnings("ignore")


def main():
    cfg.ensure_output_dirs()
    st = load_state()
    year_results, all_changes, mb_results = st["year_results"], st["all_changes"], st["mb_results"]
    grid_info, bad_mask = setup_grid()
    _, slope, slope_bins, ref_dem = load_reference_products(grid_info, bad_mask)
    outlines = list(cfg.YEAR_MASKS.values())

    # Hypsometry, mass balance and penetration diagnostics
    plot_penetration_onset(st["onset"])
    if mb_results:
        plot_sar_penetration_diagnostics(mb_results)
        plot_mass_balance_uncertainty(mb_results)
        plot_mass_balance_uncertainty_paper(mb_results, show_bin_errorbars=False)
        plot_mass_balance_uncertainty_paper(mb_results, show_bin_errorbars=True)
    plot_time_series(all_changes, year_results)
    plot_hypsometric_analysis(all_changes, year_results, mb_results=mb_results)
    plot_hypsometric_analysis_all_periods(all_changes, year_results, mb_results=mb_results)

    # Maps
    if all_changes:
        plot_uncertainty_and_filtering_maps(
            all_changes=all_changes, year_results=year_results, grid_info=grid_info,
            preloaded_slope=slope, preloaded_slope_bins=slope_bins,
            GLACIER_MASK_DIR=cfg.GLACIER_MASK_DIR, GLACIER_INVENTORY_MAP=cfg.GLACIER_INVENTORY_MAP,
            DEFAULT_GLACIER_SHP=cfg.DEFAULT_GLACIER_SHP, STABLE_TERRAIN_DIR=cfg.STABLE_TERRAIN_DIR)
        plot_spatial_maps(all_changes, grid_info, bad_mask, mb_results=mb_results,
                          shp_path=cfg.DEFAULT_GLACIER_SHP, union_shp_paths=outlines)
        plot_extended_spatial_maps(all_changes, mb_results, grid_info, bad_mask,
                                   density_A=cfg.DENSITY_A, density_B=cfg.DENSITY_B,
                                   shp_path=cfg.DEFAULT_GLACIER_SHP)
        plot_comprehensive_summary(year_results=year_results, all_changes=all_changes,
                                   mb_results=mb_results, grid_info=grid_info, bad_mask=bad_mask,
                                   shp_path=cfg.MASK_PATH, union_shp_paths=outlines)

    # Centerline profiles and multi-panel centerline figures
    plot_centerline_elevation_changes(all_changes, year_results, grid_info, cfg.CENTERLINE_DIR,
                                      preloaded_ref_dem=ref_dem, mb_results=mb_results)
    print("\n=== TanDEM-X figures complete ===")


if __name__ == "__main__":
    main()
