"""
Figures of the TanDEM-X analysis (settings: tandemx/config.py).
Requires a previous run of run_tandemx_processing.py.

1. comprehensive_summary_maps_3x3.png        DEMs, dh/dt and void-filled dh/dt
2. comprehensive_summary_hypsometry_1x3.png  area and dh/dt per elevation bin
3. centerline_analysis/combined_9panel_dem_and_dhdt_profiles_satellite.png
                                             surface and dh/dt profiles along the centerlines
4. centerline_analysis/glacier_comparison_heatmap.png
                                             mean centerline dh/dt per glacier and period
"""
import warnings

from tandemx import config as cfg
from tandemx.plotting.centerline_panels import plot_centerline_profiles_figure
from tandemx.plotting.centerlines import collect_centerline_profiles, plot_glacier_period_heatmap
from tandemx.plotting.summary import plot_comprehensive_summary
from tandemx.state import load_state
from tandemx.utils import setup_grid

warnings.filterwarnings("ignore")


def main():
    cfg.ensure_output_dirs()
    st = load_state()
    year_results, all_changes, mb_results = st["year_results"], st["all_changes"], st["mb_results"]
    grid_info, bad_mask = setup_grid()

    # Figures 1 and 2
    if all_changes:
        plot_comprehensive_summary(year_results=year_results, all_changes=all_changes,
                                   mb_results=mb_results, grid_info=grid_info, bad_mask=bad_mask,
                                   shp_path=cfg.MASK_PATH, union_shp_paths=list(cfg.YEAR_MASKS.values()))

    # Figures 3 and 4
    all_glacier_data, glacier_profiles, periods = collect_centerline_profiles(
        all_changes, year_results, grid_info, mb_results=mb_results)
    plot_glacier_period_heatmap(all_glacier_data, periods)
    plot_centerline_profiles_figure(glacier_profiles, all_glacier_data, grid_info)
    print("\n=== TanDEM-X figures complete ===")


if __name__ == "__main__":
    main()
