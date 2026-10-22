"""
TanDEM-X processing chain (settings: tandemx/config.py).

1. Master grid, study-area mask and reference-DEM products
2. DEM mosaic per year, merged mosaics for cfg.MERGED_EPOCHS
3. dh and dh/dt for every non-overlapping epoch pair
4. SAR-penetration ramp (onset from inter-period consistency)
5. Void filling, stable-terrain uncertainty and mass balance per period
6. Summary tables + processing state for run_tandemx_plotting.py
"""
import os
import warnings

import numpy as np

from tandemx import config as cfg
from tandemx.elevation_change import compute_elevation_change
from tandemx.export import (
    save_regional_mass_balance_summary,
    save_regional_mass_balance_summary_main_results,
    save_regional_mass_balance_summary_paper,
)
from tandemx.mass_balance import calculate_mass_balance_following_r_logic
from tandemx.mosaicking import merge_multiyear_dems, mosaic_year_v2
from tandemx.penetration import resolve_penetration_ramp
from tandemx.state import save_state
from tandemx.utils import (
    analysis_epochs,
    canonical_period_key,
    create_coordinate_arrays,
    is_valid_period_comparison,
    load_reference_products,
    save_raster,
    setup_grid,
)

warnings.filterwarnings("ignore")


def main():
    if not cfg.TDX_YEARS:
        raise SystemExit("Set TDX_YEARS (and the paths) in tandemx/config.py first.")
    cfg.ensure_output_dirs()

    # 1. Grid, mask, reference products
    grid_info, bad_mask = setup_grid()
    x_coords, y_coords = create_coordinate_arrays(grid_info["shape"], grid_info["transform"])
    elev_bins, slope, slope_bins, ref_dem = load_reference_products(grid_info, bad_mask)

    # 2. DEM mosaics
    year_results = {}
    for year, year_path in cfg.YEAR_DIRS.items():
        if os.path.exists(year_path):
            res = mosaic_year_v2(year_path, year, grid_info, bad_mask, x_coords, y_coords)
            if res:
                year_results[year] = res
    for pair in cfg.MERGED_EPOCHS:
        if all(y in year_results for y in pair):
            merged = merge_multiyear_dems(year_results, pair, grid_info, bad_mask, x_coords, y_coords)
            if merged:
                year_results["-".join(pair)] = merged

    # 3. Elevation change for all valid epoch pairs (slopes > 45 deg excluded)
    epochs = analysis_epochs(year_results.keys())
    all_changes = []
    for i, yA in enumerate(epochs[:-1]):
        for yB in epochs[i + 1:]:
            if not is_valid_period_comparison(yA, yB):
                continue
            res = compute_elevation_change(year_results[yA], year_results[yB], yA, yB,
                                           grid_info, bad_mask, slope_mask=slope)
            if res:
                all_changes.append(res)

    # 4. Penetration ramp (same for all periods)
    onset = resolve_penetration_ramp(all_changes, elev_bins, ref_dem=ref_dem,
                                     catchment_extent=~bad_mask)

    # 5. Mass balance per period
    mb_results = []
    for change in all_changes:
        res = calculate_mass_balance_following_r_logic(
            change_dict=change,
            year_results=year_results,
            grid_info=grid_info,
            preloaded_elev_bins=elev_bins,
            preloaded_ref_dem=ref_dem,
            preloaded_slope=slope,
            preloaded_slope_bins=slope_bins,
            GLACIER_MASK_DIR=cfg.GLACIER_MASK_DIR,
            GLACIER_INVENTORY_MAP=cfg.GLACIER_INVENTORY_MAP,
            DEFAULT_GLACIER_SHP=cfg.DEFAULT_GLACIER_SHP,
            STABLE_TERRAIN_DIR=cfg.STABLE_TERRAIN_DIR,
            sar_pen_type=cfg.SAR_PEN_TYPE,
            sar_pen_bias=cfg.SAR_PEN_BIAS,
            density_A=cfg.DENSITY_A,
            density_B=cfg.DENSITY_B,
            region_label=cfg.REGION_LABEL,
        )
        if not res:
            continue
        mb_results.append(res)
        slug = canonical_period_key(res["Period"])
        save_raster(res["dhdt_voi_raster"].astype(np.float32),
                    os.path.join(cfg.OUT_DIR, f"dhdtfilled_onice_hypsometric_{slug}.tif"),
                    grid_info, units="m a-1",
                    description=f"Void-filled elevation-change rate, {res['Period']}")
        if res.get("onice_bin_stats") is not None and not res["onice_bin_stats"].empty:
            res["onice_bin_stats"].to_csv(os.path.join(cfg.OUT_DIR, f"onice_bins_{slug}.csv"), index=False)
        if res.get("stable_slope_bin_stats") is not None and not res["stable_slope_bin_stats"].empty:
            res["stable_slope_bin_stats"].to_csv(
                os.path.join(cfg.OUT_DIR, f"stable_slope_bins_{slug}.csv"), index=False)

    # 6. Tables and state
    if mb_results:
        save_regional_mass_balance_summary(mb_results, cfg.OUT_DIR)
        save_regional_mass_balance_summary_paper(mb_results, cfg.OUT_DIR)
        save_regional_mass_balance_summary_main_results(mb_results, cfg.OUT_DIR)
    save_state(year_results, all_changes, mb_results, grid_info, onset)
    print("\n=== TanDEM-X processing complete ===")


if __name__ == "__main__":
    main()
