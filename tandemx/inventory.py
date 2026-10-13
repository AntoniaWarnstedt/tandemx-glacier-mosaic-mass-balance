"""Glacier-inventory geometries and areas (start/end/union outlines per period).
"""

import os
import re

import geopandas as gpd
import numpy as np
from shapely.ops import unary_union

from . import config as cfg
from .utils import canonical_period_key, mask_year_for_label, pixel_area_km2_from_grid, rasterize_geometry_to_grid_mask

_PERIOD_INVENTORY_OUTLINE_CACHE = {}  # (yA, yB) -> outline GeoDataFrame


def get_inventory_for_period(yA: str):
    """Inventory entry of an epoch label (accepts e.g. "2016-2017" or "2016/17")."""
    key = canonical_period_key(yA)
    m = re.fullmatch(r"(\d{4})-(\d{2})", key)
    if m:
        key = f"{m.group(1)}-{m.group(1)[:2]}{m.group(2)}"

    inventory = cfg.GLACIER_INVENTORY_MAP.get(key)
    if inventory is None:
        # fallback to single year if possible (e.g., "2016")
        inventory = cfg.GLACIER_INVENTORY_MAP.get(key.split("-")[0], {})
    return inventory if inventory is not None else {}


def load_glacier_inventory_shp_for_label(label, GLACIER_MASK_DIR, GLACIER_INVENTORY_MAP,
                                         DEFAULT_GLACIER_SHP):
    """
    Load glacier inventory shapefile path for a specific label.
    """
    inventory = get_inventory_for_period(label)
    shp_name = inventory.get("shp", None)

    if shp_name is not None:
        shp_path = os.path.join(GLACIER_MASK_DIR, shp_name)
    else:
        shp_path = DEFAULT_GLACIER_SHP

    if not os.path.exists(shp_path):
        shp_path = DEFAULT_GLACIER_SHP

    if not os.path.exists(shp_path):
        raise FileNotFoundError(f"Could not load glacier inventory shapefile for label {label}: {shp_path}")

    return shp_path


def load_glacier_inventory_gdf_for_label(label, GLACIER_MASK_DIR, GLACIER_INVENTORY_MAP,
                                         DEFAULT_GLACIER_SHP, target_crs=cfg.CRS):
    """
    Load glacier inventory shapefile for a specific label and force projected CRS for area calculations.
    """
    shp_path = load_glacier_inventory_shp_for_label(
        label, GLACIER_MASK_DIR, GLACIER_INVENTORY_MAP, DEFAULT_GLACIER_SHP
    )

    gdf = gpd.read_file(shp_path)
    if gdf.empty:
        raise ValueError(f"Inventory shapefile is empty for label {label}: {shp_path}")

    gdf = gdf[gdf.geometry.notnull()].copy()
    gdf = gdf[~gdf.geometry.is_empty].copy()

    if gdf.crs is None:
        gdf = gdf.set_crs(target_crs)
    elif gdf.crs.to_string() != target_crs:
        gdf = gdf.to_crs(target_crs)

    return gdf


def load_year_mask_geom_for_label(label, target_crs=cfg.CRS):
    """
    Resolve a period/year label to the terminus-outline geometry from
    YEAR_MASKS via MASK_ASSIGNMENT -- i.e. the SAME outline_*.shp files that
    draw the dashed terminus lines / clip the centerline segments in the
    plotting code (see get_centerline_segment_by_year, which uses the identical
    MASK_ASSIGNMENT -> YEAR_MASKS resolution).
    """
    mask_year = mask_year_for_label(label)
    mask_path = cfg.YEAR_MASKS.get(mask_year)

    if mask_path is None or not os.path.exists(mask_path):
        print(f"      ⚠️  load_year_mask_geom_for_label: no usable YEAR_MASKS path "
              f"for label={label!r} (resolved mask_year={mask_year!r}, path={mask_path!r})")
        return None

    try:
        gdf = gpd.read_file(mask_path)
        gdf = gdf[gdf.geometry.notnull() & ~gdf.geometry.is_empty].copy()
        if gdf.empty:
            return None

        # Matches build_union_outline_gdf's exact field-detection logic for
        # consistency.
        valid_field = next((c for c in ['DN', 'Id', 'ID', 'id'] if c in gdf.columns), None)
        if valid_field is not None:
            n_before = len(gdf)
            gdf = gdf[gdf[valid_field] == 1]
            print(f"      load_year_mask_geom_for_label({label!r}): filtered "
                  f"{os.path.basename(mask_path)} to {valid_field}==1 "
                  f"({len(gdf)}/{n_before} features kept)")
            if gdf.empty:
                print(f"      ⚠️  load_year_mask_geom_for_label({label!r}): no "
                      f"{valid_field}==1 features in {os.path.basename(mask_path)}")
                return None
        else:
            print(f"      ⚠️  load_year_mask_geom_for_label({label!r}): no DN/Id/ID/id "
                  f"field found in {os.path.basename(mask_path)} -- using all "
                  f"{len(gdf)} feature(s) unfiltered.")

        if gdf.crs is None:
            gdf = gdf.set_crs(target_crs)
        elif gdf.crs.to_string() != target_crs:
            gdf = gdf.to_crs(target_crs)
        return unary_union(gdf.geometry)
    except Exception as e:
        print(f"      ⚠️  load_year_mask_geom_for_label: could not load {os.path.basename(mask_path)} "
              f"for label={label!r}: {e}")
        return None


def compute_inventory_area_metrics_option_c(yA, yB, GLACIER_MASK_DIR, GLACIER_INVENTORY_MAP,
                                            DEFAULT_GLACIER_SHP, target_metric_crs=cfg.CRS,
                                            area_mask_option=None):
    """Compute start/end/mean/union/intersection inventory polygon areas in km²."""
    if area_mask_option is None:
        area_mask_option = cfg.AREA_MASK_OPTION

    start_gdf = load_glacier_inventory_gdf_for_label(
        yA, GLACIER_MASK_DIR, GLACIER_INVENTORY_MAP, DEFAULT_GLACIER_SHP, target_crs=target_metric_crs
    )
    end_gdf = load_glacier_inventory_gdf_for_label(
        yB, GLACIER_MASK_DIR, GLACIER_INVENTORY_MAP, DEFAULT_GLACIER_SHP, target_crs=target_metric_crs
    )

    start_geom = unary_union(start_gdf.geometry)
    end_geom = unary_union(end_gdf.geometry)

    union_geom = start_geom.union(end_geom)
    intersection_geom = start_geom.intersection(end_geom)

    area_start_km2 = start_geom.area * cfg.M2_TO_KM2
    area_end_km2 = end_geom.area * cfg.M2_TO_KM2
    area_mean_km2 = 0.5 * (area_start_km2 + area_end_km2)
    area_union_km2 = union_geom.area * cfg.M2_TO_KM2
    area_intersection_km2 = intersection_geom.area * cfg.M2_TO_KM2

    if area_mask_option == "B":
        # trim the end-of-period catchment with the end-year terminus outline
        year_mask_geom = load_year_mask_geom_for_label(yB, target_crs=target_metric_crs)
        trimmed_geom = None
        if year_mask_geom is not None and not year_mask_geom.is_empty:
            candidate = end_geom.intersection(year_mask_geom)
            candidate_area_km2 = candidate.area * cfg.M2_TO_KM2 if not candidate.is_empty else 0.0
            retained_fraction = (
                candidate_area_km2 / area_end_km2 if area_end_km2 > 0 else 0.0
            )
            if candidate_area_km2 > 0 and retained_fraction >= cfg.OPTION_B_MIN_RETAINED_FRACTION:
                trimmed_geom = candidate
                print(f"      Option B [{yA}-{yB}]: end_geom ({area_end_km2:.2f} km²) "
                      f"trimmed by YEAR_MASKS terminus outline -> "
                      f"{candidate_area_km2:.2f} km² ({100 * retained_fraction:.1f}% retained)")
            else:
                print(f"      ⚠️  Option B [{yA}-{yB}]: YEAR_MASKS-trimmed intersection "
                      f"only retains {100 * retained_fraction:.1f}% of end_geom's area "
                      f"({candidate_area_km2:.2f} of {area_end_km2:.2f} km²) -- below "
                      f"OPTION_B_MIN_RETAINED_FRACTION={cfg.OPTION_B_MIN_RETAINED_FRACTION}. "
                      f"Falling back to end_geom UNTRIMMED (mask_glacier_*.shp only) -- "
                      f"check that GLACIER_INVENTORY_MAP and YEAR_MASKS actually agree on "
                      f"this glacier's footprint/CRS before trusting this period's Option B result.")
        else:
            print(f"      ⚠️  Option B [{yA}-{yB}]: no usable YEAR_MASKS terminus outline for "
                  f"yB={yB!r}; falling back to end_geom UNTRIMMED (mask_glacier_*.shp only).")

        if trimmed_geom is not None:
            support_geom = trimmed_geom
            support_area_km2 = trimmed_geom.area * cfg.M2_TO_KM2
            support_label = "end_only_terminus_trimmed"
        else:
            support_geom = end_geom
            support_area_km2 = area_end_km2
            support_label = "end_only"
    elif area_mask_option == "C":
        support_geom = union_geom
        support_area_km2 = area_union_km2
        support_label = "union"
    else:
        raise ValueError(f"Unknown area_mask_option {area_mask_option!r}; expected 'B' or 'C'")

    return {
        "Area_start_km2": float(area_start_km2),
        "Area_end_km2": float(area_end_km2),
        "Area_mean_km2": float(area_mean_km2),
        "Area_union_km2": float(area_union_km2),
        "Area_intersection_km2": float(area_intersection_km2),
        "start_gdf": start_gdf,
        "end_gdf": end_gdf,
        "start_geom": start_geom,
        "end_geom": end_geom,
        "union_geom": union_geom,
        "intersection_geom": intersection_geom,
        # analysis domain of the period (see cfg.AREA_MASK_OPTION)
        "support_geom": support_geom,
        "support_area_km2": float(support_area_km2),
        "support_label": support_label,
        "area_mask_option": area_mask_option,
    }


def compute_observed_support_areas_km2(dhdt_input, grid_info, start_geom, end_geom, union_geom,
                                       intersection_geom):
    """
    Compute observed support area in km² over start/end/union/intersection
    glacier geometries.
    """
    pixel_area_km2 = pixel_area_km2_from_grid(grid_info)
    obs_mask = np.isfinite(dhdt_input)

    start_mask = rasterize_geometry_to_grid_mask(start_geom, grid_info, invert=True)
    end_mask = rasterize_geometry_to_grid_mask(end_geom, grid_info, invert=True)
    union_mask = rasterize_geometry_to_grid_mask(union_geom, grid_info, invert=True)
    intersection_mask = rasterize_geometry_to_grid_mask(intersection_geom, grid_info, invert=True)

    obs_start_mask = obs_mask & start_mask
    obs_end_mask = obs_mask & end_mask
    obs_union_mask = obs_mask & union_mask
    obs_intersection_mask = obs_mask & intersection_mask

    return {
        "Obs_area_start_km2": float(np.sum(obs_start_mask) * pixel_area_km2),
        "Obs_area_end_km2": float(np.sum(obs_end_mask) * pixel_area_km2),
        "Obs_area_union_km2": float(np.sum(obs_union_mask) * pixel_area_km2),
        "Obs_area_intersection_km2": float(np.sum(obs_intersection_mask) * pixel_area_km2),
        "obs_start_mask": obs_start_mask,
        "obs_end_mask": obs_end_mask,
        "obs_union_mask": obs_union_mask,
        "obs_intersection_mask": obs_intersection_mask,
        "obs_any_mask": obs_mask
    }


def compute_area_perimeter_error_fraction_from_geom(geom):
    """
    Compute area uncertainty fraction from a single geometry using the methods logic.
    Areafrac = (P_km / A_km2) * 0.006
    """
    if geom is None or geom.is_empty:
        return np.nan, 0.0, 0.0

    total_area_m2 = geom.area
    total_perimeter_m = geom.length

    if total_area_m2 <= 0:
        return np.nan, total_area_m2, total_perimeter_m

    # Areafrac = (P[km] / A[km^2]) * 0.006 -- perimeter m->km via M_TO_KM,
    # area m^2->km^2 via M2_TO_KM2 (see UNIT-CONVERSION CONSTANTS block)
    ref_p_a_ratio = (total_perimeter_m * cfg.M_TO_KM) / (total_area_m2 * cfg.M2_TO_KM2)
    glac_area_error = ref_p_a_ratio * 0.006

    return glac_area_error, total_area_m2, total_perimeter_m


def build_union_outline_gdf(mask_paths, target_crs):
    """
    Union multiple mask shapefiles' valid (field==1, or all features if no
    valid field) geometries into a single-row GeoDataFrame boundary, in
    target_crs.
    """
    geoms = []
    for mp in mask_paths:
        try:
            gdf = gpd.read_file(mp)
            valid_field = next((c for c in ['DN', 'Id', 'ID', 'id'] if c in gdf.columns), None)
            if valid_field is not None:
                gdf = gdf[gdf[valid_field] == 1]
            gdf = gdf.to_crs(target_crs)
            geoms.extend(g for g in gdf.geometry.tolist() if g is not None and not g.is_empty)
        except Exception as e:
            print(f"      ⚠️  Could not load {os.path.basename(mp)} for union outline: {e}")

    if not geoms:
        return None

    union_geom = unary_union(geoms)
    return gpd.GeoDataFrame(geometry=[union_geom], crs=target_crs)


def get_period_inventory_outline_gdf(yA, yB, grid_info):
    """
    Build a single-row GeoDataFrame boundary for THIS PERIOD's inventory (Ainv)
    extent -- start_geom UNION end_geom from GLACIER_INVENTORY_MAP, i.e.
    inventory_area_stats["union_geom"] from
    compute_inventory_area_metrics_option_c(yA, yB, ...).
    """
    cache_key = (yA, yB)
    if cache_key in _PERIOD_INVENTORY_OUTLINE_CACHE:
        return _PERIOD_INVENTORY_OUTLINE_CACHE[cache_key]

    gdf = None
    try:
        stats = compute_inventory_area_metrics_option_c(
            yA, yB, cfg.GLACIER_MASK_DIR, cfg.GLACIER_INVENTORY_MAP, cfg.DEFAULT_GLACIER_SHP,
            target_metric_crs=grid_info['crs'],
            area_mask_option="C",
        )
        geom = stats["union_geom"]
        if geom is not None and not geom.is_empty:
            gdf = gpd.GeoDataFrame(geometry=[geom], crs=grid_info['crs'])
    except Exception as e:
        print(f"      ⚠️  get_period_inventory_outline_gdf({yA!r}, {yB!r}): could not build "
              f"period-specific Ainv outline: {e}")
        gdf = None

    _PERIOD_INVENTORY_OUTLINE_CACHE[cache_key] = gdf
    return gdf


def get_period_inventory_exclude_mask(period_outline_gdf, grid_info):
    """
    Boolean grid mask, True OUTSIDE period_outline_gdf's geometry -- i.e. the
    pixels a displayed dh/dt (or filter-difference) raster should have set to
    NaN so it never paints past the period-specific Ainv outline drawn on top
    of it by get_period_inventory_outline_gdf. Same True = "should be NaN'd
    out" convention as bad_mask.
    """
    if period_outline_gdf is None or len(period_outline_gdf) == 0:
        return None
    try:
        geom = period_outline_gdf.geometry.iloc[0]
    except Exception:
        return None
    if geom is None or geom.is_empty:
        return None
    inside_mask = rasterize_geometry_to_grid_mask(geom, grid_info, invert=True)
    return ~inside_mask
