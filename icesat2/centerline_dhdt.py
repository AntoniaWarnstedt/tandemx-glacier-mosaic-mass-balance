"""
ICESat-2 dh/dt along the glacier centerlines: the kriged dh/dt grid is
sampled every CL_SAMPLING_DISTANCE m (mean across +-CL_BUFFER_WIDTH m) from the
terminus inland, up to CL_MAX_DISTANCE. Each centerline is clipped to the
catchment outline CL_TERMINUS_MASK and oriented so that distance 0 is its
seaward end. Settings: config.py (section 5).

Outputs (CL_OUTPUT_DIR):
  <GRID_PREFIX>_mean_dhdt_per_centerline.csv   mean dh/dt per glacier
  <GRID_PREFIX>_dhdt_profiles.csv              profiles (input of plot_centerline_dhdt.py)
"""
import glob
import os

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from shapely.geometry import LineString, Point

import config as cfg


def dhdt_raster_path():
    return cfg.CL_DHDT_RASTER or os.path.join(cfg.GRID_OUTPUT_DIR, f"{cfg.GRID_PREFIX}_dhdt.tif")


def get_longest_linestring(geom):
    """Longest LineString of a (Multi)LineString / GeometryCollection, or None."""
    if geom is None or geom.is_empty:
        return None
    if geom.geom_type == "LineString":
        return geom
    if geom.geom_type == "MultiLineString":
        lines = [g for g in geom.geoms if g.geom_type == "LineString" and not g.is_empty]
        return max(lines, key=lambda g: g.length) if lines else None
    if geom.geom_type == "GeometryCollection":
        lines = []
        for g in geom.geoms:
            if g.geom_type == "LineString" and not g.is_empty:
                lines.append(g)
            elif g.geom_type == "MultiLineString" and not g.is_empty:
                lines.extend([sg for sg in g.geoms if sg.geom_type == "LineString" and not sg.is_empty])
        return max(lines, key=lambda g: g.length) if lines else None
    return None


def align_centerline_to_terminus(line, mask_geom):
    """
    Clip the centerline to the outline and orient it so that distance 0 is at
    the terminus (the end towards the original endpoint farthest outside the
    outline, i.e. the ocean) and increases inland.
    """
    if mask_geom is None or mask_geom.is_empty:
        return line

    if mask_geom.geom_type in ['Polygon', 'MultiPolygon']:
        clipped_line = get_longest_linestring(line.intersection(mask_geom))
        if clipped_line is None:
            return line   # no intersection

        coords = list(clipped_line.coords)
        p_start, p_end = Point(coords[0]), Point(coords[-1])

        orig_p1, orig_p2 = Point(line.coords[0]), Point(line.coords[-1])
        dist1_outside = mask_geom.distance(orig_p1)
        dist2_outside = mask_geom.distance(orig_p2)
        if dist1_outside > dist2_outside:
            ocean_point = orig_p1
        elif dist2_outside > dist1_outside:
            ocean_point = orig_p2
        else:
            # neither end outside: the end closest to the outline boundary
            boundary = mask_geom.boundary
            ocean_point = orig_p1 if p_start.distance(boundary) < p_end.distance(boundary) else orig_p2

        if p_start.distance(ocean_point) > p_end.distance(ocean_point):
            clipped_line = LineString(coords[::-1])
        return clipped_line

    # outline given as terminus line(s)
    coords = list(line.coords)
    if Point(coords[0]).distance(mask_geom) > Point(coords[-1]).distance(mask_geom):
        return LineString(coords[::-1])
    return line


def sample_raster_along_line_with_buffer(raster_data, transform, line, sampling_dist,
                                         buffer_width=100, include_nan=False,
                                         max_distance=None, stat='mean'):
    """
    Sample a raster every `sampling_dist` m along `line`; each sample is the
    mean (or median) of the pixels on a perpendicular transect of
    +-buffer_width m. Returns (distances, values).
    """
    line_length = line.length if max_distance is None else min(line.length, max_distance)
    distances = np.arange(0, line_length, sampling_dist)
    if not distances.size:
        return np.array([]), np.array([])

    sampled_values, valid_distances = [], []
    pixel_size = abs(transform.a)
    buffer_pixels = int(np.ceil(buffer_width / pixel_size))

    for dist in distances:
        center_point = line.interpolate(dist)
        next_point = line.interpolate(dist + 1) if dist + 1 < line.length else line.interpolate(dist - 1)
        dx, dy = next_point.x - center_point.x, next_point.y - center_point.y
        length = np.sqrt(dx ** 2 + dy ** 2)
        if length == 0:
            continue
        perp_dx, perp_dy = -dy / length, dx / length

        buffer_values = []
        for offset in range(-buffer_pixels, buffer_pixels + 1):
            offset_m = offset * pixel_size
            col, row = ~transform * (center_point.x + offset_m * perp_dx, center_point.y + offset_m * perp_dy)
            col, row = int(round(col)), int(round(row))
            if 0 <= row < raster_data.shape[0] and 0 <= col < raster_data.shape[1]:
                value = raster_data[row, col]
                if np.isfinite(value):
                    buffer_values.append(value)

        if buffer_values:
            agg = np.median(buffer_values) if stat == 'median' else np.mean(buffer_values)
            if include_nan or np.isfinite(agg):
                sampled_values.append(agg)
                valid_distances.append(dist)

    return np.array(valid_distances), np.array(sampled_values)


def main():
    print("\n=== ICESat-2 dh/dt along the centerlines ===\n")
    os.makedirs(cfg.CL_OUTPUT_DIR, exist_ok=True)

    with rasterio.open(dhdt_raster_path()) as src:
        dhdt = src.read(1)
        transform, crs, nodata = src.transform, src.crs, src.nodata
    if nodata is not None:
        dhdt[dhdt == nodata] = np.nan

    mask_gdf = gpd.read_file(cfg.CL_TERMINUS_MASK)
    if mask_gdf.crs != crs:
        mask_gdf = mask_gdf.to_crs(crs)
    mask_geom = mask_gdf.unary_union

    centerline_files = sorted(glob.glob(os.path.join(cfg.CL_DIR, "*.shp")))
    if not centerline_files:
        print("No centerlines found.")
        return

    means, profiles = [], []
    for cl_file in centerline_files:
        centerline_name = os.path.splitext(os.path.basename(cl_file))[0]
        glacier_name = centerline_name.replace("Centerline_", "").replace("_", " ")
        print(f"Processing: {glacier_name}")
        try:
            gdf = gpd.read_file(cl_file)
            if gdf.empty:
                continue
            if gdf.crs != crs:
                gdf = gdf.to_crs(crs)
            raw_line = get_longest_linestring(gdf.unary_union)
            if raw_line is None:
                print(f"  Warning: No valid LineString found for {glacier_name}")
                continue

            line = align_centerline_to_terminus(raw_line, mask_geom)
            distances, values = sample_raster_along_line_with_buffer(
                dhdt, transform, line, cfg.CL_SAMPLING_DISTANCE, buffer_width=cfg.CL_BUFFER_WIDTH,
                max_distance=cfg.CL_MAX_DISTANCE, stat='mean')
            if len(values) == 0:
                print(f"  Warning: No valid dh/dt values found along {glacier_name}")
                continue

            mean_dhdt = float(np.nanmean(values))
            means.append((glacier_name, mean_dhdt))
            profiles.append(pd.DataFrame({"glacier": glacier_name, "distance_m": distances,
                                          "dhdt_m_a": values}))
            print(f"  Mean dh/dt = {mean_dhdt:.3f} m/yr (over {distances[-1] / 1000:.1f} km)")
        except Exception as e:
            print(f"  Error processing {glacier_name}: {e}")

    if not means:
        print("No valid results extracted.")
        return

    mean_csv = os.path.join(cfg.CL_OUTPUT_DIR, f"{cfg.GRID_PREFIX}_mean_dhdt_per_centerline.csv")
    pd.DataFrame({"glacier": [g for g, _ in means],
                  "mean_dhdt_m_per_yr": [f"{m:.4f}" for _, m in means]}).to_csv(mean_csv, index=False)
    prof_csv = os.path.join(cfg.CL_OUTPUT_DIR, f"{cfg.GRID_PREFIX}_dhdt_profiles.csv")
    pd.concat(profiles, ignore_index=True).to_csv(prof_csv, index=False)
    print(f"\n✓ Saved: {mean_csv}\n✓ Saved: {prof_csv}")


if __name__ == "__main__":
    main()
