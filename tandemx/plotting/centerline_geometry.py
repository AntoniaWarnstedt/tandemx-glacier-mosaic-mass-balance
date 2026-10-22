"""Centerline helpers: centerline files, clipping to yearly outlines and
sampling rasters along a centerline swath.
"""

import glob
import os
import re

import geopandas as gpd
import numpy as np

from .. import config as cfg
from ..utils import get_longest_linestring, mask_year_for_label


def list_centerline_files():
    """Centerline shapefiles in cfg.CENTERLINE_DIR, sorted by name; glaciers are
    numbered [1], [2], ... in this order in all figures."""
    if not cfg.CENTERLINE_DIR or not os.path.exists(cfg.CENTERLINE_DIR):
        return []
    return sorted(glob.glob(os.path.join(cfg.CENTERLINE_DIR, "*.shp")))


def glacier_name_from_file(centerline_file):
    """'Centerline_Fleming_04.shp' -> 'Fleming'."""
    name = os.path.splitext(os.path.basename(centerline_file))[0].replace('Centerline_', '')
    return re.sub(r'_\d{2}$', '', name).replace('_', ' ')


def clip_centerline_to_mask(centerline, mask_path, target_crs):
    """
    Intersect a centerline with a polygon mask and return the longest clipped LineString.
    """
    if mask_path is None or not os.path.exists(mask_path):
        return None

    try:
        mask_gdf = gpd.read_file(mask_path)
        if mask_gdf.empty:
            return None

        mask_geom = mask_gdf.to_crs(target_crs).unary_union
        clipped = centerline.intersection(mask_geom)
        return get_longest_linestring(clipped)

    except Exception as e:
        print(f"      ⚠️  Could not clip centerline with mask {os.path.basename(mask_path)}: {e}")
        return None


def get_centerline_segment_by_year(centerline, year_label, target_crs):
    """
    Return the centerline segment clipped to the mask assigned to this year/period.
    """
    mask_year = mask_year_for_label(year_label)
    mask_path = cfg.YEAR_MASKS.get(mask_year)

    if mask_path is None:
        return None

    return clip_centerline_to_mask(centerline, mask_path, target_crs)


def get_segment_distance_range_on_centerline(centerline, segment):
    """
    Return start and end distances of a clipped segment projected onto the full centerline.
    """
    if segment is None or segment.is_empty:
        return None, None

    start_pt = segment.interpolate(0)
    end_pt = segment.interpolate(segment.length)

    d0 = centerline.project(start_pt)
    d1 = centerline.project(end_pt)

    return min(d0, d1), max(d0, d1)


def sample_raster_along_line_with_buffer(raster_data, transform, line, sampling_dist,
                                         buffer_width=100, include_nan=False,
                                         max_distance=None, stat='mean'):
    """Sample raster values along a LineString with a buffer zone."""

    line_length = line.length if max_distance is None else min(line.length, max_distance)
    distances = np.arange(0, line_length, sampling_dist)

    if not distances.size:
        return np.array([]), np.array([])

    sampled_values = []
    valid_distances = []

    # Get pixel size in meters
    pixel_size = abs(transform.a)

    # Calculate how many pixels correspond to buffer width
    buffer_pixels = int(np.ceil(buffer_width / pixel_size))

    for dist in distances:
        # Get point along centerline
        center_point = line.interpolate(dist)

        # Get centerline direction at this point (using small offset for tangent)
        if dist + 1 < line.length:
            next_point = line.interpolate(dist + 1)
        else:
            next_point = line.interpolate(dist - 1)

        # Calculate perpendicular direction
        dx = next_point.x - center_point.x
        dy = next_point.y - center_point.y
        length = np.sqrt(dx ** 2 + dy ** 2)

        if length > 0:
            # Normalize and rotate 90 degrees for perpendicular
            perp_dx = -dy / length
            perp_dy = dx / length
        else:
            continue

        # Convert center point to pixel coordinates
        center_col, center_row = ~transform * (center_point.x, center_point.y)
        center_col, center_row = int(round(center_col)), int(round(center_row))

        # Sample pixels within buffer zone
        buffer_values = []

        # Create sampling grid perpendicular to centerline
        for offset in range(-buffer_pixels, buffer_pixels + 1):
            # Calculate position perpendicular to centerline
            offset_meters = offset * pixel_size

            # World coordinates of offset point
            offset_x = center_point.x + offset_meters * perp_dx
            offset_y = center_point.y + offset_meters * perp_dy

            # Convert to pixel coordinates
            col, row = ~transform * (offset_x, offset_y)
            col, row = int(round(col)), int(round(row))

            # Check if pixel is within raster bounds
            if 0 <= row < raster_data.shape[0] and 0 <= col < raster_data.shape[1]:
                value = raster_data[row, col]
                if np.isfinite(value):
                    buffer_values.append(value)

        # Compute statistic from buffer values
        if len(buffer_values) > 0:
            if stat == 'mean':
                aggregated_value = np.mean(buffer_values)
            elif stat == 'median':
                aggregated_value = np.median(buffer_values)
            elif stat == 'min':
                aggregated_value = np.min(buffer_values)
            elif stat == 'max':
                aggregated_value = np.max(buffer_values)
            elif stat == 'std':
                aggregated_value = np.std(buffer_values)
            else:
                aggregated_value = np.mean(buffer_values)

            if include_nan or np.isfinite(aggregated_value):
                sampled_values.append(aggregated_value)
                valid_distances.append(dist)

    return np.array(valid_distances), np.array(sampled_values)


def sample_raster_on_centerline_segment(raster_data, transform, full_centerline, clipped_segment,
                                        sampling_dist, buffer_width=100, include_nan=False, stat='mean',
                                        max_distance=None):
    """
    Sample raster values along the full centerline, but only keep the portion
    that lies inside the clipped segment. Returns distances referenced to the
    FULL (raw, un-shifted) centerline.
    """
    if clipped_segment is None or clipped_segment.is_empty:
        return np.array([]), np.array([])

    seg_start, seg_end = get_segment_distance_range_on_centerline(full_centerline, clipped_segment)
    if seg_start is None or seg_end is None or seg_end <= seg_start:
        return np.array([]), np.array([])

    distances, values = sample_raster_along_line_with_buffer(
        raster_data, transform, full_centerline,
        sampling_dist,
        buffer_width=buffer_width,
        include_nan=include_nan,
        max_distance=cfg.CENTERLINE_MAX_DISTANCE if max_distance is None else max_distance,
        stat=stat
    )

    if distances.size == 0:
        return distances, values

    keep = (distances >= seg_start) & (distances <= seg_end)
    return distances[keep], values[keep]


def get_centerline_plot_segment(line, max_distance=cfg.CENTERLINE_MAX_DISTANCE):
    """
    Return only the portion of a centerline from distance 0 (the start of the centerline)
    out to max_distance (or the full line, if it is shorter).
    """
    from shapely.ops import substring

    if line is None or line.is_empty:
        return line

    length = min(line.length, max_distance)
    return substring(line, 0, length)


