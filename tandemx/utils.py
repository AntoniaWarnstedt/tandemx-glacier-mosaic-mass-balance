"""General helpers: epoch labels, master grid, raster I/O, masks and the
reference-DEM products.
"""

import os
import re
import warnings
from datetime import datetime

import fiona
import geopandas as gpd
import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.features import geometry_mask, rasterize
from rasterio.transform import from_bounds
from rasterio.warp import reproject, transform_bounds
from scipy.ndimage import binary_dilation, generate_binary_structure

from . import config as cfg


def canonical_period_key(label: str) -> str:
    """Canonical key for internal lookups (always uses '-')."""
    if label is None:
        return ""
    return str(label).replace("/", "-").strip()


def parse_period(period_str):
    """Parse period string to extract start and end years."""
    parts = period_str.split('-')
    if len(parts) >= 2:
        return parts[0], parts[-1]
    return period_str, period_str


def get_display_year(year_str):
    """Convert year string to display format using / instead of -."""
    if '-' in year_str:
        parts = year_str.split('-')
        if len(parts) == 2:
            # Convert 2016-2017 to 2016/17, 2024-2025 to 2024/25
            return f"{parts[0]}/{parts[1][-2:]}"
    return year_str


def format_period_label(yA, yB):
    """Format period label for display (e.g., 2014-2016/17)."""
    yA_display = get_display_year(yA)
    yB_display = get_display_year(yB)
    return f"{yA_display}-{yB_display}"


def period_sort_key(label):
    """Chronological sort key for labels like: - '2014' - '2016-2017' - '2024/25'"""
    key = canonical_period_key(label)
    y0, y1 = parse_period(key)
    y0i = int(y0)
    y1i = int(y1)
    merged_flag = 1 if y0i != y1i else 0
    return (y0i, y1i, merged_flag, key)


def sort_period_labels(labels):
    """
    Sort period labels chronologically instead of lexically.
    """
    return sorted(labels, key=period_sort_key)


def change_sort_key(change):
    yA0, yA1 = parse_period(canonical_period_key(change["yA"]))
    yB0, yB1 = parse_period(canonical_period_key(change["yB"]))
    return (int(yA0), int(yA1), int(yB0), int(yB1))


def is_valid_period_comparison(yA, yB):
    """
    Check if a period comparison makes sense.
    Returns False if periods overlap (e.g., 2024 vs 2024-2025).
    """
    yA_start, yA_end = parse_period(yA)
    yB_start, yB_end = parse_period(yB)

    # Get all years in each period
    years_A = set(range(int(yA_start), int(yA_end) + 1))
    years_B = set(range(int(yB_start), int(yB_end) + 1))

    # Check for overlap
    if years_A & years_B:
        return False

    # Check chronological order (A should be before B)
    if int(yA_end) >= int(yB_start):
        return False

    return True


def should_skip_individual_year(year_label, epoch_labels):
    """True if year_label is part of a merged epoch that is among epoch_labels."""
    epoch_labels = set(epoch_labels)
    for years in cfg.MERGED_EPOCHS:
        if year_label in years and "-".join(years) in epoch_labels:
            return True
    return False


def analysis_epochs(labels):
    """Sorted epochs used for dh/dt: merged epochs replace their single years."""
    labels = list(labels)
    return [y for y in sort_period_labels(labels) if not should_skip_individual_year(y, labels)]


def summary_epochs(labels):
    """Three epochs of the summary figures (cfg.SUMMARY_EPOCHS or first/middle/last)."""
    if cfg.SUMMARY_EPOCHS:
        return list(cfg.SUMMARY_EPOCHS)
    eps = analysis_epochs(labels)
    if len(eps) < 3:
        return None
    return [eps[0], eps[len(eps) // 2], eps[-1]]


def mask_year_for_label(label):
    """Outline year (key of cfg.YEAR_MASKS) of an epoch or period label."""
    label = canonical_period_key(label)
    if label in cfg.MASK_ASSIGNMENT:
        return cfg.MASK_ASSIGNMENT[label]
    if label in cfg.YEAR_MASKS:
        return label
    available = sorted(cfg.YEAR_MASKS, key=int)
    if not available:
        return label
    end = int(parse_period(label)[1])
    return min(available, key=lambda y: (abs(int(y) - end), -int(y)))


def get_decimal_date(filename):
    """Extract the acquisition date from a file name as decimal year."""
    base = os.path.basename(filename)
    m = re.search(r'(20[0-9]{2})[-_](0[1-9]|1[0-2])[-_]([0-2][0-9]|3[01])', base)
    if m:
        dt = datetime(int(m.group(1)), int(m.group(2)), int(m.group(3)))
        year_start = datetime(dt.year, 1, 1)
        year_end = datetime(dt.year + 1, 1, 1)
        return float(dt.year + (dt - year_start).total_seconds() / (year_end - year_start).total_seconds())
    match = re.search(r'20[0-9]{6}', base)
    if match:
        date_str = match.group()
        dt = datetime.strptime(date_str, '%Y%m%d')
        year_start = datetime(dt.year, 1, 1)
        year_end = datetime(dt.year + 1, 1, 1)
        return float(dt.year + (dt - year_start).total_seconds() / (year_end - year_start).total_seconds())
    year_match = re.search(r'20[0-9]{2}', filename)
    if year_match:
        print(f"   ⚠️  No full date found in {base} -- using mid-year {year_match.group()}.5")
        return float(year_match.group()) + 0.5
    return 2020.0


def get_master_grid_info_union(mask_paths, resolution=30):
    """
    Like get_master_grid_info(), but sizes the master grid to the UNION of
    several mask shapefiles' extents instead of a single shapefile's bounds.
    """
    all_bounds = []
    crs = None
    for mp in mask_paths:
        with fiona.open(mp, 'r') as shp:
            all_bounds.append(shp.bounds)
            if crs is None:
                crs = rasterio.CRS.from_user_input(shp.crs)

    minx = min(b[0] for b in all_bounds)
    miny = min(b[1] for b in all_bounds)
    maxx = max(b[2] for b in all_bounds)
    maxy = max(b[3] for b in all_bounds)

    width = int(np.ceil((maxx - minx) / resolution))
    height = int(np.ceil((maxy - miny) / resolution))
    transform = from_bounds(minx, miny, maxx, maxy, width, height)

    print(f"   Union grid bounds from {len(mask_paths)} masks: "
          f"({minx:.0f}, {miny:.0f}) - ({maxx:.0f}, {maxy:.0f}), "
          f"{width} x {height} px")

    return {
        'bounds': (minx, miny, maxx, maxy),
        'crs': crs,
        'width': width,
        'height': height,
        'shape': (height, width),
        'transform': transform,
        'resolution': resolution,
    }


def create_coordinate_arrays(shape, transform):
    """Create X and Y coordinate arrays for a raster."""
    rows, cols = shape
    col_coords = np.arange(cols)
    row_coords = np.arange(rows)
    col_grid, row_grid = np.meshgrid(col_coords, row_coords)
    x = transform.c + (col_grid + 0.5) * transform.a
    y = transform.f + (row_grid + 0.5) * transform.e
    return x, y


def load_or_create_union_mask(mask_paths, grid_info, cache_path):
    """
    Like load_or_create_mask(), but KEEPS a pixel if it is valid (field == 1,
    or included if no valid field exists) in ANY of the given mask shapefiles,
    instead of requiring it to be valid in one single shapefile.
    """
    structure = generate_binary_structure(2, 2)  # 8-connected

    if os.path.exists(cache_path):
        print("   Loading cached union mask...")
        with rasterio.open(cache_path) as src:
            keep_mask = src.read(1)

        bad_mask = (keep_mask == 0)
        if np.any(bad_mask):
            bad_mask = binary_dilation(bad_mask, structure=structure, iterations=5)

        n_keep = np.sum(keep_mask == 1)
        n_exclude = np.sum(bad_mask)
        print(f"   ✓ Loaded union mask: {n_keep:,} to KEEP (=1), {n_exclude:,} excluded")
        return bad_mask

    print("   Creating union mask from all year shapefiles...")
    print(f"   Logic: KEEP pixels valid in ANY of {len(mask_paths)} year masks, EXCLUDE the rest")

    keep_mask = np.zeros(grid_info['shape'], dtype=np.uint8)

    for mp in mask_paths:
        with fiona.open(mp, 'r') as shp:
            schema_props = shp.schema['properties']
            valid_field = None
            for candidate in ['DN', 'Id', 'ID', 'id']:
                if candidate in schema_props:
                    valid_field = candidate
                    break

            if valid_field is not None:
                keep_geoms = [f['geometry'] for f in shp if f['properties'].get(valid_field) == 1]
            else:
                print(f"      ⚠️  {os.path.basename(mp)}: no valid field found (expected one of "
                      f"DN, Id, ID, id) -- including all its features")
                keep_geoms = [f['geometry'] for f in shp]

            if keep_geoms:
                year_keep = rasterize(
                    keep_geoms,
                    out_shape=grid_info['shape'],
                    transform=grid_info['transform'],
                    fill=0,
                    default_value=1,
                    dtype=np.uint8
                )
                keep_mask = np.maximum(keep_mask, year_keep)
                print(f"      {os.path.basename(mp)}: {len(keep_geoms)} polygons contributed")
            else:
                print(f"      ⚠️  {os.path.basename(mp)}: no usable polygons")

    bad_mask = (keep_mask == 0)
    if np.any(bad_mask):
        bad_mask = binary_dilation(bad_mask, structure=structure, iterations=5)

    n_keep = np.sum(keep_mask == 1)
    n_exclude = np.sum(bad_mask)
    total = keep_mask.size

    print(f"   ✓ Created union mask:")
    print(f"      - KEEP: {n_keep:,} ({n_keep/total*100:.2f}%) valid in >=1 year")
    print(f"      - EXCLUDE: {n_exclude:,} ({n_exclude/total*100:.2f}%) valid in no year")

    profile = {
        'driver': 'GTiff',
        'height': grid_info['shape'][0],
        'width': grid_info['shape'][1],
        'count': 1,
        'dtype': 'uint8',
        'crs': grid_info['crs'],
        'transform': grid_info['transform'],
        'compress': 'lzw',
        'tiled': True,
    }

    with rasterio.open(cache_path, 'w', **profile) as dst:
        dst.write(keep_mask, 1)

    print(f"   ✓ Saved union mask cache: {cache_path}")

    return bad_mask


def apply_mask_and_validate(data, bad_mask, step_name=""):
    """Apply mask and validate."""
    before_valid = np.sum(np.isfinite(data))
    data[bad_mask] = np.nan
    after_valid = np.sum(np.isfinite(data))
    masked_count = before_valid - after_valid

    if step_name and masked_count > 0:
        print(f"   [{step_name}] Masked {masked_count:,} pixels (kept: {after_valid:,})")

    return data


def rasterize_geometry_to_grid_mask(geom, grid_info, invert=True):
    """
    Rasterize a shapely geometry to the analysis grid.
    Returns boolean mask with True inside geometry.
    """
    if geom is None or geom.is_empty:
        return np.zeros(grid_info["shape"], dtype=bool)

    mask = geometry_mask(
        [geom],
        out_shape=grid_info["shape"],
        transform=grid_info["transform"],
        invert=invert
    )
    return mask.astype(bool)


def pixel_area_km2_from_grid(grid_info):
    """
    Single source of truth for 'area of one pixel, in km^2', given the grid's
    (square) resolution in metres.
    """
    return (float(grid_info["resolution"]) ** 2) * cfg.M2_TO_KM2


def load_rock_mask(shp_path, grid_info):
    """
    Loads rock outcrop shapefile and returns a boolean mask (True = Rock).
    Assumes the presence of a polygon means it is rock.
    """
    if not os.path.exists(shp_path):
        print(f"      ⚠️ Rock mask not found at {shp_path}. Using 10km buffer blindly!")
        return np.ones(grid_info['shape'], dtype=bool)

    with fiona.open(shp_path, 'r') as shp:
        geoms = [feature['geometry'] for feature in shp if feature['geometry'] is not None]

        if not geoms:
            return np.zeros(grid_info['shape'], dtype=bool)

        rock_raster = rasterize(
            geoms,
            out_shape=grid_info['shape'],
            transform=grid_info['transform'],
            fill=0,
            default_value=1,
            dtype=np.uint8
        )

    return rock_raster == 1


def resample_to_grid(src_path, grid_info, is_categorical=False):
    """Resample any raster to the master grid."""
    if src_path is None or not os.path.exists(src_path):
        return None

    resamp_method = Resampling.nearest if is_categorical else Resampling.bilinear

    with rasterio.open(src_path) as src:

        grid_crs = rasterio.CRS.from_user_input(grid_info['crs'])

        try:
            src_left, src_bottom, src_right, src_top = transform_bounds(
                src.crs, grid_crs,
                src.bounds.left, src.bounds.bottom,
                src.bounds.right, src.bounds.top,
            )
        except Exception as e:
            print(f"   [resample_to_grid] CRS transform failed for "
                  f"{os.path.basename(src_path)}: {e}")
            return None

        grid_bounds = grid_info['bounds']
        if (src_left > grid_bounds[2] or src_right < grid_bounds[0] or
                src_bottom > grid_bounds[3] or src_top < grid_bounds[1]):
            return None

        out_data = np.full(grid_info['shape'], np.nan, dtype=np.float32)

        reproject(
            source=rasterio.band(src, 1),
            destination=out_data,
            src_transform=src.transform,
            src_crs=src.crs,
            dst_transform=grid_info['transform'],
            dst_crs=grid_crs,
            resampling=resamp_method,
            src_nodata=src.nodata,
            dst_nodata=np.nan,
        )

    return out_data


def save_raster(data, path, grid_info, units=None, description=None):
    """Save a 2D numpy array as a GeoTIFF."""
    profile = {
        'driver': 'GTiff',
        'height': grid_info['shape'][0],
        'width': grid_info['shape'][1],
        'count': 1,
        'dtype': 'float32',
        'crs': grid_info['crs'],
        'transform': grid_info['transform'],
        'compress': 'lzw',
        'tiled': True,
        'nodata': -9999.0,
    }

    data_out = np.copy(data)
    data_out[~np.isfinite(data_out)] = -9999.0

    with rasterio.open(path, 'w', **profile) as dst:
        dst.write(data_out.astype(np.float32), 1)
        if units:
            dst.update_tags(units=units)
            dst.update_tags(1, units=units)
        if description:
            dst.update_tags(description=description)
            dst.set_band_description(1, description)


def fill_nodata_nearest(data, valid_extent_mask=None):
    """
    Fill NaN gaps in a CATEGORICAL raster (e.g. the elevation-bin raster from
    BIN_ELEV_PATH, or the slope-bin raster from BIN_SLOPE_PATH) by propagating
    the value of the spatially NEAREST valid (finite) pixel.
    """
    from scipy.ndimage import distance_transform_edt

    arr = np.asarray(data, dtype=float).copy()
    valid = np.isfinite(arr)

    if valid_extent_mask is not None:
        target = np.asarray(valid_extent_mask, dtype=bool) & ~valid
    else:
        target = ~valid

    if not np.any(target) or not np.any(valid):
        return arr

    _, (iy, ix) = distance_transform_edt(~valid, return_indices=True)
    nearest_vals = arr[iy, ix]
    arr[target] = nearest_vals[target]

    n_filled = int(np.sum(target))
    if n_filled > 0:
        print(f"      fill_nodata_nearest: filled {n_filled} NaN pixel(s) "
              f"via nearest-valid-bin lookup "
              f"({n_filled * cfg.RESOLUTION**2 * cfg.M2_TO_KM2:.2f} km²)")

    return arr


def prefilter_dem(data, min_elev=cfg.ELEV_MIN, max_elev=cfg.ELEV_MAX):
    """Filter extreme elevation values."""
    data = np.copy(data)
    data[(data < min_elev) | (data > max_elev)] = np.nan
    return data


def nan_median_filter(arr, size=3, chunk_rows=512):
    """
    NaN-aware median filter: each output pixel is the median of the FINITE
    values in its `size`x`size` window (NaNs are ignored, not treated as 0).
    Pixels that are NaN in the input stay NaN in the output.
    """
    from numpy.lib.stride_tricks import sliding_window_view
    a = np.asarray(arr, dtype=np.float64)
    r = size // 2
    padded = np.pad(a, r, mode="constant", constant_values=np.nan)
    out = np.full(a.shape, np.nan, dtype=np.float64)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)  # all-NaN windows
        for r0 in range(0, a.shape[0], chunk_rows):
            r1 = min(r0 + chunk_rows, a.shape[0])
            win = sliding_window_view(padded[r0:r1 + 2 * r, :], (size, size))
            out[r0:r1] = np.nanmedian(win.reshape(win.shape[0], win.shape[1], -1), axis=-1)
    out[~np.isfinite(a)] = np.nan
    return out


def reject_abs_rate(arr, max_abs=None):
    """Set |arr| > max_abs (default cfg.DHDT_MAX_ABS_RATE) to NaN (reject, not clip)."""
    max_abs = cfg.DHDT_MAX_ABS_RATE if max_abs is None else max_abs
    a = np.array(arr, dtype=np.float64, copy=True)
    with np.errstate(invalid='ignore'):
        a[np.abs(a) > max_abs] = np.nan
    return a


def remove_seam_ring(arr, ring_px=1, void_mask=None):
    """
    Set to NaN every finite pixel within `ring_px` pixels (8-connected) of a
    void, i.e. drop the outermost ring of each data patch. These seam pixels
    border data voids / scene edges, where localized phase-unwrapping errors
    are most likely. ring_px=0 disables.

    void_mask: optional boolean array of the voids to grow the ring from
    (default: the NaNs of `arr`), so that pixels rejected for an implausible
    rate do not also remove their neighbours.
    """
    from scipy.ndimage import binary_dilation
    a = np.array(arr, dtype=np.float64, copy=True)
    if ring_px <= 0:
        return a
    invalid = ~np.isfinite(a)
    voids = invalid if void_mask is None else np.asarray(void_mask, dtype=bool)
    if not np.any(voids):
        return a
    near_void = binary_dilation(voids, structure=np.ones((3, 3), dtype=bool),
                                iterations=ring_px)
    a[near_void & ~invalid] = np.nan
    return a


def horn_slope_deg(dem, res_x, res_y):
    """Slope in degrees (Horn 1981, as in GDAL/QGIS); NaN where any neighbour is NaN."""
    z = np.pad(dem.astype(np.float64), 1, mode="edge")
    a, b, c = z[:-2, :-2], z[:-2, 1:-1], z[:-2, 2:]
    d, f = z[1:-1, :-2], z[1:-1, 2:]
    g, h, i = z[2:, :-2], z[2:, 1:-1], z[2:, 2:]
    dzdx = ((c + 2 * f + i) - (a + 2 * d + g)) / (8.0 * abs(res_x))
    dzdy = ((g + 2 * h + i) - (a + 2 * b + c)) / (8.0 * abs(res_y))
    return np.degrees(np.arctan(np.hypot(dzdx, dzdy))).astype(np.float32)


def prepare_reference_dem_products(force=False):
    """
    Build the reference-DEM products used by the processing (REF_DEM_PATH,
    BIN_ELEV_PATH, SLOPE_PATH, BIN_SLOPE_PATH) from REF_DEM_SOURCE: windowed
    read of the AOI (REF_AOI_SHP buffered by REF_AOI_BUFFER_M), reprojection to
    cfg.CRS / RESOLUTION if needed, pixels outside the buffered AOI set to
    nodata, Horn slope, lower-edge binning (elevations below 0 m go to the 0 m
    bin, slopes are capped at the 85-90 deg bin).
    """
    import rasterio.windows
    import rasterio.transform
    from rasterio.windows import from_bounds as _win_from_bounds
    from rasterio.features import geometry_mask as _gmask
    outs = [cfg.REF_DEM_PATH, cfg.BIN_ELEV_PATH, cfg.SLOPE_PATH, cfg.BIN_SLOPE_PATH]
    if not force and all(os.path.exists(o) for o in outs):
        print(f"   Reference-DEM products found in {cfg.REF_PRODUCTS_DIR} (reused)")
        return
    if not os.path.exists(cfg.REF_DEM_SOURCE):
        raise FileNotFoundError(f"Reference DEM not found: {cfg.REF_DEM_SOURCE}")
    os.makedirs(cfg.REF_PRODUCTS_DIR, exist_ok=True)
    print(f"   Building reference-DEM products from {os.path.basename(cfg.REF_DEM_SOURCE)} ...")

    aoi = gpd.read_file(cfg.REF_AOI_SHP).to_crs(cfg.CRS)
    aoi_buf = aoi.geometry.union_all() if hasattr(aoi.geometry, "union_all") else aoi.unary_union
    aoi_buf = aoi_buf.buffer(cfg.REF_AOI_BUFFER_M)
    res = float(cfg.RESOLUTION)
    minx, miny, maxx, maxy = aoi_buf.bounds
    minx, miny = np.floor(minx / res) * res, np.floor(miny / res) * res
    maxx, maxy = np.ceil(maxx / res) * res, np.ceil(maxy / res) * res
    width, height = int(round((maxx - minx) / res)), int(round((maxy - miny) / res))
    dst_transform = rasterio.transform.from_origin(minx, maxy, res, res)
    dst = np.full((height, width), np.nan, dtype=np.float32)

    with rasterio.open(cfg.REF_DEM_SOURCE) as src:
        if src.crs is None:
            raise ValueError("Reference DEM has no CRS")
        b = transform_bounds(cfg.CRS, src.crs, minx, miny, maxx, maxy, densify_pts=21)
        win = _win_from_bounds(*b, transform=src.transform).round_offsets().round_lengths()
        win = win.intersection(rasterio.windows.Window(0, 0, src.width, src.height))
        data = src.read(1, window=win, masked=True).astype(np.float32).filled(np.nan)
        reproject(source=data, destination=dst,
                  src_transform=src.window_transform(win), src_crs=src.crs,
                  dst_transform=dst_transform, dst_crs=cfg.CRS,
                  resampling=Resampling.bilinear, src_nodata=np.nan, dst_nodata=np.nan)
        src_info = f"{src.crs.to_string()} {src.res[0]:.6g}x{src.res[1]:.6g}"

    slope = horn_slope_deg(dst, res, res)
    outside = _gmask([aoi_buf], out_shape=dst.shape, transform=dst_transform, invert=False)
    dst[outside] = np.nan
    slope[outside | ~np.isfinite(dst)] = np.nan
    ebin = np.where(np.isfinite(dst),
                    np.floor(np.clip(dst, 0.0, None) / cfg.REF_ELEV_BIN_M) * cfg.REF_ELEV_BIN_M, np.nan)
    sbin = np.where(np.isfinite(slope),
                    np.floor(np.clip(slope, 0.0, 89.999) / cfg.REF_SLOPE_BIN_DEG) * cfg.REF_SLOPE_BIN_DEG, np.nan)

    prof = dict(driver="GTiff", height=height, width=width, count=1, dtype="float32",
                crs=cfg.CRS, transform=dst_transform, nodata=-9999.0,
                compress="deflate", predictor=3, tiled=True, blockxsize=256, blockysize=256)
    for arr, path, desc in [
        (dst, cfg.REF_DEM_PATH, f"DInSAR reference DEM ({src_info}), AOI + {cfg.REF_AOI_BUFFER_M/1000:.0f} km, {cfg.CRS} {res:g} m"),
        (ebin, cfg.BIN_ELEV_PATH, f"Reference-DEM elevation bins, lower edge, {cfg.REF_ELEV_BIN_M:g} m"),
        (slope, cfg.SLOPE_PATH, "Reference-DEM slope (deg), Horn 1981"),
        (sbin, cfg.BIN_SLOPE_PATH, f"Reference-DEM slope bins, lower edge, {cfg.REF_SLOPE_BIN_DEG:g} deg"),
    ]:
        with rasterio.open(path, "w", **prof) as o:
            o.write(np.where(np.isfinite(arr), arr, -9999.0).astype(np.float32), 1)
            o.set_band_description(1, desc)
        print(f"   ✓ {os.path.basename(path)}")
    n = int(np.isfinite(dst).sum())
    print(f"   Reference DEM: {width}x{height} px, {n:,} valid, "
          f"elevation {np.nanmin(dst):.1f}..{np.nanmax(dst):.1f} m, slope max {np.nanmax(slope):.1f} deg")


def get_longest_linestring(geom):
    """
    Return the longest LineString from a shapely geometry.
    Handles LineString, MultiLineString, and GeometryCollection.
    Returns None if no LineString exists.
    """
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
                lines.extend([subg for subg in g.geoms if subg.geom_type == "LineString" and not subg.is_empty])
        return max(lines, key=lambda g: g.length) if lines else None

    return None


def setup_grid():
    """Master grid and study-area mask from the union of all yearly outlines."""
    mask_paths = list(cfg.YEAR_MASKS.values())
    missing = [p for p in mask_paths if not os.path.exists(p)]
    if missing:
        raise FileNotFoundError(f"Outline(s) not found: {missing}")
    grid_info = get_master_grid_info_union(mask_paths, cfg.RESOLUTION)
    bad_mask = load_or_create_union_mask(mask_paths, grid_info, cfg.MASK_RASTER_CACHE_UNION)
    return grid_info, bad_mask


def load_reference_products(grid_info, bad_mask):
    """
    Reference DEM, elevation bins, slope and slope bins on the master grid.
    NaN gaps of the bin rasters inside the study area are filled with the
    nearest valid bin so that every glacier pixel belongs to a bin.
    """
    if cfg.REF_PRODUCTS_BUILD:
        cfg.REF_PRODUCTS_DIR = os.path.join(cfg.OUT_DIR, "reference_dem")
        cfg.REF_DEM_PATH, cfg.BIN_ELEV_PATH, cfg.SLOPE_PATH, cfg.BIN_SLOPE_PATH = [
            os.path.join(cfg.REF_PRODUCTS_DIR, n) for n in
            ("ref_dem_aoi.tif", "ref_elev_binned_50m.tif", "ref_slope.tif", "ref_slope_binned_5deg.tif")]
        prepare_reference_dem_products(force=cfg.REF_PRODUCTS_REBUILD)

    elev_bins = resample_to_grid(cfg.BIN_ELEV_PATH, grid_info, is_categorical=True)
    slope = resample_to_grid(cfg.SLOPE_PATH, grid_info, is_categorical=False)
    slope_bins = resample_to_grid(cfg.BIN_SLOPE_PATH, grid_info, is_categorical=True)
    ref_dem = resample_to_grid(cfg.REF_DEM_PATH, grid_info)
    for arr, path in [(elev_bins, cfg.BIN_ELEV_PATH), (slope, cfg.SLOPE_PATH),
                      (slope_bins, cfg.BIN_SLOPE_PATH), (ref_dem, cfg.REF_DEM_PATH)]:
        if arr is None:
            raise FileNotFoundError(f"Could not load {path}")

    inside = ~bad_mask
    if np.any(inside & ~np.isfinite(elev_bins)):
        elev_bins = fill_nodata_nearest(elev_bins, valid_extent_mask=inside)
    if np.any(inside & ~np.isfinite(slope_bins)):
        slope_bins = fill_nodata_nearest(slope_bins, valid_extent_mask=inside)
    return elev_bins, slope, slope_bins, ref_dem
