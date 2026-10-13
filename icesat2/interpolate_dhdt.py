"""
Interpolate the filtered repeat-track dh/dt points (p1) and their fit
uncertainty (p1_err) to a regular grid: inverse-distance weighting of the
K nearest points, followed by a normalised Gaussian smoothing, clipped to the
outline polygon. Settings: icesat2/config.py.
"""
import os

import geopandas as gpd
import numpy as np
import rasterio
from rasterio.mask import mask
from rasterio.transform import from_bounds
from scipy.ndimage import gaussian_filter
from scipy.spatial import cKDTree

import config as cfg


def idw_weights(coords, grid_points):
    """Inverse-distance weights of the K nearest points for every grid cell."""
    k = min(cfg.K_IDW, len(coords))
    dist, idx = cKDTree(coords).query(grid_points, k=k)
    if k == 1:
        dist, idx = dist[:, None], idx[:, None]
    dist = np.maximum(dist, 1e-6)
    w = 1.0 / dist ** cfg.IDW_POWER
    w[dist > cfg.MAX_DISTANCE] = 0.0
    return w, idx


def idw_and_smooth(values, w, idx, shape):
    """IDW average followed by a Gaussian filter normalised by the data coverage."""
    sw = w.sum(axis=1)
    valid = sw > 0
    grid = np.full(len(sw), np.nan)
    grid[valid] = (values[idx] * w)[valid].sum(axis=1) / sw[valid]
    grid = grid.reshape(shape)
    valid = valid.reshape(shape).astype(float)

    coverage = gaussian_filter(valid, sigma=cfg.SIGMA, mode="constant", cval=0)
    smoothed = gaussian_filter(np.nan_to_num(grid, nan=0), sigma=cfg.SIGMA, mode="constant", cval=0)
    smoothed = np.divide(smoothed, coverage, where=coverage > 0.1, out=np.full_like(smoothed, np.nan))
    smoothed[valid == 0] = np.nan
    return np.flipud(smoothed)   # rows from north to south


def write_masked_raster(arr, bounds, crs, mask_geoms, out_path):
    """Write a grid as GeoTIFF clipped to the outline polygons."""
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    transform = from_bounds(*bounds, arr.shape[1], arr.shape[0])
    tmp = out_path.replace(".tif", "_temp.tif")
    with rasterio.open(tmp, "w", driver="GTiff", height=arr.shape[0], width=arr.shape[1], count=1,
                       dtype="float32", crs=crs, transform=transform, nodata=np.nan,
                       compress="lzw") as dst:
        dst.write(arr.astype("float32"), 1)
    with rasterio.open(tmp) as src:
        out, out_transform = mask(src, mask_geoms, crop=True, nodata=np.nan)
        meta = src.meta.copy()
    meta.update(height=out.shape[1], width=out.shape[2], transform=out_transform, nodata=np.nan)
    with rasterio.open(out_path, "w", **meta) as dst:
        dst.write(out)
    os.remove(tmp)
    print(f"Wrote {out_path}  (mean {np.nanmean(out[0]):.4f} m a-1)")


def main():
    points = gpd.read_file(cfg.INTERP_INPUT)
    outline = gpd.read_file(cfg.INTERP_MASK).to_crs(points.crs)

    # only points where both fields are valid, so both grids share the weights
    ok = points["p1"].notna() & points["p1_err"].notna()
    points = points[ok]
    coords = np.column_stack([points.geometry.x, points.geometry.y])
    print(f"{len(points)} points")

    minx, miny, maxx, maxy = outline.total_bounds
    gx, gy = np.meshgrid(np.arange(minx, maxx, cfg.RESOLUTION), np.arange(miny, maxy, cfg.RESOLUTION))
    w, idx = idw_weights(coords, np.c_[gx.ravel(), gy.ravel()])

    dhdt = idw_and_smooth(points["p1"].values, w, idx, gx.shape)
    err = idw_and_smooth(points["p1_err"].values, w, idx, gx.shape)
    if cfg.ERR_CLIP_PCTL is not None:
        err = np.clip(err, *np.nanpercentile(err, cfg.ERR_CLIP_PCTL))

    geoms = list(outline.geometry)
    bounds = (minx, miny, maxx, maxy)
    write_masked_raster(dhdt, bounds, points.crs, geoms, cfg.INTERP_OUTPUT)
    write_masked_raster(err, bounds, points.crs, geoms, cfg.INTERP_OUTPUT_ERR)


if __name__ == "__main__":
    main()
