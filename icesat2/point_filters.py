"""Point filters shared by the repeat-track and crossover cleaning scripts."""
import os

import geopandas as gpd
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree


def removed_path(output_path):
    """File holding the points removed by the filters (used by plot_filter_qc.py)."""
    return output_path.rsplit(".", 1)[0] + "_removed.gpkg"


def range_filter(gdf, col, lo, hi, label):
    """Keep rows with lo <= gdf[col] <= hi."""
    keep = (gdf[col] >= lo) & (gdf[col] <= hi)
    print(f"{label}: removed {int((~keep).sum())} points")
    return gdf[keep], gdf[~keep]


def local_mad_filter(gdf, col, k=30, n_sigma=3.0, mad_floor=0.05):
    """
    Spatial outlier filter: compare each point with its k nearest neighbours
    and keep it if it lies within median +/- n_sigma * 1.4826 * MAD of that
    neighbourhood (MAD bounded below by mad_floor). Unlike one global
    threshold, spatially coherent signals such as a fast-thinning trunk survive.
    """
    gdf = gdf.reset_index(drop=True)
    coords = np.column_stack([gdf.geometry.x.values, gdf.geometry.y.values])
    vals = gdf[col].values
    _, idx = cKDTree(coords).query(coords, k=min(k + 1, len(gdf)))  # +1: the point itself
    neigh = vals[idx]
    med = np.median(neigh, axis=1)
    mad = np.maximum(1.4826 * np.median(np.abs(neigh - med[:, None]), axis=1), mad_floor)
    keep = np.abs(vals - med) <= n_sigma * mad
    removed = gdf[~keep]
    print(f"Local outlier filter (k={k}, {n_sigma} x local MAD): removed {len(removed)} points")
    gm = np.median(vals)
    print(f"  removed below / above global median ({gm:.3f}): "
          f"{int((removed[col] < gm).sum())} / {int((removed[col] > gm).sum())}")
    return gdf[keep].copy(), removed.copy()


def save_results(clean, removed_parts, output_path):
    """Write the cleaned points and, for QC, the removed points with the filter name."""
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    clean.to_file(output_path)
    parts = [p.assign(removed_by=name) for name, p in removed_parts.items() if len(p)]
    if parts:
        rem = gpd.GeoDataFrame(pd.concat(parts, ignore_index=True), crs=clean.crs)
        rem.to_file(removed_path(output_path), driver="GPKG")
    print(f"Saved {len(clean)} points to {output_path}")
