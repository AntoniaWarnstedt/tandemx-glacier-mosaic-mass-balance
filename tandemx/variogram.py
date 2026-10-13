"""Empirical variogram (Dowd estimator), Gaussian + spherical model fit and
effective number of independent samples N_eff (Hugonnet et al. 2022).
"""

import numpy as np
import pandas as pd
from scipy.integrate import quad
from scipy.optimize import curve_fit
from scipy.spatial.distance import cdist


def _vgm_gaussian(h: np.ndarray, range_val: float, psill: float) -> np.ndarray:
    """Gaussian variogram model: γ(h) = psill * (1 - exp(-(h / range)²))"""
    a = max(float(range_val), 1e-10)
    return psill * (1.0 - np.exp(-(np.asarray(h, dtype=float) / a) ** 2))


def _vgm_spherical(h: np.ndarray, range_val: float, psill: float) -> np.ndarray:
    """
    Spherical variogram model: γ(h) = psill*(1.5*(h/r) - 0.5*(h/r)³) for h ≤ r,
    else psill
    """
    h = np.asarray(h, dtype=float)
    r = max(float(range_val), 1e-10)
    ratio = h / r
    return np.where(
        h <= r,
        float(psill) * (1.5 * ratio - 0.5 * ratio ** 3),
        float(psill)
    ).astype(float)


def _vgm_gaussian_spherical_sum(
        h: np.ndarray,
        range1: float, psill1: float,
        range2: float, psill2: float
) -> np.ndarray:
    """Sum of Gaussian (short-range) + Spherical (long-range) models.

    Hugonnet et al. (2022) Eq. 2 and Appendix B: two-range model captures
    both the noise-floor and long-wavelength DEM processing artefacts.
    """
    return _vgm_gaussian(h, range1, psill1) + _vgm_spherical(h, range2, psill2)


def _covariance_from_params(h: np.ndarray, params_df: pd.DataFrame) -> np.ndarray:
    """
    Covariance from variogram parameters.

    C(h) = total_sill - γ(h)
    Normalized correlation: ρ(h) = C(h) / total_sill ∈ [0, 1]
    """
    total_sill = float(np.sum(params_df["psill"].values))
    gamma_h = np.zeros_like(np.asarray(h, dtype=float))

    for _, row in params_df.iterrows():
        model = str(row["model"]).lower()
        r, p = float(row["range"]), float(row["psill"])

        if model == "gaussian":
            gamma_h += _vgm_gaussian(h, r, p)
        elif model == "spherical":
            gamma_h += _vgm_spherical(h, r, p)
        elif model == "fullcorr":
            # fully correlated over the glacier: no decorrelation within the
            # domain (gamma contribution 0, covariance = psill everywhere)
            pass

    return total_sill - gamma_h


def _make_corr_func(params_df: pd.DataFrame):
    """
    Return normalized correlation function ρ(h) ∈ [0,1] as a Callable.

    Used for N_eff covariance integration and for polygon-based calculations.
    """
    params_copy = params_df.copy()
    total_sill = float(np.sum(params_copy["psill"].values))

    def corr_func(h_input):
        h = np.asarray(h_input, dtype=float)
        if total_sill <= 0:
            return np.zeros_like(h)
        return np.clip(_covariance_from_params(h, params_copy) / total_sill, 0.0, 1.0)

    return corr_func


def _fallback_variogram_params() -> pd.DataFrame:
    """
    Safe single-spherical fallback when variogram estimation fails.

    1000 m range, unit sill. Used when < 500 stable points available.
    """
    return pd.DataFrame({
        "model": ["spherical"],
        "range": [1000.0],
        "psill": [1.0]
    })


def dowd_semivariance(dz: np.ndarray) -> float:
    """Dowd's robust variogram estimator for one lag bin."""
    valid = dz[np.isfinite(dz)]
    if valid.size < 2:
        return np.nan
    return 0.5 * (1.4826 * float(np.median(np.abs(valid)))) ** 2


def sample_empirical_variogram_multiscale(
        dhdt_off_ice: np.ndarray,
        x_coords: np.ndarray,
        y_coords: np.ndarray,
        sample_size: int = 10_000,
        n_pairs: int = 1_000_000,
        max_lag: float = 20_000.0,
        n_lag_bins: int = 35,
        min_pairs_per_bin: int = 50,
        random_seed: int = 42,
) -> pd.DataFrame:
    """Build empirical variogram using Dowd's estimator and exponential lag bins."""
    rng = np.random.default_rng(random_seed)

    # Mask: finite, not extreme outlier
    valid_mask = np.isfinite(dhdt_off_ice) & (np.abs(dhdt_off_ice) <= 50.0)
    n_valid = int(np.sum(valid_mask))

    if n_valid < 500:
        return pd.DataFrame(columns=["lags", "exp", "count"])

    # Extract coordinates and values
    x_v = x_coords[valid_mask]
    y_v = y_coords[valid_mask]
    z_v = dhdt_off_ice[valid_mask]

    # Subsample if needed
    if n_valid > sample_size:
        idx = rng.choice(n_valid, size=sample_size, replace=False)
        x_v, y_v, z_v = x_v[idx], y_v[idx], z_v[idx]

    n_pts = len(z_v)

    # Exponential lag bins: start at ~42 m (30√2) to skip self-correlation
    bin_edges = np.geomspace(30.0 * np.sqrt(2.0), max_lag, n_lag_bins + 1)

    # --- Random pair construction with deduplication ---
    n_draw = min(int(n_pairs * 1.5), n_pts * (n_pts - 1) // 2)
    i_raw = rng.integers(0, n_pts, size=n_draw)
    j_raw = rng.integers(0, n_pts, size=n_draw)

    # Remove self-pairs
    diff_m = i_raw != j_raw
    i_raw, j_raw = i_raw[diff_m], j_raw[diff_m]

    # Deduplicate (enforce i < j)
    lo = np.minimum(i_raw, j_raw)
    hi = np.maximum(i_raw, j_raw)
    pairs = np.unique(np.column_stack((lo, hi)), axis=0)

    # Randomly select up to n_pairs
    if len(pairs) > n_pairs:
        sel = rng.choice(len(pairs), size=n_pairs, replace=False)
        pairs = pairs[sel]

    # Compute distances and differences
    ii, jj = pairs[:, 0], pairs[:, 1]
    dx = x_v[ii] - x_v[jj]
    dy = y_v[ii] - y_v[jj]
    dist = np.sqrt(dx * dx + dy * dy)
    dz = z_v[ii] - z_v[jj]

    # Filter: valid distance and difference, within max_lag
    ok = np.isfinite(dist) & np.isfinite(dz) & (dist > 0.0) & (dist <= max_lag)
    dist, dz = dist[ok], dz[ok]

    # --- Compute Dowd semivariance per bin ---
    bin_ids = np.digitize(dist, bin_edges) - 1
    rows = []

    for k in range(n_lag_bins):
        m = bin_ids == k
        count = int(np.sum(m))

        # Only compute if bin has enough pairs
        gamma = dowd_semivariance(dz[m]) if count >= min_pairs_per_bin else np.nan

        rows.append({
            "lags": float(bin_edges[k + 1]),
            "exp": gamma,
            "count": count
        })

    return pd.DataFrame(rows)


def fit_sum_variogram_hugonnet(
        empirical_variogram: pd.DataFrame,
        max_lag: float = 20_000.0,
        n_attempts: int = 8,
        random_seed: int = 42,
) -> tuple:
    """Fit Gaussian (short) + Spherical (long) sum variogram."""
    rng = np.random.default_rng(random_seed)

    fallback_p = _fallback_variogram_params()
    fallback_c = _make_corr_func(fallback_p)

    # Filter to valid (non-NaN) semivariance
    valid = empirical_variogram[
        np.isfinite(empirical_variogram["exp"])
    ].copy()

    if len(valid) < 5:
        return fallback_p, fallback_c, False

    h_fit = valid["lags"].values.astype(float)
    gamma_fit = valid["exp"].values.astype(float)
    counts = valid["count"].values.astype(float)

    # Cressie (1985) variogram weights: sqrt(N(h)/2)
    weights = np.sqrt(np.maximum(counts / 2.0, 1.0))

    # Estimate total sill from upper-half median
    tail = h_fit > np.nanpercentile(h_fit, 50.0)
    sill_est = np.nanmedian(gamma_fit[tail]) if np.any(tail) else np.nanmax(gamma_fit)

    if not np.isfinite(sill_est) or sill_est <= 0:
        sill_est = float(np.nanmax(gamma_fit)) if np.any(np.isfinite(gamma_fit)) else 1.0
    if not np.isfinite(sill_est) or sill_est <= 0:
        sill_est = 1.0

    best_params = None
    best_cost = np.inf

    # Multiple random starts
    for attempt in range(n_attempts):
        # Random initialization
        f1 = rng.uniform(0.15, 0.55)  # fraction of sill for Gaussian
        r1_0 = rng.uniform(max_lag * 0.005, max_lag * 0.12)  # Gaussian range
        r2_0 = rng.uniform(max_lag * 0.12, max_lag * 0.85)  # Spherical range
        p0 = [r1_0, sill_est * f1, r2_0, sill_est * (1.0 - f1)]

        # Bounds
        lo = [1.0, 1e-8, max_lag * 0.005, 1e-8]
        hi = [max_lag * 0.45, sill_est * 4.0, max_lag, sill_est * 4.0]

        try:
            popt, _ = curve_fit(
                _vgm_gaussian_spherical_sum,
                h_fit, gamma_fit,
                p0=p0,
                bounds=(lo, hi),
                sigma=1.0 / weights,
                absolute_sigma=False,
                maxfev=25_000,
                method="trf",
            )

            pred = _vgm_gaussian_spherical_sum(h_fit, *popt)
            cost = float(np.sum(((gamma_fit - pred) * weights) ** 2))

            # Keep best fit
            if cost < best_cost and np.all(np.array(popt) > 0):
                best_cost, best_params = cost, popt

        except Exception:
            continue

    if best_params is None:
        return fallback_p, fallback_c, False

    r1, p1, r2, p2 = best_params

    # Enforce: Gaussian (short) ≤ Spherical (long)
    if r1 > r2:
        r1, r2, p1, p2 = r2, r1, p2, p1

    params_df = pd.DataFrame({
        "model": ["gaussian", "spherical"],
        "range": [float(r1), float(r2)],
        "psill": [float(p1), float(p2)],
    })

    return params_df, _make_corr_func(params_df), True


def neff_circular_numerical(area_m2: float, params_df: pd.DataFrame) -> float:
    """N_eff by numerical radial integration of covariance over circular area."""
    if not np.isfinite(area_m2) or area_m2 <= 0:
        return 1.0

    total_sill = float(np.sum(params_df["psill"].values))
    if total_sill <= 0:
        return 1.0

    L = np.sqrt(area_m2 / np.pi)

    def h_cov(h_val):
        """h × C(h) integrand"""
        return h_val * float(_covariance_from_params(
            np.array([h_val]), params_df
        )[0])

    try:
        # Numerical integration from 0 to L
        integral, _ = quad(h_cov, 0.0, L, limit=300, epsabs=1e-6, epsrel=1e-4)
        se_sq = 2.0 * integral / (L * L)

        if not np.isfinite(se_sq) or se_sq <= 0:
            return 1.0

        return max(1.0, float(total_sill / se_sq))

    except Exception:
        return 1.0


def neff_hugonnet_polygon(
        union_geom,
        grid_info: dict,
        corr_func,
        subsample: int = 2000,
        max_pixels: int = 5000,
        random_seed: int = 42,
) -> float:
    """N_eff for actual glacier polygon shape via Hugonnet et al. (2022) Eq. 18."""
    from rasterio.features import rasterize

    rng = np.random.default_rng(random_seed)

    if union_geom is None or union_geom.is_empty:
        return 1.0

    try:
        # Rasterize polygon to grid
        mask = np.zeros(grid_info["shape"], dtype=np.uint8)
        if hasattr(union_geom, 'geom_type'):
            if union_geom.geom_type == 'Polygon':
                geom_list = [union_geom]
            else:
                geom_list = list(union_geom.geoms) if hasattr(union_geom, 'geoms') else [union_geom]
        else:
            geom_list = [union_geom]

        mask = rasterize(
            geom_list,
            out_shape=grid_info["shape"],
            transform=grid_info["transform"],
            fill=0,
            default_value=1,
            dtype=np.uint8
        )
    except Exception:
        return 1.0

    # Get all glacier pixels
    rows_all, cols_all = np.where(mask == 1)
    n_total = int(len(rows_all))

    if n_total < 10:
        return 1.0

    # Convert pixel indices to coordinates
    tr = grid_info["transform"]
    x_a = tr.c + (cols_all + 0.5) * tr.a
    y_a = tr.f + (rows_all + 0.5) * tr.e
    coords_all = np.column_stack([x_a, y_a])

    # Cap large glaciers to manage memory
    if n_total > max_pixels:
        idx_all = rng.choice(n_total, size=max_pixels, replace=False)
        coords_all = coords_all[idx_all]
        n_total = max_pixels

    # Subsample one dimension
    m_sub = min(subsample, n_total)
    sub_idx = rng.choice(n_total, size=m_sub, replace=False)
    sub_coords = coords_all[sub_idx]

    # Distance matrix: all vs. subset
    dist_matrix = cdist(coords_all, sub_coords, metric="euclidean")
    rho_sum = float(np.sum(corr_func(dist_matrix.ravel())))

    if not np.isfinite(rho_sum) or rho_sum <= 0:
        return 1.0

    return max(1.0, float(n_total * m_sub) / rho_sum)


def _auto_max_lag(
        dhdt: np.ndarray,
        x_coords: np.ndarray,
        y_coords: np.ndarray,
        hard_cap_m: float = 200_000.0,
        glacier_max_dimension_m: float = None,
) -> float:
    """Adaptive max_lag selection for variogram fitting."""
    valid = np.isfinite(dhdt)
    if int(np.sum(valid)) < 100:
        return 20_000.0

    # Domain extent
    x_v, y_v = x_coords[valid], y_coords[valid]
    diagonal = float(np.sqrt(
        (x_v.max() - x_v.min()) ** 2 + (y_v.max() - y_v.min()) ** 2
    ))
    lag = diagonal * 0.5

    # Physical upper bound: 0.5 × shortest glacier dimension
    if glacier_max_dimension_m is not None:
        glacier_physical_cap = glacier_max_dimension_m * 0.5
        lag = min(lag, glacier_physical_cap)
        print(f"      → max_lag capped at 0.5 × glacier dimension = {glacier_physical_cap / 1000:.1f} km")

    # Hard ceiling and floor
    lag = min(lag, hard_cap_m)
    lag = max(lag, 20_000.0)

    return float(lag)


def estimate_decorrelation_length(dhdt_off_ice, x_coords, y_coords, sample_size=10_000,
                         n_pairs=1_000_000, n_lag_bins=35, random_seed=42,
                         glacier_max_dimension_m=None):
    """
    One Gaussian + spherical fit (Hugonnet et al. 2022) to the empirical
    variogram sampled once up to the shortest glacier dimension (max 200 km).
    No boundary retries; a boundary hit is only flagged.
    """
    valid = np.isfinite(dhdt_off_ice) & (np.abs(dhdt_off_ice) <= 50.0)
    n_stable = int(valid.sum())
    fb = _fallback_variogram_params()
    if glacier_max_dimension_m is not None and np.isfinite(glacier_max_dimension_m):
        max_lag = float(min(glacier_max_dimension_m, 200_000.0))
    else:
        max_lag = _auto_max_lag(dhdt_off_ice, x_coords, y_coords) * 2.0

    def _out(params, emp, corr, ok, bh):
        return {"decorrelation_length_m": float(params["range"].max()),
                "variogram_params": params, "empirical_variogram": emp,
                "correlation_func": corr, "total_sill": float(params["psill"].sum()),
                "fit_success": ok, "n_stable_pts": n_stable,
                "rolstad_equivalent_factor": np.nan, "boundary_hit": bh,
                "max_lag_used_m": max_lag, "variogram_mode": "single_fit"}

    if n_stable < 500:
        print(f"      ⚠️ Only {n_stable} stable pts — variogram fallback (1000 m).")
        return _out(fb, pd.DataFrame(columns=["lags", "exp", "count"]),
                    _make_corr_func(fb), False, False)
    print(f"      → Variogram (Dowd + exp. bins, single fit) from {min(n_stable, sample_size):,} / "
          f"{n_stable:,} stable pts, max_lag = {max_lag / 1000:.1f} km (shortest glacier dimension)…")
    emp = sample_empirical_variogram_multiscale(
        dhdt_off_ice, x_coords, y_coords, sample_size=sample_size, n_pairs=n_pairs,
        max_lag=max_lag, n_lag_bins=n_lag_bins, random_seed=random_seed)
    if emp is None or emp.empty or int(emp["exp"].notna().sum()) < 5:
        print("      ⚠️ Too few populated bins — fallback (1000 m).")
        return _out(fb, emp, _make_corr_func(fb), False, False)
    params, corr, ok = fit_sum_variogram_hugonnet(emp, max_lag=max_lag, random_seed=random_seed)
    if not ok:
        print("      ⚠️ Fit failed — fallback (1000 m).")
        return _out(fb, emp, _make_corr_func(fb), False, False)
    for _, row in params.iterrows():
        print(f"         {row['model']:10s}: range = {row['range']:8.0f} m, psill = {row['psill']:.5f}")
    bh = bool(float(params["range"].max()) >= 0.95 * max_lag)
    if bh:
        print("      ⚠️ Long range reaches max_lag (accepted, no retry) -- check the variogram CSV.")
    return _out(params, emp, corr, ok, bh)
