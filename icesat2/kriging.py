"""
Geostatistics of the ICESat-2 repeat-track dh/dt nodes: variogram, track
detection and crossover adjustment of track offsets, ordinary kriging with
node-specific noise and track-correlated errors, and cross-validation.
Used by grid_dhdt.py; settings in config.py (section 4).
"""
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
from scipy.spatial import cKDTree

import config as cfg


# ---------------------------------------------------------------------------
# Variogram
# ---------------------------------------------------------------------------
def _rho(h, a):
    """Correlation function of the fitted model (1 at h = 0)."""
    h = np.asarray(h, float)
    if cfg.VARIOGRAM_MODEL == 'spherical':
        r = h / a
        return np.where(r < 1, 1 - 1.5 * r + 0.5 * r ** 3, 0.0)
    return np.exp(-3.0 * h / a)


def _gamma_model(h, nugget, psill, a):
    return nugget + psill * (1 - _rho(h, a))


def empirical_variogram(xy, v, rng):
    """Dowd's robust estimator, gamma = 1.099 * median(|dv|)^2, per lag bin."""
    n = len(v)
    anchors = rng.choice(n, size=min(cfg.VARIOGRAM_N_ANCHORS, n), replace=False)
    tree_all = cKDTree(xy)
    sub = rng.choice(n, size=min(cfg.VARIOGRAM_LONG_SUBSAMPLE, n), replace=False)
    tree_sub = cKDTree(xy[sub])
    dd, dv = [], []
    for i in anchors:
        js = np.asarray(tree_all.query_ball_point(xy[i], cfg.VARIOGRAM_SHORT_LAG), int)
        js = js[js != i]
        d = np.hypot(*(xy[js] - xy[i]).T)
        dd.append(d); dv.append(np.abs(v[js] - v[i]))
        jl = sub[np.asarray(tree_sub.query_ball_point(xy[i], cfg.VARIOGRAM_MAX_LAG), int)]
        d = np.hypot(*(xy[jl] - xy[i]).T)
        keep = d > cfg.VARIOGRAM_SHORT_LAG
        dd.append(d[keep]); dv.append(np.abs(v[jl[keep]] - v[i]))
    dd, dv = np.concatenate(dd), np.concatenate(dv)
    rows = []
    for lo, hi in zip(np.asarray(cfg.LAG_EDGES, float)[:-1], np.asarray(cfg.LAG_EDGES, float)[1:]):
        s = (dd > lo) & (dd <= hi)
        if s.sum() < 30:
            continue
        rows.append(dict(lag_lo=lo, lag_hi=hi, lag_mean=dd[s].mean(), n_pairs=int(s.sum()),
                         gamma_dowd=1.099 * np.median(dv[s]) ** 2,
                         gamma_classic=0.5 * np.mean(dv[s] ** 2)))
    return pd.DataFrame(rows)


def fit_variogram(ev):
    fit = ev[ev['lag_lo'] >= cfg.VARIOGRAM_MIN_LAG]
    h, g = fit['lag_mean'].values, fit['gamma_dowd'].values
    g0, gmax = max(g[0], 1e-6), max(g.max(), 1e-6)
    p0 = [0.8 * g0, max(gmax - 0.8 * g0, 1e-6), 5000.0]
    bounds = ([1e-6 * gmax, 1e-6 * gmax, 300.0], [2 * gmax, 3 * gmax, 3 * cfg.VARIOGRAM_MAX_LAG])
    # Relative weighting (sigma proportional to gamma): every lag bin counts
    # equally in relative terms. Weighting by pair count would let the
    # millions of long-lag pairs decide the fit and push the nugget to zero.
    popt, _ = curve_fit(_gamma_model, h, g, p0=p0, sigma=np.maximum(g, 1e-9),
                        bounds=bounds, maxfev=20000)
    nugget, psill, a = popt
    return dict(nugget=float(nugget), psill=float(psill), range_m=float(a))


def correlation_area(a):
    """A_corr = 2 pi int rho(h) h dh  (m^2)."""
    h = np.linspace(0, 5 * a, 20001)
    return float(2 * np.pi * np.trapezoid(_rho(h, a) * h, h)) if hasattr(np, 'trapezoid') \
        else float(2 * np.pi * np.trapz(_rho(h, a) * h, h))


# ---------------------------------------------------------------------------
# Kriging and track model
# ---------------------------------------------------------------------------
def _sector_neighbours(tree, xy, t, n_nodes):
    """Indices/distances of up to N_SECTORS * N_PER_SECTOR neighbours per
    target, N_PER_SECTOR nearest per direction sector. Returns
    (idx, d, valid) of shape (m, kmax); invalid slots are padding."""
    ks = min(cfg.K_SEARCH, n_nodes)
    d, idx = tree.query(t, k=ks)
    if ks == 1:
        d, idx = d[:, None], idx[:, None]
    ok = np.isfinite(d)
    idx = np.where(ok, idx, 0)
    dxy = xy[idx] - t[:, None, :]
    sec = np.floor((np.arctan2(dxy[..., 1], dxy[..., 0]) + np.pi) / (2 * np.pi) * cfg.N_SECTORS)
    sec = np.clip(sec, 0, cfg.N_SECTORS - 1).astype(np.int64)
    onehot = (sec[..., None] == np.arange(cfg.N_SECTORS)) & ok[..., None]
    rank = np.take_along_axis(np.cumsum(onehot, axis=1), sec[..., None], axis=2)[..., 0] - 1
    keep = ok & (rank < cfg.N_PER_SECTOR)
    kmax = min(cfg.N_SECTORS * cfg.N_PER_SECTOR, ks)
    order = np.argsort(~keep, axis=1, kind='stable')[:, :kmax]   # kept first, by distance
    return (np.take_along_axis(idx, order, 1), np.take_along_axis(d, order, 1),
            np.take_along_axis(keep, order, 1))


def track_directions(xy, chunk=2000):
    """Local track direction (radians, axial) of every node by a local line
    search: for each candidate direction (every TRACK_DIR_STEP_DEG) count the
    neighbours within TRACK_DIR_RADIUS_M that lie within TRACK_PERP_TOL_M of
    the line through the node; the direction with most nodes wins. NaN if
    fewer than TRACK_DIR_MIN_NODES nodes support it. Robust to the 150 m
    node grid and to crossings (a crossing node gets one of the two tracks)."""
    tree = cKDTree(xy)
    angs = np.deg2rad(np.arange(0, 180, cfg.TRACK_DIR_STEP_DEG))
    ca, sa = np.cos(angs), np.sin(angs)
    theta = np.full(len(xy), np.nan)
    for s0 in range(0, len(xy), chunk):
        p = xy[s0:s0 + chunk]
        d, idx = tree.query(p, k=cfg.TRACK_DIR_K, distance_upper_bound=cfg.TRACK_DIR_RADIUS_M)
        ok = np.isfinite(d) & (d > 0)
        idx = np.where(ok, idx, 0)
        off = xy[idx] - p[:, None, :]                       # (m, K, 2)
        perp = np.abs(-off[..., 0, None] * sa + off[..., 1, None] * ca)   # (m, K, A)
        cnt = ((perp <= cfg.TRACK_PERP_TOL_M) & ok[..., None]).sum(1)          # (m, A)
        best = cnt.argmax(1)
        th = angs[best]
        th[cnt.max(1) < cfg.TRACK_DIR_MIN_NODES] = np.nan
        theta[s0:s0 + len(p)] = th
    return theta


def track_labels(xy, theta):
    """Group nodes into tracks: connected chains of nodes that have a clear,
    similar local direction and lie close to each other's track line.
    Nodes without a clear direction (crossings) stay on their own. Using
    labels (an equivalence relation) keeps the track-error covariance
    positive definite."""
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components
    n = len(xy)
    pairs = cKDTree(xy).query_pairs(r=cfg.TRACK_LINK_M, output_type='ndarray')
    i, j = pairs[:, 0], pairs[:, 1]
    ti, tj = theta[i], theta[j]
    sin_tol = np.sin(np.deg2rad(cfg.TRACK_ANGLE_TOL_DEG))
    d = xy[j] - xy[i]
    dist = np.hypot(d[:, 0], d[:, 1])
    phi = np.arctan2(d[:, 1], d[:, 0])
    ok = np.isfinite(ti) & np.isfinite(tj) & (np.abs(np.sin(ti - tj)) < sin_tol)
    # perpendicular offset from node i's track line (robust to the 150 m
    # node grid, whose steps are only 0/45/90 degrees)
    perp = np.abs(dist * np.sin(phi - ti))
    ok &= perp <= cfg.TRACK_PERP_TOL_M
    g = coo_matrix((np.ones(ok.sum()), (i[ok], j[ok])), shape=(n, n))
    n_lab, lab = connected_components(g, directed=False)
    return lab


def track_adjustment(xy, v, labels, theta):
    """Damped least-squares rate offset per track segment from node pairs of
    different segments within TRACK_ADJ_RADIUS_M (i.e. at crossings).
    Returns (offset per node, per-segment table)."""
    from scipy.sparse import coo_matrix
    from scipy.sparse.linalg import lsqr
    sizes = np.bincount(labels)
    big = sizes >= cfg.TRACK_ADJ_MIN_NODES
    pairs = cKDTree(xy).query_pairs(r=cfg.TRACK_ADJ_RADIUS_M, output_type='ndarray')
    i, j = pairs[:, 0], pairs[:, 1]
    li, lj = labels[i], labels[j]
    # only pairs of CROSSING tracks (different directions); fragments of the
    # same track or parallel tracks would only tie segments that share the
    # same offset anyway
    ok = (li != lj) & big[li] & big[lj] & \
        (np.abs(np.sin(theta[i] - theta[j])) > np.sin(np.deg2rad(cfg.TRACK_ADJ_MIN_CROSS_DEG)))
    i, j, li, lj = i[ok], j[ok], li[ok], lj[ok]
    seg = np.unique(np.r_[li, lj])
    col = -np.ones(labels.max() + 1, np.int64)
    col[seg] = np.arange(len(seg))
    n = len(i)
    if n == 0:
        return np.zeros(len(v)), pd.DataFrame()
    rows = np.r_[np.arange(n), np.arange(n)]
    cols = np.r_[col[li], col[lj]]
    vals = np.r_[np.ones(n), -np.ones(n)]
    A = coo_matrix((vals, (rows, cols)), shape=(n, len(seg))).tocsr()
    rhs = v[i] - v[j]
    # robust: two passes, second without pairs beyond 3 NMAD of the residual
    sol = lsqr(A, rhs, damp=cfg.TRACK_ADJ_DAMP)[0]
    r = rhs - A @ sol
    keep = np.abs(r - np.median(r)) <= 3 * nmad(r)
    sol = lsqr(A[keep], rhs[keep], damp=cfg.TRACK_ADJ_DAMP)[0]
    sol -= np.average(sol, weights=sizes[seg])          # no net shift of the field
    off = np.zeros(len(v))
    has = col[labels] >= 0
    off[has] = sol[col[labels[has]]]
    n_pairs = np.bincount(np.r_[col[li], col[lj]][np.r_[keep, keep]], minlength=len(seg))
    tab = pd.DataFrame(dict(segment=seg, n_nodes=sizes[seg], n_crossing_pairs=n_pairs, offset_myr=sol))
    r2 = rhs[keep] - A[keep] @ sol
    print(f"  track adjustment: {len(seg):,} segments, {keep.sum():,} crossing node pairs; "
          f"offsets SD {sol.std():.3f} m/a (5-95 %: {np.percentile(sol, 5):+.3f} to "
          f"{np.percentile(sol, 95):+.3f}); crossing differences NMAD "
          f"{nmad(rhs[keep]):.3f} -> {nmad(r2):.3f} m/a")
    return off, tab


def krige(targets, xy, v, noise, vg, chunk=None, labels=None):
    """Ordinary kriging with node-specific noise variance (noise filtered: estimate and
    variance refer to the smooth dh/dt field). Returns (prediction, variance,
    distance to the nearest used node)."""
    chunk = cfg.CHUNK if chunk is None else chunk
    psill, a = vg['psill'], vg['range_m']
    tvar = vg.get('track_var', 0.0) if labels is not None else 0.0
    tree = cKDTree(xy)
    pred = np.full(len(targets), np.nan)
    var = np.full(len(targets), np.nan)
    dnear = np.full(len(targets), np.nan)
    big = 1e6 * psill
    for s0 in range(0, len(targets), chunk):
        t = targets[s0:s0 + chunk]
        idx, d, valid = _sector_neighbours(tree, xy, t, len(v))
        m, k = idx.shape
        dnear[s0:s0 + m] = np.where(valid, d, np.inf).min(1)
        P = xy[idx]                                            # (m, k, 2)
        Dij = np.sqrt(((P[:, :, None, :] - P[:, None, :, :]) ** 2).sum(-1))
        vv = valid[:, :, None] & valid[:, None, :]
        A = np.ones((m, k + 1, k + 1))
        A[:, :k, :k] = np.where(vv, psill * _rho(Dij, a), 0.0)
        if tvar > 0:
            L = labels[idx]
            same = L[:, :, None] == L[:, None, :]
            A[:, :k, :k] += np.where(vv & same, tvar * np.exp(-Dij / cfg.TRACK_ERR_LENGTH_M), 0.0)
        # padding slots: no covariance, huge noise -> weight ~ 0
        A[:, np.arange(k), np.arange(k)] += np.where(valid, noise[idx] + 1e-9 * psill, big)
        A[:, k, k] = 0.0
        b = np.ones((m, k + 1))
        b[:, :k] = np.where(valid, psill * _rho(d, a), 0.0)
        sol = np.linalg.solve(A, b[..., None])[..., 0]
        w, mu = sol[:, :k], sol[:, k]
        w = np.where(valid, w, 0.0)
        pred[s0:s0 + m] = (w * v[idx]).sum(1)
        var[s0:s0 + m] = np.maximum(psill - (w * b[:, :k]).sum(1) - mu, 0.0)
    return pred, var, dnear


# ---------------------------------------------------------------------------
# Cross-validation and statistics
# ---------------------------------------------------------------------------
def make_folds(xy, rng):
    """Fold number per node: whole CV_BLOCK_M blocks go to one fold."""
    bx = np.floor(xy[:, 0] / cfg.CV_BLOCK_M).astype(np.int64)
    by = np.floor(xy[:, 1] / cfg.CV_BLOCK_M).astype(np.int64)
    blocks, inv = np.unique(np.c_[bx, by], axis=0, return_inverse=True)
    return (rng.permutation(len(blocks)) % cfg.CV_FOLDS)[inv.ravel()]


def make_track_folds(labels, rng):
    """Fold number per node: whole track segments go to one fold."""
    return (rng.permutation(labels.max() + 1) % cfg.CV_FOLDS)[labels]


def block_cross_validation(xy, v, noise, vg, fold, test_subset=None, labels=None, theta=None):
    """Leave-blocks-out CV; returns per-node table. `test_subset` (boolean)
    limits which held-out nodes are predicted (training always uses all
    nodes outside the fold)."""
    out = []
    for f in range(cfg.CV_FOLDS):
        te, tr = fold == f, fold != f
        if theta is not None:
            # leave-track-out: also drop training nodes of the same track
            # direction close to the held-out nodes (fragments of that track)
            ti = np.nonzero(tr)[0]
            dd, jj = cKDTree(xy[te]).query(xy[ti], k=1, distance_upper_bound=cfg.CV_TRACK_EXCL_M)
            near = np.isfinite(dd)
            th_te = theta[np.nonzero(te)[0]]
            same_dir = np.zeros(len(ti), bool)
            same_dir[near] = np.abs(np.sin(theta[ti[near]] - th_te[jj[near]])) < \
                np.sin(np.deg2rad(cfg.TRACK_ANGLE_TOL_DEG))
            tr = tr.copy()
            tr[ti[near & same_dir]] = False
        if test_subset is not None:
            te = te & test_subset
        if te.sum() == 0 or tr.sum() < cfg.N_PER_SECTOR * 2:
            continue
        p, s2, dn = krige(xy[te], xy[tr], v[tr], noise[tr], vg,
                          labels=None if labels is None else labels[tr])
        out.append(pd.DataFrame(dict(x=xy[te, 0], y=xy[te, 1], fold=f, observed=v[te],
                                     predicted=p, sigma_signal=np.sqrt(s2),
                                     sigma_total=np.sqrt(s2 + noise[te] + vg.get('track_var', 0.0) * (labels is not None)),
                                     dist_to_train=dn)))
    cv = pd.concat(out, ignore_index=True)
    cv['residual'] = cv['observed'] - cv['predicted']
    cv['z'] = cv['residual'] / cv['sigma_total']
    return cv


def _cv_score(res):
    res = np.asarray(res, float); res = res[np.isfinite(res)]
    if cfg.CV_TUNE_SCORE == 'rmse':
        return float(np.sqrt(np.mean(res ** 2)))
    if cfg.CV_TUNE_SCORE == 'nmad':
        return nmad(res)
    return float(np.mean(np.abs(res)))


def nmad(x):
    x = np.asarray(x, float); x = x[np.isfinite(x)]
    return float(1.4826 * np.median(np.abs(x - np.median(x)))) if x.size else np.nan


def summary(d, prefix=''):
    d = np.asarray(d, float); d = d[np.isfinite(d)]
    if d.size == 0:
        return {f'{prefix}n': 0}
    return {f'{prefix}n': d.size, f'{prefix}mean': d.mean(), f'{prefix}std': d.std(),
            f'{prefix}median': np.median(d), f'{prefix}nmad': nmad(d),
            f'{prefix}rmse': np.sqrt(np.mean(d ** 2))}
