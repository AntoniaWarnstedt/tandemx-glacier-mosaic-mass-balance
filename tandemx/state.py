"""
Hand-over between processing and plotting.

The processing run stores its in-memory results (DEM mosaics, dh/dt fields and
mass-balance results per period) in cfg.STATE_FILE; 2-D arrays are written to
compressed .npz files next to it. The plotting run reloads them.
"""
import os
import pickle

import numpy as np

from . import config as cfg


def _split_arrays(d, npz_path):
    """Move the 2-D arrays of a dict into an .npz file; aliases are kept."""
    arrays, alias, slim = {}, {}, {}
    seen = {}
    for k, v in d.items():
        if isinstance(v, np.ndarray) and v.ndim == 2:
            if id(v) in seen:
                alias[k] = seen[id(v)]
            else:
                seen[id(v)] = k
                arrays[k] = v
        else:
            slim[k] = v
    np.savez_compressed(npz_path, **arrays)
    slim["_npz"] = os.path.basename(npz_path)
    slim["_alias"] = alias
    return slim


def _join_arrays(slim, state_dir):
    d = dict(slim)
    with np.load(os.path.join(state_dir, d.pop("_npz"))) as z:
        for k in z.files:
            d[k] = z[k]
    for k, src in d.pop("_alias").items():
        d[k] = d[src]
    return d


def save_state(year_results, all_changes, mb_results, grid_info, onset):
    state_dir = os.path.dirname(cfg.STATE_FILE)
    os.makedirs(state_dir, exist_ok=True)
    changes = [_split_arrays(c, os.path.join(state_dir, f"state_change_{c['yA']}_{c['yB']}.npz"))
               for c in all_changes]
    mbs = [_split_arrays(r, os.path.join(state_dir, f"state_mb_{r['yA']}_{r['yB']}.npz"))
           for r in mb_results]
    runtime_cfg = {"SAR_PEN_START_ELEV": cfg.SAR_PEN_START_ELEV,
                   "SAR_PEN_END_ELEV": cfg.SAR_PEN_END_ELEV}
    with open(cfg.STATE_FILE, "wb") as f:
        pickle.dump({"year_results": year_results, "all_changes": changes, "mb_results": mbs,
                     "grid_info": grid_info, "onset": onset, "runtime_cfg": runtime_cfg}, f)
    print(f"   Saved processing state: {cfg.STATE_FILE}")


def load_state():
    if not os.path.exists(cfg.STATE_FILE):
        raise FileNotFoundError(f"{cfg.STATE_FILE} not found -- run run_tandemx_processing.py first.")
    state_dir = os.path.dirname(cfg.STATE_FILE)
    with open(cfg.STATE_FILE, "rb") as f:
        st = pickle.load(f)
    for k, v in st.pop("runtime_cfg").items():
        setattr(cfg, k, v)
    st["all_changes"] = [_join_arrays(c, state_dir) for c in st["all_changes"]]
    st["mb_results"] = [_join_arrays(r, state_dir) for r in st["mb_results"]]
    return st
