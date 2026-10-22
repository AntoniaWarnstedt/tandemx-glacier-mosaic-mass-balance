"""
Configuration of the ICESat-2 dh/dt processing: filtering of the repeat-track
(captoolkit fitsec.py) and crossover (captoolkit xover.py) results, kriging of
the repeat-track dh/dt to a grid and dh/dt along the glacier centerlines.

All paths are placeholders. Point DATA_DIR / OUTPUT_DIR to your own data.
"""
import os

DATA_DIR = "path/to/data"                           # root of all input data
OUTPUT_DIR = os.path.join("path/to/output", "icesat2")

# =============================================================================
# 1. Repeat-track dh/dt (fitsec.py output, columns p1 = dh/dt, p1_err)
# =============================================================================
RT_INPUT = os.path.join(DATA_DIR, "icesat2/repeat_track_dhdt.shp")   # projected CRS (m)
RT_OUTPUT = os.path.join(OUTPUT_DIR, "repeat_track_dhdt_clean.shp")
RT_MIN_DHDT = -30.0          # dh/dt outside [min, max] is discarded (m a-1)
RT_MAX_DHDT = 30.0
RT_MAX_P1_ERR = 5.0          # max. fit uncertainty p1_err (m a-1)

# =============================================================================
# 2. Crossover dh/dt (xover.py output, columns dhdt, dh_elv, dt_year, slp_1, slp_2)
# =============================================================================
XO_INPUT = os.path.join(DATA_DIR, "icesat2/crossover_dhdt.shp")         # projected CRS (m)
XO_OUTPUT = os.path.join(OUTPUT_DIR, "crossover_dhdt_clean.shp")
XO_MIN_DHDT = -30.0          # dh/dt outside [min, max] is discarded (m a-1)
XO_MAX_DHDT = 30.0
XO_MAX_DH_ELV = 15.0         # max. |elevation difference| at the crossover (m)
XO_MIN_DT_YEAR = 0.5         # min. time between the two passes (years); 0 disables the filter
XO_MAX_SLOPE_DEG = 30.0      # max. local along-track slope max(|slp_1|, |slp_2|) (deg);
                             # limits slope-induced errors (Brenner et al., 1983)

# =============================================================================
# 3. Local spatial outlier filter (both data sets)
# =============================================================================
# Each point is compared to the median of its K nearest neighbours and removed
# if it deviates by more than LOCAL_N_SIGMA x local MAD (scaled by 1.4826).
K_NEIGHBORS = 30             # neighbours forming the local reference
LOCAL_N_SIGMA = 3.0          # threshold in local robust standard deviations
MAD_FLOOR = 0.05             # lower bound of the local MAD (m a-1)

# =============================================================================
# 4. Kriging of the repeat-track dh/dt (grid_dhdt.py, plot_grid_diagnostics.py)
# =============================================================================
GRID_INPUT = RT_OUTPUT       # filtered fitsec nodes
GRID_MASK = os.path.join(DATA_DIR, "outlines/study_area.shp")   # catchment: grid extent and mean
GRID_XOVER = XO_OUTPUT       # filtered crossovers (independent check; None = skip)
GRID_OUTPUT_DIR = os.path.join(OUTPUT_DIR, "grid")
GRID_PREFIX = "is2_dhdt"     # prefix of all output files
P1_COL, P1_ERR_COL = "p1", "p1_err"   # fitsec rate and its formal error (m a-1)
XO_COL = "dhdt"              # crossover rate (m a-1)
XO_DT_COL = "dt_year"        # crossover time separation (years), optional

# Grid and neighbour search. ICESat-2 nodes are ~150 m apart along track but
# 1-3 km across track, so the plain k nearest nodes all lie on one track; the
# N_PER_SECTOR nearest nodes in each of N_SECTORS direction sectors (taken from
# the K_SEARCH nearest) bring in the tracks on both sides of every cell.
RESOLUTION = 300.0           # grid resolution (m)
MAX_DISTANCE = 9000.0        # cells farther than this from any node stay empty (m)
N_SECTORS = 8
N_PER_SECTOR = 4
K_SEARCH = 256
CHUNK = 3000                 # cells per batched kriging solve

# Variogram (Dowd's robust estimator, fitted with relative weights)
VARIOGRAM_MODEL = "spherical"   # "spherical" or "exponential"
VARIOGRAM_MAX_LAG = 30000.0  # m
VARIOGRAM_MIN_LAG = 200.0    # lags below are not fitted: fitsec nodes closer than 2 x the 100 m
                             # search radius share data (m)
VARIOGRAM_N_ANCHORS = 3000   # random nodes the pairs are built from
VARIOGRAM_SHORT_LAG = 3000.0  # pairs up to this lag use all nodes, longer lags a subsample (m)
VARIOGRAM_LONG_SUBSAMPLE = 20000
LAG_EDGES = [0.0, 200.0, 300.0, 450.0, 600.0, 800.0, 1000.0, 1300.0, 1600.0, 2000.0, 2500.0,
             3000.0, 4000.0, 5000.0, 6500.0, 8000.0, 10000.0, 13000.0, 16000.0, 20000.0,
             25000.0, 30000.0]       # lag-bin edges (m)

# Nugget and track error, chosen as fractions of the total sill (total kept
# fixed) that best predict data left out of the fit (score CV_TUNE_SCORE):
#   "track_cv"   leave-track-out cross-validation on the nodes (crossovers stay independent)
#   "crossovers" tuned against the crossover rates (then no longer independent)
#   "cv"         plain block cross-validation (prefers too small nuggets)
#   None         keep the fitted nugget
NUGGET_TUNING = "track_cv"
NUGGET_CANDIDATES = [0.02, 0.05, 0.1, 0.2]
CV_TUNE_SCORE = "mae"        # "mae", "rmse" or "nmad"
CV_TUNE_MAX_TEST = 30000     # held-out nodes per candidate (random subset)

# Track-correlated error: extra covariance track_var * exp(-d / TRACK_ERR_LENGTH_M)
# between nodes of the same track. Tracks = chains of nodes with a similar
# local direction (TRACK_ANGLE_TOL_DEG), within TRACK_PERP_TOL_M of each
# other's track line and TRACK_LINK_M of each other. The local direction comes
# from a line search through the neighbours within TRACK_DIR_RADIUS_M.
TRACK_ERROR_MODEL = True
TRACK_VAR_CANDIDATES = [0.0, 0.1, 0.2]   # track_var as fractions of the total sill
TRACK_ERR_LENGTH_M = 20000.0
TRACK_DIR_RADIUS_M = 1000.0
TRACK_DIR_K = 80             # neighbours examined for the direction search
TRACK_DIR_STEP_DEG = 2.0
TRACK_DIR_MIN_NODES = 5
TRACK_ANGLE_TOL_DEG = 6.0
TRACK_LINK_M = 1000.0        # max. gap between linked nodes of one track (bridges crossings)
TRACK_PERP_TOL_M = 120.0     # max. offset across the track direction (beam pair ~90 m apart)

# Crossover adjustment: one rate offset per track segment (>= TRACK_ADJ_MIN_NODES
# nodes) from node pairs of crossing segments within TRACK_ADJ_RADIUS_M, damped
# least squares (TRACK_ADJ_DAMP), subtracted before kriging.
TRACK_ADJUST = True
TRACK_ADJ_RADIUS_M = 400.0
TRACK_ADJ_MIN_CROSS_DEG = 20.0   # minimum crossing angle between the two tracks
TRACK_ADJ_MIN_NODES = 10
TRACK_ADJ_DAMP = 3.0

# Node noise = nugget * (p1_err / median p1_err)^2, ratio limited to
# [1/NOISE_RATIO_MAX, NOISE_RATIO_MAX].
NOISE_WEIGHTING = True
NOISE_RATIO_MAX = 25.0

# Cross-validation and calibration of sigma_IS2
CV_SCHEME = "track"          # "track" (leave-track-out, needs TRACK_ERROR_MODEL) or "block"
CV_TRACK_EXCL_M = 1000.0     # also drop training nodes of the same direction this close to a test node
CV_BLOCK_M = 3000.0          # block size of "block" cross-validation (m)
CV_FOLDS = 10
APPLY_CV_CALIBRATION = True  # scale sigma_IS2 by the 68th percentile of |z| per distance class
CV_DIST_EDGES = [0.0, 150.0, 300.0, 600.0, 1000.0, 1500.0, 2000.0, 3000.0, 5000.0, 9000.0]  # (m)
CV_MIN_PER_BIN = 30
CV_CALIBRATION_MIN = 1.0     # calibration may only increase sigma_IS2

# Catchment mean
SIGMA_SYS_MYR = 0.0          # systematic rate uncertainty added in quadrature (m a-1)
NEAR_NODE_M = 1000.0         # a second mean is reported for cells within this distance of a node

# Crossover comparison
XO_NODE_MATCH_M = 300.0      # nearest-node comparison only within this distance (m)
RANDOM_SEED = 42

# =============================================================================
# 5. dh/dt along the glacier centerlines (centerline_dhdt.py, plot_centerline_dhdt.py)
# =============================================================================
CL_DHDT_RASTER = None        # dh/dt raster; None = <GRID_OUTPUT_DIR>/<GRID_PREFIX>_dhdt.tif
CL_DIR = os.path.join(DATA_DIR, "centerlines")     # one Centerline_<Name>_NN.shp per glacier
CL_TERMINUS_MASK = GRID_MASK # outline defining the terminus (distance 0) of each centerline
CL_OUTPUT_DIR = os.path.join(OUTPUT_DIR, "centerlines")
CL_SAMPLING_DISTANCE = 100   # spacing of the samples along the centerline (m)
CL_MAX_DISTANCE = 15000      # profile length from the terminus (m)
CL_BUFFER_WIDTH = 100        # samples averaged across the centerline within this distance (m)
PERIOD_LABEL = ""            # period shown in figure titles, e.g. "2020-2025"

# --- Example: settings of Warnstedt et al. ---
# GRID_PREFIX = "is2_dhdt_2020_2025"
# PERIOD_LABEL = "2020-2025"

# =============================================================================
# 6. Figures
# =============================================================================
QC_DPI = 150
QC_DHDT_RANGE = (-3, 3)      # colour range of the QC maps (m a-1)
