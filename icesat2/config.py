"""
Configuration of the ICESat-2 dh/dt processing: filtering of the repeat-track
(captoolkit fitsec.py) and crossover (captoolkit xover.py) results and
interpolation to a grid.

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
# 4. Interpolation of the repeat-track dh/dt to a grid
# =============================================================================
INTERP_INPUT = RT_OUTPUT     # filtered points (columns p1, p1_err)
INTERP_MASK = os.path.join(DATA_DIR, "outlines/study_area.shp")  # grid extent and clip polygon
INTERP_OUTPUT = os.path.join(OUTPUT_DIR, "repeat_track_dhdt_grid.tif")      # gridded dh/dt
INTERP_OUTPUT_ERR = os.path.join(OUTPUT_DIR, "repeat_track_dhdt_err_grid.tif")  # gridded p1_err
RESOLUTION = 300             # grid resolution (m)
K_IDW = 12                   # nearest points averaged per grid cell (inverse distance weighting)
IDW_POWER = 2                # distance-weighting exponent
MAX_DISTANCE = 9000          # cells farther than this from any point stay empty (m)
SIGMA = 2.5                  # Gaussian smoothing (in grid cells)
ERR_CLIP_PCTL = (1, 99)      # percentile range the error grid is clipped to; None disables

# =============================================================================
# 5. Figures
# =============================================================================
QC_DPI = 150
QC_DHDT_RANGE = (-3, 3)      # colour range of the QC maps (m a-1)
