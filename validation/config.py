"""
Configuration of the cross-sensor validation of the TanDEM-X DEMs and dh/dt
against CryoSat-2 (CryoTEMPO-EOLIS) and ICESat-2 (ATL06).

All paths are placeholders. Point DATA_DIR / OUTPUT_DIR to your own data.
"""
import os

# =============================================================================
# 1. Paths
# =============================================================================
DATA_DIR = "path/to/data"            # root of all input data
OUTPUT_DIR = os.path.join("path/to/output", "validation")
CS2_OUTPUT_DIR = os.path.join(OUTPUT_DIR, "CS2")            # CryoSat-2 vs TanDEM-X elevation
IS2_OUTPUT_DIR = os.path.join(OUTPUT_DIR, "IS2")            # ICESat-2 vs TanDEM-X elevation
CS2_DHDT_OUTPUT_DIR = os.path.join(OUTPUT_DIR, "CS2_dhdt")  # CryoSat-2 vs TanDEM-X dh/dt
COMBINED_OUTPUT_DIR = os.path.join(OUTPUT_DIR, "combined_cs2_is2")  # combined figures
STATE_FILE = os.path.join(OUTPUT_DIR, "validation_state.pkl")  # hand-over processing -> plotting

# TanDEM-X products of run_tandemx_processing.py (dem_<epoch>.tif, dhdt_*.tif).
TDX_DIR = os.path.join("path/to/output", "tandemx")

# EGM2008 geoid grid (https://cdn.proj.org/us_nga_egm08_25.tif). The ellipsoidal
# CryoSat-2 and ICESat-2 heights are converted to EGM2008, the vertical
# reference of the TanDEM-X mosaics.
GEOID_GRID_EGM2008 = os.path.join(DATA_DIR, "geoid/us_nga_egm08_25.tif")

# CryoSat-2 point files cs2_points_<YYYY>.gpkg/.shp with columns 'elevation'
# (WGS84 ellipsoidal), 'datetime' (or 'dec_year') and 'uncertainty'/'elev_unc'.
CS2_SHP_DIR = os.path.join(DATA_DIR, "cryosat2")

# ICESat-2 ATL06 points: files found with IS2_FILE_GLOB below IS2_BASE_PATH
# ({year} = 4-digit year, '**' = any sub-folder; a merged epoch "2024-2025"
# uses the files of both years).
IS2_BASE_PATH = os.path.join(DATA_DIR, "icesat2_atl06")
IS2_FILE_GLOB = "atl06_{year}*.shp"
IS2_HEIGHT_COL = "h_elv"             # ellipsoidal (WGS84) ATL06 height h_li, converted to EGM2008
IS2_TIME_COL = "t_year"              # decimal-year time column; else 'datetime'/'date'/'time'/'delta_time'
IS2_APPLY_TIME_FILTER = False        # apply the +-IS2_MAX_TEMPORAL_OFFSET window (False: files are
                                     # already limited to the TanDEM-X dates; the window is only reported)
IS2_ROI_PATH = os.path.join(DATA_DIR, "outlines/study_area.shp")    # ICESat-2 points are clipped to it
IS2_YEARS = []                       # TanDEM-X epochs (dem_<epoch>.tif) compared with ICESat-2

# =============================================================================
# 2. Temporal matching
# =============================================================================
CS2_MAX_TEMPORAL_OFFSET = 30         # max. days between altimeter point and a TanDEM-X acquisition
IS2_MAX_TEMPORAL_OFFSET = 30

TDX_ACQUISITIONS = {}                # {year: ["YYYY-MM-DD", ...]} TanDEM-X acquisition dates per year;
                                     # a merged epoch "YYYY-YYYY" uses the dates of its years
ALL_YEARS_PERIOD_KEYS = []           # epochs pooled into "All Years"; [] = all epochs, merged epochs
                                     # replacing their single years

# =============================================================================
# 3. Co-located aggregation
# =============================================================================
AGGREGATION_GRID_SIZE_M = 500        # cell size; altimeter and TanDEM-X values are averaged per cell
AGGREGATION_ORIGIN_M = (0.0, 0.0)    # fixed grid origin -> identical cells for all sensors/epochs
AGGREGATION_MIN_POINTS_PER_CELL = 1  # min. co-located points per cell (elevation comparison)
ABS_DIFF_MAX_M = 100.0               # points with |TanDEM-X - altimeter| >= this (m) are discarded
HYPS_BIN_WIDTH_M = 100               # elevation-band width of the hypsometric statistics (m)
HYPS_MIN_SAMPLES = 10                # min. cells per elevation band
IS2_QUALITY_KEYWORDS = ("sigma", "unc", "rms", "quality", "error", "std",   # column-name keys of
                        "h_li_sigma", "dh_fit_dx_sigma", "atl06_quality",  # ATL06 quality fields
                        "snr_significance", "n_fit_photons", "w_surface_window")

# =============================================================================
# 4. CryoSat-2 dh/dt validation
# =============================================================================
USE_DHDT_CACHE = False               # reuse the matched CS2 point pairs of a previous run
DHDT_TIMESTAMP_YEARS = {}            # {epoch: [CS2 years]}; epochs not listed use their own years
                                     # ("2016-2017" -> ["2016", "2017"])
DHDT_PAIRS = []                      # [(later epoch, earlier epoch), ...]; [] = every dh/dt raster
                                     # matching DHDT_FILE_TEMPLATE in TDX_DIR
DHDT_FILE_TEMPLATE = "dhdt_filtered_3x3_{late}_minus_{early}.tif"   # TanDEM-X dh/dt raster in TDX_DIR
DHDT_MAX_PAIR_DISTANCE_M = 30        # max. distance between matched CS2 points of two epochs (m)
DHDT_MIN_PAIR_DT_YEARS = 0.5         # min. time between matched points (years)
DHDT_MAX_ABS_RATE_M_PER_YR = 30.0    # pairs with |dh/dt| above this are discarded (m a-1)
DHDT_GRID_SIZE_M = AGGREGATION_GRID_SIZE_M  # cell size of the dh/dt comparison
DHDT_MIN_PAIRS_PER_CELL = 3          # min. point pairs per cell
ENABLE_HYPSOMETRIC_DHDT = True       # elevation-banded dh/dt statistics

# =============================================================================
# 4b. Cross-checks of the TanDEM-X - altimeter offset (tables in OUTPUT_DIR/cross_checks)
# =============================================================================
# Same-cell comparison: ICESat-2 vs CryoSat-2 in identical grid cells, with
# TanDEM-X as transfer surface (IS2 - CS2 = (TDX - CS2) - (TDX - IS2)).
SAME_CELL_PERIODS = None             # epochs with both sensors; None = all such epochs
# Rock comparison: TanDEM-X minus altimeter on ice-free rock (no penetration,
# no surface change), i.e. a check of the vertical reference.
ROCK_OUTCROP_PATH = os.path.join(DATA_DIR, "rock_outcrop/rock_outcrop.shp")  # same mask as the TanDEM-X stable terrain
ROCK_DOMAIN_BUFFER_M = 10000.0       # rock inside IS2_ROI_PATH buffered by this distance (m)
ROCK_EDGE_BUFFER_M = 60.0            # rock polygons shrunk by this distance (ice/rock edge pixels) (m)
ROCK_LOW_SLOPE_DEG = 15.0            # statistics also for rock with slope <= this (deg)
ROCK_ABS_DIFF_MAX_M = 50.0           # point-level outlier threshold on rock (clouds, blunders) (m)

# =============================================================================
# 5. Figures
# =============================================================================
PLOT_DPI = 300
ROBUST_STATS = True                  # figures show median and NMAD (hypsometric panels, heatmap,
                                     # bias lines, statistics boxes) instead of / next to mean and SD
HEATMAP_SLOPE_MAX_DEG = 45.0         # slope range of the elevation x slope heatmap (deg)
HEATMAP_SLOPE_BIN_WIDTH_DEG = 5.0
DHDT_SCATTER_LIM = (-5, 5)           # axis range of the dh/dt scatter panels (m a-1)
SENSOR_COLORS = {"CryoSat-2": "#9B59B6", "CryoSat-2 dhdt": "#C71585", "ICESat-2": "#2E6DA4"}
NEUTRAL_LINE_COLOR = "#333333"       # zero and 1:1 lines
RED, GREY = "#CC2222", "#888888"      # fallback sensor colour, N-per-bin bars
SPINE_COLOR = "#333333"
GRID_KW = dict(color="#cccccc", linewidth=0.6, linestyle="-")
TICK_KW = dict(labelsize=15)
COMBINED_BIAS_YLIM = {}              # {sensor: (min, max)} y-range of the bias panels (m); missing = auto
HYPS_BIAS_YLIM = {"m": (-25, 25), "m a⁻¹": (-4, 9)}  # y-range of the hypsometric panels per unit
HYPS_N_YLIM_ALL_YEARS = None         # range of the "N per bin" axis, All-Years panel; None = auto
HYPS_N_YLIM_PANEL = {}               # {(sensor, epoch): (min, max)} per-panel overrides
HYPS_N_YLIM = {}                     # {unit: (min, max)} per unit ("m" / "m a⁻¹"); missing = auto
SCATTER_ELEV_LIM = None              # shared axis range of the elevation scatter panels (m); None = per panel
COMBINED_IS2_EPOCHS = None           # ICESat-2 epochs shown in the combined figures; None = last of IS2_YEARS
RC_PARAMS = {                        # font sizes for the large multi-panel figures
    "font.size": 16, "axes.titlesize": 18, "axes.labelsize": 17, "xtick.labelsize": 15,
    "ytick.labelsize": 15, "legend.fontsize": 14, "figure.titlesize": 24, "axes.linewidth": 1.0,
}


# =============================================================================
# Example: settings of Warnstedt et al. (Wordie Ice Shelf tributaries)
# =============================================================================
# --- Example (uncomment to use) ---
# IS2_FILE_GLOB = "winter_{year}_egm96/winter_{year}_egm96.shp"
# IS2_ROI_PATH = os.path.join(DATA_DIR, "outlines/outline_2025.shp")
# IS2_YEARS = [2022, 2023, 2024, 2025, "2024-2025"]
# TDX_ACQUISITIONS = {
#     "2014": ["2014-06-13", "2014-06-18", "2014-06-24", "2014-06-29"],
#     "2016": ["2016-07-16", "2016-07-27", "2016-08-13", "2016-08-24", "2016-09-04"],
#     "2017": ["2017-06-05", "2017-07-19", "2017-08-16"],
#     "2024": ["2024-06-12", "2024-06-17", "2024-07-26"],
#     "2025": ["2025-08-15", "2025-08-20", "2025-08-25", "2025-08-26",
#              "2025-09-05", "2025-09-11", "2025-09-16", "2025-09-27"],
# }
# ALL_YEARS_PERIOD_KEYS = ["2014", "2016-2017", "2024-2025"]
# DHDT_PAIRS = [("2016-2017", "2014"), ("2024-2025", "2014"), ("2024-2025", "2016-2017")]
# COMBINED_BIAS_YLIM = {"CryoSat-2": (-75, 75), "ICESat-2": (-20, 20)}
# HYPS_N_YLIM_ALL_YEARS = (0, 7250)
# HYPS_N_YLIM_PANEL = {("ICESat-2", "2024-2025"): (0, 750)}
# HYPS_N_YLIM = {"m": (0, 2500), "m a⁻¹": (0, 1500)}
# SCATTER_ELEV_LIM = (0.0, 2000.0)
# COMBINED_IS2_EPOCHS = ["2024-2025"]
# SAME_CELL_PERIODS = ["2024-2025"]


def ensure_output_dirs():
    for d in (OUTPUT_DIR, CS2_OUTPUT_DIR, IS2_OUTPUT_DIR, CS2_DHDT_OUTPUT_DIR, COMBINED_OUTPUT_DIR):
        os.makedirs(d, exist_ok=True)
