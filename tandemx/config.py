"""
Configuration of the TanDEM-X processing chain (mosaicking, dh/dt, void filling,
uncertainty, mass balance) and of the corresponding figures.

All paths are placeholders. Point DATA_DIR / OUTPUT_DIR (or the individual
paths below) to your own data before running.
"""
import os

os.environ.setdefault("GDAL_PAM_ENABLED", "NO")   # no .aux.xml files next to the input rasters

# =============================================================================
# 1. Paths
# =============================================================================
DATA_DIR = "path/to/data"            # root of all input data
OUTPUT_DIR = "path/to/output"        # root of all outputs

OUT_DIR = os.path.join(OUTPUT_DIR, "tandemx")         # rasters, tables, cached state
PLOT_DIR = os.path.join(OUT_DIR, "plots")              # all figures
CENTERLINE_PLOT_DIR = os.path.join(PLOT_DIR, "centerline_analysis")   # centerline figures
CENTERLINE_DATA_DIR = os.path.join(OUT_DIR, "centerline_data")       # centerline profile tables
STATE_FILE = os.path.join(OUT_DIR, "processing_state.pkl")  # hand-over processing -> plotting

# -----------------------------------------------------------------------------
# Study setup: acquisition years, epochs and outlines
# -----------------------------------------------------------------------------
# Years with TanDEM-X acquisitions (strings). Each year gets one DEM mosaic.
TDX_YEARS = []

# Years that are combined into one epoch, e.g. two consecutive years that
# together give full coverage. The merged epoch is named "<first>-<last>" and
# replaces its individual years in all dh/dt periods.
MERGED_EPOCHS = []

# Folder with the DEM strips of a year (files DEM*.tif, acquisition date
# YYYY-MM-DD or YYYYMMDD in the file name). {year} is replaced by each year.
DEM_DIR_TEMPLATE = os.path.join(DATA_DIR, "tandemx", "{year}")

# Glacier/catchment outline of a year (polygon field DN/Id = 1 marks the area
# to keep). The union of all outlines defines the processing grid and study
# mask; each epoch is clipped to the outline of its terminus position.
OUTLINE_TEMPLATE = os.path.join(DATA_DIR, "outlines", "outline_{year}.shp")

# Outline used for an epoch. Epochs not listed use the outline of their last
# year (or the closest year with an outline).
MASK_ASSIGNMENT = {}

# Glacier inventory (catchment outlines) used for areas and mass balance,
# per epoch as {epoch: {"shp": file name in GLACIER_MASK_DIR}}. Epochs not
# listed use DEFAULT_GLACIER_SHP.
GLACIER_MASK_DIR = os.path.join(DATA_DIR, "inventory")
GLACIER_INVENTORY_MAP = {}
DEFAULT_GLACIER_SHP = os.path.join(GLACIER_MASK_DIR, "glacier_inventory.shp")

REGION_LABEL = "Study region"        # region name written to the summary tables

# --- Example: settings of Warnstedt et al. (Wordie Ice Shelf tributaries) ---
# TDX_YEARS = ["2014", "2016", "2017", "2024", "2025"]
# MERGED_EPOCHS = [["2016", "2017"], ["2024", "2025"]]
# DEM_DIR_TEMPLATE = os.path.join(DATA_DIR, "tandemx", "{year}", "DEM_VAL", "all")
# GLACIER_INVENTORY_MAP = {
#     "2014": {"shp": "mask_glacier_2014.shp"},
#     "2016-2017": {"shp": "mask_glacier_2017.shp"},
#     "2024-2025": {"shp": "mask_glacier_2025.shp"},
# }
# DEFAULT_GLACIER_SHP = os.path.join(GLACIER_MASK_DIR, "mask_glacier_2014.shp")
# REGION_LABEL = "Wordie"

# Derived from the settings above (normally no need to edit).
YEAR_DIRS = {y: DEM_DIR_TEMPLATE.format(year=y) for y in TDX_YEARS}
YEAR_MASKS = {y: OUTLINE_TEMPLATE.format(year=y) for y in TDX_YEARS}
MASK_PATH = OUTLINE_TEMPLATE.format(year=TDX_YEARS[0]) if TDX_YEARS else ""  # outline for map extents

# -----------------------------------------------------------------------------
# Further inputs
# -----------------------------------------------------------------------------
CRS = "EPSG:3031"                    # projected CRS of all inputs and outputs
CENTERLINE_DIR = os.path.join(DATA_DIR, "centerlines")      # one Centerline_<Name>_NN.shp per glacier
ROCK_OUTCROP_PATH = os.path.join(DATA_DIR, "rock_outcrop/rock_outcrop.shp")  # stable terrain polygons

# Reference DEM (the DEM the TanDEM-X strips were co-registered to) and the
# four products derived from it (elevation, 50 m elevation bins, slope, 5 deg
# slope bins). With REF_PRODUCTS_BUILD = True they are built into
# OUT_DIR/reference_dem from REF_DEM_SOURCE, otherwise read from REF_PRODUCTS_DIR.
REF_DEM_SOURCE = os.path.join(DATA_DIR, "reference_dem/reference_dem_egm2008.tif")
REF_AOI_SHP = MASK_PATH              # area of interest for the reference products
REF_AOI_BUFFER_M = 10_000.0          # buffer (m) around REF_AOI_SHP
REF_PRODUCTS_DIR = os.path.join(DATA_DIR, "reference_dem")
REF_PRODUCTS_BUILD = False           # build the products instead of reading them
REF_PRODUCTS_REBUILD = False         # overwrite existing products when building
REF_DEM_PATH = os.path.join(REF_PRODUCTS_DIR, "ref_dem_aoi.tif")
BIN_ELEV_PATH = os.path.join(REF_PRODUCTS_DIR, "ref_elev_binned_50m.tif")
SLOPE_PATH = os.path.join(REF_PRODUCTS_DIR, "ref_slope.tif")
BIN_SLOPE_PATH = os.path.join(REF_PRODUCTS_DIR, "ref_slope_binned_5deg.tif")
REF_ELEV_BIN_M = 50.0                # elevation-bin width of BIN_ELEV_PATH (m)
REF_SLOPE_BIN_DEG = 5.0              # slope-bin width of BIN_SLOPE_PATH (deg)

# Stable-terrain inputs.
STABLE_TERRAIN_DIR = os.path.join(DATA_DIR, "stable_terrain")   # stable_<year>.tif (only for "rasters")
STB_SUBDIR = "stb"                   # stable-ground strips: <year folder>/../stb/DEM_STB_*.tif
STB_PREFIX = "DEM_STB_"              # file prefix of the stable-ground strips
STB_FOOTPRINT_DIR = os.path.join(DATA_DIR, "strip_footprints/{year}")  # <strip>_footprint.shp per strip

MASK_RASTER_CACHE_UNION = os.path.join(OUT_DIR, "mask_keep_union_allyears_30m.tif")  # cached study mask

# =============================================================================
# 2. Grid and DEM mosaicking
# =============================================================================
RESOLUTION = 30                      # output grid resolution (m)
ELEV_MIN = -50                       # DEM values outside [ELEV_MIN, ELEV_MAX] (m) are discarded
ELEV_MAX = 3200
NODATA_EDGE_EROSION_PX = 1           # pixels removed along strip/void edges before mosaicking
DERAMP_ORDER = 2                     # strip-to-mosaic ramp: 1 = plane, any other value = quadratic
DERAMP_SAMPLE = 20000                # max. overlap pixels sampled per ramp fit
DERAMP_MIN_PTS = 500                 # min. overlap pixels for a ramp fit, else median shift
                                     # (10**12 = never de-ramp, median shift only)
MAD_THRESHOLD = 3.0                  # outlier screen (x MAD) of the ramp-fit sample
DERAMP_MAX_ITER = 5                  # max. iterations of the ramp fit per strip
DERAMP_CONVERGENCE_M = 0.05          # stop iterating once the RMS ramp increment < this (m)
MEDIAN_SHIFT_MIN_PTS = 50            # min. overlap pixels for a median vertical shift
MERGE_MODE = "feather"               # multi-year merge: "feather" (weighted overlaps) or "fill_gaps"
FEATHER_WIDTH_PX = 50                # feathering distance from strip edges (px)
MOSAIC_ORDER = "size"                # order of the strips: "size" (most valid pixels first) or "overlap"
                                     # (reference = largest strip, then largest overlap with the mosaic)

# =============================================================================
# 3. dh/dt, void filling and inventory areas
# =============================================================================
DHDT_MAX_ABS_RATE = 30.0             # |dh/dt| above this (m a-1) is rejected (set to NaN, not clipped)
SEAM_RING_REMOVE_PX = 1              # px removed next to voids after the 3x3 median filter (0 = off)
MIN_BIN_PIXELS = 50                  # min. valid pixels for a trusted elevation-bin mean
VOID_FILL_EXTRAPOLATION = "end_bin"  # bins outside the observed elevation range: "end_bin" (mean of the
                                     # lowest/highest valid bin) or "polynomial" (cubic extrapolated)
VOID_FILL_SPATIAL_REFINEMENT = False  # extra spatial anomaly interpolation near observed pixels
VOID_FILL_BOUNDARY_SMOOTH = False    # 3x3 median along the observed/void boundary
VOID_FILL_SEARCH_PX_MODE = "fixed"   # search radius of the spatial refinement: "fixed" or "adaptive"
VOID_FILL_SEARCH_PX_FIXED = 100      # fixed search radius (px)
VOID_FILL_SEARCH_PX_MIN = 3          # bounds of the adaptive radius (px)
VOID_FILL_SEARCH_PX_MAX = 400

# Analysis domain for every period: "C" = union of start and end inventory,
# "B" = end-of-period inventory trimmed to the end-year terminus outline.
AREA_MASK_OPTION = "C"
OPTION_B_MIN_RETAINED_FRACTION = 0.10  # option B falls back to the untrimmed outline below this

# =============================================================================
# 4. SAR penetration (treated as uncertainty term, Braun et al. 2019)
# =============================================================================
SAR_PEN_BIAS = 2.0                   # max. penetration difference of the ramp (m), X- vs X-band
SAR_PEN_TYPE = "ramp"                # "ramp" (0 m at onset, linear to SAR_PEN_BIAS) or "constant"
SAR_PEN_ONSET_MODE = "auto"          # onset elevation z0: "auto" (inter-period consistency) or a number (m)
SAR_PEN_ONSET_FALLBACK_M = 1800.0    # z0 (m) if "auto" finds no significant onset; region specific
                                     # (1800 m for the Antarctic Peninsula, Seehaus et al. 2023)
SAR_PEN_ONSET_FIT_MIN_ELEV = 500.0   # bins below are excluded from the onset hinge fit (m)
SAR_PEN_RAMP_WIDTH_M = "catchment_max"  # ramp length above z0: "catchment_max" (ramp reaches SAR_PEN_BIAS
                                     # at the highest catchment elevation of the reference DEM),
                                     # "auto" (from the fitted slope) or metres
SAR_PEN_RAMP_WIDTH_FALLBACK_M = 600.0  # ramp length if the chosen option is not possible (m)
SAR_PEN_RAMP_WIDTH_MIN_M = 300.0     # bounds of the "auto" ramp length (m); None = no bound
SAR_PEN_RAMP_WIDTH_MAX_M = None
SAR_PEN_START_ELEV = 1800            # ramp start (m); overwritten by the onset estimate at run time
SAR_PEN_END_ELEV = 2400              # ramp end (m); overwritten at run time
SAR_PEN_REQUIRE_BREAK = False        # True: apply ramp only if a break in slope is detected
SAR_PEN_APPLY_BIAS_CORRECTION = False  # True: subtract the ramp from dh/dt, False: uncertainty only
SAR_PEN_RESIDUAL_UNC_FRACTION = 1.0  # fraction of the subtracted bias kept as uncertainty
SAR_PEN_COMBINE_MODE = "quadrature"  # combine penetration and measurement error: "quadrature"/"linear"
SAR_PEN_BREAK_REF_MIN_ELEV = 500     # lower edge of the reference fit of the break test (m)
SAR_PEN_BREAK_SIGMA_THRESH = 2.0     # residual threshold (robust sigma) of the break test
SAR_PEN_BREAK_MIN_CONSECUTIVE = 3    # consecutive anomalous bins needed for a break
SAR_PEN_DETECT_MIN_BIN_PIXELS = 30   # min. pixels per bin in the break test
SLOPE_BIN_WIDTH_DEG = 5.0            # slope classes of the slope-stratified break test (deg)
SLOPE_BIN_MAX_DEG = 45.0
SLOPE_STEEP_THRESHOLD_DEG = 30.0     # classes at/above count as "steep"

# =============================================================================
# 5. Stable terrain and spatial correlation (Hugonnet et al. 2022)
# =============================================================================
STABLE_TERRAIN_SOURCE = "dem_rock"   # source of the stable-terrain dh/dt:
                                     # "dem_rock"   rock pixels (ROCK_OUTCROP_PATH) of the glacier dh/dt
                                     #              mosaic itself, incl. nunataks inside the catchment
                                     # "stb_mosaic" separate stable-ground strip mosaics (DEM_STB_*)
                                     # "mosaic"     off-glacier rock pixels of the dh/dt mosaic
                                     # "rasters"    yearly stable-ground DEMs in STABLE_TERRAIN_DIR
STABLE_SIGMA_BUFFER_M = 0.0          # sigma_stable from rock inside the catchment plus this buffer (m)
STB_ALIGN_TO_DEM = True              # shift each stable strip by its median offset to its DEM strip
STB_ALIGN_MIN_PX = 1000              # min. shared pixels for that alignment
STABLE_BIAS_APPLY_CORRECTION = False  # subtract the stable-terrain residual from dh/dt
STABLE_DETREND = True                # remove a planar tilt from the stable residuals
STABLE_SYSTEMATIC_IN_BUDGET = True   # add the tilt at the glacier to the uncertainty budget
STABLE_TILT_SLOPE_MAX = "glacier_p95"  # rock used for the tilt fit: slope limit ("glacier_p95", deg, None)
STABLE_TILT_BOOT_N = 200             # block-bootstrap repetitions of the tilt fit
STABLE_TILT_BLOCK_M = 10_000.0       # bootstrap block size (m)
STANDARDIZE_VARIOGRAM = True         # variogram of dh/sigma_dh(slope) instead of raw dh

# =============================================================================
# 6. Mass balance
# =============================================================================
DENSITY_A = 850.0                    # volume-to-mass conversion factor (kg m-3), Huss (2013)
DENSITY_B = 900.0                    # alternative factor (pure ice) (kg m-3)
DENSITY_UNCERTAINTY_KGM3 = 60.0      # uncertainty of the conversion factor (kg m-3)
RHO_WATER_KGM3 = 1000.0              # water density (kg m-3)

# Unit conversions
KM2_TO_M2 = 1.0e6
M2_TO_KM2 = 1.0 / KM2_TO_M2
M3_TO_KM3 = 1.0e-9
KG_TO_GT = 1.0e-12
M_TO_KM = 1.0e-3

# =============================================================================
# 7. Figures
# =============================================================================
DPI = 300                            # resolution of all saved figures
SUMMARY_EPOCHS = None                # three epochs of the summary figures; None = first, middle, last
HYPS_SUMMARY_XLIM = None             # elevation axis of the hypsometry summary figure (m); None = auto
CENTERLINE_SAMPLING_DISTANCE = 100   # spacing of the centerline samples (m)
CENTERLINE_MAX_DISTANCE = 15000      # profile length from the terminus (m)
TERMINUS_LABEL_YEARS = None          # outline years marked as terminus lines in the profiles; None = all
TERMINUS_COLORS = {}                 # {year: colour} of the terminus lines; missing years get a grey ramp

# --- Example: figure settings of Warnstedt et al. ---
# HYPS_SUMMARY_XLIM = (-100, 2700)
# TERMINUS_LABEL_YEARS = ["2014", "2017", "2025"]
# TERMINUS_COLORS = {"2014": "skyblue", "2016": "darkgray", "2017": "dimgray",
#                    "2024": "orange", "2025": "black"}


def ensure_output_dirs():
    """Create all output folders."""
    for d in (OUT_DIR, PLOT_DIR, CENTERLINE_PLOT_DIR, CENTERLINE_DATA_DIR):
        os.makedirs(d, exist_ok=True)
