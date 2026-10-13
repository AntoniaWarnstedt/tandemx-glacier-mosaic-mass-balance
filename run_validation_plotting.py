"""
Combined cross-sensor validation figures (settings: validation/config.py).
Requires a previous run of run_validation_processing.py.
"""
from validation import config as cfg
from validation.common import load_state
from validation.plot_combined import plot_all_combined_types


def main():
    cfg.ensure_output_dirs()
    plot_all_combined_types(load_state(), cfg.COMBINED_OUTPUT_DIR)


if __name__ == "__main__":
    main()
