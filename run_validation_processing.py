"""
Cross-sensor validation of the TanDEM-X products (settings: validation/config.py).
Requires the TanDEM-X products of run_tandemx_processing.py.

1. CryoSat-2 vs. TanDEM-X surface elevation (per epoch and all epochs pooled)
2. ICESat-2 vs. TanDEM-X surface elevation
3. CryoSat-2 vs. TanDEM-X elevation-change rates
"""
from validation import config as cfg
from validation.common import save_state
from validation.cryosat2 import run_cs2_pipeline
from validation.cryosat2_dhdt import run_cs2_dhdt_pipeline
from validation.icesat2 import run_is2_pipeline


def main():
    cfg.ensure_output_dirs()
    print("=" * 70 + "\nCryoSat-2 vs TanDEM-X\n" + "=" * 70)
    cs2 = run_cs2_pipeline()
    print("=" * 70 + "\nICESat-2 vs TanDEM-X\n" + "=" * 70)
    is2 = run_is2_pipeline()
    print("=" * 70 + "\nCryoSat-2 dh/dt vs TanDEM-X dh/dt\n" + "=" * 70)
    dhdt = run_cs2_dhdt_pipeline()
    save_state({"cs2": cs2.comparisons, "is2": is2, "dhdt": dhdt})


if __name__ == "__main__":
    main()
