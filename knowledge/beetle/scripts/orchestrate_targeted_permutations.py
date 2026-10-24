"""
Orchestrator for Targeted Dual-Wavelength Permutations & Plume Comparison Suite:
Executes the remaining permutations sequentially in isolated subprocesses:
1. 10600 nm @ 40 W without plume (instructive comparison)
2. Targeted Dual Permutations:
   - (10600 nm, 1064 nm)
   - (10600 nm, 450 nm)
   - (1940 nm, 532 nm)
   - (1940 nm, 1064 nm)
   - (1940 nm, 450 nm)
   - (320 nm, 532 nm)
   - (320 nm, 1064 nm)
   - (320 nm, 450 nm)
Note: (10600 nm, 532 nm) is already complete and verified.
After each run finishes, README.md is dynamically re-synthesized.
"""

import os
import sys

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
import time
import subprocess

PYTHON_BIN = sys.executable
SCRIPT_PATH = os.path.join(REPO_ROOT, "knowledge/beetle/scripts/run_dual_permutation_suite.py")
README_GEN = os.path.join(REPO_ROOT, "knowledge/beetle/scripts/generate_comprehensive_readme.py")
RAW_DIR = os.path.join(REPO_ROOT, "knowledge/beetle/raw_arrays")

TARGET_RUNS = [
    # (name, args, expected_npz)
    ("10600nm_40w_no_plume", ["--wl1", "10600", "--p1", "40.0", "--no-plume"], "matrix_10600nm_40w_no_plume.npz"),
    ("dual_10600nm_1064nm", ["--wl1", "10600", "--wl2", "1064", "--p1", "40.0", "--p2", "20.0"], "matrix_dual_10600nm_1064nm.npz"),
    ("dual_10600nm_450nm",  ["--wl1", "10600", "--wl2", "450",  "--p1", "40.0", "--p2", "20.0"], "matrix_dual_10600nm_450nm.npz"),
    ("dual_1940nm_532nm",   ["--wl1", "1940",  "--wl2", "532",  "--p1", "40.0", "--p2", "20.0"], "matrix_dual_1940nm_532nm.npz"),
    ("dual_1940nm_1064nm",  ["--wl1", "1940",  "--wl2", "1064", "--p1", "40.0", "--p2", "20.0"], "matrix_dual_1940nm_1064nm.npz"),
    ("dual_1940nm_450nm",   ["--wl1", "1940",  "--wl2", "450",  "--p1", "40.0", "--p2", "20.0"], "matrix_dual_1940nm_450nm.npz"),
    ("dual_320nm_532nm",    ["--wl1", "320",   "--wl2", "532",  "--p1", "40.0", "--p2", "20.0"], "matrix_dual_320nm_532nm.npz"),
    ("dual_320nm_1064nm",   ["--wl1", "320",   "--wl2", "1064", "--p1", "40.0", "--p2", "20.0"], "matrix_dual_320nm_1064nm.npz"),
    ("dual_320nm_450nm",    ["--wl1", "320",   "--wl2", "450",  "--p1", "40.0", "--p2", "20.0"], "matrix_dual_320nm_450nm.npz"),
]

def main():
    print(f"\n{'='*75}")
    print(f"  TARGETED DUAL PERMUTATION & PLUME COMPARISON ORCHESTRATOR")
    print(f"  Total planned runs: {len(TARGET_RUNS)}")
    print(f"{'='*75}\n")

    for idx, (name, cmd_args, npz_name) in enumerate(TARGET_RUNS):
        npz_path = os.path.join(RAW_DIR, npz_name)
        if os.path.exists(npz_path):
            print(f"[{idx+1}/{len(TARGET_RUNS)}] {name} already exists at {npz_name}. Skipping.")
            continue

        print(f"\n[{idx+1}/{len(TARGET_RUNS)}] Launching {name} in isolated subprocess...")
        t0 = time.time()
        full_cmd = [PYTHON_BIN, SCRIPT_PATH] + cmd_args
        ret = subprocess.run(full_cmd, cwd=REPO_ROOT)
        dur = time.time() - t0

        if ret.returncode != 0:
            print(f"[ERROR] Run {name} failed with return code {ret.returncode} after {dur:.1f}s!")
            sys.exit(ret.returncode)

        print(f"[COMPLETED] {name} in {dur:.1f}s. Updating README.md...")
        subprocess.run([PYTHON_BIN, README_GEN], cwd=REPO_ROOT)

    print(f"\n{'='*75}")
    print(f"  ALL TARGETED SIMULATIONS & ANIMATIONS COMPLETE!")
    print(f"{'='*75}\n")
    subprocess.run([PYTHON_BIN, README_GEN], cwd=REPO_ROOT)

if __name__ == '__main__':
    main()
