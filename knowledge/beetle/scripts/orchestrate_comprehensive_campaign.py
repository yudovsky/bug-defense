"""
Master Orchestrator for Comprehensive Campaign:
1. Phase 1: 6 Remaining No-Plume Control Simulations at 40 W:
   - 320 nm, 450 nm, 532 nm, 1064 nm, 1470 nm, 1940 nm (10600 nm is already done)
2. Phase 2: 21 Single-Wavelength Simulations with Air Gap Plume Diffusion Fix:
   - 7 Wavelengths (320, 450, 532, 1064, 1470, 1940, 10600 nm) x 3 Powers (20W, 40W, 80W)
3. Dynamic README.md Synthesis after every run completes.
All runs execute in isolated subprocesses with 100% memory reclamation.
"""

import os
import sys

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
import time
import subprocess

PYTHON_BIN = sys.executable
DUAL_SCRIPT = os.path.join(REPO_ROOT, "knowledge/beetle/scripts/run_dual_permutation_suite.py")
SINGLE_SCRIPT = os.path.join(REPO_ROOT, "knowledge/beetle/scripts/run_highres_knowledge_suite.py")
README_GEN = os.path.join(REPO_ROOT, "knowledge/beetle/scripts/generate_comprehensive_readme.py")
RAW_DIR = os.path.join(REPO_ROOT, "knowledge/beetle/raw_arrays")
PLOTS_DIR = os.path.join(REPO_ROOT, "knowledge/beetle/plots")

NO_PLUME_TARGETS = [320, 450, 532, 1064, 1470, 1940]
ALL_WAVELENGTHS = [320, 450, 532, 1064, 1470, 1940, 10600]
ALL_POWERS = [20.0, 40.0, 80.0]

def update_readme():
    try:
        subprocess.run([PYTHON_BIN, README_GEN], cwd=REPO_ROOT, check=True)
    except Exception as e:
        print(f"[README Warning] Could not regenerate README: {e}")

def main():
    total_no_plume = len(NO_PLUME_TARGETS)
    total_single = len(ALL_WAVELENGTHS) * len(ALL_POWERS)
    total_tasks = total_no_plume + total_single

    print(f"\n{'='*75}")
    print(f"  COMPREHENSIVE MASTER SIMULATION ORCHESTRATOR")
    print(f"  Phase 1: {total_no_plume} No-Plume Control Simulations at 40 W")
    print(f"  Phase 2: {total_single} Single-Wavelength Simulations (Corrected Air Gap)")
    print(f"  Total planned runs: {total_tasks}")
    print(f"{'='*75}\n")

    t_suite_start = time.time()
    current_task = 0

    # -------------------------------------------------------------
    # PHASE 1: NO-PLUME CONTROL SUITE AT 40 W
    # -------------------------------------------------------------
    print(f"\n>>> [STARTING PHASE 1] 6 No-Plume Control Runs at 40 W")
    for wl in NO_PLUME_TARGETS:
        current_task += 1
        name = f"{wl}nm_40w_no_plume"
        npz_file = os.path.join(RAW_DIR, f"matrix_{name}.npz")
        gif_file = os.path.join(PLOTS_DIR, f"beetle_{name}.gif")

        if os.path.exists(npz_file) and os.path.exists(gif_file) and os.path.getsize(gif_file) > 1000:
            print(f"[{current_task}/{total_tasks}] {name} already complete. Skipping.")
            continue

        print(f"\n[{current_task}/{total_tasks}] Launching {name} in isolated subprocess...")
        t0 = time.time()
        cmd = [PYTHON_BIN, DUAL_SCRIPT, "--wl1", str(wl), "--p1", "40.0", "--no-plume"]
        ret = subprocess.run(cmd, cwd=REPO_ROOT)
        dur = time.time() - t0

        if ret.returncode != 0:
            print(f"[ERROR] Run {name} failed with code {ret.returncode} after {dur:.1f}s!")
        else:
            print(f"[COMPLETED] {name} in {dur:.1f}s. Updating README.md...")
            update_readme()

    # -------------------------------------------------------------
    # PHASE 2: 21 SINGLE-WAVELENGTH RUNS (CORRECTED AIR GAP)
    # -------------------------------------------------------------
    print(f"\n>>> [STARTING PHASE 2] 21 Single-Wavelength Runs (Air Gap Plume Fix)")
    for wl in ALL_WAVELENGTHS:
        for p in ALL_POWERS:
            current_task += 1
            name = f"{wl}nm_{int(p)}w"
            npz_file = os.path.join(RAW_DIR, f"matrix_{name}.npz")
            gif_file = os.path.join(PLOTS_DIR, f"beetle_native_{name}.gif")

            print(f"\n[{current_task}/{total_tasks}] Launching {name} in isolated subprocess (forced recompute)...")
            t0 = time.time()
            cmd = [
                PYTHON_BIN, SINGLE_SCRIPT,
                "--single-run",
                "-w", str(wl),
                "-p", str(int(p))
            ]
            ret = subprocess.run(cmd, cwd=REPO_ROOT)
            dur = time.time() - t0

            if ret.returncode != 0:
                print(f"[ERROR] Run {name} failed with code {ret.returncode} after {dur:.1f}s!")
            else:
                print(f"[COMPLETED] {name} in {dur:.1f}s. Updating README.md...")
                update_readme()

    total_dur = time.time() - t_suite_start
    print(f"\n{'='*75}")
    print(f"  ALL {total_tasks} COMPREHENSIVE SIMULATIONS COMPLETE in {total_dur/3600.0:.2f} hours!")
    print(f"{'='*75}\n")
    update_readme()

if __name__ == '__main__':
    main()
