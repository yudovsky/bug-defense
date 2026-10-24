#!/usr/bin/env bash
# Entry point for the beetle laser-ablation simulation.
# Usage: ./run.sh [command] [options]   (no command starts the UI; ./run.sh help for the list)
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV="$ROOT/venv"
PY="$VENV/bin/python"
SCRIPTS="$ROOT/knowledge/beetle/scripts"

usage() {
  cat <<'EOF'
Usage: ./run.sh [command] [options]

  ui [--server-only] [--port N]
                            LaserSim desktop app: set up the beetle and laser, run and watch
                              (default command; --server-only serves it at
                               http://127.0.0.1:5050/ for a browser)
  setup                     Create ./venv and install requirements
  test                      Run the unit tests
  sim [-w WL...] [-p W...]  Single-wavelength runs (default: 532 nm at 40 W)
                              e.g. ./run.sh sim -w 532 1470 -p 40 80
  dual [WL1 WL2 [P1 P2]]    Dual-pulse run: drill at WL1, then heat at WL2
                              (default: 1940 450 40 20)
  campaign                  Full single-wavelength campaign (21 runs + controls)
  permutations              Dual-pulse permutation campaign
  figures                   Rescore all runs, redraw the summary figures and the
                              minimum-energy-to-penetrate table
  gifs [--force]            Re-render the 4-panel animations from saved runs
  benchmarks [--quick]      Verification and validation benchmarks
  help                      Show this message

Runs need macOS on Apple Silicon (Metal GPU). Outputs go to
knowledge/beetle/raw_arrays/ and knowledge/beetle/plots/.
EOF
}

ensure_venv() {
  if [[ ! -x "$PY" ]]; then
    echo "No virtual environment found; running setup first."
    setup
  fi
}

setup() {
  python3 -m venv "$VENV"
  "$PY" -m pip install --upgrade pip
  "$PY" -m pip install -r "$ROOT/requirements.txt"
}

cd "$ROOT"
cmd="${1:-ui}"
shift || true

case "$cmd" in
  ui)
    ensure_venv
    "$PY" "$ROOT/app.py" "$@" ;;
  setup)
    setup ;;
  test)
    ensure_venv
    "$PY" -m pytest "$ROOT/tests" "$@" ;;
  sim)
    ensure_venv
    if [[ $# -eq 0 ]]; then set -- -w 532 -p 40; fi
    "$PY" "$SCRIPTS/run_highres_knowledge_suite.py" "$@" ;;
  dual)
    ensure_venv
    "$PY" "$SCRIPTS/run_dual_permutation_suite.py" \
      --wl1 "${1:-1940}" --wl2 "${2:-450}" --p1 "${3:-40}" --p2 "${4:-20}" ;;
  campaign)
    ensure_venv
    "$PY" "$SCRIPTS/orchestrate_comprehensive_campaign.py" "$@" ;;
  permutations)
    ensure_venv
    "$PY" "$SCRIPTS/orchestrate_targeted_permutations.py" "$@" ;;
  figures)
    ensure_venv
    "$PY" "$SCRIPTS/generate_comprehensive_readme.py"
    "$PY" "$SCRIPTS/plot_top10_metrics.py"
    "$PY" "$SCRIPTS/plot_top10_energy_volume.py"
    "$PY" "$SCRIPTS/render_450_vs_dual_boiling.py"
    "$PY" "$SCRIPTS/compute_min_energy.py" ;;
  gifs)
    ensure_venv
    "$PY" "$SCRIPTS/batch_regenerate_unmasked_gifs.py" "$@" ;;
  benchmarks)
    ensure_venv
    "$PY" "$ROOT/knowledge/simulation_lit_comparison/code/run_literature_benchmarks.py" "$@" ;;
  help|-h|--help)
    usage ;;
  *)
    echo "Unknown command: $cmd" >&2
    usage >&2
    exit 1 ;;
esac
