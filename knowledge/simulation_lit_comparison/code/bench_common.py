"""
Shared helpers for the literature / analytic benchmark suite.
"""

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

SUITE_DIR = Path(__file__).resolve().parents[1]
IMAGES_DIR = SUITE_DIR / "images"
RAW_DIR = SUITE_DIR / "raw_output"


def make_result(
    bench_id: str,
    tier: str,
    title: str,
    reference: str,
    observable: str,
    units: str,
    model: float,
    reference_value: float,
    tolerance_abs: float,
    tolerance_basis: str,
    notes: str = "",
    series: dict = None,
) -> dict:
    """
    One benchmark comparison. PASS iff |model - reference| <= tolerance_abs.
    tolerance_basis documents where the tolerance comes from (experimental SD, MC
    statistical error, discretisation estimate, ...). Model values are never clipped.
    """
    abs_err = abs(model - reference_value)
    rel_err = abs_err / abs(reference_value) * 100.0 if reference_value != 0 else float("nan")
    return {
        "id": bench_id,
        "tier": tier,
        "title": title,
        "reference": reference,
        "observable": observable,
        "units": units,
        "model": float(model),
        "reference_value": float(reference_value),
        "abs_error": float(abs_err),
        "rel_error_pct": float(rel_err),
        "tolerance_abs": float(tolerance_abs),
        "tolerance_basis": tolerance_basis,
        "pass": bool(abs_err <= tolerance_abs),
        "notes": notes,
        "series": series or {},
    }
