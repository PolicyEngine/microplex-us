"""Standalone driver for the eCPS recovery check.

Answers two questions and prints them as JSON:
  1. Does the clean PE-native loader reproduce eCPS's published broad loss (~0.166)?
  2. Does the sound optimizer fail to degrade it (old harness gave 0.544)?

Run from the microplex-us main checkout so the PolicyEngine deps resolve:
    MPR=...microplex-us-scoreboard-rebuild/rebuild
    uv run --extra dev --extra policyengine python "$MPR/run_recovery_check.py"
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))  # make mp_rebuild importable

from mp_rebuild.pe_native import load_pe_native_problem  # noqa: E402
from mp_rebuild.scoreboard import fit, score  # noqa: E402

ECPS = (
    Path.home()
    / "PolicyEngine/policyengine-us-data/policyengine_us_data/storage/enhanced_cps_2024.h5"
)


def main() -> int:
    t0 = time.time()
    pn = load_pe_native_problem(ECPS, period=2024)
    shipped = score(pn.problem, pn.weights)
    refit_w = fit(pn.problem, pn.weights, max_steps=2000)
    refit = score(pn.problem, refit_w)
    out = {
        "dataset": "enhanced_cps_2024",
        "period": 2024,
        "n_targets": pn.problem.n_targets,
        "n_households": pn.n_households,
        "shipped_loss": shipped,
        "refit_loss": refit,
        "non_increase_holds": bool(refit <= shipped + 1e-6),
        "reproduces_baseline_0166": bool(abs(shipped - 0.1664) < 0.03),
        "seconds": round(time.time() - t0, 1),
    }
    print("RECOVERY_RESULT " + json.dumps(out))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
