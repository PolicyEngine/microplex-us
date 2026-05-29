"""Honest, symmetric, held-out comparison: a Microplex candidate vs eCPS.

Both datasets are restricted to their common PE-native targets with identical
scaling, both are refit by the same sound optimizer, and the headline number is
held-out (fit on a train target split, scored on a disjoint holdout split).

This is the antidote to the old 0.094-vs-0.166 headline (which refit only the
candidate, at ~2.9x the records, on in-sample targets). It is still UNMATCHED N
(candidate ~120k vs eCPS ~41k) -- matched-N is the next step -- but it is
symmetric and held-out, which already removes two of the three confounds.

Run from the microplex-us main checkout:
    PYTHONPATH=/Users/maxghenis/PolicyEngine/policyengine-us-data \
      uv run --extra dev --extra policyengine python <this>
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from mp_rebuild.compare import result_to_dict, symmetric_holdout_compare  # noqa: E402
from mp_rebuild.pe_native import load_pe_native_raw  # noqa: E402

ECPS = (
    Path.home()
    / "PolicyEngine/policyengine-us-data/policyengine_us_data/storage/enhanced_cps_2024.h5"
)
CANDIDATE = (
    Path.home()
    / "CosilicoAI/microplex-us/artifacts/mp300k_compact_screen_20260528/mp_top120000_rescaled.h5"
)


def main() -> int:
    t0 = time.time()
    print("loading baseline eCPS ...", flush=True)
    raw_b = load_pe_native_raw(ECPS, 2024)
    print(
        f"  eCPS: {raw_b.raw_matrix.shape[0]} households, "
        f"{len(raw_b.names)} targets ({time.time() - t0:.0f}s)",
        flush=True,
    )
    print(f"loading candidate {CANDIDATE.name} ...", flush=True)
    raw_c = load_pe_native_raw(CANDIDATE, 2024)
    print(
        f"  candidate: {raw_c.raw_matrix.shape[0]} households, "
        f"{len(raw_c.names)} targets ({time.time() - t0:.0f}s)",
        flush=True,
    )

    res = symmetric_holdout_compare(
        raw_c, raw_b, holdout_fraction=0.2, seed=0, max_steps=2000
    )
    out = result_to_dict(res)
    out["seconds"] = round(time.time() - t0, 1)
    out["candidate_file"] = CANDIDATE.name
    print("COMPARISON_RESULT " + json.dumps(out))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
