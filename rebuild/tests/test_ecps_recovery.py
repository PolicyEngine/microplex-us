"""End-to-end recovery invariant against the real PE-native target estate.

The integration-level version of ``test_fit_never_increases_loss``: applying the
sound operator to the *already-calibrated* enhanced CPS must RECOVER its
PE-native broad loss (~0.166), not blow it up to ~0.544 the way the old
``run_pe_native_l0_falsification.py`` path did.

Slow (builds the PE-native matrix via PolicyEngine, a few minutes), so it is
gated behind ``MPR_RUN_SLOW=1`` to keep the fast invariant suite fast.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from mp_rebuild import fit, score

ECPS_H5 = (
    Path.home()
    / "PolicyEngine/policyengine-us-data/policyengine_us_data/storage/enhanced_cps_2024.h5"
)
RUN_SLOW = os.environ.get("MPR_RUN_SLOW") == "1"


@pytest.mark.skipif(
    not RUN_SLOW, reason="set MPR_RUN_SLOW=1 (slow: builds the PE-native matrix)"
)
@pytest.mark.skipif(not ECPS_H5.exists(), reason="enhanced_cps_2024.h5 not present")
def test_refit_ecps_recovers_shipped_loss():
    from mp_rebuild.pe_native import load_pe_native_problem

    pn = load_pe_native_problem(ECPS_H5, period=2024)

    # 1) Baseline sanity: the clean loader reproduces the published eCPS
    #    PE-native broad loss (cached baseline reports ~0.1664).
    shipped_loss = score(pn.problem, pn.weights)
    assert shipped_loss == pytest.approx(0.1664, abs=0.03), (
        f"clean loader should reproduce eCPS ~0.166, got {shipped_loss}"
    )

    # 2) Recovery / non-increase: the sound operator must not degrade a
    #    calibrated dataset. The old harness drove this to ~0.544.
    refit = fit(pn.problem, pn.weights, max_steps=2000)
    refit_loss = score(pn.problem, refit)
    assert refit_loss <= shipped_loss + 1e-6, (
        f"refit must not worsen eCPS: shipped={shipped_loss} refit={refit_loss}"
    )
