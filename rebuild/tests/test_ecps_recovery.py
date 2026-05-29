"""End-to-end recovery invariant against the real PE-native target estate.

This is the integration-level version of ``test_fit_never_increases_loss``: when
the sound operator is applied to the *already-calibrated* enhanced CPS, the
PE-native broad loss must be RECOVERED (~0.166), not blown up to ~0.544 the way
the old ``run_pe_native_l0_falsification.py`` path did.

It is skipped until the PE-native target loader + matrix builder are wired into
this clean package (next loop iteration). Wiring it here -- rather than importing
the old monolith's matrix code -- is deliberate: the bug may live in how the old
code builds the fitting vs scoring matrices, so we re-derive scoring cleanly and
verify it against the cached baseline (eCPS shipped ~0.166).
"""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.skip(
    reason="PE-native loader/matrix builder not wired into mp_rebuild yet "
    "(next iteration). Target: refit(eCPS) recovers ~0.166, not 0.544."
)


def test_refit_ecps_recovers_shipped_loss():
    # Pseudocode for the wired version:
    #   problem = load_pe_native_problem(ecps_h5, period=2024)      # clean re-derivation
    #   w0 = load_household_weights(ecps_h5, period=2024)           # shipped weights
    #   assert score(problem, w0) == pytest.approx(0.166, abs=0.01) # baseline sanity
    #   w1 = fit(problem, w0)
    #   assert score(problem, w1) <= score(problem, w0) + 1e-6      # never worse
    raise AssertionError("not wired yet")
