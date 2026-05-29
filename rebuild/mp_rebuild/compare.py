"""Symmetric, held-out comparison of two datasets on a shared PE-native target set.

Fairness guarantees (the things the old harness lacked):
  * SHARED targets: both datasets are restricted to their common kept targets.
  * IDENTICAL scaling: PE-native scaling is computed once over the common
    targets and applied to both (it depends only on the target set, not the
    dataset).
  * SYMMETRIC fit: the identical optimizer is applied to both datasets.
  * HELD-OUT headline: weights are fit on a TRAIN target split and scored on a
    disjoint HOLDOUT split (same split indices for both), so a dataset can't
    "win" just by absorbing in-sample targets into extra weight capacity.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np

from .pe_native import RawPeNative, build_scaled_problem, pe_native_scaling
from .scoreboard import fit, score


def align_common(raw_a: RawPeNative, raw_b: RawPeNative):
    """Restrict both datasets to common targets with identical shared scaling."""
    # Names must be unique: alignment maps by name, so a duplicate would silently
    # double-count one row and drop the other.
    for label, names in (("candidate", raw_a.names), ("baseline", raw_b.names)):
        if len(set(names)) != len(names):
            raise ValueError(f"{label} target names must be unique for alignment")
    pos_b = {n: i for i, n in enumerate(raw_b.names)}
    common = [n for n in raw_a.names if n in pos_b]
    if not common:
        raise ValueError("no common targets between datasets")
    idx_a = [raw_a.names.index(n) for n in common]
    idx_b = [pos_b[n] for n in common]
    targets = raw_a.targets[idx_a]
    is_national = raw_a.is_national[idx_a]
    scaling = pe_native_scaling(targets, is_national)  # shared, identical for both
    prob_a = build_scaled_problem(
        raw_a.raw_matrix[:, idx_a], targets, is_national, common, scaling=scaling
    )
    prob_b = build_scaled_problem(
        raw_b.raw_matrix[:, idx_b], targets, is_national, common, scaling=scaling
    )
    target_value_max_disagreement = float(
        np.max(np.abs(targets - raw_b.targets[idx_b]))
    )
    return prob_a, prob_b, common, target_value_max_disagreement


def _holdout_indices(n: int, fraction: float, seed: int):
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n)
    n_hold = max(1, int(round(fraction * n)))
    return np.sort(perm[n_hold:]), np.sort(perm[:n_hold])  # (train, holdout)


@dataclass
class ComparisonResult:
    n_common_targets: int
    n_train_targets: int
    n_holdout_targets: int
    n_candidate_households: int
    n_baseline_households: int
    holdout_fraction: float
    seed: int
    # headline: fit on train, score on held-out
    candidate_holdout_loss: float
    baseline_holdout_loss: float
    # diagnostics
    candidate_train_loss: float
    baseline_train_loss: float
    candidate_full_refit_loss: float
    baseline_full_refit_loss: float
    candidate_initial_full_loss: float
    baseline_initial_full_loss: float
    target_value_max_disagreement: float

    def verdict(self) -> str:
        d = self.candidate_holdout_loss - self.baseline_holdout_loss
        rel = d / self.baseline_holdout_loss if self.baseline_holdout_loss else float("nan")
        if abs(rel) < 0.02:
            return f"PARITY on held-out (candidate within {rel:+.1%} of baseline)"
        if d < 0:
            return f"candidate BETTER on held-out by {-rel:.1%}"
        return f"candidate WORSE on held-out by {rel:.1%}"


def symmetric_holdout_compare(
    raw_candidate: RawPeNative,
    raw_baseline: RawPeNative,
    *,
    holdout_fraction: float = 0.2,
    seed: int = 0,
    **fit_kwargs,
) -> ComparisonResult:
    prob_c, prob_b, common, tdis = align_common(raw_candidate, raw_baseline)
    w0_c = raw_candidate.weights
    w0_b = raw_baseline.weights

    train_idx, hold_idx = _holdout_indices(len(common), holdout_fraction, seed)
    train_c, hold_c = prob_c.subset(train_idx), prob_c.subset(hold_idx)
    train_b, hold_b = prob_b.subset(train_idx), prob_b.subset(hold_idx)

    # Headline: fit on train, score on disjoint holdout (same split for both).
    wc = fit(train_c, w0_c, **fit_kwargs)
    wb = fit(train_b, w0_b, **fit_kwargs)

    # Diagnostic: in-sample full-target refit.
    wc_full = fit(prob_c, w0_c, **fit_kwargs)
    wb_full = fit(prob_b, w0_b, **fit_kwargs)

    return ComparisonResult(
        n_common_targets=len(common),
        n_train_targets=int(len(train_idx)),
        n_holdout_targets=int(len(hold_idx)),
        n_candidate_households=int(w0_c.shape[0]),
        n_baseline_households=int(w0_b.shape[0]),
        holdout_fraction=holdout_fraction,
        seed=seed,
        candidate_holdout_loss=score(hold_c, wc),
        baseline_holdout_loss=score(hold_b, wb),
        candidate_train_loss=score(train_c, wc),
        baseline_train_loss=score(train_b, wb),
        candidate_full_refit_loss=score(prob_c, wc_full),
        baseline_full_refit_loss=score(prob_b, wb_full),
        candidate_initial_full_loss=score(prob_c, w0_c),
        baseline_initial_full_loss=score(prob_b, w0_b),
        target_value_max_disagreement=tdis,
    )


def result_to_dict(result: ComparisonResult) -> dict:
    d = asdict(result)
    d["verdict"] = result.verdict()
    return d
