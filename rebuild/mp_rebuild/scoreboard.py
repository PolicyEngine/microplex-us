"""The scoreboard: one loss, used for both fitting and scoring.

A ``CalibrationProblem`` is a linear estimation problem. Given a constraint
matrix ``M`` of shape (n_targets, n_records) and a target vector ``t``, the
estimate for a household weight vector ``w`` is ``M @ w``. The loss is the
mean squared *relative* error against ``t``.

Crucially, the SAME ``CalibrationProblem.loss`` is what ``fit`` minimises and
what ``score`` reports. There is no separate "fitting matrix" and "scoring
matrix" that can silently diverge -- that divergence is exactly what made the
old harness drive a well-calibrated eCPS from 0.166 to 0.544 while reporting a
tiny in-loop error.

``fit`` uses gradient descent with Armijo backtracking, which only ever accepts
a step that decreases the loss. Therefore:

    score(problem, fit(problem, w0)) <= score(problem, w0)   (always)

i.e. the operator can never make a dataset worse. This is the property the
recovery invariant test depends on, and the property a trustworthy calibrator
must have.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

ArrayLike = np.ndarray


@dataclass
class CalibrationProblem:
    """A linear calibration problem with a single source-of-truth loss.

    Parameters
    ----------
    matrix:
        Constraint matrix of shape (n_targets, n_records). Row ``i`` dotted with
        the weight vector produces the estimate for target ``i``.
    target:
        Target vector of shape (n_targets,).
    normalization:
        Per-target scale used to turn absolute errors into relative errors.
        Defaults to ``abs(target)`` (with zeros mapped to 1 to avoid division
        by zero), matching the PE-native "relative error" convention.
    names:
        Optional per-target names, used for family/holdout bookkeeping.
    """

    matrix: ArrayLike
    target: ArrayLike
    normalization: ArrayLike | None = None
    names: tuple[str, ...] | None = None

    def __post_init__(self) -> None:
        self.matrix = np.asarray(self.matrix, dtype=np.float64)
        self.target = np.asarray(self.target, dtype=np.float64)
        if self.matrix.ndim != 2:
            raise ValueError("matrix must be 2-D (n_targets, n_records)")
        if self.matrix.shape[0] != self.target.shape[0]:
            raise ValueError("matrix rows must match target length")
        if self.normalization is None:
            denom = np.abs(self.target)
            self.normalization = np.where(denom > 0, denom, 1.0)
        else:
            self.normalization = np.asarray(self.normalization, dtype=np.float64)
            if self.normalization.shape != self.target.shape:
                raise ValueError("normalization must match target length")
        if self.names is not None and len(self.names) != self.target.shape[0]:
            raise ValueError("names must match target length")

    @property
    def n_targets(self) -> int:
        return int(self.matrix.shape[0])

    @property
    def n_records(self) -> int:
        return int(self.matrix.shape[1])

    def estimate(self, weights: ArrayLike) -> np.ndarray:
        return self.matrix @ np.asarray(weights, dtype=np.float64)

    def relative_errors(self, weights: ArrayLike) -> np.ndarray:
        return (self.estimate(weights) - self.target) / self.normalization

    def loss(self, weights: ArrayLike) -> float:
        """THE loss: mean squared relative error. Fitting and scoring agree."""
        err = self.relative_errors(weights)
        return float(np.mean(err**2))

    def gradient(self, weights: ArrayLike) -> np.ndarray:
        """Analytic gradient of ``loss`` with respect to ``weights``."""
        err = self.relative_errors(weights)  # (n_targets,)
        scaled = err / self.normalization  # d loss / d estimate, up to 2/T
        return (2.0 / self.n_targets) * (self.matrix.T @ scaled)

    def subset(self, target_indices: ArrayLike) -> "CalibrationProblem":
        """Return the problem restricted to a subset of targets (same weights)."""
        idx = np.asarray(target_indices, dtype=int)
        names = None if self.names is None else tuple(self.names[i] for i in idx)
        return CalibrationProblem(
            matrix=self.matrix[idx, :],
            target=self.target[idx],
            normalization=self.normalization[idx],
            names=names,
        )


def score(problem: CalibrationProblem, weights: ArrayLike) -> float:
    """Score a weight vector. Scoring == the problem's own loss, by definition."""
    return problem.loss(weights)


def fit(
    problem: CalibrationProblem,
    init_weights: ArrayLike,
    *,
    max_steps: int = 2000,
    nonneg: bool = True,
    grad_tol: float = 1e-12,
    init_step: float = 1.0,
    armijo_c1: float = 1e-4,
    backtrack: float = 0.5,
    min_step: float = 1e-20,
) -> np.ndarray:
    """Minimise ``problem.loss`` from ``init_weights`` with monotone non-increase.

    Gradient descent with Armijo backtracking line search. A step is accepted
    only if it satisfies the sufficient-decrease condition, so the loss is
    non-increasing across the whole run. Consequently the returned weights never
    score worse than ``init_weights``. If ``nonneg`` is set, weights are
    projected to be non-negative (household weights can't be negative); the
    projected step is still only accepted on a decrease, so the guarantee holds.
    """
    w = np.array(init_weights, dtype=np.float64)
    if nonneg:
        w = np.maximum(w, 0.0)
    loss = problem.loss(w)

    for _ in range(max_steps):
        grad = problem.gradient(w)
        gnorm_sq = float(grad @ grad)
        if gnorm_sq < grad_tol:
            break
        step = init_step
        accepted = False
        while step > min_step:
            w_new = w - step * grad
            if nonneg:
                w_new = np.maximum(w_new, 0.0)
            new_loss = problem.loss(w_new)
            # Armijo sufficient decrease. For the projected case this is a
            # conservative accept rule; it can only ever reduce the loss.
            if new_loss <= loss - armijo_c1 * step * gnorm_sq:
                w, loss = w_new, new_loss
                accepted = True
                break
            step *= backtrack
        if not accepted:
            break

    return w


def split_targets(
    problem: CalibrationProblem,
    holdout_fraction: float,
    seed: int = 0,
) -> tuple[CalibrationProblem, CalibrationProblem]:
    """Partition targets into (train, holdout) problems over the same weights.

    Fitting on ``train`` and scoring on ``holdout`` is the headline overfitting
    check: a dataset that only "wins" by absorbing in-sample targets into a huge
    weight vector will not generalise to held-out targets.
    """
    if not 0.0 < holdout_fraction < 1.0:
        raise ValueError("holdout_fraction must be in (0, 1)")
    rng = np.random.default_rng(seed)
    n = problem.n_targets
    perm = rng.permutation(n)
    n_holdout = max(1, int(round(holdout_fraction * n)))
    holdout_idx = np.sort(perm[:n_holdout])
    train_idx = np.sort(perm[n_holdout:])
    return problem.subset(train_idx), problem.subset(holdout_idx)


def compare(
    candidate: CalibrationProblem,
    baseline: CalibrationProblem,
    candidate_init: ArrayLike,
    baseline_init: ArrayLike,
    **fit_kwargs,
) -> dict[str, float]:
    """Apply the IDENTICAL fit to both datasets and report scores.

    The comparison is symmetric by construction: there is no way to refit only
    one side. (The old harness's ``--score-candidate-only`` path is structurally
    impossible here.)
    """
    cand_w = fit(candidate, candidate_init, **fit_kwargs)
    base_w = fit(baseline, baseline_init, **fit_kwargs)
    return {
        "candidate_loss": score(candidate, cand_w),
        "baseline_loss": score(baseline, base_w),
        "candidate_loss_initial": score(candidate, candidate_init),
        "baseline_loss_initial": score(baseline, baseline_init),
    }
