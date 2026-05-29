"""Clean PE-native scoring loader.

Builds a :class:`~mp_rebuild.scoreboard.CalibrationProblem` from the canonical
``policyengine_us_data.utils.loss.build_loss_matrix`` for a dataset H5, using the
SAME target filtering and national/state-balanced scaling as the incumbent
PE-native broad loss. This reproduces the published eCPS baseline (~0.166), so
candidate and baseline are scored on an identical metric.

We reuse only *data/definition* helpers from the existing package -- the target
list (``_ENHANCED_CPS_BAD_TARGETS``) and the family classifier -- not the
optimizer. The optimizer is :func:`mp_rebuild.scoreboard.fit`, which, unlike the
old ``run_pe_native_l0_falsification.py`` path, cannot increase the scored loss.

The PE-native ``_objective`` is ``||M_scaled @ w - t_scaled||^2`` (a *sum* of
squared scaled residuals, with per-target weighting baked into the scaling), so
we wrap it as a ``reduction="sum"`` problem with unit normalization.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .scoreboard import CalibrationProblem


@dataclass
class PeNativeProblem:
    """A loaded PE-native problem plus the dataset's shipped weights."""

    problem: CalibrationProblem
    weights: np.ndarray  # shipped household weights from the dataset
    target_names: list[str]
    families: list[str]
    is_national: np.ndarray
    n_households: int


def _local_dataset(dataset_path: str, period: int):
    """Mirror the existing ``dataset_from_path`` helper without importing it."""
    from policyengine_core.data import Dataset

    class LocalDataset(Dataset):
        name = "mpr_eval_dataset"
        label = "mpr_eval_dataset"
        file_path = dataset_path
        data_format = Dataset.TIME_PERIOD_ARRAYS
        time_period = period

    return LocalDataset


def load_pe_native_problem(
    dataset_path: str | Path,
    period: int = 2024,
    bad_targets: tuple[str, ...] | None = None,
) -> PeNativeProblem:
    """Load the PE-native broad-loss problem for ``dataset_path``.

    Single-batch (fine for eCPS ~41k and the matched-N candidates we compare;
    >~50k-household candidates will need batching, tracked in PLAN.md).
    """
    from policyengine_us import Microsimulation
    from policyengine_us_data.utils.loss import build_loss_matrix

    from microplex_us.pipelines.pe_native_scores import (
        _classify_pe_native_target_family,
    )

    if bad_targets is None:
        from microplex_us.pipelines.pe_native_scores import (
            _ENHANCED_CPS_BAD_TARGETS,
        )

        bad_targets = tuple(_ENHANCED_CPS_BAD_TARGETS)

    dataset_path = str(dataset_path)
    dataset_cls = _local_dataset(dataset_path, period)

    loss_matrix, targets_array = build_loss_matrix(dataset_cls, period)
    target_names = np.asarray(loss_matrix.columns)
    targets_array = np.asarray(targets_array, dtype=np.float64)

    # Exact replica of the incumbent PE-native filtering + scaling.
    zero_mask = np.isclose(targets_array, 0.0, atol=0.1)
    bad_mask = np.isin(target_names, np.asarray(bad_targets))
    keep = ~(zero_mask | bad_mask)

    filtered = loss_matrix.loc[:, keep]
    filt_targets = targets_array[keep]
    is_national = np.asarray(
        filtered.columns.str.startswith("nation/"), dtype=bool
    )
    n_nat = int(is_national.sum())
    n_state = int((~is_national).sum())
    if n_nat == 0 or n_state == 0:
        raise ValueError(
            "PE-native broad loss needs both national and state targets"
        )

    norm_factor = np.where(is_national, 1.0 / n_nat, 1.0 / n_state).astype(
        np.float64
    )
    inv_mean = 1.0 / float(np.mean(norm_factor))
    per_target_weight = (
        inv_mean * norm_factor / float(len(filt_targets))
    ).astype(np.float64)
    denom = (filt_targets + 1.0).astype(np.float64)
    scaling = np.sqrt(per_target_weight) / denom
    scaled_matrix = filtered.to_numpy(dtype=np.float32) * scaling[
        np.newaxis, :
    ].astype(np.float32)  # (households, targets)
    scaled_target = ((filt_targets - 1.0) * scaling).astype(np.float64)

    # loss = sum((M @ w - t)^2) == the PE-native _objective, since the 1/n and
    # national/state balancing are already folded into `scaling`. normalization
    # is 1 so relative_errors are the raw scaled residuals.
    problem = CalibrationProblem(
        matrix=scaled_matrix.T,  # (targets, households)
        target=scaled_target,
        normalization=np.ones_like(scaled_target),
        names=tuple(filtered.columns.tolist()),
        reduction="sum",
    )

    sim = Microsimulation(dataset=dataset_cls)
    sim.default_calculation_period = period
    weights = (
        sim.calculate("household_weight", map_to="household", period=period)
        .values.astype(np.float64)
    )

    families = [
        _classify_pe_native_target_family(name)
        for name in filtered.columns.tolist()
    ]

    return PeNativeProblem(
        problem=problem,
        weights=weights,
        target_names=filtered.columns.tolist(),
        families=families,
        is_national=is_national,
        n_households=int(weights.shape[0]),
    )
