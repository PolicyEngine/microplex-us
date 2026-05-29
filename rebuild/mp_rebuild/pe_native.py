"""Clean PE-native scoring loader.

Builds :class:`~mp_rebuild.scoreboard.CalibrationProblem` objects from the
canonical ``policyengine_us_data.utils.loss.build_loss_matrix`` using the SAME
target filtering and national/state-balanced scaling as the incumbent PE-native
broad loss. This reproduces the published eCPS baseline (~0.166), so candidate
and baseline are scored on an identical metric.

We reuse only *data/definition* helpers from the existing package (the target
list and the family classifier), not the optimizer. The optimizer is
:func:`mp_rebuild.scoreboard.fit`, which cannot increase the scored loss.

Key design point for fair comparison: the PE-native scaling
(``sqrt(per_target_weight)/(target+1)``) depends only on the TARGET set
(value + national/state membership), not on the dataset. So a symmetric
comparison computes the scaling once over the *common* targets and applies it
identically to both datasets. ``load_pe_native_raw`` returns the unscaled
pieces for exactly that purpose; ``build_scaled_problem`` applies the scaling.

The PE-native ``_objective`` is ``sum((M_scaled @ w - t_scaled)^2)`` (per-target
weighting baked into the scaling), wrapped here as ``reduction="sum"`` with unit
normalization.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .scoreboard import CalibrationProblem


@dataclass
class RawPeNative:
    """Unscaled PE-native pieces for one dataset (kept targets only)."""

    raw_matrix: np.ndarray  # (n_households, n_targets), float32, raw contributions
    targets: np.ndarray  # (n_targets,) float64 target values (dataset-independent)
    names: list[str]
    is_national: np.ndarray  # (n_targets,) bool
    weights: np.ndarray  # (n_households,) shipped household weights


@dataclass
class PeNativeProblem:
    problem: CalibrationProblem
    weights: np.ndarray
    target_names: list[str]
    families: list[str]
    is_national: np.ndarray
    n_households: int


def _local_dataset(dataset_path: str, period: int):
    from policyengine_core.data import Dataset

    class LocalDataset(Dataset):
        name = "mpr_eval_dataset"
        label = "mpr_eval_dataset"
        file_path = dataset_path
        data_format = Dataset.TIME_PERIOD_ARRAYS
        time_period = period

    return LocalDataset


def pe_native_scaling(targets: np.ndarray, is_national: np.ndarray) -> np.ndarray:
    """The PE-native per-target scaling. Depends only on the target set.

    Identical for any dataset given the same targets, which is what makes a
    symmetric comparison well defined.
    """
    targets = np.asarray(targets, dtype=np.float64)
    is_national = np.asarray(is_national, dtype=bool)
    n_nat = int(is_national.sum())
    n_state = int((~is_national).sum())
    if n_nat == 0 or n_state == 0:
        raise ValueError("PE-native loss needs both national and state targets")
    norm_factor = np.where(is_national, 1.0 / n_nat, 1.0 / n_state).astype(np.float64)
    inv_mean = 1.0 / float(np.mean(norm_factor))
    per_target_weight = (inv_mean * norm_factor / float(len(targets))).astype(np.float64)
    denom = (targets + 1.0).astype(np.float64)
    return np.sqrt(per_target_weight) / denom


def build_scaled_problem(
    raw_matrix: np.ndarray,
    targets: np.ndarray,
    is_national: np.ndarray,
    names: list[str],
    scaling: np.ndarray | None = None,
) -> CalibrationProblem:
    """Apply PE-native scaling to raw contributions -> a CalibrationProblem.

    Pass ``scaling`` explicitly (computed over a shared/common target set) to
    guarantee both sides of a comparison use identical scaling.
    """
    if scaling is None:
        scaling = pe_native_scaling(targets, is_national)
    scaling32 = scaling[np.newaxis, :].astype(np.float32)
    scaled_matrix = np.asarray(raw_matrix, dtype=np.float32) * scaling32  # (hh, targets)
    scaled_target = ((np.asarray(targets, dtype=np.float64) - 1.0) * scaling).astype(np.float64)
    return CalibrationProblem(
        matrix=scaled_matrix.T,  # (targets, households)
        target=scaled_target,
        normalization=np.ones_like(scaled_target),
        names=tuple(names),
        reduction="sum",
    )


def load_pe_native_raw(
    dataset_path: str | Path,
    period: int = 2024,
    bad_targets: tuple[str, ...] | None = None,
) -> RawPeNative:
    """Build the filtered, UNSCALED PE-native pieces for a dataset.

    Single-batch (fine for eCPS ~41k; larger candidates may need batching --
    tracked in PLAN.md).
    """
    from policyengine_us import Microsimulation
    from policyengine_us_data.utils.loss import build_loss_matrix

    if bad_targets is None:
        from microplex_us.pipelines.pe_native_scores import _ENHANCED_CPS_BAD_TARGETS

        bad_targets = tuple(_ENHANCED_CPS_BAD_TARGETS)

    dataset_path = str(dataset_path)
    dataset_cls = _local_dataset(dataset_path, period)

    loss_matrix, targets_array = build_loss_matrix(dataset_cls, period)
    target_names = np.asarray(loss_matrix.columns)
    targets_array = np.asarray(targets_array, dtype=np.float64)

    zero_mask = np.isclose(targets_array, 0.0, atol=0.1)
    bad_mask = np.isin(target_names, np.asarray(bad_targets))
    keep = ~(zero_mask | bad_mask)

    filtered = loss_matrix.loc[:, keep]
    is_national = np.asarray(filtered.columns.str.startswith("nation/"), dtype=bool)

    sim = Microsimulation(dataset=dataset_cls)
    sim.default_calculation_period = period
    weights = (
        sim.calculate("household_weight", map_to="household", period=period)
        .values.astype(np.float64)
    )

    return RawPeNative(
        raw_matrix=filtered.to_numpy(dtype=np.float32),  # (households, targets)
        targets=targets_array[keep],
        names=list(filtered.columns),
        is_national=is_national,
        weights=weights,
    )


def load_pe_native_problem(
    dataset_path: str | Path,
    period: int = 2024,
    bad_targets: tuple[str, ...] | None = None,
) -> PeNativeProblem:
    """Single-dataset PE-native problem (scaling from its own kept targets).

    This reproduces the published eCPS baseline (~0.166).
    """
    from microplex_us.pipelines.pe_native_scores import (
        _classify_pe_native_target_family,
    )

    raw = load_pe_native_raw(dataset_path, period, bad_targets)
    problem = build_scaled_problem(
        raw.raw_matrix, raw.targets, raw.is_national, raw.names
    )
    families = [_classify_pe_native_target_family(name) for name in raw.names]
    return PeNativeProblem(
        problem=problem,
        weights=raw.weights,
        target_names=raw.names,
        families=families,
        is_national=raw.is_national,
        n_households=int(raw.weights.shape[0]),
    )
