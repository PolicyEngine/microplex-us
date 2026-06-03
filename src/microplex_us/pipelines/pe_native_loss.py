"""Shared PE-native robust loss helpers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

PE_NATIVE_ROBUST_LOSS_METRIC = "pe_native_bucketed_baseline_huber_v1"
DEFAULT_BASELINE_WEIGHT_BETA = 1.0
DEFAULT_BUCKET_EPSILON_FRACTION = 0.02
DEFAULT_HUBER_DELTA = 1.0


@dataclass(frozen=True)
class PENativeLossArrays:
    """Per-target constants for the robust PE-native loss."""

    target_names: tuple[str, ...]
    target_values: np.ndarray
    objective_target: np.ndarray
    denominator: np.ndarray
    target_weight: np.ndarray
    bucket_keys: np.ndarray
    unit_keys: np.ndarray
    scope_keys: np.ndarray
    family_keys: np.ndarray
    epsilon: np.ndarray
    beta: float
    huber_delta: float
    epsilon_fraction: float
    bucket_weight_mode: str

    def metadata(self) -> dict[str, Any]:
        unique_buckets, bucket_counts = np.unique(
            self.bucket_keys,
            return_counts=True,
        )
        return {
            "loss_metric": PE_NATIVE_ROBUST_LOSS_METRIC,
            "loss_config": {
                "baseline_weight_beta": float(self.beta),
                "bucket_epsilon_fraction": float(self.epsilon_fraction),
                "huber_delta": float(self.huber_delta),
                "bucket_key": "scope_x_unit",
                "bucket_weight_mode": self.bucket_weight_mode,
                "residual": "(estimate - target) / (abs(target) + eps_bucket)",
                "penalty": "huber",
            },
            "loss_buckets": {
                str(bucket): int(count)
                for bucket, count in zip(unique_buckets, bucket_counts, strict=True)
            },
        }

    def sidecar_rows(self) -> list[dict[str, Any]]:
        return [
            {
                "target_index": int(index),
                "target_name": str(name),
                "scope": str(self.scope_keys[index]),
                "unit": str(self.unit_keys[index]),
                "family": str(self.family_keys[index]),
                "bucket": str(self.bucket_keys[index]),
                "target_value": float(self.target_values[index]),
                "objective_target": float(self.objective_target[index]),
                "denominator": float(self.denominator[index]),
                "epsilon": float(self.epsilon[index]),
                "target_weight": float(self.target_weight[index]),
            }
            for index, name in enumerate(self.target_names)
        ]


def build_pe_native_loss_arrays(
    target_names: list[str] | tuple[str, ...] | np.ndarray,
    target_values: np.ndarray,
    *,
    beta: float = DEFAULT_BASELINE_WEIGHT_BETA,
    epsilon_fraction: float = DEFAULT_BUCKET_EPSILON_FRACTION,
    huber_delta: float = DEFAULT_HUBER_DELTA,
    bucket_weight_mode: str = "equal_bucket",
) -> PENativeLossArrays:
    """Build constants for the bucketed, baseline-weighted Huber loss."""

    names = tuple(str(name) for name in target_names)
    targets = np.asarray(target_values, dtype=np.float64)
    if targets.ndim != 1:
        raise ValueError("target_values must be 1D")
    if len(names) != targets.shape[0]:
        raise ValueError("target_names and target_values length mismatch")
    if targets.size == 0:
        raise ValueError("PE-native loss requires at least one target")
    if beta < 0.0:
        raise ValueError("baseline-weight beta must be nonnegative")
    if epsilon_fraction < 0.0:
        raise ValueError("bucket epsilon fraction must be nonnegative")
    if huber_delta <= 0.0:
        raise ValueError("Huber delta must be positive")
    if bucket_weight_mode != "equal_bucket":
        raise ValueError("Only equal_bucket weighting is implemented")

    scopes = np.asarray(
        [infer_pe_native_target_scope(name) for name in names], dtype=object
    )
    units = np.asarray(
        [infer_pe_native_target_unit(name) for name in names], dtype=object
    )
    families = np.asarray(
        [classify_pe_native_target_family(name) for name in names], dtype=object
    )
    buckets = np.asarray(
        [f"{scope}:{unit}" for scope, unit in zip(scopes, units, strict=True)],
        dtype=object,
    )
    abs_targets = np.abs(targets)
    epsilon = np.zeros_like(targets, dtype=np.float64)
    target_weight = np.zeros_like(targets, dtype=np.float64)
    unique_buckets = sorted({str(bucket) for bucket in buckets})
    bucket_budget = 1.0 / float(len(unique_buckets))
    for bucket in unique_buckets:
        mask = buckets == bucket
        bucket_targets = abs_targets[mask]
        nonzero_targets = bucket_targets[bucket_targets > 0.0]
        median_target = (
            float(np.median(nonzero_targets)) if nonzero_targets.size else 1.0
        )
        bucket_epsilon = max(float(epsilon_fraction) * median_target, 1e-12)
        epsilon[mask] = bucket_epsilon
        baseline_importance = np.power(bucket_targets + bucket_epsilon, beta)
        total_importance = float(baseline_importance.sum())
        if total_importance <= 0.0 or not np.isfinite(total_importance):
            target_weight[mask] = bucket_budget / float(mask.sum())
        else:
            target_weight[mask] = bucket_budget * baseline_importance / total_importance

    denominator = abs_targets + epsilon
    return PENativeLossArrays(
        target_names=names,
        target_values=targets,
        objective_target=targets.astype(np.float64, copy=True),
        denominator=denominator,
        target_weight=target_weight,
        bucket_keys=buckets,
        unit_keys=units,
        scope_keys=scopes,
        family_keys=families,
        epsilon=epsilon,
        beta=float(beta),
        huber_delta=float(huber_delta),
        epsilon_fraction=float(epsilon_fraction),
        bucket_weight_mode=bucket_weight_mode,
    )


def pe_native_huber_loss_terms(
    estimate: np.ndarray,
    loss_arrays: PENativeLossArrays,
) -> np.ndarray:
    rel = pe_native_relative_error(estimate, loss_arrays)
    return loss_arrays.target_weight * huber_value(rel, loss_arrays.huber_delta)


def pe_native_huber_loss(
    estimate: np.ndarray,
    loss_arrays: PENativeLossArrays,
) -> float:
    return float(pe_native_huber_loss_terms(estimate, loss_arrays).sum())


def pe_native_huber_gradient_factor(
    estimate: np.ndarray,
    loss_arrays: PENativeLossArrays,
) -> np.ndarray:
    rel = pe_native_relative_error(estimate, loss_arrays)
    return (
        loss_arrays.target_weight
        * huber_derivative(rel, loss_arrays.huber_delta)
        / loss_arrays.denominator
    )


def pe_native_relative_error(
    estimate: np.ndarray,
    loss_arrays: PENativeLossArrays,
) -> np.ndarray:
    estimate_array = np.asarray(estimate, dtype=np.float64)
    if estimate_array.shape != loss_arrays.objective_target.shape:
        raise ValueError("estimate and target shapes differ")
    return (estimate_array - loss_arrays.objective_target) / loss_arrays.denominator


def huber_value(values: np.ndarray, delta: float) -> np.ndarray:
    values_array = np.asarray(values, dtype=np.float64)
    abs_values = np.abs(values_array)
    return np.where(
        abs_values <= delta,
        0.5 * np.square(values_array),
        delta * (abs_values - 0.5 * delta),
    )


def huber_derivative(values: np.ndarray, delta: float) -> np.ndarray:
    return np.clip(np.asarray(values, dtype=np.float64), -delta, delta)


def infer_pe_native_target_scope(target_name: str) -> str:
    if target_name.startswith("nation/"):
        return "national"
    return "state"


def infer_pe_native_target_unit(target_name: str) -> str:
    normalized = target_name.lower().replace("-", "_")
    parts = normalized.split("/")
    if normalized.endswith("/snap_hhs") or normalized.endswith("/snap-hhs"):
        return "households"
    if any(part in {"amount", "total"} for part in parts):
        return "dollars"
    if (
        len(parts) >= 3
        and parts[0] == "nation"
        and parts[1] == "cbo"
        and parts[2] == "income_by_source"
    ):
        return "dollars"
    if any(part in {"count", "returns", "filers"} for part in parts):
        return "returns"
    if "spending" in normalized or "cost" in normalized or "tax" in normalized:
        return "dollars"
    if "net_worth" in normalized or "income" in normalized:
        return "dollars"
    if "enrollment" in normalized or "population" in normalized:
        return "people"
    if "/age/" in normalized or "population_by_age" in normalized:
        return "people"
    if "household" in normalized or "hhs" in normalized:
        return "households"
    return "other"


def classify_pe_native_target_family(target_name: str) -> str:
    """Classify one PE target name into broad diagnostic families."""

    parts = target_name.split("/")
    if target_name.startswith("state/census/age/"):
        return "state_age_distribution"
    if target_name.startswith("state/census/population_by_state/"):
        return "state_population"
    if target_name.startswith("state/census/population_under_5_by_state/"):
        return "state_population_under_5"
    if target_name.startswith("nation/irs/aca_spending/"):
        return "state_aca_spending"
    if target_name.startswith("state/irs/aca_enrollment/"):
        return "state_aca_enrollment"
    if target_name.startswith("irs/medicaid_enrollment/"):
        return "state_medicaid_enrollment"
    if target_name.endswith("/snap-cost"):
        return "state_snap_cost"
    if target_name.endswith("/snap-hhs"):
        return "state_snap_households"
    if target_name.startswith("state/real_estate_taxes/"):
        return "state_real_estate_taxes"
    if len(parts) >= 3 and parts[0] == "state" and parts[2] == "adjusted_gross_income":
        return "state_agi_distribution"
    if target_name.startswith("nation/jct/"):
        return "national_tax_expenditures"
    if target_name.startswith("nation/net_worth/"):
        return "national_net_worth"
    if target_name.startswith("nation/ssa/"):
        return "national_ssa"
    if target_name.startswith("nation/census/population_by_age/"):
        return "national_population_by_age"
    if target_name == "nation/census/infants":
        return "national_infants"
    if target_name.startswith("nation/census/agi_in_spm_threshold_decile_"):
        return "national_spm_threshold_agi"
    if target_name.startswith("nation/census/count_in_spm_threshold_decile_"):
        return "national_spm_threshold_count"
    if target_name.startswith("nation/census/"):
        return "national_census_other"
    if target_name.startswith("nation/irs/"):
        return "national_irs_other"
    return "other"


def loss_arrays_from_inputs(loss_inputs: dict[str, Any]) -> PENativeLossArrays | None:
    metadata = dict(loss_inputs.get("metadata") or {})
    if metadata.get("loss_metric") != PE_NATIVE_ROBUST_LOSS_METRIC:
        return None
    target_names = tuple(str(name) for name in metadata.get("target_names", ()))
    return PENativeLossArrays(
        target_names=target_names,
        target_values=np.asarray(loss_inputs["unscaled_target"], dtype=np.float64),
        objective_target=np.asarray(loss_inputs["scaled_target"], dtype=np.float64),
        denominator=np.asarray(loss_inputs["loss_denominator"], dtype=np.float64),
        target_weight=np.asarray(loss_inputs["loss_target_weight"], dtype=np.float64),
        bucket_keys=np.asarray(loss_inputs["loss_bucket"], dtype=object),
        unit_keys=np.asarray(loss_inputs["loss_unit"], dtype=object),
        scope_keys=np.asarray(loss_inputs["loss_scope"], dtype=object),
        family_keys=np.asarray(loss_inputs["loss_family"], dtype=object),
        epsilon=np.asarray(loss_inputs["loss_epsilon"], dtype=np.float64),
        beta=float(metadata.get("loss_config", {}).get("baseline_weight_beta", 1.0)),
        huber_delta=float(metadata.get("loss_config", {}).get("huber_delta", 1.0)),
        epsilon_fraction=float(
            metadata.get("loss_config", {}).get("bucket_epsilon_fraction", 0.02)
        ),
        bucket_weight_mode=str(
            metadata.get("loss_config", {}).get("bucket_weight_mode", "equal_bucket")
        ),
    )


def subset_loss_arrays(
    loss_arrays: PENativeLossArrays,
    mask: np.ndarray,
) -> PENativeLossArrays:
    mask_array = np.asarray(mask, dtype=bool)
    if mask_array.shape != loss_arrays.objective_target.shape:
        raise ValueError("loss-array mask shape mismatch")
    return PENativeLossArrays(
        target_names=tuple(
            name
            for name, keep in zip(loss_arrays.target_names, mask_array, strict=True)
            if keep
        ),
        target_values=loss_arrays.target_values[mask_array],
        objective_target=loss_arrays.objective_target[mask_array],
        denominator=loss_arrays.denominator[mask_array],
        target_weight=loss_arrays.target_weight[mask_array],
        bucket_keys=loss_arrays.bucket_keys[mask_array],
        unit_keys=loss_arrays.unit_keys[mask_array],
        scope_keys=loss_arrays.scope_keys[mask_array],
        family_keys=loss_arrays.family_keys[mask_array],
        epsilon=loss_arrays.epsilon[mask_array],
        beta=loss_arrays.beta,
        huber_delta=loss_arrays.huber_delta,
        epsilon_fraction=loss_arrays.epsilon_fraction,
        bucket_weight_mode=loss_arrays.bucket_weight_mode,
    )


__all__ = [
    "DEFAULT_BASELINE_WEIGHT_BETA",
    "DEFAULT_BUCKET_EPSILON_FRACTION",
    "DEFAULT_HUBER_DELTA",
    "PE_NATIVE_ROBUST_LOSS_METRIC",
    "PENativeLossArrays",
    "build_pe_native_loss_arrays",
    "classify_pe_native_target_family",
    "huber_derivative",
    "huber_value",
    "infer_pe_native_target_scope",
    "infer_pe_native_target_unit",
    "loss_arrays_from_inputs",
    "pe_native_huber_gradient_factor",
    "pe_native_huber_loss",
    "pe_native_huber_loss_terms",
    "pe_native_relative_error",
    "subset_loss_arrays",
]
