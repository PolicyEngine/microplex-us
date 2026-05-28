"""Reweight PE-US H5 datasets to congressional-district age targets."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import pandas as pd
from scipy.optimize import minimize

from microplex_us.pipelines.pe_native_optimization import (
    rewrite_policyengine_us_dataset_weights,
)
from microplex_us.policyengine import PolicyEngineUSDBTargetProvider
from microplex_us.policyengine.us import PolicyEngineUSConstraint


@dataclass(frozen=True)
class CDAgeTarget:
    """One congressional-district person-count-by-age target."""

    target_id: int
    district_geoid: int
    value: float
    age_constraints: tuple[PolicyEngineUSConstraint, ...]
    period: int

    @property
    def age_key(self) -> tuple[tuple[str, str], ...]:
        return tuple(
            sorted((constraint.operation, str(constraint.value)) for constraint in self.age_constraints)
        )


def normalize_at_large_cd_geoids(values: np.ndarray) -> np.ndarray:
    """Normalize statewide at-large districts from ``xx00`` to PE target ``xx01``."""
    result = np.asarray(values).copy()
    finite = np.isfinite(result.astype(float, copy=False))
    as_int = result.astype(np.int64, copy=False)
    at_large = finite & (as_int > 0) & (as_int % 100 == 0)
    result[at_large] = as_int[at_large] + 1
    return result.astype(np.int64, copy=False)


def load_cd_age_targets(
    target_db: str | Path,
    *,
    period: int = 2024,
) -> list[CDAgeTarget]:
    """Load active district person-count-by-age targets from PE's target DB."""
    provider = PolicyEngineUSDBTargetProvider(target_db)
    raw_targets = provider.load_targets(
        period=period,
        variables=["person_count"],
        domain_variables=["age"],
        geo_levels=["district"],
        active_only=True,
    )
    targets: list[CDAgeTarget] = []
    for target in raw_targets:
        district_constraints = [
            constraint
            for constraint in target.constraints
            if constraint.variable == "congressional_district_geoid"
        ]
        age_constraints = tuple(
            constraint for constraint in target.constraints if constraint.variable == "age"
        )
        if len(district_constraints) != 1 or not age_constraints:
            continue
        targets.append(
            CDAgeTarget(
                target_id=int(target.target_id),
                district_geoid=int(district_constraints[0].value),
                value=float(target.value),
                age_constraints=age_constraints,
                period=int(target.period),
            )
        )
    targets.sort(key=lambda target: (target.district_geoid, target.age_key, target.target_id))
    return targets


def reweight_h5_to_cd_age_targets(
    *,
    input_dataset: str | Path,
    target_db: str | Path,
    output_dataset: str | Path,
    period: int = 2024,
    max_iter: int = 300,
    tol: float = 1e-9,
    preserve_district_weight_sum: bool = True,
    details_output: str | Path | None = None,
) -> dict[str, Any]:
    """Apply independent per-CD entropy reweighting for age-distribution targets."""
    period_key = str(period)
    targets = load_cd_age_targets(target_db, period=period)
    if not targets:
        raise ValueError("No district person_count-by-age targets were loaded")

    with h5py.File(input_dataset, "r") as handle:
        household_ids = np.asarray(handle["household_id"][period_key])
        input_weights = np.asarray(handle["household_weight"][period_key], dtype=np.float64)
        household_cd = normalize_at_large_cd_geoids(
            np.asarray(handle["congressional_district_geoid"][period_key])
        )
        person_household_id = np.asarray(handle["person_household_id"][period_key])
        age = np.asarray(handle["age"][period_key], dtype=np.float64)

    person_household_index = _map_person_households_to_indices(
        household_ids,
        person_household_id,
    )
    unique_age_keys = sorted({target.age_key for target in targets})
    household_age_counts = _build_household_age_count_matrix(
        n_households=len(household_ids),
        person_household_index=person_household_index,
        age=age,
        age_keys=unique_age_keys,
    )
    age_key_to_col = {age_key: index for index, age_key in enumerate(unique_age_keys)}

    output_weights = input_weights.copy()
    detail_rows: list[dict[str, Any]] = []
    district_failures: list[dict[str, Any]] = []
    targets_by_district: dict[int, list[CDAgeTarget]] = {}
    for target in targets:
        targets_by_district.setdefault(target.district_geoid, []).append(target)

    for district_geoid, district_targets in sorted(targets_by_district.items()):
        household_mask = household_cd == district_geoid
        household_indices = np.flatnonzero(household_mask)
        if len(household_indices) == 0:
            district_failures.append(
                {
                    "district_geoid": district_geoid,
                    "reason": "no_households",
                    "target_count": len(district_targets),
                }
            )
            _append_detail_rows(
                detail_rows,
                targets=district_targets,
                age_key_to_col=age_key_to_col,
                household_indices=household_indices,
                household_age_counts=household_age_counts,
                input_weights=input_weights,
                output_weights=output_weights,
                status="no_households",
            )
            continue

        row_cols = [age_key_to_col[target.age_key] for target in district_targets]
        design = household_age_counts[np.ix_(household_indices, row_cols)].T.astype(
            np.float64,
            copy=False,
        )
        target_values = np.asarray([target.value for target in district_targets], dtype=np.float64)
        base_weights = input_weights[household_indices]
        fit_design = design
        fit_targets = target_values
        if preserve_district_weight_sum:
            fit_design = np.vstack(
                [
                    design,
                    np.ones((1, design.shape[1]), dtype=np.float64),
                ]
            )
            fit_targets = np.concatenate(
                [target_values, np.asarray([base_weights.sum()], dtype=np.float64)]
            )
        solution = _solve_entropy_weights(
            design=fit_design,
            base_weights=base_weights,
            targets=fit_targets,
            max_iter=max_iter,
            tol=tol,
        )
        output_weights[household_indices] = solution["weights"]
        if not solution["success"]:
            district_failures.append(
                {
                    "district_geoid": district_geoid,
                    "reason": solution["message"],
                    "target_count": len(district_targets),
                    "max_abs_relative_error": solution["max_abs_relative_error"],
                }
            )
        _append_detail_rows(
            detail_rows,
            targets=district_targets,
            age_key_to_col=age_key_to_col,
            household_indices=household_indices,
            household_age_counts=household_age_counts,
            input_weights=input_weights,
            output_weights=output_weights,
            status="ok" if solution["success"] else "not_converged",
        )

    output_path = rewrite_policyengine_us_dataset_weights(
        input_dataset_path=input_dataset,
        output_dataset_path=output_dataset,
        household_weights=output_weights,
        period=period,
    )
    _normalize_cd_geoids_in_h5(output_path, period=period)

    detail_frame = pd.DataFrame(detail_rows)
    if details_output is not None:
        detail_path = Path(details_output).expanduser().resolve()
        detail_path.parent.mkdir(parents=True, exist_ok=True)
        detail_frame.to_csv(detail_path, index=False)

    summary = _summarize_detail_frame(
        detail_frame,
        input_weight_sum=float(input_weights.sum()),
        output_weight_sum=float(output_weights.sum()),
        n_households=len(input_weights),
        n_persons=len(age),
        n_age_bins=len(unique_age_keys),
        district_failures=district_failures,
    )
    summary["preserve_district_weight_sum"] = bool(preserve_district_weight_sum)
    summary["input_dataset"] = str(Path(input_dataset).expanduser().resolve())
    summary["output_dataset"] = str(Path(output_path).expanduser().resolve())
    summary["target_db"] = str(Path(target_db).expanduser().resolve())
    summary["period"] = int(period)
    return summary


def build_cd_age_constraint_matrix(
    *,
    input_dataset: str | Path,
    target_db: str | Path,
    period: int = 2024,
    target_weight: float = 1.0,
) -> dict[str, Any]:
    """Build scaled sparse rows for CD person-count-by-age targets.

    The returned matrix has shape ``(targets, households)`` and uses the same
    ``((estimate - target + 1) / (target + 1)) ** 2`` row scaling convention as
    the PE-native broad matrix.
    """
    if target_weight <= 0:
        raise ValueError("target_weight must be positive")
    period_key = str(period)
    targets = load_cd_age_targets(target_db, period=period)
    if not targets:
        raise ValueError("No district person_count-by-age targets were loaded")

    with h5py.File(input_dataset, "r") as handle:
        household_ids = np.asarray(handle["household_id"][period_key])
        household_cd = normalize_at_large_cd_geoids(
            np.asarray(handle["congressional_district_geoid"][period_key])
        )
        person_household_id = np.asarray(handle["person_household_id"][period_key])
        age = np.asarray(handle["age"][period_key], dtype=np.float64)

    person_household_index = _map_person_households_to_indices(
        household_ids,
        person_household_id,
    )
    unique_age_keys = sorted({target.age_key for target in targets})
    household_age_counts = _build_household_age_count_matrix(
        n_households=len(household_ids),
        person_household_index=person_household_index,
        age=age,
        age_keys=unique_age_keys,
    )
    age_key_to_col = {age_key: index for index, age_key in enumerate(unique_age_keys)}

    rows: list[np.ndarray] = []
    cols: list[np.ndarray] = []
    vals: list[np.ndarray] = []
    target_values = np.asarray([target.value for target in targets], dtype=np.float64)
    scaling = np.sqrt(float(target_weight) / float(len(targets))) / (
        target_values + 1.0
    )
    target_names: list[str] = []
    for row_index, target in enumerate(targets):
        count_col = age_key_to_col[target.age_key]
        household_indices = np.flatnonzero(household_cd == target.district_geoid)
        counts = household_age_counts[household_indices, count_col]
        nonzero = counts != 0
        if nonzero.any():
            rows.append(np.full(int(nonzero.sum()), row_index, dtype=np.int32))
            cols.append(household_indices[nonzero].astype(np.int32))
            vals.append((counts[nonzero] * scaling[row_index]).astype(np.float32))
        target_names.append(
            "district/census/person_count_by_age/"
            f"{target.district_geoid}/{json.dumps(target.age_key, separators=(',', ':'))}"
        )

    if rows:
        import scipy.sparse as sp

        matrix = sp.csr_matrix(
            (
                np.concatenate(vals),
                (np.concatenate(rows), np.concatenate(cols)),
            ),
            shape=(len(targets), len(household_ids)),
            dtype=np.float32,
        )
    else:
        import scipy.sparse as sp

        matrix = sp.csr_matrix((len(targets), len(household_ids)), dtype=np.float32)

    scaled_target = ((target_values - 1.0) * scaling).astype(np.float32)
    return {
        "matrix": matrix,
        "target": scaled_target,
        "metadata": {
            "target_names": target_names,
            "n_targets_total": int(len(targets)),
            "n_targets_kept": int(len(targets)),
            "n_districts": int(len({target.district_geoid for target in targets})),
            "n_age_bins": int(len(unique_age_keys)),
            "target_weight": float(target_weight),
            "target_db": str(Path(target_db).expanduser().resolve()),
            "family": "district_age_distribution",
        },
    }


def _map_person_households_to_indices(
    household_ids: np.ndarray,
    person_household_ids: np.ndarray,
) -> np.ndarray:
    household_index = {int(household_id): index for index, household_id in enumerate(household_ids)}
    try:
        return np.asarray(
            [household_index[int(household_id)] for household_id in person_household_ids],
            dtype=np.int64,
        )
    except KeyError as exc:
        raise ValueError(f"person_household_id references missing household_id {exc}") from exc


def _build_household_age_count_matrix(
    *,
    n_households: int,
    person_household_index: np.ndarray,
    age: np.ndarray,
    age_keys: list[tuple[tuple[str, str], ...]],
) -> np.ndarray:
    counts = np.zeros((n_households, len(age_keys)), dtype=np.float32)
    for col, age_key in enumerate(age_keys):
        mask = _evaluate_age_key(age, age_key)
        np.add.at(counts[:, col], person_household_index[mask], 1.0)
    return counts


def _evaluate_age_key(
    age: np.ndarray,
    age_key: tuple[tuple[str, str], ...],
) -> np.ndarray:
    mask = np.ones(len(age), dtype=bool)
    for operation, raw_value in age_key:
        value = float(raw_value)
        if operation == "==":
            mask &= age == value
        elif operation == "!=":
            mask &= age != value
        elif operation == ">":
            mask &= age > value
        elif operation == ">=":
            mask &= age >= value
        elif operation == "<":
            mask &= age < value
        elif operation == "<=":
            mask &= age <= value
        else:
            raise ValueError(f"Unsupported age target operation: {operation!r}")
    return mask


def _solve_entropy_weights(
    *,
    design: np.ndarray,
    base_weights: np.ndarray,
    targets: np.ndarray,
    max_iter: int,
    tol: float,
) -> dict[str, Any]:
    support = design.sum(axis=1) > 0
    unsupported = (~support) & (np.abs(targets) > tol)
    if unsupported.any():
        estimates = design @ base_weights
        return {
            "weights": base_weights.copy(),
            "success": False,
            "message": "unsupported_positive_targets",
            "max_abs_relative_error": float(
                _abs_relative_error(estimates, targets).max(initial=0.0)
            ),
        }

    def objective(lam: np.ndarray) -> tuple[float, np.ndarray]:
        linear_predictor = np.clip(lam @ design, -50.0, 50.0)
        weights = base_weights * np.exp(linear_predictor)
        value = float(weights.sum() - np.dot(targets, lam))
        gradient = design @ weights - targets
        return value, gradient

    result = minimize(
        fun=lambda lam: objective(lam)[0],
        x0=np.zeros(design.shape[0], dtype=np.float64),
        jac=lambda lam: objective(lam)[1],
        method="L-BFGS-B",
        options={"maxiter": int(max_iter), "ftol": tol, "gtol": tol},
    )
    linear_predictor = np.clip(result.x @ design, -50.0, 50.0)
    weights = base_weights * np.exp(linear_predictor)
    estimates = design @ weights
    max_error = float(_abs_relative_error(estimates, targets).max(initial=0.0))
    success = bool(result.success) or max_error <= max(1e-4, tol * 100)
    return {
        "weights": weights,
        "success": success,
        "message": str(result.message),
        "max_abs_relative_error": max_error,
    }


def _append_detail_rows(
    rows: list[dict[str, Any]],
    *,
    targets: list[CDAgeTarget],
    age_key_to_col: dict[tuple[tuple[str, str], ...], int],
    household_indices: np.ndarray,
    household_age_counts: np.ndarray,
    input_weights: np.ndarray,
    output_weights: np.ndarray,
    status: str,
) -> None:
    for target in targets:
        col = age_key_to_col[target.age_key]
        counts = household_age_counts[household_indices, col]
        before = float(np.dot(counts, input_weights[household_indices]))
        after = float(np.dot(counts, output_weights[household_indices]))
        rows.append(
            {
                "target_id": target.target_id,
                "district_geoid": target.district_geoid,
                "age_key": json.dumps(target.age_key),
                "target": target.value,
                "estimate_before": before,
                "estimate_after": after,
                "relative_error_before": _relative_error(before, target.value),
                "relative_error_after": _relative_error(after, target.value),
                "abs_relative_error_before": abs(_relative_error(before, target.value)),
                "abs_relative_error_after": abs(_relative_error(after, target.value)),
                "period": target.period,
                "status": status,
            }
        )


def _relative_error(estimate: float, target: float) -> float:
    if abs(target) <= 1e-12:
        return 0.0 if abs(estimate) <= 1e-12 else float("inf")
    return float((estimate - target) / abs(target))


def _abs_relative_error(estimate: np.ndarray, target: np.ndarray) -> np.ndarray:
    denominator = np.where(np.abs(target) <= 1e-12, 1.0, np.abs(target))
    return np.abs((estimate - target) / denominator)


def _summarize_detail_frame(
    detail_frame: pd.DataFrame,
    *,
    input_weight_sum: float,
    output_weight_sum: float,
    n_households: int,
    n_persons: int,
    n_age_bins: int,
    district_failures: list[dict[str, Any]],
) -> dict[str, Any]:
    before = detail_frame["abs_relative_error_before"].to_numpy(dtype=np.float64)
    after = detail_frame["abs_relative_error_after"].to_numpy(dtype=np.float64)
    return {
        "n_targets": int(len(detail_frame)),
        "n_districts": int(detail_frame["district_geoid"].nunique()),
        "n_households": int(n_households),
        "n_persons": int(n_persons),
        "n_age_bins": int(n_age_bins),
        "input_weight_sum": float(input_weight_sum),
        "output_weight_sum": float(output_weight_sum),
        "weight_sum_relative_change": float(
            (output_weight_sum - input_weight_sum) / input_weight_sum
        ),
        "mean_abs_relative_error_before": float(before.mean()),
        "mean_abs_relative_error_after": float(after.mean()),
        "median_abs_relative_error_before": float(np.median(before)),
        "median_abs_relative_error_after": float(np.median(after)),
        "p90_abs_relative_error_before": float(np.quantile(before, 0.9)),
        "p90_abs_relative_error_after": float(np.quantile(after, 0.9)),
        "p99_abs_relative_error_before": float(np.quantile(before, 0.99)),
        "p99_abs_relative_error_after": float(np.quantile(after, 0.99)),
        "max_abs_relative_error_before": float(before.max(initial=0.0)),
        "max_abs_relative_error_after": float(after.max(initial=0.0)),
        "failed_district_count": int(len(district_failures)),
        "district_failures": district_failures,
    }


def _normalize_cd_geoids_in_h5(path: str | Path, *, period: int) -> None:
    period_key = str(period)
    with h5py.File(path, "r+") as handle:
        if "congressional_district_geoid" not in handle:
            return
        group = handle["congressional_district_geoid"]
        if period_key not in group:
            return
        values = np.asarray(group[period_key])
        group[period_key][...] = normalize_at_large_cd_geoids(values).astype(values.dtype)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dataset", required=True)
    parser.add_argument("--target-db", required=True)
    parser.add_argument("--output-dataset", required=True)
    parser.add_argument("--period", type=int, default=2024)
    parser.add_argument("--max-iter", type=int, default=300)
    parser.add_argument("--tol", type=float, default=1e-9)
    parser.add_argument(
        "--no-preserve-district-weight-sum",
        dest="preserve_district_weight_sum",
        action="store_false",
        help=(
            "Do not append a per-district household-weight preservation row. "
            "The default preserves district household totals while fitting CD-age targets."
        ),
    )
    parser.set_defaults(preserve_district_weight_sum=True)
    parser.add_argument("--summary-output")
    parser.add_argument("--details-output")
    args = parser.parse_args(argv)

    summary = reweight_h5_to_cd_age_targets(
        input_dataset=args.input_dataset,
        target_db=args.target_db,
        output_dataset=args.output_dataset,
        period=args.period,
        max_iter=args.max_iter,
        tol=args.tol,
        preserve_district_weight_sum=args.preserve_district_weight_sum,
        details_output=args.details_output,
    )
    payload = json.dumps(summary, indent=2, sort_keys=True, allow_nan=False)
    if args.summary_output:
        summary_path = Path(args.summary_output).expanduser().resolve()
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        summary_path.write_text(payload + "\n")
    print(payload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
