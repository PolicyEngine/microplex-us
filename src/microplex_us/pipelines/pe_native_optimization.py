"""Direct PE-native weight optimization for exported PolicyEngine US datasets."""

from __future__ import annotations

import json
import shutil
import subprocess
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

import h5py
import numpy as np

from microplex_us.pipelines.pe_native_scores import (
    _ENHANCED_CPS_BAD_TARGETS,
    build_policyengine_us_data_subprocess_env,
    resolve_policyengine_us_data_repo_root,
)

_PE_NATIVE_BROAD_MATRIX_SCRIPT = """
import json
import sys
from pathlib import Path

import numpy as np
from policyengine_core.data import Dataset

REPO_ROOT = sys.argv[1]
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from policyengine_us import Microsimulation
from policyengine_us_data.utils.loss import build_loss_matrix

BAD_TARGETS = tuple(json.loads(sys.argv[2]))
PERIOD = int(sys.argv[3])
DATASET_PATH = sys.argv[4]
OUTPUT_PREFIX = Path(sys.argv[5])


def dataset_from_path(dataset_path: str, dataset_name: str):
    class LocalDataset(Dataset):
        name = dataset_name
        label = dataset_name
        file_path = dataset_path
        data_format = Dataset.TIME_PERIOD_ARRAYS
        time_period = PERIOD

    return LocalDataset


dataset_cls = dataset_from_path(
    DATASET_PATH,
    Path(DATASET_PATH).stem.replace("-", "_"),
)
loss_matrix, targets_array = build_loss_matrix(dataset_cls, PERIOD)
target_names = np.asarray(loss_matrix.columns)
zero_mask = np.isclose(targets_array, 0.0, atol=0.1)
bad_mask = np.isin(target_names, BAD_TARGETS)
keep_mask = ~(zero_mask | bad_mask)

filtered = loss_matrix.loc[:, keep_mask]
filtered_targets = np.asarray(targets_array[keep_mask], dtype=np.float64)
is_national = np.asarray(filtered.columns.str.startswith("nation/"), dtype=bool)
n_national = int(is_national.sum())
n_state = int((~is_national).sum())
if n_national == 0 or n_state == 0:
    raise ValueError(
        "PE-native broad loss requires both national and state targets after filtering"
    )

normalisation_factor = np.where(
    is_national,
    1.0 / n_national,
    1.0 / n_state,
).astype(np.float64)
inv_mean_normalisation = 1.0 / float(np.mean(normalisation_factor))
per_target_weight = (
    inv_mean_normalisation * normalisation_factor / float(len(filtered_targets))
).astype(np.float64)
denominator = (filtered_targets + 1.0).astype(np.float64)
scaling = np.sqrt(per_target_weight) / denominator
scaled_matrix = filtered.to_numpy(dtype=np.float64) * scaling[np.newaxis, :]
scaled_target = (filtered_targets - 1.0) * scaling

sim = Microsimulation(dataset=dataset_cls)
sim.default_calculation_period = PERIOD
weights = sim.calculate(
    "household_weight",
    map_to="household",
    period=PERIOD,
).values.astype(np.float64)

np.save(OUTPUT_PREFIX.with_suffix(".matrix.npy"), scaled_matrix)
np.save(OUTPUT_PREFIX.with_suffix(".target.npy"), scaled_target)
np.save(OUTPUT_PREFIX.with_suffix(".weights.npy"), weights)
np.save(OUTPUT_PREFIX.with_suffix(".target_unscaled.npy"), filtered_targets)
np.save(OUTPUT_PREFIX.with_suffix(".scaling.npy"), scaling)
with open(OUTPUT_PREFIX.with_suffix(".meta.json"), "w") as handle:
    json.dump(
        {
            "target_names": filtered.columns.tolist(),
            "n_targets_total": int(len(target_names)),
            "n_targets_kept": int(keep_mask.sum()),
            "n_targets_zero_dropped": int(zero_mask.sum()),
            "n_targets_bad_dropped": int(bad_mask.sum()),
            "n_national_targets": n_national,
            "n_state_targets": n_state,
            "weight_sum": float(weights.sum()),
            "candidate_loss_before": float(
                np.square(scaled_matrix.T @ weights - scaled_target).sum()
            ),
        },
        handle,
        sort_keys=True,
    )
""".strip()


@dataclass(frozen=True)
class PolicyEngineUSNativeWeightOptimizationResult:
    metric: str
    period: int
    input_dataset: str
    output_dataset: str
    initial_loss: float
    optimized_loss: float
    loss_delta: float
    initial_weight_sum: float
    optimized_weight_sum: float
    household_count: int
    positive_household_count: int
    budget: int | None
    converged: bool
    iterations: int
    target_names: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "metric": self.metric,
            "period": self.period,
            "input_dataset": self.input_dataset,
            "output_dataset": self.output_dataset,
            "initial_loss": self.initial_loss,
            "optimized_loss": self.optimized_loss,
            "loss_delta": self.loss_delta,
            "initial_weight_sum": self.initial_weight_sum,
            "optimized_weight_sum": self.optimized_weight_sum,
            "household_count": self.household_count,
            "positive_household_count": self.positive_household_count,
            "budget": self.budget,
            "converged": self.converged,
            "iterations": self.iterations,
            "target_names": list(self.target_names),
        }


def _project_to_simplex(values: np.ndarray, total: float) -> np.ndarray:
    """Project onto {x >= 0, sum x = total}."""
    if total < 0:
        raise ValueError("total must be nonnegative")
    if len(values) == 0:
        return values.copy()
    clipped = np.maximum(values.astype(np.float64, copy=False), 0.0)
    current_sum = float(clipped.sum())
    if np.isclose(current_sum, total):
        return clipped
    if total <= 0.0:
        return np.zeros_like(clipped)

    u = np.sort(clipped)[::-1]
    cssv = np.cumsum(u) - total
    rho_candidates = u - cssv / np.arange(1, len(u) + 1) > 0
    if not np.any(rho_candidates):
        projected = np.zeros_like(clipped)
        projected[np.argmax(clipped)] = total
        return projected
    rho = int(np.nonzero(rho_candidates)[0][-1])
    theta = cssv[rho] / float(rho + 1)
    return np.maximum(clipped - theta, 0.0)


def _project_to_budget_simplex(
    values: np.ndarray,
    total: float,
    budget: int | None,
) -> np.ndarray:
    if budget is None or budget >= len(values):
        return _project_to_simplex(values, total)
    if budget <= 0:
        raise ValueError("budget must be positive when provided")
    projected = np.zeros_like(values, dtype=np.float64)
    top_idx = np.argpartition(values, -budget)[-budget:]
    projected[top_idx] = _project_to_simplex(values[top_idx], total)
    return projected


def _estimate_quadratic_lipschitz(matrix: np.ndarray, l2_penalty: float) -> float:
    if matrix.size == 0:
        return max(2.0 * l2_penalty, 1.0)
    n_households = matrix.shape[0]
    vector = np.ones(n_households, dtype=np.float64)
    vector /= np.linalg.norm(vector)
    for _ in range(25):
        transformed = matrix @ (matrix.T @ vector)
        norm = np.linalg.norm(transformed)
        if norm < 1e-12:
            return max(2.0 * l2_penalty, 1.0)
        vector = transformed / norm
    transformed = matrix @ (matrix.T @ vector)
    eigenvalue = float(np.dot(vector, transformed))
    return max(2.0 * eigenvalue + 2.0 * l2_penalty, 1e-6)


def optimize_pe_native_loss_weights(
    *,
    scaled_matrix: np.ndarray,
    scaled_target: np.ndarray,
    initial_weights: np.ndarray,
    budget: int | None = None,
    max_iter: int = 200,
    l2_penalty: float = 0.0,
    tol: float = 1e-8,
    target_total_weight: float | None = None,
    history_callback: Callable[[int, np.ndarray, float], None] | None = None,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Optimize nonnegative household weights directly on the PE-native loss matrix.

    If *target_total_weight* is provided, the simplex projection targets that
    total instead of the initial weight sum.  This allows the optimizer to
    rescale the weight budget (e.g. to match a known population total) while
    simultaneously redistributing weights to minimise the PE-native loss.
    """
    matrix = np.asarray(scaled_matrix, dtype=np.float64)
    target = np.asarray(scaled_target, dtype=np.float64)
    weights0 = np.asarray(initial_weights, dtype=np.float64)
    if matrix.ndim != 2:
        raise ValueError("scaled_matrix must be 2D")
    if target.ndim != 1 or target.shape[0] != matrix.shape[1]:
        raise ValueError("scaled_target must match scaled_matrix target dimension")
    if weights0.ndim != 1 or weights0.shape[0] != matrix.shape[0]:
        raise ValueError("initial_weights must match scaled_matrix household dimension")

    initial_weight_sum = float(weights0.sum())
    total_weight = (
        float(target_total_weight) if target_total_weight is not None else initial_weight_sum
    )
    weights = _project_to_budget_simplex(weights0, total_weight, budget)
    initial_reference = weights.copy()
    lipschitz = _estimate_quadratic_lipschitz(matrix, l2_penalty)
    step_size = 1.0 / lipschitz

    def objective(candidate: np.ndarray) -> float:
        residual = matrix.T @ candidate - target
        base = float(np.dot(residual, residual))
        if l2_penalty > 0.0:
            delta = candidate - initial_reference
            base += float(l2_penalty * np.dot(delta, delta))
        return base

    current_loss = objective(weights)
    loss_history: list[dict[str, float | int]] = [
        {
            "iteration": 0,
            "objective_loss": float(current_loss),
            "weight_sum": float(weights.sum()),
            "positive_household_count": int((weights > 1e-9).sum()),
        }
    ]
    if history_callback is not None:
        history_callback(0, weights, current_loss)
    converged = False
    completed_iter = 0
    total_backtracking_steps = 0
    for iteration in range(1, max_iter + 1):
        residual = matrix.T @ weights - target
        gradient = 2.0 * (matrix @ residual)
        if l2_penalty > 0.0:
            gradient += 2.0 * l2_penalty * (weights - initial_reference)
        completed_iter = iteration

        candidate = weights
        candidate_loss = current_loss
        accepted_descent_step = False
        iteration_step_size = step_size
        for backtrack in range(30):
            trial = _project_to_budget_simplex(
                weights - iteration_step_size * gradient,
                total_weight,
                budget,
            )
            trial_loss = objective(trial)
            if trial_loss <= current_loss:
                candidate = trial
                candidate_loss = trial_loss
                accepted_descent_step = True
                total_backtracking_steps += backtrack
                break
            iteration_step_size *= 0.5
        if not accepted_descent_step:
            converged = True
            break

        improvement = current_loss - candidate_loss
        weights = candidate
        current_loss = candidate_loss
        loss_history.append(
            {
                "iteration": int(iteration),
                "objective_loss": float(current_loss),
                "weight_sum": float(weights.sum()),
                "positive_household_count": int((weights > 1e-9).sum()),
            }
        )
        if history_callback is not None:
            history_callback(iteration, weights, current_loss)
        if improvement < tol * max(1.0, current_loss):
            converged = True
            break

    summary = {
        "initial_loss": float(objective(initial_reference)),
        "optimized_loss": float(current_loss),
        "loss_delta": float(current_loss - objective(initial_reference)),
        "initial_weight_sum": initial_weight_sum,
        "target_total_weight": total_weight,
        "optimized_weight_sum": float(weights.sum()),
        "household_count": int(len(weights)),
        "positive_household_count": int((weights > 1e-9).sum()),
        "budget": None if budget is None else int(budget),
        "iterations": int(completed_iter),
        "converged": bool(converged),
        "step_size": float(step_size),
        "line_search_backtracking_steps": int(total_backtracking_steps),
        "loss_history": loss_history,
    }
    return weights, summary


def rewrite_policyengine_us_dataset_weights(
    *,
    input_dataset_path: str | Path,
    output_dataset_path: str | Path,
    household_weights: np.ndarray,
    period: int = 2024,
) -> Path:
    """Copy a TIME_PERIOD_ARRAYS H5 and replace all exported weight arrays."""
    source = Path(input_dataset_path).expanduser().resolve()
    output = Path(output_dataset_path).expanduser().resolve()
    if source != output:
        shutil.copy2(source, output)

    period_key = str(period)
    weights = np.asarray(household_weights, dtype=np.float32)
    with h5py.File(output, "r+") as handle:
        household_ids = handle["household_id"][period_key][:]
        if len(household_ids) != len(weights):
            raise ValueError("household_weights length does not match household_id array")
        household_map = {
            int(household_id): float(weight)
            for household_id, weight in zip(household_ids, weights, strict=True)
        }
        handle["household_weight"][period_key][...] = weights

        if "person_weight" in handle and "person_household_id" in handle:
            person_households = handle["person_household_id"][period_key][:]
            person_weights = np.array(
                [household_map[int(household_id)] for household_id in person_households],
                dtype=np.float32,
            )
            handle["person_weight"][period_key][...] = person_weights

        person_households = (
            handle["person_household_id"][period_key][:]
            if "person_household_id" in handle
            else None
        )
        for group in ("tax_unit", "spm_unit", "family", "marital_unit"):
            group_weight_name = f"{group}_weight"
            group_id_name = f"{group}_id"
            person_group_name = f"person_{group}_id"
            if (
                group_weight_name not in handle
                or group_id_name not in handle
                or person_group_name not in handle
                or person_households is None
            ):
                continue
            person_group_ids = handle[person_group_name][period_key][:]
            group_to_household: dict[int, int] = {}
            for group_id, household_id in zip(
                person_group_ids,
                person_households,
                strict=True,
            ):
                group_to_household.setdefault(int(group_id), int(household_id))
            group_ids = handle[group_id_name][period_key][:]
            group_weights = np.array(
                [
                    household_map[group_to_household[int(group_id)]]
                    for group_id in group_ids
                ],
                dtype=np.float32,
            )
            handle[group_weight_name][period_key][...] = group_weights
    return output


def optimize_policyengine_us_native_loss_dataset(
    *,
    input_dataset_path: str | Path,
    output_dataset_path: str | Path,
    period: int = 2024,
    budget: int | None = None,
    max_iter: int = 200,
    l2_penalty: float = 0.0,
    tol: float = 1e-8,
    target_total_weight: float | None = None,
    policyengine_us_data_repo: str | Path | None = None,
) -> PolicyEngineUSNativeWeightOptimizationResult:
    """Optimize household weights of an exported PE-US dataset on the broad native loss."""
    resolved_repo = resolve_policyengine_us_data_repo_root(policyengine_us_data_repo)
    env = build_policyengine_us_data_subprocess_env(resolved_repo)
    with TemporaryDirectory(prefix="microplex-us-pe-native-opt-") as temp_dir:
        prefix = Path(temp_dir) / "pe_native_matrix"
        completed = subprocess.run(
            [
                "uv",
                "run",
                "--project",
                str(resolved_repo),
                "python",
                "-c",
                _PE_NATIVE_BROAD_MATRIX_SCRIPT,
                str(resolved_repo),
                json.dumps(_ENHANCED_CPS_BAD_TARGETS),
                str(int(period)),
                str(Path(input_dataset_path).expanduser().resolve()),
                str(prefix),
            ],
            cwd=resolved_repo,
            env=env,
            capture_output=True,
            text=True,
            check=False,
        )
        if completed.returncode != 0:
            detail = completed.stderr.strip() or completed.stdout.strip() or str(
                completed.returncode
            )
            raise RuntimeError(f"PE-native loss-matrix extraction failed: {detail}")

        scaled_matrix = np.load(prefix.with_suffix(".matrix.npy"))
        scaled_target = np.load(prefix.with_suffix(".target.npy"))
        initial_weights = np.load(prefix.with_suffix(".weights.npy"))
        metadata = json.loads(prefix.with_suffix(".meta.json").read_text())

        optimized_weights, summary = optimize_pe_native_loss_weights(
            scaled_matrix=scaled_matrix,
            scaled_target=scaled_target,
            initial_weights=initial_weights,
            budget=budget,
            max_iter=max_iter,
            l2_penalty=l2_penalty,
            tol=tol,
            target_total_weight=target_total_weight,
        )
        rewritten = rewrite_policyengine_us_dataset_weights(
            input_dataset_path=input_dataset_path,
            output_dataset_path=output_dataset_path,
            household_weights=optimized_weights,
            period=period,
        )
        return PolicyEngineUSNativeWeightOptimizationResult(
            metric="enhanced_cps_native_loss_weight_optimization",
            period=int(period),
            input_dataset=str(Path(input_dataset_path).expanduser().resolve()),
            output_dataset=str(rewritten),
            initial_loss=float(summary["initial_loss"]),
            optimized_loss=float(summary["optimized_loss"]),
            loss_delta=float(summary["loss_delta"]),
            initial_weight_sum=float(summary["initial_weight_sum"]),
            optimized_weight_sum=float(summary["optimized_weight_sum"]),
            household_count=int(summary["household_count"]),
            positive_household_count=int(summary["positive_household_count"]),
            budget=summary["budget"],
            converged=bool(summary["converged"]),
            iterations=int(summary["iterations"]),
            target_names=tuple(metadata["target_names"]),
        )


__all__ = [
    "PolicyEngineUSNativeWeightOptimizationResult",
    "optimize_pe_native_loss_weights",
    "optimize_policyengine_us_native_loss_dataset",
    "rewrite_policyengine_us_dataset_weights",
]
