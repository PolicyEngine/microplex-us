"""Direct PE-native weight optimization for exported PolicyEngine US datasets."""

from __future__ import annotations

import json
import math
import shutil
import subprocess
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

import h5py
import numpy as np

from microplex_us.pipelines.pe_native_loss import (
    PENativeLossArrays,
    loss_arrays_from_inputs,
    pe_native_huber_gradient_factor,
    pe_native_huber_loss,
)
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


def patch_policyengine_us_data_uprating_aliases():
    import policyengine_us_data.utils.soi as soi_utils

    original = soi_utils.create_policyengine_uprating_factors_table

    def patched_create_policyengine_uprating_factors_table(*args, **kwargs):
        table = original(*args, **kwargs)
        if (
            "employment_income" not in table.index
            and "employment_income_before_lsr" in table.index
        ):
            table.loc["employment_income"] = table.loc[
                "employment_income_before_lsr"
            ]
        return table

    soi_utils.create_policyengine_uprating_factors_table = (
        patched_create_policyengine_uprating_factors_table
    )


patch_policyengine_us_data_uprating_aliases()


def load_microplex_loss_helpers():
    import importlib.util

    for entry in sys.path:
        candidate = Path(entry) / "microplex_us" / "pipelines" / "pe_native_loss.py"
        if not candidate.exists():
            continue
        spec = importlib.util.spec_from_file_location(
            "microplex_us_pe_native_loss_standalone",
            candidate,
        )
        if spec is None or spec.loader is None:
            continue
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        return module
    raise ModuleNotFoundError("Could not load microplex_us pe_native_loss helper")


_LOSS_HELPERS = load_microplex_loss_helpers()
build_pe_native_loss_arrays = _LOSS_HELPERS.build_pe_native_loss_arrays
pe_native_huber_loss = _LOSS_HELPERS.pe_native_huber_loss

BAD_TARGETS = tuple(json.loads(sys.argv[2]))
PERIOD = int(sys.argv[3])
DATASET_PATH = sys.argv[4]
if len(sys.argv) >= 7:
    SKIP_TAX_EXPENDITURE_TARGETS = sys.argv[5] == "1"
    OUTPUT_PREFIX = Path(sys.argv[6])
else:
    SKIP_TAX_EXPENDITURE_TARGETS = False
    OUTPUT_PREFIX = Path(sys.argv[5])
TARGET_SCOPE_FILTER = sys.argv[7] if len(sys.argv) >= 8 and sys.argv[7] else None


def dataset_from_path(dataset_path: str, dataset_name: str):
    class LocalDataset(Dataset):
        name = dataset_name
        label = dataset_name
        file_path = dataset_path
        data_format = Dataset.TIME_PERIOD_ARRAYS
        time_period = PERIOD

    return LocalDataset


def scope_keep_mask(target_names):
    if TARGET_SCOPE_FILTER is None:
        return np.ones(target_names.shape, dtype=bool)
    if TARGET_SCOPE_FILTER == "national":
        return np.asarray(
            [str(name).startswith("nation/") for name in target_names],
            dtype=bool,
        )
    if TARGET_SCOPE_FILTER == "state":
        return np.asarray(
            [not str(name).startswith("nation/") for name in target_names],
            dtype=bool,
        )
    raise ValueError(f"Unsupported target scope filter: {TARGET_SCOPE_FILTER}")


dataset_cls = dataset_from_path(
    DATASET_PATH,
    Path(DATASET_PATH).stem.replace("-", "_"),
)
loss_matrix, targets_array = build_loss_matrix(dataset_cls, PERIOD)
target_names = np.asarray(loss_matrix.columns)
zero_mask = np.isclose(targets_array, 0.0, atol=0.1)
bad_mask = np.isin(target_names, BAD_TARGETS)
keep_mask = ~(zero_mask | bad_mask) & scope_keep_mask(target_names)

filtered = loss_matrix.loc[:, keep_mask]
filtered_targets = np.asarray(targets_array[keep_mask], dtype=np.float64)
if filtered_targets.size == 0:
    raise ValueError("PE-native loss matrix has no targets after filtering")
is_national = np.asarray(filtered.columns.str.startswith("nation/"), dtype=bool)
n_national = int(is_national.sum())
n_state = int((~is_national).sum())

loss_arrays = build_pe_native_loss_arrays(
    filtered.columns.tolist(),
    filtered_targets,
)
matrix = filtered.to_numpy(dtype=np.float64)
target = loss_arrays.objective_target

sim = Microsimulation(dataset=dataset_cls)
sim.default_calculation_period = PERIOD
weights = sim.calculate(
    "household_weight",
    map_to="household",
    period=PERIOD,
).values.astype(np.float64)

np.save(OUTPUT_PREFIX.with_suffix(".matrix.npy"), matrix)
np.save(OUTPUT_PREFIX.with_suffix(".target.npy"), target)
np.save(OUTPUT_PREFIX.with_suffix(".weights.npy"), weights)
np.save(OUTPUT_PREFIX.with_suffix(".target_unscaled.npy"), filtered_targets)
np.save(OUTPUT_PREFIX.with_suffix(".loss_denominator.npy"), loss_arrays.denominator)
np.save(OUTPUT_PREFIX.with_suffix(".loss_target_weight.npy"), loss_arrays.target_weight)
np.save(OUTPUT_PREFIX.with_suffix(".loss_bucket.npy"), loss_arrays.bucket_keys)
np.save(OUTPUT_PREFIX.with_suffix(".loss_unit.npy"), loss_arrays.unit_keys)
np.save(OUTPUT_PREFIX.with_suffix(".loss_scope.npy"), loss_arrays.scope_keys)
np.save(OUTPUT_PREFIX.with_suffix(".loss_family.npy"), loss_arrays.family_keys)
np.save(OUTPUT_PREFIX.with_suffix(".loss_epsilon.npy"), loss_arrays.epsilon)
with open(OUTPUT_PREFIX.with_suffix(".meta.json"), "w") as handle:
    json.dump(
        {
            **loss_arrays.metadata(),
            "target_names": filtered.columns.tolist(),
            "target_loss_metadata": loss_arrays.sidecar_rows(),
            "n_targets_total": int(len(target_names)),
            "n_targets_kept": int(keep_mask.sum()),
            "n_targets_zero_dropped": int(zero_mask.sum()),
            "n_targets_bad_dropped": int(bad_mask.sum()),
            "n_national_targets": n_national,
            "n_state_targets": n_state,
            "target_scope_filter": TARGET_SCOPE_FILTER,
            "weight_sum": float(weights.sum()),
            "candidate_loss_before": float(
                pe_native_huber_loss(matrix.T @ weights, loss_arrays)
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
    if np.isclose(current_sum, total, rtol=0.0, atol=1e-6):
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
    loss_arrays: PENativeLossArrays | None = None,
    budget: int | None = None,
    max_iter: int = 200,
    l2_penalty: float = 0.0,
    tol: float = 1e-8,
    target_total_weight: float | None = None,
    history_callback: Callable[[int, np.ndarray, float], None] | None = None,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Optimize nonnegative household weights directly on the PE-native loss matrix.

    Algorithm: **projected (proximal) gradient descent** on the least-squares
    PE-native loss ``||matrix.T @ w - target||^2`` (plus an optional L2 term
    toward the initial weights). Each iteration takes a gradient step and
    projects onto the nonnegative budget simplex (``_project_to_budget_simplex``)
    with a Lipschitz-derived step size, backtracking line search, and
    monotone-descent acceptance.

    NOTE on naming: commit history calls this the "APG refit", but it is **not**
    Nesterov-accelerated — there is no momentum/extrapolation term, so it is
    plain projected GD, not accelerated proximal gradient (APG). It is also
    distinct from the dataset-build weight calibration (microcalibrate /
    entropy / pe_l0); this routine is used for the eCPS-replacement PE-native
    *refit and scoring* (symmetric refit). See docs/calibrator-decision.md.

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
    if (
        loss_arrays is not None
        and loss_arrays.objective_target.shape[0] != matrix.shape[1]
    ):
        raise ValueError("loss_arrays target dimension must match scaled_matrix")

    initial_weight_sum = float(weights0.sum())
    total_weight = (
        float(target_total_weight)
        if target_total_weight is not None
        else initial_weight_sum
    )
    weights = _project_to_budget_simplex(weights0, total_weight, budget)
    initial_reference = weights.copy()
    if loss_arrays is None:
        lipschitz_matrix = matrix
    else:
        lipschitz_scale = np.sqrt(loss_arrays.target_weight) / loss_arrays.denominator
        lipschitz_matrix = matrix * lipschitz_scale[np.newaxis, :]
    lipschitz = _estimate_quadratic_lipschitz(lipschitz_matrix, l2_penalty)
    step_size = 1.0 / lipschitz

    def objective(candidate: np.ndarray) -> float:
        estimate = matrix.T @ candidate
        if loss_arrays is None:
            residual = estimate - target
            base = float(np.dot(residual, residual))
        else:
            base = pe_native_huber_loss(estimate, loss_arrays)
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
    momentum = 1.0
    search_weights = weights.copy()
    min_step_size = step_size * 1e-12
    max_step_size = step_size * 1e8

    def gradient_at(candidate: np.ndarray) -> np.ndarray:
        estimate = matrix.T @ candidate
        if loss_arrays is None:
            residual = estimate - target
            gradient = 2.0 * (matrix @ residual)
        else:
            gradient = matrix @ pe_native_huber_gradient_factor(
                estimate,
                loss_arrays,
            )
        if l2_penalty > 0.0:
            gradient += 2.0 * l2_penalty * (candidate - initial_reference)
        return gradient

    for iteration in range(1, max_iter + 1):
        gradient = gradient_at(search_weights)
        completed_iter = iteration

        candidate = weights
        candidate_loss = current_loss
        accepted_descent_step = False
        iteration_step_size = step_size
        accepted_backtrack = 0

        for start, start_gradient in (
            (search_weights, gradient),
            (weights, None),
        ):
            if accepted_descent_step:
                break
            if start_gradient is None:
                start_gradient = gradient_at(start)
                iteration_step_size = step_size
            for backtrack in range(40):
                trial = _project_to_budget_simplex(
                    start - iteration_step_size * start_gradient,
                    total_weight,
                    budget,
                )
                trial_loss = objective(trial)
                if trial_loss <= current_loss:
                    candidate = trial
                    candidate_loss = trial_loss
                    accepted_descent_step = True
                    accepted_backtrack = backtrack
                    total_backtracking_steps += backtrack
                    break
                iteration_step_size *= 0.5
                if iteration_step_size < min_step_size:
                    break
            if not accepted_descent_step:
                momentum = 1.0
                search_weights = weights.copy()
        if not accepted_descent_step:
            converged = True
            break

        improvement = current_loss - candidate_loss
        previous_weights = weights
        weights = candidate
        current_loss = candidate_loss
        if accepted_backtrack == 0:
            step_size = min(iteration_step_size * 1.25, max_step_size)
        else:
            step_size = iteration_step_size
        next_momentum = 0.5 * (1.0 + math.sqrt(1.0 + 4.0 * momentum**2))
        extrapolated = weights + ((momentum - 1.0) / next_momentum) * (
            weights - previous_weights
        )
        search_weights = _project_to_budget_simplex(
            extrapolated,
            total_weight,
            budget,
        )
        if float(np.dot(weights - previous_weights, search_weights - weights)) > 0.0:
            next_momentum = 1.0
            search_weights = weights.copy()
        momentum = next_momentum
        loss_history.append(
            {
                "iteration": int(iteration),
                "objective_loss": float(current_loss),
                "weight_sum": float(weights.sum()),
                "positive_household_count": int((weights > 1e-9).sum()),
                "step_size": float(step_size),
                "backtracking_steps": int(accepted_backtrack),
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
        "method": "monotone_accelerated_projected_gradient",
        "step_size": float(step_size),
        "initial_step_size": float(1.0 / lipschitz),
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
            raise ValueError(
                "household_weights length does not match household_id array"
            )
        household_map = {
            int(household_id): float(weight)
            for household_id, weight in zip(household_ids, weights, strict=True)
        }
        handle["household_weight"][period_key][...] = weights

        if "person_weight" in handle and "person_household_id" in handle:
            person_households = handle["person_household_id"][period_key][:]
            person_weights = np.array(
                [
                    household_map[int(household_id)]
                    for household_id in person_households
                ],
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
                # Production datasets (e.g. the published enhanced CPS) may
                # leave derived entity-weight groups empty because PolicyEngine
                # computes those weights from household_weight at runtime. The
                # group then exists without a value for this period, so skip
                # propagation rather than raising KeyError on [period_key].
                or period_key not in handle[group_weight_name]
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
            detail = (
                completed.stderr.strip()
                or completed.stdout.strip()
                or str(completed.returncode)
            )
            raise RuntimeError(f"PE-native loss-matrix extraction failed: {detail}")

        scaled_matrix = np.load(prefix.with_suffix(".matrix.npy"))
        scaled_target = np.load(prefix.with_suffix(".target.npy"))
        initial_weights = np.load(prefix.with_suffix(".weights.npy"))
        metadata = json.loads(prefix.with_suffix(".meta.json").read_text())
        loss_inputs = {
            "scaled_matrix": scaled_matrix,
            "scaled_target": scaled_target,
            "initial_weights": initial_weights,
            "unscaled_target": np.load(prefix.with_suffix(".target_unscaled.npy")),
            "loss_denominator": np.load(prefix.with_suffix(".loss_denominator.npy")),
            "loss_target_weight": np.load(
                prefix.with_suffix(".loss_target_weight.npy")
            ),
            "loss_bucket": np.load(
                prefix.with_suffix(".loss_bucket.npy"), allow_pickle=True
            ),
            "loss_unit": np.load(
                prefix.with_suffix(".loss_unit.npy"), allow_pickle=True
            ),
            "loss_scope": np.load(
                prefix.with_suffix(".loss_scope.npy"), allow_pickle=True
            ),
            "loss_family": np.load(
                prefix.with_suffix(".loss_family.npy"), allow_pickle=True
            ),
            "loss_epsilon": np.load(prefix.with_suffix(".loss_epsilon.npy")),
            "metadata": metadata,
        }
        loss_arrays = loss_arrays_from_inputs(loss_inputs)

        optimized_weights, summary = optimize_pe_native_loss_weights(
            scaled_matrix=scaled_matrix,
            scaled_target=scaled_target,
            initial_weights=initial_weights,
            loss_arrays=loss_arrays,
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
            metric=str(
                metadata.get(
                    "loss_metric",
                    "enhanced_cps_native_loss_weight_optimization",
                )
            ),
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
