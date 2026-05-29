"""Sound Microplex-vs-eCPS replacement comparison harness."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path
from tempfile import TemporaryDirectory
from time import perf_counter
from typing import Any

import h5py
import numpy as np

from microplex_us.pipelines.pe_native_optimization import (
    _PE_NATIVE_BROAD_MATRIX_SCRIPT,
    optimize_pe_native_loss_weights,
    rewrite_policyengine_us_dataset_weights,
)
from microplex_us.pipelines.pe_native_scores import (
    _ENHANCED_CPS_BAD_TARGETS,
    build_policyengine_us_data_subprocess_env,
    compute_us_pe_native_scores,
    resolve_policyengine_us_data_repo_root,
)
from microplex_us.pipelines.performance import (
    _write_matched_policyengine_us_baseline_dataset,
)
from microplex_us.pipelines.summarize_pe_native_family_drilldown import (
    classify_pe_native_target_family,
)

_PROTECTED_TARGET_PATTERNS: dict[str, tuple[str, ...]] = {
    "ssi": ("ssi", "supplemental_security_income"),
    "snap": ("snap",),
    "wages": ("wage", "employment_income"),
    "self_employment_income": ("self_employment", "business_income"),
    "capital_gains": ("capital_gain", "capital_gains"),
    "interest": ("interest",),
    "dividends": ("dividend",),
    "retirement_income": ("retirement", "pension", "ira", "401k", "403b"),
    "disability": ("disability", "ssdi"),
    "household_net_income": ("household_net_income", "net_income"),
}


def build_sound_ecps_replacement_comparison(
    *,
    candidate_dataset_path: str | Path,
    baseline_dataset_path: str | Path,
    output_dir: str | Path,
    period: int = 2024,
    matched_household_count: int | None = None,
    random_seed: int = 20260529,
    holdout_target_fraction: float = 0.2,
    holdout_target_seed: int = 20260529,
    optimizer_max_iter: int = 200,
    optimizer_tol: float = 1e-8,
    score_consistency_tol: float = 1e-6,
    policyengine_us_data_repo: str | Path | None = None,
    policyengine_us_data_python: str | Path | None = None,
    skip_tax_expenditure_targets: bool = False,
    force: bool = False,
) -> dict[str, Any]:
    """Build a release-contract eCPS comparison payload.

    The comparison intentionally does not accept a one-sided refit. Both the
    candidate and eCPS baseline are first matched to the same household count,
    then refit with the same dense no-gates PE-native objective, then rescored
    through the normal PE-native scorer.
    """

    started_at = perf_counter()
    candidate_path = Path(candidate_dataset_path).expanduser().resolve()
    baseline_path = Path(baseline_dataset_path).expanduser().resolve()
    destination = Path(output_dir).expanduser().resolve()
    destination.mkdir(parents=True, exist_ok=True)

    candidate_household_ids, _ = _household_weights(candidate_path, period=period)
    baseline_household_ids, _ = _household_weights(baseline_path, period=period)
    matched_count = (
        int(matched_household_count)
        if matched_household_count is not None
        else min(len(candidate_household_ids), len(baseline_household_ids))
    )
    if matched_count <= 0:
        raise ValueError("matched_household_count must be positive")
    if matched_count > len(candidate_household_ids):
        raise ValueError("matched_household_count cannot exceed candidate households")
    if matched_count > len(baseline_household_ids):
        raise ValueError("matched_household_count cannot exceed baseline households")

    matched_candidate_path = destination / "candidate_matched.h5"
    matched_baseline_path = destination / "baseline_matched.h5"
    _write_matched_dataset(
        candidate_path,
        matched_candidate_path,
        period=period,
        household_count=matched_count,
        random_seed=random_seed,
        force=force,
    )
    _write_matched_dataset(
        baseline_path,
        matched_baseline_path,
        period=period,
        household_count=matched_count,
        random_seed=random_seed + 1,
        force=force,
    )

    candidate_inputs = _extract_pe_native_loss_inputs(
        input_dataset_path=matched_candidate_path,
        period=period,
        policyengine_us_data_repo=policyengine_us_data_repo,
        policyengine_us_data_python=policyengine_us_data_python,
        skip_tax_expenditure_targets=skip_tax_expenditure_targets,
    )
    baseline_inputs = _extract_pe_native_loss_inputs(
        input_dataset_path=matched_baseline_path,
        period=period,
        policyengine_us_data_repo=policyengine_us_data_repo,
        policyengine_us_data_python=policyengine_us_data_python,
        skip_tax_expenditure_targets=skip_tax_expenditure_targets,
    )
    target_names = _validate_common_targets(candidate_inputs, baseline_inputs)
    holdout_mask = _build_holdout_target_mask(
        target_names,
        fraction=holdout_target_fraction,
        seed=holdout_target_seed,
    )

    refit_config = {
        "method": "deterministic_pe_native_projected_gradient",
        "lambda_l0": 0.0,
        "lambda_l2": 0.0,
        "use_gates": False,
        "max_iter": int(optimizer_max_iter),
        "tol": float(optimizer_tol),
        "target_total_weight": "preserve_input",
    }
    candidate_refit_path = destination / "candidate_refit.h5"
    baseline_refit_path = destination / "baseline_refit.h5"
    candidate_refit = _fit_dense_refit(
        input_dataset_path=matched_candidate_path,
        output_dataset_path=candidate_refit_path,
        loss_inputs=candidate_inputs,
        holdout_mask=holdout_mask,
        period=period,
        max_iter=optimizer_max_iter,
        tol=optimizer_tol,
    )
    baseline_refit = _fit_dense_refit(
        input_dataset_path=matched_baseline_path,
        output_dataset_path=baseline_refit_path,
        loss_inputs=baseline_inputs,
        holdout_mask=holdout_mask,
        period=period,
        max_iter=optimizer_max_iter,
        tol=optimizer_tol,
    )

    pe_native_scores = compute_us_pe_native_scores(
        candidate_dataset_path=candidate_refit_path,
        baseline_dataset_path=baseline_refit_path,
        period=period,
        policyengine_us_data_repo=policyengine_us_data_repo,
        policyengine_us_data_python=policyengine_us_data_python,
    )
    score_summary = dict(pe_native_scores.get("summary") or {})
    candidate_score_loss = score_summary.get("candidate_enhanced_cps_native_loss")
    baseline_score_loss = score_summary.get("baseline_enhanced_cps_native_loss")
    candidate_score_error = _absolute_difference(
        candidate_score_loss,
        candidate_refit["optimized_full_loss"],
    )
    baseline_score_error = _absolute_difference(
        baseline_score_loss,
        baseline_refit["optimized_full_loss"],
    )
    objective_identity_passed = (
        candidate_score_error is not None
        and baseline_score_error is not None
        and candidate_score_error <= score_consistency_tol
        and baseline_score_error <= score_consistency_tol
    )
    ecps_refit_recovery_passed = (
        baseline_refit["optimized_full_loss"]
        <= baseline_refit["initial_full_loss"] + score_consistency_tol
        and (
            baseline_score_loss is None
            or baseline_score_loss
            <= baseline_refit["initial_full_loss"] + score_consistency_tol
        )
    )

    protected_family_losses = _protected_family_losses(
        target_names=target_names,
        candidate_inputs=candidate_inputs,
        baseline_inputs=baseline_inputs,
        candidate_weights=np.asarray(candidate_refit["optimized_weights"]),
        baseline_weights=np.asarray(baseline_refit["optimized_weights"]),
    )

    score_summary.update(
        {
            "candidate_household_count": int(matched_count),
            "baseline_household_count": int(matched_count),
            "matched_household_count": True,
            "candidate_initial_enhanced_cps_native_loss": candidate_refit[
                "initial_full_loss"
            ],
            "baseline_initial_enhanced_cps_native_loss": baseline_refit[
                "initial_full_loss"
            ],
            "candidate_train_loss": candidate_refit["optimized_train_loss"],
            "baseline_train_loss": baseline_refit["optimized_train_loss"],
            "candidate_holdout_loss": candidate_refit["optimized_holdout_loss"],
            "baseline_holdout_loss": baseline_refit["optimized_holdout_loss"],
            "candidate_score_abs_error": candidate_score_error,
            "baseline_score_abs_error": baseline_score_error,
            "candidate_refit_config": refit_config,
            "baseline_refit_config": refit_config,
            "symmetric_refit": True,
            "score_candidate_only": False,
            "refit_objective_matches_scoring": objective_identity_passed,
            "ecps_refit_recovery_passed": ecps_refit_recovery_passed,
            "holdout_target_fraction": float(holdout_target_fraction),
            "holdout_targets": int(holdout_mask.sum()),
            "protected_family_losses": protected_family_losses,
        }
    )
    payload = {
        "schema_version": 1,
        "metric": "sound_ecps_replacement_comparison",
        "period": int(period),
        "candidate_dataset": _dataset_descriptor(candidate_path),
        "baseline_dataset": _dataset_descriptor(baseline_path),
        "matched_datasets": {
            "household_count": int(matched_count),
            "candidate": _dataset_descriptor(matched_candidate_path),
            "baseline": _dataset_descriptor(matched_baseline_path),
            "random_seed": int(random_seed),
        },
        "comparison_contract": {
            "matched_household_count": True,
            "symmetric_refit": True,
            "score_candidate_only": False,
            "refit_objective_matches_scoring": objective_identity_passed,
            "ecps_refit_recovery_passed": ecps_refit_recovery_passed,
            "holdout_target_fraction": float(holdout_target_fraction),
            "holdout_targets": int(holdout_mask.sum()),
            "protected_family_losses": protected_family_losses,
        },
        "summary": score_summary,
        "score": pe_native_scores,
        "candidate_refit": _strip_weights(candidate_refit),
        "baseline_refit": _strip_weights(baseline_refit),
        "target_split": {
            "holdout_target_fraction": float(holdout_target_fraction),
            "holdout_target_seed": int(holdout_target_seed),
            "train_targets": int((~holdout_mask).sum()),
            "holdout_targets": int(holdout_mask.sum()),
            "holdout_target_names": [
                name for name, holdout in zip(target_names, holdout_mask, strict=True) if holdout
            ],
        },
        "refit_config": refit_config,
        "skip_tax_expenditure_targets": bool(skip_tax_expenditure_targets),
        "elapsed_seconds": float(perf_counter() - started_at),
    }
    return payload


def write_sound_ecps_replacement_comparison(
    output_path: str | Path,
    **kwargs: Any,
) -> Path:
    """Write a sound eCPS replacement comparison payload."""

    payload = build_sound_ecps_replacement_comparison(**kwargs)
    destination = Path(output_path).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(payload, indent=2, sort_keys=True))
    return destination


def _write_matched_dataset(
    input_path: Path,
    output_path: Path,
    *,
    period: int,
    household_count: int,
    random_seed: int,
    force: bool,
) -> None:
    if output_path.exists() and not force:
        raise FileExistsError(f"{output_path} already exists; pass --force to replace it")
    _write_matched_policyengine_us_baseline_dataset(
        input_path,
        output_path,
        period=period,
        household_count=household_count,
        random_seed=random_seed,
    )


def _household_weights(
    dataset_path: str | Path,
    *,
    period: int,
) -> tuple[np.ndarray, np.ndarray]:
    path = Path(dataset_path).expanduser().resolve()
    period_key = str(period)
    with h5py.File(path, "r") as handle:
        if "household_id" not in handle or period_key not in handle["household_id"]:
            raise ValueError(f"{path} is missing household_id/{period_key}")
        if (
            "household_weight" not in handle
            or period_key not in handle["household_weight"]
        ):
            raise ValueError(f"{path} is missing household_weight/{period_key}")
        household_ids = np.asarray(handle["household_id"][period_key], dtype=np.int64)
        weights = np.asarray(
            handle["household_weight"][period_key],
            dtype=np.float64,
        )
    if household_ids.shape[0] != weights.shape[0]:
        raise ValueError(f"{path} household_id and household_weight lengths differ")
    return household_ids, weights


def _extract_pe_native_loss_inputs(
    *,
    input_dataset_path: str | Path,
    period: int,
    policyengine_us_data_repo: str | Path | None,
    policyengine_us_data_python: str | Path | None,
    skip_tax_expenditure_targets: bool,
) -> dict[str, Any]:
    if skip_tax_expenditure_targets:
        raise ValueError(
            "sound eCPS replacement comparison uses the exact PE-native broad "
            "loss target surface; skipping tax expenditure targets is unsupported"
        )
    resolved_repo = resolve_policyengine_us_data_repo_root(policyengine_us_data_repo)
    env = build_policyengine_us_data_subprocess_env(resolved_repo)
    command = (
        [str(Path(policyengine_us_data_python).expanduser())]
        if policyengine_us_data_python is not None
        else ["uv", "run", "--project", str(resolved_repo), "python"]
    )
    with TemporaryDirectory(prefix="microplex-us-ecps-comparison-") as temp_dir:
        prefix = Path(temp_dir) / "pe_native_matrix"
        completed = subprocess.run(
            [
                *command,
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
        return {
            "scaled_matrix": np.load(prefix.with_suffix(".matrix.npy")),
            "scaled_target": np.load(prefix.with_suffix(".target.npy")),
            "initial_weights": np.load(prefix.with_suffix(".weights.npy")),
            "metadata": json.loads(prefix.with_suffix(".meta.json").read_text()),
        }


def _validate_common_targets(
    candidate_inputs: dict[str, Any],
    baseline_inputs: dict[str, Any],
) -> list[str]:
    candidate_names = list(candidate_inputs["metadata"].get("target_names", ()))
    baseline_names = list(baseline_inputs["metadata"].get("target_names", ()))
    if candidate_names != baseline_names:
        raise ValueError("candidate and baseline PE-native target names differ")
    candidate_target = np.asarray(candidate_inputs["scaled_target"], dtype=np.float64)
    baseline_target = np.asarray(baseline_inputs["scaled_target"], dtype=np.float64)
    if not np.allclose(candidate_target, baseline_target):
        raise ValueError("candidate and baseline PE-native scaled targets differ")
    return candidate_names


def _build_holdout_target_mask(
    target_names: list[str],
    *,
    fraction: float,
    seed: int,
) -> np.ndarray:
    if fraction <= 0.0 or fraction >= 1.0:
        raise ValueError("holdout_target_fraction must be between 0 and 1")
    families = np.asarray(
        [classify_pe_native_target_family(name) for name in target_names]
    )
    rng = np.random.default_rng(int(seed))
    holdout_mask = np.zeros(len(target_names), dtype=bool)
    for family in sorted(set(families)):
        indices = np.flatnonzero(families == family)
        if len(indices) <= 1:
            continue
        count = int(round(len(indices) * fraction))
        count = max(1, min(count, len(indices) - 1))
        holdout_mask[rng.choice(indices, size=count, replace=False)] = True
    if not bool(holdout_mask.any()):
        raise ValueError("holdout_target_fraction did not select any targets")
    if bool(holdout_mask.all()):
        raise ValueError("holdout split selected every target")
    return holdout_mask


def _fit_dense_refit(
    *,
    input_dataset_path: Path,
    output_dataset_path: Path,
    loss_inputs: dict[str, Any],
    holdout_mask: np.ndarray,
    period: int,
    max_iter: int,
    tol: float,
) -> dict[str, Any]:
    matrix = np.asarray(loss_inputs["scaled_matrix"], dtype=np.float64)
    target = np.asarray(loss_inputs["scaled_target"], dtype=np.float64)
    initial_weights = np.asarray(loss_inputs["initial_weights"], dtype=np.float64)
    train_mask = ~holdout_mask
    loss_curve: list[dict[str, float | int]] = []

    def record_loss_curve(
        iteration: int,
        weights: np.ndarray,
        objective_loss: float,
    ) -> None:
        loss_curve.append(
            {
                "iteration": int(iteration),
                "objective_train_loss": float(objective_loss),
                "full_loss": _objective(matrix, target, weights),
                "train_loss": _objective(
                    matrix[:, train_mask],
                    target[train_mask],
                    weights,
                ),
                "holdout_loss": _objective(
                    matrix[:, holdout_mask],
                    target[holdout_mask],
                    weights,
                ),
                "weight_sum": float(weights.sum()),
                "positive_household_count": int((weights > 1e-9).sum()),
            }
        )

    optimized_weights, optimizer_summary = optimize_pe_native_loss_weights(
        scaled_matrix=matrix[:, train_mask],
        scaled_target=target[train_mask],
        initial_weights=initial_weights,
        budget=None,
        max_iter=max_iter,
        l2_penalty=0.0,
        tol=tol,
        history_callback=record_loss_curve,
    )
    rewrite_policyengine_us_dataset_weights(
        input_dataset_path=input_dataset_path,
        output_dataset_path=output_dataset_path,
        household_weights=optimized_weights,
        period=period,
    )
    return {
        "input_dataset": str(input_dataset_path.resolve()),
        "output_dataset": str(output_dataset_path.resolve()),
        "initial_full_loss": _objective(matrix, target, initial_weights),
        "optimized_full_loss": _objective(matrix, target, optimized_weights),
        "initial_train_loss": _objective(
            matrix[:, train_mask],
            target[train_mask],
            initial_weights,
        ),
        "optimized_train_loss": _objective(
            matrix[:, train_mask],
            target[train_mask],
            optimized_weights,
        ),
        "initial_holdout_loss": _objective(
            matrix[:, holdout_mask],
            target[holdout_mask],
            initial_weights,
        ),
        "optimized_holdout_loss": _objective(
            matrix[:, holdout_mask],
            target[holdout_mask],
            optimized_weights,
        ),
        "initial_weight_sum": float(initial_weights.sum()),
        "optimized_weight_sum": float(optimized_weights.sum()),
        "household_count": int(len(optimized_weights)),
        "positive_household_count": int((optimized_weights > 1e-9).sum()),
        "optimizer_summary": optimizer_summary,
        "loss_curve": loss_curve,
        "optimized_weights": optimized_weights,
    }


def _objective(matrix: np.ndarray, target: np.ndarray, weights: np.ndarray) -> float:
    residual = matrix.T @ weights - target
    return float(np.dot(residual, residual))


def _protected_family_losses(
    *,
    target_names: list[str],
    candidate_inputs: dict[str, Any],
    baseline_inputs: dict[str, Any],
    candidate_weights: np.ndarray,
    baseline_weights: np.ndarray,
) -> dict[str, dict[str, float | int]]:
    candidate_terms = _loss_terms(candidate_inputs, candidate_weights)
    baseline_terms = _loss_terms(baseline_inputs, baseline_weights)
    n_targets = float(len(target_names))
    rows: dict[str, dict[str, float | int]] = {}
    for family, patterns in _PROTECTED_TARGET_PATTERNS.items():
        indices = [
            index
            for index, name in enumerate(target_names)
            if _target_matches_protected_family(name, family, patterns)
        ]
        if not indices:
            continue
        candidate_loss = float(candidate_terms[indices].sum() / n_targets)
        baseline_loss = float(baseline_terms[indices].sum() / n_targets)
        rows[family] = {
            "n_targets": int(len(indices)),
            "candidate_loss": candidate_loss,
            "baseline_loss": baseline_loss,
            "loss_delta": candidate_loss - baseline_loss,
        }
    return rows


def _loss_terms(loss_inputs: dict[str, Any], weights: np.ndarray) -> np.ndarray:
    matrix = np.asarray(loss_inputs["scaled_matrix"], dtype=np.float64)
    target = np.asarray(loss_inputs["scaled_target"], dtype=np.float64)
    residual = matrix.T @ weights - target
    return np.square(residual)


def _target_matches_protected_family(
    target_name: str,
    family: str,
    patterns: tuple[str, ...],
) -> bool:
    normalized = (
        target_name.lower()
        .replace("-", "_")
        .replace(" ", "_")
        .replace("/", "_")
    )
    if family == "wages" and (
        "self_employment" in normalized or "business_income" in normalized
    ):
        return False
    return any(pattern in normalized for pattern in patterns)


def _absolute_difference(left: Any, right: Any) -> float | None:
    if left is None or right is None:
        return None
    return abs(float(left) - float(right))


def _strip_weights(payload: dict[str, Any]) -> dict[str, Any]:
    stripped = dict(payload)
    stripped.pop("optimized_weights", None)
    return stripped


def _dataset_descriptor(path: Path) -> dict[str, Any]:
    return {
        "path": str(path.resolve()),
        "sha256": _sha256(path),
        "size_bytes": int(path.stat().st_size),
    }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Build a sound Microplex-vs-eCPS replacement comparison payload "
            "for mp-300k artifact gates."
        )
    )
    parser.add_argument("--candidate-dataset", required=True)
    parser.add_argument("--baseline-dataset", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--output-path",
        help="Defaults to <output-dir>/sound_ecps_replacement_comparison.json.",
    )
    parser.add_argument("--period", type=int, default=2024)
    parser.add_argument("--matched-household-count", type=int)
    parser.add_argument("--random-seed", type=int, default=20260529)
    parser.add_argument("--holdout-target-fraction", type=float, default=0.2)
    parser.add_argument("--holdout-target-seed", type=int, default=20260529)
    parser.add_argument("--optimizer-max-iter", type=int, default=200)
    parser.add_argument("--optimizer-tol", type=float, default=1e-8)
    parser.add_argument("--score-consistency-tol", type=float, default=1e-6)
    parser.add_argument("--policyengine-us-data-repo")
    parser.add_argument("--policyengine-us-data-python")
    parser.add_argument("--skip-tax-expenditure-targets", action="store_true")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args(argv)

    output_dir = Path(args.output_dir).expanduser()
    output_path = (
        Path(args.output_path).expanduser()
        if args.output_path
        else output_dir / "sound_ecps_replacement_comparison.json"
    )
    written = write_sound_ecps_replacement_comparison(
        output_path,
        candidate_dataset_path=args.candidate_dataset,
        baseline_dataset_path=args.baseline_dataset,
        output_dir=output_dir,
        period=args.period,
        matched_household_count=args.matched_household_count,
        random_seed=args.random_seed,
        holdout_target_fraction=args.holdout_target_fraction,
        holdout_target_seed=args.holdout_target_seed,
        optimizer_max_iter=args.optimizer_max_iter,
        optimizer_tol=args.optimizer_tol,
        score_consistency_tol=args.score_consistency_tol,
        policyengine_us_data_repo=args.policyengine_us_data_repo,
        policyengine_us_data_python=args.policyengine_us_data_python,
        skip_tax_expenditure_targets=args.skip_tax_expenditure_targets,
        force=args.force,
    )
    print(str(written))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "build_sound_ecps_replacement_comparison",
    "write_sound_ecps_replacement_comparison",
]
