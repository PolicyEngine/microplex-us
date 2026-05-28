"""Benchmark PE-native calibration strategies on a common target surface."""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime
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
    _DEFAULT_PE_NATIVE_BASELINE_CACHE_DIR,
    _ENHANCED_CPS_BAD_TARGETS,
    build_policyengine_us_data_subprocess_env,
    compute_batch_us_pe_native_scores,
    resolve_policyengine_us_data_repo_root,
    validate_policyengine_us_data_runtime,
)


@dataclass(frozen=True)
class CalibrationBenchmarkVariant:
    """One dataset variant to score in a PE-native calibration benchmark."""

    label: str
    method: str
    dataset_path: str
    generated: bool = False
    optimization: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "label": self.label,
            "method": self.method,
            "dataset_path": self.dataset_path,
            "generated": self.generated,
            "optimization": dict(self.optimization),
        }


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


def _reference_aligned_weights(
    household_ids: np.ndarray,
    reference_dataset_path: str | Path,
    *,
    period: int,
) -> tuple[str, np.ndarray | None]:
    reference_ids, reference_weights = _household_weights(
        reference_dataset_path,
        period=period,
    )
    if household_ids.shape == reference_ids.shape and np.array_equal(
        household_ids,
        reference_ids,
    ):
        return "same_order", reference_weights
    if len(np.unique(reference_ids)) != len(reference_ids):
        return "reference_duplicate_household_ids", None
    reference_by_id = {
        int(household_id): float(weight)
        for household_id, weight in zip(reference_ids, reference_weights, strict=True)
    }
    if all(int(household_id) in reference_by_id for household_id in household_ids):
        return (
            "matched_by_household_id",
            np.asarray(
                [reference_by_id[int(household_id)] for household_id in household_ids],
                dtype=np.float64,
            ),
        )
    return "not_comparable", None


def compute_household_weight_diagnostics(
    dataset_path: str | Path,
    *,
    period: int = 2024,
    reference_dataset_path: str | Path | None = None,
) -> dict[str, Any]:
    """Summarize household weight quality and optional distance from a reference."""

    resolved = Path(dataset_path).expanduser().resolve()
    household_ids, weights = _household_weights(resolved, period=period)
    n_households = int(len(weights))
    positive = weights > 0.0
    weight_sum = float(weights.sum())
    square_sum = float(np.dot(weights, weights))
    effective_sample_size = (
        weight_sum * weight_sum / square_sum if square_sum > 0.0 else 0.0
    )
    diagnostics: dict[str, Any] = {
        "dataset_path": str(resolved),
        "period": int(period),
        "household_count": n_households,
        "positive_household_count": int(positive.sum()),
        "zero_household_count": int((weights == 0.0).sum()),
        "negative_household_count": int((weights < 0.0).sum()),
        "weight_sum": weight_sum,
        "weight_mean": float(weights.mean()) if n_households else 0.0,
        "weight_median": float(np.median(weights)) if n_households else 0.0,
        "weight_min": float(weights.min()) if n_households else 0.0,
        "weight_max": float(weights.max()) if n_households else 0.0,
        "weight_p95": float(np.quantile(weights, 0.95)) if n_households else 0.0,
        "weight_p99": float(np.quantile(weights, 0.99)) if n_households else 0.0,
        "max_to_mean_weight_ratio": (
            float(weights.max() / weights.mean())
            if n_households and weights.mean() > 0.0
            else None
        ),
        "effective_sample_size": float(effective_sample_size),
        "effective_sample_size_share": (
            float(effective_sample_size / n_households) if n_households else None
        ),
    }

    if reference_dataset_path is None:
        return diagnostics

    alignment, reference_weights = _reference_aligned_weights(
        household_ids,
        reference_dataset_path,
        period=period,
    )
    diagnostics["reference_dataset_path"] = str(
        Path(reference_dataset_path).expanduser().resolve()
    )
    diagnostics["reference_alignment"] = alignment
    if reference_weights is None:
        return diagnostics

    delta = weights - reference_weights
    reference_sum = float(reference_weights.sum())
    diagnostics.update(
        {
            "reference_weight_sum": reference_sum,
            "weight_sum_delta": float(weight_sum - reference_sum),
            "l1_delta_as_share_of_reference_sum": (
                float(np.abs(delta).sum() / abs(reference_sum))
                if reference_sum != 0.0
                else None
            ),
            "mean_abs_weight_delta": float(np.abs(delta).mean()),
            "rms_weight_delta": float(np.sqrt(np.mean(delta * delta))),
            "max_abs_weight_delta": float(np.abs(delta).max()) if len(delta) else 0.0,
            "changed_household_count": int((np.abs(delta) > 1e-9).sum()),
            "changed_household_share": (
                float((np.abs(delta) > 1e-9).mean()) if len(delta) else None
            ),
        }
    )
    return diagnostics


def _slugify_label(label: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9_.-]+", "-", label.strip()).strip("-")
    return slug or "variant"


def _log(message: str) -> None:
    timestamp = datetime.now().isoformat(timespec="seconds")
    print(f"[{timestamp}] {message}", file=sys.stderr, flush=True)


def _penalty_label(value: float) -> str:
    if value == 0.0:
        return "pe_native_unconstrained"
    return f"pe_native_l2_{value:g}".replace("+", "")


def _parse_existing_candidates(values: Sequence[str] | None) -> dict[str, Path]:
    candidates: dict[str, Path] = {}
    for value in values or ():
        if "=" not in value:
            raise ValueError(
                "--existing-candidate must be formatted as label=/path/to/file.h5"
            )
        label, path = value.split("=", 1)
        label = label.strip()
        if not label:
            raise ValueError("--existing-candidate label cannot be empty")
        candidates[label] = Path(path).expanduser()
    return candidates


def _parse_float_list(value: str | None) -> tuple[float, ...]:
    if value is None:
        return ()
    stripped = value.strip()
    if not stripped:
        return ()
    return tuple(float(item.strip()) for item in stripped.split(",") if item.strip())


def _resolve_target_total_weight(
    *,
    input_dataset_path: str | Path,
    baseline_dataset_path: str | Path,
    period: int,
    target_total_weight: float | None,
    target_total_weight_source: str,
) -> tuple[float | None, str]:
    if target_total_weight is not None:
        return float(target_total_weight), "explicit"
    if target_total_weight_source == "preserve-input":
        return None, "preserve-input"
    if target_total_weight_source == "input":
        _, input_weights = _household_weights(input_dataset_path, period=period)
        return float(input_weights.sum()), "input"
    if target_total_weight_source == "baseline":
        _, baseline_weights = _household_weights(baseline_dataset_path, period=period)
        return float(baseline_weights.sum()), "baseline"
    raise ValueError(
        "target_total_weight_source must be one of preserve-input, input, baseline"
    )


def _extract_pe_native_loss_inputs(
    *,
    input_dataset_path: str | Path,
    period: int,
    policyengine_us_data_repo: str | Path | None,
    policyengine_us_data_python: str | Path | None,
    skip_tax_expenditure_targets: bool,
) -> dict[str, Any]:
    resolved_repo = resolve_policyengine_us_data_repo_root(policyengine_us_data_repo)
    env = build_policyengine_us_data_subprocess_env(resolved_repo)
    if policyengine_us_data_python is not None:
        command = [str(Path(policyengine_us_data_python).expanduser())]
    else:
        command = ["uv", "run", "--project", str(resolved_repo), "python"]
    validate_policyengine_us_data_runtime(
        command,
        repo_root=resolved_repo,
        env=env,
    )
    _log("extracting PE-native loss matrix")
    with TemporaryDirectory(prefix="microplex-us-pe-native-benchmark-") as temp_dir:
        prefix = Path(temp_dir) / "pe_native_matrix"
        started_at = perf_counter()
        completed = subprocess.run(
            [
                *command,
                "-c",
                _PE_NATIVE_BROAD_MATRIX_SCRIPT,
                str(resolved_repo),
                json.dumps(_ENHANCED_CPS_BAD_TARGETS),
                str(int(period)),
                str(Path(input_dataset_path).expanduser().resolve()),
                "1" if skip_tax_expenditure_targets else "0",
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
        _log(f"extracted PE-native loss matrix in {perf_counter() - started_at:.1f}s")
        return {
            "scaled_matrix": np.load(prefix.with_suffix(".matrix.npy")),
            "scaled_target": np.load(prefix.with_suffix(".target.npy")),
            "initial_weights": np.load(prefix.with_suffix(".weights.npy")),
            "metadata": json.loads(prefix.with_suffix(".meta.json").read_text()),
        }


def build_policyengine_us_native_calibration_benchmark(
    *,
    input_dataset_path: str | Path,
    baseline_dataset_path: str | Path,
    output_dir: str | Path,
    period: int = 2024,
    l2_penalties: Sequence[float] = (0.0, 1e-12, 1e-10, 1e-8),
    max_iter: int = 200,
    tol: float = 1e-8,
    budget: int | None = None,
    target_total_weight: float | None = None,
    target_total_weight_source: str = "preserve-input",
    existing_candidates: Mapping[str, str | Path] | None = None,
    policyengine_us_data_repo: str | Path | None = None,
    policyengine_us_data_python: str | Path | None = None,
    batch_households: int | None = None,
    baseline_cache_dir: str | Path | None = _DEFAULT_PE_NATIVE_BASELINE_CACHE_DIR,
    skip_tax_expenditure_targets: bool = False,
    force: bool = False,
) -> dict[str, Any]:
    """Run and score PE-native calibration variants against one baseline."""

    started_at = perf_counter()
    input_path = Path(input_dataset_path).expanduser().resolve()
    baseline_path = Path(baseline_dataset_path).expanduser().resolve()
    destination = Path(output_dir).expanduser().resolve()
    destination.mkdir(parents=True, exist_ok=True)

    resolved_target_total_weight, target_total_weight_resolved_from = (
        _resolve_target_total_weight(
            input_dataset_path=input_path,
            baseline_dataset_path=baseline_path,
            period=period,
            target_total_weight=target_total_weight,
            target_total_weight_source=target_total_weight_source,
        )
    )

    variants: list[CalibrationBenchmarkVariant] = [
        CalibrationBenchmarkVariant(
            label="input",
            method="existing_input",
            dataset_path=str(input_path),
        )
    ]
    for label, path in (existing_candidates or {}).items():
        variants.append(
            CalibrationBenchmarkVariant(
                label=label,
                method="existing_candidate",
                dataset_path=str(Path(path).expanduser().resolve()),
            )
        )

    loss_inputs = (
        _extract_pe_native_loss_inputs(
            input_dataset_path=input_path,
            period=period,
            policyengine_us_data_repo=policyengine_us_data_repo,
            policyengine_us_data_python=policyengine_us_data_python,
            skip_tax_expenditure_targets=skip_tax_expenditure_targets,
        )
        if l2_penalties
        else None
    )

    for penalty in l2_penalties:
        penalty = float(penalty)
        label = _penalty_label(penalty)
        if resolved_target_total_weight is not None:
            label = f"{label}_{target_total_weight_resolved_from}_total"
        output_path = destination / f"{_slugify_label(label)}.h5"
        optimization_path = output_path.with_suffix(".optimization.json")
        if force or not output_path.exists():
            if loss_inputs is None:
                raise RuntimeError("PE-native loss inputs were not extracted")
            _log(f"optimizing {label} with l2_penalty={penalty:g}")
            optimization_started_at = perf_counter()
            optimized_weights, summary = optimize_pe_native_loss_weights(
                scaled_matrix=loss_inputs["scaled_matrix"],
                scaled_target=loss_inputs["scaled_target"],
                initial_weights=loss_inputs["initial_weights"],
                budget=budget,
                max_iter=max_iter,
                l2_penalty=penalty,
                tol=tol,
                target_total_weight=resolved_target_total_weight,
            )
            _log(
                f"optimized {label} in "
                f"{perf_counter() - optimization_started_at:.1f}s; "
                f"loss {summary['initial_loss']:.6g} -> "
                f"{summary['optimized_loss']:.6g}"
            )
            _log(f"rewriting weights for {label}")
            rewritten = rewrite_policyengine_us_dataset_weights(
                input_dataset_path=input_path,
                output_dataset_path=output_path,
                household_weights=optimized_weights,
                period=period,
            )
            optimization = {
                "metric": "enhanced_cps_native_loss_weight_optimization",
                "period": int(period),
                "input_dataset": str(input_path),
                "output_dataset": str(rewritten),
                "initial_loss": float(summary["initial_loss"]),
                "optimized_loss": float(summary["optimized_loss"]),
                "loss_delta": float(summary["loss_delta"]),
                "initial_weight_sum": float(summary["initial_weight_sum"]),
                "optimized_weight_sum": float(summary["optimized_weight_sum"]),
                "household_count": int(summary["household_count"]),
                "positive_household_count": int(
                    summary["positive_household_count"]
                ),
                "budget": summary["budget"],
                "converged": bool(summary["converged"]),
                "iterations": int(summary["iterations"]),
                "target_names": list(loss_inputs["metadata"]["target_names"]),
                "skip_tax_expenditure_targets": bool(
                    loss_inputs["metadata"].get(
                        "skip_tax_expenditure_targets",
                        skip_tax_expenditure_targets,
                    )
                ),
                "l2_penalty": penalty,
                "target_total_weight": resolved_target_total_weight,
                "target_total_weight_resolved_from": target_total_weight_resolved_from,
                "step_size": summary.get("step_size"),
                "history_interval": summary.get("history_interval"),
                "loss_history": summary.get("loss_history", []),
                "reused_existing_output": False,
            }
            optimization_path.write_text(
                json.dumps(optimization, indent=2, sort_keys=True, allow_nan=False)
            )
        else:
            _log(f"reusing existing optimized dataset for {label}")
            optimization = (
                json.loads(optimization_path.read_text())
                if optimization_path.exists()
                else {}
            )
            optimization.update(
                {
                    "l2_penalty": penalty,
                    "target_total_weight": resolved_target_total_weight,
                    "target_total_weight_resolved_from": (
                        target_total_weight_resolved_from
                    ),
                    "reused_existing_output": True,
                }
            )
        variants.append(
            CalibrationBenchmarkVariant(
                label=label,
                method="pe_native_weight_optimization",
                dataset_path=str(output_path.resolve()),
                generated=True,
                optimization=optimization,
            )
        )

    _log(f"scoring {len(variants)} calibration variants")
    scoring_started_at = perf_counter()
    scores = compute_batch_us_pe_native_scores(
        candidate_dataset_paths=[variant.dataset_path for variant in variants],
        baseline_dataset_path=baseline_path,
        period=period,
        policyengine_us_data_repo=policyengine_us_data_repo,
        policyengine_us_data_python=policyengine_us_data_python,
        batch_households=batch_households,
        baseline_cache_dir=baseline_cache_dir,
        skip_tax_expenditure_targets=skip_tax_expenditure_targets,
    )
    _log(f"scored variants in {perf_counter() - scoring_started_at:.1f}s")
    scores_by_dataset = {
        str(Path(score["broad_loss"]["candidate_dataset"]).resolve()): score
        for score in scores
    }

    rows: list[dict[str, Any]] = []
    for variant in variants:
        dataset_key = str(Path(variant.dataset_path).resolve())
        score = scores_by_dataset[dataset_key]
        broad_loss = score["broad_loss"]
        rows.append(
            {
                **variant.to_dict(),
                "score_summary": score["summary"],
                "broad_loss": broad_loss,
                "family_breakdown": score.get("family_breakdown", []),
                "weight_diagnostics": compute_household_weight_diagnostics(
                    variant.dataset_path,
                    period=period,
                    reference_dataset_path=input_path,
                ),
            }
        )

    ranked_rows = sorted(
        rows,
        key=lambda row: row["score_summary"]["candidate_enhanced_cps_native_loss"],
    )
    baseline_loss = (
        float(rows[0]["score_summary"]["baseline_enhanced_cps_native_loss"])
        if rows
        else None
    )
    payload: dict[str, Any] = {
        "schema_version": 1,
        "metric": "pe_native_calibration_strategy_benchmark",
        "period": int(period),
        "input_dataset": str(input_path),
        "baseline_dataset": str(baseline_path),
        "output_dir": str(destination),
        "skip_tax_expenditure_targets": bool(skip_tax_expenditure_targets),
        "target_total_weight": resolved_target_total_weight,
        "target_total_weight_resolved_from": target_total_weight_resolved_from,
        "budget": None if budget is None else int(budget),
        "max_iter": int(max_iter),
        "tol": float(tol),
        "l2_penalties": [float(value) for value in l2_penalties],
        "baseline_enhanced_cps_native_loss": baseline_loss,
        "best_variant_label": ranked_rows[0]["label"] if ranked_rows else None,
        "best_variant_loss": (
            float(
                ranked_rows[0]["score_summary"][
                    "candidate_enhanced_cps_native_loss"
                ]
            )
            if ranked_rows
            else None
        ),
        "variant_count": len(rows),
        "rows": rows,
        "ranking": [
            {
                "label": row["label"],
                "method": row["method"],
                "candidate_enhanced_cps_native_loss": row["score_summary"][
                    "candidate_enhanced_cps_native_loss"
                ],
                "enhanced_cps_native_loss_delta": row["score_summary"][
                    "enhanced_cps_native_loss_delta"
                ],
                "effective_sample_size_share": row["weight_diagnostics"][
                    "effective_sample_size_share"
                ],
                "l1_delta_as_share_of_reference_sum": row["weight_diagnostics"].get(
                    "l1_delta_as_share_of_reference_sum"
                ),
            }
            for row in ranked_rows
        ],
        "elapsed_seconds": perf_counter() - started_at,
    }
    return payload


def write_policyengine_us_native_calibration_benchmark(
    output_path: str | Path,
    **kwargs: Any,
) -> Path:
    """Build a PE-native calibration benchmark and write it as JSON."""

    payload = build_policyengine_us_native_calibration_benchmark(**kwargs)
    destination = Path(output_path).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(payload, indent=2, sort_keys=True))
    return destination


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Benchmark input, existing, unconstrained, and penalized PE-native "
            "calibration variants on the same PE-native broad target surface."
        )
    )
    parser.add_argument("--input-dataset", required=True)
    parser.add_argument("--baseline-dataset", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--output-path",
        help=(
            "Benchmark JSON path. Defaults to "
            "<output-dir>/pe_native_calibration_benchmark.json."
        ),
    )
    parser.add_argument("--period", type=int, default=2024)
    parser.add_argument(
        "--l2-penalties",
        default="0,1e-12,1e-10,1e-8",
        help=(
            "Comma-separated PE-native optimization penalties. "
            "Use an empty string to score only existing datasets."
        ),
    )
    parser.add_argument("--max-iter", type=int, default=200)
    parser.add_argument("--tol", type=float, default=1e-8)
    parser.add_argument("--budget", type=int)
    parser.add_argument("--target-total-weight", type=float)
    parser.add_argument(
        "--target-total-weight-source",
        choices=("preserve-input", "input", "baseline"),
        default="preserve-input",
    )
    parser.add_argument(
        "--existing-candidate",
        action="append",
        help="Add a precomputed variant as label=/path/to/candidate.h5.",
    )
    parser.add_argument("--policyengine-us-data-python")
    parser.add_argument("--policyengine-us-data-repo")
    parser.add_argument("--batch-households", type=int)
    parser.add_argument(
        "--baseline-cache-dir",
        default=str(_DEFAULT_PE_NATIVE_BASELINE_CACHE_DIR),
        help="Pass an empty string to disable PE-native baseline estimate caching.",
    )
    parser.add_argument(
        "--skip-tax-expenditure-targets",
        action="store_true",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Regenerate optimized H5 variants even if outputs already exist.",
    )
    args = parser.parse_args(argv)

    output_dir = Path(args.output_dir).expanduser()
    output_path = (
        Path(args.output_path).expanduser()
        if args.output_path
        else output_dir / "pe_native_calibration_benchmark.json"
    )
    written = write_policyengine_us_native_calibration_benchmark(
        output_path,
        input_dataset_path=args.input_dataset,
        baseline_dataset_path=args.baseline_dataset,
        output_dir=output_dir,
        period=args.period,
        l2_penalties=_parse_float_list(args.l2_penalties),
        max_iter=args.max_iter,
        tol=args.tol,
        budget=args.budget,
        target_total_weight=args.target_total_weight,
        target_total_weight_source=args.target_total_weight_source,
        existing_candidates=_parse_existing_candidates(args.existing_candidate),
        policyengine_us_data_repo=args.policyengine_us_data_repo,
        policyengine_us_data_python=args.policyengine_us_data_python,
        batch_households=args.batch_households,
        baseline_cache_dir=args.baseline_cache_dir or None,
        skip_tax_expenditure_targets=args.skip_tax_expenditure_targets,
        force=args.force,
    )
    print(str(written))
    return 0


__all__ = [
    "CalibrationBenchmarkVariant",
    "build_policyengine_us_native_calibration_benchmark",
    "compute_household_weight_diagnostics",
    "write_policyengine_us_native_calibration_benchmark",
]
