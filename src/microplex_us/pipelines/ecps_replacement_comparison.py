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

from microplex_us.pipelines.pe_native_loss import (
    classify_pe_native_target_family,
    loss_arrays_from_inputs,
    pe_native_huber_loss,
    pe_native_huber_loss_terms,
    pe_native_relative_error,
    subset_loss_arrays,
)
from microplex_us.pipelines.pe_native_optimization import (
    _PE_NATIVE_BROAD_MATRIX_SCRIPT,
    optimize_pe_native_loss_weights,
    rewrite_policyengine_us_dataset_weights,
)
from microplex_us.pipelines.pe_native_scores import (
    _ENHANCED_CPS_BAD_TARGETS,
    build_policyengine_us_data_subprocess_env,
    compute_us_pe_native_scores,
    compute_us_pe_native_support_audit,
    resolve_policyengine_us_data_repo_root,
)
from microplex_us.pipelines.performance import (
    _write_matched_policyengine_us_baseline_dataset,
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

_DATASET_DERIVED_COMPARISON_TARGETS: tuple[str, ...] = (
    "nation/source/household_count",
    "nation/source/cps_household_count",
    "nation/source/puf_clone_household_count",
)

_BASELINE_SANITY_MODES: tuple[str, ...] = ("msre", "content")

_PRODUCTION_BASELINE_REQUIRED_NONZERO_COLUMNS: tuple[str, ...] = (
    "social_security_retirement",
    "social_security_disability",
    "employment_income_before_lsr",
)


def _comparison_bad_targets() -> tuple[str, ...]:
    return tuple(
        dict.fromkeys(
            (
                *_ENHANCED_CPS_BAD_TARGETS,
                *_DATASET_DERIVED_COMPARISON_TARGETS,
            )
        )
    )


class ComparisonGateError(ValueError):
    """Raised when a comparison input or result fails a validity gate.

    These gates exist so the harness refuses to emit a misleading verdict
    instead of relying on a human noticing a mis-scored baseline or a no-op
    refit. Every recurring comparison failure should add a gate here.
    """


def _assert_refit_effective(
    label: str, refit: dict[str, Any], min_reduction: float
) -> None:
    """Fail if a refit did not move at all (a frozen no-op refit).

    A frozen refit (optimized loss == initial loss) means that side was never
    actually reweighted, so its loss is meaningless for comparison -- usually a
    degenerate loss matrix or a total-weight/population mismatch under
    ``preserve_input``. A refit that moves the loss is effective even if the
    full-set loss rises slightly: the refit minimizes the train objective, so an
    already-well-calibrated dataset can legitimately see full loss tick up from
    the held-out split. Only a frozen no-movement refit is a failure.
    """
    if not _refit_moved(refit, min_reduction):
        initial = float(refit["initial_full_loss"])
        optimized = float(refit["optimized_full_loss"])
        raise ComparisonGateError(
            f"{label} refit was a no-op: optimized loss {optimized:.6g} is "
            f"unchanged from initial {initial:.6g} (no movement beyond "
            f"{min_reduction:g}). The refit never reweighted this dataset, so the "
            f"comparison is meaningless -- likely a degenerate loss matrix or a "
            f"total-weight/population mismatch under preserve_input. Pass "
            f"assert_refit_effective=False only to deliberately accept this."
        )


def _refit_moved(refit: dict[str, Any], min_reduction: float) -> bool:
    initial = float(refit["initial_full_loss"])
    optimized = float(refit["optimized_full_loss"])
    return abs(optimized - initial) > float(min_reduction)


def _assert_baseline_sane(
    score_summary: dict[str, Any], max_msre: float
) -> dict[str, Any]:
    """Fail if the production eCPS baseline scores anomalously on this surface.

    A correctly-targeted production eCPS fits its own target surface closely; a
    large unweighted MSRE means the target DB/scorer is wrong for this baseline
    (e.g. an ad-hoc local scorer), so any verdict against it is invalid.
    """
    msre = score_summary.get("baseline_unweighted_msre")
    if msre is None:
        return {
            "mode": "msre",
            "status": "skipped",
            "reason": "baseline_unweighted_msre_absent",
        }
    if float(msre) > max_msre:
        raise ComparisonGateError(
            f"Baseline (production eCPS) scores anomalously on this target "
            f"surface: unweighted MSRE {float(msre):.3f} > {max_msre:g}. A "
            f"correctly-targeted production eCPS scores low (~0.2); a large value "
            f"means the target DB/scorer does not match the baseline, so the "
            f"comparison is invalid. Use the production target surface, or pass "
            f"assert_baseline_sane=False only to deliberately accept this."
        )
    return {
        "mode": "msre",
        "status": "passed",
        "baseline_unweighted_msre": float(msre),
        "max_baseline_unweighted_msre": float(max_msre),
    }


def _assert_production_baseline_content_sane(
    baseline_dataset_path: str | Path,
    *,
    period: int,
    required_nonzero_columns: tuple[str, ...] = (
        _PRODUCTION_BASELINE_REQUIRED_NONZERO_COLUMNS
    ),
) -> dict[str, Any]:
    """Fail if the production eCPS baseline H5 is missing required content.

    This is the right sanity gate for broad external target surfaces where a
    high eCPS MSRE may be the comparison's actual signal, not proof of a broken
    scorer. It still catches the known broken-local-eCPS failure mode by
    requiring core production columns to be present and nonzero.
    """

    path = Path(baseline_dataset_path).expanduser().resolve()
    period_key = str(period)
    missing: list[str] = []
    zero_or_nonfinite: list[str] = []
    column_summaries: dict[str, dict[str, Any]] = {}
    with h5py.File(path, "r") as handle:
        for column in required_nonzero_columns:
            if column not in handle or period_key not in handle[column]:
                missing.append(f"{column}/{period_key}")
                continue
            values = np.asarray(handle[column][period_key], dtype=np.float64)
            finite = bool(np.isfinite(values).all())
            abs_sum = float(np.abs(values).sum()) if finite else float("nan")
            nonzero_count = int(np.count_nonzero(values)) if finite else 0
            column_summaries[column] = {
                "abs_sum": abs_sum,
                "nonzero_count": nonzero_count,
                "finite": finite,
            }
            if not finite or abs_sum <= 0.0 or nonzero_count <= 0:
                zero_or_nonfinite.append(f"{column}/{period_key}")
    if missing or zero_or_nonfinite:
        details = []
        if missing:
            details.append(f"missing {', '.join(missing)}")
        if zero_or_nonfinite:
            details.append(f"zero_or_nonfinite {', '.join(zero_or_nonfinite)}")
        raise ComparisonGateError(
            "Production eCPS baseline content sanity failed: "
            + "; ".join(details)
            + ". Use the verified production eCPS blob, not a broken local H5."
        )
    return {
        "mode": "content",
        "status": "passed",
        "period": int(period),
        "required_nonzero_columns": column_summaries,
    }


def build_sound_ecps_replacement_comparison(
    *,
    candidate_dataset_path: str | Path,
    baseline_dataset_path: str | Path,
    output_dir: str | Path,
    period: int = 2024,
    matched_household_count: int | None = None,
    random_seed: int = 20260529,
    matched_sample_method: str = "uniform",
    holdout_target_fraction: float = 0.2,
    holdout_target_seed: int = 20260529,
    optimizer_max_iter: int = 200,
    optimizer_tol: float = 1e-8,
    score_consistency_tol: float = 1e-6,
    target_diagnostics_top_k: int = 50,
    include_support_audit: bool = True,
    policyengine_us_data_repo: str | Path | None = None,
    policyengine_us_data_python: str | Path | None = None,
    policyengine_targets_db_path: str | Path | None = None,
    skip_tax_expenditure_targets: bool = False,
    target_scope: str = "all",
    exact_rescore: bool = False,
    force: bool = False,
    assert_refit_effective: bool = True,
    min_refit_loss_reduction: float = 1e-9,
    assert_baseline_sane: bool = True,
    baseline_sanity_mode: str = "msre",
    max_baseline_unweighted_msre: float = 2.0,
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
    resolved_targets_db = (
        Path(policyengine_targets_db_path).expanduser().resolve()
        if policyengine_targets_db_path is not None
        else None
    )
    if resolved_targets_db is not None and not resolved_targets_db.exists():
        raise FileNotFoundError(
            f"PolicyEngine target DB not found: {resolved_targets_db}"
        )

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
        sample_method=matched_sample_method,
        force=force,
    )
    _write_matched_dataset(
        baseline_path,
        matched_baseline_path,
        period=period,
        household_count=matched_count,
        random_seed=random_seed + 1,
        sample_method=matched_sample_method,
        force=force,
    )

    candidate_inputs = _extract_pe_native_loss_inputs(
        input_dataset_path=matched_candidate_path,
        period=period,
        policyengine_us_data_repo=policyengine_us_data_repo,
        policyengine_us_data_python=policyengine_us_data_python,
        policyengine_targets_db_path=resolved_targets_db,
        skip_tax_expenditure_targets=skip_tax_expenditure_targets,
    )
    baseline_inputs = _extract_pe_native_loss_inputs(
        input_dataset_path=matched_baseline_path,
        period=period,
        policyengine_us_data_repo=policyengine_us_data_repo,
        policyengine_us_data_python=policyengine_us_data_python,
        policyengine_targets_db_path=resolved_targets_db,
        skip_tax_expenditure_targets=skip_tax_expenditure_targets,
    )
    candidate_inputs = _filter_loss_inputs_by_scope(
        candidate_inputs,
        target_scope=target_scope,
    )
    baseline_inputs = _filter_loss_inputs_by_scope(
        baseline_inputs,
        target_scope=target_scope,
    )
    target_names = _validate_common_targets(candidate_inputs, baseline_inputs)
    if exact_rescore and target_scope != "all":
        raise ValueError("exact_rescore is only supported for target_scope='all'")
    holdout_mask = _build_holdout_target_mask(
        target_names,
        fraction=holdout_target_fraction,
        seed=holdout_target_seed,
    )

    refit_config = {
        "method": "monotone_accelerated_projected_gradient",
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

    if assert_refit_effective:
        _assert_refit_effective("candidate", candidate_refit, min_refit_loss_reduction)
        _assert_refit_effective("baseline", baseline_refit, min_refit_loss_reduction)
    candidate_refit_effective_passed = _refit_moved(
        candidate_refit, min_refit_loss_reduction
    )
    baseline_refit_effective_passed = _refit_moved(
        baseline_refit, min_refit_loss_reduction
    )

    protected_family_losses = _protected_family_losses(
        target_names=target_names,
        candidate_inputs=candidate_inputs,
        baseline_inputs=baseline_inputs,
        candidate_weights=np.asarray(candidate_refit["optimized_weights"]),
        baseline_weights=np.asarray(baseline_refit["optimized_weights"]),
    )
    target_diagnostics = _target_loss_diagnostics(
        target_names=target_names,
        candidate_inputs=candidate_inputs,
        baseline_inputs=baseline_inputs,
        candidate_weights=np.asarray(candidate_refit["optimized_weights"]),
        baseline_weights=np.asarray(baseline_refit["optimized_weights"]),
        holdout_mask=holdout_mask,
        top_k=target_diagnostics_top_k,
    )

    if exact_rescore:
        pe_native_scores = compute_us_pe_native_scores(
            candidate_dataset_path=candidate_refit_path,
            baseline_dataset_path=baseline_refit_path,
            period=period,
            policyengine_us_data_repo=policyengine_us_data_repo,
            policyengine_us_data_python=policyengine_us_data_python,
            policyengine_targets_db_path=resolved_targets_db,
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
        score_source = "exact_policyengine_rescore"
        exact_rescore_status = "completed"
    else:
        score_summary = _refit_matrix_score_summary(
            target_names=target_names,
            candidate_inputs=candidate_inputs,
            baseline_inputs=baseline_inputs,
            candidate_refit=candidate_refit,
            baseline_refit=baseline_refit,
            target_diagnostics=target_diagnostics,
        )
        pe_native_scores = _refit_matrix_score_payload(
            period=period,
            candidate_dataset_path=candidate_refit_path,
            baseline_dataset_path=baseline_refit_path,
            summary=score_summary,
            target_diagnostics=target_diagnostics,
        )
        candidate_score_loss = score_summary.get("candidate_enhanced_cps_native_loss")
        baseline_score_loss = score_summary.get("baseline_enhanced_cps_native_loss")
        candidate_score_error = 0.0
        baseline_score_error = 0.0
        objective_identity_passed = True
        score_source = "refit_loss_matrix"
        exact_rescore_status = "skipped"

    if baseline_sanity_mode not in _BASELINE_SANITY_MODES:
        raise ValueError(
            "baseline_sanity_mode must be one of " + ", ".join(_BASELINE_SANITY_MODES)
        )
    baseline_sanity: dict[str, Any]
    if assert_baseline_sane:
        if baseline_sanity_mode == "msre":
            baseline_sanity = _assert_baseline_sane(
                score_summary, max_baseline_unweighted_msre
            )
        else:
            baseline_sanity = _assert_production_baseline_content_sane(
                baseline_path,
                period=period,
            )
    else:
        baseline_sanity = {
            "mode": baseline_sanity_mode,
            "status": "skipped",
        }

    ecps_refit_recovery_passed = baseline_refit[
        "optimized_full_loss"
    ] <= baseline_refit["initial_full_loss"] + score_consistency_tol and (
        baseline_score_loss is None
        or baseline_score_loss
        <= baseline_refit["initial_full_loss"] + score_consistency_tol
    )
    support_audit = (
        compute_us_pe_native_support_audit(
            candidate_dataset_path=candidate_refit_path,
            baseline_dataset_path=baseline_refit_path,
            period=period,
            policyengine_us_data_repo=policyengine_us_data_repo,
            policyengine_us_data_python=policyengine_us_data_python,
        )
        if include_support_audit
        else None
    )
    support_audit_summary = (
        _support_audit_summary(support_audit) if support_audit is not None else None
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
            "score_source": score_source,
            "exact_rescore_requested": bool(exact_rescore),
            "exact_rescore_status": exact_rescore_status,
            "candidate_refit_effective_passed": candidate_refit_effective_passed,
            "baseline_refit_effective_passed": baseline_refit_effective_passed,
            "ecps_refit_effective_passed": baseline_refit_effective_passed,
            "candidate_refit_config": refit_config,
            "baseline_refit_config": refit_config,
            "symmetric_refit": True,
            "score_candidate_only": False,
            "refit_objective_matches_scoring": objective_identity_passed,
            "ecps_refit_recovery_passed": ecps_refit_recovery_passed,
            "holdout_target_fraction": float(holdout_target_fraction),
            "holdout_targets": int(holdout_mask.sum()),
            "target_scope_filter": target_scope,
            "protected_family_losses": protected_family_losses,
            "target_diagnostics": target_diagnostics["summary"],
            "support_audit": support_audit_summary,
            "baseline_sanity": baseline_sanity,
            "policyengine_targets_db": (
                _dataset_descriptor(resolved_targets_db)
                if resolved_targets_db is not None
                else None
            ),
        }
    )
    frozen_baseline_certificate = _frozen_ecps_baseline_certificate(
        baseline_dataset_path=baseline_path,
        policyengine_targets_db_path=resolved_targets_db,
        policyengine_us_data_repo=policyengine_us_data_repo,
        period=period,
        target_names=target_names,
        target_scope=target_scope,
        holdout_target_fraction=holdout_target_fraction,
        holdout_target_seed=holdout_target_seed,
        matched_sample_method=matched_sample_method,
        refit_config=refit_config,
        skip_tax_expenditure_targets=skip_tax_expenditure_targets,
        exact_rescore=exact_rescore,
        score_source=score_source,
        baseline_sanity=baseline_sanity,
        score_summary=score_summary,
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
            "sample_method": matched_sample_method,
        },
        "comparison_contract": {
            "matched_household_count": True,
            "symmetric_refit": True,
            "score_candidate_only": False,
            "refit_objective_matches_scoring": objective_identity_passed,
            "ecps_refit_recovery_passed": ecps_refit_recovery_passed,
            "ecps_refit_effective_passed": baseline_refit_effective_passed,
            "holdout_target_fraction": float(holdout_target_fraction),
            "holdout_targets": int(holdout_mask.sum()),
            "target_scope_filter": target_scope,
            "protected_family_losses": protected_family_losses,
        },
        "frozen_ecps_baseline_certificate": frozen_baseline_certificate,
        "entity_structure": {
            "candidate_source": _entity_structure_summary(
                candidate_path,
                period=period,
            ),
            "baseline_source": _entity_structure_summary(
                baseline_path,
                period=period,
            ),
            "candidate_matched": _entity_structure_summary(
                matched_candidate_path,
                period=period,
            ),
            "baseline_matched": _entity_structure_summary(
                matched_baseline_path,
                period=period,
            ),
            "candidate_refit": _entity_structure_summary(
                candidate_refit_path,
                period=period,
            ),
            "baseline_refit": _entity_structure_summary(
                baseline_refit_path,
                period=period,
            ),
        },
        "summary": score_summary,
        "score": pe_native_scores,
        "target_diagnostics": target_diagnostics,
        "support_audit": support_audit,
        "candidate_refit": _strip_weights(candidate_refit),
        "baseline_refit": _strip_weights(baseline_refit),
        "target_split": {
            "holdout_target_fraction": float(holdout_target_fraction),
            "holdout_target_seed": int(holdout_target_seed),
            "target_scope_filter": target_scope,
            "train_targets": int((~holdout_mask).sum()),
            "holdout_targets": int(holdout_mask.sum()),
            "holdout_target_names": [
                name
                for name, holdout in zip(target_names, holdout_mask, strict=True)
                if holdout
            ],
        },
        "refit_config": refit_config,
        "skip_tax_expenditure_targets": bool(skip_tax_expenditure_targets),
        "elapsed_seconds": float(perf_counter() - started_at),
    }
    return payload


def write_sound_ecps_replacement_comparison(
    output_path: str | Path,
    target_diagnostics_path: str | Path | None = None,
    support_audit_path: str | Path | None = None,
    **kwargs: Any,
) -> Path:
    """Write a sound eCPS replacement comparison payload."""

    payload = build_sound_ecps_replacement_comparison(**kwargs)
    destination = Path(output_path).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    diagnostics_destination = (
        Path(target_diagnostics_path).expanduser().resolve()
        if target_diagnostics_path is not None
        else destination.parent / "target_loss_diagnostics.json"
    )
    diagnostics_destination.parent.mkdir(parents=True, exist_ok=True)
    diagnostics_destination.write_text(
        json.dumps(payload["target_diagnostics"], indent=2, sort_keys=True)
    )
    payload.setdefault("artifacts", {})["target_loss_diagnostics"] = (
        _dataset_descriptor(diagnostics_destination)
    )
    support_audit = payload.get("support_audit")
    if support_audit is not None:
        support_destination = (
            Path(support_audit_path).expanduser().resolve()
            if support_audit_path is not None
            else destination.parent / "support_audit.json"
        )
        support_destination.parent.mkdir(parents=True, exist_ok=True)
        support_destination.write_text(
            json.dumps(support_audit, indent=2, sort_keys=True)
        )
        payload.setdefault("artifacts", {})["support_audit"] = _dataset_descriptor(
            support_destination
        )
    destination.write_text(json.dumps(payload, indent=2, sort_keys=True))
    return destination


def _write_matched_dataset(
    input_path: Path,
    output_path: Path,
    *,
    period: int,
    household_count: int,
    random_seed: int,
    sample_method: str,
    force: bool,
) -> None:
    if output_path.exists() and not force:
        raise FileExistsError(
            f"{output_path} already exists; pass --force to replace it"
        )
    _write_matched_policyengine_us_baseline_dataset(
        input_path,
        output_path,
        period=period,
        household_count=household_count,
        random_seed=random_seed,
        sample_method=sample_method,
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


def _entity_structure_summary(
    dataset_path: str | Path,
    *,
    period: int,
) -> dict[str, Any]:
    path = Path(dataset_path).expanduser().resolve()
    period_key = str(period)
    with h5py.File(path, "r") as handle:
        household_ids = _read_period_array(handle, "household_id", period_key)
        person_ids = _read_period_array(handle, "person_id", period_key)
        person_household_ids = _read_period_array(
            handle,
            "person_household_id",
            period_key,
        )
        if person_ids.shape[0] != person_household_ids.shape[0]:
            raise ValueError(f"{path} person_id and person_household_id lengths differ")

        household_count = int(household_ids.shape[0])
        summary: dict[str, Any] = {
            "dataset": str(path),
            "period": int(period),
            "household_count": household_count,
            "person_count": int(person_ids.shape[0]),
        }
        for entity in ("tax_unit", "spm_unit", "family", "marital_unit"):
            plural = _ENTITY_PLURALS[entity]
            entity_summary = _entity_membership_summary(
                handle,
                entity=entity,
                period_key=period_key,
                person_household_ids=person_household_ids,
                household_count=household_count,
                dataset_path=path,
            )
            summary[entity] = entity_summary
            summary[f"{entity}_count"] = entity_summary["unit_count"]
            summary[f"{plural}_per_household"] = entity_summary["units_per_household"]
    return summary


_ENTITY_PLURALS = {
    "tax_unit": "tax_units",
    "spm_unit": "spm_units",
    "family": "families",
    "marital_unit": "marital_units",
}


def _read_period_array(
    handle: h5py.File,
    variable: str,
    period_key: str,
) -> np.ndarray:
    if variable not in handle or period_key not in handle[variable]:
        raise ValueError(f"Dataset is missing {variable}/{period_key}")
    return np.asarray(handle[variable][period_key], dtype=np.int64)


def _entity_membership_summary(
    handle: h5py.File,
    *,
    entity: str,
    period_key: str,
    person_household_ids: np.ndarray,
    household_count: int,
    dataset_path: Path,
) -> dict[str, Any]:
    entity_ids = _read_period_array(handle, f"{entity}_id", period_key)
    person_entity_ids = _read_period_array(
        handle,
        f"person_{entity}_id",
        period_key,
    )
    if person_entity_ids.shape[0] != person_household_ids.shape[0]:
        raise ValueError(
            f"{dataset_path} person_{entity}_id and person_household_id lengths differ"
        )
    unique_entity_ids = np.unique(entity_ids)
    duplicate_unit_id_count = int(entity_ids.shape[0] - unique_entity_ids.shape[0])
    unique_person_entity_ids, inverse = np.unique(
        person_entity_ids,
        return_inverse=True,
    )
    member_counts = np.bincount(inverse)
    singleton_count = int(np.count_nonzero(member_counts == 1))
    empty_unit_count = int(
        np.setdiff1d(unique_entity_ids, unique_person_entity_ids).size
    )
    missing_referenced_unit_count = int(
        np.setdiff1d(unique_person_entity_ids, unique_entity_ids).size
    )
    cross_household_count = _cross_household_entity_count(
        inverse,
        person_household_ids,
    )
    unit_count = int(entity_ids.shape[0])
    return {
        "unit_count": unit_count,
        "person_membership_count": int(person_entity_ids.shape[0]),
        "duplicate_unit_id_count": duplicate_unit_id_count,
        "units_per_household": (
            float(unit_count / household_count) if household_count else None
        ),
        "singleton_unit_count": singleton_count,
        "singleton_unit_share": (
            float(singleton_count / unit_count) if unit_count else None
        ),
        "empty_unit_count": empty_unit_count,
        "missing_referenced_unit_count": missing_referenced_unit_count,
        "cross_household_unit_count": cross_household_count,
    }


def _cross_household_entity_count(
    entity_inverse: np.ndarray,
    person_household_ids: np.ndarray,
) -> int:
    if entity_inverse.size == 0:
        return 0
    order = np.argsort(entity_inverse, kind="stable")
    sorted_entity = entity_inverse[order]
    sorted_household = person_household_ids[order]
    boundaries = np.concatenate(
        (
            np.asarray([0]),
            np.flatnonzero(np.diff(sorted_entity)) + 1,
            np.asarray([sorted_entity.size]),
        )
    )
    cross_household_count = 0
    for start, stop in zip(boundaries[:-1], boundaries[1:], strict=True):
        if np.unique(sorted_household[start:stop]).size > 1:
            cross_household_count += 1
    return cross_household_count


def _extract_pe_native_loss_inputs(
    *,
    input_dataset_path: str | Path,
    period: int,
    policyengine_us_data_repo: str | Path | None,
    policyengine_us_data_python: str | Path | None,
    policyengine_targets_db_path: str | Path | None,
    skip_tax_expenditure_targets: bool,
) -> dict[str, Any]:
    if skip_tax_expenditure_targets:
        raise ValueError(
            "sound eCPS replacement comparison uses the exact PE-native broad "
            "loss target surface; skipping tax expenditure targets is unsupported"
        )
    resolved_repo = resolve_policyengine_us_data_repo_root(policyengine_us_data_repo)
    env = build_policyengine_us_data_subprocess_env(resolved_repo)
    resolved_targets_db = (
        Path(policyengine_targets_db_path).expanduser().resolve()
        if policyengine_targets_db_path is not None
        else None
    )
    if resolved_targets_db is not None and not resolved_targets_db.exists():
        raise FileNotFoundError(
            f"PolicyEngine target DB not found: {resolved_targets_db}"
        )
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
                json.dumps(_comparison_bad_targets()),
                str(int(period)),
                str(Path(input_dataset_path).expanduser().resolve()),
                "1" if skip_tax_expenditure_targets else "0",
                str(prefix),
                "",
                str(resolved_targets_db) if resolved_targets_db is not None else "",
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
        return {
            "scaled_matrix": np.load(prefix.with_suffix(".matrix.npy")),
            "scaled_target": np.load(prefix.with_suffix(".target.npy")),
            "initial_weights": np.load(prefix.with_suffix(".weights.npy")),
            "unscaled_target": _load_optional_array(
                prefix.with_suffix(".target_unscaled.npy")
            ),
            "scaling": _load_optional_array(prefix.with_suffix(".scaling.npy")),
            "loss_denominator": _load_optional_array(
                prefix.with_suffix(".loss_denominator.npy")
            ),
            "loss_target_weight": _load_optional_array(
                prefix.with_suffix(".loss_target_weight.npy")
            ),
            "loss_bucket": _load_optional_array(
                prefix.with_suffix(".loss_bucket.npy"),
                allow_pickle=True,
            ),
            "loss_unit": _load_optional_array(
                prefix.with_suffix(".loss_unit.npy"),
                allow_pickle=True,
            ),
            "loss_scope": _load_optional_array(
                prefix.with_suffix(".loss_scope.npy"),
                allow_pickle=True,
            ),
            "loss_family": _load_optional_array(
                prefix.with_suffix(".loss_family.npy"),
                allow_pickle=True,
            ),
            "loss_epsilon": _load_optional_array(
                prefix.with_suffix(".loss_epsilon.npy")
            ),
            "metadata": json.loads(prefix.with_suffix(".meta.json").read_text()),
        }


def _load_optional_array(
    path: Path, *, allow_pickle: bool = False
) -> np.ndarray | None:
    return np.load(path, allow_pickle=allow_pickle) if path.exists() else None


def _filter_loss_inputs_by_scope(
    loss_inputs: dict[str, Any],
    *,
    target_scope: str,
) -> dict[str, Any]:
    if target_scope not in {"all", "national", "state"}:
        raise ValueError("target_scope must be one of all, national, or state")
    if target_scope == "all":
        return loss_inputs

    metadata = dict(loss_inputs["metadata"])
    target_names = np.asarray(metadata.get("target_names", ()), dtype=object)
    if target_names.size == 0:
        raise ValueError("PE-native loss inputs do not include target names")

    scope = loss_inputs.get("loss_scope")
    if scope is not None:
        keep_mask = np.asarray(scope, dtype=object) == target_scope
    elif target_scope == "national":
        keep_mask = np.asarray(
            [str(name).startswith("nation/") for name in target_names],
            dtype=bool,
        )
    else:
        keep_mask = np.asarray(
            [not str(name).startswith("nation/") for name in target_names],
            dtype=bool,
        )
    if not bool(keep_mask.any()):
        raise ValueError(f"target_scope={target_scope!r} selected no targets")

    filtered = dict(loss_inputs)
    filtered["scaled_matrix"] = np.asarray(loss_inputs["scaled_matrix"])[:, keep_mask]
    for key in (
        "scaled_target",
        "unscaled_target",
        "scaling",
        "loss_denominator",
        "loss_target_weight",
        "loss_bucket",
        "loss_unit",
        "loss_scope",
        "loss_family",
        "loss_epsilon",
    ):
        value = loss_inputs.get(key)
        if value is not None:
            filtered[key] = np.asarray(value)[keep_mask]

    filtered_names = target_names[keep_mask].tolist()
    metadata["target_names"] = filtered_names
    metadata["target_scope_filter"] = target_scope
    metadata["n_targets_scope_filtered_from"] = int(target_names.size)
    metadata["n_targets_kept"] = int(len(filtered_names))
    metadata["n_national_targets"] = int(
        sum(str(name).startswith("nation/") for name in filtered_names)
    )
    metadata["n_state_targets"] = int(
        len(filtered_names) - metadata["n_national_targets"]
    )
    sidecar_rows = metadata.get("target_loss_metadata")
    if isinstance(sidecar_rows, list) and len(sidecar_rows) == target_names.size:
        metadata["target_loss_metadata"] = [
            row for row, keep in zip(sidecar_rows, keep_mask, strict=True) if keep
        ]
    filtered["metadata"] = metadata
    return filtered


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
    for key in (
        "loss_denominator",
        "loss_target_weight",
        "loss_epsilon",
    ):
        left = candidate_inputs.get(key)
        right = baseline_inputs.get(key)
        if left is None and right is None:
            continue
        if left is None or right is None or not np.allclose(left, right):
            raise ValueError(f"candidate and baseline PE-native {key} differ")
    for key in ("loss_bucket", "loss_unit", "loss_scope", "loss_family"):
        left = candidate_inputs.get(key)
        right = baseline_inputs.get(key)
        if left is None and right is None:
            continue
        if left is None or right is None or not np.array_equal(left, right):
            raise ValueError(f"candidate and baseline PE-native {key} differ")
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
    loss_arrays = loss_arrays_from_inputs(loss_inputs)
    train_mask = ~holdout_mask
    train_loss_arrays = (
        subset_loss_arrays(loss_arrays, train_mask) if loss_arrays is not None else None
    )
    holdout_loss_arrays = (
        subset_loss_arrays(loss_arrays, holdout_mask)
        if loss_arrays is not None
        else None
    )
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
                "full_loss": _objective(
                    matrix,
                    target,
                    weights,
                    loss_arrays=loss_arrays,
                ),
                "train_loss": _objective(
                    matrix[:, train_mask],
                    target[train_mask],
                    weights,
                    loss_arrays=train_loss_arrays,
                ),
                "holdout_loss": _objective(
                    matrix[:, holdout_mask],
                    target[holdout_mask],
                    weights,
                    loss_arrays=holdout_loss_arrays,
                ),
                "weight_sum": float(weights.sum()),
                "positive_household_count": int((weights > 1e-9).sum()),
            }
        )

    optimized_weights, optimizer_summary = optimize_pe_native_loss_weights(
        scaled_matrix=matrix[:, train_mask],
        scaled_target=target[train_mask],
        initial_weights=initial_weights,
        loss_arrays=train_loss_arrays,
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
        "initial_full_loss": _objective(
            matrix,
            target,
            initial_weights,
            loss_arrays=loss_arrays,
        ),
        "optimized_full_loss": _objective(
            matrix,
            target,
            optimized_weights,
            loss_arrays=loss_arrays,
        ),
        "initial_train_loss": _objective(
            matrix[:, train_mask],
            target[train_mask],
            initial_weights,
            loss_arrays=train_loss_arrays,
        ),
        "optimized_train_loss": _objective(
            matrix[:, train_mask],
            target[train_mask],
            optimized_weights,
            loss_arrays=train_loss_arrays,
        ),
        "initial_holdout_loss": _objective(
            matrix[:, holdout_mask],
            target[holdout_mask],
            initial_weights,
            loss_arrays=holdout_loss_arrays,
        ),
        "optimized_holdout_loss": _objective(
            matrix[:, holdout_mask],
            target[holdout_mask],
            optimized_weights,
            loss_arrays=holdout_loss_arrays,
        ),
        "initial_weight_sum": float(initial_weights.sum()),
        "optimized_weight_sum": float(optimized_weights.sum()),
        "household_count": int(len(optimized_weights)),
        "positive_household_count": int((optimized_weights > 1e-9).sum()),
        "optimizer_summary": optimizer_summary,
        "loss_curve": loss_curve,
        "optimized_weights": optimized_weights,
    }


def _objective(
    matrix: np.ndarray,
    target: np.ndarray,
    weights: np.ndarray,
    *,
    loss_arrays: Any | None = None,
) -> float:
    estimate = matrix.T @ weights
    if loss_arrays is not None:
        return pe_native_huber_loss(estimate, loss_arrays)
    residual = estimate - target
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
    rows: dict[str, dict[str, float | int]] = {}
    for family, patterns in _PROTECTED_TARGET_PATTERNS.items():
        indices = [
            index
            for index, name in enumerate(target_names)
            if _target_matches_protected_family(name, family, patterns)
        ]
        if not indices:
            continue
        candidate_loss = float(candidate_terms[indices].sum())
        baseline_loss = float(baseline_terms[indices].sum())
        rows[family] = {
            "n_targets": int(len(indices)),
            "candidate_loss": candidate_loss,
            "baseline_loss": baseline_loss,
            "loss_delta": candidate_loss - baseline_loss,
        }
    return rows


def _target_loss_diagnostics(
    *,
    target_names: list[str],
    candidate_inputs: dict[str, Any],
    baseline_inputs: dict[str, Any],
    candidate_weights: np.ndarray,
    baseline_weights: np.ndarray,
    holdout_mask: np.ndarray,
    top_k: int,
) -> dict[str, Any]:
    candidate_terms = _loss_terms(candidate_inputs, candidate_weights)
    baseline_terms = _loss_terms(baseline_inputs, baseline_weights)
    candidate_values = _target_value_diagnostics(
        candidate_inputs,
        candidate_weights,
    )
    baseline_values = _target_value_diagnostics(
        baseline_inputs,
        baseline_weights,
    )
    if not np.array_equal(
        candidate_values["value_scale"],
        baseline_values["value_scale"],
    ):
        raise ValueError("candidate and baseline target diagnostic scales differ")
    if not np.allclose(candidate_values["target"], baseline_values["target"]):
        raise ValueError("candidate and baseline target diagnostic values differ")
    for key in ("loss_denominator", "loss_target_weight"):
        if not np.allclose(candidate_values[key], baseline_values[key]):
            raise ValueError(f"candidate and baseline target diagnostic {key} differ")
    for key in ("loss_bucket", "loss_unit", "loss_scope"):
        if not np.array_equal(candidate_values[key], baseline_values[key]):
            raise ValueError(f"candidate and baseline target diagnostic {key} differ")
    if candidate_terms.shape != baseline_terms.shape:
        raise ValueError("candidate and baseline target loss term shapes differ")
    if len(target_names) != candidate_terms.shape[0]:
        raise ValueError("target name count does not match loss terms")
    if holdout_mask.shape[0] != candidate_terms.shape[0]:
        raise ValueError("holdout mask length does not match loss terms")

    rows: list[dict[str, Any]] = []
    candidate_wins = 0
    baseline_wins = 0
    ties = 0
    candidate_loss_total = float(candidate_terms.sum())
    baseline_loss_total = float(baseline_terms.sum())
    for index, target_name in enumerate(target_names):
        candidate_loss = float(candidate_terms[index])
        baseline_loss = float(baseline_terms[index])
        loss_delta = candidate_loss - baseline_loss
        if np.isclose(candidate_loss, baseline_loss):
            winner = "tie"
            ties += 1
        elif candidate_loss < baseline_loss:
            winner = "candidate"
            candidate_wins += 1
        else:
            winner = "baseline"
            baseline_wins += 1
        rows.append(
            {
                "target_index": int(index),
                "target_name": str(target_name),
                "family": classify_pe_native_target_family(target_name),
                "loss_scope": str(candidate_values["loss_scope"][index]),
                "loss_unit": str(candidate_values["loss_unit"][index]),
                "loss_bucket": str(candidate_values["loss_bucket"][index]),
                "loss_denominator": float(candidate_values["loss_denominator"][index]),
                "loss_target_weight": float(
                    candidate_values["loss_target_weight"][index]
                ),
                "loss_epsilon": float(candidate_values["loss_epsilon"][index]),
                "split": "holdout" if bool(holdout_mask[index]) else "train",
                "value_scale": str(candidate_values["value_scale"][index]),
                "target_value": float(candidate_values["target"][index]),
                "candidate_estimate": float(candidate_values["estimate"][index]),
                "baseline_estimate": float(baseline_values["estimate"][index]),
                "candidate_error": float(candidate_values["error"][index]),
                "baseline_error": float(baseline_values["error"][index]),
                "candidate_relative_error": float(
                    candidate_values["relative_error"][index]
                ),
                "baseline_relative_error": float(
                    baseline_values["relative_error"][index]
                ),
                "candidate_loss_term": candidate_loss,
                "baseline_loss_term": baseline_loss,
                "candidate_loss_share": (
                    candidate_loss / candidate_loss_total
                    if candidate_loss_total > 0.0
                    else 0.0
                ),
                "baseline_loss_share": (
                    baseline_loss / baseline_loss_total
                    if baseline_loss_total > 0.0
                    else 0.0
                ),
                "loss_delta": float(loss_delta),
                "candidate_abs_scaled_error": float(np.sqrt(candidate_loss)),
                "baseline_abs_scaled_error": float(np.sqrt(baseline_loss)),
                "winner": winner,
            }
        )

    top_k = max(0, int(top_k))
    regressions = sorted(
        rows,
        key=lambda row: float(row["loss_delta"]),
        reverse=True,
    )[:top_k]
    improvements = sorted(rows, key=lambda row: float(row["loss_delta"]))[:top_k]
    summary = {
        "n_targets": int(len(rows)),
        "candidate_loss": candidate_loss_total,
        "baseline_loss": baseline_loss_total,
        "loss_delta": float(candidate_loss_total - baseline_loss_total),
        "candidate_max_single_target_loss_share": (
            float(candidate_terms.max() / candidate_loss_total)
            if candidate_loss_total > 0.0 and candidate_terms.size
            else 0.0
        ),
        "baseline_max_single_target_loss_share": (
            float(baseline_terms.max() / baseline_loss_total)
            if baseline_loss_total > 0.0 and baseline_terms.size
            else 0.0
        ),
        "candidate_wins": int(candidate_wins),
        "baseline_wins": int(baseline_wins),
        "ties": int(ties),
        "train_targets": int((~holdout_mask).sum()),
        "holdout_targets": int(holdout_mask.sum()),
        "top_k": int(top_k),
    }
    return {
        "schema_version": 1,
        "metric": "sound_ecps_target_loss_diagnostics",
        "summary": summary,
        "family_breakdown": _target_family_breakdown(rows, len(rows)),
        "bucket_breakdown": _target_bucket_breakdown(rows),
        "top_regressions": regressions,
        "top_improvements": improvements,
        "targets": rows,
    }


def _refit_matrix_score_summary(
    *,
    target_names: list[str],
    candidate_inputs: dict[str, Any],
    baseline_inputs: dict[str, Any],
    candidate_refit: dict[str, Any],
    baseline_refit: dict[str, Any],
    target_diagnostics: dict[str, Any],
) -> dict[str, Any]:
    candidate_loss = float(candidate_refit["optimized_full_loss"])
    baseline_loss = float(baseline_refit["optimized_full_loss"])
    candidate_msre = _diagnostic_unweighted_msre(target_diagnostics, "candidate")
    baseline_msre = _diagnostic_unweighted_msre(target_diagnostics, "baseline")
    candidate_metadata = dict(candidate_inputs.get("metadata") or {})
    baseline_metadata = dict(baseline_inputs.get("metadata") or {})
    loss_metric = str(
        candidate_metadata.get(
            "loss_metric",
            baseline_metadata.get("loss_metric", "enhanced_cps_native_loss"),
        )
    )
    n_targets_kept = int(
        candidate_metadata.get(
            "n_targets_kept",
            baseline_metadata.get("n_targets_kept", len(target_names)),
        )
    )
    summary: dict[str, Any] = {
        "loss_metric": loss_metric,
        "loss_config": candidate_metadata.get(
            "loss_config",
            baseline_metadata.get("loss_config"),
        ),
        "candidate_enhanced_cps_native_loss": candidate_loss,
        "baseline_enhanced_cps_native_loss": baseline_loss,
        "enhanced_cps_native_loss_delta": candidate_loss - baseline_loss,
        "candidate_beats_baseline": candidate_loss < baseline_loss,
        "candidate_unweighted_msre": candidate_msre,
        "baseline_unweighted_msre": baseline_msre,
        "unweighted_msre_delta": candidate_msre - baseline_msre,
        "n_targets_kept": n_targets_kept,
        "score_source": "refit_loss_matrix",
        "candidate_max_single_target_loss_share": target_diagnostics["summary"].get(
            "candidate_max_single_target_loss_share"
        ),
        "baseline_max_single_target_loss_share": target_diagnostics["summary"].get(
            "baseline_max_single_target_loss_share"
        ),
    }
    for key in (
        "n_targets_total",
        "n_targets_zero_dropped",
        "n_targets_bad_dropped",
        "n_national_targets",
        "n_state_targets",
    ):
        if key in candidate_metadata:
            summary[key] = candidate_metadata[key]
        elif key in baseline_metadata:
            summary[key] = baseline_metadata[key]
    return summary


def _refit_matrix_score_payload(
    *,
    period: int,
    candidate_dataset_path: Path,
    baseline_dataset_path: Path,
    summary: dict[str, Any],
    target_diagnostics: dict[str, Any],
) -> dict[str, Any]:
    family_breakdown = list(target_diagnostics.get("family_breakdown") or ())
    return {
        "metric": str(summary.get("loss_metric") or "enhanced_cps_native_loss"),
        "score_source": "refit_loss_matrix",
        "period": int(period),
        "candidate_dataset": str(candidate_dataset_path.resolve()),
        "baseline_dataset": str(baseline_dataset_path.resolve()),
        "summary": dict(summary),
        "family_breakdown": family_breakdown,
        "broad_loss": {
            "score_source": "refit_loss_matrix",
            "summary": dict(summary),
            "family_breakdown": family_breakdown,
        },
    }


def _diagnostic_unweighted_msre(
    target_diagnostics: dict[str, Any],
    prefix: str,
) -> float:
    rows = list(target_diagnostics.get("targets") or ())
    if not rows:
        return float("nan")
    values = np.asarray(
        [float(row[f"{prefix}_relative_error"]) for row in rows],
        dtype=np.float64,
    )
    return float(np.mean(np.square(values)))


def _target_value_diagnostics(
    loss_inputs: dict[str, Any],
    weights: np.ndarray,
) -> dict[str, np.ndarray]:
    matrix = np.asarray(loss_inputs["scaled_matrix"], dtype=np.float64)
    scaled_target = np.asarray(loss_inputs["scaled_target"], dtype=np.float64)
    scaled_estimate = matrix.T @ weights
    loss_arrays = loss_arrays_from_inputs(loss_inputs)
    if loss_arrays is not None:
        target = loss_arrays.target_values.astype(np.float64, copy=True)
        estimate = scaled_estimate.astype(np.float64, copy=True)
        error = estimate - loss_arrays.objective_target
        return {
            "value_scale": np.full(target.shape, "native", dtype=object),
            "target": target,
            "estimate": estimate,
            "error": error,
            "relative_error": pe_native_relative_error(estimate, loss_arrays),
            "loss_denominator": loss_arrays.denominator,
            "loss_target_weight": loss_arrays.target_weight,
            "loss_bucket": loss_arrays.bucket_keys,
            "loss_unit": loss_arrays.unit_keys,
            "loss_scope": loss_arrays.scope_keys,
            "loss_family": loss_arrays.family_keys,
            "loss_epsilon": loss_arrays.epsilon,
        }
    unscaled_target = loss_inputs.get("unscaled_target")
    scaling = loss_inputs.get("scaling")
    target = scaled_target.astype(np.float64, copy=True)
    estimate = scaled_estimate.astype(np.float64, copy=True)
    value_scale = np.full(target.shape, "scaled", dtype=object)
    if unscaled_target is not None and scaling is not None:
        scaling_array = np.asarray(scaling, dtype=np.float64)
        if scaling_array.shape != target.shape:
            raise ValueError("PE-native target scaling shape differs from target shape")
        native_mask = np.isfinite(scaling_array) & ~np.isclose(scaling_array, 0.0)
        if native_mask.any():
            target[native_mask] = np.asarray(unscaled_target, dtype=np.float64)[
                native_mask
            ]
            estimate[native_mask] = (
                scaled_estimate[native_mask] / scaling_array[native_mask]
            )
            value_scale[native_mask] = "native"
    if target.shape != estimate.shape:
        raise ValueError("target and estimate shapes differ")
    error = estimate - target
    relative_error = ((estimate - target) + 1.0) / (target + 1.0)
    return {
        "value_scale": value_scale,
        "target": target,
        "estimate": estimate,
        "error": error,
        "relative_error": relative_error,
        "loss_denominator": np.abs(target) + 1.0,
        "loss_target_weight": np.ones(target.shape, dtype=np.float64),
        "loss_bucket": np.full(target.shape, "legacy", dtype=object),
        "loss_unit": np.full(target.shape, "legacy", dtype=object),
        "loss_scope": np.full(target.shape, "legacy", dtype=object),
        "loss_family": np.full(target.shape, "legacy", dtype=object),
        "loss_epsilon": np.ones(target.shape, dtype=np.float64),
    }


def _target_family_breakdown(
    target_rows: list[dict[str, Any]],
    total_targets: int,
) -> list[dict[str, Any]]:
    families: dict[str, list[dict[str, Any]]] = {}
    for row in target_rows:
        families.setdefault(str(row["family"]), []).append(row)
    breakdown = []
    for family, rows in sorted(families.items()):
        candidate_loss = sum(float(row["candidate_loss_term"]) for row in rows)
        baseline_loss = sum(float(row["baseline_loss_term"]) for row in rows)
        breakdown.append(
            {
                "family": family,
                "n_targets": int(len(rows)),
                "train_targets": int(sum(1 for row in rows if row["split"] == "train")),
                "holdout_targets": int(
                    sum(1 for row in rows if row["split"] == "holdout")
                ),
                "candidate_loss_contribution": float(candidate_loss),
                "baseline_loss_contribution": float(baseline_loss),
                "loss_delta": float(candidate_loss - baseline_loss),
                "candidate_wins": int(
                    sum(1 for row in rows if row["winner"] == "candidate")
                ),
                "baseline_wins": int(
                    sum(1 for row in rows if row["winner"] == "baseline")
                ),
                "ties": int(sum(1 for row in rows if row["winner"] == "tie")),
            }
        )
    return sorted(
        breakdown, key=lambda row: abs(float(row["loss_delta"])), reverse=True
    )


def _target_bucket_breakdown(target_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    buckets: dict[str, list[dict[str, Any]]] = {}
    for row in target_rows:
        buckets.setdefault(str(row["loss_bucket"]), []).append(row)
    breakdown = []
    for bucket, rows in sorted(buckets.items()):
        candidate_loss = sum(float(row["candidate_loss_term"]) for row in rows)
        baseline_loss = sum(float(row["baseline_loss_term"]) for row in rows)
        breakdown.append(
            {
                "bucket": bucket,
                "scope": str(rows[0]["loss_scope"]),
                "unit": str(rows[0]["loss_unit"]),
                "n_targets": int(len(rows)),
                "train_targets": int(sum(1 for row in rows if row["split"] == "train")),
                "holdout_targets": int(
                    sum(1 for row in rows if row["split"] == "holdout")
                ),
                "candidate_loss_contribution": float(candidate_loss),
                "baseline_loss_contribution": float(baseline_loss),
                "loss_delta": float(candidate_loss - baseline_loss),
                "candidate_target_weight_sum": float(
                    sum(float(row["loss_target_weight"]) for row in rows)
                ),
            }
        )
    return sorted(
        breakdown, key=lambda row: abs(float(row["loss_delta"])), reverse=True
    )


def _support_audit_summary(support_audit: dict[str, Any]) -> dict[str, Any]:
    comparisons = dict(support_audit.get("comparisons") or {})
    critical_rows = list(comparisons.get("critical_input_support") or ())
    missing_stored = [
        row["variable"]
        for row in critical_rows
        if bool(row.get("baseline_stored")) and not bool(row.get("candidate_stored"))
    ]
    return {
        "missing_stored_critical_inputs": missing_stored,
        "top_critical_input_support_gaps": _sort_rows_by_abs_delta(
            critical_rows,
            "weighted_nonzero_delta",
        ),
        "top_filing_status_gaps": _sort_rows_by_abs_delta(
            list(comparisons.get("filing_status_weighted_delta") or ()),
            "weighted_count_delta",
        ),
        "top_hoh_agi_gaps": _sort_rows_by_abs_delta(
            list(comparisons.get("hoh_agi_delta") or ()),
            "weighted_count_delta",
        ),
        "top_ssi_by_age_gaps": _sort_rows_by_abs_delta(
            list(comparisons.get("ssi_by_age_delta") or ()),
            "weighted_recipient_delta",
        ),
        "top_medicare_part_b_by_age_gaps": _sort_rows_by_abs_delta(
            list(comparisons.get("medicare_part_b_premiums_by_age_delta") or ()),
            "weighted_positive_delta",
        ),
        "top_aca_ptc_spending_gaps": _sort_rows_by_abs_delta(
            list(comparisons.get("state_aca_ptc_spending_top_gaps") or ()),
            "weighted_aca_ptc_delta",
        ),
    }


def _sort_rows_by_abs_delta(
    rows: list[dict[str, Any]],
    delta_key: str,
    *,
    limit: int = 10,
) -> list[dict[str, Any]]:
    return sorted(
        rows,
        key=lambda row: abs(float(row.get(delta_key, 0.0))),
        reverse=True,
    )[:limit]


def _loss_terms(loss_inputs: dict[str, Any], weights: np.ndarray) -> np.ndarray:
    matrix = np.asarray(loss_inputs["scaled_matrix"], dtype=np.float64)
    target = np.asarray(loss_inputs["scaled_target"], dtype=np.float64)
    estimate = matrix.T @ weights
    loss_arrays = loss_arrays_from_inputs(loss_inputs)
    if loss_arrays is not None:
        return pe_native_huber_loss_terms(estimate, loss_arrays)
    residual = estimate - target
    return np.square(residual)


def _target_matches_protected_family(
    target_name: str,
    family: str,
    patterns: tuple[str, ...],
) -> bool:
    normalized = (
        target_name.lower().replace("-", "_").replace(" ", "_").replace("/", "_")
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


def _frozen_ecps_baseline_certificate(
    *,
    baseline_dataset_path: Path,
    policyengine_targets_db_path: Path | None,
    policyengine_us_data_repo: str | Path | None,
    period: int,
    target_names: list[str],
    target_scope: str,
    holdout_target_fraction: float,
    holdout_target_seed: int,
    matched_sample_method: str,
    refit_config: dict[str, Any],
    skip_tax_expenditure_targets: bool,
    exact_rescore: bool,
    score_source: str,
    baseline_sanity: dict[str, Any],
    score_summary: dict[str, Any],
) -> dict[str, Any]:
    """Freeze the eCPS baseline surface used for this numeric verdict.

    Promotion gates consume this certificate and compare it to the pinned
    benchmark manifest. That prevents a release from passing on a live
    recomputation against a different eCPS H5, target DB, scorer checkout, or
    scoring config.
    """

    scoring_config = {
        "period": int(period),
        "target_profile": "pe_native_broad",
        "target_scope": str(target_scope),
        "holdout_target_fraction": float(holdout_target_fraction),
        "holdout_target_seed": int(holdout_target_seed),
        "matched_sample_method": str(matched_sample_method),
        "refit_config": dict(refit_config),
        "skip_tax_expenditure_targets": bool(skip_tax_expenditure_targets),
        "exact_rescore": bool(exact_rescore),
        "score_source": str(score_source),
        "comparison_bad_targets": list(_comparison_bad_targets()),
    }
    baseline_metrics = {
        key: score_summary.get(key)
        for key in (
            "baseline_initial_enhanced_cps_native_loss",
            "baseline_enhanced_cps_native_loss",
            "baseline_train_loss",
            "baseline_holdout_loss",
            "baseline_unweighted_msre",
            "n_targets_kept",
            "n_national_targets",
            "n_state_targets",
        )
        if score_summary.get(key) is not None
    }
    return {
        "schema_version": 1,
        "certificate_type": "frozen_production_ecps_baseline",
        "period": int(period),
        "baseline_dataset": _dataset_descriptor(baseline_dataset_path),
        "target_db": (
            _dataset_descriptor(policyengine_targets_db_path)
            if policyengine_targets_db_path is not None
            else None
        ),
        "policyengine_us_data": _git_repo_descriptor(policyengine_us_data_repo),
        "target_surface": {
            "target_profile": "pe_native_broad",
            "target_scope": str(target_scope),
            "target_count": int(len(target_names)),
            "target_names_sha256": _canonical_json_sha256(list(target_names)),
        },
        "scoring_config": {
            **scoring_config,
            "sha256": _canonical_json_sha256(scoring_config),
        },
        "baseline_metrics": baseline_metrics,
        "baseline_sanity": dict(baseline_sanity),
    }


def _git_repo_descriptor(repo_path: str | Path | None) -> dict[str, Any] | None:
    if repo_path is None:
        return None
    repo = Path(repo_path).expanduser().resolve()
    descriptor: dict[str, Any] = {"repo": str(repo)}
    commit = _git_output_or_none(repo, "rev-parse", "HEAD")
    if commit:
        descriptor["commit"] = commit
    status = _git_output_or_none(repo, "status", "--porcelain")
    if status is not None:
        descriptor["dirty"] = bool(status)
    return descriptor


def _git_output_or_none(repo: Path, *args: str) -> str | None:
    completed = subprocess.run(
        ["git", "-C", str(repo), *args],
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        return None
    return completed.stdout.strip()


def _canonical_json_sha256(payload: Any) -> str:
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


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
    parser.add_argument(
        "--target-diagnostics-path",
        help="Defaults to <output-dir>/target_loss_diagnostics.json.",
    )
    parser.add_argument(
        "--support-audit-path",
        help="Defaults to <output-dir>/support_audit.json when enabled.",
    )
    parser.add_argument("--period", type=int, default=2024)
    parser.add_argument("--matched-household-count", type=int)
    parser.add_argument("--random-seed", type=int, default=20260529)
    parser.add_argument(
        "--matched-sample-method",
        choices=("uniform", "weight_proportional", "pps", "largest_weight"),
        default="uniform",
        help=(
            "Household thinning method used when matching a larger dataset down "
            "to the comparison household count."
        ),
    )
    parser.add_argument("--holdout-target-fraction", type=float, default=0.2)
    parser.add_argument("--holdout-target-seed", type=int, default=20260529)
    parser.add_argument("--optimizer-max-iter", type=int, default=200)
    parser.add_argument("--optimizer-tol", type=float, default=1e-8)
    parser.add_argument("--score-consistency-tol", type=float, default=1e-6)
    parser.add_argument("--target-diagnostics-top-k", type=int, default=50)
    parser.add_argument(
        "--skip-support-audit",
        action="store_true",
        help="Skip the PE-native support audit sidecar.",
    )
    parser.add_argument("--policyengine-us-data-repo")
    parser.add_argument("--policyengine-us-data-python")
    parser.add_argument(
        "--policyengine-targets-db",
        help=(
            "Explicit policy_data.db to use for PE-native comparison scoring. "
            "The scorer subprocess copies this DB into a temporary PE-US-data "
            "storage folder so the target surface is pinned."
        ),
    )
    parser.add_argument("--skip-tax-expenditure-targets", action="store_true")
    parser.add_argument(
        "--target-scope",
        choices=("all", "national", "state"),
        default="all",
        help="Restrict the PE-native refit/scoring surface by target scope.",
    )
    parser.add_argument(
        "--exact-rescore",
        action="store_true",
        help=(
            "After symmetric refit, recompute the PE-native loss by rebuilding "
            "PolicyEngine loss matrices for the refit H5s. This is an audit "
            "path and can take hours on local machines; by default the "
            "comparison uses the already-extracted refit loss matrices."
        ),
    )
    parser.add_argument("--force", action="store_true")
    parser.add_argument(
        "--no-assert-refit-effective",
        dest="assert_refit_effective",
        action="store_false",
        help="Skip the refit-effectiveness gate (allow a no-op refit).",
    )
    parser.add_argument(
        "--no-assert-baseline-sane",
        dest="assert_baseline_sane",
        action="store_false",
        help="Skip the baseline-sanity gate (allow a mis-scored production baseline).",
    )
    parser.add_argument(
        "--baseline-sanity-mode",
        choices=_BASELINE_SANITY_MODES,
        default="msre",
        help=(
            "Baseline-sanity gate to use. 'msre' requires production eCPS to "
            "score below --max-baseline-unweighted-msre on this exact surface. "
            "'content' verifies required production eCPS H5 columns are present "
            "and nonzero, for broad target surfaces where high eCPS loss is "
            "part of the comparison signal."
        ),
    )
    parser.add_argument(
        "--max-baseline-unweighted-msre",
        type=float,
        default=2.0,
        help="Baseline-sanity gate ceiling on the production eCPS unweighted MSRE.",
    )
    parser.add_argument(
        "--min-refit-loss-reduction",
        type=float,
        default=1e-9,
        help="Minimum loss reduction required by the refit-effectiveness gate.",
    )
    args = parser.parse_args(argv)

    output_dir = Path(args.output_dir).expanduser()
    output_path = (
        Path(args.output_path).expanduser()
        if args.output_path
        else output_dir / "sound_ecps_replacement_comparison.json"
    )
    written = write_sound_ecps_replacement_comparison(
        output_path,
        target_diagnostics_path=args.target_diagnostics_path,
        support_audit_path=args.support_audit_path,
        candidate_dataset_path=args.candidate_dataset,
        baseline_dataset_path=args.baseline_dataset,
        output_dir=output_dir,
        period=args.period,
        matched_household_count=args.matched_household_count,
        random_seed=args.random_seed,
        matched_sample_method=args.matched_sample_method,
        holdout_target_fraction=args.holdout_target_fraction,
        holdout_target_seed=args.holdout_target_seed,
        optimizer_max_iter=args.optimizer_max_iter,
        optimizer_tol=args.optimizer_tol,
        score_consistency_tol=args.score_consistency_tol,
        target_diagnostics_top_k=args.target_diagnostics_top_k,
        include_support_audit=not args.skip_support_audit,
        policyengine_us_data_repo=args.policyengine_us_data_repo,
        policyengine_us_data_python=args.policyengine_us_data_python,
        policyengine_targets_db_path=args.policyengine_targets_db,
        skip_tax_expenditure_targets=args.skip_tax_expenditure_targets,
        target_scope=args.target_scope,
        exact_rescore=args.exact_rescore,
        force=args.force,
        assert_refit_effective=args.assert_refit_effective,
        min_refit_loss_reduction=args.min_refit_loss_reduction,
        assert_baseline_sane=args.assert_baseline_sane,
        baseline_sanity_mode=args.baseline_sanity_mode,
        max_baseline_unweighted_msre=args.max_baseline_unweighted_msre,
    )
    print(str(written))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "build_sound_ecps_replacement_comparison",
    "write_sound_ecps_replacement_comparison",
]
