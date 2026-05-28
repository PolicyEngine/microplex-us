"""Persistent artifact quality gates for mp-300k candidates."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

import h5py

GateStatus = Literal["pass", "fail", "unmeasured"]

_REQUIRED_PERIOD_ARRAYS = (
    "household_id",
    "household_weight",
    "person_id",
    "person_household_id",
)
_DEFAULT_REQUIRED_GATES = (
    "candidate_artifact",
    "compatibility",
    "artifact_size",
    "runtime",
    "ecps_comparison",
    "benchmark_manifest",
)


def build_mp300k_artifact_gate_report(
    artifact_dir: str | Path,
    *,
    candidate_dataset_path: str | Path | None = None,
    baseline_dataset_path: str | Path | None = None,
    ecps_comparison_payload: Any = None,
    runtime_smoke_payload: dict[str, Any] | None = None,
    benchmark_manifest_path: str | Path | None = None,
    period: int = 2024,
    artifact_size_ratio_threshold: float = 2.0,
    runtime_ratio_threshold: float = 1.25,
    compute_native_scores: bool = True,
    require_ecps_comparison: bool = True,
    policyengine_us_data_repo: str | Path | None = None,
    policyengine_us_data_python: str | Path | None = None,
) -> dict[str, Any]:
    """Build a CI-friendly artifact gate report for one mp-300k candidate.

    The report is evidence-driven. It can consume precomputed runtime and
    eCPS-comparison payloads, or compute the PE-native broad score when a
    baseline dataset is available.
    """

    artifact_root = Path(artifact_dir).expanduser()
    manifest_path = artifact_root / "manifest.json"
    manifest = _load_manifest(manifest_path)
    candidate_dataset = _resolve_candidate_dataset_path(
        artifact_root,
        manifest,
        candidate_dataset_path,
    )
    baseline_dataset = (
        Path(baseline_dataset_path).expanduser()
        if baseline_dataset_path is not None
        else _manifest_baseline_dataset(artifact_root, manifest)
    )

    candidate_gate = _candidate_artifact_gate(
        manifest_path=manifest_path,
        candidate_dataset=candidate_dataset,
    )
    compatibility_gate = _compatibility_gate(candidate_dataset, period=period)
    artifact_size_gate = _artifact_size_gate(
        candidate_dataset,
        baseline_dataset=baseline_dataset,
        artifact_size_ratio_threshold=artifact_size_ratio_threshold,
    )
    resolved_ecps_comparison = _resolve_ecps_comparison_payload(
        ecps_comparison_payload,
        candidate_dataset=candidate_dataset,
        baseline_dataset=baseline_dataset,
        period=period,
        compute_native_scores=compute_native_scores,
        policyengine_us_data_repo=policyengine_us_data_repo,
        policyengine_us_data_python=policyengine_us_data_python,
    )
    ecps_comparison_gate = _ecps_comparison_gate(resolved_ecps_comparison)
    runtime_gate = _runtime_gate(
        runtime_smoke_payload,
        runtime_ratio_threshold=runtime_ratio_threshold,
    )
    benchmark_gate, benchmark_descriptor = _benchmark_manifest_gate(
        benchmark_manifest_path
    )
    gates = {
        "candidate_artifact": candidate_gate,
        "compatibility": compatibility_gate,
        "artifact_size": artifact_size_gate,
        "runtime": runtime_gate,
        "ecps_comparison": ecps_comparison_gate,
        "benchmark_manifest": benchmark_gate,
    }
    required_gates = _required_gate_names(
        require_ecps_comparison=require_ecps_comparison,
    )
    summary = _summarize_gates(gates, required_gates=required_gates)
    return {
        "schema_version": 1,
        "generated_at": datetime.now(UTC).isoformat(),
        "product": "mp-300k",
        "gate_set": "artifact_ci",
        "artifact_id": artifact_root.name,
        "artifact_dir": str(artifact_root.resolve()),
        "period": int(period),
        "required_gates": required_gates,
        "summary": summary,
        "manifest": _file_descriptor(manifest_path),
        "candidate_dataset": _optional_file_descriptor(candidate_dataset),
        "baseline_dataset": (
            _optional_file_descriptor(baseline_dataset)
            if baseline_dataset is not None
            else None
        ),
        "gates": gates,
        "ecps_comparison_payload": resolved_ecps_comparison,
        "runtime_smoke": runtime_smoke_payload,
        "benchmark_manifest": benchmark_descriptor,
    }


def write_mp300k_artifact_gate_report(
    artifact_dir: str | Path,
    *,
    output_path: str | Path | None = None,
    update_manifest: bool = True,
    **kwargs: Any,
) -> Path:
    """Write ``mp300k_artifact_gates.json`` and reference it in manifest."""

    artifact_root = Path(artifact_dir).expanduser()
    report_path = (
        Path(output_path).expanduser()
        if output_path is not None
        else artifact_root / "mp300k_artifact_gates.json"
    )
    report = build_mp300k_artifact_gate_report(artifact_root, **kwargs)
    _write_json_atomically(report_path, report)
    if update_manifest:
        manifest_path = artifact_root / "manifest.json"
        manifest = _load_manifest(manifest_path)
        artifacts = dict(manifest.get("artifacts", {}))
        artifacts["mp300k_artifact_gates"] = _relative_or_absolute(
            report_path,
            base_dir=artifact_root,
        )
        manifest["artifacts"] = artifacts
        manifest["mp300k_artifact_gates"] = {
            "status": report["summary"]["status"],
            "passing_required_gate_count": report["summary"][
                "passing_required_gate_count"
            ],
            "failed_required_gate_count": report["summary"][
                "failed_required_gate_count"
            ],
            "unmeasured_required_gate_count": report["summary"][
                "unmeasured_required_gate_count"
            ],
        }
        _write_json_atomically(manifest_path, manifest)
    return report_path


def _resolve_candidate_dataset_path(
    artifact_root: Path,
    manifest: dict[str, Any],
    explicit_path: str | Path | None,
) -> Path:
    if explicit_path is not None:
        return Path(explicit_path).expanduser()
    artifacts = dict(manifest.get("artifacts", {}))
    dataset_name = artifacts.get("policyengine_dataset")
    if not isinstance(dataset_name, str) or not dataset_name:
        raise ValueError(
            "manifest.artifacts.policyengine_dataset is required when "
            "candidate_dataset_path is not supplied"
        )
    dataset_path = Path(dataset_name).expanduser()
    if not dataset_path.is_absolute():
        dataset_path = artifact_root / dataset_path
    return dataset_path


def _manifest_baseline_dataset(
    artifact_root: Path, manifest: dict[str, Any]
) -> Path | None:
    config = dict(manifest.get("config", {}))
    value = config.get("policyengine_baseline_dataset")
    if value is None:
        return None
    baseline_path = Path(value).expanduser()
    if not baseline_path.is_absolute():
        baseline_path = artifact_root / baseline_path
    return baseline_path


def _candidate_artifact_gate(
    *,
    manifest_path: Path,
    candidate_dataset: Path,
) -> dict[str, Any]:
    missing = [
        str(path) for path in (manifest_path, candidate_dataset) if not path.exists()
    ]
    if missing:
        return _gate(
            "fail",
            "required candidate artifact files are missing",
            details={"missing": missing},
        )
    return _gate(
        "pass",
        "manifest and candidate H5 exist",
        metrics={
            "manifest_size_bytes": manifest_path.stat().st_size,
            "candidate_size_bytes": candidate_dataset.stat().st_size,
        },
    )


def _artifact_size_gate(
    candidate_dataset: Path,
    *,
    baseline_dataset: Path | None,
    artifact_size_ratio_threshold: float,
) -> dict[str, Any]:
    threshold = float(artifact_size_ratio_threshold)
    if baseline_dataset is None:
        return _gate(
            "unmeasured",
            "baseline H5 has not been attached for artifact-size comparison",
            metrics={"artifact_size_ratio_threshold": threshold},
        )
    if not candidate_dataset.exists() or not baseline_dataset.exists():
        missing = [
            str(path)
            for path in (candidate_dataset, baseline_dataset)
            if not path.exists()
        ]
        return _gate(
            "fail",
            "artifact-size comparison files are missing",
            details={"missing": missing},
            metrics={"artifact_size_ratio_threshold": threshold},
        )
    candidate_size = candidate_dataset.stat().st_size
    baseline_size = baseline_dataset.stat().st_size
    if baseline_size <= 0:
        return _gate(
            "fail",
            "baseline H5 size is nonpositive",
            metrics={
                "candidate_size_bytes": candidate_size,
                "baseline_size_bytes": baseline_size,
                "artifact_size_ratio_threshold": threshold,
            },
        )
    ratio = candidate_size / baseline_size
    return _gate(
        "pass" if ratio <= threshold else "fail",
        (
            "candidate H5 size is inside the artifact-size threshold"
            if ratio <= threshold
            else "candidate H5 size exceeds the artifact-size threshold"
        ),
        metrics={
            "candidate_size_bytes": candidate_size,
            "baseline_size_bytes": baseline_size,
            "artifact_size_ratio": ratio,
            "artifact_size_ratio_threshold": threshold,
        },
    )


def _compatibility_gate(candidate_dataset: Path, *, period: int) -> dict[str, Any]:
    try:
        missing = _missing_required_period_arrays(candidate_dataset, period=period)
        if missing:
            return _gate(
                "fail",
                "candidate H5 is missing required PolicyEngine structural arrays",
                details={"missing_arrays": missing},
            )
        from microplex_us.policyengine.us import load_policyengine_us_entity_tables

        tables = load_policyengine_us_entity_tables(
            candidate_dataset,
            period=period,
            variables=(),
        )
        household_weight_sum = float(tables.households["household_weight"].sum())
        if household_weight_sum <= 0:
            return _gate(
                "fail",
                "candidate H5 has nonpositive household weight sum",
                metrics={"household_weight_sum": household_weight_sum},
            )
        return _gate(
            "pass",
            "candidate H5 satisfies the structural PolicyEngine table contract",
            metrics={
                "household_count": int(len(tables.households)),
                "person_count": int(len(tables.persons))
                if tables.persons is not None
                else 0,
                "household_weight_sum": household_weight_sum,
            },
        )
    except Exception as exc:  # noqa: BLE001 - this is a gate report boundary.
        return _gate(
            "fail",
            "candidate H5 failed the structural PolicyEngine table contract",
            details={"error": str(exc)},
        )


def _missing_required_period_arrays(
    candidate_dataset: Path, *, period: int
) -> list[str]:
    period_key = str(int(period))
    with h5py.File(candidate_dataset, "r") as handle:
        return [
            variable
            for variable in _REQUIRED_PERIOD_ARRAYS
            if variable not in handle or period_key not in handle[variable]
        ]


def _resolve_ecps_comparison_payload(
    ecps_comparison_payload: Any,
    *,
    candidate_dataset: Path,
    baseline_dataset: Path | None,
    period: int,
    compute_native_scores: bool,
    policyengine_us_data_repo: str | Path | None,
    policyengine_us_data_python: str | Path | None,
) -> Any:
    if ecps_comparison_payload is not None:
        return ecps_comparison_payload
    if not compute_native_scores or baseline_dataset is None:
        return None
    from microplex_us.pipelines.pe_native_scores import compute_us_pe_native_scores

    return compute_us_pe_native_scores(
        candidate_dataset_path=candidate_dataset,
        baseline_dataset_path=baseline_dataset,
        period=period,
        policyengine_us_data_repo=policyengine_us_data_repo,
        policyengine_us_data_python=policyengine_us_data_python,
    )


def _ecps_comparison_gate(
    ecps_comparison_payload: dict[str, Any] | None,
) -> dict[str, Any]:
    if ecps_comparison_payload is None:
        return _gate(
            "unmeasured",
            "PE-native eCPS comparison has not been attached",
        )
    summary = _ecps_comparison_summary(ecps_comparison_payload)
    candidate_loss = summary.get("candidate_enhanced_cps_native_loss")
    baseline_loss = summary.get("baseline_enhanced_cps_native_loss")
    loss_delta = summary.get("enhanced_cps_native_loss_delta")
    reported_candidate_beats = summary.get("candidate_beats_baseline")
    details: dict[str, Any] = {}
    if candidate_loss is not None and baseline_loss is not None:
        computed_loss_delta = float(candidate_loss) - float(baseline_loss)
        if (
            loss_delta is not None
            and abs(float(loss_delta) - computed_loss_delta) > 1e-12
        ):
            details["reported_loss_delta"] = loss_delta
            details["computed_loss_delta"] = computed_loss_delta
        loss_delta = computed_loss_delta
    candidate_beats = None
    if loss_delta is not None:
        candidate_beats = float(loss_delta) < 0.0
    if (
        reported_candidate_beats is not None
        and candidate_beats is not None
        and bool(reported_candidate_beats) != candidate_beats
    ):
        details["reported_candidate_beats_baseline"] = reported_candidate_beats
        details["computed_candidate_beats_baseline"] = candidate_beats
    status: GateStatus
    if candidate_beats is None:
        status = "unmeasured"
    else:
        status = "pass" if bool(candidate_beats) else "fail"
    return _gate(
        status,
        (
            "candidate beats pinned eCPS on PE-native broad loss"
            if status == "pass"
            else (
                "candidate does not beat pinned eCPS on PE-native broad loss"
                if status == "fail"
                else "PE-native eCPS comparison payload is incomplete"
            )
        ),
        metrics={
            "candidate_enhanced_cps_native_loss": candidate_loss,
            "baseline_enhanced_cps_native_loss": baseline_loss,
            "enhanced_cps_native_loss_delta": loss_delta,
            "n_targets_kept": summary.get("n_targets_kept"),
        },
        details=details,
    )


def _ecps_comparison_summary(payload: Any) -> dict[str, Any]:
    if isinstance(payload, list):
        for item in payload:
            summary = _ecps_comparison_summary(item)
            if summary:
                return summary
        return {}
    if not isinstance(payload, dict):
        return {}
    for key in ("summary", "broad_loss", "loss_summary"):
        value = payload.get(key)
        if isinstance(value, dict):
            return dict(value)
    if "candidate_enhanced_cps_native_loss" in payload:
        return dict(payload)
    if (
        "best_variant_loss" in payload
        and "baseline_enhanced_cps_native_loss" in payload
    ):
        best_variant_loss = payload.get("best_variant_loss")
        baseline_loss = payload.get("baseline_enhanced_cps_native_loss")
        loss_delta = None
        candidate_beats = None
        if best_variant_loss is not None and baseline_loss is not None:
            loss_delta = float(best_variant_loss) - float(baseline_loss)
            candidate_beats = loss_delta < 0.0
        return {
            "candidate_enhanced_cps_native_loss": best_variant_loss,
            "baseline_enhanced_cps_native_loss": baseline_loss,
            "enhanced_cps_native_loss_delta": loss_delta,
            "candidate_beats_baseline": candidate_beats,
            "best_variant_label": payload.get("best_variant_label"),
        }
    return {}


def _runtime_gate(
    runtime_smoke_payload: dict[str, Any] | None,
    *,
    runtime_ratio_threshold: float,
) -> dict[str, Any]:
    if runtime_smoke_payload is None:
        return _gate("unmeasured", "runtime smoke benchmark has not been attached")
    payload = dict(runtime_smoke_payload)
    threshold = float(runtime_ratio_threshold)
    ratio = payload.get("runtime_ratio")
    candidate_seconds = payload.get("candidate_seconds")
    baseline_seconds = payload.get("baseline_seconds")
    if ratio is None and candidate_seconds is not None and baseline_seconds:
        ratio = float(candidate_seconds) / float(baseline_seconds)
    passes = payload.get("passes_runtime_gate")
    details: dict[str, Any] = {}
    reported_threshold = payload.get("runtime_ratio_threshold")
    reported_threshold_matches = False
    try:
        reported_threshold_matches = float(reported_threshold) == threshold
    except (TypeError, ValueError):
        pass
    if reported_threshold is not None and not reported_threshold_matches:
        details["reported_runtime_ratio_threshold"] = reported_threshold
        details["enforced_runtime_ratio_threshold"] = threshold
    if ratio is None:
        return _gate(
            "unmeasured",
            "runtime smoke payload is missing ratio or candidate/baseline seconds",
            metrics={
                "candidate_seconds": candidate_seconds,
                "baseline_seconds": baseline_seconds,
                "runtime_ratio": ratio,
                "runtime_ratio_threshold": threshold,
            },
        )
    derived_passes = float(ratio) <= threshold
    if passes is not None and bool(passes) != derived_passes:
        details["reported_passes_runtime_gate"] = passes
        details["computed_passes_runtime_gate"] = derived_passes
    return _gate(
        "pass" if derived_passes else "fail",
        (
            "candidate runtime is inside the smoke benchmark threshold"
            if derived_passes
            else "candidate runtime exceeds the smoke benchmark threshold"
        ),
        metrics={
            "candidate_seconds": candidate_seconds,
            "baseline_seconds": baseline_seconds,
            "runtime_ratio": ratio,
            "runtime_ratio_threshold": threshold,
        },
        details=details,
    )


def _benchmark_manifest_gate(
    benchmark_manifest_path: str | Path | None,
) -> tuple[dict[str, Any], dict[str, Any] | None]:
    if benchmark_manifest_path is None:
        return (
            _gate(
                "unmeasured",
                "frozen microsimulation benchmark manifest has not been attached",
            ),
            None,
        )
    manifest_path = Path(benchmark_manifest_path).expanduser()
    if not manifest_path.exists():
        return (
            _gate(
                "fail",
                "frozen microsimulation benchmark manifest path does not exist",
                details={"path": str(manifest_path)},
            ),
            None,
        )
    descriptor = _file_descriptor(manifest_path)
    return (
        _gate(
            "pass",
            "frozen microsimulation benchmark manifest attached",
            details=descriptor,
        ),
        descriptor,
    )


def _required_gate_names(*, require_ecps_comparison: bool) -> list[str]:
    required = list(_DEFAULT_REQUIRED_GATES)
    if not require_ecps_comparison:
        required.remove("ecps_comparison")
    return required


def _summarize_gates(
    gates: dict[str, dict[str, Any]], *, required_gates: list[str]
) -> dict[str, Any]:
    statuses = {name: gate["status"] for name, gate in gates.items()}
    required = set(required_gates)
    failed_required = [
        name
        for name in required_gates
        if statuses.get(name) == "fail" or statuses.get(name) is None
    ]
    unmeasured_required = [
        name
        for name in required_gates
        if statuses.get(name) == "unmeasured" and name not in failed_required
    ]
    passing_required = [name for name in required_gates if statuses.get(name) == "pass"]
    failed_optional = [
        name
        for name, status in statuses.items()
        if name not in required and status == "fail"
    ]
    unmeasured_optional = [
        name
        for name, status in statuses.items()
        if name not in required and status == "unmeasured"
    ]
    if failed_required:
        overall_status = "failed"
    elif not unmeasured_required:
        overall_status = "passed"
    else:
        overall_status = "incomplete"
    return {
        "status": overall_status,
        "passing_required_gates": passing_required,
        "failed_required_gates": failed_required,
        "unmeasured_required_gates": unmeasured_required,
        "failed_optional_gates": failed_optional,
        "unmeasured_optional_gates": unmeasured_optional,
        "passing_required_gate_count": len(passing_required),
        "failed_required_gate_count": len(failed_required),
        "unmeasured_required_gate_count": len(unmeasured_required),
    }


def _gate(
    status: GateStatus,
    summary: str,
    *,
    metrics: dict[str, Any] | None = None,
    details: dict[str, Any] | None = None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {"status": status, "summary": summary}
    if metrics:
        payload["metrics"] = metrics
    if details:
        payload["details"] = details
    return payload


def _load_manifest(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"manifest not found: {path}")
    return json.loads(path.read_text())


def _load_json_file(path: str | Path | None) -> dict[str, Any] | None:
    if path is None:
        return None
    return json.loads(Path(path).expanduser().read_text())


def _file_descriptor(path: Path) -> dict[str, Any]:
    return {
        "path": str(path.resolve()),
        "size_bytes": path.stat().st_size,
        "sha256": _sha256_file(path),
    }


def _optional_file_descriptor(path: Path) -> dict[str, Any]:
    if path.exists():
        return _file_descriptor(path)
    return {"path": str(path.resolve()), "exists": False}


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _relative_or_absolute(path: Path, *, base_dir: Path) -> str:
    try:
        return str(path.resolve().relative_to(base_dir.resolve()))
    except ValueError:
        return str(path.resolve())


def _write_json_atomically(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_name(f".{path.name}.tmp")
    temp_path.write_text(json.dumps(payload, indent=2, sort_keys=True))
    temp_path.replace(path)


def main(argv: list[str] | None = None) -> int:
    """CLI entry point for enforcing mp-300k artifact gates."""

    parser = argparse.ArgumentParser(
        description="Run persistent mp-300k artifact gates against an artifact bundle."
    )
    parser.add_argument("--artifact-dir", required=True)
    parser.add_argument("--candidate-dataset")
    parser.add_argument("--baseline-dataset")
    parser.add_argument(
        "--ecps-comparison-json",
        "--native-scores-json",
        dest="ecps_comparison_json",
    )
    parser.add_argument("--runtime-smoke-json")
    parser.add_argument("--benchmark-manifest")
    parser.add_argument("--output-json")
    parser.add_argument("--target-period", type=int, default=2024)
    parser.add_argument("--artifact-size-ratio-threshold", type=float, default=2.0)
    parser.add_argument("--runtime-ratio-threshold", type=float, default=1.25)
    parser.add_argument("--policyengine-us-data-repo")
    parser.add_argument("--policyengine-us-data-python")
    parser.add_argument(
        "--skip-ecps-computation",
        "--skip-native-score",
        dest="skip_ecps_computation",
        action="store_true",
        help=(
            "Do not compute PE-native scores when --ecps-comparison-json is absent. "
            "The eCPS comparison gate will remain unmeasured."
        ),
    )
    parser.add_argument(
        "--no-require-ecps-comparison",
        action="store_true",
        help=(
            "Keep reporting the eCPS comparison gate, but do not make it block "
            "the overall status. This is the intended deprecation path once eCPS "
            "is no longer the comparator."
        ),
    )
    parser.add_argument(
        "--allow-incomplete",
        action="store_true",
        help="Return exit code 0 when required gates are unmeasured but not failed.",
    )
    parser.add_argument(
        "--no-update-manifest",
        action="store_true",
        help="Write the gate report without adding it to manifest.json.",
    )
    args = parser.parse_args(argv)

    report_path = write_mp300k_artifact_gate_report(
        args.artifact_dir,
        output_path=args.output_json,
        update_manifest=not args.no_update_manifest,
        candidate_dataset_path=args.candidate_dataset,
        baseline_dataset_path=args.baseline_dataset,
        ecps_comparison_payload=_load_json_file(args.ecps_comparison_json),
        runtime_smoke_payload=_load_json_file(args.runtime_smoke_json),
        benchmark_manifest_path=args.benchmark_manifest,
        period=args.target_period,
        artifact_size_ratio_threshold=args.artifact_size_ratio_threshold,
        runtime_ratio_threshold=args.runtime_ratio_threshold,
        compute_native_scores=not args.skip_ecps_computation,
        require_ecps_comparison=not args.no_require_ecps_comparison,
        policyengine_us_data_repo=args.policyengine_us_data_repo,
        policyengine_us_data_python=args.policyengine_us_data_python,
    )
    print(report_path)
    report = json.loads(report_path.read_text())
    status = report["summary"]["status"]
    if status == "passed" or (status == "incomplete" and args.allow_incomplete):
        return 0
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
