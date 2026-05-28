"""Build the living Microplex diagnostic dashboard payload."""

from __future__ import annotations

import argparse
import csv
import json
import re
import subprocess
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

_ROOT = Path(__file__).resolve().parents[3]
_DEFAULT_ARTIFACT_ROOT = _ROOT / "artifacts"
_DEFAULT_OUTPUT_PATH = _DEFAULT_ARTIFACT_ROOT / "microplex_dashboard_current.json"
_DEFAULT_TARGET_DIAGNOSTICS_PATH = (
    _DEFAULT_ARTIFACT_ROOT / "pe_native_target_diagnostics_current.json"
)
_DEFAULT_POLICYENGINE_US_DATA_REPO = Path(
    "/Users/maxghenis/PolicyEngine/policyengine-us-data"
)

_PE_MODEL_SLOTS = (
    {
        "id": "policyengine_legacy_ecps",
        "label": "PE legacy enhanced CPS",
        "status": "available_as_baseline",
        "notes": (
            "The incumbent enhanced CPS is represented as the baseline side of "
            "PE-native score artifacts."
        ),
    },
    {
        "id": "policyengine_small_l0",
        "label": "PE small-L0 local model",
        "status": "missing_weight_package",
        "notes": (
            "Mapped to policyengine-us-data local_net_worth_100 when present. "
            "The weight package is not itself a scored H5 dataset."
        ),
    },
    {
        "id": "policyengine_big_l0",
        "label": "PE big-L0 local model",
        "status": "missing_weight_package",
        "notes": (
            "Mapped to policyengine-us-data local_net_worth_100_e300 when "
            "present. The weight package is not itself a scored H5 dataset."
        ),
    },
)

_PE_L0_MODEL_SPECS = (
    {
        "id": "policyengine_small_l0",
        "label": "PE small-L0 local model",
        "relative_dir": "policyengine_us_data/storage/calibration/local_net_worth_100",
    },
    {
        "id": "policyengine_big_l0",
        "label": "PE big-L0 local model",
        "relative_dir": (
            "policyengine_us_data/storage/calibration/local_net_worth_100_e300"
        ),
    },
)


@dataclass(frozen=True)
class DashboardPaths:
    """Filesystem inputs for the dashboard payload."""

    artifact_root: Path = _DEFAULT_ARTIFACT_ROOT
    target_diagnostics_path: Path = _DEFAULT_TARGET_DIAGNOSTICS_PATH
    output_path: Path = _DEFAULT_OUTPUT_PATH


def build_dashboard_payload(
    *,
    artifact_root: str | Path = _DEFAULT_ARTIFACT_ROOT,
    target_diagnostics_path: str | Path = _DEFAULT_TARGET_DIAGNOSTICS_PATH,
    policyengine_us_data_repo: str | Path | None = _DEFAULT_POLICYENGINE_US_DATA_REPO,
    include_tmux: bool = True,
) -> dict[str, Any]:
    """Collect scores, local screens, active logs, and target diagnostics."""

    artifact_root = Path(artifact_root)
    target_diagnostics_path = Path(target_diagnostics_path)
    score_runs = collect_score_runs(artifact_root)
    local_screens = collect_local_target_screens(artifact_root)
    pe_l0_models = collect_policyengine_l0_models(policyengine_us_data_repo)
    actual_l0_runs = collect_actual_l0_objective_runs(artifact_root)
    materialized_l0_scores = collect_materialized_policyengine_l0_scores(artifact_root)
    run_contracts = collect_run_contracts(artifact_root)
    active_logs = collect_recent_log_summaries(artifact_root)
    tmux_sessions = collect_tmux_sessions() if include_tmux else []
    target_diagnostics = _read_json(target_diagnostics_path)
    generated_at = datetime.now(timezone.utc).isoformat()

    return {
        "dashboard_schema_version": 1,
        "generated_at": generated_at,
        "artifact_root": str(artifact_root),
        "target_diagnostics_path": (
            str(target_diagnostics_path) if target_diagnostics is not None else None
        ),
        "target_diagnostics": target_diagnostics,
        "run_board": {
            "generated_at": generated_at,
            "score_runs": score_runs,
            "local_target_screens": local_screens,
            "policyengine_l0_models": pe_l0_models,
            "actual_l0_objective_runs": actual_l0_runs,
            "materialized_policyengine_l0_scores": materialized_l0_scores,
            "run_contracts": run_contracts,
            "active_logs": active_logs,
            "tmux_sessions": tmux_sessions,
            "comparison_matrix": build_comparison_matrix(
                score_runs,
                local_screens,
                pe_l0_models,
                materialized_l0_scores,
            ),
            "apples_to_apples": build_apples_to_apples_groups(
                score_runs,
                local_screens,
                pe_l0_models,
                materialized_l0_scores,
            ),
            "assertions": build_dashboard_assertions(
                score_runs,
                local_screens,
                pe_l0_models,
                materialized_l0_scores,
            ),
        },
    }


def collect_score_runs(artifact_root: str | Path) -> list[dict[str, Any]]:
    """Read completed PE-native score artifacts under ``artifact_root``."""

    artifact_root = Path(artifact_root)
    runs: list[dict[str, Any]] = []
    for path in sorted(_iter_score_paths(artifact_root)):
        payload = _read_json(path)
        if payload is None:
            continue
        runs.extend(_score_entries_from_payload(path, payload))
    return sorted(
        runs,
        key=lambda row: (
            row.get("candidate_loss") is None,
            row.get("candidate_loss") or float("inf"),
            row.get("artifact_path") or "",
        ),
    )


def collect_run_contracts(artifact_root: str | Path) -> list[dict[str, Any]]:
    """Read machine-readable run contract summaries under ``artifact_root``."""

    artifact_root = Path(artifact_root)
    contracts: list[dict[str, Any]] = []
    for path in sorted(artifact_root.rglob("run_summary.json")):
        summary = _read_json(path)
        if not isinstance(summary, dict):
            continue
        manifest = _read_json(path.parent / "run_manifest.json") or {}
        contracts.append(
            {
                "artifact_dir": str(path.parent),
                "summary_path": str(path),
                "manifest_path": str(path.parent / "run_manifest.json"),
                "events_path": str(path.parent / "run_events.jsonl"),
                "status_source": "contract",
                "run_id": summary.get("run_id") or manifest.get("run_id"),
                "attempt_id": summary.get("attempt_id") or manifest.get("attempt_id"),
                "status": summary.get("status"),
                "active": summary.get("active"),
                "started_at": summary.get("started_at"),
                "updated_at": summary.get("updated_at"),
                "failed_at": summary.get("failed_at"),
                "completed_at": summary.get("completed_at"),
                "failed_event_id": summary.get("failed_event_id"),
                "failure": summary.get("failure"),
                "restart": summary.get("restart"),
                "completed_stages": summary.get("completed_stages") or [],
            }
        )
    return sorted(
        contracts,
        key=lambda row: str(row.get("updated_at") or ""),
        reverse=True,
    )


def collect_local_target_screens(artifact_root: str | Path) -> list[dict[str, Any]]:
    """Read cheap matrix-side local target screen summaries."""

    artifact_root = Path(artifact_root)
    screens = []
    for path in sorted(artifact_root.rglob("split_loss_summary.json")):
        payload = _read_json(path)
        if not isinstance(payload, dict):
            continue
        score_summary = _local_screen_score_summary(path.parent / "scores.json")
        screens.append(
            {
                "label": payload.get("candidate") or path.parent.name,
                "artifact_path": str(path),
                "artifact_dir": str(path.parent),
                "metric": "latest_pe_matrix_plus_cd_age_screen",
                "status": (
                    "screen_scored_latest_pe"
                    if score_summary is not None
                    else "screen_only"
                ),
                "broad_loss": _number_or_none(
                    payload.get("broad_objective_on_latest_pe_matrix_rows")
                ),
                "pe_native_score_path": (
                    str(path.parent / "scores.json")
                    if score_summary is not None
                    else None
                ),
                "pe_native_broad_loss": (
                    score_summary.get("candidate_loss")
                    if score_summary is not None
                    else None
                ),
                "pe_native_baseline_loss": (
                    score_summary.get("baseline_loss")
                    if score_summary is not None
                    else None
                ),
                "pe_native_loss_delta": (
                    score_summary.get("loss_delta")
                    if score_summary is not None
                    else None
                ),
                "pe_native_candidate_beats_baseline": (
                    score_summary.get("candidate_beats_baseline")
                    if score_summary is not None
                    else None
                ),
                "latest_pe_baseline_broad_loss": _number_or_none(
                    payload.get("latest_pe_baseline_broad_loss")
                ),
                "latest_winner_broad_objective": _number_or_none(
                    payload.get("latest_winner_broad_objective")
                ),
                "cd_age_target_weight": _number_or_none(
                    payload.get("cd_age_target_weight")
                ),
                "cd_age_mean_abs_relative_error": _number_or_none(
                    payload.get("cd_age_mean_abs_relative_error")
                ),
                "cd_age_p90_abs_relative_error": _number_or_none(
                    payload.get("cd_age_p90_abs_relative_error")
                ),
                "cd_age_p99_abs_relative_error": _number_or_none(
                    payload.get("cd_age_p99_abs_relative_error")
                ),
                "cd_age_max_abs_relative_error": _number_or_none(
                    payload.get("cd_age_max_abs_relative_error")
                ),
                "weight_sum": _number_or_none(payload.get("weight_sum")),
                "weights_path": payload.get("weights_path"),
            }
        )
    return sorted(
        screens,
        key=lambda row: (
            row.get("cd_age_mean_abs_relative_error") is None,
            row.get("cd_age_mean_abs_relative_error") or float("inf"),
            row.get("broad_loss") or float("inf"),
        ),
    )


def _local_screen_score_summary(path: Path) -> dict[str, Any] | None:
    """Return the latest-PE score summary colocated with a local target screen."""

    payload = _read_json(path)
    if isinstance(payload, list):
        payload = payload[0] if payload else None
    if not isinstance(payload, dict):
        return None
    summary = payload.get("summary")
    if not isinstance(summary, dict):
        summary = payload
    candidate_loss = _number_or_none(summary.get("candidate_enhanced_cps_native_loss"))
    baseline_loss = _number_or_none(summary.get("baseline_enhanced_cps_native_loss"))
    if candidate_loss is None:
        return None
    return {
        "candidate_loss": candidate_loss,
        "baseline_loss": baseline_loss,
        "loss_delta": _number_or_none(summary.get("enhanced_cps_native_loss_delta")),
        "candidate_beats_baseline": summary.get("candidate_beats_baseline"),
    }


def collect_policyengine_l0_models(
    policyengine_us_data_repo: str | Path | None,
) -> list[dict[str, Any]]:
    """Collect PE local-L0 weight-package diagnostics."""

    if policyengine_us_data_repo is None:
        return []
    repo = Path(policyengine_us_data_repo)
    models = []
    for spec in _PE_L0_MODEL_SPECS:
        model_dir = repo / spec["relative_dir"]
        config = _read_json(model_dir / "unified_run_config.json")
        diagnostics = _summarize_unified_diagnostics(
            model_dir / "unified_diagnostics.csv"
        )
        weights_path = model_dir / "calibration_weights.npy"
        present = isinstance(config, dict) and diagnostics is not None
        models.append(
            {
                "id": spec["id"],
                "label": spec["label"],
                "status": (
                    "available_weight_package" if present else "missing_weight_package"
                ),
                "artifact_dir": str(model_dir),
                "weights_path": str(weights_path) if weights_path.exists() else None,
                "config_path": (
                    str(model_dir / "unified_run_config.json")
                    if isinstance(config, dict)
                    else None
                ),
                "diagnostics_path": (
                    str(model_dir / "unified_diagnostics.csv")
                    if diagnostics is not None
                    else None
                ),
                "dataset": config.get("dataset") if isinstance(config, dict) else None,
                "db_path": config.get("db_path") if isinstance(config, dict) else None,
                "n_clones": (
                    _number_or_none(config.get("n_clones"))
                    if isinstance(config, dict)
                    else None
                ),
                "epochs": (
                    _number_or_none(config.get("epochs"))
                    if isinstance(config, dict)
                    else None
                ),
                "n_targets": (
                    _number_or_none(config.get("n_targets"))
                    if isinstance(config, dict)
                    else None
                ),
                "n_records": (
                    _number_or_none(config.get("n_records"))
                    if isinstance(config, dict)
                    else None
                ),
                "weight_sum": (
                    _number_or_none(config.get("weight_sum"))
                    if isinstance(config, dict)
                    else None
                ),
                "weight_nonzero": (
                    _number_or_none(config.get("weight_nonzero"))
                    if isinstance(config, dict)
                    else None
                ),
                "mean_error_pct": (
                    _number_or_none(config.get("mean_error_pct"))
                    if isinstance(config, dict)
                    else None
                ),
                "elapsed_seconds": (
                    _number_or_none(config.get("elapsed_seconds"))
                    if isinstance(config, dict)
                    else None
                ),
                "diagnostics": diagnostics,
                "same_harness_materialization": _inspect_l0_materialization(
                    model_dir=model_dir,
                    config=config,
                    weights_path=weights_path,
                ),
                "notes": (
                    "PE local-L0 fit metrics come from unified_diagnostics.csv. "
                    "Same-harness broad/latest score remains missing until this "
                    "weight package is materialized as a scored H5."
                ),
            }
        )
    return models


def collect_actual_l0_objective_runs(
    artifact_root: str | Path,
) -> list[dict[str, Any]]:
    """Collect local unified-calibration runs scored on the actual L0 objective."""

    artifact_root = Path(artifact_root)
    runs: list[dict[str, Any]] = []
    for diagnostics_path in sorted(artifact_root.rglob("unified_diagnostics.csv")):
        diagnostics = _summarize_unified_diagnostics(diagnostics_path)
        if diagnostics is None:
            continue
        run_dir = diagnostics_path.parent
        weights_path = run_dir / "calibration_weights.npy"
        config = _read_json(run_dir / "unified_run_config.json")
        weight_summary = _weight_file_summary(weights_path)
        runs.append(
            {
                "label": run_dir.name,
                "artifact_dir": str(run_dir),
                "diagnostics_path": str(diagnostics_path),
                "config_path": (
                    str(run_dir / "unified_run_config.json")
                    if isinstance(config, dict)
                    else None
                ),
                "weights_path": str(weights_path) if weights_path.exists() else None,
                "status": "complete",
                "model_id": _infer_actual_l0_model_id(run_dir),
                "actual_l0_data_loss": diagnostics.get("actual_l0_data_loss"),
                "actual_l0_mean_abs_relative_error_pct": diagnostics.get(
                    "actual_l0_mean_abs_relative_error_pct"
                ),
                "n_targets": diagnostics.get("n_targets"),
                "n_achievable": diagnostics.get("n_achievable"),
                "n_clones": (
                    _number_or_none(config.get("n_clones"))
                    if isinstance(config, dict)
                    else None
                ),
                "epochs": (
                    _number_or_none(config.get("epochs"))
                    if isinstance(config, dict)
                    else None
                ),
                "weights": weight_summary,
                "diagnostics": diagnostics,
            }
        )
    return sorted(
        runs,
        key=lambda row: (
            row.get("actual_l0_data_loss") is None,
            row.get("actual_l0_data_loss") or float("inf"),
            row.get("artifact_dir") or "",
        ),
    )


def collect_materialized_policyengine_l0_scores(
    artifact_root: str | Path,
) -> list[dict[str, Any]]:
    """Read PE local-area L0 materializations scored through broad diagnostics."""

    artifact_root = Path(artifact_root)
    scores: list[dict[str, Any]] = []
    for path in sorted(
        artifact_root.rglob("pe_local_area_l0_state_stack_vs_legacy_ecps.json")
    ):
        payload = _read_json(path)
        if not isinstance(payload, dict):
            continue
        summary = payload.get("summary")
        if not isinstance(summary, dict):
            continue
        candidate_loss = _number_or_none(summary.get("to_loss"))
        baseline_loss = _number_or_none(summary.get("from_loss"))
        if candidate_loss is None or baseline_loss is None:
            continue
        scores.append(
            {
                "id": "policyengine_local_area_l0_state_stack",
                "label": "PE local-area L0 state stack",
                "status": "same_harness_scored_experimental",
                "artifact_path": str(path),
                "artifact_dir": str(path.parent),
                "metric": payload.get("metric")
                or "enhanced_cps_native_loss_target_delta",
                "metric_runtime": "legacy_or_patched_runtime",
                "candidate_loss": candidate_loss,
                "baseline_loss": baseline_loss,
                "candidate_beats_baseline": candidate_loss < baseline_loss,
                "loss_delta": _number_or_none(summary.get("loss_delta")),
                "n_targets": _number_or_none(summary.get("n_targets")),
                "state_score_count": _number_or_none(payload.get("state_score_count")),
                "state_weight_sum": _number_or_none(payload.get("state_weight_sum")),
                "notes": (
                    "This is an experimental materialized state-stack score. "
                    "It is a broad same-harness artifact, but it is not the "
                    "small-L0 or big-L0 weight package unless the source path "
                    "says so."
                ),
            }
        )
    return sorted(
        scores,
        key=lambda row: (
            row.get("candidate_loss") is None,
            row.get("candidate_loss") or float("inf"),
            row.get("artifact_path") or "",
        ),
    )


def collect_recent_log_summaries(
    artifact_root: str | Path, *, limit: int = 12
) -> list[dict[str, Any]]:
    """Summarize recent logs with row-batch progress lines."""

    artifact_root = Path(artifact_root)
    paths = sorted(
        (path for path in artifact_root.rglob("*.log") if path.is_file()),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )[:limit]
    summaries = []
    for path in paths:
        tail = _tail_text(path)
        progress = _parse_row_batch_progress(tail)
        summaries.append(
            {
                "path": str(path),
                "modified_at": datetime.fromtimestamp(
                    path.stat().st_mtime, timezone.utc
                ).isoformat(),
                "progress": progress,
                "last_lines": tail.splitlines()[-5:],
            }
        )
    return summaries


def collect_tmux_sessions() -> list[dict[str, Any]]:
    """Return current tmux sessions when tmux is available."""

    try:
        completed = subprocess.run(
            ["tmux", "ls"],
            check=False,
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return []
    if completed.returncode != 0:
        return []
    sessions = []
    for line in completed.stdout.splitlines():
        if not line.strip():
            continue
        name = line.split(":", 1)[0]
        if not _is_relevant_tmux_session(name):
            continue
        sessions.append({"name": name, "raw": line})
    return sorted(
        sessions, key=lambda row: (not row["name"].startswith("mp_"), row["name"])
    )


def build_comparison_matrix(
    score_runs: list[dict[str, Any]],
    local_screens: list[dict[str, Any]],
    pe_l0_models: list[dict[str, Any]],
    materialized_l0_scores: list[dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    """Build a compact answer matrix for the current PE comparison question."""

    materialized_l0_scores = materialized_l0_scores or []
    best_latest = _best_score(
        score_runs,
        predicate=lambda row: (
            row.get("metric_runtime") == "latest_policyengine_us"
            and row.get("model_id") == "microplex_current_best"
        ),
    )
    best_legacy = _best_score(
        score_runs,
        predicate=lambda row: (
            row.get("metric_runtime") == "legacy_or_patched_runtime"
            and row.get("model_id") == "microplex_current_best"
        ),
    )
    best_local = local_screens[0] if local_screens else None
    pe_l0_by_id = {row.get("id"): row for row in pe_l0_models}

    rows: list[dict[str, Any]] = []
    for slot in _PE_MODEL_SLOTS:
        row = dict(slot)
        if slot["id"] == "policyengine_legacy_ecps" and best_latest is not None:
            row.update(
                {
                    "latest_pe_broad_loss": best_latest.get("baseline_loss"),
                    "latest_pe_status": "available",
                    "legacy_metric_loss": (
                        best_legacy.get("baseline_loss")
                        if best_legacy is not None
                        else None
                    ),
                    "legacy_metric_status": (
                        "available" if best_legacy is not None else "missing"
                    ),
                }
            )
        elif slot["id"] in pe_l0_by_id:
            model = pe_l0_by_id[slot["id"]]
            diagnostics = model.get("diagnostics") or {}
            latest_score = _best_model_metric_score(
                score_runs,
                model_id=str(slot["id"]),
                metric_runtime="latest_policyengine_us",
            )
            legacy_score = _best_model_metric_score(
                score_runs,
                model_id=str(slot["id"]),
                metric_runtime="legacy_or_patched_runtime",
            )
            row.update(
                {
                    "status": (
                        "same_harness_scored"
                        if latest_score is not None or legacy_score is not None
                        else model.get("status")
                    ),
                    "artifact_dir": model.get("artifact_dir"),
                    "latest_pe_broad_loss": (
                        latest_score.get("candidate_loss")
                        if latest_score is not None
                        else None
                    ),
                    "latest_pe_status": (
                        "scored" if latest_score is not None else "missing_h5_score"
                    ),
                    "legacy_metric_loss": (
                        legacy_score.get("candidate_loss")
                        if legacy_score is not None
                        else None
                    ),
                    "legacy_metric_status": (
                        "scored" if legacy_score is not None else "missing_h5_score"
                    ),
                    "pe_local_l0_mean_abs_error_pct": diagnostics.get(
                        "mean_abs_relative_error_pct"
                    )
                    or model.get("mean_error_pct"),
                    "pe_local_l0_median_abs_error_pct": diagnostics.get(
                        "median_abs_relative_error_pct"
                    ),
                    "pe_local_l0_p90_abs_error_pct": diagnostics.get(
                        "p90_abs_relative_error_pct"
                    ),
                    "pe_local_l0_targets": diagnostics.get("n_targets")
                    or model.get("n_targets"),
                    "pe_local_l0_epochs": model.get("epochs"),
                    "pe_local_l0_weight_nonzero": model.get("weight_nonzero"),
                    "notes": (
                        "Same-harness H5 score is available."
                        if latest_score is not None or legacy_score is not None
                        else model.get("notes") or row.get("notes")
                    ),
                }
            )
        else:
            row.update(
                {
                    "latest_pe_broad_loss": None,
                    "latest_pe_status": "missing",
                    "legacy_metric_loss": None,
                    "legacy_metric_status": "missing",
                }
            )
        rows.append(row)

    best_materialized_l0 = _best_materialized_l0_score(materialized_l0_scores)
    if best_materialized_l0 is not None:
        rows.append(
            {
                "id": best_materialized_l0.get("id"),
                "label": best_materialized_l0.get("label"),
                "status": best_materialized_l0.get("status"),
                "latest_pe_broad_loss": None,
                "latest_pe_status": None,
                "legacy_metric_loss": best_materialized_l0.get("candidate_loss"),
                "legacy_metric_baseline_loss": best_materialized_l0.get(
                    "baseline_loss"
                ),
                "legacy_metric_status": (
                    "beats_legacy_pe_baseline"
                    if best_materialized_l0.get("candidate_beats_baseline")
                    else "worse_than_legacy_pe_baseline"
                ),
                "artifact_path": best_materialized_l0.get("artifact_path"),
                "notes": best_materialized_l0.get("notes"),
            }
        )

    rows.append(
        {
            "id": "microplex_current_best",
            "label": "Microplex current best",
            "status": "available" if best_latest is not None else "missing",
            "latest_pe_broad_loss": (
                best_latest.get("candidate_loss") if best_latest is not None else None
            ),
            "latest_pe_baseline_loss": (
                best_latest.get("baseline_loss") if best_latest is not None else None
            ),
            "latest_pe_status": (
                "beats_legacy_pe_baseline"
                if best_latest is not None
                and best_latest.get("candidate_beats_baseline")
                else "missing"
            ),
            "legacy_metric_loss": (
                best_legacy.get("candidate_loss") if best_legacy is not None else None
            ),
            "legacy_metric_baseline_loss": (
                best_legacy.get("baseline_loss") if best_legacy is not None else None
            ),
            "legacy_metric_status": (
                "beats_legacy_pe_baseline"
                if best_legacy is not None
                and best_legacy.get("candidate_beats_baseline")
                else "missing"
            ),
            "local_cd_age_screen_loss": (
                best_local.get("broad_loss") if best_local is not None else None
            ),
            "local_cd_age_mare": (
                best_local.get("cd_age_mean_abs_relative_error")
                if best_local is not None
                else None
            ),
            "artifact_path": (
                best_latest.get("artifact_path") if best_latest is not None else None
            ),
            "record_count_tier": (
                best_latest.get("record_count_tier")
                if best_latest is not None
                else None
            ),
            "release_smoke": (
                best_latest.get("release_smoke") if best_latest is not None else None
            ),
            "notes": (
                "This is the best completed Microplex score found locally. "
                "The CD-age row is a matrix screen until the latest-PE row-batch "
                "score finishes."
            ),
        }
    )
    return rows


def build_apples_to_apples_groups(
    score_runs: list[dict[str, Any]],
    local_screens: list[dict[str, Any]],
    pe_l0_models: list[dict[str, Any]],
    materialized_l0_scores: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Group comparisons that share an actual metric and target universe."""

    best_latest = _best_score(
        score_runs,
        predicate=lambda row: (
            row.get("metric_runtime") == "latest_policyengine_us"
            and row.get("model_id") == "microplex_current_best"
        ),
    )
    best_legacy = _best_score(
        score_runs,
        predicate=lambda row: (
            row.get("metric_runtime") == "legacy_or_patched_runtime"
            and row.get("model_id") == "microplex_current_best"
        ),
    )
    best_local = local_screens[0] if local_screens else None
    best_materialized_l0 = _best_materialized_l0_score(materialized_l0_scores)
    pe_l0_by_id = {row.get("id"): row for row in pe_l0_models}

    latest_small = _best_model_metric_score(
        score_runs,
        model_id="policyengine_small_l0",
        metric_runtime="latest_policyengine_us",
    )
    latest_big = _best_model_metric_score(
        score_runs,
        model_id="policyengine_big_l0",
        metric_runtime="latest_policyengine_us",
    )
    legacy_small = _best_model_metric_score(
        score_runs,
        model_id="policyengine_small_l0",
        metric_runtime="legacy_or_patched_runtime",
    )
    legacy_big = _best_model_metric_score(
        score_runs,
        model_id="policyengine_big_l0",
        metric_runtime="legacy_or_patched_runtime",
    )

    groups = [
        {
            "id": "latest_pe_broad",
            "label": "Latest PolicyEngine broad target loss",
            "metric_scope": "same_harness_latest_pe_broad",
            "status": (
                "complete"
                if best_latest and latest_small and latest_big
                else "partial"
                if best_latest
                else "missing"
            ),
            "rows": [
                _comparison_row(
                    model_id="policyengine_legacy_ecps",
                    label="PE legacy enhanced CPS",
                    score=(
                        best_latest.get("baseline_loss")
                        if best_latest is not None
                        else None
                    ),
                    status="scored_baseline" if best_latest else "missing",
                ),
                _comparison_row(
                    model_id="microplex_current_best",
                    label="Microplex current best",
                    score=(
                        best_latest.get("candidate_loss")
                        if best_latest is not None
                        else None
                    ),
                    status=(
                        "scored_candidate_beats_baseline"
                        if best_latest and best_latest.get("candidate_beats_baseline")
                        else "missing"
                    ),
                    artifact_path=(
                        best_latest.get("artifact_path")
                        if best_latest is not None
                        else None
                    ),
                ),
                _scored_or_missing_l0_row(
                    pe_l0_by_id,
                    "policyengine_small_l0",
                    latest_small,
                ),
                _scored_or_missing_l0_row(
                    pe_l0_by_id,
                    "policyengine_big_l0",
                    latest_big,
                ),
            ],
        },
        {
            "id": "legacy_broad",
            "label": "Legacy broad target loss",
            "metric_scope": "same_harness_legacy_broad",
            "status": (
                "complete"
                if best_legacy and legacy_small and legacy_big
                else "partial"
                if best_legacy
                else "missing"
            ),
            "rows": [
                _comparison_row(
                    model_id="policyengine_legacy_ecps",
                    label="PE legacy enhanced CPS",
                    score=(
                        best_legacy.get("baseline_loss")
                        if best_legacy is not None
                        else None
                    ),
                    status="scored_baseline" if best_legacy else "missing",
                ),
                _comparison_row(
                    model_id="microplex_current_best",
                    label="Microplex current best",
                    score=(
                        best_legacy.get("candidate_loss")
                        if best_legacy is not None
                        else None
                    ),
                    status=(
                        "scored_candidate_beats_baseline"
                        if best_legacy and best_legacy.get("candidate_beats_baseline")
                        else "missing"
                    ),
                    artifact_path=(
                        best_legacy.get("artifact_path")
                        if best_legacy is not None
                        else None
                    ),
                ),
                _comparison_row(
                    model_id="policyengine_local_area_l0_state_stack",
                    label="PE local-area L0 state stack",
                    score=(
                        best_materialized_l0.get("candidate_loss")
                        if best_materialized_l0 is not None
                        else None
                    ),
                    status=(
                        best_materialized_l0.get("status")
                        if best_materialized_l0 is not None
                        else "missing"
                    ),
                    artifact_path=(
                        best_materialized_l0.get("artifact_path")
                        if best_materialized_l0 is not None
                        else None
                    ),
                    detail=(
                        "Experimental materialization"
                        if best_materialized_l0 is not None
                        else None
                    ),
                ),
                _scored_or_missing_l0_row(
                    pe_l0_by_id,
                    "policyengine_small_l0",
                    legacy_small,
                ),
                _scored_or_missing_l0_row(
                    pe_l0_by_id,
                    "policyengine_big_l0",
                    legacy_big,
                ),
            ],
        },
        {
            "id": "pe_local_l0_native",
            "label": "PE local-L0 native target diagnostics",
            "metric_scope": "pe_native_local_l0_diagnostics",
            "status": "native_only",
            "rows": [
                _native_pe_l0_row(pe_l0_by_id, "policyengine_small_l0"),
                _native_pe_l0_row(pe_l0_by_id, "policyengine_big_l0"),
                _comparison_row(
                    model_id="microplex_cd_age_screen",
                    label="Microplex CD-age screen",
                    score=(
                        100 * best_local.get("cd_age_mean_abs_relative_error")
                        if best_local is not None
                        and best_local.get("cd_age_mean_abs_relative_error") is not None
                        else None
                    ),
                    status=(
                        "different_target_set_screen_only"
                        if best_local is not None
                        else "missing"
                    ),
                    artifact_path=(
                        best_local.get("artifact_path")
                        if best_local is not None
                        else None
                    ),
                    detail=(
                        "Displayed for tracking only; not used as a PE local-L0 "
                        "native comparison."
                    ),
                ),
            ],
        },
    ]
    return groups


def build_dashboard_assertions(
    score_runs: list[dict[str, Any]],
    local_screens: list[dict[str, Any]],
    pe_l0_models: list[dict[str, Any]],
    materialized_l0_scores: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """State which comparison claims are supported by completed artifacts."""

    materialized_l0_scores = materialized_l0_scores or []
    best_latest = _best_score(
        score_runs,
        predicate=lambda row: (
            row.get("metric_runtime") == "latest_policyengine_us"
            and row.get("model_id") == "microplex_current_best"
        ),
    )
    best_legacy = _best_score(
        score_runs,
        predicate=lambda row: (
            row.get("metric_runtime") == "legacy_or_patched_runtime"
            and row.get("model_id") == "microplex_current_best"
        ),
    )
    pe_l0_by_id = {row.get("id"): row for row in pe_l0_models}
    small_l0_present = (
        pe_l0_by_id.get("policyengine_small_l0", {}).get("status")
        == "available_weight_package"
    )
    big_l0_present = (
        pe_l0_by_id.get("policyengine_big_l0", {}).get("status")
        == "available_weight_package"
    )
    best_materialized_l0 = _best_materialized_l0_score(materialized_l0_scores)
    small_latest = _best_model_metric_score(
        score_runs,
        model_id="policyengine_small_l0",
        metric_runtime="latest_policyengine_us",
    )
    small_legacy = _best_model_metric_score(
        score_runs,
        model_id="policyengine_small_l0",
        metric_runtime="legacy_or_patched_runtime",
    )
    big_latest = _best_model_metric_score(
        score_runs,
        model_id="policyengine_big_l0",
        metric_runtime="latest_policyengine_us",
    )
    big_legacy = _best_model_metric_score(
        score_runs,
        model_id="policyengine_big_l0",
        metric_runtime="legacy_or_patched_runtime",
    )
    small_complete = bool(small_latest and small_legacy)
    big_complete = bool(big_latest and big_legacy)
    all_models_complete = bool(
        best_latest and best_legacy and small_complete and big_complete
    )
    best_latest_release_smoke = (
        best_latest.get("release_smoke") if isinstance(best_latest, dict) else None
    )
    return {
        "microplex_beats_legacy_ecps_latest_pe_broad": bool(
            best_latest and best_latest.get("candidate_beats_baseline")
        ),
        "microplex_beats_legacy_ecps_legacy_metric": bool(
            best_legacy and best_legacy.get("candidate_beats_baseline")
        ),
        "microplex_current_best_has_release_smoke": bool(best_latest_release_smoke),
        "microplex_current_best_release_smoke_passes": bool(
            isinstance(best_latest_release_smoke, dict)
            and best_latest_release_smoke.get("passes_file_size_ratio_2x")
            and best_latest_release_smoke.get("passes_runtime_ratio_1_25x")
        ),
        "microplex_vs_small_l0_complete": small_complete,
        "microplex_vs_big_l0_complete": big_complete,
        "microplex_vs_all_three_pe_models_on_both_metrics": all_models_complete,
        "policyengine_small_l0_weight_package_available": small_l0_present,
        "policyengine_big_l0_weight_package_available": big_l0_present,
        "policyengine_materialized_l0_same_harness_available": bool(
            best_materialized_l0
        ),
        "local_cd_age_screen_available": bool(local_screens),
        "apples_to_apples_groups_available": True,
        "caveat": (
            "Small-L0 and big-L0 PE weight packages are wired into the run "
            "board when available. The all-three-PE-model claim is supported "
            "only when both materialized PE L0 packages have legacy and latest "
            "same-harness scores."
        ),
    }


def _comparison_row(
    *,
    model_id: str,
    label: str,
    score: float | None,
    status: str,
    artifact_path: str | None = None,
    detail: str | None = None,
) -> dict[str, Any]:
    return {
        "model_id": model_id,
        "label": label,
        "score": _number_or_none(score),
        "status": status,
        "artifact_path": artifact_path,
        "detail": detail,
    }


def _missing_h5_row(
    pe_l0_by_id: dict[str, dict[str, Any]], model_id: str
) -> dict[str, Any]:
    model = pe_l0_by_id.get(model_id) or {}
    materialization = model.get("same_harness_materialization")
    blocker = None
    if isinstance(materialization, dict):
        blocker = materialization.get("status")
    return _comparison_row(
        model_id=model_id,
        label=str(model.get("label") or model_id),
        score=None,
        status="missing_same_harness_h5_score",
        artifact_path=model.get("artifact_dir"),
        detail=blocker,
    )


def _scored_or_missing_l0_row(
    pe_l0_by_id: dict[str, dict[str, Any]],
    model_id: str,
    score: dict[str, Any] | None,
) -> dict[str, Any]:
    if score is None:
        return _missing_h5_row(pe_l0_by_id, model_id)
    model = pe_l0_by_id.get(model_id) or {}
    return _comparison_row(
        model_id=model_id,
        label=str(model.get("label") or model_id),
        score=score.get("candidate_loss"),
        status=(
            "scored_candidate_beats_legacy_ecps"
            if score.get("candidate_beats_baseline")
            else "scored_candidate_worse_than_legacy_ecps"
        ),
        artifact_path=score.get("artifact_path"),
        detail=(
            f"{int(score['n_targets_kept']):,} targets"
            if _number_or_none(score.get("n_targets_kept")) is not None
            else None
        ),
    )


def _native_pe_l0_row(
    pe_l0_by_id: dict[str, dict[str, Any]], model_id: str
) -> dict[str, Any]:
    model = pe_l0_by_id.get(model_id) or {}
    diagnostics = model.get("diagnostics") or {}
    score = diagnostics.get("mean_abs_relative_error_pct") or model.get(
        "mean_error_pct"
    )
    targets = diagnostics.get("n_targets") or model.get("n_targets")
    return _comparison_row(
        model_id=model_id,
        label=str(model.get("label") or model_id),
        score=_number_or_none(score),
        status=(
            "native_diagnostics_available"
            if _number_or_none(score) is not None
            else "missing_native_diagnostics"
        ),
        artifact_path=model.get("diagnostics_path") or model.get("artifact_dir"),
        detail=(
            f"{format(int(targets), ',')} PE-local targets"
            if _number_or_none(targets) is not None
            else None
        ),
    )


def _best_materialized_l0_score(
    rows: list[dict[str, Any]],
) -> dict[str, Any] | None:
    candidates = [
        row for row in rows if _number_or_none(row.get("candidate_loss")) is not None
    ]
    if not candidates:
        return None
    return min(candidates, key=lambda row: row["candidate_loss"])


def _weight_file_summary(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        import numpy as np

        weights = np.asarray(np.load(path), dtype=float)
    except Exception:  # pragma: no cover - defensive artifact read
        return {"status": "unreadable", "path": str(path)}
    return {
        "status": "ok",
        "path": str(path),
        "records": int(weights.size),
        "nonzero": int((weights > 0.0).sum()),
        "greater_than_1": int((weights > 1.0).sum()),
        "greater_than_100": int((weights > 100.0).sum()),
        "sum": float(weights.sum()),
    }


def _infer_actual_l0_model_id(run_dir: Path) -> str:
    text = str(run_dir).lower()
    if "microplex" in text or "mp_" in text:
        return "microplex_actual_l0"
    if "local_net_worth_100_e300" in text:
        return "policyengine_big_l0"
    if "local_net_worth_100" in text:
        return "policyengine_small_l0"
    return "unknown_actual_l0"


def _inspect_l0_materialization(
    *,
    model_dir: Path,
    config: Any,
    weights_path: Path,
) -> dict[str, Any]:
    """Return a cheap compatibility check for materializing a PE-L0 package."""

    result: dict[str, Any] = {"status": "unknown"}
    if not weights_path.exists():
        result["status"] = "missing_weights"
        return result

    try:
        import numpy as np

        weights = np.load(weights_path, mmap_mode="r")
        weight_count = int(weights.shape[0])
        result["weight_count"] = weight_count
    except Exception as error:  # pragma: no cover - defensive artifact read
        result["status"] = "weights_unreadable"
        result["error"] = str(error)
        return result

    geography_path = model_dir / "geography.npz"
    if geography_path.exists():
        try:
            import numpy as np

            with np.load(geography_path, allow_pickle=True) as geography:
                if "block_geoid" in geography:
                    result["geography_row_count"] = int(
                        geography["block_geoid"].shape[0]
                    )
                if "n_records" in geography:
                    result["geography_n_records"] = int(geography["n_records"][0])
                if "n_clones" in geography:
                    result["geography_n_clones"] = int(geography["n_clones"][0])
        except Exception as error:  # pragma: no cover - defensive artifact read
            result["geography_error"] = str(error)

    dataset_path = None
    if isinstance(config, dict) and config.get("dataset"):
        dataset_path = Path(str(config["dataset"]))
        result["dataset_path"] = str(dataset_path)
    if dataset_path is None or not dataset_path.exists():
        result["status"] = "source_h5_missing"
        return result

    household_count = _h5_period_length(dataset_path, "household_id")
    result["source_household_count"] = household_count
    if household_count is None:
        result["status"] = "source_h5_unreadable"
        return result

    if household_count > 0 and weight_count % household_count == 0:
        result["status"] = "materializable_against_current_source_h5"
        result["implied_clone_count"] = weight_count // household_count
    else:
        result["status"] = "incompatible_current_source_h5"
        result["detail"] = (
            "Weight count is not divisible by the current source H5 household "
            "count; same-harness scoring needs the matching source dataset or "
            "a regenerated L0 package."
        )
    return result


def _h5_period_length(path: Path, variable: str) -> int | None:
    try:
        import h5py

        with h5py.File(path, "r") as handle:
            if variable not in handle:
                return None
            obj = handle[variable]
            if hasattr(obj, "keys"):
                keys = list(obj.keys())
                if not keys:
                    return None
                return int(obj[keys[0]].shape[0])
            return int(obj.shape[0])
    except Exception:  # pragma: no cover - defensive artifact read
        return None


def write_dashboard_payload(
    output_path: str | Path = _DEFAULT_OUTPUT_PATH,
    *,
    artifact_root: str | Path = _DEFAULT_ARTIFACT_ROOT,
    target_diagnostics_path: str | Path = _DEFAULT_TARGET_DIAGNOSTICS_PATH,
    policyengine_us_data_repo: str | Path | None = _DEFAULT_POLICYENGINE_US_DATA_REPO,
    include_tmux: bool = True,
) -> Path:
    """Write the living dashboard JSON payload."""

    payload = build_dashboard_payload(
        artifact_root=artifact_root,
        target_diagnostics_path=target_diagnostics_path,
        policyengine_us_data_repo=policyengine_us_data_repo,
        include_tmux=include_tmux,
    )
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2, sort_keys=True))
    return output_path


def _iter_score_paths(artifact_root: Path) -> list[Path]:
    paths = list(artifact_root.rglob("scores.json"))
    paths.extend(artifact_root.rglob("policyengine_native_scores.json"))
    paths.extend(artifact_root.rglob("*_score.json"))
    return [path for path in paths if path.is_file()]


def _score_entries_from_payload(path: Path, payload: Any) -> list[dict[str, Any]]:
    if isinstance(payload, list):
        raw_entries = payload
    elif isinstance(payload, dict) and "broad_loss" in payload:
        raw_entries = [payload]
    elif isinstance(payload, dict) and "summary" in payload:
        raw_entries = [payload]
    elif isinstance(payload, dict) and "candidate_enhanced_cps_native_loss" in payload:
        raw_entries = [payload]
    else:
        return []

    entries = []
    for index, item in enumerate(raw_entries):
        if not isinstance(item, dict):
            continue
        if "candidate_enhanced_cps_native_loss" in item:
            summary = item
            broad_loss = item
        else:
            summary = (
                item.get("summary") if isinstance(item.get("summary"), dict) else {}
            )
            broad_loss = (
                item.get("broad_loss")
                if isinstance(item.get("broad_loss"), dict)
                else {}
            )
        candidate_loss = _number_or_none(
            summary.get("candidate_enhanced_cps_native_loss")
        )
        baseline_loss = _number_or_none(
            summary.get("baseline_enhanced_cps_native_loss")
        )
        if candidate_loss is None or baseline_loss is None:
            continue
        candidate_dataset = broad_loss.get("candidate_dataset")
        baseline_dataset = broad_loss.get("baseline_dataset")
        metric_runtime = _infer_metric_runtime(path, summary)
        model_id = _infer_score_model_id(path, candidate_dataset)
        label = _score_label(path, candidate_dataset, index)
        release_smoke = _release_smoke_summary(path.parent)
        record_count_tier = _infer_record_count_tier(
            path, candidate_dataset
        ) or _infer_record_count_tier_from_release_smoke(release_smoke)
        entries.append(
            {
                "label": label,
                "model_id": model_id,
                "record_count_tier": record_count_tier,
                "artifact_path": str(path),
                "artifact_dir": str(path.parent),
                "entry_index": index,
                "metric": item.get("metric") or "pe_native_broad_loss",
                "metric_runtime": metric_runtime,
                "period": item.get("period") or summary.get("period") or 2024,
                "candidate_dataset": candidate_dataset,
                "baseline_dataset": baseline_dataset,
                "candidate_loss": candidate_loss,
                "baseline_loss": baseline_loss,
                "loss_delta": _number_or_none(
                    summary.get("enhanced_cps_native_loss_delta")
                ),
                "candidate_beats_baseline": bool(
                    summary.get("candidate_beats_baseline")
                ),
                "candidate_unweighted_msre": _number_or_none(
                    summary.get("candidate_unweighted_msre")
                ),
                "baseline_unweighted_msre": _number_or_none(
                    summary.get("baseline_unweighted_msre")
                ),
                "n_targets_kept": _number_or_none(summary.get("n_targets_kept")),
                "n_targets_total": _number_or_none(summary.get("n_targets_total")),
                "candidate_weight_sum": _number_or_none(
                    broad_loss.get("candidate_weight_sum")
                ),
                "baseline_weight_sum": _number_or_none(
                    broad_loss.get("baseline_weight_sum")
                ),
                "release_smoke": release_smoke,
                "source_kind": "scores_json",
            }
        )
    return entries


def _release_smoke_summary(artifact_dir: Path) -> dict[str, Any] | None:
    """Read colocated lightweight release gate smoke output when present."""

    path = artifact_dir / "runtime_smoke_loader.json"
    payload = _read_json(path)
    if not isinstance(payload, dict):
        return None

    candidate = payload.get("candidate")
    baseline = payload.get("baseline")
    if not isinstance(candidate, dict) or not isinstance(baseline, dict):
        return None

    runtime_ratio = _number_or_none(
        payload.get("median_runtime_ratio") or payload.get("runtime_ratio")
    )
    file_size_ratio = _number_or_none(payload.get("file_size_ratio"))
    household_ratio = _number_or_none(payload.get("household_ratio"))
    return {
        "artifact_path": str(path),
        "benchmark": payload.get("benchmark"),
        "candidate_households": _number_or_none(candidate.get("households")),
        "baseline_households": _number_or_none(baseline.get("households")),
        "household_ratio": household_ratio,
        "candidate_file_size_bytes": _number_or_none(candidate.get("file_size_bytes")),
        "baseline_file_size_bytes": _number_or_none(baseline.get("file_size_bytes")),
        "file_size_ratio": file_size_ratio,
        "median_runtime_ratio": runtime_ratio,
        "candidate_median_elapsed_seconds": _number_or_none(
            candidate.get("median_elapsed_seconds") or candidate.get("elapsed_seconds")
        ),
        "baseline_median_elapsed_seconds": _number_or_none(
            baseline.get("median_elapsed_seconds") or baseline.get("elapsed_seconds")
        ),
        "raw_candidate_household_weight_sum": _number_or_none(
            candidate.get("raw_household_weight_sum")
        ),
        "raw_baseline_household_weight_sum": _number_or_none(
            baseline.get("raw_household_weight_sum")
        ),
        "passes_file_size_ratio_2x": (
            None if file_size_ratio is None else file_size_ratio <= 2.0
        ),
        "passes_runtime_ratio_1_25x": (
            None if runtime_ratio is None else runtime_ratio <= 1.25
        ),
    }


def _summarize_unified_diagnostics(path: Path) -> dict[str, Any] | None:
    try:
        with path.open(newline="") as file:
            rows = list(csv.DictReader(file))
    except OSError:
        return None
    if not rows:
        return None

    abs_errors = []
    actual_l0_abs_errors = []
    actual_l0_squared_errors = []
    achievable_count = 0
    for row in rows:
        if str(row.get("achievable", "")).lower() == "true":
            achievable_count += 1
        error = _number_or_none(row.get("abs_rel_error"))
        if error is not None:
            abs_errors.append(error)
        estimate = _number_or_none(row.get("estimate"))
        true_value = _number_or_none(row.get("true_value"))
        if estimate is not None and true_value is not None:
            actual_error = (estimate - true_value) / (true_value + 1.0)
            actual_l0_abs_errors.append(abs(actual_error))
            actual_l0_squared_errors.append(actual_error * actual_error)

    sorted_errors = sorted(abs_errors)
    return {
        "n_targets": len(rows),
        "n_achievable": achievable_count,
        "actual_l0_objective": ("sum(((estimate - target) / (target + 1)) ** 2)"),
        "actual_l0_data_loss": (
            sum(actual_l0_squared_errors) if actual_l0_squared_errors else None
        ),
        "actual_l0_mean_abs_relative_error_pct": (
            100 * sum(actual_l0_abs_errors) / len(actual_l0_abs_errors)
            if actual_l0_abs_errors
            else None
        ),
        "mean_abs_relative_error_pct": (
            100 * sum(abs_errors) / len(abs_errors) if abs_errors else None
        ),
        "median_abs_relative_error_pct": _percentile(sorted_errors, 0.5),
        "p90_abs_relative_error_pct": _percentile(sorted_errors, 0.9),
        "p99_abs_relative_error_pct": _percentile(sorted_errors, 0.99),
        "max_abs_relative_error_pct": (
            100 * sorted_errors[-1] if sorted_errors else None
        ),
        "share_under_10pct": _share_under(abs_errors, 0.10),
        "share_under_25pct": _share_under(abs_errors, 0.25),
    }


def _score_label(path: Path, candidate_dataset: Any, index: int) -> str:
    artifact = path.parent.name
    if isinstance(candidate_dataset, str):
        dataset_name = Path(candidate_dataset).name
        if dataset_name != "policyengine_us.h5":
            return f"{artifact} / {dataset_name}"
    if index:
        return f"{artifact} / candidate {index + 1}"
    return artifact


def _infer_metric_runtime(path: Path, summary: dict[str, Any]) -> str:
    text = str(path).lower()
    n_targets = _number_or_none(summary.get("n_targets_kept"))
    baseline_loss = _number_or_none(summary.get("baseline_enhanced_cps_native_loss"))
    if "legacy_targets" in text:
        return "legacy_or_patched_runtime"
    if "new_targets" in text:
        return "latest_policyengine_us"
    if n_targets == 2805 and baseline_loss == 0.09774356788921322:
        return "legacy_or_patched_runtime"
    if (
        "latest_us_data" in text
        or n_targets in {2814, 2818}
        or (baseline_loss is not None and baseline_loss > 0.15)
    ):
        return "latest_policyengine_us"
    return "legacy_or_patched_runtime"


def _infer_score_model_id(path: Path, candidate_dataset: Any) -> str:
    text_parts = [str(path).lower()]
    if isinstance(candidate_dataset, str):
        text_parts.append(candidate_dataset.lower())
        text_parts.append(Path(candidate_dataset).name.lower())
    text = " ".join(text_parts)
    if "pe_small_l0" in text or "local_net_worth_100/" in text:
        return "policyengine_small_l0"
    if "pe_big_l0" in text or "local_net_worth_100_e300" in text:
        return "policyengine_big_l0"
    if "policyengine_local_area_l0" in text or "state_stack" in text:
        return "policyengine_local_area_l0_state_stack"
    return "microplex_current_best"


def _infer_record_count_tier(path: Path, candidate_dataset: Any) -> str | None:
    """Infer product-style record-count tier labels such as ``mp-120k``."""

    text_parts = [str(path).lower()]
    if isinstance(candidate_dataset, str):
        text_parts.append(candidate_dataset.lower())
        text_parts.append(Path(candidate_dataset).name.lower())
    text = " ".join(text_parts)
    match = re.search(r"\bmp[-_]?(\d+(?:k|m))(?:\b|_)", text)
    if match:
        return f"mp-{match.group(1)}"
    return None


def _infer_record_count_tier_from_release_smoke(
    release_smoke: dict[str, Any] | None,
) -> str | None:
    """Infer a product-style tier from measured household rows when available."""

    if not isinstance(release_smoke, dict):
        return None
    households = _number_or_none(release_smoke.get("candidate_households"))
    if households is None or households <= 0:
        return None
    if households >= 1_000_000:
        return f"mp-{households / 1_000_000:.1f}m".replace(".0m", "m")
    return f"mp-{round(households / 1_000)}k"


def _percentile(sorted_values: list[float], quantile: float) -> float | None:
    if not sorted_values:
        return None
    if len(sorted_values) == 1:
        return 100 * sorted_values[0]
    position = quantile * (len(sorted_values) - 1)
    lower = int(position)
    upper = min(lower + 1, len(sorted_values) - 1)
    weight = position - lower
    return 100 * (sorted_values[lower] * (1 - weight) + sorted_values[upper] * weight)


def _share_under(values: list[float], threshold: float) -> float | None:
    if not values:
        return None
    return sum(value < threshold for value in values) / len(values)


def _best_score(rows: list[dict[str, Any]], *, predicate: Any) -> dict[str, Any] | None:
    candidates = [
        row
        for row in rows
        if predicate(row) and _number_or_none(row.get("candidate_loss")) is not None
    ]
    if not candidates:
        return None
    return min(candidates, key=lambda row: row["candidate_loss"])


def _best_model_metric_score(
    rows: list[dict[str, Any]],
    *,
    model_id: str,
    metric_runtime: str,
) -> dict[str, Any] | None:
    return _best_score(
        rows,
        predicate=lambda row: (
            row.get("model_id") == model_id
            and row.get("metric_runtime") == metric_runtime
        ),
    )


def _parse_row_batch_progress(text: str) -> dict[str, Any] | None:
    pattern = re.compile(
        r"PE-native row batch (?P<dataset>[^:]+): "
        r"(?P<done>\d+)/(?P<total>\d+) households "
        r"\((?P<elapsed>[0-9.]+)s\)"
    )
    matches = list(pattern.finditer(text))
    if not matches:
        return None
    match = matches[-1]
    done = int(match.group("done"))
    total = int(match.group("total"))
    return {
        "dataset": match.group("dataset"),
        "households_done": done,
        "households_total": total,
        "fraction": done / total if total else None,
        "elapsed_seconds": float(match.group("elapsed")),
    }


def _is_relevant_tmux_session(name: str) -> bool:
    lowered = name.lower()
    return (
        lowered.startswith("mp_")
        or "microplex" in lowered
        or lowered.startswith("dashboard")
    )


def _tail_text(path: Path, max_bytes: int = 8192) -> str:
    try:
        with path.open("rb") as file:
            file.seek(0, 2)
            size = file.tell()
            file.seek(max(size - max_bytes, 0))
            return file.read().decode("utf-8", errors="replace")
    except OSError:
        return ""


def _read_json(path: Path) -> Any | None:
    try:
        return json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return None


def _number_or_none(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if number != number:
        return None
    return number


def main(argv: list[str] | None = None) -> int:
    """CLI for the living Microplex dashboard payload."""

    parser = argparse.ArgumentParser(
        description="Build the living Microplex diagnostic dashboard JSON."
    )
    parser.add_argument("--artifact-root", default=str(_DEFAULT_ARTIFACT_ROOT))
    parser.add_argument(
        "--target-diagnostics-path",
        default=str(_DEFAULT_TARGET_DIAGNOSTICS_PATH),
        help="Existing per-target diagnostics JSON to embed when available.",
    )
    parser.add_argument(
        "--policyengine-us-data-repo",
        default=str(_DEFAULT_POLICYENGINE_US_DATA_REPO),
        help=(
            "Local policyengine-us-data checkout used to discover PE local-L0 "
            "weight packages. Pass an empty string to skip discovery."
        ),
    )
    parser.add_argument("--output-path", default=str(_DEFAULT_OUTPUT_PATH))
    parser.add_argument(
        "--no-tmux",
        action="store_true",
        help="Skip tmux session discovery for deterministic tests.",
    )
    args = parser.parse_args(argv)
    output = write_dashboard_payload(
        args.output_path,
        artifact_root=args.artifact_root,
        target_diagnostics_path=args.target_diagnostics_path,
        policyengine_us_data_repo=args.policyengine_us_data_repo or None,
        include_tmux=not args.no_tmux,
    )
    print(output)
    return 0
