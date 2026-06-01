"""Write non-gating transparency sidecars for Microplex artifact bundles."""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any

TRANSPARENCY_SIDECAR_SCHEMA_VERSION = 1
DEFAULT_CONTRACT_PATH = Path(__file__).with_name("ecps_export_contract.json")
DEFAULT_OUTPUT_DIRNAME = "transparency"

_TIMESTAMPED_LINE_RE = re.compile(r"^\[(?P<timestamp>[^\]]+)\]\s*(?P<message>.*)$")
_KEY_RE = re.compile(r"(?P<key>[A-Za-z_][A-Za-z0-9_]*)=")


def write_transparency_sidecars(
    artifact_root: str | Path,
    *,
    dataset_path: str | Path | None = None,
    log_paths: list[str | Path] | tuple[str | Path, ...] | None = None,
    contract_path: str | Path = DEFAULT_CONTRACT_PATH,
    output_dir: str | Path | None = None,
) -> dict[str, Any]:
    """Write source, row, column, imputation, and calibration sidecars.

    These sidecars are observability artifacts. They intentionally do not
    decide whether a dataset is production-ready; release performance remains
    governed by the loss comparison.
    """

    root = Path(artifact_root).expanduser().resolve()
    dataset = (
        Path(dataset_path).expanduser().resolve()
        if dataset_path is not None
        else root / "policyengine_us.h5"
    )
    logs = _resolve_log_paths(root, log_paths)
    destination = (
        Path(output_dir).expanduser().resolve()
        if output_dir is not None
        else root / DEFAULT_OUTPUT_DIRNAME
    )
    destination.mkdir(parents=True, exist_ok=True)

    generated_at = _now_iso()
    log_summary = _parse_logs(logs)
    outputs: dict[str, Path] = {}

    common = {
        "schema_version": TRANSPARENCY_SIDECAR_SCHEMA_VERSION,
        "generated_at": generated_at,
        "artifact_root": str(root),
        "non_gating": True,
        "production_performance_gate": "loss",
    }

    source_manifest = {
        **common,
        "logs": [str(path) for path in logs],
        "wrapper_events": log_summary["wrapper_events"],
        "build_config": log_summary["build_config"],
        "source_events": log_summary["source_events"],
        "failures": log_summary["failures"],
    }
    outputs["source_manifest"] = _write_json(
        destination / "source_manifest.json",
        source_manifest,
    )

    imputation_manifest = {
        **common,
        "logs": [str(path) for path in logs],
        "donor_integration": log_summary["donor_integration"],
    }
    outputs["imputation_manifest"] = _write_json(
        destination / "imputation_manifest.json",
        imputation_manifest,
    )

    column_manifest = _build_column_manifest(
        dataset,
        contract_path=Path(contract_path),
        common=common,
    )
    outputs["column_manifest"] = _write_json(
        destination / "column_manifest.json",
        column_manifest,
    )

    row_count_manifest = _build_row_count_manifest(dataset, root=root, common=common)
    outputs["row_count_manifest"] = _write_json(
        destination / "row_count_manifest.json",
        row_count_manifest,
    )

    calibration_trace = _build_calibration_trace(root, common=common)
    outputs["calibration_trace"] = _write_json(
        destination / "calibration_trace.json",
        calibration_trace,
    )

    summary = {
        **common,
        "output_dir": str(destination),
        "sidecars": {key: str(path) for key, path in sorted(outputs.items())},
        "dataset_path": str(dataset),
        "dataset_available": dataset.exists(),
        "log_paths": [str(path) for path in logs],
    }
    outputs["summary"] = _write_json(
        destination / "transparency_summary.json",
        summary,
    )
    return summary


def _now_iso() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def _write_json(path: Path, payload: dict[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_name(f".{path.name}.tmp")
    temp_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temp_path.replace(path)
    return path


def _resolve_log_paths(
    artifact_root: Path,
    log_paths: list[str | Path] | tuple[str | Path, ...] | None,
) -> list[Path]:
    if log_paths:
        return [Path(path).expanduser().resolve() for path in log_paths]
    candidates = [
        artifact_root / "logs" / "gate1_build.log",
        artifact_root / "logs" / "rebuild.log",
        artifact_root / "resume_correct_targets" / "logs" / "resume.log",
    ]
    return [path for path in candidates if path.exists()]


def _parse_logs(paths: list[Path]) -> dict[str, Any]:
    wrapper_events: list[dict[str, Any]] = []
    source_events: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    donor_sources: dict[str, dict[str, Any]] = {}
    build_config: dict[str, Any] = {}

    for path in paths:
        if not path.exists():
            failures.append({"log_path": str(path), "reason": "missing_log"})
            continue
        for line_number, raw_line in enumerate(path.read_text().splitlines(), start=1):
            line = raw_line.strip()
            if not line:
                continue
            timestamp, message = _split_timestamped_line(line)
            if message.startswith(("Starting ", "Shape:", "Artifact root:")) or (
                message.startswith("Microplex git") or message.startswith("Free disk:")
            ):
                wrapper_events.append(
                    {
                        "log_path": str(path),
                        "line": line_number,
                        "timestamp": timestamp,
                        "message": message,
                    }
                )
            if line.startswith("PE-US-data rebuild checkpoint: starting build"):
                build_config = _parse_last_bracket_payload(line)
                build_config["log_path"] = str(path)
                build_config["line"] = line_number
            elif _is_source_event(message):
                source_events.append(
                    {
                        "log_path": str(path),
                        "line": line_number,
                        "timestamp": timestamp,
                        "message": message,
                    }
                )
            elif "Traceback (most recent call last)" in line:
                failures.append(
                    {
                        "log_path": str(path),
                        "line": line_number,
                        "timestamp": timestamp,
                        "reason": "traceback",
                    }
                )
            elif re.match(r"^[A-Za-z_][A-Za-z0-9_]*(Error|Exception):", line):
                failures.append(
                    {
                        "log_path": str(path),
                        "line": line_number,
                        "timestamp": timestamp,
                        "reason": "exception",
                        "message": line,
                    }
                )

            if line.startswith("US microplex donor integration:"):
                _record_donor_event(
                    donor_sources,
                    path=path,
                    line_number=line_number,
                    timestamp=timestamp,
                    message=line,
                )

    return {
        "wrapper_events": wrapper_events,
        "build_config": build_config,
        "source_events": source_events,
        "failures": failures,
        "donor_integration": {
            "sources": [
                _finalize_donor_source(source)
                for source in sorted(
                    donor_sources.values(),
                    key=lambda item: str(item.get("donor_source", "")),
                )
            ]
        },
    }


def _split_timestamped_line(line: str) -> tuple[str | None, str]:
    match = _TIMESTAMPED_LINE_RE.match(line)
    if match is None:
        return None, line
    return match.group("timestamp"), match.group("message")


def _is_source_event(message: str) -> bool:
    return message.startswith(
        (
            "Downloading ",
            "Downloaded ",
            "Parsing ",
            "Cached processed data",
            "Using repo-local PUF",
            "Loading PUF",
            "Loading demographics",
            "Loading processed CPS",
            "Expanded ",
            "  Raw records:",
            "  After demographics merge:",
        )
    )


def _parse_last_bracket_payload(line: str) -> dict[str, Any]:
    start = line.rfind("[")
    end = line.rfind("]")
    if start == -1 or end == -1 or end <= start:
        return {}
    return _parse_key_values(line[start + 1 : end])


def _parse_key_values(payload: str) -> dict[str, Any]:
    matches = list(_KEY_RE.finditer(payload))
    fields: dict[str, Any] = {}
    for index, match in enumerate(matches):
        value_start = match.end()
        value_end = (
            matches[index + 1].start() if index + 1 < len(matches) else len(payload)
        )
        value = payload[value_start:value_end].strip().rstrip(",").strip()
        fields[match.group("key")] = _coerce_scalar(value)
    return fields


def _coerce_scalar(value: str) -> Any:
    if value in {"true", "True"}:
        return True
    if value in {"false", "False"}:
        return False
    if value in {"none", "None", "null"}:
        return None
    if re.fullmatch(r"-?\d+", value):
        try:
            return int(value)
        except ValueError:
            return value
    if re.fullmatch(r"-?\d+\.\d+", value):
        try:
            return float(value)
        except ValueError:
            return value
    return value


def _record_donor_event(
    donor_sources: dict[str, dict[str, Any]],
    *,
    path: Path,
    line_number: int,
    timestamp: str | None,
    message: str,
) -> None:
    payload = _parse_last_bracket_payload(message)
    if payload.get("donor_source") is None:
        return
    donor_source = str(payload["donor_source"])
    source = donor_sources.setdefault(
        donor_source,
        {
            "donor_source": donor_source,
            "source_events": [],
            "entity_id_events": [],
            "blocks": {},
        },
    )
    event = {
        "log_path": str(path),
        "line": line_number,
        "timestamp": timestamp,
        "fields": payload,
    }
    if (
        "source start" in message
        or "source ready" in message
        or "source complete" in message
    ):
        source["source_events"].append(event)
        if "source ready" in message:
            source["ready"] = payload
        if "source complete" in message:
            source["complete"] = payload
        return
    if "entity ids " in message:
        source["entity_id_events"].append(event)
        return
    if "block " not in message:
        return

    block_name = str(payload.get("block") or "unknown")
    block = source["blocks"].setdefault(
        block_name,
        {
            "block": block_name,
            "restored": payload.get("restored"),
            "started_count": 0,
            "completed_count": 0,
            "run_events": [],
            "complete_events": [],
        },
    )
    if "block start" in message:
        block["started_count"] += 1
        block["last_start"] = event
        block["restored"] = payload.get("restored", block.get("restored"))
    elif "block run" in message:
        block["run_events"].append(event)
        block["last_run"] = event
    elif "block complete" in message:
        block["completed_count"] += 1
        block["complete_events"].append(event)
        block["last_complete"] = event
        block["integrated_vars"] = payload.get("integrated_vars")


def _finalize_donor_source(source: dict[str, Any]) -> dict[str, Any]:
    blocks = [
        block
        for block in sorted(
            source.get("blocks", {}).values(),
            key=lambda item: str(item.get("block", "")),
        )
    ]
    completed = [
        block["block"]
        for block in blocks
        if int(block.get("completed_count") or 0)
        >= int(block.get("started_count") or 0)
    ]
    active = [
        block["block"]
        for block in blocks
        if int(block.get("completed_count") or 0) < int(block.get("started_count") or 0)
    ]
    return {
        **source,
        "block_count": len(blocks),
        "completed_block_count": len(completed),
        "completed_blocks": completed,
        "active_blocks": active,
        "blocks": blocks,
    }


def _build_column_manifest(
    dataset_path: Path,
    *,
    contract_path: Path,
    common: dict[str, Any],
) -> dict[str, Any]:
    contract = _load_contract(contract_path)
    required = set(contract.get("required", []))
    forbidden = set(contract.get("forbidden", []))
    optional = set(contract.get("ecps_internal_optional", []))
    formula_owned_excluded = set(contract.get("formula_owned_excluded", []))
    if not dataset_path.exists():
        return {
            **common,
            "dataset_path": str(dataset_path),
            "available": False,
            "reason": "dataset_not_found",
            "contract_path": str(contract_path),
            "required_count": len(required),
            "forbidden_count": len(forbidden),
        }

    present = _h5_top_level_columns(dataset_path)
    known = required | forbidden | optional | formula_owned_excluded
    missing_required = sorted(required - present)
    forbidden_present = sorted(forbidden & present)
    formula_owned_excluded_present = sorted(formula_owned_excluded & present)
    return {
        **common,
        "dataset_path": str(dataset_path),
        "available": True,
        "contract_path": str(contract_path),
        "present_count": len(present),
        "required_count": len(required),
        "forbidden_count": len(forbidden),
        "missing_required_count": len(missing_required),
        "forbidden_present_count": len(forbidden_present),
        "formula_owned_excluded_present_count": len(formula_owned_excluded_present),
        "missing_required": missing_required,
        "forbidden_present": forbidden_present,
        "formula_owned_excluded_present": formula_owned_excluded_present,
        "extra_unknown": sorted(present - known),
        "present_columns": sorted(present),
        "diagnostic_status": "clean"
        if not missing_required and not forbidden_present
        else "needs_attention",
    }


def _load_contract(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def _h5_top_level_columns(dataset_path: Path) -> set[str]:
    import h5py

    with h5py.File(dataset_path, "r") as h5:
        return {str(key).split("/")[0] for key in h5.keys()}


def _build_row_count_manifest(
    dataset_path: Path,
    *,
    root: Path,
    common: dict[str, Any],
) -> dict[str, Any]:
    checkpoint_counts = _load_checkpoint_counts(root)
    if not dataset_path.exists():
        return {
            **common,
            "dataset_path": str(dataset_path),
            "available": False,
            "reason": "dataset_not_found",
            "checkpoint_counts": checkpoint_counts,
        }

    h5_summary = _summarize_h5_variables(dataset_path)
    return {
        **common,
        "dataset_path": str(dataset_path),
        "available": True,
        "checkpoint_counts": checkpoint_counts,
        "shape_counts": h5_summary["shape_counts"],
        "variables": h5_summary["variables"],
    }


def _summarize_h5_variables(dataset_path: Path) -> dict[str, Any]:
    import h5py

    variables: list[dict[str, Any]] = []
    shape_counter: Counter[str] = Counter()
    shape_examples: dict[str, list[str]] = {}
    with h5py.File(dataset_path, "r") as h5:
        for name in sorted(h5.keys()):
            obj = h5[name]
            periods = []
            if isinstance(obj, h5py.Dataset):
                periods.append(_dataset_summary("flat", obj))
            else:
                for period in sorted(obj.keys()):
                    child = obj[period]
                    if isinstance(child, h5py.Dataset):
                        periods.append(_dataset_summary(str(period), child))
            for period in periods:
                shape_key = "x".join(str(part) for part in period["shape"])
                shape_counter[shape_key] += 1
                shape_examples.setdefault(shape_key, []).append(name)
            variables.append({"name": name, "periods": periods})
    shape_counts = [
        {
            "shape": key,
            "variable_count": count,
            "example_variables": shape_examples.get(key, [])[:12],
        }
        for key, count in sorted(
            shape_counter.items(), key=lambda item: (-item[1], item[0])
        )
    ]
    return {"variables": variables, "shape_counts": shape_counts}


def _dataset_summary(period: str, dataset: Any) -> dict[str, Any]:
    shape = [int(part) for part in dataset.shape]
    return {
        "period": period,
        "shape": shape,
        "rows": shape[0] if shape else None,
        "dtype": str(dataset.dtype),
    }


def _load_checkpoint_counts(root: Path) -> dict[str, Any]:
    candidates = [
        root / "record_count_probe.json",
        root / "resume_summary.json",
        root / "resume_correct_targets" / "resume_summary.json",
        root / "post-microsim" / "metadata.json",
        root / "resume_correct_targets" / "post-microsim" / "metadata.json",
        root / "checkpoints" / "post-microsim" / "metadata.json",
        root / "checkpoints" / "post-imputation" / "metadata.json",
    ]
    loaded: dict[str, Any] = {}
    for path in candidates:
        if not path.exists():
            continue
        try:
            payload = json.loads(path.read_text())
        except json.JSONDecodeError:
            continue
        loaded[str(path)] = _extract_row_counts(payload)
    return loaded


def _extract_row_counts(payload: dict[str, Any]) -> dict[str, Any]:
    counts: dict[str, Any] = {}
    for key in ("households", "persons", "tax_units", "families", "spm_units"):
        value = payload.get(key)
        if isinstance(value, dict) and "rows" in value:
            counts[key] = value["rows"]
        elif isinstance(value, int | float):
            counts[key] = int(value)
    for key in ("calibrated_rows", "row_count", "person_count", "household_count"):
        if key in payload:
            counts[key] = payload[key]
    return counts


def _build_calibration_trace(root: Path, *, common: dict[str, Any]) -> dict[str, Any]:
    candidates = [
        root / "calibration_summary.json",
        root / "resume_correct_targets" / "calibration_summary.json",
    ]
    summaries = []
    for path in candidates:
        if not path.exists():
            continue
        try:
            payload = json.loads(path.read_text())
        except json.JSONDecodeError:
            continue
        summaries.append(
            {
                "path": str(path),
                "backend": payload.get("backend"),
                "period": payload.get("period"),
                "converged": payload.get("converged"),
                "n_loaded_targets": payload.get("n_loaded_targets"),
                "n_supported_targets": payload.get("n_supported_targets"),
                "n_unsupported_targets": payload.get("n_unsupported_targets"),
                "full_oracle_capped_mean_abs_relative_error": payload.get(
                    "full_oracle_capped_mean_abs_relative_error"
                ),
                "feasibility_filter": payload.get("feasibility_filter"),
                "calibration_stages": payload.get("calibration_stages", []),
            }
        )
    return {
        **common,
        "available": bool(summaries),
        "summaries": summaries,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="microplex-us-write-transparency-sidecars",
        description=(
            "Write non-gating source, row, column, imputation, and calibration "
            "sidecars for one Microplex artifact root."
        ),
    )
    parser.add_argument("artifact_root", help="Microplex artifact root to inspect.")
    parser.add_argument(
        "--dataset",
        help="PolicyEngine H5 path. Defaults to ARTIFACT_ROOT/policyengine_us.h5.",
    )
    parser.add_argument(
        "--log",
        action="append",
        dest="logs",
        help="Log path to parse. May be repeated. Defaults to common artifact logs.",
    )
    parser.add_argument(
        "--contract",
        default=str(DEFAULT_CONTRACT_PATH),
        help="eCPS column contract path.",
    )
    parser.add_argument(
        "--output-dir",
        help="Output directory. Defaults to ARTIFACT_ROOT/transparency.",
    )
    args = parser.parse_args(argv)
    summary = write_transparency_sidecars(
        args.artifact_root,
        dataset_path=args.dataset,
        log_paths=args.logs,
        contract_path=args.contract,
        output_dir=args.output_dir,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
