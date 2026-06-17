"""Tests for the CI runner script (scripts/run_ecps_eval.py).

These validate the orchestration logic *without* running the heavy eval:
command assembly, baseline/candidate resolution, DRYRUN behaviour, step-summary
rendering (including the honest-reporting caveat), and the clone-floor gate
short-circuit. The actual comparison subprocess is monkeypatched.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest

from microplex_us.pipelines.ecps_clone_floor import CloneFloorGateResult


def _load_runner() -> ModuleType:
    """Import scripts/run_ecps_eval.py as a module by path."""
    script = Path(__file__).resolve().parent.parent / "scripts" / "run_ecps_eval.py"
    spec = importlib.util.spec_from_file_location("run_ecps_eval", script)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


runner = _load_runner()


def _write_diag(path: Path, share: float) -> None:
    path.write_text(json.dumps({"clone_household_weight_share": share}))


# --------------------------------------------------------------------------- #
# Command assembly
# --------------------------------------------------------------------------- #
def test_build_comparison_command_has_sound_flags(tmp_path: Path) -> None:
    cmd = runner.build_comparison_command(
        candidate=tmp_path / "cand.h5",
        baseline=tmp_path / "base.h5",
        output_dir=tmp_path / "out",
    )
    assert cmd[0] == runner.COMPARISON_CONSOLE_SCRIPT
    joined = " ".join(cmd)
    assert "--candidate-dataset" in joined
    assert "--baseline-dataset" in joined
    assert "--matched-sample-method uniform" in joined
    assert "--holdout-target-fraction 0.2" in joined
    assert "--optimizer-max-iter 200" in joined
    assert "--period 2024" in joined
    assert "--force" in joined


def test_build_comparison_command_passes_through_pe_data_repo(tmp_path: Path) -> None:
    cmd = runner.build_comparison_command(
        candidate=tmp_path / "cand.h5",
        baseline=tmp_path / "base.h5",
        output_dir=tmp_path / "out",
        policyengine_us_data_repo="/some/repo",
        policyengine_us_data_python="/some/python",
    )
    joined = " ".join(cmd)
    assert "--policyengine-us-data-repo /some/repo" in joined
    assert "--policyengine-us-data-python /some/python" in joined


# --------------------------------------------------------------------------- #
# Baseline / candidate resolution
# --------------------------------------------------------------------------- #
def test_resolve_baseline_local_path(tmp_path: Path) -> None:
    h5 = tmp_path / "enhanced_cps_2024.h5"
    h5.write_text("x")
    resolved_h5, resolved_diag = runner.resolve_baseline(str(h5), tmp_path)
    assert resolved_h5 == h5
    assert resolved_diag.name == runner.BASELINE_DIAGNOSTICS_FILENAME
    assert resolved_diag.parent == h5.parent


def test_resolve_baseline_latest_is_dryrun_safe(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """'latest' under DRYRUN returns expected paths without downloading."""
    monkeypatch.setenv("DRYRUN", "1")
    h5, diag = runner.resolve_baseline("latest published eCPS", tmp_path)
    assert h5.name == runner.BASELINE_H5_FILENAME
    assert diag.name == runner.BASELINE_DIAGNOSTICS_FILENAME


def test_resolve_candidate_local_path(tmp_path: Path) -> None:
    cand = tmp_path / "candidate.h5"
    assert runner.resolve_candidate(str(cand), tmp_path) == cand


def test_resolve_candidate_http_dryrun(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("DRYRUN", "1")
    target = runner.resolve_candidate("https://example.com/data/candidate.h5", tmp_path)
    assert target == tmp_path / "candidate.h5"


def test_resolve_candidate_hf_dryrun(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("DRYRUN", "1")
    target = runner.resolve_candidate(
        "hf://policyengine/policyengine-us-data/enhanced_cps_2024.h5", tmp_path
    )
    assert target == tmp_path / "enhanced_cps_2024.h5"


# --------------------------------------------------------------------------- #
# Step-summary rendering
# --------------------------------------------------------------------------- #
def _summary_result() -> dict[str, Any]:
    return {
        "summary": {
            "matched_household_count": 41000,
            "baseline_enhanced_cps_native_loss": 1.0,
            "candidate_enhanced_cps_native_loss": 0.5,
            "baseline_train_loss": 0.9,
            "candidate_train_loss": 0.4,
            "baseline_holdout_loss": 0.95,
            "candidate_holdout_loss": 0.45,
            "holdout_target_fraction": 0.2,
            "candidate_beats_baseline": True,
            "symmetric_refit": True,
            "score_candidate_only": False,
            "refit_objective_matches_scoring": True,
            "ecps_refit_recovery_passed": True,
        }
    }


def test_render_summary_with_result_includes_everything() -> None:
    gate = CloneFloorGateResult(
        passed=True,
        message="baseline eCPS clone share 42.0% >= 5.0% floor - healthy",
        clone_weight_share=0.42,
        floor=0.05,
    )
    text = runner.render_step_summary(gate, _summary_result())
    assert "Matched N (households): 41000" in text
    assert "Baseline (eCPS) loss: 1.0" in text
    assert "Candidate (MP) loss: 0.5" in text
    assert "train loss: 0.9" in text
    assert "train loss: 0.4" in text
    assert "holdout loss: 0.95" in text
    assert "holdout loss: 0.45" in text
    assert "candidate_beats_baseline: **True**" in text
    assert "symmetric_refit: PASS" in text
    assert "ecps_refit_recovery_passed: PASS" in text
    assert "Clone-floor baseline gate: PASS" in text
    # Honest-reporting caveat must appear verbatim.
    assert runner.HONEST_REPORTING_CAVEAT in text
    assert "#113" in text


def test_render_summary_gate_failed_says_not_run() -> None:
    gate = CloneFloorGateResult(
        passed=False,
        message="baseline eCPS degraded; clone share 2.0% < 5.0% floor",
        clone_weight_share=0.02,
        floor=0.05,
    )
    text = runner.render_step_summary(gate, result=None)
    assert "Clone-floor baseline gate: FAIL" in text
    assert "was **not run**" in text
    # Caveat is always present.
    assert runner.HONEST_REPORTING_CAVEAT in text


# --------------------------------------------------------------------------- #
# End-to-end orchestration (subprocess monkeypatched)
# --------------------------------------------------------------------------- #
def test_run_dryrun_prints_command_and_runs_nothing(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """DRYRUN prints the would-run command and never calls subprocess."""
    monkeypatch.setenv("DRYRUN", "1")

    def _fail_run(*args: object, **kwargs: object) -> None:
        raise AssertionError("subprocess.run must not be called in DRYRUN")

    monkeypatch.setattr(runner.subprocess, "run", _fail_run)

    cand = tmp_path / "candidate.h5"
    base = tmp_path / "enhanced_cps_2024.h5"
    args = runner.build_arg_parser().parse_args(
        [
            "--candidate",
            str(cand),
            "--baseline-source",
            str(base),
            "--work-dir",
            str(tmp_path / "work"),
            "--output-dir",
            str(tmp_path / "out"),
        ]
    )
    with caplog.at_level("INFO"):
        rc = runner.run(args)
    assert rc == 0
    assert "would run" in caplog.text
    assert runner.COMPARISON_CONSOLE_SCRIPT in caplog.text


def test_run_gate_failure_short_circuits(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A degraded baseline makes run() return 1 without invoking the eval."""
    monkeypatch.delenv("DRYRUN", raising=False)

    def _fail_run(*args: object, **kwargs: object) -> None:
        raise AssertionError("comparison must not run when the gate fails")

    monkeypatch.setattr(runner.subprocess, "run", _fail_run)

    cand = tmp_path / "candidate.h5"
    cand.write_text("x")
    base = tmp_path / "enhanced_cps_2024.h5"
    base.write_text("x")
    _write_diag(base.with_name(runner.BASELINE_DIAGNOSTICS_FILENAME), share=0.02)

    summary_file = tmp_path / "summary.md"
    monkeypatch.setenv("GITHUB_STEP_SUMMARY", str(summary_file))

    args = runner.build_arg_parser().parse_args(
        [
            "--candidate",
            str(cand),
            "--baseline-source",
            str(base),
            "--work-dir",
            str(tmp_path / "work"),
            "--output-dir",
            str(tmp_path / "out"),
        ]
    )
    rc = runner.run(args)
    assert rc == 1
    written = summary_file.read_text()
    assert "Clone-floor baseline gate: FAIL" in written
    assert "degraded" in written


def test_run_happy_path_invokes_eval_and_summarizes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A healthy baseline runs the eval (mocked) and writes the summary."""
    monkeypatch.delenv("DRYRUN", raising=False)

    output_dir = tmp_path / "out"

    class _Completed:
        returncode = 0

    def _fake_run(cmd: list[str], **kwargs: object) -> _Completed:
        # Simulate the comparison writing its result JSON.
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / runner.RESULT_JSON_FILENAME).write_text(
            json.dumps(_summary_result())
        )
        return _Completed()

    monkeypatch.setattr(runner.subprocess, "run", _fake_run)

    cand = tmp_path / "candidate.h5"
    cand.write_text("x")
    base = tmp_path / "enhanced_cps_2024.h5"
    base.write_text("x")
    _write_diag(base.with_name(runner.BASELINE_DIAGNOSTICS_FILENAME), share=0.42)

    summary_file = tmp_path / "summary.md"
    monkeypatch.setenv("GITHUB_STEP_SUMMARY", str(summary_file))

    args = runner.build_arg_parser().parse_args(
        [
            "--candidate",
            str(cand),
            "--baseline-source",
            str(base),
            "--work-dir",
            str(tmp_path / "work"),
            "--output-dir",
            str(output_dir),
        ]
    )
    rc = runner.run(args)
    assert rc == 0
    written = summary_file.read_text()
    assert "Clone-floor baseline gate: PASS" in written
    assert "candidate_beats_baseline: **True**" in written
    assert runner.HONEST_REPORTING_CAVEAT in written
