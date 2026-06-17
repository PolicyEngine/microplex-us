"""Run the sound eCPS-replacement comparison end to end (for CI).

This is the orchestration layer the ``ecps-eval`` GitHub Actions workflow shells
out to, so the workflow YAML stays thin and the logic stays unit-tested. It:

1. **Resolves the baseline** Enhanced CPS - an explicit local path, or (by
   default) the latest published artifact on the Hugging Face Hub
   (``policyengine/policyengine-us-data``). Its clone-diagnostics sidecar is
   resolved alongside it.
2. **Resolves the candidate** Microplex H5 - a local path or a downloadable URI.
3. **Runs the clone-floor baseline gate** (the core lesson). Before spending
   ~20-30 minutes benchmarking, refuse to benchmark against a degraded (or
   unverifiable) baseline. See
   :mod:`microplex_us.pipelines.ecps_clone_floor`.
4. **Runs the comparison** via the
   ``microplex-us-ecps-replacement-comparison`` console script with the sound
   flags.
5. **Emits a GitHub Step Summary** with matched N, both losses, train/holdout
   losses, ``candidate_beats_baseline``, every soundness gate, and the
   honest-reporting caveat (#113).

Set ``DRYRUN=1`` to print the comparison command that *would* run without
executing it (and without downloading anything that needs credentials).

This script intentionally does **not** build a candidate dataset. A full
candidate build is GPU-heavy and would run on Modal; see the workflow for where
that path would plug in.

Usage (typical CI invocation)::

    uv run python scripts/run_ecps_eval.py \\
        --candidate hf://policyengine/policyengine-us-data/enhanced_cps_2024.h5 \\
        --baseline-source latest \\
        --output-dir comparison_output
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

# Allow running both as an installed module and as a bare script from a checkout
# where ``src/`` is not yet importable.
try:
    from microplex_us.pipelines.ecps_clone_floor import (
        DEFAULT_CLONE_FLOOR,
        CloneFloorGateResult,
        evaluate_clone_floor_gate,
    )
except ModuleNotFoundError:  # pragma: no cover - exercised only outside tests
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
    from microplex_us.pipelines.ecps_clone_floor import (
        DEFAULT_CLONE_FLOOR,
        CloneFloorGateResult,
        evaluate_clone_floor_gate,
    )

logger = logging.getLogger("run_ecps_eval")

# The published baseline artifact and its diagnostics sidecar.
HF_DATA_REPO = "policyengine/policyengine-us-data"
BASELINE_H5_FILENAME = "enhanced_cps_2024.h5"
BASELINE_DIAGNOSTICS_FILENAME = "enhanced_cps_2024.clone_diagnostics.json"

# Default soundness flags for the comparison (kept in one place).
DEFAULT_PERIOD = 2024
DEFAULT_MATCHED_SAMPLE_METHOD = "uniform"
DEFAULT_HOLDOUT_TARGET_FRACTION = 0.2
DEFAULT_OPTIMIZER_MAX_ITER = 200

COMPARISON_CONSOLE_SCRIPT = "microplex-us-ecps-replacement-comparison"
RESULT_JSON_FILENAME = "sound_ecps_replacement_comparison.json"

# The honest-reporting caveat, surfaced verbatim in the step summary.
HONEST_REPORTING_CAVEAT = (
    "candidate_beats_baseline=true is only meaningful if the baseline passed "
    "the clone-floor gate AND the candidate does not share tax-unit "
    "construction code with the baseline (see microplex-us #113 re: microunit "
    "convergence)."
)

# Values of DRYRUN that mean "off".
_FALSE_VALUES = ("", "0", "false", "no", "off")


def _dryrun_enabled() -> bool:
    """Return ``True`` if ``DRYRUN`` is set to a truthy value."""
    return os.environ.get("DRYRUN", "").strip().lower() not in _FALSE_VALUES


def _is_published_baseline(baseline_source: str | None) -> bool:
    """Return ``True`` if the baseline should be pulled from Hugging Face."""
    if baseline_source is None:
        return True
    return baseline_source.strip().lower() in (
        "latest",
        "latest published ecps",
    )


def resolve_baseline(
    baseline_source: str | None,
    work_dir: Path,
) -> tuple[Path, Path]:
    """Resolve the baseline H5 and its clone-diagnostics sidecar.

    Parameters
    ----------
    baseline_source:
        Either a local path to an Enhanced CPS H5, or ``None`` / ``"latest"`` /
        ``"latest published eCPS"`` to pull the latest published artifact.
    work_dir:
        Directory to download artifacts into when pulling from the Hub.

    Returns
    -------
    tuple[Path, Path]
        ``(baseline_h5, baseline_diagnostics_json)``.

    Notes
    -----
    In ``DRYRUN`` mode nothing is downloaded; the *expected* local paths are
    returned so the caller can show the command without touching the network or
    needing a Hugging Face token.
    """
    if not _is_published_baseline(baseline_source):
        assert baseline_source is not None  # narrowed by _is_published_baseline
        h5 = Path(baseline_source).expanduser()
        diagnostics = h5.with_name(BASELINE_DIAGNOSTICS_FILENAME)
        return h5, diagnostics

    work_dir.mkdir(parents=True, exist_ok=True)
    expected_h5 = work_dir / BASELINE_H5_FILENAME
    expected_diag = work_dir / BASELINE_DIAGNOSTICS_FILENAME

    if _dryrun_enabled():
        logger.info(
            "[dryrun] would download %s and %s from %s",
            BASELINE_H5_FILENAME,
            BASELINE_DIAGNOSTICS_FILENAME,
            HF_DATA_REPO,
        )
        return expected_h5, expected_diag

    from huggingface_hub import hf_hub_download

    token = os.environ.get("HUGGING_FACE_TOKEN")
    h5 = Path(
        hf_hub_download(
            repo_id=HF_DATA_REPO,
            repo_type="dataset",
            filename=BASELINE_H5_FILENAME,
            local_dir=str(work_dir),
            token=token,
        )
    )
    # The diagnostics sidecar may not be published yet; download it if present,
    # otherwise leave the expected path so the gate fails closed on absence.
    try:
        diagnostics = Path(
            hf_hub_download(
                repo_id=HF_DATA_REPO,
                repo_type="dataset",
                filename=BASELINE_DIAGNOSTICS_FILENAME,
                local_dir=str(work_dir),
                token=token,
            )
        )
    except Exception as exc:  # noqa: BLE001 - any failure means "treat as absent"
        logger.warning(
            "could not fetch %s (%s); clone-floor gate will fail closed",
            BASELINE_DIAGNOSTICS_FILENAME,
            exc,
        )
        diagnostics = expected_diag

    return h5, diagnostics


def resolve_candidate(candidate: str, work_dir: Path) -> Path:
    """Resolve the candidate H5 to a local path.

    A local path is returned as-is. An ``http(s)://`` or ``hf://`` URI is
    downloaded into ``work_dir`` (skipped in ``DRYRUN`` mode).
    """
    if candidate.startswith(("http://", "https://")):
        target = work_dir / Path(candidate.split("?", 1)[0]).name
        if _dryrun_enabled():
            logger.info("[dryrun] would download candidate from %s", candidate)
            return target
        work_dir.mkdir(parents=True, exist_ok=True)
        from urllib.request import urlretrieve

        urlretrieve(candidate, target)  # noqa: S310 - trusted CI input
        return target

    if candidate.startswith("hf://"):
        # Form: hf://<repo_id>/<filename> (dataset repo assumed).
        remainder = candidate[len("hf://") :]
        repo_id, _, filename = remainder.rpartition("/")
        target = work_dir / Path(filename).name
        if _dryrun_enabled():
            logger.info(
                "[dryrun] would download candidate %s from hf dataset %s",
                filename,
                repo_id,
            )
            return target
        work_dir.mkdir(parents=True, exist_ok=True)
        from huggingface_hub import hf_hub_download

        return Path(
            hf_hub_download(
                repo_id=repo_id,
                repo_type="dataset",
                filename=filename,
                local_dir=str(work_dir),
                token=os.environ.get("HUGGING_FACE_TOKEN"),
            )
        )

    return Path(candidate).expanduser()


def build_comparison_command(
    candidate: Path,
    baseline: Path,
    output_dir: Path,
    period: int = DEFAULT_PERIOD,
    matched_sample_method: str = DEFAULT_MATCHED_SAMPLE_METHOD,
    holdout_target_fraction: float = DEFAULT_HOLDOUT_TARGET_FRACTION,
    optimizer_max_iter: int = DEFAULT_OPTIMIZER_MAX_ITER,
    policyengine_us_data_repo: str | None = None,
    policyengine_us_data_python: str | None = None,
) -> list[str]:
    """Assemble the comparison console-script command with the sound flags."""
    command = [
        COMPARISON_CONSOLE_SCRIPT,
        "--candidate-dataset",
        str(candidate),
        "--baseline-dataset",
        str(baseline),
        "--output-dir",
        str(output_dir),
        "--period",
        str(period),
        "--matched-sample-method",
        matched_sample_method,
        "--holdout-target-fraction",
        str(holdout_target_fraction),
        "--optimizer-max-iter",
        str(optimizer_max_iter),
        "--force",
    ]
    if policyengine_us_data_repo:
        command += ["--policyengine-us-data-repo", policyengine_us_data_repo]
    if policyengine_us_data_python:
        command += ["--policyengine-us-data-python", policyengine_us_data_python]
    return command


def render_step_summary(
    gate: CloneFloorGateResult,
    result: dict[str, Any] | None,
) -> str:
    """Render the full GitHub Step Summary as Markdown.

    Includes matched N, both losses, train/holdout losses,
    ``candidate_beats_baseline``, every soundness gate, the clone-floor gate
    outcome, and the honest-reporting caveat. A pure function so the exact
    reporting text can be asserted in tests.
    """
    lines: list[str] = ["## Sound eCPS-replacement comparison", ""]

    # Clone-floor baseline gate.
    floor_status = "PASS" if gate.passed else "FAIL"
    lines.append(f"### Clone-floor baseline gate: {floor_status}")
    lines.append("")
    lines.append(f"- {gate.message}")
    lines.append("")

    if result is None:
        lines.append(
            "> Comparison was **not run** because the clone-floor gate failed."
        )
        lines.append("")
    else:
        summary = result.get("summary", {})
        beats = summary.get("candidate_beats_baseline")
        candidate_loss = summary.get("candidate_enhanced_cps_native_loss")
        baseline_loss = summary.get("baseline_enhanced_cps_native_loss")
        lines.extend(
            [
                "### Results",
                "",
                f"- Matched N (households): {summary.get('matched_household_count')}",
                f"- Baseline (eCPS) loss: {baseline_loss}",
                f"- Candidate (MP) loss: {candidate_loss}",
                f"- Baseline (eCPS) train loss: {summary.get('baseline_train_loss')}",
                f"- Candidate (MP) train loss: {summary.get('candidate_train_loss')}",
                f"- Baseline (eCPS) holdout loss: {summary.get('baseline_holdout_loss')}",
                f"- Candidate (MP) holdout loss: {summary.get('candidate_holdout_loss')}",
                f"- Holdout target fraction: {summary.get('holdout_target_fraction')}",
                f"- candidate_beats_baseline: **{beats}**",
            ]
        )
        gates = _soundness_gates(summary)
        if gates:
            lines.append("- Soundness gates:")
            for name, passed in gates.items():
                lines.append(f"  - {name}: {'PASS' if passed else 'FAIL'}")
        lines.append("")

    lines.append("### Honest-reporting caveat")
    lines.append("")
    lines.append(f"> {HONEST_REPORTING_CAVEAT}")
    lines.append("")
    return "\n".join(lines)


def _soundness_gates(summary: dict[str, Any]) -> dict[str, Any]:
    """Pull the comparison's soundness-gate booleans out of its summary.

    The comparison reports these as individual keys in ``summary``; surface the
    ones present so the step summary lists exactly what the run enforced.
    """
    gate_keys = (
        "matched_household_count",
        "symmetric_refit",
        "score_candidate_only",
        "refit_objective_matches_scoring",
        "ecps_refit_recovery_passed",
    )
    return {key: summary[key] for key in gate_keys if key in summary}


def _write_step_summary(text: str) -> None:
    """Append ``text`` to ``$GITHUB_STEP_SUMMARY`` if set; always log it."""
    logger.info("\n%s", text)
    summary_path = os.environ.get("GITHUB_STEP_SUMMARY")
    if summary_path:
        with open(summary_path, "a", encoding="utf-8") as handle:
            handle.write(text + "\n")


def _load_result(output_dir: Path) -> dict[str, Any]:
    """Load the comparison result JSON from ``output_dir``."""
    result_path = output_dir / RESULT_JSON_FILENAME
    return json.loads(result_path.read_text())


def run(args: argparse.Namespace) -> int:
    """Resolve inputs, run the gate, run the comparison, emit the summary."""
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    work_dir = Path(args.work_dir)
    output_dir = Path(args.output_dir)
    dryrun = _dryrun_enabled()

    baseline_h5, baseline_diag = resolve_baseline(args.baseline_source, work_dir)
    candidate_h5 = resolve_candidate(args.candidate, work_dir)

    logger.info("Baseline H5:          %s", baseline_h5)
    logger.info("Baseline diagnostics: %s", baseline_diag)
    logger.info("Candidate H5:         %s", candidate_h5)

    command = build_comparison_command(
        candidate=candidate_h5,
        baseline=baseline_h5,
        output_dir=output_dir,
        period=args.period,
        matched_sample_method=args.matched_sample_method,
        holdout_target_fraction=args.holdout_target_fraction,
        optimizer_max_iter=args.optimizer_max_iter,
        policyengine_us_data_repo=args.policyengine_us_data_repo,
        policyengine_us_data_python=args.policyengine_us_data_python,
    )

    if dryrun:
        # Show the clone-floor gate decision if the sidecar happens to exist
        # locally, then print the command that would run and exit.
        gate = evaluate_clone_floor_gate(baseline_diag, floor=args.clone_floor)
        logger.info("[dryrun] clone-floor gate: %s", gate.message)
        logger.info("[dryrun] would run:\n  %s", " ".join(command))
        return 0

    # --- Clone-floor baseline gate (fail closed) --------------------------- #
    gate = evaluate_clone_floor_gate(baseline_diag, floor=args.clone_floor)
    if not gate.passed:
        logger.error("CLONE-FLOOR GATE FAILED: %s", gate.message)
        _write_step_summary(render_step_summary(gate, result=None))
        return 1
    logger.info("Clone-floor gate passed: %s", gate.message)

    # --- Run the comparison ------------------------------------------------ #
    logger.info("Running comparison:\n  %s", " ".join(command))
    completed = subprocess.run(command, check=False)
    if completed.returncode != 0:
        logger.error("comparison exited %s", completed.returncode)
        return completed.returncode

    result = _load_result(output_dir)
    _write_step_summary(render_step_summary(gate, result=result))
    return 0


def build_arg_parser() -> argparse.ArgumentParser:
    """Build the argument parser for the runner."""
    parser = argparse.ArgumentParser(
        description=(
            "Resolve baseline + candidate, run the clone-floor gate, run the "
            "sound eCPS-replacement comparison, and emit a GitHub Step Summary."
        )
    )
    parser.add_argument(
        "--candidate",
        required=True,
        help=("Candidate Microplex H5: local path, http(s):// URI, or hf://repo/file."),
    )
    parser.add_argument(
        "--baseline-source",
        default=None,
        help=(
            "Baseline source: a local Enhanced CPS H5 path, or 'latest' "
            "(default) to pull the latest published eCPS from Hugging Face."
        ),
    )
    parser.add_argument(
        "--work-dir",
        default="eval_work",
        help="Directory for downloaded artifacts.",
    )
    parser.add_argument(
        "--output-dir",
        default="comparison_output",
        help="Directory for comparison artifacts.",
    )
    parser.add_argument("--period", type=int, default=DEFAULT_PERIOD)
    parser.add_argument(
        "--matched-sample-method",
        choices=("uniform", "weight_proportional", "pps", "largest_weight"),
        default=DEFAULT_MATCHED_SAMPLE_METHOD,
    )
    parser.add_argument(
        "--holdout-target-fraction",
        type=float,
        default=DEFAULT_HOLDOUT_TARGET_FRACTION,
    )
    parser.add_argument(
        "--optimizer-max-iter",
        type=int,
        default=DEFAULT_OPTIMIZER_MAX_ITER,
    )
    parser.add_argument(
        "--clone-floor",
        type=float,
        default=DEFAULT_CLONE_FLOOR,
        help="Minimum acceptable baseline clone weight share (default 0.05).",
    )
    parser.add_argument(
        "--policyengine-us-data-repo",
        default=None,
        help="Optional path/URL passed through to the comparison.",
    )
    parser.add_argument(
        "--policyengine-us-data-python",
        default=None,
        help="Optional interpreter path passed through to the comparison.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """CLI entry point for the runner."""
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
