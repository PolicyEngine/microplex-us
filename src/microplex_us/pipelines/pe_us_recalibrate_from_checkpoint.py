"""Recalibrate a saved US microplex checkpoint with a new calibration config.

Load a ``post_imputation`` or ``post_microsim`` pipeline checkpoint
previously saved via
``pe_us_data_rebuild_checkpoint --pipeline-checkpoint-save-post-imputation-path``
(or ``--pipeline-checkpoint-save-post-microsim-path``) and rerun the
calibration stage without repeating the ~11 hours of synthesis + donor
imputation. A ``post_microsim`` checkpoint additionally skips the
microsim materialization step because the materialized vars are
already on the bundle as columns.

Intended for rapid iteration on calibration backends / target sets /
sparsity schedules: change one flag, run for ~30 min
(``post_imputation``) or ~1–2 min + calibration fit
(``post_microsim``) instead of half a day.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections.abc import Sequence
from pathlib import Path

from microplex_us.pipelines.us import (
    USMicroplexBuildConfig,
    USMicroplexPipeline,
    recalibrate_policyengine_us_from_checkpoint,
)


def _prepare_output_root(output_root: Path) -> Path:
    if not output_root.exists():
        raise FileNotFoundError(f"--output-root does not exist: {output_root}")
    if not output_root.is_dir():
        raise NotADirectoryError(f"--output-root is not a directory: {output_root}")
    if not os.access(output_root, os.W_OK | os.X_OK):
        raise PermissionError(f"--output-root is not writable: {output_root}")
    return output_root


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Rerun US microplex calibration from a saved checkpoint. Works "
            "with both post_imputation (skips ~11 h synthesis) and "
            "post_microsim (additionally skips ~30 min microsim) stages."
        ),
    )
    parser.add_argument(
        "--checkpoint-path",
        type=Path,
        required=True,
        help=(
            "Path to a directory written by the main pipeline with "
            "--pipeline-checkpoint-save-post-imputation-path or "
            "--pipeline-checkpoint-save-post-microsim-path."
        ),
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        required=True,
        help="Existing output directory for the recalibrated bundle and summary.",
    )
    parser.add_argument(
        "--targets-db",
        type=Path,
        required=True,
        help="Path to the PolicyEngine US targets SQLite database.",
    )
    parser.add_argument(
        "--arch-targets-db",
        type=Path,
        action="append",
        default=[],
        help=(
            "Path to an Arch target artifact. May be supplied multiple times. "
            "Required when --calibration-target-source=arch."
        ),
    )
    parser.add_argument(
        "--target-period",
        type=int,
        default=None,
        help="Calendar year for calibration targets (default: config default).",
    )
    parser.add_argument(
        "--target-profile",
        type=str,
        default=None,
        help="PolicyEngine target profile used for scoring/ledger context.",
    )
    parser.add_argument(
        "--calibration-target-source",
        choices=["policyengine", "arch"],
        default=None,
        help=(
            "Source for calibration targets. Defaults to the build config default "
            "unless supplied."
        ),
    )
    parser.add_argument(
        "--calibration-target-profile",
        type=str,
        default=None,
        help="Target profile used to select calibration constraints.",
    )
    parser.add_argument(
        "--calibration-backend",
        type=str,
        default="pe_l0",
        help="Calibration backend (pe_l0, microcalibrate, hardconcrete, etc.).",
    )
    parser.add_argument(
        "--calibration-max-iter",
        type=int,
        default=None,
        help="Max iterations / epochs for the calibration solver.",
    )
    parser.add_argument(
        "--policyengine-materialize-batch-size",
        type=int,
        default=100_000,
        help=(
            "Batch size for PE variable materialization (default 100_000; "
            "keeps a single Microsimulation under a few GB at 1.5M-household scale)."
        ),
    )
    parser.add_argument(
        "--pipeline-checkpoint-save-post-microsim-path",
        type=Path,
        default=None,
        help=(
            "If set, also save a post-microsim checkpoint during this "
            "recalibration so the next iteration can skip microsim too."
        ),
    )
    parser.add_argument(
        "--policyengine-dataset-output",
        type=Path,
        default=None,
        help=(
            "Optional PolicyEngine-readable H5 output path. When set, the "
            "recalibrated entity tables are exported after calibration."
        ),
    )
    args = parser.parse_args(argv)
    output_root = _prepare_output_root(args.output_root)

    config_kwargs: dict[str, object] = {
        "calibration_backend": args.calibration_backend,
        "policyengine_targets_db": args.targets_db,
        "policyengine_materialize_batch_size": int(
            args.policyengine_materialize_batch_size
        ),
    }
    if args.arch_targets_db:
        config_kwargs["arch_targets_db"] = tuple(
            str(path) for path in args.arch_targets_db
        )
    if args.target_period is not None:
        config_kwargs["policyengine_target_period"] = int(args.target_period)
    if args.target_profile is not None:
        config_kwargs["policyengine_target_profile"] = args.target_profile
    if args.calibration_target_source is not None:
        config_kwargs["calibration_target_source"] = args.calibration_target_source
    if args.calibration_target_profile is not None:
        config_kwargs["policyengine_calibration_target_profile"] = (
            args.calibration_target_profile
        )
    if args.calibration_max_iter is not None:
        config_kwargs["calibration_max_iter"] = int(args.calibration_max_iter)
    if args.pipeline_checkpoint_save_post_microsim_path is not None:
        config_kwargs["pipeline_checkpoint_save_post_microsim_path"] = (
            args.pipeline_checkpoint_save_post_microsim_path
        )

    config = USMicroplexBuildConfig(**config_kwargs)
    result = recalibrate_policyengine_us_from_checkpoint(config, args.checkpoint_path)

    result.calibrated_data.to_parquet(output_root / "calibrated_data.parquet")
    result.policyengine_tables.households.to_parquet(
        output_root / "households.parquet"
    )
    if result.policyengine_tables.persons is not None:
        result.policyengine_tables.persons.to_parquet(
            output_root / "persons.parquet"
        )
    if result.policyengine_tables.tax_units is not None:
        result.policyengine_tables.tax_units.to_parquet(
            output_root / "tax_units.parquet"
        )
    if result.policyengine_tables.spm_units is not None:
        result.policyengine_tables.spm_units.to_parquet(
            output_root / "spm_units.parquet"
        )
    if result.policyengine_tables.families is not None:
        result.policyengine_tables.families.to_parquet(
            output_root / "families.parquet"
        )
    if result.policyengine_tables.marital_units is not None:
        result.policyengine_tables.marital_units.to_parquet(
            output_root / "marital_units.parquet"
        )
    (output_root / "calibration_summary.json").write_text(
        json.dumps(result.calibration_summary, indent=2, default=str)
    )
    if args.policyengine_dataset_output is not None:
        USMicroplexPipeline(config).export_policyengine_dataset(
            result,
            args.policyengine_dataset_output,
            period=args.target_period,
        )
    print(
        f"Recalibrated from {args.checkpoint_path} → {output_root} "
        f"(stage={result.loaded_stage}, "
        f"rows={len(result.calibrated_data)})"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
