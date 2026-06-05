"""CLI for the PE-US-data checkpoint rebuild runner."""

from __future__ import annotations

import argparse
import json

from microplex_us.pipelines.pe_us_data_rebuild_checkpoint_runner import (
    run_policyengine_us_data_rebuild_checkpoint,
)
from microplex_us.pipelines.stage_contracts import US_CANONICAL_STAGE_IDS
from microplex_us.pipelines.stage_run import parse_us_stage_input_override


def main(argv: list[str] | None = None) -> None:
    """CLI entry point for one PE-US-data rebuild checkpoint."""

    parser = argparse.ArgumentParser(
        description="Run a versioned PE-US-data rebuild checkpoint in microplex-us."
    )
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--baseline-dataset", required=True)
    parser.add_argument("--targets-db", required=True)
    parser.add_argument("--policyengine-us-data-repo")
    parser.add_argument("--policyengine-us-data-python")
    parser.add_argument("--version-id")
    parser.add_argument("--target-period", type=int, default=2024)
    parser.add_argument("--target-profile", default="pe_native_broad")
    parser.add_argument("--calibration-target-profile")
    parser.add_argument(
        "--calibration-target-source",
        choices=["policyengine", "arch"],
        default="policyengine",
        help=(
            "Target provider used for calibration. Use 'arch' with "
            "--arch-targets-db for MP production calibration while keeping "
            "--target-profile on the PE/eCPS comparison surface."
        ),
    )
    parser.add_argument(
        "--arch-targets-db",
        action="append",
        default=[],
        help=(
            "Arch targets SQLite DB or consumer_facts.jsonl path for "
            "calibration. May be supplied more than once."
        ),
    )
    parser.add_argument("--n-synthetic", type=int, default=100_000)
    parser.add_argument("--random-seed", type=int, default=42)
    parser.add_argument("--donor-imputer-condition-selection")
    parser.add_argument(
        "--donor-imputer-backend",
        choices=["maf", "qrf", "zi_qrf", "regime_aware"],
        default=None,
        help=(
            "Donor imputer backend. `zi_qrf` activates the zero-inflated "
            "QRF path that skips predict() on gate-predicted-zero rows, "
            "which is a large wall-clock win on heavy-zero PUF tax "
            "variables. See docs/next-run-plan.md."
        ),
    )
    parser.add_argument("--cps-source-year", type=int, default=2023)
    parser.add_argument("--puf-target-year", type=int)
    parser.add_argument("--puf-cps-reference-year", type=int)
    parser.add_argument("--acs-year", type=int, default=2024)
    parser.add_argument("--sipp-year", type=int, default=2023)
    parser.add_argument("--scf-year", type=int, default=2022)
    parser.add_argument("--cps-cache-dir")
    parser.add_argument("--puf-cache-dir")
    parser.add_argument("--donor-cache-dir")
    parser.add_argument("--puf-path")
    parser.add_argument("--puf-demographics-path")
    parser.add_argument("--cps-sample-n", type=int)
    parser.add_argument("--puf-sample-n", type=int)
    parser.add_argument("--donor-sample-n", type=int)
    parser.add_argument("--query-random-seed", type=int, default=0)
    parser.add_argument("--target-variable", action="append", default=[])
    parser.add_argument("--target-domain", action="append", default=[])
    parser.add_argument("--target-geo-level", action="append", default=[])
    parser.add_argument("--calibration-target-variable", action="append", default=[])
    parser.add_argument("--calibration-target-domain", action="append", default=[])
    parser.add_argument("--calibration-target-geo-level", action="append", default=[])
    parser.add_argument(
        "--include-donor-surveys",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--include-sipp",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Include SIPP donor providers. Defaults to --include-donor-surveys.",
    )
    parser.add_argument(
        "--include-scf",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Include the SCF donor provider. Defaults to --include-donor-surveys.",
    )
    parser.add_argument("--no-cps-download", action="store_true")
    parser.add_argument("--no-puf-expand-persons", action="store_true")
    parser.add_argument("--defer-policyengine-harness", action="store_true")
    parser.add_argument("--defer-policyengine-native-score", action="store_true")
    parser.add_argument("--defer-native-audit", action="store_true")
    parser.add_argument("--defer-imputation-ablation", action="store_true")
    parser.add_argument("--require-policyengine-native-score", action="store_true")
    parser.add_argument(
        "--calibration-backend",
        choices=[
            "entropy",
            "ipf",
            "chi2",
            "sparse",
            "hardconcrete",
            "pe_l0",
            "microcalibrate",
            "none",
        ],
        default=None,
        help=(
            "Weighting/calibration backend. Default is the config default "
            "(entropy). Use `microcalibrate` for the identity-preserving "
            "gradient-descent chi-squared backend that survived the v6 OOM."
        ),
    )
    parser.add_argument(
        "--calibration-max-iter",
        type=int,
        default=None,
        help=(
            "Max iterations / epochs for the calibration solver. Passed "
            "through to USMicroplexBuildConfig.calibration_max_iter."
        ),
    )
    parser.add_argument(
        "--policyengine-materialize-batch-size",
        type=int,
        default=None,
        help=(
            "If set, splits PolicyEngine variable materialization into "
            "household chunks of this size. At 1.5M-household scale a "
            "single Microsimulation is 25-35 GB; batch_size=100_000 "
            "drops peak to a few GB. Required for workstation runs; "
            "unset (full-dataset) path targeted Modal GPU."
        ),
    )
    parser.add_argument(
        "--pipeline-checkpoint-save-post-imputation-path",
        type=str,
        default=None,
        help=(
            "If set, save a post-imputation pipeline checkpoint to this "
            "directory (right after donor imputation + PE tables build, "
            "before microsim). A rerun can resume from this checkpoint "
            "to skip the ~11 h synthesis stage."
        ),
    )
    parser.add_argument(
        "--policyengine-export-column-contract-path",
        type=str,
        default=None,
        help=(
            "If set, check the eCPS export-column contract from the "
            "post-imputation PE entity tables before microsimulation and "
            "calibration."
        ),
    )
    parser.add_argument(
        "--pipeline-checkpoint-save-post-microsim-path",
        type=str,
        default=None,
        help=(
            "If set, save a post-microsim pipeline checkpoint to this "
            "directory (after target variables are materialized, before "
            "the calibration fit loop). A rerun can resume from this "
            "checkpoint to skip both synthesis and microsim, leaving "
            "only the calibration fit."
        ),
    )
    parser.add_argument(
        "--resume-from-stage",
        choices=US_CANONICAL_STAGE_IDS,
        default=None,
        help=(
            "Resume an existing saved run from this canonical stage. Requires "
            "--version-id unless --output-root points directly at a saved artifact "
            "directory. The runner validates required durable artifacts before "
            "starting."
        ),
    )
    parser.add_argument(
        "--capital-gains-lots",
        action=argparse.BooleanOptionalAction,
        default=None,
        help=(
            "Write an anchor-preserving synthetic capital-gains lot SQLite "
            "sidecar artifact from PolicyEngine person tables."
        ),
    )
    parser.add_argument("--capital-gains-lots-max-lots-per-person", type=int)
    parser.add_argument("--capital-gains-lots-random-seed", type=int)
    parser.add_argument(
        "--allow-stage-input-overrides",
        action="store_true",
        help=(
            "Allow typed stage manifests to consume explicit CLI input overrides "
            "instead of the immediately previous stage manifest."
        ),
    )
    parser.add_argument(
        "--stage-input-override",
        action="append",
        default=[],
        metavar="STAGE_ID.KEY=PATH",
        help=("Explicit stage input override. Requires --allow-stage-input-overrides."),
    )
    args = parser.parse_args(argv)
    stage_input_overrides = tuple(
        parse_us_stage_input_override(value) for value in args.stage_input_override
    )
    if stage_input_overrides and not args.allow_stage_input_overrides:
        parser.error("--stage-input-override requires --allow-stage-input-overrides")

    config_overrides = {
        "n_synthetic": int(args.n_synthetic),
        "random_seed": int(args.random_seed),
    }
    if args.donor_imputer_condition_selection is not None:
        config_overrides["donor_imputer_condition_selection"] = (
            args.donor_imputer_condition_selection
        )
    if args.donor_imputer_backend is not None:
        config_overrides["donor_imputer_backend"] = args.donor_imputer_backend
    if args.calibration_backend is not None:
        config_overrides["calibration_backend"] = args.calibration_backend
    if args.calibration_max_iter is not None:
        config_overrides["calibration_max_iter"] = int(args.calibration_max_iter)
    if args.policyengine_materialize_batch_size is not None:
        config_overrides["policyengine_materialize_batch_size"] = int(
            args.policyengine_materialize_batch_size
        )
    if args.pipeline_checkpoint_save_post_imputation_path is not None:
        config_overrides["pipeline_checkpoint_save_post_imputation_path"] = (
            args.pipeline_checkpoint_save_post_imputation_path
        )
    if args.policyengine_export_column_contract_path is not None:
        config_overrides["policyengine_export_column_contract_path"] = (
            args.policyengine_export_column_contract_path
        )
    if args.pipeline_checkpoint_save_post_microsim_path is not None:
        config_overrides["pipeline_checkpoint_save_post_microsim_path"] = (
            args.pipeline_checkpoint_save_post_microsim_path
        )
    if args.capital_gains_lots is not None:
        config_overrides["capital_gains_lots_enabled"] = bool(args.capital_gains_lots)
    if args.capital_gains_lots_max_lots_per_person is not None:
        config_overrides["capital_gains_lots_max_lots_per_person"] = int(
            args.capital_gains_lots_max_lots_per_person
        )
    if args.capital_gains_lots_random_seed is not None:
        config_overrides["capital_gains_lots_random_seed"] = int(
            args.capital_gains_lots_random_seed
        )

    result = run_policyengine_us_data_rebuild_checkpoint(
        output_root=args.output_root,
        policyengine_baseline_dataset=args.baseline_dataset,
        policyengine_targets_db=args.targets_db,
        arch_targets_db=(tuple(args.arch_targets_db) if args.arch_targets_db else None),
        calibration_target_source=args.calibration_target_source,
        target_period=args.target_period,
        target_profile=args.target_profile,
        calibration_target_profile=args.calibration_target_profile,
        target_variables=tuple(args.target_variable),
        target_domains=tuple(args.target_domain),
        target_geo_levels=tuple(args.target_geo_level),
        calibration_target_variables=tuple(args.calibration_target_variable),
        calibration_target_domains=tuple(args.calibration_target_domain),
        calibration_target_geo_levels=tuple(args.calibration_target_geo_level),
        config_overrides=config_overrides,
        cps_source_year=args.cps_source_year,
        cps_cache_dir=args.cps_cache_dir,
        cps_download=not args.no_cps_download,
        puf_target_year=args.puf_target_year,
        puf_cps_reference_year=args.puf_cps_reference_year,
        puf_cache_dir=args.puf_cache_dir,
        puf_path=args.puf_path,
        puf_demographics_path=args.puf_demographics_path,
        puf_expand_persons=not args.no_puf_expand_persons,
        include_donor_surveys=args.include_donor_surveys,
        include_sipp=args.include_sipp,
        include_scf=args.include_scf,
        acs_year=args.acs_year,
        sipp_year=args.sipp_year,
        scf_year=args.scf_year,
        donor_cache_dir=args.donor_cache_dir,
        policyengine_us_data_repo=args.policyengine_us_data_repo,
        policyengine_us_data_python=args.policyengine_us_data_python,
        cps_sample_n=args.cps_sample_n,
        puf_sample_n=args.puf_sample_n,
        donor_sample_n=args.donor_sample_n,
        query_random_seed=args.query_random_seed,
        version_id=args.version_id,
        defer_policyengine_harness=args.defer_policyengine_harness,
        require_policyengine_native_score=args.require_policyengine_native_score,
        defer_policyengine_native_score=args.defer_policyengine_native_score,
        defer_native_audit=args.defer_native_audit,
        defer_imputation_ablation=args.defer_imputation_ablation,
        allow_stage_input_overrides=args.allow_stage_input_overrides,
        stage_input_overrides=stage_input_overrides,
        resume_from_stage=args.resume_from_stage,
    )

    print(result.artifacts.artifact_paths.output_dir)
    print(result.parity_path)
    print(json.dumps(result.parity_payload["verdict"], indent=2, sort_keys=True))
