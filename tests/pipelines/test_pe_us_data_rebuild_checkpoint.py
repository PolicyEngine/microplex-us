"""Tests for the PE-US-data rebuild checkpoint runner."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import h5py
import pandas as pd
from microplex.core import SourceQuery

import microplex_us.pipelines.pe_us_data_rebuild_checkpoint as checkpoint_module
from microplex_us.pipelines.artifacts import (
    USMicroplexArtifactPaths,
    USMicroplexVersionedBuildArtifacts,
)
from microplex_us.pipelines.pe_us_data_rebuild import (
    default_policyengine_us_data_rebuild_source_providers,
)
from microplex_us.pipelines.pe_us_data_rebuild_checkpoint import (
    attach_policyengine_us_data_rebuild_checkpoint_evidence,
    default_policyengine_us_data_rebuild_checkpoint_config,
    default_policyengine_us_data_rebuild_queries,
    run_policyengine_us_data_rebuild_checkpoint,
)
from microplex_us.pipelines.registry import load_us_microplex_run_registry


def test_default_policyengine_us_data_rebuild_checkpoint_config_sets_pe_context() -> (
    None
):
    config = default_policyengine_us_data_rebuild_checkpoint_config(
        policyengine_baseline_dataset="/tmp/enhanced_cps_2024.h5",
        policyengine_targets_db="/tmp/policy_data.db",
        target_period=2024,
        target_profile="pe_native_broad",
        n_synthetic=500,
        random_seed=123,
    )

    assert config.synthesis_backend == "seed"
    assert config.calibration_backend == "entropy"
    assert config.policyengine_calibration_min_active_households == 20
    assert config.policyengine_calibration_deferred_stage_min_active_households == (
        10,
        1,
    )
    assert config.policyengine_calibration_deferred_stage_max_constraints == 24
    assert (
        config.policyengine_calibration_deferred_stage_min_full_oracle_capped_mean_abs_relative_error
        is None
    )
    assert config.policyengine_calibration_deferred_stage_top_family_count == 7
    assert config.policyengine_calibration_deferred_stage_top_geography_count == 4
    assert config.donor_imputer_backend == "qrf"
    assert config.donor_imputer_condition_selection == "pe_prespecified"
    assert config.donor_imputer_excluded_variables == ()
    assert config.policyengine_baseline_dataset == "/tmp/enhanced_cps_2024.h5"
    assert config.policyengine_targets_db == "/tmp/policy_data.db"
    assert config.policyengine_dataset_year == 2024
    assert config.policyengine_target_period == 2024
    assert config.policyengine_target_profile == "pe_native_broad"
    assert config.policyengine_calibration_target_profile == "pe_native_broad"
    assert config.policyengine_calibration_target_variables == ()
    assert config.policyengine_oracle_relative_error_cap == 10.0
    assert config.policyengine_direct_override_variables == (
        "health_savings_account_ald",
        "non_sch_d_capital_gains",
    )
    assert config.policyengine_prefer_existing_tax_unit_ids is True
    assert config.n_synthetic == 500
    assert config.random_seed == 123


def test_default_policyengine_us_data_rebuild_checkpoint_config_preserves_explicit_calibration_scope() -> (
    None
):
    config = default_policyengine_us_data_rebuild_checkpoint_config(
        policyengine_baseline_dataset="/tmp/enhanced_cps_2024.h5",
        policyengine_targets_db="/tmp/policy_data.db",
        calibration_target_variables=("snap",),
    )

    assert config.policyengine_calibration_target_variables == ("snap",)


def test_default_policyengine_us_data_rebuild_checkpoint_config_infers_total_weight_targets(
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        "microplex_us.pipelines.pe_us_data_rebuild_checkpoint._infer_policyengine_baseline_household_weight_sum",
        lambda dataset, *, target_period: 150_000_000.0,
    )

    config = default_policyengine_us_data_rebuild_checkpoint_config(
        policyengine_baseline_dataset="/tmp/enhanced_cps_2024.h5",
        policyengine_targets_db="/tmp/policy_data.db",
        target_period=2024,
    )

    assert config.policyengine_calibration_target_total_weight == 150_000_000.0
    assert config.policyengine_calibration_rescale_to_target_total_weight is True
    assert config.policyengine_selection_target_total_weight == 150_000_000.0


def test_default_policyengine_us_data_rebuild_checkpoint_config_respects_explicit_total_weight_overrides(
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        "microplex_us.pipelines.pe_us_data_rebuild_checkpoint._infer_policyengine_baseline_household_weight_sum",
        lambda dataset, *, target_period: 150_000_000.0,
    )

    config = default_policyengine_us_data_rebuild_checkpoint_config(
        policyengine_baseline_dataset="/tmp/enhanced_cps_2024.h5",
        policyengine_targets_db="/tmp/policy_data.db",
        target_period=2024,
        policyengine_calibration_target_total_weight=123.0,
        policyengine_selection_target_total_weight=456.0,
    )

    assert config.policyengine_calibration_target_total_weight == 123.0
    assert config.policyengine_selection_target_total_weight == 456.0


def test_default_policyengine_us_data_rebuild_checkpoint_config_skips_calibration_total_weight_when_rescaling_to_input_sum(
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        "microplex_us.pipelines.pe_us_data_rebuild_checkpoint._infer_policyengine_baseline_household_weight_sum",
        lambda dataset, *, target_period: 150_000_000.0,
    )

    config = default_policyengine_us_data_rebuild_checkpoint_config(
        policyengine_baseline_dataset="/tmp/enhanced_cps_2024.h5",
        policyengine_targets_db="/tmp/policy_data.db",
        target_period=2024,
        policyengine_calibration_rescale_to_input_weight_sum=True,
    )

    assert config.policyengine_calibration_target_total_weight is None
    assert config.policyengine_calibration_rescale_to_target_total_weight is False
    assert config.policyengine_selection_target_total_weight == 150_000_000.0


def test_default_policyengine_us_data_rebuild_checkpoint_config_skips_inferred_total_weight_targets_for_no_calibration(
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        "microplex_us.pipelines.pe_us_data_rebuild_checkpoint._infer_policyengine_baseline_household_weight_sum",
        lambda dataset, *, target_period: 150_000_000.0,
    )

    config = default_policyengine_us_data_rebuild_checkpoint_config(
        policyengine_baseline_dataset="/tmp/enhanced_cps_2024.h5",
        policyengine_targets_db="/tmp/policy_data.db",
        target_period=2024,
        calibration_backend="none",
    )

    assert config.calibration_backend == "none"
    assert config.policyengine_calibration_target_total_weight is None
    assert config.policyengine_calibration_rescale_to_target_total_weight is False
    assert config.policyengine_selection_target_total_weight is None


def test_infer_policyengine_baseline_household_weight_sum_returns_none_when_weight_array_missing(
    tmp_path,
) -> None:
    from microplex_us.pipelines.pe_us_data_rebuild_checkpoint import (
        _infer_policyengine_baseline_household_weight_sum,
    )

    dataset_path = tmp_path / "baseline.h5"
    with h5py.File(dataset_path, "w") as handle:
        household_id = handle.create_group("household_id")
        household_id.create_dataset("2024", data=[1, 2, 3])

    inferred = _infer_policyengine_baseline_household_weight_sum(
        dataset_path,
        target_period=2024,
    )

    assert inferred is None


def test_default_policyengine_us_data_rebuild_queries_assign_sample_sizes_by_provider_type() -> (
    None
):
    providers = default_policyengine_us_data_rebuild_source_providers(
        include_donor_surveys=True,
        cps_download=False,
    )

    queries = default_policyengine_us_data_rebuild_queries(
        providers,
        cps_sample_n=11,
        puf_sample_n=22,
        donor_sample_n=33,
        random_seed=7,
    )

    assert queries[providers[0].descriptor.name].provider_filters == {
        "sample_n": 11,
        "random_seed": 7,
        "state_age_floor": 1,
    }
    assert queries[providers[1].descriptor.name].provider_filters == {
        "sample_n": 22,
        "random_seed": 7,
    }
    for provider in providers[2:]:
        assert queries[provider.descriptor.name].provider_filters == {
            "sample_n": 33,
            "random_seed": 7,
            "state_age_floor": 1,
        }


def test_default_policyengine_us_data_rebuild_queries_derive_donor_sample_size_from_sampled_sources() -> (
    None
):
    providers = default_policyengine_us_data_rebuild_source_providers(
        include_donor_surveys=True,
        cps_download=False,
    )

    queries = default_policyengine_us_data_rebuild_queries(
        providers,
        cps_sample_n=11,
        puf_sample_n=22,
        random_seed=7,
    )

    assert queries[providers[0].descriptor.name].provider_filters == {
        "sample_n": 11,
        "random_seed": 7,
        "state_age_floor": 1,
    }
    for provider in providers[2:]:
        assert queries[provider.descriptor.name].provider_filters == {
            "sample_n": 22,
            "random_seed": 7,
            "state_age_floor": 1,
        }


def test_default_policyengine_us_data_rebuild_queries_can_disable_cps_state_age_floor() -> (
    None
):
    providers = default_policyengine_us_data_rebuild_source_providers(
        include_donor_surveys=False,
        cps_download=False,
    )

    queries = default_policyengine_us_data_rebuild_queries(
        providers,
        cps_sample_n=11,
        puf_sample_n=22,
        cps_state_age_floor=None,
        random_seed=7,
    )

    assert queries[providers[0].descriptor.name].provider_filters == {
        "sample_n": 11,
        "random_seed": 7,
    }


@dataclass(frozen=True)
class _FakeProvider:
    descriptor: Any


def test_run_policyengine_us_data_rebuild_checkpoint_builds_bundle_and_parity(
    monkeypatch,
    tmp_path,
    caplog,
) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(
        "microplex_us.pipelines.pe_us_data_rebuild_checkpoint._infer_policyengine_baseline_household_weight_sum",
        lambda dataset, *, target_period: 150_000_000.0,
    )
    artifact_dir = tmp_path / "artifacts" / "run-1"
    artifact_dir.mkdir(parents=True)
    provider = _FakeProvider(descriptor=SimpleNamespace(name="fake_source"))
    query = SourceQuery(provider_filters={"sample_n": 5})
    captured: dict[str, Any] = {}

    def fake_build_and_save_versioned_us_microplex_from_source_providers(
        *,
        providers,
        output_root,
        config,
        queries,
        version_id,
        frontier_metric,
        policyengine_comparison_cache,
        policyengine_target_provider,
        policyengine_baseline_dataset,
        policyengine_harness_slices,
        policyengine_harness_metadata,
        policyengine_us_data_repo,
        defer_policyengine_harness,
        require_policyengine_native_score,
        defer_policyengine_native_score,
        precomputed_policyengine_harness_payload,
        precomputed_policyengine_native_scores,
        run_registry_path,
        run_index_path,
        run_registry_metadata,
        enable_child_tax_unit_agi_drift,
        allow_stage_input_overrides,
        stage_input_overrides,
    ):
        captured.update(
            {
                "providers": providers,
                "output_root": output_root,
                "config": config,
                "queries": queries,
                "version_id": version_id,
                "frontier_metric": frontier_metric,
                "policyengine_baseline_dataset": policyengine_baseline_dataset,
                "policyengine_harness_metadata": policyengine_harness_metadata,
                "run_registry_metadata": run_registry_metadata,
                "defer_policyengine_harness": defer_policyengine_harness,
                "defer_policyengine_native_score": defer_policyengine_native_score,
                "enable_child_tax_unit_agi_drift": enable_child_tax_unit_agi_drift,
                "allow_stage_input_overrides": allow_stage_input_overrides,
                "stage_input_overrides": stage_input_overrides,
            }
        )
        manifest = {
            "created_at": "2026-04-06T00:00:00+00:00",
            "config": config.to_dict(),
            "rows": {"seed": 10, "synthetic": 20, "calibrated": 20},
            "weights": {"nonzero": 20, "total": 20.0},
            "targets": {"n_marginal_groups": 1, "n_continuous": 0},
            "synthesis": {
                "scaffold_source": "fake_source",
                "source_names": ["fake_source"],
                "backend": "seed",
                "condition_vars": [],
                "target_vars": [],
                "donor_integrated_variables": [],
                "state_program_support_proxies": {"available": [], "missing": []},
            },
            "calibration": {
                "converged": True,
                "n_loaded_targets": 1,
                "n_supported_targets": 1,
                "full_oracle_capped_mean_abs_relative_error": 0.12,
                "full_oracle_mean_abs_relative_error": 0.12,
            },
            "artifacts": {
                "seed_data": "seed_data.parquet",
                "synthetic_data": "synthetic_data.parquet",
                "calibrated_data": "calibrated_data.parquet",
                "targets": "targets.json",
                "policyengine_dataset": "policyengine_us.h5",
            },
        }
        (artifact_dir / "manifest.json").write_text(json.dumps(manifest))
        (artifact_dir / "policyengine_us.h5").write_text("dataset")
        return USMicroplexVersionedBuildArtifacts(
            build_result=SimpleNamespace(config=config),
            artifact_paths=USMicroplexArtifactPaths(
                output_dir=artifact_dir,
                version_id="run-1",
                seed_data=artifact_dir / "seed_data.parquet",
                synthetic_data=artifact_dir / "synthetic_data.parquet",
                calibrated_data=artifact_dir / "calibrated_data.parquet",
                targets=artifact_dir / "targets.json",
                manifest=artifact_dir / "manifest.json",
            ),
        )

    def fake_write_policyengine_us_data_rebuild_parity_artifact(
        artifact_dir_arg,
        output_path=None,
        *,
        program=None,
        manifest_payload=None,
        harness_payload=None,
        native_scores_payload=None,
    ) -> Path:
        assert manifest_payload is None
        assert harness_payload is None
        assert native_scores_payload is None
        path = (
            Path(output_path)
            if output_path is not None
            else Path(artifact_dir_arg) / "pe_us_data_rebuild_parity.json"
        )
        path.write_text(
            json.dumps(
                {
                    "program": {"programId": program.program_id},
                    "verdict": {"hasRealPolicyEngineComparison": False},
                }
            )
        )
        return path

    def fake_build_policyengine_us_data_rebuild_parity_artifact(
        artifact_dir_arg,
        *,
        program=None,
        manifest_payload=None,
        harness_payload=None,
        native_scores_payload=None,
    ) -> dict[str, Any]:
        assert manifest_payload is None
        assert harness_payload is None
        assert native_scores_payload is None
        return {
            "artifactId": Path(artifact_dir_arg).name,
            "program": {"programId": program.program_id},
            "verdict": {"hasRealPolicyEngineComparison": False},
        }

    module_name = "microplex_us.pipelines.pe_us_data_rebuild_checkpoint"
    monkeypatch.setattr(
        f"{module_name}.build_and_save_versioned_us_microplex_from_source_providers",
        fake_build_and_save_versioned_us_microplex_from_source_providers,
    )

    def fake_attach_policyengine_us_data_rebuild_checkpoint_evidence(
        artifact_dir_arg,
        **kwargs,
    ):
        captured["attach_kwargs"] = kwargs
        artifact_root = Path(artifact_dir_arg)
        registry_path = tmp_path / "artifacts" / "run_registry.jsonl"
        run_index_path = tmp_path / "artifacts" / "run_index.duckdb"
        manifest_path = artifact_root / "manifest.json"
        manifest = json.loads(manifest_path.read_text())
        manifest["artifacts"]["policyengine_harness"] = "policyengine_harness.json"
        (artifact_root / "policyengine_harness.json").write_text(
            json.dumps(
                {
                    "summary": {
                        "candidate_mean_abs_relative_error": 0.08,
                        "baseline_mean_abs_relative_error": 0.10,
                        "mean_abs_relative_error_delta": -0.02,
                    }
                }
            )
        )
        manifest["policyengine_harness"] = {
            "candidate_mean_abs_relative_error": 0.08,
            "baseline_mean_abs_relative_error": 0.10,
            "mean_abs_relative_error_delta": -0.02,
        }
        registry_path.write_text(
            json.dumps(
                {
                    "created_at": "2026-04-06T00:00:00+00:00",
                    "artifact_id": "run-1",
                    "artifact_dir": str(artifact_root.resolve()),
                    "manifest_path": str(manifest_path.resolve()),
                    "policyengine_harness_path": str(
                        (artifact_root / "policyengine_harness.json").resolve()
                    ),
                    "full_oracle_capped_mean_abs_relative_error": 0.12,
                    "full_oracle_mean_abs_relative_error": 0.12,
                    "enhanced_cps_native_loss_delta": 0.5,
                }
            )
            + "\n"
        )
        run_index_path.write_text("")
        manifest["run_registry"] = {
            "path": "artifacts/run_registry.jsonl",
            "artifact_id": "run-1",
        }
        manifest["run_index"] = {
            "path": "artifacts/run_index.duckdb",
            "artifact_id": "run-1",
        }
        if kwargs.get("precomputed_imputation_ablation_payload") is not None:
            manifest["artifacts"]["imputation_ablation"] = "imputation_ablation.json"
            manifest["imputation_ablation"] = dict(
                kwargs["precomputed_imputation_ablation_payload"].get("summary", {})
            )
            (artifact_root / "imputation_ablation.json").write_text(
                json.dumps(kwargs["precomputed_imputation_ablation_payload"])
            )
        manifest["artifacts"]["policyengine_native_audit"] = (
            "pe_us_data_rebuild_native_audit.json"
        )
        manifest["policyengine_native_audit"] = {
            "largestRegressingFamily": None,
        }
        (artifact_root / "pe_us_data_rebuild_native_audit.json").write_text(
            json.dumps({"verdictHints": {"largestRegressingFamily": None}})
        )
        manifest_path.write_text(json.dumps(manifest))
        return SimpleNamespace(
            artifact_dir=artifact_root,
            manifest_path=manifest_path,
            harness_path=artifact_root / "policyengine_harness.json",
            native_scores_path=None,
            parity_path=fake_write_policyengine_us_data_rebuild_parity_artifact(
                artifact_dir_arg,
                program=kwargs.get("program"),
            ),
            parity_payload=fake_build_policyengine_us_data_rebuild_parity_artifact(
                artifact_dir_arg,
                program=kwargs.get("program"),
            ),
            native_audit_path=artifact_root / "pe_us_data_rebuild_native_audit.json",
            native_audit_payload={"verdictHints": {"largestRegressingFamily": None}},
            imputation_ablation_path=(
                artifact_root / "imputation_ablation.json"
                if kwargs.get("precomputed_imputation_ablation_payload") is not None
                else None
            ),
            imputation_ablation_payload=kwargs.get(
                "precomputed_imputation_ablation_payload"
            ),
        )

    monkeypatch.setattr(
        f"{module_name}.attach_policyengine_us_data_rebuild_checkpoint_evidence",
        fake_attach_policyengine_us_data_rebuild_checkpoint_evidence,
    )

    caplog.set_level(
        logging.INFO,
        logger="microplex_us.pipelines.pe_us_data_rebuild_checkpoint",
    )
    result = run_policyengine_us_data_rebuild_checkpoint(
        output_root=tmp_path / "artifacts",
        policyengine_baseline_dataset="/tmp/enhanced_cps_2024.h5",
        policyengine_targets_db="/tmp/policy_data.db",
        providers=(provider,),
        queries={"fake_source": query},
        version_id="run-1",
    )

    assert result.provider_names == ("fake_source",)
    assert result.queries == {"fake_source": query}
    assert result.parity_path == artifact_dir / "pe_us_data_rebuild_parity.json"
    assert result.parity_payload["program"]["programId"] == "pe-us-data-rebuild-v1"
    assert captured["providers"] == [provider]
    assert captured["queries"] == {"fake_source": query}
    assert captured["version_id"] == "run-1"
    assert captured["frontier_metric"] == "full_oracle_capped_mean_abs_relative_error"
    assert captured["policyengine_baseline_dataset"] == "/tmp/enhanced_cps_2024.h5"
    assert captured["config"].policyengine_targets_db == "/tmp/policy_data.db"
    assert (
        captured["config"].policyengine_calibration_target_total_weight == 150_000_000.0
    )
    assert (
        captured["config"].policyengine_calibration_rescale_to_target_total_weight
        is True
    )
    assert (
        captured["config"].policyengine_selection_target_total_weight == 150_000_000.0
    )
    assert captured["defer_policyengine_harness"] is True
    assert captured["defer_policyengine_native_score"] is True
    assert captured["enable_child_tax_unit_agi_drift"] is True
    assert captured["allow_stage_input_overrides"] is False
    assert captured["stage_input_overrides"] == ()
    assert captured["policyengine_harness_metadata"]["rebuild_checkpoint"] is True
    assert captured["policyengine_harness_metadata"]["rebuild_program_id"] == (
        "pe-us-data-rebuild-v1"
    )
    assert captured["policyengine_harness_metadata"]["rebuild_provider_names"] == [
        "fake_source"
    ]
    assert captured["run_registry_metadata"]["rebuild_profile_expected"] is True
    assert captured["attach_kwargs"]["build_result"].config == captured["config"]
    assert captured["attach_kwargs"]["compute_native_audit"] is True
    assert captured["attach_kwargs"]["compute_imputation_ablation"] is True
    assert captured["attach_kwargs"]["precomputed_imputation_ablation_payload"] is None
    assert (
        result.artifacts.artifact_paths.policyengine_harness
        == artifact_dir / "policyengine_harness.json"
    )
    assert (
        result.artifacts.artifact_paths.run_registry
        == tmp_path / "artifacts" / "run_registry.jsonl"
    )
    assert (
        result.artifacts.artifact_paths.run_index_db
        == tmp_path / "artifacts" / "run_index.duckdb"
    )
    assert (
        result.artifacts.artifact_paths.policyengine_native_audit
        == artifact_dir / "pe_us_data_rebuild_native_audit.json"
    )
    assert result.artifacts.current_entry is not None
    assert result.artifacts.current_entry.artifact_id == "run-1"
    assert result.artifacts.frontier_entry is not None
    assert result.artifacts.frontier_entry.artifact_id == "run-1"
    assert result.artifacts.frontier_delta == 0.0
    assert (
        result.native_audit_path
        == artifact_dir / "pe_us_data_rebuild_native_audit.json"
    )
    assert result.native_audit_payload == {
        "verdictHints": {"largestRegressingFamily": None}
    }
    assert result.imputation_ablation_path is None
    assert result.imputation_ablation_payload is None
    log_messages = [record.getMessage() for record in caplog.records]
    assert any(
        "PE-US-data rebuild checkpoint: starting build" in message
        and "version_id=run-1" in message
        and "providers=fake_source" in message
        for message in log_messages
    )
    assert any(
        "PE-US-data rebuild checkpoint: build complete" in message
        and str(artifact_dir) in message
        for message in log_messages
    )
    assert any(
        "PE-US-data rebuild checkpoint: attaching PE evidence" in message
        and "compute_native_audit=True" in message
        for message in log_messages
    )
    assert any(
        "PE-US-data rebuild checkpoint: evidence complete" in message
        and "pe_us_data_rebuild_parity.json" in message
        for message in log_messages
    )
    assert any(
        "PE-US-data rebuild checkpoint: checkpoint ready" in message
        and str(artifact_dir) in message
        for message in log_messages
    )


def test_emit_checkpoint_progress_falls_back_to_stderr_when_no_logger_handlers(
    monkeypatch,
    capsys,
) -> None:
    emitted: list[str] = []

    class _FakeLogger:
        handlers: list[object] = []

        def info(self, message: str) -> None:
            emitted.append(message)

    monkeypatch.setattr(checkpoint_module, "LOGGER", _FakeLogger())
    monkeypatch.setattr(checkpoint_module, "_root_logger_has_handlers", lambda: False)

    checkpoint_module._emit_checkpoint_progress(
        "PE-US-data rebuild checkpoint: starting build",
        version_id="run-1",
        providers="fake_source",
    )

    stderr = capsys.readouterr().err
    assert emitted == [
        "PE-US-data rebuild checkpoint: starting build "
        "[version_id=run-1, providers=fake_source]"
    ]
    assert (
        stderr == "PE-US-data rebuild checkpoint: starting build "
        "[version_id=run-1, providers=fake_source]\n"
    )


def test_main_passes_donor_condition_selection_override(monkeypatch, capsys) -> None:
    captured: dict[str, Any] = {}
    artifact_dir = Path("/tmp/artifacts/run-1")
    parity_path = artifact_dir / "pe_us_data_rebuild_parity.json"

    def fake_run_policyengine_us_data_rebuild_checkpoint(**kwargs):
        captured.update(kwargs)
        return SimpleNamespace(
            artifacts=SimpleNamespace(
                artifact_paths=SimpleNamespace(output_dir=artifact_dir)
            ),
            parity_path=parity_path,
            parity_payload={
                "verdict": {"hasRealPolicyEngineComparison": True},
            },
        )

    monkeypatch.setattr(
        checkpoint_module,
        "run_policyengine_us_data_rebuild_checkpoint",
        fake_run_policyengine_us_data_rebuild_checkpoint,
    )

    checkpoint_module.main(
        [
            "--output-root",
            "/tmp/artifacts",
            "--baseline-dataset",
            "/tmp/enhanced_cps_2024.h5",
            "--targets-db",
            "/tmp/policy_data.db",
            "--version-id",
            "run-1",
            "--donor-imputer-condition-selection",
            "pe_plus_puf_native_challenger",
            "--defer-native-audit",
            "--defer-imputation-ablation",
        ]
    )

    assert captured["config_overrides"]["donor_imputer_condition_selection"] == (
        "pe_plus_puf_native_challenger"
    )
    assert captured["config_overrides"]["n_synthetic"] == 100_000
    assert captured["config_overrides"]["random_seed"] == 42
    assert captured["defer_native_audit"] is True
    assert captured["defer_imputation_ablation"] is True
    stdout = capsys.readouterr().out
    assert "/tmp/artifacts/run-1" in stdout
    assert "hasRealPolicyEngineComparison" in stdout


def test_run_policyengine_us_data_rebuild_checkpoint_rejects_empty_provider_sequence(
    tmp_path,
) -> None:
    try:
        run_policyengine_us_data_rebuild_checkpoint(
            output_root=tmp_path / "artifacts",
            policyengine_baseline_dataset="/tmp/enhanced_cps_2024.h5",
            policyengine_targets_db="/tmp/policy_data.db",
            providers=(),
        )
    except ValueError as exc:
        assert "non-empty provider sequence" in str(exc)
    else:
        raise AssertionError("Expected empty providers to fail closed")


def test_run_policyengine_us_data_rebuild_checkpoint_rejects_unknown_query_keys(
    tmp_path,
) -> None:
    provider = _FakeProvider(descriptor=SimpleNamespace(name="fake_source"))
    try:
        run_policyengine_us_data_rebuild_checkpoint(
            output_root=tmp_path / "artifacts",
            policyengine_baseline_dataset="/tmp/enhanced_cps_2024.h5",
            policyengine_targets_db="/tmp/policy_data.db",
            providers=(provider,),
            queries={"typo_source": SourceQuery(provider_filters={"sample_n": 5})},
        )
    except ValueError as exc:
        assert "unknown provider keys" in str(exc)
        assert "fake_source" in str(exc)
    else:
        raise AssertionError("Expected unknown query keys to fail")


def test_run_policyengine_us_data_rebuild_checkpoint_rejects_mismatched_explicit_config(
    tmp_path,
) -> None:
    config = default_policyengine_us_data_rebuild_checkpoint_config(
        policyengine_baseline_dataset="/tmp/enhanced_cps_2024.h5",
        policyengine_targets_db="/tmp/policy_data.db",
        target_period=2024,
    )
    provider = _FakeProvider(descriptor=SimpleNamespace(name="fake_source"))

    try:
        run_policyengine_us_data_rebuild_checkpoint(
            output_root=tmp_path / "artifacts",
            policyengine_baseline_dataset="/tmp/other_baseline.h5",
            policyengine_targets_db="/tmp/policy_data.db",
            config=config,
            providers=(provider,),
            queries={"fake_source": SourceQuery(provider_filters={"sample_n": 5})},
        )
    except ValueError as exc:
        assert "does not match the requested PE rebuild context" in str(exc)
        assert "policyengine_baseline_dataset" in str(exc)
    else:
        raise AssertionError("Expected mismatched explicit config to fail")


def test_run_policyengine_us_data_rebuild_checkpoint_accepts_matching_explicit_config_default_calibration_scope(
    tmp_path,
) -> None:
    config = default_policyengine_us_data_rebuild_checkpoint_config(
        policyengine_baseline_dataset="/tmp/enhanced_cps_2024.h5",
        policyengine_targets_db="/tmp/policy_data.db",
        target_period=2024,
    )

    try:
        run_policyengine_us_data_rebuild_checkpoint(
            output_root=tmp_path / "artifacts",
            policyengine_baseline_dataset="/tmp/enhanced_cps_2024.h5",
            policyengine_targets_db="/tmp/policy_data.db",
            config=config,
            providers=(),
        )
    except ValueError as exc:
        assert "non-empty provider sequence" in str(exc)
        assert "requested PE rebuild context" not in str(exc)
    else:
        raise AssertionError("Expected empty providers to fail after validation")


def test_run_policyengine_us_data_rebuild_checkpoint_rejects_custom_python_without_native_defer(
    tmp_path,
) -> None:
    provider = _FakeProvider(descriptor=SimpleNamespace(name="fake_source"))
    try:
        run_policyengine_us_data_rebuild_checkpoint(
            output_root=tmp_path / "artifacts",
            policyengine_baseline_dataset="/tmp/enhanced_cps_2024.h5",
            policyengine_targets_db="/tmp/policy_data.db",
            providers=(provider,),
            queries={"fake_source": SourceQuery(provider_filters={"sample_n": 5})},
            policyengine_us_data_python="/tmp/venv/bin/python",
        )
    except ValueError as exc:
        assert "defer_policyengine_native_score=True" in str(exc)
    else:
        raise AssertionError("Expected unsupported custom PE Python path to fail")


def test_attach_policyengine_us_data_rebuild_checkpoint_evidence_updates_manifest(
    monkeypatch,
    tmp_path,
) -> None:
    artifact_dir = tmp_path / "artifact"
    artifact_dir.mkdir()
    manifest = {
        "created_at": "2026-04-06T00:00:00+00:00",
        "config": default_policyengine_us_data_rebuild_checkpoint_config(
            policyengine_baseline_dataset="/tmp/enhanced_cps_2024.h5",
            policyengine_targets_db="/tmp/policy_data.db",
            target_period=2024,
        ).to_dict(),
        "rows": {"seed": 10, "synthetic": 20, "calibrated": 20},
        "weights": {"nonzero": 20, "total": 20.0},
        "targets": {"n_marginal_groups": 1, "n_continuous": 0},
        "synthesis": {
            "scaffold_source": "cps_asec_2023",
            "source_names": ["cps_asec_2023", "irs_soi_puf"],
            "backend": "seed",
            "condition_vars": [],
            "target_vars": [],
            "donor_integrated_variables": [],
            "state_program_support_proxies": {"available": [], "missing": []},
        },
        "calibration": {
            "converged": True,
            "n_loaded_targets": 1,
            "n_supported_targets": 1,
            "full_oracle_capped_mean_abs_relative_error": 0.12,
            "full_oracle_mean_abs_relative_error": 0.12,
        },
        "artifacts": {
            "seed_data": "seed_data.parquet",
            "synthetic_data": "synthetic_data.parquet",
            "calibrated_data": "calibrated_data.parquet",
            "targets": "targets.json",
            "policyengine_dataset": "policyengine_us.h5",
        },
    }
    (artifact_dir / "manifest.json").write_text(json.dumps(manifest))
    (artifact_dir / "data_flow_snapshot.json").write_text(
        json.dumps(
            {
                "schemaVersion": 1,
                "stages": [
                    {
                        "id": "benchmark",
                        "status": "missing",
                        "metrics": [],
                        "outputs": [],
                    }
                ],
            }
        )
    )
    for name in (
        "seed_data.parquet",
        "synthetic_data.parquet",
        "calibrated_data.parquet",
        "targets.json",
        "policyengine_us.h5",
    ):
        (artifact_dir / name).write_text("{}")

    harness_payload = {
        "candidate_label": "microplex",
        "baseline_label": "policyengine_us_data",
        "period": 2024,
        "metadata": {"slice_profile": "pe_native_broad"},
        "summary": {
            "candidate_mean_abs_relative_error": 0.08,
            "baseline_mean_abs_relative_error": 0.10,
            "mean_abs_relative_error_delta": -0.02,
            "candidate_composite_parity_loss": 0.14,
            "baseline_composite_parity_loss": 0.15,
            "composite_parity_loss_delta": -0.01,
            "slice_win_rate": 0.55,
            "target_win_rate": 0.58,
            "supported_target_rate": 0.98,
            "baseline_supported_target_rate": 0.99,
            "tag_summaries": {},
            "parity_scorecard": {},
            "attribute_cell_summaries": {},
        },
    }
    native_scores_payload = {
        "metric": "enhanced_cps_native_loss",
        "period": 2024,
        "summary": {
            "candidate_enhanced_cps_native_loss": 0.30,
            "baseline_enhanced_cps_native_loss": 0.20,
            "enhanced_cps_native_loss_delta": 0.10,
            "candidate_beats_baseline": False,
        },
    }
    imputation_ablation_payload = {
        "schema_version": 1,
        "artifact_id": "artifact",
        "production_variant": "structured_pe_conditioning",
        "summary": {
            "source_count": 1,
            "skipped_source_count": 0,
            "target_count": 3,
            "production_variant": "structured_pe_conditioning",
            "production_mean_weighted_mae": 0.21,
            "production_mean_support_f1": 0.88,
            "best_mean_weighted_mae_variant": "structured_pe_conditioning",
            "best_mean_support_f1_variant": "structured_pe_conditioning",
            "variant_scorecard": {
                "structured_pe_conditioning": {
                    "source_count": 1,
                    "mean_weighted_mae": 0.21,
                    "mean_support_f1": 0.88,
                }
            },
        },
        "source_reports": {},
        "skipped_sources": [],
    }

    module_name = "microplex_us.pipelines.pe_us_data_rebuild_checkpoint"
    monkeypatch.setattr(
        f"{module_name}.write_policyengine_us_data_rebuild_parity_artifact",
        lambda artifact_dir_arg, **kwargs: (
            Path(artifact_dir_arg) / "pe_us_data_rebuild_parity.json"
        ),
    )
    monkeypatch.setattr(
        f"{module_name}.build_policyengine_us_data_rebuild_parity_artifact",
        lambda artifact_dir_arg, **kwargs: {
            "artifactId": Path(artifact_dir_arg).name,
            "verdict": {"hasRealPolicyEngineComparison": True},
        },
    )
    native_audit_payload = {
        "artifactId": "artifact",
        "period": 2024,
        "targetDelta": {
            "metric": "enhanced_cps_native_loss_target_delta",
            "period": 2024,
            "from_dataset": "/tmp/enhanced_cps_2024.h5",
            "to_dataset": "/tmp/policyengine_us.h5",
            "summary": {"n_targets": 1, "to_win_rate": 1.0},
            "family_summaries": [{"target_family": "national_irs_other"}],
            "scope_summaries": [{"target_scope": "national"}],
            "targets": [
                {
                    "target_name": "nation/irs/example",
                    "target_family": "national_irs_other",
                    "target_scope": "national",
                    "winner": "to",
                    "weighted_term_delta": -1.0,
                    "from_weighted_term": 2.0,
                    "to_weighted_term": 1.0,
                    "target_value": 100.0,
                    "from_estimate": 90.0,
                    "to_estimate": 95.0,
                    "from_rel_error": 0.2,
                    "to_rel_error": 0.1,
                }
            ],
            "top_regressions": [],
            "top_improvements": [],
        },
        "verdictHints": {
            "productionImputationVariantIsMaeWinner": True,
            "productionImputationVariantIsSupportWinner": True,
        },
    }
    monkeypatch.setattr(
        f"{module_name}.build_policyengine_us_data_rebuild_native_audit",
        lambda artifact_dir_arg, **kwargs: native_audit_payload,
    )

    result = attach_policyengine_us_data_rebuild_checkpoint_evidence(
        artifact_dir,
        compute_harness=False,
        compute_native_scores=False,
        precomputed_policyengine_harness_payload=harness_payload,
        precomputed_policyengine_native_scores=native_scores_payload,
        precomputed_imputation_ablation_payload=imputation_ablation_payload,
        run_registry_path=tmp_path / "run_registry.jsonl",
        run_index_path=tmp_path,
        run_registry_metadata={"checkpoint_test": True},
    )

    written_manifest = json.loads((artifact_dir / "manifest.json").read_text())
    refreshed_snapshot = json.loads(
        (artifact_dir / "data_flow_snapshot.json").read_text()
    )
    benchmark_stage = next(
        stage
        for stage in refreshed_snapshot["stages"]
        if stage["id"] == "09_validation_benchmarking"
    )
    registry_entries = load_us_microplex_run_registry(tmp_path / "run_registry.jsonl")
    assert result.harness_path == artifact_dir / "policyengine_harness.json"
    assert result.native_scores_path == artifact_dir / "policyengine_native_scores.json"
    assert (
        result.native_audit_path
        == artifact_dir / "pe_us_data_rebuild_native_audit.json"
    )
    assert (
        result.native_target_diagnostics_path
        == artifact_dir / "pe_native_target_diagnostics.json"
    )
    assert result.native_audit_payload == native_audit_payload
    assert result.native_target_diagnostics_payload is not None
    assert result.imputation_ablation_path == artifact_dir / "imputation_ablation.json"
    written_native_audit = json.loads(
        (artifact_dir / "pe_us_data_rebuild_native_audit.json").read_text()
    )
    written_target_diagnostics = json.loads(
        (artifact_dir / "pe_native_target_diagnostics.json").read_text()
    )
    assert written_target_diagnostics["artifact_id"] == "artifact"
    assert written_target_diagnostics["run_id"] == "artifact"
    assert written_target_diagnostics["targets"][0]["artifact_id"] == "artifact"
    assert (
        written_manifest["artifacts"]["policyengine_harness"]
        == "policyengine_harness.json"
    )
    assert (
        written_manifest["artifacts"]["policyengine_native_scores"]
        == "policyengine_native_scores.json"
    )
    assert (
        written_manifest["artifacts"]["policyengine_native_audit"]
        == "pe_us_data_rebuild_native_audit.json"
    )
    assert (
        written_manifest["artifacts"]["policyengine_native_target_diagnostics"]
        == "pe_native_target_diagnostics.json"
    )
    assert (
        written_manifest["artifacts"]["imputation_ablation"]
        == "imputation_ablation.json"
    )
    assert (
        written_manifest["policyengine_harness"]["mean_abs_relative_error_delta"]
        == -0.02
    )
    assert (
        written_manifest["policyengine_native_scores"]["enhanced_cps_native_loss_delta"]
        == 0.10
    )
    assert written_manifest["run_registry"]["default_frontier_metric"] == (
        "full_oracle_capped_mean_abs_relative_error"
    )
    assert (
        written_manifest["imputation_ablation"]["production_mean_weighted_mae"] == 0.21
    )
    assert (
        written_manifest["policyengine_native_audit"][
            "productionImputationVariantIsMaeWinner"
        ]
        is True
    )
    assert (
        written_native_audit["verdictHints"]["productionImputationVariantIsMaeWinner"]
        is True
    )
    assert written_target_diagnostics["diagnostic_schema_version"] == 1
    assert written_target_diagnostics["dataset_labels"] == {
        "from": "policyengine-us-data",
        "to": "microplex-us",
    }
    first_target = written_target_diagnostics["targets"][0]
    assert first_target["target_id"] == "nation/irs/example"
    assert first_target["us_data_absolute_error"] == 10.0
    assert first_target["microplex_absolute_error"] == 5.0
    assert first_target["delta_absolute_error"] == -5.0
    assert written_manifest["run_registry"]["artifact_id"] == "artifact"
    assert written_manifest["run_index"]["artifact_id"] == "artifact"
    assert (tmp_path / "run_index.duckdb").exists()
    assert len(registry_entries) == 1
    assert registry_entries[0].artifact_id == "artifact"
    assert registry_entries[0].full_oracle_capped_mean_abs_relative_error == 0.12
    assert registry_entries[0].full_oracle_mean_abs_relative_error == 0.12
    assert registry_entries[0].metadata["checkpoint_test"] is True
    assert benchmark_stage["status"] == "ready"
    assert benchmark_stage["outputs"] == [
        "policyengine_harness.json",
        "policyengine_native_scores.json",
        "imputation_ablation.json",
        "pe_us_data_rebuild_native_audit.json",
        "pe_native_target_diagnostics.json",
    ]
    assert {metric["label"]: metric["value"] for metric in benchmark_stage["metrics"]}[
        "Capped full oracle loss"
    ] == 0.12
    assert {metric["label"]: metric["value"] for metric in benchmark_stage["metrics"]}[
        "Full oracle loss"
    ] == 0.12
    assert {metric["label"]: metric["value"] for metric in benchmark_stage["metrics"]}[
        "Imputation MAE"
    ] == 0.21
    assert {metric["label"]: metric["value"] for metric in benchmark_stage["metrics"]}[
        "Imputation F1"
    ] == 0.88


def test_attach_policyengine_us_data_rebuild_checkpoint_evidence_registers_calibration_only_runs(
    monkeypatch,
    tmp_path,
) -> None:
    artifact_dir = tmp_path / "artifact"
    artifact_dir.mkdir()
    manifest = {
        "created_at": "2026-04-06T00:00:00+00:00",
        "config": default_policyengine_us_data_rebuild_checkpoint_config(
            policyengine_baseline_dataset="/tmp/enhanced_cps_2024.h5",
            policyengine_targets_db="/tmp/policy_data.db",
            target_period=2024,
        ).to_dict(),
        "rows": {"seed": 10, "synthetic": 20, "calibrated": 20},
        "weights": {"nonzero": 20, "total": 20.0},
        "targets": {"n_marginal_groups": 1, "n_continuous": 0},
        "synthesis": {"source_names": ["cps_asec_2023", "irs_soi_puf"]},
        "calibration": {
            "converged": True,
            "n_loaded_targets": 1,
            "n_supported_targets": 1,
            "full_oracle_capped_mean_abs_relative_error": 0.12,
            "full_oracle_mean_abs_relative_error": 0.12,
        },
        "artifacts": {
            "seed_data": "seed_data.parquet",
            "synthetic_data": "synthetic_data.parquet",
            "calibrated_data": "calibrated_data.parquet",
            "targets": "targets.json",
            "policyengine_dataset": "policyengine_us.h5",
        },
    }
    (artifact_dir / "manifest.json").write_text(json.dumps(manifest))
    (artifact_dir / "data_flow_snapshot.json").write_text(
        json.dumps({"schemaVersion": 1, "stages": []})
    )
    for name in (
        "seed_data.parquet",
        "synthetic_data.parquet",
        "calibrated_data.parquet",
        "targets.json",
        "policyengine_us.h5",
    ):
        (artifact_dir / name).write_text("{}")

    module_name = "microplex_us.pipelines.pe_us_data_rebuild_checkpoint"
    monkeypatch.setattr(
        f"{module_name}.write_policyengine_us_data_rebuild_parity_artifact",
        lambda artifact_dir_arg, **kwargs: (
            Path(artifact_dir_arg) / "pe_us_data_rebuild_parity.json"
        ),
    )
    monkeypatch.setattr(
        f"{module_name}.build_policyengine_us_data_rebuild_parity_artifact",
        lambda artifact_dir_arg, **kwargs: {
            "artifactId": Path(artifact_dir_arg).name,
            "verdict": {"hasRealPolicyEngineComparison": False},
        },
    )

    attach_policyengine_us_data_rebuild_checkpoint_evidence(
        artifact_dir,
        compute_harness=False,
        compute_native_scores=False,
        compute_native_audit=False,
        compute_imputation_ablation=False,
        run_registry_path=tmp_path / "run_registry.jsonl",
        run_index_path=tmp_path,
    )

    written_manifest = json.loads((artifact_dir / "manifest.json").read_text())
    registry_entries = load_us_microplex_run_registry(tmp_path / "run_registry.jsonl")

    assert written_manifest["run_registry"]["default_frontier_metric"] == (
        "full_oracle_capped_mean_abs_relative_error"
    )
    assert registry_entries[0].artifact_id == "artifact"
    assert registry_entries[0].full_oracle_capped_mean_abs_relative_error == 0.12
    assert registry_entries[0].full_oracle_mean_abs_relative_error == 0.12


def test_load_checkpoint_versioned_artifacts_hydrates_stage_sidecar_paths(
    tmp_path,
) -> None:
    artifact_dir = tmp_path / "artifact"
    artifact_dir.mkdir()
    stage_artifacts = artifact_dir / "stage_artifacts"
    for path in (
        artifact_dir / "seed_data.parquet",
        artifact_dir / "synthetic_data.parquet",
        artifact_dir / "calibrated_data.parquet",
        artifact_dir / "targets.json",
        artifact_dir / "policyengine_us.h5",
        artifact_dir / "stage_manifest.json",
        artifact_dir / "data_flow_snapshot.json",
        stage_artifacts / "03_source_planning" / "source_plan.json",
        stage_artifacts / "04_seed_scaffold" / "scaffold_seed_data.parquet",
        stage_artifacts / "06_policyengine_entities" / "metadata.json",
        stage_artifacts / "07_calibration" / "calibration_summary.json",
        stage_artifacts / "09_validation_benchmarking" / "evidence_manifest.json",
        stage_artifacts / "artifact_inventory.json",
        stage_artifacts / "conditional_readiness.json",
        artifact_dir / "policyengine_native_scores.json",
        artifact_dir / "source_weight_diagnostics.json",
    ):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("{}")
    manifest = {
        "artifacts": {
            "seed_data": "seed_data.parquet",
            "synthetic_data": "synthetic_data.parquet",
            "calibrated_data": "calibrated_data.parquet",
            "targets": "targets.json",
            "policyengine_dataset": "policyengine_us.h5",
            "stage_manifest": "stage_manifest.json",
            "data_flow_snapshot": "data_flow_snapshot.json",
            "source_plan": "stage_artifacts/03_source_planning/source_plan.json",
            "scaffold_seed_data": (
                "stage_artifacts/04_seed_scaffold/scaffold_seed_data.parquet"
            ),
            "policyengine_entity_tables": (
                "stage_artifacts/06_policyengine_entities/metadata.json"
            ),
            "calibration_summary": (
                "stage_artifacts/07_calibration/calibration_summary.json"
            ),
            "validation_evidence": (
                "stage_artifacts/09_validation_benchmarking/evidence_manifest.json"
            ),
            "artifact_inventory": "stage_artifacts/artifact_inventory.json",
            "conditional_readiness": "stage_artifacts/conditional_readiness.json",
            "policyengine_native_scores": "policyengine_native_scores.json",
            "source_weight_diagnostics": "source_weight_diagnostics.json",
        }
    }
    (artifact_dir / "manifest.json").write_text(json.dumps(manifest))

    loaded = checkpoint_module._load_checkpoint_versioned_artifacts(
        build_result=SimpleNamespace(),
        artifact_root=artifact_dir,
        frontier_metric="full_oracle_mean_abs_relative_error",
    )
    paths = loaded.artifact_paths

    assert paths.stage_manifest == artifact_dir / "stage_manifest.json"
    assert paths.data_flow_snapshot == artifact_dir / "data_flow_snapshot.json"
    assert paths.artifact_inventory == stage_artifacts / "artifact_inventory.json"
    assert paths.conditional_readiness == stage_artifacts / "conditional_readiness.json"
    assert (
        paths.source_plan == stage_artifacts / "03_source_planning" / "source_plan.json"
    )
    assert paths.scaffold_seed_data == (
        stage_artifacts / "04_seed_scaffold" / "scaffold_seed_data.parquet"
    )
    assert paths.policyengine_entity_tables == (
        stage_artifacts / "06_policyengine_entities" / "metadata.json"
    )
    assert paths.calibration_summary == (
        stage_artifacts / "07_calibration" / "calibration_summary.json"
    )
    assert paths.validation_evidence == (
        stage_artifacts / "09_validation_benchmarking" / "evidence_manifest.json"
    )
    assert paths.policyengine_native_scores == (
        artifact_dir / "policyengine_native_scores.json"
    )
    assert paths.source_weight_diagnostics == (
        artifact_dir / "source_weight_diagnostics.json"
    )


def test_attach_policyengine_us_data_rebuild_checkpoint_evidence_computes_imputation_ablation_with_build_result(
    monkeypatch,
    tmp_path,
) -> None:
    artifact_dir = tmp_path / "artifact"
    artifact_dir.mkdir()
    manifest = {
        "created_at": "2026-04-06T00:00:00+00:00",
        "config": default_policyengine_us_data_rebuild_checkpoint_config(
            policyengine_baseline_dataset="/tmp/enhanced_cps_2024.h5",
            policyengine_targets_db="/tmp/policy_data.db",
            target_period=2024,
        ).to_dict(),
        "rows": {"seed": 10, "synthetic": 20, "calibrated": 20},
        "weights": {"nonzero": 20, "total": 20.0},
        "targets": {"n_marginal_groups": 1, "n_continuous": 0},
        "synthesis": {
            "scaffold_source": "cps_asec_2023",
            "source_names": ["cps_asec_2023", "irs_soi_puf"],
            "backend": "seed",
            "condition_vars": [],
            "target_vars": [],
            "donor_integrated_variables": [],
            "state_program_support_proxies": {"available": [], "missing": []},
        },
        "calibration": {
            "converged": True,
            "n_loaded_targets": 1,
            "n_supported_targets": 1,
            "full_oracle_capped_mean_abs_relative_error": 0.12,
            "full_oracle_mean_abs_relative_error": 0.12,
        },
        "artifacts": {
            "seed_data": "seed_data.parquet",
            "synthetic_data": "synthetic_data.parquet",
            "calibrated_data": "calibrated_data.parquet",
            "targets": "targets.json",
            "policyengine_dataset": "policyengine_us.h5",
        },
    }
    (artifact_dir / "manifest.json").write_text(json.dumps(manifest))
    for name in (
        "seed_data.parquet",
        "synthetic_data.parquet",
        "calibrated_data.parquet",
        "targets.json",
        "policyengine_us.h5",
    ):
        (artifact_dir / name).write_text("{}")

    harness_payload = {
        "summary": {
            "candidate_mean_abs_relative_error": 0.08,
            "baseline_mean_abs_relative_error": 0.10,
            "mean_abs_relative_error_delta": -0.02,
        }
    }
    native_scores_payload = {
        "summary": {
            "candidate_enhanced_cps_native_loss": 0.30,
            "enhanced_cps_native_loss_delta": 0.10,
        }
    }
    imputation_ablation_payload = {
        "schema_version": 1,
        "artifact_id": "artifact",
        "production_variant": "structured_pe_conditioning",
        "summary": {
            "source_count": 1,
            "production_mean_weighted_mae": 0.19,
            "production_mean_support_f1": 0.91,
        },
        "source_reports": {},
        "skipped_sources": [],
    }
    captured: dict[str, Any] = {}
    build_result = SimpleNamespace(
        config=SimpleNamespace(donor_imputer_condition_selection="pe_prespecified")
    )

    module_name = "microplex_us.pipelines.pe_us_data_rebuild_checkpoint"
    monkeypatch.setattr(
        f"{module_name}.write_policyengine_us_data_rebuild_parity_artifact",
        lambda artifact_dir_arg, **kwargs: (
            Path(artifact_dir_arg) / "pe_us_data_rebuild_parity.json"
        ),
    )
    monkeypatch.setattr(
        f"{module_name}.build_policyengine_us_data_rebuild_parity_artifact",
        lambda artifact_dir_arg, **kwargs: {
            "artifactId": Path(artifact_dir_arg).name,
            "verdict": {"hasRealPolicyEngineComparison": True},
        },
    )
    native_audit_payload = {
        "artifactId": "artifact",
        "verdictHints": {
            "productionImputationVariantIsMaeWinner": False,
            "productionImputationVariantIsSupportWinner": True,
        },
    }
    monkeypatch.setattr(
        f"{module_name}.build_policyengine_us_data_rebuild_native_audit",
        lambda artifact_dir_arg, **kwargs: native_audit_payload,
    )

    def fake_build_checkpoint_imputation_ablation_payload(
        build_result_arg,
        *,
        artifact_id,
        manifest,
    ):
        captured["build_result"] = build_result_arg
        captured["artifact_id"] = artifact_id
        captured["manifest"] = manifest
        return imputation_ablation_payload

    monkeypatch.setattr(
        f"{module_name}._build_checkpoint_imputation_ablation_payload",
        fake_build_checkpoint_imputation_ablation_payload,
    )

    result = attach_policyengine_us_data_rebuild_checkpoint_evidence(
        artifact_dir,
        build_result=build_result,
        compute_harness=False,
        compute_native_scores=False,
        compute_imputation_ablation=True,
        precomputed_policyengine_harness_payload=harness_payload,
        precomputed_policyengine_native_scores=native_scores_payload,
    )

    written_manifest = json.loads((artifact_dir / "manifest.json").read_text())
    assert captured["build_result"] is build_result
    assert captured["artifact_id"] == "artifact"
    assert (
        captured["manifest"]["policyengine_harness"]["mean_abs_relative_error_delta"]
        == -0.02
    )
    assert (
        captured["manifest"]["policyengine_native_scores"][
            "enhanced_cps_native_loss_delta"
        ]
        == 0.10
    )
    assert result.imputation_ablation_payload == imputation_ablation_payload
    assert result.native_audit_payload == native_audit_payload
    assert (
        result.native_audit_path
        == artifact_dir / "pe_us_data_rebuild_native_audit.json"
    )
    assert result.imputation_ablation_path == artifact_dir / "imputation_ablation.json"
    assert (
        written_manifest["artifacts"]["policyengine_native_audit"]
        == "pe_us_data_rebuild_native_audit.json"
    )
    assert (
        written_manifest["policyengine_native_audit"][
            "productionImputationVariantIsSupportWinner"
        ]
        is True
    )
    assert (
        written_manifest["artifacts"]["imputation_ablation"]
        == "imputation_ablation.json"
    )


def test_build_checkpoint_imputation_ablation_payload_returns_none_when_no_donor_reports(
    monkeypatch,
) -> None:
    from microplex_us.pipelines.pe_us_data_rebuild_checkpoint import (
        _build_checkpoint_imputation_ablation_payload,
    )

    class FakePipeline:
        def __init__(self, config):
            self.config = config

        def prepare_source_input(self, frame):
            return SimpleNamespace(frame=frame)

        def prepare_seed_data_from_source(self, source_input):
            return pd.DataFrame(
                {"household_id": [1], "person_id": [1], "hh_weight": [1.0]}
            )

    monkeypatch.setattr(
        "microplex_us.pipelines.us.USMicroplexPipeline",
        FakePipeline,
    )
    scaffold_frame = SimpleNamespace(source=SimpleNamespace(name="scaffold"))

    payload = _build_checkpoint_imputation_ablation_payload(
        SimpleNamespace(
            config=SimpleNamespace(
                donor_imputer_condition_selection="pe_prespecified",
            ),
            source_frame=scaffold_frame,
            source_frames=(scaffold_frame,),
        ),
        artifact_id="artifact",
        manifest={},
    )

    assert payload is None
