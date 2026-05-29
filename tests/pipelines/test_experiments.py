"""Tests for source-mix PE-US experiment runners."""

from __future__ import annotations

import json
from pathlib import Path

from microplex_us.pipelines.artifacts import USMicroplexArtifactPaths
from microplex_us.pipelines.experiments import (
    USMicroplexExperimentReport,
    USMicroplexExperimentResult,
    USMicroplexSourceExperimentSpec,
    _refresh_experiment_results_from_registry,
    build_us_n_synthetic_sweep_experiments,
    default_us_source_mix_experiments,
    run_us_microplex_n_synthetic_sweep,
    run_us_microplex_source_experiments,
)
from microplex_us.pipelines.performance import USMicroplexPerformanceHarnessConfig
from microplex_us.pipelines.registry import USMicroplexRunRegistryEntry
from microplex_us.pipelines.us import USMicroplexBuildConfig


class _DummyProvider:
    def __init__(self, name: str):
        self.descriptor = type("Descriptor", (), {"name": name})()


def _artifact_paths(root: Path, name: str) -> USMicroplexArtifactPaths:
    output_dir = root / name
    return USMicroplexArtifactPaths(
        output_dir=output_dir,
        version_id=name,
        seed_data=output_dir / "seed.parquet",
        synthetic_data=output_dir / "synthetic.parquet",
        calibrated_data=output_dir / "calibrated.parquet",
        targets=output_dir / "targets.json",
        manifest=output_dir / "manifest.json",
        synthesizer=None,
        policyengine_dataset=output_dir / "policyengine.h5",
        data_flow_snapshot=output_dir / "data_flow_snapshot.json",
        artifact_inventory=output_dir / "stage_artifacts" / "artifact_inventory.json",
        conditional_readiness=(
            output_dir / "stage_artifacts" / "conditional_readiness.json"
        ),
        policyengine_harness=output_dir / "policyengine_harness.json",
        policyengine_native_scores=output_dir / "policyengine_native_scores.json",
        policyengine_native_audit=output_dir / "pe_us_data_rebuild_native_audit.json",
        run_registry=root / "run_registry.jsonl",
        run_index_db=root / "run_index.duckdb",
    )


def _entry(
    artifact_id: str,
    *,
    composite_loss: float,
    source_names: tuple[str, ...],
) -> USMicroplexRunRegistryEntry:
    return USMicroplexRunRegistryEntry(
        created_at="2026-03-25T12:00:00+00:00",
        artifact_id=artifact_id,
        artifact_dir=f"/tmp/{artifact_id}",
        manifest_path=f"/tmp/{artifact_id}/manifest.json",
        candidate_mean_abs_relative_error=composite_loss + 0.1,
        baseline_mean_abs_relative_error=0.5,
        mean_abs_relative_error_delta=composite_loss - 0.25,
        candidate_composite_parity_loss=composite_loss,
        baseline_composite_parity_loss=0.5,
        composite_parity_loss_delta=composite_loss - 0.5,
        source_names=source_names,
    )


def test_run_us_microplex_source_experiments_saves_report_and_sorts(monkeypatch, tmp_path):
    call_log: list[dict[str, object]] = []

    def fake_build_and_save(
        providers,
        output_root,
        *,
        config=None,
        queries=None,
        frontier_metric="candidate_composite_parity_loss",
        policyengine_comparison_cache=None,
        policyengine_target_provider=None,
        policyengine_baseline_dataset=None,
        policyengine_harness_slices=None,
        policyengine_harness_metadata=None,
        run_registry_path=None,
        run_registry_metadata=None,
    ):
        experiment_name = run_registry_metadata["experiment_name"]
        call_log.append(
            {
                "name": experiment_name,
                "frontier_metric": frontier_metric,
                "policyengine_comparison_cache": policyengine_comparison_cache,
                "run_registry_metadata": dict(run_registry_metadata),
                "policyengine_harness_metadata": dict(policyengine_harness_metadata),
            }
        )
        composite_loss = 0.35 if experiment_name == "cps+puf" else 0.45
        current_entry = _entry(
            experiment_name,
            composite_loss=composite_loss,
            source_names=tuple(provider.descriptor.name for provider in providers),
        )
        return type(
            "FakeArtifacts",
            (),
            {
                "artifact_paths": _artifact_paths(Path(output_root), experiment_name),
                "current_entry": current_entry,
                "frontier_entry": current_entry,
                "frontier_delta": 0.0,
            },
        )()

    monkeypatch.setattr(
        "microplex_us.pipelines.experiments.build_and_save_versioned_us_microplex_from_source_providers",
        fake_build_and_save,
    )

    report = run_us_microplex_source_experiments(
        [
            USMicroplexSourceExperimentSpec(
                name="cps-only",
                providers=(_DummyProvider("cps"),),
                metadata={"family": "baseline"},
            ),
            USMicroplexSourceExperimentSpec(
                name="cps+puf",
                providers=(_DummyProvider("cps"), _DummyProvider("puf")),
                metadata={"family": "tax"},
            ),
        ],
        tmp_path / "experiments",
        metadata={"suite": "parity-search"},
    )

    assert report.best_result is not None
    assert report.best_result.name == "cps+puf"
    assert [result.name for result in report.leaderboard] == ["cps+puf", "cps-only"]
    assert report.metadata["suite"] == "parity-search"
    assert len(call_log) == 2
    assert call_log[0]["frontier_metric"] == "candidate_composite_parity_loss"
    assert call_log[0]["policyengine_comparison_cache"] is call_log[1]["policyengine_comparison_cache"]
    assert call_log[0]["run_registry_metadata"]["experiment_name"] == "cps-only"
    assert call_log[1]["policyengine_harness_metadata"]["experiment_name"] == "cps+puf"

    report_path = tmp_path / "experiments" / "experiment_report.json"
    assert report_path.exists()
    loaded = USMicroplexExperimentReport.load(report_path)
    assert loaded.best_result is not None
    assert loaded.best_result.name == "cps+puf"
    assert loaded.leaderboard[0].current_entry is not None
    assert loaded.leaderboard[0].current_entry.candidate_composite_parity_loss == 0.35
    assert loaded.leaderboard[0].artifact_paths.data_flow_snapshot is not None
    assert loaded.leaderboard[0].artifact_paths.artifact_inventory is not None
    assert loaded.leaderboard[0].artifact_paths.conditional_readiness is not None
    assert loaded.leaderboard[0].artifact_paths.policyengine_native_scores is not None
    assert loaded.leaderboard[0].artifact_paths.policyengine_native_audit is not None
    assert loaded.leaderboard[0].artifact_paths.run_index_db is not None


def test_run_us_microplex_source_experiments_requires_at_least_one_experiment(tmp_path):
    try:
        run_us_microplex_source_experiments([], tmp_path / "experiments")
    except ValueError as exc:
        assert "at least one experiment" in str(exc)
    else:
        raise AssertionError("Expected ValueError for empty experiment batch")


def test_build_us_n_synthetic_sweep_experiments_updates_names_and_config():
    base_experiment = USMicroplexSourceExperimentSpec(
        name="cps+puf",
        providers=(_DummyProvider("cps"), _DummyProvider("puf")),
        config=USMicroplexBuildConfig(n_synthetic=500, random_seed=11),
        metadata={"family": "tax"},
    )

    sweep = build_us_n_synthetic_sweep_experiments(base_experiment, [2000, 10000])

    assert [experiment.name for experiment in sweep] == [
        "cps+puf-n2000",
        "cps+puf-n10000",
    ]
    assert [experiment.config.n_synthetic for experiment in sweep] == [2000, 10000]
    assert all(experiment.config.random_seed == 11 for experiment in sweep)
    assert sweep[0].metadata["family"] == "tax"
    assert sweep[0].metadata["base_experiment_name"] == "cps+puf"
    assert sweep[1].metadata["n_synthetic"] == 10000


def test_run_us_microplex_n_synthetic_sweep_expands_metadata(monkeypatch, tmp_path):
    captured: dict[str, object] = {}

    def fake_run_source_experiments(
        experiments,
        output_root,
        *,
        frontier_metric="candidate_composite_parity_loss",
        policyengine_target_provider=None,
        policyengine_baseline_dataset=None,
        policyengine_comparison_cache=None,
        policyengine_harness_slices=None,
        policyengine_harness_metadata=None,
        run_registry_path=None,
        report_path=None,
        performance_harness_config=None,
        performance_session=None,
        metadata=None,
    ):
        captured["experiments"] = experiments
        captured["output_root"] = Path(output_root)
        captured["metadata"] = dict(metadata or {})
        return USMicroplexExperimentReport(
            output_root=Path(output_root),
            frontier_metric=frontier_metric,
            results=(),
            metadata=dict(metadata or {}),
        )

    monkeypatch.setattr(
        "microplex_us.pipelines.experiments.run_us_microplex_source_experiments",
        fake_run_source_experiments,
    )

    report = run_us_microplex_n_synthetic_sweep(
        USMicroplexSourceExperimentSpec(
            name="cps+puf",
            providers=(_DummyProvider("cps"), _DummyProvider("puf")),
        ),
        [2000, 10000],
        tmp_path / "scale-sweep",
        metadata={"suite": "size-sweep"},
    )

    sweep_experiments = captured["experiments"]
    assert [experiment.name for experiment in sweep_experiments] == [
        "cps+puf-n2000",
        "cps+puf-n10000",
    ]
    assert captured["metadata"] == {
        "base_experiment_name": "cps+puf",
        "n_synthetic_values": [2000, 10000],
        "sweep_parameter": "n_synthetic",
        "suite": "size-sweep",
    }
    assert report.metadata["suite"] == "size-sweep"


def test_default_us_source_mix_experiments_builds_standard_ladder():
    cps_provider = _DummyProvider("cps")
    puf_provider = _DummyProvider("puf")
    psid_provider = _DummyProvider("psid")

    experiments = default_us_source_mix_experiments(
        cps_provider=cps_provider,
        puf_provider=puf_provider,
        psid_provider=psid_provider,
    )

    assert [experiment.name for experiment in experiments] == [
        "cps-only",
        "cps+puf",
        "cps+psid",
        "cps+puf+psid",
    ]
    assert experiments[0].metadata["sources"] == ["cps"]
    assert experiments[-1].metadata["sources"] == ["cps", "puf", "psid"]


def test_run_us_microplex_source_experiments_can_use_performance_session(
    monkeypatch,
    tmp_path,
):
    calls: dict[str, list[object]] = {
        "warm": [],
        "run": [],
        "save": [],
    }

    class FakeSession:
        def __init__(self):
            self.comparison_cache = object()

        def warm_parity_cache(self, *, config):
            calls["warm"].append(config)

        def run(self, providers, *, config, queries=None):
            calls["run"].append(
                {
                    "providers": tuple(provider.descriptor.name for provider in providers),
                    "config": config,
                    "queries": queries,
                }
            )
            return type(
                "FakePerformanceResult",
                (),
                {
                    "build_result": f"build:{'+'.join(provider.descriptor.name for provider in providers)}",
                    "parity_run": type(
                        "FakeParityRun",
                        (),
                        {
                            "to_dict": lambda self: {
                                "summary": {
                                    "candidate_composite_parity_loss": 0.4,
                                    "baseline_composite_parity_loss": 0.5,
                                }
                            }
                        },
                    )(),
                    "pe_native_scores": {
                        "summary": {
                            "candidate_enhanced_cps_native_loss": 0.9,
                            "baseline_enhanced_cps_native_loss": 1.1,
                            "enhanced_cps_native_loss_delta": -0.2,
                        }
                    },
                },
            )()

    def fake_save_build_result(
        build_result,
        output_root,
        *,
        frontier_metric="candidate_composite_parity_loss",
        policyengine_comparison_cache=None,
        policyengine_target_provider=None,
        policyengine_baseline_dataset=None,
        policyengine_harness_slices=None,
        policyengine_harness_metadata=None,
        precomputed_policyengine_harness_payload=None,
        defer_policyengine_harness=False,
        precomputed_policyengine_native_scores=None,
        defer_policyengine_native_score=False,
        run_registry_path=None,
        run_registry_metadata=None,
        version_id=None,
    ):
        experiment_name = run_registry_metadata["experiment_name"]
        calls["save"].append(
            {
                "build_result": build_result,
                "output_root": Path(output_root),
                "frontier_metric": frontier_metric,
                "policyengine_comparison_cache": policyengine_comparison_cache,
                "policyengine_harness_metadata": dict(policyengine_harness_metadata),
                "precomputed_policyengine_harness_payload": precomputed_policyengine_harness_payload,
                "defer_policyengine_harness": defer_policyengine_harness,
                "precomputed_policyengine_native_scores": precomputed_policyengine_native_scores,
                "defer_policyengine_native_score": defer_policyengine_native_score,
                "run_registry_metadata": dict(run_registry_metadata),
                "version_id": version_id,
            }
        )
        current_entry = _entry(
            experiment_name,
            composite_loss=0.3 if experiment_name == "cps+puf" else 0.4,
            source_names=tuple(run_registry_metadata["sources"])
            if "sources" in run_registry_metadata
            else (experiment_name,),
        )
        return type(
            "FakeArtifacts",
            (),
            {
                "artifact_paths": _artifact_paths(Path(output_root), experiment_name),
                "current_entry": current_entry,
                "frontier_entry": current_entry,
                "frontier_delta": 0.0,
            },
        )()

    monkeypatch.setattr(
        "microplex_us.pipelines.experiments.save_versioned_us_microplex_build_result",
        fake_save_build_result,
    )
    registry_entries = [
        _entry("cps-only", composite_loss=0.4, source_names=("cps",)),
        _entry("cps+puf", composite_loss=0.3, source_names=("cps", "puf")),
    ]
    monkeypatch.setattr(
        "microplex_us.pipelines.experiments.backfill_us_pe_native_scores_bundles",
        lambda artifact_dirs, **kwargs: [
            Path(path) / "manifest.json" for path in artifact_dirs
        ],
    )
    calls["backfill_native_audit"] = []
    monkeypatch.setattr(
        "microplex_us.pipelines.experiments.backfill_us_pe_native_audit_bundles",
        lambda artifact_dirs, **kwargs: calls["backfill_native_audit"].append(
            {
                "artifact_dirs": [Path(path) for path in artifact_dirs],
                "kwargs": dict(kwargs),
            }
        ),
    )
    monkeypatch.setattr(
        "microplex_us.pipelines.experiments.load_us_microplex_run_registry",
        lambda _path: registry_entries,
    )
    monkeypatch.setattr(
        "microplex_us.pipelines.experiments.select_us_microplex_frontier_entry",
        lambda _path, *, metric="candidate_composite_parity_loss": min(
            registry_entries,
            key=lambda entry: getattr(entry, metric),
        ),
    )

    session = FakeSession()
    performance_config = USMicroplexPerformanceHarnessConfig(
        sample_n=25,
        n_synthetic=25,
        targets_db="/tmp/policy_data.db",
        baseline_dataset="/tmp/baseline.h5",
        policyengine_us_data_repo="/tmp/policyengine-us-data",
        evaluate_parity=True,
        evaluate_pe_native_loss=True,
    )
    report = run_us_microplex_source_experiments(
        [
            USMicroplexSourceExperimentSpec(
                name="cps-only",
                providers=(_DummyProvider("cps"),),
                config=USMicroplexBuildConfig(n_synthetic=2000, random_seed=7),
                metadata={"sources": ["cps"]},
            ),
            USMicroplexSourceExperimentSpec(
                name="cps+puf",
                providers=(_DummyProvider("cps"), _DummyProvider("puf")),
                config=USMicroplexBuildConfig(n_synthetic=4000, random_seed=9),
                metadata={"sources": ["cps", "puf"]},
            ),
        ],
        tmp_path / "experiments",
        performance_harness_config=performance_config,
        performance_session=session,
    )

    assert report.best_result is not None
    assert report.best_result.name == "cps+puf"
    assert len(calls["warm"]) == 1
    assert len(calls["run"]) == 2
    assert len(calls["save"]) == 2
    assert calls["run"][0]["config"].evaluate_parity is False
    assert calls["run"][0]["config"].evaluate_pe_native_loss is False
    assert calls["run"][0]["config"].n_synthetic == 2000
    assert calls["run"][0]["config"].random_seed == 7
    assert calls["run"][1]["config"].n_synthetic == 4000
    assert calls["run"][1]["config"].random_seed == 9
    assert calls["save"][0]["policyengine_comparison_cache"] is session.comparison_cache
    assert calls["save"][1]["run_registry_metadata"]["experiment_name"] == "cps+puf"
    assert calls["save"][1]["policyengine_harness_metadata"]["experiment_name"] == "cps+puf"
    assert calls["save"][0]["precomputed_policyengine_harness_payload"] == {
        "summary": {
            "candidate_composite_parity_loss": 0.4,
            "baseline_composite_parity_loss": 0.5,
        }
    }
    assert calls["save"][0]["defer_policyengine_harness"] is False
    assert calls["save"][0]["precomputed_policyengine_native_scores"] is None
    assert calls["save"][0]["defer_policyengine_native_score"] is True
    assert len(calls["backfill_native_audit"]) == 1
    assert calls["backfill_native_audit"][0]["artifact_dirs"] == [
        tmp_path / "experiments" / "cps-only",
        tmp_path / "experiments" / "cps+puf",
    ]
    assert (
        calls["backfill_native_audit"][0]["kwargs"]["policyengine_us_data_repo"]
        == "/tmp/policyengine-us-data"
    )
    assert report.best_result.current_entry is not None
    assert report.best_result.current_entry.artifact_id == "cps+puf"


def test_refresh_experiment_results_from_registry_returns_original_results_when_empty(
    tmp_path,
):
    registry_path = tmp_path / "run_registry.jsonl"
    results = (
        USMicroplexExperimentResult(
            name="cps-only",
            artifact_paths=_artifact_paths(tmp_path, "cps-only"),
            frontier_metric="candidate_composite_parity_loss",
            frontier_delta=None,
        ),
    )

    loaded = _refresh_experiment_results_from_registry(
        results,
        run_registry_path=registry_path,
        frontier_metric="candidate_composite_parity_loss",
    )

    assert loaded == results


def test_refresh_experiment_results_from_registry_refreshes_backfilled_artifact_paths(
    monkeypatch,
    tmp_path,
) -> None:
    output_dir = tmp_path / "cps-only"
    output_dir.mkdir()
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "artifacts": {
                    "data_flow_snapshot": "data_flow_snapshot.json",
                    "artifact_inventory": "stage_artifacts/artifact_inventory.json",
                    "conditional_readiness": (
                        "stage_artifacts/conditional_readiness.json"
                    ),
                    "policyengine_native_scores": "policyengine_native_scores.json",
                    "policyengine_native_audit": "pe_us_data_rebuild_native_audit.json",
                }
            }
        )
    )
    for name in (
        "data_flow_snapshot.json",
        "policyengine_native_scores.json",
        "pe_us_data_rebuild_native_audit.json",
    ):
        (output_dir / name).write_text("{}")
    (output_dir / "stage_artifacts").mkdir()
    for name in ("artifact_inventory.json", "conditional_readiness.json"):
        (output_dir / "stage_artifacts" / name).write_text("{}")
    registry_path = tmp_path / "run_registry.jsonl"
    result = USMicroplexExperimentResult(
        name="cps-only",
        artifact_paths=USMicroplexArtifactPaths(
            output_dir=output_dir,
            version_id="cps-only",
            seed_data=output_dir / "seed.parquet",
            synthetic_data=output_dir / "synthetic.parquet",
            calibrated_data=output_dir / "calibrated.parquet",
            targets=output_dir / "targets.json",
            manifest=manifest_path,
            policyengine_native_scores=None,
            policyengine_native_audit=None,
            data_flow_snapshot=None,
        ),
        frontier_metric="candidate_composite_parity_loss",
        frontier_delta=None,
    )
    registry_entries = [
        _entry("cps-only", composite_loss=0.4, source_names=("cps",)),
    ]
    monkeypatch.setattr(
        "microplex_us.pipelines.experiments.load_us_microplex_run_registry",
        lambda _path: registry_entries,
    )
    monkeypatch.setattr(
        "microplex_us.pipelines.experiments.select_us_microplex_frontier_entry",
        lambda _path, *, metric="candidate_composite_parity_loss": registry_entries[0]
    )
    loaded = _refresh_experiment_results_from_registry(
        (result,),
        run_registry_path=registry_path,
        frontier_metric="candidate_composite_parity_loss",
    )

    assert loaded[0].artifact_paths.data_flow_snapshot == output_dir / "data_flow_snapshot.json"
    assert loaded[0].artifact_paths.artifact_inventory == (
        output_dir / "stage_artifacts" / "artifact_inventory.json"
    )
    assert loaded[0].artifact_paths.conditional_readiness == (
        output_dir / "stage_artifacts" / "conditional_readiness.json"
    )
    assert (
        loaded[0].artifact_paths.policyengine_native_scores
        == output_dir / "policyengine_native_scores.json"
    )
    assert (
        loaded[0].artifact_paths.policyengine_native_audit
        == output_dir / "pe_us_data_rebuild_native_audit.json"
    )


def test_run_us_microplex_source_experiments_requires_performance_config_for_session(
    tmp_path,
):
    class FakeSession:
        comparison_cache = object()

    try:
        run_us_microplex_source_experiments(
            [
                USMicroplexSourceExperimentSpec(
                    name="cps-only",
                    providers=(_DummyProvider("cps"),),
                )
            ],
            tmp_path / "experiments",
            performance_session=FakeSession(),
        )
    except ValueError as exc:
        assert "performance_harness_config is required" in str(exc)
    else:
        raise AssertionError("Expected ValueError when session is provided without config")
