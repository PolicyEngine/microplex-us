"""Experiment runners for PE-US parity optimization."""

from __future__ import annotations

import json
from dataclasses import dataclass, field, replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from microplex.core import SourceProvider, SourceQuery

from microplex_us.pipelines.artifacts import (
    USMicroplexArtifactPaths,
    build_and_save_versioned_us_microplex_from_source_providers,
    save_versioned_us_microplex_build_result,
)
from microplex_us.pipelines.backfill_pe_native_audit import (
    backfill_us_pe_native_audit_bundles,
)
from microplex_us.pipelines.backfill_pe_native_scores import (
    backfill_us_pe_native_scores_bundles,
)
from microplex_us.pipelines.performance import (
    USMicroplexPerformanceHarnessConfig,
    USMicroplexPerformanceSession,
)
from microplex_us.pipelines.registry import (
    FrontierMetric,
    USMicroplexRunRegistryEntry,
    load_us_microplex_run_registry,
    select_us_microplex_frontier_entry,
)
from microplex_us.pipelines.us import USMicroplexBuildConfig
from microplex_us.policyengine.harness import (
    PolicyEngineUSComparisonCache,
    PolicyEngineUSHarnessSlice,
)


@dataclass(frozen=True)
class USMicroplexSourceExperimentSpec:
    """One named source-mix experiment to run through the PE-US parity harness."""

    name: str
    providers: tuple[SourceProvider, ...]
    config: USMicroplexBuildConfig | None = None
    queries: dict[str, SourceQuery] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)


def default_us_source_mix_experiments(
    *,
    cps_provider: SourceProvider,
    base_config: USMicroplexBuildConfig | None = None,
    cps_query: SourceQuery | None = None,
    puf_provider: SourceProvider | None = None,
    puf_query: SourceQuery | None = None,
    psid_provider: SourceProvider | None = None,
    psid_query: SourceQuery | None = None,
) -> tuple[USMicroplexSourceExperimentSpec, ...]:
    """Build a standard ladder of US source-mix experiments."""
    experiments = [
        USMicroplexSourceExperimentSpec(
            name="cps-only",
            providers=(cps_provider,),
            config=base_config,
            queries=(
                {cps_provider.descriptor.name: cps_query}
                if cps_query is not None
                else {}
            ),
            metadata={"sources": [cps_provider.descriptor.name]},
        )
    ]

    if puf_provider is not None:
        experiments.append(
            USMicroplexSourceExperimentSpec(
                name="cps+puf",
                providers=(cps_provider, puf_provider),
                config=base_config,
                queries={
                    **(
                        {cps_provider.descriptor.name: cps_query}
                        if cps_query is not None
                        else {}
                    ),
                    **(
                        {puf_provider.descriptor.name: puf_query}
                        if puf_query is not None
                        else {}
                    ),
                },
                metadata={
                    "sources": [
                        cps_provider.descriptor.name,
                        puf_provider.descriptor.name,
                    ]
                },
            )
        )

    if psid_provider is not None:
        experiments.append(
            USMicroplexSourceExperimentSpec(
                name="cps+psid",
                providers=(cps_provider, psid_provider),
                config=base_config,
                queries={
                    **(
                        {cps_provider.descriptor.name: cps_query}
                        if cps_query is not None
                        else {}
                    ),
                    **(
                        {psid_provider.descriptor.name: psid_query}
                        if psid_query is not None
                        else {}
                    ),
                },
                metadata={
                    "sources": [
                        cps_provider.descriptor.name,
                        psid_provider.descriptor.name,
                    ]
                },
            )
        )

    if puf_provider is not None and psid_provider is not None:
        experiments.append(
            USMicroplexSourceExperimentSpec(
                name="cps+puf+psid",
                providers=(cps_provider, puf_provider, psid_provider),
                config=base_config,
                queries={
                    **(
                        {cps_provider.descriptor.name: cps_query}
                        if cps_query is not None
                        else {}
                    ),
                    **(
                        {puf_provider.descriptor.name: puf_query}
                        if puf_query is not None
                        else {}
                    ),
                    **(
                        {psid_provider.descriptor.name: psid_query}
                        if psid_query is not None
                        else {}
                    ),
                },
                metadata={
                    "sources": [
                        cps_provider.descriptor.name,
                        puf_provider.descriptor.name,
                        psid_provider.descriptor.name,
                    ]
                },
            )
        )

    return tuple(experiments)


def build_us_n_synthetic_sweep_experiments(
    experiment: USMicroplexSourceExperimentSpec,
    n_synthetic_values: tuple[int, ...] | list[int],
    *,
    name_template: str = "{base_name}-n{n_synthetic}",
) -> tuple[USMicroplexSourceExperimentSpec, ...]:
    """Expand one experiment into a deterministic n_synthetic sweep."""
    if not n_synthetic_values:
        raise ValueError(
            "build_us_n_synthetic_sweep_experiments requires at least one n_synthetic value"
        )

    base_config = experiment.config or USMicroplexBuildConfig()
    seen_values: set[int] = set()
    sweep_experiments: list[USMicroplexSourceExperimentSpec] = []
    for raw_value in n_synthetic_values:
        n_synthetic = int(raw_value)
        if n_synthetic <= 0:
            raise ValueError("n_synthetic sweep values must be positive integers")
        if n_synthetic in seen_values:
            raise ValueError(
                f"Duplicate n_synthetic sweep value supplied: {n_synthetic}"
            )
        seen_values.add(n_synthetic)
        sweep_experiments.append(
            USMicroplexSourceExperimentSpec(
                name=name_template.format(
                    base_name=experiment.name,
                    n_synthetic=n_synthetic,
                ),
                providers=experiment.providers,
                config=replace(base_config, n_synthetic=n_synthetic),
                queries=dict(experiment.queries),
                metadata={
                    **dict(experiment.metadata),
                    "base_experiment_name": experiment.name,
                    "n_synthetic": n_synthetic,
                    "sweep_parameter": "n_synthetic",
                },
            )
        )
    return tuple(sweep_experiments)


@dataclass(frozen=True)
class USMicroplexExperimentResult:
    """Persistable summary for one completed source-mix experiment."""

    name: str
    artifact_paths: USMicroplexArtifactPaths
    frontier_metric: FrontierMetric
    frontier_delta: float | None
    current_entry: USMicroplexRunRegistryEntry | None = None
    frontier_entry: USMicroplexRunRegistryEntry | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def metric_value(self) -> float | None:
        if self.current_entry is None:
            return None
        return getattr(self.current_entry, self.frontier_metric, None)

    @property
    def source_names(self) -> tuple[str, ...]:
        if self.current_entry is not None and self.current_entry.source_names:
            return self.current_entry.source_names
        return ()

    def to_dict(self) -> dict[str, Any]:
        """Serialize the experiment result to a JSON-compatible payload."""
        return {
            "name": self.name,
            "artifact_paths": {
                "output_dir": str(self.artifact_paths.output_dir),
                "seed_data": str(self.artifact_paths.seed_data),
                "synthetic_data": str(self.artifact_paths.synthetic_data),
                "calibrated_data": str(self.artifact_paths.calibrated_data),
                "targets": str(self.artifact_paths.targets),
                "manifest": str(self.artifact_paths.manifest),
                "version_id": self.artifact_paths.version_id,
                "synthesizer": (
                    str(self.artifact_paths.synthesizer)
                    if self.artifact_paths.synthesizer is not None
                    else None
                ),
                "policyengine_dataset": (
                    str(self.artifact_paths.policyengine_dataset)
                    if self.artifact_paths.policyengine_dataset is not None
                    else None
                ),
                "data_flow_snapshot": (
                    str(self.artifact_paths.data_flow_snapshot)
                    if self.artifact_paths.data_flow_snapshot is not None
                    else None
                ),
                "policyengine_harness": (
                    str(self.artifact_paths.policyengine_harness)
                    if self.artifact_paths.policyengine_harness is not None
                    else None
                ),
                "policyengine_native_scores": (
                    str(self.artifact_paths.policyengine_native_scores)
                    if self.artifact_paths.policyengine_native_scores is not None
                    else None
                ),
                "policyengine_native_audit": (
                    str(self.artifact_paths.policyengine_native_audit)
                    if self.artifact_paths.policyengine_native_audit is not None
                    else None
                ),
                "capital_gains_lots": (
                    str(self.artifact_paths.capital_gains_lots)
                    if self.artifact_paths.capital_gains_lots is not None
                    else None
                ),
                "run_registry": (
                    str(self.artifact_paths.run_registry)
                    if self.artifact_paths.run_registry is not None
                    else None
                ),
                "run_index_db": (
                    str(self.artifact_paths.run_index_db)
                    if self.artifact_paths.run_index_db is not None
                    else None
                ),
            },
            "frontier_metric": self.frontier_metric,
            "frontier_delta": self.frontier_delta,
            "metric_value": self.metric_value,
            "source_names": list(self.source_names),
            "current_entry": (
                self.current_entry.to_dict() if self.current_entry is not None else None
            ),
            "frontier_entry": (
                self.frontier_entry.to_dict() if self.frontier_entry is not None else None
            ),
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> USMicroplexExperimentResult:
        """Restore an experiment result from serialized JSON payload."""
        artifact_paths = payload["artifact_paths"]
        return cls(
            name=payload["name"],
            artifact_paths=USMicroplexArtifactPaths(
                output_dir=Path(artifact_paths["output_dir"]),
                seed_data=Path(artifact_paths["seed_data"]),
                synthetic_data=Path(artifact_paths["synthetic_data"]),
                calibrated_data=Path(artifact_paths["calibrated_data"]),
                targets=Path(artifact_paths["targets"]),
                manifest=Path(artifact_paths["manifest"]),
                version_id=artifact_paths.get("version_id"),
                synthesizer=(
                    Path(artifact_paths["synthesizer"])
                    if artifact_paths.get("synthesizer") is not None
                    else None
                ),
                policyengine_dataset=(
                    Path(artifact_paths["policyengine_dataset"])
                    if artifact_paths.get("policyengine_dataset") is not None
                    else None
                ),
                data_flow_snapshot=(
                    Path(artifact_paths["data_flow_snapshot"])
                    if artifact_paths.get("data_flow_snapshot") is not None
                    else None
                ),
                policyengine_harness=(
                    Path(artifact_paths["policyengine_harness"])
                    if artifact_paths.get("policyengine_harness") is not None
                    else None
                ),
                policyengine_native_scores=(
                    Path(artifact_paths["policyengine_native_scores"])
                    if artifact_paths.get("policyengine_native_scores") is not None
                    else None
                ),
                policyengine_native_audit=(
                    Path(artifact_paths["policyengine_native_audit"])
                    if artifact_paths.get("policyengine_native_audit") is not None
                    else None
                ),
                capital_gains_lots=(
                    Path(artifact_paths["capital_gains_lots"])
                    if artifact_paths.get("capital_gains_lots") is not None
                    else None
                ),
                run_registry=(
                    Path(artifact_paths["run_registry"])
                    if artifact_paths.get("run_registry") is not None
                    else None
                ),
                run_index_db=(
                    Path(artifact_paths["run_index_db"])
                    if artifact_paths.get("run_index_db") is not None
                    else None
                ),
            ),
            frontier_metric=payload["frontier_metric"],
            frontier_delta=payload.get("frontier_delta"),
            current_entry=(
                USMicroplexRunRegistryEntry.from_dict(payload["current_entry"])
                if payload.get("current_entry") is not None
                else None
            ),
            frontier_entry=(
                USMicroplexRunRegistryEntry.from_dict(payload["frontier_entry"])
                if payload.get("frontier_entry") is not None
                else None
            ),
            metadata=dict(payload.get("metadata", {})),
        )


@dataclass(frozen=True)
class USMicroplexExperimentReport:
    """Persistable report for a batch of source-mix experiments."""

    output_root: Path
    frontier_metric: FrontierMetric
    results: tuple[USMicroplexExperimentResult, ...]
    created_at: str = field(
        default_factory=lambda: datetime.now(UTC).replace(microsecond=0).isoformat()
    )
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def leaderboard(self) -> tuple[USMicroplexExperimentResult, ...]:
        """Return results sorted by the configured frontier metric."""

        def sort_key(
            result: USMicroplexExperimentResult,
        ) -> tuple[bool, float, str]:
            metric_value = result.metric_value
            if metric_value is None:
                return (True, float("inf"), result.name)
            return (False, metric_value, result.name)

        return tuple(sorted(self.results, key=sort_key))

    @property
    def best_result(self) -> USMicroplexExperimentResult | None:
        leaderboard = self.leaderboard
        if not leaderboard:
            return None
        if leaderboard[0].metric_value is None:
            return None
        return leaderboard[0]

    def to_dict(self) -> dict[str, Any]:
        """Serialize the report to a JSON-compatible dict."""
        best_result = self.best_result
        return {
            "created_at": self.created_at,
            "output_root": str(self.output_root),
            "frontier_metric": self.frontier_metric,
            "summary": {
                "best_experiment": best_result.name if best_result is not None else None,
                "best_metric_value": (
                    best_result.metric_value if best_result is not None else None
                ),
                "n_results": len(self.results),
            },
            "metadata": dict(self.metadata),
            "results": [result.to_dict() for result in self.results],
        }

    def save(self, path: str | Path) -> Path:
        """Persist the experiment report to disk."""
        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(self.to_dict(), indent=2, sort_keys=True))
        return output_path

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> USMicroplexExperimentReport:
        """Restore a report from serialized JSON."""
        return cls(
            output_root=Path(payload["output_root"]),
            frontier_metric=payload["frontier_metric"],
            results=tuple(
                USMicroplexExperimentResult.from_dict(result)
                for result in payload.get("results", [])
            ),
            created_at=payload["created_at"],
            metadata=dict(payload.get("metadata", {})),
        )

    @classmethod
    def load(cls, path: str | Path) -> USMicroplexExperimentReport:
        """Load a persisted experiment report."""
        return cls.from_dict(json.loads(Path(path).read_text()))


def run_us_microplex_source_experiments(
    experiments: list[USMicroplexSourceExperimentSpec]
    | tuple[USMicroplexSourceExperimentSpec, ...],
    output_root: str | Path,
    *,
    frontier_metric: FrontierMetric = "candidate_composite_parity_loss",
    policyengine_target_provider: Any | None = None,
    policyengine_baseline_dataset: str | Path | None = None,
    policyengine_comparison_cache: PolicyEngineUSComparisonCache | None = None,
    policyengine_harness_slices: (
        tuple[PolicyEngineUSHarnessSlice, ...] | list[PolicyEngineUSHarnessSlice] | None
    ) = None,
    policyengine_harness_metadata: dict[str, Any] | None = None,
    run_registry_path: str | Path | None = None,
    report_path: str | Path | None = None,
    performance_harness_config: USMicroplexPerformanceHarnessConfig | None = None,
    performance_session: USMicroplexPerformanceSession | None = None,
    metadata: dict[str, Any] | None = None,
) -> USMicroplexExperimentReport:
    """Run a batch of source-mix experiments through the versioned PE-US build loop."""
    if not experiments:
        raise ValueError("run_us_microplex_source_experiments requires at least one experiment")
    if performance_session is not None and performance_harness_config is None:
        raise ValueError(
            "performance_harness_config is required when providing performance_session"
        )

    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    results: list[USMicroplexExperimentResult] = []
    active_performance_session = performance_session
    if performance_harness_config is not None and active_performance_session is None:
        active_performance_session = USMicroplexPerformanceSession()

    shared_comparison_cache = (
        policyengine_comparison_cache
        or (
            active_performance_session.comparison_cache
            if active_performance_session is not None
            else None
        )
        or PolicyEngineUSComparisonCache()
    )
    if (
        active_performance_session is not None
        and performance_harness_config is not None
        and performance_harness_config.targets_db is not None
        and performance_harness_config.baseline_dataset is not None
    ):
        active_performance_session.warm_parity_cache(config=performance_harness_config)
    batch_native_scoring = (
        active_performance_session is not None
        and performance_harness_config is not None
        and performance_harness_config.evaluate_pe_native_loss
        and len(experiments) > 1
    )

    for experiment in experiments:
        harness_metadata = {
            "experiment_name": experiment.name,
            **dict(policyengine_harness_metadata or {}),
            **dict(experiment.metadata),
        }
        registry_metadata = {
            "experiment_name": experiment.name,
            **dict(experiment.metadata),
        }
        if (
            active_performance_session is not None
            and performance_harness_config is not None
        ):
            harness_config = _resolve_experiment_performance_config(
                experiment,
                performance_harness_config,
            )
            if batch_native_scoring:
                harness_config = replace(harness_config, evaluate_pe_native_loss=False)
            performance_result = active_performance_session.run(
                list(experiment.providers),
                config=harness_config,
                queries=experiment.queries or None,
            )
            artifacts = save_versioned_us_microplex_build_result(
                performance_result.build_result,
                output_root,
                frontier_metric=frontier_metric,
                policyengine_comparison_cache=shared_comparison_cache,
                policyengine_target_provider=policyengine_target_provider,
                policyengine_baseline_dataset=policyengine_baseline_dataset,
                policyengine_harness_slices=policyengine_harness_slices,
                policyengine_harness_metadata=harness_metadata,
                precomputed_policyengine_harness_payload=(
                    performance_result.parity_run.to_dict()
                    if performance_result.parity_run is not None
                    else None
                ),
                defer_policyengine_harness=performance_result.parity_run is None,
                precomputed_policyengine_native_scores=(
                    None if batch_native_scoring else performance_result.pe_native_scores
                ),
                defer_policyengine_native_score=batch_native_scoring,
                run_registry_path=run_registry_path,
                run_registry_metadata=registry_metadata,
            )
        else:
            artifacts = build_and_save_versioned_us_microplex_from_source_providers(
                list(experiment.providers),
                output_root,
                config=experiment.config,
                queries=experiment.queries or None,
                frontier_metric=frontier_metric,
                policyengine_comparison_cache=shared_comparison_cache,
                policyengine_target_provider=policyengine_target_provider,
                policyengine_baseline_dataset=policyengine_baseline_dataset,
                policyengine_harness_slices=policyengine_harness_slices,
                policyengine_harness_metadata=harness_metadata,
                run_registry_path=run_registry_path,
                run_registry_metadata=registry_metadata,
            )
        results.append(
            USMicroplexExperimentResult(
                name=experiment.name,
                artifact_paths=artifacts.artifact_paths,
                frontier_metric=frontier_metric,
                frontier_delta=artifacts.frontier_delta,
                current_entry=artifacts.current_entry,
                frontier_entry=artifacts.frontier_entry,
                metadata=dict(experiment.metadata),
            )
        )

    resolved_run_registry_path = Path(run_registry_path or output_root / "run_registry.jsonl")
    if batch_native_scoring:
        backfill_us_pe_native_scores_bundles(
            [result.artifact_paths.output_dir for result in results],
            baseline_dataset=performance_harness_config.baseline_dataset,
            policyengine_us_data_repo=performance_harness_config.policyengine_us_data_repo,
            rebuild_registry=True,
        )
        backfill_us_pe_native_audit_bundles(
            [result.artifact_paths.output_dir for result in results],
            policyengine_us_data_repo=performance_harness_config.policyengine_us_data_repo,
        )
        results = list(
            _refresh_experiment_results_from_registry(
                results,
                run_registry_path=resolved_run_registry_path,
                frontier_metric=frontier_metric,
            )
        )

    report = USMicroplexExperimentReport(
        output_root=output_root,
        frontier_metric=frontier_metric,
        results=tuple(results),
        metadata=dict(metadata or {}),
    )
    report.save(report_path or output_root / "experiment_report.json")
    return report


def run_us_microplex_n_synthetic_sweep(
    experiment: USMicroplexSourceExperimentSpec,
    n_synthetic_values: tuple[int, ...] | list[int],
    output_root: str | Path,
    *,
    name_template: str = "{base_name}-n{n_synthetic}",
    frontier_metric: FrontierMetric = "candidate_composite_parity_loss",
    policyengine_target_provider: Any | None = None,
    policyengine_baseline_dataset: str | Path | None = None,
    policyengine_comparison_cache: PolicyEngineUSComparisonCache | None = None,
    policyengine_harness_slices: (
        tuple[PolicyEngineUSHarnessSlice, ...] | list[PolicyEngineUSHarnessSlice] | None
    ) = None,
    policyengine_harness_metadata: dict[str, Any] | None = None,
    run_registry_path: str | Path | None = None,
    report_path: str | Path | None = None,
    performance_harness_config: USMicroplexPerformanceHarnessConfig | None = None,
    performance_session: USMicroplexPerformanceSession | None = None,
    metadata: dict[str, Any] | None = None,
) -> USMicroplexExperimentReport:
    """Run one base experiment across multiple n_synthetic values."""
    sweep_experiments = build_us_n_synthetic_sweep_experiments(
        experiment,
        n_synthetic_values,
        name_template=name_template,
    )
    sweep_values = [spec.metadata["n_synthetic"] for spec in sweep_experiments]
    return run_us_microplex_source_experiments(
        sweep_experiments,
        output_root,
        frontier_metric=frontier_metric,
        policyengine_target_provider=policyengine_target_provider,
        policyengine_baseline_dataset=policyengine_baseline_dataset,
        policyengine_comparison_cache=policyengine_comparison_cache,
        policyengine_harness_slices=policyengine_harness_slices,
        policyengine_harness_metadata=policyengine_harness_metadata,
        run_registry_path=run_registry_path,
        report_path=report_path,
        performance_harness_config=performance_harness_config,
        performance_session=performance_session,
        metadata={
            "base_experiment_name": experiment.name,
            "n_synthetic_values": sweep_values,
            "sweep_parameter": "n_synthetic",
            **dict(metadata or {}),
        },
    )


def _resolve_experiment_performance_config(
    experiment: USMicroplexSourceExperimentSpec,
    base_config: USMicroplexPerformanceHarnessConfig,
) -> USMicroplexPerformanceHarnessConfig:
    build_config = experiment.config or base_config.build_config
    resolved = replace(
        base_config,
        build_config=build_config,
        evaluate_parity=False,
    )
    if build_config is None:
        return resolved
    return replace(
        resolved,
        n_synthetic=build_config.n_synthetic,
        random_seed=build_config.random_seed,
    )


def _refresh_experiment_results_from_registry(
    results: list[USMicroplexExperimentResult] | tuple[USMicroplexExperimentResult, ...],
    *,
    run_registry_path: str | Path,
    frontier_metric: FrontierMetric,
) -> tuple[USMicroplexExperimentResult, ...]:
    registry_entries = load_us_microplex_run_registry(run_registry_path)
    if not registry_entries:
        return tuple(results)

    frontier_entry = select_us_microplex_frontier_entry(
        run_registry_path,
        metric=frontier_metric,
    )
    entries_by_artifact_id = {entry.artifact_id: entry for entry in registry_entries}
    run_index_path = Path(run_registry_path).parent / "run_index.duckdb"

    refreshed: list[USMicroplexExperimentResult] = []
    for result in results:
        version_id = result.artifact_paths.version_id or result.artifact_paths.output_dir.name
        current_entry = entries_by_artifact_id.get(version_id)
        current_value = (
            getattr(current_entry, frontier_metric, None)
            if current_entry is not None
            else None
        )
        frontier_value = (
            getattr(frontier_entry, frontier_metric, None)
            if frontier_entry is not None
            else None
        )
        frontier_delta = (
            current_value - frontier_value
            if current_value is not None and frontier_value is not None
            else None
        )
        refreshed.append(
            replace(
                result,
                artifact_paths=_refresh_experiment_artifact_paths(
                    result.artifact_paths,
                    run_registry_path=Path(run_registry_path),
                    run_index_path=run_index_path,
                ),
                current_entry=current_entry,
                frontier_entry=frontier_entry,
                frontier_delta=frontier_delta,
            )
        )
    return tuple(refreshed)


def _refresh_experiment_artifact_paths(
    artifact_paths: USMicroplexArtifactPaths,
    *,
    run_registry_path: Path,
    run_index_path: Path,
) -> USMicroplexArtifactPaths:
    manifest_payload = _load_optional_json(artifact_paths.manifest)
    artifacts = dict(manifest_payload.get("artifacts", {})) if manifest_payload else {}
    artifact_root = artifact_paths.output_dir
    return replace(
        artifact_paths,
        data_flow_snapshot=_resolve_optional_result_artifact_path(
            artifact_root,
            artifacts.get("data_flow_snapshot"),
            fallback="data_flow_snapshot.json",
        ),
        policyengine_harness=_resolve_optional_result_artifact_path(
            artifact_root,
            artifacts.get("policyengine_harness"),
        ),
        policyengine_native_scores=_resolve_optional_result_artifact_path(
            artifact_root,
            artifacts.get("policyengine_native_scores"),
        ),
        policyengine_native_audit=_resolve_optional_result_artifact_path(
            artifact_root,
            artifacts.get("policyengine_native_audit"),
        ),
        run_registry=Path(run_registry_path),
        run_index_db=run_index_path,
    )


def _resolve_optional_result_artifact_path(
    artifact_root: Path,
    artifact_name: str | None,
    *,
    fallback: str | None = None,
) -> Path | None:
    if artifact_name:
        path = artifact_root / artifact_name
        return path if path.exists() else None
    if fallback is None:
        return None
    fallback_path = artifact_root / fallback
    return fallback_path if fallback_path.exists() else None


def _load_optional_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    return json.loads(path.read_text())
