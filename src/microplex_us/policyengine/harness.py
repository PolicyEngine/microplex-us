"""Persistent comparison harness for PE-US target slices."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
from microplex.targets import (
    BatchBenchmarkResultEvaluator,
    BenchmarkResult,
    BenchmarkSliceComparison,
    BenchmarkSliceSpec,
    BenchmarkSuiteResult,
    FilterOperator,
    TargetFilter,
    TargetProvider,
    TargetQuery,
    TargetSet,
    build_benchmark_suite_from_results,
    build_benchmark_suite_result,
    evaluate_benchmark_slice_results,
    filter_nonempty_benchmark_slices,
    load_benchmark_slice_target_sets,
    union_target_sets,
)

from microplex_us.pipeline_metadata import pipeline_node
from microplex_us.policyengine.comparison import (
    POLICYENGINE_US_BENCHMARK_GROUP_FIELDS,
    PolicyEngineUSComparisonCache,
    PolicyEngineUSTargetComparisonReport,
    PolicyEngineUSTargetEvaluation,
    PolicyEngineUSTargetEvaluationReport,
    evaluate_policyengine_us_target_set,
    evaluate_policyengine_us_target_sets,
    slice_policyengine_us_target_evaluation_report,
)
from microplex_us.policyengine.us import (
    PolicyEngineUSEntityTableBundle,
    load_policyengine_us_entity_tables,
)

COMPOSITE_PARITY_LOSS_WEIGHTS = {
    "micro": 0.35,
    "attribute_macro": 0.35,
    "attribute_tail": 0.20,
    "support_gap": 0.10,
}
ATTRIBUTE_TAIL_FRACTION = 0.10
UNSPECIFIED_ATTRIBUTE = "__unknown__"


PolicyEngineUSHarnessSlice = BenchmarkSliceSpec


@dataclass
class PolicyEngineUSHarnessSliceResult:
    """Comparison result for one named harness slice."""

    slice: PolicyEngineUSHarnessSlice
    comparison: PolicyEngineUSTargetComparisonReport

    @property
    def candidate_mean_abs_relative_error(self) -> float | None:
        return self.comparison.candidate.mean_abs_relative_error

    @property
    def baseline_mean_abs_relative_error(self) -> float | None:
        if self.comparison.baseline is None:
            return None
        return self.comparison.baseline.mean_abs_relative_error

    @property
    def mean_abs_relative_error_delta(self) -> float | None:
        return self.comparison.mean_abs_relative_error_delta

    @property
    def candidate_beats_baseline(self) -> bool | None:
        delta = self.mean_abs_relative_error_delta
        if delta is None:
            return None
        return delta < 0.0

    @property
    def benchmark_slice_result(self) -> BenchmarkSliceComparison | None:
        comparison = self.comparison.benchmark_comparison
        if comparison is None:
            return None
        return BenchmarkSliceComparison(
            slice=BenchmarkSliceSpec(
                name=self.slice.name,
                query=self.slice.query,
                description=self.slice.description,
                tags=self.slice.tags,
            ),
            comparison=comparison,
        )


@dataclass
class PolicyEngineUSHarnessRun:
    """Persistent PE-US harness evaluation across multiple target slices."""

    candidate_label: str
    baseline_label: str
    period: int | str
    slice_results: list[PolicyEngineUSHarnessSliceResult] = field(default_factory=list)
    created_at: str = field(
        default_factory=lambda: datetime.now(UTC).replace(microsecond=0).isoformat()
    )
    metadata: dict[str, Any] = field(default_factory=dict)
    _benchmark_suite: BenchmarkSuiteResult | None = field(default=None, repr=False)

    @property
    def benchmark_suite(self) -> BenchmarkSuiteResult:
        if self._benchmark_suite is not None:
            return self._benchmark_suite
        comparable_results = [
            result for result in self.slice_results if result.comparison.baseline is not None
        ]
        return build_benchmark_suite_from_results(
            candidate_label=self.candidate_label,
            baseline_label=self.baseline_label,
            period=self.period,
            slices=[result.slice for result in comparable_results],
            candidate_results={
                result.slice.name: result.comparison.candidate.benchmark_result
                for result in comparable_results
            },
            baseline_results={
                result.slice.name: result.comparison.baseline.benchmark_result
                for result in comparable_results
                if result.comparison.baseline is not None
            },
            group_fields=POLICYENGINE_US_BENCHMARK_GROUP_FIELDS,
            created_at=self.created_at,
            metadata=dict(self.metadata),
        )

    @property
    def candidate_mean_abs_relative_error(self) -> float | None:
        return self.benchmark_suite.candidate_mean_abs_relative_error

    @property
    def baseline_mean_abs_relative_error(self) -> float | None:
        return self.benchmark_suite.baseline_mean_abs_relative_error

    @property
    def mean_abs_relative_error_delta(self) -> float | None:
        return self.benchmark_suite.mean_abs_relative_error_delta

    @property
    def slice_win_rate(self) -> float | None:
        return self.benchmark_suite.slice_win_rate

    @property
    def target_win_rate(self) -> float | None:
        return self.benchmark_suite.target_win_rate

    @property
    def supported_target_rate(self) -> float | None:
        return self.benchmark_suite.supported_target_rate

    @property
    def baseline_supported_target_rate(self) -> float | None:
        return self.benchmark_suite.baseline_supported_target_rate

    @property
    def candidate_micro_mean_abs_relative_error(self) -> float | None:
        return self.candidate_micro_mean_abs_relative_error_for_tag(None)

    @property
    def baseline_micro_mean_abs_relative_error(self) -> float | None:
        return self.baseline_micro_mean_abs_relative_error_for_tag(None)

    @property
    def candidate_attribute_macro_mean_abs_relative_error(self) -> float | None:
        return self.candidate_attribute_macro_mean_abs_relative_error_for_tag(None)

    @property
    def baseline_attribute_macro_mean_abs_relative_error(self) -> float | None:
        return self.baseline_attribute_macro_mean_abs_relative_error_for_tag(None)

    @property
    def candidate_attribute_tail_mean_abs_relative_error(self) -> float | None:
        return self.candidate_attribute_tail_mean_abs_relative_error_for_tag(None)

    @property
    def baseline_attribute_tail_mean_abs_relative_error(self) -> float | None:
        return self.baseline_attribute_tail_mean_abs_relative_error_for_tag(None)

    @property
    def candidate_composite_parity_loss(self) -> float | None:
        return self.candidate_composite_parity_loss_for_tag(None)

    @property
    def baseline_composite_parity_loss(self) -> float | None:
        return self.baseline_composite_parity_loss_for_tag(None)

    @property
    def composite_parity_loss_delta(self) -> float | None:
        candidate_loss = self.candidate_composite_parity_loss
        baseline_loss = self.baseline_composite_parity_loss
        if candidate_loss is None or baseline_loss is None:
            return None
        return candidate_loss - baseline_loss

    @property
    def attribute_cell_summaries(self) -> dict[str, dict[str, Any]]:
        return self.attribute_cell_summaries_for_tag(None)

    @property
    def tag_summaries(self) -> dict[str, dict[str, float | None]]:
        tags = tuple(
            dict.fromkeys(
                tag
                for result in self.slice_results
                for tag in result.slice.tags
            )
        )
        return {
            tag: {
                "candidate_mean_abs_relative_error": self.candidate_mean_abs_relative_error_for_tag(
                    tag
                ),
                "baseline_mean_abs_relative_error": self.baseline_mean_abs_relative_error_for_tag(
                    tag
                ),
                "mean_abs_relative_error_delta": self.mean_abs_relative_error_delta_for_tag(
                    tag
                ),
                "slice_win_rate": self.slice_win_rate_for_tag(tag),
                "target_win_rate": self.target_win_rate_for_tag(tag),
                "supported_target_rate": self.supported_target_rate_for_tag(tag),
                "baseline_supported_target_rate": self._supported_target_rate_for_tag(
                    tag,
                    kind="baseline",
                ),
                "candidate_micro_mean_abs_relative_error": self.candidate_micro_mean_abs_relative_error_for_tag(
                    tag
                ),
                "baseline_micro_mean_abs_relative_error": self.baseline_micro_mean_abs_relative_error_for_tag(
                    tag
                ),
                "candidate_attribute_macro_mean_abs_relative_error": self.candidate_attribute_macro_mean_abs_relative_error_for_tag(
                    tag
                ),
                "baseline_attribute_macro_mean_abs_relative_error": self.baseline_attribute_macro_mean_abs_relative_error_for_tag(
                    tag
                ),
                "candidate_attribute_tail_mean_abs_relative_error": self.candidate_attribute_tail_mean_abs_relative_error_for_tag(
                    tag
                ),
                "baseline_attribute_tail_mean_abs_relative_error": self.baseline_attribute_tail_mean_abs_relative_error_for_tag(
                    tag
                ),
                "candidate_composite_parity_loss": self.candidate_composite_parity_loss_for_tag(
                    tag
                ),
                "baseline_composite_parity_loss": self.baseline_composite_parity_loss_for_tag(
                    tag
                ),
                "composite_parity_loss_delta": self.composite_parity_loss_delta_for_tag(
                    tag
                ),
            }
            for tag in tags
        }

    @property
    def parity_scorecard(self) -> dict[str, dict[str, float | bool | None]]:
        scopes = {
            "overall": None,
            "national": "national",
            "local": "local",
            "state": "state",
            "district": "district",
        }
        scorecard: dict[str, dict[str, float | bool | None]] = {}
        for scope, tag in scopes.items():
            if tag is not None and not self._slice_results_for_tag(tag):
                continue
            scorecard[scope] = {
                "candidate_mean_abs_relative_error": self.candidate_mean_abs_relative_error_for_tag(
                    tag
                ),
                "baseline_mean_abs_relative_error": self.baseline_mean_abs_relative_error_for_tag(
                    tag
                ),
                "mean_abs_relative_error_delta": self.mean_abs_relative_error_delta_for_tag(
                    tag
                ),
                "slice_win_rate": self.slice_win_rate_for_tag(tag),
                "target_win_rate": self.target_win_rate_for_tag(tag),
                "supported_target_rate": self.supported_target_rate_for_tag(tag),
                "baseline_supported_target_rate": self._supported_target_rate_for_tag(
                    tag,
                    kind="baseline",
                ),
                "candidate_micro_mean_abs_relative_error": self.candidate_micro_mean_abs_relative_error_for_tag(
                    tag
                ),
                "baseline_micro_mean_abs_relative_error": self.baseline_micro_mean_abs_relative_error_for_tag(
                    tag
                ),
                "candidate_attribute_macro_mean_abs_relative_error": self.candidate_attribute_macro_mean_abs_relative_error_for_tag(
                    tag
                ),
                "baseline_attribute_macro_mean_abs_relative_error": self.baseline_attribute_macro_mean_abs_relative_error_for_tag(
                    tag
                ),
                "candidate_attribute_tail_mean_abs_relative_error": self.candidate_attribute_tail_mean_abs_relative_error_for_tag(
                    tag
                ),
                "baseline_attribute_tail_mean_abs_relative_error": self.baseline_attribute_tail_mean_abs_relative_error_for_tag(
                    tag
                ),
                "candidate_composite_parity_loss": self.candidate_composite_parity_loss_for_tag(
                    tag
                ),
                "baseline_composite_parity_loss": self.baseline_composite_parity_loss_for_tag(
                    tag
                ),
                "composite_parity_loss_delta": self.composite_parity_loss_delta_for_tag(
                    tag
                ),
                "candidate_beats_baseline": self._candidate_beats_baseline_for_tag(tag),
            }
        return scorecard

    def candidate_mean_abs_relative_error_for_tag(self, tag: str | None) -> float | None:
        return self.benchmark_suite.candidate_mean_abs_relative_error_for_tag(tag)

    def baseline_mean_abs_relative_error_for_tag(self, tag: str | None) -> float | None:
        return self.benchmark_suite.baseline_mean_abs_relative_error_for_tag(tag)

    def mean_abs_relative_error_delta_for_tag(self, tag: str | None) -> float | None:
        return self.benchmark_suite.mean_abs_relative_error_delta_for_tag(tag)

    def slice_win_rate_for_tag(self, tag: str | None) -> float | None:
        return self.benchmark_suite.slice_win_rate_for_tag(tag)

    def target_win_rate_for_tag(self, tag: str | None) -> float | None:
        return self.benchmark_suite.target_win_rate_for_tag(tag)

    def supported_target_rate_for_tag(self, tag: str | None) -> float | None:
        return self.benchmark_suite.supported_target_rate_for_tag(tag)

    def candidate_micro_mean_abs_relative_error_for_tag(self, tag: str | None) -> float | None:
        return self._micro_mean_abs_relative_error_for_tag(tag, kind="candidate")

    def baseline_micro_mean_abs_relative_error_for_tag(self, tag: str | None) -> float | None:
        return self._micro_mean_abs_relative_error_for_tag(tag, kind="baseline")

    def candidate_attribute_macro_mean_abs_relative_error_for_tag(
        self,
        tag: str | None,
    ) -> float | None:
        return self._attribute_macro_mean_abs_relative_error_for_tag(
            tag,
            kind="candidate",
        )

    def baseline_attribute_macro_mean_abs_relative_error_for_tag(
        self,
        tag: str | None,
    ) -> float | None:
        return self._attribute_macro_mean_abs_relative_error_for_tag(
            tag,
            kind="baseline",
        )

    def candidate_attribute_tail_mean_abs_relative_error_for_tag(
        self,
        tag: str | None,
    ) -> float | None:
        return self._attribute_tail_mean_abs_relative_error_for_tag(
            tag,
            kind="candidate",
        )

    def baseline_attribute_tail_mean_abs_relative_error_for_tag(
        self,
        tag: str | None,
    ) -> float | None:
        return self._attribute_tail_mean_abs_relative_error_for_tag(
            tag,
            kind="baseline",
        )

    def candidate_composite_parity_loss_for_tag(self, tag: str | None) -> float | None:
        return self._composite_parity_loss_for_tag(tag, kind="candidate")

    def baseline_composite_parity_loss_for_tag(self, tag: str | None) -> float | None:
        return self._composite_parity_loss_for_tag(tag, kind="baseline")

    def composite_parity_loss_delta_for_tag(self, tag: str | None) -> float | None:
        candidate_loss = self.candidate_composite_parity_loss_for_tag(tag)
        baseline_loss = self.baseline_composite_parity_loss_for_tag(tag)
        if candidate_loss is None or baseline_loss is None:
            return None
        return candidate_loss - baseline_loss

    def attribute_cell_summaries_for_tag(
        self,
        tag: str | None,
    ) -> dict[str, dict[str, Any]]:
        candidate_records = self._target_records_for_tag(tag, kind="candidate")
        baseline_records = self._target_records_for_tag(tag, kind="baseline")
        cells: dict[str, dict[str, Any]] = {}
        for kind, records in (
            ("candidate", candidate_records),
            ("baseline", baseline_records),
        ):
            for record in records.values():
                attrs = self._target_attribute_summary(record["target"])
                cell_key = attrs["cell_key"]
                summary = cells.setdefault(
                    cell_key,
                    {
                        **attrs,
                        "candidate_target_count": 0,
                        "candidate_supported_target_count": 0,
                        "baseline_target_count": 0,
                        "baseline_supported_target_count": 0,
                        "_candidate_errors": [],
                        "_baseline_errors": [],
                    },
                )
                summary[f"{kind}_target_count"] += 1
                if record["supported"]:
                    summary[f"{kind}_supported_target_count"] += 1
                relative_error = record["relative_error"]
                if relative_error is not None:
                    summary[f"_{kind}_errors"].append(abs(relative_error))

        for summary in cells.values():
            for kind in ("candidate", "baseline"):
                errors = summary.pop(f"_{kind}_errors")
                target_count = summary[f"{kind}_target_count"]
                supported_count = summary[f"{kind}_supported_target_count"]
                summary[f"{kind}_mean_abs_relative_error"] = (
                    float(np.mean(errors)) if errors else None
                )
                summary[f"{kind}_support_rate"] = (
                    supported_count / target_count if target_count else None
                )
            candidate_error = summary["candidate_mean_abs_relative_error"]
            baseline_error = summary["baseline_mean_abs_relative_error"]
            summary["mean_abs_relative_error_delta"] = (
                candidate_error - baseline_error
                if candidate_error is not None and baseline_error is not None
                else None
            )
        return dict(sorted(cells.items()))

    def _candidate_beats_baseline_for_tag(self, tag: str | None) -> bool | None:
        delta = self.mean_abs_relative_error_delta_for_tag(tag)
        if delta is None:
            return None
        return delta < 0.0

    def _slice_results_for_tag(
        self,
        tag: str | None,
    ) -> list[PolicyEngineUSHarnessSliceResult]:
        if tag is None:
            return list(self.slice_results)
        return [
            result for result in self.slice_results if tag in result.slice.tags
        ]

    def _mean_abs_relative_error(
        self,
        *,
        tag: str | None,
        kind: str,
    ) -> float | None:
        suite = self.benchmark_suite
        if kind == "candidate":
            return suite.candidate_mean_abs_relative_error_for_tag(tag)
        return suite.baseline_mean_abs_relative_error_for_tag(tag)

    def _supported_target_rate_for_tag(
        self,
        tag: str | None,
        *,
        kind: str,
    ) -> float | None:
        suite = self.benchmark_suite
        if kind == "candidate":
            return suite.supported_target_rate_for_tag(tag)
        return suite.baseline_supported_target_rate_for_tag(tag)

    def _micro_mean_abs_relative_error_for_tag(
        self,
        tag: str | None,
        *,
        kind: str,
    ) -> float | None:
        errors = [
            abs(record["relative_error"])
            for record in self._target_records_for_tag(tag, kind=kind).values()
            if record["relative_error"] is not None
        ]
        if not errors:
            return None
        return float(np.mean(errors))

    def _attribute_macro_mean_abs_relative_error_for_tag(
        self,
        tag: str | None,
        *,
        kind: str,
    ) -> float | None:
        errors = [
            cell[f"{kind}_mean_abs_relative_error"]
            for cell in self.attribute_cell_summaries_for_tag(tag).values()
            if cell[f"{kind}_mean_abs_relative_error"] is not None
        ]
        if not errors:
            return None
        return float(np.mean(errors))

    def _attribute_tail_mean_abs_relative_error_for_tag(
        self,
        tag: str | None,
        *,
        kind: str,
    ) -> float | None:
        errors = sorted(
            (
                cell[f"{kind}_mean_abs_relative_error"]
                for cell in self.attribute_cell_summaries_for_tag(tag).values()
                if cell[f"{kind}_mean_abs_relative_error"] is not None
            ),
            reverse=True,
        )
        if not errors:
            return None
        n_tail = max(1, int(np.ceil(len(errors) * ATTRIBUTE_TAIL_FRACTION)))
        return float(np.mean(errors[:n_tail]))

    def _composite_parity_loss_for_tag(
        self,
        tag: str | None,
        *,
        kind: str,
    ) -> float | None:
        micro = self._micro_mean_abs_relative_error_for_tag(tag, kind=kind)
        macro = self._attribute_macro_mean_abs_relative_error_for_tag(tag, kind=kind)
        tail = self._attribute_tail_mean_abs_relative_error_for_tag(tag, kind=kind)
        support_rate = self._supported_target_rate_for_tag(tag, kind=kind)
        if (
            micro is None
            or macro is None
            or tail is None
            or support_rate is None
        ):
            return None
        weights = COMPOSITE_PARITY_LOSS_WEIGHTS
        return (
            weights["micro"] * micro
            + weights["attribute_macro"] * macro
            + weights["attribute_tail"] * tail
            + weights["support_gap"] * (1.0 - support_rate)
        )

    def _target_records_for_tag(
        self,
        tag: str | None,
        *,
        kind: str,
    ) -> dict[tuple[str, str], dict[str, Any]]:
        records: dict[tuple[str, str], dict[str, Any]] = {}
        for result in self._slice_results_for_tag(tag):
            report = (
                result.comparison.candidate
                if kind == "candidate"
                else result.comparison.baseline
            )
            if report is None:
                continue
            for target in report.unsupported_targets:
                record_key = (result.slice.name, target.name)
                records.setdefault(
                    record_key,
                    {
                        "target": target,
                        "supported": False,
                        "relative_error": None,
                    },
                )
            for evaluation in report.evaluations:
                records[(result.slice.name, evaluation.target.name)] = {
                    "target": evaluation.target,
                    "supported": True,
                    "relative_error": evaluation.relative_error,
                }
        return records

    def _target_attribute_summary(self, target: Any) -> dict[str, str]:
        metadata = dict(getattr(target, "metadata", {}) or {})
        geo_level = str(metadata.get("geo_level") or UNSPECIFIED_ATTRIBUTE)
        entity = target.entity.value
        aggregation = target.aggregation.value
        feature = str(
            target.measure
            or metadata.get("variable")
            or f"{entity}_count"
        )
        domain_variable = str(
            metadata.get("domain_variable") or UNSPECIFIED_ATTRIBUTE
        )
        cell_key = "|".join(
            (
                f"geo={geo_level}",
                f"entity={entity}",
                f"aggregation={aggregation}",
                f"feature={feature}",
                f"domain={domain_variable}",
            )
        )
        return {
            "cell_key": cell_key,
            "geo_level": geo_level,
            "entity": entity,
            "aggregation": aggregation,
            "feature": feature,
            "domain_variable": domain_variable,
        }

    def to_dict(self) -> dict[str, Any]:
        """Serialize the harness run to a JSON-compatible dict."""
        return {
            "candidate_label": self.candidate_label,
            "baseline_label": self.baseline_label,
            "period": self.period,
            "created_at": self.created_at,
            "metadata": dict(self.metadata),
            "summary": {
                "candidate_mean_abs_relative_error": self.candidate_mean_abs_relative_error,
                "baseline_mean_abs_relative_error": self.baseline_mean_abs_relative_error,
                "mean_abs_relative_error_delta": self.mean_abs_relative_error_delta,
                "slice_win_rate": self.slice_win_rate,
                "target_win_rate": self.target_win_rate,
                "supported_target_rate": self.supported_target_rate,
                "baseline_supported_target_rate": self.baseline_supported_target_rate,
                "candidate_micro_mean_abs_relative_error": self.candidate_micro_mean_abs_relative_error,
                "baseline_micro_mean_abs_relative_error": self.baseline_micro_mean_abs_relative_error,
                "candidate_attribute_macro_mean_abs_relative_error": self.candidate_attribute_macro_mean_abs_relative_error,
                "baseline_attribute_macro_mean_abs_relative_error": self.baseline_attribute_macro_mean_abs_relative_error,
                "candidate_attribute_tail_mean_abs_relative_error": self.candidate_attribute_tail_mean_abs_relative_error,
                "baseline_attribute_tail_mean_abs_relative_error": self.baseline_attribute_tail_mean_abs_relative_error,
                "candidate_composite_parity_loss": self.candidate_composite_parity_loss,
                "baseline_composite_parity_loss": self.baseline_composite_parity_loss,
                "composite_parity_loss_delta": self.composite_parity_loss_delta,
                "tag_summaries": self.tag_summaries,
                "parity_scorecard": self.parity_scorecard,
                "attribute_cell_summaries": self.attribute_cell_summaries,
            },
            "slices": [
                {
                    **result.slice.to_dict(),
                    "summary": _slice_result_summary(result),
                    "candidate": _report_to_dict(result.comparison.candidate),
                    "baseline": (
                        _report_to_dict(result.comparison.baseline)
                        if result.comparison.baseline is not None
                        else None
                    ),
                }
                for result in self.slice_results
            ],
        }

    def save(self, path: str | Path) -> Path:
        """Persist the harness run as JSON."""
        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(self.to_dict(), indent=2, sort_keys=True))
        return output_path

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> PolicyEngineUSHarnessRun:
        """Restore a harness run from serialized JSON payload."""
        return cls(
            candidate_label=payload["candidate_label"],
            baseline_label=payload["baseline_label"],
            period=payload["period"],
            created_at=payload["created_at"],
            metadata=dict(payload.get("metadata", {})),
            slice_results=[
                PolicyEngineUSHarnessSliceResult(
                    slice=PolicyEngineUSHarnessSlice.from_dict(slice_payload),
                    comparison=PolicyEngineUSTargetComparisonReport(
                        candidate=_report_from_dict(slice_payload["candidate"]),
                        baseline=(
                            _report_from_dict(slice_payload["baseline"])
                            if slice_payload.get("baseline") is not None
                            else None
                        ),
                    ),
                )
                for slice_payload in payload.get("slices", [])
            ],
        )

    @classmethod
    def load(cls, path: str | Path) -> PolicyEngineUSHarnessRun:
        """Load a persisted harness run from JSON."""
        return cls.from_dict(json.loads(Path(path).read_text()))


@dataclass
class _PolicyEngineUSCandidateBatchResultEvaluator(BatchBenchmarkResultEvaluator):
    tables: PolicyEngineUSEntityTableBundle
    period: int | str
    dataset_year: int | None
    simulation_cls: Any | None
    label: str
    strict_materialization: bool
    direct_override_variables: tuple[str, ...] = ()
    last_reports: dict[str, PolicyEngineUSTargetEvaluationReport] = field(
        default_factory=dict,
        init=False,
    )

    def evaluate_target_sets(
        self,
        target_sets: dict[str, TargetSet],
        slices: tuple[BenchmarkSliceSpec, ...],
    ) -> dict[str, BenchmarkResult]:
        del slices
        reports = evaluate_policyengine_us_target_sets(
            self.tables,
            target_sets,
            period=self.period,
            dataset_year=self.dataset_year,
            simulation_cls=self.simulation_cls,
            label=self.label,
            strict_materialization=self.strict_materialization,
            direct_override_variables=self.direct_override_variables,
        )
        self.last_reports = reports
        return {name: report.benchmark_result for name, report in reports.items()}


@dataclass
class _PolicyEngineUSBaselineBatchResultEvaluator(BatchBenchmarkResultEvaluator):
    baseline_dataset: str | Any
    period: int | str
    dataset_year: int | None
    simulation_cls: Any | None
    baseline_label: str
    strict_materialization: bool
    cache: PolicyEngineUSComparisonCache | None
    last_reports: dict[str, PolicyEngineUSTargetEvaluationReport] = field(
        default_factory=dict,
        init=False,
    )

    def evaluate_target_sets(
        self,
        target_sets: dict[str, TargetSet],
        slices: tuple[BenchmarkSliceSpec, ...],
    ) -> dict[str, BenchmarkResult]:
        del slices
        reports = _evaluate_policyengine_us_baseline_target_sets(
            target_sets,
            baseline_dataset=self.baseline_dataset,
            period=self.period,
            dataset_year=self.dataset_year,
            simulation_cls=self.simulation_cls,
            baseline_label=self.baseline_label,
            strict_materialization=self.strict_materialization,
            cache=self.cache,
        )
        self.last_reports = reports
        return {name: report.benchmark_result for name, report in reports.items()}


@pipeline_node(
    id="us.stage9.evaluate_policyengine_us_harness",
    label="Evaluate PE target harness",
    description="Evaluate candidate entity tables against PolicyEngine target slices and baseline evidence.",
    artifacts_in=("policyengine_entity_tables", "policyengine_targets"),
    artifacts_out=("policyengine_harness",),
)
def evaluate_policyengine_us_harness(
    candidate_tables: PolicyEngineUSEntityTableBundle,
    provider: TargetProvider,
    slices: list[PolicyEngineUSHarnessSlice] | tuple[PolicyEngineUSHarnessSlice, ...],
    *,
    baseline_dataset: str | Any,
    dataset_year: int | None = None,
    simulation_cls: Any | None = None,
    candidate_label: str = "microplex",
    baseline_label: str = "policyengine_baseline",
    metadata: dict[str, Any] | None = None,
    strict_materialization: bool = True,
    cache: PolicyEngineUSComparisonCache | None = None,
    candidate_direct_override_variables: tuple[str, ...] = (),
) -> PolicyEngineUSHarnessRun:
    """Evaluate a candidate bundle against a baseline across named target slices."""
    if not slices:
        raise ValueError("PolicyEngineUSHarness requires at least one slice")

    slice_target_sets = load_benchmark_slice_target_sets(
        provider,
        slices,
        loader=(
            (lambda effective_provider, query: cache.load_target_set(effective_provider, query))
            if cache is not None
            else None
        ),
    )
    period = slices[0].query.period if slices[0].query.period is not None else 2024
    candidate_result_evaluator = _PolicyEngineUSCandidateBatchResultEvaluator(
        tables=candidate_tables,
        period=period,
        dataset_year=dataset_year,
        simulation_cls=simulation_cls,
        label=candidate_label,
        strict_materialization=strict_materialization,
        direct_override_variables=candidate_direct_override_variables,
    )
    candidate_results = evaluate_benchmark_slice_results(
        slice_target_sets,
        slices,
        batch_evaluator=candidate_result_evaluator,
    )
    baseline_result_evaluator = _PolicyEngineUSBaselineBatchResultEvaluator(
        baseline_dataset=baseline_dataset,
        period=period,
        dataset_year=dataset_year,
        simulation_cls=simulation_cls,
        baseline_label=baseline_label,
        strict_materialization=strict_materialization,
        cache=cache,
    )
    baseline_results = evaluate_benchmark_slice_results(
        slice_target_sets,
        slices,
        batch_evaluator=baseline_result_evaluator,
    )
    candidate_reports = candidate_result_evaluator.last_reports
    baseline_reports = baseline_result_evaluator.last_reports
    slice_results = [
        PolicyEngineUSHarnessSliceResult(
            slice=slice_spec,
            comparison=PolicyEngineUSTargetComparisonReport(
                candidate=candidate_reports[slice_spec.name],
                baseline=baseline_reports[slice_spec.name],
            ),
        )
        for slice_spec in slices
    ]
    comparable_slice_results = [
        result for result in slice_results if result.comparison.benchmark_comparison is not None
    ]
    suite_metadata = dict(metadata or {})
    if len(comparable_slice_results) != len(slice_results):
        suite_metadata["excluded_slice_names"] = [
            result.slice.name
            for result in slice_results
            if result.comparison.benchmark_comparison is None
        ]

    return PolicyEngineUSHarnessRun(
        candidate_label=candidate_label,
        baseline_label=baseline_label,
        period=period,
        slice_results=slice_results,
        metadata=suite_metadata,
        _benchmark_suite=(
            build_benchmark_suite_from_results(
                candidate_label=candidate_label,
                baseline_label=baseline_label,
                period=period,
                slices=[result.slice for result in comparable_slice_results],
                candidate_results={
                    result.slice.name: candidate_results[result.slice.name]
                    for result in comparable_slice_results
                },
                baseline_results={
                    result.slice.name: baseline_results[result.slice.name]
                    for result in comparable_slice_results
                },
                group_fields=POLICYENGINE_US_BENCHMARK_GROUP_FIELDS,
                metadata=suite_metadata,
            )
            if comparable_slice_results
            else build_benchmark_suite_result(
                candidate_label=candidate_label,
                baseline_label=baseline_label,
                period=period,
                slice_results=[],
                metadata=suite_metadata,
            )
        ),
    )


def _evaluate_policyengine_us_baseline_target_sets(
    target_sets: dict[str, TargetSet],
    *,
    baseline_dataset: str | Any,
    period: int | str,
    dataset_year: int | None,
    simulation_cls: Any | None,
    baseline_label: str,
    strict_materialization: bool,
    cache: PolicyEngineUSComparisonCache | None,
) -> dict[str, PolicyEngineUSTargetEvaluationReport]:
    union_target_set = union_target_sets(target_sets)
    baseline_union_report = (
        cache.load_baseline_report(
            target_set=union_target_set,
            baseline_dataset=baseline_dataset,
            period=period,
            dataset_year=dataset_year,
            simulation_cls=simulation_cls,
            baseline_label=baseline_label,
            strict_materialization=strict_materialization,
        )
        if cache is not None
        else evaluate_policyengine_us_target_set(
            load_policyengine_us_entity_tables(
                baseline_dataset,
                period=period,
            ),
            union_target_set,
            period=period,
            dataset_year=dataset_year,
            simulation_cls=simulation_cls,
            label=baseline_label,
            strict_materialization=strict_materialization,
        )
    )
    return {
        name: slice_policyengine_us_target_evaluation_report(
            baseline_union_report,
            target_set,
        )
        for name, target_set in target_sets.items()
    }


def filter_nonempty_policyengine_us_harness_slices(
    provider: TargetProvider,
    slices: list[PolicyEngineUSHarnessSlice] | tuple[PolicyEngineUSHarnessSlice, ...],
    *,
    cache: PolicyEngineUSComparisonCache | None = None,
) -> tuple[PolicyEngineUSHarnessSlice, ...]:
    """Drop harness slices that resolve to no canonical targets."""
    return filter_nonempty_benchmark_slices(
        provider,
        slices,
        loader=(
            (lambda effective_provider, query: cache.load_target_set(effective_provider, query))
            if cache is not None
            else None
        ),
    )


def default_policyengine_us_harness_slices(
    *,
    period: int,
) -> tuple[PolicyEngineUSHarnessSlice, ...]:
    """Return a small default PE-US harness focused on parity-critical aggregates."""
    return (
        PolicyEngineUSHarnessSlice(
            name="household_count",
            description="National household count target slice",
            query=TargetQuery(period=period, names=("household_count",)),
        ),
        PolicyEngineUSHarnessSlice(
            name="snap",
            description="National SNAP amount and recipient slices",
            query=TargetQuery(period=period, names=("snap", "household_count")),
        ),
        PolicyEngineUSHarnessSlice(
            name="california",
            description="California geographic subset",
            query=TargetQuery(
                period=period,
                metadata_filters={"geo_level": "state"},
            ),
        ),
    )


def default_policyengine_us_db_all_target_slices(
    *,
    period: int,
    reform_id: int = 0,
) -> tuple[PolicyEngineUSHarnessSlice, ...]:
    """Return one benchmark slice spanning all active PE-US DB targets for a period."""
    return (
        PolicyEngineUSHarnessSlice(
            name="all_targets",
            description="All active PE-US DB targets for this benchmark period",
            tags=("benchmark", "all_targets"),
            query=TargetQuery(
                period=period,
                provider_filters={"reform_id": reform_id},
            ),
        ),
    )


def default_policyengine_us_db_harness_slices(
    *,
    period: int,
    variables: tuple[str, ...] = (),
    domain_variables: tuple[str, ...] = (),
    geo_levels: tuple[str, ...] = (),
    reform_id: int = 0,
) -> tuple[PolicyEngineUSHarnessSlice, ...]:
    """Return DB-backed default PE-US harness slices derived from target filters."""
    base_provider_filters = {
        "reform_id": reform_id,
        "variables": list(variables) if variables else None,
        "domain_variables": list(domain_variables) if domain_variables else None,
        "geo_levels": list(geo_levels) if geo_levels else None,
    }
    slices = [
        PolicyEngineUSHarnessSlice(
            name="all_targets",
            description="All PE-US DB targets selected for this build",
            tags=("benchmark", "all_targets"),
            query=TargetQuery(
                period=period,
                provider_filters={
                    key: value
                    for key, value in base_provider_filters.items()
                    if value is not None
                },
            ),
        )
    ]
    for variable in variables:
        slices.append(
            PolicyEngineUSHarnessSlice(
                name=variable,
                description=f"{variable} targets selected for this build",
                query=TargetQuery(
                    period=period,
                    provider_filters={
                        key: value
                        for key, value in {
                            **base_provider_filters,
                            "variables": [variable],
                        }.items()
                        if value is not None
                    },
                ),
            )
        )
    return tuple(slices)


def default_policyengine_us_db_parity_slices(
    *,
    period: int,
    variables: tuple[str, ...] = (),
    domain_variables: tuple[str, ...] = (),
    geo_levels: tuple[str, ...] = (),
    reform_id: int = 0,
) -> tuple[PolicyEngineUSHarnessSlice, ...]:
    """Return the default PE-US parity suite split across national and local loss."""
    slice_specs = [
        {
            "name": "national_aggregate_core",
            "description": "National aggregate calibration targets from PE-US production",
            "geo_levels": ("national",),
            "variables": (
                "adjusted_gross_income",
                "child_support_expense",
                "child_support_received",
                "health_insurance_premiums_without_medicare_part_b",
                "income_tax_positive",
                "medicaid",
                "medicare_part_b_premiums",
                "other_medical_expenses",
                "over_the_counter_health_expenses",
                "qualified_business_income_deduction",
                "rent",
                "salt_deduction",
                "snap",
                "social_security",
                "social_security_disability",
                "social_security_retirement",
                "spm_unit_capped_housing_subsidy",
                "spm_unit_capped_work_childcare_expenses",
                "ssi",
                "tanf",
                "tip_income",
                "unemployment_compensation",
            ),
            "domain_variable_is_null": True,
            "tags": ("parity", "national", "aggregate"),
        },
        {
            "name": "national_soi_amounts",
            "description": "National IRS SOI amount targets used in production calibration",
            "geo_levels": ("national",),
            "variables": (
                "income_tax_before_credits",
                "dividend_income",
                "net_capital_gains",
                "qualified_business_income_deduction",
                "qualified_dividend_income",
                "rental_income",
                "salt",
                "self_employment_income",
                "tax_exempt_interest_income",
                "tax_unit_partnership_s_corp_income",
                "taxable_interest_income",
                "taxable_ira_distributions",
                "taxable_pension_income",
                "taxable_social_security",
                "unemployment_compensation",
            ),
            "domain_variable_values": (
                "income_tax_before_credits",
                "dividend_income",
                "net_capital_gains",
                "qualified_business_income_deduction",
                "qualified_dividend_income",
                "rental_income",
                "salt",
                "self_employment_income",
                "tax_exempt_interest_income",
                "tax_unit_partnership_s_corp_income",
                "taxable_interest_income",
                "taxable_ira_distributions",
                "taxable_pension_income",
                "taxable_social_security",
                "unemployment_compensation",
            ),
            "tags": ("parity", "national", "tax", "soi_amounts"),
        },
        {
            "name": "national_soi_filer_counts",
            "description": "National IRS SOI filer-count targets used in production calibration",
            "geo_levels": ("national",),
            "variables": ("tax_unit_count",),
            "domain_variable_values": (
                "dividend_income",
                "income_tax",
                "income_tax_before_credits",
                "medical_expense_deduction",
                "net_capital_gains",
                "qualified_business_income_deduction",
                "qualified_dividend_income",
                "real_estate_taxes",
                "rental_income",
                "salt",
                "self_employment_income",
                "tax_exempt_interest_income",
                "tax_unit_partnership_s_corp_income",
                "taxable_interest_income",
                "taxable_ira_distributions",
                "taxable_pension_income",
                "taxable_social_security",
                "unemployment_compensation",
            ),
            "tags": ("parity", "national", "counts", "soi_filers"),
        },
        {
            "name": "state_programs_core",
            "description": "State SNAP household counts and Medicaid recipiency counts from production calibration",
            "geo_levels": ("state",),
            "variables": ("household_count", "person_count"),
            "domain_variable_values": ("snap", "medicaid_enrolled"),
            "tags": ("parity", "local", "state", "programs"),
        },
        {
            "name": "district_age_counts",
            "description": "District age-band person counts from production calibration",
            "geo_levels": ("district",),
            "variables": ("person_count",),
            "domain_variable_values": ("age",),
            "tags": ("parity", "local", "district", "counts", "age"),
        },
        {
            "name": "district_agi_counts",
            "description": "District AGI-band person counts from production calibration",
            "geo_levels": ("district",),
            "variables": ("person_count",),
            "domain_variable_values": ("adjusted_gross_income",),
            "tags": ("parity", "local", "district", "counts", "agi"),
        },
        {
            "name": "district_snap_households",
            "description": "District SNAP-recipient household counts from production calibration",
            "geo_levels": ("district",),
            "variables": ("household_count",),
            "domain_variable_values": ("snap",),
            "tags": ("parity", "local", "district", "programs", "snap"),
        },
        {
            "name": "district_income_core",
            "description": "District income-component totals from production calibration",
            "geo_levels": ("district",),
            "variables": (
                "real_estate_taxes",
                "self_employment_income",
                "taxable_pension_income",
                "unemployment_compensation",
            ),
            "domain_variable_values": (
                "real_estate_taxes",
                "self_employment_income",
                "taxable_pension_income",
                "unemployment_compensation",
            ),
            "tags": ("parity", "local", "district", "income"),
        },
    ]
    slices: list[PolicyEngineUSHarnessSlice] = []
    for spec in slice_specs:
        provider_filters = _build_parity_provider_filters(
            base_variables=variables,
            base_domain_variables=domain_variables,
            base_geo_levels=geo_levels,
            spec_variables=spec.get("variables"),
            spec_domain_variables=spec.get("domain_variables"),
            spec_domain_variable_values=spec.get("domain_variable_values"),
            spec_domain_variable_is_null=spec.get("domain_variable_is_null"),
            spec_geo_levels=spec.get("geo_levels"),
            reform_id=reform_id,
        )
        if provider_filters is None:
            continue
        slices.append(
            PolicyEngineUSHarnessSlice(
                name=spec["name"],
                description=spec["description"],
                tags=spec["tags"],
                query=TargetQuery(period=period, provider_filters=provider_filters),
            )
        )
    return tuple(slices)


def _build_parity_provider_filters(
    *,
    base_variables: tuple[str, ...],
    base_domain_variables: tuple[str, ...],
    base_geo_levels: tuple[str, ...],
    spec_variables: tuple[str, ...] | None,
    spec_domain_variables: tuple[str, ...] | None,
    spec_domain_variable_values: tuple[str, ...] | None,
    spec_domain_variable_is_null: bool | None,
    spec_geo_levels: tuple[str, ...] | None,
    reform_id: int,
) -> dict[str, Any] | None:
    resolved_variables = _intersect_optional_filters(base_variables, spec_variables)
    resolved_domain_variable_values = _intersect_optional_filters(
        base_domain_variables,
        spec_domain_variable_values,
    )
    resolved_domain_variables = (
        None
        if spec_domain_variable_values is not None
        else _intersect_optional_filters(
            base_domain_variables,
            spec_domain_variables,
        )
    )
    resolved_geo_levels = _intersect_optional_filters(base_geo_levels, spec_geo_levels)
    if (
        resolved_variables == ()
        or resolved_domain_variables == ()
        or resolved_domain_variable_values == ()
        or resolved_geo_levels == ()
        or (base_domain_variables and spec_domain_variable_is_null is True)
    ):
        return None
    return {
        key: value
        for key, value in {
            "reform_id": reform_id,
            "variables": list(resolved_variables) if resolved_variables else None,
            "domain_variables": (
                list(resolved_domain_variables) if resolved_domain_variables else None
            ),
            "domain_variable_values": (
                list(resolved_domain_variable_values)
                if resolved_domain_variable_values
                else None
            ),
            "domain_variable_is_null": (
                spec_domain_variable_is_null
                if spec_domain_variable_is_null is not None
                else None
            ),
            "geo_levels": list(resolved_geo_levels) if resolved_geo_levels else None,
        }.items()
        if value is not None
    }


def _intersect_optional_filters(
    base_values: tuple[str, ...],
    spec_values: tuple[str, ...] | None,
) -> tuple[str, ...] | None:
    if not base_values and not spec_values:
        return None
    if not base_values:
        return tuple(spec_values or ())
    if spec_values is None:
        return tuple(base_values)
    intersection = tuple(value for value in spec_values if value in set(base_values))
    return intersection


def _report_to_dict(report: Any) -> dict[str, Any]:
    return {
        "label": report.label,
        "period": report.period,
        "summary": {
            "supported_target_count": report.supported_target_count,
            "unsupported_target_count": len(report.unsupported_targets),
            "mean_abs_relative_error": report.mean_abs_relative_error,
            "max_abs_relative_error": report.max_abs_relative_error,
        },
        "materialized_variables": list(report.materialized_variables),
        "materialization_failures": dict(report.materialization_failures),
        "unsupported_targets": [_target_to_dict(target) for target in report.unsupported_targets],
        "evaluations": [
            {
                "target": _target_to_dict(evaluation.target),
                "actual_value": evaluation.actual_value,
                "absolute_error": evaluation.absolute_error,
                "relative_error": evaluation.relative_error,
            }
            for evaluation in report.evaluations
        ],
    }


def _report_from_dict(payload: dict[str, Any]) -> Any:
    return PolicyEngineUSTargetEvaluationReport(
        label=payload["label"],
        period=payload["period"],
        evaluations=[
            PolicyEngineUSTargetEvaluation(
                target=_target_from_dict(item["target"]),
                actual_value=float(item["actual_value"]),
            )
            for item in payload.get("evaluations", [])
        ],
        unsupported_targets=[
            _target_from_dict(target)
            for target in payload.get("unsupported_targets", [])
        ],
        materialized_variables=tuple(payload.get("materialized_variables", [])),
        materialization_failures=dict(payload.get("materialization_failures", {})),
    )


def _slice_result_summary(
    result: PolicyEngineUSHarnessSliceResult,
) -> dict[str, float | int | bool | dict[str, str] | None]:
    candidate = result.comparison.candidate
    baseline = result.comparison.baseline
    return {
        "candidate_supported_target_count": candidate.supported_target_count,
        "candidate_unsupported_target_count": len(candidate.unsupported_targets),
        "candidate_mean_abs_relative_error": candidate.mean_abs_relative_error,
        "candidate_max_abs_relative_error": candidate.max_abs_relative_error,
        "candidate_materialization_failures": dict(candidate.materialization_failures),
        "baseline_supported_target_count": (
            baseline.supported_target_count if baseline is not None else None
        ),
        "baseline_unsupported_target_count": (
            len(baseline.unsupported_targets) if baseline is not None else None
        ),
        "baseline_mean_abs_relative_error": (
            baseline.mean_abs_relative_error if baseline is not None else None
        ),
        "baseline_max_abs_relative_error": (
            baseline.max_abs_relative_error if baseline is not None else None
        ),
        "baseline_materialization_failures": (
            dict(baseline.materialization_failures)
            if baseline is not None
            else {}
        ),
        "mean_abs_relative_error_delta": result.mean_abs_relative_error_delta,
        "candidate_beats_baseline": result.candidate_beats_baseline,
    }


def _target_to_dict(target: Any) -> dict[str, Any]:
    return {
        "name": target.name,
        "entity": target.entity.value,
        "value": float(target.value),
        "period": target.period,
        "measure": target.measure,
        "aggregation": target.aggregation.value,
        "filters": [
            {
                "feature": target_filter.feature,
                "operator": target_filter.operator.value,
                "value": target_filter.value,
            }
            for target_filter in target.filters
        ],
        "tolerance": target.tolerance,
        "source": target.source,
        "units": target.units,
        "description": target.description,
        "metadata": dict(target.metadata),
    }


def _target_from_dict(payload: dict[str, Any]) -> Any:
    from microplex.core import EntityType
    from microplex.targets import TargetAggregation, TargetSpec

    return TargetSpec(
        name=payload["name"],
        entity=EntityType(payload["entity"]),
        value=float(payload["value"]),
        period=payload["period"],
        measure=payload.get("measure"),
        aggregation=TargetAggregation(payload["aggregation"]),
        filters=tuple(
            TargetFilter(
                item["feature"] if "feature" in item else item["variable"],
                FilterOperator(item["operator"]),
                item["value"],
            )
            for item in payload.get("filters", [])
        ),
        tolerance=payload.get("tolerance"),
        source=payload.get("source"),
        units=payload.get("units"),
        description=payload.get("description"),
        metadata=dict(payload.get("metadata", {})),
    )
