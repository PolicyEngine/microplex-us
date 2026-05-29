"""Tests for the US microplex performance harness."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
from microplex.core import EntityType
from microplex.targets import TargetQuery, TargetSet, TargetSpec

from microplex_us.pipelines.pe_native_optimization import (
    PolicyEngineUSNativeWeightOptimizationResult,
)
from microplex_us.pipelines.performance import (
    USMicroplexPerformanceHarnessConfig,
    USMicroplexPerformanceHarnessRequest,
    USMicroplexPerformanceHarnessResult,
    USMicroplexPerformanceSession,
    _calibration_build_config_key,
    _precalibration_build_config_key,
    _sample_matched_household_ids,
    _write_matched_policyengine_us_baseline_dataset,
    default_fast_calibration_target_variables,
    run_us_microplex_performance_harness,
    warm_us_microplex_parity_cache,
)
from microplex_us.pipelines.us import USMicroplexBuildConfig
from microplex_us.policyengine import (
    PolicyEngineUSComparisonCache,
    PolicyEngineUSEntityTableBundle,
    load_policyengine_us_entity_tables,
    write_policyengine_us_time_period_dataset,
)


class _DummyProvider:
    def __init__(self, name: str):
        self.descriptor = SimpleNamespace(name=name)

    def load_frame(self, query=None):
        _ = query
        return SimpleNamespace(source=SimpleNamespace(name=self.descriptor.name))


@dataclass
class _FakeHarnessRun:
    candidate_composite_parity_loss: float = 0.4
    baseline_composite_parity_loss: float = 0.5
    candidate_mean_abs_relative_error: float = 0.2
    baseline_mean_abs_relative_error: float = 0.25
    target_win_rate: float = 0.75
    slice_win_rate: float = 1.0


class _FakePipeline:
    def __init__(
        self,
        config=None,
        stage_log: list[str] | None = None,
        stage_log_style: str = "full",
    ):
        self.config = config
        self.stage_log = stage_log
        self.stage_log_style = stage_log_style

    def _log(self, message: str) -> None:
        if self.stage_log is not None:
            self.stage_log.append(message)

    def prepare_source_input(self, frame):
        if self.stage_log_style == "short":
            self._log(f"prepare_source_input:{frame.source.name}")
        else:
            self._log("prepare_source_input")
        return SimpleNamespace(frame=frame)

    def _select_scaffold_source(self, source_inputs):
        self._log("select_scaffold")
        return source_inputs[0]

    def prepare_seed_data_from_source(self, source_input):
        _ = source_input
        self._log("prepare_seed" if self.stage_log_style == "short" else "prepare_seed_data")
        return pd.DataFrame({"household_id": [1], "income": [1.0], "hh_weight": [1.0]})

    def _integrate_donor_sources(self, seed_data, *, scaffold_input, donor_inputs):
        _ = scaffold_input
        _ = donor_inputs
        self._log(
            "integrate_donors"
            if self.stage_log_style == "short"
            else "integrate_donor_sources"
        )
        if self.stage_log is not None:
            return {
                "seed_data": seed_data.assign(dividend_income=[2.0]),
                "integrated_variables": ["dividend_income"],
            }
        return {"seed_data": seed_data, "integrated_variables": []}

    def build_targets(self, seed_data):
        _ = seed_data
        self._log("build_targets")
        return SimpleNamespace(marginal={}, continuous={})

    def _resolve_synthesis_variables(
        self,
        source_input=None,
        *,
        fusion_plan=None,
        include_all_observed_targets=False,
        available_columns=None,
    ):
        _ = source_input
        _ = fusion_plan
        _ = include_all_observed_targets
        _ = available_columns
        self._log("resolve_synthesis_variables")
        return SimpleNamespace(condition_vars=("age",), target_vars=("income",))

    def synthesize(self, seed_data, synthesis_variables=None):
        _ = synthesis_variables
        self._log("synthesize")
        return seed_data.assign(weight=[1.0]), None, {"backend": "bootstrap"}

    def ensure_target_support(self, synthetic_data, seed_data, targets):
        _ = seed_data
        _ = targets
        self._log("ensure_target_support")
        return synthetic_data

    def build_policyengine_entity_tables(self, population):
        _ = population
        self._log(
            "build_policyengine_entity_tables"
            if self.stage_log_style == "short"
            else "build_policyengine_tables"
        )
        if self.stage_log is not None:
            return PolicyEngineUSEntityTableBundle(
                households=pd.DataFrame({"household_id": [1], "household_weight": [1.0]}),
                persons=None,
                tax_units=None,
                spm_units=None,
                families=None,
                marital_units=None,
            )
        return SimpleNamespace(households=pd.DataFrame({"household_id": [1]}))

    def calibrate_policyengine_tables(self, tables):
        _ = tables
        self._log("calibrate_policyengine_tables")
        if self.stage_log is not None:
            return (
                PolicyEngineUSEntityTableBundle(
                    households=pd.DataFrame({"household_id": [1], "household_weight": [1.0]}),
                    persons=None,
                    tax_units=None,
                    spm_units=None,
                    families=None,
                    marital_units=None,
                ),
                pd.DataFrame({"weight": [1.0]}),
                {"backend": "policyengine_db_entropy"},
            )
        return (
            SimpleNamespace(households=pd.DataFrame({"household_id": [1]})),
            pd.DataFrame({"weight": [1.0]}),
            {"backend": "policyengine_db_entropy"},
        )

    def export_policyengine_dataset(
        self,
        result,
        path,
        *,
        period=None,
        direct_override_variables=None,
    ):
        _ = result
        _ = period
        self._log(f"export_policyengine_dataset:{tuple(direct_override_variables or ())}")
        path.write_text("stub")
        return path


def _patch_fake_harness(
    monkeypatch,
    *,
    stage_log: list[str] | None = None,
    stage_log_style: str = "full",
) -> None:
    monkeypatch.setattr(
        "microplex_us.pipelines.performance.USMicroplexPipeline",
        lambda config=None: _FakePipeline(
            config=config,
            stage_log=stage_log,
            stage_log_style=stage_log_style,
        ),
    )
    monkeypatch.setattr(
        "microplex_us.pipelines.performance.FusionPlan.from_sources",
        lambda sources: SimpleNamespace(source_names=tuple(source.name for source in sources)),
    )


def test_default_fast_calibration_target_variables_prefers_income_tax_over_agi():
    assert default_fast_calibration_target_variables(
        ("adjusted_gross_income", "income_tax", "dividend_income")
    ) == ("income_tax", "dividend_income")
    assert default_fast_calibration_target_variables(
        ("adjusted_gross_income", "dividend_income")
    ) == ("adjusted_gross_income", "dividend_income")


def test_run_us_microplex_performance_harness_returns_stage_timings(monkeypatch):
    stage_log: list[str] = []
    cache_refs: list[object] = []
    _patch_fake_harness(
        monkeypatch,
        stage_log=stage_log,
        stage_log_style="short",
    )
    monkeypatch.setattr(
        "microplex_us.pipelines.performance.PolicyEngineUSDBTargetProvider",
        lambda path: SimpleNamespace(path=path),
    )
    monkeypatch.setattr(
        "microplex_us.pipelines.performance.default_policyengine_us_db_harness_slices",
        lambda **kwargs: (SimpleNamespace(name="all_targets", query=SimpleNamespace(period=kwargs["period"])),),
    )
    monkeypatch.setattr(
        "microplex_us.pipelines.performance.filter_nonempty_policyengine_us_harness_slices",
        lambda provider, slices, cache=None: cache_refs.append(cache) or tuple(slices),
    )
    monkeypatch.setattr(
        "microplex_us.pipelines.performance.evaluate_policyengine_us_harness",
        lambda *args, **kwargs: cache_refs.append(kwargs.get("cache")) or _FakeHarnessRun(),
    )
    comparison_cache = PolicyEngineUSComparisonCache()

    result = run_us_microplex_performance_harness(
        providers=[_DummyProvider("cps"), _DummyProvider("puf")],
        config=USMicroplexPerformanceHarnessConfig(
            targets_db="/tmp/policy_data.db",
            baseline_dataset="/tmp/enhanced_cps.h5",
        ),
        comparison_cache=comparison_cache,
    )

    assert result.source_names == ("cps", "puf")
    assert result.candidate_composite_parity_loss == 0.4
    assert result.baseline_composite_parity_loss == 0.5
    assert result.target_win_rate == 0.75
    assert result.slice_win_rate == 1.0
    assert result.total_seconds >= 0.0
    assert cache_refs == [comparison_cache, comparison_cache]
    assert set(result.stage_timings) >= {
        "load_frames",
        "prepare_source_inputs",
        "prepare_seed_data",
        "integrate_donor_sources",
        "build_targets",
        "resolve_synthesis_variables",
        "synthesize",
        "ensure_target_support",
        "build_policyengine_tables",
        "calibrate_policyengine_tables",
        "evaluate_parity_harness",
    }
    assert stage_log == [
        "prepare_source_input:cps",
        "prepare_source_input:puf",
        "select_scaffold",
        "prepare_seed",
        "integrate_donors",
        "build_targets",
        "resolve_synthesis_variables",
        "synthesize",
        "ensure_target_support",
        "build_policyengine_entity_tables",
        "calibrate_policyengine_tables",
    ]


def test_run_us_microplex_performance_harness_requires_targets_db_and_baseline_for_parity():
    try:
        run_us_microplex_performance_harness(
            providers=[_DummyProvider("cps")],
            config=USMicroplexPerformanceHarnessConfig(),
        )
    except ValueError as exc:
        assert "requires both targets_db and baseline_dataset" in str(exc)
    else:
        raise AssertionError("Expected ValueError when parity inputs are missing")


def test_run_us_microplex_performance_harness_can_skip_parity(monkeypatch):
    _patch_fake_harness(monkeypatch)

    result = run_us_microplex_performance_harness(
        providers=[_DummyProvider("cps")],
        config=USMicroplexPerformanceHarnessConfig(evaluate_parity=False),
    )

    assert result.parity_run is None
    assert "evaluate_parity_harness" not in result.stage_timings


def test_run_us_microplex_performance_harness_can_enable_fast_calibration_targets(
    monkeypatch,
):
    _patch_fake_harness(monkeypatch)

    result = run_us_microplex_performance_harness(
        providers=[_DummyProvider("cps")],
        config=USMicroplexPerformanceHarnessConfig(
            evaluate_parity=False,
            fast_inner_loop_calibration=True,
            target_variables=("adjusted_gross_income", "income_tax", "dividend_income"),
        ),
    )

    assert result.build_config.policyengine_target_variables == (
        "adjusted_gross_income",
        "income_tax",
        "dividend_income",
    )
    assert result.build_config.policyengine_calibration_target_variables == (
        "income_tax",
        "dividend_income",
    )


def test_run_us_microplex_performance_harness_can_keep_exact_calibration_targets(
    monkeypatch,
):
    _patch_fake_harness(monkeypatch)

    result = run_us_microplex_performance_harness(
        providers=[_DummyProvider("cps")],
        config=USMicroplexPerformanceHarnessConfig(
            evaluate_parity=False,
            fast_inner_loop_calibration=False,
            target_variables=("adjusted_gross_income", "income_tax", "dividend_income"),
        ),
    )

    assert result.build_config.policyengine_calibration_target_variables == (
        "adjusted_gross_income",
        "income_tax",
        "dividend_income",
    )


def test_run_us_microplex_performance_harness_preserves_target_profiles(monkeypatch):
    _patch_fake_harness(monkeypatch)

    result = run_us_microplex_performance_harness(
        providers=[_DummyProvider("cps")],
        config=USMicroplexPerformanceHarnessConfig(
            evaluate_parity=False,
            target_profile="pe_native_broad",
            calibration_target_profile="pe_native_broad",
        ),
    )

    assert result.build_config.policyengine_target_profile == "pe_native_broad"
    assert result.build_config.policyengine_calibration_target_profile == "pe_native_broad"
    assert result.build_config.policyengine_target_variables == ()
    assert result.build_config.policyengine_target_geo_levels == ()
    assert result.build_config.policyengine_calibration_target_variables == ()
    assert result.build_config.policyengine_calibration_target_geo_levels == ()


def test_calibration_cache_key_includes_iteration_and_tolerance_settings():
    base = USMicroplexBuildConfig(
        calibration_backend="entropy",
        calibration_tol=1e-6,
        calibration_max_iter=100,
    )
    updated = USMicroplexBuildConfig(
        calibration_backend="entropy",
        calibration_tol=1e-5,
        calibration_max_iter=500,
    )

    assert _precalibration_build_config_key(base) == _precalibration_build_config_key(
        updated
    )
    assert _calibration_build_config_key(base) != _calibration_build_config_key(
        updated
    )


def test_calibration_cache_key_includes_household_budget_selection():
    base = USMicroplexBuildConfig(
        calibration_backend="entropy",
        policyengine_selection_household_budget=None,
    )
    updated = USMicroplexBuildConfig(
        calibration_backend="entropy",
        policyengine_selection_household_budget=29_999,
    )

    assert _precalibration_build_config_key(base) == _precalibration_build_config_key(
        updated
    )
    assert _calibration_build_config_key(base) != _calibration_build_config_key(
        updated
    )


def test_calibration_cache_key_includes_pe_native_selection_hyperparameters():
    base = USMicroplexBuildConfig(
        calibration_backend="entropy",
        policyengine_selection_backend="pe_native_loss",
        policyengine_selection_household_budget=29_999,
        policyengine_selection_max_iter=200,
        policyengine_selection_tol=1e-8,
        policyengine_selection_l2_penalty=0.0,
    )
    updated = USMicroplexBuildConfig(
        calibration_backend="entropy",
        policyengine_selection_backend="pe_native_loss",
        policyengine_selection_household_budget=29_999,
        policyengine_selection_max_iter=1_000,
        policyengine_selection_tol=1e-7,
        policyengine_selection_l2_penalty=1e-5,
    )

    assert _precalibration_build_config_key(base) == _precalibration_build_config_key(
        updated
    )
    assert _calibration_build_config_key(base) != _calibration_build_config_key(
        updated
    )


def test_run_us_microplex_performance_harness_allows_full_source_queries(monkeypatch):
    captured_queries: list[dict[str, object]] = []
    _patch_fake_harness(monkeypatch)

    class CapturingProvider(_DummyProvider):
        def load_frame(self, query=None):
            provider_filters = dict(getattr(query, "provider_filters", {}) or {})
            captured_queries.append(provider_filters)
            return super().load_frame(query)

    run_us_microplex_performance_harness(
        providers=[CapturingProvider("cps")],
        config=USMicroplexPerformanceHarnessConfig(
            sample_n=None,
            evaluate_parity=False,
        ),
    )

    assert captured_queries == [{"sample_n": None, "random_seed": 42}]


def test_run_us_microplex_performance_harness_can_evaluate_native_loss(monkeypatch):
    _patch_fake_harness(monkeypatch)
    monkeypatch.setattr(
        "microplex_us.pipelines.performance.compute_us_pe_native_scores",
        lambda **kwargs: {
            "summary": {
                "candidate_enhanced_cps_native_loss": 0.2,
                "baseline_enhanced_cps_native_loss": 0.3,
                "enhanced_cps_native_loss_delta": -0.1,
            },
            "kwargs": kwargs,
        },
    )

    result = run_us_microplex_performance_harness(
        providers=[_DummyProvider("cps")],
        config=USMicroplexPerformanceHarnessConfig(
            evaluate_parity=False,
            evaluate_pe_native_loss=True,
            baseline_dataset="/tmp/enhanced_cps.h5",
            policyengine_us_data_repo="/tmp/policyengine-us-data",
        ),
    )

    assert result.candidate_enhanced_cps_native_loss == 0.2
    assert result.baseline_enhanced_cps_native_loss == 0.3
    assert result.enhanced_cps_native_loss_delta == -0.1
    assert "evaluate_pe_native_loss" in result.stage_timings


def test_run_us_microplex_performance_harness_can_evaluate_matched_native_loss(
    monkeypatch,
    tmp_path,
):
    _patch_fake_harness(monkeypatch)
    matched_calls: list[dict[str, object]] = []
    score_calls: list[dict[str, object]] = []

    def _fake_write_matched_baseline(
        baseline_dataset_path,
        output_dataset_path,
        *,
        period,
        household_count,
        random_seed,
    ):
        matched_calls.append(
            {
                "baseline_dataset_path": baseline_dataset_path,
                "period": period,
                "household_count": household_count,
                "random_seed": random_seed,
            }
        )
        path = Path(output_dataset_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("matched")
        return str(path.resolve())

    def _fake_score(**kwargs):
        score_calls.append(kwargs)
        return {
            "summary": {
                "candidate_enhanced_cps_native_loss": 0.18,
                "baseline_enhanced_cps_native_loss": 0.22,
                "enhanced_cps_native_loss_delta": -0.04,
            }
        }

    monkeypatch.setattr(
        "microplex_us.pipelines.performance._write_matched_policyengine_us_baseline_dataset",
        _fake_write_matched_baseline,
    )
    monkeypatch.setattr(
        "microplex_us.pipelines.performance.compute_us_pe_native_scores",
        _fake_score,
    )

    baseline_output = tmp_path / "matched_baseline.h5"
    result = run_us_microplex_performance_harness(
        providers=[_DummyProvider("cps")],
        config=USMicroplexPerformanceHarnessConfig(
            evaluate_parity=False,
            evaluate_matched_pe_native_loss=True,
            baseline_dataset="/tmp/enhanced_cps.h5",
            policyengine_us_data_repo="/tmp/policyengine-us-data",
            output_matched_baseline_dataset_path=baseline_output,
        ),
    )

    assert matched_calls == [
        {
            "baseline_dataset_path": "/tmp/enhanced_cps.h5",
            "period": 2024,
            "household_count": 1,
            "random_seed": 42,
        }
    ]
    assert score_calls[0]["baseline_dataset_path"] == str(baseline_output.resolve())
    assert result.matched_pe_native_scores is not None
    assert result.matched_baseline_dataset_path == str(baseline_output.resolve())
    assert "build_matched_baseline_dataset" in result.stage_timings
    assert "evaluate_matched_pe_native_loss" in result.stage_timings


def test_run_us_microplex_performance_harness_can_reweight_matched_native_loss(
    monkeypatch,
    tmp_path,
):
    _patch_fake_harness(monkeypatch)
    matched_calls: list[dict[str, object]] = []
    reweight_calls: list[dict[str, object]] = []
    score_calls: list[dict[str, object]] = []

    def _fake_write_matched_baseline(
        baseline_dataset_path,
        output_dataset_path,
        *,
        period,
        household_count,
        random_seed,
    ):
        matched_calls.append(
            {
                "baseline_dataset_path": baseline_dataset_path,
                "period": period,
                "household_count": household_count,
                "random_seed": random_seed,
            }
        )
        path = Path(output_dataset_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("matched")
        return str(path.resolve())

    def _fake_reweight(
        input_dataset_path,
        output_dataset_path,
        *,
        period,
        epochs,
        l0_lambda,
        seed,
        policyengine_us_data_repo,
    ):
        reweight_calls.append(
            {
                "input_dataset_path": input_dataset_path,
                "output_dataset_path": str(Path(output_dataset_path)),
                "period": period,
                "epochs": epochs,
                "l0_lambda": l0_lambda,
                "seed": seed,
                "policyengine_us_data_repo": policyengine_us_data_repo,
            }
        )
        path = Path(output_dataset_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("reweighted")
        return str(path.resolve())

    def _fake_score(**kwargs):
        score_calls.append(kwargs)
        return {
            "summary": {
                "candidate_enhanced_cps_native_loss": 0.17,
                "baseline_enhanced_cps_native_loss": 0.21,
                "enhanced_cps_native_loss_delta": -0.04,
            }
        }

    monkeypatch.setattr(
        "microplex_us.pipelines.performance._write_matched_policyengine_us_baseline_dataset",
        _fake_write_matched_baseline,
    )
    monkeypatch.setattr(
        "microplex_us.pipelines.performance._reweight_matched_policyengine_us_baseline_dataset",
        _fake_reweight,
    )
    monkeypatch.setattr(
        "microplex_us.pipelines.performance.compute_us_pe_native_scores",
        _fake_score,
    )

    baseline_output = tmp_path / "matched_baseline_reweighted.h5"
    result = run_us_microplex_performance_harness(
        providers=[_DummyProvider("cps")],
        config=USMicroplexPerformanceHarnessConfig(
            evaluate_parity=False,
            evaluate_matched_pe_native_loss=True,
            reweight_matched_pe_native_loss=True,
            matched_baseline_reweight_epochs=300,
            matched_baseline_reweight_l0_lambda=1e-6,
            matched_baseline_reweight_seed=123,
            baseline_dataset="/tmp/enhanced_cps.h5",
            policyengine_us_data_repo="/tmp/policyengine-us-data",
            output_matched_baseline_dataset_path=baseline_output,
        ),
    )

    assert len(matched_calls) == 1
    assert len(reweight_calls) == 1
    assert reweight_calls[0]["epochs"] == 300
    assert reweight_calls[0]["l0_lambda"] == 1e-6
    assert reweight_calls[0]["seed"] == 123
    assert score_calls[0]["baseline_dataset_path"] == str(baseline_output.resolve())
    assert result.matched_baseline_dataset_path == str(baseline_output.resolve())
    assert "reweight_matched_baseline_dataset" in result.stage_timings


def test_write_matched_policyengine_us_baseline_dataset_preserves_variables(
    tmp_path,
):
    baseline_path = tmp_path / "baseline.h5"
    matched_path = tmp_path / "matched.h5"
    full_copy_path = tmp_path / "matched_full.h5"

    write_policyengine_us_time_period_dataset(
        {
            "household_id": {"2024": [10, 20]},
            "household_weight": {"2024": [1.5, 2.5]},
            "person_id": {"2024": [1, 2, 3]},
            "person_household_id": {"2024": [10, 10, 20]},
            "person_weight": {"2024": [1.5, 1.5, 2.5]},
            "tax_unit_id": {"2024": [100, 200]},
            "person_tax_unit_id": {"2024": [100, 100, 200]},
            "tax_unit_weight": {"2024": [1.5, 2.5]},
            "state_code": {"2024": [1, 2]},
            "age": {"2024": [34, 12, 45]},
            "employment_income": {"2024": [100.0, 0.0, 55.0]},
        },
        baseline_path,
    )

    matched_dataset_path = _write_matched_policyengine_us_baseline_dataset(
        baseline_path,
        matched_path,
        period=2024,
        household_count=1,
        random_seed=42,
    )
    matched_tables = load_policyengine_us_entity_tables(matched_dataset_path, period=2024)
    assert "state_code" in matched_tables.households.columns
    assert "age" in matched_tables.persons.columns
    assert "employment_income" in matched_tables.persons.columns

    copied_dataset_path = _write_matched_policyengine_us_baseline_dataset(
        baseline_path,
        full_copy_path,
        period=2024,
        household_count=2,
        random_seed=42,
    )
    assert Path(copied_dataset_path).read_bytes() == baseline_path.read_bytes()


def test_sample_matched_household_ids_supports_weighted_methods():
    household_ids = np.asarray([10, 20, 30])
    weights = np.asarray([0.0, 0.0, 5.0])

    assert _sample_matched_household_ids(
        household_ids,
        weights,
        household_count=1,
        random_seed=42,
        sample_method="weight_proportional",
    ).tolist() == [30]
    assert _sample_matched_household_ids(
        household_ids,
        np.asarray([1.0, 9.0, 2.0]),
        household_count=2,
        random_seed=42,
        sample_method="largest_weight",
    ).tolist() == [20, 30]


def test_run_us_microplex_performance_harness_can_write_output_bundle(monkeypatch, tmp_path):
    _patch_fake_harness(monkeypatch)

    result_path = tmp_path / "result.json"
    dataset_path = tmp_path / "candidate.h5"

    result = run_us_microplex_performance_harness(
        providers=[_DummyProvider("cps")],
        config=USMicroplexPerformanceHarnessConfig(
            evaluate_parity=False,
            output_json_path=result_path,
            output_policyengine_dataset_path=dataset_path,
        ),
    )

    assert result_path.exists()
    assert dataset_path.exists()
    assert result.policyengine_dataset_path == str(dataset_path)

    payload = json.loads(result_path.read_text())
    assert payload["policyengine_dataset_path"] == str(dataset_path)
    assert payload["source_names"] == ["cps"]
    assert payload["calibration_summary"]["backend"] == "policyengine_db_entropy"


def test_run_us_microplex_performance_harness_can_write_pe_native_target_delta_output(
    monkeypatch,
    tmp_path,
):
    _patch_fake_harness(monkeypatch)

    delta_path = tmp_path / "target_deltas.json"
    compare_calls: list[dict[str, object]] = []

    def _fake_compare(**kwargs):
        compare_calls.append(kwargs)
        return {
            "metric": "enhanced_cps_native_loss_target_delta",
            "top_regressions": [{"target_name": "nation/foo", "weighted_term_delta": 0.5}],
            "top_improvements": [{"target_name": "state/bar", "weighted_term_delta": -0.25}],
        }

    monkeypatch.setattr(
        "microplex_us.pipelines.performance.compare_us_pe_native_target_deltas",
        _fake_compare,
    )

    result = run_us_microplex_performance_harness(
        providers=[_DummyProvider("cps")],
        config=USMicroplexPerformanceHarnessConfig(
            evaluate_parity=False,
            baseline_dataset="/tmp/enhanced_cps.h5",
            policyengine_us_data_repo="/tmp/policyengine-us-data",
            output_pe_native_target_delta_path=delta_path,
            pe_native_target_delta_top_k=7,
        ),
    )

    assert result.pe_native_target_deltas is not None
    assert delta_path.exists()
    assert compare_calls[0]["from_dataset_path"] == "/tmp/enhanced_cps.h5"
    assert compare_calls[0]["top_k"] == 7
    payload = json.loads(delta_path.read_text())
    assert payload["metric"] == "enhanced_cps_native_loss_target_delta"
    assert payload["top_regressions"][0]["target_name"] == "nation/foo"
    assert "evaluate_pe_native_target_deltas" in result.stage_timings
    assert "write_pe_native_target_delta_json" in result.stage_timings


def test_run_us_microplex_performance_harness_can_write_pe_native_support_audit_output(
    monkeypatch,
    tmp_path,
):
    _patch_fake_harness(monkeypatch)

    audit_path = tmp_path / "support_audit.json"
    audit_calls: list[dict[str, object]] = []

    def _fake_audit(**kwargs):
        audit_calls.append(kwargs)
        return {
            "metric": "enhanced_cps_support_audit",
            "comparisons": {
                "critical_input_support": [
                    {
                        "variable": "child_support_expense",
                        "candidate_stored": False,
                        "baseline_stored": True,
                    }
                ]
            },
        }

    monkeypatch.setattr(
        "microplex_us.pipelines.performance.compute_us_pe_native_support_audit",
        _fake_audit,
    )

    result = run_us_microplex_performance_harness(
        providers=[_DummyProvider("cps")],
        config=USMicroplexPerformanceHarnessConfig(
            evaluate_parity=False,
            baseline_dataset="/tmp/enhanced_cps.h5",
            policyengine_us_data_repo="/tmp/policyengine-us-data",
            output_pe_native_support_audit_path=audit_path,
        ),
    )

    assert result.pe_native_support_audit is not None
    assert audit_path.exists()
    assert audit_calls[0]["baseline_dataset_path"] == "/tmp/enhanced_cps.h5"
    payload = json.loads(audit_path.read_text())
    assert payload["metric"] == "enhanced_cps_support_audit"
    assert (
        payload["comparisons"]["critical_input_support"][0]["variable"]
        == "child_support_expense"
    )
    assert "evaluate_pe_native_support_audit" in result.stage_timings
    assert "write_pe_native_support_audit_json" in result.stage_timings


def test_run_us_microplex_performance_harness_passes_export_direct_overrides(monkeypatch):
    stage_log: list[str] = []
    _patch_fake_harness(monkeypatch, stage_log=stage_log)
    monkeypatch.setattr(
        "microplex_us.pipelines.performance.compute_us_pe_native_scores",
        lambda **kwargs: {"summary": {}, "kwargs": kwargs},
    )

    run_us_microplex_performance_harness(
        providers=[_DummyProvider("cps")],
        config=USMicroplexPerformanceHarnessConfig(
            evaluate_parity=False,
            evaluate_pe_native_loss=True,
            baseline_dataset="/tmp/enhanced_cps.h5",
            policyengine_us_data_repo="/tmp/policyengine-us-data",
            build_config=USMicroplexBuildConfig(
                policyengine_direct_override_variables=("filing_status", "snap")
            ),
        ),
    )

    assert "export_policyengine_dataset:('filing_status', 'snap')" in stage_log


def test_run_us_microplex_performance_harness_can_optimize_native_loss(monkeypatch):
    stage_log: list[str] = []
    _patch_fake_harness(monkeypatch, stage_log=stage_log)
    optimization_calls: list[dict[str, object]] = []
    score_calls: list[dict[str, object]] = []

    def _fake_optimize(**kwargs):
        optimization_calls.append(kwargs)
        Path(kwargs["output_dataset_path"]).write_text("optimized")
        return PolicyEngineUSNativeWeightOptimizationResult(
            metric="enhanced_cps_native_loss_weight_optimization",
            period=2024,
            input_dataset=str(kwargs["input_dataset_path"]),
            output_dataset=str(Path(kwargs["output_dataset_path"]).resolve()),
            initial_loss=0.4,
            optimized_loss=0.2,
            loss_delta=-0.2,
            initial_weight_sum=10.0,
            optimized_weight_sum=10.0,
            household_count=3,
            positive_household_count=2,
            budget=2,
            converged=True,
            iterations=12,
            target_names=("nation/foo", "state/bar"),
        )

    def _fake_score(**kwargs):
        score_calls.append(kwargs)
        return {
            "summary": {
                "candidate_enhanced_cps_native_loss": 0.2,
                "baseline_enhanced_cps_native_loss": 0.3,
                "enhanced_cps_native_loss_delta": -0.1,
            }
        }

    monkeypatch.setattr(
        "microplex_us.pipelines.performance.optimize_policyengine_us_native_loss_dataset",
        _fake_optimize,
    )
    monkeypatch.setattr(
        "microplex_us.pipelines.performance.compute_us_pe_native_scores",
        _fake_score,
    )

    result = run_us_microplex_performance_harness(
        providers=[_DummyProvider("cps")],
        config=USMicroplexPerformanceHarnessConfig(
            evaluate_parity=False,
            evaluate_pe_native_loss=True,
            optimize_pe_native_loss=True,
            pe_native_household_budget=2,
            pe_native_optimizer_max_iter=50,
            pe_native_optimizer_l2_penalty=0.25,
            pe_native_optimizer_tol=1e-6,
            baseline_dataset="/tmp/enhanced_cps.h5",
            policyengine_us_data_repo="/tmp/policyengine-us-data",
        ),
    )

    assert len(optimization_calls) == 1
    assert optimization_calls[0]["budget"] == 2
    assert optimization_calls[0]["max_iter"] == 50
    assert optimization_calls[0]["l2_penalty"] == 0.25
    assert optimization_calls[0]["tol"] == 1e-6
    assert str(score_calls[0]["candidate_dataset_path"]).endswith(
        "candidate_policyengine_us_optimized.h5"
    )
    assert result.pe_native_scores is not None
    assert result.pe_native_scores["optimization"]["optimized_loss"] == 0.2
    assert result.pe_native_scores["optimization"]["rescored_loss_abs_error"] == 0.0
    assert "optimize_pe_native_loss_weights" in result.stage_timings


def test_run_us_microplex_performance_harness_writes_optimized_dataset_output(
    monkeypatch,
    tmp_path,
):
    _patch_fake_harness(monkeypatch)

    def _fake_optimize(**kwargs):
        Path(kwargs["output_dataset_path"]).write_text("optimized")
        return PolicyEngineUSNativeWeightOptimizationResult(
            metric="enhanced_cps_native_loss_weight_optimization",
            period=2024,
            input_dataset=str(kwargs["input_dataset_path"]),
            output_dataset=str(Path(kwargs["output_dataset_path"]).resolve()),
            initial_loss=0.4,
            optimized_loss=0.2,
            loss_delta=-0.2,
            initial_weight_sum=10.0,
            optimized_weight_sum=10.0,
            household_count=3,
            positive_household_count=2,
            budget=2,
            converged=True,
            iterations=12,
            target_names=("nation/foo", "state/bar"),
        )

    monkeypatch.setattr(
        "microplex_us.pipelines.performance.optimize_policyengine_us_native_loss_dataset",
        _fake_optimize,
    )
    monkeypatch.setattr(
        "microplex_us.pipelines.performance.compute_us_pe_native_scores",
        lambda **kwargs: {
            "summary": {
                "candidate_enhanced_cps_native_loss": 0.2,
                "baseline_enhanced_cps_native_loss": 0.3,
                "enhanced_cps_native_loss_delta": -0.1,
            }
        },
    )

    dataset_path = tmp_path / "candidate_optimized.h5"
    result = run_us_microplex_performance_harness(
        providers=[_DummyProvider("cps")],
        config=USMicroplexPerformanceHarnessConfig(
            evaluate_parity=False,
            evaluate_pe_native_loss=True,
            optimize_pe_native_loss=True,
            pe_native_household_budget=2,
            baseline_dataset="/tmp/enhanced_cps.h5",
            policyengine_us_data_repo="/tmp/policyengine-us-data",
            output_policyengine_dataset_path=dataset_path,
        ),
    )

    assert result.policyengine_dataset_path == str(dataset_path.resolve())
    assert dataset_path.read_text() == "optimized"


def test_run_us_microplex_performance_harness_rejects_native_optimization_without_scoring(
    monkeypatch,
):
    _patch_fake_harness(monkeypatch)

    try:
        run_us_microplex_performance_harness(
            providers=[_DummyProvider("cps")],
            config=USMicroplexPerformanceHarnessConfig(
                evaluate_parity=False,
                evaluate_pe_native_loss=False,
                optimize_pe_native_loss=True,
            ),
        )
    except ValueError as exc:
        assert "evaluate_pe_native_loss" in str(exc)
    else:  # pragma: no cover - defensive assertion
        raise AssertionError("expected optimize_pe_native_loss validation error")


def test_run_us_microplex_performance_harness_rejects_native_loss_mismatch(monkeypatch):
    _patch_fake_harness(monkeypatch)

    monkeypatch.setattr(
        "microplex_us.pipelines.performance.optimize_policyengine_us_native_loss_dataset",
        lambda **kwargs: PolicyEngineUSNativeWeightOptimizationResult(
            metric="enhanced_cps_native_loss_weight_optimization",
            period=2024,
            input_dataset=str(kwargs["input_dataset_path"]),
            output_dataset=str(Path(kwargs["output_dataset_path"]).resolve()),
            initial_loss=0.4,
            optimized_loss=0.2,
            loss_delta=-0.2,
            initial_weight_sum=10.0,
            optimized_weight_sum=10.0,
            household_count=3,
            positive_household_count=2,
            budget=None,
            converged=True,
            iterations=12,
            target_names=("nation/foo", "state/bar"),
        ),
    )
    monkeypatch.setattr(
        "microplex_us.pipelines.performance.compute_us_pe_native_scores",
        lambda **kwargs: {
            "summary": {
                "candidate_enhanced_cps_native_loss": 0.25,
                "baseline_enhanced_cps_native_loss": 0.3,
                "enhanced_cps_native_loss_delta": -0.05,
            }
        },
    )

    try:
        run_us_microplex_performance_harness(
            providers=[_DummyProvider("cps")],
            config=USMicroplexPerformanceHarnessConfig(
                evaluate_parity=False,
                evaluate_pe_native_loss=True,
                optimize_pe_native_loss=True,
                pe_native_score_consistency_tol=1e-6,
                baseline_dataset="/tmp/enhanced_cps.h5",
                policyengine_us_data_repo="/tmp/policyengine-us-data",
            ),
        )
    except ValueError as exc:
        assert "does not match rescored loss" in str(exc)
    else:  # pragma: no cover - defensive assertion
        raise AssertionError("expected PE-native loss consistency validation error")


def test_run_us_microplex_performance_harness_rejects_nonpositive_target_delta_top_k(
    monkeypatch,
):
    _patch_fake_harness(monkeypatch)

    try:
        run_us_microplex_performance_harness(
            providers=[_DummyProvider("cps")],
            config=USMicroplexPerformanceHarnessConfig(
                evaluate_parity=False,
                pe_native_target_delta_top_k=0,
            ),
        )
    except ValueError as exc:
        assert "pe_native_target_delta_top_k" in str(exc)
    else:  # pragma: no cover - defensive assertion
        raise AssertionError("expected target delta top-k validation error")


def test_run_us_microplex_performance_harness_rejects_nonpositive_matched_baseline_household_count(
    monkeypatch,
):
    _patch_fake_harness(monkeypatch)

    try:
        run_us_microplex_performance_harness(
            providers=[_DummyProvider("cps")],
            config=USMicroplexPerformanceHarnessConfig(
                evaluate_parity=False,
                matched_baseline_household_count=0,
            ),
        )
    except ValueError as exc:
        assert "matched_baseline_household_count" in str(exc)
    else:  # pragma: no cover - defensive assertion
        raise AssertionError(
            "expected matched baseline household count validation error"
        )


def test_run_us_microplex_performance_harness_rejects_reweighted_matched_loss_without_matched_loss(
    monkeypatch,
):
    _patch_fake_harness(monkeypatch)

    try:
        run_us_microplex_performance_harness(
            providers=[_DummyProvider("cps")],
            config=USMicroplexPerformanceHarnessConfig(
                evaluate_parity=False,
                reweight_matched_pe_native_loss=True,
            ),
        )
    except ValueError as exc:
        assert "evaluate_matched_pe_native_loss" in str(exc)
    else:  # pragma: no cover - defensive assertion
        raise AssertionError(
            "expected reweighted matched baseline validation error"
        )


def test_warm_us_microplex_parity_cache_preloads_baseline(monkeypatch):
    cache = PolicyEngineUSComparisonCache()
    load_target_set_calls: list[tuple[int, tuple[str, ...] | None]] = []
    baseline_calls: list[dict[str, object]] = []

    class FakeProvider:
        def load_target_set(self, query=None):
            period = query.period if query is not None else 2024
            names = tuple(query.names) if query is not None else ()
            load_target_set_calls.append((period, names or None))
            return TargetSet(
                [
                    TargetSpec(
                        name="policyengine_us_target_1",
                        entity=EntityType.HOUSEHOLD,
                        value=1.0,
                        period=period,
                        aggregation="count",
                    )
                ]
            )

    monkeypatch.setattr(
        "microplex_us.pipelines.performance.PolicyEngineUSDBTargetProvider",
        lambda path: FakeProvider(),
    )
    monkeypatch.setattr(
        "microplex_us.pipelines.performance.default_policyengine_us_db_harness_slices",
        lambda **kwargs: (
            SimpleNamespace(
                name="all_targets",
                query=TargetQuery(period=kwargs["period"]),
            ),
        ),
    )
    monkeypatch.setattr(
        "microplex_us.pipelines.performance.PolicyEngineUSComparisonCache.load_baseline_report",
        lambda self, **kwargs: baseline_calls.append(kwargs) or SimpleNamespace(),
    )

    warmed_cache = warm_us_microplex_parity_cache(
        config=USMicroplexPerformanceHarnessConfig(
            targets_db="/tmp/policy_data.db",
            baseline_dataset="/tmp/enhanced_cps.h5",
        ),
        comparison_cache=cache,
    )

    assert warmed_cache is cache
    assert load_target_set_calls == [(2024, None)]
    assert baseline_calls
    assert baseline_calls[0]["baseline_dataset"] == "/tmp/enhanced_cps.h5"


def test_warm_us_microplex_parity_cache_uses_resolved_scope_for_named_profile(monkeypatch):
    cache = PolicyEngineUSComparisonCache()
    slice_kwargs: dict[str, object] = {}
    baseline_calls: list[dict[str, object]] = []

    class FakeProvider:
        def load_target_set(self, query=None):
            period = query.period if query is not None else 2024
            return TargetSet(
                [
                    TargetSpec(
                        name="policyengine_us_target_1",
                        entity=EntityType.HOUSEHOLD,
                        value=1.0,
                        period=period,
                        aggregation="count",
                    )
                ]
            )

    monkeypatch.setattr(
        "microplex_us.pipelines.performance.PolicyEngineUSDBTargetProvider",
        lambda path: FakeProvider(),
    )
    monkeypatch.setattr(
        "microplex_us.pipelines.performance.default_policyengine_us_db_harness_slices",
        lambda **kwargs: slice_kwargs.update(kwargs)
        or (
            SimpleNamespace(
                name="all_targets",
                query=TargetQuery(period=kwargs["period"]),
            ),
        ),
    )
    monkeypatch.setattr(
        "microplex_us.pipelines.performance.PolicyEngineUSComparisonCache.load_baseline_report",
        lambda self, **kwargs: baseline_calls.append(kwargs) or SimpleNamespace(),
    )

    warm_us_microplex_parity_cache(
        config=USMicroplexPerformanceHarnessConfig(
            targets_db="/tmp/policy_data.db",
            baseline_dataset="/tmp/enhanced_cps.h5",
            target_profile="pe_native_broad",
            calibration_target_profile="pe_native_broad",
            build_config=USMicroplexBuildConfig(
                policyengine_target_profile="pe_native_broad",
                policyengine_calibration_target_profile="pe_native_broad",
            ),
        ),
        comparison_cache=cache,
    )

    assert slice_kwargs["variables"] == ()
    assert slice_kwargs["domain_variables"] == ()
    assert slice_kwargs["geo_levels"] == ()
    assert baseline_calls


def test_us_microplex_performance_session_reuses_comparison_cache(monkeypatch):
    session = USMicroplexPerformanceSession()
    run_calls: list[tuple[PolicyEngineUSComparisonCache, object, object, object]] = []

    def fake_run(
        providers,
        *,
        config,
        queries=None,
        comparison_cache=None,
        frame_cache=None,
        precalibration_cache=None,
        calibration_cache=None,
    ):
        _ = providers
        _ = config
        _ = queries
        run_calls.append(
            (
                comparison_cache,
                frame_cache,
                precalibration_cache,
                calibration_cache,
            )
        )
        return "ok"

    monkeypatch.setattr(
        "microplex_us.pipelines.performance.run_us_microplex_performance_harness",
        fake_run,
    )

    result = session.run(
        [_DummyProvider("cps")],
        config=USMicroplexPerformanceHarnessConfig(evaluate_parity=False),
    )

    assert result == "ok"
    assert run_calls == [
        (
            session.comparison_cache,
            session.frame_cache,
            session.precalibration_cache,
            session.calibration_cache,
        )
    ]


def test_us_microplex_performance_session_run_batch_uses_native_batch_scorer(
    monkeypatch,
    tmp_path,
):
    session = USMicroplexPerformanceSession()
    run_configs: list[USMicroplexPerformanceHarnessConfig] = []
    batch_calls: list[dict[str, object]] = []

    fake_build_result = SimpleNamespace(calibration_summary={"backend": "entropy"})
    fake_build_config = USMicroplexBuildConfig()

    def fake_run(
        providers,
        *,
        config,
        queries=None,
        comparison_cache=None,
        frame_cache=None,
        precalibration_cache=None,
        calibration_cache=None,
    ):
        _ = providers
        _ = queries
        _ = comparison_cache
        _ = frame_cache
        _ = precalibration_cache
        _ = calibration_cache
        run_configs.append(config)
        dataset_path = Path(config.output_policyengine_dataset_path)
        dataset_path.parent.mkdir(parents=True, exist_ok=True)
        dataset_path.write_text("stub")
        return USMicroplexPerformanceHarnessResult(
            config=config,
            build_config=fake_build_config,
            build_result=fake_build_result,
            source_names=("cps",),
            stage_timings={"load_frames": 0.0},
            total_seconds=0.0,
            parity_run=None,
            pe_native_scores=None,
            pe_native_target_deltas=None,
            policyengine_dataset_path=str(dataset_path),
        )

    def fake_batch_score(**kwargs):
        batch_calls.append(kwargs)
        return [
            {
                "summary": {
                    "candidate_enhanced_cps_native_loss": 0.2,
                    "baseline_enhanced_cps_native_loss": 0.3,
                    "enhanced_cps_native_loss_delta": -0.1,
                },
                "timing": {
                    "batch_elapsed_seconds": 1.25,
                    "batch_candidate_count": 2,
                },
            },
            {
                "summary": {
                    "candidate_enhanced_cps_native_loss": 0.25,
                    "baseline_enhanced_cps_native_loss": 0.3,
                    "enhanced_cps_native_loss_delta": -0.05,
                },
                "timing": {
                    "batch_elapsed_seconds": 1.25,
                    "batch_candidate_count": 2,
                },
            },
        ]

    monkeypatch.setattr(
        "microplex_us.pipelines.performance.run_us_microplex_performance_harness",
        fake_run,
    )
    monkeypatch.setattr(
        "microplex_us.pipelines.performance.compute_batch_us_pe_native_scores",
        fake_batch_score,
    )

    requests = (
        USMicroplexPerformanceHarnessRequest(
            providers=(_DummyProvider("cps"),),
            config=USMicroplexPerformanceHarnessConfig(
                evaluate_parity=False,
                evaluate_pe_native_loss=True,
                baseline_dataset="/tmp/enhanced_cps.h5",
                policyengine_us_data_repo="/tmp/policyengine-us-data",
                output_policyengine_dataset_path=tmp_path / "candidate_a.h5",
            ),
        ),
        USMicroplexPerformanceHarnessRequest(
            providers=(_DummyProvider("cps"),),
            config=USMicroplexPerformanceHarnessConfig(
                evaluate_parity=False,
                evaluate_pe_native_loss=True,
                baseline_dataset="/tmp/enhanced_cps.h5",
                policyengine_us_data_repo="/tmp/policyengine-us-data",
                output_policyengine_dataset_path=tmp_path / "candidate_b.h5",
            ),
        ),
    )

    results = session.run_batch(requests)

    assert len(run_configs) == 2
    assert all(config.evaluate_pe_native_loss is False for config in run_configs)
    assert len(batch_calls) == 1
    assert batch_calls[0]["baseline_dataset_path"] == "/tmp/enhanced_cps.h5"
    assert batch_calls[0]["candidate_dataset_paths"] == [
        str(tmp_path / "candidate_a.h5"),
        str(tmp_path / "candidate_b.h5"),
    ]
    assert results[0].pe_native_scores["summary"]["candidate_enhanced_cps_native_loss"] == 0.2
    assert results[1].pe_native_scores["summary"]["candidate_enhanced_cps_native_loss"] == 0.25
    assert results[0].stage_timings["evaluate_pe_native_loss"] == 1.25


def test_us_microplex_performance_session_reuses_loaded_frames(monkeypatch):
    session = USMicroplexPerformanceSession()
    load_calls: list[str] = []

    class CountingProvider(_DummyProvider):
        def load_frame(self, query=None):
            _ = query
            load_calls.append(self.descriptor.name)
            return SimpleNamespace(source=SimpleNamespace(name=self.descriptor.name))
    _patch_fake_harness(monkeypatch)

    provider = CountingProvider("cps")
    config = USMicroplexPerformanceHarnessConfig(evaluate_parity=False)

    first = session.run([provider], config=config)
    second = session.run([provider], config=config)

    assert first.source_names == ("cps",)
    assert second.source_names == ("cps",)
    assert load_calls == ["cps"]


def test_us_microplex_performance_session_reuses_precalibration_state(monkeypatch):
    stage_calls: list[str] = []

    class CountingProvider(_DummyProvider):
        def load_frame(self, query=None):
            _ = query
            return SimpleNamespace(source=SimpleNamespace(name=self.descriptor.name))
    _patch_fake_harness(monkeypatch, stage_log=stage_calls)

    provider = CountingProvider("cps")
    config = USMicroplexPerformanceHarnessConfig(evaluate_parity=False)

    frame_cache = {}
    precalibration_cache = {}

    first = run_us_microplex_performance_harness(
        [provider],
        config=config,
        frame_cache=frame_cache,
        precalibration_cache=precalibration_cache,
        calibration_cache=None,
    )
    second = run_us_microplex_performance_harness(
        [provider],
        config=config,
        frame_cache=frame_cache,
        precalibration_cache=precalibration_cache,
        calibration_cache=None,
    )

    assert first.source_names == ("cps",)
    assert second.source_names == ("cps",)
    assert stage_calls.count("prepare_source_input") == 1
    assert stage_calls.count("prepare_seed_data") == 1
    assert stage_calls.count("integrate_donor_sources") == 1
    assert stage_calls.count("build_targets") == 1
    assert stage_calls.count("resolve_synthesis_variables") == 1
    assert stage_calls.count("synthesize") == 1
    assert stage_calls.count("ensure_target_support") == 1
    assert stage_calls.count("build_policyengine_tables") == 1
    assert stage_calls.count("calibrate_policyengine_tables") == 2


def test_us_microplex_performance_session_reuses_calibration_state(monkeypatch):
    session = USMicroplexPerformanceSession()
    stage_calls: list[str] = []

    class CountingProvider(_DummyProvider):
        def load_frame(self, query=None):
            _ = query
            return SimpleNamespace(source=SimpleNamespace(name=self.descriptor.name))
    _patch_fake_harness(monkeypatch, stage_log=stage_calls)

    provider = CountingProvider("cps")
    config = USMicroplexPerformanceHarnessConfig(evaluate_parity=False)

    first = session.run([provider], config=config)
    second = session.run([provider], config=config)

    assert first.source_names == ("cps",)
    assert second.source_names == ("cps",)
    assert stage_calls.count("prepare_source_input") == 1
    assert stage_calls.count("calibrate_policyengine_tables") == 1
    assert len(session.calibration_cache) == 1
    assert first.stage_timings["calibrate_policyengine_tables"] >= 0.0
    assert second.stage_timings["calibrate_policyengine_tables"] == 0.0
