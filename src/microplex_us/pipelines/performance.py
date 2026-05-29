"""Performance harness for iterative US microplex optimization."""

from __future__ import annotations

import json
import shutil
import subprocess
from collections.abc import Iterable
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from tempfile import TemporaryDirectory
from time import perf_counter

import h5py
import numpy as np
import pandas as pd
from microplex.core import EntityType, ObservationFrame, SourceProvider, SourceQuery
from microplex.fusion import FusionPlan
from microplex.targets import TargetSet

from microplex_us.pipelines.pe_native_optimization import (
    optimize_policyengine_us_native_loss_dataset,
    rewrite_policyengine_us_dataset_weights,
)
from microplex_us.pipelines.pe_native_scores import (
    _ENHANCED_CPS_BAD_TARGETS,
    build_policyengine_us_data_subprocess_env,
    compare_us_pe_native_target_deltas,
    compute_batch_us_pe_native_scores,
    compute_us_pe_native_scores,
    compute_us_pe_native_support_audit,
    resolve_policyengine_us_data_repo_root,
)
from microplex_us.pipelines.us import (
    USMicroplexBuildConfig,
    USMicroplexBuildResult,
    USMicroplexPipeline,
    USMicroplexTargets,
)
from microplex_us.policyengine.harness import (
    PolicyEngineUSComparisonCache,
    PolicyEngineUSHarnessRun,
    default_policyengine_us_db_harness_slices,
    evaluate_policyengine_us_harness,
    filter_nonempty_policyengine_us_harness_slices,
)
from microplex_us.policyengine.us import (
    PolicyEngineUSDBTargetProvider,
    PolicyEngineUSEntityTableBundle,
    _infer_policyengine_array_entity,
    _load_policyengine_us_period_arrays,
    _resolve_policyengine_us_tax_benefit_system,
    write_policyengine_us_time_period_dataset,
)

CacheValue = object
BuildConfigCacheKey = tuple[tuple[str, CacheValue], ...]
SourceQueryCacheKey = tuple[
    str,
    tuple[tuple[str, CacheValue], ...],
    int | str | None,
    tuple[tuple[str, CacheValue], ...],
]
PreCalibrationCacheKey = tuple[tuple[SourceQueryCacheKey, ...], BuildConfigCacheKey]
CalibrationCacheKey = tuple[PreCalibrationCacheKey, BuildConfigCacheKey]

PRECALIBRATION_STAGE_NAMES = (
    "prepare_source_inputs",
    "prepare_seed_data",
    "integrate_donor_sources",
    "build_targets",
    "resolve_synthesis_variables",
    "synthesize",
    "ensure_target_support",
    "build_policyengine_tables",
)

_MATCHED_BASELINE_REWEIGHT_SCRIPT = """
import json
import sys
from pathlib import Path

import numpy as np
from policyengine_core.data import Dataset

REPO_ROOT = sys.argv[1]
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from policyengine_us import Microsimulation
from policyengine_us_data.datasets.cps.enhanced_cps import reweight
from policyengine_us_data.utils.loss import build_loss_matrix

BAD_TARGETS = tuple(json.loads(sys.argv[2]))
PERIOD = int(sys.argv[3])
DATASET_PATH = sys.argv[4]
OUTPUT_WEIGHTS = Path(sys.argv[5])
EPOCHS = int(sys.argv[6])
L0_LAMBDA = float(sys.argv[7])
SEED = int(sys.argv[8])


def dataset_from_path(dataset_path: str, dataset_name: str):
    class LocalDataset(Dataset):
        name = dataset_name
        label = dataset_name
        file_path = dataset_path
        data_format = Dataset.TIME_PERIOD_ARRAYS
        time_period = PERIOD

    return LocalDataset


dataset_cls = dataset_from_path(
    DATASET_PATH,
    Path(DATASET_PATH).stem.replace("-", "_"),
)
sim = Microsimulation(dataset=dataset_cls)
original_weights = sim.calculate(
    "household_weight",
    map_to="household",
    period=PERIOD,
).values.astype(np.float64)
rng = np.random.default_rng(SEED)
original_weights = original_weights + rng.normal(1.0, 0.1, len(original_weights))
loss_matrix, targets_array = build_loss_matrix(dataset_cls, PERIOD)
zero_mask = np.isclose(targets_array, 0.0, atol=0.1)
bad_mask = loss_matrix.columns.isin(BAD_TARGETS)
keep_mask = ~(zero_mask | bad_mask)
loss_matrix_clean = loss_matrix.loc[:, keep_mask].astype(np.float32)
targets_array_clean = targets_array[keep_mask]
optimized_weights = reweight(
    original_weights,
    loss_matrix_clean,
    targets_array_clean,
    log_path=None,
    epochs=EPOCHS,
    l0_lambda=L0_LAMBDA,
    seed=SEED,
)
np.save(OUTPUT_WEIGHTS, optimized_weights.astype(np.float32))
""".strip()


def default_fast_calibration_target_variables(
    variables: tuple[str, ...],
) -> tuple[str, ...]:
    """Drop redundant targets when a downstream PE tax aggregate already subsumes them."""
    filtered = list(variables)
    if "income_tax" in filtered and "adjusted_gross_income" in filtered:
        filtered = [
            variable for variable in filtered if variable != "adjusted_gross_income"
        ]
    return tuple(filtered) or variables


@dataclass(frozen=True)
class USMicroplexPerformanceHarnessConfig:
    """Configuration for a repeatable local optimization harness."""

    sample_n: int | None = 100
    n_synthetic: int = 100
    random_seed: int = 42
    targets_db: str | Path | None = None
    baseline_dataset: str | Path | None = None
    target_period: int = 2024
    target_variables: tuple[str, ...] | None = None
    target_domains: tuple[str, ...] | None = None
    target_geo_levels: tuple[str, ...] | None = None
    target_profile: str | None = None
    calibration_target_variables: tuple[str, ...] | None = None
    calibration_target_domains: tuple[str, ...] | None = None
    calibration_target_geo_levels: tuple[str, ...] | None = None
    calibration_target_profile: str | None = None
    build_config: USMicroplexBuildConfig | None = None
    evaluate_parity: bool = True
    evaluate_pe_native_loss: bool = False
    evaluate_matched_pe_native_loss: bool = False
    reweight_matched_pe_native_loss: bool = False
    optimize_pe_native_loss: bool = False
    pe_native_household_budget: int | None = None
    pe_native_optimizer_max_iter: int = 200
    pe_native_optimizer_l2_penalty: float = 0.0
    pe_native_optimizer_tol: float = 1e-8
    pe_native_score_consistency_tol: float = 1e-6
    pe_native_target_delta_top_k: int = 25
    matched_baseline_household_count: int | None = None
    matched_baseline_random_seed: int = 42
    matched_baseline_reweight_epochs: int = 250
    matched_baseline_reweight_l0_lambda: float = 2.6445e-07
    matched_baseline_reweight_seed: int = 1456
    policyengine_us_data_repo: str | Path | None = None
    strict_materialization: bool = True
    fast_inner_loop_calibration: bool = False
    output_json_path: str | Path | None = None
    output_policyengine_dataset_path: str | Path | None = None
    output_pe_native_target_delta_path: str | Path | None = None
    output_pe_native_support_audit_path: str | Path | None = None
    output_matched_baseline_dataset_path: str | Path | None = None


@dataclass(frozen=True)
class USMicroplexPerformanceHarnessResult:
    """Stage timings plus parity metrics for one performance-harness run."""

    config: USMicroplexPerformanceHarnessConfig
    build_config: USMicroplexBuildConfig
    build_result: USMicroplexBuildResult
    source_names: tuple[str, ...]
    stage_timings: dict[str, float]
    total_seconds: float
    parity_run: PolicyEngineUSHarnessRun | None = None
    pe_native_scores: dict[str, object] | None = None
    matched_pe_native_scores: dict[str, object] | None = None
    pe_native_target_deltas: dict[str, object] | None = None
    pe_native_support_audit: dict[str, object] | None = None
    policyengine_dataset_path: str | None = None
    matched_baseline_dataset_path: str | None = None

    def _parity_run_attr(self, name: str) -> float | None:
        if self.parity_run is None:
            return None
        return getattr(self.parity_run, name)

    def _pe_native_score_attr(self, name: str) -> float | bool | int | None:
        if self.pe_native_scores is None:
            return None
        summary = self.pe_native_scores.get("summary")
        if not isinstance(summary, dict):
            return None
        return summary.get(name)

    @property
    def candidate_composite_parity_loss(self) -> float | None:
        return self._parity_run_attr("candidate_composite_parity_loss")

    @property
    def baseline_composite_parity_loss(self) -> float | None:
        return self._parity_run_attr("baseline_composite_parity_loss")

    @property
    def target_win_rate(self) -> float | None:
        return self._parity_run_attr("target_win_rate")

    @property
    def slice_win_rate(self) -> float | None:
        return self._parity_run_attr("slice_win_rate")

    @property
    def candidate_enhanced_cps_native_loss(self) -> float | None:
        value = self._pe_native_score_attr("candidate_enhanced_cps_native_loss")
        return float(value) if value is not None else None

    @property
    def baseline_enhanced_cps_native_loss(self) -> float | None:
        value = self._pe_native_score_attr("baseline_enhanced_cps_native_loss")
        return float(value) if value is not None else None

    @property
    def enhanced_cps_native_loss_delta(self) -> float | None:
        value = self._pe_native_score_attr("enhanced_cps_native_loss_delta")
        return float(value) if value is not None else None

    def to_dict(self) -> dict[str, object]:
        payload: dict[str, object] = {
            "config": _json_compatible_value(asdict(self.config)),
            "build_config": self.build_config.to_dict(),
            "source_names": list(self.source_names),
            "stage_timings": dict(self.stage_timings),
            "total_seconds": float(self.total_seconds),
            "calibration_summary": _json_compatible_value(
                self.build_result.calibration_summary
            ),
            "policyengine_dataset_path": self.policyengine_dataset_path,
        }
        if self.parity_run is not None:
            payload["parity_run"] = self.parity_run.to_dict()
        if self.pe_native_scores is not None:
            payload["pe_native_scores"] = _json_compatible_value(
                self.pe_native_scores
            )
        if self.matched_pe_native_scores is not None:
            payload["matched_pe_native_scores"] = _json_compatible_value(
                self.matched_pe_native_scores
            )
        if self.pe_native_target_deltas is not None:
            payload["pe_native_target_deltas"] = _json_compatible_value(
                self.pe_native_target_deltas
            )
        if self.pe_native_support_audit is not None:
            payload["pe_native_support_audit"] = _json_compatible_value(
                self.pe_native_support_audit
            )
        if self.matched_baseline_dataset_path is not None:
            payload["matched_baseline_dataset_path"] = self.matched_baseline_dataset_path
        return payload

    def save(self, path: str | Path) -> Path:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(json.dumps(self.to_dict(), indent=2, sort_keys=True))
        return destination


@dataclass(frozen=True)
class USMicroplexPerformanceHarnessRequest:
    """One performance-harness request for shared-session batch execution."""

    providers: tuple[SourceProvider, ...]
    config: USMicroplexPerformanceHarnessConfig
    queries: dict[str, SourceQuery] | None = None


@dataclass
class USMicroplexPreCalibrationCacheEntry:
    """Reusable upstream build state prior to PE-US calibration."""

    seed_data: pd.DataFrame
    synthetic_data: pd.DataFrame
    synthetic_tables: PolicyEngineUSEntityTableBundle
    targets: USMicroplexTargets
    synthesis_metadata: dict[str, object]
    synthesizer: object | None
    source_frame: ObservationFrame
    source_frames: tuple[ObservationFrame, ...]
    fusion_plan: FusionPlan


@dataclass
class USMicroplexCalibrationCacheEntry:
    """Reusable PE-US calibration result for a fixed synthetic table bundle."""

    policyengine_tables: PolicyEngineUSEntityTableBundle
    calibrated_data: pd.DataFrame
    calibration_summary: dict[str, object]


def _stage(stage_timings: dict[str, float], name: str) -> float:
    start = perf_counter()
    stage_timings.setdefault(name, 0.0)
    return start


def _finish_stage(stage_timings: dict[str, float], name: str, start: float) -> None:
    stage_timings[name] += perf_counter() - start


def _copy_output_file(source: str | Path, destination: str | Path) -> str:
    source_path = Path(source).expanduser().resolve()
    destination_path = Path(destination).expanduser().resolve()
    destination_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source_path, destination_path)
    return str(destination_path)


def _normalize_source_query_value(value: object) -> object:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, dict):
        return tuple(
            sorted(
                (str(key), _normalize_source_query_value(item))
                for key, item in value.items()
            )
        )
    if isinstance(value, (list, tuple)):
        return tuple(_normalize_source_query_value(item) for item in value)
    if callable(value):
        return f"{value.__module__}.{value.__qualname__}"
    return str(value)


def _json_compatible_value(value: object) -> object:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {
            str(key): _json_compatible_value(item)
            for key, item in value.items()
        }
    if isinstance(value, (list, tuple)):
        return [_json_compatible_value(item) for item in value]
    if hasattr(value, "item") and callable(getattr(value, "item")):
        try:
            return _json_compatible_value(value.item())
        except (TypeError, ValueError):
            pass
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def _sorted_normalized_items(
    items: Iterable[tuple[str, object]],
) -> tuple[tuple[str, object], ...]:
    return tuple(
        sorted(
            (str(key), _normalize_source_query_value(value))
            for key, value in items
        )
    )


_POLICYENGINE_SELECTION_CACHE_FIELDS = frozenset(
    {
        "policyengine_selection_backend",
        "policyengine_selection_household_budget",
        "policyengine_selection_max_iter",
        "policyengine_selection_tol",
        "policyengine_selection_l2_penalty",
    }
)

PRECALIBRATION_EXCLUDED_BUILD_CONFIG_FIELDS = frozenset(
    {
        "calibration_backend",
        "calibration_tol",
        "calibration_max_iter",
        "target_sparsity",
        "device",
        "policyengine_baseline_dataset",
        "policyengine_targets_db",
        "policyengine_target_period",
        "policyengine_target_variables",
        "policyengine_target_domains",
        "policyengine_target_geo_levels",
        "policyengine_calibration_target_variables",
        "policyengine_calibration_target_domains",
        "policyengine_calibration_target_geo_levels",
        "policyengine_target_reform_id",
        "policyengine_simulation_cls",
    }
) | _POLICYENGINE_SELECTION_CACHE_FIELDS

CALIBRATION_INCLUDED_BUILD_CONFIG_FIELDS = frozenset(
    {
        "calibration_backend",
        "calibration_tol",
        "calibration_max_iter",
        "target_sparsity",
        "device",
        "policyengine_targets_db",
        "policyengine_target_period",
        "policyengine_target_variables",
        "policyengine_target_domains",
        "policyengine_target_geo_levels",
        "policyengine_calibration_target_variables",
        "policyengine_calibration_target_domains",
        "policyengine_calibration_target_geo_levels",
        "policyengine_target_reform_id",
        "policyengine_simulation_cls",
        "policyengine_dataset_year",
    }
) | _POLICYENGINE_SELECTION_CACHE_FIELDS


def _provider_cache_identity(provider: SourceProvider) -> tuple[str, tuple[tuple[str, object], ...]]:
    provider_type = f"{provider.__class__.__module__}.{provider.__class__.__qualname__}"
    public_items = _sorted_normalized_items(
        (key, value)
        for key, value in vars(provider).items()
        if not key.startswith("_") and not callable(value)
    )
    return provider_type, public_items


def _build_config_key(
    build_config: USMicroplexBuildConfig,
    *,
    included_fields: frozenset[str] | None = None,
    excluded_fields: frozenset[str] = frozenset(),
) -> tuple[tuple[str, object], ...]:
    return _sorted_normalized_items(
        (key, value)
        for key, value in build_config.to_dict().items()
        if (included_fields is None or key in included_fields)
        and key not in excluded_fields
    )


def _precalibration_build_config_key(
    build_config: USMicroplexBuildConfig,
) -> BuildConfigCacheKey:
    return _build_config_key(
        build_config,
        excluded_fields=PRECALIBRATION_EXCLUDED_BUILD_CONFIG_FIELDS,
    )


def _calibration_build_config_key(
    build_config: USMicroplexBuildConfig,
) -> BuildConfigCacheKey:
    return _build_config_key(
        build_config,
        included_fields=CALIBRATION_INCLUDED_BUILD_CONFIG_FIELDS,
    )


def _source_query_cache_key(
    provider: SourceProvider,
    query: SourceQuery | None,
) -> SourceQueryCacheKey:
    query = query or SourceQuery()
    provider_type, provider_items = _provider_cache_identity(provider)
    return (
        provider_type,
        provider_items,
        query.period,
        _sorted_normalized_items(query.provider_filters.items()),
    )


def _copy_optional_table(table: pd.DataFrame | None) -> pd.DataFrame | None:
    if table is None:
        return None
    return table.copy()


def _clone_policyengine_us_tables(
    tables: PolicyEngineUSEntityTableBundle,
) -> PolicyEngineUSEntityTableBundle:
    if not isinstance(tables, PolicyEngineUSEntityTableBundle):
        return tables
    return PolicyEngineUSEntityTableBundle(
        households=tables.households.copy(),
        persons=_copy_optional_table(tables.persons),
        tax_units=_copy_optional_table(tables.tax_units),
        spm_units=_copy_optional_table(tables.spm_units),
        families=_copy_optional_table(tables.families),
        marital_units=_copy_optional_table(tables.marital_units),
    )


def _clone_calibration_cache_entry(
    entry: USMicroplexCalibrationCacheEntry,
) -> USMicroplexCalibrationCacheEntry:
    return USMicroplexCalibrationCacheEntry(
        policyengine_tables=_clone_policyengine_us_tables(entry.policyengine_tables),
        calibrated_data=entry.calibrated_data.copy(deep=True),
        calibration_summary=dict(entry.calibration_summary),
    )


def _filter_policyengine_tables_to_households(
    tables: PolicyEngineUSEntityTableBundle,
    household_ids: pd.Index,
) -> PolicyEngineUSEntityTableBundle:
    household_id_set = set(household_ids.tolist())

    def _filter_table(table: pd.DataFrame | None) -> pd.DataFrame | None:
        if table is None:
            return None
        if "household_id" not in table.columns:
            return table.copy()
        return table.loc[table["household_id"].isin(household_id_set)].copy()

    households = tables.households.loc[
        tables.households["household_id"].isin(household_id_set)
    ].copy()
    return PolicyEngineUSEntityTableBundle(
        households=households,
        persons=_filter_table(tables.persons),
        tax_units=_filter_table(tables.tax_units),
        spm_units=_filter_table(tables.spm_units),
        families=_filter_table(tables.families),
        marital_units=_filter_table(tables.marital_units),
    )


def _write_matched_policyengine_us_baseline_dataset(
    baseline_dataset_path: str | Path,
    output_dataset_path: str | Path,
    *,
    period: int,
    household_count: int,
    random_seed: int,
    sample_method: str = "uniform",
) -> str:
    period_key = str(period)
    arrays = _load_policyengine_us_period_arrays(
        baseline_dataset_path,
        period_key=period_key,
        variables=None,
    )
    required_structural = {
        "household_id",
        "household_weight",
        "person_id",
        "person_household_id",
    }
    missing = sorted(required_structural - set(arrays))
    if missing:
        raise ValueError(
            "matched baseline dataset is missing required structural arrays: "
            + ", ".join(missing)
        )

    household_ids = np.asarray(arrays["household_id"])
    household_weights = np.asarray(arrays["household_weight"], dtype=np.float64)
    if household_count <= 0:
        raise ValueError("matched baseline household_count must be positive")
    if household_count > len(household_ids):
        raise ValueError(
            "matched baseline household_count cannot exceed baseline household rows"
        )

    resolved_baseline_path = Path(baseline_dataset_path).expanduser().resolve()
    resolved_output_path = Path(output_dataset_path).expanduser().resolve()
    if household_count == len(household_ids):
        if resolved_baseline_path != resolved_output_path:
            resolved_output_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(resolved_baseline_path, resolved_output_path)
        return str(resolved_output_path)

    sampled_household_ids = _sample_matched_household_ids(
        household_ids,
        household_weights,
        household_count=household_count,
        random_seed=random_seed,
        sample_method=sample_method,
    )
    household_mask = np.isin(household_ids, sampled_household_ids)
    person_mask = np.isin(
        np.asarray(arrays["person_household_id"]),
        sampled_household_ids,
    )

    entity_masks: dict[object, np.ndarray] = {
        EntityType.HOUSEHOLD: household_mask,
        EntityType.PERSON: person_mask,
    }
    entity_lengths: dict[EntityType, int] = {
        EntityType.HOUSEHOLD: len(household_ids),
        EntityType.PERSON: len(np.asarray(arrays["person_id"])),
    }
    for entity_type, id_name, person_membership_name in (
        (EntityType.TAX_UNIT, "tax_unit_id", "person_tax_unit_id"),
        (EntityType.SPM_UNIT, "spm_unit_id", "person_spm_unit_id"),
        (EntityType.FAMILY, "family_id", "person_family_id"),
        ("marital_unit", "marital_unit_id", "person_marital_unit_id"),
    ):
        entity_ids = arrays.get(id_name)
        person_entity_ids = arrays.get(person_membership_name)
        if entity_ids is None or person_entity_ids is None:
            continue
        entity_ids = np.asarray(entity_ids)
        person_entity_ids = np.asarray(person_entity_ids)
        selected_entity_ids = np.unique(person_entity_ids[person_mask])
        entity_masks[entity_type] = np.isin(entity_ids, selected_entity_ids)
        entity_lengths[entity_type] = len(entity_ids)

    original_weight_sum = float(household_weights.sum())
    sampled_weight_sum = float(household_weights[household_mask].sum())
    if sampled_weight_sum <= 0.0:
        raise ValueError("matched baseline sample produced nonpositive household weight sum")
    weight_scale = original_weight_sum / sampled_weight_sum

    structural_entities: dict[str, object] = {
        "household_id": EntityType.HOUSEHOLD,
        "household_weight": EntityType.HOUSEHOLD,
        "person_id": EntityType.PERSON,
        "person_household_id": EntityType.PERSON,
        "person_weight": EntityType.PERSON,
        "tax_unit_id": EntityType.TAX_UNIT,
        "person_tax_unit_id": EntityType.PERSON,
        "tax_unit_weight": EntityType.TAX_UNIT,
        "spm_unit_id": EntityType.SPM_UNIT,
        "person_spm_unit_id": EntityType.PERSON,
        "spm_unit_weight": EntityType.SPM_UNIT,
        "family_id": EntityType.FAMILY,
        "person_family_id": EntityType.PERSON,
        "family_weight": EntityType.FAMILY,
        "marital_unit_id": "marital_unit",
        "person_marital_unit_id": EntityType.PERSON,
        "marital_unit_weight": "marital_unit",
    }
    scaled_weight_variables = {
        "household_weight",
        "person_weight",
        "tax_unit_weight",
        "spm_unit_weight",
        "family_weight",
        "marital_unit_weight",
    }
    try:
        tax_benefit_system = _resolve_policyengine_us_tax_benefit_system(None)
    except (ImportError, ValueError):
        tax_benefit_system = None

    sampled_arrays: dict[str, dict[str, np.ndarray]] = {}
    with h5py.File(resolved_baseline_path, "r") as handle:
        for variable_name, group in handle.items():
            if not isinstance(group, h5py.Group):
                continue
            period_values = {
                stored_period: np.asarray(dataset)
                for stored_period, dataset in group.items()
            }
            if not period_values:
                continue

            representative_values = next(iter(period_values.values()))
            entity = structural_entities.get(variable_name)
            if entity is None:
                entity = _infer_policyengine_array_entity(
                    variable_name=variable_name,
                    values=representative_values,
                    entity_lengths=entity_lengths,
                    tax_benefit_system=tax_benefit_system,
                )
            mask = entity_masks.get(entity)
            if mask is None:
                continue

            sampled_periods: dict[str, np.ndarray] = {}
            for stored_period, values in period_values.items():
                sampled_values = values[mask]
                if variable_name in scaled_weight_variables:
                    sampled_values = sampled_values.astype(np.float64) * weight_scale
                sampled_periods[stored_period] = sampled_values
            sampled_arrays[variable_name] = sampled_periods

    return str(
        write_policyengine_us_time_period_dataset(
            sampled_arrays,
            output_dataset_path,
        ).resolve()
    )


def _sample_matched_household_ids(
    household_ids: np.ndarray,
    household_weights: np.ndarray,
    *,
    household_count: int,
    random_seed: int,
    sample_method: str,
) -> np.ndarray:
    """Choose household IDs for a matched-size PE dataset copy."""

    method = sample_method.lower().replace("-", "_")
    if method == "uniform":
        return (
            pd.Series(household_ids)
            .sample(
                n=household_count,
                replace=False,
                random_state=random_seed,
            )
            .to_numpy()
        )
    if method in {"weight_proportional", "pps"}:
        positive_weights = np.clip(household_weights.astype(np.float64), 0.0, None)
        weight_sum = float(positive_weights.sum())
        if weight_sum <= 0.0:
            raise ValueError(
                "weight-proportional matched sample requires positive household weights"
            )
        rng = np.random.default_rng(random_seed)
        return rng.choice(
            household_ids,
            size=household_count,
            replace=False,
            p=positive_weights / weight_sum,
        )
    if method == "largest_weight":
        frame = pd.DataFrame(
            {"household_id": household_ids, "household_weight": household_weights}
        )
        return (
            frame.sort_values(
                ["household_weight", "household_id"],
                ascending=[False, True],
                kind="mergesort",
            )["household_id"]
            .head(household_count)
            .to_numpy()
        )
    raise ValueError(
        "matched sample_method must be one of: uniform, weight_proportional, "
        "largest_weight"
    )


def _reweight_matched_policyengine_us_baseline_dataset(
    input_dataset_path: str | Path,
    output_dataset_path: str | Path,
    *,
    period: int,
    epochs: int,
    l0_lambda: float,
    seed: int,
    policyengine_us_data_repo: str | Path | None = None,
) -> str:
    resolved_repo = resolve_policyengine_us_data_repo_root(policyengine_us_data_repo)
    env = build_policyengine_us_data_subprocess_env(resolved_repo)
    with TemporaryDirectory(prefix="microplex-us-matched-baseline-reweight-") as temp_dir:
        weights_path = Path(temp_dir) / "matched_baseline_weights.npy"
        completed = subprocess.run(
            [
                "uv",
                "run",
                "--project",
                str(resolved_repo),
                "python",
                "-c",
                _MATCHED_BASELINE_REWEIGHT_SCRIPT,
                str(resolved_repo),
                json.dumps(_ENHANCED_CPS_BAD_TARGETS),
                str(int(period)),
                str(Path(input_dataset_path).expanduser().resolve()),
                str(weights_path),
                str(int(epochs)),
                str(float(l0_lambda)),
                str(int(seed)),
            ],
            cwd=resolved_repo,
            env=env,
            capture_output=True,
            text=True,
            check=False,
        )
        if completed.returncode != 0:
            stderr = completed.stderr.strip()
            stdout = completed.stdout.strip()
            detail = stderr or stdout or f"exit code {completed.returncode}"
            raise RuntimeError(f"Matched baseline reweighting failed: {detail}")
        optimized_weights = np.load(weights_path)
    rewritten = rewrite_policyengine_us_dataset_weights(
        input_dataset_path=input_dataset_path,
        output_dataset_path=output_dataset_path,
        household_weights=optimized_weights,
        period=period,
    )
    return str(rewritten.resolve())


def _default_provider_query(
    config: USMicroplexPerformanceHarnessConfig,
) -> SourceQuery:
    return SourceQuery(
        provider_filters={
            "sample_n": config.sample_n,
            "random_seed": config.random_seed,
        }
    )


def _load_frames(
    providers: list[SourceProvider],
    provider_queries: dict[str, SourceQuery],
    *,
    frame_cache: dict[SourceQueryCacheKey, ObservationFrame] | None,
) -> tuple[list[ObservationFrame], list[SourceQueryCacheKey]]:
    frames: list[ObservationFrame] = []
    frame_keys: list[SourceQueryCacheKey] = []
    for provider in providers:
        provider_query = provider_queries.get(provider.descriptor.name)
        cache_key = _source_query_cache_key(provider, provider_query)
        frame_keys.append(cache_key)
        if frame_cache is not None and cache_key in frame_cache:
            frames.append(frame_cache[cache_key])
            continue
        frame = provider.load_frame(provider_query)
        if frame_cache is not None:
            frame_cache[cache_key] = frame
        frames.append(frame)
    return frames, frame_keys


def _mark_skipped_stages(stage_timings: dict[str, float], stage_names: tuple[str, ...]) -> None:
    for stage_name in stage_names:
        stage_timings.setdefault(stage_name, 0.0)


def _resolve_build_config(config: USMicroplexPerformanceHarnessConfig) -> USMicroplexBuildConfig:
    default_base_kwargs: dict[str, object] = {
        "synthesis_backend": "bootstrap",
        "calibration_backend": "entropy",
    }
    if config.target_profile is None and config.calibration_target_profile is None:
        default_base_kwargs.update(
            {
                "policyengine_target_variables": (
                    "adjusted_gross_income",
                    "income_tax",
                    "dividend_income",
                    "taxable_interest_income",
                    "self_employment_income",
                ),
                "policyengine_target_geo_levels": ("national",),
            }
        )
    base = config.build_config or USMicroplexBuildConfig(**default_base_kwargs)
    target_variables = (
        base.policyengine_target_variables
        if config.target_variables is None
        else config.target_variables
    )
    target_domains = (
        base.policyengine_target_domains
        if config.target_domains is None
        else config.target_domains
    )
    target_geo_levels = (
        base.policyengine_target_geo_levels
        if config.target_geo_levels is None
        else config.target_geo_levels
    )
    calibration_target_variables = (
        config.calibration_target_variables
        if config.calibration_target_variables is not None
        else base.policyengine_calibration_target_variables
        or (
            default_fast_calibration_target_variables(target_variables)
            if config.fast_inner_loop_calibration and target_variables
            else target_variables
        )
    )
    calibration_target_domains = (
        config.calibration_target_domains
        if config.calibration_target_domains is not None
        else base.policyengine_calibration_target_domains or target_domains
    )
    calibration_target_geo_levels = (
        config.calibration_target_geo_levels
        if config.calibration_target_geo_levels is not None
        else base.policyengine_calibration_target_geo_levels or target_geo_levels
    )
    return replace(
        base,
        n_synthetic=config.n_synthetic,
        random_seed=config.random_seed,
        policyengine_targets_db=(
            str(config.targets_db)
            if config.targets_db is not None
            else base.policyengine_targets_db
        ),
        policyengine_baseline_dataset=(
            str(config.baseline_dataset)
            if config.baseline_dataset is not None
            else base.policyengine_baseline_dataset
        ),
        policyengine_target_period=config.target_period,
        policyengine_target_variables=target_variables,
        policyengine_target_domains=target_domains,
        policyengine_target_geo_levels=target_geo_levels,
        policyengine_target_profile=config.target_profile or base.policyengine_target_profile,
        policyengine_calibration_target_variables=calibration_target_variables,
        policyengine_calibration_target_domains=calibration_target_domains,
        policyengine_calibration_target_geo_levels=calibration_target_geo_levels,
        policyengine_calibration_target_profile=(
            config.calibration_target_profile
            or base.policyengine_calibration_target_profile
        ),
        policyengine_dataset_year=base.policyengine_dataset_year or config.target_period,
    )


@dataclass
class USMicroplexPerformanceSession:
    """Reusable local optimization session with a persistent PE-US comparison cache."""

    comparison_cache: PolicyEngineUSComparisonCache = field(
        default_factory=PolicyEngineUSComparisonCache
    )
    frame_cache: dict[SourceQueryCacheKey, ObservationFrame] = field(default_factory=dict)
    precalibration_cache: dict[PreCalibrationCacheKey, USMicroplexPreCalibrationCacheEntry] = field(
        default_factory=dict
    )
    calibration_cache: dict[CalibrationCacheKey, USMicroplexCalibrationCacheEntry] = field(
        default_factory=dict
    )

    def warm_parity_cache(
        self,
        *,
        config: USMicroplexPerformanceHarnessConfig,
    ) -> PolicyEngineUSComparisonCache:
        return warm_us_microplex_parity_cache(
            config=config,
            comparison_cache=self.comparison_cache,
        )

    def run(
        self,
        providers: list[SourceProvider],
        *,
        config: USMicroplexPerformanceHarnessConfig,
        queries: dict[str, SourceQuery] | None = None,
    ) -> USMicroplexPerformanceHarnessResult:
        return run_us_microplex_performance_harness(
            providers,
            config=config,
            queries=queries,
            comparison_cache=self.comparison_cache,
            frame_cache=self.frame_cache,
            precalibration_cache=self.precalibration_cache,
            calibration_cache=self.calibration_cache,
        )

    def run_batch(
        self,
        requests: list[USMicroplexPerformanceHarnessRequest]
        | tuple[USMicroplexPerformanceHarnessRequest, ...],
    ) -> tuple[USMicroplexPerformanceHarnessResult, ...]:
        """Run multiple requests with shared caches and grouped PE-native batch scoring."""

        if not requests:
            return ()

        indexed_results: list[USMicroplexPerformanceHarnessResult] = []
        batch_groups: dict[
            tuple[str, int, str | None],
            list[tuple[int, str, USMicroplexPerformanceHarnessConfig]],
        ] = {}
        pending_target_deltas: list[tuple[int, str, USMicroplexPerformanceHarnessConfig]] = []

        with TemporaryDirectory(prefix="microplex-us-harness-batch-") as temp_dir:
            temp_root = Path(temp_dir)
            for index, request in enumerate(requests):
                original_config = request.config
                should_batch_native_loss = (
                    original_config.evaluate_pe_native_loss
                    and not original_config.optimize_pe_native_loss
                    and original_config.baseline_dataset is not None
                )
                dataset_output_path = original_config.output_policyengine_dataset_path
                if should_batch_native_loss and dataset_output_path is None:
                    dataset_output_path = temp_root / f"candidate_{index}.h5"

                run_config = original_config
                if should_batch_native_loss:
                    run_config = replace(
                        original_config,
                        evaluate_pe_native_loss=False,
                        output_json_path=None,
                        output_pe_native_target_delta_path=None,
                        output_policyengine_dataset_path=dataset_output_path,
                    )

                result = self.run(
                    list(request.providers),
                    config=run_config,
                    queries=request.queries,
                )
                if should_batch_native_loss:
                    if result.policyengine_dataset_path is None:
                        raise ValueError(
                            "Batched PE-native scoring requires an exported policyengine dataset path"
                        )
                    result = replace(result, config=original_config)
                    group_key = (
                        str(original_config.baseline_dataset),
                        (
                            result.build_config.policyengine_dataset_year
                            or original_config.target_period
                        ),
                        (
                            str(original_config.policyengine_us_data_repo)
                            if original_config.policyengine_us_data_repo is not None
                            else None
                        ),
                    )
                    batch_groups.setdefault(group_key, []).append(
                        (index, result.policyengine_dataset_path, original_config)
                    )
                    if original_config.output_pe_native_target_delta_path is not None:
                        pending_target_deltas.append(
                            (index, result.policyengine_dataset_path, original_config)
                        )
                indexed_results.append(result)

            for group_key, group_items in batch_groups.items():
                baseline_dataset, period, policyengine_us_data_repo = group_key
                payloads = compute_batch_us_pe_native_scores(
                    candidate_dataset_paths=[
                        candidate_path for _, candidate_path, _ in group_items
                    ],
                    baseline_dataset_path=baseline_dataset,
                    period=period,
                    policyengine_us_data_repo=policyengine_us_data_repo,
                )
                if len(payloads) != len(group_items):
                    raise ValueError(
                        "PE-native batch scorer returned a different number of payloads than requests"
                    )
                for (result_index, _candidate_path, original_config), payload in zip(
                    group_items,
                    payloads,
                    strict=True,
                ):
                    stage_timings = dict(indexed_results[result_index].stage_timings)
                    timing = payload.get("timing")
                    if isinstance(timing, dict):
                        batch_elapsed = timing.get("batch_elapsed_seconds")
                        if batch_elapsed is not None:
                            stage_timings["evaluate_pe_native_loss"] = float(batch_elapsed)
                    updated_result = replace(
                        indexed_results[result_index],
                        config=original_config,
                        stage_timings=stage_timings,
                        pe_native_scores=payload,
                    )
                    indexed_results[result_index] = updated_result

            for result_index, candidate_path, original_config in pending_target_deltas:
                target_deltas = compare_us_pe_native_target_deltas(
                    from_dataset_path=str(original_config.baseline_dataset),
                    to_dataset_path=candidate_path,
                    period=(
                        indexed_results[result_index].build_config.policyengine_dataset_year
                        or original_config.target_period
                    ),
                    top_k=original_config.pe_native_target_delta_top_k,
                    policyengine_us_data_repo=original_config.policyengine_us_data_repo,
                )
                destination = Path(original_config.output_pe_native_target_delta_path)
                destination.parent.mkdir(parents=True, exist_ok=True)
                destination.write_text(
                    json.dumps(
                        _json_compatible_value(target_deltas),
                        indent=2,
                        sort_keys=True,
                    )
                )
                stage_timings = dict(indexed_results[result_index].stage_timings)
                stage_timings.setdefault("evaluate_pe_native_target_deltas", 0.0)
                stage_timings.setdefault("write_pe_native_target_delta_json", 0.0)
                indexed_results[result_index] = replace(
                    indexed_results[result_index],
                    stage_timings=stage_timings,
                    pe_native_target_deltas=target_deltas,
                )

            final_results: list[USMicroplexPerformanceHarnessResult] = []
            for result in indexed_results:
                if result.config.output_json_path is not None:
                    result.save(result.config.output_json_path)
                final_results.append(result)

        return tuple(final_results)


def _union_target_set(target_sets: dict[str, TargetSet]) -> TargetSet:
    union = TargetSet()
    seen_names: set[str] = set()
    for target_set in target_sets.values():
        for target in target_set.targets:
            if target.name in seen_names:
                continue
            seen_names.add(target.name)
            union.add(target)
    return union


def warm_us_microplex_parity_cache(
    *,
    config: USMicroplexPerformanceHarnessConfig,
    comparison_cache: PolicyEngineUSComparisonCache | None = None,
) -> PolicyEngineUSComparisonCache:
    """Preload target slices and the baseline PE-US report into a reusable cache."""
    if config.targets_db is None or config.baseline_dataset is None:
        raise ValueError(
            "warm_us_microplex_parity_cache requires both targets_db and baseline_dataset"
        )

    cache = comparison_cache or PolicyEngineUSComparisonCache()
    build_config = _resolve_build_config(config)
    target_provider = PolicyEngineUSDBTargetProvider(str(config.targets_db))
    slices = default_policyengine_us_db_harness_slices(
        period=config.target_period,
        variables=build_config.policyengine_target_variables,
        domain_variables=build_config.policyengine_target_domains,
        geo_levels=build_config.policyengine_target_geo_levels,
        reform_id=build_config.policyengine_target_reform_id,
    )
    slices = filter_nonempty_policyengine_us_harness_slices(
        target_provider,
        slices,
        cache=cache,
    )
    slice_target_sets = {
        slice_spec.name: cache.load_target_set(target_provider, slice_spec.query)
        for slice_spec in slices
    }
    union_target_set = _union_target_set(slice_target_sets)
    cache.load_baseline_report(
        target_set=union_target_set,
        baseline_dataset=str(config.baseline_dataset),
        period=config.target_period,
        dataset_year=build_config.policyengine_dataset_year,
        simulation_cls=build_config.policyengine_simulation_cls,
        baseline_label="policyengine_baseline",
        strict_materialization=config.strict_materialization,
    )
    return cache


def run_us_microplex_performance_harness(
    providers: list[SourceProvider],
    *,
    config: USMicroplexPerformanceHarnessConfig,
    queries: dict[str, SourceQuery] | None = None,
    comparison_cache: PolicyEngineUSComparisonCache | None = None,
    frame_cache: dict[SourceQueryCacheKey, ObservationFrame] | None = None,
    precalibration_cache: dict[PreCalibrationCacheKey, USMicroplexPreCalibrationCacheEntry]
    | None = None,
    calibration_cache: dict[CalibrationCacheKey, USMicroplexCalibrationCacheEntry]
    | None = None,
) -> USMicroplexPerformanceHarnessResult:
    """Run a repeatable build+parity loop with stage-level timings."""
    if not providers:
        raise ValueError("USMicroplex performance harness requires at least one provider")
    if config.evaluate_parity and (
        config.targets_db is None or config.baseline_dataset is None
    ):
        raise ValueError(
            "USMicroplex performance harness requires both targets_db and baseline_dataset"
        )
    if config.evaluate_pe_native_loss and config.baseline_dataset is None:
        raise ValueError(
            "USMicroplex performance harness requires baseline_dataset for PE-native loss scoring"
        )
    if config.evaluate_matched_pe_native_loss and config.baseline_dataset is None:
        raise ValueError(
            "USMicroplex performance harness requires baseline_dataset for matched PE-native loss scoring"
        )
    if config.reweight_matched_pe_native_loss and not config.evaluate_matched_pe_native_loss:
        raise ValueError(
            "reweight_matched_pe_native_loss requires evaluate_matched_pe_native_loss"
        )
    if config.optimize_pe_native_loss and not config.evaluate_pe_native_loss:
        raise ValueError(
            "USMicroplex performance harness requires evaluate_pe_native_loss when optimize_pe_native_loss is enabled"
        )
    if (
        config.pe_native_household_budget is not None
        and config.pe_native_household_budget <= 0
    ):
        raise ValueError("pe_native_household_budget must be positive when provided")
    if config.pe_native_score_consistency_tol <= 0.0:
        raise ValueError("pe_native_score_consistency_tol must be positive")
    if config.pe_native_target_delta_top_k <= 0:
        raise ValueError("pe_native_target_delta_top_k must be positive")
    if (
        config.matched_baseline_household_count is not None
        and config.matched_baseline_household_count <= 0
    ):
        raise ValueError("matched_baseline_household_count must be positive")
    if config.matched_baseline_reweight_epochs <= 0:
        raise ValueError("matched_baseline_reweight_epochs must be positive")
    if config.matched_baseline_reweight_l0_lambda < 0.0:
        raise ValueError("matched_baseline_reweight_l0_lambda must be nonnegative")
    if (
        config.output_pe_native_target_delta_path is not None
        and config.baseline_dataset is None
    ):
        raise ValueError(
            "USMicroplex performance harness requires baseline_dataset for PE-native target deltas"
        )
    if (
        config.output_pe_native_support_audit_path is not None
        and config.baseline_dataset is None
    ):
        raise ValueError(
            "USMicroplex performance harness requires baseline_dataset for PE-native support audit"
        )

    build_config = _resolve_build_config(config)
    pipeline = USMicroplexPipeline(build_config)
    provider_queries = dict(queries or {})
    for provider in providers:
        provider_queries.setdefault(
            provider.descriptor.name,
            _default_provider_query(config),
        )

    stage_timings: dict[str, float] = {}
    total_start = perf_counter()

    start = _stage(stage_timings, "load_frames")
    frames, frame_keys = _load_frames(
        providers,
        provider_queries,
        frame_cache=frame_cache,
    )
    _finish_stage(stage_timings, "load_frames", start)
    precalibration_key = (
        tuple(frame_keys),
        _precalibration_build_config_key(build_config),
    )
    precalibration = (
        precalibration_cache.get(precalibration_key)
        if precalibration_cache is not None
        else None
    )

    if precalibration is None:
        start = _stage(stage_timings, "prepare_source_inputs")
        source_inputs = [pipeline.prepare_source_input(frame) for frame in frames]
        fusion_plan = FusionPlan.from_sources([frame.source for frame in frames])
        scaffold_input = pipeline._select_scaffold_source(source_inputs)
        _finish_stage(stage_timings, "prepare_source_inputs", start)

        start = _stage(stage_timings, "prepare_seed_data")
        seed_data = pipeline.prepare_seed_data_from_source(scaffold_input)
        _finish_stage(stage_timings, "prepare_seed_data", start)

        start = _stage(stage_timings, "integrate_donor_sources")
        donor_integration = pipeline._integrate_donor_sources(
            seed_data,
            scaffold_input=scaffold_input,
            donor_inputs=[
                source for source in source_inputs if source is not scaffold_input
            ],
        )
        seed_data = donor_integration["seed_data"]
        _finish_stage(stage_timings, "integrate_donor_sources", start)

        start = _stage(stage_timings, "build_targets")
        targets = pipeline.build_targets(seed_data)
        _finish_stage(stage_timings, "build_targets", start)

        start = _stage(stage_timings, "resolve_synthesis_variables")
        synthesis_variables = pipeline._resolve_synthesis_variables(
            scaffold_input,
            fusion_plan=fusion_plan,
            include_all_observed_targets=len(source_inputs) > 1,
            available_columns=set(seed_data.columns),
        )
        _finish_stage(stage_timings, "resolve_synthesis_variables", start)

        start = _stage(stage_timings, "synthesize")
        synthetic_data, synthesizer, synthesis_metadata = pipeline.synthesize(
            seed_data,
            synthesis_variables=synthesis_variables,
        )
        _finish_stage(stage_timings, "synthesize", start)

        start = _stage(stage_timings, "ensure_target_support")
        synthetic_data = pipeline.ensure_target_support(
            synthetic_data,
            seed_data,
            targets,
        )
        _finish_stage(stage_timings, "ensure_target_support", start)

        start = _stage(stage_timings, "build_policyengine_tables")
        synthetic_tables = pipeline.build_policyengine_entity_tables(synthetic_data)
        _finish_stage(stage_timings, "build_policyengine_tables", start)

        synthesis_metadata = {
            **synthesis_metadata,
            "source_names": fusion_plan.source_names,
            "condition_vars": list(synthesis_variables.condition_vars),
            "target_vars": list(synthesis_variables.target_vars),
            "scaffold_source": scaffold_input.frame.source.name,
            "donor_integrated_variables": donor_integration["integrated_variables"],
        }
        precalibration = USMicroplexPreCalibrationCacheEntry(
            seed_data=seed_data,
            synthetic_data=synthetic_data,
            synthetic_tables=synthetic_tables,
            targets=targets,
            synthesis_metadata=synthesis_metadata,
            synthesizer=synthesizer,
            source_frame=scaffold_input.frame,
            source_frames=tuple(frames),
            fusion_plan=fusion_plan,
        )
        if precalibration_cache is not None:
            precalibration_cache[precalibration_key] = precalibration
    else:
        _mark_skipped_stages(stage_timings, PRECALIBRATION_STAGE_NAMES)

    calibration_key = (
        precalibration_key,
        _calibration_build_config_key(build_config),
    )
    calibration = (
        calibration_cache.get(calibration_key)
        if calibration_cache is not None
        else None
    )
    if calibration is None:
        start = _stage(stage_timings, "calibrate_policyengine_tables")
        policyengine_tables, calibrated_data, calibration_summary = (
            pipeline.calibrate_policyengine_tables(
                _clone_policyengine_us_tables(precalibration.synthetic_tables)
            )
        )
        _finish_stage(stage_timings, "calibrate_policyengine_tables", start)
        calibration = USMicroplexCalibrationCacheEntry(
            policyengine_tables=_clone_policyengine_us_tables(policyengine_tables),
            calibrated_data=calibrated_data.copy(deep=True),
            calibration_summary=dict(calibration_summary),
        )
        if calibration_cache is not None:
            calibration_cache[calibration_key] = calibration
    else:
        stage_timings.setdefault("calibrate_policyengine_tables", 0.0)

    calibration = _clone_calibration_cache_entry(calibration)

    build_result = USMicroplexBuildResult(
        config=build_config,
        seed_data=precalibration.seed_data,
        synthetic_data=precalibration.synthetic_data,
        calibrated_data=calibration.calibrated_data,
        targets=precalibration.targets,
        calibration_summary=calibration.calibration_summary,
        synthesis_metadata=dict(precalibration.synthesis_metadata),
        synthesizer=precalibration.synthesizer,
        policyengine_tables=calibration.policyengine_tables,
        source_frame=precalibration.source_frame,
        source_frames=precalibration.source_frames,
        fusion_plan=precalibration.fusion_plan,
    )

    parity_run = None
    if config.evaluate_parity:
        start = _stage(stage_timings, "evaluate_parity_harness")
        target_provider = PolicyEngineUSDBTargetProvider(str(config.targets_db))
        slices = default_policyengine_us_db_harness_slices(
            period=config.target_period,
            variables=build_config.policyengine_target_variables,
            domain_variables=build_config.policyengine_target_domains,
            geo_levels=build_config.policyengine_target_geo_levels,
            reform_id=build_config.policyengine_target_reform_id,
        )
        slices = filter_nonempty_policyengine_us_harness_slices(
            target_provider,
            slices,
            cache=comparison_cache,
        )
        parity_run = evaluate_policyengine_us_harness(
            build_result.policyengine_tables,
            target_provider,
            slices,
            baseline_dataset=str(config.baseline_dataset),
            dataset_year=build_config.policyengine_dataset_year,
            simulation_cls=build_config.policyengine_simulation_cls,
            metadata={
                "sample_n": config.sample_n,
                "n_synthetic": config.n_synthetic,
                "source_names": precalibration.fusion_plan.source_names,
                "target_variables": list(build_config.policyengine_target_variables),
                "calibration_target_variables": list(
                    build_config.policyengine_calibration_target_variables
                ),
            },
            strict_materialization=config.strict_materialization,
            cache=comparison_cache,
        )
        _finish_stage(stage_timings, "evaluate_parity_harness", start)

    policyengine_dataset_path = None
    pe_native_scores = None
    matched_pe_native_scores = None
    pe_native_target_deltas = None
    pe_native_support_audit = None
    needs_pe_dataset = (
        config.evaluate_pe_native_loss
        or config.evaluate_matched_pe_native_loss
        or config.output_pe_native_target_delta_path is not None
        or config.output_pe_native_support_audit_path is not None
    )
    if needs_pe_dataset:
        with TemporaryDirectory(prefix="microplex-us-native-score-") as temp_dir:
            pe_stage_started = False
            candidate_dataset_path = pipeline.export_policyengine_dataset(
                build_result,
                Path(temp_dir) / "candidate_policyengine_us.h5",
                direct_override_variables=build_config.policyengine_direct_override_variables,
            )
            dataset_to_score = candidate_dataset_path
            pe_native_optimization = None
            if config.optimize_pe_native_loss:
                if not pe_stage_started:
                    start = _stage(stage_timings, "evaluate_pe_native_loss")
                    pe_stage_started = True
                optimize_start = _stage(
                    stage_timings,
                    "optimize_pe_native_loss_weights",
                )
                optimized_dataset_path = (
                    Path(temp_dir) / "candidate_policyengine_us_optimized.h5"
                )
                optimization_result = optimize_policyengine_us_native_loss_dataset(
                    input_dataset_path=candidate_dataset_path,
                    output_dataset_path=optimized_dataset_path,
                    period=build_config.policyengine_dataset_year or config.target_period,
                    budget=config.pe_native_household_budget,
                    max_iter=config.pe_native_optimizer_max_iter,
                    l2_penalty=config.pe_native_optimizer_l2_penalty,
                    tol=config.pe_native_optimizer_tol,
                    policyengine_us_data_repo=config.policyengine_us_data_repo,
                )
                dataset_to_score = optimized_dataset_path
                pe_native_optimization = optimization_result.to_dict()
                _finish_stage(
                    stage_timings,
                    "optimize_pe_native_loss_weights",
                    optimize_start,
                )
            if config.evaluate_pe_native_loss:
                if not pe_stage_started:
                    start = _stage(stage_timings, "evaluate_pe_native_loss")
                    pe_stage_started = True
                pe_native_scores = compute_us_pe_native_scores(
                    candidate_dataset_path=dataset_to_score,
                    baseline_dataset_path=str(config.baseline_dataset),
                    period=build_config.policyengine_dataset_year or config.target_period,
                    policyengine_us_data_repo=config.policyengine_us_data_repo,
                )
                if pe_native_optimization is not None:
                    summary = pe_native_scores.get("summary")
                    if not isinstance(summary, dict):
                        raise ValueError(
                            "PE-native optimization requires score summary metadata for consistency validation"
                        )
                    rescored_loss = summary.get("candidate_enhanced_cps_native_loss")
                    if rescored_loss is None:
                        raise ValueError(
                            "PE-native optimization consistency validation requires candidate_enhanced_cps_native_loss"
                        )
                    abs_error = abs(
                        float(rescored_loss) - float(pe_native_optimization["optimized_loss"])
                    )
                    if abs_error > config.pe_native_score_consistency_tol:
                        raise ValueError(
                            "PE-native optimized loss does not match rescored loss within tolerance: "
                            f"{abs_error:.6g} > {config.pe_native_score_consistency_tol:.6g}"
                        )
                    pe_native_scores = dict(pe_native_scores)
                    pe_native_optimization = dict(pe_native_optimization)
                    pe_native_optimization["rescored_loss_abs_error"] = abs_error
                    pe_native_scores["optimization"] = pe_native_optimization
            matched_baseline_dataset_path = None
            if config.evaluate_matched_pe_native_loss:
                matched_start = _stage(
                    stage_timings,
                    "build_matched_baseline_dataset",
                )
                candidate_household_count = len(build_result.policyengine_tables.households)
                matched_baseline_dataset_path = _write_matched_policyengine_us_baseline_dataset(
                    config.baseline_dataset,
                    config.output_matched_baseline_dataset_path
                    or (Path(temp_dir) / "matched_baseline_policyengine_us.h5"),
                    period=build_config.policyengine_dataset_year or config.target_period,
                    household_count=(
                        config.matched_baseline_household_count
                        or candidate_household_count
                    ),
                    random_seed=config.matched_baseline_random_seed,
                )
                _finish_stage(
                    stage_timings,
                    "build_matched_baseline_dataset",
                    matched_start,
                )
                if config.reweight_matched_pe_native_loss:
                    reweight_start = _stage(
                        stage_timings,
                        "reweight_matched_baseline_dataset",
                    )
                    matched_baseline_dataset_path = _reweight_matched_policyengine_us_baseline_dataset(
                        matched_baseline_dataset_path,
                        config.output_matched_baseline_dataset_path
                        or (Path(temp_dir) / "matched_baseline_policyengine_us_reweighted.h5"),
                        period=build_config.policyengine_dataset_year or config.target_period,
                        epochs=config.matched_baseline_reweight_epochs,
                        l0_lambda=config.matched_baseline_reweight_l0_lambda,
                        seed=config.matched_baseline_reweight_seed,
                        policyengine_us_data_repo=config.policyengine_us_data_repo,
                    )
                    _finish_stage(
                        stage_timings,
                        "reweight_matched_baseline_dataset",
                        reweight_start,
                    )
                matched_score_start = _stage(
                    stage_timings,
                    "evaluate_matched_pe_native_loss",
                )
                matched_pe_native_scores = compute_us_pe_native_scores(
                    candidate_dataset_path=dataset_to_score,
                    baseline_dataset_path=matched_baseline_dataset_path,
                    period=build_config.policyengine_dataset_year or config.target_period,
                    policyengine_us_data_repo=config.policyengine_us_data_repo,
                )
                _finish_stage(
                    stage_timings,
                    "evaluate_matched_pe_native_loss",
                    matched_score_start,
                )
            if config.output_policyengine_dataset_path is not None:
                policyengine_dataset_path = _copy_output_file(
                    dataset_to_score,
                    config.output_policyengine_dataset_path,
                )
            if config.output_pe_native_target_delta_path is not None:
                delta_start = _stage(stage_timings, "evaluate_pe_native_target_deltas")
                pe_native_target_deltas = compare_us_pe_native_target_deltas(
                    from_dataset_path=str(config.baseline_dataset),
                    to_dataset_path=dataset_to_score,
                    period=build_config.policyengine_dataset_year or config.target_period,
                    top_k=config.pe_native_target_delta_top_k,
                    policyengine_us_data_repo=config.policyengine_us_data_repo,
                )
                _finish_stage(
                    stage_timings,
                    "evaluate_pe_native_target_deltas",
                    delta_start,
                )
                write_delta_start = _stage(
                    stage_timings,
                    "write_pe_native_target_delta_json",
                )
                destination = Path(config.output_pe_native_target_delta_path)
                destination.parent.mkdir(parents=True, exist_ok=True)
                destination.write_text(
                    json.dumps(
                        _json_compatible_value(pe_native_target_deltas),
                        indent=2,
                        sort_keys=True,
                    )
                )
                _finish_stage(
                    stage_timings,
                    "write_pe_native_target_delta_json",
                    write_delta_start,
                )
            if config.output_pe_native_support_audit_path is not None:
                support_start = _stage(
                    stage_timings,
                    "evaluate_pe_native_support_audit",
                )
                pe_native_support_audit = compute_us_pe_native_support_audit(
                    candidate_dataset_path=dataset_to_score,
                    baseline_dataset_path=str(config.baseline_dataset),
                    period=build_config.policyengine_dataset_year or config.target_period,
                    policyengine_us_data_repo=config.policyengine_us_data_repo,
                )
                _finish_stage(
                    stage_timings,
                    "evaluate_pe_native_support_audit",
                    support_start,
                )
                write_support_start = _stage(
                    stage_timings,
                    "write_pe_native_support_audit_json",
                )
                destination = Path(config.output_pe_native_support_audit_path)
                destination.parent.mkdir(parents=True, exist_ok=True)
                destination.write_text(
                    json.dumps(
                        _json_compatible_value(pe_native_support_audit),
                        indent=2,
                        sort_keys=True,
                    )
                )
                _finish_stage(
                    stage_timings,
                    "write_pe_native_support_audit_json",
                    write_support_start,
                )
            if pe_stage_started:
                _finish_stage(stage_timings, "evaluate_pe_native_loss", start)
    elif config.output_policyengine_dataset_path is not None:
        start = _stage(stage_timings, "write_policyengine_dataset")
        policyengine_dataset_path = str(
            pipeline.export_policyengine_dataset(
                build_result,
                config.output_policyengine_dataset_path,
                direct_override_variables=build_config.policyengine_direct_override_variables,
            )
        )
        _finish_stage(stage_timings, "write_policyengine_dataset", start)

    total_seconds = perf_counter() - total_start
    result = USMicroplexPerformanceHarnessResult(
        config=config,
        build_config=build_config,
        build_result=build_result,
        source_names=precalibration.fusion_plan.source_names,
        stage_timings=stage_timings,
        total_seconds=total_seconds,
        parity_run=parity_run,
        pe_native_scores=pe_native_scores,
        matched_pe_native_scores=matched_pe_native_scores,
        pe_native_target_deltas=pe_native_target_deltas,
        pe_native_support_audit=pe_native_support_audit,
        policyengine_dataset_path=policyengine_dataset_path,
        matched_baseline_dataset_path=(
            matched_baseline_dataset_path if needs_pe_dataset else None
        ),
    )
    if config.output_json_path is not None:
        start = _stage(stage_timings, "write_output_json")
        result.save(config.output_json_path)
        _finish_stage(stage_timings, "write_output_json", start)
    return result


__all__ = [
    "USMicroplexPerformanceHarnessConfig",
    "USMicroplexPerformanceHarnessRequest",
    "USMicroplexPerformanceHarnessResult",
    "USMicroplexCalibrationCacheEntry",
    "USMicroplexPreCalibrationCacheEntry",
    "USMicroplexPerformanceSession",
    "default_fast_calibration_target_variables",
    "run_us_microplex_performance_harness",
    "warm_us_microplex_parity_cache",
]
