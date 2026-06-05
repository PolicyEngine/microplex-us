"""Configuration and query helpers for PE-US-data checkpoint rebuilds."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

import h5py
from microplex.core import SourceQuery

from microplex_us.pipelines.pe_us_data_rebuild import (
    default_policyengine_us_data_rebuild_config,
)
from microplex_us.pipelines.us import USMicroplexBuildConfig

if TYPE_CHECKING:
    from microplex.core import SourceProvider

DEFAULT_ARCH_CALIBRATION_TARGET_PROFILE = "pe_native_broad_source_backed"


def _resolve_checkpoint_calibration_target_variables(
    calibration_target_variables: tuple[str, ...],
) -> tuple[str, ...]:
    return tuple(calibration_target_variables)


def _normalize_path_value(value: str | Path | None) -> str | None:
    if value is None:
        return None
    return str(Path(value).expanduser())


def _normalize_arch_targets_db_value(
    value: str | Path | tuple[str | Path, ...] | None,
) -> str | tuple[str, ...] | None:
    if value is None:
        return None
    if isinstance(value, (str, Path)):
        return str(Path(value).expanduser())
    return tuple(str(Path(path).expanduser()) for path in value)


def _validate_checkpoint_config_context(
    config: USMicroplexBuildConfig,
    *,
    policyengine_baseline_dataset: str | Path,
    policyengine_targets_db: str | Path,
    arch_targets_db: str | Path | tuple[str | Path, ...] | None,
    calibration_target_source: Literal["policyengine", "arch"],
    target_period: int,
    target_profile: str,
    calibration_target_profile: str | None,
    target_variables: tuple[str, ...],
    target_domains: tuple[str, ...],
    target_geo_levels: tuple[str, ...],
    calibration_target_variables: tuple[str, ...],
    calibration_target_domains: tuple[str, ...],
    calibration_target_geo_levels: tuple[str, ...],
) -> None:
    expected_pairs = {
        "policyengine_baseline_dataset": _normalize_path_value(
            policyengine_baseline_dataset
        ),
        "policyengine_targets_db": _normalize_path_value(policyengine_targets_db),
        "arch_targets_db": _normalize_arch_targets_db_value(arch_targets_db),
        "calibration_target_source": calibration_target_source,
        "policyengine_dataset_year": int(target_period),
        "policyengine_target_period": int(target_period),
        "policyengine_target_profile": target_profile,
        "policyengine_calibration_target_profile": (
            calibration_target_profile
            or (
                DEFAULT_ARCH_CALIBRATION_TARGET_PROFILE
                if calibration_target_source == "arch"
                else target_profile
            )
        ),
        "policyengine_target_variables": tuple(target_variables),
        "policyengine_target_domains": tuple(target_domains),
        "policyengine_target_geo_levels": tuple(target_geo_levels),
        "policyengine_calibration_target_variables": (
            _resolve_checkpoint_calibration_target_variables(
                calibration_target_variables
            )
        ),
        "policyengine_calibration_target_domains": tuple(calibration_target_domains),
        "policyengine_calibration_target_geo_levels": tuple(
            calibration_target_geo_levels
        ),
    }
    for key, expected in expected_pairs.items():
        observed = getattr(config, key)
        if observed != expected:
            raise ValueError(
                "Explicit config does not match the requested PE rebuild context for "
                f"{key}: expected {expected!r}, observed {observed!r}"
            )


def _validate_query_keys(
    provider_names: tuple[str, ...],
    queries: dict[str, SourceQuery],
) -> None:
    unexpected = sorted(set(queries) - set(provider_names))
    if unexpected:
        allowed = ", ".join(provider_names)
        unexpected_text = ", ".join(unexpected)
        raise ValueError(
            "Checkpoint queries include unknown provider keys: "
            f"{unexpected_text}. Expected one of: {allowed}"
        )


def _infer_policyengine_baseline_household_weight_sum(
    baseline_dataset: str | Path,
    *,
    target_period: int,
) -> float | None:
    """Best-effort household-weight target inferred from the PE baseline dataset."""

    dataset_path = Path(baseline_dataset).expanduser()
    if not dataset_path.exists():
        return None
    try:
        with h5py.File(dataset_path, "r") as handle:
            weights = handle.get("household_weight")
            if weights is None:
                return None
            period_key = str(int(target_period))
            if period_key not in weights:
                return None
            weight_sum = float(weights[period_key][...].sum())
    except (FileNotFoundError, OSError, ValueError):
        return None
    return weight_sum if weight_sum > 0.0 else None


def default_policyengine_us_data_rebuild_checkpoint_config(
    *,
    policyengine_baseline_dataset: str | Path,
    policyengine_targets_db: str | Path,
    arch_targets_db: str | Path | tuple[str | Path, ...] | None = None,
    calibration_target_source: Literal["policyengine", "arch"] = "policyengine",
    target_period: int = 2024,
    target_profile: str = "pe_native_broad",
    calibration_target_profile: str | None = None,
    target_variables: tuple[str, ...] = (),
    target_domains: tuple[str, ...] = (),
    target_geo_levels: tuple[str, ...] = (),
    calibration_target_variables: tuple[str, ...] = (),
    calibration_target_domains: tuple[str, ...] = (),
    calibration_target_geo_levels: tuple[str, ...] = (),
    **overrides: Any,
) -> USMicroplexBuildConfig:
    """Return the canonical rebuild config with required PE comparison context."""

    resolved_target_period = int(target_period)
    if calibration_target_source not in {"policyengine", "arch"}:
        raise ValueError(
            "calibration_target_source must be 'policyengine' or 'arch', "
            f"got {calibration_target_source!r}"
        )
    resolved_arch_targets_db = _normalize_arch_targets_db_value(arch_targets_db)
    if calibration_target_source == "arch" and resolved_arch_targets_db is None:
        raise ValueError(
            "arch_targets_db is required when calibration_target_source='arch'"
        )
    resolved_calibration_target_profile = calibration_target_profile or (
        DEFAULT_ARCH_CALIBRATION_TARGET_PROFILE
        if calibration_target_source == "arch"
        else target_profile
    )
    resolved_baseline_weight_sum = _infer_policyengine_baseline_household_weight_sum(
        policyengine_baseline_dataset,
        target_period=resolved_target_period,
    )
    resolved_overrides = dict(overrides)
    infer_total_weight_targets = (
        resolved_baseline_weight_sum is not None
        and resolved_overrides.get("calibration_backend") != "none"
    )
    if infer_total_weight_targets:
        resolved_overrides.setdefault(
            "policyengine_selection_target_total_weight",
            resolved_baseline_weight_sum,
        )
        if not resolved_overrides.get(
            "policyengine_calibration_rescale_to_input_weight_sum",
            False,
        ):
            resolved_overrides.setdefault(
                "policyengine_calibration_target_total_weight",
                resolved_baseline_weight_sum,
            )
            resolved_overrides.setdefault(
                "policyengine_calibration_rescale_to_target_total_weight",
                True,
            )
    return default_policyengine_us_data_rebuild_config(
        policyengine_baseline_dataset=str(policyengine_baseline_dataset),
        policyengine_targets_db=str(policyengine_targets_db),
        arch_targets_db=resolved_arch_targets_db,
        calibration_target_source=calibration_target_source,
        policyengine_dataset_year=resolved_target_period,
        policyengine_target_period=resolved_target_period,
        policyengine_target_profile=target_profile,
        policyengine_calibration_target_profile=resolved_calibration_target_profile,
        policyengine_target_variables=tuple(target_variables),
        policyengine_target_domains=tuple(target_domains),
        policyengine_target_geo_levels=tuple(target_geo_levels),
        policyengine_calibration_target_variables=(
            _resolve_checkpoint_calibration_target_variables(
                calibration_target_variables
            )
        ),
        policyengine_calibration_target_domains=tuple(calibration_target_domains),
        policyengine_calibration_target_geo_levels=tuple(calibration_target_geo_levels),
        **resolved_overrides,
    )


def default_policyengine_us_data_rebuild_queries(
    providers: tuple[SourceProvider, ...] | list[SourceProvider],
    *,
    cps_sample_n: int | None = None,
    puf_sample_n: int | None = None,
    donor_sample_n: int | None = None,
    cps_state_age_floor: int | None = 1,
    donor_state_age_floor: int | None = 1,
    random_seed: int = 0,
) -> dict[str, SourceQuery]:
    """Return default provider queries for a rebuild checkpoint smoke run."""

    from microplex_us.data_sources.cps import CPSASECSourceProvider
    from microplex_us.data_sources.donor_surveys import DonorSurveySourceProvider
    from microplex_us.data_sources.puf import PUFSourceProvider

    resolved_donor_sample_n = donor_sample_n
    if resolved_donor_sample_n is None:
        source_sample_sizes = tuple(
            int(sample_n)
            for sample_n in (cps_sample_n, puf_sample_n)
            if sample_n is not None
        )
        if source_sample_sizes:
            resolved_donor_sample_n = max(source_sample_sizes)

    queries: dict[str, SourceQuery] = {}
    for provider in providers:
        sample_n: int | None = None
        if isinstance(provider, CPSASECSourceProvider):
            sample_n = cps_sample_n
        elif isinstance(provider, PUFSourceProvider):
            sample_n = puf_sample_n
        elif isinstance(provider, DonorSurveySourceProvider):
            sample_n = resolved_donor_sample_n
        if sample_n is None:
            continue
        provider_filters = {
            "sample_n": int(sample_n),
            "random_seed": int(random_seed),
        }
        if (
            isinstance(provider, CPSASECSourceProvider)
            and cps_state_age_floor is not None
        ):
            provider_filters["state_age_floor"] = int(cps_state_age_floor)
        elif (
            isinstance(provider, DonorSurveySourceProvider)
            and donor_state_age_floor is not None
        ):
            provider_filters["state_age_floor"] = int(donor_state_age_floor)
        queries[provider.descriptor.name] = SourceQuery(
            provider_filters=provider_filters
        )
    return queries
