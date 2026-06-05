# ruff: noqa: F401
"""Concrete checkpoint runner for the PE-US-data rebuild profile.

This module is intentionally thin. Implementation lives in thematic
``pe_us_data_rebuild_checkpoint_*`` modules while this file preserves the public
import path and ``python -m`` entry point.
"""

from __future__ import annotations

from microplex_us.pipelines.artifacts import (
    build_and_save_versioned_us_microplex_from_source_providers,
)
from microplex_us.pipelines.pe_us_data_rebuild import (
    default_policyengine_us_data_rebuild_source_providers,
)
from microplex_us.pipelines.pe_us_data_rebuild_checkpoint_ablation import (
    _build_checkpoint_imputation_ablation_payload,
)
from microplex_us.pipelines.pe_us_data_rebuild_checkpoint_artifacts import (
    _load_checkpoint_manifest,
    _load_checkpoint_manifest_if_available,
    _load_checkpoint_versioned_artifacts,
    _load_resume_dataframe_artifact,
    _load_resume_json_artifact,
    _load_resume_policyengine_tables,
    _load_resume_targets,
)
from microplex_us.pipelines.pe_us_data_rebuild_checkpoint_cli import main
from microplex_us.pipelines.pe_us_data_rebuild_checkpoint_common import (
    LOGGER,
    _emit_checkpoint_progress,
    _resolve_policyengine_us_runtime_version,
    _root_logger_has_handlers,
    _write_json_atomically,
)
from microplex_us.pipelines.pe_us_data_rebuild_checkpoint_config import (
    _infer_policyengine_baseline_household_weight_sum,
    _normalize_arch_targets_db_value,
    _normalize_path_value,
    _resolve_checkpoint_calibration_target_variables,
    _validate_checkpoint_config_context,
    _validate_query_keys,
    default_policyengine_us_data_rebuild_checkpoint_config,
    default_policyengine_us_data_rebuild_queries,
)
from microplex_us.pipelines.pe_us_data_rebuild_checkpoint_evidence import (
    _refresh_checkpoint_data_flow_snapshot,
    attach_policyengine_us_data_rebuild_checkpoint_evidence,
)
from microplex_us.pipelines.pe_us_data_rebuild_checkpoint_resume import (
    _checkpoint_resume_extra_artifact_requirements,
    _complete_resume_run_profile_stage,
    _is_artifact_backed_checkpoint_resume_stage,
    _load_checkpoint_source_frames,
    _resolve_checkpoint_resume_artifact_root,
    _resume_checkpoint_build_from_saved_stage,
    _resume_checkpoint_build_from_source_stage,
    _resume_provider_context_from_manifest,
    _run_checkpoint_calibration_resume_stage,
    _run_checkpoint_policyengine_entity_resume_stage,
    _run_policyengine_us_data_rebuild_checkpoint_resume,
)
from microplex_us.pipelines.pe_us_data_rebuild_checkpoint_runner import (
    run_policyengine_us_data_rebuild_checkpoint,
)
from microplex_us.pipelines.pe_us_data_rebuild_checkpoint_types import (
    PEUSDataRebuildCheckpointEvidenceResult,
    PEUSDataRebuildCheckpointResult,
)
from microplex_us.pipelines.versioned_artifacts import (
    _finalize_versioned_build_artifacts,
)

__all__ = [
    "PEUSDataRebuildCheckpointEvidenceResult",
    "PEUSDataRebuildCheckpointResult",
    "attach_policyengine_us_data_rebuild_checkpoint_evidence",
    "default_policyengine_us_data_rebuild_checkpoint_config",
    "default_policyengine_us_data_rebuild_queries",
    "main",
    "run_policyengine_us_data_rebuild_checkpoint",
]


if __name__ == "__main__":
    main()
