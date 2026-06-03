import json

import h5py
import numpy as np

from microplex_us.pipelines.export_lineage_manifest import (
    build_export_lineage_manifest,
)


def _columns_by_name(payload):
    return {column["column"]: column for column in payload["columns"]}


def test_export_lineage_manifest_tracks_source_backed_blocks():
    payload = build_export_lineage_manifest()
    columns = _columns_by_name(payload)

    for column in (
        "business_is_sstb",
        "home_mortgage_interest",
        "reported_has_medicaid_health_coverage_at_interview",
        "ssn_card_type",
        "weekly_hours_worked_before_lsr",
    ):
        assert columns[column]["has_source_lineage"]

    assert (
        columns["selected_marketplace_plan_benchmark_ratio"]["export_path_status"]
        == "default_only"
    )
    assert not columns["selected_marketplace_plan_benchmark_ratio"][
        "has_source_lineage"
    ]


def test_export_lineage_manifest_flags_populated_ecps_default_only_column(tmp_path):
    contract_path = tmp_path / "contract.json"
    contract_path.write_text(
        json.dumps(
            {
                "required": [
                    "selected_marketplace_plan_benchmark_ratio",
                    "weekly_hours_worked_before_lsr",
                ],
                "forbidden": [],
            }
        )
    )
    baseline_path = tmp_path / "baseline.h5"
    with h5py.File(baseline_path, "w") as handle:
        selected = handle.create_group("selected_marketplace_plan_benchmark_ratio")
        selected.create_dataset("2024", data=np.array([0.8, 1.0]))
        weekly_hours = handle.create_group("weekly_hours_worked_before_lsr")
        weekly_hours.create_dataset("2024", data=np.array([0.0, 40.0]))

    payload = build_export_lineage_manifest(
        contract_path=contract_path,
        support_baseline=baseline_path,
    )

    issues = {issue["column"]: issue for issue in payload["issues"]}
    assert issues == {
        "selected_marketplace_plan_benchmark_ratio": {
            "column": "selected_marketplace_plan_benchmark_ratio",
            "ecps_support_requirement": "numeric_nonzero",
            "export_path_status": "default_only",
            "issue": "ecps_populated_export_has_no_source_lineage",
        }
    }
