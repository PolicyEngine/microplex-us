import json

import numpy as np
import pytest

from microplex_us.pipelines.dashboard import build_dashboard_payload
from microplex_us.pipelines.run_contract import RunContractWriter


def test_dashboard_payload_marks_missing_pe_l0_comparators(tmp_path):
    artifacts = tmp_path / "artifacts"
    run_dir = artifacts / "latest"
    run_dir.mkdir(parents=True)
    (run_dir / "scores.json").write_text(
        json.dumps(
            [
                {
                    "metric": "pe_native_broad_loss",
                    "period": 2024,
                    "summary": {
                        "baseline_enhanced_cps_native_loss": 0.0977,
                        "candidate_beats_baseline": True,
                        "candidate_enhanced_cps_native_loss": 0.0252,
                        "enhanced_cps_native_loss_delta": -0.0725,
                        "n_targets_kept": 2805,
                        "n_targets_total": 2816,
                    },
                    "broad_loss": {
                        "baseline_dataset": "enhanced_cps_2024.h5",
                        "candidate_dataset": "pe_l0_candidate.h5",
                        "baseline_weight_sum": 153.8,
                        "candidate_weight_sum": 153.7,
                    },
                }
            ]
        )
    )
    screen_dir = artifacts / "local_screen"
    screen_dir.mkdir()
    (screen_dir / "split_loss_summary.json").write_text(
        json.dumps(
            {
                "candidate": "cd_age_w8",
                "broad_objective_on_latest_pe_matrix_rows": 0.0262,
                "latest_pe_baseline_broad_loss": 0.0977,
                "cd_age_mean_abs_relative_error": 0.0155,
            }
        )
    )
    (screen_dir / "scores.json").write_text(
        json.dumps(
            [
                {
                    "summary": {
                        "baseline_enhanced_cps_native_loss": 0.0977,
                        "candidate_beats_baseline": True,
                        "candidate_enhanced_cps_native_loss": 0.0263,
                        "enhanced_cps_native_loss_delta": -0.0714,
                    }
                }
            ]
        )
    )
    local_l0_dir = artifacts / "pe_local_area_l0_compare"
    local_l0_dir.mkdir()
    (local_l0_dir / "pe_local_area_l0_state_stack_vs_legacy_ecps.json").write_text(
        json.dumps(
            {
                "metric": "enhanced_cps_native_loss_target_delta",
                "from_dataset": "legacy-pe-ecps",
                "to_dataset": "pe-local-area-l0-state-stack",
                "state_score_count": 51,
                "state_weight_sum": 121.0,
                "summary": {
                    "n_targets": 2814,
                    "from_loss": 0.1747,
                    "to_loss": 3.0,
                    "loss_delta": 2.8253,
                },
            }
        )
    )
    microplex_l0_dir = artifacts / "microplex_actual_l0"
    microplex_l0_dir.mkdir()
    (microplex_l0_dir / "unified_diagnostics.csv").write_text(
        "\n".join(
            [
                "target,true_value,estimate,rel_error,abs_rel_error,achievable",
                "a,100,90,-0.10,0.10,True",
                "b,100,100,0.00,0.00,True",
            ]
        )
    )
    (microplex_l0_dir / "unified_run_config.json").write_text(
        json.dumps({"n_clones": 10, "epochs": 300})
    )
    np.save(microplex_l0_dir / "calibration_weights.npy", np.array([1.0, 0.0, 200.0]))
    target_diagnostics = artifacts / "pe_native_target_diagnostics_current.json"
    target_diagnostics.write_text(
        json.dumps(
            {
                "dataset_labels": {"from": "PE", "to": "Microplex"},
                "summary": {"n_targets": 0},
                "targets": [],
            }
        )
    )
    pe_repo = tmp_path / "policyengine-us-data"
    for dirname, epochs, mean_error in [
        ("local_net_worth_100", 100, 5.5),
        ("local_net_worth_100_e300", 300, 2.5),
    ]:
        model_dir = (
            pe_repo
            / "policyengine_us_data"
            / "storage"
            / "calibration"
            / dirname
        )
        model_dir.mkdir(parents=True)
        (model_dir / "unified_run_config.json").write_text(
            json.dumps(
                {
                    "dataset": "source_imputed_stratified_extended_cps_2024.h5",
                    "db_path": "policy_data.db",
                    "n_clones": 430,
                    "epochs": epochs,
                    "n_targets": 2,
                    "n_records": 3_000_000,
                    "weight_sum": 153.0,
                    "weight_nonzero": 1000,
                    "mean_error_pct": mean_error,
                }
            )
        )
        (model_dir / "unified_diagnostics.csv").write_text(
            "\n".join(
                [
                    "target,true_value,estimate,rel_error,abs_rel_error,achievable",
                    "a,100,95,-0.05,0.05,True",
                    "b,100,80,-0.20,0.20,True",
                ]
            )
        )

    payload = build_dashboard_payload(
        artifact_root=artifacts,
        target_diagnostics_path=target_diagnostics,
        policyengine_us_data_repo=pe_repo,
        include_tmux=False,
    )

    assertions = payload["run_board"]["assertions"]
    assert assertions["microplex_beats_legacy_ecps_latest_pe_broad"] is True
    assert assertions["policyengine_small_l0_weight_package_available"] is True
    assert assertions["policyengine_big_l0_weight_package_available"] is True
    assert assertions["microplex_vs_small_l0_complete"] is False
    assert assertions["microplex_vs_big_l0_complete"] is False
    assert (
        assertions["microplex_vs_all_three_pe_models_on_both_metrics"] is False
    )
    assert assertions["policyengine_materialized_l0_same_harness_available"] is True
    assert assertions["apples_to_apples_groups_available"] is True
    assert payload["run_board"]["score_runs"][0]["candidate_loss"] == 0.0252
    assert payload["run_board"]["local_target_screens"][0]["label"] == "cd_age_w8"
    assert payload["run_board"]["local_target_screens"][0]["status"] == (
        "screen_scored_latest_pe"
    )
    assert payload["run_board"]["local_target_screens"][0][
        "pe_native_broad_loss"
    ] == 0.0263
    assert (
        payload["run_board"]["materialized_policyengine_l0_scores"][0][
            "candidate_loss"
        ]
        == 3.0
    )
    actual_l0_runs = payload["run_board"]["actual_l0_objective_runs"]
    assert actual_l0_runs[0]["model_id"] == "microplex_actual_l0"
    assert actual_l0_runs[0]["actual_l0_data_loss"] == pytest.approx(100 / (101**2))
    assert actual_l0_runs[0]["weights"]["nonzero"] == 2
    groups = {row["id"]: row for row in payload["run_board"]["apples_to_apples"]}
    assert groups["latest_pe_broad"]["rows"][0]["score"] == 0.0977
    assert groups["legacy_broad"]["rows"][2]["model_id"] == (
        "policyengine_local_area_l0_state_stack"
    )
    models = {
        row["id"]: row for row in payload["run_board"]["policyengine_l0_models"]
    }
    assert models["policyengine_small_l0"]["epochs"] == 100
    assert (
        models["policyengine_big_l0"]["diagnostics"][
            "mean_abs_relative_error_pct"
        ]
        == 12.5
    )
    assert models["policyengine_big_l0"]["diagnostics"][
        "actual_l0_objective"
    ] == "sum(((estimate - target) / (target + 1)) ** 2)"
    assert models["policyengine_big_l0"]["diagnostics"][
        "actual_l0_data_loss"
    ] == pytest.approx(425 / (101**2))


def test_dashboard_payload_wires_materialized_pe_l0_score_jsons(tmp_path):
    artifacts = tmp_path / "artifacts"
    artifacts.mkdir()
    latest_dir = artifacts / "latest_microplex"
    legacy_dir = artifacts / "legacy_microplex"
    latest_dir.mkdir()
    legacy_dir.mkdir()
    (latest_dir / "scores.json").write_text(
        json.dumps(
            [
                {
                    "metric": "pe_native_broad_loss",
                    "summary": {
                        "baseline_enhanced_cps_native_loss": 0.10,
                        "candidate_beats_baseline": True,
                        "candidate_enhanced_cps_native_loss": 0.03,
                        "n_targets_kept": 2805,
                    },
                    "broad_loss": {
                        "candidate_dataset": "microplex_latest.h5",
                        "baseline_dataset": "enhanced_cps_2024.h5",
                    },
                }
            ]
        )
    )
    (legacy_dir / "scores.json").write_text(
        json.dumps(
            [
                {
                    "metric": "pe_native_broad_loss",
                    "summary": {
                        "baseline_enhanced_cps_native_loss": 0.17,
                        "candidate_beats_baseline": True,
                        "candidate_enhanced_cps_native_loss": 0.06,
                        "n_targets_kept": 2814,
                    },
                    "broad_loss": {
                        "candidate_dataset": "microplex_legacy.h5",
                        "baseline_dataset": "enhanced_cps_2024.h5",
                    },
                }
            ]
        )
    )
    score_dir = artifacts / "pe_l0_clone_apples_to_apples"
    score_dir.mkdir()
    for metric, targets, small_loss, big_loss in [
        ("legacy_targets", 2814, 0.15, 0.12),
        ("new_targets", 2805, 0.09, 0.08),
    ]:
        for label, loss in [
            ("pe_small_l0", small_loss),
            ("pe_big_l0", big_loss),
        ]:
            (score_dir / f"{metric}_{label}_score.json").write_text(
                json.dumps(
                    {
                        "metric": "enhanced_cps_native_loss",
                        "candidate_dataset": f"/tmp/{label}.h5",
                        "baseline_dataset": "/tmp/enhanced_cps_2024.h5",
                        "baseline_enhanced_cps_native_loss": (
                            0.10 if metric == "new_targets" else 0.17
                        ),
                        "candidate_beats_baseline": loss < (
                            0.10 if metric == "new_targets" else 0.17
                        ),
                        "candidate_enhanced_cps_native_loss": loss,
                        "enhanced_cps_native_loss_delta": loss
                        - (0.10 if metric == "new_targets" else 0.17),
                        "n_targets_kept": targets,
                        "n_targets_total": targets + 10,
                    }
                )
            )

    pe_repo = tmp_path / "policyengine-us-data"
    for dirname in ["local_net_worth_100", "local_net_worth_100_e300"]:
        model_dir = (
            pe_repo
            / "policyengine_us_data"
            / "storage"
            / "calibration"
            / dirname
        )
        model_dir.mkdir(parents=True)
        (model_dir / "unified_run_config.json").write_text(
            json.dumps({"n_targets": 2, "epochs": 100})
        )
        (model_dir / "unified_diagnostics.csv").write_text(
            "\n".join(
                [
                    "target,true_value,estimate,rel_error,abs_rel_error,achievable",
                    "a,100,95,-0.05,0.05,True",
                ]
            )
        )

    payload = build_dashboard_payload(
        artifact_root=artifacts,
        target_diagnostics_path=artifacts / "missing.json",
        policyengine_us_data_repo=pe_repo,
        include_tmux=False,
    )

    assertions = payload["run_board"]["assertions"]
    assert assertions["microplex_vs_small_l0_complete"] is True
    assert assertions["microplex_vs_big_l0_complete"] is True
    assert assertions["microplex_vs_all_three_pe_models_on_both_metrics"] is True
    groups = {row["id"]: row for row in payload["run_board"]["apples_to_apples"]}
    latest_rows = {
        row["model_id"]: row for row in groups["latest_pe_broad"]["rows"]
    }
    legacy_rows = {
        row["model_id"]: row for row in groups["legacy_broad"]["rows"]
    }
    assert latest_rows["policyengine_small_l0"]["score"] == 0.09
    assert latest_rows["policyengine_big_l0"]["score"] == 0.08
    assert legacy_rows["policyengine_small_l0"]["score"] == 0.15
    assert legacy_rows["policyengine_big_l0"]["score"] == 0.12


def test_dashboard_payload_reads_run_contract_summaries(tmp_path):
    artifacts = tmp_path / "artifacts"
    writer = RunContractWriter(
        artifacts / "contracted_run",
        run_id="contracted-run",
        attempt_id="attempt-1",
    )
    with writer.stage("preflight"):
        pass

    payload = build_dashboard_payload(
        artifact_root=artifacts,
        policyengine_us_data_repo=None,
        include_tmux=False,
    )

    contracts = payload["run_board"]["run_contracts"]
    assert len(contracts) == 1
    assert contracts[0]["status_source"] == "contract"
    assert contracts[0]["run_id"] == "contracted-run"
    assert contracts[0]["status"] == "running"
    assert contracts[0]["completed_stages"] == ["preflight"]
