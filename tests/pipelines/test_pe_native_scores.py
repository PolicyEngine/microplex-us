"""Tests for PE-native scoring helpers."""

from __future__ import annotations

import json
import os
from types import SimpleNamespace

from microplex_us.pipelines.pe_native_scores import (
    PolicyEngineUSEnhancedCPSNativeScores,
    annotate_pe_native_target_db_matches,
    build_policyengine_us_data_pythonpath,
    build_policyengine_us_data_subprocess_env,
    compare_us_pe_native_target_deltas,
    compute_batch_us_pe_native_scores,
    compute_batch_us_pe_native_support_audits,
    compute_batch_us_pe_native_target_deltas,
    compute_us_pe_native_scores,
    compute_us_pe_native_support_audit,
    parse_pe_native_target_lookup_key,
    resolve_policyengine_us_data_python,
    write_us_pe_native_scores,
    write_us_pe_native_target_diagnostics,
)

_MICROPLEX_SRC = __import__("microplex_us").__path__[0].rsplit("/microplex_us", 1)[0]


def test_compute_us_pe_native_scores_wraps_broad_loss(monkeypatch, tmp_path) -> None:
    candidate = tmp_path / "candidate.h5"
    baseline = tmp_path / "baseline.h5"
    candidate.write_text("candidate")
    baseline.write_text("baseline")

    monkeypatch.setattr(
        "microplex_us.pipelines.pe_native_scores.compute_policyengine_us_enhanced_cps_native_scores",
        lambda *_args, **_kwargs: PolicyEngineUSEnhancedCPSNativeScores(
            metric="enhanced_cps_native_loss",
            period=2024,
            candidate_dataset=str(candidate),
            baseline_dataset=str(baseline),
            candidate_enhanced_cps_native_loss=0.25,
            baseline_enhanced_cps_native_loss=0.5,
            enhanced_cps_native_loss_delta=-0.25,
            candidate_unweighted_msre=0.3,
            baseline_unweighted_msre=0.6,
            unweighted_msre_delta=-0.3,
            n_targets_total=2863,
            n_targets_kept=2853,
            n_targets_zero_dropped=10,
            n_targets_bad_dropped=10,
            n_national_targets=2000,
            n_state_targets=853,
            candidate_weight_sum=100.0,
            baseline_weight_sum=200.0,
            family_breakdown=(
                {
                    "family": "state_age_distribution",
                    "n_targets": 900,
                    "candidate_loss_contribution": 0.1,
                    "baseline_loss_contribution": 0.05,
                    "loss_contribution_delta": 0.05,
                    "candidate_mean_weighted_loss": 0.2,
                    "baseline_mean_weighted_loss": 0.1,
                    "candidate_mean_unweighted_msre": 0.3,
                    "baseline_mean_unweighted_msre": 0.2,
                    "unweighted_msre_delta": 0.1,
                },
            ),
        ),
    )

    payload = compute_us_pe_native_scores(
        candidate_dataset_path=candidate,
        baseline_dataset_path=baseline,
        period=2024,
    )

    assert payload["metric"] == "enhanced_cps_native_loss"
    assert payload["summary"]["candidate_enhanced_cps_native_loss"] == 0.25
    assert payload["summary"]["baseline_enhanced_cps_native_loss"] == 0.5
    assert payload["summary"]["enhanced_cps_native_loss_delta"] == -0.25
    assert payload["summary"]["candidate_beats_baseline"] is True
    assert payload["summary"]["candidate_unweighted_msre"] == 0.3
    assert payload["summary"]["n_targets_kept"] == 2853
    assert payload["family_breakdown"][0]["family"] == "state_age_distribution"
    assert payload["broad_loss"]["family_breakdown"][0]["n_targets"] == 900


def test_write_us_pe_native_scores_persists_payload(monkeypatch, tmp_path) -> None:
    candidate = tmp_path / "candidate.h5"
    baseline = tmp_path / "baseline.h5"
    output_path = tmp_path / "native.json"
    candidate.write_text("candidate")
    baseline.write_text("baseline")

    monkeypatch.setattr(
        "microplex_us.pipelines.pe_native_scores.compute_us_pe_native_scores",
        lambda **_kwargs: {
            "metric": "enhanced_cps_native_loss",
            "summary": {
                "candidate_enhanced_cps_native_loss": 0.2,
                "baseline_enhanced_cps_native_loss": 0.4,
                "enhanced_cps_native_loss_delta": -0.2,
            },
        },
    )

    written = write_us_pe_native_scores(
        output_path,
        candidate_dataset_path=candidate,
        baseline_dataset_path=baseline,
    )

    assert written == output_path
    assert output_path.exists()


def test_compute_batch_us_pe_native_scores_wraps_multiple_candidates(
    monkeypatch,
    tmp_path,
) -> None:
    candidate_a = tmp_path / "candidate-a.h5"
    candidate_b = tmp_path / "candidate-b.h5"
    baseline = tmp_path / "baseline.h5"
    for path in (candidate_a, candidate_b, baseline):
        path.write_text(path.stem)

    payload = [
        {
            "metric": "enhanced_cps_native_loss",
            "period": 2024,
            "candidate_dataset": str(candidate_a),
            "baseline_dataset": str(baseline),
            "candidate_enhanced_cps_native_loss": 0.25,
            "baseline_enhanced_cps_native_loss": 0.5,
            "enhanced_cps_native_loss_delta": -0.25,
            "candidate_beats_baseline": True,
            "candidate_unweighted_msre": 0.3,
            "baseline_unweighted_msre": 0.6,
            "unweighted_msre_delta": -0.3,
            "n_targets_total": 2865,
            "n_targets_kept": 2853,
            "n_targets_zero_dropped": 10,
            "n_targets_bad_dropped": 10,
            "n_national_targets": 677,
            "n_state_targets": 2176,
            "candidate_weight_sum": 100.0,
            "baseline_weight_sum": 200.0,
            "family_breakdown": [
                {
                    "family": "state_age_distribution",
                    "n_targets": 900,
                    "candidate_loss_contribution": 0.1,
                    "baseline_loss_contribution": 0.05,
                    "loss_contribution_delta": 0.05,
                    "candidate_mean_weighted_loss": 0.2,
                    "baseline_mean_weighted_loss": 0.1,
                    "candidate_mean_unweighted_msre": 0.3,
                    "baseline_mean_unweighted_msre": 0.2,
                    "unweighted_msre_delta": 0.1,
                }
            ],
        },
        {
            "metric": "enhanced_cps_native_loss",
            "period": 2024,
            "candidate_dataset": str(candidate_b),
            "baseline_dataset": str(baseline),
            "candidate_enhanced_cps_native_loss": 0.75,
            "baseline_enhanced_cps_native_loss": 0.5,
            "enhanced_cps_native_loss_delta": 0.25,
            "candidate_beats_baseline": False,
            "candidate_unweighted_msre": 0.8,
            "baseline_unweighted_msre": 0.6,
            "unweighted_msre_delta": 0.2,
            "n_targets_total": 2865,
            "n_targets_kept": 2853,
            "n_targets_zero_dropped": 10,
            "n_targets_bad_dropped": 10,
            "n_national_targets": 677,
            "n_state_targets": 2176,
            "candidate_weight_sum": 120.0,
            "baseline_weight_sum": 200.0,
            "family_breakdown": [
                {
                    "family": "state_agi_distribution",
                    "n_targets": 918,
                    "candidate_loss_contribution": 0.2,
                    "baseline_loss_contribution": 0.05,
                    "loss_contribution_delta": 0.15,
                    "candidate_mean_weighted_loss": 0.4,
                    "baseline_mean_weighted_loss": 0.1,
                    "candidate_mean_unweighted_msre": 0.5,
                    "baseline_mean_unweighted_msre": 0.2,
                    "unweighted_msre_delta": 0.3,
                }
            ],
        },
    ]

    monkeypatch.setattr(
        "microplex_us.pipelines.pe_native_scores.subprocess.run",
        lambda *_args, **_kwargs: SimpleNamespace(
            returncode=0,
            stdout=json.dumps(payload),
            stderr="",
        ),
    )
    monkeypatch.setattr(
        "microplex_us.pipelines.pe_native_scores.resolve_policyengine_us_data_repo_root",
        lambda _repo=None: tmp_path,
    )

    results = compute_batch_us_pe_native_scores(
        candidate_dataset_paths=[candidate_a, candidate_b],
        baseline_dataset_path=baseline,
        period=2024,
        policyengine_us_data_repo=tmp_path,
        policyengine_us_data_python=tmp_path / "python",
    )

    assert len(results) == 2
    assert results[0]["summary"]["candidate_beats_baseline"] is True
    assert results[1]["summary"]["candidate_beats_baseline"] is False
    assert results[1]["broad_loss"]["enhanced_cps_native_loss_delta"] == 0.25
    assert results[0]["family_breakdown"][0]["family"] == "state_age_distribution"
    assert (
        results[1]["broad_loss"]["family_breakdown"][0]["family"]
        == "state_agi_distribution"
    )
    assert results[0]["timing"]["batch_candidate_count"] == 2
    assert results[0]["timing"]["batch_elapsed_seconds"] >= 0.0


def test_build_policyengine_us_data_pythonpath_includes_sibling_microimpute(
    tmp_path,
) -> None:
    repo = tmp_path / "policyengine-us-data"
    (repo / "policyengine_us_data").mkdir(parents=True)
    microimpute = tmp_path / "microimpute"
    (microimpute / "microimpute").mkdir(parents=True)

    pythonpath = build_policyengine_us_data_pythonpath(
        repo,
        existing_pythonpath="/tmp/existing-one:/tmp/existing-two",
    )

    assert pythonpath.split(os.pathsep) == [
        str(repo),
        _MICROPLEX_SRC,
        str(microimpute),
        "/tmp/existing-one",
        "/tmp/existing-two",
    ]


def test_build_policyengine_us_data_subprocess_env_strips_outer_uv_markers(
    tmp_path,
) -> None:
    repo = tmp_path / "policyengine-us-data"
    (repo / "policyengine_us_data").mkdir(parents=True)
    microimpute = tmp_path / "microimpute"
    (microimpute / "microimpute").mkdir(parents=True)

    env = build_policyengine_us_data_subprocess_env(
        repo,
        base_env={
            "HOME": "/tmp/home",
            "PATH": "/usr/bin:/bin",
            "VIRTUAL_ENV": "/tmp/outer-venv",
            "UV_RUN_RECURSION_DEPTH": "1",
            "PYTHONPATH": "/tmp/existing",
            "KEEP_ME": "yes",
        },
    )

    assert env["HOME"] == "/tmp/home"
    assert env["PATH"] == "/usr/bin:/bin"
    assert "KEEP_ME" not in env
    assert "VIRTUAL_ENV" not in env
    assert "UV_RUN_RECURSION_DEPTH" not in env
    assert env["PYTHONPATH"].split(os.pathsep) == [
        str(repo),
        _MICROPLEX_SRC,
        str(microimpute),
        "/tmp/existing",
    ]


def test_resolve_policyengine_us_data_python_preserves_venv_symlink_path(
    tmp_path,
) -> None:
    repo = tmp_path / "policyengine-us-data"
    (repo / "policyengine_us_data").mkdir(parents=True)
    real_python = tmp_path / "real-python"
    real_python.write_text("#!/bin/sh\nexit 0\n")
    real_python.chmod(0o755)
    venv_python = repo / ".venv" / "bin" / "python"
    venv_python.parent.mkdir(parents=True)
    venv_python.symlink_to(real_python)

    resolved = resolve_policyengine_us_data_python(repo_root=repo)

    assert resolved == venv_python


def test_compare_us_pe_native_target_deltas_wraps_subprocess_payload(
    monkeypatch,
    tmp_path,
) -> None:
    before = tmp_path / "before.h5"
    after = tmp_path / "after.h5"
    for path in (before, after):
        path.write_text(path.stem)

    payload = {
        "metric": "enhanced_cps_native_loss_target_delta",
        "period": 2024,
        "from_dataset": str(before),
        "to_dataset": str(after),
        "summary": {
            "n_targets": 2,
            "from_wins": 1,
            "to_wins": 1,
            "ties": 0,
            "from_win_rate": 0.5,
            "to_win_rate": 0.5,
            "from_loss": 0.5,
            "to_loss": 1.0,
            "loss_delta": 0.5,
            "mean_weighted_term_delta": 0.5,
        },
        "family_summaries": [
            {
                "target_family": "national_irs_other",
                "n_targets": 1,
                "to_win_rate": 0.0,
                "loss_delta": 1.5,
            }
        ],
        "scope_summaries": [
            {
                "target_scope": "national",
                "n_targets": 1,
                "to_win_rate": 0.0,
                "loss_delta": 1.5,
            }
        ],
        "targets": [
            {
                "target_name": "nation/irs/example",
                "target_family": "national_irs_other",
                "target_scope": "national",
                "winner": "from",
                "weighted_term_delta": 1.5,
                "from_weighted_term": 0.2,
                "to_weighted_term": 1.7,
                "target_value": 10.0,
                "from_estimate": 1.0,
                "to_estimate": 0.0,
                "from_rel_error": 0.3,
                "to_rel_error": 1.0,
                "from_abs_pct_error": 90.0,
                "to_abs_pct_error": 100.0,
            }
        ],
        "top_regressions": [
            {
                "target_name": "nation/irs/example",
                "target_family": "national_irs_other",
                "target_scope": "national",
                "winner": "from",
                "weighted_term_delta": 1.5,
                "from_weighted_term": 0.2,
                "to_weighted_term": 1.7,
                "target_value": 10.0,
                "from_estimate": 1.0,
                "to_estimate": 0.0,
                "from_rel_error": 0.3,
                "to_rel_error": 1.0,
            }
        ],
        "top_improvements": [
            {
                "target_name": "state/example",
                "target_family": "other",
                "target_scope": "state",
                "winner": "to",
                "weighted_term_delta": -0.5,
                "from_weighted_term": 0.8,
                "to_weighted_term": 0.3,
                "target_value": 12.0,
                "from_estimate": 4.0,
                "to_estimate": 8.0,
                "from_rel_error": 0.7,
                "to_rel_error": 0.2,
            }
        ],
    }

    monkeypatch.setattr(
        "microplex_us.pipelines.pe_native_scores.subprocess.run",
        lambda *_args, **_kwargs: SimpleNamespace(
            returncode=0,
            stdout=json.dumps(payload),
            stderr="",
        ),
    )
    monkeypatch.setattr(
        "microplex_us.pipelines.pe_native_scores.resolve_policyengine_us_data_repo_root",
        lambda _repo=None: tmp_path,
    )

    result = compare_us_pe_native_target_deltas(
        from_dataset_path=before,
        to_dataset_path=after,
        period=2024,
        top_k=10,
        policyengine_us_data_repo=tmp_path,
        policyengine_us_data_python=tmp_path / "python",
    )

    assert result["metric"] == "enhanced_cps_native_loss_target_delta"
    assert result["top_regressions"][0]["target_name"] == "nation/irs/example"
    assert result["summary"]["to_win_rate"] == 0.5
    assert result["targets"][0]["target_family"] == "national_irs_other"


def test_parse_pe_native_target_lookup_key_maps_eitc_agi_child_labels() -> None:
    amount_key = parse_pe_native_target_lookup_key("nation/irs/eitc/amount/c3_1_1k")
    returns_key = parse_pe_native_target_lookup_key("nation/irs/eitc/returns/c2_1_1k")

    assert amount_key is not None
    assert amount_key.variable == "eitc"
    assert amount_key.count_children == 3
    assert amount_key.agi_lower == 1.0
    assert amount_key.agi_upper == 1_000.0
    assert amount_key.expected_target()["domain_variable"] == (
        "adjusted_gross_income,eitc,eitc_child_count"
    )
    assert returns_key is not None
    assert returns_key.variable == "tax_unit_count"
    assert returns_key.count_children == 2


def test_annotate_pe_native_target_db_matches_marks_matches_and_gaps(
    monkeypatch,
    tmp_path,
) -> None:
    db_path = tmp_path / "policy_data.db"
    db_path.write_text("stub")
    matched_name = "nation/irs/eitc/amount/c3_1_1k"
    matched_key = parse_pe_native_target_lookup_key(matched_name)
    assert matched_key is not None

    monkeypatch.setattr(
        "microplex_us.pipelines.pe_native_scores._load_policyengine_target_match_index",
        lambda *_args, **_kwargs: {
            matched_key.as_tuple(): [
                {
                    "target_id": 123,
                    "variable": "eitc",
                    "period": 2022,
                    "value": 2_079_000.0,
                    "source": "IRS SOI",
                    "notes": "Table 2.5",
                    "geo_level": "national",
                    "geographic_id": "US",
                    "domain_variable": ("adjusted_gross_income,eitc,eitc_child_count"),
                    "constraints": [
                        {
                            "variable": "eitc_child_count",
                            "operation": ">",
                            "value": "2",
                        }
                    ],
                }
            ]
        },
    )
    payload = {
        "targets": [
            {"target_name": matched_name},
            {"target_name": "nation/irs/eitc/returns/c2_1_1k"},
            {"target_name": "nation/census/infants"},
        ],
        "top_improvements": [{"target_name": matched_name}],
        "top_regressions": [],
    }

    annotate_pe_native_target_db_matches(
        payload,
        target_db_path=db_path,
        period=2024,
    )

    assert payload["targets"][0]["policyengine_target_match"] == "matched"
    assert payload["targets"][0]["policyengine_target_id"] == 123
    assert payload["targets"][1]["policyengine_target_match"] == "legacy_only"
    assert payload["targets"][1]["policyengine_target_expected"]["variable"] == (
        "tax_unit_count"
    )
    assert payload["targets"][2]["policyengine_target_match"] == "unparsed"
    assert payload["top_improvements"][0]["policyengine_target_match"] == "matched"
    assert payload["target_db_summary"]["matched"] == 1
    assert payload["target_db_summary"]["legacy_only"] == 1
    assert payload["target_db_summary"]["unparsed"] == 1
    assert payload["target_db_summary"]["match_rate"] == 0.5


def test_write_us_pe_native_target_diagnostics_persists_full_payload(
    monkeypatch,
    tmp_path,
) -> None:
    before = tmp_path / "before.h5"
    after = tmp_path / "after.h5"
    output_path = tmp_path / "diagnostics.json"
    for path in (before, after):
        path.write_text(path.stem)

    monkeypatch.setattr(
        "microplex_us.pipelines.pe_native_scores.compare_us_pe_native_target_deltas",
        lambda **_kwargs: {
            "metric": "enhanced_cps_native_loss_target_delta",
            "period": 2024,
            "from_dataset": str(before),
            "to_dataset": str(after),
            "summary": {"n_targets": 1, "to_win_rate": 1.0},
            "family_summaries": [{"target_family": "national_irs_other"}],
            "scope_summaries": [{"target_scope": "national"}],
            "targets": [
                {
                    "target_name": "nation/irs/example",
                    "target_family": "national_irs_other",
                    "winner": "to",
                    "weighted_term_delta": -1.0,
                }
            ],
            "top_regressions": [],
            "top_improvements": [],
        },
    )

    written = write_us_pe_native_target_diagnostics(
        output_path,
        from_dataset_path=before,
        to_dataset_path=after,
        from_label="baseline",
        to_label="candidate",
        policyengine_targets_db_path=tmp_path / "missing.db",
    )

    payload = json.loads(written.read_text())
    assert written == output_path
    assert payload["diagnostic_schema_version"] == 1
    assert payload["dataset_labels"] == {"from": "baseline", "to": "candidate"}
    assert payload["targets"][0]["target_name"] == "nation/irs/example"
    assert payload["targets"][0]["policyengine_target_match"] == "unparsed"
    assert payload["target_db_summary"]["unparsed"] == 1


def test_compute_batch_us_pe_native_target_deltas_wraps_multiple_candidates(
    monkeypatch,
    tmp_path,
) -> None:
    candidate_a = tmp_path / "candidate-a.h5"
    candidate_b = tmp_path / "candidate-b.h5"
    baseline = tmp_path / "baseline.h5"
    for path in (candidate_a, candidate_b, baseline):
        path.write_text(path.stem)

    payload = [
        {
            "metric": "enhanced_cps_native_loss_target_delta",
            "period": 2024,
            "from_dataset": str(baseline),
            "to_dataset": str(candidate_a),
            "top_regressions": [{"target_name": "nation/irs/example-a"}],
            "top_improvements": [{"target_name": "state/example-a"}],
        },
        {
            "metric": "enhanced_cps_native_loss_target_delta",
            "period": 2024,
            "from_dataset": str(baseline),
            "to_dataset": str(candidate_b),
            "top_regressions": [{"target_name": "nation/irs/example-b"}],
            "top_improvements": [{"target_name": "state/example-b"}],
        },
    ]

    monkeypatch.setattr(
        "microplex_us.pipelines.pe_native_scores.subprocess.run",
        lambda *_args, **_kwargs: SimpleNamespace(
            returncode=0,
            stdout=json.dumps(payload),
            stderr="",
        ),
    )
    monkeypatch.setattr(
        "microplex_us.pipelines.pe_native_scores.resolve_policyengine_us_data_repo_root",
        lambda _repo=None: tmp_path,
    )

    results = compute_batch_us_pe_native_target_deltas(
        candidate_dataset_paths=[candidate_a, candidate_b],
        baseline_dataset_path=baseline,
        period=2024,
        top_k=10,
        policyengine_us_data_repo=tmp_path,
        policyengine_us_data_python=tmp_path / "python",
    )

    assert len(results) == 2
    assert results[0]["to_dataset"] == str(candidate_a)
    assert results[1]["top_regressions"][0]["target_name"] == "nation/irs/example-b"


def test_compute_us_pe_native_support_audit_wraps_subprocess_payload(
    monkeypatch,
    tmp_path,
) -> None:
    candidate = tmp_path / "candidate.h5"
    baseline = tmp_path / "baseline.h5"
    for path in (candidate, baseline):
        path.write_text(path.stem)

    payload = {
        "metric": "enhanced_cps_support_audit",
        "period": 2024,
        "candidate_dataset": str(candidate),
        "baseline_dataset": str(baseline),
        "candidate": {
            "critical_input_support": {
                "child_support_expense": {
                    "stored": False,
                    "weighted_nonzero": 0.0,
                }
            }
        },
        "baseline": {
            "critical_input_support": {
                "child_support_expense": {
                    "stored": True,
                    "weighted_nonzero": 10.0,
                }
            }
        },
        "comparisons": {
            "critical_input_support": [
                {
                    "variable": "child_support_expense",
                    "candidate_stored": False,
                    "baseline_stored": True,
                    "weighted_nonzero_delta": -10.0,
                }
            ]
        },
    }

    monkeypatch.setattr(
        "microplex_us.pipelines.pe_native_scores.subprocess.run",
        lambda *_args, **_kwargs: SimpleNamespace(
            returncode=0,
            stdout=json.dumps(payload),
            stderr="",
        ),
    )
    monkeypatch.setattr(
        "microplex_us.pipelines.pe_native_scores.resolve_policyengine_us_data_repo_root",
        lambda _repo=None: tmp_path,
    )

    result = compute_us_pe_native_support_audit(
        candidate_dataset_path=candidate,
        baseline_dataset_path=baseline,
        period=2024,
        policyengine_us_data_repo=tmp_path,
        policyengine_us_data_python=tmp_path / "python",
    )

    assert result["metric"] == "enhanced_cps_support_audit"
    assert (
        result["comparisons"]["critical_input_support"][0]["variable"]
        == "child_support_expense"
    )


def test_compute_batch_us_pe_native_support_audits_wraps_multiple_candidates(
    monkeypatch,
    tmp_path,
) -> None:
    candidate_a = tmp_path / "candidate-a.h5"
    candidate_b = tmp_path / "candidate-b.h5"
    baseline = tmp_path / "baseline.h5"
    for path in (candidate_a, candidate_b, baseline):
        path.write_text(path.stem)

    payload = {
        "metric": "enhanced_cps_support_audit_batch",
        "period": 2024,
        "baseline_dataset": str(baseline),
        "baseline": {
            "critical_input_support": {
                "child_support_expense": {"stored": True, "weighted_nonzero": 10.0}
            }
        },
        "results": [
            {
                "candidate_dataset": str(candidate_a),
                "candidate": {
                    "critical_input_support": {
                        "child_support_expense": {
                            "stored": False,
                            "weighted_nonzero": 0.0,
                        }
                    }
                },
                "comparisons": {
                    "critical_input_support": [
                        {
                            "variable": "child_support_expense",
                            "candidate_stored": False,
                            "baseline_stored": True,
                            "weighted_nonzero_delta": -10.0,
                        }
                    ]
                },
            },
            {
                "candidate_dataset": str(candidate_b),
                "candidate": {
                    "critical_input_support": {
                        "child_support_expense": {
                            "stored": True,
                            "weighted_nonzero": 12.0,
                        }
                    }
                },
                "comparisons": {
                    "critical_input_support": [
                        {
                            "variable": "child_support_expense",
                            "candidate_stored": True,
                            "baseline_stored": True,
                            "weighted_nonzero_delta": 2.0,
                        }
                    ]
                },
            },
        ],
    }

    monkeypatch.setattr(
        "microplex_us.pipelines.pe_native_scores.subprocess.run",
        lambda *_args, **_kwargs: SimpleNamespace(
            returncode=0,
            stdout=json.dumps(payload),
            stderr="",
        ),
    )
    monkeypatch.setattr(
        "microplex_us.pipelines.pe_native_scores.resolve_policyengine_us_data_repo_root",
        lambda _repo=None: tmp_path,
    )

    results = compute_batch_us_pe_native_support_audits(
        candidate_dataset_paths=[candidate_a, candidate_b],
        baseline_dataset_path=baseline,
        period=2024,
        policyengine_us_data_repo=tmp_path,
        policyengine_us_data_python=tmp_path / "python",
    )

    assert len(results) == 2
    assert results[0]["baseline_dataset"] == str(baseline)
    assert results[0]["candidate_dataset"] == str(candidate_a)
    assert (
        results[1]["comparisons"]["critical_input_support"][0]["weighted_nonzero_delta"]
        == 2.0
    )
