"""PolicyEngine-native scoring helpers for US Microplex artifacts."""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from time import perf_counter
from typing import Any

_DEFAULT_PE_US_DATA_REPO = Path.home() / "PolicyEngine" / "policyengine-us-data"
_PE_US_DATA_PYTHON_ENV = "MICROPLEX_US_POLICYENGINE_US_DATA_PYTHON"
_PE_US_DATA_REPO_ENV = "MICROPLEX_US_POLICYENGINE_US_DATA_REPO"
_PE_NATIVE_SCORE_BASE_ENV_VARS: tuple[str, ...] = (
    "HOME",
    "PATH",
    "TMPDIR",
    "LANG",
    "LC_ALL",
    "TZ",
)
_EITC_AGI_CHILD_DOMAIN_VARIABLE = "adjusted_gross_income,eitc,eitc_child_count"
_EITC_AGI_CHILD_LABEL = re.compile(
    r"^nation/irs/eitc/(?P<metric>returns|amount)/"
    r"c(?P<count_children>\d+)_(?P<agi_lower>[^_]+)_(?P<agi_upper>[^/]+)$"
)

_ENHANCED_CPS_BAD_TARGETS: tuple[str, ...] = (
    "nation/irs/adjusted gross income/total/AGI in 10k-15k/taxable/Head of Household",
    "nation/irs/adjusted gross income/total/AGI in 15k-20k/taxable/Head of Household",
    "nation/irs/adjusted gross income/total/AGI in 10k-15k/taxable/Married Filing Jointly/Surviving Spouse",
    "nation/irs/adjusted gross income/total/AGI in 15k-20k/taxable/Married Filing Jointly/Surviving Spouse",
    "nation/irs/count/count/AGI in 10k-15k/taxable/Head of Household",
    "nation/irs/count/count/AGI in 15k-20k/taxable/Head of Household",
    "nation/irs/count/count/AGI in 10k-15k/taxable/Married Filing Jointly/Surviving Spouse",
    "nation/irs/count/count/AGI in 15k-20k/taxable/Married Filing Jointly/Surviving Spouse",
    "state/RI/adjusted_gross_income/amount/-inf_1",
    "nation/irs/exempt interest/count/AGI in -inf-inf/taxable/All",
)

_PE_NATIVE_BROAD_SCORE_SCRIPT = """
import json
import sys
from pathlib import Path

import numpy as np
from policyengine_core.data import Dataset

REPO_ROOT = sys.argv[1]
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from policyengine_us import Microsimulation
from policyengine_us_data.utils.loss import build_loss_matrix

BAD_TARGETS = tuple(json.loads(sys.argv[2]))
PERIOD = int(sys.argv[3])
CANDIDATE_DATASET = sys.argv[4]
BASELINE_DATASET = sys.argv[5]


def dataset_from_path(dataset_path: str, dataset_name: str):
    class LocalDataset(Dataset):
        name = dataset_name
        label = dataset_name
        file_path = dataset_path
        data_format = Dataset.TIME_PERIOD_ARRAYS
        time_period = PERIOD

    return LocalDataset


def classify_target_family(target_name: str) -> str:
    parts = target_name.split("/")
    if target_name.startswith("state/census/age/"):
        return "state_age_distribution"
    if target_name.startswith("state/census/population_by_state/"):
        return "state_population"
    if target_name.startswith("state/census/population_under_5_by_state/"):
        return "state_population_under_5"
    if target_name.startswith("nation/irs/aca_spending/"):
        return "state_aca_spending"
    if target_name.startswith("state/irs/aca_enrollment/"):
        return "state_aca_enrollment"
    if target_name.startswith("irs/medicaid_enrollment/"):
        return "state_medicaid_enrollment"
    if target_name.endswith("/snap-cost"):
        return "state_snap_cost"
    if target_name.endswith("/snap-hhs"):
        return "state_snap_households"
    if target_name.startswith("state/real_estate_taxes/"):
        return "state_real_estate_taxes"
    if len(parts) >= 3 and parts[0] == "state" and parts[2] == "adjusted_gross_income":
        return "state_agi_distribution"
    if target_name.startswith("nation/jct/"):
        return "national_tax_expenditures"
    if target_name.startswith("nation/net_worth/"):
        return "national_net_worth"
    if target_name.startswith("nation/ssa/"):
        return "national_ssa"
    if target_name.startswith("nation/census/population_by_age/"):
        return "national_population_by_age"
    if target_name == "nation/census/infants":
        return "national_infants"
    if target_name.startswith("nation/census/agi_in_spm_threshold_decile_"):
        return "national_spm_threshold_agi"
    if target_name.startswith("nation/census/count_in_spm_threshold_decile_"):
        return "national_spm_threshold_count"
    if target_name.startswith("nation/census/"):
        return "national_census_other"
    if target_name.startswith("nation/irs/"):
        return "national_irs_other"
    return "other"


def build_family_breakdown(target_names, candidate_terms, baseline_terms, candidate_rel_error, baseline_rel_error):
    family_rows = []
    target_names = list(target_names)
    unique_families = sorted({classify_target_family(name) for name in target_names})
    n_targets_total = float(len(target_names))
    for family in unique_families:
        idx = [i for i, name in enumerate(target_names) if classify_target_family(name) == family]
        if not idx:
            continue
        candidate_slice = candidate_terms[idx]
        baseline_slice = baseline_terms[idx]
        candidate_rel_slice = candidate_rel_error[idx]
        baseline_rel_slice = baseline_rel_error[idx]
        family_rows.append(
            {
                "family": family,
                "n_targets": int(len(idx)),
                "candidate_loss_contribution": float(candidate_slice.sum() / n_targets_total),
                "baseline_loss_contribution": float(baseline_slice.sum() / n_targets_total),
                "loss_contribution_delta": float((candidate_slice.sum() - baseline_slice.sum()) / n_targets_total),
                "candidate_mean_weighted_loss": float(candidate_slice.mean()),
                "baseline_mean_weighted_loss": float(baseline_slice.mean()),
                "candidate_mean_unweighted_msre": float(candidate_rel_slice.mean()),
                "baseline_mean_unweighted_msre": float(baseline_rel_slice.mean()),
                "unweighted_msre_delta": float(candidate_rel_slice.mean() - baseline_rel_slice.mean()),
            }
        )
    family_rows.sort(key=lambda row: row["loss_contribution_delta"], reverse=True)
    return family_rows


def compute(dataset_path: str) -> dict[str, float | int]:
    dataset_cls = dataset_from_path(
        dataset_path,
        Path(dataset_path).stem.replace("-", "_"),
    )
    loss_matrix, targets_array = build_loss_matrix(dataset_cls, PERIOD)
    target_names = np.asarray(loss_matrix.columns)
    zero_mask = np.isclose(targets_array, 0.0, atol=0.1)
    bad_mask = np.isin(target_names, BAD_TARGETS)
    keep_mask = ~(zero_mask | bad_mask)

    filtered = loss_matrix.loc[:, keep_mask]
    filtered_targets = np.asarray(targets_array[keep_mask], dtype=np.float64)
    is_national = np.asarray(filtered.columns.str.startswith("nation/"), dtype=bool)
    n_national = int(is_national.sum())
    n_state = int((~is_national).sum())
    if n_national == 0 or n_state == 0:
        raise ValueError(
            "PE-native broad loss requires both national and state targets after filtering"
        )

    normalisation_factor = np.where(
        is_national,
        1.0 / n_national,
        1.0 / n_state,
    ).astype(np.float64)
    inv_mean_normalisation = 1.0 / float(np.mean(normalisation_factor))

    sim = Microsimulation(dataset=dataset_cls)
    sim.default_calculation_period = PERIOD
    weights = sim.calculate(
        "household_weight",
        map_to="household",
        period=PERIOD,
    ).values.astype(np.float64)

    estimate = weights @ filtered.to_numpy(dtype=np.float64)
    rel_error = (((estimate - filtered_targets) + 1.0) / (filtered_targets + 1.0)) ** 2
    weighted_terms = inv_mean_normalisation * rel_error * normalisation_factor
    loss_value = float(weighted_terms.mean())
    unweighted_msre = float(rel_error.mean())

    return {
        "loss": loss_value,
        "unweighted_msre": unweighted_msre,
        "n_targets_total": int(len(target_names)),
        "n_targets_kept": int(keep_mask.sum()),
        "n_targets_zero_dropped": int(zero_mask.sum()),
        "n_targets_bad_dropped": int(bad_mask.sum()),
        "n_national_targets": n_national,
        "n_state_targets": n_state,
        "weight_sum": float(weights.sum()),
        "target_names": filtered.columns.tolist(),
        "weighted_terms": weighted_terms.tolist(),
        "rel_error": rel_error.tolist(),
    }


candidate = compute(CANDIDATE_DATASET)
baseline = compute(BASELINE_DATASET)

if candidate["n_targets_kept"] != baseline["n_targets_kept"]:
    raise ValueError(
        "Candidate and baseline produced different target counts after filtering: "
        f"{candidate['n_targets_kept']} vs {baseline['n_targets_kept']}"
    )
if candidate["target_names"] != baseline["target_names"]:
    raise ValueError("Candidate and baseline produced different target names after filtering")

payload = {
    "metric": "enhanced_cps_native_loss",
    "period": PERIOD,
    "candidate_dataset": CANDIDATE_DATASET,
    "baseline_dataset": BASELINE_DATASET,
    "candidate_enhanced_cps_native_loss": candidate["loss"],
    "baseline_enhanced_cps_native_loss": baseline["loss"],
    "enhanced_cps_native_loss_delta": candidate["loss"] - baseline["loss"],
    "candidate_unweighted_msre": candidate["unweighted_msre"],
    "baseline_unweighted_msre": baseline["unweighted_msre"],
    "unweighted_msre_delta": (
        candidate["unweighted_msre"] - baseline["unweighted_msre"]
    ),
    "n_targets_total": candidate["n_targets_total"],
    "n_targets_kept": candidate["n_targets_kept"],
    "n_targets_zero_dropped": candidate["n_targets_zero_dropped"],
    "n_targets_bad_dropped": candidate["n_targets_bad_dropped"],
    "n_national_targets": candidate["n_national_targets"],
    "n_state_targets": candidate["n_state_targets"],
    "candidate_weight_sum": candidate["weight_sum"],
    "baseline_weight_sum": baseline["weight_sum"],
    "family_breakdown": build_family_breakdown(
        candidate["target_names"],
        np.asarray(candidate["weighted_terms"], dtype=np.float64),
        np.asarray(baseline["weighted_terms"], dtype=np.float64),
        np.asarray(candidate["rel_error"], dtype=np.float64),
        np.asarray(baseline["rel_error"], dtype=np.float64),
    ),
}
print(json.dumps(payload, sort_keys=True))
""".strip()

_PE_NATIVE_BROAD_BATCH_SCORE_SCRIPT = """
import json
import sys
from pathlib import Path

import numpy as np
from policyengine_core.data import Dataset

REPO_ROOT = sys.argv[1]
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from policyengine_us import Microsimulation
from policyengine_us_data.utils.loss import build_loss_matrix

BAD_TARGETS = tuple(json.loads(sys.argv[2]))
PERIOD = int(sys.argv[3])
BASELINE_DATASET = sys.argv[4]
CANDIDATE_DATASETS = tuple(json.loads(sys.argv[5]))


def dataset_from_path(dataset_path: str, dataset_name: str):
    class LocalDataset(Dataset):
        name = dataset_name
        label = dataset_name
        file_path = dataset_path
        data_format = Dataset.TIME_PERIOD_ARRAYS
        time_period = PERIOD

    return LocalDataset


def classify_target_family(target_name: str) -> str:
    parts = target_name.split("/")
    if target_name.startswith("state/census/age/"):
        return "state_age_distribution"
    if target_name.startswith("state/census/population_by_state/"):
        return "state_population"
    if target_name.startswith("state/census/population_under_5_by_state/"):
        return "state_population_under_5"
    if target_name.startswith("nation/irs/aca_spending/"):
        return "state_aca_spending"
    if target_name.startswith("state/irs/aca_enrollment/"):
        return "state_aca_enrollment"
    if target_name.startswith("irs/medicaid_enrollment/"):
        return "state_medicaid_enrollment"
    if target_name.endswith("/snap-cost"):
        return "state_snap_cost"
    if target_name.endswith("/snap-hhs"):
        return "state_snap_households"
    if target_name.startswith("state/real_estate_taxes/"):
        return "state_real_estate_taxes"
    if len(parts) >= 3 and parts[0] == "state" and parts[2] == "adjusted_gross_income":
        return "state_agi_distribution"
    if target_name.startswith("nation/jct/"):
        return "national_tax_expenditures"
    if target_name.startswith("nation/net_worth/"):
        return "national_net_worth"
    if target_name.startswith("nation/ssa/"):
        return "national_ssa"
    if target_name.startswith("nation/census/population_by_age/"):
        return "national_population_by_age"
    if target_name == "nation/census/infants":
        return "national_infants"
    if target_name.startswith("nation/census/agi_in_spm_threshold_decile_"):
        return "national_spm_threshold_agi"
    if target_name.startswith("nation/census/count_in_spm_threshold_decile_"):
        return "national_spm_threshold_count"
    if target_name.startswith("nation/census/"):
        return "national_census_other"
    if target_name.startswith("nation/irs/"):
        return "national_irs_other"
    return "other"


def build_family_breakdown(target_names, candidate_terms, baseline_terms, candidate_rel_error, baseline_rel_error):
    family_rows = []
    target_names = list(target_names)
    unique_families = sorted({classify_target_family(name) for name in target_names})
    n_targets_total = float(len(target_names))
    for family in unique_families:
        idx = [i for i, name in enumerate(target_names) if classify_target_family(name) == family]
        if not idx:
            continue
        candidate_slice = candidate_terms[idx]
        baseline_slice = baseline_terms[idx]
        candidate_rel_slice = candidate_rel_error[idx]
        baseline_rel_slice = baseline_rel_error[idx]
        family_rows.append(
            {
                "family": family,
                "n_targets": int(len(idx)),
                "candidate_loss_contribution": float(candidate_slice.sum() / n_targets_total),
                "baseline_loss_contribution": float(baseline_slice.sum() / n_targets_total),
                "loss_contribution_delta": float((candidate_slice.sum() - baseline_slice.sum()) / n_targets_total),
                "candidate_mean_weighted_loss": float(candidate_slice.mean()),
                "baseline_mean_weighted_loss": float(baseline_slice.mean()),
                "candidate_mean_unweighted_msre": float(candidate_rel_slice.mean()),
                "baseline_mean_unweighted_msre": float(baseline_rel_slice.mean()),
                "unweighted_msre_delta": float(candidate_rel_slice.mean() - baseline_rel_slice.mean()),
            }
        )
    family_rows.sort(key=lambda row: row["loss_contribution_delta"], reverse=True)
    return family_rows


def compute(dataset_path: str) -> dict[str, float | int]:
    dataset_cls = dataset_from_path(
        dataset_path,
        Path(dataset_path).stem.replace("-", "_"),
    )
    loss_matrix, targets_array = build_loss_matrix(dataset_cls, PERIOD)
    target_names = np.asarray(loss_matrix.columns)
    zero_mask = np.isclose(targets_array, 0.0, atol=0.1)
    bad_mask = np.isin(target_names, BAD_TARGETS)
    keep_mask = ~(zero_mask | bad_mask)

    filtered = loss_matrix.loc[:, keep_mask]
    filtered_targets = np.asarray(targets_array[keep_mask], dtype=np.float64)
    is_national = np.asarray(filtered.columns.str.startswith("nation/"), dtype=bool)
    n_national = int(is_national.sum())
    n_state = int((~is_national).sum())
    if n_national == 0 or n_state == 0:
        raise ValueError(
            "PE-native broad loss requires both national and state targets after filtering"
        )

    normalisation_factor = np.where(
        is_national,
        1.0 / n_national,
        1.0 / n_state,
    ).astype(np.float64)
    inv_mean_normalisation = 1.0 / float(np.mean(normalisation_factor))

    sim = Microsimulation(dataset=dataset_cls)
    sim.default_calculation_period = PERIOD
    weights = sim.calculate(
        "household_weight",
        map_to="household",
        period=PERIOD,
    ).values.astype(np.float64)

    estimate = weights @ filtered.to_numpy(dtype=np.float64)
    rel_error = (((estimate - filtered_targets) + 1.0) / (filtered_targets + 1.0)) ** 2
    weighted_terms = inv_mean_normalisation * rel_error * normalisation_factor
    loss_value = float(weighted_terms.mean())
    unweighted_msre = float(rel_error.mean())

    return {
        "dataset": dataset_path,
        "loss": loss_value,
        "unweighted_msre": unweighted_msre,
        "n_targets_total": int(len(target_names)),
        "n_targets_kept": int(keep_mask.sum()),
        "n_targets_zero_dropped": int(zero_mask.sum()),
        "n_targets_bad_dropped": int(bad_mask.sum()),
        "n_national_targets": n_national,
        "n_state_targets": n_state,
        "weight_sum": float(weights.sum()),
        "target_names": filtered.columns.tolist(),
        "weighted_terms": weighted_terms.tolist(),
        "rel_error": rel_error.tolist(),
    }


baseline = compute(BASELINE_DATASET)
payload = []
for candidate_dataset in CANDIDATE_DATASETS:
    candidate = compute(candidate_dataset)
    if candidate["n_targets_kept"] != baseline["n_targets_kept"]:
        raise ValueError(
            "Candidate and baseline produced different target counts after filtering: "
            f"{candidate['n_targets_kept']} vs {baseline['n_targets_kept']}"
        )
    if candidate["target_names"] != baseline["target_names"]:
        raise ValueError("Candidate and baseline produced different target names after filtering")
    payload.append(
        {
            "metric": "enhanced_cps_native_loss",
            "period": PERIOD,
            "candidate_dataset": candidate_dataset,
            "baseline_dataset": BASELINE_DATASET,
            "candidate_enhanced_cps_native_loss": candidate["loss"],
            "baseline_enhanced_cps_native_loss": baseline["loss"],
            "enhanced_cps_native_loss_delta": candidate["loss"] - baseline["loss"],
            "candidate_beats_baseline": candidate["loss"] < baseline["loss"],
            "candidate_unweighted_msre": candidate["unweighted_msre"],
            "baseline_unweighted_msre": baseline["unweighted_msre"],
            "unweighted_msre_delta": (
                candidate["unweighted_msre"] - baseline["unweighted_msre"]
            ),
            "n_targets_total": candidate["n_targets_total"],
            "n_targets_kept": candidate["n_targets_kept"],
            "n_targets_zero_dropped": candidate["n_targets_zero_dropped"],
            "n_targets_bad_dropped": candidate["n_targets_bad_dropped"],
            "n_national_targets": candidate["n_national_targets"],
            "n_state_targets": candidate["n_state_targets"],
            "candidate_weight_sum": candidate["weight_sum"],
            "baseline_weight_sum": baseline["weight_sum"],
            "family_breakdown": build_family_breakdown(
                candidate["target_names"],
                np.asarray(candidate["weighted_terms"], dtype=np.float64),
                np.asarray(baseline["weighted_terms"], dtype=np.float64),
                np.asarray(candidate["rel_error"], dtype=np.float64),
                np.asarray(baseline["rel_error"], dtype=np.float64),
            ),
        }
    )
print(json.dumps(payload, sort_keys=True))
""".strip()

_PE_NATIVE_TARGET_DELTA_SCRIPT = """
import json
import sys
from pathlib import Path

import numpy as np
from policyengine_core.data import Dataset

REPO_ROOT = sys.argv[1]
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from policyengine_us import Microsimulation
from policyengine_us_data.utils.loss import build_loss_matrix

BAD_TARGETS = tuple(json.loads(sys.argv[2]))
PERIOD = int(sys.argv[3])
FROM_DATASET = sys.argv[4]
TO_DATASET = sys.argv[5]
TOP_K = int(sys.argv[6])


def dataset_from_path(dataset_path: str, dataset_name: str):
    class LocalDataset(Dataset):
        name = dataset_name
        label = dataset_name
        file_path = dataset_path
        data_format = Dataset.TIME_PERIOD_ARRAYS
        time_period = PERIOD

    return LocalDataset


def compute(dataset_path: str):
    dataset_cls = dataset_from_path(
        dataset_path,
        Path(dataset_path).stem.replace("-", "_"),
    )
    loss_matrix, targets_array = build_loss_matrix(dataset_cls, PERIOD)
    target_names = np.asarray(loss_matrix.columns)
    zero_mask = np.isclose(targets_array, 0.0, atol=0.1)
    bad_mask = np.isin(target_names, BAD_TARGETS)
    keep_mask = ~(zero_mask | bad_mask)

    filtered = loss_matrix.loc[:, keep_mask]
    filtered_targets = np.asarray(targets_array[keep_mask], dtype=np.float64)
    is_national = np.asarray(filtered.columns.str.startswith("nation/"), dtype=bool)
    n_national = int(is_national.sum())
    n_state = int((~is_national).sum())
    if n_national == 0 or n_state == 0:
        raise ValueError(
            "PE-native broad loss requires both national and state targets after filtering"
        )

    normalisation_factor = np.where(
        is_national,
        1.0 / n_national,
        1.0 / n_state,
    ).astype(np.float64)
    inv_mean_normalisation = 1.0 / float(np.mean(normalisation_factor))

    sim = Microsimulation(dataset=dataset_cls)
    sim.default_calculation_period = PERIOD
    weights = sim.calculate(
        "household_weight",
        map_to="household",
        period=PERIOD,
    ).values.astype(np.float64)

    estimate = weights @ filtered.to_numpy(dtype=np.float64)
    rel_error = (((estimate - filtered_targets) + 1.0) / (filtered_targets + 1.0)) ** 2
    weighted_terms = inv_mean_normalisation * rel_error * normalisation_factor
    return {
        "target_names": filtered.columns.tolist(),
        "targets": filtered_targets.tolist(),
        "estimate": estimate.tolist(),
        "rel_error": rel_error.tolist(),
        "weighted_terms": weighted_terms.tolist(),
    }


def classify_target_family(target_name: str) -> str:
    parts = target_name.split("/")
    if target_name.startswith("state/census/age/"):
        return "state_age_distribution"
    if target_name.startswith("state/census/population_by_state/"):
        return "state_population"
    if target_name.startswith("state/census/population_under_5_by_state/"):
        return "state_population_under_5"
    if target_name.startswith("nation/irs/aca_spending/"):
        return "state_aca_spending"
    if target_name.startswith("state/irs/aca_enrollment/"):
        return "state_aca_enrollment"
    if target_name.startswith("irs/medicaid_enrollment/"):
        return "state_medicaid_enrollment"
    if target_name.endswith("/snap-cost"):
        return "state_snap_cost"
    if target_name.endswith("/snap-hhs"):
        return "state_snap_households"
    if target_name.startswith("state/real_estate_taxes/"):
        return "state_real_estate_taxes"
    if len(parts) >= 3 and parts[0] == "state" and parts[2] == "adjusted_gross_income":
        return "state_agi_distribution"
    if target_name.startswith("nation/jct/"):
        return "national_tax_expenditures"
    if target_name.startswith("nation/net_worth/"):
        return "national_net_worth"
    if target_name.startswith("nation/ssa/"):
        return "national_ssa"
    if target_name.startswith("nation/census/population_by_age/"):
        return "national_population_by_age"
    if target_name == "nation/census/infants":
        return "national_infants"
    if target_name.startswith("nation/census/agi_in_spm_threshold_decile_"):
        return "national_spm_threshold_agi"
    if target_name.startswith("nation/census/count_in_spm_threshold_decile_"):
        return "national_spm_threshold_count"
    if target_name.startswith("nation/census/"):
        return "national_census_other"
    if target_name.startswith("nation/irs/"):
        return "national_irs_other"
    return "other"


def target_scope(target_name: str) -> str:
    if target_name.startswith("nation/"):
        return "national"
    if target_name.startswith("state/") or target_name.endswith("/snap-cost") or target_name.endswith("/snap-hhs"):
        return "state"
    return "other"


def abs_pct_error(estimate: float, target: float) -> float:
    return abs(estimate - target) / max(abs(target), 1.0) * 100.0


def build_target_rows(from_payload, to_payload):
    rows = []
    for idx, name in enumerate(from_payload["target_names"]):
        from_term = float(from_payload["weighted_terms"][idx])
        to_term = float(to_payload["weighted_terms"][idx])
        from_error = float(from_payload["rel_error"][idx])
        to_error = float(to_payload["rel_error"][idx])
        target_value = float(from_payload["targets"][idx])
        from_estimate = float(from_payload["estimate"][idx])
        to_estimate = float(to_payload["estimate"][idx])
        if to_error < from_error:
            winner = "to"
        elif from_error < to_error:
            winner = "from"
        else:
            winner = "tie"
        rows.append(
            {
                "target_name": name,
                "target_family": classify_target_family(name),
                "target_scope": target_scope(name),
                "winner": winner,
                "weighted_term_delta": to_term - from_term,
                "from_weighted_term": from_term,
                "to_weighted_term": to_term,
                "target_value": target_value,
                "from_estimate": from_estimate,
                "to_estimate": to_estimate,
                "from_rel_error": from_error,
                "to_rel_error": to_error,
                "from_abs_pct_error": abs_pct_error(from_estimate, target_value),
                "to_abs_pct_error": abs_pct_error(to_estimate, target_value),
            }
        )
    return rows


def summarize_target_rows(rows, *, group_field=None):
    if group_field is None:
        grouped = [("all", rows)]
    else:
        values = sorted({row[group_field] for row in rows})
        grouped = [(value, [row for row in rows if row[group_field] == value]) for value in values]

    summaries = []
    for value, group_rows in grouped:
        n_targets = len(group_rows)
        from_wins = sum(1 for row in group_rows if row["winner"] == "from")
        to_wins = sum(1 for row in group_rows if row["winner"] == "to")
        ties = n_targets - from_wins - to_wins
        from_loss = float(np.mean([row["from_weighted_term"] for row in group_rows]))
        to_loss = float(np.mean([row["to_weighted_term"] for row in group_rows]))
        summary = {
            "n_targets": n_targets,
            "from_wins": from_wins,
            "to_wins": to_wins,
            "ties": ties,
            "from_win_rate": from_wins / n_targets if n_targets else None,
            "to_win_rate": to_wins / n_targets if n_targets else None,
            "from_loss": from_loss,
            "to_loss": to_loss,
            "loss_delta": to_loss - from_loss,
            "mean_weighted_term_delta": float(
                np.mean([row["weighted_term_delta"] for row in group_rows])
            ),
        }
        if group_field is not None:
            summary[group_field] = value
        summaries.append(summary)
    return summaries[0] if group_field is None else summaries


from_payload = compute(FROM_DATASET)
to_payload = compute(TO_DATASET)

if from_payload["target_names"] != to_payload["target_names"]:
    raise ValueError("Datasets produced different target names after filtering")

rows = build_target_rows(from_payload, to_payload)
rows.sort(key=lambda row: row["weighted_term_delta"], reverse=True)
payload = {
    "metric": "enhanced_cps_native_loss_target_delta",
    "period": PERIOD,
    "from_dataset": FROM_DATASET,
    "to_dataset": TO_DATASET,
    "summary": summarize_target_rows(rows),
    "family_summaries": summarize_target_rows(rows, group_field="target_family"),
    "scope_summaries": summarize_target_rows(rows, group_field="target_scope"),
    "targets": rows,
    "top_regressions": rows[:TOP_K],
    "top_improvements": list(reversed(rows[-TOP_K:])),
}
print(json.dumps(payload, sort_keys=True))
""".strip()

_PE_NATIVE_TARGET_DELTA_BATCH_SCRIPT = """
import json
import sys
from pathlib import Path

import numpy as np
from policyengine_core.data import Dataset

REPO_ROOT = sys.argv[1]
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from policyengine_us import Microsimulation
from policyengine_us_data.utils.loss import build_loss_matrix

BAD_TARGETS = tuple(json.loads(sys.argv[2]))
PERIOD = int(sys.argv[3])
BASELINE_DATASET = sys.argv[4]
CANDIDATE_DATASETS = json.loads(sys.argv[5])
TOP_K = int(sys.argv[6])


def dataset_from_path(dataset_path: str, dataset_name: str):
    class LocalDataset(Dataset):
        name = dataset_name
        label = dataset_name
        file_path = dataset_path
        data_format = Dataset.TIME_PERIOD_ARRAYS
        time_period = PERIOD

    return LocalDataset


def compute(dataset_path: str):
    dataset_cls = dataset_from_path(
        dataset_path,
        Path(dataset_path).stem.replace("-", "_"),
    )
    loss_matrix, targets_array = build_loss_matrix(dataset_cls, PERIOD)
    target_names = np.asarray(loss_matrix.columns)
    zero_mask = np.isclose(targets_array, 0.0, atol=0.1)
    bad_mask = np.isin(target_names, BAD_TARGETS)
    keep_mask = ~(zero_mask | bad_mask)

    filtered = loss_matrix.loc[:, keep_mask]
    filtered_targets = np.asarray(targets_array[keep_mask], dtype=np.float64)
    is_national = np.asarray(filtered.columns.str.startswith("nation/"), dtype=bool)
    n_national = int(is_national.sum())
    n_state = int((~is_national).sum())
    if n_national == 0 or n_state == 0:
        raise ValueError(
            "PE-native broad loss requires both national and state targets after filtering"
        )

    normalisation_factor = np.where(
        is_national,
        1.0 / n_national,
        1.0 / n_state,
    ).astype(np.float64)
    inv_mean_normalisation = 1.0 / float(np.mean(normalisation_factor))

    sim = Microsimulation(dataset=dataset_cls)
    sim.default_calculation_period = PERIOD
    weights = sim.calculate(
        "household_weight",
        map_to="household",
        period=PERIOD,
    ).values.astype(np.float64)

    estimate = weights @ filtered.to_numpy(dtype=np.float64)
    rel_error = (((estimate - filtered_targets) + 1.0) / (filtered_targets + 1.0)) ** 2
    weighted_terms = inv_mean_normalisation * rel_error * normalisation_factor
    return {
        "target_names": filtered.columns.tolist(),
        "targets": filtered_targets.tolist(),
        "estimate": estimate.tolist(),
        "rel_error": rel_error.tolist(),
        "weighted_terms": weighted_terms.tolist(),
    }


def classify_target_family(target_name: str) -> str:
    parts = target_name.split("/")
    if target_name.startswith("state/census/age/"):
        return "state_age_distribution"
    if target_name.startswith("state/census/population_by_state/"):
        return "state_population"
    if target_name.startswith("state/census/population_under_5_by_state/"):
        return "state_population_under_5"
    if target_name.startswith("nation/irs/aca_spending/"):
        return "state_aca_spending"
    if target_name.startswith("state/irs/aca_enrollment/"):
        return "state_aca_enrollment"
    if target_name.startswith("irs/medicaid_enrollment/"):
        return "state_medicaid_enrollment"
    if target_name.endswith("/snap-cost"):
        return "state_snap_cost"
    if target_name.endswith("/snap-hhs"):
        return "state_snap_households"
    if target_name.startswith("state/real_estate_taxes/"):
        return "state_real_estate_taxes"
    if len(parts) >= 3 and parts[0] == "state" and parts[2] == "adjusted_gross_income":
        return "state_agi_distribution"
    if target_name.startswith("nation/jct/"):
        return "national_tax_expenditures"
    if target_name.startswith("nation/net_worth/"):
        return "national_net_worth"
    if target_name.startswith("nation/ssa/"):
        return "national_ssa"
    if target_name.startswith("nation/census/population_by_age/"):
        return "national_population_by_age"
    if target_name == "nation/census/infants":
        return "national_infants"
    if target_name.startswith("nation/census/agi_in_spm_threshold_decile_"):
        return "national_spm_threshold_agi"
    if target_name.startswith("nation/census/count_in_spm_threshold_decile_"):
        return "national_spm_threshold_count"
    if target_name.startswith("nation/census/"):
        return "national_census_other"
    if target_name.startswith("nation/irs/"):
        return "national_irs_other"
    return "other"


def target_scope(target_name: str) -> str:
    if target_name.startswith("nation/"):
        return "national"
    if target_name.startswith("state/") or target_name.endswith("/snap-cost") or target_name.endswith("/snap-hhs"):
        return "state"
    return "other"


def abs_pct_error(estimate: float, target: float) -> float:
    return abs(estimate - target) / max(abs(target), 1.0) * 100.0


def build_target_rows(from_payload, to_payload):
    rows = []
    for idx, name in enumerate(from_payload["target_names"]):
        from_term = float(from_payload["weighted_terms"][idx])
        to_term = float(to_payload["weighted_terms"][idx])
        from_error = float(from_payload["rel_error"][idx])
        to_error = float(to_payload["rel_error"][idx])
        target_value = float(from_payload["targets"][idx])
        from_estimate = float(from_payload["estimate"][idx])
        to_estimate = float(to_payload["estimate"][idx])
        if to_error < from_error:
            winner = "to"
        elif from_error < to_error:
            winner = "from"
        else:
            winner = "tie"
        rows.append(
            {
                "target_name": name,
                "target_family": classify_target_family(name),
                "target_scope": target_scope(name),
                "winner": winner,
                "weighted_term_delta": to_term - from_term,
                "from_weighted_term": from_term,
                "to_weighted_term": to_term,
                "target_value": target_value,
                "from_estimate": from_estimate,
                "to_estimate": to_estimate,
                "from_rel_error": from_error,
                "to_rel_error": to_error,
                "from_abs_pct_error": abs_pct_error(from_estimate, target_value),
                "to_abs_pct_error": abs_pct_error(to_estimate, target_value),
            }
        )
    return rows


def summarize_target_rows(rows, *, group_field=None):
    if group_field is None:
        grouped = [("all", rows)]
    else:
        values = sorted({row[group_field] for row in rows})
        grouped = [(value, [row for row in rows if row[group_field] == value]) for value in values]

    summaries = []
    for value, group_rows in grouped:
        n_targets = len(group_rows)
        from_wins = sum(1 for row in group_rows if row["winner"] == "from")
        to_wins = sum(1 for row in group_rows if row["winner"] == "to")
        ties = n_targets - from_wins - to_wins
        from_loss = float(np.mean([row["from_weighted_term"] for row in group_rows]))
        to_loss = float(np.mean([row["to_weighted_term"] for row in group_rows]))
        summary = {
            "n_targets": n_targets,
            "from_wins": from_wins,
            "to_wins": to_wins,
            "ties": ties,
            "from_win_rate": from_wins / n_targets if n_targets else None,
            "to_win_rate": to_wins / n_targets if n_targets else None,
            "from_loss": from_loss,
            "to_loss": to_loss,
            "loss_delta": to_loss - from_loss,
            "mean_weighted_term_delta": float(
                np.mean([row["weighted_term_delta"] for row in group_rows])
            ),
        }
        if group_field is not None:
            summary[group_field] = value
        summaries.append(summary)
    return summaries[0] if group_field is None else summaries


baseline_payload = compute(BASELINE_DATASET)
results = []
for candidate_dataset in CANDIDATE_DATASETS:
    candidate_payload = compute(candidate_dataset)
    if baseline_payload["target_names"] != candidate_payload["target_names"]:
        raise ValueError("Datasets produced different target names after filtering")

    rows = build_target_rows(baseline_payload, candidate_payload)
    rows.sort(key=lambda row: row["weighted_term_delta"], reverse=True)
    results.append(
        {
            "metric": "enhanced_cps_native_loss_target_delta",
            "period": PERIOD,
            "from_dataset": BASELINE_DATASET,
            "to_dataset": candidate_dataset,
            "summary": summarize_target_rows(rows),
            "family_summaries": summarize_target_rows(rows, group_field="target_family"),
            "scope_summaries": summarize_target_rows(rows, group_field="target_scope"),
            "targets": rows,
            "top_regressions": rows[:TOP_K],
            "top_improvements": list(reversed(rows[-TOP_K:])),
        }
    )

print(json.dumps(results, sort_keys=True))
""".strip()

_PE_NATIVE_SUPPORT_AUDIT_SCRIPT = """
import json
import sys
from pathlib import Path

import h5py
import numpy as np
from policyengine_core.data import Dataset

REPO_ROOT = sys.argv[1]
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from policyengine_us import Microsimulation

PERIOD = int(sys.argv[2])
CANDIDATE_DATASET = sys.argv[3]
BASELINE_DATASET = sys.argv[4]

STATE_FIPS_TO_ABBR = {
    1: "AL", 2: "AK", 4: "AZ", 5: "AR", 6: "CA", 8: "CO", 9: "CT", 10: "DE",
    11: "DC", 12: "FL", 13: "GA", 15: "HI", 16: "ID", 17: "IL", 18: "IN",
    19: "IA", 20: "KS", 21: "KY", 22: "LA", 23: "ME", 24: "MD", 25: "MA",
    26: "MI", 27: "MN", 28: "MS", 29: "MO", 30: "MT", 31: "NE", 32: "NV",
    33: "NH", 34: "NJ", 35: "NM", 36: "NY", 37: "NC", 38: "ND", 39: "OH",
    40: "OK", 41: "OR", 42: "PA", 44: "RI", 45: "SC", 46: "SD", 47: "TN",
    48: "TX", 49: "UT", 50: "VT", 51: "VA", 53: "WA", 54: "WV", 55: "WI",
    56: "WY",
}
CRITICAL_PERSON_VARIABLES = (
    "has_marketplace_health_coverage",
    "has_esi",
    "medicare_part_b_premiums",
    "child_support_expense",
    "self_employment_income_before_lsr",
    "rental_income",
    "non_sch_d_capital_gains",
)
HIGH_SIGNAL_MFS_AGI_BINS = (
    ("75k_to_100k", 75_000.0, 100_000.0),
    ("100k_to_200k", 100_000.0, 200_000.0),
    ("200k_to_500k", 200_000.0, 500_000.0),
    ("500k_plus", 500_000.0, np.inf),
)
HIGH_SIGNAL_HOH_AGI_BINS = (
    ("20k_to_25k", 20_000.0, 25_000.0),
    ("25k_to_30k", 25_000.0, 30_000.0),
    ("30k_to_40k", 30_000.0, 40_000.0),
    ("200k_to_500k", 200_000.0, 500_000.0),
    ("500k_to_1m", 500_000.0, 1_000_000.0),
    ("1m_plus", 1_000_000.0, np.inf),
)
AGE_BUCKETS = (
    ("0_to_4", 0, 5),
    ("5_to_17", 5, 18),
    ("18_to_29", 18, 30),
    ("30_to_44", 30, 45),
    ("45_to_64", 45, 65),
    ("65_plus", 65, np.inf),
)
SSI_AGE_BUCKETS = (
    ("all", -np.inf, np.inf),
    ("under_18", 0, 18),
    ("18_to_64", 18, 65),
    ("65_plus", 65, np.inf),
)
MEDICARE_PART_B_AGE_BUCKETS = (
    ("age_0_to_9", 0, 10),
    ("age_10_to_19", 10, 20),
    ("age_20_to_29", 20, 30),
    ("age_30_to_39", 30, 40),
    ("age_40_to_49", 40, 50),
    ("age_50_to_59", 50, 60),
    ("age_60_to_64", 60, 65),
    ("age_65_plus", 65, np.inf),
)


def dataset_from_path(dataset_path: str, dataset_name: str):
    class LocalDataset(Dataset):
        name = dataset_name
        label = dataset_name
        file_path = dataset_path
        data_format = Dataset.TIME_PERIOD_ARRAYS
        time_period = PERIOD

    return LocalDataset


def stored_variables_for(dataset_path: str) -> set[str]:
    with h5py.File(dataset_path, "r") as handle:
        return set(handle.keys())


def state_abbr(value) -> str:
    if value is None:
        return "NA"
    try:
        numeric = int(value)
    except (TypeError, ValueError):
        return str(value)
    return STATE_FIPS_TO_ABBR.get(numeric, str(numeric))


def normalize_status(value) -> str:
    if hasattr(value, "name"):
        return str(value.name)
    text = str(value)
    if "." in text:
        text = text.rsplit(".", 1)[-1]
    normalized = text.strip().upper().replace(" ", "_")
    if normalized in {
        "SINGLE",
        "JOINT",
        "SEPARATE",
        "HEAD_OF_HOUSEHOLD",
        "SURVIVING_SPOUSE",
    }:
        return normalized
    return normalized


def summarize_numeric(values, weights, *, stored: bool) -> dict[str, float | int | bool]:
    arr = np.nan_to_num(np.asarray(values, dtype=np.float64), nan=0.0)
    w = np.asarray(weights, dtype=np.float64)
    positive = arr > 0.0
    negative = arr < 0.0
    nonzero = arr != 0.0
    return {
        "stored": bool(stored),
        "nonzero_count": int(nonzero.sum()),
        "positive_count": int(positive.sum()),
        "negative_count": int(negative.sum()),
        "weighted_nonzero": float(w[nonzero].sum()),
        "weighted_positive": float(w[positive].sum()),
        "weighted_negative": float(w[negative].sum()),
        "value_sum": float((arr * w).sum()),
    }


def summarize_bool(values, weights, *, stored: bool) -> dict[str, float | int | bool]:
    arr = np.asarray(values).astype(bool)
    w = np.asarray(weights, dtype=np.float64)
    return {
        "stored": bool(stored),
        "true_count": int(arr.sum()),
        "false_count": int((~arr).sum()),
        "weighted_true": float(w[arr].sum()),
        "weighted_false": float(w[~arr].sum()),
    }


def build_snapshot(dataset_path: str) -> dict:
    dataset_cls = dataset_from_path(
        dataset_path,
        Path(dataset_path).stem.replace("-", "_"),
    )
    stored_variables = stored_variables_for(dataset_path)
    sim = Microsimulation(dataset=dataset_cls)
    sim.default_calculation_period = PERIOD

    person_weights = sim.calculate("person_weight", period=PERIOD).values.astype(np.float64)
    household_weights = sim.calculate("household_weight", period=PERIOD).values.astype(np.float64)
    tax_unit_weights = sim.calculate("tax_unit_weight", period=PERIOD).values.astype(np.float64)
    person_state = sim.calculate("state_fips", map_to="person", period=PERIOD).values
    household_state = sim.calculate("state_fips", map_to="household", period=PERIOD).values
    person_age = sim.calculate("age", period=PERIOD).values.astype(np.float64)
    marketplace = sim.calculate("has_marketplace_health_coverage", period=PERIOD).values
    filing_status = sim.calculate("filing_status", period=PERIOD).values
    adjusted_gross_income = sim.calculate("adjusted_gross_income", period=PERIOD).values.astype(np.float64)
    ssi = sim.calculate("ssi", period=PERIOD).values.astype(np.float64)
    medicare_part_b_premiums = sim.calculate("medicare_part_b_premiums", period=PERIOD).values.astype(np.float64)
    aca_ptc_household = sim.calculate("aca_ptc", map_to="household", period=PERIOD).values.astype(np.float64)

    critical_support = {}
    for variable in CRITICAL_PERSON_VARIABLES:
        values = sim.calculate(variable, period=PERIOD).values
        if np.asarray(values).dtype == np.bool_:
            critical_support[variable] = summarize_bool(
                values,
                person_weights,
                stored=variable in stored_variables,
            )
        else:
            critical_support[variable] = summarize_numeric(
                values,
                person_weights,
                stored=variable in stored_variables,
            )

    normalized_filing_status = np.asarray([normalize_status(value) for value in filing_status])
    filing_status_counts = {}
    for status in ("SINGLE", "JOINT", "SEPARATE", "HEAD_OF_HOUSEHOLD", "SURVIVING_SPOUSE"):
        mask = normalized_filing_status == status
        filing_status_counts[status] = {
            "count": int(mask.sum()),
            "weighted_count": float(tax_unit_weights[mask].sum()),
        }

    def agi_support_for_status(status: str, bins) -> list[dict]:
        status_mask = normalized_filing_status == status
        rows = []
        for label, lower, upper in bins:
            mask = status_mask & (adjusted_gross_income >= lower) & (adjusted_gross_income < upper)
            rows.append(
                {
                    "agi_bin": label,
                    "count": int(mask.sum()),
                    "weighted_count": float(tax_unit_weights[mask].sum()),
                    "weighted_agi": float((adjusted_gross_income[mask] * tax_unit_weights[mask]).sum()),
                }
            )
        return rows

    def person_value_by_age(values, buckets) -> list[dict]:
        arr = np.nan_to_num(np.asarray(values, dtype=np.float64), nan=0.0)
        rows = []
        for label, lower, upper in buckets:
            age_mask = (person_age >= lower) & (person_age < upper)
            positive = age_mask & (arr > 0.0)
            rows.append(
                {
                    "age_bucket": label,
                    "person_count": int(age_mask.sum()),
                    "positive_count": int(positive.sum()),
                    "weighted_people": float(person_weights[age_mask].sum()),
                    "weighted_positive": float(person_weights[positive].sum()),
                    "value_sum": float((arr[age_mask] * person_weights[age_mask]).sum()),
                }
            )
        return rows

    mfs_agi_support = agi_support_for_status("SEPARATE", HIGH_SIGNAL_MFS_AGI_BINS)
    hoh_agi_support = agi_support_for_status("HEAD_OF_HOUSEHOLD", HIGH_SIGNAL_HOH_AGI_BINS)
    ssi_by_age = person_value_by_age(ssi, SSI_AGE_BUCKETS)
    medicare_part_b_by_age = person_value_by_age(
        medicare_part_b_premiums,
        MEDICARE_PART_B_AGE_BUCKETS,
    )

    state_aca_ptc = {}
    for state in sorted({state_abbr(value) for value in household_state}):
        state_mask = np.asarray([state_abbr(value) == state for value in household_state], dtype=bool)
        positive = state_mask & (aca_ptc_household > 0.0)
        state_aca_ptc[state] = {
            "weighted_households": float(household_weights[state_mask].sum()),
            "weighted_positive_households": float(household_weights[positive].sum()),
            "weighted_aca_ptc": float((aca_ptc_household[state_mask] * household_weights[state_mask]).sum()),
        }

    states = sorted({state_abbr(value) for value in person_state})
    state_marketplace = {}
    state_age_bucket = {}
    marketplace_bool = np.asarray(marketplace).astype(bool)
    for state in states:
        state_mask = np.asarray([state_abbr(value) == state for value in person_state], dtype=bool)
        enrolled = state_mask & marketplace_bool
        state_marketplace[state] = {
            "weighted_people": float(person_weights[state_mask].sum()),
            "weighted_marketplace_enrollment": float(person_weights[enrolled].sum()),
        }
        bucket_weights = {}
        nonempty = 0
        for label, lower, upper in AGE_BUCKETS:
            mask = state_mask & (person_age >= lower) & (person_age < upper)
            weight = float(person_weights[mask].sum())
            bucket_weights[label] = weight
            if weight > 0.0:
                nonempty += 1
        state_age_bucket[state] = {
            "nonempty_buckets": int(nonempty),
            "bucket_weights": bucket_weights,
        }

    return {
        "dataset": dataset_path,
        "stored_variable_count": int(len(stored_variables)),
        "stored_variables": sorted(stored_variables),
        "critical_input_support": critical_support,
        "filing_status_weighted_counts": filing_status_counts,
        "mfs_high_agi_support": mfs_agi_support,
        "hoh_agi_support": hoh_agi_support,
        "ssi_by_age": ssi_by_age,
        "medicare_part_b_premiums_by_age": medicare_part_b_by_age,
        "state_aca_ptc_spending": state_aca_ptc,
        "state_marketplace_enrollment": state_marketplace,
        "state_age_bucket_support": state_age_bucket,
    }


def compare_snapshots(candidate: dict, baseline: dict) -> dict:
    critical_rows = []
    for variable in CRITICAL_PERSON_VARIABLES:
        candidate_row = candidate["critical_input_support"][variable]
        baseline_row = baseline["critical_input_support"][variable]
        candidate_weighted = candidate_row.get("weighted_nonzero", candidate_row.get("weighted_true", 0.0))
        baseline_weighted = baseline_row.get("weighted_nonzero", baseline_row.get("weighted_true", 0.0))
        critical_rows.append(
            {
                "variable": variable,
                "candidate_stored": bool(candidate_row.get("stored", False)),
                "baseline_stored": bool(baseline_row.get("stored", False)),
                "candidate_weighted_nonzero": float(candidate_weighted),
                "baseline_weighted_nonzero": float(baseline_weighted),
                "weighted_nonzero_delta": float(candidate_weighted - baseline_weighted),
            }
        )

    filing_status_rows = []
    for status in ("SINGLE", "JOINT", "SEPARATE", "HEAD_OF_HOUSEHOLD", "SURVIVING_SPOUSE"):
        candidate_row = candidate["filing_status_weighted_counts"][status]
        baseline_row = baseline["filing_status_weighted_counts"][status]
        filing_status_rows.append(
            {
                "filing_status": status,
                "candidate_weighted_count": float(candidate_row["weighted_count"]),
                "baseline_weighted_count": float(baseline_row["weighted_count"]),
                "weighted_count_delta": float(candidate_row["weighted_count"] - baseline_row["weighted_count"]),
            }
        )

    baseline_bins = {row["agi_bin"]: row for row in baseline["mfs_high_agi_support"]}
    mfs_rows = []
    for row in candidate["mfs_high_agi_support"]:
        other = baseline_bins[row["agi_bin"]]
        mfs_rows.append(
            {
                "agi_bin": row["agi_bin"],
                "candidate_weighted_count": float(row["weighted_count"]),
                "baseline_weighted_count": float(other["weighted_count"]),
                "weighted_count_delta": float(row["weighted_count"] - other["weighted_count"]),
                "candidate_weighted_agi": float(row["weighted_agi"]),
                "baseline_weighted_agi": float(other["weighted_agi"]),
                "weighted_agi_delta": float(row["weighted_agi"] - other["weighted_agi"]),
            }
        )

    baseline_bins = {row["agi_bin"]: row for row in baseline["hoh_agi_support"]}
    hoh_rows = []
    for row in candidate["hoh_agi_support"]:
        other = baseline_bins[row["agi_bin"]]
        hoh_rows.append(
            {
                "agi_bin": row["agi_bin"],
                "candidate_weighted_count": float(row["weighted_count"]),
                "baseline_weighted_count": float(other["weighted_count"]),
                "weighted_count_delta": float(row["weighted_count"] - other["weighted_count"]),
                "candidate_weighted_agi": float(row["weighted_agi"]),
                "baseline_weighted_agi": float(other["weighted_agi"]),
                "weighted_agi_delta": float(row["weighted_agi"] - other["weighted_agi"]),
            }
        )

    def age_value_delta(name: str) -> list[dict]:
        baseline_bins = {row["age_bucket"]: row for row in baseline[name]}
        rows = []
        for row in candidate[name]:
            other = baseline_bins[row["age_bucket"]]
            rows.append(
                {
                    "age_bucket": row["age_bucket"],
                    "candidate_weighted_positive": float(row["weighted_positive"]),
                    "baseline_weighted_positive": float(other["weighted_positive"]),
                    "weighted_positive_delta": float(row["weighted_positive"] - other["weighted_positive"]),
                    "candidate_value_sum": float(row["value_sum"]),
                    "baseline_value_sum": float(other["value_sum"]),
                    "value_sum_delta": float(row["value_sum"] - other["value_sum"]),
                }
            )
        return rows

    ssi_rows = age_value_delta("ssi_by_age")
    for row in ssi_rows:
        row["candidate_weighted_recipients"] = row.pop("candidate_weighted_positive")
        row["baseline_weighted_recipients"] = row.pop("baseline_weighted_positive")
        row["weighted_recipient_delta"] = row.pop("weighted_positive_delta")
        row["candidate_ssi"] = row.pop("candidate_value_sum")
        row["baseline_ssi"] = row.pop("baseline_value_sum")
        row["ssi_delta"] = row.pop("value_sum_delta")

    medicare_part_b_rows = age_value_delta("medicare_part_b_premiums_by_age")

    all_states = sorted(
        set(candidate["state_aca_ptc_spending"])
        | set(baseline["state_aca_ptc_spending"])
    )
    state_aca_ptc_rows = []
    for state in all_states:
        candidate_row = candidate["state_aca_ptc_spending"].get(
            state,
            {"weighted_aca_ptc": 0.0, "weighted_positive_households": 0.0},
        )
        baseline_row = baseline["state_aca_ptc_spending"].get(
            state,
            {"weighted_aca_ptc": 0.0, "weighted_positive_households": 0.0},
        )
        state_aca_ptc_rows.append(
            {
                "state": state,
                "candidate_weighted_aca_ptc": float(candidate_row["weighted_aca_ptc"]),
                "baseline_weighted_aca_ptc": float(baseline_row["weighted_aca_ptc"]),
                "weighted_aca_ptc_delta": float(candidate_row["weighted_aca_ptc"] - baseline_row["weighted_aca_ptc"]),
                "candidate_weighted_positive_households": float(candidate_row["weighted_positive_households"]),
                "baseline_weighted_positive_households": float(baseline_row["weighted_positive_households"]),
                "weighted_positive_household_delta": float(
                    candidate_row["weighted_positive_households"]
                    - baseline_row["weighted_positive_households"]
                ),
            }
        )
    state_aca_ptc_rows.sort(
        key=lambda row: abs(row["weighted_aca_ptc_delta"]),
        reverse=True,
    )

    all_states = sorted(
        set(candidate["state_marketplace_enrollment"])
        | set(baseline["state_marketplace_enrollment"])
    )
    state_marketplace_rows = []
    for state in all_states:
        candidate_row = candidate["state_marketplace_enrollment"].get(
            state,
            {"weighted_marketplace_enrollment": 0.0},
        )
        baseline_row = baseline["state_marketplace_enrollment"].get(
            state,
            {"weighted_marketplace_enrollment": 0.0},
        )
        state_marketplace_rows.append(
            {
                "state": state,
                "candidate_weighted_marketplace_enrollment": float(candidate_row["weighted_marketplace_enrollment"]),
                "baseline_weighted_marketplace_enrollment": float(baseline_row["weighted_marketplace_enrollment"]),
                "weighted_marketplace_enrollment_delta": float(
                    candidate_row["weighted_marketplace_enrollment"]
                    - baseline_row["weighted_marketplace_enrollment"]
                ),
            }
        )
    state_marketplace_rows.sort(
        key=lambda row: abs(row["weighted_marketplace_enrollment_delta"]),
        reverse=True,
    )

    all_states = sorted(
        set(candidate["state_age_bucket_support"])
        | set(baseline["state_age_bucket_support"])
    )
    state_age_rows = []
    for state in all_states:
        candidate_row = candidate["state_age_bucket_support"].get(
            state,
            {"bucket_weights": {}},
        )
        baseline_row = baseline["state_age_bucket_support"].get(
            state,
            {"bucket_weights": {}},
        )
        for label, _lower, _upper in AGE_BUCKETS:
            candidate_weight = float(candidate_row["bucket_weights"].get(label, 0.0))
            baseline_weight = float(baseline_row["bucket_weights"].get(label, 0.0))
            state_age_rows.append(
                {
                    "state": state,
                    "age_bucket": label,
                    "candidate_weight": candidate_weight,
                    "baseline_weight": baseline_weight,
                    "weight_delta": candidate_weight - baseline_weight,
                }
            )
    state_age_rows.sort(key=lambda row: abs(row["weight_delta"]), reverse=True)

    return {
        "critical_input_support": critical_rows,
        "filing_status_weighted_delta": filing_status_rows,
        "mfs_high_agi_delta": mfs_rows,
        "hoh_agi_delta": hoh_rows,
        "ssi_by_age_delta": ssi_rows,
        "medicare_part_b_premiums_by_age_delta": medicare_part_b_rows,
        "state_aca_ptc_spending_top_gaps": state_aca_ptc_rows[:15],
        "state_marketplace_enrollment_top_gaps": state_marketplace_rows[:15],
        "state_age_bucket_top_gaps": state_age_rows[:20],
    }


candidate = build_snapshot(CANDIDATE_DATASET)
baseline = build_snapshot(BASELINE_DATASET)
payload = {
    "metric": "enhanced_cps_support_audit",
    "period": PERIOD,
    "candidate_dataset": CANDIDATE_DATASET,
    "baseline_dataset": BASELINE_DATASET,
    "candidate": candidate,
    "baseline": baseline,
    "comparisons": compare_snapshots(candidate, baseline),
}
print(json.dumps(payload, sort_keys=True))
""".strip()

_PE_NATIVE_SUPPORT_AUDIT_BATCH_SCRIPT = """
import json
import sys
from pathlib import Path

import h5py
import numpy as np
from policyengine_core.data import Dataset

REPO_ROOT = sys.argv[1]
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from policyengine_us import Microsimulation

PERIOD = int(sys.argv[2])
BASELINE_DATASET = sys.argv[3]
CANDIDATE_DATASETS = json.loads(sys.argv[4])

STATE_FIPS_TO_ABBR = {
    1: "AL", 2: "AK", 4: "AZ", 5: "AR", 6: "CA", 8: "CO", 9: "CT", 10: "DE",
    11: "DC", 12: "FL", 13: "GA", 15: "HI", 16: "ID", 17: "IL", 18: "IN",
    19: "IA", 20: "KS", 21: "KY", 22: "LA", 23: "ME", 24: "MD", 25: "MA",
    26: "MI", 27: "MN", 28: "MS", 29: "MO", 30: "MT", 31: "NE", 32: "NV",
    33: "NH", 34: "NJ", 35: "NM", 36: "NY", 37: "NC", 38: "ND", 39: "OH",
    40: "OK", 41: "OR", 42: "PA", 44: "RI", 45: "SC", 46: "SD", 47: "TN",
    48: "TX", 49: "UT", 50: "VT", 51: "VA", 53: "WA", 54: "WV", 55: "WI",
    56: "WY",
}
CRITICAL_PERSON_VARIABLES = (
    "has_marketplace_health_coverage",
    "has_esi",
    "medicare_part_b_premiums",
    "child_support_expense",
    "self_employment_income_before_lsr",
    "rental_income",
    "non_sch_d_capital_gains",
)
HIGH_SIGNAL_MFS_AGI_BINS = (
    ("75k_to_100k", 75_000.0, 100_000.0),
    ("100k_to_200k", 100_000.0, 200_000.0),
    ("200k_to_500k", 200_000.0, 500_000.0),
    ("500k_plus", 500_000.0, np.inf),
)
HIGH_SIGNAL_HOH_AGI_BINS = (
    ("20k_to_25k", 20_000.0, 25_000.0),
    ("25k_to_30k", 25_000.0, 30_000.0),
    ("30k_to_40k", 30_000.0, 40_000.0),
    ("200k_to_500k", 200_000.0, 500_000.0),
    ("500k_to_1m", 500_000.0, 1_000_000.0),
    ("1m_plus", 1_000_000.0, np.inf),
)
AGE_BUCKETS = (
    ("0_to_4", 0, 5),
    ("5_to_17", 5, 18),
    ("18_to_29", 18, 30),
    ("30_to_44", 30, 45),
    ("45_to_64", 45, 65),
    ("65_plus", 65, np.inf),
)
SSI_AGE_BUCKETS = (
    ("all", -np.inf, np.inf),
    ("under_18", 0, 18),
    ("18_to_64", 18, 65),
    ("65_plus", 65, np.inf),
)
MEDICARE_PART_B_AGE_BUCKETS = (
    ("age_0_to_9", 0, 10),
    ("age_10_to_19", 10, 20),
    ("age_20_to_29", 20, 30),
    ("age_30_to_39", 30, 40),
    ("age_40_to_49", 40, 50),
    ("age_50_to_59", 50, 60),
    ("age_60_to_64", 60, 65),
    ("age_65_plus", 65, np.inf),
)


def dataset_from_path(dataset_path: str, dataset_name: str):
    class LocalDataset(Dataset):
        name = dataset_name
        label = dataset_name
        file_path = dataset_path
        data_format = Dataset.TIME_PERIOD_ARRAYS
        time_period = PERIOD

    return LocalDataset


def stored_variables_for(dataset_path: str) -> set[str]:
    with h5py.File(dataset_path, "r") as handle:
        return set(handle.keys())


def state_abbr(value) -> str:
    if value is None:
        return "NA"
    try:
        numeric = int(value)
    except (TypeError, ValueError):
        return str(value)
    return STATE_FIPS_TO_ABBR.get(numeric, str(numeric))


def normalize_status(value) -> str:
    if hasattr(value, "name"):
        return str(value.name)
    text = str(value)
    if "." in text:
        text = text.rsplit(".", 1)[-1]
    normalized = text.strip().upper().replace(" ", "_")
    if normalized in {
        "SINGLE",
        "JOINT",
        "SEPARATE",
        "HEAD_OF_HOUSEHOLD",
        "SURVIVING_SPOUSE",
    }:
        return normalized
    return normalized


def summarize_numeric(values, weights, *, stored: bool) -> dict[str, float | int | bool]:
    arr = np.nan_to_num(np.asarray(values, dtype=np.float64), nan=0.0)
    w = np.asarray(weights, dtype=np.float64)
    positive = arr > 0.0
    negative = arr < 0.0
    nonzero = arr != 0.0
    return {
        "stored": bool(stored),
        "nonzero_count": int(nonzero.sum()),
        "positive_count": int(positive.sum()),
        "negative_count": int(negative.sum()),
        "weighted_nonzero": float(w[nonzero].sum()),
        "weighted_positive": float(w[positive].sum()),
        "weighted_negative": float(w[negative].sum()),
        "value_sum": float((arr * w).sum()),
    }


def summarize_bool(values, weights, *, stored: bool) -> dict[str, float | int | bool]:
    arr = np.asarray(values).astype(bool)
    w = np.asarray(weights, dtype=np.float64)
    return {
        "stored": bool(stored),
        "true_count": int(arr.sum()),
        "false_count": int((~arr).sum()),
        "weighted_true": float(w[arr].sum()),
        "weighted_false": float(w[~arr].sum()),
    }


def build_snapshot(dataset_path: str) -> dict:
    dataset_cls = dataset_from_path(
        dataset_path,
        Path(dataset_path).stem.replace("-", "_"),
    )
    stored_variables = stored_variables_for(dataset_path)
    sim = Microsimulation(dataset=dataset_cls)
    sim.default_calculation_period = PERIOD

    person_weights = sim.calculate("person_weight", period=PERIOD).values.astype(np.float64)
    household_weights = sim.calculate("household_weight", period=PERIOD).values.astype(np.float64)
    tax_unit_weights = sim.calculate("tax_unit_weight", period=PERIOD).values.astype(np.float64)
    person_state = sim.calculate("state_fips", map_to="person", period=PERIOD).values
    household_state = sim.calculate("state_fips", map_to="household", period=PERIOD).values
    person_age = sim.calculate("age", period=PERIOD).values.astype(np.float64)
    marketplace = sim.calculate("has_marketplace_health_coverage", period=PERIOD).values
    filing_status = sim.calculate("filing_status", period=PERIOD).values
    adjusted_gross_income = sim.calculate("adjusted_gross_income", period=PERIOD).values.astype(np.float64)
    ssi = sim.calculate("ssi", period=PERIOD).values.astype(np.float64)
    medicare_part_b_premiums = sim.calculate("medicare_part_b_premiums", period=PERIOD).values.astype(np.float64)
    aca_ptc_household = sim.calculate("aca_ptc", map_to="household", period=PERIOD).values.astype(np.float64)

    critical_support = {}
    for variable in CRITICAL_PERSON_VARIABLES:
        values = sim.calculate(variable, period=PERIOD).values
        if np.asarray(values).dtype == np.bool_:
            critical_support[variable] = summarize_bool(
                values,
                person_weights,
                stored=variable in stored_variables,
            )
        else:
            critical_support[variable] = summarize_numeric(
                values,
                person_weights,
                stored=variable in stored_variables,
            )

    normalized_filing_status = np.asarray([normalize_status(value) for value in filing_status])
    filing_status_counts = {}
    for status in ("SINGLE", "JOINT", "SEPARATE", "HEAD_OF_HOUSEHOLD", "SURVIVING_SPOUSE"):
        mask = normalized_filing_status == status
        filing_status_counts[status] = {
            "count": int(mask.sum()),
            "weighted_count": float(tax_unit_weights[mask].sum()),
        }

    def agi_support_for_status(status: str, bins) -> list[dict]:
        status_mask = normalized_filing_status == status
        rows = []
        for label, lower, upper in bins:
            mask = status_mask & (adjusted_gross_income >= lower) & (adjusted_gross_income < upper)
            rows.append(
                {
                    "agi_bin": label,
                    "count": int(mask.sum()),
                    "weighted_count": float(tax_unit_weights[mask].sum()),
                    "weighted_agi": float((adjusted_gross_income[mask] * tax_unit_weights[mask]).sum()),
                }
            )
        return rows

    def person_value_by_age(values, buckets) -> list[dict]:
        arr = np.nan_to_num(np.asarray(values, dtype=np.float64), nan=0.0)
        rows = []
        for label, lower, upper in buckets:
            age_mask = (person_age >= lower) & (person_age < upper)
            positive = age_mask & (arr > 0.0)
            rows.append(
                {
                    "age_bucket": label,
                    "person_count": int(age_mask.sum()),
                    "positive_count": int(positive.sum()),
                    "weighted_people": float(person_weights[age_mask].sum()),
                    "weighted_positive": float(person_weights[positive].sum()),
                    "value_sum": float((arr[age_mask] * person_weights[age_mask]).sum()),
                }
            )
        return rows

    mfs_agi_support = agi_support_for_status("SEPARATE", HIGH_SIGNAL_MFS_AGI_BINS)
    hoh_agi_support = agi_support_for_status("HEAD_OF_HOUSEHOLD", HIGH_SIGNAL_HOH_AGI_BINS)
    ssi_by_age = person_value_by_age(ssi, SSI_AGE_BUCKETS)
    medicare_part_b_by_age = person_value_by_age(
        medicare_part_b_premiums,
        MEDICARE_PART_B_AGE_BUCKETS,
    )

    state_aca_ptc = {}
    for state in sorted({state_abbr(value) for value in household_state}):
        state_mask = np.asarray([state_abbr(value) == state for value in household_state], dtype=bool)
        positive = state_mask & (aca_ptc_household > 0.0)
        state_aca_ptc[state] = {
            "weighted_households": float(household_weights[state_mask].sum()),
            "weighted_positive_households": float(household_weights[positive].sum()),
            "weighted_aca_ptc": float((aca_ptc_household[state_mask] * household_weights[state_mask]).sum()),
        }

    states = sorted({state_abbr(value) for value in person_state})
    state_marketplace = {}
    state_age_bucket = {}
    marketplace_bool = np.asarray(marketplace).astype(bool)
    for state in states:
        state_mask = np.asarray([state_abbr(value) == state for value in person_state], dtype=bool)
        enrolled = state_mask & marketplace_bool
        state_marketplace[state] = {
            "weighted_people": float(person_weights[state_mask].sum()),
            "weighted_marketplace_enrollment": float(person_weights[enrolled].sum()),
        }
        bucket_weights = {}
        nonempty = 0
        for label, lower, upper in AGE_BUCKETS:
            mask = state_mask & (person_age >= lower) & (person_age < upper)
            weight = float(person_weights[mask].sum())
            bucket_weights[label] = weight
            if weight > 0.0:
                nonempty += 1
        state_age_bucket[state] = {
            "nonempty_buckets": int(nonempty),
            "bucket_weights": bucket_weights,
        }

    return {
        "dataset": dataset_path,
        "stored_variable_count": int(len(stored_variables)),
        "stored_variables": sorted(stored_variables),
        "critical_input_support": critical_support,
        "filing_status_weighted_counts": filing_status_counts,
        "mfs_high_agi_support": mfs_agi_support,
        "hoh_agi_support": hoh_agi_support,
        "ssi_by_age": ssi_by_age,
        "medicare_part_b_premiums_by_age": medicare_part_b_by_age,
        "state_aca_ptc_spending": state_aca_ptc,
        "state_marketplace_enrollment": state_marketplace,
        "state_age_bucket_support": state_age_bucket,
    }


def compare_snapshots(candidate: dict, baseline: dict) -> dict:
    critical_rows = []
    for variable in CRITICAL_PERSON_VARIABLES:
        candidate_row = candidate["critical_input_support"][variable]
        baseline_row = baseline["critical_input_support"][variable]
        candidate_weighted = candidate_row.get("weighted_nonzero", candidate_row.get("weighted_true", 0.0))
        baseline_weighted = baseline_row.get("weighted_nonzero", baseline_row.get("weighted_true", 0.0))
        critical_rows.append(
            {
                "variable": variable,
                "candidate_stored": bool(candidate_row.get("stored", False)),
                "baseline_stored": bool(baseline_row.get("stored", False)),
                "candidate_weighted_nonzero": float(candidate_weighted),
                "baseline_weighted_nonzero": float(baseline_weighted),
                "weighted_nonzero_delta": float(candidate_weighted - baseline_weighted),
            }
        )

    filing_status_rows = []
    for status in ("SINGLE", "JOINT", "SEPARATE", "HEAD_OF_HOUSEHOLD", "SURVIVING_SPOUSE"):
        candidate_row = candidate["filing_status_weighted_counts"][status]
        baseline_row = baseline["filing_status_weighted_counts"][status]
        filing_status_rows.append(
            {
                "filing_status": status,
                "candidate_weighted_count": float(candidate_row["weighted_count"]),
                "baseline_weighted_count": float(baseline_row["weighted_count"]),
                "weighted_count_delta": float(candidate_row["weighted_count"] - baseline_row["weighted_count"]),
            }
        )

    baseline_bins = {row["agi_bin"]: row for row in baseline["mfs_high_agi_support"]}
    mfs_rows = []
    for row in candidate["mfs_high_agi_support"]:
        other = baseline_bins[row["agi_bin"]]
        mfs_rows.append(
            {
                "agi_bin": row["agi_bin"],
                "candidate_weighted_count": float(row["weighted_count"]),
                "baseline_weighted_count": float(other["weighted_count"]),
                "weighted_count_delta": float(row["weighted_count"] - other["weighted_count"]),
                "candidate_weighted_agi": float(row["weighted_agi"]),
                "baseline_weighted_agi": float(other["weighted_agi"]),
                "weighted_agi_delta": float(row["weighted_agi"] - other["weighted_agi"]),
            }
        )

    baseline_bins = {row["agi_bin"]: row for row in baseline["hoh_agi_support"]}
    hoh_rows = []
    for row in candidate["hoh_agi_support"]:
        other = baseline_bins[row["agi_bin"]]
        hoh_rows.append(
            {
                "agi_bin": row["agi_bin"],
                "candidate_weighted_count": float(row["weighted_count"]),
                "baseline_weighted_count": float(other["weighted_count"]),
                "weighted_count_delta": float(row["weighted_count"] - other["weighted_count"]),
                "candidate_weighted_agi": float(row["weighted_agi"]),
                "baseline_weighted_agi": float(other["weighted_agi"]),
                "weighted_agi_delta": float(row["weighted_agi"] - other["weighted_agi"]),
            }
        )

    def age_value_delta(name: str) -> list[dict]:
        baseline_bins = {row["age_bucket"]: row for row in baseline[name]}
        rows = []
        for row in candidate[name]:
            other = baseline_bins[row["age_bucket"]]
            rows.append(
                {
                    "age_bucket": row["age_bucket"],
                    "candidate_weighted_positive": float(row["weighted_positive"]),
                    "baseline_weighted_positive": float(other["weighted_positive"]),
                    "weighted_positive_delta": float(row["weighted_positive"] - other["weighted_positive"]),
                    "candidate_value_sum": float(row["value_sum"]),
                    "baseline_value_sum": float(other["value_sum"]),
                    "value_sum_delta": float(row["value_sum"] - other["value_sum"]),
                }
            )
        return rows

    ssi_rows = age_value_delta("ssi_by_age")
    for row in ssi_rows:
        row["candidate_weighted_recipients"] = row.pop("candidate_weighted_positive")
        row["baseline_weighted_recipients"] = row.pop("baseline_weighted_positive")
        row["weighted_recipient_delta"] = row.pop("weighted_positive_delta")
        row["candidate_ssi"] = row.pop("candidate_value_sum")
        row["baseline_ssi"] = row.pop("baseline_value_sum")
        row["ssi_delta"] = row.pop("value_sum_delta")

    medicare_part_b_rows = age_value_delta("medicare_part_b_premiums_by_age")

    all_states = sorted(
        set(candidate["state_aca_ptc_spending"])
        | set(baseline["state_aca_ptc_spending"])
    )
    state_aca_ptc_rows = []
    for state in all_states:
        candidate_row = candidate["state_aca_ptc_spending"].get(
            state,
            {"weighted_aca_ptc": 0.0, "weighted_positive_households": 0.0},
        )
        baseline_row = baseline["state_aca_ptc_spending"].get(
            state,
            {"weighted_aca_ptc": 0.0, "weighted_positive_households": 0.0},
        )
        state_aca_ptc_rows.append(
            {
                "state": state,
                "candidate_weighted_aca_ptc": float(candidate_row["weighted_aca_ptc"]),
                "baseline_weighted_aca_ptc": float(baseline_row["weighted_aca_ptc"]),
                "weighted_aca_ptc_delta": float(candidate_row["weighted_aca_ptc"] - baseline_row["weighted_aca_ptc"]),
                "candidate_weighted_positive_households": float(candidate_row["weighted_positive_households"]),
                "baseline_weighted_positive_households": float(baseline_row["weighted_positive_households"]),
                "weighted_positive_household_delta": float(
                    candidate_row["weighted_positive_households"]
                    - baseline_row["weighted_positive_households"]
                ),
            }
        )
    state_aca_ptc_rows.sort(
        key=lambda row: abs(row["weighted_aca_ptc_delta"]),
        reverse=True,
    )

    all_states = sorted(
        set(candidate["state_marketplace_enrollment"])
        | set(baseline["state_marketplace_enrollment"])
    )
    state_marketplace_rows = []
    for state in all_states:
        candidate_row = candidate["state_marketplace_enrollment"].get(
            state,
            {"weighted_marketplace_enrollment": 0.0},
        )
        baseline_row = baseline["state_marketplace_enrollment"].get(
            state,
            {"weighted_marketplace_enrollment": 0.0},
        )
        state_marketplace_rows.append(
            {
                "state": state,
                "candidate_weighted_marketplace_enrollment": float(candidate_row["weighted_marketplace_enrollment"]),
                "baseline_weighted_marketplace_enrollment": float(baseline_row["weighted_marketplace_enrollment"]),
                "weighted_marketplace_enrollment_delta": float(
                    candidate_row["weighted_marketplace_enrollment"]
                    - baseline_row["weighted_marketplace_enrollment"]
                ),
            }
        )
    state_marketplace_rows.sort(
        key=lambda row: abs(row["weighted_marketplace_enrollment_delta"]),
        reverse=True,
    )

    all_states = sorted(
        set(candidate["state_age_bucket_support"])
        | set(baseline["state_age_bucket_support"])
    )
    state_age_rows = []
    for state in all_states:
        candidate_row = candidate["state_age_bucket_support"].get(
            state,
            {"bucket_weights": {}},
        )
        baseline_row = baseline["state_age_bucket_support"].get(
            state,
            {"bucket_weights": {}},
        )
        for label, _lower, _upper in AGE_BUCKETS:
            candidate_weight = float(candidate_row["bucket_weights"].get(label, 0.0))
            baseline_weight = float(baseline_row["bucket_weights"].get(label, 0.0))
            state_age_rows.append(
                {
                    "state": state,
                    "age_bucket": label,
                    "candidate_weight": candidate_weight,
                    "baseline_weight": baseline_weight,
                    "weight_delta": candidate_weight - baseline_weight,
                }
            )
    state_age_rows.sort(key=lambda row: abs(row["weight_delta"]), reverse=True)

    return {
        "critical_input_support": critical_rows,
        "filing_status_weighted_delta": filing_status_rows,
        "mfs_high_agi_delta": mfs_rows,
        "hoh_agi_delta": hoh_rows,
        "ssi_by_age_delta": ssi_rows,
        "medicare_part_b_premiums_by_age_delta": medicare_part_b_rows,
        "state_aca_ptc_spending_top_gaps": state_aca_ptc_rows[:15],
        "state_marketplace_enrollment_top_gaps": state_marketplace_rows[:15],
        "state_age_bucket_top_gaps": state_age_rows[:20],
    }


baseline = build_snapshot(BASELINE_DATASET)
results = []
for candidate_dataset in CANDIDATE_DATASETS:
    candidate = build_snapshot(candidate_dataset)
    results.append(
        {
            "candidate_dataset": candidate_dataset,
            "candidate": candidate,
            "comparisons": compare_snapshots(candidate, baseline),
        }
    )

payload = {
    "metric": "enhanced_cps_support_audit_batch",
    "period": PERIOD,
    "baseline_dataset": BASELINE_DATASET,
    "baseline": baseline,
    "results": results,
}
print(json.dumps(payload, sort_keys=True))
""".strip()


@dataclass(frozen=True)
class PolicyEngineUSEnhancedCPSNativeScores:
    """Exact enhanced-CPS native-loss comparison for one candidate/baseline pair."""

    metric: str
    period: int
    candidate_dataset: str
    baseline_dataset: str
    candidate_enhanced_cps_native_loss: float
    baseline_enhanced_cps_native_loss: float
    enhanced_cps_native_loss_delta: float
    candidate_unweighted_msre: float
    baseline_unweighted_msre: float
    unweighted_msre_delta: float
    n_targets_total: int
    n_targets_kept: int
    n_targets_zero_dropped: int
    n_targets_bad_dropped: int
    n_national_targets: int
    n_state_targets: int
    candidate_weight_sum: float
    baseline_weight_sum: float
    family_breakdown: tuple[dict[str, Any], ...] = field(default_factory=tuple)

    def to_dict(self) -> dict[str, Any]:
        return {
            "metric": self.metric,
            "period": self.period,
            "candidate_dataset": self.candidate_dataset,
            "baseline_dataset": self.baseline_dataset,
            "candidate_enhanced_cps_native_loss": (
                self.candidate_enhanced_cps_native_loss
            ),
            "baseline_enhanced_cps_native_loss": (
                self.baseline_enhanced_cps_native_loss
            ),
            "enhanced_cps_native_loss_delta": self.enhanced_cps_native_loss_delta,
            "candidate_unweighted_msre": self.candidate_unweighted_msre,
            "baseline_unweighted_msre": self.baseline_unweighted_msre,
            "unweighted_msre_delta": self.unweighted_msre_delta,
            "n_targets_total": self.n_targets_total,
            "n_targets_kept": self.n_targets_kept,
            "n_targets_zero_dropped": self.n_targets_zero_dropped,
            "n_targets_bad_dropped": self.n_targets_bad_dropped,
            "n_national_targets": self.n_national_targets,
            "n_state_targets": self.n_state_targets,
            "candidate_weight_sum": self.candidate_weight_sum,
            "baseline_weight_sum": self.baseline_weight_sum,
            "family_breakdown": list(self.family_breakdown),
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> PolicyEngineUSEnhancedCPSNativeScores:
        return cls(
            metric=str(payload["metric"]),
            period=int(payload["period"]),
            candidate_dataset=str(payload["candidate_dataset"]),
            baseline_dataset=str(payload["baseline_dataset"]),
            candidate_enhanced_cps_native_loss=float(
                payload["candidate_enhanced_cps_native_loss"]
            ),
            baseline_enhanced_cps_native_loss=float(
                payload["baseline_enhanced_cps_native_loss"]
            ),
            enhanced_cps_native_loss_delta=float(
                payload["enhanced_cps_native_loss_delta"]
            ),
            candidate_unweighted_msre=float(payload["candidate_unweighted_msre"]),
            baseline_unweighted_msre=float(payload["baseline_unweighted_msre"]),
            unweighted_msre_delta=float(payload["unweighted_msre_delta"]),
            n_targets_total=int(payload["n_targets_total"]),
            n_targets_kept=int(payload["n_targets_kept"]),
            n_targets_zero_dropped=int(payload["n_targets_zero_dropped"]),
            n_targets_bad_dropped=int(payload["n_targets_bad_dropped"]),
            n_national_targets=int(payload["n_national_targets"]),
            n_state_targets=int(payload["n_state_targets"]),
            candidate_weight_sum=float(payload["candidate_weight_sum"]),
            baseline_weight_sum=float(payload["baseline_weight_sum"]),
            family_breakdown=tuple(payload.get("family_breakdown", ())),
        )


PolicyEngineUSNativeBroadLossScore = PolicyEngineUSEnhancedCPSNativeScores


def resolve_policyengine_us_data_repo_root(
    repo_root: str | Path | None = None,
) -> Path:
    """Resolve the local policyengine-us-data checkout used for native scoring."""

    candidates: list[Path] = []
    if repo_root is not None:
        candidates.append(Path(repo_root))
    env_repo = os.environ.get(_PE_US_DATA_REPO_ENV)
    if env_repo:
        candidates.append(Path(env_repo))
    candidates.append(_DEFAULT_PE_US_DATA_REPO)

    for candidate in candidates:
        resolved = candidate.expanduser().resolve()
        if (resolved / "policyengine_us_data").exists():
            return resolved
    searched = ", ".join(str(path.expanduser()) for path in candidates)
    raise FileNotFoundError(
        "Could not resolve policyengine-us-data repo root. "
        f"Searched: {searched}"
    )


def resolve_policyengine_us_data_python(
    python_executable: str | Path | None = None,
    *,
    repo_root: str | Path | None = None,
) -> Path:
    """Resolve a Python executable with policyengine-us-data installed."""

    candidates: list[Path] = []
    if python_executable is not None:
        candidates.append(Path(python_executable))
    env_python = os.environ.get(_PE_US_DATA_PYTHON_ENV)
    if env_python:
        candidates.append(Path(env_python))
    resolved_repo = resolve_policyengine_us_data_repo_root(repo_root)
    candidates.extend(
        (
            resolved_repo / ".venv" / "bin" / "python",
            resolved_repo / "venv" / "bin" / "python",
        )
    )

    for candidate in candidates:
        expanded = candidate.expanduser()
        if expanded.exists() and os.access(expanded, os.X_OK):
            return expanded
    searched = ", ".join(str(path.expanduser()) for path in candidates)
    raise FileNotFoundError(
        "Could not resolve a usable policyengine-us-data Python executable. "
        f"Searched: {searched}"
    )


def build_policyengine_us_data_pythonpath(
    repo_root: str | Path | None = None,
    *,
    existing_pythonpath: str | None = None,
) -> str:
    """Build the native-scoring PYTHONPATH for local PE-US-data checkouts."""

    resolved_repo = resolve_policyengine_us_data_repo_root(repo_root)
    path_entries: list[str] = [str(resolved_repo)]

    sibling_microimpute = resolved_repo.parent / "microimpute"
    if (sibling_microimpute / "microimpute").exists():
        path_entries.append(str(sibling_microimpute))

    if existing_pythonpath:
        path_entries.extend(
            entry for entry in existing_pythonpath.split(os.pathsep) if entry
        )
    return os.pathsep.join(path_entries)


def build_policyengine_us_data_subprocess_env(
    repo_root: str | Path | None = None,
    *,
    base_env: dict[str, str] | None = None,
) -> dict[str, str]:
    """Build a clean subprocess env for PE-native scoring helpers."""

    source_env = dict(os.environ if base_env is None else base_env)
    env = {
        key: source_env[key]
        for key in _PE_NATIVE_SCORE_BASE_ENV_VARS
        if key in source_env and source_env[key]
    }
    env["PYTHONPATH"] = build_policyengine_us_data_pythonpath(
        repo_root,
        existing_pythonpath=source_env.get("PYTHONPATH"),
    )
    return env


def compute_policyengine_us_enhanced_cps_native_scores(
    candidate_dataset: str | Path,
    baseline_dataset: str | Path,
    *,
    period: int = 2024,
    policyengine_us_data_python: str | Path | None = None,
    policyengine_us_data_repo: str | Path | None = None,
) -> PolicyEngineUSEnhancedCPSNativeScores:
    """Score one candidate and baseline under the exact enhanced-CPS loss."""
    resolved_repo = resolve_policyengine_us_data_repo_root(policyengine_us_data_repo)
    env = build_policyengine_us_data_subprocess_env(resolved_repo)
    if policyengine_us_data_python is not None:
        command = [str(Path(policyengine_us_data_python).expanduser())]
    else:
        command = ["uv", "run", "--project", str(resolved_repo), "python"]
    completed = subprocess.run(
        [
            *command,
            "-c",
            _PE_NATIVE_BROAD_SCORE_SCRIPT,
            str(resolved_repo),
            json.dumps(_ENHANCED_CPS_BAD_TARGETS),
            str(int(period)),
            str(Path(candidate_dataset).expanduser().resolve()),
            str(Path(baseline_dataset).expanduser().resolve()),
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
        raise RuntimeError(f"PE-native broad loss scoring failed: {detail}")
    payload = json.loads(completed.stdout)
    return PolicyEngineUSEnhancedCPSNativeScores.from_dict(payload)


def score_policyengine_us_native_broad_loss(
    candidate_dataset: str | Path,
    baseline_dataset: str | Path,
    *,
    period: int = 2024,
    python_executable: str | Path | None = None,
    repo_root: str | Path | None = None,
) -> PolicyEngineUSEnhancedCPSNativeScores:
    """Backward-compatible alias for the exact enhanced-CPS loss scorer."""
    return compute_policyengine_us_enhanced_cps_native_scores(
        candidate_dataset,
        baseline_dataset,
        period=period,
        policyengine_us_data_python=python_executable,
        policyengine_us_data_repo=repo_root,
    )


def compute_us_pe_native_scores(
    *,
    candidate_dataset_path: str | Path,
    baseline_dataset_path: str | Path,
    period: int = 2024,
    policyengine_us_data_repo: str | Path | None = None,
    policyengine_us_data_python: str | Path | None = None,
) -> dict[str, Any]:
    """Build the saved manifest payload for PE-native broad scoring."""

    score = compute_policyengine_us_enhanced_cps_native_scores(
        candidate_dataset_path,
        baseline_dataset_path,
        period=period,
        policyengine_us_data_python=policyengine_us_data_python,
        policyengine_us_data_repo=policyengine_us_data_repo,
    )
    return {
        "metric": score.metric,
        "period": score.period,
        "summary": {
            "candidate_enhanced_cps_native_loss": (
                score.candidate_enhanced_cps_native_loss
            ),
            "baseline_enhanced_cps_native_loss": (
                score.baseline_enhanced_cps_native_loss
            ),
            "enhanced_cps_native_loss_delta": score.enhanced_cps_native_loss_delta,
            "candidate_beats_baseline": score.enhanced_cps_native_loss_delta < 0.0,
            "candidate_unweighted_msre": score.candidate_unweighted_msre,
            "baseline_unweighted_msre": score.baseline_unweighted_msre,
            "unweighted_msre_delta": score.unweighted_msre_delta,
            "n_targets_total": score.n_targets_total,
            "n_targets_kept": score.n_targets_kept,
            "n_targets_zero_dropped": score.n_targets_zero_dropped,
            "n_targets_bad_dropped": score.n_targets_bad_dropped,
            "n_national_targets": score.n_national_targets,
            "n_state_targets": score.n_state_targets,
        },
        "broad_loss": score.to_dict(),
        "family_breakdown": list(score.family_breakdown),
    }


def compute_batch_us_pe_native_scores(
    *,
    candidate_dataset_paths: list[str | Path] | tuple[str | Path, ...],
    baseline_dataset_path: str | Path,
    period: int = 2024,
    policyengine_us_data_repo: str | Path | None = None,
    policyengine_us_data_python: str | Path | None = None,
) -> list[dict[str, Any]]:
    """Score multiple candidates against one baseline in a single PE-native subprocess."""

    if not candidate_dataset_paths:
        return []
    started_at = perf_counter()
    resolved_repo = resolve_policyengine_us_data_repo_root(policyengine_us_data_repo)
    env = build_policyengine_us_data_subprocess_env(resolved_repo)
    if policyengine_us_data_python is not None:
        command = [str(Path(policyengine_us_data_python).expanduser())]
    else:
        command = ["uv", "run", "--project", str(resolved_repo), "python"]
    completed = subprocess.run(
        [
            *command,
            "-c",
            _PE_NATIVE_BROAD_BATCH_SCORE_SCRIPT,
            str(resolved_repo),
            json.dumps(_ENHANCED_CPS_BAD_TARGETS),
            str(int(period)),
            str(Path(baseline_dataset_path).expanduser().resolve()),
            json.dumps(
                [
                    str(Path(candidate_path).expanduser().resolve())
                    for candidate_path in candidate_dataset_paths
                ]
            ),
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
        raise RuntimeError(f"PE-native batch broad loss scoring failed: {detail}")
    payload = json.loads(completed.stdout)
    elapsed_seconds = perf_counter() - started_at
    results = [
        {
            "metric": item["metric"],
            "period": int(item["period"]),
            "summary": {
                "candidate_enhanced_cps_native_loss": float(
                    item["candidate_enhanced_cps_native_loss"]
                ),
                "baseline_enhanced_cps_native_loss": float(
                    item["baseline_enhanced_cps_native_loss"]
                ),
                "enhanced_cps_native_loss_delta": float(
                    item["enhanced_cps_native_loss_delta"]
                ),
                "candidate_beats_baseline": bool(
                    item["candidate_beats_baseline"]
                ),
                "candidate_unweighted_msre": float(item["candidate_unweighted_msre"]),
                "baseline_unweighted_msre": float(item["baseline_unweighted_msre"]),
                "unweighted_msre_delta": float(item["unweighted_msre_delta"]),
                "n_targets_total": int(item["n_targets_total"]),
                "n_targets_kept": int(item["n_targets_kept"]),
                "n_targets_zero_dropped": int(item["n_targets_zero_dropped"]),
                "n_targets_bad_dropped": int(item["n_targets_bad_dropped"]),
                "n_national_targets": int(item["n_national_targets"]),
                "n_state_targets": int(item["n_state_targets"]),
            },
            "broad_loss": {
                "metric": item["metric"],
                "period": int(item["period"]),
                "candidate_dataset": str(item["candidate_dataset"]),
                "baseline_dataset": str(item["baseline_dataset"]),
                "candidate_enhanced_cps_native_loss": float(
                    item["candidate_enhanced_cps_native_loss"]
                ),
                "baseline_enhanced_cps_native_loss": float(
                    item["baseline_enhanced_cps_native_loss"]
                ),
                "enhanced_cps_native_loss_delta": float(
                    item["enhanced_cps_native_loss_delta"]
                ),
                "candidate_beats_baseline": bool(
                    item["candidate_beats_baseline"]
                ),
                "candidate_unweighted_msre": float(item["candidate_unweighted_msre"]),
                "baseline_unweighted_msre": float(item["baseline_unweighted_msre"]),
                "unweighted_msre_delta": float(item["unweighted_msre_delta"]),
                "n_targets_total": int(item["n_targets_total"]),
                "n_targets_kept": int(item["n_targets_kept"]),
                "n_targets_zero_dropped": int(item["n_targets_zero_dropped"]),
                "n_targets_bad_dropped": int(item["n_targets_bad_dropped"]),
                "n_national_targets": int(item["n_national_targets"]),
                "n_state_targets": int(item["n_state_targets"]),
                "candidate_weight_sum": float(item["candidate_weight_sum"]),
                "baseline_weight_sum": float(item["baseline_weight_sum"]),
                "family_breakdown": list(item.get("family_breakdown", [])),
            },
            "family_breakdown": list(item.get("family_breakdown", [])),
        }
        for item in payload
    ]
    for item in results:
        item["timing"] = {
            "batch_elapsed_seconds": float(elapsed_seconds),
            "batch_candidate_count": len(candidate_dataset_paths),
        }
    return results


@dataclass(frozen=True)
class PENativeTargetLookupKey:
    """Structured lookup key for a legacy PE-native target label."""

    variable: str
    count_children: int
    agi_lower: float
    agi_upper: float

    def as_tuple(self) -> tuple[str, int, float, float]:
        return (self.variable, self.count_children, self.agi_lower, self.agi_upper)

    @staticmethod
    def _json_safe_bound(value: float) -> float | str:
        if value == float("inf"):
            return "inf"
        if value == float("-inf"):
            return "-inf"
        return value

    def expected_constraints(self) -> list[dict[str, str | float | int]]:
        if self.count_children < 3:
            child_constraint: dict[str, str | float | int] = {
                "variable": "eitc_child_count",
                "operation": "==",
                "value": self.count_children,
            }
        else:
            child_constraint = {
                "variable": "eitc_child_count",
                "operation": ">",
                "value": 2,
            }
        return [
            {"variable": "tax_unit_is_filer", "operation": "==", "value": 1},
            {"variable": "eitc", "operation": ">", "value": 0},
            child_constraint,
            {
                "variable": "adjusted_gross_income",
                "operation": ">=",
                "value": self._json_safe_bound(self.agi_lower),
            },
            {
                "variable": "adjusted_gross_income",
                "operation": "<",
                "value": self._json_safe_bound(self.agi_upper),
            },
        ]

    def expected_target(self) -> dict[str, Any]:
        return {
            "variable": self.variable,
            "geo_level": "national",
            "geographic_id": "US",
            "domain_variable": _EITC_AGI_CHILD_DOMAIN_VARIABLE,
            "constraints": self.expected_constraints(),
        }


def _parse_pe_native_numeric_token(token: str) -> float:
    if token == "-inf":
        return float("-inf")
    if token == "inf":
        return float("inf")
    multipliers = {
        "bn": 1_000_000_000.0,
        "m": 1_000_000.0,
        "k": 1_000.0,
    }
    for suffix, multiplier in multipliers.items():
        if token.endswith(suffix):
            return float(token[: -len(suffix)]) * multiplier
    return float(token)


def parse_pe_native_target_lookup_key(
    target_name: str,
) -> PENativeTargetLookupKey | None:
    """Parse PE-native labels that now have structured DB equivalents."""

    match = _EITC_AGI_CHILD_LABEL.match(target_name)
    if match is None:
        return None
    metric = match.group("metric")
    variable = "tax_unit_count" if metric == "returns" else "eitc"
    return PENativeTargetLookupKey(
        variable=variable,
        count_children=int(match.group("count_children")),
        agi_lower=_parse_pe_native_numeric_token(match.group("agi_lower")),
        agi_upper=_parse_pe_native_numeric_token(match.group("agi_upper")),
    )


def _constraint_value_as_float(value: str) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _target_lookup_key_from_policyengine_target(
    target: Any,
) -> tuple[str, int, float, float] | None:
    if target.geo_level != "national":
        return None
    if target.variable not in {"eitc", "tax_unit_count"}:
        return None
    if target.domain_variable != _EITC_AGI_CHILD_DOMAIN_VARIABLE:
        return None

    agi_lower: float | None = None
    agi_upper: float | None = None
    count_children: int | None = None
    has_eitc_positive_constraint = False

    for constraint in target.constraints:
        value = str(constraint.value)
        numeric_value = _constraint_value_as_float(value)
        if (
            constraint.variable == "adjusted_gross_income"
            and constraint.operation == ">="
            and numeric_value is not None
        ):
            agi_lower = numeric_value
        elif (
            constraint.variable == "adjusted_gross_income"
            and constraint.operation == "<"
            and numeric_value is not None
        ):
            agi_upper = numeric_value
        elif constraint.variable == "eitc" and constraint.operation == ">":
            has_eitc_positive_constraint = numeric_value == 0
        elif constraint.variable == "eitc_child_count" and numeric_value is not None:
            if constraint.operation == "==":
                count_children = int(numeric_value)
            elif constraint.operation == ">" and numeric_value == 2:
                count_children = 3
            elif constraint.operation == ">=" and numeric_value == 3:
                count_children = 3

    if (
        agi_lower is None
        or agi_upper is None
        or count_children is None
        or not has_eitc_positive_constraint
    ):
        return None
    return (target.variable, count_children, agi_lower, agi_upper)


def _policyengine_target_payload(target: Any) -> dict[str, Any]:
    return {
        "target_id": target.target_id,
        "variable": target.variable,
        "period": target.period,
        "value": target.value,
        "source": target.source,
        "notes": target.notes,
        "geo_level": target.geo_level,
        "geographic_id": target.geographic_id,
        "domain_variable": target.domain_variable,
        "constraints": [
            {
                "variable": constraint.variable,
                "operation": constraint.operation,
                "value": constraint.value,
            }
            for constraint in target.constraints
        ],
    }


def _load_policyengine_target_match_index(
    target_db_path: str | Path,
    *,
    period: int,
) -> dict[tuple[str, int, float, float], list[dict[str, Any]]]:
    from microplex_us.policyengine.us import PolicyEngineUSDBTargetProvider

    provider = PolicyEngineUSDBTargetProvider(target_db_path, validate=False)
    targets = provider.load_targets(
        period=period,
        variables=["eitc", "tax_unit_count"],
        domain_variable_values=[_EITC_AGI_CHILD_DOMAIN_VARIABLE],
        geo_levels=["national"],
    )
    matches: dict[tuple[str, int, float, float], list[dict[str, Any]]] = {}
    for target in targets:
        key = _target_lookup_key_from_policyengine_target(target)
        if key is None:
            continue
        matches.setdefault(key, []).append(_policyengine_target_payload(target))
    return matches


def _default_policyengine_targets_db_path(
    policyengine_us_data_repo: str | Path | None,
) -> Path | None:
    try:
        repo = resolve_policyengine_us_data_repo_root(policyengine_us_data_repo)
    except FileNotFoundError:
        return None
    path = repo / "policyengine_us_data" / "storage" / "calibration" / "policy_data.db"
    return path if path.exists() else None


def annotate_pe_native_target_db_matches(
    payload: dict[str, Any],
    *,
    target_db_path: str | Path | None,
    period: int,
) -> dict[str, Any]:
    """Attach structured PolicyEngine target DB matches to diagnostic rows."""

    rows = list(payload.get("targets") or [])
    resolved_db_path = Path(target_db_path).expanduser() if target_db_path else None
    match_index: dict[tuple[str, int, float, float], list[dict[str, Any]]] = {}
    target_db_error = None
    if resolved_db_path is not None and resolved_db_path.exists():
        try:
            match_index = _load_policyengine_target_match_index(
                resolved_db_path,
                period=period,
            )
        except Exception as exc:  # pragma: no cover - defensive diagnostic path
            target_db_error = str(exc)

    counts = {
        "matched": 0,
        "legacy_only": 0,
        "unparsed": 0,
        "ambiguous": 0,
        "db_unavailable": 0,
    }
    annotations_by_name: dict[str, dict[str, Any]] = {}
    for row in rows:
        target_name = str(row.get("target_name", ""))
        key = parse_pe_native_target_lookup_key(target_name)
        if key is None:
            annotation: dict[str, Any] = {"policyengine_target_match": "unparsed"}
        elif resolved_db_path is None or not resolved_db_path.exists() or target_db_error:
            annotation = {
                "policyengine_target_match": "db_unavailable",
                "policyengine_target_expected": key.expected_target(),
            }
        else:
            matches = match_index.get(key.as_tuple(), [])
            if len(matches) == 1:
                match = matches[0]
                annotation = {
                    "policyengine_target_match": "matched",
                    "policyengine_target_id": match["target_id"],
                    "policyengine_target_variable": match["variable"],
                    "policyengine_target_period": match["period"],
                    "policyengine_target_value": match["value"],
                    "policyengine_target_source": match["source"],
                    "policyengine_target_domain_variable": match["domain_variable"],
                    "policyengine_target_constraints": match["constraints"],
                }
            elif len(matches) > 1:
                annotation = {
                    "policyengine_target_match": "ambiguous",
                    "policyengine_target_match_count": len(matches),
                    "policyengine_target_matches": matches,
                    "policyengine_target_expected": key.expected_target(),
                }
            else:
                annotation = {
                    "policyengine_target_match": "legacy_only",
                    "policyengine_target_expected": key.expected_target(),
                }
        counts[annotation["policyengine_target_match"]] += 1
        row.update(annotation)
        annotations_by_name[target_name] = annotation

    for list_name in ("top_improvements", "top_regressions"):
        for row in payload.get(list_name) or []:
            annotation = annotations_by_name.get(str(row.get("target_name", "")))
            if annotation:
                row.update(annotation)

    parsed_total = counts["matched"] + counts["legacy_only"] + counts["ambiguous"]
    payload["target_db_summary"] = {
        "target_db_path": str(resolved_db_path) if resolved_db_path else None,
        "target_db_error": target_db_error,
        **counts,
        "parsed_targets": parsed_total,
        "match_rate": counts["matched"] / parsed_total if parsed_total else None,
    }
    return payload


def compare_us_pe_native_target_deltas(
    *,
    from_dataset_path: str | Path,
    to_dataset_path: str | Path,
    period: int = 2024,
    top_k: int = 25,
    policyengine_us_data_repo: str | Path | None = None,
    policyengine_us_data_python: str | Path | None = None,
) -> dict[str, Any]:
    """Compare per-target PE-native weighted-loss terms between two datasets."""

    resolved_repo = resolve_policyengine_us_data_repo_root(policyengine_us_data_repo)
    env = build_policyengine_us_data_subprocess_env(resolved_repo)
    if policyengine_us_data_python is not None:
        command = [str(Path(policyengine_us_data_python).expanduser())]
    else:
        command = ["uv", "run", "--project", str(resolved_repo), "python"]
    completed = subprocess.run(
        [
            *command,
            "-c",
            _PE_NATIVE_TARGET_DELTA_SCRIPT,
            str(resolved_repo),
            json.dumps(_ENHANCED_CPS_BAD_TARGETS),
            str(int(period)),
            str(Path(from_dataset_path).expanduser().resolve()),
            str(Path(to_dataset_path).expanduser().resolve()),
            str(int(top_k)),
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
        raise RuntimeError(f"PE-native target delta comparison failed: {detail}")
    return json.loads(completed.stdout)


def compute_batch_us_pe_native_target_deltas(
    *,
    candidate_dataset_paths: list[str | Path] | tuple[str | Path, ...],
    baseline_dataset_path: str | Path,
    period: int = 2024,
    top_k: int = 25,
    policyengine_us_data_repo: str | Path | None = None,
    policyengine_us_data_python: str | Path | None = None,
) -> list[dict[str, Any]]:
    """Compare PE-native weighted-loss targets for many candidates against one baseline."""

    if not candidate_dataset_paths:
        return []
    resolved_repo = resolve_policyengine_us_data_repo_root(policyengine_us_data_repo)
    env = build_policyengine_us_data_subprocess_env(resolved_repo)
    if policyengine_us_data_python is not None:
        command = [str(Path(policyengine_us_data_python).expanduser())]
    else:
        command = ["uv", "run", "--project", str(resolved_repo), "python"]
    completed = subprocess.run(
        [
            *command,
            "-c",
            _PE_NATIVE_TARGET_DELTA_BATCH_SCRIPT,
            str(resolved_repo),
            json.dumps(_ENHANCED_CPS_BAD_TARGETS),
            str(int(period)),
            str(Path(baseline_dataset_path).expanduser().resolve()),
            json.dumps(
                [
                    str(Path(candidate_path).expanduser().resolve())
                    for candidate_path in candidate_dataset_paths
                ]
            ),
            str(int(top_k)),
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
        raise RuntimeError(f"PE-native batch target delta comparison failed: {detail}")
    return list(json.loads(completed.stdout))


def compute_us_pe_native_support_audit(
    *,
    candidate_dataset_path: str | Path,
    baseline_dataset_path: str | Path,
    period: int = 2024,
    policyengine_us_data_repo: str | Path | None = None,
    policyengine_us_data_python: str | Path | None = None,
) -> dict[str, Any]:
    """Compare candidate vs baseline structural support on selected PE surfaces."""

    resolved_repo = resolve_policyengine_us_data_repo_root(policyengine_us_data_repo)
    env = build_policyengine_us_data_subprocess_env(resolved_repo)
    if policyengine_us_data_python is not None:
        command = [str(Path(policyengine_us_data_python).expanduser())]
    else:
        command = ["uv", "run", "--project", str(resolved_repo), "python"]
    completed = subprocess.run(
        [
            *command,
            "-c",
            _PE_NATIVE_SUPPORT_AUDIT_SCRIPT,
            str(resolved_repo),
            str(int(period)),
            str(Path(candidate_dataset_path).expanduser().resolve()),
            str(Path(baseline_dataset_path).expanduser().resolve()),
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
        raise RuntimeError(f"PE-native support audit failed: {detail}")
    return json.loads(completed.stdout)


def compute_batch_us_pe_native_support_audits(
    *,
    candidate_dataset_paths: list[str | Path] | tuple[str | Path, ...],
    baseline_dataset_path: str | Path,
    period: int = 2024,
    policyengine_us_data_repo: str | Path | None = None,
    policyengine_us_data_python: str | Path | None = None,
) -> list[dict[str, Any]]:
    """Compare PE support structure for many candidates against one baseline."""

    if not candidate_dataset_paths:
        return []
    resolved_repo = resolve_policyengine_us_data_repo_root(policyengine_us_data_repo)
    env = build_policyengine_us_data_subprocess_env(resolved_repo)
    if policyengine_us_data_python is not None:
        command = [str(Path(policyengine_us_data_python).expanduser())]
    else:
        command = ["uv", "run", "--project", str(resolved_repo), "python"]
    completed = subprocess.run(
        [
            *command,
            "-c",
            _PE_NATIVE_SUPPORT_AUDIT_BATCH_SCRIPT,
            str(resolved_repo),
            str(int(period)),
            str(Path(baseline_dataset_path).expanduser().resolve()),
            json.dumps(
                [
                    str(Path(candidate_path).expanduser().resolve())
                    for candidate_path in candidate_dataset_paths
                ]
            ),
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
        raise RuntimeError(f"PE-native batch support audit failed: {detail}")

    payload = json.loads(completed.stdout)
    baseline_dataset = str(payload["baseline_dataset"])
    baseline_snapshot = payload["baseline"]
    period_value = int(payload["period"])
    return [
        {
            "metric": "enhanced_cps_support_audit",
            "period": period_value,
            "candidate_dataset": str(item["candidate_dataset"]),
            "baseline_dataset": baseline_dataset,
            "candidate": item["candidate"],
            "baseline": baseline_snapshot,
            "comparisons": item["comparisons"],
        }
        for item in payload.get("results", ())
    ]


def write_us_pe_native_scores(
    output_path: str | Path,
    *,
    candidate_dataset_path: str | Path,
    baseline_dataset_path: str | Path,
    period: int = 2024,
    policyengine_us_data_repo: str | Path | None = None,
    policyengine_us_data_python: str | Path | None = None,
) -> Path:
    """Write PE-native broad scoring payload to disk."""

    payload = compute_us_pe_native_scores(
        candidate_dataset_path=candidate_dataset_path,
        baseline_dataset_path=baseline_dataset_path,
        period=period,
        policyengine_us_data_repo=policyengine_us_data_repo,
        policyengine_us_data_python=policyengine_us_data_python,
    )
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False)
    )
    return destination


def write_us_pe_native_target_diagnostics(
    output_path: str | Path,
    *,
    from_dataset_path: str | Path,
    to_dataset_path: str | Path,
    period: int = 2024,
    top_k: int = 50,
    from_label: str = "policyengine-us-data",
    to_label: str = "microplex-us",
    policyengine_us_data_repo: str | Path | None = None,
    policyengine_us_data_python: str | Path | None = None,
    policyengine_targets_db_path: str | Path | None = None,
) -> Path:
    """Write the full PE-native per-target diagnostic dataset to disk."""

    payload = compare_us_pe_native_target_deltas(
        from_dataset_path=from_dataset_path,
        to_dataset_path=to_dataset_path,
        period=period,
        top_k=top_k,
        policyengine_us_data_repo=policyengine_us_data_repo,
        policyengine_us_data_python=policyengine_us_data_python,
    )
    payload["diagnostic_schema_version"] = 1
    payload["dataset_labels"] = {
        "from": from_label,
        "to": to_label,
    }
    target_db_path = (
        Path(policyengine_targets_db_path).expanduser()
        if policyengine_targets_db_path is not None
        else _default_policyengine_targets_db_path(policyengine_us_data_repo)
    )
    annotate_pe_native_target_db_matches(
        payload,
        target_db_path=target_db_path,
        period=period,
    )
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(payload, indent=2, sort_keys=True))
    return destination


def main(argv: list[str] | None = None) -> int:
    """CLI for exact broad PE-native loss scoring."""

    parser = argparse.ArgumentParser(
        description="Score a candidate and baseline under PE-US's enhanced-CPS native loss."
    )
    parser.add_argument("--candidate-dataset", required=True)
    parser.add_argument("--baseline-dataset", required=True)
    parser.add_argument("--period", type=int, default=2024)
    parser.add_argument("--policyengine-us-data-python")
    parser.add_argument("--policyengine-us-data-repo")
    args = parser.parse_args(argv)

    score = compute_policyengine_us_enhanced_cps_native_scores(
        args.candidate_dataset,
        args.baseline_dataset,
        period=args.period,
        policyengine_us_data_python=args.policyengine_us_data_python,
        policyengine_us_data_repo=args.policyengine_us_data_repo,
    )
    print(json.dumps(score.to_dict(), indent=2, sort_keys=True))
    return 0


def main_target_diagnostics(argv: list[str] | None = None) -> int:
    """CLI for full PE-native per-target diagnostics."""

    parser = argparse.ArgumentParser(
        description=(
            "Write a full per-target PE-native diagnostic JSON comparing a "
            "baseline dataset to a Microplex candidate."
        )
    )
    parser.add_argument("--from-dataset", required=True)
    parser.add_argument("--to-dataset", required=True)
    parser.add_argument("--output-path", required=True)
    parser.add_argument("--period", type=int, default=2024)
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--from-label", default="policyengine-us-data")
    parser.add_argument("--to-label", default="microplex-us")
    parser.add_argument("--policyengine-us-data-python")
    parser.add_argument("--policyengine-us-data-repo")
    parser.add_argument("--policyengine-targets-db")
    args = parser.parse_args(argv)

    path = write_us_pe_native_target_diagnostics(
        args.output_path,
        from_dataset_path=args.from_dataset,
        to_dataset_path=args.to_dataset,
        period=args.period,
        top_k=args.top_k,
        from_label=args.from_label,
        to_label=args.to_label,
        policyengine_us_data_python=args.policyengine_us_data_python,
        policyengine_us_data_repo=args.policyengine_us_data_repo,
        policyengine_targets_db_path=args.policyengine_targets_db,
    )
    print(str(path))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
