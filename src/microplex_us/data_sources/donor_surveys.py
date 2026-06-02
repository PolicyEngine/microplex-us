"""Spec-driven donor survey providers aligned with PE-US-data source-impute."""

from __future__ import annotations

import json
import pickle
import subprocess
import tempfile
from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path
from textwrap import dedent

import h5py
import numpy as np
import pandas as pd
from microplex.core import (
    EntityObservation,
    EntityRelationship,
    EntityType,
    ObservationFrame,
    RelationshipCardinality,
    Shareability,
    SourceArchetype,
    SourceDescriptor,
    SourceQuery,
    TimeStructure,
    apply_source_query,
)

from microplex_us.data_sources.sampling import (
    sample_frame_with_state_floor,
    sample_frame_without_replacement,
)
from microplex_us.pe_source_impute_specs import (
    PEPolicyengineDatasetLoaderSpec,
    PESourceImputeBlockSpec,
    apply_pe_source_impute_loader_postprocess,
    get_pe_source_impute_block_spec,
    resolve_sipp_source_impute_block_spec,
)
from microplex_us.pipelines.pe_native_scores import (
    build_policyengine_us_data_subprocess_env,
    resolve_policyengine_us_data_python,
    resolve_policyengine_us_data_repo_root,
)
from microplex_us.source_registry import resolve_source_variable_capabilities

try:
    from huggingface_hub import hf_hub_download

    HF_AVAILABLE = True
except ImportError:
    HF_AVAILABLE = False

PERSON_OBSERVATION_EXCLUDED_COLUMNS = (
    "person_id",
    "household_id",
    "weight",
    "year",
)
HOUSEHOLD_OBSERVATION_EXCLUDED_COLUMNS = (
    "household_id",
    "household_weight",
    "year",
)

DONOR_UPRATING_EXCLUDED_COLUMNS = {
    "person_id",
    "household_id",
    "weight",
    "year",
    "age",
    "sex",
    "is_female",
    "is_male",
    "is_married",
    "is_household_head",
    "cps_race",
    "state_fips",
    "tenure",
    "tenure_type",
    "own_children_in_household",
    "count_under_18",
    "count_under_6",
    "household_size",
}

DONOR_UPRATING_FACTOR_ALIASES = {
    "employment_income": "employment_income_before_lsr",
    "income": "employment_income_before_lsr",
    "interest_dividend_income": "taxable_interest_income",
    "social_security_pension_income": "social_security_retirement",
    "scf_certificates_of_deposit": "bank_account_assets",
    "scf_savings_bonds": "bond_assets",
    "scf_retirement_assets": "net_worth",
    "scf_cash_value_life_insurance": "net_worth",
    "scf_other_managed_assets": "stock_assets",
    "scf_other_financial_assets": "net_worth",
    "scf_primary_residence_value": "home_equity",
    "scf_other_residential_real_estate": "household_other_real_estate_value",
    "scf_nonresidential_real_estate_equity": "household_other_real_estate_equity",
    "scf_business_equity": "household_business_assets_equity",
    "scf_other_nonfinancial_assets": "net_worth",
    "scf_mortgage_debt": "first_home_mortgage_balance",
    "scf_other_residential_debt": "household_other_real_estate_debt",
    "scf_other_lines_of_credit": "household_vehicles_debt",
    "scf_credit_card_debt": "household_vehicles_debt",
    "scf_vehicle_installment_debt": "household_vehicles_debt",
    "scf_student_loan_debt": "household_vehicles_debt",
    "scf_other_installment_debt": "household_vehicles_debt",
    "scf_other_debt": "household_vehicles_debt",
}

TARGET_YEAR_UPRATED_SURVEYS = {"sipp", "scf"}


@dataclass(frozen=True)
class DonorSurveyTables:
    """Canonical household/person tables for one donor survey block."""

    households: pd.DataFrame
    persons: pd.DataFrame


DonorSurveyTablesLoader = Callable[..., DonorSurveyTables]


def _descriptor_from_tables(
    *,
    households: pd.DataFrame,
    persons: pd.DataFrame,
    name: str,
    shareability: Shareability,
    archetype: SourceArchetype | None,
) -> SourceDescriptor:
    household_variables = tuple(
        column
        for column in households.columns
        if column not in HOUSEHOLD_OBSERVATION_EXCLUDED_COLUMNS
    )
    person_variables = tuple(
        column
        for column in persons.columns
        if column not in PERSON_OBSERVATION_EXCLUDED_COLUMNS
    )
    return SourceDescriptor(
        name=name,
        shareability=shareability,
        time_structure=TimeStructure.REPEATED_CROSS_SECTION,
        archetype=archetype,
        observations=(
            EntityObservation(
                entity=EntityType.HOUSEHOLD,
                key_column="household_id",
                variable_names=household_variables,
                weight_column="household_weight"
                if "household_weight" in households.columns
                else None,
                period_column="year" if "year" in households.columns else None,
            ),
            EntityObservation(
                entity=EntityType.PERSON,
                key_column="person_id",
                variable_names=person_variables,
                weight_column="weight" if "weight" in persons.columns else None,
                period_column="year" if "year" in persons.columns else None,
            ),
        ),
        variable_capabilities=resolve_source_variable_capabilities(
            name,
            (*household_variables, *person_variables),
        ),
    )


def _build_static_descriptor(
    *,
    spec: PESourceImputeBlockSpec,
    shareability: Shareability,
) -> SourceDescriptor:
    return SourceDescriptor(
        name=spec.descriptor_name,
        shareability=shareability,
        time_structure=TimeStructure.REPEATED_CROSS_SECTION,
        archetype=spec.archetype,
        observations=(
            EntityObservation(
                entity=EntityType.HOUSEHOLD,
                key_column="household_id",
                variable_names=spec.household_variables,
                weight_column="household_weight",
                period_column="year",
            ),
            EntityObservation(
                entity=EntityType.PERSON,
                key_column="person_id",
                variable_names=spec.person_variables,
                weight_column="weight",
                period_column="year",
            ),
        ),
    )


def _ensure_person_ids(persons: pd.DataFrame) -> pd.DataFrame:
    result = persons.copy()
    if "person_id" not in result.columns:
        if "household_id" in result.columns:
            result["person_id"] = (
                result["household_id"].astype(str)
                + ":"
                + result.groupby("household_id").cumcount().add(1).astype(str)
            )
            return result
        result["person_id"] = np.arange(len(result)).astype(str)
        return result

    if not result["person_id"].duplicated().any():
        return result

    if "household_id" in result.columns:
        composite = (
            result["household_id"].astype(str) + ":" + result["person_id"].astype(str)
        )
        if not composite.duplicated().any():
            result["person_id"] = composite
            return result
        result["person_id"] = (
            result["household_id"].astype(str)
            + ":"
            + result.groupby("household_id").cumcount().add(1).astype(str)
        )
        return result

    result["person_id"] = np.arange(len(result)).astype(str)
    return result


def _sample_households_and_persons(
    *,
    households: pd.DataFrame,
    persons: pd.DataFrame,
    sample_n: int | None,
    random_seed: int,
    state_floor: int | None = None,
    state_age_floor: int | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    households = households.reset_index(drop=True)
    persons = persons.reset_index(drop=True)
    if sample_n is None or sample_n >= len(households):
        return households, persons
    sampled_households = _sample_donor_households(
        households=households,
        persons=persons,
        sample_n=sample_n,
        random_seed=random_seed,
        state_floor=state_floor,
        state_age_floor=state_age_floor,
    )
    keep = set(sampled_households["household_id"])
    sampled_persons = persons[persons["household_id"].isin(keep)].copy()
    return (
        sampled_households.sort_values(["household_id"]).reset_index(drop=True),
        sampled_persons.sort_values(["household_id", "person_id"]).reset_index(
            drop=True
        ),
    )


def _sample_donor_households(
    *,
    households: pd.DataFrame,
    persons: pd.DataFrame,
    sample_n: int | None,
    random_seed: int,
    state_floor: int | None = None,
    state_age_floor: int | None = None,
) -> pd.DataFrame:
    resolved_state_age_floor = int(state_age_floor or 0)
    if (
        resolved_state_age_floor <= 0
        or "state_fips" not in households.columns
        or "age" not in persons.columns
        or "household_id" not in households.columns
        or "household_id" not in persons.columns
    ):
        return sample_frame_with_state_floor(
            households,
            sample_n=sample_n,
            random_seed=random_seed,
            weight_col="household_weight",
            state_floor=state_floor,
            positive_only_when_weighted=True,
        )

    coverage = persons[["household_id", "age"]].merge(
        households[["household_id", "state_fips"]],
        on="household_id",
        how="inner",
    )
    coverage["age_band"] = coverage["age"].map(_donor_age_band_key)
    coverage["state_fips"] = pd.to_numeric(
        coverage["state_fips"], errors="coerce"
    ).astype("Int64")
    coverage = coverage.dropna(subset=["state_fips", "age_band"]).copy()
    if coverage.empty:
        return sample_frame_with_state_floor(
            households,
            sample_n=sample_n,
            random_seed=random_seed,
            weight_col="household_weight",
            state_floor=state_floor,
            positive_only_when_weighted=True,
        )

    rng = np.random.default_rng(random_seed)
    selected_ids: set[str | int] = set()
    for _, group in coverage.groupby(["state_fips", "age_band"], sort=True):
        group_household_ids = pd.Index(group["household_id"].unique())
        already_selected = [hid for hid in group_household_ids if hid in selected_ids]
        missing = resolved_state_age_floor - len(already_selected)
        if missing <= 0:
            continue
        available_ids = [hid for hid in group_household_ids if hid not in selected_ids]
        if not available_ids:
            continue
        candidate_households = households[
            households["household_id"].isin(available_ids)
        ].copy()
        sampled = sample_frame_without_replacement(
            candidate_households,
            sample_n=min(missing, len(candidate_households)),
            random_seed=int(rng.integers(0, np.iinfo(np.int32).max)),
            weight_col="household_weight",
            positive_only_when_weighted=True,
        )
        selected_ids.update(sampled["household_id"].tolist())

    if sample_n is not None and len(selected_ids) > sample_n:
        raise ValueError(
            "state_age_floor requires more sampled donor households than sample_n allows: "
            f"selected={len(selected_ids)}, sample_n={sample_n}"
        )

    if not selected_ids:
        return sample_frame_with_state_floor(
            households,
            sample_n=sample_n,
            random_seed=random_seed,
            weight_col="household_weight",
            state_floor=state_floor,
            positive_only_when_weighted=True,
        )

    selected = households[households["household_id"].isin(selected_ids)].copy()
    remaining_n = int(sample_n) - len(selected)
    if remaining_n <= 0:
        return selected

    remainder = households[~households["household_id"].isin(selected_ids)].copy()
    remainder_sample = sample_frame_without_replacement(
        remainder,
        sample_n=remaining_n,
        random_seed=int(rng.integers(0, np.iinfo(np.int32).max)),
        weight_col="household_weight",
        positive_only_when_weighted=True,
    )
    return pd.concat([selected, remainder_sample], axis=0, ignore_index=False)


def _donor_age_band_key(age: float | int | None) -> str | None:
    value = pd.to_numeric(pd.Series([age]), errors="coerce").iloc[0]
    if pd.isna(value):
        return None
    age_int = int(value)
    if age_int < 0:
        return None
    if age_int >= 85:
        return "85_plus"
    lower = (age_int // 5) * 5
    upper = lower + 5
    return f"{lower}_{upper}"


def _build_observation_frame(
    *,
    households: pd.DataFrame,
    persons: pd.DataFrame,
    source_name: str,
    shareability: Shareability,
    archetype: SourceArchetype | None,
) -> ObservationFrame:
    normalized_households = households.copy()
    normalized_persons = _ensure_person_ids(persons)
    descriptor = _descriptor_from_tables(
        households=normalized_households,
        persons=normalized_persons,
        name=source_name,
        shareability=shareability,
        archetype=archetype,
    )
    frame = ObservationFrame(
        source=descriptor,
        tables={
            EntityType.HOUSEHOLD: normalized_households,
            EntityType.PERSON: normalized_persons,
        },
        relationships=(
            EntityRelationship(
                parent_entity=EntityType.HOUSEHOLD,
                child_entity=EntityType.PERSON,
                parent_key="household_id",
                child_key="household_id",
                cardinality=RelationshipCardinality.ONE_TO_MANY,
            ),
        ),
    )
    frame.validate()
    return frame


def _build_policyengine_dataset_loader_script(
    spec: PEPolicyengineDatasetLoaderSpec,
    *,
    year: int,
) -> str:
    payload = json.dumps(
        {
            "year": int(year),
            "dataset_loader": asdict(spec),
        }
    )
    return dedent(
        f"""
import importlib
import json
import pickle
import sys
import numpy as np
import pandas as pd

payload = json.loads({payload!r})
spec = payload["dataset_loader"]
out_path = sys.argv[1]
sample_n = None if sys.argv[2] == "None" else int(sys.argv[2])
random_seed = int(sys.argv[3])

module = importlib.import_module(spec["module"])
dataset_cls = getattr(module, spec["class_name"])
data = dataset_cls().load_dataset()

def _numeric(values):
    return pd.to_numeric(pd.Series(np.asarray(values)), errors="coerce").fillna(0.0)

def _boolean_float(values):
    return pd.Series(np.asarray(values)).astype(bool).astype(float)

def _text(values):
    return pd.Series(np.asarray(values)).map(
        lambda value: value.decode() if isinstance(value, (bytes, bytearray)) else str(value)
    )

def _mapped_text(values, mapping):
    return _text(values).map(mapping).fillna(0).astype(int)

def _load_fallback(keys):
    for key in keys:
        if key in data:
            return pd.Series(np.asarray(data[key]))
    raise KeyError(f"Missing fallback keys {{keys}} in dataset payload")

def _build_persons():
    if spec["builder_kind"] == "household_rows":
        household_index = pd.Index(data[spec["household_index_key"]])
        person_households = pd.Index(data[spec["person_household_key"]])
        household_to_row = pd.Series(
            np.arange(len(household_index), dtype=np.int64),
            index=household_index,
        )
        household_rows = household_to_row.loc[person_households].to_numpy()
        persons = pd.DataFrame({{"household_id": person_households.to_numpy()}})
        if spec["person_id_key"] is not None:
            persons["person_id"] = np.asarray(data[spec["person_id_key"]])
        for target, source in spec["direct_person_columns"].items():
            persons[target] = _numeric(data[source])
        for target, source in spec["boolean_person_columns"].items():
            persons[target] = _boolean_float(data[source])
        for target, source in spec["row_indexed_person_columns"].items():
            persons[target] = _numeric(np.asarray(data[source])[household_rows])
        for target, source in spec["mapped_row_person_columns"].items():
            persons[target] = _mapped_text(
                np.asarray(data[source])[household_rows],
                spec["mapped_value_tables"][target],
            )
    elif spec["builder_kind"] == "single_person_households":
        base_length = len(data[spec["length_source_key"]])
        if spec["generated_household_ids"]:
            household_ids = np.arange(base_length, dtype=np.int64) + 1
        else:
            household_ids = np.asarray(data[spec["household_index_key"]])
        persons = pd.DataFrame({{"household_id": household_ids}})
        if spec["person_id_from_household_id"]:
            persons["person_id"] = persons["household_id"]
        elif spec["person_id_key"] is not None:
            persons["person_id"] = np.asarray(data[spec["person_id_key"]])
        for target, source in spec["direct_person_columns"].items():
            persons[target] = _numeric(data[source])
        for target, source in spec["boolean_person_columns"].items():
            persons[target] = _boolean_float(data[source])
    else:
        raise ValueError(f"Unsupported dataset loader builder kind: {{spec['builder_kind']}}")

    for target, keys in spec["fallback_person_columns"].items():
        persons[target] = _numeric(_load_fallback(keys))
    if spec["sex_from_boolean_source"] is not None:
        source = spec["sex_from_boolean_source"]
        source_values = pd.Series(persons[source]).astype(bool).to_numpy()
        persons["sex"] = np.where(
            source_values,
            spec["sex_true_value"],
            spec["sex_false_value"],
        )
    for target, source in spec["copy_person_columns"].items():
        persons[target] = persons[source]
    for target, value in spec["constant_person_columns"].items():
        persons[target] = value
    if spec["income_sum_columns"]:
        persons["income"] = sum(
            _numeric(persons[column]) for column in spec["income_sum_columns"]
        )
    for column in spec["int_person_columns"]:
        if column in persons.columns:
            persons[column] = (
                pd.to_numeric(persons[column], errors="coerce")
                .fillna(0)
                .astype(int)
            )
    persons["year"] = int(payload["year"])
    return persons

persons = _build_persons()
households = (
    persons[
        ["household_id", "state_fips", "tenure", "weight", "year"]
    ]
    .rename(columns={{"weight": "household_weight"}})
    .drop_duplicates(subset=["household_id"])
)

if sample_n is not None and sample_n < len(households):
    sampled = households.sample(
        n=sample_n,
        random_state=random_seed,
        replace=False,
        weights=households["household_weight"],
    ).copy()
    keep = set(sampled["household_id"])
    households = sampled.sort_values(["household_id"]).reset_index(drop=True)
    persons = (
        persons[persons["household_id"].isin(keep)]
        .sort_values(["household_id", "person_id"])
        .reset_index(drop=True)
    )
else:
    households = households.sort_values(["household_id"]).reset_index(drop=True)
    persons = persons.sort_values(["household_id", "person_id"]).reset_index(drop=True)

with open(out_path, "wb") as handle:
    pickle.dump({{"households": households, "persons": persons}}, handle)
"""
    )


def _decode_h5_values(values: np.ndarray) -> np.ndarray:
    """Decode fixed-width HDF5 byte strings to ordinary Python strings."""
    if values.dtype.kind not in {"S", "O"}:
        return values
    return np.asarray(
        [
            value.decode() if isinstance(value, (bytes, bytearray)) else value
            for value in values
        ]
    )


def _load_policyengine_us_data_h5_dataset(
    *,
    filename: str,
    policyengine_us_data_repo: str | Path | None,
    cache_dir: Path | None,
) -> dict[str, np.ndarray]:
    if policyengine_us_data_repo is None:
        h5_path = _download_policyengine_us_data_file(
            filename=filename,
            cache_dir=cache_dir,
        )
    else:
        repo_root = resolve_policyengine_us_data_repo_root(policyengine_us_data_repo)
        h5_path = repo_root / "policyengine_us_data" / "storage" / filename
    if not h5_path.exists():
        raise FileNotFoundError(f"Missing PolicyEngine US-data H5 file: {h5_path}")
    with h5py.File(h5_path, "r") as h5:
        return {key: _decode_h5_values(np.asarray(h5[key])) for key in h5.keys()}


def _build_policyengine_dataset_tables_from_arrays(
    *,
    data: dict[str, np.ndarray],
    dataset_loader: PEPolicyengineDatasetLoaderSpec,
    year: int,
) -> DonorSurveyTables:
    spec = asdict(dataset_loader)

    def _numeric(values):
        return pd.to_numeric(pd.Series(np.asarray(values)), errors="coerce").fillna(0.0)

    def _boolean_float(values):
        return pd.Series(np.asarray(values)).astype(bool).astype(float)

    def _text(values):
        return pd.Series(_decode_h5_values(np.asarray(values))).astype(str)

    def _mapped_text(values, mapping):
        return _text(values).map(mapping).fillna(0).astype(int)

    def _load_fallback(keys):
        for key in keys:
            if key in data:
                return pd.Series(np.asarray(data[key]))
        raise KeyError(f"Missing fallback keys {keys} in dataset payload")

    if spec["builder_kind"] == "household_rows":
        household_index = pd.Index(data[spec["household_index_key"]])
        person_households = pd.Index(data[spec["person_household_key"]])
        household_to_row = pd.Series(
            np.arange(len(household_index), dtype=np.int64),
            index=household_index,
        )
        household_rows = household_to_row.loc[person_households].to_numpy()
        persons = pd.DataFrame({"household_id": person_households.to_numpy()})
        if spec["person_id_key"] is not None:
            persons["person_id"] = np.asarray(data[spec["person_id_key"]])
        for target, source in spec["direct_person_columns"].items():
            persons[target] = _numeric(data[source])
        for target, source in spec["boolean_person_columns"].items():
            persons[target] = _boolean_float(data[source])
        for target, source in spec["row_indexed_person_columns"].items():
            persons[target] = _numeric(np.asarray(data[source])[household_rows])
        for target, source in spec["mapped_row_person_columns"].items():
            persons[target] = _mapped_text(
                np.asarray(data[source])[household_rows],
                spec["mapped_value_tables"][target],
            )
    elif spec["builder_kind"] == "single_person_households":
        base_length = len(data[spec["length_source_key"]])
        if spec["generated_household_ids"]:
            household_ids = np.arange(base_length, dtype=np.int64) + 1
        else:
            household_ids = np.asarray(data[spec["household_index_key"]])
        persons = pd.DataFrame({"household_id": household_ids})
        if spec["person_id_from_household_id"]:
            persons["person_id"] = persons["household_id"]
        elif spec["person_id_key"] is not None:
            persons["person_id"] = np.asarray(data[spec["person_id_key"]])
        for target, source in spec["direct_person_columns"].items():
            persons[target] = _numeric(data[source])
        for target, source in spec["boolean_person_columns"].items():
            persons[target] = _boolean_float(data[source])
    else:
        raise ValueError(
            f"Unsupported dataset loader builder kind: {spec['builder_kind']}"
        )

    for target, keys in spec["fallback_person_columns"].items():
        persons[target] = _numeric(_load_fallback(keys))
    if spec["sex_from_boolean_source"] is not None:
        source = spec["sex_from_boolean_source"]
        source_values = pd.Series(persons[source]).astype(bool).to_numpy()
        persons["sex"] = np.where(
            source_values,
            spec["sex_true_value"],
            spec["sex_false_value"],
        )
    for target, source in spec["copy_person_columns"].items():
        persons[target] = persons[source]
    for target, value in spec["constant_person_columns"].items():
        persons[target] = value
    if spec["income_sum_columns"]:
        persons["income"] = sum(
            _numeric(persons[column]) for column in spec["income_sum_columns"]
        )
    for column in spec["int_person_columns"]:
        if column in persons.columns:
            persons[column] = (
                pd.to_numeric(persons[column], errors="coerce").fillna(0).astype(int)
            )
    persons["year"] = int(year)
    households = (
        persons[["household_id", "state_fips", "tenure", "weight", "year"]]
        .rename(columns={"weight": "household_weight"})
        .drop_duplicates(subset=["household_id"])
    )
    return DonorSurveyTables(
        households=households.sort_values(["household_id"]).reset_index(drop=True),
        persons=persons.sort_values(["household_id", "person_id"]).reset_index(
            drop=True
        ),
    )


def _run_policyengine_dataset_loader(
    *,
    script: str,
    sample_n: int | None,
    random_seed: int,
    policyengine_us_data_repo: str | Path | None = None,
    policyengine_us_data_python: str | Path | None = None,
) -> DonorSurveyTables:
    resolved_repo = resolve_policyengine_us_data_repo_root(policyengine_us_data_repo)
    resolved_python = resolve_policyengine_us_data_python(
        policyengine_us_data_python,
        repo_root=resolved_repo,
    )
    env = build_policyengine_us_data_subprocess_env(resolved_repo)
    with tempfile.TemporaryDirectory(prefix="microplex-us-donor-") as tempdir:
        payload_path = Path(tempdir) / "tables.pkl"
        subprocess.run(
            [
                str(resolved_python),
                "-c",
                script,
                str(payload_path),
                "None" if sample_n is None else str(int(sample_n)),
                str(int(random_seed)),
            ],
            check=True,
            cwd=resolved_repo,
            env=env,
        )
        with payload_path.open("rb") as handle:
            payload = pickle.load(handle)
    return DonorSurveyTables(
        households=payload["households"],
        persons=payload["persons"],
    )


def _run_policyengine_dataset_loader_from_spec(
    *,
    spec: PESourceImputeBlockSpec,
    year: int,
    sample_n: int | None,
    random_seed: int,
    policyengine_us_data_repo: str | Path | None = None,
    policyengine_us_data_python: str | Path | None = None,
) -> DonorSurveyTables:
    dataset_loader = spec.dataset_loader
    if dataset_loader is None:
        raise ValueError(
            f"PE source-impute block '{spec.key}' is missing a dataset loader spec"
        )
    return _run_policyengine_dataset_loader(
        script=_build_policyengine_dataset_loader_script(dataset_loader, year=year),
        sample_n=sample_n,
        random_seed=random_seed,
        policyengine_us_data_repo=policyengine_us_data_repo,
        policyengine_us_data_python=policyengine_us_data_python,
    )


def _default_acs_tables_loader(
    *,
    year: int,
    sample_n: int | None,
    random_seed: int,
    state_floor: int | None = None,
    state_age_floor: int | None = None,
    cache_dir: Path | None = None,
    policyengine_us_data_repo: str | Path | None = None,
    policyengine_us_data_python: str | Path | None = None,
) -> DonorSurveyTables:
    spec = get_pe_source_impute_block_spec("acs")
    if int(year) != spec.default_year and policyengine_us_data_repo is None:
        raise ValueError(
            f"{spec.descriptor_name} provider supports non-default years only "
            "when policyengine_us_data_repo is provided"
        )
    if int(year) == spec.default_year:
        tables = _run_policyengine_dataset_loader_from_spec(
            spec=spec,
            year=year,
            sample_n=None if state_floor else sample_n,
            random_seed=random_seed,
            policyengine_us_data_repo=policyengine_us_data_repo,
            policyengine_us_data_python=policyengine_us_data_python,
        )
    else:
        dataset_loader = spec.dataset_loader
        if dataset_loader is None:
            raise ValueError(
                f"PE source-impute block '{spec.key}' is missing a dataset loader spec"
            )
        data = _load_policyengine_us_data_h5_dataset(
            filename=f"acs_{int(year)}.h5",
            policyengine_us_data_repo=policyengine_us_data_repo,
            cache_dir=cache_dir,
        )
        tables = _build_policyengine_dataset_tables_from_arrays(
            data=data,
            dataset_loader=dataset_loader,
            year=year,
        )
    households = (
        tables.households.drop_duplicates(subset=["household_id"])
        .sort_values(["household_id"])
        .reset_index(drop=True)
    )
    persons = (
        tables.persons[
            tables.persons["household_id"].isin(set(households["household_id"]))
        ]
        .sort_values(["household_id", "person_id"])
        .reset_index(drop=True)
    )
    households, persons = _sample_households_and_persons(
        households=households,
        persons=persons,
        sample_n=sample_n,
        random_seed=random_seed,
        state_floor=state_floor,
        state_age_floor=state_age_floor,
    )
    return DonorSurveyTables(households=households, persons=persons)


def _default_scf_tables_loader(
    *,
    year: int,
    sample_n: int | None,
    random_seed: int,
    state_floor: int | None = None,
    state_age_floor: int | None = None,
    cache_dir: Path | None = None,
    policyengine_us_data_repo: str | Path | None = None,
    policyengine_us_data_python: str | Path | None = None,
) -> DonorSurveyTables:
    _ = cache_dir
    spec = get_pe_source_impute_block_spec("scf")
    if int(year) != spec.default_year:
        raise ValueError(
            f"{spec.descriptor_name} provider currently supports year={spec.default_year} only"
        )
    tables = _run_policyengine_dataset_loader_from_spec(
        spec=spec,
        year=year,
        sample_n=None if state_floor else sample_n,
        random_seed=random_seed,
        policyengine_us_data_repo=policyengine_us_data_repo,
        policyengine_us_data_python=policyengine_us_data_python,
    )
    households, persons = _sample_households_and_persons(
        households=tables.households,
        persons=tables.persons,
        sample_n=sample_n,
        random_seed=random_seed,
        state_floor=state_floor,
        state_age_floor=state_age_floor,
    )
    return DonorSurveyTables(households=households, persons=persons)


def _download_policyengine_us_data_file(
    *,
    filename: str,
    cache_dir: Path | None,
) -> Path:
    if cache_dir is None:
        cache_dir = Path.home() / ".cache" / "microplex"
    cache_dir.mkdir(parents=True, exist_ok=True)
    destination = cache_dir / filename
    if destination.exists():
        return destination
    if not HF_AVAILABLE:
        raise ImportError("huggingface_hub required: pip install huggingface_hub")
    downloaded = hf_hub_download(
        repo_id="PolicyEngine/policyengine-us-data",
        filename=filename,
        repo_type="model",
        local_dir=cache_dir,
    )
    return Path(downloaded)


def _load_policyengine_uprating_factors(
    *,
    policyengine_us_data_repo: str | Path | None,
    cache_dir: Path | None,
) -> pd.DataFrame:
    if policyengine_us_data_repo is None:
        factors_path = _download_policyengine_us_data_file(
            filename="uprating_factors.csv",
            cache_dir=cache_dir,
        )
    else:
        repo_root = resolve_policyengine_us_data_repo_root(policyengine_us_data_repo)
        factors_path = (
            repo_root / "policyengine_us_data" / "storage" / "uprating_factors.csv"
        )
    if not factors_path.exists():
        raise FileNotFoundError(
            f"Missing PolicyEngine US-data uprating factors: {factors_path}"
        )
    return pd.read_csv(factors_path).set_index("Variable")


def _uprate_donor_tables_to_target_year(
    tables: DonorSurveyTables,
    *,
    spec: PESourceImputeBlockSpec,
    source_year: int,
    target_year: int | None,
    policyengine_us_data_repo: str | Path | None,
    cache_dir: str | Path | None,
) -> DonorSurveyTables:
    if (
        target_year is None
        or int(target_year) == int(source_year)
        or spec.survey_name not in TARGET_YEAR_UPRATED_SURVEYS
    ):
        return tables

    resolved_cache_dir = None if cache_dir is None else Path(cache_dir)
    factors = _load_policyengine_uprating_factors(
        policyengine_us_data_repo=policyengine_us_data_repo,
        cache_dir=resolved_cache_dir,
    )
    start_column = str(int(source_year))
    end_column = str(int(target_year))
    if start_column not in factors.columns or end_column not in factors.columns:
        raise ValueError(
            "PolicyEngine US-data uprating factors do not cover "
            f"{source_year}->{target_year}"
        )

    persons = tables.persons.copy()
    for column in persons.columns:
        if column in DONOR_UPRATING_EXCLUDED_COLUMNS:
            continue
        factor_name = DONOR_UPRATING_FACTOR_ALIASES.get(column, column)
        if factor_name not in factors.index:
            continue
        start = float(factors.loc[factor_name, start_column])
        end = float(factors.loc[factor_name, end_column])
        if start == 0:
            raise ValueError(f"Zero uprating base for {factor_name} in {source_year}")
        persons[column] = pd.to_numeric(persons[column], errors="coerce").fillna(
            0.0
        ) * (end / start)

    if "year" in persons.columns:
        persons["year"] = int(target_year)
    households = tables.households.copy()
    if "year" in households.columns:
        households["year"] = int(target_year)
    return DonorSurveyTables(households=households, persons=persons)


def _build_joined_raw_identifier(
    frame: pd.DataFrame,
    *,
    parts: tuple[str, ...],
) -> pd.Series:
    if not parts:
        raise ValueError("Raw identifier spec must include at least one part")
    values = frame.loc[:, list(parts)].astype(str)
    return values.iloc[:, 0] if len(parts) == 1 else values.agg(":".join, axis=1)


def _load_sipp_tables_from_spec(
    *,
    spec: PESourceImputeBlockSpec,
    year: int,
    sample_n: int | None,
    random_seed: int,
    state_floor: int | None = None,
    state_age_floor: int | None = None,
    cache_dir: Path | None,
) -> DonorSurveyTables:
    raw_loader = spec.raw_loader
    if raw_loader is None:
        raise ValueError(
            f"PE source-impute block '{spec.key}' is missing a raw loader spec"
        )
    if int(year) != spec.default_year:
        raise ValueError(
            f"{spec.descriptor_name} provider currently supports year={spec.default_year} only"
        )
    sipp_path = _download_policyengine_us_data_file(
        filename=raw_loader.filename,
        cache_dir=cache_dir,
    )
    read_csv_kwargs: dict[str, object] = {}
    if raw_loader.delimiter is not None:
        read_csv_kwargs["delimiter"] = raw_loader.delimiter
    if raw_loader.usecols:
        read_csv_kwargs["usecols"] = raw_loader.usecols
    df = pd.read_csv(sipp_path, **read_csv_kwargs)

    for variable, source_column in raw_loader.direct_columns.items():
        values = pd.to_numeric(df[source_column], errors="coerce").fillna(0.0)
        if variable in set(raw_loader.int_columns):
            df[variable] = values.astype(int)
        else:
            df[variable] = values.astype(float)
    for variable, contains in raw_loader.sum_columns_contains.items():
        matched_columns = [column for column in df.columns if contains in column]
        df[variable] = (
            df[matched_columns].fillna(0).sum(axis=1) if matched_columns else 0.0
        )
    for variable, indicator in raw_loader.indicator_columns.items():
        raw_values = pd.to_numeric(df[indicator.column], errors="coerce").fillna(0.0)
        df[variable] = raw_values.eq(indicator.equals).astype(float)
    for variable, value in raw_loader.constant_columns.items():
        df[variable] = value

    df["year"] = int(year)
    df["household_id"] = _build_joined_raw_identifier(
        df,
        parts=raw_loader.household_id_parts,
    )
    df["person_id"] = _build_joined_raw_identifier(
        df,
        parts=raw_loader.person_id_parts,
    )
    for variable, source_variable in raw_loader.copy_columns.items():
        df[variable] = df[source_variable]

    df = apply_pe_source_impute_loader_postprocess(df, spec)
    for variable, source_variable in raw_loader.copy_columns.items():
        if source_variable in df.columns:
            df[variable] = df[source_variable]
    households = (
        df[["household_id", "weight", "state_fips", "tenure", "year"]]
        .rename(columns={"weight": "household_weight"})
        .drop_duplicates(subset=["household_id"])
        .reset_index(drop=True)
    )
    persons = df[
        [
            "person_id",
            "household_id",
            *spec.person_variables,
            "weight",
            "year",
        ]
    ].copy()
    households, persons = _sample_households_and_persons(
        households=households,
        persons=persons,
        sample_n=sample_n,
        random_seed=random_seed,
        state_floor=state_floor,
        state_age_floor=state_age_floor,
    )
    return DonorSurveyTables(households=households, persons=persons)


def _default_sipp_tips_tables_loader(
    *,
    year: int,
    sample_n: int | None,
    random_seed: int,
    state_floor: int | None = None,
    state_age_floor: int | None = None,
    cache_dir: Path | None = None,
    policyengine_us_data_repo: str | Path | None = None,
    policyengine_us_data_python: str | Path | None = None,
) -> DonorSurveyTables:
    _ = policyengine_us_data_repo, policyengine_us_data_python
    return _load_sipp_tables_from_spec(
        spec=get_pe_source_impute_block_spec("sipp_tips"),
        year=year,
        sample_n=sample_n,
        random_seed=random_seed,
        state_floor=state_floor,
        state_age_floor=state_age_floor,
        cache_dir=cache_dir,
    )


def _default_sipp_assets_tables_loader(
    *,
    year: int,
    sample_n: int | None,
    random_seed: int,
    state_floor: int | None = None,
    state_age_floor: int | None = None,
    cache_dir: Path | None = None,
    policyengine_us_data_repo: str | Path | None = None,
    policyengine_us_data_python: str | Path | None = None,
) -> DonorSurveyTables:
    _ = policyengine_us_data_repo, policyengine_us_data_python
    return _load_sipp_tables_from_spec(
        spec=get_pe_source_impute_block_spec("sipp_assets"),
        year=year,
        sample_n=sample_n,
        random_seed=random_seed,
        state_floor=state_floor,
        state_age_floor=state_age_floor,
        cache_dir=cache_dir,
    )


BLOCK_LOADERS: dict[str, DonorSurveyTablesLoader] = {
    "acs": _default_acs_tables_loader,
    "sipp_tips": _default_sipp_tips_tables_loader,
    "sipp_assets": _default_sipp_assets_tables_loader,
    "scf": _default_scf_tables_loader,
}


DonorSurveyProviderSpec = PESourceImputeBlockSpec


def _default_loader_for_spec(spec: PESourceImputeBlockSpec) -> DonorSurveyTablesLoader:
    return BLOCK_LOADERS[spec.key]


def resolve_sipp_donor_survey_spec(block: str) -> DonorSurveyProviderSpec:
    return resolve_sipp_source_impute_block_spec(block)


class DonorSurveySourceProvider:
    """Generic source provider for one donor survey block."""

    def __init__(
        self,
        *,
        spec: DonorSurveyProviderSpec,
        year: int | None = None,
        cache_dir: str | Path | None = None,
        shareability: Shareability = Shareability.PUBLIC,
        loader: DonorSurveyTablesLoader | None = None,
        policyengine_us_data_repo: str | Path | None = None,
        policyengine_us_data_python: str | Path | None = None,
        target_year: int | None = None,
    ) -> None:
        self.spec = spec
        self.year = int(spec.default_year if year is None else year)
        self.target_year = None if target_year is None else int(target_year)
        self.cache_dir = None if cache_dir is None else Path(cache_dir)
        self.shareability = shareability
        self.loader = loader
        self.policyengine_us_data_repo = policyengine_us_data_repo
        self.policyengine_us_data_python = policyengine_us_data_python
        self._descriptor_cache: SourceDescriptor | None = None

    @property
    def descriptor(self) -> SourceDescriptor:
        if self._descriptor_cache is not None:
            return self._descriptor_cache
        return _build_static_descriptor(
            spec=self.spec,
            shareability=self.shareability,
        )

    def load_frame(self, query: SourceQuery | None = None) -> ObservationFrame:
        query = query or SourceQuery()
        provider_filters = query.provider_filters
        loader = self.loader or _default_loader_for_spec(self.spec)
        year = int(provider_filters.get("year", self.year))
        target_year = provider_filters.get("target_year", self.target_year)
        resolved_target_year = None if target_year is None else int(target_year)
        cache_dir = provider_filters.get("cache_dir", self.cache_dir)
        policyengine_us_data_repo = provider_filters.get(
            "policyengine_us_data_repo",
            self.policyengine_us_data_repo,
        )
        tables = loader(
            year=year,
            sample_n=provider_filters.get("sample_n"),
            random_seed=int(provider_filters.get("random_seed", 0)),
            state_floor=provider_filters.get("state_floor"),
            state_age_floor=provider_filters.get("state_age_floor"),
            cache_dir=cache_dir,
            policyengine_us_data_repo=policyengine_us_data_repo,
            policyengine_us_data_python=provider_filters.get(
                "policyengine_us_data_python",
                self.policyengine_us_data_python,
            ),
        )
        tables = _uprate_donor_tables_to_target_year(
            tables,
            spec=self.spec,
            source_year=year,
            target_year=resolved_target_year,
            policyengine_us_data_repo=policyengine_us_data_repo,
            cache_dir=cache_dir,
        )
        frame = _build_observation_frame(
            households=tables.households,
            persons=tables.persons,
            source_name=self.spec.source_name(year),
            shareability=self.shareability,
            archetype=self.spec.archetype,
        )
        self._descriptor_cache = frame.source
        return apply_source_query(frame, query)


class ACSSourceProvider(DonorSurveySourceProvider):
    """PolicyEngine-aligned ACS donor provider."""

    def __init__(
        self,
        *,
        year: int = get_pe_source_impute_block_spec("acs").default_year,
        shareability: Shareability = Shareability.PUBLIC,
        loader: DonorSurveyTablesLoader | None = None,
        policyengine_us_data_repo: str | Path | None = None,
        policyengine_us_data_python: str | Path | None = None,
        target_year: int | None = None,
    ) -> None:
        super().__init__(
            spec=get_pe_source_impute_block_spec("acs"),
            year=year,
            shareability=shareability,
            loader=loader,
            policyengine_us_data_repo=policyengine_us_data_repo,
            policyengine_us_data_python=policyengine_us_data_python,
            target_year=target_year,
        )


class SIPPSourceProvider(DonorSurveySourceProvider):
    """PolicyEngine-aligned SIPP donor provider with block-level specs."""

    def __init__(
        self,
        *,
        block: str,
        year: int | None = None,
        cache_dir: str | Path | None = None,
        shareability: Shareability = Shareability.PUBLIC,
        loader: DonorSurveyTablesLoader | None = None,
        policyengine_us_data_repo: str | Path | None = None,
        policyengine_us_data_python: str | Path | None = None,
        target_year: int | None = None,
    ) -> None:
        self.block = block
        super().__init__(
            spec=resolve_sipp_donor_survey_spec(block),
            year=year,
            cache_dir=cache_dir,
            shareability=shareability,
            loader=loader,
            policyengine_us_data_repo=policyengine_us_data_repo,
            policyengine_us_data_python=policyengine_us_data_python,
            target_year=target_year,
        )


class SIPPTipsSourceProvider(SIPPSourceProvider):
    """Backward-compatible alias for the SIPP tips donor block."""

    def __init__(
        self,
        *,
        year: int | None = None,
        cache_dir: str | Path | None = None,
        shareability: Shareability = Shareability.PUBLIC,
        loader: DonorSurveyTablesLoader | None = None,
        policyengine_us_data_repo: str | Path | None = None,
        policyengine_us_data_python: str | Path | None = None,
        target_year: int | None = None,
    ) -> None:
        super().__init__(
            block="tips",
            year=year,
            cache_dir=cache_dir,
            shareability=shareability,
            loader=loader,
            policyengine_us_data_repo=policyengine_us_data_repo,
            policyengine_us_data_python=policyengine_us_data_python,
            target_year=target_year,
        )


class SIPPAssetsSourceProvider(SIPPSourceProvider):
    """Backward-compatible alias for the SIPP asset donor block."""

    def __init__(
        self,
        *,
        year: int | None = None,
        cache_dir: str | Path | None = None,
        shareability: Shareability = Shareability.PUBLIC,
        loader: DonorSurveyTablesLoader | None = None,
        policyengine_us_data_repo: str | Path | None = None,
        policyengine_us_data_python: str | Path | None = None,
        target_year: int | None = None,
    ) -> None:
        super().__init__(
            block="assets",
            year=year,
            cache_dir=cache_dir,
            shareability=shareability,
            loader=loader,
            policyengine_us_data_repo=policyengine_us_data_repo,
            policyengine_us_data_python=policyengine_us_data_python,
            target_year=target_year,
        )


class SCFSourceProvider(DonorSurveySourceProvider):
    """PolicyEngine-aligned SCF donor provider."""

    def __init__(
        self,
        *,
        year: int = get_pe_source_impute_block_spec("scf").default_year,
        shareability: Shareability = Shareability.PUBLIC,
        loader: DonorSurveyTablesLoader | None = None,
        policyengine_us_data_repo: str | Path | None = None,
        policyengine_us_data_python: str | Path | None = None,
        target_year: int | None = None,
    ) -> None:
        super().__init__(
            spec=get_pe_source_impute_block_spec("scf"),
            year=year,
            shareability=shareability,
            loader=loader,
            policyengine_us_data_repo=policyengine_us_data_repo,
            policyengine_us_data_python=policyengine_us_data_python,
            target_year=target_year,
        )
