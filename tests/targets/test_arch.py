from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import pytest
from microplex.core import EntityType
from microplex.targets import TargetAggregation, TargetQuery

from microplex_us.geography import US_STATE_ABBR_BY_FIPS
from microplex_us.pipelines.us import USMicroplexBuildConfig, USMicroplexPipeline
from microplex_us.policyengine.target_profiles import PolicyEngineUSTargetCell
from microplex_us.targets import (
    ArchConsumerFactJSONLTargetProvider,
    ArchSQLiteTargetProvider,
    summarize_arch_target_gap_queue,
    summarize_arch_target_profile_coverage,
)
from microplex_us.targets.arch import (
    ArchTargetRecord,
    arch_target_record_to_canonical_spec,
    main_gaps,
    main_refresh,
)


def _create_arch_targets_db(path: Path) -> None:
    conn = sqlite3.connect(path)
    conn.executescript(
        """
        CREATE TABLE strata (
            id INTEGER PRIMARY KEY,
            name TEXT,
            jurisdiction TEXT,
            parent_id INTEGER,
            definition_hash TEXT
        );

        CREATE TABLE stratum_constraints (
            id INTEGER PRIMARY KEY,
            stratum_id INTEGER NOT NULL,
            variable TEXT NOT NULL,
            operator TEXT NOT NULL,
            value TEXT NOT NULL
        );

        CREATE TABLE targets (
            id INTEGER PRIMARY KEY,
            stratum_id INTEGER NOT NULL,
            variable TEXT NOT NULL,
            period INTEGER NOT NULL,
            value REAL NOT NULL,
            target_type TEXT NOT NULL,
            geographic_level TEXT,
            source TEXT NOT NULL,
            source_table TEXT,
            source_url TEXT,
            notes TEXT
        );
        """
    )
    conn.executemany(
        """
        INSERT INTO strata (id, name, jurisdiction, definition_hash)
        VALUES (?, ?, ?, ?)
        """,
        [
            (1, "US", "US", "root"),
            (2, "US All Filers", "US", "filers"),
            (3, "CA Filers AGI $50k-$75k", "US", "ca_50k_75k"),
        ],
    )
    conn.executemany(
        """
        INSERT INTO stratum_constraints (
            stratum_id,
            variable,
            operator,
            value
        ) VALUES (?, ?, ?, ?)
        """,
        [
            (2, "is_tax_filer", "==", "1"),
            (3, "is_tax_filer", "==", "1"),
            (3, "state_fips", "==", "06"),
            (3, "agi_bracket", "==", "50k_to_75k"),
        ],
    )
    conn.executemany(
        """
        INSERT INTO targets (
            id,
            stratum_id,
            variable,
            period,
            value,
            target_type,
            geographic_level,
            source,
            source_table,
            source_url,
            notes
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        [
            (
                1,
                2,
                "tax_exempt_interest_returns",
                2023,
                10.0,
                "COUNT",
                None,
                "IRS_SOI",
                "SOI",
                None,
                None,
            ),
            (
                2,
                2,
                "tax_exempt_interest_amount",
                2023,
                100.0,
                "AMOUNT",
                None,
                "IRS_SOI",
                "SOI",
                None,
                None,
            ),
            (
                3,
                2,
                "adjusted_gross_income",
                2022,
                1_000.0,
                "AMOUNT",
                None,
                "IRS_SOI",
                "SOI",
                None,
                None,
            ),
            (
                4,
                2,
                "adjusted_gross_income",
                2023,
                1_100.0,
                "AMOUNT",
                None,
                "IRS_SOI",
                "SOI",
                None,
                None,
            ),
            (
                5,
                1,
                "labor_force_count",
                2023,
                100.0,
                "COUNT",
                None,
                "BLS",
                "BLS",
                None,
                None,
            ),
            (6, 1, "labor_force", 2024, 110.0, "COUNT", None, "CBO", "CBO", None, None),
            (
                7,
                3,
                "tax_unit_count",
                2023,
                20.0,
                "COUNT",
                "STATE",
                "IRS_SOI",
                "SOI",
                None,
                None,
            ),
        ],
    )
    conn.commit()
    conn.close()


def _insert_multi_domain_soi_targets(path: Path) -> None:
    conn = sqlite3.connect(path)
    conn.execute(
        """
        INSERT INTO strata (id, name, jurisdiction, definition_hash)
        VALUES (?, ?, ?, ?)
        """,
        (4, "US Filers AGI $50k-$75k", "US", "national_50k_75k"),
    )
    conn.executemany(
        """
        INSERT INTO stratum_constraints (
            stratum_id,
            variable,
            operator,
            value
        ) VALUES (?, ?, ?, ?)
        """,
        [
            (4, "is_tax_filer", "==", "1"),
            (4, "agi_bracket", "==", "50k_to_75k"),
        ],
    )
    conn.executemany(
        """
        INSERT INTO targets (
            id,
            stratum_id,
            variable,
            period,
            value,
            target_type,
            geographic_level,
            source,
            source_table,
            source_url,
            notes
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        [
            (
                8,
                4,
                "tax_exempt_interest_returns",
                2023,
                5.0,
                "COUNT",
                None,
                "IRS_SOI",
                "SOI",
                None,
                None,
            ),
            (
                9,
                4,
                "adjusted_gross_income",
                2023,
                500.0,
                "AMOUNT",
                None,
                "IRS_SOI",
                "SOI",
                None,
                None,
            ),
        ],
    )
    conn.commit()
    conn.close()


def _insert_w2_tip_income_target(path: Path) -> None:
    conn = sqlite3.connect(path)
    conn.execute(
        """
        INSERT INTO strata (id, name, jurisdiction, definition_hash)
        VALUES (?, ?, ?, ?)
        """,
        (5, "US taxpayers with Form W-2 social security tips", "US", "w2_tips"),
    )
    conn.execute(
        """
        INSERT INTO stratum_constraints (
            stratum_id,
            variable,
            operator,
            value
        ) VALUES (?, ?, ?, ?)
        """,
        (5, "tip_income", ">", "0"),
    )
    conn.executemany(
        """
        INSERT INTO targets (
            id,
            stratum_id,
            variable,
            period,
            value,
            target_type,
            geographic_level,
            source,
            source_table,
            source_url,
            notes
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        [
            (
                10,
                5,
                "tip_income",
                2020,
                80.0,
                "AMOUNT",
                "NATIONAL",
                "IRS_SOI",
                "W-2",
                None,
                None,
            ),
            (
                11,
                2,
                "adjusted_gross_income",
                2020,
                800.0,
                "AMOUNT",
                None,
                "IRS_SOI",
                "SOI",
                None,
                None,
            ),
        ],
    )
    conn.commit()
    conn.close()


def _insert_irs_soi_itemized_deduction_targets(path: Path) -> None:
    conn = sqlite3.connect(path)
    conn.executemany(
        """
        INSERT INTO targets (
            id,
            stratum_id,
            variable,
            period,
            value,
            target_type,
            geographic_level,
            source,
            source_table,
            source_url,
            notes
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        [
            (
                12,
                2,
                "medical_amount",
                2023,
                100.0,
                "AMOUNT",
                None,
                "IRS_SOI",
                "SOI Individual Returns - Itemized Deductions",
                None,
                None,
            ),
            (
                13,
                2,
                "real_estate_taxes_amount",
                2023,
                200.0,
                "AMOUNT",
                None,
                "IRS_SOI",
                "SOI Individual Returns - Itemized Deductions",
                None,
                None,
            ),
            (
                14,
                2,
                "salt_amount",
                2023,
                300.0,
                "AMOUNT",
                None,
                "IRS_SOI",
                "SOI Individual Returns - Itemized Deductions",
                None,
                None,
            ),
            (
                15,
                2,
                "medical_claims",
                2023,
                10.0,
                "COUNT",
                None,
                "IRS_SOI",
                "SOI Individual Returns - Itemized Deductions",
                None,
                None,
            ),
            (
                16,
                2,
                "real_estate_taxes_claims",
                2023,
                20.0,
                "COUNT",
                None,
                "IRS_SOI",
                "SOI Individual Returns - Itemized Deductions",
                None,
                None,
            ),
            (
                17,
                2,
                "salt_claims",
                2023,
                30.0,
                "COUNT",
                None,
                "IRS_SOI",
                "SOI Individual Returns - Itemized Deductions",
                None,
                None,
            ),
        ],
    )
    conn.commit()
    conn.close()


def _insert_complete_state_rollup_targets(path: Path) -> None:
    conn = sqlite3.connect(path)
    state_fips_values = sorted(
        state_fips for state_fips in US_STATE_ABBR_BY_FIPS if state_fips != "72"
    )
    ctc_strata = [
        (1_000 + index, f"{state_fips} CTC filers", "US", f"ctc_{state_fips}")
        for index, state_fips in enumerate(state_fips_values)
    ]
    aca_strata = [
        (2_000 + index, f"{state_fips} ACA marketplace", "US", f"aca_{state_fips}")
        for index, state_fips in enumerate(state_fips_values)
    ]
    conn.executemany(
        """
        INSERT INTO strata (id, name, jurisdiction, definition_hash)
        VALUES (?, ?, ?, ?)
        """,
        [*ctc_strata, *aca_strata],
    )
    conn.executemany(
        """
        INSERT INTO stratum_constraints (
            stratum_id,
            variable,
            operator,
            value
        ) VALUES (?, ?, ?, ?)
        """,
        [
            *((stratum_id, "is_tax_filer", "==", "1") for stratum_id, *_ in ctc_strata),
            *(
                (stratum_id, "state_fips", "==", state_fips)
                for stratum_id, _, _, definition_hash in ctc_strata
                for state_fips in (definition_hash.removeprefix("ctc_"),)
            ),
            *(
                (stratum_id, "state_fips", "==", state_fips)
                for stratum_id, _, _, definition_hash in aca_strata
                for state_fips in (definition_hash.removeprefix("aca_"),)
            ),
        ],
    )
    ctc_targets = [
        (
            10_000 + index * 2,
            stratum_id,
            "ctc_amount",
            2024,
            1_000.0 + index,
            "AMOUNT",
            None,
            "IRS_SOI",
            "State Data FY",
            None,
            None,
        )
        for index, (stratum_id, *_rest) in enumerate(ctc_strata)
    ]
    ctc_count_targets = [
        (
            10_001 + index * 2,
            stratum_id,
            "ctc_claims",
            2024,
            100.0 + index,
            "COUNT",
            None,
            "IRS_SOI",
            "State Data FY",
            None,
            None,
        )
        for index, (stratum_id, *_rest) in enumerate(ctc_strata)
    ]
    aca_targets = [
        (
            20_000 + index,
            stratum_id,
            "aca_aptc_amount",
            2024,
            10_000.0 + index,
            "AMOUNT",
            "STATE",
            "CMS_ACA",
            "2024 OEP State-Level Public Use File",
            None,
            None,
        )
        for index, (stratum_id, *_rest) in enumerate(aca_strata)
    ]
    conn.executemany(
        """
        INSERT INTO targets (
            id,
            stratum_id,
            variable,
            period,
            value,
            target_type,
            geographic_level,
            source,
            source_table,
            source_url,
            notes
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        [*ctc_targets, *ctc_count_targets, *aca_targets],
    )
    conn.commit()
    conn.close()


def test_arch_provider_ages_soi_and_maps_return_counts_to_positive_amounts(tmp_path):
    db_path = tmp_path / "arch_targets.db"
    _create_arch_targets_db(db_path)

    provider = ArchSQLiteTargetProvider(db_path)
    target_set = provider.load_target_set(
        TargetQuery(
            period=2024,
            provider_filters={
                "sources": ["IRS_SOI"],
                "variables": ["tax_exempt_interest_income"],
                "entity_overrides": {
                    "tax_exempt_interest_income": EntityType.PERSON,
                },
            },
        )
    )

    assert len(target_set.targets) == 2
    count_target = next(
        target
        for target in target_set.targets
        if target.aggregation is TargetAggregation.COUNT
    )
    amount_target = next(
        target
        for target in target_set.targets
        if target.aggregation is TargetAggregation.SUM
    )
    assert count_target.entity is EntityType.TAX_UNIT
    assert count_target.name == "arch_target_1"
    assert count_target.description == (
        "Tax-exempt interest returns for US All Filers (IRS SOI, 2024)"
    )
    assert count_target.metadata["display_label"] == count_target.description
    assert count_target.metadata["target_semantic"] == "count"
    assert count_target.metadata["model_variable_role"] == "preserved_input"
    assert count_target.measure is None
    assert count_target.value == pytest.approx(11.0)
    assert {
        (target_filter.feature, target_filter.operator.value, target_filter.value)
        for target_filter in count_target.filters
    } == {
        ("tax_unit_is_filer", "==", "1"),
        ("tax_exempt_interest_income", ">", 0),
    }

    assert amount_target.entity is EntityType.PERSON
    assert amount_target.description == (
        "Tax-exempt interest amount for US All Filers (IRS SOI, 2024)"
    )
    assert amount_target.metadata["display_label"] == amount_target.description
    assert amount_target.metadata["target_semantic"] == "amount"
    assert amount_target.metadata["model_variable_role"] == "preserved_input"
    assert amount_target.measure == "tax_exempt_interest_income"
    assert amount_target.value == pytest.approx(110.0)
    assert amount_target.metadata["arch_source_period"] == 2023
    assert amount_target.metadata["arch_aging_amount_method"] == (
        "soi_total_agi_last_growth_extrapolation"
    )


def test_arch_provider_maps_agi_bracket_constraints_to_agi_ranges(tmp_path):
    db_path = tmp_path / "arch_targets.db"
    _create_arch_targets_db(db_path)

    provider = ArchSQLiteTargetProvider(db_path)
    target_set = provider.load_target_set(
        TargetQuery(
            period=2024,
            provider_filters={
                "sources": ["IRS_SOI"],
                "variables": ["tax_unit_count"],
            },
        )
    )

    assert {target.metadata["target_id"] for target in target_set.targets} == {7}
    bracket_target = next(
        target for target in target_set.targets if target.metadata["target_id"] == 7
    )
    assert bracket_target.name == "arch_target_7"
    assert bracket_target.description == (
        "Tax unit count for CA Filers AGI $50k-$75k (IRS SOI, 2024)"
    )
    assert bracket_target.metadata["display_label"] == bracket_target.description
    assert bracket_target.value == pytest.approx(22.0)
    assert {
        (target_filter.feature, target_filter.operator.value, target_filter.value)
        for target_filter in bracket_target.filters
    } == {
        ("tax_unit_is_filer", "==", "1"),
        ("state_fips", "==", "06"),
        ("adjusted_gross_income", ">=", 50_000),
        ("adjusted_gross_income", "<", 75_000),
    }


def test_arch_provider_includes_parent_stratum_constraints(tmp_path):
    db_path = tmp_path / "arch_targets.db"
    _create_arch_targets_db(db_path)
    conn = sqlite3.connect(db_path)
    conn.execute(
        """
        INSERT INTO strata (id, name, jurisdiction, parent_id, definition_hash)
        VALUES (?, ?, ?, ?, ?)
        """,
        (4, "US Filers AGI $75k-$100k", "US", 2, "national_75k_100k"),
    )
    conn.execute(
        """
        INSERT INTO stratum_constraints (
            stratum_id,
            variable,
            operator,
            value
        ) VALUES (?, ?, ?, ?)
        """,
        (4, "agi_bracket", "==", "75k_to_100k"),
    )
    conn.execute(
        """
        INSERT INTO targets (
            id,
            stratum_id,
            variable,
            period,
            value,
            target_type,
            geographic_level,
            source,
            source_table,
            source_url,
            notes
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            8,
            4,
            "adjusted_gross_income",
            2023,
            750.0,
            "AMOUNT",
            None,
            "IRS_SOI",
            "SOI",
            None,
            None,
        ),
    )
    conn.commit()
    conn.close()

    provider = ArchSQLiteTargetProvider(db_path)
    target_set = provider.load_target_set(
        TargetQuery(
            period=2024,
            provider_filters={
                "sources": ["IRS_SOI"],
                "variables": ["adjusted_gross_income"],
            },
        )
    )

    child_target = next(
        target for target in target_set.targets if target.metadata["target_id"] == 8
    )
    assert {
        (target_filter.feature, target_filter.operator.value, target_filter.value)
        for target_filter in child_target.filters
    } == {
        ("tax_unit_is_filer", "==", "1"),
        ("adjusted_gross_income", ">=", 75_000),
        ("adjusted_gross_income", "<", 100_000),
    }


def test_arch_provider_honors_policyengine_target_cells(tmp_path):
    db_path = tmp_path / "arch_targets.db"
    _create_arch_targets_db(db_path)

    provider = ArchSQLiteTargetProvider(db_path)
    target_set = provider.load_target_set(
        TargetQuery(
            period=2024,
            provider_filters={
                "sources": ["IRS_SOI"],
                "target_cells": [
                    {
                        "variable": "tax_unit_count",
                        "geo_level": "national",
                        "domain_variable": "tax_exempt_interest_income",
                    }
                ],
            },
        )
    )

    assert [target.metadata["target_id"] for target in target_set.targets] == [1]
    target = target_set.targets[0]
    assert target.aggregation is TargetAggregation.COUNT
    assert target.measure is None

    target_set = provider.load_target_set(
        TargetQuery(
            period=2024,
            provider_filters={
                "sources": ["IRS_SOI"],
                "target_cells": [
                    {
                        "variable": "tax_exempt_interest_income",
                        "geo_level": "national",
                        "domain_variable": "tax_exempt_interest_income",
                    }
                ],
            },
        )
    )

    assert [target.metadata["target_id"] for target in target_set.targets] == [2]


def test_arch_provider_target_cell_domain_match_is_exact(tmp_path):
    db_path = tmp_path / "arch_targets.db"
    _create_arch_targets_db(db_path)
    _insert_multi_domain_soi_targets(db_path)

    provider = ArchSQLiteTargetProvider(db_path)
    target_set = provider.load_target_set(
        TargetQuery(
            period=2024,
            provider_filters={
                "sources": ["IRS_SOI"],
                "target_cells": [
                    {
                        "variable": "tax_unit_count",
                        "geo_level": "national",
                        "domain_variable": "tax_exempt_interest_income",
                    }
                ],
            },
        )
    )

    assert [target.metadata["target_id"] for target in target_set.targets] == [1]


def test_arch_provider_matches_aliased_amount_self_domain_target_cells(tmp_path):
    db_path = tmp_path / "arch_targets.db"
    _create_arch_targets_db(db_path)
    conn = sqlite3.connect(db_path)
    conn.execute(
        """
        INSERT INTO targets (
            id,
            stratum_id,
            variable,
            period,
            value,
            target_type,
            geographic_level,
            source,
            source_table,
            source_url,
            notes
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            8,
            2,
            "income_tax_liability",
            2023,
            80.0,
            "AMOUNT",
            None,
            "IRS_SOI",
            "SOI",
            None,
            None,
        ),
    )
    conn.commit()
    conn.close()

    provider = ArchSQLiteTargetProvider(db_path)
    target_set = provider.load_target_set(
        TargetQuery(
            period=2024,
            provider_filters={
                "sources": ["IRS_SOI"],
                "target_cells": [
                    {
                        "variable": "income_tax",
                        "geo_level": "national",
                        "domain_variable": "income_tax",
                    }
                ],
            },
        )
    )

    assert [target.metadata["target_id"] for target in target_set.targets] == [8]

    target_set = provider.load_target_set(
        TargetQuery(
            period=2024,
            provider_filters={
                "sources": ["IRS_SOI"],
                "target_cells": [
                    {
                        "variable": "income_tax_positive",
                        "geo_level": "national",
                        "domain_variable": None,
                    }
                ],
            },
        )
    )

    assert [target.metadata["target_id"] for target in target_set.targets] == [8]
    assert target_set.targets[0].measure == "income_tax"
    assert target_set.targets[0].metadata["arch_variable"] == "income_tax_liability"


def test_arch_provider_matches_current_profile_aliases(tmp_path):
    db_path = tmp_path / "arch_targets.db"
    _create_arch_targets_db(db_path)
    conn = sqlite3.connect(db_path)
    conn.executemany(
        """
        INSERT INTO targets (
            id,
            stratum_id,
            variable,
            period,
            value,
            target_type,
            geographic_level,
            source,
            source_table,
            source_url,
            notes
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        [
            (
                8,
                2,
                "alimony_received_amount",
                2023,
                20.0,
                "AMOUNT",
                None,
                "IRS_SOI",
                "SOI",
                None,
                None,
            ),
            (
                9,
                2,
                "schedule_c_income_amount",
                2023,
                30.0,
                "AMOUNT",
                None,
                "IRS_SOI",
                "SOI",
                None,
                None,
            ),
            (
                10,
                1,
                "medicaid_total_enrollment",
                2024,
                40.0,
                "COUNT",
                None,
                "CMS_MEDICAID",
                "CMS",
                None,
                None,
            ),
            (
                11,
                2,
                "wages_salaries_amount",
                2023,
                50.0,
                "AMOUNT",
                None,
                "IRS_SOI",
                "SOI",
                None,
                None,
            ),
            (
                12,
                2,
                "wages_salaries_returns",
                2023,
                60.0,
                "COUNT",
                None,
                "IRS_SOI",
                "SOI",
                None,
                None,
            ),
            (
                13,
                2,
                "schedule_c_income_returns",
                2023,
                70.0,
                "COUNT",
                None,
                "IRS_SOI",
                "SOI",
                None,
                None,
            ),
        ],
    )
    conn.commit()
    conn.close()

    provider = ArchSQLiteTargetProvider(db_path)
    target_set = provider.load_target_set(
        TargetQuery(
            period=2024,
            provider_filters={
                "target_cells": [
                    {
                        "variable": "alimony_income",
                        "geo_level": "national",
                        "domain_variable": None,
                    },
                    {
                        "variable": "self_employment_income",
                        "geo_level": "national",
                        "domain_variable": "self_employment_income",
                    },
                    {
                        "variable": "total_self_employment_income",
                        "geo_level": "national",
                        "domain_variable": "total_self_employment_income",
                    },
                    {
                        "variable": "tax_unit_count",
                        "geo_level": "national",
                        "domain_variable": "total_self_employment_income",
                    },
                    {
                        "variable": "person_count",
                        "geo_level": "national",
                        "domain_variable": "medicaid",
                    },
                    {
                        "variable": "employment_income",
                        "geo_level": "national",
                        "domain_variable": "employment_income",
                    },
                    {
                        "variable": "tax_unit_count",
                        "geo_level": "national",
                        "domain_variable": "employment_income",
                    },
                ],
            },
        )
    )

    assert {target.metadata["target_id"] for target in target_set.targets} == {
        8,
        9,
        10,
        11,
        12,
        13,
    }
    variables_by_id = {
        target.metadata["target_id"]: target.metadata["variable"]
        for target in target_set.targets
    }
    assert variables_by_id == {
        8: "alimony_income",
        9: "self_employment_income",
        10: "person_count",
        11: "employment_income",
        12: "employment_income",
        13: "self_employment_income",
    }


def test_arch_target_rejects_broad_proprietors_income_as_self_employment():
    record = ArchTargetRecord(
        target_id=1,
        stratum_id=1,
        variable="schedule_c_income_amount",
        period=2024,
        value=2_023_080_000_000,
        target_type="AMOUNT",
        geographic_level=None,
        geography_id=None,
        source="BEA",
        source_table="NIPA annual personal income components",
        source_url=None,
        notes=None,
        stratum_name="US",
        jurisdiction="US",
        constraints=(),
        concept=(
            "bea_nipa.proprietors_income_with_inventory_valuation_and_capital_"
            "consumption_adjustments"
        ),
        source_concept=(
            "bea_nipa.a041rc_proprietors_income_with_inventory_valuation_and_"
            "capital_consumption_adjustments"
        ),
    )

    with pytest.raises(
        ValueError,
        match="cannot be exposed as plain self_employment_income",
    ):
        arch_target_record_to_canonical_spec(record)


def test_arch_provider_maps_eitc_child_count_constraints(tmp_path):
    db_path = tmp_path / "arch_targets.db"
    _create_arch_targets_db(db_path)
    conn = sqlite3.connect(db_path)
    conn.execute(
        """
        INSERT INTO strata (id, name, jurisdiction, definition_hash)
        VALUES (?, ?, ?, ?)
        """,
        (4, "US EITC 3+ Children", "US", "eitc_3plus_children"),
    )
    conn.executemany(
        """
        INSERT INTO stratum_constraints (
            stratum_id,
            variable,
            operator,
            value
        ) VALUES (?, ?, ?, ?)
        """,
        [
            (4, "is_tax_filer", "==", "1"),
            (4, "eitc_qualifying_children", ">=", "3"),
        ],
    )
    conn.execute(
        """
        INSERT INTO targets (
            id,
            stratum_id,
            variable,
            period,
            value,
            target_type,
            geographic_level,
            source,
            source_table,
            source_url,
            notes
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            8,
            4,
            "eitc_amount",
            2023,
            15.0,
            "AMOUNT",
            None,
            "IRS_SOI",
            "EITC",
            None,
            None,
        ),
    )
    conn.commit()
    conn.close()

    provider = ArchSQLiteTargetProvider(db_path)
    target_set = provider.load_target_set(
        TargetQuery(
            period=2024,
            provider_filters={
                "sources": ["IRS_SOI"],
                "target_cells": [
                    {
                        "variable": "eitc",
                        "geo_level": "national",
                        "domain_variable": "eitc_child_count",
                    }
                ],
            },
        )
    )

    assert [target.metadata["target_id"] for target in target_set.targets] == [8]
    target = target_set.targets[0]
    assert target.measure == "eitc"
    assert {
        (target_filter.feature, target_filter.operator.value, target_filter.value)
        for target_filter in target.filters
    } == {
        ("tax_unit_is_filer", "==", "1"),
        ("eitc_child_count", ">=", "3"),
    }


def test_arch_provider_matches_eitc_count_and_multi_domain_cells(tmp_path):
    db_path = tmp_path / "arch_targets.db"
    _create_arch_targets_db(db_path)
    conn = sqlite3.connect(db_path)
    conn.executemany(
        """
        INSERT INTO strata (id, name, jurisdiction, definition_hash)
        VALUES (?, ?, ?, ?)
        """,
        [
            (4, "US EITC 3+ Children", "US", "eitc_3plus_children"),
            (5, "US AGI 1_to_10k EITC 1 Child", "US", "eitc_1_child_agi"),
        ],
    )
    conn.executemany(
        """
        INSERT INTO stratum_constraints (
            stratum_id,
            variable,
            operator,
            value
        ) VALUES (?, ?, ?, ?)
        """,
        [
            (4, "is_tax_filer", "==", "1"),
            (4, "eitc_qualifying_children", ">=", "3"),
            (5, "is_tax_filer", "==", "1"),
            (5, "agi_bracket", "==", "1_to_10k"),
            (5, "eitc_qualifying_children", "==", "1"),
        ],
    )
    conn.executemany(
        """
        INSERT INTO targets (
            id,
            stratum_id,
            variable,
            period,
            value,
            target_type,
            geographic_level,
            source,
            source_table,
            source_url,
            notes
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        [
            (
                8,
                4,
                "eitc_claims",
                2023,
                10.0,
                "COUNT",
                None,
                "IRS_SOI",
                "EITC",
                None,
                None,
            ),
            (
                9,
                5,
                "eitc_amount",
                2023,
                20.0,
                "AMOUNT",
                None,
                "IRS_SOI",
                "EITC",
                None,
                None,
            ),
            (
                10,
                5,
                "eitc_claims",
                2023,
                30.0,
                "COUNT",
                None,
                "IRS_SOI",
                "EITC",
                None,
                None,
            ),
        ],
    )
    conn.commit()
    conn.close()

    provider = ArchSQLiteTargetProvider(db_path)
    target_set = provider.load_target_set(
        TargetQuery(
            period=2024,
            provider_filters={
                "sources": ["IRS_SOI"],
                "target_cells": [
                    {
                        "variable": "tax_unit_count",
                        "geo_level": "national",
                        "domain_variable": "eitc_child_count",
                    },
                    {
                        "variable": "eitc",
                        "geo_level": "national",
                        "domain_variable": (
                            "adjusted_gross_income,eitc,eitc_child_count"
                        ),
                    },
                    {
                        "variable": "tax_unit_count",
                        "geo_level": "national",
                        "domain_variable": (
                            "adjusted_gross_income,eitc,eitc_child_count"
                        ),
                    },
                ],
            },
        )
    )

    assert {target.metadata["target_id"] for target in target_set.targets} == {
        8,
        9,
        10,
    }


def test_arch_provider_maps_census_stc_state_income_tax(tmp_path):
    db_path = tmp_path / "arch_targets.db"
    _create_arch_targets_db(db_path)
    conn = sqlite3.connect(db_path)
    conn.execute(
        """
        INSERT INTO strata (id, name, jurisdiction, definition_hash)
        VALUES (?, ?, ?, ?)
        """,
        (4, "CA state government", "US", "ca_state_government"),
    )
    conn.execute(
        """
        INSERT INTO stratum_constraints (
            stratum_id,
            variable,
            operator,
            value
        ) VALUES (?, ?, ?, ?)
        """,
        (4, "state_fips", "==", "06"),
    )
    conn.execute(
        """
        INSERT INTO targets (
            id,
            stratum_id,
            variable,
            period,
            value,
            target_type,
            geographic_level,
            source,
            source_table,
            source_url,
            notes
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            8,
            4,
            "state_individual_income_tax_collections",
            2024,
            123.0,
            "AMOUNT",
            "STATE",
            "CENSUS_STC",
            "STC T40",
            None,
            None,
        ),
    )
    conn.commit()
    conn.close()

    provider = ArchSQLiteTargetProvider(db_path)
    target_set = provider.load_target_set(
        TargetQuery(
            period=2024,
            provider_filters={
                "target_cells": [
                    {
                        "variable": "state_income_tax",
                        "geo_level": "state",
                        "domain_variable": None,
                    }
                ],
            },
        )
    )

    assert [target.metadata["target_id"] for target in target_set.targets] == [8]
    target = target_set.targets[0]
    assert target.measure == "state_income_tax"
    assert target.entity is EntityType.TAX_UNIT
    assert target.aggregation is TargetAggregation.SUM
    assert target.metadata["source"] == "CENSUS_STC"
    assert {
        (target_filter.feature, target_filter.operator.value, target_filter.value)
        for target_filter in target.filters
    } == {("state_fips", "==", "06")}


def test_arch_provider_maps_soi_itemized_deduction_targets(tmp_path):
    db_path = tmp_path / "arch_targets.db"
    _create_arch_targets_db(db_path)
    conn = sqlite3.connect(db_path)
    conn.executemany(
        """
        INSERT INTO targets (
            id,
            stratum_id,
            variable,
            period,
            value,
            target_type,
            geographic_level,
            source,
            source_table,
            source_url,
            notes
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        [
            (
                8,
                2,
                "limited_state_local_taxes_amount",
                2024,
                122.0,
                "AMOUNT",
                None,
                "IRS_SOI",
                "Historic Table 2",
                None,
                None,
            ),
            (
                9,
                2,
                "interest_paid_deduction_amount",
                2024,
                169.0,
                "AMOUNT",
                None,
                "IRS_SOI",
                "Historic Table 2",
                None,
                "Composed from Schedule A interest lines.",
            ),
        ],
    )
    conn.commit()
    conn.close()

    provider = ArchSQLiteTargetProvider(db_path)
    target_set = provider.load_target_set(
        TargetQuery(
            period=2024,
            provider_filters={
                "sources": ["IRS_SOI"],
                "target_cells": [
                    {
                        "variable": "salt_deduction",
                        "geo_level": "national",
                        "domain_variable": None,
                    },
                    {
                        "variable": "interest_deduction",
                        "geo_level": "national",
                        "domain_variable": None,
                    },
                ],
            },
        )
    )

    targets_by_measure = {target.measure: target for target in target_set.targets}
    assert set(targets_by_measure) == {"interest_deduction", "salt_deduction"}

    salt_target = targets_by_measure["salt_deduction"]
    assert salt_target.metadata["target_id"] == 8
    assert salt_target.entity is EntityType.TAX_UNIT
    assert salt_target.aggregation is TargetAggregation.SUM

    interest_target = targets_by_measure["interest_deduction"]
    assert interest_target.metadata["target_id"] == 9
    assert interest_target.entity is EntityType.TAX_UNIT
    assert interest_target.metadata["notes"] == (
        "Composed from Schedule A interest lines."
    )


def test_arch_provider_infers_geo_level_from_constraints(tmp_path):
    db_path = tmp_path / "arch_targets.db"
    _create_arch_targets_db(db_path)
    conn = sqlite3.connect(db_path)
    conn.execute(
        """
        INSERT INTO strata (id, name, jurisdiction, definition_hash)
        VALUES (?, ?, ?, ?)
        """,
        (4, "CA Filers", "US", "ca_filers"),
    )
    conn.executemany(
        """
        INSERT INTO stratum_constraints (
            stratum_id,
            variable,
            operator,
            value
        ) VALUES (?, ?, ?, ?)
        """,
        [
            (4, "is_tax_filer", "==", "1"),
            (4, "state_fips", "==", "06"),
        ],
    )
    conn.execute(
        """
        INSERT INTO targets (
            id,
            stratum_id,
            variable,
            period,
            value,
            target_type,
            geographic_level,
            source,
            source_table,
            source_url,
            notes
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            8,
            4,
            "adjusted_gross_income",
            2023,
            500.0,
            "AMOUNT",
            None,
            "IRS_SOI",
            "SOI",
            None,
            None,
        ),
    )
    conn.commit()
    conn.close()

    provider = ArchSQLiteTargetProvider(db_path)
    target_set = provider.load_target_set(
        TargetQuery(
            period=2024,
            provider_filters={
                "sources": ["IRS_SOI"],
                "target_cells": [
                    {
                        "variable": "adjusted_gross_income",
                        "geo_level": "state",
                        "geographic_id": "6",
                        "domain_variable": None,
                    }
                ],
            },
        )
    )

    assert [target.metadata["target_id"] for target in target_set.targets] == [8]
    assert target_set.targets[0].metadata["geo_level"] == "state"

    target_set = provider.load_target_set(
        TargetQuery(
            period=2024,
            provider_filters={
                "sources": ["IRS_SOI"],
                "variables": ["adjusted_gross_income"],
                "geo_levels": ["national"],
            },
        )
    )

    assert 8 not in {target.metadata["target_id"] for target in target_set.targets}


def test_arch_provider_maps_program_indicator_constraints_to_support_filters(tmp_path):
    db_path = tmp_path / "arch_targets.db"
    _create_arch_targets_db(db_path)
    conn = sqlite3.connect(db_path)
    conn.execute(
        """
        INSERT INTO strata (id, name, jurisdiction, definition_hash)
        VALUES (?, ?, ?, ?)
        """,
        (4, "SNAP households", "US", "snap_households"),
    )
    conn.execute(
        """
        INSERT INTO stratum_constraints (
            stratum_id,
            variable,
            operator,
            value
        ) VALUES (?, ?, ?, ?)
        """,
        (4, "snap", "==", "1"),
    )
    conn.executemany(
        """
        INSERT INTO targets (
            id,
            stratum_id,
            variable,
            period,
            value,
            target_type,
            geographic_level,
            source,
            source_table,
            source_url,
            notes
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        [
            (
                8,
                4,
                "snap_household_count",
                2024,
                10.0,
                "COUNT",
                None,
                "USDA_SNAP",
                "USDA",
                None,
                None,
            ),
            (
                9,
                4,
                "snap_benefits",
                2024,
                500.0,
                "AMOUNT",
                None,
                "USDA_SNAP",
                "USDA",
                None,
                None,
            ),
        ],
    )
    conn.commit()
    conn.close()

    provider = ArchSQLiteTargetProvider(db_path)
    target_set = provider.load_target_set(
        TargetQuery(
            period=2024,
            provider_filters={
                "sources": ["USDA_SNAP"],
                "target_cells": [
                    {
                        "variable": "snap",
                        "geo_level": "national",
                        "domain_variable": "snap",
                    },
                    {
                        "variable": "household_count",
                        "geo_level": "national",
                        "domain_variable": "snap",
                    },
                ],
            },
        )
    )

    assert {target.metadata["target_id"] for target in target_set.targets} == {8, 9}
    for target in target_set.targets:
        assert [
            (target_filter.feature, target_filter.operator.value, target_filter.value)
            for target_filter in target.filters
        ] == [("snap", ">", 0)]

    target_set = provider.load_target_set(
        TargetQuery(
            period=2024,
            provider_filters={
                "sources": ["USDA_SNAP"],
                "target_cells": [
                    {
                        "variable": "snap",
                        "geo_level": "national",
                        "domain_variable": None,
                    },
                ],
            },
        )
    )

    assert [target.metadata["target_id"] for target in target_set.targets] == [9]


def test_arch_provider_normalizes_congressional_district_constraints(tmp_path):
    db_path = tmp_path / "arch_targets.db"
    _create_arch_targets_db(db_path)
    conn = sqlite3.connect(db_path)
    conn.execute(
        """
        INSERT INTO strata (id, name, jurisdiction, definition_hash)
        VALUES (?, ?, ?, ?)
        """,
        (4, "CA-01 Filers", "US", "ca_01_filers"),
    )
    conn.executemany(
        """
        INSERT INTO stratum_constraints (
            stratum_id,
            variable,
            operator,
            value
        ) VALUES (?, ?, ?, ?)
        """,
        [
            (4, "is_tax_filer", "=", "1"),
            (4, "state_fips", "=", "06"),
            (4, "congressional_district", "=", "01"),
        ],
    )
    conn.execute(
        """
        INSERT INTO targets (
            id,
            stratum_id,
            variable,
            period,
            value,
            target_type,
            geographic_level,
            source,
            source_table,
            source_url,
            notes
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            8,
            4,
            "adjusted_gross_income",
            2023,
            500.0,
            "AMOUNT",
            "CONGRESSIONAL_DISTRICT",
            "IRS_SOI",
            "SOI",
            None,
            None,
        ),
    )
    conn.commit()
    conn.close()

    provider = ArchSQLiteTargetProvider(db_path)
    target_set = provider.load_target_set(
        TargetQuery(
            period=2024,
            provider_filters={
                "sources": ["IRS_SOI"],
                "target_cells": [
                    {
                        "variable": "adjusted_gross_income",
                        "geo_level": "district",
                        "geographic_id": "0601",
                        "domain_variable": None,
                    }
                ],
            },
        )
    )

    assert [target.metadata["target_id"] for target in target_set.targets] == [8]
    target = target_set.targets[0]
    assert target.metadata["geo_level"] == "district"
    assert (
        "congressional_district_geoid",
        "==",
        "0601",
    ) in {
        (target_filter.feature, target_filter.operator.value, target_filter.value)
        for target_filter in target.filters
    }
    assert (
        "tax_unit_is_filer",
        "==",
        "1",
    ) in {
        (target_filter.feature, target_filter.operator.value, target_filter.value)
        for target_filter in target.filters
    }

    target_set = provider.load_target_set(
        TargetQuery(
            period=2024,
            provider_filters={
                "sources": ["IRS_SOI"],
                "variables": ["adjusted_gross_income"],
                "geo_levels": ["state"],
            },
        )
    )

    assert 8 not in {target.metadata["target_id"] for target in target_set.targets}

    target_set = provider.load_target_set(
        TargetQuery(
            period=2024,
            provider_filters={
                "sources": ["IRS_SOI"],
                "variables": ["adjusted_gross_income"],
                "geo_levels": ["congressional_district"],
            },
        )
    )

    assert [target.metadata["target_id"] for target in target_set.targets] == [8]

    target_set = provider.load_target_set(
        TargetQuery(
            period=2024,
            provider_filters={
                "sources": ["IRS_SOI"],
                "variables": ["adjusted_gross_income"],
                "geo_levels": ["congressional-district"],
            },
        )
    )

    assert [target.metadata["target_id"] for target in target_set.targets] == [8]


def test_arch_provider_no_domain_target_cell_excludes_domain_strata(tmp_path):
    db_path = tmp_path / "arch_targets.db"
    _create_arch_targets_db(db_path)
    _insert_multi_domain_soi_targets(db_path)

    provider = ArchSQLiteTargetProvider(db_path)
    target_set = provider.load_target_set(
        TargetQuery(
            period=2024,
            provider_filters={
                "sources": ["IRS_SOI"],
                "target_cells": [
                    {
                        "variable": "adjusted_gross_income",
                        "geo_level": "national",
                        "domain_variable": None,
                    }
                ],
            },
        )
    )

    assert [target.metadata["target_id"] for target in target_set.targets] == [4]

    target_set = provider.load_target_set(
        TargetQuery(
            period=2024,
            provider_filters={
                "sources": ["IRS_SOI"],
                "target_cells": [
                    {
                        "variable": "adjusted_gross_income",
                        "geo_level": "national",
                        "domain_variable": "",
                    }
                ],
            },
        )
    )

    assert [target.metadata["target_id"] for target in target_set.targets] == [4]


def test_arch_provider_current_year_partial_soi_falls_back_to_latest_soi(tmp_path):
    db_path = tmp_path / "arch_targets.db"
    _create_arch_targets_db(db_path)
    conn = sqlite3.connect(db_path)
    conn.executemany(
        """
        INSERT INTO targets (
            id,
            stratum_id,
            variable,
            period,
            value,
            target_type,
            geographic_level,
            source,
            source_table,
            source_url,
            notes
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        [
            (
                8,
                2,
                "tax_unit_count",
                2024,
                25.0,
                "COUNT",
                None,
                "IRS_SOI",
                "SOI",
                None,
                None,
            ),
            (
                9,
                2,
                "adjusted_gross_income",
                2024,
                1_200.0,
                "AMOUNT",
                None,
                "IRS_SOI",
                "SOI",
                None,
                None,
            ),
            (
                10,
                2,
                "income_tax_liability",
                2024,
                80.0,
                "AMOUNT",
                None,
                "IRS_SOI",
                "SOI",
                None,
                None,
            ),
        ],
    )
    conn.commit()
    conn.close()

    provider = ArchSQLiteTargetProvider(db_path)
    target_set = provider.load_target_set(
        TargetQuery(
            period=2024,
            provider_filters={
                "sources": ["IRS_SOI"],
                "variables": ["tax_exempt_interest_income"],
            },
        )
    )

    assert {target.metadata["target_id"] for target in target_set.targets} == {1, 2}
    assert {target.metadata["arch_source_period"] for target in target_set.targets} == {
        2023
    }


def test_arch_provider_uses_latest_soi_record_per_composition(tmp_path):
    db_path = tmp_path / "arch_targets.db"
    _create_arch_targets_db(db_path)
    conn = sqlite3.connect(db_path)
    conn.execute(
        """
        INSERT INTO strata (id, name, jurisdiction, definition_hash)
        VALUES (?, ?, ?, ?)
        """,
        (4, "CA Filers", "US", "ca_filers"),
    )
    conn.executemany(
        """
        INSERT INTO stratum_constraints (
            stratum_id,
            variable,
            operator,
            value
        ) VALUES (?, ?, ?, ?)
        """,
        [
            (4, "is_tax_filer", "==", "1"),
            (4, "state_fips", "==", "06"),
        ],
    )
    conn.executemany(
        """
        INSERT INTO targets (
            id,
            stratum_id,
            variable,
            period,
            value,
            target_type,
            geographic_level,
            source,
            source_table,
            source_url,
            notes
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        [
            (
                8,
                1,
                "labor_force_count",
                2022,
                100.0,
                "COUNT",
                None,
                "BLS",
                "BLS",
                None,
                None,
            ),
            (
                9,
                2,
                "tax_unit_count",
                2024,
                25.0,
                "COUNT",
                None,
                "IRS_SOI",
                "SOI",
                None,
                None,
            ),
            (
                10,
                2,
                "adjusted_gross_income",
                2024,
                1_200.0,
                "AMOUNT",
                None,
                "IRS_SOI",
                "SOI",
                None,
                None,
            ),
            (
                11,
                2,
                "income_tax_liability",
                2024,
                80.0,
                "AMOUNT",
                None,
                "IRS_SOI",
                "SOI",
                None,
                None,
            ),
            (
                12,
                4,
                "wages_salaries_amount",
                2022,
                90.0,
                "AMOUNT",
                "STATE",
                "IRS_SOI",
                "SOI",
                None,
                None,
            ),
        ],
    )
    conn.commit()
    conn.close()

    provider = ArchSQLiteTargetProvider(db_path)
    target_set = provider.load_target_set(
        TargetQuery(
            period=2024,
            provider_filters={
                "sources": ["IRS_SOI"],
                "target_cells": [
                    {
                        "variable": "employment_income",
                        "geo_level": "state",
                        "domain_variable": "employment_income",
                    }
                ],
            },
        )
    )

    assert [target.metadata["target_id"] for target in target_set.targets] == [12]
    assert target_set.targets[0].period == 2024
    assert target_set.targets[0].metadata["arch_source_period"] == 2022


def test_arch_provider_maps_income_tax_before_credits_targets(tmp_path):
    db_path = tmp_path / "arch_targets.db"
    _create_arch_targets_db(db_path)
    conn = sqlite3.connect(db_path)
    conn.executemany(
        """
        INSERT INTO targets (
            id,
            stratum_id,
            variable,
            period,
            value,
            target_type,
            geographic_level,
            source,
            source_table,
            source_url,
            notes
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        [
            (
                8,
                2,
                "income_tax_before_credits_returns",
                2023,
                50.0,
                "COUNT",
                None,
                "IRS_SOI",
                "SOI",
                None,
                None,
            ),
            (
                9,
                2,
                "income_tax_before_credits_amount",
                2023,
                500.0,
                "AMOUNT",
                None,
                "IRS_SOI",
                "SOI",
                None,
                None,
            ),
        ],
    )
    conn.commit()
    conn.close()

    provider = ArchSQLiteTargetProvider(db_path)
    target_set = provider.load_target_set(
        TargetQuery(
            period=2024,
            provider_filters={
                "sources": ["IRS_SOI"],
                "target_cells": [
                    {
                        "variable": "tax_unit_count",
                        "geo_level": "national",
                        "domain_variable": "income_tax_before_credits",
                    },
                    {
                        "variable": "income_tax_before_credits",
                        "geo_level": "national",
                        "domain_variable": "income_tax_before_credits",
                    },
                ],
            },
        )
    )

    assert {target.metadata["target_id"] for target in target_set.targets} == {8, 9}
    count_target = next(
        target
        for target in target_set.targets
        if target.aggregation is TargetAggregation.COUNT
    )
    assert {
        (target_filter.feature, target_filter.operator.value, target_filter.value)
        for target_filter in count_target.filters
    } == {
        ("tax_unit_is_filer", "==", "1"),
        ("income_tax_before_credits", ">", 0),
    }


def test_arch_provider_maps_real_estate_tax_targets(tmp_path):
    db_path = tmp_path / "arch_targets.db"
    _create_arch_targets_db(db_path)
    conn = sqlite3.connect(db_path)
    conn.executemany(
        """
        INSERT INTO targets (
            id,
            stratum_id,
            variable,
            period,
            value,
            target_type,
            geographic_level,
            source,
            source_table,
            source_url,
            notes
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        [
            (
                8,
                2,
                "real_estate_taxes_claims",
                2023,
                12.0,
                "COUNT",
                None,
                "IRS_SOI",
                "SOI",
                None,
                None,
            ),
            (
                9,
                2,
                "real_estate_taxes_amount",
                2023,
                120.0,
                "AMOUNT",
                None,
                "IRS_SOI",
                "SOI",
                None,
                None,
            ),
        ],
    )
    conn.commit()
    conn.close()

    provider = ArchSQLiteTargetProvider(db_path)
    target_set = provider.load_target_set(
        TargetQuery(
            period=2024,
            provider_filters={
                "sources": ["IRS_SOI"],
                "target_cells": [
                    {
                        "variable": "tax_unit_count",
                        "geo_level": "national",
                        "domain_variable": "real_estate_taxes",
                    },
                    {
                        "variable": "real_estate_taxes",
                        "geo_level": "national",
                        "domain_variable": "real_estate_taxes",
                    },
                ],
            },
        )
    )

    assert {target.metadata["target_id"] for target in target_set.targets} == {8, 9}
    assert {target.metadata["variable"] for target in target_set.targets} == {
        "real_estate_taxes"
    }


def test_arch_provider_maps_aca_aptc_amount_targets(tmp_path):
    db_path = tmp_path / "arch_targets.db"
    _create_arch_targets_db(db_path)
    conn = sqlite3.connect(db_path)
    conn.execute(
        """
        INSERT INTO strata (id, name, jurisdiction, definition_hash)
        VALUES (?, ?, ?, ?)
        """,
        (4, "CA ACA Marketplace", "US", "ca_aca"),
    )
    conn.execute(
        """
        INSERT INTO stratum_constraints (
            stratum_id,
            variable,
            operator,
            value
        ) VALUES (?, ?, ?, ?)
        """,
        (4, "state_fips", "==", "06"),
    )
    conn.executemany(
        """
        INSERT INTO targets (
            id,
            stratum_id,
            variable,
            period,
            value,
            target_type,
            geographic_level,
            source,
            source_table,
            source_url,
            notes
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        [
            (
                8,
                4,
                "aca_aptc_amount",
                2024,
                100.0,
                "AMOUNT",
                "STATE",
                "CMS_ACA",
                "CMS OEP",
                None,
                None,
            ),
            (
                9,
                4,
                "aca_marketplace_enrollment",
                2024,
                200.0,
                "COUNT",
                "STATE",
                "CMS_ACA",
                "CMS OEP",
                None,
                None,
            ),
        ],
    )
    conn.commit()
    conn.close()

    provider = ArchSQLiteTargetProvider(db_path)
    target_set = provider.load_target_set(
        TargetQuery(
            period=2024,
            provider_filters={
                "sources": ["CMS_ACA"],
                "target_cells": [
                    {
                        "variable": "aca_ptc",
                        "geo_level": "state",
                        "domain_variable": None,
                    }
                ],
            },
        )
    )

    assert [target.metadata["target_id"] for target in target_set.targets] == [8]
    assert target_set.targets[0].measure == "aca_ptc"

    target_set = provider.load_target_set(
        TargetQuery(
            period=2024,
            provider_filters={
                "sources": ["CMS_ACA"],
                "target_cells": [
                    {
                        "variable": "person_count",
                        "geo_level": "state",
                        "domain_variable": "aca_ptc,is_aca_ptc_eligible",
                    }
                ],
            },
        )
    )

    assert [target.metadata["target_id"] for target in target_set.targets] == [9]


def test_arch_provider_maps_soi_aca_ptc_return_counts(tmp_path):
    db_path = tmp_path / "arch_targets.db"
    _create_arch_targets_db(db_path)
    conn = sqlite3.connect(db_path)
    conn.execute(
        """
        INSERT INTO targets (
            id,
            stratum_id,
            variable,
            period,
            value,
            target_type,
            geographic_level,
            source,
            source_table,
            source_url,
            notes
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            8,
            2,
            "aca_ptc_returns",
            2023,
            7_841_370.0,
            "COUNT",
            None,
            "IRS_SOI",
            "Historic Table 2",
            None,
            None,
        ),
    )
    conn.commit()
    conn.close()

    provider = ArchSQLiteTargetProvider(db_path)
    target_set = provider.load_target_set(
        TargetQuery(
            period=2024,
            provider_filters={
                "sources": ["IRS_SOI"],
                "target_cells": [
                    {
                        "variable": "tax_unit_count",
                        "geo_level": "national",
                        "domain_variable": "used_aca_ptc",
                    }
                ],
            },
        )
    )

    assert [target.metadata["target_id"] for target in target_set.targets] == [8]
    target = target_set.targets[0]
    assert target.aggregation is TargetAggregation.COUNT
    assert target.entity is EntityType.TAX_UNIT
    assert {
        (target_filter.feature, target_filter.operator.value, target_filter.value)
        for target_filter in target.filters
    } == {
        ("tax_unit_is_filer", "==", "1"),
        ("aca_ptc", ">", 0),
    }

    target_set = provider.load_target_set(
        TargetQuery(
            period=2024,
            provider_filters={
                "sources": ["IRS_SOI"],
                "target_cells": [
                    {
                        "variable": "tax_unit_count",
                        "geo_level": "national",
                        "domain_variable": (
                            "selected_marketplace_plan_benchmark_ratio,used_aca_ptc"
                        ),
                    }
                ],
            },
        )
    )

    assert [target.metadata["target_id"] for target in target_set.targets] == [8]


def test_arch_provider_maps_soi_tax_filer_individual_counts(tmp_path):
    db_path = tmp_path / "arch_targets.db"
    _create_arch_targets_db(db_path)
    conn = sqlite3.connect(db_path)
    conn.execute(
        """
        INSERT INTO strata (id, name, jurisdiction, definition_hash)
        VALUES (?, ?, ?, ?)
        """,
        (4, "CA AGI 1_to_10k", "US", "ca_agi_1_to_10k"),
    )
    conn.executemany(
        """
        INSERT INTO stratum_constraints (
            stratum_id,
            variable,
            operator,
            value
        ) VALUES (?, ?, ?, ?)
        """,
        [
            (4, "is_tax_filer", "==", "1"),
            (4, "state_fips", "==", "06"),
            (4, "agi_bracket", "==", "1_to_10k"),
        ],
    )
    conn.execute(
        """
        INSERT INTO targets (
            id,
            stratum_id,
            variable,
            period,
            value,
            target_type,
            geographic_level,
            source,
            source_table,
            source_url,
            notes
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            8,
            4,
            "tax_filer_individual_count",
            2023,
            1_930_150.0,
            "COUNT",
            "STATE",
            "IRS_SOI",
            "Historic Table 2",
            None,
            "SOI number of individuals does not represent full population.",
        ),
    )
    conn.commit()
    conn.close()

    provider = ArchSQLiteTargetProvider(db_path)
    target_set = provider.load_target_set(
        TargetQuery(
            period=2024,
            provider_filters={
                "sources": ["IRS_SOI"],
                "target_cells": [
                    {
                        "variable": "person_count",
                        "geo_level": "state",
                        "domain_variable": "adjusted_gross_income",
                    }
                ],
            },
        )
    )

    assert [target.metadata["target_id"] for target in target_set.targets] == [8]
    target = target_set.targets[0]
    assert target.aggregation is TargetAggregation.COUNT
    assert target.entity is EntityType.PERSON
    assert {
        (target_filter.feature, target_filter.operator.value, target_filter.value)
        for target_filter in target.filters
    } == {
        ("tax_unit_is_filer", "==", "1"),
        ("state_fips", "==", "06"),
        ("adjusted_gross_income", ">=", 1),
        ("adjusted_gross_income", "<", 10_000),
    }


def test_arch_provider_maps_medicaid_benefit_targets(tmp_path):
    db_path = tmp_path / "arch_targets.db"
    _create_arch_targets_db(db_path)
    conn = sqlite3.connect(db_path)
    conn.execute(
        """
        INSERT INTO targets (
            id,
            stratum_id,
            variable,
            period,
            value,
            target_type,
            geographic_level,
            source,
            source_table,
            source_url,
            notes
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            8,
            1,
            "medicaid_benefits",
            2024,
            931_692_000_000.0,
            "AMOUNT",
            "NATIONAL",
            "CMS_MEDICAID",
            "CMS NHE",
            None,
            None,
        ),
    )
    conn.commit()
    conn.close()

    provider = ArchSQLiteTargetProvider(db_path)
    target_set = provider.load_target_set(
        TargetQuery(
            period=2024,
            provider_filters={
                "sources": ["CMS_MEDICAID"],
                "target_cells": [
                    {
                        "variable": "medicaid",
                        "geo_level": "national",
                        "domain_variable": None,
                    }
                ],
            },
        )
    )

    assert [target.metadata["target_id"] for target in target_set.targets] == [8]
    target = target_set.targets[0]
    assert target.measure == "medicaid"
    assert target.entity is EntityType.PERSON


def test_arch_consumer_fact_provider_maps_wealth_and_part_b_targets(tmp_path):
    jsonl_path = tmp_path / "consumer_facts.jsonl"
    rows = [
        {
            "schema_version": "arch.consumer_fact.v1",
            "aggregate_fact_key": "arch.aggregate_fact.v2:net_worth",
            "semantic_fact_key": "arch.semantic_fact.v2:net_worth",
            "concept_alignment": {
                "canonical_concept": (
                    "federal_reserve.z1.households_nonprofits_net_worth"
                ),
                "source_concept": "federal_reserve.z1.fl152090005",
                "relation": "source_label",
                "authority": "arch-us",
            },
            "geography": {
                "id": "0100000US",
                "level": "country",
            },
            "label": "United States household net worth",
            "observed_measure": {
                "source_concept": "federal_reserve.z1.fl152090005",
                "source_measure_id": "amount_outstanding",
                "source_name": "federal_reserve",
                "source_table": (
                    "Z.1 B.101 Households and nonprofit organizations"
                ),
                "unit": "usd",
            },
            "period": {"type": "calendar_year", "value": 2024},
            "source": {
                "source_name": "federal_reserve",
                "source_table": (
                    "Z.1 B.101 Households and nonprofit organizations"
                ),
                "url": "https://www.federalreserve.gov/releases/z1/",
            },
            "universe_constraints": {"domain": "household_balance_sheet"},
            "value": 169_619_200_000_000,
        },
        {
            "schema_version": "arch.consumer_fact.v1",
            "aggregate_fact_key": "arch.aggregate_fact.v2:part_b",
            "semantic_fact_key": "arch.semantic_fact.v2:part_b",
            "concept_alignment": {
                "canonical_concept": "cms_medicare.part_b_premium_income",
                "source_concept": "cms_medicare.part_b_premium_income",
            },
            "geography": {
                "id": "0100000US",
                "level": "country",
            },
            "label": "United States Medicare Part B premium income",
            "observed_measure": {
                "source_concept": "cms_medicare.part_b_premium_income",
                "source_measure_id": "actual_amount",
                "source_name": "cms_medicare",
                "source_table": "2025 Medicare Trustees Report Table III.C3",
                "unit": "usd",
            },
            "period": {"type": "calendar_year", "value": 2024},
            "source": {
                "source_name": "cms_medicare",
                "source_table": "2025 Medicare Trustees Report Table III.C3",
                "url": "https://www.cms.gov/oact/tr/2025",
            },
            "universe_constraints": {
                "domain": "medicare_financing",
                "constraints": [
                    {
                        "operator": "==",
                        "role": "filter",
                        "value": "actual",
                        "variable": "amount_basis",
                    },
                    {
                        "operator": "==",
                        "role": "filter",
                        "value": "part_b",
                        "variable": "medicare.part",
                    },
                    {
                        "operator": "==",
                        "role": "filter",
                        "value": "premiums_from_enrollees",
                        "variable": "medicare.financing_component",
                    },
                ],
            },
            "value": 139_837_000_000,
        },
    ]
    jsonl_path.write_text(
        "".join(f"{json.dumps(row, sort_keys=True)}\n" for row in rows)
    )

    provider = ArchConsumerFactJSONLTargetProvider(jsonl_path)
    target_set = provider.load_target_set(
        TargetQuery(
            period=2024,
            provider_filters={
                "target_cells": [
                    {
                        "variable": "net_worth",
                        "geo_level": "national",
                        "domain_variable": None,
                    },
                    {
                        "variable": "medicare_part_b_premiums",
                        "geo_level": "national",
                        "domain_variable": None,
                    },
                ],
            },
        )
    )

    targets_by_measure = {target.measure: target for target in target_set.targets}
    assert set(targets_by_measure) == {"medicare_part_b_premiums", "net_worth"}

    net_worth = targets_by_measure["net_worth"]
    assert net_worth.entity is EntityType.HOUSEHOLD
    assert net_worth.aggregation is TargetAggregation.SUM
    assert net_worth.value == pytest.approx(169_619_200_000_000)
    assert net_worth.filters == ()
    assert net_worth.metadata["source"] == "FEDERAL_RESERVE"
    assert net_worth.metadata["arch_source_concept"] == (
        "federal_reserve.z1.fl152090005"
    )

    part_b = targets_by_measure["medicare_part_b_premiums"]
    assert part_b.entity is EntityType.PERSON
    assert part_b.aggregation is TargetAggregation.SUM
    assert part_b.value == pytest.approx(139_837_000_000)
    assert part_b.filters == ()
    assert part_b.metadata["source"] == "CMS_MEDICARE"
    assert part_b.metadata["arch_concept"] == "cms_medicare.part_b_premium_income"


def test_arch_provider_maps_ssa_benefit_targets(tmp_path):
    db_path = tmp_path / "arch_targets.db"
    _create_arch_targets_db(db_path)
    conn = sqlite3.connect(db_path)
    conn.executemany(
        """
        INSERT INTO targets (
            id,
            stratum_id,
            variable,
            period,
            value,
            target_type,
            geographic_level,
            source,
            source_table,
            source_url,
            notes
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        [
            (
                8,
                1,
                "social_security_benefits",
                2024,
                1_471_195_000_000.0,
                "AMOUNT",
                "NATIONAL",
                "SSA",
                "SSA Supplement",
                None,
                None,
            ),
            (
                9,
                1,
                "social_security_retirement_benefits",
                2024,
                1_111_728_000_000.0,
                "AMOUNT",
                "NATIONAL",
                "SSA",
                "SSA Supplement",
                None,
                None,
            ),
            (
                10,
                1,
                "ssi_payments",
                2024,
                63_079_493_000.0,
                "AMOUNT",
                "NATIONAL",
                "SSA",
                "SSA Supplement",
                None,
                None,
            ),
        ],
    )
    conn.commit()
    conn.close()

    provider = ArchSQLiteTargetProvider(db_path)
    target_set = provider.load_target_set(
        TargetQuery(
            period=2024,
            provider_filters={
                "sources": ["SSA"],
                "target_cells": [
                    {
                        "variable": "social_security",
                        "geo_level": "national",
                        "domain_variable": None,
                    },
                    {
                        "variable": "social_security_retirement",
                        "geo_level": "national",
                        "domain_variable": None,
                    },
                    {
                        "variable": "ssi",
                        "geo_level": "national",
                        "domain_variable": None,
                    },
                ],
            },
        )
    )

    assert {target.metadata["target_id"] for target in target_set.targets} == {
        8,
        9,
        10,
    }
    assert {target.measure for target in target_set.targets} == {
        "social_security",
        "social_security_retirement",
        "ssi",
    }
    assert {target.entity for target in target_set.targets} == {EntityType.PERSON}


def test_arch_provider_maps_tanf_cash_assistance_target(tmp_path):
    db_path = tmp_path / "arch_targets.db"
    _create_arch_targets_db(db_path)
    conn = sqlite3.connect(db_path)
    conn.executemany(
        """
        INSERT INTO targets (
            id,
            stratum_id,
            variable,
            period,
            value,
            target_type,
            geographic_level,
            source,
            source_table,
            source_url,
            notes
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        [
            (
                8,
                1,
                "tanf_cash_assistance",
                2024,
                7_788_317_474.55,
                "AMOUNT",
                "NATIONAL",
                "HHS_ACF_TANF",
                "ACF TANF Financial Data",
                None,
                None,
            ),
            (
                9,
                1,
                "tanf_family_count",
                2024,
                841_208.67,
                "COUNT",
                "NATIONAL",
                "HHS_ACF_TANF",
                "ACF TANF Caseload Data",
                None,
                None,
            ),
        ],
    )
    conn.commit()
    conn.close()

    provider = ArchSQLiteTargetProvider(db_path)
    target_set = provider.load_target_set(
        TargetQuery(
            period=2024,
            provider_filters={
                "sources": ["HHS_ACF_TANF"],
                "target_cells": [
                    {
                        "variable": "tanf",
                        "geo_level": "national",
                        "domain_variable": None,
                    },
                    {
                        "variable": "spm_unit_count",
                        "geo_level": "national",
                        "domain_variable": "tanf",
                    },
                ],
            },
        )
    )

    assert [target.metadata["target_id"] for target in target_set.targets] == [8, 9]
    targets_by_id = {
        target.metadata["target_id"]: target for target in target_set.targets
    }
    assert targets_by_id[8].measure == "tanf"
    assert targets_by_id[8].entity is EntityType.SPM_UNIT
    assert targets_by_id[9].measure is None
    assert targets_by_id[9].metadata["variable"] == "spm_unit_count"
    assert targets_by_id[9].entity is EntityType.SPM_UNIT


def test_arch_provider_maps_w2_tip_income_without_source_year_labor_force(tmp_path):
    db_path = tmp_path / "arch_targets.db"
    _create_arch_targets_db(db_path)
    _insert_w2_tip_income_target(db_path)

    provider = ArchSQLiteTargetProvider(db_path)
    target_set = provider.load_target_set(
        TargetQuery(
            period=2024,
            provider_filters={
                "sources": ["IRS_SOI"],
                "variables": ["tip_income"],
            },
        )
    )

    assert len(target_set.targets) == 1
    target = target_set.targets[0]
    assert target.entity is EntityType.PERSON
    assert target.measure == "tip_income"
    assert target.aggregation is TargetAggregation.SUM
    assert target.value == pytest.approx(121.0)
    assert {
        (target_filter.feature, target_filter.operator.value, target_filter.value)
        for target_filter in target.filters
    } == {("tip_income", ">", "0")}
    assert target.metadata["arch_source_period"] == 2020
    assert target.metadata["arch_aging_count_method"] == "not_required"
    assert target.metadata["arch_aging_amount_method"] == (
        "soi_total_agi_last_growth_extrapolation"
    )


def test_arch_provider_maps_ira_contribution_targets(tmp_path):
    db_path = tmp_path / "arch_targets.db"
    _create_arch_targets_db(db_path)
    conn = sqlite3.connect(db_path)
    conn.executemany(
        """
        INSERT INTO strata (id, name, jurisdiction, definition_hash)
        VALUES (?, ?, ?, ?)
        """,
        [
            (12, "US taxpayers with traditional IRA contributions", "US", "trad_ira"),
            (13, "US taxpayers with Roth IRA contributions", "US", "roth_ira"),
        ],
    )
    conn.executemany(
        """
        INSERT INTO stratum_constraints (
            stratum_id,
            variable,
            operator,
            value
        ) VALUES (?, ?, ?, ?)
        """,
        [
            (12, "traditional_ira_contributions", ">", "0"),
            (13, "roth_ira_contributions", ">", "0"),
        ],
    )
    conn.executemany(
        """
        INSERT INTO targets (
            id,
            stratum_id,
            variable,
            period,
            value,
            target_type,
            geographic_level,
            source,
            source_table,
            source_url,
            notes
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        [
            (
                12,
                12,
                "traditional_ira_contributions",
                2022,
                50.0,
                "AMOUNT",
                "NATIONAL",
                "IRS_SOI",
                "IRA",
                None,
                None,
            ),
            (
                13,
                13,
                "roth_ira_contributions",
                2022,
                75.0,
                "AMOUNT",
                "NATIONAL",
                "IRS_SOI",
                "IRA",
                None,
                None,
            ),
        ],
    )
    conn.commit()
    conn.close()

    provider = ArchSQLiteTargetProvider(db_path)
    target_set = provider.load_target_set(
        TargetQuery(
            period=2024,
            provider_filters={
                "sources": ["IRS_SOI"],
                "target_cells": [
                    {
                        "variable": "traditional_ira_contributions",
                        "geo_level": "national",
                        "domain_variable": None,
                    },
                    {
                        "variable": "roth_ira_contributions",
                        "geo_level": "national",
                        "domain_variable": None,
                    },
                ],
            },
        )
    )

    assert {target.measure for target in target_set.targets} == {
        "traditional_ira_contributions",
        "roth_ira_contributions",
    }
    assert {target.entity for target in target_set.targets} == {EntityType.PERSON}
    assert {
        target.metadata["arch_aging_count_method"] for target in target_set.targets
    } == {"not_required"}


def test_us_pipeline_can_select_arch_target_provider(tmp_path):
    db_path = tmp_path / "arch_targets.db"
    _create_arch_targets_db(db_path)

    pipeline = USMicroplexPipeline(
        USMicroplexBuildConfig(
            arch_targets_db=str(db_path),
            calibration_target_source="arch",
        )
    )

    provider, source = pipeline._resolve_calibration_target_provider()

    assert source == "arch"
    assert isinstance(provider, ArchSQLiteTargetProvider)


def test_arch_target_profile_coverage_summarizes_custom_cells(tmp_path):
    db_path = tmp_path / "arch_targets.db"
    _create_arch_targets_db(db_path)

    provider = ArchSQLiteTargetProvider(db_path)
    report = summarize_arch_target_profile_coverage(
        provider,
        period=2024,
        profile_name="custom",
        target_cells=(
            PolicyEngineUSTargetCell(
                "adjusted_gross_income",
                geo_level="national",
                domain_variable=None,
            ),
            PolicyEngineUSTargetCell(
                "tax_unit_count",
                geo_level="national",
                domain_variable="tax_exempt_interest_income",
            ),
            PolicyEngineUSTargetCell(
                "employment_income",
                geo_level="national",
                domain_variable="employment_income",
            ),
        ),
    )

    assert report.target_cell_count == 3
    assert report.covered_cell_count == 2
    assert report.uncovered_cell_count == 1
    assert report.coverage_rate == pytest.approx(2 / 3)
    assert report.by_geo_level == {
        "national": {
            "target_cell_count": 3,
            "covered_cell_count": 2,
            "uncovered_cell_count": 1,
        }
    }
    assert report.by_variable["adjusted_gross_income"]["covered_cell_count"] == 1
    assert report.by_variable["employment_income"]["uncovered_cell_count"] == 1

    payload = report.to_dict()
    assert payload["profile_name"] == "custom"
    assert payload["cells"][0]["target_ids"] == [4]
    assert payload["cells"][1]["target_ids"] == [1]
    assert payload["cells"][2]["covered"] is False


def test_arch_target_profile_coverage_accepts_soi_itemized_domain(tmp_path):
    db_path = tmp_path / "arch_targets.db"
    _create_arch_targets_db(db_path)
    _insert_irs_soi_itemized_deduction_targets(db_path)

    provider = ArchSQLiteTargetProvider(db_path)
    report = summarize_arch_target_profile_coverage(
        provider,
        period=2024,
        profile_name="custom",
        target_cells=(
            PolicyEngineUSTargetCell(
                "medical_expense_deduction",
                geo_level="national",
                domain_variable="medical_expense_deduction",
            ),
            PolicyEngineUSTargetCell(
                "medical_expense_deduction",
                geo_level="national",
                domain_variable=("medical_expense_deduction,tax_unit_itemizes"),
            ),
            PolicyEngineUSTargetCell(
                "real_estate_taxes",
                geo_level="national",
                domain_variable="real_estate_taxes,tax_unit_itemizes",
            ),
            PolicyEngineUSTargetCell(
                "salt",
                geo_level="national",
                domain_variable="salt,tax_unit_itemizes",
            ),
            PolicyEngineUSTargetCell(
                "tax_unit_count",
                geo_level="national",
                domain_variable=("medical_expense_deduction,tax_unit_itemizes"),
            ),
            PolicyEngineUSTargetCell(
                "tax_unit_count",
                geo_level="national",
                domain_variable="real_estate_taxes,tax_unit_itemizes",
            ),
            PolicyEngineUSTargetCell(
                "tax_unit_count",
                geo_level="national",
                domain_variable="salt,tax_unit_itemizes",
            ),
        ),
    )

    assert report.target_cell_count == 7
    assert report.covered_cell_count == 7
    assert {
        (cell.cell["variable"], cell.cell["domain_variable"]): cell.target_ids
        for cell in report.cells
    } == {
        (
            "medical_expense_deduction",
            "medical_expense_deduction",
        ): (12,),
        (
            "medical_expense_deduction",
            "medical_expense_deduction,tax_unit_itemizes",
        ): (12,),
        (
            "real_estate_taxes",
            "real_estate_taxes,tax_unit_itemizes",
        ): (13,),
        ("salt", "salt,tax_unit_itemizes"): (14,),
        (
            "tax_unit_count",
            "medical_expense_deduction,tax_unit_itemizes",
        ): (15,),
        (
            "tax_unit_count",
            "real_estate_taxes,tax_unit_itemizes",
        ): (16,),
        ("tax_unit_count", "salt,tax_unit_itemizes"): (17,),
    }


def test_arch_target_profile_coverage_rolls_complete_state_targets_to_national(
    tmp_path,
):
    db_path = tmp_path / "arch_targets.db"
    _create_arch_targets_db(db_path)
    _insert_complete_state_rollup_targets(db_path)

    provider = ArchSQLiteTargetProvider(db_path)
    report = summarize_arch_target_profile_coverage(
        provider,
        period=2024,
        profile_name="custom",
        target_cells=(
            PolicyEngineUSTargetCell(
                "non_refundable_ctc",
                geo_level="national",
                domain_variable="non_refundable_ctc",
            ),
            PolicyEngineUSTargetCell(
                "non_refundable_ctc",
                geo_level="national",
                domain_variable="adjusted_gross_income,non_refundable_ctc",
            ),
            PolicyEngineUSTargetCell(
                "tax_unit_count",
                geo_level="national",
                domain_variable="non_refundable_ctc",
            ),
            PolicyEngineUSTargetCell(
                "tax_unit_count",
                geo_level="national",
                domain_variable="adjusted_gross_income,non_refundable_ctc",
            ),
            PolicyEngineUSTargetCell(
                "aca_ptc",
                geo_level="national",
                domain_variable="aca_ptc",
            ),
        ),
    )

    assert report.target_cell_count == 5
    assert report.covered_cell_count == 5
    target_set = provider.load_target_set(
        TargetQuery(
            period=2024,
            provider_filters={
                "target_cells": [cell.cell for cell in report.cells],
            },
        )
    )
    rollup_targets = {
        (target.measure or target.metadata["variable"], target.aggregation): target
        for target in target_set.targets
        if target.metadata["geo_level"] == "national"
        and str(target.metadata["target_id"]).startswith("-")
    }
    assert rollup_targets[
        ("non_refundable_ctc", TargetAggregation.SUM)
    ].value == pytest.approx(sum(1_000.0 + index for index in range(51)))
    assert rollup_targets[
        ("non_refundable_ctc", TargetAggregation.COUNT)
    ].value == pytest.approx(sum(100.0 + index for index in range(51)))
    assert rollup_targets[("aca_ptc", TargetAggregation.SUM)].value == pytest.approx(
        sum(10_000.0 + index for index in range(51))
    )


def test_arch_target_profile_coverage_reports_current_pe_profile(tmp_path):
    db_path = tmp_path / "arch_targets.db"
    _create_arch_targets_db(db_path)

    provider = ArchSQLiteTargetProvider(db_path)
    report = summarize_arch_target_profile_coverage(
        provider,
        period=2024,
        profile_name="pe_native_broad",
    )

    assert report.target_cell_count == 189
    assert report.covered_cell_count == 4
    assert report.uncovered_cell_count == 185
    assert report.by_geo_level["national"]["covered_cell_count"] == 3
    assert report.by_geo_level["state"]["covered_cell_count"] == 1

    covered_cells = {
        (
            cell.cell["variable"],
            cell.cell["geo_level"],
            cell.cell["domain_variable"],
        ): cell.target_ids
        for cell in report.cells
        if cell.covered
    }
    assert covered_cells == {
        ("adjusted_gross_income", "national", None): (4,),
        (
            "tax_exempt_interest_income",
            "national",
            "tax_exempt_interest_income",
        ): (2,),
        ("tax_unit_count", "national", "tax_exempt_interest_income"): (1,),
        ("tax_unit_count", "state", "adjusted_gross_income"): (7,),
    }


def test_arch_target_gap_queue_describes_missing_loader_rows(tmp_path):
    db_path = tmp_path / "arch_targets.db"
    _create_arch_targets_db(db_path)

    provider = ArchSQLiteTargetProvider(db_path)
    report = summarize_arch_target_gap_queue(
        provider,
        period=2024,
        profile_name="custom",
        target_cells=(
            PolicyEngineUSTargetCell(
                "employment_income",
                geo_level="national",
                domain_variable="employment_income",
            ),
            PolicyEngineUSTargetCell(
                "tax_unit_count",
                geo_level="national",
                domain_variable="employment_income",
            ),
        ),
    )

    assert report.row_count == 2
    assert report.covered_row_count == 0
    assert report.uncovered_row_count == 2
    assert report.by_loader_status == {"missing_arch_target_record": 2}
    assert report.by_gap_category == {"ready_primary_loader": 2}

    rows_by_variable = {row.variable: row for row in report.rows}
    amount_row = rows_by_variable["employment_income"]
    assert amount_row.priority == 1
    assert amount_row.expected_source == "IRS_SOI"
    assert amount_row.expected_source_table == "IRS SOI Publication 1304 Table 1.4"
    assert amount_row.expected_arch_variable == "wages_salaries_amount"
    assert amount_row.expected_target_type == "AMOUNT"
    assert amount_row.expected_entity == "person"
    assert amount_row.expected_aggregation == "sum"
    assert amount_row.gap_category == "ready_primary_loader"
    assert amount_row.expected_filters == (
        {
            "kind": "domain",
            "feature": "employment_income",
            "operator": ">",
            "value": 0,
        },
    )
    assert amount_row.agent_task_kind == "add_arch_source_loader_or_target_record"

    count_row = rows_by_variable["tax_unit_count"]
    assert count_row.expected_arch_variable == "wages_salaries_returns"
    assert count_row.expected_target_type == "COUNT"
    assert count_row.expected_entity == "tax_unit"
    assert count_row.expected_aggregation == "count"
    assert count_row.gap_category == "ready_primary_loader"


def test_arch_target_gap_queue_points_full_population_amounts_to_bea(tmp_path):
    db_path = tmp_path / "arch_targets.db"
    _create_arch_targets_db(db_path)

    provider = ArchSQLiteTargetProvider(db_path)
    report = summarize_arch_target_gap_queue(
        provider,
        period=2024,
        profile_name="custom",
        target_cells=(
            PolicyEngineUSTargetCell(
                "employment_income",
                geo_level="national",
                domain_variable=None,
            ),
            PolicyEngineUSTargetCell(
                "self_employment_income",
                geo_level="national",
                domain_variable=None,
            ),
            PolicyEngineUSTargetCell(
                "dividend_income",
                geo_level="national",
                domain_variable=None,
            ),
            PolicyEngineUSTargetCell(
                "self_employment_income",
                geo_level="state",
                domain_variable=None,
            ),
        ),
    )

    rows_by_cell = {(row.variable, row.geo_level): row for row in report.rows}
    assert rows_by_cell[("employment_income", "national")].expected_source == "BEA"
    assert rows_by_cell[("employment_income", "national")].expected_arch_variable == (
        "wages_salaries_amount"
    )
    assert rows_by_cell[("employment_income", "national")].expected_source_table == (
        "BEA NIPA annual total wages and salaries"
    )
    assert (
        rows_by_cell[("self_employment_income", "national")].expected_source
        == "IRS_SOI"
    )
    assert rows_by_cell[
        ("self_employment_income", "national")
    ].expected_arch_variable == ("schedule_c_income_amount")
    assert rows_by_cell[
        ("self_employment_income", "national")
    ].expected_source_table == ("IRS SOI Publication 1304")
    assert rows_by_cell[("dividend_income", "national")].expected_source == "BEA"
    assert rows_by_cell[("dividend_income", "national")].expected_arch_variable == (
        "personal_dividend_income_amount"
    )
    assert (
        rows_by_cell[("self_employment_income", "state")].expected_source == "IRS_SOI"
    )
    assert rows_by_cell[("self_employment_income", "state")].expected_source_table == (
        "IRS SOI Publication 1304"
    )


def test_arch_target_gap_queue_marks_multi_domain_rows_for_review(tmp_path):
    db_path = tmp_path / "arch_targets.db"
    _create_arch_targets_db(db_path)

    provider = ArchSQLiteTargetProvider(db_path)
    report = summarize_arch_target_gap_queue(
        provider,
        period=2024,
        profile_name="custom",
        target_cells=(
            PolicyEngineUSTargetCell(
                "tax_unit_count",
                geo_level="national",
                domain_variable="adjusted_gross_income,medical_expense_deduction",
            ),
        ),
    )

    row = report.rows[0]
    assert row.expected_source == "IRS_SOI"
    assert row.expected_arch_variable is None
    assert row.loader_status == "needs_source_mapping_review"
    assert row.gap_category == "source_mapping_review"
    assert row.agent_task_kind == "review_source_mapping"
    assert "multi-domain cells" in row.notes


def test_arch_target_gap_queue_points_eitc_child_rows_to_soi_table_2(tmp_path):
    db_path = tmp_path / "arch_targets.db"
    _create_arch_targets_db(db_path)

    provider = ArchSQLiteTargetProvider(db_path)
    report = summarize_arch_target_gap_queue(
        provider,
        period=2024,
        profile_name="custom",
        target_cells=(
            PolicyEngineUSTargetCell(
                "tax_unit_count",
                geo_level="national",
                domain_variable="eitc_child_count",
            ),
            PolicyEngineUSTargetCell(
                "tax_unit_count",
                geo_level="national",
                domain_variable="adjusted_gross_income,eitc,eitc_child_count",
            ),
        ),
    )

    assert {row.expected_arch_variable for row in report.rows} == {"eitc_claims"}
    assert {row.expected_source_table for row in report.rows} == {
        "IRS SOI Historic Table 2"
    }
    assert {row.expected_target_type for row in report.rows} == {"COUNT"}
    assert {row.loader_status for row in report.rows} == {"missing_arch_target_record"}
    assert {row.gap_category for row in report.rows} == {"ready_primary_loader"}


def test_arch_target_gap_queue_points_aca_ptc_counts_to_soi_table_2(tmp_path):
    db_path = tmp_path / "arch_targets.db"
    _create_arch_targets_db(db_path)

    provider = ArchSQLiteTargetProvider(db_path)
    report = summarize_arch_target_gap_queue(
        provider,
        period=2024,
        profile_name="custom",
        target_cells=(
            PolicyEngineUSTargetCell(
                "tax_unit_count",
                geo_level="national",
                domain_variable="used_aca_ptc",
            ),
        ),
    )

    row = report.rows[0]
    assert row.expected_source == "IRS_SOI"
    assert row.expected_source_table == "IRS SOI Historic Table 2"
    assert row.expected_arch_variable == "aca_ptc_returns"
    assert row.expected_target_type == "COUNT"
    assert row.expected_entity == "tax_unit"
    assert row.loader_status == "missing_arch_target_record"
    assert row.gap_category == "ready_primary_loader"


def test_arch_target_gap_queue_points_income_tax_return_counts_to_soi_table_2(
    tmp_path,
):
    db_path = tmp_path / "arch_targets.db"
    _create_arch_targets_db(db_path)

    provider = ArchSQLiteTargetProvider(db_path)
    report = summarize_arch_target_gap_queue(
        provider,
        period=2024,
        profile_name="custom",
        target_cells=(
            PolicyEngineUSTargetCell(
                "tax_unit_count",
                geo_level="national",
                domain_variable=(
                    "adjusted_gross_income,income_tax_before_credits"
                ),
            ),
            PolicyEngineUSTargetCell(
                "tax_unit_count",
                geo_level="national",
                domain_variable=(
                    "adjusted_gross_income,filing_status,"
                    "income_tax_before_credits"
                ),
            ),
        ),
    )

    assert {row.expected_source for row in report.rows} == {"IRS_SOI"}
    assert {row.expected_source_table for row in report.rows} == {
        "IRS SOI Historic Table 2"
    }
    assert {row.expected_arch_variable for row in report.rows} == {
        "income_tax_before_credits_returns"
    }
    assert {row.expected_target_type for row in report.rows} == {"COUNT"}
    assert {row.expected_entity for row in report.rows} == {"tax_unit"}


def test_arch_target_gap_queue_points_energy_subsidy_households_to_liheap(
    tmp_path,
):
    db_path = tmp_path / "arch_targets.db"
    _create_arch_targets_db(db_path)

    provider = ArchSQLiteTargetProvider(db_path)
    report = summarize_arch_target_gap_queue(
        provider,
        period=2024,
        profile_name="custom",
        target_cells=(
            PolicyEngineUSTargetCell(
                "household_count",
                geo_level="national",
                domain_variable="spm_unit_energy_subsidy_reported",
            ),
        ),
    )

    row = report.rows[0]
    assert row.expected_source == "HHS_ACF_LIHEAP"
    assert row.expected_source_table == "HHS ACF LIHEAP National Profile"
    assert row.expected_arch_variable == "liheap_household_count"
    assert row.expected_target_type == "COUNT"
    assert row.expected_entity == "household"
    assert row.loader_status == "missing_arch_target_record"
    assert row.gap_category == "ready_primary_loader"


def test_arch_target_gap_queue_points_retirement_contributions_to_soi(
    tmp_path,
):
    db_path = tmp_path / "arch_targets.db"
    _create_arch_targets_db(db_path)

    provider = ArchSQLiteTargetProvider(db_path)
    report = summarize_arch_target_gap_queue(
        provider,
        period=2024,
        profile_name="custom",
        target_cells=(
            PolicyEngineUSTargetCell(
                "traditional_401k_contributions",
                geo_level="national",
            ),
            PolicyEngineUSTargetCell("roth_401k_contributions", geo_level="national"),
            PolicyEngineUSTargetCell(
                "self_employed_pension_contribution_ald",
                geo_level="national",
            ),
        ),
    )

    rows_by_variable = {row.variable: row for row in report.rows}
    traditional = rows_by_variable["traditional_401k_contributions"]
    roth = rows_by_variable["roth_401k_contributions"]
    self_employed = rows_by_variable["self_employed_pension_contribution_ald"]

    assert {row.expected_source for row in report.rows} == {"IRS_SOI"}
    assert traditional.expected_source_table == "IRS SOI Form W-2 Statistics Table 4.B"
    assert traditional.expected_arch_variable == "traditional_401k_contributions"
    assert traditional.expected_entity == "person"
    assert roth.expected_source_table == "IRS SOI Form W-2 Statistics Table 4.B"
    assert roth.expected_arch_variable == "roth_401k_contributions"
    assert roth.expected_entity == "person"
    assert self_employed.expected_source_table == (
        "IRS SOI Publication 1304 Table 1.4"
    )
    assert self_employed.expected_arch_variable == (
        "self_employed_pension_contribution_ald"
    )
    assert self_employed.expected_entity == "tax_unit"
    assert {row.gap_category for row in report.rows} == {"ready_primary_loader"}


def test_arch_target_gap_queue_points_agi_person_counts_to_soi_table_2(tmp_path):
    db_path = tmp_path / "arch_targets.db"
    _create_arch_targets_db(db_path)

    provider = ArchSQLiteTargetProvider(db_path)
    report = summarize_arch_target_gap_queue(
        provider,
        period=2024,
        profile_name="custom",
        target_cells=(
            PolicyEngineUSTargetCell(
                "person_count",
                geo_level="state",
                domain_variable="adjusted_gross_income",
            ),
        ),
    )

    row = report.rows[0]
    assert row.expected_source == "IRS_SOI"
    assert row.expected_source_table == "IRS SOI Historic Table 2"
    assert row.expected_arch_variable == "tax_filer_individual_count"
    assert row.expected_target_type == "COUNT"
    assert row.expected_entity == "person"
    assert row.loader_status == "missing_arch_target_record"
    assert row.gap_category == "ready_primary_loader"


def test_arch_target_gap_queue_points_state_income_tax_to_census_stc(tmp_path):
    db_path = tmp_path / "arch_targets.db"
    _create_arch_targets_db(db_path)

    provider = ArchSQLiteTargetProvider(db_path)
    report = summarize_arch_target_gap_queue(
        provider,
        period=2024,
        profile_name="custom",
        target_cells=(PolicyEngineUSTargetCell("state_income_tax", geo_level="state"),),
    )

    row = report.rows[0]
    assert row.expected_source == "CENSUS_STC"
    assert row.expected_source_table == "Census State Tax Collections item T40"
    assert row.expected_arch_variable == "state_individual_income_tax_collections"
    assert row.expected_target_type == "AMOUNT"
    assert row.expected_entity == "tax_unit"
    assert row.loader_status == "missing_arch_target_record"
    assert row.gap_category == "ready_primary_loader"


def test_arch_target_gap_queue_points_itemized_deductions_to_soi_table_2(tmp_path):
    db_path = tmp_path / "arch_targets.db"
    _create_arch_targets_db(db_path)

    provider = ArchSQLiteTargetProvider(db_path)
    report = summarize_arch_target_gap_queue(
        provider,
        period=2024,
        profile_name="custom",
        target_cells=(
            PolicyEngineUSTargetCell("salt_deduction", geo_level="national"),
            PolicyEngineUSTargetCell("interest_deduction", geo_level="national"),
            PolicyEngineUSTargetCell(
                "tax_unit_count",
                geo_level="national",
                domain_variable="salt,tax_unit_itemizes",
            ),
        ),
    )

    rows_by_variable = {row.variable: row for row in report.rows}
    rows_by_cell = {(row.variable, row.domain_variable): row for row in report.rows}
    salt_row = rows_by_variable["salt_deduction"]
    assert salt_row.expected_source == "IRS_SOI"
    assert salt_row.expected_source_table == "IRS SOI Historic Table 2"
    assert salt_row.expected_arch_variable == "limited_state_local_taxes_amount"
    assert salt_row.expected_target_type == "AMOUNT"
    assert salt_row.expected_entity == "tax_unit"
    assert salt_row.loader_status == "missing_arch_target_record"
    assert salt_row.gap_category == "ready_primary_loader"

    interest_row = rows_by_variable["interest_deduction"]
    assert interest_row.expected_source == "IRS_SOI"
    assert interest_row.expected_source_table == "IRS SOI Historic Table 2"
    assert interest_row.expected_arch_variable == "interest_paid_deduction_amount"
    assert interest_row.expected_target_type == "AMOUNT"
    assert interest_row.expected_entity == "tax_unit"
    assert interest_row.loader_status == "missing_arch_target_record"
    assert interest_row.gap_category == "ready_primary_loader"

    salt_count_row = rows_by_cell[("tax_unit_count", "salt,tax_unit_itemizes")]
    assert salt_count_row.expected_source == "IRS_SOI"
    assert salt_count_row.expected_source_table == (
        "IRS SOI itemized deduction or credit tables"
    )
    assert salt_count_row.expected_arch_variable == "salt_claims"
    assert salt_count_row.expected_target_type == "COUNT"
    assert salt_count_row.expected_entity == "tax_unit"
    assert salt_count_row.loader_status == "missing_arch_target_record"
    assert salt_count_row.gap_category == "ready_primary_loader"


def test_arch_target_gap_queue_points_income_tax_positive_to_soi_liability(tmp_path):
    db_path = tmp_path / "arch_targets.db"
    _create_arch_targets_db(db_path)

    provider = ArchSQLiteTargetProvider(db_path)
    report = summarize_arch_target_gap_queue(
        provider,
        period=2024,
        profile_name="custom",
        target_cells=(
            PolicyEngineUSTargetCell("income_tax_positive", geo_level="national"),
        ),
    )

    row = report.rows[0]
    assert row.expected_source == "IRS_SOI"
    assert row.expected_source_table == (
        "IRS SOI Publication 1304 Table 1.1 or Historic Table 2"
    )
    assert row.expected_arch_variable == "income_tax_liability"
    assert row.expected_target_type == "AMOUNT"
    assert row.expected_entity == "tax_unit"
    assert row.loader_status == "missing_arch_target_record"
    assert row.gap_category == "ready_primary_loader"


def test_arch_target_gap_queue_deprioritizes_survey_or_model_inputs(tmp_path):
    db_path = tmp_path / "arch_targets.db"
    _create_arch_targets_db(db_path)

    provider = ArchSQLiteTargetProvider(db_path)
    report = summarize_arch_target_gap_queue(
        provider,
        period=2024,
        profile_name="custom",
        target_cells=(
            PolicyEngineUSTargetCell("rent", geo_level="national"),
            PolicyEngineUSTargetCell(
                "person_count",
                geo_level="national",
                domain_variable="ssn_card_type",
            ),
        ),
    )

    assert report.by_gap_category == {"survey_or_model_input_deprioritized": 2}
    assert {row.gap_category for row in report.rows} == {
        "survey_or_model_input_deprioritized"
    }
    assert {row.agent_task_kind for row in report.rows} == {
        "defer_or_review_non_primary_source"
    }
    assert all(
        "survey/model-input proxy deprioritized" in row.notes for row in report.rows
    )


def test_arch_target_gap_queue_classifies_loaded_wrong_geography(tmp_path):
    db_path = tmp_path / "arch_targets.db"
    _create_arch_targets_db(db_path)

    provider = ArchSQLiteTargetProvider(db_path)
    report = summarize_arch_target_gap_queue(
        provider,
        period=2024,
        profile_name="custom",
        target_cells=(
            PolicyEngineUSTargetCell(
                "tax_unit_count",
                geo_level="national",
                domain_variable="adjusted_gross_income",
            ),
        ),
    )

    row = report.rows[0]
    assert row.expected_source == "IRS_SOI"
    assert row.expected_arch_variable == "tax_unit_count"
    assert row.loader_status == "loaded_arch_variable_missing_geography"
    assert row.gap_category == "ready_rollup_or_geography"
    assert row.agent_task_kind == "add_arch_rollup_or_geography_records"


def test_arch_target_gap_queue_can_include_covered_rows(tmp_path):
    db_path = tmp_path / "arch_targets.db"
    _create_arch_targets_db(db_path)

    provider = ArchSQLiteTargetProvider(db_path)
    report = summarize_arch_target_gap_queue(
        provider,
        period=2024,
        profile_name="custom",
        include_covered=True,
        target_cells=(
            PolicyEngineUSTargetCell(
                "adjusted_gross_income",
                geo_level="national",
                domain_variable=None,
            ),
        ),
    )

    assert report.row_count == 1
    row = report.rows[0]
    assert row.covered is True
    assert row.target_ids == (4,)
    assert row.expected_filters == ()
    assert row.loader_status == "covered"
    assert row.gap_category == "covered"
    assert row.agent_task_kind == "none"


def test_arch_target_gap_queue_cli_writes_csv(tmp_path):
    db_path = tmp_path / "arch_targets.db"
    output_path = tmp_path / "gaps.csv"
    _create_arch_targets_db(db_path)

    exit_code = main_gaps(
        [
            "--arch-targets-db",
            str(db_path),
            "--period",
            "2024",
            "--profile",
            "pe_native_broad",
            "--format",
            "csv",
            "--output",
            str(output_path),
        ]
    )

    assert exit_code == 0
    text = output_path.read_text()
    assert text.startswith("priority,profile_name,period,variable")
    assert "gap_category" in text
    assert "employment_income" in text
    assert "missing_arch_target_record" in text


def test_arch_target_refresh_cli_discovers_artifact_and_writes_snapshot(tmp_path):
    artifact_root = tmp_path / "artifacts"
    artifact_root.mkdir()
    db_path = artifact_root / "arch_targets_fixture.db"
    output_dir = tmp_path / "snapshot"
    _create_arch_targets_db(db_path)

    exit_code = main_refresh(
        [
            "--artifact-root",
            str(artifact_root),
            "--period",
            "2024",
            "--profile",
            "pe_native_broad",
            "--output-dir",
            str(output_dir),
        ]
    )

    assert exit_code == 0

    coverage_path = output_dir / "pe_native_broad_2024_coverage.json"
    gaps_json_path = output_dir / "pe_native_broad_2024_gaps.json"
    gaps_csv_path = output_dir / "pe_native_broad_2024_gaps.csv"
    summary_path = output_dir / "pe_native_broad_2024_summary.md"

    coverage = json.loads(coverage_path.read_text())
    gaps = json.loads(gaps_json_path.read_text())
    gaps_csv = gaps_csv_path.read_text()
    summary = summary_path.read_text()

    assert coverage["target_cell_count"] == 189
    assert coverage["covered_cell_count"] == 4
    assert gaps["uncovered_row_count"] == 185
    assert gaps_csv.startswith("priority,profile_name,period,variable")
    assert "Coverage rate" in summary
    assert str(db_path.resolve()) in summary
