from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any

import pytest
from microplex.targets import TargetQuery

import microplex_us.targets.arch as arch_module
from microplex_us.pipelines.us import USMicroplexBuildConfig, USMicroplexPipeline
from microplex_us.policyengine.target_profiles import PolicyEngineUSTargetCell
from microplex_us.targets import (
    ArchCompositeSQLiteTargetProvider,
    ArchConsumerFactJSONLTargetProvider,
    ArchFactSQLiteTargetProvider,
    ArchSQLiteTargetProvider,
    resolve_arch_sqlite_target_provider,
    summarize_arch_target_gap_queue,
    summarize_arch_target_profile_coverage,
)
from microplex_us.targets.arch import main_parity, main_smoke


def _create_value_constraint_target_db(path: Path) -> None:
    conn = sqlite3.connect(path)
    conn.executescript(
        """
        CREATE TABLE strata (
            id INTEGER PRIMARY KEY,
            name TEXT,
            jurisdiction TEXT,
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
            (1, "US All Filers", "US", "all"),
            (2, "US Filers AGI 1_to_5k", "US", "1_to_5k"),
        ],
    )
    conn.executemany(
        """
        INSERT INTO stratum_constraints (stratum_id, variable, operator, value)
        VALUES (?, ?, ?, ?)
        """,
        [
            (1, "is_tax_filer", "==", "1"),
            (2, "is_tax_filer", "==", "1"),
            (2, "adjusted_gross_income", ">=", "1"),
            (2, "adjusted_gross_income", "<", "5000"),
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
                1,
                "tax_unit_count",
                2023,
                160_602_107,
                "COUNT",
                "NATIONAL",
                "IRS_SOI",
                "Publication 1304 Table 1.1",
                "https://www.irs.gov/pub/irs-soi/23in11si.xls",
                None,
            ),
            (
                2,
                1,
                "adjusted_gross_income",
                2023,
                15_286_017_359_000,
                "AMOUNT",
                "NATIONAL",
                "IRS_SOI",
                "Publication 1304 Table 1.1",
                "https://www.irs.gov/pub/irs-soi/23in11si.xls",
                None,
            ),
            (
                3,
                1,
                "income_tax_liability",
                2023,
                2_147_909_818_000,
                "AMOUNT",
                "NATIONAL",
                "IRS_SOI",
                "Publication 1304 Table 1.1",
                "https://www.irs.gov/pub/irs-soi/23in11si.xls",
                None,
            ),
            (
                4,
                2,
                "tax_unit_count",
                2023,
                7_357_751,
                "COUNT",
                "NATIONAL",
                "IRS_SOI",
                "Publication 1304 Table 1.1",
                "https://www.irs.gov/pub/irs-soi/23in11si.xls",
                None,
            ),
            (
                5,
                2,
                "adjusted_gross_income",
                2023,
                20_372_694_000,
                "AMOUNT",
                "NATIONAL",
                "IRS_SOI",
                "Publication 1304 Table 1.1",
                "https://www.irs.gov/pub/irs-soi/23in11si.xls",
                None,
            ),
        ],
    )
    conn.commit()
    conn.close()


def _create_arch_fact_db(path: Path) -> None:
    conn = sqlite3.connect(path)
    conn.executescript(
        """
        CREATE TABLE aggregate_facts (
            fact_key TEXT PRIMARY KEY,
            source_record_id TEXT,
            value_numeric REAL,
            value_text TEXT,
            value_json TEXT NOT NULL,
            period_value TEXT NOT NULL,
            geography_level TEXT NOT NULL,
            geography_id TEXT NOT NULL,
            geography_name TEXT,
            measure_concept TEXT NOT NULL,
            measure_source_concept TEXT,
            measure_concept_relation TEXT,
            measure_concept_authority TEXT,
            measure_concept_evidence_url TEXT,
            measure_concept_evidence_notes TEXT,
            measure_legal_vintage TEXT,
            measure_unit TEXT NOT NULL,
            aggregation_method TEXT NOT NULL,
            domain TEXT NOT NULL,
            filters_json TEXT NOT NULL,
            label TEXT,
            source_name TEXT,
            source_table TEXT,
            source_url TEXT,
            source_method_notes TEXT
        );

        CREATE TABLE aggregate_constraints (
            fact_key TEXT NOT NULL,
            ordinal INTEGER NOT NULL,
            variable TEXT NOT NULL,
            operator TEXT NOT NULL,
            value_text TEXT,
            value_numeric REAL,
            value_json TEXT NOT NULL,
            unit TEXT,
            role TEXT NOT NULL,
            label TEXT,
            PRIMARY KEY (fact_key, ordinal)
        );

        CREATE TABLE fact_source_cells (
            fact_key TEXT NOT NULL,
            source_cell_key TEXT NOT NULL,
            ordinal INTEGER NOT NULL,
            PRIMARY KEY (fact_key, source_cell_key)
        );

        CREATE TABLE fact_source_rows (
            fact_key TEXT NOT NULL,
            source_row_key TEXT NOT NULL,
            ordinal INTEGER NOT NULL,
            PRIMARY KEY (fact_key, source_row_key)
        );
        """
    )

    def fact(
        key: str,
        *,
        concept: str,
        value: float,
        aggregation: str,
        income_range: str,
        unit: str,
        source_concept: str | None = None,
    ) -> tuple[Any, ...]:
        return (
            key,
            f"irs_soi.ty2023.table_1_1.{income_range}.{concept.rsplit('.', 1)[-1]}",
            value,
            str(int(value)) if float(value).is_integer() else str(value),
            json.dumps(value),
            "2023",
            "country",
            "0100000US",
            "United States",
            concept,
            source_concept,
            "exact" if source_concept else None,
            "arch-us" if source_concept else None,
            "https://uscode.house.gov/view.xhtml?req=(title:26%20section:62%20edition:prelim)"
            if source_concept
            else None,
            "IRS SOI Table 1.1 reports adjusted gross income.",
            "tax_year_2023" if source_concept else None,
            unit,
            aggregation,
            "all_individual_income_tax_returns",
            json.dumps({"filing_status": "all", "income_range": income_range}),
            f"{income_range} {concept}",
            "irs_soi",
            "Publication 1304 Table 1.1",
            "https://www.irs.gov/pub/irs-soi/23in11si.xls",
            "Source-package aggregate fact fixture.",
        )

    conn.executemany(
        """
        INSERT INTO aggregate_facts (
            fact_key,
            source_record_id,
            value_numeric,
            value_text,
            value_json,
            period_value,
            geography_level,
            geography_id,
            geography_name,
            measure_concept,
            measure_source_concept,
            measure_concept_relation,
            measure_concept_authority,
            measure_concept_evidence_url,
            measure_concept_evidence_notes,
            measure_legal_vintage,
            measure_unit,
            aggregation_method,
            domain,
            filters_json,
            label,
            source_name,
            source_table,
            source_url,
            source_method_notes
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        [
            fact(
                "arch.fact.v1:all-count",
                concept="irs_soi.individual_income_tax_returns",
                value=160_602_107,
                aggregation="count",
                income_range="all",
                unit="count",
            ),
            fact(
                "arch.fact.v1:all-agi",
                concept="us:statutes/26/62#adjusted_gross_income",
                source_concept="irs_soi.adjusted_gross_income",
                value=15_286_017_359_000,
                aggregation="sum",
                income_range="all",
                unit="usd",
            ),
            fact(
                "arch.fact.v1:all-tax",
                concept="irs_soi.total_income_tax",
                value=2_147_909_818_000,
                aggregation="sum",
                income_range="all",
                unit="usd",
            ),
            fact(
                "arch.fact.v1:1-to-5k-count",
                concept="irs_soi.individual_income_tax_returns",
                value=7_357_751,
                aggregation="count",
                income_range="1_to_5k",
                unit="count",
            ),
            fact(
                "arch.fact.v1:1-to-5k-agi",
                concept="us:statutes/26/62#adjusted_gross_income",
                source_concept="irs_soi.adjusted_gross_income",
                value=20_372_694_000,
                aggregation="sum",
                income_range="1_to_5k",
                unit="usd",
            ),
        ],
    )
    conn.executemany(
        """
        INSERT INTO aggregate_constraints (
            fact_key,
            ordinal,
            variable,
            operator,
            value_text,
            value_numeric,
            value_json,
            unit,
            role,
            label
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        [
            (
                key,
                ordinal,
                "us:statutes/26/62#adjusted_gross_income",
                operator,
                str(value),
                float(value),
                json.dumps(value),
                "usd",
                "filter",
                "Adjusted gross income bound",
            )
            for key in ("arch.fact.v1:1-to-5k-count", "arch.fact.v1:1-to-5k-agi")
            for ordinal, operator, value in ((0, ">=", 1), (1, "<", 5000))
        ],
    )
    conn.executemany(
        """
        INSERT INTO fact_source_cells (fact_key, source_cell_key, ordinal)
        VALUES (?, ?, ?)
        """,
        [
            ("arch.fact.v1:all-agi", "arch.source_cell.v1:agi", 0),
            ("arch.fact.v1:all-count", "arch.source_cell.v1:count", 0),
        ],
    )
    conn.execute(
        """
        INSERT INTO fact_source_rows (fact_key, source_row_key, ordinal)
        VALUES (?, ?, ?)
        """,
        ("arch.fact.v1:all-agi", "arch.source_row.v1:all", 0),
    )
    conn.commit()
    conn.close()


def _insert_arch_table_1_1_reference_totals(
    path: Path,
    *,
    year: int,
    return_count: float,
    adjusted_gross_income: float,
) -> None:
    conn = sqlite3.connect(path)

    def fact(
        key: str,
        *,
        concept: str,
        value: float,
        aggregation: str,
        unit: str,
        source_concept: str | None = None,
    ) -> tuple[Any, ...]:
        return (
            key,
            f"irs_soi.ty{year}.table_1_1.all.{concept.rsplit('.', 1)[-1]}",
            value,
            str(int(value)) if float(value).is_integer() else str(value),
            json.dumps(value),
            str(year),
            "country",
            "0100000US",
            "United States",
            concept,
            source_concept,
            "exact" if source_concept else None,
            "arch-us" if source_concept else None,
            "https://uscode.house.gov/view.xhtml?req=(title:26%20section:62%20edition:prelim)"
            if source_concept
            else None,
            "IRS SOI Table 1.1 reports adjusted gross income.",
            f"tax_year_{year}" if source_concept else None,
            unit,
            aggregation,
            "all_individual_income_tax_returns",
            json.dumps({"filing_status": "all", "income_range": "all"}),
            f"{year} all {concept}",
            "irs_soi",
            "Publication 1304 Table 1.1",
            f"https://www.irs.gov/pub/irs-soi/{str(year)[-2:]}in11si.xls",
            "Source-package aggregate fact aging reference fixture.",
        )

    conn.executemany(
        """
        INSERT INTO aggregate_facts (
            fact_key,
            source_record_id,
            value_numeric,
            value_text,
            value_json,
            period_value,
            geography_level,
            geography_id,
            geography_name,
            measure_concept,
            measure_source_concept,
            measure_concept_relation,
            measure_concept_authority,
            measure_concept_evidence_url,
            measure_concept_evidence_notes,
            measure_legal_vintage,
            measure_unit,
            aggregation_method,
            domain,
            filters_json,
            label,
            source_name,
            source_table,
            source_url,
            source_method_notes
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        [
            fact(
                f"arch.fact.v1:{year}-all-count",
                concept="irs_soi.individual_income_tax_returns",
                value=return_count,
                aggregation="count",
                unit="count",
            ),
            fact(
                f"arch.fact.v1:{year}-all-agi",
                concept="us:statutes/26/62#adjusted_gross_income",
                source_concept="irs_soi.adjusted_gross_income",
                value=adjusted_gross_income,
                aggregation="sum",
                unit="usd",
            ),
        ],
    )
    conn.commit()
    conn.close()


def _insert_arch_table_1_4_facts(path: Path) -> None:
    conn = sqlite3.connect(path)

    def fact(
        key: str,
        *,
        concept: str,
        value: float,
        aggregation: str,
        income_range: str,
        unit: str,
        source_concept: str | None = None,
        concept_relation: str | None = None,
    ) -> tuple[Any, ...]:
        slug = concept.split("#")[-1].rsplit(".", 1)[-1].replace(":", "_")
        return (
            key,
            f"irs_soi.ty2023.table_1_4.{income_range}.{slug}",
            value,
            str(int(value)) if float(value).is_integer() else str(value),
            json.dumps(value),
            "2023",
            "country",
            "0100000US",
            "United States",
            concept,
            source_concept,
            concept_relation,
            "arch-us" if concept_relation else None,
            "https://www.irs.gov/statistics/soi-tax-stats-individual-income-tax-returns-complete-report-publication-1304-basic-tables-part-1"
            if concept_relation
            else None,
            "SOI Table 1.4 source concept alignment fixture."
            if concept_relation
            else None,
            "tax_year_2023" if concept_relation else None,
            unit,
            aggregation,
            "all_individual_income_tax_returns",
            json.dumps({"filing_status": "all", "income_range": income_range}),
            f"{income_range} {concept}",
            "irs_soi",
            "Publication 1304 Table 1.4",
            "https://www.irs.gov/pub/irs-soi/23in14ar.xls",
            "Source-package aggregate fact fixture.",
        )

    conn.executemany(
        """
        INSERT INTO aggregate_facts (
            fact_key,
            source_record_id,
            value_numeric,
            value_text,
            value_json,
            period_value,
            geography_level,
            geography_id,
            geography_name,
            measure_concept,
            measure_source_concept,
            measure_concept_relation,
            measure_concept_authority,
            measure_concept_evidence_url,
            measure_concept_evidence_notes,
            measure_legal_vintage,
            measure_unit,
            aggregation_method,
            domain,
            filters_json,
            label,
            source_name,
            source_table,
            source_url,
            source_method_notes
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        [
            fact(
                "arch.fact.v1:t14-all-wages-returns",
                concept="irs_soi.returns_with_total_wages",
                value=132_000_000,
                aggregation="count",
                income_range="all",
                unit="count",
            ),
            fact(
                "arch.fact.v1:t14-all-wages-amount",
                concept="us:statutes/26/62#input.wages",
                source_concept="irs_soi.total_wages",
                concept_relation="broad_match",
                value=10_500_000_000_000,
                aggregation="sum",
                income_range="all",
                unit="usd",
            ),
            fact(
                "arch.fact.v1:t14-all-capital-gains-returns",
                concept="irs_soi.returns_with_taxable_net_capital_gains",
                value=27_000_000,
                aggregation="count",
                income_range="all",
                unit="count",
            ),
            fact(
                "arch.fact.v1:t14-all-capital-gains-amount",
                concept="irs_soi.taxable_net_capital_gains",
                value=1_100_000_000_000,
                aggregation="sum",
                income_range="all",
                unit="usd",
            ),
            fact(
                "arch.fact.v1:t14-all-ira-returns",
                concept="irs_soi.returns_with_taxable_ira_distributions",
                value=18_000_000,
                aggregation="count",
                income_range="all",
                unit="count",
            ),
            fact(
                "arch.fact.v1:t14-all-ira-amount",
                concept="irs_soi.taxable_ira_distributions",
                value=420_000_000_000,
                aggregation="sum",
                income_range="all",
                unit="usd",
            ),
            fact(
                "arch.fact.v1:t14-all-pension-returns",
                concept="irs_soi.returns_with_taxable_pension_income",
                value=30_000_000,
                aggregation="count",
                income_range="all",
                unit="count",
            ),
            fact(
                "arch.fact.v1:t14-all-pension-amount",
                concept="irs_soi.taxable_pension_income",
                value=740_000_000_000,
                aggregation="sum",
                income_range="all",
                unit="usd",
            ),
            fact(
                "arch.fact.v1:t14-all-uc-returns",
                concept="irs_soi.returns_with_unemployment_compensation",
                value=7_000_000,
                aggregation="count",
                income_range="all",
                unit="count",
            ),
            fact(
                "arch.fact.v1:t14-all-uc-amount",
                concept="irs_soi.unemployment_compensation",
                value=62_000_000_000,
                aggregation="sum",
                income_range="all",
                unit="usd",
            ),
            fact(
                "arch.fact.v1:t14-all-taxable-ss-returns",
                concept="irs_soi.returns_with_taxable_social_security_benefits",
                value=29_000_000,
                aggregation="count",
                income_range="all",
                unit="count",
            ),
            fact(
                "arch.fact.v1:t14-all-taxable-ss-amount",
                concept="irs_soi.taxable_social_security_benefits",
                value=510_000_000_000,
                aggregation="sum",
                income_range="all",
                unit="usd",
            ),
            fact(
                "arch.fact.v1:t14-1-to-5k-wages-amount",
                concept="us:statutes/26/62#input.wages",
                source_concept="irs_soi.total_wages",
                concept_relation="broad_match",
                value=4_200_000_000,
                aggregation="sum",
                income_range="1_to_5k",
                unit="usd",
            ),
        ],
    )
    conn.executemany(
        """
        INSERT INTO aggregate_constraints (
            fact_key,
            ordinal,
            variable,
            operator,
            value_text,
            value_numeric,
            value_json,
            unit,
            role,
            label
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        [
            (
                "arch.fact.v1:t14-1-to-5k-wages-amount",
                ordinal,
                "us:statutes/26/62#adjusted_gross_income",
                operator,
                str(value),
                float(value),
                json.dumps(value),
                "usd",
                "filter",
                "Adjusted gross income bound",
            )
            for ordinal, operator, value in ((0, ">=", 1), (1, "<", 5000))
        ],
    )
    conn.executemany(
        """
        INSERT INTO fact_source_cells (fact_key, source_cell_key, ordinal)
        VALUES (?, ?, ?)
        """,
        [
            (
                "arch.fact.v1:t14-all-wages-amount",
                "arch.source_cell.v1:t14-wages-amount",
                0,
            ),
            (
                "arch.fact.v1:t14-all-wages-returns",
                "arch.source_cell.v1:t14-wages-returns",
                0,
            ),
        ],
    )
    conn.execute(
        """
        INSERT INTO fact_source_rows (fact_key, source_row_key, ordinal)
        VALUES (?, ?, ?)
        """,
        (
            "arch.fact.v1:t14-all-wages-amount",
            "arch.source_row.v1:t14-all",
            0,
        ),
    )
    conn.commit()
    conn.close()


def _write_consumer_fact_jsonl(path: Path) -> None:
    def row(
        key: str,
        *,
        semantic_key: str,
        concept: str,
        value: float,
        aggregation: str,
        income_range: str,
        unit: str,
        source_concept: str | None = None,
    ) -> dict[str, Any]:
        observed_concept = source_concept or concept
        constraints = []
        if income_range == "1_to_5k":
            constraints = [
                {
                    "variable": "us:statutes/26/62#adjusted_gross_income",
                    "operator": ">=",
                    "value": 1,
                    "unit": "usd",
                    "role": "filter",
                },
                {
                    "variable": "us:statutes/26/62#adjusted_gross_income",
                    "operator": "<",
                    "value": 5000,
                    "unit": "usd",
                    "role": "filter",
                },
            ]
        payload: dict[str, Any] = {
            "schema_version": "arch.consumer_fact.v1",
            "aggregate_fact_key": key,
            "semantic_fact_key": semantic_key,
            "legacy_fact_key": key.replace("aggregate_fact.v2", "fact.v1"),
            "value": value,
            "value_type": "integer",
            "period": {"type": "tax_year", "value": 2023},
            "geography": {
                "id": "0100000US",
                "level": "country",
                "vintage": "2020_census",
            },
            "entity": {"name": "tax_unit", "role": "filing_unit"},
            "aggregation": {"method": aggregation},
            "observed_measure": {
                "source_concept": observed_concept,
                "source_measure_id": observed_concept.rsplit(".", 1)[-1],
                "source_name": "irs_soi",
                "source_table": "Publication 1304 Table 1.1",
                "unit": unit,
            },
            "dimensions": {"filing_status": "all", "income_range": income_range},
            "universe_constraints": {
                "domain": "all_individual_income_tax_returns",
                "constraints": constraints,
            },
            "source": {
                "source_name": "irs_soi",
                "source_table": "Publication 1304 Table 1.1",
                "url": "https://www.irs.gov/pub/irs-soi/23in11si.xls",
                "method_notes": "Consumer-contract fact fixture.",
            },
            "lineage": {
                "source_record_id": (
                    f"irs_soi.ty2023.table_1_1.{income_range}."
                    f"{observed_concept.rsplit('.', 1)[-1]}"
                ),
                "source_cell_keys": [f"arch.source_cell.v1:{key.rsplit(':', 1)[-1]}"],
                "source_row_keys": [f"arch.source_row.v1:{income_range}"],
            },
            "label": f"{income_range} {concept}",
        }
        if source_concept is not None:
            payload["concept_alignment"] = {
                "concept_alignment_key": "arch.concept_alignment.v2:agi",
                "source_concept": source_concept,
                "canonical_concept": concept,
                "relation": "exact",
                "authority": "arch-us",
                "evidence_url": (
                    "https://uscode.house.gov/view.xhtml?"
                    "req=(title:26%20section:62%20edition:prelim)"
                ),
                "evidence_notes": "IRS SOI Table 1.1 reports adjusted gross income.",
                "legal_vintage": "tax_year_2023",
            }
        return payload

    rows = [
        row(
            "arch.aggregate_fact.v2:all-count",
            semantic_key="arch.semantic_fact.v2:all-count",
            concept="irs_soi.individual_income_tax_returns",
            value=160_602_107,
            aggregation="count",
            income_range="all",
            unit="count",
        ),
        row(
            "arch.aggregate_fact.v2:all-agi",
            semantic_key="arch.semantic_fact.v2:all-agi",
            concept="us:statutes/26/62#adjusted_gross_income",
            source_concept="irs_soi.adjusted_gross_income",
            value=15_286_017_359_000,
            aggregation="sum",
            income_range="all",
            unit="usd",
        ),
        row(
            "arch.aggregate_fact.v2:all-tax",
            semantic_key="arch.semantic_fact.v2:all-tax",
            concept="irs_soi.total_income_tax",
            value=2_147_909_818_000,
            aggregation="sum",
            income_range="all",
            unit="usd",
        ),
        row(
            "arch.aggregate_fact.v2:1-to-5k-count",
            semantic_key="arch.semantic_fact.v2:1-to-5k-count",
            concept="irs_soi.individual_income_tax_returns",
            value=7_357_751,
            aggregation="count",
            income_range="1_to_5k",
            unit="count",
        ),
        row(
            "arch.aggregate_fact.v2:1-to-5k-agi",
            semantic_key="arch.semantic_fact.v2:1-to-5k-agi",
            concept="us:statutes/26/62#adjusted_gross_income",
            source_concept="irs_soi.adjusted_gross_income",
            value=20_372_694_000,
            aggregation="sum",
            income_range="1_to_5k",
            unit="usd",
        ),
    ]
    path.write_text("\n".join(json.dumps(item, sort_keys=True) for item in rows) + "\n")


def _consumer_fact(
    key: str,
    *,
    concept: str,
    domain: str,
    source_name: str,
    source_table: str,
    value: float,
    period: dict[str, Any] | None = None,
    geography: dict[str, Any] | None = None,
    constraints: tuple[dict[str, Any], ...] = (),
    unit: str = "count",
) -> dict[str, Any]:
    return {
        "schema_version": "arch.consumer_fact.v1",
        "aggregate_fact_key": f"arch.aggregate_fact.v2:{key}",
        "semantic_fact_key": f"arch.semantic_fact.v2:{key}",
        "value": value,
        "period": period or {"type": "calendar_year", "value": 2024},
        "geography": geography
        or {"level": "country", "id": "0100000US", "name": "United States"},
        "observed_measure": {
            "source_concept": concept,
            "source_measure_id": concept.rsplit(".", 1)[-1],
            "source_name": source_name,
            "source_table": source_table,
            "unit": unit,
        },
        "universe_constraints": {
            "domain": domain,
            "constraints": list(constraints),
        },
        "source": {
            "source_name": source_name,
            "source_table": source_table,
            "url": f"https://example.test/{key}",
            "method_notes": "US admin source-family fixture.",
        },
        "lineage": {
            "source_record_id": f"{source_name}.{key}",
            "source_cell_keys": [f"arch.source_cell.v1:{key}"],
            "source_row_keys": [f"arch.source_row.v1:{key}"],
        },
        "label": key,
    }


def _target_filter_tuples(target: Any) -> set[tuple[str, str, str]]:
    return {
        (
            str(target_filter.feature),
            str(getattr(target_filter.operator, "value", target_filter.operator)),
            str(target_filter.value),
        )
        for target_filter in target.filters
    }


def _normalize_target_behavior(target_set) -> list[tuple[Any, ...]]:
    rows = []
    for target in target_set.targets:
        filters = tuple(
            sorted(
                (
                    str(target_filter.feature),
                    str(
                        getattr(target_filter.operator, "value", target_filter.operator)
                    ),
                    str(target_filter.value),
                )
                for target_filter in target.filters
            )
        )
        rows.append(
            (
                str(target.entity.value),
                str(getattr(target.aggregation, "value", target.aggregation)),
                target.measure,
                round(float(target.value), 6),
                int(target.period),
                str(target.source),
                target.metadata["variable"],
                target.metadata["geo_level"],
                filters,
            )
        )
    return sorted(rows)


def test_arch_fact_provider_matches_value_constraint_soi_targets(
    tmp_path: Path,
) -> None:
    value_db = tmp_path / "value_targets.db"
    fact_db = tmp_path / "arch_facts.db"
    _create_value_constraint_target_db(value_db)
    _create_arch_fact_db(fact_db)

    query = TargetQuery(period=2023)
    value_targets = ArchSQLiteTargetProvider(value_db).load_target_set(query)
    fact_targets = ArchFactSQLiteTargetProvider(fact_db).load_target_set(query)

    assert _normalize_target_behavior(fact_targets) == _normalize_target_behavior(
        value_targets
    )


def test_arch_fact_provider_skips_unsupported_source_package_facts(
    tmp_path: Path,
) -> None:
    fact_db = tmp_path / "arch_facts.db"
    _create_arch_fact_db(fact_db)
    conn = sqlite3.connect(fact_db)
    conn.execute(
        """
        INSERT INTO aggregate_facts (
            fact_key,
            source_record_id,
            value_numeric,
            value_text,
            value_json,
            period_value,
            geography_level,
            geography_id,
            geography_name,
            measure_concept,
            measure_source_concept,
            measure_concept_relation,
            measure_concept_authority,
            measure_concept_evidence_url,
            measure_concept_evidence_notes,
            measure_legal_vintage,
            measure_unit,
            aggregation_method,
            domain,
            filters_json,
            label,
            source_name,
            source_table,
            source_url,
            source_method_notes
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            "arch.fact.v1:bea-defined-contribution-pensions",
            "bea_nipa.nipa.7.20.line_5",
            274_439_000_000,
            "274439000000",
            json.dumps(274_439_000_000),
            "2023",
            "country",
            "0100000US",
            "United States",
            "bea_nipa.defined_contribution_employer_contributions",
            "bea_nipa.W351RC",
            "source_label",
            "bea",
            "https://apps.bea.gov/iTable/",
            "BEA NIPA pension contribution source-package fixture.",
            None,
            "usd",
            "sum",
            "defined_contribution_pension_plans",
            json.dumps({}),
            "Defined contribution employer contributions",
            "bea_nipa",
            "NIPA Table 7.20",
            "https://apps.bea.gov/iTable/",
            "Unsupported source package fact fixture.",
        ),
    )
    conn.commit()
    conn.close()

    target_set = ArchFactSQLiteTargetProvider(fact_db).load_target_set(
        TargetQuery(period=2023)
    )

    assert target_set.targets
    assert all(
        target.metadata.get("arch_concept")
        != "bea_nipa.defined_contribution_employer_contributions"
        for target in target_set.targets
    )


def test_arch_consumer_fact_jsonl_provider_matches_value_constraint_soi_targets(
    tmp_path: Path,
) -> None:
    value_db = tmp_path / "value_targets.db"
    consumer_jsonl = tmp_path / "consumer_facts.jsonl"
    _create_value_constraint_target_db(value_db)
    _write_consumer_fact_jsonl(consumer_jsonl)

    query = TargetQuery(period=2023)
    value_targets = ArchSQLiteTargetProvider(value_db).load_target_set(query)
    consumer_targets = ArchConsumerFactJSONLTargetProvider(
        consumer_jsonl
    ).load_target_set(query)

    assert _normalize_target_behavior(consumer_targets) == _normalize_target_behavior(
        value_targets
    )


def test_arch_consumer_fact_jsonl_provider_skips_unsupported_source_package_facts(
    tmp_path: Path,
) -> None:
    consumer_jsonl = tmp_path / "consumer_facts.jsonl"
    rows = [
        _consumer_fact(
            "soi-wages",
            concept="irs_soi.total_wages",
            domain="all_individual_income_tax_returns",
            source_name="irs_soi",
            source_table="Publication 1304 Table 1.1",
            period={"type": "tax_year", "value": 2024},
            value=10_000_000_000_000,
            unit="usd",
        ),
        _consumer_fact(
            "bea-defined-contribution-pensions",
            concept="bea_nipa.defined_contribution_employer_contributions",
            domain="defined_contribution_pension_plans",
            source_name="bea_nipa",
            source_table="NIPA Table 7.20",
            period={"type": "calendar_year", "value": 2024},
            value=274_439_000_000,
            unit="usd",
        ),
    ]
    consumer_jsonl.write_text(
        "\n".join(json.dumps(row, sort_keys=True) for row in rows) + "\n"
    )

    target_set = ArchConsumerFactJSONLTargetProvider(consumer_jsonl).load_target_set(
        TargetQuery(period=2024)
    )

    assert len(target_set.targets) == 1
    target = target_set.targets[0]
    assert target.metadata["arch_variable"] == "wages_salaries_amount"
    assert target.measure == "employment_income"
    assert target.metadata["arch_concept"] == "irs_soi.total_wages"


def test_arch_fact_provider_preserves_fact_provenance(tmp_path: Path) -> None:
    fact_db = tmp_path / "arch_facts.db"
    _create_arch_fact_db(fact_db)

    target_set = ArchFactSQLiteTargetProvider(fact_db).load_target_set(
        TargetQuery(
            period=2023,
            provider_filters={
                "target_cells": [
                    {
                        "variable": "adjusted_gross_income",
                        "geo_level": "national",
                        "domain_variable": None,
                    }
                ]
            },
        )
    )

    all_agi = next(
        target
        for target in target_set.targets
        if target.metadata["arch_aggregate_fact_key"] == "arch.fact.v1:all-agi"
    )
    assert all_agi.metadata["arch_semantic_fact_key"].startswith(
        "arch.semantic_fact.v1|us:statutes/26/62#adjusted_gross_income"
    )
    assert all_agi.metadata["arch_source_record_id"].startswith(
        "irs_soi.ty2023.table_1_1.all"
    )
    assert all_agi.metadata["arch_source_cell_keys"] == ["arch.source_cell.v1:agi"]
    assert all_agi.metadata["arch_source_row_keys"] == ["arch.source_row.v1:all"]
    assert all_agi.metadata["arch_source_concept"] == "irs_soi.adjusted_gross_income"
    assert all_agi.metadata["arch_concept_relation"] == "exact"
    assert all_agi.metadata["unit"] == "usd"


def test_arch_consumer_fact_jsonl_provider_preserves_contract_keys(
    tmp_path: Path,
) -> None:
    consumer_jsonl = tmp_path / "consumer_facts.jsonl"
    _write_consumer_fact_jsonl(consumer_jsonl)

    target_set = ArchConsumerFactJSONLTargetProvider(consumer_jsonl).load_target_set(
        TargetQuery(period=2023)
    )

    all_agi = next(
        target
        for target in target_set.targets
        if target.metadata["arch_aggregate_fact_key"]
        == "arch.aggregate_fact.v2:all-agi"
    )
    assert all_agi.metadata["arch_semantic_fact_key"] == "arch.semantic_fact.v2:all-agi"
    assert all_agi.metadata["arch_source_record_id"].startswith(
        "irs_soi.ty2023.table_1_1.all"
    )
    assert all_agi.metadata["arch_source_cell_keys"] == ["arch.source_cell.v1:all-agi"]
    assert all_agi.metadata["arch_source_row_keys"] == ["arch.source_row.v1:all"]
    assert all_agi.metadata["arch_source_concept"] == "irs_soi.adjusted_gross_income"
    assert all_agi.metadata["arch_concept_relation"] == "exact"
    assert all_agi.metadata["unit"] == "usd"


def test_arch_consumer_fact_jsonl_provider_maps_income_tax_after_credits_returns(
    tmp_path: Path,
) -> None:
    consumer_jsonl = tmp_path / "consumer_facts.jsonl"
    _write_consumer_fact_jsonl(consumer_jsonl)
    rows = [json.loads(line) for line in consumer_jsonl.read_text().splitlines()]
    row = json.loads(json.dumps(rows[0]))
    row["aggregate_fact_key"] = "arch.aggregate_fact.v2:all-income-tax-returns"
    row["semantic_fact_key"] = "arch.semantic_fact.v2:all-income-tax-returns"
    row["legacy_fact_key"] = "arch.fact.v1:all-income-tax-returns"
    row["value"] = 111_545_061
    row["observed_measure"] = {
        **row["observed_measure"],
        "source_concept": "irs_soi.returns_with_income_tax_after_credits",
        "source_measure_id": "income_tax_after_credits_returns",
        "unit": "count",
    }
    row["lineage"]["source_record_id"] = (
        "irs_soi.ty2023.table_1_1.all.income_tax_after_credits_returns"
    )
    consumer_jsonl.write_text(json.dumps(row, sort_keys=True) + "\n")

    target_set = ArchConsumerFactJSONLTargetProvider(consumer_jsonl).load_target_set(
        TargetQuery(period=2023)
    )
    target = target_set.targets[0]
    filters = {
        (
            target_filter.feature,
            str(getattr(target_filter.operator, "value", target_filter.operator)),
            str(target_filter.value),
        )
        for target_filter in target.filters
    }

    assert target.metadata["arch_variable"] == "income_tax_liability_returns"
    assert target.metadata["variable"] == "tax_unit_count"
    assert target.aggregation.value == "count"
    assert filters == {
        ("income_tax", ">", "0"),
        ("tax_unit_is_filer", "==", "1"),
    }


def test_arch_consumer_fact_jsonl_provider_maps_tax_exempt_interest(
    tmp_path: Path,
) -> None:
    consumer_jsonl = tmp_path / "consumer_facts.jsonl"
    _write_consumer_fact_jsonl(consumer_jsonl)
    template = json.loads(consumer_jsonl.read_text().splitlines()[0])
    rows = []
    for suffix, concept, measure_id, value, unit in (
        (
            "qualified-dividends-returns",
            "irs_soi.returns_with_qualified_dividends",
            "qualified_dividends_returns",
            38_000_000,
            "count",
        ),
        (
            "qualified-dividends-amount",
            "irs_soi.qualified_dividends",
            "qualified_dividends_amount",
            350_000_000_000,
            "usd",
        ),
        (
            "returns",
            "irs_soi.returns_with_tax_exempt_interest",
            "tax_exempt_interest_returns",
            6_837_120,
            "count",
        ),
        (
            "amount",
            "irs_soi.tax_exempt_interest",
            "tax_exempt_interest_amount",
            89_000_000_000,
            "usd",
        ),
    ):
        row = json.loads(json.dumps(template))
        row["aggregate_fact_key"] = f"arch.aggregate_fact.v2:tax-exempt-{suffix}"
        row["semantic_fact_key"] = f"arch.semantic_fact.v2:tax-exempt-{suffix}"
        row["legacy_fact_key"] = f"arch.fact.v1:tax-exempt-{suffix}"
        row["period"] = {"type": "tax_year", "value": 2022}
        row["value"] = value
        row["source"] = {**row["source"], "source_table": "Historic Table 2"}
        row["observed_measure"] = {
            **row["observed_measure"],
            "source_concept": concept,
            "source_measure_id": measure_id,
            "source_table": "Historic Table 2",
            "unit": unit,
        }
        row["aggregation"] = {"method": "count" if unit == "count" else "sum"}
        row["lineage"]["source_record_id"] = (
            f"irs_soi.ty2022.historic_table_2.us.all.{measure_id}"
        )
        rows.append(row)
    consumer_jsonl.write_text(
        "\n".join(json.dumps(row, sort_keys=True) for row in rows) + "\n"
    )

    target_set = ArchConsumerFactJSONLTargetProvider(consumer_jsonl).load_target_set(
        TargetQuery(period=2022)
    )
    targets_by_arch_variable = {
        target.metadata["arch_variable"]: target for target in target_set.targets
    }
    returns = targets_by_arch_variable["tax_exempt_interest_returns"]
    amount = targets_by_arch_variable["tax_exempt_interest_amount"]
    qualified_returns = targets_by_arch_variable["qualified_dividends_returns"]
    qualified_amount = targets_by_arch_variable["qualified_dividends_amount"]

    assert returns.metadata["variable"] == "tax_exempt_interest_income"
    assert returns.aggregation.value == "count"
    assert {
        (
            target_filter.feature,
            str(getattr(target_filter.operator, "value", target_filter.operator)),
            str(target_filter.value),
        )
        for target_filter in returns.filters
    } == {
        ("tax_exempt_interest_income", ">", "0"),
        ("tax_unit_is_filer", "==", "1"),
    }
    assert amount.metadata["variable"] == "tax_exempt_interest_income"
    assert amount.measure == "tax_exempt_interest_income"
    assert qualified_returns.metadata["variable"] == "qualified_dividend_income"
    assert qualified_returns.aggregation.value == "count"
    assert {
        (
            target_filter.feature,
            str(getattr(target_filter.operator, "value", target_filter.operator)),
            str(target_filter.value),
        )
        for target_filter in qualified_returns.filters
    } == {
        ("qualified_dividend_income", ">", "0"),
        ("tax_unit_is_filer", "==", "1"),
    }
    assert qualified_amount.metadata["variable"] == "qualified_dividend_income"
    assert qualified_amount.measure == "qualified_dividend_income"


def test_arch_consumer_fact_jsonl_provider_maps_schedule_c_self_employment(
    tmp_path: Path,
) -> None:
    consumer_jsonl = tmp_path / "consumer_facts.jsonl"
    rows = [
        _consumer_fact(
            "soi-schedule-c-returns",
            concept="irs_soi.returns_with_schedule_c_income",
            domain="all_individual_income_tax_returns",
            source_name="irs_soi",
            source_table="Historic Table 2",
            period={"type": "tax_year", "value": 2022},
            value=28_000_000,
            unit="count",
        ),
        _consumer_fact(
            "soi-schedule-c-income",
            concept="irs_soi.schedule_c_income",
            domain="all_individual_income_tax_returns",
            source_name="irs_soi",
            source_table="Historic Table 2",
            period={"type": "tax_year", "value": 2022},
            value=512_000_000_000,
            unit="usd",
        ),
        _consumer_fact(
            "soi-partnership-scorp-returns",
            concept="irs_soi.returns_with_partnership_scorp_income",
            domain="all_individual_income_tax_returns",
            source_name="irs_soi",
            source_table="Historic Table 2",
            period={"type": "tax_year", "value": 2022},
            value=12_000_000,
            unit="count",
        ),
        _consumer_fact(
            "soi-partnership-scorp-income",
            concept="irs_soi.partnership_scorp_income",
            domain="all_individual_income_tax_returns",
            source_name="irs_soi",
            source_table="Historic Table 2",
            period={"type": "tax_year", "value": 2022},
            value=1_200_000_000_000,
            unit="usd",
        ),
    ]
    consumer_jsonl.write_text(
        "\n".join(json.dumps(row, sort_keys=True) for row in rows) + "\n"
    )

    target_set = ArchConsumerFactJSONLTargetProvider(consumer_jsonl).load_target_set(
        TargetQuery(period=2022)
    )
    targets_by_arch_variable = {
        target.metadata["arch_variable"]: target for target in target_set.targets
    }
    returns = targets_by_arch_variable["schedule_c_income_returns"]
    amount = targets_by_arch_variable["schedule_c_income_amount"]
    partnership_returns = targets_by_arch_variable["partnership_scorp_income_returns"]
    partnership_amount = targets_by_arch_variable["partnership_scorp_income_amount"]

    assert returns.metadata["variable"] == "self_employment_income"
    assert {
        (
            target_filter.feature,
            str(getattr(target_filter.operator, "value", target_filter.operator)),
            str(target_filter.value),
        )
        for target_filter in returns.filters
    } == {
        ("self_employment_income", ">", "0"),
        ("tax_unit_is_filer", "==", "1"),
    }
    assert amount.metadata["variable"] == "self_employment_income"
    assert amount.measure == "self_employment_income"
    assert (
        partnership_returns.metadata["variable"]
        == "tax_unit_partnership_s_corp_income"
    )
    assert {
        (
            target_filter.feature,
            str(getattr(target_filter.operator, "value", target_filter.operator)),
            str(target_filter.value),
        )
        for target_filter in partnership_returns.filters
    } == {
        ("tax_unit_is_filer", "==", "1"),
        ("tax_unit_partnership_s_corp_income", ">", "0"),
    }
    assert (
        partnership_amount.metadata["variable"]
        == "tax_unit_partnership_s_corp_income"
    )
    assert partnership_amount.measure == "tax_unit_partnership_s_corp_income"


def test_arch_consumer_fact_jsonl_provider_maps_historic_table_2_concepts(
    tmp_path: Path,
) -> None:
    consumer_jsonl = tmp_path / "consumer_facts.jsonl"
    _write_consumer_fact_jsonl(consumer_jsonl)
    template = json.loads(consumer_jsonl.read_text().splitlines()[0])
    rows = []
    for index, (concept, measure_id, value) in enumerate(
        (
            (
                "irs_soi.returns_with_premium_tax_credit",
                "premium_tax_credit_returns",
                7_841_370,
            ),
            ("irs_soi.earned_income_credit", "eitc_amount", 59_204_588_000),
            (
                "irs_soi.tax_filer_individuals",
                "tax_filer_individual_count",
                293_617_150,
            ),
        ),
        start=1,
    ):
        row = json.loads(json.dumps(template))
        row["aggregate_fact_key"] = f"arch.aggregate_fact.v2:historic-table-2-{index}"
        row["semantic_fact_key"] = f"arch.semantic_fact.v2:historic-table-2-{index}"
        row["legacy_fact_key"] = f"arch.fact.v1:historic-table-2-{index}"
        row["period"] = {"type": "tax_year", "value": 2022}
        row["value"] = value
        row["source"] = {**row["source"], "source_table": "Historic Table 2"}
        row["observed_measure"] = {
            **row["observed_measure"],
            "source_concept": concept,
            "source_measure_id": measure_id,
            "source_table": "Historic Table 2",
            "unit": "usd" if concept == "irs_soi.earned_income_credit" else "count",
        }
        row["aggregation"] = {
            "method": "sum" if concept == "irs_soi.earned_income_credit" else "count"
        }
        row["lineage"]["source_record_id"] = (
            f"irs_soi.ty2022.historic_table_2.us.all.{measure_id}"
        )
        rows.append(row)
    consumer_jsonl.write_text(
        "\n".join(json.dumps(row, sort_keys=True) for row in rows) + "\n"
    )

    target_set = ArchConsumerFactJSONLTargetProvider(consumer_jsonl).load_target_set(
        TargetQuery(period=2022)
    )
    targets_by_arch_variable = {
        target.metadata["arch_variable"]: target for target in target_set.targets
    }

    premium_tax_credit = targets_by_arch_variable["aca_ptc_returns"]
    assert premium_tax_credit.metadata["variable"] == "tax_unit_count"
    assert premium_tax_credit.aggregation.value == "count"
    assert {
        (
            target_filter.feature,
            str(getattr(target_filter.operator, "value", target_filter.operator)),
            str(target_filter.value),
        )
        for target_filter in premium_tax_credit.filters
    } == {
        ("aca_ptc", ">", "0"),
        ("tax_unit_is_filer", "==", "1"),
    }

    eitc = targets_by_arch_variable["eitc_amount"]
    assert eitc.metadata["variable"] == "eitc"
    assert eitc.measure == "eitc"
    assert eitc.aggregation.value == "sum"

    tax_filer_individuals = targets_by_arch_variable["tax_filer_individual_count"]
    assert tax_filer_individuals.metadata["variable"] == "person_count"
    assert tax_filer_individuals.aggregation.value == "count"


def test_arch_consumer_fact_jsonl_provider_maps_table_2_1_itemized_details(
    tmp_path: Path,
) -> None:
    consumer_jsonl = tmp_path / "consumer_facts.jsonl"
    period = {"type": "tax_year", "value": 2023}
    source_table = "Publication 1304 Table 2.1"
    domain = "individual_income_tax_returns_with_itemized_deductions"
    rows = [
        _consumer_fact(
            "soi-charitable-deduction",
            concept="irs_soi.contributions_deduction",
            domain=domain,
            source_name="irs_soi",
            source_table=source_table,
            value=211_975_123_000,
            period=period,
            unit="usd",
        ),
        _consumer_fact(
            "soi-charitable-returns",
            concept="irs_soi.returns_with_contributions_deduction",
            domain=domain,
            source_name="irs_soi",
            source_table=source_table,
            value=11_747_949,
            period=period,
        ),
        _consumer_fact(
            "soi-interest-deduction",
            concept="irs_soi.interest_paid_deduction",
            domain=domain,
            source_name="irs_soi",
            source_table=source_table,
            value=208_176_768_000,
            period=period,
            unit="usd",
        ),
        _consumer_fact(
            "soi-state-local-total",
            concept="irs_soi.state_and_local_taxes",
            domain=domain,
            source_name="irs_soi",
            source_table=source_table,
            value=331_823_221_000,
            period=period,
            unit="usd",
        ),
        _consumer_fact(
            "soi-state-local-income-sales",
            concept="irs_soi.state_local_income_or_sales_taxes",
            domain=domain,
            source_name="irs_soi",
            source_table=source_table,
            value=218_543_083_000,
            period=period,
            unit="usd",
        ),
        _consumer_fact(
            "soi-real-estate-taxes",
            concept="irs_soi.real_estate_taxes",
            domain=domain,
            source_name="irs_soi",
            source_table=source_table,
            value=108_606_373_000,
            period=period,
            unit="usd",
        ),
        _consumer_fact(
            "soi-mortgage-financial",
            concept="irs_soi.home_mortgage_interest_paid_to_financial_institutions",
            domain=domain,
            source_name="irs_soi",
            source_table=source_table,
            value=167_675_863_000,
            period=period,
            unit="usd",
        ),
        _consumer_fact(
            "soi-mortgage-individual",
            concept="irs_soi.home_mortgage_interest_paid_to_individuals",
            domain=domain,
            source_name="irs_soi",
            source_table=source_table,
            value=3_688_924_000,
            period=period,
            unit="usd",
        ),
        _consumer_fact(
            "soi-deductible-points",
            concept="irs_soi.deductible_points",
            domain=domain,
            source_name="irs_soi",
            source_table=source_table,
            value=1_027_127_000,
            period=period,
            unit="usd",
        ),
        _consumer_fact(
            "soi-investment-interest",
            concept="irs_soi.investment_interest_expense_deduction",
            domain=domain,
            source_name="irs_soi",
            source_table=source_table,
            value=35_768_354_000,
            period=period,
            unit="usd",
        ),
    ]
    consumer_jsonl.write_text(
        "\n".join(json.dumps(row, sort_keys=True) for row in rows) + "\n"
    )

    provider = ArchConsumerFactJSONLTargetProvider(consumer_jsonl)
    target_set = provider.load_target_set(TargetQuery(period=2023))
    targets_by_arch_variable = {
        target.metadata["arch_variable"]: target
        for target in target_set.targets
        if target.metadata.get("arch_variable") is not None
    }
    targets_by_measure = {
        str(target.measure): target
        for target in target_set.targets
        if target.measure is not None
    }

    charitable = targets_by_arch_variable["charitable_amount"]
    assert charitable.measure == "charitable_deduction"
    assert charitable.entity.value == "tax_unit"
    assert charitable.value == 211_975_123_000
    assert ("itemized_deductions", ">", "0") in _target_filter_tuples(charitable)

    charitable_count = targets_by_arch_variable["charitable_returns"]
    assert charitable_count.metadata["variable"] == "charitable_deduction"
    assert charitable_count.aggregation.value == "count"
    assert ("charitable_deduction", ">", "0") in _target_filter_tuples(
        charitable_count
    )
    assert ("itemized_deductions", ">", "0") in _target_filter_tuples(
        charitable_count
    )

    assert targets_by_arch_variable["interest_paid_deduction_amount"].measure == (
        "interest_deduction"
    )
    assert "total_state_local_taxes_amount" not in targets_by_arch_variable
    state_local_income_sales = targets_by_arch_variable[
        "state_local_income_or_sales_tax_amount"
    ]
    assert state_local_income_sales.measure == "state_and_local_sales_or_income_tax"
    assert targets_by_arch_variable["real_estate_taxes_amount"].measure == (
        "real_estate_taxes"
    )
    salt = targets_by_measure["salt"]
    assert salt.measure == "salt"
    assert salt.value == 327_149_456_000
    assert salt.metadata["variable"] == "salt"
    assert targets_by_arch_variable["mortgage_interest_paid_amount"].measure == (
        "deductible_mortgage_interest"
    )
    assert targets_by_arch_variable["home_mortgage_personal_seller_amount"].measure == (
        "deductible_mortgage_interest"
    )
    assert targets_by_arch_variable["deductible_points_amount"].measure == (
        "deductible_mortgage_interest"
    )
    assert targets_by_arch_variable["investment_interest_paid_amount"].measure == (
        "investment_interest_expense"
    )

    coverage = summarize_arch_target_profile_coverage(
        provider,
        period=2023,
        profile_name="custom",
        target_cells=(
            PolicyEngineUSTargetCell(
                "charitable_deduction",
                geo_level="national",
            ),
            PolicyEngineUSTargetCell(
                "charitable_deduction",
                geo_level="national",
                domain_variable="charitable_deduction,tax_unit_itemizes",
            ),
            PolicyEngineUSTargetCell(
                "tax_unit_count",
                geo_level="national",
                domain_variable="charitable_deduction,tax_unit_itemizes",
            ),
            PolicyEngineUSTargetCell(
                "interest_deduction",
                geo_level="national",
            ),
            PolicyEngineUSTargetCell(
                "deductible_mortgage_interest",
                geo_level="national",
            ),
            PolicyEngineUSTargetCell(
                "state_and_local_sales_or_income_tax",
                geo_level="national",
            ),
            PolicyEngineUSTargetCell(
                "salt",
                geo_level="national",
                domain_variable="salt,tax_unit_itemizes",
            ),
        ),
    )
    assert coverage.covered_cell_count == coverage.target_cell_count


def test_arch_consumer_fact_jsonl_provider_maps_state_soi_rows(
    tmp_path: Path,
) -> None:
    consumer_jsonl = tmp_path / "consumer_facts.jsonl"
    rows = [
        _consumer_fact(
            "state-ca-agi-50k-75k",
            concept="irs_soi.adjusted_gross_income",
            domain="all_individual_income_tax_returns",
            source_name="irs_soi",
            source_table="Historic Table 2 state AGI facts",
            period={"type": "tax_year", "value": 2022},
            geography={"level": "state", "id": "0400000US06", "name": "California"},
            value=123_456_000_000,
            unit="usd",
            constraints=(
                {
                    "variable": "us:statutes/26/62#adjusted_gross_income",
                    "operator": ">=",
                    "value": 50_000,
                    "unit": "usd",
                    "role": "filter",
                },
                {
                    "variable": "us:statutes/26/62#adjusted_gross_income",
                    "operator": "<",
                    "value": 75_000,
                    "unit": "usd",
                    "role": "filter",
                },
            ),
        ),
        _consumer_fact(
            "state-ca-eitc-amount",
            concept="irs_soi.earned_income_credit",
            domain="individual_income_tax_returns",
            source_name="irs_soi",
            source_table="Historic Table 2 state EITC totals",
            period={"type": "tax_year", "value": 2022},
            geography={"level": "state", "id": "0400000US06", "name": "California"},
            value=5_770_703_000,
            unit="usd",
        ),
    ]
    consumer_jsonl.write_text(
        "\n".join(json.dumps(row, sort_keys=True) for row in rows) + "\n"
    )

    target_set = ArchConsumerFactJSONLTargetProvider(consumer_jsonl).load_target_set(
        TargetQuery(period=2022)
    )
    targets_by_arch_variable = {
        target.metadata["arch_variable"]: target for target in target_set.targets
    }

    agi = targets_by_arch_variable["adjusted_gross_income"]
    assert agi.metadata["variable"] == "adjusted_gross_income"
    assert agi.metadata["geo_level"] == "state"
    assert agi.metadata["geography_id"] == "0400000US06"
    assert agi.measure == "adjusted_gross_income"
    assert agi.aggregation.value == "sum"
    assert _target_filter_tuples(agi) == {
        ("tax_unit_is_filer", "==", "1"),
        ("adjusted_gross_income", ">=", "50000"),
        ("adjusted_gross_income", "<", "75000"),
        ("state_fips", "==", "06"),
    }

    eitc = targets_by_arch_variable["eitc_amount"]
    assert eitc.metadata["variable"] == "eitc"
    assert eitc.metadata["geo_level"] == "state"
    assert eitc.measure == "eitc"
    assert eitc.aggregation.value == "sum"
    assert _target_filter_tuples(eitc) == {
        ("tax_unit_is_filer", "==", "1"),
        ("state_fips", "==", "06"),
    }


def test_arch_consumer_fact_jsonl_provider_maps_acs_district_age_rows(
    tmp_path: Path,
) -> None:
    consumer_jsonl = tmp_path / "consumer_facts.jsonl"
    rows = [
        _consumer_fact(
            "acs-cd-al01-age-0-4",
            concept="census_acs.person_count",
            domain="total_population",
            source_name="census_acs",
            source_table="ACS S0101 congressional district age",
            period={"type": "calendar_year", "value": 2024},
            geography={
                "level": "congressional_district",
                "id": "5001900US0101",
                "name": "Congressional District 1 (119th Congress), Alabama",
            },
            value=39_908,
            constraints=(
                {
                    "variable": "age",
                    "operator": ">=",
                    "value": 0,
                    "unit": "years",
                    "role": "filter",
                },
                {
                    "variable": "age",
                    "operator": "<",
                    "value": 5,
                    "unit": "years",
                    "role": "filter",
                },
            ),
        ),
    ]
    consumer_jsonl.write_text(
        "\n".join(json.dumps(row, sort_keys=True) for row in rows) + "\n"
    )

    target_set = ArchConsumerFactJSONLTargetProvider(consumer_jsonl).load_target_set(
        TargetQuery(
            period=2024,
            provider_filters={
                "sources": ["CENSUS_ACS"],
                "target_cells": [
                    {
                        "variable": "person_count",
                        "geo_level": "district",
                        "geographic_id": "0101",
                        "domain_variable": "age",
                    },
                ],
            },
        )
    )

    assert len(target_set.targets) == 1
    target = target_set.targets[0]
    assert target.value == 39_908
    assert target.metadata["variable"] == "person_count"
    assert target.metadata["geo_level"] == "district"
    assert target.metadata["source"] == "CENSUS_ACS"
    assert _target_filter_tuples(target) == {
        ("age", ">=", "0"),
        ("age", "<", "5"),
        ("congressional_district_geoid", "==", "0101"),
    }


def test_arch_consumer_fact_jsonl_provider_normalizes_117th_district_geos(
    tmp_path: Path,
) -> None:
    consumer_jsonl = tmp_path / "consumer_facts.jsonl"
    rows = [
        _consumer_fact(
            "soi-cd-al01-agi",
            concept="us:statutes/26/62#adjusted_gross_income",
            domain="all_individual_income_tax_returns",
            source_name="irs_soi",
            source_table="SOI Congressional District Data 2022",
            period={"type": "tax_year", "value": 2022},
            geography={
                "level": "congressional_district",
                "id": "5001700US0101",
                "name": "Alabama Congressional District 1",
            },
            value=22_915_824_000,
            unit="usd",
        ),
    ]
    consumer_jsonl.write_text(
        "\n".join(json.dumps(row, sort_keys=True) for row in rows) + "\n"
    )

    target_set = ArchConsumerFactJSONLTargetProvider(consumer_jsonl).load_target_set(
        TargetQuery(
            period=2022,
            provider_filters={
                "sources": ["IRS_SOI"],
                "target_cells": [
                    {
                        "variable": "adjusted_gross_income",
                        "geo_level": "district",
                        "geographic_id": "0101",
                        "domain_variable": None,
                    },
                ],
            },
        )
    )

    assert len(target_set.targets) == 1
    target = target_set.targets[0]
    assert target.value == 22_915_824_000
    assert target.metadata["variable"] == "adjusted_gross_income"
    assert target.metadata["geo_level"] == "district"
    assert _target_filter_tuples(target) == {
        ("tax_unit_is_filer", "==", "1"),
        ("congressional_district_geoid", "==", "0101"),
    }


def test_arch_consumer_fact_jsonl_provider_maps_acs_state_age_sex_rows(
    tmp_path: Path,
) -> None:
    consumer_jsonl = tmp_path / "consumer_facts.jsonl"
    row = _consumer_fact(
        "acs-ca-female-40-44",
        concept="census_acs.person_count",
        domain="total_population",
        source_name="census_acs",
        source_table="ACS B01001 state female age",
        period={"type": "calendar_year", "value": 2023},
        geography={"level": "state", "id": "0400000US06", "name": "California"},
        value=1_300_307,
        constraints=(
            {
                "variable": "age",
                "operator": ">=",
                "value": 40,
                "unit": "years",
                "role": "filter",
            },
            {
                "variable": "age",
                "operator": "<",
                "value": 45,
                "unit": "years",
                "role": "filter",
            },
            {
                "variable": "sex",
                "operator": "==",
                "value": "female",
                "role": "filter",
            },
        ),
    )
    consumer_jsonl.write_text(json.dumps(row, sort_keys=True) + "\n")

    target_set = ArchConsumerFactJSONLTargetProvider(consumer_jsonl).load_target_set(
        TargetQuery(period=2023)
    )

    assert len(target_set.targets) == 1
    target = target_set.targets[0]
    assert target.value == 1_300_307
    assert target.metadata["variable"] == "person_count"
    assert target.metadata["geo_level"] == "state"
    assert _target_filter_tuples(target) == {
        ("age", ">=", "40"),
        ("age", "<", "45"),
        ("is_female", "==", "1"),
        ("state_fips", "==", "06"),
    }


def test_arch_consumer_fact_jsonl_provider_maps_acs_district_snap_rows(
    tmp_path: Path,
) -> None:
    consumer_jsonl = tmp_path / "consumer_facts.jsonl"
    geography = {
        "level": "congressional_district",
        "id": "5001900US0101",
        "name": "Congressional District 1 (119th Congress), Alabama",
    }
    rows = [
        _consumer_fact(
            "acs-cd-al01-households-total",
            concept="census_acs.household_count",
            domain="households",
            source_name="census_acs",
            source_table="ACS S2201 congressional district SNAP households",
            period={"type": "calendar_year", "value": 2024},
            geography=geography,
            value=300_636,
        ),
        _consumer_fact(
            "acs-cd-al01-households-snap",
            concept="census_acs.household_count",
            domain="households",
            source_name="census_acs",
            source_table="ACS S2201 congressional district SNAP households",
            period={"type": "calendar_year", "value": 2024},
            geography=geography,
            value=34_742,
            constraints=(
                {
                    "variable": "snap_receipt_status",
                    "operator": "==",
                    "value": "receiving_food_stamps_snap",
                    "role": "filter",
                },
            ),
        ),
    ]
    consumer_jsonl.write_text(
        "\n".join(json.dumps(row, sort_keys=True) for row in rows) + "\n"
    )

    target_set = ArchConsumerFactJSONLTargetProvider(consumer_jsonl).load_target_set(
        TargetQuery(
            period=2024,
            provider_filters={
                "sources": ["CENSUS_ACS"],
                "target_cells": [
                    {
                        "variable": "household_count",
                        "geo_level": "district",
                        "geographic_id": "0101",
                        "domain_variable": "snap",
                    },
                ],
            },
        )
    )

    assert len(target_set.targets) == 1
    target = target_set.targets[0]
    assert target.value == 34_742
    assert target.metadata["variable"] == "household_count"
    assert target.metadata["geo_level"] == "district"
    assert target.metadata["source"] == "CENSUS_ACS"
    assert _target_filter_tuples(target) == {
        ("congressional_district_geoid", "==", "0101"),
        ("snap", ">", "0"),
    }


def test_arch_consumer_fact_jsonl_provider_maps_state_broad_soi_concepts(
    tmp_path: Path,
) -> None:
    consumer_jsonl = tmp_path / "consumer_facts.jsonl"
    geography = {"level": "state", "id": "0400000US06", "name": "California"}
    rows = [
        _consumer_fact(
            "state-ca-qualified-dividends",
            concept="irs_soi.qualified_dividends",
            domain="all_individual_income_tax_returns",
            source_name="irs_soi",
            source_table="Historic Table 2 state broad totals",
            period={"type": "tax_year", "value": 2022},
            geography=geography,
            value=93_000_000_000,
            unit="usd",
        ),
        _consumer_fact(
            "state-ca-schedule-c-returns",
            concept="irs_soi.returns_with_schedule_c_income",
            domain="all_individual_income_tax_returns",
            source_name="irs_soi",
            source_table="Historic Table 2 state broad totals",
            period={"type": "tax_year", "value": 2022},
            geography=geography,
            value=3_617_080,
        ),
        _consumer_fact(
            "state-ca-partnership-scorp",
            concept="irs_soi.partnership_scorp_income",
            domain="all_individual_income_tax_returns",
            source_name="irs_soi",
            source_table="Historic Table 2 state broad totals",
            period={"type": "tax_year", "value": 2022},
            geography=geography,
            value=125_930_370_000,
            unit="usd",
        ),
        _consumer_fact(
            "state-ca-medical-dental",
            concept="irs_soi.medical_dental_expense_deduction",
            domain="all_individual_income_tax_returns",
            source_name="irs_soi",
            source_table="Historic Table 2 state broad totals",
            period={"type": "tax_year", "value": 2022},
            geography=geography,
            value=11_456_144_000,
            unit="usd",
        ),
        _consumer_fact(
            "state-ca-qbi-returns",
            concept="irs_soi.returns_with_qualified_business_income_deduction",
            domain="all_individual_income_tax_returns",
            source_name="irs_soi",
            source_table="Historic Table 2 state broad totals",
            period={"type": "tax_year", "value": 2022},
            geography=geography,
            value=499_080,
        ),
        _consumer_fact(
            "state-ca-qbi",
            concept="irs_soi.qualified_business_income_deduction",
            domain="all_individual_income_tax_returns",
            source_name="irs_soi",
            source_table="Historic Table 2 state broad totals",
            period={"type": "tax_year", "value": 2022},
            geography=geography,
            value=4_400_400_000,
            unit="usd",
        ),
        _consumer_fact(
            "state-ca-rental-returns",
            concept="irs_soi.returns_with_rental_royalty_income",
            domain="all_individual_income_tax_returns",
            source_name="irs_soi",
            source_table="Historic Table 2 state broad totals",
            period={"type": "tax_year", "value": 2022},
            geography=geography,
            value=1_315_410,
        ),
        _consumer_fact(
            "state-ca-rental",
            concept="irs_soi.rental_royalty_income",
            domain="all_individual_income_tax_returns",
            source_name="irs_soi",
            source_table="Historic Table 2 state broad totals",
            period={"type": "tax_year", "value": 2022},
            geography=geography,
            value=14_331_993_000,
            unit="usd",
        ),
        _consumer_fact(
            "state-ca-ctc-returns",
            concept="irs_soi.returns_with_child_tax_credit",
            domain="all_individual_income_tax_returns",
            source_name="irs_soi",
            source_table="Historic Table 2 state broad totals",
            period={"type": "tax_year", "value": 2022},
            geography=geography,
            value=4_626_510,
        ),
        _consumer_fact(
            "state-ca-ctc",
            concept="irs_soi.child_tax_credit",
            domain="all_individual_income_tax_returns",
            source_name="irs_soi",
            source_table="Historic Table 2 state broad totals",
            period={"type": "tax_year", "value": 2022},
            geography=geography,
            value=9_724_583_000,
            unit="usd",
        ),
        _consumer_fact(
            "state-ca-actc-returns",
            concept="irs_soi.returns_with_additional_child_tax_credit",
            domain="all_individual_income_tax_returns",
            source_name="irs_soi",
            source_table="Historic Table 2 state broad totals",
            period={"type": "tax_year", "value": 2022},
            geography=geography,
            value=1_933_500,
        ),
        _consumer_fact(
            "state-ca-actc",
            concept="irs_soi.additional_child_tax_credit",
            domain="all_individual_income_tax_returns",
            source_name="irs_soi",
            source_table="Historic Table 2 state broad totals",
            period={"type": "tax_year", "value": 2022},
            geography=geography,
            value=3_605_628_000,
            unit="usd",
        ),
    ]
    consumer_jsonl.write_text(
        "\n".join(json.dumps(row, sort_keys=True) for row in rows) + "\n"
    )

    target_set = ArchConsumerFactJSONLTargetProvider(consumer_jsonl).load_target_set(
        TargetQuery(period=2022)
    )
    targets_by_arch_variable = {
        target.metadata["arch_variable"]: target for target in target_set.targets
    }

    qualified_dividends = targets_by_arch_variable["qualified_dividends_amount"]
    assert qualified_dividends.metadata["variable"] == "qualified_dividend_income"
    assert qualified_dividends.measure == "qualified_dividend_income"
    assert _target_filter_tuples(qualified_dividends) == {
        ("tax_unit_is_filer", "==", "1"),
        ("state_fips", "==", "06"),
    }

    schedule_c_returns = targets_by_arch_variable["schedule_c_income_returns"]
    assert schedule_c_returns.metadata["variable"] == "self_employment_income"
    assert schedule_c_returns.aggregation.value == "count"
    assert ("self_employment_income", ">", "0") in _target_filter_tuples(
        schedule_c_returns
    )

    partnership = targets_by_arch_variable["partnership_scorp_income_amount"]
    assert (
        partnership.metadata["variable"] == "tax_unit_partnership_s_corp_income"
    )
    assert partnership.measure == "tax_unit_partnership_s_corp_income"

    medical = targets_by_arch_variable["medical_dental_expense_amount"]
    assert medical.metadata["variable"] == "medical_expense_deduction"
    assert medical.measure == "medical_expense_deduction"

    qbi = targets_by_arch_variable["qbi_amount"]
    assert qbi.metadata["variable"] == "qualified_business_income_deduction"
    assert qbi.measure == "qualified_business_income_deduction"

    qbi_claims = targets_by_arch_variable["qbi_claims"]
    assert (
        qbi_claims.metadata["variable"]
        == "qualified_business_income_deduction"
    )
    assert qbi_claims.aggregation.value == "count"
    assert (
        "qualified_business_income_deduction",
        ">",
        "0",
    ) in _target_filter_tuples(qbi_claims)

    rental = targets_by_arch_variable["rental_royalty_income_amount"]
    assert rental.metadata["variable"] == "rental_income"
    assert rental.measure == "rental_income"

    rental_returns = targets_by_arch_variable["rental_royalty_income_returns"]
    assert rental_returns.metadata["variable"] == "rental_income"
    assert rental_returns.aggregation.value == "count"
    assert ("rental_income", ">", "0") in _target_filter_tuples(
        rental_returns
    )

    ctc = targets_by_arch_variable["ctc_amount"]
    assert ctc.metadata["variable"] == "non_refundable_ctc"
    assert ctc.measure == "non_refundable_ctc"

    ctc_claims = targets_by_arch_variable["ctc_claims"]
    assert ctc_claims.metadata["variable"] == "non_refundable_ctc"
    assert ctc_claims.aggregation.value == "count"
    assert ("non_refundable_ctc", ">", "0") in _target_filter_tuples(
        ctc_claims
    )

    actc = targets_by_arch_variable["actc_amount"]
    assert actc.metadata["variable"] == "refundable_ctc"
    assert actc.measure == "refundable_ctc"

    actc_claims = targets_by_arch_variable["actc_claims"]
    assert actc_claims.metadata["variable"] == "refundable_ctc"
    assert actc_claims.aggregation.value == "count"
    assert ("refundable_ctc", ">", "0") in _target_filter_tuples(
        actc_claims
    )


def test_arch_consumer_fact_jsonl_provider_maps_soi_alimony_concepts(
    tmp_path: Path,
) -> None:
    consumer_jsonl = tmp_path / "consumer_facts.jsonl"
    rows = [
        _consumer_fact(
            "soi-alimony-received-returns",
            concept="irs_soi.returns_with_alimony_received",
            domain="all_individual_income_tax_returns",
            source_name="irs_soi",
            source_table="Publication 1304 Table 1.4",
            period={"type": "tax_year", "value": 2023},
            value=183_582,
        ),
        _consumer_fact(
            "soi-alimony-received-amount",
            concept="irs_soi.alimony_received",
            domain="all_individual_income_tax_returns",
            source_name="irs_soi",
            source_table="Publication 1304 Table 1.4",
            period={"type": "tax_year", "value": 2023},
            value=6_686_429_000,
            unit="usd",
        ),
        _consumer_fact(
            "soi-alimony-paid-returns",
            concept="irs_soi.returns_with_alimony_paid",
            domain="all_individual_income_tax_returns",
            source_name="irs_soi",
            source_table="Publication 1304 Table 1.4",
            period={"type": "tax_year", "value": 2023},
            value=278_541,
        ),
        _consumer_fact(
            "soi-alimony-paid-amount",
            concept="irs_soi.alimony_paid",
            domain="all_individual_income_tax_returns",
            source_name="irs_soi",
            source_table="Publication 1304 Table 1.4",
            period={"type": "tax_year", "value": 2023},
            value=7_497_135_000,
            unit="usd",
        ),
    ]
    consumer_jsonl.write_text(
        "\n".join(json.dumps(row, sort_keys=True) for row in rows) + "\n"
    )

    target_set = ArchConsumerFactJSONLTargetProvider(consumer_jsonl).load_target_set(
        TargetQuery(period=2023)
    )
    targets_by_arch_variable = {
        target.metadata["arch_variable"]: target for target in target_set.targets
    }

    received_amount = targets_by_arch_variable["alimony_received_amount"]
    assert received_amount.metadata["variable"] == "alimony_income"
    assert received_amount.measure == "alimony_income"

    received_returns = targets_by_arch_variable["alimony_received_returns"]
    assert received_returns.metadata["variable"] == "tax_unit_count"
    assert received_returns.aggregation.value == "count"
    assert ("alimony_income", ">", "0") in _target_filter_tuples(
        received_returns
    )

    paid_amount = targets_by_arch_variable["alimony_paid_amount"]
    assert paid_amount.metadata["variable"] == "alimony_expense"
    assert paid_amount.measure == "alimony_expense"

    paid_returns = targets_by_arch_variable["alimony_paid_returns"]
    assert paid_returns.metadata["variable"] == "tax_unit_count"
    assert paid_returns.aggregation.value == "count"
    assert ("alimony_expense", ">", "0") in _target_filter_tuples(paid_returns)


def test_arch_consumer_fact_jsonl_provider_maps_eitc_by_agi_and_children(
    tmp_path: Path,
) -> None:
    consumer_jsonl = tmp_path / "consumer_facts.jsonl"
    row = _consumer_fact(
        "eitc-three-child-50k-75k-returns",
        concept="irs_soi.returns_with_total_earned_income_credit",
        domain="individual_income_tax_returns_with_earned_income_credit",
        source_name="irs_soi",
        source_table="Publication 1304 Table 2.5 EITC by AGI and qualifying children",
        period={"type": "tax_year", "value": 2022},
        value=97_411,
        constraints=(
            {
                "variable": "us:statutes/26/62#adjusted_gross_income",
                "operator": ">=",
                "value": 50_000,
                "unit": "usd",
                "role": "filter",
            },
            {
                "variable": "us:statutes/26/62#adjusted_gross_income",
                "operator": "<",
                "value": 75_000,
                "unit": "usd",
                "role": "filter",
            },
            {
                "variable": "us.tax.earned_income_credit_qualifying_children",
                "operator": "==",
                "value": 3,
                "unit": "count",
                "role": "filter",
            },
        ),
    )
    consumer_jsonl.write_text(json.dumps(row, sort_keys=True) + "\n")

    target_set = ArchConsumerFactJSONLTargetProvider(consumer_jsonl).load_target_set(
        TargetQuery(period=2022)
    )
    target = target_set.targets[0]

    assert target.metadata["arch_variable"] == "eitc_claims"
    assert target.metadata["variable"] == "eitc"
    assert target.aggregation.value == "count"
    assert _target_filter_tuples(target) == {
        ("eitc", ">", "0"),
        ("adjusted_gross_income", ">=", "50000"),
        ("adjusted_gross_income", "<", "75000"),
        ("eitc_child_count", "==", "3"),
    }


def test_arch_consumer_fact_coverage_accepts_eitc_child_count_totals(
    tmp_path: Path,
) -> None:
    consumer_jsonl = tmp_path / "consumer_facts.jsonl"
    rows = [
        _consumer_fact(
            "eitc-one-child-total-returns",
            concept="irs_soi.returns_with_total_earned_income_credit",
            domain="individual_income_tax_returns_with_earned_income_credit",
            source_name="irs_soi",
            source_table=(
                "Publication 1304 Table 2.5 EITC by AGI and qualifying children"
            ),
            period={"type": "tax_year", "value": 2022},
            value=8_490_417,
            constraints=(
                {
                    "variable": "us.tax.earned_income_credit_qualifying_children",
                    "operator": "==",
                    "value": 1,
                    "unit": "count",
                    "role": "filter",
                },
            ),
        ),
        _consumer_fact(
            "eitc-one-child-total-amount",
            concept="irs_soi.total_earned_income_credit",
            domain="individual_income_tax_returns_with_earned_income_credit",
            source_name="irs_soi",
            source_table=(
                "Publication 1304 Table 2.5 EITC by AGI and qualifying children"
            ),
            period={"type": "tax_year", "value": 2022},
            value=21_182_747_000,
            unit="usd",
            constraints=(
                {
                    "variable": "us.tax.earned_income_credit_qualifying_children",
                    "operator": "==",
                    "value": 1,
                    "unit": "count",
                    "role": "filter",
                },
            ),
        ),
    ]
    consumer_jsonl.write_text(
        "\n".join(json.dumps(row, sort_keys=True) for row in rows) + "\n"
    )
    provider = ArchConsumerFactJSONLTargetProvider(consumer_jsonl)

    report = summarize_arch_target_profile_coverage(
        provider,
        period=2022,
        profile_name="custom",
        target_cells=(
            {
                "variable": "eitc",
                "geo_level": "national",
                "domain_variable": "eitc_child_count",
            },
            {
                "variable": "tax_unit_count",
                "geo_level": "national",
                "domain_variable": "eitc_child_count",
            },
        ),
    )

    assert report.covered_cell_count == 2


def test_arch_consumer_fact_jsonl_provider_maps_us_admin_source_families(
    tmp_path: Path,
) -> None:
    consumer_jsonl = tmp_path / "consumer_facts.jsonl"
    rows = [
        _consumer_fact(
            "kff-aca-effectuated",
            concept="cms_aca.marketplace_effectuated_enrollment",
            domain="aca_marketplace_effectuated_enrollment",
            source_name="kff",
            source_table="Marketplace Effectuated Enrollment",
            geography={"level": "state", "id": "0400000US06", "name": "California"},
            value=1_795_695,
        ),
        _consumer_fact(
            "cms-medicaid-monthly",
            concept="cms_medicaid.total_medicaid_enrollment",
            domain="medicaid_chip_enrollment",
            source_name="cms_medicaid",
            source_table="Monthly Medicaid and CHIP Enrollment",
            geography={"level": "state", "id": "0400000US06", "name": "California"},
            period={"type": "month", "value": "2024-12"},
            value=13_500_000,
        ),
        _consumer_fact(
            "cms-nhe-medicaid",
            concept="cms_nhe.medicaid_title_xix_expenditures",
            domain="national_health_expenditures",
            source_name="cms_nhe",
            source_table="National Health Expenditures",
            value=931_692_000_000,
            unit="usd",
        ),
        _consumer_fact(
            "snap-benefits",
            concept="usda_snap.total_benefits",
            domain="supplemental_nutrition_assistance_program",
            source_name="usda_snap",
            source_table="SNAP fiscal year benefits",
            value=100_000_000_000,
            unit="usd",
        ),
        _consumer_fact(
            "snap-households",
            concept="usda_snap.average_monthly_households",
            domain="supplemental_nutrition_assistance_program",
            source_name="usda_snap",
            source_table="SNAP fiscal year participation",
            geography={"level": "state", "id": "0400000US06", "name": "California"},
            value=2_100_000,
        ),
        _consumer_fact(
            "tanf-cash",
            concept="hhs_acf_tanf.cash_assistance_expenditures",
            domain="tanf_cash_assistance",
            source_name="hhs_acf_tanf",
            source_table="TANF Financial Data",
            period={"type": "fiscal_year", "value": 2024},
            value=7_788_317_475,
            unit="usd",
        ),
        _consumer_fact(
            "tanf-total-families",
            concept="hhs_acf_tanf.average_monthly_tanf_total_families",
            domain="tanf_caseload",
            source_name="hhs_acf_tanf",
            source_table="TANF Caseload Data 2024",
            period={"type": "fiscal_year", "value": 2024},
            value=841_209,
        ),
        _consumer_fact(
            "liheap-households",
            concept="hhs_acf_liheap.households_served_by_state_programs",
            domain="liheap_state_programs",
            source_name="hhs_acf_liheap",
            source_table="LIHEAP FY2024 National Profile (All States)",
            period={"type": "fiscal_year", "value": 2024},
            value=5_876_646,
            constraints=(
                {"variable": "program", "operator": "==", "value": "liheap"},
                {
                    "variable": "administering_entity",
                    "operator": "==",
                    "value": "state_programs",
                },
            ),
        ),
        _consumer_fact(
            "stc-income-tax",
            concept="census_stc.individual_income_tax_collections",
            domain="state_government_tax_collections",
            source_name="census_stc",
            source_table="FY2024 STC Flat File item T40",
            geography={"level": "state", "id": "0400000US06", "name": "California"},
            period={"type": "fiscal_year", "value": 2024},
            value=123_101_651_000,
            unit="usd",
        ),
        _consumer_fact(
            "ssa-retirement",
            concept="ssa.annual_oasdi_or_ssi_payment_amount",
            domain="social_security_and_ssi_payments",
            source_name="ssa",
            source_table="Annual Statistical Supplement",
            value=1_111_728_000_000,
            unit="usd",
            constraints=(
                {
                    "variable": "us_social_security_and_ssi.program_payment_type",
                    "operator": "==",
                    "value": "social_security_retirement_benefits",
                },
            ),
        ),
        _consumer_fact(
            "ssa-ssi",
            concept="ssa.annual_oasdi_or_ssi_payment_amount",
            domain="social_security_and_ssi_payments",
            source_name="ssa",
            source_table="Annual Statistical Supplement",
            value=63_079_493_000,
            unit="usd",
            constraints=(
                {
                    "variable": "us_social_security_and_ssi.program_payment_type",
                    "operator": "==",
                    "value": "ssi_payments",
                },
            ),
        ),
        _consumer_fact(
            "pep-age",
            concept="census_pep.resident_population",
            domain="resident_population",
            source_name="census_pep",
            source_table="Annual Estimates by Age and Sex",
            value=18_599_314,
            constraints=(
                {"variable": "age", "operator": ">=", "value": 0, "unit": "years"},
                {"variable": "age", "operator": "<", "value": 5, "unit": "years"},
            ),
        ),
        _consumer_fact(
            "aca-oep-average-aptc",
            concept="cms_aca.average_monthly_aptc",
            domain="aca_marketplace_qhp_selections",
            source_name="cms_aca",
            source_table="OEP State-Level Public Use File",
            geography={"level": "state", "id": "0400000US06", "name": "California"},
            value=526,
            unit="usd",
        ),
        _consumer_fact(
            "w2-traditional-401k",
            concept="irs_soi.form_w2_401k_elective_deferrals",
            domain="form_w2_items",
            source_name="irs_soi",
            source_table="Form W-2 Statistics Table 4.B",
            period={"type": "tax_year", "value": 2024},
            value=277_859_181_000,
            unit="usd",
        ),
        _consumer_fact(
            "w2-roth-401k",
            concept="irs_soi.form_w2_designated_roth_401k_contributions",
            domain="form_w2_items",
            source_name="irs_soi",
            source_table="Form W-2 Statistics Table 4.B",
            period={"type": "tax_year", "value": 2024},
            value=32_302_509_000,
            unit="usd",
        ),
        _consumer_fact(
            "soi-keogh",
            concept="irs_soi.payments_to_keogh_plan",
            domain="all_individual_income_tax_returns",
            source_name="irs_soi",
            source_table="Publication 1304 Table 1.4",
            period={"type": "tax_year", "value": 2024},
            value=30_130_848_000,
            unit="usd",
        ),
    ]
    consumer_jsonl.write_text(
        "\n".join(json.dumps(row, sort_keys=True) for row in rows) + "\n"
    )
    provider = ArchConsumerFactJSONLTargetProvider(consumer_jsonl)

    report = summarize_arch_target_profile_coverage(
        provider,
        period=2024,
        profile_name="custom",
        target_cells=(
            {
                "variable": "person_count",
                "geo_level": "state",
                "domain_variable": "aca_ptc",
            },
            {
                "variable": "person_count",
                "geo_level": "state",
                "domain_variable": "medicaid_enrolled",
            },
            {"variable": "medicaid", "geo_level": "national", "domain_variable": None},
            {"variable": "snap", "geo_level": "national", "domain_variable": None},
            {
                "variable": "household_count",
                "geo_level": "state",
                "domain_variable": "snap",
            },
            {"variable": "tanf", "geo_level": "national", "domain_variable": None},
            {
                "variable": "spm_unit_count",
                "geo_level": "national",
                "domain_variable": "tanf",
            },
            {
                "variable": "household_count",
                "geo_level": "national",
                "domain_variable": "spm_unit_energy_subsidy_reported",
            },
            {
                "variable": "state_income_tax",
                "geo_level": "state",
                "domain_variable": None,
            },
            {
                "variable": "social_security_retirement",
                "geo_level": "national",
                "domain_variable": None,
            },
            {"variable": "ssi", "geo_level": "national", "domain_variable": None},
            {
                "variable": "person_count",
                "geo_level": "national",
                "domain_variable": "age",
            },
            {
                "variable": "traditional_401k_contributions",
                "geo_level": "national",
                "domain_variable": None,
            },
            {
                "variable": "roth_401k_contributions",
                "geo_level": "national",
                "domain_variable": None,
            },
            {
                "variable": "self_employed_pension_contribution_ald",
                "geo_level": "national",
                "domain_variable": None,
            },
        ),
    )

    assert report.target_cell_count == 15
    assert report.covered_cell_count == 15

    target_set = provider.load_target_set(TargetQuery(period=2024))
    targets_by_arch_variable = {
        target.metadata["arch_variable"]: target for target in target_set.targets
    }
    assert (
        targets_by_arch_variable["aca_marketplace_enrollment"].metadata["variable"]
        == "person_count"
    )
    assert (
        targets_by_arch_variable["medicaid_total_enrollment"].metadata["variable"]
        == "person_count"
    )
    assert targets_by_arch_variable["medicaid_benefits"].measure == "medicaid"
    assert targets_by_arch_variable["snap_benefits"].measure == "snap"
    assert (
        targets_by_arch_variable["snap_household_count"].metadata["variable"]
        == "household_count"
    )
    assert targets_by_arch_variable["tanf_cash_assistance"].measure == "tanf"
    assert (
        targets_by_arch_variable["tanf_family_count"].metadata["variable"]
        == "spm_unit_count"
    )
    liheap_target = targets_by_arch_variable["liheap_household_count"]
    assert liheap_target.metadata["variable"] == "household_count"
    assert {
        (target_filter.feature, target_filter.operator.value, target_filter.value)
        for target_filter in liheap_target.filters
    } == {("spm_unit_energy_subsidy_reported", ">", 0)}
    assert (
        targets_by_arch_variable["state_individual_income_tax_collections"].measure
        == "state_income_tax"
    )
    assert (
        targets_by_arch_variable["social_security_retirement_benefits"].measure
        == "social_security_retirement"
    )
    assert targets_by_arch_variable["ssi_payments"].measure == "ssi"
    traditional_401k = targets_by_arch_variable["traditional_401k_contributions"]
    assert traditional_401k.measure == "traditional_401k_contributions"
    assert traditional_401k.entity.value == "person"
    roth_401k = targets_by_arch_variable["roth_401k_contributions"]
    assert roth_401k.measure == "roth_401k_contributions"
    assert roth_401k.entity.value == "person"
    self_employed_pension = targets_by_arch_variable[
        "self_employed_pension_contribution_ald"
    ]
    assert self_employed_pension.measure == "self_employed_pension_contribution_ald"
    assert self_employed_pension.entity.value == "tax_unit"
    assert "aca_average_monthly_aptc" not in targets_by_arch_variable


def test_arch_consumer_fact_jsonl_provider_maps_medicare_part_b_premiums(
    tmp_path: Path,
) -> None:
    consumer_jsonl = tmp_path / "consumer_facts.jsonl"
    rows = [
        _consumer_fact(
            "cms-medicare-part-b-premiums",
            concept="cms_medicare.part_b_premium_income",
            domain="medicare_financing",
            source_name="cms_medicare",
            source_table="2025 Medicare Trustees Report Table III.C3",
            period={"type": "calendar_year", "value": 2024},
            value=139_837_000_000,
            unit="usd",
            constraints=(
                {"variable": "amount_basis", "operator": "==", "value": "actual"},
                {"variable": "medicare.part", "operator": "==", "value": "part_b"},
                {
                    "variable": "medicare.financing_component",
                    "operator": "==",
                    "value": "premiums_from_enrollees",
                },
            ),
        ),
    ]
    consumer_jsonl.write_text(
        "\n".join(json.dumps(row, sort_keys=True) for row in rows) + "\n"
    )

    target_set = ArchConsumerFactJSONLTargetProvider(consumer_jsonl).load_target_set(
        TargetQuery(period=2024)
    )

    target = target_set.targets[0]
    assert target.metadata["arch_variable"] == "medicare_part_b_premiums"
    assert target.metadata["variable"] == "medicare_part_b_premiums"
    assert target.measure == "medicare_part_b_premiums"
    assert target.entity.value == "person"
    assert target.filters == ()


def test_arch_consumer_fact_jsonl_provider_maps_ssi_detail_targets(
    tmp_path: Path,
) -> None:
    consumer_jsonl = tmp_path / "consumer_facts.jsonl"
    rows = [
        _consumer_fact(
            "ssa-ssi-aged-recipients",
            concept="ssa.ssi_recipient_count",
            domain="social_security_and_ssi_payments",
            source_name="ssa",
            source_table="SSI Annual Statistical Report 2024",
            period={"type": "calendar_year", "value": 2024},
            value=1_160_608,
            unit="count",
            constraints=(
                {"variable": "ssi_category", "operator": "==", "value": "aged"},
            ),
        ),
        _consumer_fact(
            "ssa-ca-ssi-payments",
            concept="ssa.ssi_payment_amount",
            domain="social_security_and_ssi_payments",
            source_name="ssa",
            source_table="SSI Annual Statistical Report 2024",
            geography={"level": "state", "id": "0400000US06", "name": "California"},
            period={"type": "calendar_year", "value": 2024},
            value=12_800_000_000,
            unit="usd",
        ),
        _consumer_fact(
            "ssa-ca-ssi-disabled-recipients",
            concept="ssa.ssi_recipient_count",
            domain="social_security_and_ssi_payments",
            source_name="ssa",
            source_table="SSI Annual Statistical Report 2024",
            geography={"level": "state", "id": "0400000US06", "name": "California"},
            period={"type": "calendar_year", "value": 2024},
            value=877_000,
            unit="count",
            constraints=(
                {"variable": "ssi_category", "operator": "==", "value": "disabled"},
            ),
        ),
    ]
    consumer_jsonl.write_text(
        "\n".join(json.dumps(row, sort_keys=True) for row in rows) + "\n"
    )

    target_set = ArchConsumerFactJSONLTargetProvider(consumer_jsonl).load_target_set(
        TargetQuery(period=2024)
    )

    def find_target(
        arch_variable: str,
        required_filters: set[tuple[str, str, object]],
    ):
        for target in target_set.targets:
            if target.metadata["arch_variable"] != arch_variable:
                continue
            filters = {
                (target_filter.feature, target_filter.operator.value, target_filter.value)
                for target_filter in target.filters
            }
            if required_filters.issubset(filters):
                return target
        raise AssertionError(
            f"Missing {arch_variable} target with filters {required_filters}"
        )

    aged_count = find_target(
        "ssi_recipients",
        {("is_ssi_aged", "==", 1), ("ssi", ">", 0)},
    )
    assert aged_count.measure is None
    assert aged_count.entity.value == "person"
    assert aged_count.value == pytest.approx(1_160_608)
    assert aged_count.metadata["arch_variable"] == "ssi_recipients"
    assert {
        (target_filter.feature, target_filter.operator.value, target_filter.value)
        for target_filter in aged_count.filters
    } == {("is_ssi_aged", "==", 1), ("ssi", ">", 0)}

    ca_payments = find_target(
        "ssi_total_payments",
        {("state_fips", "==", "06")},
    )
    assert ca_payments.measure == "ssi"
    assert ca_payments.entity.value == "person"
    assert ca_payments.value == pytest.approx(12_800_000_000)
    assert ca_payments.metadata["arch_variable"] == "ssi_total_payments"
    assert {
        (target_filter.feature, target_filter.operator.value, target_filter.value)
        for target_filter in ca_payments.filters
    } == {("state_fips", "==", "06")}

    ca_disabled_count = find_target(
        "ssi_recipients",
        {("is_ssi_disabled", "==", 1), ("ssi", ">", 0), ("state_fips", "==", "06")},
    )
    assert ca_disabled_count.measure is None
    assert ca_disabled_count.value == pytest.approx(877_000)
    assert {
        (target_filter.feature, target_filter.operator.value, target_filter.value)
        for target_filter in ca_disabled_count.filters
    } == {
        ("is_ssi_disabled", "==", 1),
        ("ssi", ">", 0),
        ("state_fips", "==", "06"),
    }


def test_arch_consumer_fact_jsonl_provider_maps_fed_household_net_worth(
    tmp_path: Path,
) -> None:
    consumer_jsonl = tmp_path / "consumer_facts.jsonl"
    rows = [
        _consumer_fact(
            "fed-z1-net-worth",
            concept="federal_reserve.z1.households_nonprofits_net_worth",
            domain="household_balance_sheet",
            source_name="federal_reserve",
            source_table="Z.1 B.101 Households and nonprofit organizations",
            period={"type": "calendar_year", "value": 2024},
            value=169_619_200_000_000,
            unit="usd",
        ),
    ]
    consumer_jsonl.write_text(
        "\n".join(json.dumps(row, sort_keys=True) for row in rows) + "\n"
    )

    target_set = ArchConsumerFactJSONLTargetProvider(consumer_jsonl).load_target_set(
        TargetQuery(period=2024)
    )

    target = target_set.targets[0]
    assert target.metadata["arch_variable"] == "net_worth_amount"
    assert target.metadata["variable"] == "net_worth"
    assert target.measure == "net_worth"
    assert target.entity.value == "household"
    assert target.filters == ()


def test_arch_consumer_fact_jsonl_provider_maps_decennial_sld_facts(
    tmp_path: Path,
) -> None:
    consumer_jsonl = tmp_path / "consumer_facts.jsonl"
    rows = [
        _consumer_fact(
            "census-cd119-sldu-population",
            concept="census_decennial.resident_population",
            domain="resident_population",
            source_name="census_decennial",
            source_table="2020 Census CD119 California SLD P1",
            geography={
                "level": "state_legislative_district_upper",
                "id": "610U900US06001",
                "name": "State Senate District 1",
            },
            value=943_108,
        ),
        _consumer_fact(
            "census-cd119-sldl-households",
            concept="census_decennial.occupied_housing_units",
            domain="households",
            source_name="census_decennial",
            source_table="2020 Census CD119 California SLD H3",
            geography={
                "level": "state_legislative_district_lower",
                "id": "620L900US06080",
                "name": "Assembly District 80",
            },
            value=154_291,
        ),
    ]
    consumer_jsonl.write_text(
        "\n".join(json.dumps(row, sort_keys=True) for row in rows) + "\n"
    )

    provider = ArchConsumerFactJSONLTargetProvider(consumer_jsonl)
    report = summarize_arch_target_profile_coverage(
        provider,
        period=2024,
        profile_name="custom",
        target_cells=(
            {
                "variable": "person_count",
                "geo_level": "sldu",
                "geographic_id": "CA-SLDU-001",
                "domain_variable": None,
            },
            {
                "variable": "household_count",
                "geo_level": "sldl",
                "geographic_id": "CA-SLDL-080",
                "domain_variable": None,
            },
        ),
    )

    assert report.covered_cell_count == 2
    target_set = provider.load_target_set(TargetQuery(period=2024))
    targets_by_arch_variable = {
        target.metadata["arch_variable"]: target for target in target_set.targets
    }
    population = targets_by_arch_variable["population"]
    households = targets_by_arch_variable["household_count"]

    assert population.value == 943_108
    assert population.metadata["source"] == "CENSUS_DECENNIAL"
    assert population.metadata["geo_level"] == "sldu"
    assert {
        (
            target_filter.feature,
            str(getattr(target_filter.operator, "value", target_filter.operator)),
            str(target_filter.value),
        )
        for target_filter in population.filters
    } == {("sldu_id", "==", "CA-SLDU-001")}
    assert households.value == 154_291
    assert households.metadata["geo_level"] == "sldl"
    assert {
        (
            target_filter.feature,
            str(getattr(target_filter.operator, "value", target_filter.operator)),
            str(target_filter.value),
        )
        for target_filter in households.filters
    } == {("sldl_id", "==", "CA-SLDL-080")}


def test_arch_consumer_fact_jsonl_provider_maps_acs_cd_age_population(
    tmp_path: Path,
) -> None:
    consumer_jsonl = tmp_path / "consumer_facts.jsonl"
    rows = [
        _consumer_fact(
            "acs-cd119-age-population",
            concept="census_acs.person_count",
            domain="total_population",
            source_name="census_acs",
            source_table="ACS 2024 1-year subject table S0101",
            geography={
                "level": "congressional_district",
                "id": "5001900US0101",
                "name": "Alabama Congressional District 1",
            },
            value=39_908,
            constraints=(
                {"variable": "age", "operator": ">=", "value": 0, "unit": "years"},
                {"variable": "age", "operator": "<", "value": 5, "unit": "years"},
            ),
        ),
        _consumer_fact(
            "acs-cd119-households",
            concept="census_acs.household_count",
            domain="households",
            source_name="census_acs",
            source_table="ACS 2024 1-year subject table S2201",
            geography={
                "level": "congressional_district",
                "id": "5001900US0101",
                "name": "Alabama Congressional District 1",
            },
            value=300_636,
        ),
        _consumer_fact(
            "acs-cd119-snap-households",
            concept="census_acs.household_count",
            domain="households",
            source_name="census_acs",
            source_table="ACS 2024 1-year subject table S2201",
            geography={
                "level": "congressional_district",
                "id": "5001900US0101",
                "name": "Alabama Congressional District 1",
            },
            value=34_742,
            constraints=(
                {
                    "variable": "snap_receipt_status",
                    "operator": "==",
                    "value": "receiving_food_stamps_snap",
                },
            ),
        ),
    ]
    consumer_jsonl.write_text(
        "\n".join(json.dumps(row, sort_keys=True) for row in rows) + "\n"
    )

    provider = ArchConsumerFactJSONLTargetProvider(consumer_jsonl)
    report = summarize_arch_target_profile_coverage(
        provider,
        period=2024,
        profile_name="custom",
        target_cells=(
            {
                "variable": "person_count",
                "geo_level": "district",
                "geographic_id": "0101",
                "domain_variable": "age",
            },
            {
                "variable": "household_count",
                "geo_level": "district",
                "geographic_id": "0101",
                "domain_variable": None,
            },
            {
                "variable": "household_count",
                "geo_level": "district",
                "geographic_id": "0101",
                "domain_variable": "snap",
            },
        ),
    )

    assert report.covered_cell_count == 3
    target_set = provider.load_target_set(TargetQuery(period=2024))
    targets_by_key = {
        target.metadata["arch_source_record_id"]: target
        for target in target_set.targets
    }
    target = targets_by_key["census_acs.acs-cd119-age-population"]
    households = targets_by_key["census_acs.acs-cd119-households"]
    snap_households = targets_by_key["census_acs.acs-cd119-snap-households"]

    assert target.metadata["source"] == "CENSUS_ACS"
    assert target.metadata["arch_variable"] == "population"
    assert target.metadata["variable"] == "person_count"
    assert target.metadata["geo_level"] == "district"
    assert {
        (
            target_filter.feature,
            str(getattr(target_filter.operator, "value", target_filter.operator)),
            str(target_filter.value),
        )
        for target_filter in target.filters
    } == {
        ("age", ">=", "0"),
        ("age", "<", "5"),
        ("congressional_district_geoid", "==", "0101"),
    }
    assert households.metadata["arch_variable"] == "household_count"
    assert households.metadata["variable"] == "household_count"
    assert {
        (
            target_filter.feature,
            str(getattr(target_filter.operator, "value", target_filter.operator)),
            str(target_filter.value),
        )
        for target_filter in households.filters
    } == {("congressional_district_geoid", "==", "0101")}
    assert snap_households.metadata["arch_variable"] == "household_count"
    assert {
        (
            target_filter.feature,
            str(getattr(target_filter.operator, "value", target_filter.operator)),
            str(target_filter.value),
        )
        for target_filter in snap_households.filters
    } == {
        ("congressional_district_geoid", "==", "0101"),
        ("snap", ">", "0"),
    }


def test_arch_consumer_fact_jsonl_provider_maps_census_population_projection(
    tmp_path: Path,
) -> None:
    consumer_jsonl = tmp_path / "consumer_facts.jsonl"
    rows = [
        _consumer_fact(
            "census-popproj-age-0",
            concept="census.population_projection",
            domain="population_projection",
            source_name="census_population_projections",
            source_table="2023 National Population Projections Main Series",
            period={"type": "calendar_year", "value": 2024},
            value=3_636_897,
            constraints=(
                {"variable": "age", "operator": ">=", "value": 0, "unit": "years"},
                {"variable": "age", "operator": "<", "value": 1, "unit": "years"},
            ),
        ),
    ]
    consumer_jsonl.write_text(
        "\n".join(json.dumps(row, sort_keys=True) for row in rows) + "\n"
    )

    target_set = ArchConsumerFactJSONLTargetProvider(consumer_jsonl).load_target_set(
        TargetQuery(period=2024)
    )

    target = target_set.targets[0]
    assert target.metadata["arch_variable"] == "population"
    assert target.metadata["variable"] == "person_count"
    assert {
        (
            target_filter.feature,
            str(getattr(target_filter.operator, "value", target_filter.operator)),
            str(target_filter.value),
        )
        for target_filter in target.filters
    } == {("age", ">=", "0"), ("age", "<", "1")}


def test_arch_consumer_fact_jsonl_provider_normalizes_legacy_sld_ids(
    tmp_path: Path,
) -> None:
    consumer_jsonl = tmp_path / "consumer_facts.jsonl"
    rows = [
        _consumer_fact(
            "legacy-sldu-population",
            concept="census_decennial.resident_population",
            domain="resident_population",
            source_name="census_decennial",
            source_table="Legacy SLD fixture",
            geography={
                "level": "state_senate_district",
                "id": "CA-SD-1",
                "name": "State Senate District 1",
            },
            value=943_108,
        ),
        _consumer_fact(
            "legacy-sldl-households",
            concept="census_decennial.occupied_housing_units",
            domain="households",
            source_name="census_decennial",
            source_table="Legacy SLD fixture",
            geography={
                "level": "state_house_district",
                "id": "NY-AD-65",
                "name": "Assembly District 65",
            },
            value=154_291,
        ),
    ]
    consumer_jsonl.write_text(
        "\n".join(json.dumps(row, sort_keys=True) for row in rows) + "\n"
    )

    provider = ArchConsumerFactJSONLTargetProvider(consumer_jsonl)
    report = summarize_arch_target_profile_coverage(
        provider,
        period=2024,
        profile_name="custom",
        target_cells=(
            {
                "variable": "person_count",
                "geo_level": "sldu",
                "geographic_id": "06001",
                "domain_variable": None,
            },
            {
                "variable": "household_count",
                "geo_level": "sldl",
                "geographic_id": "36065",
                "domain_variable": None,
            },
        ),
    )

    assert report.covered_cell_count == 2
    target_set = provider.load_target_set(TargetQuery(period=2024))
    targets_by_arch_variable = {
        target.metadata["arch_variable"]: target for target in target_set.targets
    }

    assert {
        (target_filter.feature, str(target_filter.value))
        for target_filter in targets_by_arch_variable["population"].filters
    } == {("sldu_id", "CA-SLDU-001")}
    assert {
        (target_filter.feature, str(target_filter.value))
        for target_filter in targets_by_arch_variable["household_count"].filters
    } == {("sldl_id", "NY-SLDL-065")}


def test_arch_consumer_fact_jsonl_provider_maps_bea_full_population_amounts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        arch_module,
        "ARCH_NATIONAL_ROLLUP_STATE_FIPS",
        frozenset({"06", "36"}),
    )
    consumer_jsonl = tmp_path / "consumer_facts.jsonl"
    rows = [
        _consumer_fact(
            "bea-nipa-wages",
            concept="bea_nipa.wages_and_salaries",
            domain="personal_income",
            source_name="bea",
            source_table="NIPA annual total wages and salaries",
            value=11_000_000_000_000,
            unit="usd",
        ),
        _consumer_fact(
            "bea-nipa-proprietors",
            concept=(
                "bea_nipa.proprietors_income_with_inventory_valuation_and_capital_consumption_adjustments"
            ),
            domain="personal_income",
            source_name="bea",
            source_table="NIPA annual personal income components",
            value=2_000_000_000_000,
            unit="usd",
        ),
        _consumer_fact(
            "bea-regional-us-wages",
            concept="bea_regional.wages_and_salaries",
            domain="personal_income",
            source_name="bea",
            source_table="SAINC5N",
            value=12_300_000_000_000,
            unit="usd",
        ),
        _consumer_fact(
            "bea-regional-us-proprietors",
            concept="bea_regional.proprietors_income",
            domain="personal_income",
            source_name="bea",
            source_table="SAINC5N",
            value=2_020_000_000_000,
            unit="usd",
        ),
        _consumer_fact(
            "bea-regional-ca-wages",
            concept="bea_regional.wages_and_salaries",
            domain="personal_income",
            source_name="bea",
            source_table="SAINC5N",
            geography={"level": "state", "id": "0400000US06", "name": "California"},
            value=1_500_000_000_000,
            unit="usd",
        ),
        _consumer_fact(
            "bea-regional-ca-supplements",
            concept="bea_regional.supplements_to_wages_and_salaries",
            domain="personal_income",
            source_name="bea",
            source_table="SAINC5N",
            geography={"level": "state", "id": "0400000US06", "name": "California"},
            value=300_000_000_000,
            unit="usd",
        ),
        _consumer_fact(
            "bea-regional-ca-contributions",
            concept="bea_regional.contributions_for_government_social_insurance",
            domain="personal_income",
            source_name="bea",
            source_table="SAINC5N",
            geography={"level": "state", "id": "0400000US06", "name": "California"},
            value=200_000_000_000,
            unit="usd",
        ),
        _consumer_fact(
            "bea-regional-ca-residence",
            concept="bea_regional.residence_adjustment",
            domain="personal_income",
            source_name="bea",
            source_table="SAINC5N",
            geography={"level": "state", "id": "0400000US06", "name": "California"},
            value=40_000_000_000,
            unit="usd",
        ),
        _consumer_fact(
            "bea-regional-ny-wages",
            concept="bea_regional.wages_and_salaries",
            domain="personal_income",
            source_name="bea",
            source_table="SAINC5N",
            geography={"level": "state", "id": "0400000US36", "name": "New York"},
            value=2_000_000_000_000,
            unit="usd",
        ),
        _consumer_fact(
            "bea-regional-ny-supplements",
            concept="bea_regional.supplements_to_wages_and_salaries",
            domain="personal_income",
            source_name="bea",
            source_table="SAINC5N",
            geography={"level": "state", "id": "0400000US36", "name": "New York"},
            value=400_000_000_000,
            unit="usd",
        ),
        _consumer_fact(
            "bea-regional-ny-contributions",
            concept="bea_regional.contributions_for_government_social_insurance",
            domain="personal_income",
            source_name="bea",
            source_table="SAINC5N",
            geography={"level": "state", "id": "0400000US36", "name": "New York"},
            value=100_000_000_000,
            unit="usd",
        ),
        _consumer_fact(
            "bea-regional-ny-residence",
            concept="bea_regional.residence_adjustment",
            domain="personal_income",
            source_name="bea",
            source_table="SAINC5N",
            geography={"level": "state", "id": "0400000US36", "name": "New York"},
            value=-50_000_000_000,
            unit="usd",
        ),
        _consumer_fact(
            "bea-regional-ca-proprietors",
            concept="bea_regional.proprietors_income",
            domain="personal_income",
            source_name="bea",
            source_table="SAINC5N",
            geography={"level": "state", "id": "0400000US06", "name": "California"},
            value=180_000_000_000,
            unit="usd",
        ),
        _consumer_fact(
            "bea-nipa-dividends",
            concept="bea_nipa.personal_dividend_income",
            domain="personal_income",
            source_name="bea",
            source_table="NIPA annual personal income components",
            value=2_100_000_000_000,
            unit="usd",
        ),
        _consumer_fact(
            "bea-nipa-rental",
            concept=(
                "bea_nipa.rental_income_of_persons_with_capital_consumption_adjustment"
            ),
            domain="personal_income",
            source_name="bea",
            source_table="NIPA annual personal income components",
            value=1_000_000_000_000,
            unit="usd",
        ),
        _consumer_fact(
            "bea-nipa-social-security",
            concept="bea_nipa.social_security_benefits",
            domain="personal_current_transfer_receipts",
            source_name="bea",
            source_table="NIPA annual personal income components",
            value=1_500_000_000_000,
            unit="usd",
        ),
        _consumer_fact(
            "bea-nipa-medicaid",
            concept="bea_nipa.medicaid_benefits",
            domain="personal_current_transfer_receipts",
            source_name="bea",
            source_table="NIPA annual personal income components",
            value=900_000_000_000,
            unit="usd",
        ),
        _consumer_fact(
            "bea-nipa-ui",
            concept="bea_nipa.unemployment_insurance_benefits",
            domain="personal_current_transfer_receipts",
            source_name="bea",
            source_table="NIPA annual personal income components",
            value=30_000_000_000,
            unit="usd",
        ),
        _consumer_fact(
            "bea-nipa-saving-rate",
            concept="bea_nipa.personal_saving_rate",
            domain="personal_income",
            source_name="bea",
            source_table="NIPA annual personal income disposition",
            value=3.8,
            unit="percent",
        ),
    ]
    consumer_jsonl.write_text(
        "\n".join(json.dumps(row, sort_keys=True) for row in rows) + "\n"
    )
    provider = ArchConsumerFactJSONLTargetProvider(consumer_jsonl)

    report = summarize_arch_target_profile_coverage(
        provider,
        period=2024,
        profile_name="custom",
        target_cells=(
            {
                "variable": "employment_income_before_lsr",
                "geo_level": "national",
                "domain_variable": None,
            },
            {
                "variable": "employment_income_before_lsr",
                "geo_level": "state",
                "domain_variable": None,
            },
            {
                "variable": "self_employment_income",
                "geo_level": "national",
                "domain_variable": None,
            },
            {
                "variable": "self_employment_income",
                "geo_level": "state",
                "domain_variable": None,
            },
            {
                "variable": "dividend_income",
                "geo_level": "national",
                "domain_variable": None,
            },
            {
                "variable": "rental_income",
                "geo_level": "national",
                "domain_variable": None,
            },
            {
                "variable": "social_security",
                "geo_level": "national",
                "domain_variable": None,
            },
            {"variable": "medicaid", "geo_level": "national", "domain_variable": None},
            {
                "variable": "unemployment_compensation",
                "geo_level": "national",
                "domain_variable": None,
            },
        ),
    )

    assert report.target_cell_count == 9
    assert report.covered_cell_count == 7

    target_set = provider.load_target_set(TargetQuery(period=2024))
    targets_by_source_record = {
        target.metadata["arch_source_record_id"]: target
        for target in target_set.targets
    }
    assert set(targets_by_source_record) == {
        "bea.bea-nipa-wages",
        "bea.bea-nipa-proprietors",
        "microplex.derived.bea_state_wages.2024.06",
        "microplex.derived.bea_state_wages.2024.36",
        "bea.bea-regional-ca-proprietors",
        "bea.bea-nipa-dividends",
        "bea.bea-nipa-rental",
        "bea.bea-nipa-social-security",
        "bea.bea-nipa-medicaid",
        "bea.bea-nipa-ui",
    }
    assert targets_by_source_record["bea.bea-nipa-wages"].measure == (
        "employment_income_before_lsr"
    )
    assert (
        targets_by_source_record["bea.bea-nipa-wages"].metadata["arch_variable"]
        == "employment_income_before_lsr_amount"
    )
    assert targets_by_source_record["bea.bea-nipa-wages"].filters == ()
    assert targets_by_source_record["bea.bea-nipa-proprietors"].measure == (
        "proprietors_income_amount"
    )
    assert targets_by_source_record["bea.bea-nipa-proprietors"].metadata[
        "arch_concept"
    ] == (
        "bea_nipa.proprietors_income_with_inventory_valuation_and_capital_consumption_adjustments"
    )
    assert targets_by_source_record["bea.bea-nipa-proprietors"].filters == ()
    ca_state_wages = targets_by_source_record[
        "microplex.derived.bea_state_wages.2024.06"
    ]
    ny_state_wages = targets_by_source_record[
        "microplex.derived.bea_state_wages.2024.36"
    ]
    assert ca_state_wages.measure == (
        "employment_income_before_lsr"
    )
    assert ca_state_wages.metadata["arch_variable"] == (
        "employment_income_before_lsr_amount"
    )
    ca_adjusted = 1_500_000_000_000 + 40_000_000_000 * (
        1_500_000_000_000 / 2_000_000_000_000
    )
    ny_adjusted = 2_000_000_000_000 - 50_000_000_000 * (
        2_000_000_000_000 / 2_500_000_000_000
    )
    scale = 11_000_000_000_000 / (ca_adjusted + ny_adjusted)
    assert ca_state_wages.value == pytest.approx(ca_adjusted * scale)
    assert ny_state_wages.value == pytest.approx(ny_adjusted * scale)
    assert ca_state_wages.value + ny_state_wages.value == pytest.approx(
        11_000_000_000_000
    )
    assert targets_by_source_record["bea.bea-regional-ca-proprietors"].measure == (
        "proprietors_income_amount"
    )
    assert {
        (
            target_filter.feature,
            str(getattr(target_filter.operator, "value", target_filter.operator)),
            str(target_filter.value),
        )
        for target_filter in ca_state_wages.filters
    } == {("state_fips", "==", "06")}
    assert targets_by_source_record["bea.bea-nipa-dividends"].source == "BEA"
    assert "bea.bea-regional-us-wages" not in targets_by_source_record
    assert "bea.bea-regional-ca-wages" not in targets_by_source_record
    assert "bea.bea-regional-ca-supplements" not in targets_by_source_record
    assert "bea.bea-regional-ca-contributions" not in targets_by_source_record
    assert "bea.bea-regional-ca-residence" not in targets_by_source_record
    assert "bea.bea-regional-us-proprietors" not in targets_by_source_record
    assert not provider.load_target_set(
        TargetQuery(
            period=2024,
            provider_filters={"variables": ("self_employment_income",)},
        )
    ).targets
    assert all(
        target.metadata["arch_variable"] != "personal_saving_rate"
        for target in target_set.targets
    )


def test_arch_consumer_fact_jsonl_provider_skips_cbo_projection_concepts(
    tmp_path: Path,
) -> None:
    consumer_jsonl = tmp_path / "consumer_facts.jsonl"
    control = _consumer_fact(
        "soi-wages",
        concept="irs_soi.total_wages",
        domain="all_individual_income_tax_returns",
        source_name="irs_soi",
        source_table="Publication 1304 Table 1.1",
        period={"type": "tax_year", "value": 2024},
        value=10_000_000_000_000,
        unit="usd",
    )
    cbo_concepts = (
        "cbo.adjusted_gross_income_projection",
        "cbo.wages_and_salaries_projection",
        (
            "cbo.taxable_interest_and_ordinary_dividends_excluding_qualified_"
            "dividends_projection"
        ),
        "cbo.qualified_dividend_income_projection",
        "cbo.net_capital_gain_projection",
        "cbo.net_business_income_projection",
    )
    rows = [control]
    for concept in cbo_concepts:
        row = _consumer_fact(
            concept.rsplit(".", 1)[-1],
            concept=concept,
            domain="individual_income_tax_returns",
            source_name="cbo",
            source_table=(
                "Revenue Projections, by Category, February 2026, "
                "sheet 3.Individual Income Tax Details"
            ),
            period={"type": "tax_year", "value": 2024},
            value=1_000_000_000,
            unit="usd",
        )
        row["concept_alignment"] = {
            "canonical_concept": concept,
            "source_concept": concept.replace("_projection", ""),
            "relation": "source_label",
            "authority": "cbo",
            "evidence_notes": "Projection fixture.",
        }
        rows.append(row)
    consumer_jsonl.write_text(
        "\n".join(json.dumps(row, sort_keys=True) for row in rows) + "\n"
    )

    target_set = ArchConsumerFactJSONLTargetProvider(consumer_jsonl).load_target_set(
        TargetQuery(period=2024)
    )

    assert len(target_set.targets) == 1
    target = target_set.targets[0]
    assert target.metadata["arch_variable"] == "wages_salaries_amount"
    assert target.measure == "employment_income"
    assert target.metadata["arch_source_concept"] == "irs_soi.total_wages"


def test_arch_target_smoke_cli_reports_consumer_fact_jsonl_counts(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    consumer_jsonl = tmp_path / "consumer_facts.jsonl"
    _write_consumer_fact_jsonl(consumer_jsonl)

    exit_code = main_smoke(
        [
            "--arch-targets-db",
            str(consumer_jsonl),
            "--period",
            "2023",
            "--expected-target-count",
            "5",
            "--no-compose-model-year-targets",
        ]
    )
    payload = json.loads(capsys.readouterr().out)

    assert exit_code == 0
    assert payload["valid"]
    assert payload["target_count"] == 5
    assert payload["by_source"] == {"IRS_SOI": 5}
    assert payload["by_variable"] == {
        "adjusted_gross_income": 2,
        "income_tax": 1,
        "tax_unit_count": 2,
    }
    assert payload["errors"] == []
    assert payload["sample_targets"][0]["metadata"]["arch_aggregate_fact_key"]


def test_arch_target_smoke_cli_rejects_unexpected_target_count(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    consumer_jsonl = tmp_path / "consumer_facts.jsonl"
    _write_consumer_fact_jsonl(consumer_jsonl)

    exit_code = main_smoke(
        [
            "--arch-targets-db",
            str(consumer_jsonl),
            "--period",
            "2023",
            "--expected-target-count",
            "6",
            "--no-compose-model-year-targets",
        ]
    )
    payload = json.loads(capsys.readouterr().out)

    assert exit_code == 1
    assert not payload["valid"]
    assert payload["target_count"] == 5
    assert payload["errors"] == [
        {
            "code": "unexpected_target_count",
            "message": "Expected 6 targets, loaded 5.",
        }
    ]


def test_arch_target_parity_cli_accepts_matching_consumer_fact_jsonl(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    value_db = tmp_path / "value_targets.db"
    consumer_jsonl = tmp_path / "consumer_facts.jsonl"
    _create_value_constraint_target_db(value_db)
    _write_consumer_fact_jsonl(consumer_jsonl)

    exit_code = main_parity(
        [
            "--incumbent-arch-targets-db",
            str(value_db),
            "--candidate-arch-targets-db",
            str(consumer_jsonl),
            "--period",
            "2023",
            "--no-compose-model-year-targets",
        ]
    )
    payload = json.loads(capsys.readouterr().out)

    assert exit_code == 0
    assert payload["valid"]
    assert payload["counts"] == {
        "candidate_only_count": 0,
        "candidate_target_count": 5,
        "duplicate_identity_count": 0,
        "incumbent_only_count": 0,
        "incumbent_target_count": 5,
        "matched_count": 5,
        "value_mismatch_count": 0,
    }
    assert payload["errors"] == []
    assert payload["rows"][0]["status"] == "matched"


def test_arch_target_parity_cli_rejects_value_mismatch(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    value_db = tmp_path / "value_targets.db"
    consumer_jsonl = tmp_path / "consumer_facts.jsonl"
    _create_value_constraint_target_db(value_db)
    _write_consumer_fact_jsonl(consumer_jsonl)
    rows = [json.loads(line) for line in consumer_jsonl.read_text().splitlines()]
    rows[1]["value"] += 1_000
    consumer_jsonl.write_text(
        "\n".join(json.dumps(row, sort_keys=True) for row in rows) + "\n"
    )

    exit_code = main_parity(
        [
            "--incumbent-arch-targets-db",
            str(value_db),
            "--candidate-arch-targets-db",
            str(consumer_jsonl),
            "--period",
            "2023",
            "--no-compose-model-year-targets",
        ]
    )
    payload = json.loads(capsys.readouterr().out)

    assert exit_code == 1
    assert not payload["valid"]
    assert payload["counts"]["matched_count"] == 4
    assert payload["counts"]["value_mismatch_count"] == 1
    assert payload["errors"][0]["code"] == "value_mismatch"
    assert payload["errors"][0]["absolute_delta"] == 1_000


def test_arch_target_parity_cli_rejects_duplicate_candidate_identity(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    value_db = tmp_path / "value_targets.db"
    consumer_jsonl = tmp_path / "consumer_facts.jsonl"
    _create_value_constraint_target_db(value_db)
    _write_consumer_fact_jsonl(consumer_jsonl)
    lines = consumer_jsonl.read_text().splitlines()
    consumer_jsonl.write_text("\n".join([*lines, lines[0]]) + "\n")

    exit_code = main_parity(
        [
            "--incumbent-arch-targets-db",
            str(value_db),
            "--candidate-arch-targets-db",
            str(consumer_jsonl),
            "--period",
            "2023",
            "--no-compose-model-year-targets",
        ]
    )
    payload = json.loads(capsys.readouterr().out)

    assert exit_code == 1
    assert not payload["valid"]
    assert payload["counts"]["duplicate_identity_count"] == 1
    assert payload["errors"][0]["code"] == "duplicate_identity"
    assert payload["errors"][0]["candidate_target_count"] == 2


def test_arch_fact_provider_composes_latest_source_facts_to_model_year(
    tmp_path: Path,
) -> None:
    fact_db = tmp_path / "arch_facts.db"
    _create_arch_fact_db(fact_db)

    target_set = ArchFactSQLiteTargetProvider(fact_db).load_target_set(
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

    all_agi = next(
        target
        for target in target_set.targets
        if target.metadata["arch_aggregate_fact_key"] == "arch.fact.v1:all-agi"
    )
    assert all_agi.period == 2024
    assert all_agi.value == 15_286_017_359_000
    assert all_agi.metadata["arch_source_period"] == 2023
    assert all_agi.metadata["arch_model_period"] == 2024
    assert all_agi.metadata["arch_aging_amount_factor"] == 1
    assert all_agi.metadata["arch_aging_amount_method"] == (
        "source_fact_carry_forward_no_amount_reference"
    )


def test_arch_composite_source_facts_age_across_artifacts(
    tmp_path: Path,
) -> None:
    table_1_1_db = tmp_path / "arch_table_1_1.db"
    table_1_4_db = tmp_path / "arch_table_1_4.db"
    _create_arch_fact_db(table_1_1_db)
    _insert_arch_table_1_1_reference_totals(
        table_1_1_db,
        year=2022,
        return_count=160_602_107 / 1.1,
        adjusted_gross_income=15_286_017_359_000 / 1.1,
    )
    _create_arch_fact_db(table_1_4_db)
    _insert_arch_table_1_4_facts(table_1_4_db)
    provider = resolve_arch_sqlite_target_provider((table_1_1_db, table_1_4_db))

    target_set = provider.load_target_set(
        TargetQuery(
            period=2024,
            provider_filters={
                "sources": ["IRS_SOI"],
                "target_cells": [
                    {
                        "variable": "employment_income",
                        "geo_level": "national",
                        "domain_variable": "employment_income",
                    }
                ],
            },
        )
    )

    wages = next(
        target
        for target in target_set.targets
        if target.metadata["arch_aggregate_fact_key"]
        == "arch.fact.v1:t14-all-wages-amount"
    )
    assert wages.period == 2024
    assert wages.value == 10_500_000_000_000 * 1.1
    assert wages.metadata["arch_source_period"] == 2023
    assert wages.metadata["arch_aging_amount_factor"] == 1.1
    assert wages.metadata["arch_aging_amount_method"] == (
        "soi_total_agi_last_growth_extrapolation"
    )
    assert wages.metadata["arch_source_db_path"] == str(table_1_4_db)


def test_arch_provider_resolver_detects_source_fact_schema(tmp_path: Path) -> None:
    fact_db = tmp_path / "arch_facts.db"
    _create_arch_fact_db(fact_db)

    provider = resolve_arch_sqlite_target_provider(fact_db)

    assert isinstance(provider, ArchFactSQLiteTargetProvider)


def test_arch_provider_resolver_detects_consumer_fact_jsonl(tmp_path: Path) -> None:
    consumer_jsonl = tmp_path / "consumer_facts.jsonl"
    _write_consumer_fact_jsonl(consumer_jsonl)

    provider = resolve_arch_sqlite_target_provider(consumer_jsonl)

    assert isinstance(provider, ArchConsumerFactJSONLTargetProvider)


def test_arch_provider_resolver_combines_multiple_source_fact_dbs(
    tmp_path: Path,
) -> None:
    table_1_1_db = tmp_path / "arch_table_1_1.db"
    table_1_4_db = tmp_path / "arch_table_1_4.db"
    _create_arch_fact_db(table_1_1_db)
    _create_arch_fact_db(table_1_4_db)
    _insert_arch_table_1_4_facts(table_1_4_db)

    provider = resolve_arch_sqlite_target_provider(
        (str(table_1_1_db), str(table_1_4_db))
    )
    target_set = provider.load_target_set(TargetQuery(period=2023))

    assert isinstance(provider, ArchCompositeSQLiteTargetProvider)
    assert len(target_set.targets) == 18
    assert len({target.name for target in target_set.targets}) == 18
    assert {target.metadata["target_id"] for target in target_set.targets} == set(
        range(1, 19)
    )
    assert all(
        "arch_source_db_path" in target.metadata for target in target_set.targets
    )


def test_us_pipeline_arch_target_provider_accepts_source_fact_db(
    tmp_path: Path,
) -> None:
    fact_db = tmp_path / "arch_facts.db"
    _create_arch_fact_db(fact_db)
    pipeline = USMicroplexPipeline(
        USMicroplexBuildConfig(
            arch_targets_db=str(fact_db),
            calibration_target_source="arch",
        )
    )

    provider, source = pipeline._resolve_calibration_target_provider()

    assert source == "arch"
    assert isinstance(provider, ArchFactSQLiteTargetProvider)


def test_us_pipeline_arch_target_provider_accepts_consumer_fact_jsonl(
    tmp_path: Path,
) -> None:
    consumer_jsonl = tmp_path / "consumer_facts.jsonl"
    _write_consumer_fact_jsonl(consumer_jsonl)
    pipeline = USMicroplexPipeline(
        USMicroplexBuildConfig(
            arch_targets_db=str(consumer_jsonl),
            calibration_target_source="arch",
        )
    )

    provider, source = pipeline._resolve_calibration_target_provider()

    assert source == "arch"
    assert isinstance(provider, ArchConsumerFactJSONLTargetProvider)


def test_us_pipeline_arch_target_provider_accepts_multiple_source_fact_dbs(
    tmp_path: Path,
) -> None:
    table_1_1_db = tmp_path / "arch_table_1_1.db"
    table_1_4_db = tmp_path / "arch_table_1_4.db"
    _create_arch_fact_db(table_1_1_db)
    _create_arch_fact_db(table_1_4_db)
    _insert_arch_table_1_4_facts(table_1_4_db)
    pipeline = USMicroplexPipeline(
        USMicroplexBuildConfig(
            arch_targets_db=(str(table_1_1_db), str(table_1_4_db)),
            calibration_target_source="arch",
        )
    )

    provider, source = pipeline._resolve_calibration_target_provider()
    target_set = provider.load_target_set(TargetQuery(period=2023))

    assert source == "arch"
    assert isinstance(provider, ArchCompositeSQLiteTargetProvider)
    assert len(target_set.targets) == 18


def test_arch_fact_provider_maps_soi_table_1_4_income_source_facts(
    tmp_path: Path,
) -> None:
    fact_db = tmp_path / "arch_facts.db"
    _create_arch_fact_db(fact_db)
    _insert_arch_table_1_4_facts(fact_db)

    target_set = ArchFactSQLiteTargetProvider(fact_db).load_target_set(
        TargetQuery(period=2023)
    )
    table_1_4_targets = [
        target
        for target in target_set.targets
        if target.metadata["source_table"] == "Publication 1304 Table 1.4"
    ]

    arch_variables = {target.metadata["arch_variable"] for target in table_1_4_targets}
    assert arch_variables >= {
        "wages_salaries_returns",
        "wages_salaries_amount",
        "net_capital_gains_returns",
        "net_capital_gains_amount",
        "taxable_ira_distributions_returns",
        "taxable_ira_distributions_amount",
        "taxable_pension_income_returns",
        "taxable_pension_income_amount",
        "unemployment_compensation_returns",
        "unemployment_compensation_amount",
        "taxable_social_security_returns",
        "taxable_social_security_amount",
    }

    wages_amount = next(
        target
        for target in table_1_4_targets
        if target.metadata["arch_aggregate_fact_key"]
        == "arch.fact.v1:t14-all-wages-amount"
    )
    assert wages_amount.measure == "employment_income"
    assert getattr(wages_amount.aggregation, "value", wages_amount.aggregation) == "sum"
    assert getattr(wages_amount.entity, "value", wages_amount.entity) == "person"
    assert wages_amount.metadata["variable"] == "employment_income"
    assert wages_amount.metadata["arch_source_concept"] == "irs_soi.total_wages"
    assert wages_amount.metadata["arch_concept_relation"] == "broad_match"
    assert wages_amount.metadata["arch_source_cell_keys"] == [
        "arch.source_cell.v1:t14-wages-amount"
    ]

    wages_returns = next(
        target
        for target in table_1_4_targets
        if target.metadata["arch_aggregate_fact_key"]
        == "arch.fact.v1:t14-all-wages-returns"
    )
    assert wages_returns.measure is None
    assert (
        getattr(wages_returns.aggregation, "value", wages_returns.aggregation)
        == "count"
    )
    assert getattr(wages_returns.entity, "value", wages_returns.entity) == "tax_unit"
    assert wages_returns.metadata["variable"] == "employment_income"
    assert (
        "employment_income",
        ">",
        "0",
    ) in {
        (
            str(target_filter.feature),
            str(getattr(target_filter.operator, "value", target_filter.operator)),
            str(target_filter.value),
        )
        for target_filter in wages_returns.filters
    }

    capital_gains_amount = next(
        target
        for target in table_1_4_targets
        if target.metadata["arch_aggregate_fact_key"]
        == "arch.fact.v1:t14-all-capital-gains-amount"
    )
    assert (
        "net_capital_gains",
        ">",
        "0",
    ) in {
        (
            str(target_filter.feature),
            str(getattr(target_filter.operator, "value", target_filter.operator)),
            str(target_filter.value),
        )
        for target_filter in capital_gains_amount.filters
    }

    bracket_wages = next(
        target
        for target in table_1_4_targets
        if target.metadata["arch_aggregate_fact_key"]
        == "arch.fact.v1:t14-1-to-5k-wages-amount"
    )
    assert {
        (
            str(target_filter.feature),
            str(getattr(target_filter.operator, "value", target_filter.operator)),
            str(target_filter.value),
        )
        for target_filter in bracket_wages.filters
    } >= {
        ("adjusted_gross_income", ">=", "1"),
        ("adjusted_gross_income", "<", "5000"),
    }


def test_arch_fact_profile_coverage_accepts_soi_table_1_4_facts(
    tmp_path: Path,
) -> None:
    fact_db = tmp_path / "arch_facts.db"
    _create_arch_fact_db(fact_db)
    _insert_arch_table_1_4_facts(fact_db)
    provider = ArchFactSQLiteTargetProvider(fact_db)

    report = summarize_arch_target_profile_coverage(
        provider,
        period=2023,
        profile_name="custom",
        target_cells=(
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
            {
                "variable": "taxable_social_security",
                "geo_level": "national",
                "domain_variable": "taxable_social_security",
            },
            {
                "variable": "tax_unit_count",
                "geo_level": "national",
                "domain_variable": "taxable_social_security",
            },
        ),
    )

    assert report.target_cell_count == 4
    assert report.covered_cell_count == 4
    assert report.coverage_rate == 1


def test_arch_composite_profile_coverage_combines_table_1_1_and_1_4(
    tmp_path: Path,
) -> None:
    table_1_1_db = tmp_path / "arch_table_1_1.db"
    table_1_4_db = tmp_path / "arch_table_1_4.db"
    _create_arch_fact_db(table_1_1_db)
    _create_arch_fact_db(table_1_4_db)
    _insert_arch_table_1_4_facts(table_1_4_db)
    provider = resolve_arch_sqlite_target_provider((table_1_1_db, table_1_4_db))

    report = summarize_arch_target_profile_coverage(
        provider,
        period=2023,
        profile_name="custom",
        target_cells=(
            {
                "variable": "adjusted_gross_income",
                "geo_level": "national",
                "domain_variable": None,
            },
            {
                "variable": "income_tax",
                "geo_level": "national",
                "domain_variable": None,
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
        ),
    )

    assert report.target_cell_count == 4
    assert report.covered_cell_count == 4
    assert report.coverage_rate == 1


def test_arch_fact_gap_queue_uses_source_fact_loaded_catalog(
    tmp_path: Path,
) -> None:
    fact_db = tmp_path / "arch_facts.db"
    _create_arch_fact_db(fact_db)
    _insert_arch_table_1_4_facts(fact_db)
    provider = ArchFactSQLiteTargetProvider(fact_db)

    report = summarize_arch_target_gap_queue(
        provider,
        period=2023,
        profile_name="custom",
        target_cells=(
            {
                "variable": "employment_income",
                "geo_level": "state",
                "domain_variable": "employment_income",
            },
        ),
    )

    assert report.row_count == 1
    assert report.rows[0].expected_arch_variable == "wages_salaries_amount"
    assert report.rows[0].loader_status == "loaded_arch_variable_missing_geography"


def test_arch_fact_gap_queue_expected_filters_normalize_geography_ids(
    tmp_path: Path,
) -> None:
    fact_db = tmp_path / "arch_facts.db"
    _create_arch_fact_db(fact_db)
    provider = ArchFactSQLiteTargetProvider(fact_db)

    report = summarize_arch_target_gap_queue(
        provider,
        period=2023,
        profile_name="custom",
        target_cells=(
            {
                "variable": "person_count",
                "geo_level": "state",
                "geographic_id": "06",
                "domain_variable": None,
            },
            {
                "variable": "person_count",
                "geo_level": "sldu",
                "geographic_id": "06001",
                "domain_variable": None,
            },
            {
                "variable": "household_count",
                "geo_level": "sldl",
                "geographic_id": "36065",
                "domain_variable": None,
            },
        ),
    )

    filters_by_level = {
        row.geo_level: {
            item["feature"]: item["value"]
            for item in row.expected_filters
            if item["kind"] == "geography"
        }
        for row in report.rows
    }

    assert filters_by_level == {
        "state": {"state_fips": "06"},
        "sldu": {"sldu_id": "CA-SLDU-001"},
        "sldl": {"sldl_id": "NY-SLDL-065"},
    }
