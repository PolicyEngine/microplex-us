from __future__ import annotations

from microplex_us.policyengine.target_profiles import (
    policyengine_us_target_profile_exclusion_reasons,
    policyengine_us_target_profile_names,
    resolve_policyengine_us_target_profile,
)


def test_policyengine_us_target_profile_names_include_no_state_aca_variant() -> None:
    assert "pe_native_broad" in policyengine_us_target_profile_names()
    assert "pe_native_broad_no_state_aca" in policyengine_us_target_profile_names()
    assert "pe_native_broad_source_backed" in policyengine_us_target_profile_names()


def test_broad_profile_includes_soi_employment_income_cells() -> None:
    broad = resolve_policyengine_us_target_profile("pe_native_broad")
    broad_cells = {
        (cell.variable, cell.geo_level, cell.domain_variable, cell.geographic_id)
        for cell in broad
    }

    assert (
        "employment_income",
        "national",
        "employment_income",
        None,
    ) in broad_cells
    assert (
        "tax_unit_count",
        "national",
        "employment_income",
        None,
    ) in broad_cells
    assert (
        "employment_income",
        "state",
        "employment_income",
        None,
    ) in broad_cells
    assert (
        "tax_unit_count",
        "state",
        "employment_income",
        None,
    ) in broad_cells


def test_broad_profile_includes_bea_full_population_amount_cells() -> None:
    broad = resolve_policyengine_us_target_profile("pe_native_broad")
    broad_cells = {
        (cell.variable, cell.geo_level, cell.domain_variable, cell.geographic_id)
        for cell in broad
    }

    assert (
        "dividend_income",
        "national",
        None,
        None,
    ) in broad_cells
    assert (
        "employment_income",
        "national",
        None,
        None,
    ) in broad_cells
    assert (
        "rental_income",
        "national",
        None,
        None,
    ) in broad_cells
    assert (
        "self_employment_income",
        "national",
        None,
        None,
    ) in broad_cells
    assert (
        "employment_income",
        "state",
        None,
        None,
    ) in broad_cells
    assert (
        "self_employment_income",
        "state",
        None,
        None,
    ) in broad_cells


def test_broad_profile_covers_current_policyengine_target_db_cells() -> None:
    broad = resolve_policyengine_us_target_profile("pe_native_broad")
    broad_cells = {
        (cell.variable, cell.geo_level, cell.domain_variable, cell.geographic_id)
        for cell in broad
    }

    added_policyengine_cells = {
        ("aca_ptc", "national", "aca_ptc", None),
        ("adjusted_gross_income", "national", "adjusted_gross_income", None),
        (
            "adjusted_gross_income",
            "national",
            "adjusted_gross_income,filing_status,income_tax_before_credits",
            None,
        ),
        (
            "adjusted_gross_income",
            "national",
            "adjusted_gross_income,income_tax_before_credits",
            None,
        ),
        ("childcare_expenses", "national", None, None),
        ("deductible_mortgage_interest", "national", None, None),
        ("household_count", "national", "spm_unit_energy_subsidy_reported", None),
        (
            "medical_expense_deduction",
            "national",
            "medical_expense_deduction,tax_unit_itemizes",
            None,
        ),
        (
            "non_refundable_ctc",
            "national",
            "adjusted_gross_income,non_refundable_ctc",
            None,
        ),
        ("non_refundable_ctc", "national", "non_refundable_ctc", None),
        (
            "real_estate_taxes",
            "national",
            "real_estate_taxes,tax_unit_itemizes",
            None,
        ),
        (
            "refundable_ctc",
            "national",
            "adjusted_gross_income,refundable_ctc",
            None,
        ),
        ("roth_401k_contributions", "national", None, None),
        ("salt", "national", "salt,tax_unit_itemizes", None),
        ("self_employed_pension_contribution_ald", "national", None, None),
        ("spm_unit_count", "national", "tanf", None),
        ("tanf", "national", "tanf", None),
        ("tax_unit_count", "national", "adjusted_gross_income", None),
        (
            "tax_unit_count",
            "national",
            "adjusted_gross_income,filing_status,income_tax_before_credits",
            None,
        ),
        (
            "tax_unit_count",
            "national",
            "adjusted_gross_income,income_tax_before_credits",
            None,
        ),
        (
            "tax_unit_count",
            "national",
            "adjusted_gross_income,non_refundable_ctc",
            None,
        ),
        (
            "tax_unit_count",
            "national",
            "adjusted_gross_income,refundable_ctc",
            None,
        ),
        (
            "tax_unit_count",
            "national",
            "medical_expense_deduction,tax_unit_itemizes",
            None,
        ),
        ("tax_unit_count", "national", "non_refundable_ctc", None),
        (
            "tax_unit_count",
            "national",
            "real_estate_taxes,tax_unit_itemizes",
            None,
        ),
        ("tax_unit_count", "national", "salt,tax_unit_itemizes", None),
        ("tax_unit_count", "national", "total_self_employment_income", None),
        (
            "total_self_employment_income",
            "national",
            "total_self_employment_income",
            None,
        ),
        ("traditional_401k_contributions", "national", None, None),
        ("aca_ptc", "state", "aca_ptc", None),
        ("adjusted_gross_income", "state", "adjusted_gross_income", None),
        (
            "medical_expense_deduction",
            "state",
            "medical_expense_deduction,tax_unit_itemizes",
            None,
        ),
        ("non_refundable_ctc", "state", "non_refundable_ctc", None),
        ("person_count", "state", "aca_ptc,is_aca_ptc_eligible", None),
        ("person_count", "state", "is_pregnant", None),
        (
            "real_estate_taxes",
            "state",
            "real_estate_taxes,tax_unit_itemizes",
            None,
        ),
        ("salt", "state", "salt,tax_unit_itemizes", None),
        ("spm_unit_count", "state", "tanf", None),
        ("tanf", "state", "tanf", None),
        (
            "tax_unit_count",
            "state",
            "medical_expense_deduction,tax_unit_itemizes",
            None,
        ),
        ("tax_unit_count", "state", "non_refundable_ctc", None),
        (
            "tax_unit_count",
            "state",
            "real_estate_taxes,tax_unit_itemizes",
            None,
        ),
        ("tax_unit_count", "state", "salt,tax_unit_itemizes", None),
        (
            "tax_unit_count",
            "state",
            "selected_marketplace_plan_benchmark_ratio,used_aca_ptc",
            None,
        ),
        ("tax_unit_count", "state", "total_self_employment_income", None),
        ("tax_unit_count", "state", "used_aca_ptc", None),
        (
            "total_self_employment_income",
            "state",
            "total_self_employment_income",
            None,
        ),
    }

    assert added_policyengine_cells <= broad_cells


def test_broad_profile_has_no_duplicate_cells() -> None:
    broad = resolve_policyengine_us_target_profile("pe_native_broad")
    broad_cells = [
        (cell.variable, cell.geo_level, cell.domain_variable, cell.geographic_id)
        for cell in broad
    ]

    assert len(broad_cells) == len(set(broad_cells))


def test_no_state_aca_profile_excludes_only_state_aca_cells() -> None:
    broad = resolve_policyengine_us_target_profile("pe_native_broad")
    no_state_aca = resolve_policyengine_us_target_profile(
        "pe_native_broad_no_state_aca"
    )

    broad_cells = {
        (cell.variable, cell.geo_level, cell.domain_variable, cell.geographic_id)
        for cell in broad
    }
    no_state_aca_cells = {
        (cell.variable, cell.geo_level, cell.domain_variable, cell.geographic_id)
        for cell in no_state_aca
    }

    assert (
        "aca_ptc",
        "state",
        None,
        None,
    ) in broad_cells
    assert (
        "tax_unit_count",
        "state",
        "aca_ptc",
        None,
    ) in broad_cells
    assert (
        "aca_ptc",
        "state",
        "aca_ptc",
        None,
    ) in broad_cells
    assert (
        "person_count",
        "state",
        "aca_ptc,is_aca_ptc_eligible",
        None,
    ) in broad_cells
    assert (
        "tax_unit_count",
        "state",
        "used_aca_ptc",
        None,
    ) in broad_cells
    assert (
        "tax_unit_count",
        "national",
        "aca_ptc",
        None,
    ) in no_state_aca_cells
    assert (
        "aca_ptc",
        "state",
        None,
        None,
    ) not in no_state_aca_cells
    assert (
        "tax_unit_count",
        "state",
        "aca_ptc",
        None,
    ) not in no_state_aca_cells
    assert (
        "aca_ptc",
        "state",
        "aca_ptc",
        None,
    ) not in no_state_aca_cells
    assert (
        "person_count",
        "state",
        "aca_ptc",
        None,
    ) not in no_state_aca_cells
    assert (
        "person_count",
        "state",
        "aca_ptc,is_aca_ptc_eligible",
        None,
    ) not in no_state_aca_cells
    assert (
        "tax_unit_count",
        "state",
        "selected_marketplace_plan_benchmark_ratio,used_aca_ptc",
        None,
    ) not in no_state_aca_cells
    assert (
        "tax_unit_count",
        "state",
        "used_aca_ptc",
        None,
    ) not in no_state_aca_cells


def test_source_backed_profile_excludes_only_documented_non_source_cells() -> None:
    broad = resolve_policyengine_us_target_profile("pe_native_broad")
    source_backed = resolve_policyengine_us_target_profile(
        "pe_native_broad_source_backed"
    )
    exclusion_reasons = policyengine_us_target_profile_exclusion_reasons(
        "pe_native_broad_source_backed"
    )

    broad_cells = {
        (cell.variable, cell.geo_level, cell.domain_variable, cell.geographic_id)
        for cell in broad
    }
    source_backed_cells = {
        (cell.variable, cell.geo_level, cell.domain_variable, cell.geographic_id)
        for cell in source_backed
    }

    assert len(broad_cells) == 189
    assert len(exclusion_reasons) == 15
    assert all(reason for reason in exclusion_reasons.values())
    assert set(exclusion_reasons) <= broad_cells
    assert len(source_backed_cells) == 174
    assert source_backed_cells == broad_cells - set(exclusion_reasons)
    assert (
        "childcare_expenses",
        "national",
        None,
        None,
    ) not in source_backed_cells
    assert (
        "person_count",
        "state",
        "is_pregnant",
        None,
    ) not in source_backed_cells
    assert (
        "employment_income",
        "national",
        None,
        None,
    ) in source_backed_cells
    assert (
        "medicare_part_b_premiums",
        "national",
        None,
        None,
    ) in source_backed_cells
    assert (
        "net_worth",
        "national",
        None,
        None,
    ) in source_backed_cells
