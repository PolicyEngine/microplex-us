"""Tests for the CPS Social Security reason-code split, focused on the
``social_security_retirement`` leaf that drives the ``national_ssa`` loss family.

``social_security_retirement`` is constructed in ``_process_persons`` by
splitting the bundled CPS ``SS_VAL`` (mapped to ``social_security``) across the
four benefit reasons using the ASEC ``RESNSS1``/``RESNSS2`` reason codes, with an
age-62 fallback for records whose reason is unclassified. This mirrors eCPS
``policyengine_us_data/datasets/cps/cps.py`` (the SS reason-code split + age-62
fallback). The leaf is produced today but, before this change, was missing from
the policyengine-us export allowlist, so it never reached the exported H5.

These tests exercise the real split (they do NOT stub ``_process_persons``) and
assert reconciliation, dominance, fallback, the both-reasons-present drop
edge case, and export-allowlist
properties. They run on tiny synthetic frames, so no weighting is involved;
national-total accuracy against SSA/IRS targets is validated downstream by the
eCPS comparison harness, not here.
"""

import polars as pl

from microplex_us.data_sources.cps import (
    MINIMUM_RETIREMENT_AGE,
    SOCIAL_SECURITY_DEPENDENT_REASON_CODES,
    SOCIAL_SECURITY_DISABILITY_REASON_CODE,
    SOCIAL_SECURITY_RETIREMENT_REASON_CODE,
    SOCIAL_SECURITY_SURVIVOR_REASON_CODES,
    _process_persons,
)

# The four leaves the SS_VAL reason-code split produces.
_SS_COMPONENTS = (
    "social_security_retirement",
    "social_security_disability",
    "social_security_survivors",
    "social_security_dependents",
)


def _raw_person_frame(rows: list[dict]) -> pl.DataFrame:
    """Build a raw CPS-style person frame with the columns the split consumes.

    Census column names are used because ``_process_persons`` selects/renames via
    ``PERSON_VARIABLES`` before running the split.
    """
    n = len(rows)
    return pl.DataFrame(
        {
            "PH_SEQ": [1] * n,
            "A_LINENO": list(range(1, n + 1)),
            "A_FNLWGT": [100.0] * n,
            "A_AGE": [row["age"] for row in rows],
            "SS_VAL": [row["ss"] for row in rows],
            "RESNSS1": [row.get("r1", 0) for row in rows],
            "RESNSS2": [row.get("r2", 0) for row in rows],
        }
    )


def test_reason_codes_match_ecps_constants():
    """The reason-code constants mirror eCPS cps.py classification."""
    assert SOCIAL_SECURITY_RETIREMENT_REASON_CODE == 1
    assert SOCIAL_SECURITY_DISABILITY_REASON_CODE == 2
    assert SOCIAL_SECURITY_SURVIVOR_REASON_CODES == (3, 5)
    assert SOCIAL_SECURITY_DEPENDENT_REASON_CODES == (4, 6, 7)
    assert MINIMUM_RETIREMENT_AGE == 62


def test_split_components_sum_to_total_social_security():
    """The four reason-coded components reconstruct the bundled SS_VAL total.

    Every recipient carries a classified reason (or an age-based fallback), so
    summing the four components must equal ``social_security`` person-by-person.
    """
    rows = [
        {"age": 70, "ss": 20_000.0, "r1": 1},  # retirement
        {"age": 50, "ss": 15_000.0, "r1": 2},  # disability
        {"age": 40, "ss": 12_000.0, "r1": 3},  # survivor
        {"age": 10, "ss": 8_000.0, "r1": 4},  # dependent
        {"age": 67, "ss": 18_000.0, "r1": 0},  # unclassified -> age>=62 retirement
        {"age": 45, "ss": 9_000.0, "r1": 0},  # unclassified -> age<62 disability
        {"age": 30, "ss": 0.0, "r1": 0},  # non-recipient
    ]
    result = _process_persons(_raw_person_frame(rows), 2023)

    for component in _SS_COMPONENTS:
        assert component in result.columns, f"{component} not produced"

    component_sum = result.select(
        sum(pl.col(component) for component in _SS_COMPONENTS).alias("total")
    )["total"]
    total_ss = result["social_security"]
    for got, expected in zip(component_sum.to_list(), total_ss.to_list()):
        assert abs(got - expected) < 1e-6


def test_retirement_is_the_dominant_component():
    """On a retiree-heavy aged population, retirement dominates the SS split.

    SSA program data: OASI retirement benefits are by far the largest Social
    Security component, so a population skewed to ages 62+ must produce a
    retirement total larger than each of disability/survivors/dependents.
    """
    rows = [{"age": age, "ss": 20_000.0, "r1": 1} for age in (66, 68, 70, 72, 75)]
    rows += [{"age": age, "ss": 18_000.0, "r1": 0} for age in (63, 67, 71)]
    rows += [
        {"age": 50, "ss": 14_000.0, "r1": 2},  # disability
        {"age": 35, "ss": 10_000.0, "r1": 3},  # survivor
    ]
    result = _process_persons(_raw_person_frame(rows), 2023)

    totals = {
        component: float(result[component].sum()) for component in _SS_COMPONENTS
    }
    assert totals["social_security_retirement"] > totals["social_security_disability"]
    assert totals["social_security_retirement"] > totals["social_security_survivors"]
    assert totals["social_security_retirement"] > totals["social_security_dependents"]


def test_retirement_values_are_non_degenerate():
    """Retirement is neither all-zero nor a single constant across recipients."""
    rows = [
        {"age": 66, "ss": 12_000.0, "r1": 1},
        {"age": 70, "ss": 24_000.0, "r1": 1},
        {"age": 64, "ss": 18_000.0, "r1": 0},  # age>=62 fallback -> retirement
        {"age": 50, "ss": 15_000.0, "r1": 2},  # disability, NOT retirement
    ]
    result = _process_persons(_raw_person_frame(rows), 2023)

    retirement = result["social_security_retirement"]
    positive = [value for value in retirement.to_list() if value > 0]
    assert len(positive) >= 2, "retirement should be positive for several records"
    assert len(set(positive)) >= 2, "retirement values should not be a single constant"
    assert float(retirement.sum()) > 0.0


def test_age_62_fallback_routes_unclassified_by_age():
    """Unclassified recipients route to retirement iff age >= 62, else disability."""
    rows = [
        {"age": 62, "ss": 10_000.0, "r1": 0},  # exactly 62 -> retirement
        {"age": 61, "ss": 10_000.0, "r1": 0},  # 61 -> disability
    ]
    result = _process_persons(_raw_person_frame(rows), 2023)

    retirement = result["social_security_retirement"].to_list()
    disability = result["social_security_disability"].to_list()
    assert retirement[0] == 10_000.0
    assert disability[0] == 0.0
    assert retirement[1] == 0.0
    assert disability[1] == 10_000.0


def test_explicit_reason_code_overrides_age_fallback():
    """A classified disability reason stays disability even at retirement age."""
    rows = [{"age": 70, "ss": 16_000.0, "r1": 2}]  # disability code at age 70
    result = _process_persons(_raw_person_frame(rows), 2023)

    assert result["social_security_disability"].to_list()[0] == 16_000.0
    assert result["social_security_retirement"].to_list()[0] == 0.0


def test_both_retirement_and_disability_reasons_is_a_drop_edge_case():
    """When BOTH retirement and disability reasons are present, the value drops.

    The split gates retirement on ``has_retirement & ~has_disability`` and
    disability on ``has_disability & ~has_retirement`` (and the record is not
    "unclassified" because it carries reasons), so a record coded for BOTH lands
    in none of the four components: every component is 0 even though
    ``social_security`` is positive. This is a rare degenerate ASEC coding
    (simultaneous retirement+disability reason) and is documented here as the
    leaf's deterministic behavior rather than silently assumed. It is the one
    case where the four components do not reconstruct the total. Do not "fix" by
    giving retirement priority without first confirming the matching eCPS
    version's behavior.
    """
    rows = [{"age": 68, "ss": 22_000.0, "r1": 1, "r2": 2}]
    result = _process_persons(_raw_person_frame(rows), 2023)

    assert result["social_security_retirement"].to_list()[0] == 0.0
    assert result["social_security_disability"].to_list()[0] == 0.0
    assert result["social_security_survivors"].to_list()[0] == 0.0
    assert result["social_security_dependents"].to_list()[0] == 0.0
    # The total is still the bundled SS_VAL; only the component split drops it.
    assert result["social_security"].to_list()[0] == 22_000.0


def test_social_security_retirement_in_export_allowlist():
    """The leaf must be in the export allowlist or it never reaches the H5.

    The alias map must NOT remap it to a different (reported) companion leaf.
    """
    from microplex_us.policyengine.us import (
        POLICYENGINE_US_EXPORT_COLUMN_ALIASES,
        SAFE_POLICYENGINE_US_EXPORT_VARIABLES,
    )

    assert "social_security_retirement" in SAFE_POLICYENGINE_US_EXPORT_VARIABLES
    assert "social_security_disability" in SAFE_POLICYENGINE_US_EXPORT_VARIABLES
    assert "social_security_survivors" in SAFE_POLICYENGINE_US_EXPORT_VARIABLES
    assert "social_security_dependents" in SAFE_POLICYENGINE_US_EXPORT_VARIABLES
    assert (
        POLICYENGINE_US_EXPORT_COLUMN_ALIASES.get("social_security_retirement")
        is None
    )


def test_social_security_retirement_survives_computed_export_guard():
    """A future pe-us that re-adds a formula must not silently drop the leaf.

    The export path filters out PolicyEngine-computed variables unless they are
    in POLICYENGINE_US_ALLOWED_COMPUTED_EXPORT_VARIABLES. Allowlisting the leaf
    alone is not enough: if a pe-us version re-introduces the historical
    fallback formula, the computed-export guard would strip it before the
    hard-raise validation runs, leaving the column silently missing. Pin the
    insurance: the leaf is in the override set, so the guard always keeps it.
    """
    from microplex_us.policyengine.us import (
        POLICYENGINE_US_ALLOWED_COMPUTED_EXPORT_VARIABLES,
        POLICYENGINE_US_DATA_OVERRIDABLE_COMPUTED_EXPORT_VARIABLES,
        detect_policyengine_computed_export_variables,
    )

    assert (
        "social_security_retirement"
        in POLICYENGINE_US_DATA_OVERRIDABLE_COMPUTED_EXPORT_VARIABLES
    )
    assert (
        "social_security_retirement"
        in POLICYENGINE_US_ALLOWED_COMPUTED_EXPORT_VARIABLES
    )

    # Simulate a pe-us where the leaf has a formula: the computed-export
    # detector must NOT mark it for exclusion, because it is allow-listed.
    class _FormulaVar:
        formulas = {"2024": object()}
        adds = None
        subtracts = None

    class _SystemWithFormula:
        variables = {"social_security_retirement": _FormulaVar()}

    excluded = detect_policyengine_computed_export_variables(
        _SystemWithFormula(), ["social_security_retirement"]
    )
    assert "social_security_retirement" not in excluded


# Standalone runner so the suite executes without pytest installed in the env.
if __name__ == "__main__":
    import traceback

    funcs = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    passed = 0
    failed = 0
    for fn in funcs:
        try:
            fn()
            print(f"PASS {fn.__name__}")
            passed += 1
        except Exception:  # noqa: BLE001
            print(f"FAIL {fn.__name__}")
            traceback.print_exc()
            failed += 1
    print(f"SUMMARY passed={passed} failed={failed}")
    raise SystemExit(1 if failed else 0)
