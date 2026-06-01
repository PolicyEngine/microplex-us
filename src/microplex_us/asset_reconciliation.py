"""SCF net-worth component reconciliation.

This module ports the self-contained numpy net-worth reconciliation logic
from policyengine-us-data's enhanced-CPS (eCPS) asset imputation so that the
microplex-us SCF donor block can reconcile its 19 imputed balance-sheet
component leaves to a direct SCF ``net_worth`` anchor, exactly as eCPS does.

Mirrored verbatim (signs, fallbacks, scaling) from
``policyengine_us_data/utils/asset_imputation.py`` on
``PolicyEngine/policyengine-us-data`` ``upstream/main``:

    * ``SCF_NET_WORTH_COMPONENT_TARGETS``                 -> L40-66
    * ``SCF_NET_WORTH_COMPONENT_VARIABLES``               -> L57
    * ``SCF_OTHER_ASSET_COMPONENT`` / ``..._DEBT_COMPONENT`` -> L58-59
    * ``NET_WORTH_COMPONENT_SIGNS``                       -> L77-86
    * ``compute_net_worth_from_components``               -> L373-398 (def)
    * ``rebalance_scf_net_worth_components``              -> L401-490 (def)

Storage convention (matches eCPS): every component leaf is stored as a
**positive magnitude** -- including the debt leaves. The ``-1`` debt sign in
``NET_WORTH_COMPONENT_SIGNS`` is applied only when reconstructing net worth
(``compute_net_worth_from_components``) and when rebalancing
(``rebalance_scf_net_worth_components``), never to the stored leaf values. The
SCF raw debt columns (mrthel, ccbal, edn_inst, ...) are themselves stored as
non-negative balances, so no sign flip happens at load time either.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np

# Leaf -> raw SCF summary-extract source column(s). Single source column per
# leaf; mirrors SCF_NET_WORTH_COMPONENT_TARGETS in eCPS asset_imputation.py
# L40-66. Kept here as documentation / a single source of truth for tests.
SCF_NET_WORTH_COMPONENT_TARGETS: dict[str, tuple[str, ...]] = {
    "scf_certificates_of_deposit": ("cds",),
    "scf_savings_bonds": ("savbnd",),
    "scf_retirement_assets": ("retqliq",),
    "scf_cash_value_life_insurance": ("cashli",),
    "scf_other_managed_assets": ("othma",),
    "scf_other_financial_assets": ("othfin",),
    "scf_primary_residence_value": ("houses",),
    "scf_other_residential_real_estate": ("oresre",),
    "scf_nonresidential_real_estate_equity": ("nnresre",),
    "scf_business_equity": ("bus",),
    "scf_other_nonfinancial_assets": ("othnfin",),
    "scf_mortgage_debt": ("mrthel",),
    "scf_other_residential_debt": ("resdbt",),
    "scf_other_lines_of_credit": ("othloc",),
    "scf_credit_card_debt": ("ccbal",),
    "scf_vehicle_installment_debt": ("veh_inst",),
    "scf_student_loan_debt": ("edn_inst",),
    "scf_other_installment_debt": ("oth_inst",),
    "scf_other_debt": ("odebt",),
}
SCF_NET_WORTH_COMPONENT_VARIABLES: tuple[str, ...] = tuple(
    SCF_NET_WORTH_COMPONENT_TARGETS
)

# Asset / debt fallback sinks used when an entire side of the balance sheet is
# unexpectedly empty for a household. Mirrors eCPS L58-59.
SCF_OTHER_ASSET_COMPONENT = "scf_other_financial_assets"
SCF_OTHER_DEBT_COMPONENT = "scf_other_debt"

# Signed component map. Debt leaves enter net worth with -1; every asset leaf is
# +1 by default. Mirrors eCPS NET_WORTH_COMPONENT_SIGNS L77-86. ``auto_loan_balance``
# is included because it is a SIPP/SCF-blended debt leaf that also subtracts.
NET_WORTH_COMPONENT_SIGNS: dict[str, float] = {
    "auto_loan_balance": -1.0,
    "scf_mortgage_debt": -1.0,
    "scf_other_residential_debt": -1.0,
    "scf_other_lines_of_credit": -1.0,
    "scf_credit_card_debt": -1.0,
    "scf_vehicle_installment_debt": -1.0,
    "scf_student_loan_debt": -1.0,
    "scf_other_installment_debt": -1.0,
    "scf_other_debt": -1.0,
}

# SIPP/SCF-blended policy leaves that the rebalance must preserve (it only
# rescales the SCF-only components). Mirrors eCPS protected_variables default
# (SIPP_LIQUID_ASSET_VARIABLES + SIPP_VEHICLE_ASSET_VARIABLES).
PROTECTED_BLENDED_COMPONENT_VARIABLES: tuple[str, ...] = (
    "bank_account_assets",
    "stock_assets",
    "bond_assets",
    "household_vehicles_value",
)


def compute_net_worth_from_components(
    *,
    components: Mapping[str, Sequence[float]],
    component_signs: Mapping[str, float] = NET_WORTH_COMPONENT_SIGNS,
) -> np.ndarray:
    """Compute household net worth from signed balance-sheet components.

    Verbatim port of eCPS ``compute_net_worth_from_components``
    (asset_imputation.py L373-398).
    """
    iterator = iter(components.items())
    try:
        first_variable, first_values = next(iterator)
    except StopIteration:
        return np.array([], dtype=np.float32)

    first_values = np.asarray(first_values, dtype=np.float32)
    component_total = (component_signs.get(first_variable, 1.0) * first_values).astype(
        np.float32
    )

    for variable, values in iterator:
        values = np.asarray(values, dtype=np.float32)
        if values.shape != component_total.shape:
            raise ValueError(
                f"{variable} has shape {values.shape}, but expected "
                f"{component_total.shape}."
            )
        component_total += component_signs.get(variable, 1.0) * values

    return component_total.astype(np.float32)


def rebalance_scf_net_worth_components(
    *,
    components: Mapping[str, Sequence[float]],
    target_net_worth: Sequence[float],
    adjustable_variables: Sequence[str] = SCF_NET_WORTH_COMPONENT_VARIABLES,
    protected_variables: Sequence[str] = PROTECTED_BLENDED_COMPONENT_VARIABLES,
    component_signs: Mapping[str, float] = NET_WORTH_COMPONENT_SIGNS,
) -> dict[str, np.ndarray]:
    """Rebalance SCF-only leaves so the component formula matches net worth.

    Component QRFs are fit sequentially but still predict each leaf separately,
    so their sum can drift from the direct SCF net worth distribution. Preserve
    the final SIPP/SCF-blended policy leaves and proportionally scale SCF-only
    same-sign leaves to the direct SCF net worth anchor.

    Verbatim port of eCPS ``rebalance_scf_net_worth_components``
    (asset_imputation.py L401-490).
    """
    adjusted = {
        variable: np.asarray(values, dtype=np.float32).copy()
        for variable, values in components.items()
    }
    if not adjusted:
        return adjusted

    target_net_worth = np.asarray(target_net_worth, dtype=np.float32)
    first_shape = target_net_worth.shape
    for variable, values in adjusted.items():
        if values.shape != first_shape:
            raise ValueError(
                f"{variable} has shape {values.shape}, but target_net_worth "
                f"has shape {first_shape}."
            )

    protected_variables = set(protected_variables)
    adjustable_variables = tuple(
        variable
        for variable in adjustable_variables
        if variable in adjusted and variable not in protected_variables
    )
    if not adjustable_variables:
        return adjusted

    fixed_total = np.zeros_like(target_net_worth, dtype=np.float32)
    for variable, values in adjusted.items():
        if variable not in adjustable_variables:
            fixed_total += component_signs.get(variable, 1.0) * values

    asset_variables = [
        variable
        for variable in adjustable_variables
        if component_signs.get(variable, 1.0) >= 0
    ]
    debt_variables = [
        variable
        for variable in adjustable_variables
        if component_signs.get(variable, 1.0) < 0
    ]

    asset_total = np.zeros_like(target_net_worth, dtype=np.float32)
    for variable in asset_variables:
        asset_total += adjusted[variable]

    debt_total = np.zeros_like(target_net_worth, dtype=np.float32)
    for variable in debt_variables:
        debt_total += adjusted[variable]

    desired_adjustable_total = target_net_worth - fixed_total
    positive_target = desired_adjustable_total >= 0

    required_assets = np.maximum(desired_adjustable_total + debt_total, 0)
    asset_scale = np.divide(
        required_assets,
        asset_total,
        out=np.ones_like(required_assets, dtype=np.float32),
        where=(asset_total > 0) & positive_target,
    )
    for variable in asset_variables:
        adjusted[variable][positive_target] *= asset_scale[positive_target]

    needs_asset_fallback = positive_target & (asset_total <= 0) & (required_assets > 0)
    if needs_asset_fallback.any() and SCF_OTHER_ASSET_COMPONENT in adjusted:
        adjusted[SCF_OTHER_ASSET_COMPONENT][needs_asset_fallback] = required_assets[
            needs_asset_fallback
        ]

    required_debts = np.maximum(asset_total - desired_adjustable_total, 0)
    debt_scale = np.divide(
        required_debts,
        debt_total,
        out=np.ones_like(required_debts, dtype=np.float32),
        where=(debt_total > 0) & ~positive_target,
    )
    for variable in debt_variables:
        adjusted[variable][~positive_target] *= debt_scale[~positive_target]

    needs_debt_fallback = (~positive_target) & (debt_total <= 0) & (required_debts > 0)
    if needs_debt_fallback.any() and SCF_OTHER_DEBT_COMPONENT in adjusted:
        adjusted[SCF_OTHER_DEBT_COMPONENT][needs_debt_fallback] = required_debts[
            needs_debt_fallback
        ]

    return adjusted
