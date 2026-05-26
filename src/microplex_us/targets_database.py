"""
Legacy US calibration targets database models.

This module is US-specific and remains separate from the canonical
`microplex.targets` abstractions used for cross-country target specs.
"""

from dataclasses import dataclass, field
from enum import Enum

import numpy as np
import pandas as pd


class TargetCategory(Enum):
    """Categories of legacy US calibration targets."""

    AGI_DISTRIBUTION = "agi_distribution"
    INCOME_SOURCES = "income_sources"
    DEDUCTIONS = "deductions"
    TAX_LIABILITY = "tax_liability"
    EITC = "eitc"
    CTC = "ctc"
    ACTC = "actc"
    OTHER_CREDITS = "other_credits"
    SNAP = "snap"
    MEDICAID = "medicaid"
    HOUSING = "housing"
    SSI = "ssi"
    TANF = "tanf"
    UNEMPLOYMENT = "unemployment"
    POPULATION = "population"
    HOUSEHOLD_STRUCTURE = "household_structure"
    AGE_DISTRIBUTION = "age_distribution"
    EMPLOYMENT = "employment"


@dataclass
class Target:
    """A legacy US calibration target."""

    name: str
    category: TargetCategory
    value: float
    year: int
    source: str
    source_url: str | None = None
    geography: str = "US"
    state_fips: str | None = None
    filing_status: str | None = None
    agi_lower: float = -np.inf
    agi_upper: float = np.inf
    is_count: bool = True
    is_taxable_only: bool = False
    rac_variable: str | None = None
    rac_statute: str | None = None
    microdata_column: str | None = None
    notes: str | None = None
    last_updated: str | None = None


@dataclass
class TargetsDatabase:
    """Database of legacy US calibration targets."""

    targets: list[Target] = field(default_factory=list)
    _by_category: dict[TargetCategory, list[Target]] = field(default_factory=dict)
    _by_geography: dict[str, list[Target]] = field(default_factory=dict)

    def add(self, target: Target):
        self.targets.append(target)
        if target.category not in self._by_category:
            self._by_category[target.category] = []
        self._by_category[target.category].append(target)
        if target.geography not in self._by_geography:
            self._by_geography[target.geography] = []
        self._by_geography[target.geography].append(target)

    def add_many(self, targets: list[Target]):
        for target in targets:
            self.add(target)

    def get_by_category(self, category: TargetCategory) -> list[Target]:
        return self._by_category.get(category, [])

    def get_by_geography(self, geography: str) -> list[Target]:
        return self._by_geography.get(geography, [])

    def get_national(self) -> list[Target]:
        return self.get_by_geography("US")

    def get_state(self, state_fips: str) -> list[Target]:
        return [target for target in self.targets if target.state_fips == state_fips]

    def get_with_rac_mapping(self) -> list[Target]:
        return [target for target in self.targets if target.rac_variable is not None]

    def to_dataframe(self) -> pd.DataFrame:
        rows = []
        for target in self.targets:
            rows.append(
                {
                    "name": target.name,
                    "category": target.category.value,
                    "value": target.value,
                    "year": target.year,
                    "source": target.source,
                    "geography": target.geography,
                    "state_fips": target.state_fips,
                    "filing_status": target.filing_status,
                    "agi_lower": target.agi_lower,
                    "agi_upper": target.agi_upper,
                    "is_count": target.is_count,
                    "rac_variable": target.rac_variable,
                    "rac_statute": target.rac_statute,
                    "microdata_column": target.microdata_column,
                }
            )
        return pd.DataFrame(rows)

    def to_calibration_format(
        self,
        geography: str = "US",
        year: int = 2021,
    ) -> tuple[dict[str, dict], dict[str, float]]:
        marginal_targets: dict[str, dict] = {}
        continuous_targets: dict[str, float] = {}

        for target in self.targets:
            if target.geography != geography or target.year != year:
                continue
            if target.microdata_column is None:
                continue
            if target.is_count:
                variable = target.microdata_column
                if variable not in marginal_targets:
                    marginal_targets[variable] = {}
                if target.agi_lower != -np.inf or target.agi_upper != np.inf:
                    category = f"{target.agi_lower:.0f}_to_{target.agi_upper:.0f}"
                else:
                    category = "all"
                marginal_targets[variable][category] = target.value
            else:
                continuous_targets[target.microdata_column] = target.value

        return marginal_targets, continuous_targets

    def compare_to_policyengine(self, pe_targets: pd.DataFrame) -> pd.DataFrame:
        our_df = self.to_dataframe()
        comparison = our_df.merge(
            pe_targets,
            left_on=["name", "year"],
            right_on=["Variable", "Year"],
            how="outer",
            suffixes=("_policyengine", "_pe"),
        )
        comparison["difference"] = comparison["value"] - comparison["Value"]
        comparison["pct_difference"] = comparison["difference"] / comparison["Value"] * 100
        return comparison

    def coverage_summary(self) -> dict[str, int]:
        return {
            category.value: len(self.get_by_category(category))
            for category in TargetCategory
        }

    def __len__(self) -> int:
        return len(self.targets)

    def __repr__(self) -> str:
        coverage = self.coverage_summary()
        non_zero = {key: value for key, value in coverage.items() if value > 0}
        return f"TargetsDatabase({len(self)} targets across {len(non_zero)} categories)"
