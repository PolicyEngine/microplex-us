"""Synthetic capital-gains lot generation and relational persistence."""

from __future__ import annotations

import json
import sqlite3
from dataclasses import asdict, dataclass
from hashlib import blake2b
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

CAPITAL_GAINS_LOT_COLUMNS: tuple[str, ...] = (
    "lot_id",
    "person_id",
    "tax_unit_id",
    "household_id",
    "tax_year",
    "lot_index",
    "sale_time",
    "holding_period",
    "purchase_time",
    "sale_proceeds",
    "basis",
    "gain_or_loss",
    "asset_type",
    "is_gain_lot",
)


@dataclass(frozen=True)
class SyntheticCapitalGainsLotConfig:
    """Controls the first-pass synthetic-lot imputation."""

    random_seed: int = 42
    max_lots_per_person: int = 4
    high_gain_threshold: float = 100_000.0
    medium_gain_threshold: float = 10_000.0
    annual_nominal_return: float = 0.07
    gain_basis_ratio_floor: float = 0.05
    gain_basis_ratio_ceiling: float = 0.95
    loss_basis_ratio_floor: float = 1.05
    max_holding_period_years: int = 35


def generate_synthetic_capital_gains_lots(
    persons: pd.DataFrame,
    *,
    period: int,
    config: SyntheticCapitalGainsLotConfig | None = None,
    gain_column: str = "long_term_capital_gains_before_response",
) -> pd.DataFrame:
    """Generate deterministic synthetic long-term capital-gains lots.

    The generator is anchored to the existing person-level PolicyEngine input:
    lots aggregate exactly back to each person's capital-gains amount. It is a
    relational artifact scaffold, not a SOCA-calibrated production imputation.
    """

    resolved = config or SyntheticCapitalGainsLotConfig()
    if gain_column not in persons.columns:
        raise ValueError(f"persons is missing required column {gain_column!r}")
    if resolved.max_lots_per_person < 1:
        raise ValueError("max_lots_per_person must be at least 1")

    rows: list[dict[str, Any]] = []
    for position, (_, person) in enumerate(persons.iterrows()):
        raw_gain = pd.to_numeric(person[gain_column], errors="coerce")
        gain = 0.0 if pd.isna(raw_gain) else float(raw_gain)
        if np.isclose(gain, 0.0):
            continue

        person_id = _optional_int(person.get("person_id"))
        tax_unit_id = _optional_int(person.get("tax_unit_id"))
        household_id = _optional_int(person.get("household_id"))
        sign = "gain" if gain > 0 else "loss"
        stable_key = person_id if person_id is not None else f"row-{position}"
        rng = np.random.default_rng(
            _stable_seed(
                resolved.random_seed,
                period,
                stable_key,
                tax_unit_id,
                sign,
                "synthetic-capital-gains-lots-v1",
            )
        )
        n_lots = _lot_count(abs(gain), resolved)
        shares = _deterministic_lot_shares(n_lots, rng)
        signed_lot_gains = shares * gain
        holding_periods = _draw_holding_periods(n_lots, rng, resolved)

        for lot_index, (lot_gain, holding_period) in enumerate(
            zip(signed_lot_gains, holding_periods, strict=True)
        ):
            sale_proceeds, basis = _basis_and_proceeds(
                lot_gain,
                int(holding_period),
                resolved,
            )
            sale_time = float(period) + 0.5
            rows.append(
                {
                    "lot_id": 0,
                    "person_id": person_id,
                    "tax_unit_id": tax_unit_id,
                    "household_id": household_id,
                    "tax_year": int(period),
                    "lot_index": int(lot_index),
                    "sale_time": sale_time,
                    "holding_period": float(holding_period),
                    "purchase_time": sale_time - float(holding_period),
                    "sale_proceeds": float(sale_proceeds),
                    "basis": float(basis),
                    "gain_or_loss": float(lot_gain),
                    "asset_type": "unknown",
                    "is_gain_lot": bool(lot_gain > 0),
                }
            )

    lots = pd.DataFrame(rows, columns=CAPITAL_GAINS_LOT_COLUMNS)
    if lots.empty:
        return lots
    lots = lots.sort_values(
        ["person_id", "tax_year", "lot_index"], kind="stable"
    ).reset_index(drop=True)
    lots["lot_id"] = np.arange(1, len(lots) + 1, dtype=np.int64)
    return lots.astype(
        {
            "lot_id": "int64",
            "tax_year": "int64",
            "lot_index": "int64",
            "sale_time": "float64",
            "holding_period": "float64",
            "purchase_time": "float64",
            "sale_proceeds": "float64",
            "basis": "float64",
            "gain_or_loss": "float64",
            "asset_type": "string",
            "is_gain_lot": "bool",
        }
    )


def validate_capital_gains_lot_anchors(
    persons: pd.DataFrame,
    lots: pd.DataFrame,
    *,
    gain_column: str = "long_term_capital_gains_before_response",
    tolerance: float = 1e-5,
    relative_tolerance: float = 1e-9,
) -> None:
    """Raise if lot totals do not reconcile to person-level capital gains."""

    if gain_column not in persons.columns:
        raise ValueError(f"persons is missing required column {gain_column!r}")
    if "person_id" not in persons.columns:
        raise ValueError("persons is missing required column 'person_id'")
    missing_lot_columns = {"person_id", "gain_or_loss"} - set(lots.columns)
    if missing_lot_columns:
        raise ValueError(f"lots is missing columns: {sorted(missing_lot_columns)}")

    anchors = (
        persons[["person_id", gain_column]]
        .assign(
            person_id=lambda df: pd.to_numeric(df["person_id"], errors="coerce"),
            _anchor=lambda df: pd.to_numeric(df[gain_column], errors="coerce").fillna(
                0.0
            ),
        )
        .groupby("person_id", dropna=False)["_anchor"]
        .sum()
    )
    lot_totals = (
        lots.assign(
            person_id=pd.to_numeric(lots["person_id"], errors="coerce"),
            gain_or_loss=pd.to_numeric(lots["gain_or_loss"], errors="coerce").fillna(
                0.0
            ),
        )
        .groupby("person_id", dropna=False)["gain_or_loss"]
        .sum()
    )
    combined = pd.concat([anchors, lot_totals], axis=1).fillna(0.0)
    combined.columns = ["anchor", "lot_total"]
    deltas = (combined["anchor"] - combined["lot_total"]).abs()
    bad = deltas[
        ~np.isclose(
            combined["anchor"],
            combined["lot_total"],
            atol=tolerance,
            rtol=relative_tolerance,
        )
    ]
    if not bad.empty:
        worst_person = bad.idxmax()
        raise ValueError(
            "Synthetic capital-gains lots do not reconcile to person anchors; "
            f"worst person_id={worst_person!r}, delta={float(bad.max())}"
        )


def write_capital_gains_lots_sqlite(
    lots: pd.DataFrame,
    path: str | Path,
    *,
    metadata: dict[str, Any] | None = None,
    if_exists: str = "replace",
) -> Path:
    """Persist synthetic lots to a compact SQLite artifact."""

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    missing = set(CAPITAL_GAINS_LOT_COLUMNS) - set(lots.columns)
    if missing:
        raise ValueError(f"lots is missing columns: {sorted(missing)}")
    with sqlite3.connect(output_path) as conn:
        lots.loc[:, CAPITAL_GAINS_LOT_COLUMNS].to_sql(
            "capital_gains_lots",
            conn,
            index=False,
            if_exists=if_exists,
        )
        conn.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_capital_gains_lots_person_period
            ON capital_gains_lots (person_id, tax_year)
            """
        )
        conn.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_capital_gains_lots_tax_unit_period
            ON capital_gains_lots (tax_unit_id, tax_year)
            """
        )
        conn.execute("DROP TABLE IF EXISTS capital_gains_lot_metadata")
        conn.execute(
            """
            CREATE TABLE capital_gains_lot_metadata (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL
            )
            """
        )
        for key, value in (metadata or {}).items():
            conn.execute(
                """
                INSERT INTO capital_gains_lot_metadata (key, value)
                VALUES (?, ?)
                """,
                (str(key), json.dumps(value, sort_keys=True)),
            )
    return output_path


def read_capital_gains_lots_sqlite(path: str | Path) -> pd.DataFrame:
    """Read a synthetic capital-gains lot SQLite artifact."""

    with sqlite3.connect(Path(path)) as conn:
        return pd.read_sql_query(
            """
            SELECT *
            FROM capital_gains_lots
            ORDER BY lot_id
            """,
            conn,
        )


def synthetic_capital_gains_lot_metadata(
    config: SyntheticCapitalGainsLotConfig,
    *,
    period: int,
    source_gain_column: str = "long_term_capital_gains_before_response",
) -> dict[str, Any]:
    """Build metadata for the current synthetic-lot artifact contract."""

    return {
        "format_version": 1,
        "tax_year": int(period),
        "source_gain_column": source_gain_column,
        "config": asdict(config),
        "method": "deterministic_anchor_preserving_synthetic_lots_phase_1",
        "capital_gains_lots_issue": (
            "https://github.com/PolicyEngine/policyengine-us-data/issues/1127"
        ),
        "limitations": (
            "Phase 1 prototype: no SOCA calibration, no asset type assignment, "
            "and no mixed gross gain/loss reconstruction."
        ),
    }


def _lot_count(amount: float, config: SyntheticCapitalGainsLotConfig) -> int:
    if amount >= config.high_gain_threshold:
        return min(config.max_lots_per_person, 4)
    if amount >= config.medium_gain_threshold:
        return min(config.max_lots_per_person, 2)
    return 1


def _deterministic_lot_shares(n_lots: int, rng: np.random.Generator) -> np.ndarray:
    if n_lots == 1:
        return np.array([1.0], dtype=float)
    shares = rng.dirichlet(np.full(n_lots, 1.5))
    shares[-1] = 1.0 - float(shares[:-1].sum())
    return shares


def _draw_holding_periods(
    n_lots: int,
    rng: np.random.Generator,
    config: SyntheticCapitalGainsLotConfig,
) -> np.ndarray:
    buckets = np.array([2, 3, 5, 8, 12, 20, 30], dtype=int)
    weights = np.array([0.12, 0.16, 0.2, 0.18, 0.16, 0.12, 0.06], dtype=float)
    holding_periods = rng.choice(buckets, size=n_lots, replace=True, p=weights)
    return np.clip(holding_periods, 2, config.max_holding_period_years)


def _basis_and_proceeds(
    gain_or_loss: float,
    holding_period_years: int,
    config: SyntheticCapitalGainsLotConfig,
) -> tuple[float, float]:
    if gain_or_loss > 0:
        raw_basis_ratio = 1.0 / (
            (1.0 + config.annual_nominal_return) ** holding_period_years
        )
        basis_ratio = float(
            np.clip(
                raw_basis_ratio,
                config.gain_basis_ratio_floor,
                config.gain_basis_ratio_ceiling,
            )
        )
        basis = gain_or_loss * basis_ratio / (1.0 - basis_ratio)
        return basis + gain_or_loss, basis

    loss = abs(gain_or_loss)
    basis_ratio = max(
        1.0 + config.annual_nominal_return * holding_period_years / 4.0,
        config.loss_basis_ratio_floor,
    )
    basis = loss * basis_ratio / (basis_ratio - 1.0)
    return basis - loss, basis


def _optional_int(value: Any) -> int | None:
    if value is None or pd.isna(value):
        return None
    return int(value)


def _stable_seed(*parts: Any) -> int:
    payload = "|".join(str(part) for part in parts).encode()
    digest = blake2b(payload, digest_size=8).digest()
    return int.from_bytes(digest, byteorder="little", signed=False)
