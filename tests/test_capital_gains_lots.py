from __future__ import annotations

import sqlite3

import numpy as np
import pandas as pd
import pytest

from microplex_us.capital_gains_lots import (
    CAPITAL_GAINS_LOT_COLUMNS,
    SyntheticCapitalGainsLotConfig,
    generate_synthetic_capital_gains_lots,
    read_capital_gains_lots_sqlite,
    synthetic_capital_gains_lot_metadata,
    validate_capital_gains_lot_anchors,
    write_capital_gains_lots_sqlite,
)
from microplex_us.pipelines.artifacts import _maybe_write_capital_gains_lot_artifact
from microplex_us.pipelines.us import (
    USMicroplexBuildConfig,
    USMicroplexBuildResult,
    USMicroplexTargets,
)
from microplex_us.policyengine.us import PolicyEngineUSEntityTableBundle


def test_generate_synthetic_capital_gains_lots_preserves_person_anchors():
    persons = pd.DataFrame(
        {
            "person_id": [1, 2, 3],
            "tax_unit_id": [10, 20, 20],
            "household_id": [100, 200, 200],
            "long_term_capital_gains_before_response": [150_000.0, -20_000.0, 0.0],
        }
    )
    config = SyntheticCapitalGainsLotConfig(random_seed=7, max_lots_per_person=3)

    lots = generate_synthetic_capital_gains_lots(
        persons,
        period=2026,
        config=config,
    )

    assert list(lots.columns) == list(CAPITAL_GAINS_LOT_COLUMNS)
    assert set(lots["person_id"]) == {1, 2}
    assert lots.groupby("person_id")["gain_or_loss"].sum().to_dict() == pytest.approx(
        {1: 150_000.0, 2: -20_000.0}
    )
    np.testing.assert_allclose(
        lots["sale_proceeds"] - lots["basis"],
        lots["gain_or_loss"],
    )
    assert set(lots["asset_type"]) == {"unknown"}
    assert (lots["sale_time"] == 2026.5).all()
    np.testing.assert_allclose(
        lots["sale_time"] - lots["holding_period"],
        lots["purchase_time"],
    )
    validate_capital_gains_lot_anchors(persons, lots)


def test_synthetic_capital_gains_lots_are_deterministic():
    persons = pd.DataFrame(
        {
            "person_id": [1, 2],
            "tax_unit_id": [10, 20],
            "household_id": [100, 200],
            "long_term_capital_gains_before_response": [250_000.0, 125_000.0],
        }
    )
    config = SyntheticCapitalGainsLotConfig(random_seed=99)

    first = generate_synthetic_capital_gains_lots(persons, period=2026, config=config)
    second = generate_synthetic_capital_gains_lots(persons, period=2026, config=config)

    pd.testing.assert_frame_equal(first, second)

    shuffled = persons.sample(frac=1.0, random_state=2).reset_index(drop=True)
    shuffled_lots = generate_synthetic_capital_gains_lots(
        shuffled,
        period=2026,
        config=config,
    )

    pd.testing.assert_frame_equal(first, shuffled_lots)


def test_validate_capital_gains_lot_anchors_allows_float_roundoff():
    persons = pd.DataFrame(
        {
            "person_id": [1],
            "long_term_capital_gains_before_response": [6_005.71],
        }
    )
    lots = pd.DataFrame(
        {
            "person_id": [1, 1],
            "gain_or_loss": [3_000.0, 3_005.7100076293945],
        }
    )

    validate_capital_gains_lot_anchors(persons, lots)


def test_validate_capital_gains_lot_anchors_rejects_material_mismatch():
    persons = pd.DataFrame(
        {
            "person_id": [1],
            "long_term_capital_gains_before_response": [6_005.71],
        }
    )
    lots = pd.DataFrame({"person_id": [1], "gain_or_loss": [6_005.72]})

    with pytest.raises(ValueError, match="do not reconcile"):
        validate_capital_gains_lot_anchors(persons, lots)


def test_write_and_read_capital_gains_lots_sqlite(tmp_path):
    persons = pd.DataFrame(
        {
            "person_id": [1],
            "tax_unit_id": [10],
            "household_id": [100],
            "long_term_capital_gains_before_response": [15_000.0],
        }
    )
    config = SyntheticCapitalGainsLotConfig(random_seed=1)
    lots = generate_synthetic_capital_gains_lots(persons, period=2026, config=config)
    db_path = tmp_path / "capital_gains_lots.db"

    write_capital_gains_lots_sqlite(
        lots,
        db_path,
        metadata=synthetic_capital_gains_lot_metadata(config, period=2026),
    )
    restored = read_capital_gains_lots_sqlite(db_path)

    assert restored["gain_or_loss"].sum() == pytest.approx(15_000.0)
    with sqlite3.connect(db_path) as conn:
        metadata = dict(
            conn.execute("SELECT key, value FROM capital_gains_lot_metadata")
        )
        index_rows = conn.execute(
            """
            SELECT name
            FROM sqlite_master
            WHERE type = 'index'
            ORDER BY name
            """
        ).fetchall()
    assert "config" in metadata
    assert "limitations" in metadata
    assert ("idx_capital_gains_lots_person_period",) in index_rows
    assert ("idx_capital_gains_lots_tax_unit_period",) in index_rows


def test_capital_gains_lot_artifact_sidecar_is_config_gated(tmp_path):
    persons = pd.DataFrame(
        {
            "person_id": [1, 2],
            "tax_unit_id": [10, 20],
            "household_id": [100, 200],
            "long_term_capital_gains_before_response": [25_000.0, 0.0],
        }
    )
    result = USMicroplexBuildResult(
        config=USMicroplexBuildConfig(
            capital_gains_lots_enabled=True,
            capital_gains_lots_max_lots_per_person=2,
            policyengine_dataset_year=2026,
        ),
        seed_data=pd.DataFrame(),
        synthetic_data=pd.DataFrame(),
        calibrated_data=pd.DataFrame(),
        targets=USMicroplexTargets(marginal={}, continuous={}),
        calibration_summary={},
        policyengine_tables=PolicyEngineUSEntityTableBundle(
            households=pd.DataFrame({"household_id": [100, 200]}),
            persons=persons,
        ),
    )

    path, summary = _maybe_write_capital_gains_lot_artifact(result, tmp_path)

    assert path == tmp_path / "capital_gains_lots.sqlite"
    assert path.exists()
    assert summary == {
        "enabled": True,
        "written": True,
        "path": "capital_gains_lots.sqlite",
        "person_rows": 2,
        "nonzero_person_rows": 1,
        "lot_rows": 2,
        "source_gain_column": "long_term_capital_gains_before_response",
        "max_lots_per_person": 2,
    }
    restored = read_capital_gains_lots_sqlite(path)
    validate_capital_gains_lot_anchors(persons, restored)
