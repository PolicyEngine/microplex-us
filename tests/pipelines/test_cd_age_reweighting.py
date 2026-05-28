from __future__ import annotations

import sqlite3

import h5py
import numpy as np

from microplex_us.pipelines.cd_age_reweighting import (
    normalize_at_large_cd_geoids,
    reweight_h5_to_cd_age_targets,
)


def test_normalize_at_large_cd_geoids_maps_statewide_zero_to_one() -> None:
    values = np.asarray([200, 201, 1000, 3601, 0], dtype=np.int64)

    normalized = normalize_at_large_cd_geoids(values)

    np.testing.assert_array_equal(
        normalized,
        np.asarray([201, 201, 1001, 3601, 0], dtype=np.int64),
    )


def test_reweight_h5_to_cd_age_targets_matches_simple_at_large_targets(tmp_path) -> None:
    dataset = tmp_path / "input.h5"
    output = tmp_path / "output.h5"
    db = tmp_path / "policy_data.db"
    _write_minimal_h5(dataset)
    _write_cd_age_target_db(db)

    summary = reweight_h5_to_cd_age_targets(
        input_dataset=dataset,
        target_db=db,
        output_dataset=output,
        period=2024,
        max_iter=100,
        preserve_district_weight_sum=False,
    )

    assert summary["n_targets"] == 2
    assert summary["max_abs_relative_error_after"] < 1e-5
    with h5py.File(output, "r") as handle:
        np.testing.assert_allclose(
            handle["household_weight"]["2024"][:],
            np.asarray([10.0, 20.0], dtype=np.float32),
            rtol=1e-5,
        )
        np.testing.assert_array_equal(
            handle["congressional_district_geoid"]["2024"][:],
            np.asarray([201, 201]),
        )


def _write_minimal_h5(path):
    with h5py.File(path, "w") as handle:
        _write_period(handle, "household_id", [1, 2])
        _write_period(handle, "household_weight", [1.0, 1.0])
        _write_period(handle, "congressional_district_geoid", [200, 200])
        _write_period(handle, "person_household_id", [1, 2])
        _write_period(handle, "age", [4, 40])


def _write_period(handle, variable, values):
    group = handle.create_group(variable)
    group.create_dataset("2024", data=np.asarray(values))


def _write_cd_age_target_db(path):
    conn = sqlite3.connect(path)
    try:
        conn.executescript(
            """
            CREATE TABLE targets (
                target_id INTEGER PRIMARY KEY,
                variable TEXT,
                period INTEGER,
                stratum_id INTEGER,
                reform_id INTEGER DEFAULT 0,
                value REAL,
                active INTEGER DEFAULT 1,
                tolerance REAL,
                source TEXT,
                notes TEXT
            );
            CREATE TABLE strata (
                stratum_id INTEGER PRIMARY KEY,
                definition_hash TEXT,
                parent_stratum_id INTEGER
            );
            CREATE TABLE stratum_constraints (
                stratum_id INTEGER,
                constraint_variable TEXT,
                operation TEXT,
                value TEXT
            );
            CREATE VIEW target_overview AS
            SELECT
                target_id,
                stratum_id,
                variable,
                value,
                period,
                active,
                'district' AS geo_level,
                '201' AS geographic_id,
                'age' AS domain_variable
            FROM targets;
            """
        )
        _insert_target(conn, 1, 101, 10.0, [("age", "<", "18"), ("age", ">", "-1")])
        _insert_target(conn, 2, 102, 20.0, [("age", ">=", "18")])
        conn.commit()
    finally:
        conn.close()


def _insert_target(conn, target_id, stratum_id, value, constraints):
    conn.execute(
        """
        INSERT INTO targets
            (target_id, variable, period, stratum_id, reform_id, value, active)
        VALUES (?, 'person_count', 2024, ?, 0, ?, 1)
        """,
        (target_id, stratum_id, value),
    )
    conn.execute("INSERT INTO strata (stratum_id) VALUES (?)", (stratum_id,))
    for constraint in [
        ("congressional_district_geoid", "==", "201"),
        *constraints,
    ]:
        conn.execute(
            """
            INSERT INTO stratum_constraints
                (stratum_id, constraint_variable, operation, value)
            VALUES (?, ?, ?, ?)
            """,
            (stratum_id, *constraint),
        )
