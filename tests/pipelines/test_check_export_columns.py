"""Tests for the fast eCPS column-parity check CLI.

The module under test is loaded directly from its file path (not via
``import microplex_us...``) so these tests run with only ``pytest`` /
``h5py`` / ``numpy`` installed -- importing the ``microplex_us`` package
would pull ``microplex`` and torch. This mirrors the loader pattern in
``test_mp300k_artifact_gates.py``.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

_MODULE_PATH = (
    Path(__file__).resolve().parents[2]
    / "src"
    / "microplex_us"
    / "pipelines"
    / "check_export_columns.py"
)
_spec = importlib.util.spec_from_file_location("check_export_columns", _MODULE_PATH)
cec = importlib.util.module_from_spec(_spec)
# Register before exec so the module's @dataclass can resolve its module.
sys.modules["check_export_columns"] = cec
_spec.loader.exec_module(cec)

DEFAULT_CONTRACT_PATH = cec.DEFAULT_CONTRACT_PATH
DEFAULT_SPEC_PATH = cec.DEFAULT_SPEC_PATH
compute_column_diff = cec.compute_column_diff
compute_spec_variable_manifest_diff = cec.compute_spec_variable_manifest_diff
load_contract = cec.load_contract
main = cec.main

# A tiny self-contained contract so most tests do not depend on the
# (large) committed contract.
TINY_CONTRACT = {
    "required": ["age", "snap", "employment_income"],
    "ecps_internal_optional": ["person_is_puf_clone"],
    "forbidden": ["snap_reported", "ssi_reported"],
}


def _write_json(path: Path, obj) -> Path:
    path.write_text(json.dumps(obj))
    return path


@pytest.fixture
def contract_path(tmp_path: Path) -> Path:
    return _write_json(tmp_path / "contract.json", TINY_CONTRACT)


def _run_columns(
    tmp_path: Path,
    contract_path: Path,
    columns: list[str],
) -> int:
    cols_path = _write_json(tmp_path / "cols.json", columns)
    return main(
        [
            "--columns-json",
            str(cols_path),
            "--contract",
            str(contract_path),
        ]
    )


def _write_period_h5(path: Path, columns: dict[str, list[object]]) -> Path:
    h5py = pytest.importorskip("h5py")
    import numpy as np

    with h5py.File(path, "w") as f:
        for column, values in columns.items():
            f.create_dataset(f"{column}/2024", data=np.asarray(values))
    return path


def test_main_clean_list_returns_zero(tmp_path, contract_path):
    # required + optional, no forbidden -> pass.
    cols = ["age", "snap", "employment_income", "person_is_puf_clone"]
    assert _run_columns(tmp_path, contract_path, cols) == 0


def test_main_missing_required_returns_one(tmp_path, contract_path):
    # Drop a required column.
    cols = ["age", "snap"]  # missing employment_income
    assert _run_columns(tmp_path, contract_path, cols) == 1


def test_main_forbidden_present_returns_one(tmp_path, contract_path):
    # All required present, but a forbidden column is exported.
    cols = ["age", "snap", "employment_income", "snap_reported"]
    assert _run_columns(tmp_path, contract_path, cols) == 1


def test_columns_json_path_collapses_period_suffix(tmp_path, contract_path):
    # "name/period" entries collapse to the base name and still pass.
    cols = ["age/2024", "snap/2024", "employment_income/2024"]
    assert _run_columns(tmp_path, contract_path, cols) == 0


def test_optional_column_is_neither_required_nor_forbidden(tmp_path, contract_path):
    # Omitting an optional column does not fail; it is not "missing".
    cols = ["age", "snap", "employment_income"]
    assert _run_columns(tmp_path, contract_path, cols) == 0


def test_main_h5_path_returns_zero_when_clean(tmp_path, contract_path):
    h5py = pytest.importorskip("h5py")
    import numpy as np

    # Mirror the eCPS export layout: each column is a group <col>/<period>.
    h5_path = tmp_path / "export.h5"
    with h5py.File(h5_path, "w") as f:
        for col in ["age", "snap", "employment_income"]:
            f.create_dataset(f"{col}/2024", data=np.array([1, 2, 3]))
    rc = main([str(h5_path), "--contract", str(contract_path)])
    assert rc == 0


def test_main_h5_path_flags_missing_required(tmp_path, contract_path):
    h5py = pytest.importorskip("h5py")
    import numpy as np

    h5_path = tmp_path / "export.h5"
    with h5py.File(h5_path, "w") as f:
        # missing employment_income
        for col in ["age", "snap"]:
            f.create_dataset(f"{col}/2024", data=np.array([1, 2, 3]))
    rc = main([str(h5_path), "--contract", str(contract_path)])
    assert rc == 1


def test_main_h5_path_flags_forbidden_present(tmp_path, contract_path):
    h5py = pytest.importorskip("h5py")
    import numpy as np

    h5_path = tmp_path / "export.h5"
    with h5py.File(h5_path, "w") as f:
        for col in ["age", "snap", "employment_income", "snap_reported"]:
            f.create_dataset(f"{col}/2024", data=np.array([1, 2, 3]))
    rc = main([str(h5_path), "--contract", str(contract_path)])
    assert rc == 1


def test_main_h5_path_accepts_flat_datasets(tmp_path, contract_path):
    # A flat dataset layout (no period sub-group) is also accepted.
    h5py = pytest.importorskip("h5py")
    import numpy as np

    h5_path = tmp_path / "export.h5"
    with h5py.File(h5_path, "w") as f:
        for col in ["age", "snap", "employment_income"]:
            f.create_dataset(col, data=np.array([1, 2, 3]))
    rc = main([str(h5_path), "--contract", str(contract_path)])
    assert rc == 0


def test_support_baseline_rejects_numeric_column_eCPS_populates(
    tmp_path,
    contract_path,
):
    candidate = _write_period_h5(
        tmp_path / "candidate.h5",
        {
            "age": [34, 42, 50],
            "snap": [False, True, False],
            "employment_income": [0.0, 0.0, 0.0],
        },
    )
    baseline = _write_period_h5(
        tmp_path / "baseline.h5",
        {
            "age": [34, 42, 50],
            "snap": [False, True, False],
            "employment_income": [0.0, 12_000.0, 0.0],
        },
    )

    rc = main(
        [
            str(candidate),
            "--contract",
            str(contract_path),
            "--support-baseline",
            str(baseline),
        ]
    )

    assert rc == 1


def test_support_baseline_rejects_missing_numeric_sign_support(
    tmp_path,
):
    contract_path = _write_json(
        tmp_path / "contract.json",
        {
            "required": ["age", "snap", "rental_income"],
            "ecps_internal_optional": [],
            "forbidden": [],
        },
    )
    candidate = _write_period_h5(
        tmp_path / "candidate.h5",
        {
            "age": [34, 42, 50],
            "snap": [False, True, False],
            "rental_income": [0.0, 12_000.0, 0.0],
        },
    )
    baseline = _write_period_h5(
        tmp_path / "baseline.h5",
        {
            "age": [34, 42, 50],
            "snap": [False, True, False],
            "rental_income": [-200.0, 12_000.0, 0.0],
        },
    )
    diagnostics = tmp_path / "support.json"

    rc = main(
        [
            str(candidate),
            "--contract",
            str(contract_path),
            "--support-baseline",
            str(baseline),
            "--support-diagnostics-json",
            str(diagnostics),
        ]
    )

    assert rc == 1
    payload = json.loads(diagnostics.read_text())
    assert payload["issues"][0]["column"] == "rental_income"
    assert payload["issues"][0]["requirement"] == "numeric_signed"
    assert payload["issues"][0]["baseline"]["negative_count"] == 1
    assert payload["issues"][0]["candidate"]["negative_count"] == 0


def test_support_baseline_accepts_negative_noise_for_unsigned_numeric(
    tmp_path,
    contract_path,
):
    candidate = _write_period_h5(
        tmp_path / "candidate.h5",
        {
            "age": [34, 42, 50],
            "snap": [False, True, False],
            "employment_income": [0.0, 12_000.0, 0.0],
        },
    )
    baseline = _write_period_h5(
        tmp_path / "baseline.h5",
        {
            "age": [34, 42, 50],
            "snap": [False, True, False],
            "employment_income": [-200.0, 12_000.0, 0.0],
        },
    )

    rc = main(
        [
            str(candidate),
            "--contract",
            str(contract_path),
            "--support-baseline",
            str(baseline),
        ]
    )

    assert rc == 0


def test_support_baseline_rejects_categorical_column_eCPS_varies(
    tmp_path,
    contract_path,
):
    candidate = _write_period_h5(
        tmp_path / "candidate.h5",
        {
            "age": [34, 42, 50],
            "snap": [False, False, False],
            "employment_income": [0.0, 12_000.0, 0.0],
        },
    )
    baseline = _write_period_h5(
        tmp_path / "baseline.h5",
        {
            "age": [34, 42, 50],
            "snap": [False, True, False],
            "employment_income": [0.0, 12_000.0, 0.0],
        },
    )

    rc = main(
        [
            str(candidate),
            "--contract",
            str(contract_path),
            "--support-baseline",
            str(baseline),
        ]
    )

    assert rc == 1


def test_support_baseline_ignores_ecps_filler_columns(tmp_path, contract_path):
    candidate = _write_period_h5(
        tmp_path / "candidate.h5",
        {
            "age": [34, 42, 50],
            "snap": [False, True, False],
            "employment_income": [0.0, 0.0, 0.0],
        },
    )
    baseline = _write_period_h5(
        tmp_path / "baseline.h5",
        {
            "age": [34, 42, 50],
            "snap": [False, True, False],
            "employment_income": [0.0, 0.0, 0.0],
        },
    )

    rc = main(
        [
            str(candidate),
            "--contract",
            str(contract_path),
            "--support-baseline",
            str(baseline),
        ]
    )

    assert rc == 0


def test_support_baseline_accepts_candidate_categorical_support_for_numeric_ecps(
    tmp_path,
    contract_path,
):
    candidate = _write_period_h5(
        tmp_path / "candidate.h5",
        {
            "age": [b"34", b"42", b"50"],
            "snap": [False, True, False],
            "employment_income": [0.0, 12_000.0, 0.0],
        },
    )
    baseline = _write_period_h5(
        tmp_path / "baseline.h5",
        {
            "age": [34, 42, 50],
            "snap": [False, True, False],
            "employment_income": [0.0, 12_000.0, 0.0],
        },
    )

    rc = main(
        [
            str(candidate),
            "--contract",
            str(contract_path),
            "--support-baseline",
            str(baseline),
        ]
    )

    assert rc == 0


def test_support_baseline_writes_diagnostics_and_honors_explicit_exemption(
    tmp_path,
    contract_path,
):
    candidate = _write_period_h5(
        tmp_path / "candidate.h5",
        {
            "age": [34, 42, 50],
            "snap": [False, True, False],
            "employment_income": [0.0, 0.0, 0.0],
        },
    )
    baseline = _write_period_h5(
        tmp_path / "baseline.h5",
        {
            "age": [34, 42, 50],
            "snap": [False, True, False],
            "employment_income": [0.0, 12_000.0, 0.0],
        },
    )
    diagnostics = tmp_path / "support.json"

    rc = main(
        [
            str(candidate),
            "--contract",
            str(contract_path),
            "--support-baseline",
            str(baseline),
            "--support-exempt-column",
            "employment_income",
            "--support-diagnostics-json",
            str(diagnostics),
        ]
    )

    assert rc == 0
    payload = json.loads(diagnostics.read_text())
    assert payload["issues"] == []
    assert payload["exempt_columns"] == ["employment_income"]


def test_main_entity_tables_path_uses_schema_columns(
    tmp_path, contract_path, monkeypatch
):
    checkpoint_dir = tmp_path / "post-imputation"

    def fake_columns(path, *, direct_override_variables):
        assert path == checkpoint_dir
        assert direct_override_variables == ("non_sch_d_capital_gains",)
        return {"age", "snap", "employment_income"}

    monkeypatch.setattr(cec, "_columns_from_entity_tables", fake_columns)

    rc = main(
        [
            "--entity-tables",
            str(checkpoint_dir),
            "--direct-override-variable",
            "non_sch_d_capital_gains",
            "--contract",
            str(contract_path),
        ]
    )

    assert rc == 0


def test_main_explicit_spec_variable_manifest_failure_returns_one(
    tmp_path,
    contract_path,
):
    spec_path = tmp_path / "spec.yaml"
    spec_path.write_text(
        """
meta: {country: us, model_year: 2024}
imputation:
  - onto: synthetic_puf
    from: puf
    vars: [employment_income]
variables:
  age:
    mp_spec: {method: passthrough}
  snap:
    mp_spec: {method: passthrough}
""",
        encoding="utf-8",
    )

    cols_path = _write_json(
        tmp_path / "cols.json",
        ["age", "snap", "employment_income"],
    )
    rc = main(
        [
            "--columns-json",
            str(cols_path),
            "--contract",
            str(contract_path),
            "--spec",
            str(spec_path),
        ]
    )

    assert rc == 1


def test_main_requires_exactly_one_input(tmp_path, contract_path):
    # Neither input -> argparse error (SystemExit code 2).
    with pytest.raises(SystemExit) as exc:
        main(["--contract", str(contract_path)])
    assert exc.value.code == 2

    # Both inputs -> argparse error.
    cols_path = _write_json(tmp_path / "c.json", ["age"])
    with pytest.raises(SystemExit) as exc:
        main(
            [
                str(tmp_path / "x.h5"),
                "--columns-json",
                str(cols_path),
                "--contract",
                str(contract_path),
            ]
        )
    assert exc.value.code == 2

    with pytest.raises(SystemExit) as exc:
        main(
            [
                "--columns-json",
                str(cols_path),
                "--support-baseline",
                str(tmp_path / "baseline.h5"),
                "--contract",
                str(contract_path),
            ]
        )
    assert exc.value.code == 2


def test_compute_column_diff_categories():
    diff = compute_column_diff(
        {"age", "snap", "snap_reported", "mystery"},
        required={"age", "snap", "wages"},
        forbidden={"snap_reported"},
        optional={"person_is_puf_clone"},
    )
    assert diff.missing_required == ["wages"]
    assert diff.forbidden_present == ["snap_reported"]
    assert diff.extra_unknown == ["mystery"]
    assert diff.ok is False


def test_load_contract_rejects_missing_keys(tmp_path):
    bad = _write_json(tmp_path / "bad.json", {"required": ["age"]})
    with pytest.raises(ValueError, match="forbidden"):
        load_contract(bad)


def test_spec_variable_manifest_diff_covers_committed_spec():
    diff = compute_spec_variable_manifest_diff(
        contract=load_contract(DEFAULT_CONTRACT_PATH),
        spec_path=DEFAULT_SPEC_PATH,
    )

    assert diff.ok
    assert diff.required_contract_count == 252
    assert diff.declared_imputation_count == 76
    assert diff.variable_manifest_count == 278
    assert diff.missing_required == []
    assert diff.missing_declared_imputation == []
    assert diff.extra_variables == []


def test_spec_variable_manifest_diff_flags_missing_required_and_imputation(tmp_path):
    contract = {
        "required": ["age", "snap"],
        "forbidden": [],
        "ecps_internal_optional": [],
    }
    spec_path = tmp_path / "spec.yaml"
    spec_path.write_text(
        """
meta: {country: us, model_year: 2024}
imputation:
  - onto: synthetic_puf
    from: puf
    vars: [employment_income]
variables:
  age:
    mp_spec: {method: passthrough}
""",
        encoding="utf-8",
    )

    diff = compute_spec_variable_manifest_diff(
        contract=contract,
        spec_path=spec_path,
    )

    assert diff.ok is False
    assert diff.missing_required == ["snap"]
    assert diff.missing_declared_imputation == ["employment_income"]
    assert diff.extra_variables == []


def test_committed_contract_parses_with_expected_categories():
    contract = load_contract(DEFAULT_CONTRACT_PATH)
    for key in (
        "required",
        "ecps_internal_optional",
        "forbidden",
        "formula_owned_excluded",
    ):
        assert key in contract, f"contract missing '{key}'"
        assert isinstance(contract[key], list)
    # Category sizes of the eCPS contract, aligned to the clone-correct baseline
    # H5 (postfix_clonecorrect plus target-source gap fixes): required exports
    # both the 5 capped retirement account inputs and the 5 *_desired
    # retirement inputs, forbids the
    # PUF_REPORTED_CALCULATED_TAX_OUTPUT_VARIABLES tax-credit outputs, and
    # excludes only weeks_worked (the lone pe-us formula var the baseline does
    # not persist). Structural/overridable computed fields
    # (has_tin/has_itin/in_nyc/fsla_overtime_premium/meets_ssi_disability_criteria)
    # are REQUIRED, matching the in-tree _column_contract_gate.
    # Sizes sum to the 258-column source-backed baseline: 252 + 5 + 1.
    assert len(contract["required"]) == 252
    assert len(contract["ecps_internal_optional"]) == 5
    assert len(contract["forbidden"]) == 22
    assert len(contract["formula_owned_excluded"]) == 1
    # Categories must be disjoint.
    req = set(contract["required"])
    opt = set(contract["ecps_internal_optional"])
    forb = set(contract["forbidden"])
    excl = set(contract["formula_owned_excluded"])
    assert req.isdisjoint(opt)
    assert req.isdisjoint(forb)
    assert opt.isdisjoint(forb)
    assert excl.isdisjoint(req)
    assert excl.isdisjoint(forb)
    # The clone-bookkeeping flags are optional, not required.
    assert "person_is_puf_clone" in opt
    assert "person_is_puf_clone" not in req
    # Structural/overridable computed fields are REQUIRED (in-tree gate parity),
    # NOT excluded; only weeks_worked is excluded.
    for structural in (
        "has_tin",
        "has_itin",
        "in_nyc",
        "fsla_overtime_premium",
        "meets_ssi_disability_criteria",
        "difficulty_hearing",
    ):
        assert structural in req
    assert excl == {"weeks_worked"}


def test_committed_clean_fixture_passes_committed_contract(capsys):
    # The CI fixture must be a clean, passing set against the real
    # contract so the green CI path proves the gate passes on good data.
    fixture = Path(__file__).parent / "fixtures" / "ecps_clean_columns.json"
    rc = main(["--columns-json", str(fixture)])
    report = capsys.readouterr().out
    assert rc == 0
    assert "spec variable manifest" in report


def test_committed_contract_covers_every_baseline_column():
    # Completeness invariant: every column the clean baseline fixture exports
    # must be accounted for by some contract category, so a baseline-shaped
    # export produces no extra_unknown columns. This pins the contract to the
    # real baseline and catches silent under-specification of `required`.
    contract = load_contract(DEFAULT_CONTRACT_PATH)
    fixture = Path(__file__).parent / "fixtures" / "ecps_clean_columns.json"
    present = set(json.loads(fixture.read_text()))
    diff = compute_column_diff(
        present,
        required=set(contract["required"]),
        forbidden=set(contract["forbidden"]),
        optional=set(contract["ecps_internal_optional"]),
        excluded=set(contract["formula_owned_excluded"]),
    )
    assert diff.extra_unknown == []
    assert diff.missing_required == []
    assert diff.forbidden_present == []


def test_default_contract_path_is_packaged():
    # The contract ships next to the module so the default path resolves.
    assert DEFAULT_CONTRACT_PATH.name == "ecps_export_contract.json"
    assert DEFAULT_CONTRACT_PATH.exists()
    assert callable(cec.main)
