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
compute_column_diff = cec.compute_column_diff
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


def test_committed_contract_parses_with_expected_categories():
    contract = load_contract(DEFAULT_CONTRACT_PATH)
    for key in ("required", "ecps_internal_optional", "forbidden"):
        assert key in contract, f"contract missing '{key}'"
        assert isinstance(contract[key], list)
    # Category sizes of the frozen eCPS contract (244 = 239 + 5).
    assert len(contract["required"]) == 239
    assert len(contract["ecps_internal_optional"]) == 5
    assert len(contract["forbidden"]) == 15
    # Categories must be disjoint.
    req = set(contract["required"])
    opt = set(contract["ecps_internal_optional"])
    forb = set(contract["forbidden"])
    assert req.isdisjoint(opt)
    assert req.isdisjoint(forb)
    assert opt.isdisjoint(forb)
    # The clone-bookkeeping flags are optional, not required.
    assert "person_is_puf_clone" in opt
    assert "person_is_puf_clone" not in req


def test_committed_clean_fixture_passes_committed_contract():
    # The CI fixture must be a clean, passing set against the real
    # contract so the green CI path proves the gate passes on good data.
    fixture = Path(__file__).parent / "fixtures" / "ecps_clean_columns.json"
    rc = main(["--columns-json", str(fixture)])
    assert rc == 0


def test_default_contract_path_is_packaged():
    # The contract ships next to the module so the default path resolves.
    assert DEFAULT_CONTRACT_PATH.name == "ecps_export_contract.json"
    assert DEFAULT_CONTRACT_PATH.exists()
    assert callable(cec.main)
