"""Fast eCPS column-parity check for exported datasets.

This is the cheap, millisecond gate that should pass *before* the
expensive MP-300k build. It compares the column set of a candidate export
against a frozen contract describing what the enhanced CPS (eCPS) baseline
exports, so column drift is catchable locally and in CI without producing
any data.

The required/forbidden column diff here mirrors the one inside
``_column_contract_gate`` in ``mp300k_artifact_gates`` (``required
- present`` and ``forbidden & present``) -- but that gate only runs deep
in the slow artifact path. This module surfaces the same check as a
one-line local command and the first, cheap CI job. It also reuses that
module's ``_h5_top_level_columns`` helper for H5 parsing so the two cannot
read columns differently.

The contract (``ecps_export_contract.json``) defines three categories:

- ``required`` -- columns MP must export to be a drop-in eCPS replacement.
- ``ecps_internal_optional`` -- eCPS clone-bookkeeping columns MP need not
  export (neither required nor forbidden).
- ``forbidden`` -- transient takeup-input columns eCPS drops and MP must
  not export.

Heavy imports (``h5py``, via the gate helper) are deferred so importing
this module and running the ``--columns-json`` path stay cheap.

Usage::

    python -m microplex_us.pipelines.check_export_columns export.h5
    python -m microplex_us.pipelines.check_export_columns \\
        --columns-json columns.json
    python -m microplex_us.pipelines.check_export_columns export.h5 \\
        --contract custom_contract.json

Exits 1 if any required column is missing or any forbidden column is
present; exits 0 otherwise.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path

# Path to the committed contract shipped alongside this module.
DEFAULT_CONTRACT_PATH = Path(__file__).with_name("ecps_export_contract.json")


@dataclass
class ColumnDiff:
    """Result of comparing a present column set against a contract."""

    missing_required: list[str]
    forbidden_present: list[str]
    extra_unknown: list[str]

    @property
    def ok(self) -> bool:
        """True when no required column is missing and none forbidden."""
        return not self.missing_required and not self.forbidden_present


def compute_column_diff(
    present: set[str],
    *,
    required: set[str],
    forbidden: set[str],
    optional: frozenset[str] | set[str] = frozenset(),
    excluded: frozenset[str] | set[str] = frozenset(),
) -> ColumnDiff:
    """Compare a present column set against contract categories.

    Mirrors the required/forbidden diff in ``_column_contract_gate`` in
    ``mp300k_artifact_gates`` (``required - present`` and ``forbidden &
    present``). ``optional`` (clone-bookkeeping flags) and ``excluded``
    (formula-owned columns MP need not export) are recognized categories, so
    they never appear in ``extra_unknown``. ``extra_unknown`` is informational
    only: columns present that are in no known category.
    """
    missing_required = required - present
    forbidden_present = forbidden & present
    known = required | forbidden | set(optional) | set(excluded)
    extra_unknown = present - known
    return ColumnDiff(
        missing_required=sorted(missing_required),
        forbidden_present=sorted(forbidden_present),
        extra_unknown=sorted(extra_unknown),
    )


def load_contract(path: Path) -> dict:
    """Load and validate the column-parity contract JSON."""
    with open(path) as f:
        contract = json.load(f)
    for key in ("required", "forbidden"):
        if key not in contract:
            raise ValueError(f"Contract {path} is missing required key '{key}'.")
    contract.setdefault("ecps_internal_optional", [])
    contract.setdefault("formula_owned_excluded", [])
    return contract


def _gate_h5_top_level_columns():
    """Return the artifact gate's ``_h5_top_level_columns`` helper.

    Loaded from the sibling module *by file path* (not by package import)
    so neither importing this module nor running it as a script pulls the
    heavy ``microplex_us`` package ``__init__`` (and ``microplex`` /
    torch). ``h5py`` is imported only as a side effect of executing the
    gate module here, on the H5 path. The module is registered in
    ``sys.modules`` before execution so its dataclasses resolve.
    """
    import importlib.util

    module_name = "_mp300k_artifact_gates_for_columns"
    cached = sys.modules.get(module_name)
    if cached is not None:
        return cached._h5_top_level_columns

    gate_path = Path(__file__).with_name("mp300k_artifact_gates.py")
    spec = importlib.util.spec_from_file_location(module_name, gate_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module._h5_top_level_columns


def _columns_from_h5(h5_path: Path) -> set[str]:
    """Return top-level base column names from an exported H5.

    Reuses the artifact-gate helper so H5 parsing stays identical. Columns
    may be datasets named ``<column>`` or groups ``<column>/<period>``;
    both collapse to the base name.
    """
    return _gate_h5_top_level_columns()(h5_path)


def _columns_from_json(json_path: Path) -> set[str]:
    """Return base column names from a JSON list (no data file needed)."""
    with open(json_path) as f:
        names = json.load(f)
    if not isinstance(names, list):
        raise ValueError(
            f"--columns-json {json_path} must contain a JSON list of column names."
        )
    return {str(name).split("/")[0] for name in names}


def _bullet_lines(items: list[str]) -> list[str]:
    """Render a list as indented bullets, or a placeholder if empty."""
    if not items:
        return ["    (none)"]
    return [f"    - {item}" for item in items]


def _format_report(
    diff: ColumnDiff,
    *,
    source: str,
    n_present: int,
    n_required: int,
    n_forbidden: int,
) -> str:
    """Build a human-readable report for the diff."""
    lines = [
        "eCPS column-parity check",
        f"  source:               {source}",
        f"  columns present:      {n_present}",
        f"  required (contract):  {n_required}",
        f"  forbidden (contract): {n_forbidden}",
        "",
        f"  missing_required ({len(diff.missing_required)}):",
        *_bullet_lines(diff.missing_required),
        f"  forbidden_present ({len(diff.forbidden_present)}):",
        *_bullet_lines(diff.forbidden_present),
        f"  extra_unknown (informational, {len(diff.extra_unknown)}):",
        *_bullet_lines(diff.extra_unknown),
        "",
        "  RESULT: " + ("PASS" if diff.ok else "FAIL"),
    ]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    """Run the column-parity check; return the process exit code."""
    parser = argparse.ArgumentParser(
        prog="check_export_columns",
        description=(
            "Fast eCPS column-parity check: compare a candidate export's "
            "columns to the frozen eCPS contract. Produces no data."
        ),
    )
    parser.add_argument(
        "h5path",
        nargs="?",
        help="Path to an exported H5 whose columns are checked.",
    )
    parser.add_argument(
        "--columns-json",
        metavar="FILE",
        help=(
            "Path to a JSON list of column names to check instead of an "
            "H5 (the no-data CI path). Mutually exclusive with h5path."
        ),
    )
    parser.add_argument(
        "--contract",
        metavar="FILE",
        default=str(DEFAULT_CONTRACT_PATH),
        help="Override the contract JSON (default: committed contract).",
    )
    args = parser.parse_args(argv)

    if bool(args.h5path) == bool(args.columns_json):
        parser.error("provide exactly one of an H5 path or --columns-json.")

    contract = load_contract(Path(args.contract))
    required = set(contract["required"])
    forbidden = set(contract["forbidden"])
    optional = set(contract["ecps_internal_optional"])
    excluded = set(contract.get("formula_owned_excluded", []))

    if args.columns_json:
        source = args.columns_json
        present = _columns_from_json(Path(args.columns_json))
    else:
        source = args.h5path
        present = _columns_from_h5(Path(args.h5path))

    diff = compute_column_diff(
        present,
        required=required,
        forbidden=forbidden,
        optional=optional,
        excluded=excluded,
    )
    print(
        _format_report(
            diff,
            source=source,
            n_present=len(present),
            n_required=len(required),
            n_forbidden=len(forbidden),
        )
    )
    return 0 if diff.ok else 1


if __name__ == "__main__":
    sys.exit(main())
