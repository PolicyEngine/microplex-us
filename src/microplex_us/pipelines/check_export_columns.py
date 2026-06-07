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
one-line local command and the first, cheap CI job.

The contract (``ecps_export_contract.json``) defines three categories:

- ``required`` -- columns MP must export to be a drop-in eCPS replacement.
- ``ecps_internal_optional`` -- eCPS clone-bookkeeping columns MP need not
  export (neither required nor forbidden).
- ``forbidden`` -- transient takeup-input columns eCPS drops and MP must
  not export.

Heavy imports (``h5py``) are deferred so importing this module and running
the ``--columns-json`` path stay cheap.

Usage::

    python -m microplex_us.pipelines.check_export_columns export.h5
    python -m microplex_us.pipelines.check_export_columns \\
        --columns-json columns.json
    python -m microplex_us.pipelines.check_export_columns \\
        --entity-tables checkpoints/post-imputation
    python -m microplex_us.pipelines.check_export_columns export.h5 \\
        --contract custom_contract.json

Exits 1 if any required column is missing or any forbidden column is
present; exits 0 otherwise.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

# Path to the committed contract shipped alongside this module.
DEFAULT_CONTRACT_PATH = Path(__file__).with_name("ecps_export_contract.json")
DEFAULT_SPEC_PATH = Path(__file__).resolve().parents[1] / "specs" / "us-2024.yaml"

SIGNED_NUMERIC_SUPPORT_COLUMNS = frozenset(
    {
        "farm_income",
        "farm_operations_income",
        "partnership_s_corp_income",
        "rental_income",
        "self_employment_income_before_lsr",
    }
)


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


@dataclass
class ColumnSupportStats:
    """Compact support/variation summary for one exported H5 column."""

    column: str
    kind: str
    row_count: int
    nonzero_count: int | None
    positive_count: int | None
    negative_count: int | None
    unique_count: int


@dataclass
class ColumnSupportIssue:
    """One eCPS-populated column missing equivalent MP support."""

    column: str
    requirement: str
    baseline: ColumnSupportStats
    candidate: ColumnSupportStats | None


@dataclass
class SupportDiff:
    """Result of comparing candidate support against eCPS support."""

    issues: list[ColumnSupportIssue]
    checked_columns: list[str]
    baseline_populated_columns: list[str]
    baseline_filler_columns: list[str]
    exempt_columns: list[str]

    @property
    def ok(self) -> bool:
        """True when every eCPS-populated column has candidate support."""
        return not self.issues


@dataclass
class SpecVariableManifestDiff:
    """Result of checking ``spec.variables`` against the frozen contract."""

    spec_path: str
    required_contract_count: int
    declared_imputation_count: int
    variable_manifest_count: int
    missing_required: list[str]
    missing_declared_imputation: list[str]
    extra_variables: list[str]

    @property
    def ok(self) -> bool:
        """True when the manifest exactly covers required and declared vars."""
        return not (
            self.missing_required
            or self.missing_declared_imputation
            or self.extra_variables
        )


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


def compute_support_diff(
    candidate_h5: Path,
    *,
    baseline_h5: Path,
    period: int,
    required_columns: set[str],
    exempt_columns: frozenset[str] | set[str] = frozenset(),
) -> SupportDiff:
    """Compare candidate support against eCPS support for required columns.

    Presence is not enough for release parity. If the pinned eCPS baseline
    *populates* a required exported column, MP must populate it too:

    - numeric columns: eCPS has at least one nonzero value, so MP must also
      have at least one nonzero value. Declared signed-income exports must
      also preserve positive/negative support when eCPS has it;
    - boolean/string/categorical columns: eCPS has more than one unique value,
      so MP must also vary.

    Columns where eCPS itself is all-zero/single-valued are treated as fillers
    and do not require MP support. Explicit exemptions are reserved for known
    rare, computed-downstream, or intentionally absent variables.
    """
    period_key = str(int(period))
    exempt = {str(column) for column in exempt_columns}
    checked_columns: list[str] = []
    baseline_populated_columns: list[str] = []
    baseline_filler_columns: list[str] = []
    issues: list[ColumnSupportIssue] = []

    import h5py

    with (
        h5py.File(candidate_h5, "r") as candidate,
        h5py.File(baseline_h5, "r") as baseline,
    ):
        for column in sorted(required_columns):
            if column in exempt:
                continue
            baseline_values = _h5_column_values(
                baseline,
                column,
                period_key=period_key,
            )
            if baseline_values is None:
                continue
            checked_columns.append(column)
            baseline_stats = _support_stats(column, baseline_values)
            requirement = _support_requirement(
                baseline_stats,
                require_signed_numeric=column in SIGNED_NUMERIC_SUPPORT_COLUMNS,
            )
            if requirement is None:
                baseline_filler_columns.append(column)
                continue
            baseline_populated_columns.append(column)
            candidate_values = _h5_column_values(
                candidate,
                column,
                period_key=period_key,
            )
            candidate_stats = (
                None
                if candidate_values is None
                else _support_stats(column, candidate_values)
            )
            if not _satisfies_support_requirement(
                candidate_stats,
                requirement=requirement,
            ):
                issues.append(
                    ColumnSupportIssue(
                        column=column,
                        requirement=requirement,
                        baseline=baseline_stats,
                        candidate=candidate_stats,
                    )
                )

    return SupportDiff(
        issues=issues,
        checked_columns=checked_columns,
        baseline_populated_columns=baseline_populated_columns,
        baseline_filler_columns=baseline_filler_columns,
        exempt_columns=sorted(exempt & set(required_columns)),
    )


def compute_spec_variable_manifest_diff(
    *,
    contract: dict,
    spec_path: Path = DEFAULT_SPEC_PATH,
) -> SpecVariableManifestDiff:
    """Compare ``spec.variables`` with required exports and declared imputations."""
    text = spec_path.read_text(encoding="utf-8")
    variables = _parse_top_level_mapping_keys(text, "variables")
    if not variables:
        raise ValueError(f"Spec {spec_path} is missing a variables mapping.")

    required = {str(column) for column in contract["required"]}
    declared_imputation = _parse_imputation_vars(text)
    expected = required | declared_imputation
    return SpecVariableManifestDiff(
        spec_path=str(spec_path),
        required_contract_count=len(required),
        declared_imputation_count=len(declared_imputation),
        variable_manifest_count=len(variables),
        missing_required=sorted(required - variables),
        missing_declared_imputation=sorted(declared_imputation - variables),
        extra_variables=sorted(variables - expected),
    )


def _top_level_section_lines(text: str, section: str) -> list[str]:
    """Return lines in a simple top-level YAML section.

    This module is intentionally importable with only the column-parity
    job's minimal dependencies, so the fast manifest gate avoids PyYAML.
    The parser only needs the committed spec's shape: top-level sections,
    mapping keys under ``variables:``, and imputation ``vars`` lists.
    """
    section_header = f"{section}:"
    lines = text.splitlines()
    for index, line in enumerate(lines):
        if line.strip() == section_header and not line.startswith((" ", "\t")):
            body: list[str] = []
            for candidate in lines[index + 1 :]:
                stripped = candidate.strip()
                if (
                    stripped
                    and not candidate.startswith((" ", "\t"))
                    and re.match(r"^[A-Za-z_][A-Za-z0-9_-]*:", stripped)
                ):
                    break
                body.append(candidate)
            return body
    return []


def _parse_top_level_mapping_keys(text: str, section: str) -> set[str]:
    """Parse direct mapping keys from a top-level section."""
    keys: set[str] = set()
    for line in _top_level_section_lines(text, section):
        match = re.match(r"^  ([A-Za-z_][A-Za-z0-9_]*):(?:\s|$)", line)
        if match:
            keys.add(match.group(1))
    return keys


def _parse_inline_list(raw: str) -> list[str]:
    """Parse the simple YAML inline list form used in tests."""
    stripped = raw.strip()
    if not stripped.startswith("[") or not stripped.endswith("]"):
        return []
    body = stripped[1:-1].strip()
    if not body:
        return []
    return [
        parsed
        for item in body.split(",")
        if (parsed := _parse_simple_yaml_scalar(item)) is not None
    ]


def _parse_simple_yaml_scalar(raw: str) -> str | None:
    """Parse a simple YAML scalar variable name with optional inline comment."""
    value = raw.strip()
    quote: str | None = None
    unquoted = []
    for index, char in enumerate(value):
        if char in {"'", '"'}:
            if quote is None:
                quote = char
            elif quote == char:
                quote = None
        if char == "#" and quote is None and (index == 0 or value[index - 1].isspace()):
            break
        unquoted.append(char)
    value = "".join(unquoted).strip()
    if (
        len(value) >= 2
        and value[0] == value[-1]
        and value[0] in {"'", '"'}
    ):
        value = value[1:-1].strip()
    if re.match(r"^[A-Za-z_][A-Za-z0-9_]*$", value):
        return value
    return None


def _parse_imputation_vars(text: str) -> set[str]:
    """Parse variable names from imputation step ``vars`` lists."""
    variables: set[str] = set()
    in_vars_block = False
    for line in _top_level_section_lines(text, "imputation"):
        if re.match(r"^  -\s", line):
            in_vars_block = False

        inline_match = re.match(r"^    vars:\s*(\[.*\])(?:\s+#.*)?$", line)
        if inline_match:
            variables.update(_parse_inline_list(inline_match.group(1)))
            in_vars_block = False
            continue

        if re.match(r"^    vars:\s*$", line):
            in_vars_block = True
            continue

        if in_vars_block:
            item_match = re.match(r"^      -\s+(.+?)\s*$", line)
            if item_match:
                if parsed := _parse_simple_yaml_scalar(item_match.group(1)):
                    variables.add(parsed)
                continue
            if re.match(r"^    [A-Za-z_][A-Za-z0-9_-]*:", line):
                in_vars_block = False

    return variables


def _h5_column_values(
    handle: Any,
    column: str,
    *,
    period_key: str,
):
    """Return one H5 column's values, supporting grouped and flat layouts."""
    if column not in handle:
        return None
    item = handle[column]
    import h5py
    import numpy as np

    if isinstance(item, h5py.Group):
        if period_key not in item:
            return None
        item = item[period_key]
    if not isinstance(item, h5py.Dataset):
        return None
    return np.asarray(item)


def _support_stats(column: str, values) -> ColumnSupportStats:
    """Summarize nonzero support and uniqueness for an exported column."""
    import numpy as np

    array = np.asarray(values)
    flattened = array.reshape(-1)
    unique_count = int(len(np.unique(flattened))) if flattened.size else 0
    kind = _support_kind(flattened)
    nonzero_count: int | None = None
    positive_count: int | None = None
    negative_count: int | None = None
    if kind == "numeric":
        numeric = flattened
        if np.issubdtype(numeric.dtype, np.floating):
            numeric = numeric[np.isfinite(numeric)]
        nonzero_count = int(np.count_nonzero(numeric))
        positive_count = int(np.count_nonzero(numeric > 0))
        negative_count = int(np.count_nonzero(numeric < 0))
    return ColumnSupportStats(
        column=column,
        kind=kind,
        row_count=int(flattened.size),
        nonzero_count=nonzero_count,
        positive_count=positive_count,
        negative_count=negative_count,
        unique_count=unique_count,
    )


def _support_kind(values) -> str:
    """Classify a NumPy array for support checking."""
    import numpy as np

    dtype = np.asarray(values).dtype
    if np.issubdtype(dtype, np.bool_):
        return "categorical"
    if np.issubdtype(dtype, np.number):
        return "numeric"
    return "categorical"


def _support_requirement(
    stats: ColumnSupportStats,
    *,
    require_signed_numeric: bool = True,
) -> str | None:
    """Return the support MP must match for an eCPS column, if any."""
    if stats.kind == "numeric":
        if (stats.nonzero_count or 0) <= 0:
            return None
        has_positive = (stats.positive_count or 0) > 0
        has_negative = (stats.negative_count or 0) > 0
        if require_signed_numeric and has_positive and has_negative:
            return "numeric_signed"
        if has_positive:
            return "numeric_positive"
        if has_negative:
            return "numeric_negative"
        return "numeric_nonzero"
    return "categorical_variation" if stats.unique_count > 1 else None


def _satisfies_support_requirement(
    stats: ColumnSupportStats | None,
    *,
    requirement: str,
) -> bool:
    """Return whether candidate stats meet an eCPS-derived requirement."""
    if stats is None:
        return False
    if requirement in {
        "numeric_nonzero",
        "numeric_positive",
        "numeric_negative",
        "numeric_signed",
    }:
        if stats.kind != "numeric":
            return stats.unique_count > 1
        if requirement == "numeric_nonzero":
            return (stats.nonzero_count or 0) > 0
        if requirement == "numeric_positive":
            return (stats.positive_count or 0) > 0
        if requirement == "numeric_negative":
            return (stats.negative_count or 0) > 0
        return (stats.positive_count or 0) > 0 and (stats.negative_count or 0) > 0
    if requirement == "categorical_variation":
        return stats.unique_count > 1
    raise ValueError(f"Unknown support requirement: {requirement}")


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


def _columns_from_h5(h5_path: Path) -> set[str]:
    """Return top-level base column names from an exported H5.

    Columns may be datasets named ``<column>`` or groups ``<column>/<period>``;
    both collapse to the base name. This intentionally duplicates the tiny
    parser used by the artifact gate so the fast column CI can run without
    importing the full Microplex stack.
    """
    import h5py

    with h5py.File(h5_path, "r") as handle:
        return {name.split("/")[0] for name in handle.keys()}


def _columns_from_json(json_path: Path) -> set[str]:
    """Return base column names from a JSON list (no data file needed)."""
    with open(json_path) as f:
        names = json.load(f)
    if not isinstance(names, list):
        raise ValueError(
            f"--columns-json {json_path} must contain a JSON list of column names."
        )
    return {str(name).split("/")[0] for name in names}


def _columns_from_entity_tables(
    entity_tables_path: Path,
    *,
    direct_override_variables: tuple[str, ...] = (),
) -> set[str]:
    """Return export column names from a saved PE entity-table checkpoint.

    This is the pre-calibration path: post-imputation entity tables already
    determine the final H5 schema, while calibration only changes weights.
    Imports stay deferred so the JSON/H5 fast paths do not import Microplex.
    """
    from microplex_us.policyengine.us import (
        build_policyengine_us_export_column_names,
        load_us_pipeline_checkpoint,
    )

    tables, _metadata = load_us_pipeline_checkpoint(entity_tables_path)
    return build_policyengine_us_export_column_names(
        tables,
        direct_override_variables=direct_override_variables,
    )


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
    support_diff: SupportDiff | None = None,
    spec_diff: SpecVariableManifestDiff | None = None,
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
    ]
    if support_diff is not None:
        lines.extend(
            [
                "",
                "  eCPS support parity:",
                f"    checked_columns:             {len(support_diff.checked_columns)}",
                f"    eCPS-populated columns:      {len(support_diff.baseline_populated_columns)}",
                f"    eCPS filler columns:         {len(support_diff.baseline_filler_columns)}",
                f"    explicit support exemptions: {len(support_diff.exempt_columns)}",
                f"    unsupported_populated ({len(support_diff.issues)}):",
                *_bullet_lines(
                    [
                        f"{issue.column} ({issue.requirement}; "
                        f"eCPS={_compact_stats(issue.baseline)}, "
                        f"candidate={_compact_stats(issue.candidate)})"
                        for issue in support_diff.issues
                    ]
                ),
            ]
        )
    if spec_diff is not None:
        lines.extend(
            [
                "",
                "  spec variable manifest:",
                f"    spec:                         {spec_diff.spec_path}",
                f"    required contract variables:  {spec_diff.required_contract_count}",
                f"    declared imputation variables: {spec_diff.declared_imputation_count}",
                f"    spec.variables count:         {spec_diff.variable_manifest_count}",
                f"    missing_required ({len(spec_diff.missing_required)}):",
                *_bullet_lines(spec_diff.missing_required),
                "    missing_declared_imputation "
                f"({len(spec_diff.missing_declared_imputation)}):",
                *_bullet_lines(spec_diff.missing_declared_imputation),
                f"    extra_variables ({len(spec_diff.extra_variables)}):",
                *_bullet_lines(spec_diff.extra_variables),
            ]
        )
    ok = (
        diff.ok
        and (support_diff is None or support_diff.ok)
        and (spec_diff is None or spec_diff.ok)
    )
    lines.extend(["", "  RESULT: " + ("PASS" if ok else "FAIL")])
    return "\n".join(lines)


def _compact_stats(stats: ColumnSupportStats | None) -> str:
    """Render support stats compactly for CLI output."""
    if stats is None:
        return "missing"
    if stats.kind == "numeric":
        return (
            f"nonzero {stats.nonzero_count}/{stats.row_count}; "
            f"+{stats.positive_count}, -{stats.negative_count}"
        )
    return f"unique {stats.unique_count}/{stats.row_count}"


def support_diff_to_dict(diff: SupportDiff) -> dict[str, Any]:
    """Return a JSON-serializable support parity payload."""
    payload = asdict(diff)
    return payload


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
        "--entity-tables",
        metavar="DIR",
        help=(
            "Path to a saved PolicyEngine entity-table checkpoint/stage "
            "directory (for example checkpoints/post-imputation). Checks "
            "the export schema before microsimulation/calibration/H5."
        ),
    )
    parser.add_argument(
        "--direct-override-variable",
        action="append",
        default=[],
        metavar="VARIABLE",
        help=(
            "PolicyEngine formula variable intentionally exported from source "
            "data. Repeat for each override used by the build."
        ),
    )
    parser.add_argument(
        "--contract",
        metavar="FILE",
        default=str(DEFAULT_CONTRACT_PATH),
        help="Override the contract JSON (default: committed contract).",
    )
    parser.add_argument(
        "--spec",
        metavar="FILE",
        help=(
            "Spec YAML whose variables block must cover the contract and "
            "declared imputation vars. Defaults to the committed US spec when "
            "using the committed contract."
        ),
    )
    parser.add_argument(
        "--skip-spec-variable-manifest",
        action="store_true",
        help="Skip the spec.variables manifest coverage check.",
    )
    parser.add_argument(
        "--support-baseline",
        metavar="H5",
        help=(
            "Pinned eCPS baseline H5. When supplied with an H5 candidate, "
            "also fail if eCPS has nonzero/variant support for a required "
            "exported column and the candidate is all-zero/constant."
        ),
    )
    parser.add_argument(
        "--period",
        type=int,
        default=2024,
        help="Tax year period to inspect for H5 support parity (default: 2024).",
    )
    parser.add_argument(
        "--support-exempt-column",
        action="append",
        default=[],
        metavar="COLUMN",
        help=(
            "Required export column exempt from support parity because it is "
            "declared rare, computed downstream, or intentionally absent. "
            "Repeat for each explicit exception."
        ),
    )
    parser.add_argument(
        "--support-diagnostics-json",
        metavar="FILE",
        help="Optional path to write support-parity diagnostics JSON.",
    )
    args = parser.parse_args(argv)

    selected_inputs = [
        bool(args.h5path),
        bool(args.columns_json),
        bool(args.entity_tables),
    ]
    if sum(selected_inputs) != 1:
        parser.error(
            "provide exactly one of an H5 path, --columns-json, or --entity-tables."
        )
    if args.support_baseline and not args.h5path:
        parser.error("--support-baseline requires an H5 candidate path.")

    contract = load_contract(Path(args.contract))
    required = set(contract["required"])
    forbidden = set(contract["forbidden"])
    optional = set(contract["ecps_internal_optional"])
    excluded = set(contract.get("formula_owned_excluded", []))

    if args.columns_json:
        source = args.columns_json
        present = _columns_from_json(Path(args.columns_json))
    elif args.entity_tables:
        source = args.entity_tables
        present = _columns_from_entity_tables(
            Path(args.entity_tables),
            direct_override_variables=tuple(args.direct_override_variable),
        )
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
    contract_path = Path(args.contract).resolve()
    spec_path = None
    if not args.skip_spec_variable_manifest:
        if args.spec:
            spec_path = Path(args.spec)
        elif contract_path == DEFAULT_CONTRACT_PATH.resolve():
            spec_path = DEFAULT_SPEC_PATH
    spec_diff = (
        None
        if spec_path is None
        else compute_spec_variable_manifest_diff(
            contract=contract,
            spec_path=Path(spec_path),
        )
    )
    support_diff = None
    if args.support_baseline:
        support_exempt = set(contract.get("support_exemptions", [])) | set(
            args.support_exempt_column
        )
        support_diff = compute_support_diff(
            Path(args.h5path),
            baseline_h5=Path(args.support_baseline),
            period=int(args.period),
            required_columns=required,
            exempt_columns=support_exempt,
        )
        if args.support_diagnostics_json:
            output_path = Path(args.support_diagnostics_json)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_text(
                json.dumps(support_diff_to_dict(support_diff), indent=2) + "\n"
            )
    print(
        _format_report(
            diff,
            source=source,
            n_present=len(present),
            n_required=len(required),
            n_forbidden=len(forbidden),
            support_diff=support_diff,
            spec_diff=spec_diff,
        )
    )
    return (
        0
        if (
            diff.ok
            and (support_diff is None or support_diff.ok)
            and (spec_diff is None or spec_diff.ok)
        )
        else 1
    )


if __name__ == "__main__":
    sys.exit(main())
