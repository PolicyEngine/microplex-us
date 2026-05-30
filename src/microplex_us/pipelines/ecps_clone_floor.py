"""Clone-floor baseline gate for the eCPS-replacement eval.

Before the CI eval spends ~20-30 minutes benchmarking a candidate against a
baseline Enhanced CPS, it must confirm the baseline is actually worth measuring
against. A degraded baseline - one whose clone (PUF-support / donor-replay)
records have collapsed back toward the bare CPS-ASEC weights - makes any
``candidate_beats_baseline`` result meaningless: the candidate is then "beating"
a broken artifact, not a real eCPS.

This module reads the baseline's ``enhanced_cps_2024.clone_diagnostics.json``
sidecar and decides whether the clone household-weight share clears a floor
(default 5%). It is the single, unit-tested decision point the workflow shells
out to, so the gate logic is testable without running the heavy comparison.

The sidecar schema is intentionally tolerant. It accepts, in priority order:

1. A top-level ``clone_household_weight_share`` (or any of the equivalent keys
   the repo already uses elsewhere, e.g. ``mp300k_artifact_gates`` reads
   ``clone_household_weight_share`` / ``puf_clone_household_weight_share`` /
   ``support_household_weight_share``).
2. The same keys nested under a ``summary`` object.
3. A ``sources`` list (or dict) of per-source rows; the clone share is then the
   summed ``household_weight_share`` of every row whose ``source_class`` /
   ``source_name`` marks it as a clone/support/donor-replay source (matching
   the existing convention in ``mp300k_artifact_gates``).

Missing or malformed sidecars fail the gate closed: an unverifiable baseline is
treated exactly like a degraded one, never silently benchmarked.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

# Default minimum acceptable clone household-weight share. Below this, the
# baseline eCPS has decayed toward the un-enhanced CPS and must not be used as a
# benchmark target.
DEFAULT_CLONE_FLOOR = 0.05

# Keys that, anywhere we look, carry a precomputed clone/support weight share.
# Ordered by specificity; the first present numeric value wins.
_CLONE_SHARE_KEYS = (
    "clone_household_weight_share",
    "puf_clone_household_weight_share",
    "support_household_weight_share",
    "puf_support_household_weight_share",
    "clone_weight_share",
    "support_weight_share",
)

# Per-source-row keys that hold that row's household weight share.
_ENTRY_SHARE_KEYS = ("household_weight_share", "weight_share", "share")

# Tokens that mark a source row as a clone / support / donor-replay source.
_CLONE_SOURCE_TOKENS = ("clone", "support", "donor_replay")

# Tokens that explicitly mark a source row as NOT a clone (a real base source).
_NON_CLONE_SOURCE_TOKENS = ("fixed", "forbes")


@dataclass(frozen=True)
class CloneFloorGateResult:
    """Outcome of the clone-floor baseline gate.

    Attributes
    ----------
    passed:
        ``True`` only when the baseline is trustworthy enough to benchmark
        against: a readable sidecar whose clone weight share meets the floor.
    message:
        Human-readable explanation - loud and specific on failure so CI logs
        make the reason obvious.
    clone_weight_share:
        The observed clone household-weight share, or ``None`` when it could
        not be read.
    floor:
        The floor the share was checked against.
    """

    passed: bool
    message: str
    clone_weight_share: float | None
    floor: float


def _coerce_float(value: object) -> float | None:
    """Return ``value`` as a float, or ``None`` if it is not finite/numeric."""
    if value is None or isinstance(value, bool):
        return None
    try:
        result = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None
    if result != result or result in (float("inf"), float("-inf")):
        return None
    return result


def _first_share_in_mapping(mapping: dict[str, object]) -> float | None:
    """Return the first recognized precomputed clone-share value in a mapping."""
    for key in _CLONE_SHARE_KEYS:
        if key in mapping:
            share = _coerce_float(mapping[key])
            if share is not None:
                return share
    return None


def _source_rows(payload: dict[str, object]) -> list[dict[str, object]]:
    """Extract per-source rows from any of the supported container keys."""
    for key in ("sources", "source_classes", "source_weight_shares"):
        value = payload.get(key)
        if isinstance(value, list):
            return [row for row in value if isinstance(row, dict)]
        if isinstance(value, dict):
            rows: list[dict[str, object]] = []
            for name, row in value.items():
                if isinstance(row, dict):
                    rows.append({"source_name": name, **row})
            return rows
    return []


def _row_is_clone(row: dict[str, object]) -> bool:
    """Return ``True`` if a source row represents a clone/support source."""
    source_class = str(
        row.get("source_class")
        or row.get("class")
        or row.get("kind")
        or row.get("category")
        or ""
    ).lower()
    source_name = str(row.get("source_name") or row.get("name") or "").lower()
    haystack = f"{source_class} {source_name}"
    if any(token in haystack for token in _NON_CLONE_SOURCE_TOKENS):
        return False
    return any(token in haystack for token in _CLONE_SOURCE_TOKENS)


def _row_share(row: dict[str, object]) -> float | None:
    """Return a source row's household weight share, if present."""
    for key in _ENTRY_SHARE_KEYS:
        if key in row:
            share = _coerce_float(row[key])
            if share is not None:
                return share
    return None


def extract_clone_weight_share(payload: dict[str, object]) -> float | None:
    """Extract the clone household-weight share from a diagnostics payload.

    Tries, in order: top-level precomputed keys, the same keys nested under a
    ``summary`` object, then summing the shares of clone source rows.

    Returns ``None`` if no clone share can be determined (the caller then fails
    the gate closed).
    """
    direct = _first_share_in_mapping(payload)
    if direct is not None:
        return direct

    summary = payload.get("summary")
    if isinstance(summary, dict):
        nested = _first_share_in_mapping(summary)
        if nested is not None:
            return nested

    rows = _source_rows(payload)
    if rows:
        clone_shares = [
            share
            for row in rows
            if _row_is_clone(row) and (share := _row_share(row)) is not None
        ]
        if clone_shares:
            return float(sum(clone_shares))

    return None


def load_clone_diagnostics(path: Path) -> dict[str, object]:
    """Load and parse a clone-diagnostics JSON sidecar.

    Raises
    ------
    FileNotFoundError
        If the sidecar does not exist.
    ValueError
        If the sidecar is not valid JSON or is not a JSON object.
    """
    if not path.exists():
        raise FileNotFoundError(path)
    try:
        payload = json.loads(path.read_text())
    except json.JSONDecodeError as exc:
        raise ValueError(f"{path} is not valid JSON: {exc}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return payload


def evaluate_clone_floor_gate(
    path: Path,
    floor: float = DEFAULT_CLONE_FLOOR,
) -> CloneFloorGateResult:
    """Evaluate the clone-floor baseline gate for a diagnostics sidecar.

    Folds together the three cases CI must distinguish:

    * **Healthy** - the sidecar exists, parses, exposes a clone share, and that
      share meets ``floor``. The gate passes.
    * **Degraded** - the sidecar exists and exposes a clone share, but it has
      fallen below ``floor``. The baseline has decayed back toward the bare CPS
      (see #113); the gate fails *loudly*.
    * **Missing or malformed** - the sidecar is absent, unparseable, or exposes
      no clone share. We cannot prove the baseline is healthy, so we *fail
      closed*: an unverifiable baseline is treated exactly like a bad one.

    Parameters
    ----------
    path:
        Path to the baseline's ``*.clone_diagnostics.json`` sidecar.
    floor:
        Minimum acceptable clone household-weight share (default 5%).

    Returns
    -------
    CloneFloorGateResult
        Structured outcome with a loud message on failure.
    """
    try:
        payload = load_clone_diagnostics(path)
    except FileNotFoundError:
        return CloneFloorGateResult(
            passed=False,
            message=(
                f"baseline eCPS clone diagnostics not found at {path} - "
                "refusing to benchmark against an unverifiable baseline "
                "(fail closed; see #113)"
            ),
            clone_weight_share=None,
            floor=floor,
        )
    except ValueError as exc:
        return CloneFloorGateResult(
            passed=False,
            message=(
                f"baseline eCPS clone diagnostics at {path} are malformed "
                f"({exc}) - refusing to benchmark against an unverifiable "
                "baseline (fail closed; see #113)"
            ),
            clone_weight_share=None,
            floor=floor,
        )

    share = extract_clone_weight_share(payload)
    if share is None:
        return CloneFloorGateResult(
            passed=False,
            message=(
                f"baseline eCPS clone diagnostics at {path} expose no clone "
                "household-weight share - refusing to benchmark against an "
                "unverifiable baseline (fail closed; see #113)"
            ),
            clone_weight_share=None,
            floor=floor,
        )

    if share >= floor:
        return CloneFloorGateResult(
            passed=True,
            message=(
                f"baseline eCPS clone share {share:.1%} >= {floor:.1%} floor "
                "- baseline is healthy"
            ),
            clone_weight_share=share,
            floor=floor,
        )

    return CloneFloorGateResult(
        passed=False,
        message=(
            f"baseline eCPS degraded; clone share {share:.1%} < {floor:.1%} "
            "floor - refusing to benchmark against a bad baseline (see #113)"
        ),
        clone_weight_share=share,
        floor=floor,
    )


__all__ = [
    "DEFAULT_CLONE_FLOOR",
    "CloneFloorGateResult",
    "evaluate_clone_floor_gate",
    "extract_clone_weight_share",
    "load_clone_diagnostics",
]
