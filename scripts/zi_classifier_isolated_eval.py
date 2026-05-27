"""Isolated per-column ZI classifier evaluation.

Answers the diagnostic question behind the 5-way ZI-QDNN coverage tie: if we
strip the downstream draw network out of the loop and evaluate only the
zero/non-zero classifier's own calibration and discrimination, do the five
candidates still look equivalent?

Protocol
--------

- Same data as the coverage benchmark: enhanced_cps_2024, 77,006 persons, 14
  conditioning columns, 36 target columns, seed 42.
- Same outer 80/20 train/holdout split used by ScaleUpRunner.
- For each target column with training-set zero-fraction >= 10% (the upstream
  ZI trigger) and at least 10 zero + 10 non-zero training rows, further split
  training 80/20 (seed 42) into fit / val.
- Label is (~at_min).astype(int), matching `_MultiSourceBase.fit`.
- Fit each of 5 classifiers on (X_fit, label_fit), predict P(y>0) on X_val.
- Report: log-loss, Brier, ECE (10 equal-width bins), ROC-AUC, fit seconds.

Aggregation
-----------

For each classifier, report column-count-weighted mean and median across the
eligible target columns. The RF default should be the baseline everything else
is compared against, since it is what the coverage benchmark locked in.
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score

from microplex_us.bakeoff.local_methods import (
    _dnn_factory,
    _hgb_factory,
    _logistic_factory,
    _rf_calibrated_factory,
)
from microplex_us.bakeoff.scale_up import (
    DEFAULT_CONDITION_COLS,
    DEFAULT_ENHANCED_CPS_PATH,
    DEFAULT_TARGET_COLS,
    _load_enhanced_cps,
)

LOGGER = logging.getLogger(__name__)


def _rf_default_factory():
    from sklearn.ensemble import RandomForestClassifier

    return RandomForestClassifier(n_estimators=50, random_state=42, n_jobs=-1)


CLASSIFIERS: dict[str, Callable[[], Any]] = {
    "RF_default": _rf_default_factory,
    "Logistic": _logistic_factory,
    "HistGB": _hgb_factory,
    "RF_calibrated": _rf_calibrated_factory,
    "DNN": _dnn_factory,
}


def _expected_calibration_error(
    y_true: np.ndarray, p_hat: np.ndarray, n_bins: int = 10
) -> float:
    """Equal-width ECE: sum over bins of (n_bin/N) * |acc - conf|."""
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    n = len(y_true)
    for i in range(n_bins):
        lo, hi = edges[i], edges[i + 1]
        if i == n_bins - 1:
            mask = (p_hat >= lo) & (p_hat <= hi)
        else:
            mask = (p_hat >= lo) & (p_hat < hi)
        if not mask.any():
            continue
        bin_conf = float(p_hat[mask].mean())
        bin_acc = float(y_true[mask].mean())
        ece += (mask.sum() / n) * abs(bin_conf - bin_acc)
    return float(ece)


def _positive_class_proba(clf: Any, X: np.ndarray) -> np.ndarray:
    """Return P(y == 1 | x) regardless of how the classifier orders classes."""
    proba = clf.predict_proba(X)
    classes = np.asarray(clf.classes_)
    pos_idx = int(np.where(classes == 1)[0][0])
    return proba[:, pos_idx]


def evaluate_column(
    col: str,
    X_fit: np.ndarray,
    y_fit_label: np.ndarray,
    X_val: np.ndarray,
    y_val_label: np.ndarray,
) -> dict[str, dict[str, float]]:
    """Fit every classifier on (X_fit, y_fit_label); score on val."""
    results: dict[str, dict[str, float]] = {}
    for name, factory in CLASSIFIERS.items():
        clf = factory()
        t0 = time.perf_counter()
        clf.fit(X_fit, y_fit_label)
        fit_s = time.perf_counter() - t0
        p_hat = _positive_class_proba(clf, X_val)
        p_hat = np.clip(p_hat, 1e-6, 1 - 1e-6)
        ll = float(log_loss(y_val_label, p_hat, labels=[0, 1]))
        brier = float(brier_score_loss(y_val_label, p_hat))
        ece = _expected_calibration_error(y_val_label, p_hat, n_bins=10)
        try:
            auc = float(roc_auc_score(y_val_label, p_hat))
        except ValueError:
            auc = float("nan")
        results[name] = {
            "log_loss": ll,
            "brier": brier,
            "ece": ece,
            "auc": auc,
            "fit_s": fit_s,
        }
    return results


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__ or "")
    parser.add_argument(
        "--data-path", type=Path, default=DEFAULT_ENHANCED_CPS_PATH
    )
    parser.add_argument("--year", default="2024")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--holdout-frac", type=float, default=0.2)
    parser.add_argument("--inner-val-frac", type=float, default=0.2)
    parser.add_argument("--zero-threshold", type=float, default=0.1)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "/Users/maxghenis/PolicyEngine/microplex-us/artifacts/"
            "zi_classifier_isolated_eval.json"
        ),
    )
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    columns = list(DEFAULT_CONDITION_COLS) + list(DEFAULT_TARGET_COLS)
    df = _load_enhanced_cps(args.data_path, args.year, columns)
    df = df.astype(np.float32)
    LOGGER.info("loaded %d rows x %d cols", len(df), len(df.columns))

    rng = np.random.default_rng(args.seed)
    idx = rng.permutation(len(df))
    cut = int(len(df) * (1.0 - args.holdout_frac))
    train = df.iloc[idx[:cut]].reset_index(drop=True)
    LOGGER.info("outer split: %d train rows (holdout discarded, not needed here)", len(train))

    inner_rng = np.random.default_rng(args.seed + 1)
    inner_idx = inner_rng.permutation(len(train))
    inner_cut = int(len(train) * (1.0 - args.inner_val_frac))
    fit_idx, val_idx = inner_idx[:inner_cut], inner_idx[inner_cut:]
    LOGGER.info("inner split: %d fit / %d val", len(fit_idx), len(val_idx))

    cond = list(DEFAULT_CONDITION_COLS)
    X_train_all = train[cond].to_numpy()
    X_fit_all = X_train_all[fit_idx]
    X_val_all = X_train_all[val_idx]

    per_col: dict[str, Any] = {}
    eligible: list[str] = []
    skipped: list[dict[str, Any]] = []

    for col in DEFAULT_TARGET_COLS:
        y = train[col].to_numpy()
        min_val = float(np.nanmin(y))
        at_min = np.isclose(y, min_val, atol=1e-6)
        zero_frac = float(at_min.mean())
        label = (~at_min).astype(int)

        fit_label = label[fit_idx]
        val_label = label[val_idx]
        n_zero_fit = int((fit_label == 0).sum())
        n_pos_fit = int((fit_label == 1).sum())
        n_zero_val = int((val_label == 0).sum())
        n_pos_val = int((val_label == 1).sum())

        if zero_frac < args.zero_threshold:
            skipped.append(
                {"col": col, "reason": "below_zero_threshold", "zero_frac": zero_frac}
            )
            continue
        if n_zero_fit < 10 or n_pos_fit < 10:
            skipped.append(
                {
                    "col": col,
                    "reason": "insufficient_class_counts_fit",
                    "n_zero_fit": n_zero_fit,
                    "n_pos_fit": n_pos_fit,
                }
            )
            continue
        if n_zero_val < 1 or n_pos_val < 1:
            skipped.append(
                {
                    "col": col,
                    "reason": "insufficient_class_counts_val",
                    "n_zero_val": n_zero_val,
                    "n_pos_val": n_pos_val,
                }
            )
            continue

        LOGGER.info(
            "== %s == zero_frac=%.3f fit=%d/%d val=%d/%d (zero/pos)",
            col,
            zero_frac,
            n_zero_fit,
            n_pos_fit,
            n_zero_val,
            n_pos_val,
        )

        col_result = evaluate_column(
            col=col,
            X_fit=X_fit_all,
            y_fit_label=fit_label,
            X_val=X_val_all,
            y_val_label=val_label,
        )

        per_col[col] = {
            "zero_frac_train": zero_frac,
            "min_val": min_val,
            "n_zero_fit": n_zero_fit,
            "n_pos_fit": n_pos_fit,
            "n_zero_val": n_zero_val,
            "n_pos_val": n_pos_val,
            "classifiers": col_result,
        }
        eligible.append(col)

        summary = " ".join(
            f"{clf}=ll{m['log_loss']:.4f}/auc{m['auc']:.3f}"
            for clf, m in col_result.items()
        )
        LOGGER.info("  %s", summary)

    # Aggregate across eligible columns
    aggregate: dict[str, dict[str, float]] = {}
    for clf in CLASSIFIERS:
        rows = [per_col[c]["classifiers"][clf] for c in eligible]
        if not rows:
            continue
        agg = {
            "log_loss_mean": float(np.mean([r["log_loss"] for r in rows])),
            "log_loss_median": float(np.median([r["log_loss"] for r in rows])),
            "brier_mean": float(np.mean([r["brier"] for r in rows])),
            "ece_mean": float(np.mean([r["ece"] for r in rows])),
            "auc_mean": float(np.nanmean([r["auc"] for r in rows])),
            "auc_median": float(np.nanmedian([r["auc"] for r in rows])),
            "fit_s_total": float(np.sum([r["fit_s"] for r in rows])),
        }
        aggregate[clf] = agg

    out = {
        "config": {
            "data_path": str(args.data_path),
            "year": args.year,
            "seed": args.seed,
            "holdout_frac": args.holdout_frac,
            "inner_val_frac": args.inner_val_frac,
            "zero_threshold": args.zero_threshold,
            "n_train_rows": len(train),
            "n_fit_rows": len(fit_idx),
            "n_val_rows": len(val_idx),
            "condition_cols": list(DEFAULT_CONDITION_COLS),
            "target_cols": list(DEFAULT_TARGET_COLS),
            "eligible_cols": eligible,
            "skipped": skipped,
        },
        "per_column": per_col,
        "aggregate": aggregate,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(out, indent=2, default=str))
    LOGGER.info("wrote %s", args.output)

    print()
    print(f"Eligible columns (zero_frac >= {args.zero_threshold}): {len(eligible)}")
    print(f"Skipped columns: {len(skipped)}")
    print()
    print(
        f"{'classifier':>15}  {'log_loss':>9}  {'log_loss_med':>12}  "
        f"{'brier':>7}  {'ece':>7}  {'auc':>6}  {'auc_med':>7}  {'total_fit_s':>11}"
    )
    ordered = sorted(aggregate.items(), key=lambda kv: kv[1]["log_loss_mean"])
    for clf, agg in ordered:
        print(
            f"{clf:>15}  {agg['log_loss_mean']:9.4f}  {agg['log_loss_median']:12.4f}  "
            f"{agg['brier_mean']:7.4f}  {agg['ece_mean']:7.4f}  "
            f"{agg['auc_mean']:6.3f}  {agg['auc_median']:7.3f}  "
            f"{agg['fit_s_total']:11.1f}"
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
