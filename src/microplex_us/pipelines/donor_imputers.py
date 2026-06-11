"""Donor imputer implementations for US pipeline donor synthesis."""

from __future__ import annotations

import importlib.util
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier


class ColumnwiseQRFDonorImputer:
    """Columnwise QRF donor imputer, optionally with zero-inflated support."""

    def __init__(
        self,
        *,
        condition_vars: list[str],
        target_vars: list[str],
        n_estimators: int = 100,
        zero_inflated_vars: set[str] | None = None,
        nonnegative_vars: set[str] | None = None,
        zero_threshold: float = 0.05,
        quantiles: tuple[float, ...] = (0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95),
    ) -> None:
        self.condition_vars = list(condition_vars)
        self.target_vars = list(target_vars)
        self.n_estimators = int(n_estimators)
        self.zero_inflated_vars = set(zero_inflated_vars or ())
        self.nonnegative_vars = set(nonnegative_vars or ())
        self.zero_threshold = float(zero_threshold)
        self.quantiles = tuple(float(value) for value in quantiles)
        self._models: dict[str, Any] = {}
        self._zero_models: dict[str, RandomForestClassifier] = {}

    def fit(
        self,
        data: pd.DataFrame,
        *,
        weight_col: str | None = "weight",
        epochs: int | None = None,
        batch_size: int | None = None,
        learning_rate: float | None = None,
        verbose: bool = False,
    ) -> ColumnwiseQRFDonorImputer:
        del weight_col, epochs, batch_size, learning_rate, verbose
        if importlib.util.find_spec("quantile_forest") is None:
            raise ImportError(
                "quantile-forest is required for donor_imputer_backend='qrf'"
            )
        from quantile_forest import RandomForestQuantileRegressor

        self._models = {}
        self._zero_models = {}
        for column in self.target_vars:
            subset = data[self.condition_vars + [column]].dropna()
            if len(subset) < 25:
                continue
            x_values = subset[self.condition_vars].to_numpy(dtype=float)
            y_values = subset[column].to_numpy(dtype=float)
            if (
                column in self.zero_inflated_vars
                and (y_values == 0).mean() >= self.zero_threshold
                and (y_values == 0).sum() >= 10
                and (y_values != 0).sum() >= 10
            ):
                # Gate trained as zero vs nonzero (both signs), not as
                # zero-or-negative vs positive. The old `y > 0` label
                # silently dropped every negative training row along
                # with zeros, so the QRF below only ever saw positive
                # rows and could never emit a negative prediction - the
                # v7 bug that blanked the negative tail of capital
                # gains, partnership income, farm income, etc. The
                # `!= 0` label is the minimal fix; the full upgrade to
                # `microimpute.ZeroInflatedImputer` (regime-aware
                # tripartite routing with separate positive / negative
                # QRFs) is tracked as a follow-up.
                zero_model = RandomForestClassifier(
                    n_estimators=max(50, self.n_estimators // 2),
                    random_state=42,
                    n_jobs=-1,
                )
                zero_model.fit(x_values, (y_values != 0).astype(int))
                self._zero_models[column] = zero_model
                x_values = x_values[y_values != 0]
                y_values = y_values[y_values != 0]
            if len(y_values) < 25:
                continue
            model = RandomForestQuantileRegressor(
                n_estimators=self.n_estimators,
                random_state=42,
                n_jobs=-1,
            )
            model.fit(x_values, y_values)
            self._models[column] = model
        return self

    def generate(
        self,
        conditions: pd.DataFrame,
        seed: int | None = None,
    ) -> pd.DataFrame:
        rng = np.random.RandomState(seed or 42)
        synthetic = conditions.copy().reset_index(drop=True)
        x_values = synthetic[self.condition_vars].to_numpy(dtype=float)
        for column in self.target_vars:
            model = self._models.get(column)
            if model is None:
                synthetic[column] = np.nan
                continue
            values = np.zeros(len(synthetic), dtype=float)
            target_rows = np.ones(len(synthetic), dtype=bool)
            zero_model = self._zero_models.get(column)
            if zero_model is not None:
                probabilities = zero_model.predict_proba(x_values)
                positive_probs = (
                    probabilities[:, 1]
                    if probabilities.shape[1] > 1
                    else np.zeros(len(synthetic), dtype=float)
                )
                target_rows = rng.random(len(synthetic)) < positive_probs
                values[:] = 0.0
            if target_rows.any():
                predictions = model.predict(
                    x_values[target_rows],
                    quantiles=list(self.quantiles),
                )
                quantile_choices = rng.choice(
                    len(self.quantiles), size=target_rows.sum()
                )
                draws = predictions[np.arange(target_rows.sum()), quantile_choices]
                if column in self.nonnegative_vars:
                    draws = np.maximum(draws, 0.0)
                values[target_rows] = draws
            synthetic[column] = values
        return synthetic


class RegimeAwareDonorImputer:
    """Donor imputer that wraps `microimpute.ZeroInflatedImputer` per column.

    Each target is fit with an independent `ZeroInflatedImputer`, which
    auto-detects one of seven regimes (THREE_SIGN / ZI_POSITIVE /
    ZI_NEGATIVE / SIGN_ONLY / POSITIVE_ONLY / NEGATIVE_ONLY /
    DEGENERATE_ZERO) from the training distribution and composes a
    gate classifier + one or two base imputers as appropriate.

    Key advantages over `ColumnwiseQRFDonorImputer`:

    1. Negative values in training are preserved in predictions for
       three-sign targets (capital gains, partnership/S-corp income,
       farm income, rental income). The v7 `y > 0` bug is structurally
       impossible under regime-aware routing.
    2. Predictions on three-sign targets never land in the interior
       band between ``max(train_neg)`` and ``min(train_pos)`` - the
       tripartite gate routes to sign-specific base imputers that each
       see only one sign of training data.

    This class is a thin columnwise adapter: one `ZeroInflatedImputer`
    is fit per target, using `microimpute.QRF` as the base. Fit and
    generate work column-by-column so memory scales with the single
    largest base imputer, not with the total target count.
    """

    def __init__(
        self,
        condition_vars: list[str],
        target_vars: list[str],
        n_estimators: int = 100,
        classifier_type: str = "hist_gb",
        seed: int = 42,
    ) -> None:
        self.condition_vars = list(condition_vars)
        self.target_vars = list(target_vars)
        self.n_estimators = int(n_estimators)
        self.classifier_type = str(classifier_type)
        self.seed = int(seed)
        self._fitted: dict[str, Any] = {}
        self._regimes: dict[str, str] = {}

    def fit(
        self,
        data: pd.DataFrame,
        *,
        weight_col: str | None = "weight",
        epochs: int | None = None,
        batch_size: int | None = None,
        learning_rate: float | None = None,
        verbose: bool = False,
    ) -> RegimeAwareDonorImputer:
        del weight_col, epochs, batch_size, learning_rate, verbose

        if importlib.util.find_spec("microimpute.models.zero_inflated") is None:
            raise ImportError(
                "microimpute with microimpute.models.zero_inflated is required "
                "for donor_imputer_backend='regime_aware'."
            )
        if importlib.util.find_spec("quantile_forest") is None:
            raise ImportError(
                "quantile-forest is required for the RegimeAwareDonorImputer base QRF."
            )

        from microimpute.models.qrf import QRF
        from microimpute.models.zero_inflated import ZeroInflatedImputer

        self._fitted = {}
        self._regimes = {}
        for column in self.target_vars:
            subset = data[self.condition_vars + [column]].dropna()
            if len(subset) < 25:
                continue
            # base_imputer_kwargs={} because microimpute 2.x's
            # ZeroInflatedImputer._fit_base_single already passes
            # log_level="ERROR" to the base, and duplicating it here
            # raises TypeError. Upstream fix tracked.
            wrapper = ZeroInflatedImputer(
                base_imputer_class=QRF,
                base_imputer_kwargs={},
                classifier_type=self.classifier_type,
                seed=self.seed,
            )
            fitted = wrapper.fit(
                subset,
                predictors=list(self.condition_vars),
                imputed_variables=[column],
            )
            self._fitted[column] = fitted
            self._regimes[column] = wrapper.get_regime(column)
        return self

    def generate(
        self,
        conditions: pd.DataFrame,
        seed: int | None = None,
    ) -> pd.DataFrame:
        synthetic = conditions.copy().reset_index(drop=True)
        master_seed = self.seed if seed is None else int(seed)
        master_rng = np.random.default_rng(master_seed)
        for column in self.target_vars:
            fitted = self._fitted.get(column)
            if fitted is None:
                synthetic[column] = np.nan
                continue
            column_seed = int(
                master_rng.integers(0, np.iinfo(np.int32).max, dtype=np.int64)
            )
            self._reset_prediction_rngs(fitted, seed=column_seed)
            preds = fitted.predict(synthetic[self.condition_vars])
            synthetic[column] = preds[column].to_numpy(dtype=float)
        return synthetic

    def _reset_prediction_rngs(
        self,
        obj: Any,
        *,
        seed: int,
        visited: set[int] | None = None,
    ) -> None:
        if visited is None:
            visited = set()
        if obj is None or isinstance(obj, (str, bytes, int, float, bool)):
            return
        object_id = id(obj)
        if object_id in visited:
            return
        visited.add(object_id)

        if hasattr(obj, "_rng"):
            obj._rng = np.random.default_rng(seed)
        child_rng = np.random.default_rng(seed)

        if isinstance(obj, dict):
            children = list(obj.values())
        elif isinstance(obj, (list, tuple, set)):
            children = list(obj)
        else:
            children = []
            for attr_name in ("models", "_per_variable", "_non_numeric_bundle"):
                child = getattr(obj, attr_name, None)
                if child is not None:
                    children.append(child)

        for child in children:
            child_seed = int(
                child_rng.integers(0, np.iinfo(np.int32).max, dtype=np.int64)
            )
            self._reset_prediction_rngs(
                child,
                seed=child_seed,
                visited=visited,
            )


class MicroimputeDonorImputer:
    """Donor imputer backed by microimpute's canonical chained ``Imputer``.

    Unlike ``ColumnwiseQRFDonorImputer`` and ``RegimeAwareDonorImputer`` -
    which fit one independent model per target - this adapter fits the
    entire donor-target block as ONE sequentially chained call. microimpute's
    ``Imputer`` (regime-gated, QRF-based) imputes the targets in the order
    they are listed in ``target_vars``: each target conditions on the
    original predictors plus the targets imputed before it, preserving the
    cross-variable joint structure (e.g. capital losses conditioned on the
    already-imputed wage/income spine).

    Mirrors the interface of the other donor imputers so the block engine can
    swap it in transparently:

    - ``__init__(*, condition_vars, target_vars, zero_inflated_vars=None,
      seed=...)``
    - ``.fit(data, weight_col=None, **kwargs)``
    - ``.generate(conditions, seed=None) -> pd.DataFrame``

    ``zero_inflated_vars`` is accepted for interface symmetry but is not
    needed: microimpute's ``Imputer`` auto-detects each target's sign regime
    (THREE_SIGN / ZI_POSITIVE / ZI_NEGATIVE / SIGN_ONLY / POSITIVE_ONLY /
    NEGATIVE_ONLY / DEGENERATE_ZERO) from the training distribution, so
    zero-inflation handling is intrinsic rather than configured per column.
    MAF-only fit kwargs (``epochs``/``batch_size``/``learning_rate``/
    ``verbose``) are accepted and ignored via ``**kwargs``.
    """

    def __init__(
        self,
        *,
        condition_vars: list[str],
        target_vars: list[str],
        zero_inflated_vars: set[str] | None = None,
        nonnegative_vars: set[str] | None = None,
        seed: int = 42,
        classifier_type: str = "hist_gb",
    ) -> None:
        # zero_inflated_vars / nonnegative_vars are part of the shared donor
        # imputer interface but are not used here: regime detection is
        # intrinsic to microimpute's Imputer (see class docstring).
        del zero_inflated_vars, nonnegative_vars
        self.condition_vars = list(condition_vars)
        self.target_vars = list(target_vars)
        self.seed = int(seed)
        self.classifier_type = str(classifier_type)
        self._fitted: Any = None

    def fit(
        self,
        data: pd.DataFrame,
        *,
        weight_col: str | None = "weight",
        epochs: int | None = None,
        batch_size: int | None = None,
        learning_rate: float | None = None,
        verbose: bool = False,
    ) -> MicroimputeDonorImputer:
        # MAF-only knobs are accepted for interface symmetry and ignored.
        del epochs, batch_size, learning_rate, verbose

        if importlib.util.find_spec("microimpute") is None:
            raise ImportError(
                "microimpute is required for donor_imputer_backend='microimpute'."
            )
        from microimpute import Imputer

        columns = list(self.condition_vars) + list(self.target_vars)
        if weight_col is not None and weight_col in data.columns:
            columns = columns + [weight_col]
        # Chained-equations fitting needs one consistent training frame, so
        # drop rows with any missing predictor/target/weight value listwise
        # rather than per column. This mirrors the dropna() the columnwise
        # imputer applies per target, but keeps a single aligned design
        # matrix across the whole chain.
        present_columns = [column for column in columns if column in data.columns]
        fit_frame = data[present_columns].replace([np.inf, -np.inf], np.nan).dropna()

        effective_weight_col = (
            weight_col
            if weight_col is not None and weight_col in fit_frame.columns
            else None
        )
        if effective_weight_col is not None:
            positive = fit_frame[effective_weight_col] > 0
            fit_frame = fit_frame.loc[positive]

        # Construct the Imputer with the effective seed so prediction draws
        # (which read the fitted result's RNG, seeded from this value) are
        # reproducible.
        imputer = Imputer(
            seed=self.seed,
            classifier_type=self.classifier_type,
            log_level="ERROR",
        )
        self._fitted = imputer.fit(
            X_train=fit_frame,
            predictors=list(self.condition_vars),
            imputed_variables=list(self.target_vars),
            weight_col=effective_weight_col,
        )
        return self

    def generate(
        self,
        conditions: pd.DataFrame,
        seed: int | None = None,
    ) -> pd.DataFrame:
        synthetic = conditions.copy().reset_index(drop=True)
        if self._fitted is None:
            for variable in self.target_vars:
                synthetic[variable] = np.nan
            return synthetic

        # Re-seed the fitted result's prediction RNG so a caller-supplied
        # seed makes generate reproducible (the engine passes a fixed seed).
        effective_seed = self.seed if seed is None else int(seed)
        if hasattr(self._fitted, "_rng"):
            self._fitted._rng = np.random.default_rng(effective_seed)

        predictions = self._fitted.predict(
            synthetic[self.condition_vars].copy()
        )
        if isinstance(predictions, dict):
            predictions = next(iter(predictions.values()))
        predictions = predictions.reset_index(drop=True)
        for variable in self.target_vars:
            if variable in predictions.columns:
                synthetic[variable] = pd.to_numeric(
                    predictions[variable], errors="coerce"
                ).to_numpy(dtype=float)
            else:
                synthetic[variable] = np.nan
        return synthetic
