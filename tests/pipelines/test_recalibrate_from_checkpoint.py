"""Recalibrate-from-checkpoint helper.

Loads a post-imputation bundle previously saved by
``save_us_pipeline_checkpoint`` and calls
``pipeline.calibrate_policyengine_tables`` on it. Used by operators to
iterate on calibration config (backend, lambda schedule, targets)
without paying the ~11 h synthesis + donor-imputation cost that
produced the bundle.

These tests drive:

1. The helper loads a post-imputation checkpoint and dispatches the
   bundle to a fresh pipeline's calibrate method.
2. The helper also accepts post-microsim checkpoints, where materialized
   target columns already exist on the bundle.
3. The helper raises a clear error if the checkpoint directory is
   missing.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

from microplex_us.pipelines.us import USMicroplexBuildConfig
from microplex_us.policyengine.us import (
    PolicyEngineUSEntityTableBundle,
    save_us_pipeline_checkpoint,
)


def _make_bundle(n: int = 50) -> PolicyEngineUSEntityTableBundle:
    rng = np.random.default_rng(0)
    household_ids = np.arange(n) + 1
    return PolicyEngineUSEntityTableBundle(
        households=pd.DataFrame(
            {
                "household_id": household_ids,
                "household_weight": rng.uniform(0.5, 2.0, size=n),
            }
        ),
        persons=pd.DataFrame(
            {
                "person_id": household_ids * 10,
                "household_id": household_ids,
                "age": rng.integers(0, 85, size=n),
            }
        ),
    )


class TestRecalibrateFromPipelineCheckpoint:
    @pytest.mark.parametrize("stage", ["post_imputation", "post_microsim"])
    def test_checkpoint_dispatches_to_calibrate(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        stage: str,
    ) -> None:
        """Both supported stages load their bundle and dispatch to calibrate.

        For ``post_microsim``, microsim is skipped inside
        ``_resolve_policyengine_calibration_targets`` because all
        materialized vars are present as columns; for
        ``post_imputation``, microsim runs normally. The helper only
        orchestrates the load and hand-off, so the parametrized test
        covers both paths.
        """
        from microplex_us.pipelines.us import (
            recalibrate_policyengine_us_from_checkpoint,
        )

        bundle = _make_bundle(n=40)
        save_us_pipeline_checkpoint(
            bundle, tmp_path / "checkpoint", stage=stage
        )

        observed_tables: list[PolicyEngineUSEntityTableBundle] = []

        def _fake_calibrate(
            self: Any,
            tables: PolicyEngineUSEntityTableBundle,
        ) -> tuple[PolicyEngineUSEntityTableBundle, pd.DataFrame, dict[str, Any]]:
            observed_tables.append(tables)
            return (
                tables,
                tables.households.assign(weight=tables.households["household_weight"]),
                {"mock": True},
            )

        monkeypatch.setattr(
            "microplex_us.pipelines.us.USMicroplexPipeline.calibrate_policyengine_tables",
            _fake_calibrate,
        )

        cfg = USMicroplexBuildConfig(
            calibration_backend="pe_l0",
            policyengine_targets_db=tmp_path / "targets.db",
        )
        result = recalibrate_policyengine_us_from_checkpoint(cfg, tmp_path / "checkpoint")

        assert len(observed_tables) == 1
        pd.testing.assert_frame_equal(
            observed_tables[0].households, bundle.households
        )
        assert result.calibration_summary == {"mock": True}
        assert result.loaded_stage == stage
        pd.testing.assert_frame_equal(
            result.policyengine_tables.households, bundle.households
        )

    def test_unsupported_stage_raises(self, tmp_path: Path) -> None:
        """A metadata.json with an unknown stage is rejected."""
        from microplex_us.pipelines.us import (
            recalibrate_policyengine_us_from_checkpoint,
        )

        (tmp_path / "checkpoint").mkdir()
        import json

        (tmp_path / "checkpoint" / "metadata.json").write_text(
            json.dumps({"format_version": 1, "stage": "bogus"})
        )
        cfg = USMicroplexBuildConfig(policyengine_targets_db=tmp_path / "targets.db")
        with pytest.raises(ValueError, match="Cannot resume"):
            recalibrate_policyengine_us_from_checkpoint(cfg, tmp_path / "checkpoint")

    def test_missing_checkpoint_raises(self, tmp_path: Path) -> None:
        from microplex_us.pipelines.us import (
            recalibrate_policyengine_us_from_checkpoint,
        )

        cfg = USMicroplexBuildConfig(policyengine_targets_db=tmp_path / "targets.db")
        with pytest.raises(FileNotFoundError):
            recalibrate_policyengine_us_from_checkpoint(cfg, tmp_path / "nope")


class TestRecalibrateFromCheckpointCli:
    def test_prepare_output_root_accepts_existing_empty_directory(
        self,
        tmp_path: Path,
    ) -> None:
        from microplex_us.pipelines.pe_us_recalibrate_from_checkpoint import (
            _prepare_output_root,
        )

        output_root = tmp_path / "output"
        output_root.mkdir()

        assert _prepare_output_root(output_root) == output_root
        assert output_root.is_dir()
        assert list(output_root.iterdir()) == []

    def test_prepare_output_root_rejects_missing_directory(
        self,
        tmp_path: Path,
    ) -> None:
        from microplex_us.pipelines.pe_us_recalibrate_from_checkpoint import (
            _prepare_output_root,
        )

        output_root = tmp_path / "output"

        with pytest.raises(FileNotFoundError, match="--output-root does not exist"):
            _prepare_output_root(output_root)
        assert not output_root.exists()

    def test_prepare_output_root_rejects_unwritable_directory(
        self,
        tmp_path: Path,
    ) -> None:
        from microplex_us.pipelines.pe_us_recalibrate_from_checkpoint import (
            _prepare_output_root,
        )

        output_root = tmp_path / "output"
        output_root.mkdir()
        original_mode = output_root.stat().st_mode
        try:
            output_root.chmod(0o500)
            if os.access(output_root, os.W_OK | os.X_OK):
                pytest.skip("current platform still reports chmod 0500 as writable")
            with pytest.raises(PermissionError, match="--output-root is not writable"):
                _prepare_output_root(output_root)
        finally:
            output_root.chmod(original_mode)

    def test_main_rejects_output_file_before_recalibration(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        import microplex_us.pipelines.pe_us_recalibrate_from_checkpoint as cli

        called = False

        def _fail_if_called(*args: Any, **kwargs: Any) -> None:
            nonlocal called
            called = True
            raise AssertionError("recalibration should not start")

        monkeypatch.setattr(
            cli,
            "recalibrate_policyengine_us_from_checkpoint",
            _fail_if_called,
        )
        output_root = tmp_path / "output"
        output_root.write_text("not a directory")

        with pytest.raises(NotADirectoryError, match="--output-root is not a directory"):
            cli.main(
                [
                    "--checkpoint-path",
                    str(tmp_path / "checkpoint"),
                    "--output-root",
                    str(output_root),
                    "--targets-db",
                    str(tmp_path / "targets.db"),
                ]
            )

        assert called is False

    def test_main_rejects_missing_output_directory_before_recalibration(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        import microplex_us.pipelines.pe_us_recalibrate_from_checkpoint as cli

        called = False

        def _fail_if_called(*args: Any, **kwargs: Any) -> None:
            nonlocal called
            called = True
            raise AssertionError("recalibration should not start")

        monkeypatch.setattr(
            cli,
            "recalibrate_policyengine_us_from_checkpoint",
            _fail_if_called,
        )
        output_root = tmp_path / "output"

        with pytest.raises(FileNotFoundError, match="--output-root does not exist"):
            cli.main(
                [
                    "--checkpoint-path",
                    str(tmp_path / "checkpoint"),
                    "--output-root",
                    str(output_root),
                    "--targets-db",
                    str(tmp_path / "targets.db"),
                ]
            )

        assert called is False
        assert not output_root.exists()
