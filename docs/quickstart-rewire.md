# Quickstart — `spec-based-ecps-rewire` tools

*Walk through every piece of tooling that landed on the rewire branch overnight, in the order you'd actually use them.*

## 1. Set up

```bash
cd microplex-us
git checkout spec-based-ecps-rewire
uv pip install -e .[dev]
uv pip install microcalibrate prdc
```

Python 3.13+ required (microcalibrate dep). All tests should pass:

```bash
uv run pytest tests/calibration tests/bakeoff -q
# Expected: 21 passed in ~10 s
```

## 2. Calibration: the G1 unblocker

`microplex_us.calibration.MicrocalibrateAdapter` is the production calibrator
from now on. It's wired into `USMicroplexBuildConfig.calibration_backend`:

```bash
uv run python -m microplex_us.pipelines.pe_us_data_rebuild_checkpoint \
    --baseline-dataset ~/PolicyEngine/policyengine-us-data/policyengine_us_data/storage/enhanced_cps_2024.h5 \
    --targets-db ~/PolicyEngine/policyengine-us-data/policyengine_us_data/storage/calibration/policy_data.db \
    --policyengine-us-data-repo ~/PolicyEngine/policyengine-us-data \
    --output-root artifacts/live_pe_us_data_rebuild_checkpoint_20260417_microcalibrate \
    --version-id v7 \
    --calibration-backend microcalibrate
```

The `--calibration-backend microcalibrate` flag is the only meaningful change
from the v4/v5/v6 launch commands. Everything else stays identical.

Expected change from v6: the OOM at `backend=entropy` during
`calibrate_policyengine_tables` is gone. Pipeline should complete and write
`pe_us_data_rebuild_parity.json`.

### Verify dispatch without running the whole pipeline

```python
from microplex_us.pipelines.us import USMicroplexBuildConfig, USMicroplexPipeline
from microplex_us.calibration import MicrocalibrateAdapter

cfg = USMicroplexBuildConfig(calibration_backend="microcalibrate")
pipeline = USMicroplexPipeline(cfg)
calibrator = pipeline._build_weight_calibrator()
assert isinstance(calibrator, MicrocalibrateAdapter)
```

Covered by `tests/calibration/test_us_pipeline_dispatch.py`.

## 3. Synthesizer scale-up benchmark

```bash
# Defaults: ZI-QRF + ZI-MAF + ZI-QDNN, all 77k rows × 50 columns
uv run python -m microplex_us.bakeoff \
    --stage stage1 \
    --methods ZI-QRF ZI-MAF ZI-QDNN \
    --output artifacts/scale_up_stage1.json

# Completes in ~6 minutes on a 48 GB M3.
# Per-method results land in artifacts/scale_up_stage1.json.partial.jsonl
# as soon as each method finishes.
```

### Run a single method at a smaller scale

```python
from pathlib import Path
from microplex_us.bakeoff import ScaleUpRunner, ScaleUpStageConfig, stage1_config

base = stage1_config()
cfg = ScaleUpStageConfig(
    stage="quick_zi_qrf",
    n_rows=20_000,
    methods=("ZI-QRF",),
    condition_cols=base.condition_cols,
    target_cols=base.target_cols,
    holdout_frac=0.2,
    seed=42,
    k=5,
    n_generate=16_000,
    data_path=base.data_path,
    year=base.year,
    rare_cell_checks=base.rare_cell_checks,
    prdc_max_samples=15_000,
)
results = ScaleUpRunner(cfg).run(incremental_path=Path("artifacts/quick.jsonl"))
for r in results:
    print(r.method, r.coverage, r.fit_wall_seconds)
```

### Tune per-method hyperparameters

```python
cfg = ScaleUpStageConfig(
    # ... other fields ...
    method_kwargs={
        "ZI-MAF": {"n_layers": 8, "hidden_dim": 128, "epochs": 200, "lr": 5e-4},
    },
)
```

Every field in the method class's `__init__` signature can be overridden.

### Interpret the result

`ScaleUpResult` fields:

- `coverage` — PRDC coverage (fraction of real records with a synthetic neighbor within k-NN). Higher is better. Sample-size sensitive (see the PRDC cap note below).
- `precision`, `density` — other PRDC metrics.
- `fit_wall_seconds`, `generate_wall_seconds` — timing.
- `peak_rss_gb_during_fit` — process RSS (on macOS, corrected for the bytes-vs-KB units bug).
- `zero_rate_mae` — scalar mean absolute error in per-column zero-rate.
- `zero_rate_per_column` — per-column `{real, synth, abs_diff}`. Identifies which specific columns drive the error.
- `rare_cell_ratios` — synth-count / real-count for designated rare subpopulations (elderly self-employed, young dividend, disabled SSDI, top-1 % employment).

### Known quirks

- **PRDC sample size matters.** Coverage drops as real sample grows (tighter k-NN radius). Compare across stages only when `prdc_max_samples` is the same.
- **ZI-MAF / ZI-QDNN at default settings are not competitive** on real ECPS. Stage-1 result: ZI-QRF 0.256 >> ZI-QDNN 0.147 >> ZI-MAF 0.014 at 77k × 50. Hyperparameter tuning is an open investigation (see `docs/stage-1-pilot-results.md`).

## 4. Embedding-PRDC validation (optional)

Standalone script that settles whether stage-1's ordering is a metric artifact from 50-dim PRDC:

```bash
uv run python scripts/embedding_prdc_compare.py \
    --n-rows 40000 \
    --output artifacts/embedding_prdc_compare.json
```

Trains a 16-dim autoencoder on the holdout, then computes PRDC in both raw and latent space. Takes ~5 min.

If ordering is preserved in latent space: stage-1 finding is robust. If it changes: raw PRDC in 50-dim was noise and the stage-1 winners need re-examination in a less dimensionality-sensitive metric.

## 5. Diagnostics

### PSID coverage = 0 reproduction

```python
import pandas as pd
import numpy as np

df = pd.read_parquet("~/PolicyEngine/microplex/data/stacked_comprehensive.parquet")
exclude = {"weight", "person_id", "household_id", "interview_number"}

survey_dfs = {}
for src in ["sipp", "cps", "psid"]:
    sub = df[df["_survey"] == src].drop(columns=["_survey"]).copy()
    num = [c for c in sub.columns
           if sub[c].dtype.kind in "fiu" and sub[c].isna().mean() < 0.05]
    survey_dfs[src] = sub[num].dropna().reset_index(drop=True)

first = next(iter(survey_dfs.values()))
shared = [c for c in first.columns
          if c not in exclude and all(c in d.columns for d in survey_dfs.values())]
print("shared_cols:", shared)  # ['is_male', 'age'] — 2 variables
```

Full diagnosis in `docs/psid-coverage-zero-diagnosis.md`.

## 6. What to look at for planning the next step

Read these in order:

1. `docs/v6-postmortem.md` — what killed v6 and why
2. `docs/calibrator-decision.md` — why microcalibrate is mainline
3. `docs/core-wiring-audit.md` — what's in microplex core, what's wired, what to swap
4. `docs/synthesizer-benchmark-scale-up.md` — how to think about scale-up
5. `docs/stage-1-pilot-results.md` — the actual numbers and what they mean
6. `docs/microcalibrate-wiring-plan.md` — rollout of the G1 unblocker
7. `docs/overnight-session-2026-04-16.md` — full session audit trail
8. `docs/psid-coverage-zero-diagnosis.md` — the PSID = 0 finding

## 7. Production next steps

Ordered by expected value:

1. Launch a v7 run with `--calibration-backend microcalibrate`. Expected outcome: pipeline completes and writes parity artifact. If it OOMs, the OOM is in a *different* stage than calibration, which is a new finding.
2. After v7 completes: parse the parity artifact and compare against `broader-donors-ssn-card-type-v1` (baseline 0.6955 full-oracle capped loss). If v7 lands below that, G1 is cleared.
3. While v7 runs: execute stage-2 scale-up (1M rows × 50 cols) on the rewire branch. Requires a larger data source than ECPS (77k limit); the natural candidate is a clone-and-assign of ECPS to 1M, matching PE-US-data's local-area pattern.
4. If ZI-MAF tuning recovered it (see `artifacts/zi_maf_tuning.json` once the overnight run completes): lock in the best config as the new `ZI-MAF` default in `method_kwargs`.

## 8. Cleanup tasks from the session

These are tracked as follow-ups and do not block G1:

- `disabled_ssdi` zero-rate diverges to 0.0 on all methods. Investigate per-column breakdown (now exposed) to find which other columns break.
- ZI-QRF OOM at the loky-worker level above 61k×50. Already worked around (PRDC cap). Root-cause fix would be switching `n_jobs=-1` to a bounded pool or a worker-recycling wrapper.
- MPS / CUDA for ZI-MAF + ZI-QDNN in the benchmark method classes. Would shrink fit time 3–5× but is a separate refactor of `microplex.eval.benchmark`.
- Per-method benchmark at v6 scale (1.5 M household entity table) once the v7 pipeline gives us that artifact to measure against.

## 9. Don't do

- Don't launch another v6-style run with `backend=entropy`. Known-OOM. Use `microcalibrate`.
- Don't take the small-benchmark (10k × 7 synthetic) ordering at face value for G1 defaults. Stage-1 evidence overturned it.
- Don't trust raw PRDC coverage in 50 dimensions as an absolute number across stages. Ordering across methods at the same stage/config is fine; absolute numbers across stages need the same PRDC cap.
