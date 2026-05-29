"""Clean-slate, scoreboard-first rebuild of the microplex-US calibration loop.

This package exists to settle one question honestly: does a Microplex-built
US microdataset calibrate better than the incumbent enhanced CPS (eCPS), on the
real PolicyEngine-native target estate, under a SOUND and SYMMETRIC measurement
contract?

Design principles (see rebuild/PLAN.md):
  1. The scoreboard is the product. Build it first, make it adversarial.
  2. The fitting objective IS the scored objective (one loss, no divergence).
  3. The calibration operator can never make a dataset worse on that loss
     (monotone non-increase, guaranteed by line search). The old harness
     violated this: refitting eCPS drove its loss 0.166 -> 0.544.
  4. Comparisons are symmetric (identical operation applied to both datasets)
     and matched on N.
  5. Held-out targets are the headline number; in-sample is a diagnostic.
"""

from .scoreboard import (
    CalibrationProblem,
    compare,
    fit,
    score,
    split_targets,
)

__all__ = [
    "CalibrationProblem",
    "compare",
    "fit",
    "score",
    "split_targets",
]
