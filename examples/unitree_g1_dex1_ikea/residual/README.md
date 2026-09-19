# Residual BC on the DAgger intervention data

The deployed baseline (`..._stage1v2_absarm` / `..._stage1_absarm_subtask`) is the
best policy this project has put on the robot. Two attempts to improve it by
merging the human-intervention clips into the training set and retraining from
scratch both scored 1-3% BETTER on the offline scan and WORSE on the robot, in
the specific way the operator described: the leg is not seated, and the policy
lets go anyway, with no press and no fine adjustment.

Six mechanisms were measured and none explains it (normalisation drift 0.3% on
the dims the config uses; per-frame sharding, so the expert really is 4.6% of
windows; command |dq| 0.0151 vs the demos' 0.0145; frozen-feature distance 101.4%
of the demos' own spread; the expert presses *harder*, -38.3 mm against -31.2;
release-containing windows over-represented only 1.4x). What is left is that a
full retrain is a new model, and the baseline is the one that survived the robot.

So this does not retrain. It freezes the baseline and learns a correction:

    a_deploy = a_base(o) + delta(o)

delta = 0 is exactly the baseline, which is the property the two SFT runs did not
have. The backbone is frozen in the baseline (`tune_llm`, `tune_visual` both
False, `tune_top_llm_layers` 0), so a given observation always produces the same
features: `cache.py` pays for them once and everything after it trains in minutes.

## Targets

`delta = a_human - a_base(o)`, and `a_base` has to be recomputed offline because
the policy was not running while the human drove. The baseline's head is flow
matching, so a single sample carries its own noise into every target; `cache.py`
averages 8.

## Composition

Training only on the corrections would leave delta undefined everywhere else, so
the "leave it alone" frames are in the set explicitly, and they are frames where
the policy itself acted, which makes their target close to zero by construction:

| source | frames | target |
| --- | ---: | --- |
| `intervention_full` human segments | 12,887 | `a_human - a_base` |
| `intervention_full` policy segments | 48,335 | `a_policy - a_base`, ~0 |
| `deploy_success` | 16,345 | `a_policy - a_base`, ~0 |
| `intervention_full` hold segments | 18,396 | **excluded** -- the arm is frozen at a fixed target and the action is constant |

That is 17% correction, 83% leave-alone, without any hand-set ratio.

## Why not flow matching for the residual

The residual is near zero almost everywhere and single-peaked where it is not.
Fitting it with a second flow head would reintroduce exactly the mode averaging
that the merged SFT runs are suspected of. Plain MSE, and the output is clipped to
+-0.1 rad per joint so a runaway residual cannot leave the baseline's neighbourhood.

## How this gets judged

Not by the scan. The offline scan called both failed SFT runs an improvement, and
its validation split holds only demonstration episodes and no intervention states
at all. Instead:

1. on held-out normal frames, `||delta||` has to stay near zero -- if it does not,
   this degrades the baseline everywhere and must not reach the robot;
2. on held-out intervention episodes, `delta` has to point the way the human
   actually corrected (cosine similarity against the human's own correction);
3. the robot.
