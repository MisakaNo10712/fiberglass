# Unit sanity and reproducibility notes

## Why global_step must be checkpointed
- The curriculum schedule uses global_step to compute lambda_kappa and lambda_hf.
- verify_collapse_fix calls Trainer._scheduled_lambdas(), so an incorrect global_step makes the reported schedule and loss ratios misleading.
- If a checkpoint lacks global_step and verify defaults to 0, the schedule will look like stage start even if training was far past warmup.

## How to read the new diagnostics
- x_range/Lx and y_range/Ly should be close to 1 if coordinates and Lx/Ly are in the same unit system. Values >>1 or <<1 suggest a unit mismatch (for example mm vs m) or a wrong Lx/Ly.
- kappa_meas vs kappa_pred absolute scale:
  - abs_max, mean, std show raw scale.
  - scale_ratio_std and scale_ratio_max highlight mismatches; values >1e3 or <1e-3 are red flags.
- Standardized stats (kappa_*_hat) use dataset mean/std. If kappa_pred_hat is far from O(1), the loss scale is likely inconsistent.
- masked_*_max_abs_diff avoids padding influence; prefer these when comparing two samples of different lengths.

## Physical interpretation hints
- The curvature operator uses DCT basis with kx = m*pi/Lx and ky = n*pi/Ly, so the second derivative introduces kx^2/ky^2.
- If x/y units are off relative to Lx/Ly, kappa_pred can blow up by a squared scale factor.
- If kappa_meas is missing a thickness or proportionality factor, scale_ratio_* will show a large mismatch.

## Scale knobs (default 1.0)
- data.coord_scale: multiply x/y in the dataset before feeding the model and curvature operator.
- data.kappa_meas_scale: multiply kappa_meas (and its mean/std) for consistent normalization.
- These are for quick sanity tests; they are reported in the unit sanity summary.
