# k=2 embedding error-rate figures

These figures use the complete `S(n,2)` matrices for
`n = 20, 40, ..., 240`, both InfoNCE and sigmoid, every dimension
`d = 5, ..., 30`, seed 0, and 100,000 optimization steps without early
stopping. See `experiments/k2_error_rates` for the training and analysis code.

## Paper-style reproductions

- **`k2_minimum_positive_dimension`**: the smallest tested dimension whose final
  global margin is strictly positive. InfoNCE needs increasing dimension as
  `n` grows and does not separate by `d=30` for `n=220,240`; sigmoid separates
  between `d=5` and `d=10` throughout.
- **`k2_maximal_margin`**: the largest checkpointed positive margin
  during 100,000 steps for the paper's dimensions `d=6,18,30`.

## Final classification errors

All probability figures use

```text
FPR = count(A_ij=0 and score>threshold) / count(A_ij=0)
FNR = count(A_ij=1 and score<=threshold) / count(A_ij=1)
```

- **`k2_false_positive_negative_rates`**: fixed threshold 0. This is appropriate for the
  sigmoid objective but exposes that InfoNCE scores are not calibrated around
  zero.
- **`k2_midpoint_false_positive_negative_rates`**: threshold halfway between the positive and
  negative class-mean scores. This removes the arbitrary InfoNCE score offset,
  but the midpoint need not fall inside an existing separation gap.
- **`k2_extrema_midpoint_false_positive_negative_rates`**: threshold halfway between the minimum positive
  and maximum negative score. Zero error occurs exactly when the global margin
  is strictly positive.
- **`k2_extrema_midpoint_combined_error_rates`**: the same extrema-midpoint data with
  losses overlaid in one panel per `n`. InfoNCE is solid and sigmoid is dotted;
  blue denotes false positives and orange denotes false negatives.

Each figure is provided as PNG and PDF. The accompanying CSV files contain the
full-precision values, integer counts, denominators, and threshold statistics.
