# k=2 embedding error-rate reproduction

This directory reproduces the `k=2` synthetic embedding experiments from
[arXiv:2605.23556](https://arxiv.org/pdf/2605.23556) and adds final
false-positive and false-negative measurements.

## Experiment

- Matrices: the complete `S(n,2)` incidence matrices, with
  `n = 20, 40, ..., 240` and `N = binom(n,2)` rows.
- Losses: InfoNCE and sigmoid.
- Dimensions: every `d = 5, 6, ..., 30`.
- Optimization: 100,000 steps, seed 0, fixed sigmoid relative bias 0, and no
  early stopping.
- Execution: all 26 dimensions have independent `U_d`, `V_d`, temperatures,
  and Adam states, but are trained concurrently in one batched GPU process.
- Checkpointing: atomic resume state and margin history every 1,000 steps.

## Error thresholds

The final scripts report three classification conventions for
`s_ij = <U_i,V_j>`.

1. **Zero:** predict positive when `s_ij > 0`.
2. **Class-mean midpoint:**

   ```text
   threshold = (mean(s_ij | A_ij=1) + mean(s_ij | A_ij=0)) / 2
   ```

3. **Extrema midpoint:**

   ```text
   threshold = (min(s_ij | A_ij=1) + max(s_ij | A_ij=0)) / 2
   ```

For every threshold, a positive score equal to the threshold counts as a false
negative. The extrema midpoint is directly tied to the paper's margin: a
strictly positive margin implies zero false positives and zero false negatives.

## Files

- `batched_d_embed_errors.py`: trains all independent dimensions together and
  saves final embeddings, margins, and threshold-zero error counts.
- `run_k2_batched_errors_array.sh`: Slurm array wrapper.
- `plot_k2_results.py` / `plot_k2_results.sh`: paper-style margin figures and
  the threshold-zero error grid.
- `recompute_midpoint_error_rates.py`: recomputes class-mean or extrema-midpoint
  error rates from saved final `U,V`, without retraining.
- `run_k2_midpoint_postprocess.sh`: CPU Slurm wrapper for saved-embedding
  recomputation.
- `plot_combined_error_rates.py`: overlays InfoNCE (solid) and sigmoid (dotted)
  in a single panel for each `n`.
- `validate_k2_outputs.py`: verifies result completeness, finite values,
  count/rate identities, and completed histories.

The corresponding figures and full-precision tables are in
`diagrams/k2_error_rates`.
