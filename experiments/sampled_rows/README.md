# Fixed sampled-row TopK experiment

This directory contains the scripts used for the `n=100` sampled-row
experiment in `diagrams/sampled_rows_n100`.

## Experimental design

- `n=100`
- `k in {4, 5, 6}`
- `N in {100, 200, 400, 800, 1600, 3200, 6400, 12800, 25600}`
- seed `0`
- InfoNCE and sigmoid objectives
- embedding dimensions `d=5,...,30`
- 100,000 optimization steps, with no early stopping
- margins recorded every 1,000 steps

For each `(n,k,seed)`, `generate_iid_sparse_graphs.py` samples one sequence of
iid uniform `k`-subsets. Sampling is with replacement across rows and without
replacement within each row. The matrix at each larger `N` extends the smaller
matrix by taking a longer prefix of the same sampled sequence. The exact saved
matrix is reused across all dimensions and both losses.

`batched_d_embed.py` trains all requested dimensions concurrently on one GPU.
The dimension axis is batched only for throughput: every `d` has independent
`U_d`, `V_d`, temperature, optimizer state, and normalized active coordinates.
Checkpoints are written atomically every 1,000 steps, so preempted Slurm jobs
can safely requeue and resume.

## Files

- `generate_iid_sparse_graphs.py`: generate and save the fixed nested matrices.
- `batched_d_embed.py`: train independent dimensions in one batched GPU process.
- `run_sampled_batched_d_array.sh`: map Slurm array tasks to `(loss,k,N)`.
- `plot_sampled_min_d.py`: plot the minimum dimension with positive final margin.
- `plot_sampled_margin.py`: plot the largest checkpointed positive margin.
- `plot_sampled_full.sh`: generate both figure families and their CSV summaries.

The Slurm scripts record the exact cluster paths used for this run. Set
`SAMPLED_GRAPH_ROOT`, `SAMPLED_OUT_ROOT`, and the script-path environment
variables before submission, and adjust the partition/GPU directives for a
different cluster.

## Example graph generation

```bash
python generate_iid_sparse_graphs.py \
  --n 100 \
  --k-values "4 5 6" \
  --N-values "100 200 400 800 1600 3200 6400 12800 25600" \
  --seed 0 \
  --output-dir /path/to/graphs
```

The committed figures were generated from repository commit
`81b82ca49eb7714bd435e98e850b1c5afe643a3a`.
