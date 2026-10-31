# Mean-UCB Cost-Aware Five-Fold Paper Experiments

This isolated package runs the final paper experiment agreed with the RA:

- five MolAS training folds;
- seeds `0`, `1`, and `2`;
- every checkpoint from 1% through 100%;
- plain Mean-UCB plus five cost-aware variants;
- leakage-safe runtime prediction from queried runtimes only;
- molecular embeddings concatenated with four physical descriptors and algorithm one-hot encoding.

## Required additional inputs

Place these RA-provided files in `inputs/`:

- `moad_physical_descriptors_filtered.csv`
- `moad_5fold_split_seed42.csv`

The runner validates identifiers, fold membership, dimensions, and missing values before writing any experiment output.

## Methods

- `plain_mean_ucb`: cost-unaware reference.
- `direct_cost_uc`: raw Mean-UCB utility divided by predicted runtime.
- `runtime_uncertainty`: utility divided by the square root of conservative predicted cost (`mean + SD`).
- `cost_tiers_5`: utility divided by a five-level predicted-cost tier.
- `relaxed_cost_power_05`: utility divided by square-root predicted cost.
- `log_cost`: utility divided by `log1p` transformed predicted cost.

## Commands

```bash
python audit_inputs.py
python runtime_feature_ablation.py
python run_experiment.py --smoke-test --folds 0 --seeds 0
python launch_all.py
python finalize_audit.py
```

All writes are atomic. Each micro-batch and checkpoint saves resumable state. Re-running `launch_all.py` skips complete runs and resumes incomplete runs.
