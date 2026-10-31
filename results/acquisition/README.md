# Frozen acquisition results

The canonical archive contains **90 runs**: six formulations × five folds ×
three acquisition seeds. Each run has 100 acquisition checkpoints. The paper's
downstream comparison uses acquisition seed 0.

All **27,000 original CSV identities** were verified against commit
`f11931c704650b8ecf5ad86a0ae69d87c82a89ec` of the existing
`SeanWan514/PORT-AS_cost_aware` repository:

- 9,000 completed strict-score matrices;
- 9,000 original incremental query logs;
- 9,000 cumulative query logs.

Payloads were recovered from the original `cost_aware_relaxed_six_methods.zip`.
`provenance/pinned_acquisition_summary.json` identifies the source and archive
hashes; `pinned_acquisition_files.jsonl.gz` identifies every logical CSV by its
Git blob and SHA-256. No acquisition experiment was rerun.

## Storage

Each `<policy>/foldN_seedS.tar.xz` stores all matrices and original incremental
logs, the final cumulative ledger, and available local diagnostics. Earlier
cumulative CSVs are stored **losslessly as prefixes of the final ledger**.
Every prefix's exact bytes match its original Git blob, including headers,
numeric serialization, and line endings. There were zero prefix mismatches.
This removes repeated storage without changing observations or results.

The 90 canonical archives total **335,727,308 bytes**. Raw matrices and query
logs expand to substantially more space. Small historical reports and aggregate
diagnostic tables are separately preserved in `historical_summaries.zip`.
Some per-checkpoint auxiliary workspace diagnostics were cloud-only; their
absence is inventoried separately and does not mean a primary CSV is missing.
Redundant full restart JSON states, smoke tests, and unrelated pilots are omitted.

## Inspect one exact original CSV

From the repository root:

```sh
python3 scripts/read_acquisition_artifact.py --method cost_tiers_5 --fold 0 --seed 0 --checkpoint 17 --kind newly --output scratch/original_incremental_017.csv
python3 scripts/read_acquisition_artifact.py --method plain_mean_ucb --fold 0 --seed 0 --checkpoint 25 --kind cumulative --output scratch/original_cumulative_025.csv
```

`--kind matrix` selects a completed matrix. Omitting `--output` writes to stdout.
The utility verifies content identities and refuses to overwrite an existing
output file.

The recovered analysis notebook expects ordinary on-disk CSV paths. Estimate
and then explicitly materialize the seed-0 archive into a new inspection area:

```sh
python3 scripts/materialize_acquisition.py --destination scratch/acquisition_seed0
python3 scripts/materialize_acquisition.py --destination scratch/acquisition_seed0 --write
```

This only unpacks original data; it does not train a model. Other seeds can be
selected explicitly. Do not confuse acquired-cell percentage with elapsed
docking cost when analyzing the checkpoints.

## Historical log defect

The original cost-tiers, fold-0, seed-0 incremental export at checkpoint 17
repeats 178 records from the prior checkpoint after a resume. It is deliberately
preserved. The cumulative ledger has unique query orders. The corrected release
runner fixes future incremental exports; it does not rewrite historical results.

The recovered calibration notebook's `load_master_events` reads the highest
available **cumulative** ledger, then removes warm-start records. Thus this
incremental duplication does not enter that recovered workflow. Final figure
assembly/export identity is documented separately in the figure-lineage map.
Use cumulative ledgers for unique query/cost accounting; do not concatenate
incremental files without auditing query-order uniqueness.
