# Offline acquisition and matrix export

This directory packages the six-policy acquisition implementation from
`MolAS_AL_restart/AL_MolAS/mean_ucb_cost_5fold_paper`. It queries existing docking
tables in simulation; it does not execute docking engines or train the downstream
MolAS selector. The numerical scoring, completion, runtime-surrogate, and
acquisition functions are preserved from the inspected source. Publication
packaging did not rerun acquisition or docking experiments or replace the
historical feature cache.

The original source files available during packaging are preserved separately in
[`../../provenance/historical_acquisition_source`](../../provenance/historical_acquisition_source/).
Its manifest records hashes for all twelve recovered historical core text files;
no auxiliary source file remains unavailable. This is a snapshot of inspected source, not a claim that its exact
revision generated every paper figure. The archived outputs provide additional
lineage evidence described in the repository's provenance documentation.

## Files and dependency boundary

| File | Purpose |
| --- | --- |
| `run_experiment.py` | Six-policy acquisition, completion, and checkpoint exports |
| `release_support.py` | Portable paths, run identity, and resume safeguards |
| `launch_all.py` | Bounded parallel launcher; invoking it starts experiments |
| `audit_inputs.py` | Read-only schema and feature-order validation |
| `finalize_audit.py` | Read-only output completeness and optional incremental-log audit |
| `summarize_results.py` | Aggregate existing acquisition diagnostics; not downstream analysis |
| `formula_tests.py` | Preserved legacy finiteness checks |
| `tests/test_release.py` | Fabricated-data regression checks; no AL loop or model fitting |

Python 3.10 or newer is required. Runtime dependencies are NumPy, Pandas, and
scikit-learn; their normal transitive dependencies are also required. Install in
an isolated environment:

```sh
python -m venv .venv
.venv/bin/python -m pip install -r code/acquisition/requirements.txt
```

`requirements.txt` preserves the original minimum-version specification. It is
not a recovered historical lockfile and does not promise identical floating-point
results on every platform. The release's validation record identifies the
environment used for its fabricated checks. Every new run records its actual
Python/package versions and source hashes.

The acquisition runner imports no other `AL_MolAS` source package. PyTorch,
torch-geometric, pretrained feature models, and DAL-toolbox are not needed when
the prepared feature array is supplied. The original feature-building functions
are preserved as a **non-executable provenance excerpt**, with dependency notes.
Data validation compares the existing cache with its source transformation; the
archived cache itself is retained unchanged.

The preserved feature builder normalizes over the supplied complex IDs before
random projection. The archived cache therefore must not be described as having
fold-local feature preprocessing. This is separate from the runtime surrogate's
`StandardScaler`, which is fitted only on already observed cells. The runner
consumes the prepared cache unchanged. The data provenance directory records the
independent alignment check of that cache against the existing molecular inputs.

## Required inputs

Paths are relative to the repository by default and can be overridden explicitly.

| Option | Default | Contract |
| --- | --- | --- |
| `--data-dir` | `data/raw` | Three CSV matrices listed below |
| `--features` | `data/features/complex_features_ligand_protein_d256_seed0.npy` | Finite `N × 256` numeric array, no pickle |
| `--feature-ids` | `data/features/complex_feature_ids.csv` | One-column CSV with producer-verified complex IDs in feature-row order |
| `--descriptors` | `data/descriptors/moad_physical_descriptors_filtered.csv` | Identifier plus exactly four numeric descriptors |
| `--fold-split` / `--fold-file` | `data/splits/moad_5fold_split_seed42.csv` | Identifier plus one integer fold column, encoded 0–4 or 1–5 |
| `--output-dir` | `outputs/acquisition` | New outputs; keep historical archives separate |

The three raw matrix names are `moad_rmsd_trimmed.csv`,
`moad_pb_ratio_trimmed.csv`, and `moad_runtime_trimmed.csv`. Their first column
contains complex IDs. Algorithm columns use suffixes `_rmsd`, `_pb_ratio`, and
`_runtime`, respectively, for `surfdock`, `unimol`, `diffdock`, `diffdock_l`,
`gnina`, `autodock`, and `smina`. Runtime values are finite nonnegative seconds.
The three matrices must have identical ordered, unique identifiers.

Descriptors and folds are aligned by identifier. Molecular features require an
**independently established ordered-ID mapping**, not merely the correct number
of rows. The runner fails before acquisition if the feature ID file is missing
or mismatched. Do not manufacture this file from raw-matrix order when the
feature-cache order is unverified. The repository's raw-matrix ID list does not
by itself prove cache alignment. All required acquisition inputs and the independently verified feature-ID mapping
are included in this package; no synthetic input is substituted.

If using the documented recovered held-out membership file, supply it explicitly:
`--fold-split data/splits/recovered_heldout_folds_from_downstream.csv`. This is not
represented as recovery of the original seed-42 splitting procedure.

## Validation without experiments

From the repository root, these commands do not execute an acquisition loop:

```sh
python -m unittest discover -s code/acquisition/tests -v
python code/acquisition/run_experiment.py --help
python code/acquisition/audit_inputs.py --help
```

Once all authentic inputs and their ordered feature mapping are available:

```sh
python code/acquisition/audit_inputs.py --fold-split PATH_TO_VERIFIED_FOLDS
python code/acquisition/run_experiment.py --validate-only --fold-split PATH_TO_VERIFIED_FOLDS
```

The unit suite uses temporary fabricated arrays/records. Runtime-predictor tests
replace fitting with a fake predictor. No docking, SVD fitting, MLP fitting, real
benchmark acquisition, or downstream selector training occurs in these checks.

## Explicit experiment commands

The following commands **would run new offline acquisition experiments**. They
are provided for future use and were not run during publication packaging.
They require all authentic inputs described above; add path overrides as needed.

```sh
# One method/fold/seed, in a fresh output directory:
python code/acquisition/run_experiment.py --methods plain_mean_ucb --folds 0 --seeds 0 --output-dir outputs/new_acquisition

# All six methods, five folds, three acquisition seeds (90 runs):
python code/acquisition/launch_all.py --workers 3 --output-dir outputs/new_acquisition
```

`--smoke-test` is also a real acquisition run on the first 120 training complexes,
not a data-free unit test. It writes under `smoke/` rather than `runs/` and is
explicitly identified in metadata. The launcher retains the historical
single-thread numerical-library limits in each worker. It no longer suppresses
all warnings. Completed runs are skipped only after matching input/code/
environment/job provenance; an existing legacy output directory is never adopted
as a validated release run.

For existing release outputs, use:

```sh
python code/acquisition/finalize_audit.py --output-dir outputs/new_acquisition --check-incremental
python code/acquisition/summarize_results.py --output-dir outputs/new_acquisition
```

Both accept explicit `--methods`, `--folds`, and `--seeds`. The summary retains the
historical descriptive fold-seed means/standard deviations and requires the
plain Mean-UCB baseline. These are not confidence intervals over independent
replicates. The completeness audit reports file-count scope unless
`--check-incremental` is supplied; that option additionally reads incremental logs
and detects repeated prior blocks across checkpoints. Neither audit certifies
historical figure lineage or downstream training.

## Numerical definitions retained

- Strict score is the clipped product of the normalized RMSD sigmoid with
  `pb_ratio == 1.0`; lambda is 3. Partial PB validity is not accepted as valid.
- Acquisition uses SVD ranks 1, 2, 3, and 5, each with four completion iterations;
  dispersion is population standard deviation. Checkpoint matrices in this
  inspected implementation use rank 3 and six iterations. These code facts are
  not an unsupported attribution of the exact historical figure-generating run.
- Mean-UCB uses beta 1. Plain and direct cost-utility use raw utility; the other
  four variants use ascending average-tie percentile ranks, floored at 0.001.
- Candidate costs are min–max scaled to `[1, 5]` where used; numerically equal
  extrema map to 1. Five-tier boundaries are candidate-cost quintiles, with ties
  assigned to the lower tier. Direct division adds `1e-6` to predicted seconds.
- Runtime prediction fits only revealed runtimes. Each prediction is transformed
  from log seconds and clipped to observed 5th/95th percentile bounds. The
  uncertainty policy uses three models; their mean plus population SD is clipped
  again before cost scaling. Other policies use one model.
- The runtime MLP retains widths `(32, 16)` and at most ten iterations per refit.
  Acquisition re-ranks after batches of at most 50 cells, truncated at integer
  percentage checkpoint boundaries. Cumulative cost includes the warm start.

## Publication changes and provenance limits

1. **Portable inputs and output location.** The inherited sibling-folder paths
   became explicit CLI options. Ordered feature IDs, finite feature values,
   integer fold labels, unique matrix IDs, and valid runtimes are checked. Valid
   scientific computations are unchanged.
2. **Correct incremental export after resume.** The next export starts after the
   last completed checkpoint, rather than the checkpoint before it. The historical
   defect was observed in cost-tiers/fold-0/seed-0 checkpoint 17: its incremental
   CSV repeated 178 earlier records. Its cumulative records and matrices remained
   correct. The release does not alter that archived CSV or assert that the defect
   changed any published figure.
3. **Keep pending audit records on resume.** Round diagnostics for a partly
   completed checkpoint are included in saved state, so resuming does not discard
   the earlier rounds of that checkpoint's diagnostic JSON.
4. **Actual scope and run identity.** Each invocation gets a separate manifest
   containing its selected methods/folds/seeds and smoke/resume mode. Per-run
   manifests bind input hashes, source hashes, actual environment, and job
   settings. A completed marker alone cannot silently skip a changed run. A
   legacy directory without release provenance is rejected; use a fresh output
   location. `--no-resume` also refuses to overwrite an existing run.
5. **Focused verification.** Fabricated checks cover exact formula contracts,
   clipping order, data identity, resume-export uniqueness, and metadata. An AST
   comparison verifies the scientific functions against the preserved snapshot.

The historical runtime-feature-ablation script is not the paper's engine-specific
ridge-probe implementation. Neither it nor the downstream MolAS/figure pipeline
is fabricated here. The exact historical training environment, source revision,
and every source-to-figure link have not been recovered. Consult the repository's
data and downstream documentation for available artifacts and remaining inputs.
