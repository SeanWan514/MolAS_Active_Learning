# Downstream results and reproduction status

The release now includes the recovered completed-matrix MolAS training/evaluation source, two relevant analysis notebooks, saved downstream results, and the eight final rendered figure assets. A later Downloads audit superseded the initial workspace-only source gaps. Recovery preserves existing files; no docking, acquisition, selector training, ridge fitting, statistical testing, or plot regeneration was performed during packaging.

The recovered source snapshot does not establish the exact executed revision/settings for every saved run. Several final figure components can now be matched exactly to saved analysis outputs; the remaining final rendering/assembly links are identified separately below.

## Saved result scope

The historical source directory is `port_as_results_260829`. Six paper policies have saved output directories labelled `seed0`, with five outer-fold directories per checkpoint. This label identifies the acquisition archive. The recovered launcher specifies selector seed 42, but a per-run configuration record for every archived result has not been recovered. Acquisition artifacts for other seeds elsewhere in the release are not additional downstream replications.

| Policy | Saved checkpoints | Original CSVs | Packaged CSVs | Unavailable CSVs |
|---|---:|---:|---:|---:|
| `plain_mean_ucb` | 30 | 840 | 840 | 0 |
| `direct_cost_uc` | 27 | 756 | 756 | 0 |
| `relaxed_cost_power_05` | 27 | 756 | 756 | 0 |
| `log_cost` | 27 | 756 | 756 | 0 |
| `cost_tiers_5` | 27 | 756 | 756 | 0 |
| `runtime_uncertainty` | 23 | 644 | 644 | 0 |
| **Total** | **161** | **4,508** | **4,508** | **0** |

All 4,508 original CSV payloads were recovered and packaged on 31 October 2026. The original transferred `port_as_results_260829.zip` supplied 1,122 payloads that were unavailable in the workspace because of iCloud placeholders; all 3,386 readable workspace CSVs matched the original ZIP byte for byte. See `provenance/downstream_original_archive_recovery.json` for the original archive SHA-256, every recovered member hash, and the 144 excluded nonpaper hybrid CSVs. The six release ZIPs contain 240,471,093 bytes of original CSV payloads and total 72,399,169 bytes compressed. The packaging input was a verified extraction of that archive: manifest relative paths identify the historical result tree, while recorded filesystem flags describe the packaging input. `results/downstream/verification.json` confirms every packaged member and summary mirror; source files were not regenerated.

The four 27-checkpoint policies contain `b001_pct`–`b011_pct`, followed by `b013_pct`, `b015_pct`, `b017_pct`, `b019_pct`, `b020_pct`, `b025_pct`, `b030_pct`, `b035_pct`, `b040_pct`, `b045_pct`, `b050_pct`, `b060_pct`, `b070_pct`, `b080_pct`, `b090_pct`, and `b100_pct`. `plain_mean_ucb` additionally contains `b012_pct`, `b014_pct`, and `b016_pct`. The saved `runtime_uncertainty` directory lacks `b060_pct`, `b080_pct`, `b090_pct`, and `b100_pct` from that 27-checkpoint set. These labels denote acquired-cell percentages, **not hours or prespecified realized-cost stopping points**. Development directories `hybrid_b1` and `hybrid_b50` are excluded because they are not among the paper's six policies.

## Files and inspection

- `results/downstream/<policy>_seed0.zip` contains original, byte-preserved CSVs under `<policy>/seed0/bNNN_pct/test_composite/default/...`.
- `results/downstream/summary_tables/` mirrors available small endpoint summaries without modifying their contents.
- `results/downstream/file_manifest.csv.gz` records every expected source CSV, its status, archive member, SHA-256 when available, original byte count, and observed CSV columns/row count.
- `results/downstream/checkpoint_coverage.csv` records the available fold-level metric, prediction, and descriptive tables at each checkpoint. A fold directory's existence alone does not imply that all its files were packaged.
- `results/downstream/archive_summary.json` records archive hashes, totals, exclusions, and the distinction between acquisition labels and unverified selector seeds.

For example, from the repository root:

```bash
python3 -m zipfile -l results/downstream/cost_tiers_5_seed0.zip
python3 -m zipfile -e results/downstream/cost_tiers_5_seed0.zip /your/results-inspection-directory
```

Every saved checkpoint is fully packaged, including all 28 original CSVs at `cost_tiers_5/seed0/b001_pct` and `plain_mean_ucb/seed0/b100_pct`. The split recovery in `data/splits/recovered_heldout_folds_from_downstream.csv` uses saved held-out IDs, not a newly generated random split; see `data/provenance/recovered_split.json` for the primary source hashes and cross-checks. The original `data/splits/moad_5fold_split_seed42.csv` was subsequently recovered, and all 3,179 memberships agree exactly with this independent reconstruction; see `data/provenance/hydrated_original_inputs.json`. The held-out fold sizes are 636, 636, 636, 636, and 635.

## Table meanings and interpretation boundaries

Each complete checkpoint contains three overall CSVs plus five files per fold. `*_predicted_scores.csv` contains IDs, seven engine score columns, and the selected engine; preserve the raw score scale rather than treating it as probabilities. `*_descriptive.csv` contains per-complex outcomes and selections. `*_full.csv` contains endpoint values for engines and selectors. The unsuffixed `best-epoch=...csv` records selected-endpoint/gap metrics. `overall_summary_metrics.csv` has a different orientation from the fold-level summaries. Filenames alone do not recover hyperparameters or training seeds.

The recovered primary notebook reads pose-success endpoints from each fold's `*_full.csv` and composite outcomes from `*_descriptive.csv`. It normalizes SBS–VBS gaps within each fold before averaging. A fold-normalized mean generally differs from a ratio of pooled endpoint means. Its cost curves use linear interpolation on each method's common five-fold realized-cost support, without extrapolation; an interpolated point is not a separately trained selector. Overlapping training folds and repeated checkpoints are not independent replications.

## Recovered training and evaluation adapter

`code/downstream/historical_molas/` contains 27 unchanged source/configuration/notebook files from `Downloads/MolAS_AL-main`, including the original MIT license. `historical_molas_manifest.json` records file hashes and source paths. The source snapshot has no recovered Git commit. Python bytecode caches were excluded. The primary analysis notebook also contains exploratory sections beyond the paper; their presence is not a claim that those analyses were reported or newly executed.

The recovered `main.sh`, `main.py`, and `data/data_interface.py` establish this implemented workflow:

1. The launcher verifies that the five saved fold matrices contain precisely the exported outer-training IDs, in order, and that the exported folds agree with its configured five-fold split.
2. Training uses `label_source=al` and an explicit fold/checkpoint `score_matrix_path`. `test_size=0` avoids splitting the already fold-specific training matrix into new outer folds. Internally, 10% of those rows form the validation subset; their targets also come from the completed matrix.
3. Held-out testing is a separate call using `label_source=composite`, the raw outcome tables, and the corresponding outer fold. Thus the recovered adapter supports completed-target training and true held-out evaluation.
4. The recovered launcher specifies model `MolAS`, selector seed 42, learning rate 0.001, batch size 64, and maximum 300 epochs. These are observed source settings, not a recovered per-run configuration for all 161 saved checkpoints. Its current active list contains only `runtime_uncertainty` and `cost_tiers_5` at 70/80/90/100%.
5. The source keeps the top three checkpoints monitored by validation `val_as_acc`; the shell selects the highest-epoch file among available `best-...` checkpoint files for testing. This selection rule should be preserved or explicitly changed in future work, rather than replaced silently with a different convention.

The source's seed 42 should not be confused with the acquisition directory label `seed0`. The original outer split is independently corroborated by every recovered test membership; this does not, by itself, prove every training RNG setting or environment. No model checkpoints or per-run hyperparameter YAMLs were found in the audited saved-results/source directories.

## Two distinct target definitions

The recovered downstream `data/dataset.py:rmsd_score` uses the normalized sigmoid with lambda 3 but sets scores to zero when RMSD exceeds 5 Å or is negative; the PoseBusters gate requires exact validity. The acquisition implementation uses the normalized sigmoid times that gate and clips to [0,1], without the same >5 Å truncation. Do not silently alias these target roles.

The primary notebook explicitly separates `GROUND_TRUTH_PATH` (native MolAS evaluation scores) from `ACQUISITION_GROUND_TRUTH_PATH` (the acquisition oracle used for matrix diagnostics). Its reconstruction comparisons use the latter; its downstream endpoints are read from saved held-out results.

`data/targets/moad_score_native_reconstructed.csv` is explicitly reconstructed from the unchanged raw tables using the recovered native formula. All 22,253 per-engine values were checked against the five saved descriptive tables covering all 3,179 complexes, with maximum absolute discrepancy 1.11e-16; see `data/provenance/native_evaluation_target.json`. It is not presented as the recovered original `moad_score_trimmed.csv` byte stream. Acquisition strict targets remain separate, with their own provenance. Path configuration requires both roles explicitly rather than guessing an alias.

## Figure sources and remaining rendering links

Cell references below are **zero-based notebook indices**. `provenance/downstream_notebook_index.json` records source cells, execution counters, and saved image hashes; a saved counter/output is not proof that later-edited cell source was executed unchanged. No notebook was executed during packaging.

| Figure | Recovered source/evidence | Remaining boundary |
|---|---|---|
| 1 | `port_as_analysis.ipynb` cells 10, 17, 27: saved downstream endpoints, common-cost interpolation, and `downstream_performance_vs_cost` plotting. | Exact final `Fig1.png` rendering/export not byte-matched. |
| 2 | Primary notebook cells 8, 34, 35: matrix diagnostics, cost alignment, and within-fold Spearman correlations across available methods (at least four). | Exact final `Fig2.png` rendering/export not byte-matched. |
| 3 | Primary notebook cell 16 and the saved `runtime_distribution.png` analysis export. | The final asset is `runtime_distribution-REV.png`; it is not identical to that earlier export. |
| 4 | `moad_runtime_physical_and_embedding_analysis.ipynb` cells 23, 25, 26: embeddings, ridge probes, and heatmap. The final PDF's embedded raster is pixel-identical to cell 26's saved output. | Probe source/configuration is recovered; the precise executed source revision is not independently rerun/verified. |
| 5 | Primary notebook cells 45, 46, 48: cumulative-log calibration and signed-error trend. Its saved table reports 88,117 records per cost-aware policy, totaling 440,585. The PDF's bottom raster exactly matches the saved signed-calibration PNG. | Final top/middle raster versions and PDF assembly are not exactly matched. |
| 6 | Primary notebook cells 18–19: cumulative cost share and cost-phase query allocation. The PDF's top raster exactly matches saved `cumulative_cost_share.png`. | Exact lower heatmap version and PDF assembly remain unverified. |
| 7 | Primary notebook cell 17. The final image is byte-identical to saved `queried_cells_vs_cost.png`. | Plot source is recovered; no new execution was performed. |
| 8 | Primary notebook cell 27 provides the downstream plot family and common-cost summaries. | The precise first-20-hours export/layout command was not recovered. |

The ridge source uses natural log runtime, five shuffled outer folds with random state 42, a training-fold `StandardScaler` followed by `RidgeCV` over 13 alphas from 1e-3 to 1e3, and separate physical/protein/ligand/combined feature sets. These are the full 1,536-dimensional protein-plus-ligand representations for this diagnostic, not the projected 256-dimensional acquisition cache. Saved probe values and the raster match are retained without refitting.

The recovered calibration code reads the highest available **cumulative** query file per method/fold, computes cumulative realized runtime, excludes round-zero warm-start records, and drops unavailable runtime predictions. The known repeated block in one historical **incremental** checkpoint file is therefore not an input to this recovered calibration workflow. Original records remain unchanged. Signed-error trend bins use pooled realized-cost quantiles with a minimum bin size; records can repeat the same historical cells across policies and overlapping folds.

Figure-component evidence is in `provenance/notebook_figure_pixel_comparison.json`; all eight asset hashes and source references are in `provenance/figure_lineage.json`. Eighteen existing analysis PNG exports are preserved under `results/downstream/recovered_analysis_figures/`, with their own manifest. Two additional manuscript runtime scripts are preserved under `code/downstream/historical_manuscript_scripts/`; they are not automatically attributed to a final figure.

## Portable setup without execution

Use the standard-library `code/downstream/prepare_local_copy.py` with `local_paths.example.json` to configure a new source copy. It changes paths only, creates raw/graph/ligand aliases, and never executes models or notebooks. The byte-preserved historical source remains untouched. The primary notebook is configured only with two explicit distinct target files and an expanded saved-results root; its configured copy's old outputs are cleared. The separate runtime-probe notebook retains historical paths and is not automatically configured. See `code/downstream/README.md` for exact preparation commands.

Acquisition archives are `results/acquisition/<method>/fold<fold>_seed<seed>.tar.xz`. `scripts/read_acquisition_artifact.py` accesses the original CSV identities, including cumulative checkpoints represented as losslessly verified prefixes of the full ledger. Follow the repository's materialization instructions to prepare the `<method>/fold<fold>/seed0/` tree expected by the recovered launcher/notebook. Expand the six downstream ZIPs into one root retaining their original policy-relative paths. These archives contain complete payloads and do not depend on the original workspace's cloud-placeholder state.

Future use of `main.sh` starts new training/testing. It loops over selected methods/checkpoints and five outer folds; to cover the paper's saved scope, use the six policy names and their actual checkpoint sets from the coverage inventory. The recovered defaults select only a subset, so do not interpret one launch as all 161 saved checkpoint evaluations. Configuring paths does not recover missing per-run execution records or create a validated full reproduction.

`code/downstream/upstream_molas/` is an additional pinned official reference source, not a substitute for the recovered completed-matrix adapter. All saved CSV payloads in the six-policy scope are included. The remaining reproducibility gaps are per-run execution/configuration records and the unmatched final rendering/assembly details. Available source, preserved outputs, and exact historical regeneration remain distinct claims.
