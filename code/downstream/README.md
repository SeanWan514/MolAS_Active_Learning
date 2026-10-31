# MolAS downstream code and saved analyses

`historical_molas/` preserves the recovered `Downloads/MolAS_AL-main` source snapshot byte for byte: the completed-matrix training adapter, held-out evaluator, launcher, model/loss modules, dependency files, and analysis notebooks. Its original MIT license is included. `historical_molas_manifest.json` records all 27 source files and hashes; Python bytecode caches are excluded. The snapshot includes exploratory analyses beyond the paper. Its presence establishes available code, not execution of every notebook section or the exact configuration of every archived run.

`historical_manuscript_scripts/` preserves two additional runtime-plotting scripts from the manuscript source folder. Their final-figure role is not established. `upstream_molas/` is a separate, unchanged official MolAS reference snapshot at commit `a6c417216ddb992f7dc513d511d0429aece4bd61`, with MIT license and `upstream_molas_manifest.json`; it lacks the recovered completed-matrix adapter and is not the primary AI4DD workflow.

See [`../../docs/downstream_reproduction.md`](../../docs/downstream_reproduction.md) for the recovered training protocol, source-to-figure evidence, saved CSV coverage, and remaining provenance limits. [`../../provenance/figure_lineage.json`](../../provenance/figure_lineage.json) maps all eight figures to preserved assets and relevant notebook cells.

## Prepare portable paths without executing experiments

`prepare_local_copy.py` creates a **new configured copy** of the recovered sources. It never imports MolAS/PyTorch, executes a notebook, fits a model, or changes historical sources. Copy `local_paths.example.json` to your own JSON, supply authentic paths, and run:

```bash
python3 code/downstream/prepare_local_copy.py --config /absolute/path/to/your_paths.json --output /absolute/path/to/a_new_configured_molas_directory
```

Paths in the JSON are resolved relative to that JSON file. The example's relative paths assume it remains in `code/downstream`; update them if you move it. The acquisition path must contain expanded `<method>/fold<fold>/seed0/strict_score_matrices/strict_score_NNNpct.csv` directories. Use the acquisition artifact reader/extraction instructions to obtain those existing files; passing an archive directory alone is insufficient.

The utility configures dataset roots, raw/graph/ligand file links, and launcher paths. It preserves the recovered launcher's scientific settings and currently active **two-method, four-checkpoint subset**, rather than silently enabling a large new experiment. Review `main.sh` before any future execution. Activating its launcher would start new training and testing; preparing the copy does not.

Add `--include-analysis` only after resolving both target paths and the expanded downstream output root. `molas_score_matrix` and `acquisition_score_matrix` are deliberately separate required roles: the native MolAS evaluation score truncates RMSD above 5 Å to zero, while the acquisition strict score does not. The example leaves them unset to prevent an incorrect automatic alias. The native reconstruction in `data/targets/moad_score_native_reconstructed.csv` is independently checked against saved per-complex outcomes; see its provenance before selecting it. The primary notebook's paths are updated and its **configured copy's** old outputs are cleared. The original notebook, including saved outputs, remains unchanged. The separate runtime-probe notebook retains historical paths and is not automatically configured.

## Saved-output packaging utility

`catalog_saved_outputs.py` was written for this release. It inventories and byte-copies existing CSVs; it is not the historical trainer/evaluator or plotting code.

```bash
python3 code/downstream/catalog_saved_outputs.py --source /absolute/path/to/port_as_results_260829 --output /absolute/path/to/MolAS_NIPS/results/downstream
python3 code/downstream/verify_saved_outputs.py --write-report --require-complete --original-archive-manifest provenance/downstream_original_archive_recovery.json
```

The verifier checks every archive/member hash, member CRC, and generated summary mirror. With the original archive manifest supplied, it also checks every member against the independently recovered original ZIP identities. It distinguishes integrity from completeness; `--require-complete` returns nonzero while original CSV payloads remain unavailable. Without `--write-report`, it does not modify the package.

The release contains all 4,508 saved CSVs for the six paper policies. The verified original `port_as_results_260829.zip` resolved earlier cloud-only gaps; archive/member evidence is recorded in `provenance/downstream_original_archive_recovery.json`. To rebuild, use a fully readable extraction of that historical result tree as `--source`.

Python's standard library is sufficient for these utilities. Source and archive output must be separate non-nested directories. iCloud placeholders are explicitly inventoried rather than treated as recovered payloads. No scientific experiments were executed during release packaging.
