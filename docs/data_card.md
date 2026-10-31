# MolAS active-acquisition dataset card

This card documents the GitHub release's data schema and provenance. The author confirmed applicable redistribution terms; see the [confirmation record](../provenance/author_redistribution_confirmation.json). Exact end-to-end historical reproduction has not been newly verified. Original acquisition inputs have been recovered and hash-verified; numerical values are preserved. Existing feature arithmetic was independently repeated only as an integrity check. No docking, active-learning runs, or learned-model training was performed.

## Scope

The benchmark comprises 3,179 MOAD-curated protein–ligand complex identifiers and seven docking algorithms: `surfdock`, `unimol`, `diffdock`, `diffdock_l`, `gnina`, `autodock`, and `smina`. Each of the three raw tables contains 22,253 measurements in the same row/algorithm order. These are historical outcomes used as an offline oracle for cell-wise acquisition. Their public availability must not be confused with permission for an acquisition policy to access unrevealed outcomes or held-out labels.

The package supports inspecting the existing outcome/runtime matrix, auditing saved acquisition/downstream results, and preparing reproducible inputs. It does not establish performance on other benchmarks, hardware, portfolios, or prospective docking campaigns.

## Data dictionary

| File/field | Meaning |
|---|---|
| All raw tables: `PDB_CCD_ID` | Original compound complex identifier, e.g. `19gs_BSP_A_210`. Preserve it as a string; it includes PDB, ligand/CCD, chain and residue components. |
| `raw/moad_rmsd_trimmed.csv`: `<algorithm>_rmsd` | Historical docking RMSD. The manuscript interprets RMSD in Angstrom. Values are preserved exactly, including extreme finite values. |
| `raw/moad_pb_ratio_trimmed.csv`: `<algorithm>_pb_ratio` | Historical PoseBusters validity ratio. PB validity for scoring is the exact rule `ratio == 1.0`, not a tolerance or rounding rule. |
| `raw/moad_runtime_trimmed.csv`: `<algorithm>_runtime` | Historical per-cell runtime, treated as seconds by the experiment code (summed seconds / 3600 for hours). Hardware and precise timing boundaries still require original acquisition documentation. |
| `targets/full_strict_score_matrix.csv`: `<algorithm>_strict_score` | Original restart target; deterministic score derived from RMSD and PB validity, higher is better. |
| `targets/strict_score_matrix.csv`: `<algorithm>` | Earlier target table retained unchanged, with slightly different decimal serialization and column names. |
| `targets/moad_score_native_reconstructed.csv`: `<algorithm>` | Reconstructed **native downstream evaluation** score using the recovered MolAS source's RMSD >5 or <0 zeroing rule. Distinct from the acquisition targets above. |
| `ordered_complex_ids.csv`: `PDB_CCD_ID` | The raw-table row sequence only. Do not use it as unverified feature provenance. |
| `splits/moad_5fold_split_seed42.csv`: `row_index`, `protein`, `ligand`, `instance_id`, `test_fold` | Original split; zero-based row index, original compound ID, and test-fold label `0`–`4`. `protein + '_' + ligand == instance_id`. |
| `splits/recovered_heldout_folds_from_downstream.csv`: `PDB_CCD_ID`, `fold` | Recovered existing test-set membership, with folds `0`–`4`; a row is held out when its fold matches the evaluated outer fold. |
| `descriptors/moad_physical_descriptors_filtered.csv` | `PDB_CCD_ID` plus `ligand_heavy_atoms`, `ligand_rotatable_bonds`, `protein_residues`, and `pocket_heavy_atoms_r6`. These are the original four numeric descriptor columns; exact generation conventions should be taken from the upstream preprocessing documentation. |
| `features/complex_features_ligand_protein_d256_seed0.npy` | Original 3,179 × 256 float32 cache, unchanged. |
| `features/complex_feature_ids.csv`: `PDB_CCD_ID` | Cache row order verified against the source transformation for all 3,179 rows. |
| `features/ligand_chemberta_moad3179.pt` | Original PyTorch ZIP archive with `ligand_ids` and a 3,179 × 384 float embedding tensor. |
| `features/ligand_feature_ids.csv`: `PDB_CCD_ID` | The ligand archive's verified embedded ID sequence. It equals the raw table order. |
| `provenance/protein_graph_inventory.csv` | One filename/complex, logical bytes, and cloud-placeholder status for the historical restart copy; no graph values or hashes are inferred from filenames. |

The continuous acquisition score is

`strict_score = ((1 + exp(-6)) / (1 + exp(3 * (rmsd - 2)))) * (pb_ratio == 1.0)`.

Both target files are unchanged copies of their recorded sources: the original restart `ground_truth/full_strict_score_matrix.csv` and the earlier `MolAS_AL/data/processed/pilot1a/strict_score_matrix.csv`. Both contain all 22,253 values and are checked against the raw-data formula. Their column suffixes and decimal serialization differ; they are not claimed to be byte-identical. Runtime does not enter this target formula.

## Native evaluation target is separate

The recovered downstream source, `MolAS_AL-main/data/dataset.py`, defines `rmsd_score` as zero when RMSD is greater than 5 or less than 0; otherwise it uses the same normalized sigmoid. `pb_score` again requires the ratio to equal 1 exactly. An RMSD of exactly 5 retains its sigmoid value. This native evaluation definition differs from the continuous acquisition target and must not be silently interchanged with it.

The historical analysis notebook explicitly separates `GROUND_TRUTH_PATH` (server-side `moad_score_trimmed.csv`) from `ACQUISITION_GROUND_TRUTH_PATH` (a full acquisition score matrix). The original `moad_score_trimmed.csv` was not found in the relevant local source/data folders. `data/targets/moad_score_native_reconstructed.csv` therefore has an explicit reconstruction name and provenance: it was deterministically calculated from the unchanged raw RMSD/PB tables using the recovered native source rules. It is not represented as the recovered original CSV.

All 22,253 reconstructed native engine scores were independently compared with the seven engine-score columns in the five saved `cost_tiers_5/seed0/b001_pct` downstream descriptive tables. These tables jointly cover all 3,179 complexes exactly once. The maximum absolute difference is 1.11e-16, with zero mismatches at a 1e-12 tolerance. The continuous acquisition and native definitions differ by up to approximately 0.00012097 in these data; source tables and existing scientific results were not modified. Source-function excerpts, file hashes, and validation references are in `data/provenance/native_evaluation_target.json` and `native_score_functions.py.txt`.

`scripts/reconstruct_native_evaluation_target.py --check-only` verifies this derived file. Without `--check-only`, the utility writes only the explicitly named reconstructed target. This is deterministic data preparation, not a docking or model-training experiment. Supply native and acquisition paths separately to analysis tooling; do not silently alias either to the other.

## Sources and recovery

The three raw matrices were copied byte-for-byte from the resident `MolAS_AL/data/raw` directory. All three are now confirmed hash-identical to the corresponding recovered `MolAS_AL_restart/AL_MolAS/data/raw` inputs. The three raw matrices, original split, original descriptors, and combined cache also match all six hashes in the preserved original `input_audit.json` and `experiment_manifest.json`. `data/provenance/raw_sources.json` and `hydrated_original_inputs.json` record exact sources. Initial bounded-read failures remain recorded as history and no longer describe current input availability.

While the original split was initially unavailable, existing test membership was recovered from the five `cost_tiers_5/seed0/b001_pct/test_composite/default/fold_*/..._descriptive.csv` files. Their `protein` and `ligand` fields reconstruct the original complex IDs. The resulting sets are disjoint, cover all 3,179 raw IDs, and contain 636/636/636/636/635 rows. Ordered IDs also agree with ten additional resident checkpoint files. The original split was subsequently hydrated: every original `instance_id`/`test_fold` pair matches the independent recovery exactly, with no label offset required. Original `row_index` is 0–3178 and follows raw-table order. Both files are retained. The recovery did not generate a new split. Evidence is in `data/provenance/recovered_split.json` and `hydrated_original_inputs.json`.

The ligand archive was copied unchanged from the restart structural-data folder. ZIP entries and pickle opcodes were inspected without unpickling or executing objects. Its explicit 3,179 `ligand_ids` match the raw ID order, and the single float storage contains 1,220,736 finite values consistent with the recorded 3,179 × 384 shape. The combined projected feature cache was checked separately as described below.

## Combined-feature verification and structural inputs

The original four-descriptor CSV and combined 256-dimensional NPY cache were subsequently hydrated and copied unchanged. The cache's SHA-256 is `10b4a50f5cfc3de000ab06d0b8112146486e63571ad9db252455dd5af3976a91`, matching the historical input audit. Its row order was independently verified, rather than inferred from filename or shape.

`scripts/verify_feature_cache_alignment.py` uses an allowlisted metadata decoder that replaces PyTorch/PyG classes with inert records and reads numeric tensor ZIP storages; it does not deserialize executable PyTorch/PyG objects. It reads the explicit ligand IDs, pools existing graph features in the original ID order, and repeats the existing producer's normalization and seed-0 Gaussian projection. The seven original producer functions are preserved in `provenance/historical_acquisition_source/feature_builder_excerpt.py.txt`, with original-file hashes and line ranges in `feature_builder_provenance.json`. All 813,824 cache values across all 3,179 rows agree within the predeclared combined tolerance `abs(error) <= 2e-5 + 2e-5 * abs(reference)`. Maximum absolute difference is 2.19345e-5 and RMS difference is 2.32080e-6, consistent with float32 reduction differences between NumPy and the original PyTorch operations. The arrays are not bit-identical. The original cache was never rewritten. `complex_feature_ids.csv` was emitted only after this comparison passed; the full evidence and hashes are in `feature_cache_comparison.json` and `combined_feature_alignment.json` under `data/provenance`.

This check also identifies the historical cache preprocessing: normalization uses the fixed full set of 3,179 feature rows before projection. It must not be described as a scaler fitted separately within each training fold. No alternative representation or ligand-only cache was substituted.

Large protein graphs comprise 3,179 files and 14,985,036,224 logical bytes. The restart directory contains 2,414 cloud placeholders, but a separate fully resident source was located at `/Users/wanweilin/Downloads/data/MolAS/protein_graphs_esmc_600m`. The packaging process cloned that source locally into `data/external/protein_graphs_esmc_600m`: all filenames and logical sizes agree, and the local clone has zero cloud placeholders. These large assets are kept outside the small Git-tracked dataset and have not been publicly uploaded. Refer to the parent package's graph hash manifest for payload verification. The original compressed structural archive is also outside this package. Graph availability and cache lineage are separate checks; the independent numerical cache comparison is documented above, and no replacement cache was substituted.

No physical descriptors were invented, no embedding model was run, and no existing feature cache was replaced. Deterministic feature arithmetic was repeated solely to compare against the existing cache. No docking results or random splits were generated. Check `data/provenance/validation_report.json` for current completeness rather than assuming a directory name means its input is available.

## Data quality and interpretation

All three raw tables contain 3,179 unique IDs, aligned row order, and finite numeric values. Runtime is strictly positive and PB ratios lie in [0,1]. Raw RMSD includes extreme finite values, with a maximum approximately 1.5964e17 Angstrom. These were preserved; their possible failed-run or sentinel meaning has not been established. They must not be silently removed, clipped, or reinterpreted as validated physical poses.

Runtime is hardware/workflow dependent. The exact engine versions, compute allocation, startup/preprocessing inclusion, repeated-run aggregation, and failure/timeout conventions require source-author documentation. The five folds are existing complex-level memberships; this package does not certify protein-family or ligand-scaffold separation. The `seed0` labels in downstream directories do not independently establish the MolAS training RNG seed.

## Validation and reproducibility boundaries

`python scripts/validate_data.py` checks raw schemas and numeric domains, exact cross-table ID order, both acquisition score tables, the separate native-evaluation reconstruction, original/recovered fold agreement, descriptor/feature schemas, ligand internal IDs, cache-comparison evidence, six historical input hashes, and recorded checksums. It uses the Python standard library and never docks, trains, downloads, or deserializes PyTorch objects. `verify_feature_cache_alignment.py` is the separate NumPy-only source-to-cache comparison utility. Use `--require-complete` for a nonzero exit when acquisition inputs are missing. Acquisition inputs are complete. The numeric validator does not verify licenses; the author's redistribution confirmation is recorded separately from these integrity checks.

## Attribution and publication

Upstream MolAS: [code](https://github.com/BradWangW/MolAS), [processed data/checkpoints](https://zenodo.org/records/17760688), and [paper](https://link.springer.com/article/10.1186/s13321-026-01168-8). Cite the camera-ready active-learning paper using its verified final bibliographic record and cite MolAS and the original dataset/feature sources. Do not invent a DOI or release URL.

No new blanket dataset license has been assigned. The author confirmed applicable redistribution terms; see `third_party.md` and the confirmation record. The complete [HF dataset](https://huggingface.co/datasets/SeanWan05/MolAS-Active-Learning) states its component-specific terms and provides nine explicit Viewer configurations plus the original binary inputs and archives. Its author-owned data contributions use CC BY 4.0. The [official dataset-card](https://huggingface.co/docs/hub/datasets-cards), [file-configuration](https://huggingface.co/docs/hub/datasets-data-files-configuration), and [license](https://huggingface.co/docs/hub/repositories-licenses) guidance does not itself grant upstream redistribution rights.
