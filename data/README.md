# Packaged dataset inputs

These are local copies of existing data and explicitly identified recoveries. No docking, model training, embedding inference, or new split generation was performed. Existing feature arithmetic was repeated only to validate the unchanged historical cache.

| Path | Contents and status |
|---|---|
| `raw/moad_rmsd_trimmed.csv` | Existing RMSD table; byte-identical to the subsequently recovered restart input. |
| `raw/moad_pb_ratio_trimmed.csv` | Existing PoseBusters-ratio table; bytes match the restart copy. |
| `raw/moad_runtime_trimmed.csv` | Existing runtime table; bytes match the restart copy. |
| `ordered_complex_ids.csv` | Raw-table row order. This is **not** evidence of combined-feature row order. |
| `targets/full_strict_score_matrix.csv` | Original restart target table, copied unchanged and formula-checked. |
| `targets/strict_score_matrix.csv` | Earlier target table retained with its original bare-algorithm column names; also formula-checked. |
| `targets/moad_score_native_reconstructed.csv` | Explicitly reconstructed native MolAS evaluation target: the native source sets scores to zero for RMSD >5 or <0. Verified against all 22,253 saved downstream engine scores. This is distinct from the acquisition target. |
| `splits/moad_5fold_split_seed42.csv` | Recovered original split CSV; its memberships exactly match the independent downstream recovery. |
| `splits/recovered_heldout_folds_from_downstream.csv` | Existing held-out membership recovered from saved downstream results, with provenance and cross-checks. Not a newly generated seed42 split. |
| `descriptors/moad_physical_descriptors_filtered.csv` | Recovered original four-descriptor CSV, aligned by original complex IDs. |
| `features/complex_features_ligand_protein_d256_seed0.npy` | Unchanged historical 3,179 × 256 feature cache; its hash matches the original input audit. |
| `features/complex_feature_ids.csv` | Row IDs verified by comparing the existing cache with the original deterministic source transformation across every row. |
| `features/ligand_chemberta_moad3179.pt` | Existing ligand embeddings with explicit embedded IDs; copied unchanged. |
| `features/ligand_feature_ids.csv` | ID sequence verified against the ligand archive's explicit `ligand_ids`. |
| `provenance/` | Exact source locations, hashes, bounded-read outcomes, graph inventory, recovered-fold evidence, and validation report. |

All six core acquisition inputs now match the original input-audit hashes. Initial cloud-placeholder failures remain recorded as historical attempts and were resolved by hydration. Combined-feature IDs were produced only after every row passed a deterministic source-to-cache comparison, with explicit float32 tolerances. All 3,179 large protein graphs are available locally under `external/protein_graphs_esmc_600m`, separately from the small Git-tracked dataset; their original upstream location and inventory are documented in the data card. Local availability does not mean full experiment reproduction, public publication, or license clearance.

Run `python scripts/validate_data.py --report data/provenance/validation_report.json` from the package root. Add `--require-complete` to fail when acquisition inputs are incomplete. A pass for available files does not establish full reproduction or license clearance.

The analysis notebook uses separate native-evaluation and acquisition ground truths. Use the explicitly named native reconstruction for the former and the continuous acquisition targets for the latter. The original server-side `moad_score_trimmed.csv` bytes were not recovered; no alias with that name is supplied. `python scripts/reconstruct_native_evaluation_target.py --check-only` verifies the reconstruction without changing it.

Read `../docs/data_card.md` and `../docs/third_party.md` before publishing. No new dataset license has been assigned.
