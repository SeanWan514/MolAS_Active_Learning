# Selection of paper artifacts

The source workspace was inventoried by filename, size, and cloud-placeholder
status before copying. Its research branches are not all part of the paper.
No docking, acquisition campaign, selector training, or new scientific comparison
was run during packaging. Original workspace files were not edited.

| Source area | Decision and reason |
|---|---|
| `MolAS_AL_restart/AL_MolAS/mean_ucb_cost_5fold_paper` | Primary acquisition implementation candidate and six-policy, five-fold, three-acquisition-seed archive. Preserve historical source; make separately documented portable release changes. The downstream paper uses acquisition seed 0. |
| `port_as_results_260829/{six paper policies}` | Preserve downstream CSVs, including per-complex predictions, fold summaries, and aggregate tables. Missing cloud contents remain inventoried. |
| `PORT-AS_cost_aware_bare.git` and original cost-aware remote | Versioned acquisition matrix/query-log evidence. Pin the exact historical commit and content identities; do not replace it with a moving branch. |
| `camera_ready_final_2026-10-31` and user-supplied `SRS_2026_AL_MolAS_Docking-8.pdf` | Preserve the audited source, style, bibliography, eight figures, and supplied 12-page camera-ready PDF. |
| `MolAS_AL/data/raw` | Readable original raw tables; PB/runtime agree byte-for-byte with restart copies. |
| `MolAS_AL` legacy strict-score output and restart `ground_truth` | Preserve both existing target matrices with explicit provenance. The restart original was subsequently recovered; its values and the legacy representation were checked against the raw data without conflating their distinct file identities. |
| `Downloads/data/MolAS` | Original resident ligand/protein features. Copy needed structural inputs into the local data area; retain hashes and separate large-data publication requirements. |
| `Downloads/cost_aware_relaxed_six_methods.zip` | Complete original six-policy archive. Verify all recovered CSV identities against the pinned Git tree; losslessly deduplicate cumulative prefixes in the release. |
| `Downloads/MolAS_AL-main` | Recovered historical MolAS training adapter, launcher, model/data modules, embedding-generation source, and analysis notebooks. Preserve unchanged with the supplied license and hashes; it is an unversioned downloaded tree. |
| Original paper download's `NeurIPS2026/scripts` and `Downloads/main_results` | Additional plotting source and rendered analysis assets; only demonstrated relationships are used in the figure-lineage map. |
| `cost_aware_design2` and `dal_toolbox_design2` | Only the feature-cache lineage and original feature-building source are relevant dependencies. Their unrelated AL campaigns are not paper baselines. |
| `mean_ucb_cost_balance_100checkpoints` | Earlier design; not substituted for the paper-specific six-policy archive. |
| `budget_projected_gain_hybrid`, `hybrid_b1`, `hybrid_b50`, `PORT-AS_cost_aware_log_upload`, `PORT-AS_cost_aware_upload`, incomplete upload copy | Development/hybrid runs outside the evaluated six-policy study; excluded. |
| `design1_raw_metric_reconstruction`, `design2_score_matrix_reconstruction`, `paper_query_svd_design2`, `dl_design2_qin2025`, `cost_aware_predicted_runtime_design2` | Earlier experiments and alternative formulations. Their existence does not establish controlled baseline comparisons in the paper. Excluded from reported results. |
| Generic `audit` and `runtime_audit` | Earlier-design audits; not mislabeled as the paper's final runtime-probe or plotting implementation. |
| Early `MolAS_AL` literature/pilot reports, top-level library catalog, `docs`, `scripts/fetch_github_meta.py`, `data/repositories.yml` | Background/development materials rather than dependencies of the reported study. |
| `deliverables` | Earlier research report, abstract, poster-planning materials; not final paper artifacts. |
| Earlier camera-ready folders, compilation scratch, auxiliary LaTeX files | Superseded drafts/build products; excluded. |
| Virtual environments, vendored unrelated libraries, caches, nested Git metadata, `.DS_Store` | Not copied into the publication source tree. |

The audit includes the full accessible workspace's metadata, not a claim that
every cloud-only file was read. A cloud placeholder contains a name and logical
size but may not provide readable bytes. Missing files are explicitly identified
in artifact manifests rather than represented by empty files.

## Evidence boundaries

The acquisition implementation is supported by formula checks against saved
records. A historical code commit and exact dependency lock were not archived
with every reported run. The recovered downstream source and launch script add
substantial protocol evidence, but launcher settings and saved filenames alone
do not prove every executed training configuration. The repository documents
these distinctions and does not claim a new end-to-end reproduction run.

The original acquisition report's three-seed coverage must not be confused with
independent selector-seed replication. The archived downstream folders are
`seed0`, and the manuscript describes that scope.
