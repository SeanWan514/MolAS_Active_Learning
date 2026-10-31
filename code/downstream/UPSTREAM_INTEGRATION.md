# Official MolAS reference and recovered AI4DD adapter

The separate `upstream_molas/` directory preserves the official [MolAS repository at commit `a6c417216ddb992f7dc513d511d0429aece4bd61`](https://github.com/BradWangW/MolAS/tree/a6c417216ddb992f7dc513d511d0429aece4bd61), retrieved on 31 October 2026. Its 24 small source/configuration files (286,198 bytes) are unchanged, including the MIT license. `upstream_molas_manifest.json` records their original Git blob IDs and SHA-256 hashes. The large upstream `analysis.ipynb` with rendered outputs is excluded from this compact reference copy.

The AI4DD completed-matrix adapter was subsequently recovered separately in `historical_molas/`. Use that recovered source and its provenance for the paper workflow; do not substitute current upstream defaults. No environment installation, model execution, or statistical testing was performed during packaging.

| Integration point | Pinned official upstream | Recovered AI4DD source |
|---|---|---|
| Target loading | Builds complete targets from raw RMSD/PoseBusters inputs. | Adds `--label_source` and `--score_matrix_path` for fold-specific completed training targets. |
| Outer folds | Generates K-fold membership from dataset order and a seed. | Launcher verifies the exported fold mapping and each completed training matrix before fitting. |
| Held-out evaluation | Exports engine/selector outcomes and predictions. | Launcher uses a separate native-composite test call on the matching outer held-out fold. |
| Model configuration | Current source defaults/examples cover several architectures and portfolios. | Recovered `main.sh` gives an actual source configuration for MolAS, seven engines, seed 42, and a selected subset of acquisition checkpoints; per-run execution records remain separate evidence. |
| Paths | Contains hard-coded paths to upstream data and graphs. | `prepare_local_copy.py` configures a new copy of the recovered adapter without changing numerical functions or executing it. |

The native evaluation score and acquisition strict score have different RMSD-tail handling. Follow the two explicit target roles in `local_paths.example.json` and the data provenance; do not rename the strict target into the native target's role. The native reconstruction is separately identified and checked against saved outcomes.

The upstream `stats_test.py` belongs to that upstream project. Its presence does not mean additional tests were performed for this AI4DD study or that its assumptions apply to repeated acquisition checkpoints. The recovered primary notebook likewise contains exploratory sections beyond the paper; packaging preserves them without presenting them as new experiments or new manuscript evidence.
