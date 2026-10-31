# MolAS: active learning under heterogeneous docking runtime

Research artifacts for **Active Learning for Molecular-Docking Algorithm
Selection under Heterogeneous Runtime**, accepted at the NeurIPS 2026 AI for
Drug Discovery workshop.

**Sean Wan, Jiabao Brad Wang, and Mustafa Misir — Duke Kunshan University.**

[GitHub repository](https://github.com/SeanWan514/MolAS_Active_Learning) ·
[Paper PDF](paper/AI4DD52_camera_ready.pdf) ·
[OpenReview](https://openreview.net/forum?id=tSz3wBLABm) ·
[Data card](docs/data_card.md) ·
[Figure lineage](provenance/figure_lineage.json) ·
[Reproduction status](docs/downstream_reproduction.md)

This study compares Mean-UCB with five predicted-runtime formulations for
retrospective, cell-wise acquisition of molecular-docking training matrices.
It uses 3,179 complexes, seven docking engines, and five held-out folds.
The contribution is an empirical comparison under heterogeneous runtime;
it does not establish superiority over untested acquisition baselines or
performance in a prospective docking campaign.

Partial acquisition supports useful selectors in the reported comparison, while
the best runtime formulation depends on the budget and downstream endpoint.
Matrix reconstruction diagnostics do not consistently track held-out selector
quality. The figure below is an unchanged asset from the paper.

![Reported downstream performance versus realized docking cost](paper/source/Fig1.png)

## Release status

This repository contains the **AI4DD 2026 camera-ready research artifacts**: source
code, small input data and feature caches, archived acquisition/downstream
results, manuscript assets, and integrity checks. The separately distributed
structural dataset is in preparation. File integrity and historical provenance
are documented in [`provenance/release_status.json`](provenance/release_status.json);
exact end-to-end scientific reproduction has not been newly verified.

The historical MolAS training adapter, launch script, analysis notebooks, runtime
probe source, and additional plotting scripts have been recovered from the
original downloads. Per-run training configurations and final figure-export
identity are documented to the extent supported by those records; a launcher
default alone is not proof of every historical invocation. All eight rendered
paper figures are included. The current official MolAS source is also included
as a separate pinned upstream reference with its MIT license. No new training
or end-to-end scientific reproduction run was performed during packaging.

The project's original software and associated software documentation use the
[MIT License](LICENSE). Datasets, research results, embeddings, model weights,
the manuscript, and figures retain separate terms; third-party software keeps
its supplied notices. See [NOTICE.md](NOTICE.md) and
[`docs/third_party.md`](docs/third_party.md) for the license scope and data
provenance. The author confirmed the applicable redistribution terms for this
GitHub release; the confirmation is recorded in
[`provenance/author_redistribution_confirmation.json`](provenance/author_redistribution_confirmation.json).
The separate dataset release will provide its own version and download reference.

## What is included

| Location | Contents |
|---|---|
| `paper/` | Supplied camera-ready PDF, matching manuscript source, bibliography, workshop style, and eight unchanged figure assets |
| `code/acquisition/` | Portable six-policy acquisition/completion implementation, input validation, launcher, summaries, and focused regression checks |
| `provenance/historical_acquisition_source/` | Byte-preserved inspected source, hashes, feature-building lineage, and distinction from the release copy |
| `code/downstream/` | Recovered historical training/analysis source, saved-result inventory utility, and separately pinned official upstream MolAS reference |
| `data/` | Original score/runtime inputs, existing strict targets, fixed folds, descriptors, feature artifacts, schema, and validation evidence |
| `results/acquisition/` | Compressed historical acquisition matrices, query records, and available diagnostics, with source/content manifests |
| `results/downstream/` | Compressed historical downstream predictions and metrics, readable summaries, and per-checkpoint coverage |
| `provenance/` | Artifact identities, figure-to-evidence map, file inventories, release status, and validation results |
| `docs/` | Reproduction instructions, data card, third-party status, workspace selection, and limitations |
| `scripts/` | Packaging and read-only integrity utilities; these do not launch scientific campaigns |

The six policies are `plain_mean_ucb`, `direct_cost_uc`,
`relaxed_cost_power_05`, `log_cost`, `runtime_uncertainty`, and `cost_tiers_5`.
The acquisition archive spans three seeds; the reported downstream folders use
acquisition **seed 0**. The five outer folds are not independent selector-seed
replications. Earlier hybrid/DAL/pilot experiments are excluded from the paper
results; their presence in the original workspace did not make them evaluated
baselines in this study. See [workspace selection](docs/workspace_audit.md).

## Inspect and validate without experiments

Python **3.11 or newer** is recommended for the repository's validation tools.
To obtain this repository and verify its files:

```sh
git clone https://github.com/SeanWan514/MolAS_Active_Learning.git
cd MolAS_Active_Learning
python3 scripts/verify_repository.py
python3 scripts/validate_data.py
```

The first command checks the frozen small-file/archive hashes and reports
integrity separately from reproduction completeness. `--strict-release` returns
nonzero if the recorded GitHub release is not ready for publication. Scientific
reproduction status is reported separately.
To additionally verify the large local structural files, use
`python3 scripts/verify_repository.py --include-large-data`.

For acquisition software checks, install the declared dependencies in an
isolated environment:

```sh
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python -m unittest discover -s code/acquisition/tests -v
.venv/bin/python code/acquisition/run_experiment.py --help
```

These unit checks use fabricated inputs and do not run docking, an acquisition
campaign, or selector training. Dependency constraints and the recorded release
test environment are not a reconstructed historical environment lock.

Instructions for genuine **new acquisition runs**, including the input contract
and explicit output directory, are in
[`code/acquisition/README.md`](code/acquisition/README.md). They were not executed
during this release preparation. Historical result archives remain unchanged by
the corrected release runner. In particular, the known incremental-log resume
defect is documented rather than silently repaired inside original records.

## Data boundaries

Algorithm order is `surfdock`, `unimol`, `diffdock`, `diffdock_l`, `gnina`,
`autodock`, `smina`. Raw matrices contain original measured outcomes; completed
checkpoint matrices contain observed scores plus imputed entries. Do not mix
these as if all values were observed docking measurements.

The full matrices are retrospective oracles. Availability of their values does
not authorize using unqueried labels or held-out outcomes during acquisition.
Runtime fitting uses previously observed training-pool runtimes. Preserve the
original complex-level folds and account for warm-start docking cost.

Raw numeric values have not been silently filtered or replaced. Very large RMSD
values in the source are preserved and documented in the data card; their
failure/sentinel semantics are not inferred from magnitude alone.

The original protein graphs comprise **3,179 files (about 15 GB)**. They are
**excluded from this GitHub release** and retained locally under
`data/external/protein_graphs_esmc_600m/`. A separate dataset release is in
preparation; its versioned URL and checksums will be added here when available.
The included small feature cache supports the acquisition input contract, while
the historical downstream MolAS workflow also requires the structural data.

## Manuscript

The frozen supplied PDF has 12 pages: main text through page 5, references on
pages 5–7, and appendix on pages 8–12. Its availability statement remains the
one in the supplied camera-ready version; packaging does not alter the paper or
claim that publication has occurred. Compile `paper/source/[AI4DD]main.tex`
with the included style, bibliography, sections, and figures in a suitable LaTeX
environment. The source is not replaced with an earlier draft or a new abstract.

Use [`CITATION.cff`](CITATION.cff) for citation metadata. The code release is
versioned as `v1.0.0`; the separate dataset identifier will be added after
publication and verification.
