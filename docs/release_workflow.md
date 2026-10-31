# Release and verification

Repository: https://github.com/SeanWan514/MolAS_Active_Learning

The GitHub release includes source code, small input data/features, original saved
results, manuscript assets, and verification/provenance records. Large structural
graphs remain excluded from Git for a separate dataset release. The author
confirmed the applicable redistribution terms on 2026-10-31; see the
[confirmation record](../provenance/author_redistribution_confirmation.json).
Original software uses the [MIT License](../LICENSE); preserve third-party notices
and the component boundaries in [NOTICE.md](../NOTICE.md).

## Verify this release

```sh
git clone https://github.com/SeanWan514/MolAS_Active_Learning.git
cd MolAS_Active_Learning
python3 scripts/verify_repository.py --strict-release
python3 scripts/validate_data.py --require-complete
```

The first command checks the frozen Git-visible file identities and the recorded
GitHub-publication readiness. The data validator checks the included acquisition
inputs. Exact historical end-to-end reproduction is a separate status; neither
command launches docking, acquisition campaigns, or selector training.

For the existing focused software checks, install the declared dependencies in
an isolated environment and run:

```sh
python3 -m unittest discover -s code/acquisition/tests -v
```

See [acquisition instructions](../code/acquisition/README.md) and
[downstream reproduction records](downstream_reproduction.md) for input contracts,
source snapshots, configuration helpers, and historical execution limits.

## Separate structural dataset

The complete original graph set comprises 3,179 files and 14,985,036,224 bytes.
It is retained locally under `data/external/protein_graphs_esmc_600m/` and excluded
by `.gitignore`. The complete dataset is published at [https://huggingface.co/datasets/SeanWan05/MolAS-Active-Learning](https://huggingface.co/datasets/SeanWan05/MolAS-Active-Learning) with snapshot `v1.0.0`. Anonymous access, all uploaded file identities, and all nine Viewer configurations have been verified. Its immutable revision is recorded in `provenance/hf_dataset_publication.json`; graph hashes remain in `provenance/external_graph_checksums.json`.

## Frozen artifacts and future updates

The supplied paper, scientific results, and historical source snapshots are
preserved. Corrected portable software is identified separately. No scientific
campaign was rerun during preparation. File integrity and exact historical
reproduction remain separate claims; retain the documented per-run configuration
and figure-export provenance limits.

After an intentional edit, rerun the appropriate component checks and regenerate
`provenance/repository_files.json` with `python3 scripts/freeze_manifest.py --write`.
Do not suppress checksum failures or replace historical results with new outputs.
Code releases use Git tags; the initial version is `v1.0.0`. Verify public access
to the repository and exact tag before adding its link to the manuscript. The
frozen supplied manuscript is not automatically edited by this publication.
