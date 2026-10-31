# Local structural data

`protein_graphs_esmc_600m/` contains the 3,179 original graph files copied from
the supplied MolAS feature download. Total logical size: 14,985,036,224 bytes.
The copy uses APFS cloning to avoid duplicating the stored payload on this Mac.
Every destination file has been hashed and compared with its source.
See `../../provenance/external_graph_checksums.json`.

These files are present locally but deliberately excluded from ordinary Git
commits. They are too large as a collection for a practical source repository.
The author confirmed applicable redistribution terms; see
`../../provenance/author_redistribution_confirmation.json`. Their separate dataset
release is in preparation. Its actual versioned download reference will be added
when published. The small ligand archive is in `../features/`.

The historical combined 256-dimensional feature cache has also been recovered
unchanged and its row alignment independently checked; it is in `../features/`.
The graphs and their original cache remain distinct artifacts. The recovered
downstream source documents training/evaluation settings, while the graph files
alone do not establish the exact configuration of every historical run.
