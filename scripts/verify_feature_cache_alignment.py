#!/usr/bin/env python3
"""Compare an existing cache with its deterministic source transformation.

No docking, learned-model fitting, embedding inference, or cache replacement.
Requires NumPy. PyTorch/PyG objects are NOT deserialized: an allowlisted pickle
reader replaces their classes/functions with inert metadata records. Only the
referenced numeric ZIP storage is read. Unrecognized classes fail closed.
"""
import argparse
from collections import OrderedDict
import csv
import hashlib
import io
import json
import math
from pathlib import Path
import pickle
import time
import zipfile

import numpy as np


class InertRecord:
    def __setstate__(self, state):
        self.state = state


class TensorMetadata:
    def __init__(self, storage, offset, size, stride, *_):
        self.storage = storage
        self.offset = int(offset)
        self.size = tuple(int(x) for x in size)
        self.stride = tuple(int(x) for x in stride)


class MetadataReader(pickle.Unpickler):
    def find_class(self, module, name):
        if module == 'torch_geometric.data.data' and name in ('Data', 'DataEdgeAttr', 'DataTensorAttr'):
            return InertRecord
        if module == 'torch_geometric.data.storage' and name == 'GlobalStorage':
            return InertRecord
        if module == 'torch._utils' and name == '_rebuild_tensor_v2':
            return TensorMetadata
        if module == 'torch' and name in ('FloatStorage', 'LongStorage', 'DoubleStorage'):
            return {'FloatStorage': 'f4', 'LongStorage': 'i8', 'DoubleStorage': 'f8'}[name]
        if module == 'collections' and name == 'OrderedDict':
            return OrderedDict
        raise pickle.UnpicklingError(f'Disallowed global: {module}.{name}')

    def persistent_load(self, reference):
        assert isinstance(reference, tuple) and len(reference) == 5 and reference[0] == 'storage'
        _, dtype, key, location, count = reference
        assert dtype in ('f4', 'i8', 'f8') and isinstance(key, str) and key.isdigit()
        assert isinstance(count, int) and 0 <= count <= 500_000_000
        return {'dtype': dtype, 'key': key, 'count': count}


def decode(path):
    archive = zipfile.ZipFile(path)
    pickle_names = [x for x in archive.namelist() if x.endswith('/data.pkl')]
    assert len(pickle_names) == 1
    name = pickle_names[0]
    assert archive.getinfo(name).file_size < 2_000_000
    metadata = MetadataReader(io.BytesIO(archive.read(name))).load()
    prefix = name.rsplit('/', 1)[0]
    byteorder = archive.read(prefix + '/byteorder').decode().strip()
    assert byteorder in ('little', 'big')
    return archive, prefix, byteorder, metadata


def numeric_tensor(archive, prefix, byteorder, metadata):
    assert isinstance(metadata, TensorMetadata)
    storage = metadata.storage
    dtype = np.dtype(('<' if byteorder == 'little' else '>') + storage['dtype'])
    payload = archive.read(prefix + '/data/' + storage['key'])
    values = np.frombuffer(payload, dtype=dtype)
    assert len(values) == storage['count']
    assert metadata.offset >= 0 and all(x >= 0 for x in metadata.size + metadata.stride)
    last = metadata.offset + sum((n - 1) * s for n, s in zip(metadata.size, metadata.stride))
    assert 0 <= last < len(values)
    view = np.ndarray(metadata.size, dtype=dtype, buffer=payload,
                      offset=metadata.offset * dtype.itemsize,
                      strides=tuple(x * dtype.itemsize for x in metadata.stride))
    return np.asarray(view, dtype=np.float32)


def pooled_protein(path):
    archive, prefix, byteorder, obj = decode(path)
    try:
        if isinstance(obj, dict):
            candidates = list(obj.values())
        else:
            mapping = obj.state['_store'].state['_mapping']
            candidates = [mapping[k] for k in ('x', 'node_s', 'node_features', 'esm', 'emb', 'features') if k in mapping]
        numeric = []
        for candidate in candidates:
            if not isinstance(candidate, TensorMetadata):
                continue
            arr = numeric_tensor(archive, prefix, byteorder, candidate)
            if arr.ndim == 1:
                numeric.append(arr)
            elif arr.ndim >= 2:
                numeric.append(arr.reshape(arr.shape[0], -1).mean(axis=0, dtype=np.float32))
        assert numeric, f'No feature tensor: {path}'
        # Same largest-pooled-vector choice as the existing producer function.
        return max(numeric, key=lambda x: x.size)
    finally:
        archive.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument('--write-ids', action='store_true', help='Write an ID index only after every cache row passes comparison')
    parser.add_argument('--report', type=Path)
    args = parser.parse_args()
    root = args.root.resolve()
    with (root / 'data/ordered_complex_ids.csv').open(newline='') as handle:
        ids = [x['PDB_CCD_ID'] for x in csv.DictReader(handle)]
    cache_path = root / 'data/features/complex_features_ligand_protein_d256_seed0.npy'
    cache = np.load(cache_path, allow_pickle=False)
    assert cache.shape == (3179, 256) and cache.dtype == np.float32
    ligand_path = root / 'data/features/ligand_chemberta_moad3179.pt'
    archive, prefix, byteorder, ligand_meta = decode(ligand_path)
    try:
        embedded_ids = ligand_meta['ligand_ids']
        ligand = numeric_tensor(archive, prefix, byteorder, ligand_meta['embeddings'])
        assert embedded_ids == ids and ligand.shape == (3179, 384)
    finally:
        archive.close()
    graph_dir = root / 'data/external/protein_graphs_esmc_600m'
    proteins = []
    started = time.time()
    for index, identifier in enumerate(ids):
        proteins.append(pooled_protein(graph_dir / f'pyg_graph_{identifier}_esmc_600m.pt'))
        if (index + 1) % 400 == 0:
            print(f'Checked existing graph storage {index + 1}/{len(ids)} ({time.time()-started:.1f}s)', flush=True)
    protein = np.stack(proteins).astype(np.float32)
    assert protein.shape == (3179, 1152)
    # Reproduce existing normalize_features/random_project equations; no fit of a learned model.
    features = np.asarray(np.concatenate([ligand, protein], axis=1), dtype=np.float32)
    mean = np.nanmean(features, axis=0, keepdims=True)
    std = np.nanstd(features, axis=0, keepdims=True)
    std[std < 1e-8] = 1.0
    normalized = np.nan_to_num((features - mean) / std).astype(np.float32)
    projection = np.random.default_rng(0).normal(0.0, 1.0 / math.sqrt(256), size=(features.shape[1], 256)).astype(np.float32)
    expected = (normalized @ projection).astype(np.float32)
    # Predeclared tight float32 tolerance allows different CPU reduction implementations.
    atol = rtol = 2e-5
    matching = np.isclose(cache, expected, atol=atol, rtol=rtol)
    difference = np.abs(cache.astype(np.float64) - expected.astype(np.float64))
    passed = bool(matching.all())
    report = {'scope': 'deterministic integrity comparison of existing data; no AL, docking, model training, or cache replacement',
              'cache_sha256': hashlib.sha256(cache_path.read_bytes()).hexdigest(),
              'raw_id_index_sha256': hashlib.sha256((root / 'data/ordered_complex_ids.csv').read_bytes()).hexdigest(),
              'ligand_sha256': hashlib.sha256(ligand_path.read_bytes()).hexdigest(),
              'producer_reference': 'dal_toolbox_design2/run_dal_toolbox_design2.py functions normalize_features, random_project, load_ligand_features, pooled_graph_feature, build_complex_features',
              'packaged_producer_excerpt': 'provenance/historical_acquisition_source/feature_builder_excerpt.py.txt',
              'packaged_producer_provenance': 'provenance/historical_acquisition_source/feature_builder_provenance.json',
              'shape': list(cache.shape), 'numpy_version': np.__version__, 'projection_seed': 0,
              'absolute_tolerance': atol, 'relative_tolerance': rtol,
              'maximum_absolute_difference': float(difference.max()),
              'root_mean_square_difference': float(np.sqrt(np.mean(difference ** 2))),
              'matching_values': int(matching.sum()), 'total_values': int(matching.size),
              'matching_rows': int(matching.all(axis=1).sum()), 'total_rows': len(ids),
              'bitwise_numeric_equal': bool(np.array_equal(cache, expected)), 'passed': passed,
              'note': 'NumPy float32 node pooling can differ in rounding from original PyTorch pooling; all comparisons are explicitly quantified.',
              'elapsed_seconds': round(time.time() - started, 2)}
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2), flush=True)
    if passed and args.write_ids:
        id_path = root / 'data/features/complex_feature_ids.csv'
        with id_path.open('w', newline='') as handle:
            writer = csv.writer(handle)
            writer.writerow(['PDB_CCD_ID'])
            writer.writerows((identifier,) for identifier in ids)
        evidence = {'producer_verified': True,
                    'verification_kind': 'existing producer transformation independently checked against every cached row',
                    'evidence': 'data/provenance/feature_cache_comparison.json',
                    'cache_sha256': report['cache_sha256'],
                    'feature_ids_sha256': hashlib.sha256(id_path.read_bytes()).hexdigest(),
                    'absolute_tolerance': atol, 'relative_tolerance': rtol,
                    'verified_rows': len(ids), 'verified_values': int(matching.size),
                    'not_a_claim_of_byte_identical_regeneration': not report['bitwise_numeric_equal']}
        (root / 'data/provenance/combined_feature_alignment.json').write_text(json.dumps(evidence, indent=2) + '\n')
    return 0 if passed else 2


if __name__ == '__main__':
    raise SystemExit(main())
