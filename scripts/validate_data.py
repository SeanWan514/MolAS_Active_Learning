#!/usr/bin/env python3
"""Validate packaged data only. Never train, dock, download, or unpickle objects.

Default exit status: 0 when available required-core data pass, 1 on corruption.
With --require-complete, exit 2 when acquisition inputs remain incomplete.
"""
import argparse
import array
import ast
import csv
import hashlib
import json
import math
from pathlib import Path
import pickletools
import struct
import sys
import zipfile

ALGORITHMS = ['surfdock', 'unimol', 'diffdock', 'diffdock_l', 'gnina', 'autodock', 'smina']


def table(path):
    with path.open(newline='', encoding='utf-8-sig') as handle:
        reader = csv.reader(handle)
        header = next(reader)
        rows = list(reader)
    assert len(header) == len(set(header)), f'Duplicate columns: {path}'
    assert all(len(row) == len(header) for row in rows), f'Ragged rows: {path}'
    return header, rows


def finite(value):
    number = float(value)
    assert math.isfinite(number), f'Nonfinite numeric value: {value}'
    return number


def score(rmsd, pb):
    z = 3.0 * (rmsd - 2.0)
    return ((1.0 + math.exp(-6.0)) / (1.0 + math.exp(z)) if z < 700 else 0.0) * (pb == 1.0)


def matching_id_column(header, rows, expected):
    candidates = [i for i in range(len(header))
                  if len({row[i] for row in rows}) == len(rows)
                  and {row[i] for row in rows} == set(expected)]
    assert len(candidates) == 1, f'Expected exactly one ID column; found {candidates}'
    return candidates[0]


def npy_metadata(path):
    with path.open('rb') as handle:
        assert handle.read(6) == b'\x93NUMPY', 'Not an NPY file'
        major, minor = handle.read(2)
        assert major in (1, 2, 3), f'Unsupported NPY version: {major}.{minor}'
        size = struct.unpack('<H' if major == 1 else '<I', handle.read(2 if major == 1 else 4))[0]
        header = ast.literal_eval(handle.read(size).decode('utf-8' if major == 3 else 'latin1'))
        dtype = header['descr']
        assert dtype in ('<f4', '<f8', '>f4', '>f8'), f'Unsupported numeric dtype: {dtype}'
        values = array.array('f' if dtype.endswith('4') else 'd')
        values.frombytes(handle.read())
        if (dtype[0] == '<') != (sys.byteorder == 'little'):
            values.byteswap()
        assert len(values) == math.prod(header['shape']), 'NPY payload size mismatch'
        assert all(math.isfinite(x) for x in values), 'NPY contains nonfinite values'
        return header


def validate(root):
    report = {'scope': 'file integrity and existing-data consistency; no scientific experiments',
              'checks': {}, 'errors': [], 'warnings': [], 'missing_acquisition_inputs': []}
    checks = report['checks']
    try:
        matrices = {}
        ids = None
        for stem, suffix in [('rmsd', '_rmsd'), ('pb_ratio', '_pb_ratio'), ('runtime', '_runtime')]:
            path = root / 'data/raw' / f'moad_{stem}_trimmed.csv'
            h, rows = table(path)
            assert h == ['PDB_CCD_ID'] + [a + suffix for a in ALGORITHMS], f'Unexpected schema: {path}'
            current = [row[0] for row in rows]
            assert len(current) == 3179 and len(set(current)) == 3179, f'Invalid ID coverage: {path}'
            if ids is not None:
                assert ids == current, f'Row order differs: {path}'
            ids = current
            vals = [[finite(x) for x in row[1:]] for row in rows]
            assert all(x >= 0 for row in vals for x in row), f'Negative value: {path}'
            if stem == 'runtime':
                assert all(x > 0 for row in vals for x in row), 'Runtime must be positive'
            if stem == 'pb_ratio':
                assert all(x <= 1 for row in vals for x in row), 'PB ratio exceeds 1'
            matrices[stem] = vals
            checks[stem] = {'rows': len(rows), 'algorithm_columns': 7, 'finite_values': 22253,
                            'minimum': min(map(min, vals)), 'maximum': max(map(max, vals))}
            if stem == 'rmsd' and max(map(max, vals)) > 1000:
                extreme_count = sum(x > 1000 for row in vals for x in row)
                checks['rmsd']['values_above_1000_angstrom'] = extreme_count
                report['warnings'].append(f'Raw RMSD contains {extreme_count} values above 1000 Angstrom; preserved unchanged. Failure/sentinel semantics are not established.')
        h, rows = table(root / 'data/ordered_complex_ids.csv')
        assert h == ['PDB_CCD_ID'] and [r[0] for r in rows] == ids, 'Raw ID index mismatch'
        checks['raw_matrix_id_order'] = 'identical across all three matrices and raw ID index'
        target = root / 'data/targets/strict_score_matrix.csv'
        h, rows = table(target)
        assert h == ['PDB_CCD_ID'] + ALGORITHMS and [r[0] for r in rows] == ids, 'Target alignment mismatch'
        max_error = 0.0
        for i, row in enumerate(rows):
            for j, value in enumerate(row[1:]):
                observed = finite(value)
                assert 0 <= observed <= 1, 'Strict score outside [0,1]'
                max_error = max(max_error, abs(observed - score(matrices['rmsd'][i][j], matrices['pb_ratio'][i][j])))
        assert max_error <= 1e-12, f'Strict-score formula mismatch: {max_error}'
        checks['strict_score'] = {'maximum_absolute_formula_error': max_error, 'tolerance': 1e-12,
                                 'lambda': 3.0, 'pb_valid': 'pb_ratio == 1.0'}
        original_target = root / 'data/targets/full_strict_score_matrix.csv'
        if original_target.exists():
            h, rows = table(original_target)
            assert h == ['PDB_CCD_ID'] + [a + '_strict_score' for a in ALGORITHMS], 'Original target schema mismatch'
            assert [r[0] for r in rows] == ids, 'Original target ID order mismatch'
            max_error = max(abs(finite(row[j + 1]) - score(matrices['rmsd'][i][j], matrices['pb_ratio'][i][j]))
                            for i, row in enumerate(rows) for j in range(7))
            assert max_error <= 1e-12, f'Original target formula mismatch: {max_error}'
            checks['original_strict_score'] = {'maximum_absolute_formula_error': max_error, 'tolerance': 1e-12}
        native_target = root / 'data/targets/moad_score_native_reconstructed.csv'
        if native_target.exists():
            h, rows = table(native_target)
            assert h == ['PDB_CCD_ID'] + ALGORITHMS and [r[0] for r in rows] == ids, 'Native evaluation target alignment mismatch'
            max_error = 0.0
            for i, row in enumerate(rows):
                for j in range(7):
                    rmsd = matrices['rmsd'][i][j]
                    expected = 0.0 if rmsd > 5 or rmsd < 0 else score(rmsd, matrices['pb_ratio'][i][j])
                    max_error = max(max_error, abs(finite(row[j + 1]) - expected))
            assert max_error <= 1e-12, f'Native evaluation formula mismatch: {max_error}'
            native_provenance = json.loads((root / 'data/provenance/native_evaluation_target.json').read_text())
            assert hashlib.sha256(native_target.read_bytes()).hexdigest() == native_provenance['sha256'], 'Native target provenance hash mismatch'
            validation = native_provenance['saved_downstream_validation']
            assert validation['complete_unique_complexes'] == 3179 and validation['engine_score_values_checked'] == 22253 and validation['mismatches'] == 0
            checks['native_evaluation_target'] = {'status': 'reconstructed, not recovered original CSV',
                                                  'maximum_absolute_formula_error': max_error,
                                                  'saved_engine_values_independently_checked': 22253,
                                                  'maximum_saved_engine_value_difference': validation['maximum_absolute_difference'],
                                                  'distinct_from_acquisition_target': True}

        original_split = root / 'data/splits/moad_5fold_split_seed42.csv'
        recovered_split = root / 'data/splits/recovered_heldout_folds_from_downstream.csv'
        memberships = {}
        for path in [original_split, recovered_split]:
            if not path.exists():
                continue
            h, rows = table(path)
            id_col = matching_id_column(h, rows, ids)
            fold_cols = [i for i, name in enumerate(h) if 'fold' in name.lower() and i != id_col]
            assert len(fold_cols) == 1, f'Ambiguous fold column: {path}'
            col = fold_cols[0]
            raw_folds = [finite(row[col]) for row in rows]
            assert all(v == int(v) for v in raw_folds), 'Noninteger fold label'
            offset = 1 if set(raw_folds) == {1, 2, 3, 4, 5} else 0
            assignment = {row[id_col]: int(v) - offset for row, v in zip(rows, raw_folds)}
            assert set(assignment.values()) == set(range(5)), 'Expected five folds'
            counts = {str(k): sum(v == k for v in assignment.values()) for k in range(5)}
            assert sorted(counts.values()) == [635, 636, 636, 636, 636], 'Unexpected fold sizes'
            memberships[path.name] = assignment
            checks[path.name] = {'id_column': h[id_col], 'fold_column': h[col], 'counts': counts,
                                 'normalization_offset': offset}
        if len(memberships) == 2:
            assert memberships[original_split.name] == memberships[recovered_split.name], 'Original/recovered fold mismatch'
            checks['original_recovered_folds_agree'] = True
        if not original_split.exists():
            report['warnings'].append('Original seed42 split file unavailable; recovered file records existing downstream membership, not a new random split.')
        if not memberships:
            report['missing_acquisition_inputs'].append('original or validated recovered outer-fold assignments')

        descriptor = root / 'data/descriptors/moad_physical_descriptors_filtered.csv'
        if descriptor.exists():
            h, rows = table(descriptor)
            id_col = matching_id_column(h, rows, ids)
            numeric = []
            for j in range(len(h)):
                if j == id_col:
                    continue
                try:
                    [finite(row[j]) for row in rows]
                    numeric.append(h[j])
                except (ValueError, AssertionError):
                    pass
            assert len(numeric) == 4, f'Expected four finite numeric descriptors, found {numeric}'
            checks['physical_descriptors'] = {'id_column': h[id_col], 'numeric_columns': numeric, 'rows': len(rows)}
        else:
            report['missing_acquisition_inputs'].append(str(descriptor.relative_to(root)))

        feature = root / 'data/features/complex_features_ligand_protein_d256_seed0.npy'
        feature_ids = root / 'data/features/complex_feature_ids.csv'
        if feature.exists():
            metadata = npy_metadata(feature)
            assert tuple(metadata['shape']) == (3179, 256), 'Combined feature shape mismatch'
            checks['combined_feature_numeric_payload'] = metadata
        else:
            report['missing_acquisition_inputs'].append(str(feature.relative_to(root)))
        if feature_ids.exists():
            h, rows = table(feature_ids)
            assert len(h) == 1 and [x[0] for x in rows] == ids, 'Combined feature ID order mismatch'
            provenance = root / 'data/provenance/combined_feature_alignment.json'
            assert provenance.exists(), 'Producer-verified combined-feature alignment evidence is required'
            evidence = json.loads(provenance.read_text())
            assert evidence.get('producer_verified') is True and evidence.get('evidence'), 'Incomplete feature alignment evidence'
            assert hashlib.sha256(feature.read_bytes()).hexdigest() == evidence['cache_sha256'], 'Feature alignment evidence refers to a different cache'
            assert hashlib.sha256(feature_ids.read_bytes()).hexdigest() == evidence['feature_ids_sha256'], 'Feature alignment evidence refers to different IDs'
            comparison = json.loads((root / evidence['evidence']).read_text())
            assert comparison['passed'] and comparison['matching_rows'] == 3179 and comparison['matching_values'] == 813824, 'Incomplete cache-value comparison'
            checks['combined_feature_ids'] = {'ordered_ids': 3179, 'independently_compared_cache_values': 813824,
                                             'maximum_absolute_difference': comparison['maximum_absolute_difference'],
                                             'absolute_tolerance': comparison['absolute_tolerance'],
                                             'relative_tolerance': comparison['relative_tolerance']}
        else:
            report['missing_acquisition_inputs'].append('producer-verified data/features/complex_feature_ids.csv and alignment evidence')

        ligand = root / 'data/features/ligand_chemberta_moad3179.pt'
        if ligand.exists():
            with zipfile.ZipFile(ligand) as archive:
                prefix = archive.namelist()[0].split('/')[0]
                ops = pickletools.genops(archive.read(prefix + '/data.pkl'))
                strings = [arg for op, arg, _ in ops if op.name in ('BINUNICODE', 'SHORT_BINUNICODE', 'UNICODE')]
                id_set = set(ids)
                embedded_ids = [s for s in strings if s in id_set]
                assert embedded_ids == ids, 'Embedded ligand IDs do not match raw ID order'
                values = array.array('f')
                values.frombytes(archive.read(prefix + '/data/0'))
                if archive.read(prefix + '/byteorder').decode().strip() != sys.byteorder:
                    values.byteswap()
                assert len(values) == 3179 * 384 and all(math.isfinite(x) for x in values), 'Ligand numeric payload invalid'
            h, rows = table(root / 'data/features/ligand_feature_ids.csv')
            assert h == ['PDB_CCD_ID'] and [x[0] for x in rows] == embedded_ids, 'Ligand index mismatch'
            checks['ligand_embeddings'] = {'embedded_ordered_ids': 3179, 'finite_float_values': len(values),
                                          'inspection': 'ZIP/pickle opcodes only; no object deserialization'}

        graph_dir = root / 'data/external/protein_graphs_esmc_600m'
        if graph_dir.exists():
            files = list(graph_dir.glob('pyg_graph_*_esmc_600m.pt'))
            graph_ids = [p.name.removeprefix('pyg_graph_').removesuffix('_esmc_600m.pt') for p in files]
            assert len(files) == 3179 and set(graph_ids) == set(ids), 'External protein graph ID coverage mismatch'
            stats = [p.stat() for p in files]
            placeholders = sum(bool(getattr(s, 'st_flags', 0) & 0x40000000) for s in stats)
            assert placeholders == 0, 'External protein graph directory contains cloud placeholders'
            checks['external_protein_graphs'] = {'files': len(files), 'logical_bytes': sum(s.st_size for s in stats),
                                               'cloud_placeholders': placeholders,
                                               'check_scope': 'filenames and filesystem metadata, not payload deserialization'}

        historical_audit = root / 'data/provenance/original_input_audit.json'
        if historical_audit.exists():
            audit = json.loads(historical_audit.read_text())
            current_files = {p.name: p for p in [*(root / 'data/raw').glob('*.csv'), original_split,
                                                descriptor, feature] if p.exists()}
            matched = 0
            for original_path, expected_hash in audit['sha256'].items():
                filename = Path(original_path).name
                assert filename in current_files, f'Historical audited input missing: {filename}'
                assert hashlib.sha256(current_files[filename].read_bytes()).hexdigest() == expected_hash, f'Historical input hash mismatch: {filename}'
                matched += 1
            checks['original_input_audit_hashes'] = {'matching_inputs': matched,
                                                    'note': 'Comparison to preserved historical hashes; historical experiments were not rerun.'}

        checksum_file = root / 'data/CHECKSUMS.sha256'
        if checksum_file.exists():
            count = 0
            for line in checksum_file.read_text().splitlines():
                digest, rel = line.split('  ', 1)
                path = root / rel
                assert path.is_relative_to(root) and path.is_file(), f'Missing checksum target: {rel}'
                assert hashlib.sha256(path.read_bytes()).hexdigest() == digest, f'Checksum mismatch: {rel}'
                count += 1
            checks['checksums'] = {'verified_files': count}
    except (AssertionError, ValueError, KeyError, OSError, csv.Error, zipfile.BadZipFile) as error:
        report['errors'].append(str(error))
    report['available_data_integrity_passed'] = not report['errors']
    report['acquisition_inputs_complete'] = not report['errors'] and not report['missing_acquisition_inputs']
    report['full_downstream_reproduction_verified'] = False
    report['license_clearance_verified'] = False
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument('--report', type=Path, help='Optional path for a JSON integrity report')
    parser.add_argument('--require-complete', action='store_true')
    args = parser.parse_args()
    result = validate(args.root.resolve())
    text = json.dumps(result, indent=2, allow_nan=False) + '\n'
    print(text, end='')
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(text)
    if result['errors']:
        return 1
    if args.require_complete and not result['acquisition_inputs_complete']:
        return 2
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
