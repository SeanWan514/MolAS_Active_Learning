#!/usr/bin/env python3
"""Reconstruct native MolAS evaluation labels from unchanged raw tables.

This is deterministic data preparation, not docking or model training. The
output is explicitly reconstructed; it is not the missing original server CSV.
Acquisition uses a different, continuous target and is never overwritten here.
"""
import argparse
import csv
import hashlib
import math
from pathlib import Path

ALGORITHMS = ['surfdock', 'unimol', 'diffdock', 'diffdock_l', 'gnina', 'autodock', 'smina']


def native_score(rmsd, pb_ratio):
    if rmsd > 5 or rmsd < 0:
        return 0.0
    return (1 + math.exp(-6)) / (1 + math.exp(3 * (rmsd - 2))) * (pb_ratio == 1.0)


def read(path):
    with path.open(newline='') as handle:
        return list(csv.DictReader(handle))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument('--output', type=Path, help='Explicit reconstructed target destination')
    parser.add_argument('--check-only', action='store_true', help='Compare with the existing reconstructed file without writing')
    args = parser.parse_args()
    root = args.root.resolve()
    output = args.output or root / 'data/targets/moad_score_native_reconstructed.csv'
    protected = [root / f'data/raw/moad_{name}_trimmed.csv' for name in ['rmsd', 'pb_ratio', 'runtime']]
    protected += [root / 'data/targets/strict_score_matrix.csv', root / 'data/targets/full_strict_score_matrix.csv']
    if output.resolve() in {path.resolve() for path in protected}:
        parser.error('Refusing to overwrite raw data or an original acquisition target')
    rmsd = read(root / 'data/raw/moad_rmsd_trimmed.csv')
    pb = read(root / 'data/raw/moad_pb_ratio_trimmed.csv')
    ids = [row['PDB_CCD_ID'] for row in rmsd]
    assert len(ids) == len(set(ids)) == 3179
    assert ids == [row['PDB_CCD_ID'] for row in pb]
    records = [[identifier] + [native_score(float(r[a + '_rmsd']), float(p[a + '_pb_ratio']))
                               for a in ALGORITHMS]
               for identifier, r, p in zip(ids, rmsd, pb)]
    if args.check_only:
        existing = read(output)
        assert [row['PDB_CCD_ID'] for row in existing] == ids
        largest = max(abs(float(row[a]) - record[j + 1])
                      for row, record in zip(existing, records) for j, a in enumerate(ALGORITHMS))
        assert largest <= 1e-12, f'Native target mismatch: {largest}'
        print(f'Checked 3179 x 7 native labels; maximum absolute formula difference: {largest}')
    else:
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open('w', newline='') as handle:
            writer = csv.writer(handle)
            writer.writerow(['PDB_CCD_ID'] + ALGORITHMS)
            writer.writerows([row[0]] + [repr(x) for x in row[1:]] for row in records)
        print(f'Wrote explicitly reconstructed native evaluation target: {output}')
    print(f'SHA256: {hashlib.sha256(output.read_bytes()).hexdigest()}')


if __name__ == '__main__':
    main()
