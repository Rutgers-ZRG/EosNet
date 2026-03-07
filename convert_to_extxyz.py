#!/usr/bin/env python3
"""Convert EOSNet POSCAR (.vasp) dataset to extxyz format.

Usage:
    python convert_to_extxyz.py /path/to/dataset [--output data.extxyz]

Input:  root_dir/ with id_prop.csv + *.vasp files
Output: root_dir/data.extxyz (all structures, targets in atoms.info)
"""

import argparse
import csv
import os
import sys
from ase.io import read as ase_read, write as ase_write


def main():
    parser = argparse.ArgumentParser(
        description='Convert EOSNet .vasp dataset to extxyz format')
    parser.add_argument('root_dir', help='Path to dataset directory')
    parser.add_argument('--output', default=None,
                        help='Output file (default: root_dir/data.extxyz)')
    parser.add_argument('--target-key', default='target',
                        help='Key name for target in extxyz info (default: target)')
    args = parser.parse_args()

    root_dir = args.root_dir
    output_path = args.output or os.path.join(root_dir, 'data.extxyz')

    id_prop_file = os.path.join(root_dir, 'id_prop.csv')
    if not os.path.isfile(id_prop_file):
        print(f"ERROR: {id_prop_file} not found")
        sys.exit(1)

    with open(id_prop_file) as f:
        reader = csv.reader(f, delimiter=',')
        id_prop_data = [row for row in reader]

    print(f"Found {len(id_prop_data)} entries in id_prop.csv")

    all_atoms = []
    errors = []
    for struct_id, target in id_prop_data:
        vasp_file = os.path.join(root_dir, struct_id + '.vasp')
        if not os.path.isfile(vasp_file):
            errors.append(struct_id)
            continue
        try:
            atoms = ase_read(vasp_file)
            atoms.info['struct_id'] = struct_id
            atoms.info[args.target_key] = float(target)
            all_atoms.append(atoms)
        except Exception as e:
            print(f"WARNING: Failed to read {struct_id}: {e}")
            errors.append(struct_id)

    if errors:
        print(f"WARNING: {len(errors)} structures could not be read: "
              f"{errors[:5]}{'...' if len(errors) > 5 else ''}")

    ase_write(output_path, all_atoms, format='extxyz')
    print(f"Wrote {len(all_atoms)} structures to {output_path}")

    # Verify round-trip
    check = ase_read(output_path, index=':')
    assert len(check) == len(all_atoms), \
        f"Round-trip check failed: wrote {len(all_atoms)}, read back {len(check)}"
    print("Round-trip verification passed.")


if __name__ == '__main__':
    main()
