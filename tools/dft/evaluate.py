"""Compare completed fixed-RHO and reference SIESTA calculations."""

import glob
import math
from pathlib import Path
import pickle
import re

import numpy as np

from config_loader import get_args


FLOAT_PATTERN = r'[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[Ee][-+]?\d+)?'


def find_one(directory, pattern):
    matches = sorted(glob.glob(str(Path(directory) / pattern)))
    if len(matches) != 1:
        raise ValueError(
            f'Expected one {pattern} file in {directory}, found {len(matches)}'
        )
    return Path(matches[0])


def get_eigs(path):
    eig_file = find_one(path, '*.EIG')
    with eig_file.open() as handle:
        lines = handle.readlines()

    ef = float(lines[0].split()[0])
    neig, nspin, nkpt = map(int, lines[1].split())
    kblock = int(math.ceil(float(neig * nspin) / 10))
    energies = [
        [
            float(lines[2 + kblock * ik + j // 10].split()[1 + j % 10
                  if j < 10 else j % 10])
            for j in range(neig * nspin)
        ]
        for ik in range(nkpt)
    ]
    energies = np.asarray(energies).reshape((nkpt, nspin, neig))
    occupied = energies[energies <= ef]
    unoccupied = energies[energies > ef]
    if occupied.size == 0 or unoccupied.size == 0:
        raise ValueError(f'Cannot identify both occupied and unoccupied states in {eig_file}')
    return float(np.max(occupied)), float(np.min(unoccupied))


def get_atomic_forces(path):
    force_file = find_one(path, '*.FA')
    with force_file.open() as handle:
        number_of_atoms = int(handle.readline())
        rows = [handle.readline().split() for _ in range(number_of_atoms)]
    if any(len(row) < 4 for row in rows):
        raise ValueError(f'Malformed force file: {force_file}')
    forces = np.asarray([[float(row[1]), float(row[2]), float(row[3])]
                         for row in rows])
    return forces


def get_energy(path, label):
    stdout = Path(path) / 'stdout.txt'
    if not stdout.is_file():
        raise FileNotFoundError(f'Cannot find SIESTA output: {stdout}')
    pattern = re.compile(
        rf'^\s*siesta:\s+{re.escape(label)}\s*=\s*({FLOAT_PATTERN})\s*$',
        re.MULTILINE,
    )
    matches = pattern.findall(stdout.read_text(errors='replace'))
    if not matches:
        raise ValueError(f'Energy label {label!r} not found in {stdout}')
    return float(matches[-1])


def get_number_of_atoms(path):
    stdout = Path(path) / 'stdout.txt'
    if not stdout.is_file():
        raise FileNotFoundError(f'Cannot find SIESTA output: {stdout}')
    pattern = re.compile(
        r'initatomlists:\s+Number of atoms[^:]*:\s*([0-9]+)',
    )
    matches = pattern.findall(stdout.read_text(errors='replace'))
    if not matches:
        raise ValueError(f'Number of atoms not found in {stdout}')
    return int(matches[-1])


def output_directory(root, name, subdirectory):
    path = Path(root) / name
    if subdirectory:
        path = path / subdirectory
    return path


def main(args):
    predicted_root = Path(args.dft.predicted).resolve()
    reference_root = Path(args.dft.reference).resolve()
    names = sorted(path.name for path in predicted_root.iterdir() if path.is_dir())
    if not names:
        raise ValueError(f'No calculation directories found in {predicted_root}')

    predicted_energies = []
    reference_energies = []
    predicted_forces = []
    reference_forces = []
    predicted_homos = []
    reference_homos = []
    predicted_lumos = []
    reference_lumos = []
    numbers_of_atoms = []

    for name in names:
        predicted = output_directory(
            predicted_root, name, args.dft.predicted_output_subdir
        )
        reference = output_directory(
            reference_root, name, args.dft.reference_output_subdir
        )
        if not reference.is_dir():
            raise FileNotFoundError(f'Missing reference calculation: {reference}')

        number_of_atoms = get_number_of_atoms(predicted)
        reference_number_of_atoms = get_number_of_atoms(reference)
        if number_of_atoms != reference_number_of_atoms:
            raise ValueError(
                f'Atom-count mismatch for {name}: '
                f'{number_of_atoms} != {reference_number_of_atoms}'
            )

        predicted_force = get_atomic_forces(predicted)
        reference_force = get_atomic_forces(reference)
        if predicted_force.shape != reference_force.shape:
            raise ValueError(f'Force-shape mismatch for {name}')
        predicted_homo, predicted_lumo = get_eigs(predicted)
        reference_homo, reference_lumo = get_eigs(reference)

        numbers_of_atoms.append(number_of_atoms)
        predicted_energies.append(
            get_energy(predicted, args.dft.predicted_energy_label)
            / number_of_atoms
        )
        reference_energies.append(
            get_energy(reference, args.dft.reference_energy_label)
            / number_of_atoms
        )
        predicted_forces.extend(predicted_force)
        reference_forces.extend(reference_force)
        predicted_homos.append(predicted_homo)
        reference_homos.append(reference_homo)
        predicted_lumos.append(predicted_lumo)
        reference_lumos.append(reference_lumo)

    predicted_forces = np.asarray(predicted_forces)
    reference_forces = np.asarray(reference_forces)
    data = {
        'names': np.asarray(names),
        'nas': np.asarray(numbers_of_atoms),
        'tes': np.asarray(predicted_energies),
        'fxs': predicted_forces[:, 0],
        'fys': predicted_forces[:, 1],
        'fzs': predicted_forces[:, 2],
        'homos': np.asarray(predicted_homos),
        'lumos': np.asarray(predicted_lumos),
        'tes0': np.asarray(reference_energies),
        'fxs0': reference_forces[:, 0],
        'fys0': reference_forces[:, 1],
        'fzs0': reference_forces[:, 2],
        'homos0': np.asarray(reference_homos),
        'lumos0': np.asarray(reference_lumos),
        'metadata': {
            'predicted_energy_label': str(args.dft.predicted_energy_label),
            'reference_energy_label': str(args.dft.reference_energy_label),
            'energy_unit': 'eV/atom',
            'force_unit': 'eV/Ang',
        },
    }

    with Path(args.dft.result).open('wb') as handle:
        pickle.dump(data, handle)


if __name__ == '__main__':
    main(get_args())
