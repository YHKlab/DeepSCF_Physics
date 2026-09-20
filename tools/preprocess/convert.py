"""Convert SIESTA grid outputs into per-channel HDF5 files."""

import argparse
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import h5py

from utils import siestaio


DEFAULT_FEATURES = (
    'rhoatom',
    'Molecular.IOCH',
    'rhoover',
    'gradDRho',
    'gradDRho_r',
    'lapDRho',
    'tau',
    'tau_tf',
    'tau_w',
    'tau_tfr',
    'tau_wr',
    'alpha',
)


def write_value(path, value):
    with h5py.File(str(path), 'w') as handle:
        handle.create_dataset('value', data=value)


def convert_one(calculation, output, features):
    calculation = Path(calculation)
    output = Path(output)
    name = calculation.name

    _, _, target = siestaio.readGrid(str(calculation / 'Molecular.RHO'))
    write_value(output / 'target' / f'{name}.h5', target[0])

    for index, feature in enumerate(features, start=1):
        _, _, grid = siestaio.readGrid(str(calculation / feature))
        write_value(output / f'feature{index}' / f'{name}.h5', grid[0])

    return name


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('calculations', type=Path,
                        help='directory whose direct children are calculations')
    parser.add_argument('output', type=Path,
                        help='output directory for per-channel HDF5 files')
    parser.add_argument('--features', nargs='+', default=list(DEFAULT_FEATURES),
                        help='SIESTA grid filenames in feature-channel order')
    parser.add_argument('--workers', type=int, default=1)
    return parser.parse_args()


def main():
    args = parse_args()
    calculations = sorted(path for path in args.calculations.iterdir()
                          if path.is_dir())
    if not calculations:
        raise ValueError(f'No calculation directories found in {args.calculations}')
    if args.workers < 1:
        raise ValueError('--workers must be at least 1')

    (args.output / 'target').mkdir(parents=True, exist_ok=True)
    for index in range(1, len(args.features) + 1):
        (args.output / f'feature{index}').mkdir(parents=True, exist_ok=True)

    if args.workers == 1:
        for calculation in calculations:
            print(convert_one(calculation, args.output, args.features))
    else:
        with ProcessPoolExecutor(max_workers=args.workers) as executor:
            futures = [executor.submit(convert_one, calculation, args.output,
                                       args.features)
                       for calculation in calculations]
            for future in futures:
                print(future.result())


if __name__ == '__main__':
    main()
