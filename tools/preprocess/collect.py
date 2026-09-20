"""Standardize input channels and assemble DeepSCF HDF5 samples."""

import argparse
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import h5py
import numpy as np
import yaml


def read_value(path):
    with h5py.File(str(path), 'r') as handle:
        return handle['value'][:]


def collect_one(name, dataset, means, standard_deviations, num_features):
    dataset = Path(dataset)
    target = read_value(dataset / 'target' / f'{name}.h5')
    features = []
    for index in range(1, num_features):
        if standard_deviations[index] == 0.0:
            raise ValueError(f'Feature {index} has zero standard deviation')
        value = read_value(dataset / f'feature{index}' / f'{name}.h5')
        features.append(
            (value - means[index]) / standard_deviations[index]
        )

    with h5py.File(str(dataset / 'total' / f'{name}.h5'), 'w') as handle:
        handle.create_dataset('target', data=target)
        handle.create_dataset('feature', data=np.asarray(features))
    return name


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('dataset', type=Path,
                        help='directory containing scale.yaml and channel data')
    parser.add_argument('--workers', type=int, default=1)
    return parser.parse_args()


def main():
    args = parse_args()
    if args.workers < 1:
        raise ValueError('--workers must be at least 1')
    with (args.dataset / 'scale.yaml').open(encoding='utf-8') as handle:
        scale = yaml.safe_load(handle)

    num_features = int(scale['num_of_feat'])
    means = np.asarray([scale[f'mean{i}'] for i in range(num_features)])
    standard_deviations = np.asarray(
        [scale[f'std{i}'] for i in range(num_features)]
    )

    names = [path.stem for path in sorted((args.dataset / 'target').glob('*.h5'))]
    if not names:
        raise ValueError(f'No target files found in {args.dataset / "target"}')
    for index in range(1, num_features):
        feature_names = [path.stem for path in sorted(
            (args.dataset / f'feature{index}').glob('*.h5')
        )]
        if feature_names != names:
            raise ValueError(f'Feature {index} does not match target samples')

    (args.dataset / 'total').mkdir(exist_ok=True)
    if args.workers == 1:
        for name in names:
            print(collect_one(name, args.dataset, means,
                              standard_deviations, num_features))
    else:
        with ProcessPoolExecutor(max_workers=args.workers) as executor:
            futures = [executor.submit(collect_one, name, args.dataset, means,
                                       standard_deviations, num_features)
                       for name in names]
            for future in futures:
                print(future.result())


if __name__ == '__main__':
    main()
