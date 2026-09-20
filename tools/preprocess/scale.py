"""Calculate the channel scaling metadata used by DeepSCF."""

import argparse
from pathlib import Path

import h5py
import numpy as np
import yaml


def channel_files(dataset, index):
    directory = dataset / ('target' if index == 0 else f'feature{index}')
    files = sorted(directory.glob('*.h5'))
    if not files:
        raise ValueError(f'No HDF5 files found in {directory}')
    return files


def channel_statistics(files):
    means = []
    standard_deviations = []
    for path in files:
        with h5py.File(str(path), 'r') as handle:
            value = handle['value'][:]
        means.append(np.mean(value))
        standard_deviations.append(np.std(value))
    return float(np.mean(means)), float(np.mean(standard_deviations))


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('dataset', type=Path,
                        help='directory produced by tools.preprocess.convert')
    return parser.parse_args()


def main():
    args = parse_args()
    feature_directories = sorted(
        path for path in args.dataset.glob('feature*') if path.is_dir()
    )
    num_features = len(feature_directories) + 1

    output = {'num_of_feat': num_features}
    expected_names = None
    for index in range(num_features):
        files = channel_files(args.dataset, index)
        names = [path.stem for path in files]
        if expected_names is None:
            expected_names = names
        elif names != expected_names:
            raise ValueError(f'Channel {index} does not match target samples')
        mean, standard_deviation = channel_statistics(files)
        output[f'mean{index}'] = mean
        output[f'std{index}'] = standard_deviation

    with (args.dataset / 'scale.yaml').open('w', encoding='utf-8') as handle:
        yaml.safe_dump(output, handle, sort_keys=False)


if __name__ == '__main__':
    main()
