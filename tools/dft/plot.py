"""Plot parity comparisons from tools.dft.evaluate output."""

from pathlib import Path
import pickle

import matplotlib.pyplot as plt
import numpy as np

from config_loader import get_args


def parity_plot(reference, predicted, color, name):
    all_values = np.concatenate([reference, predicted])
    lower = float(all_values.min())
    upper = float(all_values.max())
    padding = (upper - lower) * 0.05
    if padding == 0.0:
        padding = max(abs(lower) * 0.05, 1.0e-12)
    lower -= padding
    upper += padding

    figure, axis = plt.subplots(figsize=(5, 5))
    axis.plot([lower, upper], [lower, upper], color='black',
              linewidth=1.5, linestyle='--')
    axis.scatter(reference, predicted, s=20, alpha=0.5, color=color)
    axis.set_xlim(lower, upper)
    axis.set_ylim(lower, upper)
    axis.set_xlabel('Reference')
    axis.set_ylabel('DeepSCF fixed-RHO')
    figure.tight_layout()
    figure.savefig(name, dpi=300, transparent=True)
    plt.close(figure)


def main(args):
    result = Path(args.dft.result)
    with result.open('rb') as handle:
        data = pickle.load(handle)

    features = ('tes', 'fxs', 'fys', 'fzs', 'homos', 'lumos')
    colors = ('tab:blue', 'tab:orange', 'tab:green',
              'tab:red', 'tab:purple', 'tab:brown')
    summary = result.with_name(f'{result.stem}_summary.txt')
    with summary.open('w') as handle:
        for key, color in zip(features, colors):
            predicted = np.asarray(data[key], dtype=float).ravel()
            reference = np.asarray(data[f'{key}0'], dtype=float).ravel()
            mae = np.abs(reference - predicted).mean()
            line = f'{key}: MAE = {mae:.6f}'
            print(line)
            handle.write(f'{line}\n')
            parity_plot(
                reference,
                predicted,
                color,
                result.with_name(f'{result.stem}_{key}.png'),
            )


if __name__ == '__main__':
    main(get_args())
