# DeepSCF Physics Update

<p align="center">
  <img height="240" src="./logo/shematics.png" alt="DeepSCF workflow"/>
</p>

DeepSCF learns the map from an initial electron density and grid-projected
atomic features to a self-consistent electron density. This `physics-update`
branch extends the original implementation with a physics-aware training path,
reproducible preprocessing tools, fixed-density SIESTA postprocessing, and a
band-sum-based energy diagnostic for predicted densities.

The method is described in [*Convolutional network learning of self-consistent
electron density via grid-projected atomic fingerprints*](https://doi.org/10.1038/s41524-024-01433-0).

## What is included

- The current multi-file DeepSCF model and standard training path.
- `train_physics.py` for density-gradient regularization.
- SIESTA-grid-to-HDF5 preprocessing with recorded channel scaling.
- Prediction of SIESTA-format `.RHO` files.
- Read-only comparison of completed fixed-RHO and self-consistent calculations.
- A version-specific SIESTA 4.1.5 patch that prints `Dxc_pcc` and `Etot_bs`.

This repository does not include trained weights, datasets, pseudopotentials,
completed DFT results, or calculation-submission scripts.

## Installation

The code preserves the Python 3.7 / PyTorch 1.12 environment of the original
implementation.

```bash
git clone --branch physics-update \
  https://github.com/YHKlab-MSJeong/DeepSCF_Physics.git
cd DeepSCF_Physics
python -m pip install -r requirements.txt
```

For GPU use, install a PyTorch build matching the local CUDA stack rather than
blindly replacing a working site installation.

## Configuration

All entry points merge a user YAML file into [`config/input.yaml`](config/input.yaml).
Start from the documented six-channel example:

```bash
cp examples/input.yaml input.yaml
```

Run commands from the repository root. A different file can be selected with
`--input path/to/input.yaml`; `--default` may be used to replace the base
configuration.

The most important fields are:

- `dataset.path`: directory containing assembled `.h5` samples.
- `model.input_layers`: number of selected input channels.
- `model.input_layers_index`: zero-based channel indices loaded from each HDF5
  `feature` array.
- `model.rho0_index`: position of the standardized initial density within the
  selected model input.
- `model.rho0_mean` and `model.rho0_std`: scaling used to reconstruct the
  physical initial density.
- `train.loss.regularization`: optional physics losses; the example enables
  `gradient_3p`.

## Dataset preparation

Each completed source calculation must be a direct child of a calculation
root and contain `Molecular.RHO` plus the requested grid-feature files. The
default feature order is defined in `tools/preprocess/convert.py` and can be
overridden with `--features`.

```bash
python -m tools.preprocess.convert calculations dataset_raw
python -m tools.preprocess.scale dataset_raw
python -m tools.preprocess.collect dataset_raw
```

The first command writes one HDF5 file per sample and channel. The second
writes `dataset_raw/scale.yaml`; its mean and standard deviation are the
averages of the corresponding per-sample statistics, matching the source
workflow. The final command writes assembled samples under
`dataset_raw/total/` with:

- `target`: self-consistent density, shape `[x, y, z]`, unstandardized.
- `feature`: standardized inputs, shape `[channel, x, y, z]`.

Use `--workers N` with `convert` or `collect` only after confirming that local
I/O capacity can support parallel readers. These tools never alter the source
SIESTA calculations.

Copy the initial-density channel's entries from `scale.yaml` to
`model.rho0_mean` and `model.rho0_std`. Channel numbering in `scale.yaml`
includes the target as channel 0, whereas assembled model features start from
the converted `feature1` channel.

## Training and evaluation

Standard density training:

```bash
python train.py --input input.yaml
```

Physics-aware training with the regularizers selected in the YAML file:

```bash
python train_physics.py --input input.yaml
```

Evaluate density error or density-gradient error:

```bash
python evaluate.py --input input.yaml
python evaluate_grad.py --input input.yaml
```

Checkpoints named `<epoch>_model.pt` contain model, optimizer, scheduler, and
epoch state. The final `model.pt` contains only the model state. Dataset split
metadata is written to `data.pkl`. These generated files are ignored by Git.

## Predict a SIESTA density

Set `predict.model`, `predict.target.path`, and
`predict.target.grid_spacing` in the input file, then run:

```bash
python predict.py --input input.yaml
```

The target may be one assembled HDF5 file, a directory of such files, or the
saved `data.pkl`. Prediction writes one `.RHO` grid in the current directory
for each selected sample. The current writer constructs an orthorhombic cell
from the mesh and scalar grid spacing; use a system-specific writer when the
actual cell is non-orthorhombic or cannot be reconstructed this way.

## Fixed-RHO SIESTA evaluation

A predicted density is evaluated in a non-self-consistent SIESTA calculation
that reads the external `.RHO` and keeps it fixed. For a consistent comparison,
predicted and reference calculations must use the same geometry, species,
pseudopotentials, basis, mesh, k points, occupations, and energy settings.

The ordinary fixed-density `Etot` mixes terms evaluated from the supplied
density with a band sum from the Hamiltonian built from that density. This is
not generally the most useful energy comparison for a non-self-consistent
machine-learned density. The added diagnostic reconstructs the basic
Kohn-Sham energy from the band sum:

```text
Etot_bs = Ebs - Uscf + Dxc_pcc + Ena + Uatm - Enaatm - Eions
```

Here `Dxc_pcc`, rather than the raw `Dxc`, restores the valence-density
expectation value of the exchange-correlation potential when nonlinear core
corrections are present. The derivation, source mapping, assumptions, and
limitations are documented in [`docs/energy_evaluation.md`](docs/energy_evaluation.md).

### Apply the SIESTA 4.1.5 output patch

The patch targets
[`YHKlab-RGLee/SIESTA-for-DeepSCF`](https://github.com/YHKlab-RGLee/SIESTA-for-DeepSCF)
commit `43eb5a811d3391c6b9e7c6367c7624363cb3b907`. From the top of that SIESTA
source tree, run:

```bash
/path/to/DeepSCF_Physics/tools/siesta/apply_dxc_pcc_etot_bs_siesta415.sh
```

The script verifies the source version and expected files, performs a dry run,
creates non-overwriting dated backups, and applies the patch. A second run is
an idempotent no-op. It does not compile SIESTA or start a calculation. See
[`tools/siesta/README.md`](tools/siesta/README.md) for the exact support scope.

### Compare completed calculations

Set the two calculation roots in `dft.predicted` and `dft.reference`. Direct
children are paired by directory name. By default, the fixed-RHO output is
read from each child's `OUT/` directory and the reference directly from its
child directory.

```bash
python -m tools.dft.evaluate --input input.yaml
python -m tools.dft.plot --input input.yaml
```

The evaluator is read-only with respect to calculation directories. It parses
`Etot_bs` for predicted calculations and `Etot` for references, plus one `.FA`
and one `.EIG` file in each output directory. The result pickle and parity
plots are analysis outputs, not proof of physical convergence; convergence
must be established from the original SIESTA outputs.

`Etot_bs` implements the basic Kohn-Sham decomposition used here. Optional
contributions such as external fields, DFT+U, charged-cell corrections,
molecular mechanics, metadynamics, or custom constraints require an explicit
term-by-term audit. It is an energy diagnostic, not a force functional and not
a universal replacement for SIESTA's total energy.

## Citation

```bibtex
@article{deepscf,
  author  = {Lee, Ryong-Gyu and Kim, Yong-Hoon},
  title   = {Convolutional network learning of self-consistent electron density via grid-projected atomic fingerprints},
  journal = {npj Computational Materials},
  volume  = {10},
  number  = {1},
  pages   = {248},
  year    = {2024},
  doi     = {10.1038/s41524-024-01433-0}
}
```

## License

See [`LICENSE`](LICENSE).
