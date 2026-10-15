# DeepSCF Physics

Electron-density prediction with physics-aware training and SIESTA-based evaluation.

<p align="center">
  <img height="240" src="./logo/shematics.png" alt="DeepSCF workflow"/>
</p>

DeepSCF Physics predicts self-consistent electron densities from an initial
electron density and grid-projected atomic features. The workflow covers
SIESTA-grid preprocessing, standard or physics-aware training, density
prediction and evaluation, and fixed-density SIESTA energy analysis using
the [`Etot_bs` band-sum reconstruction](#why-etot_bs-for-a-fixed-density).

This repository extends the original DeepSCF implementation. The underlying
method is described in [*Convolutional network learning of self-consistent
electron density via grid-projected atomic fingerprints*](https://doi.org/10.1038/s41524-024-01433-0).

## Features

- SIESTA-grid-to-HDF5 preprocessing with recorded channel scaling.
- Standard density training and optional gradient or Laplacian regularization
  with 3-point or 5-point finite-difference operators.
- Density-error evaluation for saved train or test splits.
- Prediction of SIESTA-format `.RHO` files.
- [`Etot_bs` energy evaluation](#siesta-energy-evaluation) for predicted
  densities, including partial-core XC bookkeeping through `Dxc_pcc`, a
  version-specific SIESTA 4.1.5 output patch, and read-only comparison of
  completed predicted-density and self-consistent calculations.

This repository does not include trained weights, datasets, pseudopotentials,
completed DFT results, or calculation-submission scripts.

## Installation

The code preserves the Python 3.7 / PyTorch 1.12 environment of the original
implementation.

```bash
git clone --branch physics-update \
  https://github.com/YHKlab/DeepSCF_Physics.git
cd DeepSCF_Physics
python -m pip install -r requirements.txt
```

For GPU use, install a PyTorch build matching the local CUDA stack rather than
blindly replacing a working site installation.

## Configuration

Training, density evaluation, prediction, and DFT postprocessing merge a user
YAML file into [`config/input.yaml`](config/input.yaml). Start from the
documented six-channel example:

```bash
cp examples/input.yaml input.yaml
```

Run the preprocessing, training, density-evaluation, and prediction examples
from the repository root. DFT postprocessing instead runs from the calculation
case directory, as shown below. For YAML-based commands, select a different
file with `--input path/to/input.yaml`; `--default` replaces the base
configuration. Preprocessing uses positional arguments, not this YAML file.

The most important fields are:

- `dataset.path`: directory containing assembled `.h5` samples.
- `model.input_layers`: number of selected input channels.
- `model.input_layers_index`: zero-based channel indices loaded from each HDF5
  `feature` array.
- `model.rho0_index`: zero-based index of the standardized initial density in
  the full HDF5 `feature` array, before `model.input_layers_index` is applied.
- `model.rho0_mean` and `model.rho0_std`: scaling used to reconstruct the
  physical initial density.
- `train.loss.regularization`: optional physics losses; the example enables
  `gradient_3p`.

## Dataset preparation

Each completed source calculation must be a direct child of a calculation
root and contain `Molecular.RHO` plus the requested grid-feature files. The
default feature order is defined in `tools/preprocess/convert.py` and can be
overridden with `--features`.

These tools consume existing grids; they do not run SIESTA or generate missing
feature files. The converter takes the first spin component (`grid[0]`) of
each grid, rather than summing or preserving multiple spin components. Check
that this matches the intended density representation before conversion.

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
SIESTA calculations, but existing generated HDF5 files and `scale.yaml` can
be overwritten. Use a fresh output directory for a new preprocessing run.

Copy the initial-density channel's entries from `scale.yaml` to
`model.rho0_mean` and `model.rho0_std`. Channel numbering in `scale.yaml`
includes the target as channel 0, whereas assembled model features start from
the converted `feature1` channel. With the default order, `feature1` is
`rhoatom`, so its scaling is `mean1` / `std1` and its model index is 0.
The example's selected indices `[0, 1, 2, 7, 8, 11]` correspond to `rhoatom`,
`Molecular.IOCH`, `rhoover`, `tau_tf`, `tau_w`, and `alpha`. Replace the example
scaling values with those from your own dataset and preserve the same channel
order and scaling when using a trained checkpoint.

## Training

Standard density training:

```bash
python train.py --input input.yaml
```

Physics-aware training with the regularizers selected in the YAML file:

```bash
python train_physics.py --input input.yaml
```

Supported regularizers are `gradient_3p`, `gradient_5p`, `laplacian_3p`, and
`laplacian_5p`. Set their weights with
`train.loss.regularization_weight.gradient` or
`train.loss.regularization_weight.laplacian`, respectively. The example uses
`gradient_3p`; `train.py` uses only the density loss.

Checkpoints named `<epoch>_model.pt` contain model, optimizer, scheduler, and
epoch state. The final `model.pt` contains only the model state. Dataset split
metadata is written to `data.pkl`. These generated files are ignored by Git.

## Density evaluation

Set `evaluate.model` to a saved checkpoint and `evaluate.target` to the saved
`data.pkl`. Use `is_test: true` for the test split or `is_test: false` for the
train split, then run:

```bash
python evaluate.py --input input.yaml
```

The evaluator writes per-sample relative-L1 density errors (%) to
`evaluate.txt` and an aggregate error to `summary.txt`. The aggregate is
100 times the summed absolute density error divided by the summed reference
density over the selected samples, not the mean or median of per-sample
percentages. Evaluation measures density accuracy independently of the
regularizer used during training.

Both Test and Train evaluations write the same `evaluate.txt` and
`summary.txt` in the current directory. Preserve those outputs before changing
the split or checkpoint and rerunning; they are overwritten, not appended as
separate split results. Evaluation can also overwrite a training
`summary.txt` if run in the training directory.

## Density prediction

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
`predict.target.grid_spacing` is in angstroms. Existing same-name `.RHO` files
and `predict.txt` in the current directory can be overwritten; do not run
prediction in a directory containing source or reference density files.

## SIESTA energy evaluation

A predicted density is evaluated in a non-self-consistent SIESTA calculation
that reads the external `.RHO` and keeps it fixed. For a consistent comparison,
predicted and reference calculations must use the same geometry, species,
pseudopotentials, basis, mesh, k points, occupations, and energy settings.

### Why `Etot_bs` for a fixed density?

Keeping the supplied density fixed does not mean that diagonalizing its
Hamiltonian reproduces that density. Distinguish:

- $n_0$: predicted valence density read from the external `.RHO` file;
- $n_c$: frozen partial-core density used by a nonlinear core correction;
- $H_0=H[n_0]$: Hamiltonian constructed from the supplied density;
- $D_1$: occupied-state density matrix obtained by diagonalizing $H_0$;
- $n_1$: valence density represented by $D_1$, generally different from $n_0$.

In the fixed-RHO control flow considered here, kinetic and nonlocal terms can
be evaluated using $D_1$, while Hartree and XC grid terms use the restored
$n_0$. The ordinary printed `Etot` can therefore mix quantities from two
different densities. This is the motivation for a separate band-sum
diagnostic, not a claim that SIESTA's ordinary self-consistent energy formula
is generally incorrect. In particular, the absence of `Dxc` from the native
direct-energy formula is not itself an error; adding it there would double
count XC terms.

### Band-sum reconstruction

The occupied eigenvalue sum is $E_{\mathrm{bs}}=\operatorname{Tr}(D_1H_0)$.
It contains potential-energy expectations as well as kinetic energy, so it
is not a total energy on its own. For the basic pseudopotential Kohn-Sham
decomposition, the Harris-Foulkes-type reconstruction is, with spin indices
suppressed,

$$
\begin{aligned}
E_{\mathrm{HF}}[n_0,D_1]
&=E_{\mathrm{bs}}[n_0,D_1]-E_H[n_0]+E_{\mathrm{xc}}[n_0+n_c] \\
&\quad-\int n_0(\mathbf r)V_{\mathrm{xc}}[n_0+n_c](\mathbf r)\,d\mathbf r
+E_{\mathrm{II}}.
\end{aligned}
$$

Here $E_H$ is the Hartree energy and $E_{\mathrm{II}}$ the ionic contribution.
At self-consistency, the Hartree-potential expectation in the band sum is
$2E_H$, so one Hartree energy must be subtracted; the XC-potential expectation
is replaced by the XC functional energy. For a fixed input density, the
correction terms above are evaluated from $n_0$, while the band sum comes from
$D_1$. Thus this diagnostic depends on both $n_0$ and $D_1$; it is not simply
the direct density functional $E[n_0]$.

In SIESTA's neutral-atom bookkeeping, the implemented expression is:

```text
Etot_bs = Ebs - Uscf + Dxc_pcc + Ena + Uatm - Enaatm - Eions
```

| Printed or stored quantity | Meaning in the reconstruction |
|---|---|
| `Ebs` | Occupied-state band sum, $\operatorname{Tr}(D_1H_0)$. |
| `Uscf` | Full valence Hartree energy of $n_0$, subtracted once. |
| `Dxc_pcc` | XC functional energy minus the valence XC-potential integral evaluated using $n_0$. |
| `Ena + Uatm - Enaatm - Eions` | Ionic contribution expressed through SIESTA's neutral-atom reference terms. |

Use `Uscf`, not `DUscf`: the latter is the electrostatic energy of the density
difference from the neutral-atom density, not the full Hartree energy.

### Why `Dxc_pcc`, not raw `Dxc`?

With nonlinear core corrections, `cellXC` evaluates XC using valence plus
partial-core density. In the same spin-suppressed notation, its raw
double-counting correction is

$$
D_{\mathrm{xc}}
=E_{\mathrm{xc}}[n_0+n_c]
-\int (n_0+n_c)V_{\mathrm{xc}}[n_0+n_c]\,d\mathbf r.
$$

The partial-core density affects the potential but is not formed by the
occupied valence orbitals. The band sum therefore has no separate
partial-core expectation to subtract. Using raw `Dxc` would subtract that
contribution anyway. The required correction restores it:

$$
\begin{aligned}
D_{\mathrm{xc,pcc}}
&=D_{\mathrm{xc}}+\int n_cV_{\mathrm{xc}}[n_0+n_c]\,d\mathbf r \\
&=E_{\mathrm{xc}}[n_0+n_c]-\int n_0V_{\mathrm{xc}}[n_0+n_c]\,d\mathbf r.
\end{aligned}
$$

For spin-polarized calculations, the partial-core integral is summed over
spin channels with $n_c/n_{\mathrm{sd}}$ in each diagonal channel; the detailed
note gives the full expression. Without a partial-core contribution,
`Dxc_pcc` reduces to raw `Dxc`.

In the [output patch](tools/siesta/siesta-4.1.5-dxc-pcc-etot-bs.patch),
`Src/dhscf.F` accumulates this partial-core integral immediately after
`cellXC`, while `Vscf` still contains only the XC potential, and performs the
MPI reduction. The result is propagated through the energy routines and grid
sampling. `Src/write_subs.F` assembles the expression above from available
terms and prints `Dxc`, `Dxc_pcc`, and `Etot_bs` in eV. The patch leaves native
`Etot` and the SCF iteration unchanged; it adds a separate diagnostic.

### Interpretation and limitations

Near self-consistency, the Harris-Foulkes rearrangement cancels first-order
Hartree/XC density-response terms under the usual smoothness and occupation
assumptions. This motivates its use for approximate input densities, but does
not guarantee a smaller error for an arbitrary prediction. At a fully
self-consistent solution, the direct and band-sum forms agree when the same
physical terms are included on both sides.

The short `Etot_bs` expression covers the basic Kohn-Sham decomposition.
External fields and dipole corrections, DFT+U, charged-cell corrections,
custom constraints, spin-orbit terms, molecular mechanics, metadynamics, and
nonequilibrium extensions require a term-by-term audit. With fractional
occupations, compare internal energies consistently; `Etot_bs` does not
automatically include the entropy correction of an electronic free energy.
It also does not make printed forces variational derivatives of this
diagnostic. Forces require their own validation.

See [`docs/energy_evaluation.md`](docs/energy_evaluation.md) for the longer
derivation, full spin-dependent PCC expression, complete source mapping, and
validation protocol.

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
[`tools/siesta/README.md`](tools/siesta/README.md) for supported sources and
rebuild instructions. Rebuild SIESTA and run the fixed-density calculations
with the patched executable before evaluating their outputs.

### Compare completed calculations

Set `is_test` and the reference calculation root in `dft.reference`. The
evaluator automatically reads predicted calculations from `calc_test` when
`is_test: true`, or `calc_train` when `is_test: false`, relative to the current
working directory. Direct children are paired by directory name. By default,
the fixed-RHO output is read from each child's `OUT/` directory and the
reference directly from its child directory.

Run from the directory containing `calc_test` or `calc_train`, not from inside
one of those directories. Make the cloned code importable without moving the
calculation data, and use an absolute YAML path:

```bash
DEEPSCF_REPO=/path/to/DeepSCF_Physics
cd /path/to/evaluation_case
PYTHONPATH="$DEEPSCF_REPO${PYTHONPATH:+:$PYTHONPATH}" \
  python -m tools.dft.evaluate --input /path/to/evaluation_case/input.yaml
PYTHONPATH="$DEEPSCF_REPO${PYTHONPATH:+:$PYTHONPATH}" \
  python -m tools.dft.plot --input /path/to/evaluation_case/input.yaml
```

Use absolute data paths in the YAML as well: relative paths are resolved from
the working directory, not from the YAML's location. For `is_test: true` and
the default output subdirectories, matching sample `0001` requires:

```text
/path/to/evaluation_case/calc_test/0001/OUT/
    stdout.txt, one *.FA, one *.EIG
<dft.reference>/0001/
    stdout.txt, one *.FA, one *.EIG
```

By default, `dft.predicted_energy_label` is `Etot_bs` and
`dft.reference_energy_label` is `Etot`. `stdout.txt` must contain the selected
energy label and atom count. The evaluator takes the last matching energy
entry and fails if the requested label is absent; it does not silently fall
back to ordinary `Etot`. Force and eigenvalue files are required even when
only the energy comparison is of interest. Verify that the last matching
entry belongs to the intended completed calculation.

The evaluator writes `dft_results_test.pkl` or `dft_results_train.pkl` for the
selected split, and the plotting command reads the same file automatically.
The stored `tes` and `tes0` arrays are already normalized to **eV/atom**.
Compute absolute energy errors in meV/atom as
`1000 * abs(tes - tes0)`; do not divide by the atom count again. This conversion
does not change the stored units. The plotting command writes
`dft_results_<split>_summary.txt` and `dft_results_<split>_<quantity>.png` parity
plots. Its `tes` MAE remains in eV/atom; forces are in eV/angstrom and
HOMO/LUMO energies in eV. Repeating either command can overwrite its analysis
outputs.

The evaluator is read-only with respect to calculation directories. It parses
`Etot_bs` for predicted calculations and `Etot` for references, plus one `.FA`
and one `.EIG` file in each output directory. The result pickle and parity
plots are analysis outputs, not proof of physical convergence; convergence
must be established from the original SIESTA outputs.

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
