# Fixed-density total energies from `Ebs`

## Purpose

DeepSCF supplies a predicted valence density to SIESTA. SIESTA builds a
Kohn-Sham Hamiltonian from that density and diagonalizes it without converging
the usual density-mixing loop. In this setting, the ordinary printed `Etot`
can combine density-matrix terms from the diagonalization with grid terms from
the supplied density. The `Etot_bs` diagnostic added here instead reconstructs
the basic Kohn-Sham energy from a band sum whose Hamiltonian was built from the
same supplied density.

The central result is

```text
Etot_bs = Ebs - Uscf + Dxc_pcc + Ena + Uatm - Enaatm - Eions.
```

This is a diagnostic for the basic pseudopotential Kohn-Sham decomposition.
It is not an unconditional replacement for every SIESTA `Etot`; optional
Hamiltonian and energy terms must be audited as described below.

## Density and Hamiltonian definitions

We use the following notation:

```text
n0(r) = predicted valence density supplied in the external RHO file
nc(r) = frozen partial-core density of an NLCC pseudopotential
H0    = H[n0], the Hamiltonian constructed from n0
D1    = output density matrix obtained by diagonalizing H0
n1(r) = valence density represented by D1
dn(r) = n1(r) - n0(r)
```

Thus `Etot_bs` is not a density-only functional. It depends on both the input
density that creates the potential and the output density matrix that supplies
the occupied-state band sum:

```text
Etot_bs = Etot_bs[n0, D1].
```

## Band-sum derivation

Suppressing spin indices, the basic pseudopotential Kohn-Sham energy is

```text
E[n] = Ts[n] + Eext[n] + EH[n] + Exc[n + nc] + EII,
```

where

```text
EH[n] = (1/2) integral n(r) VH[n](r) dr.
```

The occupied eigenvalue sum of a Hamiltonian constructed from the same density
contains the Hartree and exchange-correlation potentials:

```text
Ebs = Sum_i fi epsilon_i
    = Ts + Eext
    + integral n VH dr
    + integral n Vxc dr.
```

Because `integral n VH = 2 EH`, substituting the band sum into the direct
functional gives

```text
E[n] = Ebs
     - EH[n]
     + Exc[n + nc] - integral n(r) Vxc[n + nc](r) dr
     + EII.
```

The band sum is therefore not a total energy by itself. One Hartree
contribution must be removed, and the XC-potential expectation already in
`Ebs` must be replaced by the XC energy functional.

In SIESTA's neutral-atom bookkeeping, the terms map as follows:

| Mathematical term | SIESTA quantity |
|---|---|
| occupied eigenvalue sum, `Tr(D1 H0)` | `Ebs` |
| full valence Hartree energy of `n0` | `Uscf` |
| XC functional minus valence XC-potential expectation | `Dxc_pcc` |
| ionic and neutral-atom reference contribution | `Ena + Uatm - Enaatm - Eions` |

This mapping produces the implemented expression for `Etot_bs`.

## Why raw `Dxc` is not enough with partial-core charge

For an NLCC pseudopotential, the XC functional is evaluated using valence plus
partial-core density. With `nsd` diagonal spin components, SIESTA's raw
`cellXC` correction has the form

```text
Dxc = Exc[nv + nc]
    - Sum_s integral (nv,s + nc/nsd) Vxc,s dr.
```

The partial-core density affects the XC potential, but it is not made from
occupied Kohn-Sham orbitals. Consequently, the band sum contains only the
valence expectation `Sum_s integral nv,s Vxc,s dr`. Combining raw `Dxc` with
`Ebs` would subtract the partial-core expectation even though that expectation
was never present in the band sum.

The required correction is

```text
Epcc,xc = Sum_s integral (nc/nsd) Vxc,s dr

Dxc_pcc = Dxc + Epcc,xc
        = Exc[nv + nc] - Sum_s integral nv,s Vxc,s dr.
```

The patch accumulates `Epcc,xc` immediately after `cellXC`, while `Vscf` still
contains the pure XC potential, and then returns both the existing raw `Dxc`
and the new `Dxc_pcc`. This is why the added quantity is named `Dxc_pcc`: it is
the XC double-counting correction aligned with a valence-state band sum after
the partial-core term has been restored.

For one previously inspected 16-atom fixed-RHO example, the values were

```text
Dxc     = 171.6403799321 eV
Dxc_pcc =  85.9058457549 eV.
```

Using raw `Dxc` in that case shifts the reconstructed energy by about
`85.7345 eV`. This is a single diagnostic example, not a transferable estimate
of the NLCC correction for another material.

## Why `Etot_bs` helps for a fixed input density

The Hamiltonian is constructed from `n0`, while diagonalization gives `D1` and
`n1`. The corresponding Harris-Foulkes-type expression is

```text
EHF[n0, D1] = Ebs[n0, D1]
            - EH[n0]
            + Exc[n0 + nc]
            - integral n0 Vxc[n0 + nc] dr
            + EII.
```

Let `dn = n1 - n0`. Expanding the Hartree and XC functionals around `n0`
shows that their linear changes are

```text
delta EH  = integral VH[n0] dn dr
delta Exc = integral Vxc[n0 + nc] dn dr.
```

The same linear potential expectations are already present in the band sum.
They cancel in the Harris-Foulkes rearrangement, leaving an error that begins
at second order in `dn` under the usual assumptions. This stationary behavior
is the reason the band-sum reconstruction is useful when the supplied density
is close to, but not exactly, the density obtained by diagonalizing its
Hamiltonian.

By contrast, in the inspected fixed-RHO control flow, final `Ekin` and `Enl`
can be evaluated using `D1` while Hartree/XC grid quantities continue to be
evaluated from the externally restored `n0`. The ordinary direct `Etot` then
acts as a density-inconsistent hybrid rather than a direct functional of one
common density.

For the same single 16-atom example:

```text
Etot_ref = -1708.6134131156 eV
Etot     = -1706.5886172904 eV   error = +2.0247958252 eV
Etot_bs  = -1708.6457966828 eV   error = -0.0323835672 eV
```

The absolute error is about 62.5 times smaller in that example. This number is
not a dataset average and does not establish a universal improvement factor.
A dataset-level claim requires matched calculations, common convergence
criteria, and statistics over every retained sample.

## Relation to the ordinary SIESTA total energy

The native direct total energy uses quantities such as `Ekin`, `Enl`,
electrostatic differences, and `Exc` directly. It therefore does not need
`Dxc`, and the absence of `Dxc` from the native formula is not itself a bug.
Adding `Dxc` to that direct formula would double count XC terms.

`Dxc_pcc` is needed only for the alternative reconstruction that starts from
`Ebs`. At a fully self-consistent solution, and when identical optional terms
are included on both sides, the direct and band-sum forms are rearrangements
of the same Kohn-Sham energy.

The SIESTA neutral-atom variables should also not be confused:

```text
DUscf = electrostatic energy of the density difference from rho_atm
Uscf  = full valence-density Hartree energy

DEna    = Enascf - Enaatm       at final energy assembly
Ion-ion = Ena + Uatm - Enaatm - Eions.
```

The `Etot_bs` formula uses `Uscf`, not `DUscf`.

## Source implementation

The public patch is
[`tools/siesta/siesta-4.1.5-dxc-pcc-etot-bs.patch`](../tools/siesta/siesta-4.1.5-dxc-pcc-etot-bs.patch).
It targets SIESTA 4.1.5 at the exact commit documented in
[`tools/siesta/README.md`](../tools/siesta/README.md).

| Source file | Role in the patch |
|---|---|
| `Src/dhscf.F` | Integrates the partial-core XC-potential expectation after `cellXC`, performs MPI reduction, and exposes optional `Dxc_pcc_out`. |
| `Src/grdsam.F` | Carries and averages `Dxc_pcc` across `GridCellSampling` points. |
| `Src/m_energies.F90` | Adds stored energy member `Dxc_pcc`. |
| `Src/setup_hamiltonian.F` | Receives `Dxc_pcc` during Hamiltonian construction. |
| `Src/compute_energies.F90` | Propagates `Dxc_pcc` in energy-update paths. |
| `Src/final_H_f_stress.F` | Propagates the value in final-Hamiltonian evaluation. |
| `Src/local_DOS.F` | Keeps the extended `dhscf` interface consistent for local-DOS paths. |
| `Src/m_pexsi_local_dos.F90` | Keeps the extended interface consistent for PEXSI local-DOS paths. |
| `Src/siesta_analysis.F` | Keeps the extended interface consistent for analysis paths. |
| `Src/write_subs.F` | Forms `Etot_bs` and prints `Dxc`, `Dxc_pcc`, and `Etot_bs`. |

The output-side implementation is conceptually

```fortran
Etot_bs = Ebs - Uscf + Dxc_pcc
Etot_bs = Etot_bs + Ena + Uatm - Enaatm - Eions
```

and prints energies in eV following the surrounding SIESTA style. The patch
does not change the SCF loop, replace `Etot`, or integrate a new density inside
`write_subs.F`; it only transports the already computed diagnostic and prints
the reconstruction.

## Optional terms that require a separate audit

The short expression is deliberately limited. If a potential `Va` is added to
the Hamiltonian and its desired functional energy is `Ea`, a band-sum formula
usually needs

```text
Ca = Ea - Tr(D Va),
```

not simply `+Ea`. A contribution absent from the electronic Hamiltonian, such
as a purely ionic bias, is instead added directly. The following cases must be
checked term by term before treating `Etot_bs` as a complete total energy:

- external electric fields and dipole corrections;
- DFT+U or other orbital-dependent corrections;
- charged periodic cells and Madelung/background corrections;
- order-N charge constraints;
- spin-orbit paths not represented identically in the chosen band sum;
- molecular-mechanics terms, metadynamics, or other ionic bias energies;
- custom constraints or locally modified Hamiltonian terms;
- transport or nonequilibrium energy definitions.

For fractional occupations, distinguish internal energy from electronic free
energy. If SIESTA reports `FreeE = Etot - Temp*Entropy`, the corresponding
entropy term must be handled consistently before comparing states. Do not
compare one calculation's internal energy with another's free energy.

Finally, `Etot_bs` does not make printed forces variational derivatives of the
new diagnostic. Forces from a non-self-consistent predicted density require
their own theoretical and implementation audit.

## Recommended validation protocol

1. Use identical structure, basis, pseudopotentials, mesh, k points,
   occupations, and optional physics in the fixed-RHO and reference runs.
2. Confirm from the output that the supplied density was actually read and
   kept fixed by the intended executable.
3. Record `Ebs`, `Uscf`, `Dxc`, `Dxc_pcc`, `Ion-ion`, `Etot_bs`, and native
   `Etot`; verify the printed formula numerically.
4. Compare `Etot_bs` against a converged self-consistent reference for every
   matched sample, reporting both signed and absolute errors per system or per
   atom as appropriate.
5. Treat malformed, unconverged, mismatched, or missing-output calculations as
   failures rather than silently dropping them.
6. State explicitly which optional terms were disabled or audited.

The utilities in `tools/dft/` implement read-only parsing and matched-directory
pairing. They do not determine whether a SIESTA calculation is physically
converged; that remains a prerequisite checked from the calculation outputs.
