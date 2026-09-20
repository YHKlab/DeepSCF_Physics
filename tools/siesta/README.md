# SIESTA 4.1.5 energy-output patch

This directory adds the `Dxc_pcc` and `Etot_bs` diagnostics required by the
fixed-density energy workflow documented in
[`docs/energy_evaluation.md`](../../docs/energy_evaluation.md).

The patch targets the public
[`YHKlab-RGLee/SIESTA-for-DeepSCF`](https://github.com/YHKlab-RGLee/SIESTA-for-DeepSCF)
source at commit `43eb5a811d3391c6b9e7c6367c7624363cb3b907` (`version.info` =
`4.1.5`). It is distinct from the earlier patch for the private
SIESTA 4.1-b4 MS-DFT source family.

Run the script from the top of a clean SIESTA source tree:

```bash
/path/to/DeepSCF_Physics/tools/siesta/apply_dxc_pcc_etot_bs_siesta415.sh
```

The script checks the source state, performs a dry run, creates non-overwriting
dated backups in `Src/`, and then applies the version-specific patch. It does
not rebuild SIESTA or run a calculation. Re-running it after a complete
application is a no-op.

Regenerate the build directory from the patched `Src/Makefile` before
compilation (normally with `Src/obj_setup.sh` in a new Obj directory), then add
the site's `arch.make` and build there. Do not rely on the tracked
`Obj_test/Makefile` at the supported commit: it predates three feature-source
objects (`rhoov.o`, `kindens.o`, and `grddens.o`) that are already present in
`Src/Makefile`. This mismatch is independent of the energy patch.

For example, after preserving the site's compiler and library configuration:

```bash
mkdir Obj_dxc
cd Obj_dxc
../Src/obj_setup.sh
# Install or generate arch.make for this machine, then:
make -j2 siesta
```

Do not interpret `Etot_bs` as a complete total energy when optional terms such
as external fields, DFT+U, charged-cell corrections, molecular mechanics, or
metadynamics are active. See the detailed energy note for the supported basic
Kohn-Sham decomposition and finite-temperature caveats.
