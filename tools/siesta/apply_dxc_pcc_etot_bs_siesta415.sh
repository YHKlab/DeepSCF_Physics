#!/bin/sh

set -eu

script_dir=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
patch_file="$script_dir/siesta-4.1.5-dxc-pcc-etot-bs.patch"

source_files="Src/m_energies.F90 Src/dhscf.F Src/grdsam.F
Src/final_H_f_stress.F Src/setup_hamiltonian.F Src/compute_energies.F90
Src/local_DOS.F Src/m_pexsi_local_dos.F90 Src/siesta_analysis.F
Src/write_subs.F"

if [ ! -f version.info ] || [ "$(tr -d '[:space:]' < version.info)" != "4.1.5" ]; then
    echo "ERROR: run this script from the top of a SIESTA 4.1.5 tree." >&2
    exit 1
fi

if [ ! -f "$patch_file" ]; then
    echo "ERROR: patch file not found: $patch_file" >&2
    exit 1
fi

for file in $source_files; do
    if [ ! -f "$file" ]; then
        echo "ERROR: source file not found: $file" >&2
        exit 1
    fi
done

base_marker_count=0
check_marker()
{
    if grep -Fq "$2" "$1"; then
        base_marker_count=$((base_marker_count + 1))
    fi
}

check_marker Src/m_energies.F90 "real(dp):: Dxc_pcc"
check_marker Src/dhscf.F "Dxc_pcc_out"
check_marker Src/grdsam.F "avDxc_pcc"
check_marker Src/final_H_f_stress.F "Exc, Dxc, Dxc_pcc"
check_marker Src/setup_hamiltonian.F "Dxc_pcc_out=Dxc_pcc"
check_marker Src/compute_energies.F90 "Dxc_pcc_out=Dxc_pcc"
check_marker Src/local_DOS.F "Dxc_pcc_out=Dxc_pcc"
check_marker Src/m_pexsi_local_dos.F90 "Dxc_pcc_out=Dxc_pcc"
check_marker Src/siesta_analysis.F "Dxc_pcc_out=Dxc_pcc"
check_marker Src/write_subs.F "siesta: Dxc_pcc ="

etot_marker_count=0
for marker in \
    "Etot_bs = Ebs - Uscf + Dxc_pcc" \
    "'siesta: Etot_bs ='" \
    "'siesta: ',       'Etot_bs ='"
do
    if grep -Fq "$marker" Src/write_subs.F; then
        etot_marker_count=$((etot_marker_count + 1))
    fi
done

if [ "$base_marker_count" -eq 10 ] && [ "$etot_marker_count" -eq 3 ]; then
    echo "Dxc_pcc and Etot_bs patch is already applied; no files were changed."
    exit 0
fi

if [ "$base_marker_count" -ne 0 ] || [ "$etot_marker_count" -ne 0 ]; then
    echo "ERROR: a partial or incompatible Dxc_pcc/Etot_bs patch was detected." >&2
    echo "No files were changed. Review or restore the source first." >&2
    exit 1
fi

if ! command -v patch >/dev/null 2>&1; then
    echo "ERROR: the 'patch' command is required." >&2
    exit 1
fi

date_tag=$(date +%y%m%d)

backup_name()
{
    file=$1
    directory=${file%/*}
    name=${file##*/}
    stem=${name%.*}
    extension=${name##*.}
    printf '%s/%s_%s.%s\n' "$directory" "$stem" "$date_tag" "$extension"
}

for file in $source_files; do
    backup=$(backup_name "$file")
    if [ -e "$backup" ]; then
        echo "ERROR: backup already exists: $backup" >&2
        echo "No files were changed and no backups were overwritten." >&2
        exit 1
    fi
    if [ -e "$file.rej" ]; then
        echo "ERROR: pre-existing reject file found: $file.rej" >&2
        exit 1
    fi
done

if ! patch --dry-run --batch --forward -p1 < "$patch_file"; then
    echo "ERROR: the 4.1.5 Dxc_pcc/Etot_bs patch does not apply cleanly." >&2
    echo "No source files or backups were changed." >&2
    exit 1
fi

echo "Creating dated backups with tag $date_tag ..."
for file in $source_files; do
    backup=$(backup_name "$file")
    cp -p -- "$file" "$backup"
    echo "  $file -> $backup"
done

if ! patch --batch --forward --no-backup-if-mismatch -p1 < "$patch_file"; then
    echo "ERROR: patch application failed; restoring dated backups." >&2
    for file in $source_files; do
        backup=$(backup_name "$file")
        cp -p -- "$backup" "$file"
        if [ -f "$file.rej" ]; then
            rm -f -- "$file.rej"
        fi
    done
    exit 1
fi

echo "Dxc_pcc and Etot_bs source patch applied successfully."
echo "The Obj directory, executable, and calculation data were not changed."
echo "Rebuild the intended Obj tree separately before using the new output."
