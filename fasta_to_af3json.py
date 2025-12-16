#!/usr/bin/env python3
import argparse
import json
import os
import re
from pathlib import Path


AMINO_ALPHABET = set(list("ACDEFGHIKLMNPQRSTVWYBXZJUO*-"))


def read_fasta(path: str):
    """Parse FASTA and yield (header, sequence) tuples.

    Robust to stray lines: only accept sequence lines that consist solely of
    letters in AMINO_ALPHABET (after uppercasing) and '-','*'. Others are ignored.
    """
    header = None
    seq_parts = []
    with open(path, 'r') as f:
        for raw in f:
            line = raw.strip()
            if not line:
                continue
            if line.startswith('>'):
                if header is not None and seq_parts:
                    yield header, ''.join(seq_parts)
                header = line[1:].strip()
                seq_parts = []
                continue
            # Only accept valid protein letters (uppercased) and '-' or '*'
            up = line.upper().replace(' ', '')
            if all(ch in AMINO_ALPHABET for ch in up):
                seq_parts.append(up)
            else:
                # ignore stray non-FASTA content (e.g., SMILES lines) gracefully
                continue
        # flush
        if header is not None and seq_parts:
            yield header, ''.join(seq_parts)


def sanitize_name(header: str) -> str:
    # take up to first whitespace
    base = header.split()[0]
    # replace problematic chars
    base = re.sub(r"[^A-Za-z0-9._-]+", "_", base)
    return base


def read_mapping_lines(path: str, kind: str):
    """Read mapping lines for ligand/rna/dna/ccd.

    - Accept lines like "NAME: VALUE" or just "VALUE".
    - Strip whitespace; skip empty/comment lines.
    - For CCD, uppercase the code(s); for nucleic, uppercase sequence; for ligand keep as-is.
    Returns list[str].
    """
    vals = []
    with open(path, 'r') as f:
        for raw in f:
            line = raw.strip()
            if not line or line.startswith('#'):
                continue
            if ':' in line:
                # NAME: VALUE or KEY:VALUE
                _, val = line.split(':', 1)
                val = val.strip()
            else:
                val = line
            if kind == 'ccd':
                vals.append(val.strip().upper())
            elif kind in ('rna', 'dna'):
                vals.append(val.strip().upper().replace(' ', ''))
            else:  # ligand (SMILES)
                vals.append(val.strip())
    return vals


def build_raw_cases(records,
                    ligand_list=None,
                    ligand_single=None,
                    ccd_list=None,
                    ccd_single=None,
                    rna_list=None,
                    rna_single=None,
                    dna_list=None,
                    dna_single=None,
                    name_prefix: str | None = None):
    """Build raw input cases expected by AlphaFoldModel.(single|batch)_prepare_sequences.

    Output schema per entry:
      {"name": str, "protein": str, ["ligand"|"ccd"|"rna"|"dna"]: str, ...}
    """
    cases = []
    n = len(records)

    def pick(lst, single):
        if lst is not None:
            if len(lst) != n:
                raise SystemExit(
                    f"Mapping lines count mismatch: got {len(lst)} values for {n} FASTA records.")
            return lst
        if single is None:
            return [None] * n
        return [single] * n

    ligands = pick(ligand_list, ligand_single)
    ccds = pick(ccd_list, ccd_single)
    rnas = pick(rna_list, rna_single)
    dnas = pick(dna_list, dna_single)

    for i, (header, protein_seq) in enumerate(records):
        name = sanitize_name(header)
        if name_prefix:
            name = f"{name_prefix}_{name}"

        # Put 'name' last to avoid affecting index-based chain IDs
        entry = {"protein": protein_seq}
        if rnas[i]:
            entry["rna"] = rnas[i]
        if dnas[i]:
            entry["dna"] = dnas[i]
        if ligands[i]:
            entry["ligand"] = ligands[i]
        if ccds[i]:
            entry["ccd"] = ccds[i]
        entry["name"] = name
        cases.append(entry)

    return cases


def main():
    ap = argparse.ArgumentParser(
        description="Convert FASTA to raw JSON for AlphaFoldModel.prepare_sequences.")
    ap.add_argument("fasta", help="Input FASTA of protein sequences")
    ap.add_argument("-o", "--out", default=None,
                    help="Output JSON path (default: <fasta_basename>.raw.json)")
    ap.add_argument("--name-prefix", default=None,
                    help="Optional prefix for job names")

    # One-to-one mapping files
    ap.add_argument("--ligand-file", help="File with SMILES, one per FASTA record")
    ap.add_argument("--ccd-file", help="File with CCD code, one per FASTA record")
    ap.add_argument("--rna-file", help="File with RNA sequence, one per FASTA record")
    ap.add_argument("--dna-file", help="File with DNA sequence, one per FASTA record")

    # Broadcast singles
    ap.add_argument("--ligand", help="Single SMILES to broadcast to all records")
    ap.add_argument("--ccd", help="Single CCD code to broadcast to all records")
    ap.add_argument("--rna", help="Single RNA sequence to broadcast to all records")
    ap.add_argument("--dna", help="Single DNA sequence to broadcast to all records")

    args = ap.parse_args()

    records = list(read_fasta(args.fasta))
    if not records:
        raise SystemExit("No FASTA records parsed. Check input.")

    lig_list = read_mapping_lines(args.ligand_file, 'ligand') if args.ligand_file else None
    ccd_list = read_mapping_lines(args.ccd_file, 'ccd') if args.ccd_file else None
    rna_list = read_mapping_lines(args.rna_file, 'rna') if args.rna_file else None
    dna_list = read_mapping_lines(args.dna_file, 'dna') if args.dna_file else None

    cases = build_raw_cases(
        records,
        ligand_list=lig_list,
        ligand_single=args.ligand,
        ccd_list=ccd_list,
        ccd_single=args.ccd,
        rna_list=rna_list,
        rna_single=args.rna,
        dna_list=dna_list,
        dna_single=args.dna,
        name_prefix=args.name_prefix,
    )

    out_path = args.out
    if out_path is None:
        stem = Path(args.fasta).name
        out_path = os.path.splitext(stem)[0] + ".raw.json"

    with open(out_path, 'w') as f:
        json.dump(cases, f, indent=2)

    print(f"Wrote {len(cases)} case(s) to {out_path}")


if __name__ == "__main__":
    main()
