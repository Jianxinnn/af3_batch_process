# AlphaFold3 local batch prediction (core workflow)

## What you do (3 steps)

1) Start from your initial input (FASTA or JSON)  
2) Build/prepare AF3 input JSON (raw → prepared)  
3) Split across GPUs and run, then collect output folders

---

## One-shot run (recommended entrypoint)

Use `af3_run_from_json.py` to run the whole pipeline; it auto-detects whether your JSON is raw or prepared:

```bash
python af3_run_from_json.py <input.json> --device cuda:<id[,id...]> [--output-dir <dir>] \
  [--model-seeds <seed[,seed...]>] [--dialect alphafoldserver|alphafold3] [--version <N>] [--name-prefix <prefix>] [--single]
```

- Raw cases (no `sequences`) → prepare → predict
- Prepared jobs (has `sequences`) → predict

---

## What your starting input looks like (pick one)

### Start A: you have a protein FASTA

1) FASTA → raw cases JSON:

```bash
python fasta_to_af3json.py <proteins.fasta> -o <cases.raw.json> \
  [--name-prefix <prefix>] [--ligand/--rna/--dna/--ccd ...] [--ligand-file/--rna-file/--dna-file/--ccd-file ...]
```

FASTA looks like:

```fasta
>job1
MKT...
>job2
GAA...
```

2) raw cases → predict:

```bash
python af3_run_from_json.py <cases.raw.json> --device cuda:<id[,id...]>
```

### Start B: you already have a raw cases JSON

Raw cases is a `List[Dict]`; each item is a job with at least `name` and `protein` (optionally `ligand/ccd/rna/dna`).

```bash
python af3_run_from_json.py <cases.raw.json> --device cuda:<id[,id...]>
```

Raw cases JSON looks like:

```json
[
  {
    "name": "job1",
    "protein": "MKT..."
  },
  {
    "name": "job2",
    "protein": "GAA...",
    "ligand": "CC(=O)N...",
    "ccd": "ATP,MG",
    "rna": "GCAG...",
    "dna": "GCTC..."
  }
]
```

### Start C: you already have a prepared jobs JSON

Prepared jobs is a `List[Dict]`; each item contains `name/modelSeeds/sequences/dialect/version`.

```bash
python af3_run_from_json.py <jobs.prepared.json> --device cuda:<id[,id...]>
```

Prepared jobs JSON looks like (example uses `alphafoldserver` dialect):

```json
[
  {
    "name": "job1",
    "modelSeeds": [234321],
    "dialect": "alphafoldserver",
    "version": 1,
    "sequences": [
      {"proteinChain": {"sequence": "MKT...", "useStructureTemplate": true, "count": 1}},
      {"ligand": {"smiles": "CC(=O)N..."}}
    ]
  }
]
```

---

## How to scale batch runs

- Single GPU: `--device cuda:0`
- Multi GPU: `--device cuda:0,1,2,...` (the script splits jobs into N input files and launches N `run_alphafold.py` processes)
- Incremental reruns: append new jobs to your JSON and rerun; finished jobs are auto-skipped (based on the 3 output files being present and non-empty)

---

## What to configure (`config.yaml`)

`alphafold3_localbase.py` reads `config.yaml` in this directory. Key fields:
- `model.weights_path`, `model.database_path`, `model.env_name`
- `paths.input_dir`, `paths.output_dir`
- `binaries.jackhmmer/hmmbuild/hmmsearch/nhmmer`
- `execution.base_dir` (directory that contains `run_alphafold.py`)

---

## Where outputs go (layout)

Output root: `config.yaml: paths.output_dir` (or `af3_run_from_json.py --output-dir`).  
Each job writes into: `<output_dir>/<name.lower()>/`:
- `<name.lower()>_model.cif`
- `<name.lower()>_confidences.json`
- `<name.lower()>_summary_confidences.json`

---

## Practical recommendations

- Same ligand/RNA/DNA for all proteins in one FASTA: use Start A and pass single values (`--ligand/--rna/--dna`) to broadcast to every FASTA record.
- Different ligand/RNA/DNA/CCD per protein: use Start A with per-record mapping files (`--ligand-file/--rna-file/--dna-file/--ccd-file`) so each FASTA entry gets its own values (line count must match FASTA records).
- If your inputs already come from a table/DB (e.g., CSV): generate a raw cases JSON directly (Start B) and run `af3_run_from_json.py` without going through FASTA mapping.

