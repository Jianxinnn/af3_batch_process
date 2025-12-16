#!/usr/bin/env python3
import argparse
import json
import os

from alphafold3_localbase import AlphaFoldModel


def is_prepared(job_like):
    """Heuristic: prepared entries have 'sequences' array; raw entries have 'protein' etc."""
    if isinstance(job_like, dict) and "sequences" in job_like:
        return True
    return False


def main():
    ap = argparse.ArgumentParser(description="Run AlphaFold3 from JSON, supporting raw -> prepare -> predict.")
    ap.add_argument("json_path", help="Path to JSON. Can be raw (name/protein/...) or prepared (sequences/...) list.")
    ap.add_argument("--device", default="cuda:0", help="CUDA device spec, e.g. cuda:0 or cuda:0,1")
    ap.add_argument("--model-seeds", default="234321", help="Comma-separated model seeds used when preparing from raw JSON")
    ap.add_argument("--single", action="store_true", help="Use single_prepare_sequences for raw JSON; default uses batch_prepare_sequences")
    ap.add_argument("--name-prefix", default=None, help="Optional prefix when preparing from raw JSON")
    ap.add_argument("--dialect", choices=["alphafoldserver", "alphafold3"], default="alphafoldserver", help="Output dialect for prepared JSON")
    ap.add_argument("--version", type=int, default=None, help="Top-level version to emit (default: 1 for server; 4 for alphafold3)")
    ap.add_argument("--output-dir", default=None, help="Output directory (not used in batch mode)")
    args = ap.parse_args()

    if not os.path.exists(args.json_path):
        raise SystemExit(f"JSON path not found: {args.json_path}")

    with open(args.json_path) as f:
        raw = json.load(f)

    # Normalize input to a list for downstream prep; accept a single dict as well
    if isinstance(raw, list):
        data = raw
    elif isinstance(raw, dict):
        data = [raw]
    else:
        raise SystemExit("Input JSON must be a list or a dict")
    
    if args.output_dir:
        os.makedirs(args.output_dir, exist_ok=True)
        model = AlphaFoldModel(input_dir=args.output_dir, output_dir=args.output_dir)
    else:
        model = AlphaFoldModel()

    # Decide whether data is raw or already prepared
    if data and is_prepared(data[0]):
        # Already AF3 prepared; hand off directly as batch jobs
        prepared = data
    else:
        # Treat as raw input; convert via prepare_sequences
        # Respect name prefix by injecting into each entry if requested
        if args.name_prefix:
            for case in data:
                if isinstance(case, dict) and "name" in case:
                    case["name"] = f"{args.name_prefix}_{case['name']}"

        seeds = ",".join([p.strip() for p in args.model_seeds.split(',') if p.strip()])
        if args.single:
            prepared = model.single_prepare_sequences(data, seeds, dialect=args.dialect, version=args.version)
        else:
            prepared = model.batch_prepare_sequences(data, seeds, dialect=args.dialect, version=args.version)

    # Prepare input files and run prediction
    # For multi-GPU, allow comma-separated device ids like cuda:0,1
    gpu_spec = args.device
    if ":" in gpu_spec:
        ids = gpu_spec.split(":")[1]
        gpu_ids = ids.split(",")
        num = len(gpu_ids)
    else:
        num = 1

    input_info = model.prepare_input(prepared, batch_mode=True, num_gpus=num, name_prefix="batch")

    res = model.run_prediction(input_info, device=args.device)
    if res:
        print(res)


if __name__ == "__main__":
    main()
