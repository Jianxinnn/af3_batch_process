from typing import Dict, List, Optional
from pathlib import Path
import json
import os
import yaml
import sys
import re
import shutil
sys.path.append(os.path.dirname(__file__))
from execution import run_alphafold

# Default modelSeeds if none provided in input
DEFAULT_MODEL_SEEDS = [234321]


class AlphaFoldModel:
    def __init__(self, input_dir: str = None, output_dir: str = None):
        self.config = self._load_config()
        self.INPUT_DIR = self.config["paths"]["input_dir"] if input_dir is None else input_dir
        self.OUTPUT_DIR = self.config["paths"]["output_dir"] if output_dir is None else output_dir
        
    def _load_config(self) -> Dict:
        """Load configuration from config.yaml"""
        config_path = os.path.join(os.path.dirname(__file__), "config.yaml")
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    
    @property
    def name(self) -> str:
        return "AlphaFold 3"
        
    @property
    def description(self) -> str:
        return "DeepMind's AlphaFold 3 for protein structure prediction"
        
    @property
    def required_env(self) -> str:
        return self.config["model"]["env_name"]
        
    @property
    def model_weights_path(self) -> str:
        return self.config["model"]["weights_path"]
    
    @property
    def database_path(self) -> str:
        return self.config["model"]["database_path"]
    
    @property
    def output_structure_pattern(self) -> str:
        return "{}_model.cif"   
    
    @property
    def output_confidence_pattern(self) -> str:
        return "{}_confidences.json"   
    
    @property
    def output_summary_pattern(self) -> str:
        return "{}_summary_confidences.json"   
    
    def input_template(self, job_name, model_seeds, sequences, dialect: str = "alphafoldserver", version: Optional[int] = None, extras: Optional[Dict] = None) -> Dict:
        """Construct top-level input JSON for AF3.

        dialect: 'alphafoldserver' (default) or 'alphafold3'
        version: if None → 1 for 'alphafoldserver', 4 for 'alphafold3'
        extras: optional dict with keys among {'bondedAtomPairs','userCCD','userCCDPath'}
        """
        if version is None:
            version = 4 if dialect == "alphafold3" else 1
        tpl = {
            "name": job_name,
            "modelSeeds": model_seeds,
            "sequences": sequences,
            "dialect": dialect,
            "version": version,
        }
        if extras:
            for k in ("bondedAtomPairs", "userCCD", "userCCDPath"):
                if k in extras and extras[k] is not None:
                    tpl[k] = extras[k]
        return tpl
    
    def single_prepare_sequences(self, sequences: List[Dict], model_seeds, dialect: str = "alphafold3", version: Optional[int] = None) -> List[Dict]:
        """Prepare a list of jobs from raw cases for a single-GPU style run.

        Raw case: {'protein': '...', ['rna'|'dna'|'ligand'|'ccd']: '...', 'name': '...'}
        - dialect 'alphafold3': entities include 'id' letters and use keys 'protein'/'rna'/'dna'/'ligand'.
        - dialect 'alphafoldserver': entities use 'proteinChain'/'rnaSequence'/'dnaSequence'/'ligand'/'ccdCode'.
        """
        prepared = []
        model_seeds = [int(seed.strip()) for seed in str(model_seeds).split(",") if seed.strip().isdigit()]

        for case in sequences:
            job_name = case.get('name', 'job')
            prepared_case = []
            extras = {
                'bondedAtomPairs': case.get('bondedAtomPairs'),
                'userCCD': case.get('userCCD'),
                'userCCDPath': case.get('userCCDPath'),
            }

            chain_idx = 0
            for key, seq in case.items():
                if key in ('name', 'bondedAtomPairs', 'userCCD', 'userCCDPath'):
                    continue
                if seq is None:
                    continue
                cleaned = seq.upper().replace(' ', '').replace('\n', '') if key in ("protein", "rna", "dna", "ccd") else seq

                if dialect == 'alphafold3':
                    chain_id = chr(65 + chain_idx)
                    if key == 'ligand':
                        entity = {"ligand": {"id": chain_id, "smiles": seq}}
                    elif key == 'ccd':
                        entity = {"ligand": {"id": chain_id, "ccdCodes": list(self.re_split_codes(cleaned))}}
                    elif key == 'rna':
                        entity = {"rna": {"id": chain_id, "sequence": cleaned}}
                    elif key == 'dna':
                        entity = {"dna": {"id": chain_id, "sequence": cleaned}}
                    else:
                        entity = {"protein": {"id": chain_id, "sequence": cleaned}}
                else:
                    if key == 'ligand':
                        entity = {"ligand": {"smiles": seq}}
                    elif key == 'ccd':
                        entity = {"ccdCode": {"ccdCodes": cleaned}}
                    elif key == 'rna':
                        entity = {"rnaSequence": {"sequence": cleaned, "count": 1}}
                    elif key == 'dna':
                        entity = {"dnaSequence": {"sequence": cleaned, "count": 1}}
                    else:
                        entity = {"proteinChain": {"sequence": cleaned, "useStructureTemplate": True, "count": 1}}

                prepared_case.append(entity)
                chain_idx += 1

            job = self.input_template(job_name, model_seeds, prepared_case, dialect, version, extras)
            prepared.append(job)
        return prepared
    
    def batch_prepare_sequences(self, sequences: List[Dict], model_seeds, dialect: str = "alphafoldserver", version: Optional[int] = None) -> List[Dict]:
        """Prepare a list of jobs from raw cases for batch mode with selectable dialect."""
        prepared = []
        model_seeds = [int(seed.strip()) for seed in str(model_seeds).split(",") if seed.strip().isdigit()]

        for case in sequences:
            job_name = case.get('name', 'job')
            prepared_case = []
            extras = {
                'bondedAtomPairs': case.get('bondedAtomPairs'),
                'userCCD': case.get('userCCD'),
                'userCCDPath': case.get('userCCDPath'),
            }

            chain_idx = 0
            for key, seq in case.items():
                if key in ('name', 'bondedAtomPairs', 'userCCD', 'userCCDPath'):
                    continue
                if seq is None:
                    continue
                cleaned_seq = seq.upper().replace(' ', '').replace('\n', '') if key in ("protein", "rna", "dna", "ccd") else seq

                if dialect == 'alphafold3':
                    chain_id = chr(65 + chain_idx)
                    if key == 'ligand':
                        entity = {"ligand": {"id": chain_id, "smiles": seq}}
                    elif key == 'ccd':
                        entity = {"ligand": {"id": chain_id, "ccdCodes": list(self.re_split_codes(cleaned_seq))}}
                    elif key == 'rna':
                        entity = {"rna": {"id": chain_id, "sequence": cleaned_seq}}
                    elif key == 'dna':
                        entity = {"dna": {"id": chain_id, "sequence": cleaned_seq}}
                    else:
                        entity = {"protein": {"id": chain_id, "sequence": cleaned_seq}}
                else:
                    if key == 'ligand':
                        entity = {"ligand": {"smiles": seq}}
                    elif key == 'ccd':
                        entity = {"ccdCode": {"ccdCodes": cleaned_seq}}
                    elif key == 'rna':
                        entity = {"rnaSequence": {"sequence": cleaned_seq, "count": 1}}
                    elif key == 'dna':
                        entity = {"dnaSequence": {"sequence": cleaned_seq, "count": 1}}
                    else:
                        entity = {"proteinChain": {"sequence": cleaned_seq, "useStructureTemplate": True, "count": 1}}

                prepared_case.append(entity)
                chain_idx += 1

            case_template = self.input_template(job_name, model_seeds, prepared_case, dialect, version, extras)
            prepared.append(case_template)
        return prepared

    # End of class

    # Helpers
    @staticmethod
    def re_split_codes(text: str):
        """Split CCD codes string into list of uppercase tokens.

        Accept separators: comma, whitespace, semicolon. E.g. "ATP,MG" → ["ATP","MG"]
        """
        tokens = re.split(r"[\s,;]+", text.strip()) if text else []
        return [t.upper() for t in tokens if t]

    @staticmethod
    def _ensure_model_seeds(job: Dict) -> Dict:
        """Ensure 'modelSeeds' exists and is non-empty for a job dict.

        Mutates and returns the job dict. Uses DEFAULT_MODEL_SEEDS when absent/empty.
        """
        seeds = job.get("modelSeeds")
        if not seeds:
            job["modelSeeds"] = list(DEFAULT_MODEL_SEEDS)
        return job

    def prepare_input(self, alphafold_input, batch_mode=False, num_gpus=1, name_prefix="test123") -> Dict:
        # Save input JSON
        os.makedirs(self.INPUT_DIR, exist_ok=True)
        
        if batch_mode and num_gpus > 1:
            # Split data across GPUs
            input_paths = []
            total_jobs = len(alphafold_input)
            jobs_per_gpu = total_jobs // num_gpus
            remainder = total_jobs % num_gpus
            
            current_idx = 0
            for gpu_id in range(num_gpus):
                # Calculate number of jobs for this GPU
                n_jobs = jobs_per_gpu + (1 if gpu_id < remainder else 0)
                if n_jobs == 0:
                    continue
                    
                # Get jobs for this GPU
                gpu_jobs = alphafold_input[current_idx:current_idx + n_jobs]
                current_idx += n_jobs
                
                # Save to file
                input_path = os.path.join(self.INPUT_DIR, f"{name_prefix}_fold_input_gpu{gpu_id}.json")
                with open(input_path, "w") as f:
                    json.dump(gpu_jobs, f, indent=2)
                input_paths.append(input_path)
                
            return {
                "input_paths": input_paths,
                "batch_mode": True
            }
        else:
            # Single file for all jobs or single GPU
            input_path = os.path.join(self.INPUT_DIR, f"{name_prefix}_fold_input.json")
            with open(input_path, "w") as f:
                if batch_mode:
                    json.dump(alphafold_input, f, indent=2)
                else:
                    json.dump(alphafold_input[0], f, indent=2)
            return {
                "input_path": input_path,
                "batch_mode": batch_mode
            }

    def check_prediction_exists(self, job_name: str) -> bool:
        """Check if prediction results already exist for a given job"""
        job_dir = os.path.join(self.OUTPUT_DIR, job_name.lower())
        if not os.path.exists(job_dir):
            return False
            
        structure_path = os.path.join(job_dir, self.output_structure_pattern.format(job_name.lower()))
        confidence_path = os.path.join(job_dir, self.output_confidence_pattern.format(job_name.lower()))
        summary_path = os.path.join(job_dir, self.output_summary_pattern.format(job_name.lower()))
        
        # Check if files exist and are not empty
        def is_valid_file(path):
            return os.path.exists(path) and os.path.getsize(path) > 0
            
        return all(is_valid_file(p) for p in [structure_path, confidence_path, summary_path])

    def run_prediction(self, input_data: Dict, placeholder=None, device="cuda:0") -> str:
        import json

        def _build_command(json_path: str, gpu_id: str) -> str:
            return (
                f"cd {self.config['execution']['base_dir']} && "
                f"CUDA_VISIBLE_DEVICES={gpu_id} conda run -n {self.required_env} "
                f"python run_alphafold.py "
                f"--json_path={json_path} "
                f"--model_dir={self.model_weights_path} "
                f"--output_dir={self.OUTPUT_DIR} "
                f"--db_dir={self.database_path} "
                f"--jackhmmer_binary_path={self.config['binaries']['jackhmmer']} "
                f"--hmmbuild_binary_path={self.config['binaries']['hmmbuild']} "
                f"--hmmsearch_binary_path={self.config['binaries']['hmmsearch']} "
                f"--nhmmer_binary_path={self.config['binaries']['nhmmer']} "
            )

        # Unified stash for per-job AF3 input JSONs
        af3_stash_dir = os.path.join(self.INPUT_DIR, "af3_input_jsons")
        os.makedirs(af3_stash_dir, exist_ok=True)

        if input_data.get("batch_mode", False) and ":" in device:
            gpu_ids = device.split(":")[1].split(",")
            num_gpus = len(gpu_ids)
            
            if num_gpus >= 2:
                # Multi-GPU parallel processing
                from concurrent.futures import ThreadPoolExecutor
                # import json
                
                def process_job(args):
                    input_path, gpu_id = args
                    # Read jobs info to check if predictions exist
                    with open(input_path) as f:
                        jobs_data = json.load(f)
                    # Normalize to list
                    if not isinstance(jobs_data, list):
                        jobs = [jobs_data]
                    else:
                        jobs = jobs_data

                    # Partition jobs: AF3 vs others
                    af3_jobs = []
                    other_jobs = []
                    for job in jobs:
                        if job.get("dialect") == "alphafold3":
                            af3_jobs.append(job)
                        else:
                            other_jobs.append(job)

                    outputs = []

                    # Handle alphafold3: per-job single-dict JSON inputs
                    if af3_jobs:
                        to_run = []
                        for job in af3_jobs:
                            if self.check_prediction_exists(job["name"]):
                                print(f"jobs name {job['name']} already exist, skipping...")
                                continue
                            job = self._ensure_model_seeds(job)
                            # Save per-job JSON into unified stash dir
                            safe_name = job["name"].lower().replace(' ', '_')
                            job_json = os.path.join(af3_stash_dir, f"{safe_name}.input.json")
                            with open(job_json, "w") as jf:
                                json.dump(job, jf, indent=2)
                            to_run.append(job_json)

                        for job_json in to_run:
                            cmd = _build_command(job_json, gpu_id)
                            out = run_alphafold(cmd, placeholder)
                            outputs.append(out)
                            # After run, copy the input json into its result folder
                            try:
                                with open(job_json) as jf:
                                    job_obj = json.load(jf)
                                job_dir = os.path.join(self.OUTPUT_DIR, job_obj["name"].lower())
                                os.makedirs(job_dir, exist_ok=True)
                                shutil.copy2(job_json, os.path.join(job_dir, "input.json"))
                            except Exception as e:
                                print(f"Warning: failed to copy input json for {job_json}: {e}")

                    # Handle other dialects in batch (list) mode as before
                    if other_jobs:
                        # Filter out completed
                        jobs_to_process = [j for j in other_jobs if not self.check_prediction_exists(j["name"])]
                        if jobs_to_process:
                            # Overwrite file with the remaining non-AF3 jobs
                            with open(input_path, 'w') as f:
                                json.dump(jobs_to_process, f, indent=2)
                            cmd = _build_command(input_path, gpu_id)
                            outputs.append(run_alphafold(cmd, placeholder))
                        else:
                            print(f"All non-AF3 predictions in {input_path} already exist, skipping...")

                    return "\n".join([o for o in outputs if o])

                # Create job-GPU pairs
                job_gpu_pairs = list(zip(input_data["input_paths"], gpu_ids))

                # Run jobs in parallel
                with ThreadPoolExecutor(max_workers=num_gpus) as executor:
                    results = list(executor.map(process_job, job_gpu_pairs))
                
                return "\n".join(filter(None, results))
            
        # Single GPU processing
        input_path = input_data.get("input_path") or input_data["input_paths"][0]
        
        # Check if predictions exist
        with open(input_path) as f:
            jobs_data = json.load(f)

        # Normalize to list if batch_mode
        if not isinstance(jobs_data, list) and input_data.get("batch_mode", True):
            jobs = [jobs_data]
        elif isinstance(jobs_data, list):
            jobs = jobs_data
        else:
            jobs = [jobs_data]

        # Split by dialect
        af3_jobs = [j for j in jobs if j.get("dialect") == "alphafold3"]
        other_jobs = [j for j in jobs if j.get("dialect") != "alphafold3"]

        gpu_id = device.split(":")[1] if ":" in device else device

        outputs = []

        # Alphafold3: per-job JSON dicts
        if af3_jobs:
            to_run = []
            for job in af3_jobs:
                if self.check_prediction_exists(job["name"]):
                    print(f"jobs name {job['name']} already exist, skipping...")
                    continue
                job = self._ensure_model_seeds(job)
                # Save per-job JSON into unified stash dir
                safe_name = job["name"].lower().replace(' ', '_')
                job_json = os.path.join(af3_stash_dir, f"{safe_name}.input.json")
                with open(job_json, "w") as jf:
                    json.dump(job, jf, indent=2)
                to_run.append(job_json)

            for job_json in to_run:
                cmd = _build_command(job_json, gpu_id)
                out = run_alphafold(cmd, placeholder)
                outputs.append(out)
                # After run, copy the input json into its result folder
                try:
                    with open(job_json) as jf:
                        job_obj = json.load(jf)
                    job_dir = os.path.join(self.OUTPUT_DIR, job_obj["name"].lower())
                    os.makedirs(job_dir, exist_ok=True)
                    shutil.copy2(job_json, os.path.join(job_dir, "input.json"))
                except Exception as e:
                    print(f"Warning: failed to copy input json for {job_json}: {e}")

        # Other dialects: retain existing list-based batching behavior
        if other_jobs:
            jobs_to_process = [j for j in other_jobs if not self.check_prediction_exists(j["name"])]
            if not jobs_to_process:
                print("All jobs already exist, nothing to process")
            else:
                # Overwrite the input file with remaining non-AF3 jobs and run once
                with open(input_path, 'w') as f:
                    json.dump(jobs_to_process, f, indent=2)
                cmd = _build_command(input_path, gpu_id)
                outputs.append(run_alphafold(cmd, placeholder))

        return "\n".join([o for o in outputs if o])
        
    def process_output(self, output_dir: str, config: Dict) -> Dict:
        # 查找最新的输出文件
        structure_path = os.path.join(output_dir, config['job_name'].lower(), self.output_structure_pattern.format(config['job_name'].lower()))
        confidence_path = os.path.join(output_dir, config['job_name'].lower(), self.output_confidence_pattern.format(config['job_name'].lower()))
        summary_path = os.path.join(output_dir, config['job_name'].lower(), self.output_summary_pattern.format(config['job_name'].lower()))
        return {
            "format": "alphafold",
            "structure_files": [structure_path],
            "confidence_files": [confidence_path],
            "summary_files": [summary_path],
            'job_download_basename': config['job_name'].lower().replace(' ', '_')
        } 
    

if __name__ == "__main__":
    model = AlphaFoldModel()
    sequences = [
        {"name": "test1", "protein": "MALWMRLLPLLALLALWGPDPAAA", "rna": "GCTCGCGATGCTAGAGGGCTCTGC"},
        {"name": "test2", "protein": "MALWMRLLPLLALLALWGPDPAAA", "dna": "GCTCGCGATGCTAGAGGGCTCTGC", "rna": "GCAGAGCCCUCCAGCAUCGCGAGC"}
    ]
    batch_mode = False
    if batch_mode:
        input_data = model.batch_prepare_sequences(sequences, "234321")
    else:
        input_data = model.single_prepare_sequences(sequences, "234321")
    

    
    gpu_ids = "0"
    gpu_num = len(gpu_ids.split(","))
    input_data = model.prepare_input(input_data, batch_mode=batch_mode, num_gpus=gpu_num, name_prefix="aha_batch")
    model.run_prediction(input_data, device=f"cuda:{gpu_ids}")
