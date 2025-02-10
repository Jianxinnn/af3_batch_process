from typing import Dict, List
from pathlib import Path
import json
import os
import yaml

from execution import run_alphafold


class AlphaFoldModel:
    def __init__(self):
        self.config = self._load_config()
        self.INPUT_DIR = self.config["paths"]["input_dir"]
        self.OUTPUT_DIR = self.config["paths"]["output_dir"]
        
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
    
    def input_template(self, job_name, model_seeds, sequences, dialect="alphafoldserver") -> Dict:
        return {
            "name": job_name,
            "modelSeeds": model_seeds,
            "sequences": sequences,
            "dialect": dialect,
            "version": 1
        }
    
    def single_prepare_sequences(self, sequences: List[Dict], model_seeds) -> List[List[Dict]]:
        """
        Convert input sequences to AlphaFold3 format.
        Input format: [{'protein':'XX', 'rna':'XX'}, {'protein':'XX', 'dna':'XX', 'rna':'XX'}, ...]
        Output format: [[{"protein": {"sequence": "XX", "id": "A"}}, {"rna": {"sequence": "XX", "id": "B"}}], ...]
        """
        prepared_sequences = []
        model_seeds = [int(seed.strip()) for seed in model_seeds.split(",") if seed.strip().isdigit()]
        dialect = "alphafold3"
        for case in sequences:
            prepared_case = []
            
            for i, (seq_type, seq) in enumerate(case.items()):
                if seq_type == 'name':
                    job_name = seq
                    continue
                # Clean and format sequence
                cleaned_seq = seq.upper().replace(' ', '').replace('\n', '')
                
                # Create entity with ID (A, B, C, etc.)
                if seq_type == "ligand":
                    entity = {
                        "ligand": {
                            "smiles": cleaned_seq,
                            "id": chr(65 + i)  # 65 is ASCII for 'A'
                        }
                    }
                elif seq_type == "ccd":
                    entity = {
                        "ligand": {
                            "ccdCodes": cleaned_seq,
                            "id": chr(65 + i)  # 65 is ASCII for 'A'
                        }
                    }
                else:
                    entity = {
                        seq_type: {
                            "sequence": cleaned_seq,
                            "id": chr(65 + i)  # 65 is ASCII for 'A'
                        }
                    }
                prepared_case.append(entity)

            case_template = self.input_template(job_name, model_seeds, prepared_case, dialect)
            prepared_sequences.append(case_template)
        return prepared_sequences
    
    def batch_prepare_sequences(self, sequences: List[Dict], model_seeds) -> List[List[Dict]]:
        """
        Convert input sequences to AlphaFold3 format.
        Input format: [{'protein':'XX', 'rna':'XX'}, {'protein':'XX', 'dna':'XX', 'rna':'XX'}, ...]
        Output format: [[{"proteinChain": {"sequence": "XX", "id": "A"}}, {"rnaSequence": {"sequence": "XX", "id": "B"}}], ...]
        """
        prepared_sequences = []
        model_seeds = [int(seed.strip()) for seed in model_seeds.split(",") if seed.strip().isdigit()]

        for case in sequences:
            prepared_case = []
            for i, (seq_type, seq) in enumerate(case.items()):
                if seq_type == 'name':
                    job_name = seq
                    continue

                # Clean and format sequence
                cleaned_seq = seq.upper().replace(' ', '').replace('\n', '')
                
                # Create entity with ID (A, B, C, etc.)
                if seq_type == "protein":
                    entity = {
                        "proteinChain": {
                            "sequence": cleaned_seq,
                            "useStructureTemplate": True,
                            "count": 1
                        }
                    }
                elif seq_type == "rna":
                    entity = {
                        "rnaSequence": {
                            "sequence": cleaned_seq,
                            "count": 1
                        }
                    }   
                elif seq_type == "dna":
                    entity = {
                        "dnaSequence": {
                            "sequence": cleaned_seq,
                            "count": 1
                        }
                    }
                elif seq_type == "ligand":
                    entity = {
                        "ligand": {
                            "smiles": cleaned_seq
                        }
                    }
                elif seq_type == "ccd":
                    entity = {
                        "ccdCode": {
                            "ccdCodes": cleaned_seq
                        }
                    }
                else:
                    raise ValueError(f"Unsupported sequence type: {seq_type}")
                prepared_case.append(entity)
            
            case_template = self.input_template(job_name, model_seeds, prepared_case)
            prepared_sequences.append(case_template)
        return prepared_sequences

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
                    
                    # Create a new list for jobs that need processing
                    jobs_to_process = []
                    for job in jobs_data:
                        if not self.check_prediction_exists(job["name"]):
                            jobs_to_process.append(job)
                    
                    if not jobs_to_process:
                        print(f"All predictions in {input_path} already exist, skipping...")
                        return
                        
                    # Write filtered jobs back to file
                    with open(input_path, 'w') as f:
                        json.dump(jobs_to_process, f, indent=2)
                    
                    command = (
                        f"cd {self.config['execution']['base_dir']} && "
                        f"CUDA_VISIBLE_DEVICES={gpu_id} conda run -n {self.required_env} "
                        f"python run_alphafold.py "
                        f"--json_path={input_path} "
                        f"--model_dir={self.model_weights_path} "
                        f"--output_dir={self.OUTPUT_DIR} "
                        f"--db_dir={self.database_path} "
                        f"--jackhmmer_binary_path={self.config['binaries']['jackhmmer']} "
                        f"--hmmbuild_binary_path={self.config['binaries']['hmmbuild']} "
                        f"--hmmsearch_binary_path={self.config['binaries']['hmmsearch']} "
                        f"--nhmmer_binary_path={self.config['binaries']['nhmmer']} "
                    )
                    return run_alphafold(command, placeholder)

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

        if not isinstance(jobs_data, list) and input_data.get("batch_mode", True):
            jobs_data = [jobs_data]
            
        if input_data.get("batch_mode", True):
            # Create a new list to store jobs that need to be processed
            jobs_to_process = []
            for job in jobs_data:
                check = self.check_prediction_exists(job["name"])
                if check is True:
                    print(f"jobs name {job['name']} already exist, skipping...")
                else:
                    jobs_to_process.append(job)
            
            if not jobs_to_process:
                print("All jobs already exist, nothing to process")
                return ""
                
            # Write the filtered jobs back to the input file
            with open(input_path, 'w') as f:
                json.dump(jobs_to_process, f, indent=2)
        else:
            check = self.check_prediction_exists(jobs_data["name"])
            if check is True:
                print(f"jobs name {jobs_data['name']} already exist, skipping...")
                return ""

        command = (
            f"cd {self.config['execution']['base_dir']} && "
            f"CUDA_VISIBLE_DEVICES={device.split(':')[1]} conda run -n {self.required_env} "
            f"python run_alphafold.py "
            f"--json_path={input_path} "
            f"--model_dir={self.model_weights_path} "
            f"--output_dir={self.OUTPUT_DIR} "
            f"--db_dir={self.database_path} "
            f"--jackhmmer_binary_path={self.config['binaries']['jackhmmer']} "
            f"--hmmbuild_binary_path={self.config['binaries']['hmmbuild']} "
            f"--hmmsearch_binary_path={self.config['binaries']['hmmsearch']} "
            f"--nhmmer_binary_path={self.config['binaries']['nhmmer']} "
        )
        
        return run_alphafold(command, placeholder)
        
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