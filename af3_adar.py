from alphafold3_localbase import AlphaFoldModel
from typing import Dict, List
import os
import json
import pandas as pd


class AF3Adar(AlphaFoldModel):

    INPUT_DIR = "/dataStor/home/jxtang/tmp/adar_search/input"
    OUTPUT_DIR = "/dataStor/home/jxtang/tmp/adar_search/output"

    def __init__(self, data_path):
        super().__init__()
        self.data_path = data_path
        self.sequences = self.load_data()

    def load_data(self):
        self.dataset = pd.read_csv(self.data_path)
        sequences = []
        analysis = {
            'total': 0,
            'total_len': 0,
            'total_len_list': []
        }
        for _, row in self.dataset.iterrows():
            analysis['total'] += 1
            analysis['total_len'] += len(row['db_aln'])
            analysis['total_len_list'].append(len(row['db_aln']))
            sequences.append({
                'name': f"{row['Target']}_{row['probs']}",
                'protein': row['db_aln'],
                'rna': 'GCAGAGCCCUCCAGCAUCGCGAGC',
                'dna': 'GCTCGCGATGCTAGAGGGCTCTGC',
            })
        
        print(f'total: {analysis["total"]}, total_len: {analysis["total_len"]/analysis["total"]}, max_len: {max(analysis["total_len_list"])}, min_len: {min(analysis["total_len_list"])}')
        return sequences
    
    def batch_run(self, gpu_ids="2,3,4,5,6,7"):
        input_data = self.batch_prepare_sequences(self.sequences, "19989898")
        num_gpus = len(gpu_ids.split(","))
        input_data = self.prepare_input(input_data, batch_mode=True, num_gpus=num_gpus, name_prefix="ADAR_res_700")
        self.run_prediction(input_data, device=f"cuda:{gpu_ids}")


if __name__ == "__main__":
    af3_adar = AF3Adar("./foldseek_res.csv")
    af3_adar.batch_run(gpu_ids="3,4,5,6,7")


