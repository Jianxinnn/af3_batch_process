input_path=/dataStor/home/jxtang/tmp/structure_input/fold_input.json
output_dir=/dataStor/home/jxtang/tmp/structure_output
device=5
required_env=alphafold3_venv_a6000
model_weights_path=/dataStor/share/weights/alphafold3
database_path=/dataStor/share/data/af3_deepmind_db


cd /dataStor/home/jxtang/project/protein/structure/alphafold3 && \
CUDA_VISIBLE_DEVICES=${device} conda run -n ${required_env} \
python run_alphafold.py \
--json_path=${input_path} \
--model_dir=${model_weights_path} \
--output_dir=${output_dir} \
--db_dir=${database_path} \
--jackhmmer_binary_path=/dataStor/home/jxtang/software/hmmer/bin/jackhmmer \
--hmmbuild_binary_path=/dataStor/home/jxtang/software/hmmer/bin/hmmbuild \
--hmmsearch_binary_path=/dataStor/home/jxtang/software/hmmer/bin/hmmsearch \
--nhmmer_binary_path=/dataStor/home/jxtang/software/hmmer/bin/nhmmer \
