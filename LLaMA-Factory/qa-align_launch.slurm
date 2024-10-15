#!/bin/bash
#SBATCH --job-name=QAalign    #作业名称
#SBATCH --partition=q_intel_gpu_nvidia_h20   #指定队列
#SBATCH --output=/home/export/base/ycsc_chenkh/hitici_02/online1/enhance_alignment/LLaMA-Factory/logs/run-job%j.out     #正常输出
#SBATCH --error=/home/export/base/ycsc_chenkh/hitici_02/online1/enhance_alignment/LLaMA-Factory/logs/run-job%j.err   #错误输出
#SBATCH --gres=gpu:6       #指定GPU数量
#SBATCH --ntasks-per-node=1
#SBATCH --nodelist=gpu031
#SBATCH --nodes=1

export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=8
export LD_LIBRARY_PATH=/home/export/base/ycsc_chenkh/hitici_02/online1/cuda-11.8/lib64:$LD_LIBRARY_PATH
export WANDB_MODE=offline
export WANDB__SERVICE_WAIT=600
export FORCE_TORCHRUN=1
export MKL_SERVICE_FORCE_INTEL=1

module load amd/gcc_compiler/11.3.0 amd/cuda/12.2
nvcc -V
source activate
conda activate new_llama_factory

project_dir=/home/export/base/ycsc_chenkh/hitici_02/online1/enhance_alignment/LLaMA-Factory
cd ${project_dir}
save_dir=${project_dir}/saves

# 1st stage sft
config_file=${project_dir}/examples/train_full/llama2_QAalign_1st-stage_full_sft_ds2.yaml
exp_run_name=QAalign_1st-stage_full_sft
output_dir=${save_dir}/llama2-7b/full/baseline/${exp_run_name}
sed -i '/^output_dir: /s|output_dir: .*|output_dir: '${output_dir}'|' ${config_file}
# llamafactory-cli train ${config_file}

# 2nd stage sft
config_file=${project_dir}/examples/train_full/llama2_QAalign_2nd-stage_full_sft_ds2.yaml
ckpt_path=${output_dir}
exp_run_name=QAalign_2nd-stage_full_sft
output_dir=${save_dir}/llama2-7b/full/baseline/${exp_run_name}
sed -i '/^output_dir: /s|output_dir: .*|output_dir: '${output_dir}'|' ${config_file}
sed -i '/^model_name_or_path: /s|model_name_or_path: .*|model_name_or_path: '${ckpt_path}'|' ${config_file}
llamafactory-cli train ${config_file}
