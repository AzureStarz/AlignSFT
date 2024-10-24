#!/bin/bash
set -ex

project_dir=/home/export/base/ycsc_chenkh/hitici_02/online1/AlignSFT
ckpt_paths=(
    # "${project_dir}/LLaMA-Factory/saves/llama2-7b/full/baseline/hf-MAmmoTH-7B" "mammoth"
    "${project_dir}/LLaMA-Factory/saves/llama2-7b/full/baseline/hf-Mathoctopus-Parallel-7B" "mathoctopus"
    "${project_dir}/LLaMA-Factory/saves/llama2-7b/full/baseline/hf-QAlign-MetaMathQA-7B" "metamath"
    "${project_dir}/LLaMA-Factory/saves/llama2-7b/full/baseline/hf-WizardMath-7B-V1.0" "metamath"
    "${project_dir}/LLaMA-Factory/saves/llama2-7b/full/baseline/hf-MathOctopus-MAPO-DPO-7B" "mathoctopus"
    "${project_dir}/LLaMA-Factory/saves/llama2-7b/full/baseline/hf-MetaMathOctopus-MAPO-DPO-7B" "mathoctopus"
    "${project_dir}/LLaMA-Factory/saves/llama2-7b/full/baseline/mCoT" "mcot"
    "/home/export/base/ycsc_chenkh/hitici_02/online1/data/pretrained-models/MetaMath-7B-V1.0" "metamath"
    # 你可以继续添加更多路径
)

# 每个子数组的长度（每个路径和名称对）
subarray_length=2

num_of_gpu=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)

for ((i = 0; i < ${#ckpt_paths[@]}; i += subarray_length)); do
    ckpt_path="${ckpt_paths[i]}"
    template_name="${ckpt_paths[i + 1]}"
    model_name=$(echo ${ckpt_path} | rev | cut -d'/' -f1 | rev)
    echo "Evaluating model: ${model_name}"
    # python infer.py \
    #     --model ${ckpt_path} \
    #     --template_name ${template_name} \
    #     --task mgsm \
    #     --inp_path ./eval_data \
    #     --out_path ./eval_outputs \
    #     --num_gpus ${num_of_gpu}

    # python infer.py \
    #     --model ${ckpt_path} \
    #     --template_name ${template_name} \
    #     --task msvamp \
    #     --inp_path ./eval_data \
    #     --out_path ./eval_outputs \
    #     --num_gpus ${num_of_gpu}
    
    python infer.py \
        --model ${ckpt_path} \
        --template_name ${template_name} \
        --task m_asdiv_mawps \
        --inp_path ./eval_data \
        --out_path ./eval_outputs \
        --num_gpus ${num_of_gpu}
done