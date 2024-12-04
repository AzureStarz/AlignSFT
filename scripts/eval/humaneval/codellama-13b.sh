#!/bin/env bash
humaneval_data_dir=/home/export/base/ycsc_chenkh/hitici_02/online1/LangBridge/data/humaneval

ckpt_path=${1:-"/home/export/base/ycsc_chenkh/hitici_02/online1/data/pretrained-models/CodeLlama-13b-Python-hf"}
output_path=${2:-"/home/export/base/ycsc_chenkh/hitici_02/online1/AlignSFT/eval_outputs/humaneval/codellama-13b/results.json"}

# --tasks humaneval#en,humaneval#en_anon,humaneval#sw,humaneval#sw_anon,humaneval#bn,humaneval#bn_anon,humaneval#mr,humaneval#mr_anon,humaneval#ne,humaneval#ne_anon,humaneval#pa,humaneval#pa_anon,humaneval#te,humaneval#te_anon,humaneval#ur,humaneval#ur_anon \
# --load_data_path ${humaneval_data_dir}/humaneval_english.json,${humaneval_data_dir}/humaneval_english_anon.json,${humaneval_data_dir}/humaneval_swahili.json,${humaneval_data_dir}/humaneval_swahili_anon.json,${humaneval_data_dir}/humaneval_bengali.json,${humaneval_data_dir}/humaneval_bengali_anon.json,${humaneval_data_dir}/humaneval_marathi.json,${humaneval_data_dir}/humaneval_marathi_anon.json,${humaneval_data_dir}/humaneval_nepali.json,${humaneval_data_dir}/humaneval_nepali_anon.json,${humaneval_data_dir}/humaneval_punjabi.json,${humaneval_data_dir}/humaneval_punjabi_anon.json,${humaneval_data_dir}/humaneval_telugu.json,${humaneval_data_dir}/humaneval_telugu_anon.json,${humaneval_data_dir}/humaneval_urdu.json,${humaneval_data_dir}/humaneval_urdu_anon.json \


accelerate launch python_scripts/eval_code.py \
  --model ${ckpt_path} \
  --trust_remote_code \
  --tasks humaneval#en \
  --load_data_path ${humaneval_data_dir}/humaneval_english.json \
  --do_sample False \
  --n_samples 1 \
  --batch_size 1 \
  --allow_code_execution \
  --precision bf16 \
  --save_generations \
  --metric_output_path ${output_path} \