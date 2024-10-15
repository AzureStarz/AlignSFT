#!/bin/env bash
ckpt_path=${1:-"/home/export/base/ycsc_chenkh/hitici_02/online1/LangBridge/checkpoints/debug"}
output_path=${2:-"/home/export/base/ycsc_chenkh/hitici_02/online1/LangBridge/eval_outputs/mgsm/debug"}

python python_scripts/eval_baseline.py \
  --model hf-causal-experimental \
  --model_args dtype="bfloat16",pretrained=${ckpt_path} \
  --tasks mgsm_en,mgsm_es,mgsm_fr,mgsm_de,mgsm_ru,mgsm_zh,mgsm_ja,mgsm_th,mgsm_sw,mgsm_bn,mgsm_te\
  --instruction_template mathoctopus \
  --batch_size 1 \
  --output_path ${output_path}  \
  --device cuda:0 \
  --no_cache