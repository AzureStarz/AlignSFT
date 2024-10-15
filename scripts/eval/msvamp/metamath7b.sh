#!/bin/env bash
ckpt_path=${1:-"/home/export/base/ycsc_chenkh/hitici_02/online1/LangBridge/checkpoints/debug"}
output_path=${2:-"/home/export/base/ycsc_chenkh/hitici_02/online1/LangBridge/eval_outputs/mgsm/debug"}

python python_scripts/eval_baseline.py \
  --model hf-causal-experimental \
  --model_args dtype="bfloat16",pretrained=${ckpt_path} \
  --tasks msvamp_en,msvamp_es,msvamp_fr,msvamp_de,msvamp_ru,msvamp_zh,msvamp_ja,msvamp_th,msvamp_sw,msvamp_bn \
  --instruction_template metamath \
  --batch_size 1 \
  --output_path ${output_path}  \
  --device cuda:0 \
  --no_cache