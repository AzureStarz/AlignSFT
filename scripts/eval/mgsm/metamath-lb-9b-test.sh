#!/bin/env bash
ckpt_path=${1:-"/home/export/base/ycsc_chenkh/hitici_02/online1/LangBridge/checkpoints/debug"}
output_path=${2:-"/home/export/base/ycsc_chenkh/hitici_02/online1/LangBridge/eval_outputs/mgsm/debug"}
enc_tokenizer_path=${3:-"/home/export/base/ycsc_chenkh/hitici_02/online1/LangBridge/pretrained-models/mt5-xl-lm-adapt"}

python python_scripts/eval_langbridge.py \
  --checkpoint_path ${ckpt_path} \
  --enc_tokenizer ${enc_tokenizer_path} \
  --tasks mgsm_en \
  --instruction_template metamath \
  --batch_size 1 \
  --output_path ${output_path} \
  --device cuda:0 \
  --no_cache