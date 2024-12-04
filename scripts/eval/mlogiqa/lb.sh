#!/bin/env bash
ckpt_path=${1:-"/home/export/base/ycsc_chenkh/hitici_02/online1/LangBridge/checkpoints/debug"}
output_path=${2:-"/home/export/base/ycsc_chenkh/hitici_02/online1/LangBridge/eval_outputs/logiqa/debug"}
enc_tokenizer_path=${3:-"/home/export/base/ycsc_chenkh/hitici_02/online1/LangBridge/pretrained-models/mt5-xl-lm-adapt"}

batch_size=32

python python_scripts/eval_langbridge.py \
  --checkpoint_path ${ckpt_path} \
  --enc_tokenizer ${enc_tokenizer_path} \
  --tasks mlogiqa_ar,mlogiqa_en,mlogiqa_es,mlogiqa_fr,mlogiqa_ja,mlogiqa_ko,mlogiqa_pt,mlogiqa_th,mlogiqa_vi,mlogiqa_zh \
  --instruction_template base \
  --batch_size ${batch_size} \
  --output_path ${output_path} \
  --device cuda:0 \
  --no_cache