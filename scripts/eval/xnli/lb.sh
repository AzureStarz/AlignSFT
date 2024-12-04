#!/bin/env bash
ckpt_path=${1:-"/home/export/base/ycsc_chenkh/hitici_02/online1/LangBridge/checkpoints/debug"}
output_path=${2:-"/home/export/base/ycsc_chenkh/hitici_02/online1/LangBridge/eval_outputs/xnli/debug"}
enc_tokenizer_path=${3:-"/home/export/base/ycsc_chenkh/hitici_02/online1/LangBridge/pretrained-models/mt5-xl-lm-adapt"}

batch_size=128

python python_scripts/eval_langbridge.py \
  --checkpoint_path ${ckpt_path} \
  --enc_tokenizer ${enc_tokenizer_path} \
  --tasks xnli_ar,xnli_bg,xnli_de,xnli_el,xnli_en,xnli_es,xnli_fr,xnli_hi,xnli_ru,xnli_sw,xnli_th,xnli_tr,xnli_ur,xnli_vi,xnli_zh \
  --instruction_template base \
  --batch_size ${batch_size} \
  --output_path ${output_path} \
  --device cuda:0 \
  --no_cache