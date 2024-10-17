#!/bin/env bash
ckpt_path=${1:-"/home/export/base/ycsc_chenkh/hitici_02/online1/LangBridge/checkpoints/debug"}
output_path=${2:-"/home/export/base/ycsc_chenkh/hitici_02/online1/LangBridge/eval_outputs/mgsm/debug"}
enc_tokenizer_path=${3:-"/home/export/base/ycsc_chenkh/hitici_02/online1/LangBridge/pretrained-models/mt5-xl-lm-adapt"}

_LANG="ar bn ca da de en es eu fr gu hi hr hu hy id is it ja kn ml mr nb ne nl pt ro ru sk sr sv sw ta te th uk vi zh"
task=$(echo $_LANG | sed 's/ /,m_mmlu_math_/g')
task="m_mmlu_math_${task}"

python python_scripts/eval_langbridge.py \
  --checkpoint_path ${ckpt_path} \
  --enc_tokenizer ${enc_tokenizer_path} \
  --tasks ${task} \
  --instruction_template metamath \
  --batch_size 1 \
  --output_path ${output_path} \
  --device cuda:0 \
  --no_cache