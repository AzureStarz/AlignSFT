#!/bin/env bash
ckpt_path=${1:-"/home/export/base/ycsc_chenkh/hitici_02/online1/LangBridge/checkpoints/debug"}
output_path=${2:-"/home/export/base/ycsc_chenkh/hitici_02/online1/LangBridge/eval_outputs/xnli/debug"}
enc_tokenizer_path=${3:-"/home/export/base/ycsc_chenkh/hitici_02/online1/LangBridge/pretrained-models/mt5-xl-lm-adapt"}

batch_size=1


# --tasks csqa_ar,csqa_de,csqa_en,csqa_es,csqa_fr,csqa_hi,csqa_it,csqa_ja,csqa_nl,csqa_pl,csqa_pt,csqa_ru,csqa_sw,csqa_ur,csqa_vi,csqa_zh \

python python_scripts/eval_langbridge.py \
  --checkpoint_path ${ckpt_path} \
  --enc_tokenizer ${enc_tokenizer_path} \
  --tasks csqa_en \
  --instruction_template base \
  --batch_size ${batch_size} \
  --output_path ${output_path} \
  --device cuda:0 \
  --no_cache