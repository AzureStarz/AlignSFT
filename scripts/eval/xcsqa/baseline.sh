#!/bin/env bash
ckpt_path=${1:-"/home/export/base/ycsc_chenkh/hitici_02/online1/LangBridge/checkpoints/debug"}
template_name=${2:-"base"}
output_path=${3:-"/home/export/base/ycsc_chenkh/hitici_02/online1/LangBridge/eval_outputs/mgsm/debug"}

python python_scripts/eval_baseline.py \
  --model hf-causal-experimental \
  --model_args dtype="bfloat16",pretrained=${ckpt_path} \
  --tasks csqa_ar,csqa_de,csqa_en,csqa_es,csqa_fr,csqa_hi,csqa_it,csqa_ja,csqa_nl,csqa_pl,csqa_pt,csqa_ru,csqa_sw,csqa_ur,csqa_vi,csqa_zh \
  --instruction_template ${template_name} \
  --batch_size 1 \
  --output_path ${output_path}  \
  --device cuda:0 \
  --no_cache