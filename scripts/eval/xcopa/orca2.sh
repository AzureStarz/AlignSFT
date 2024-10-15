#!/bin/env bash
ckpt_path=${1:-"/home/export/base/ycsc_chenkh/hitici_02/online1/LangBridge/checkpoints/debug"}
output_path=${2:-"/home/export/base/ycsc_chenkh/hitici_02/online1/LangBridge/eval_outputs/mgsm/debug"}

batch_size=128

python python_scripts/eval_baseline.py \
  --model hf-causal-experimental \
  --model_args dtype="bfloat16",pretrained=${ckpt_path} \
  --tasks copa,xcopa_et,xcopa_ht,xcopa_it,xcopa_id,xcopa_qu,xcopa_sw,xcopa_zh,xcopa_ta,xcopa_th,xcopa_tr,xcopa_vi \
  --instruction_template orca \
  --batch_size ${batch_size} \
  --output_path ${output_path}  \
  --device cuda:0 \
  --no_cache