#!/bin/env bash
ckpt_path=${1:-"/home/export/base/ycsc_chenkh/hitici_02/online1/LangBridge/checkpoints/debug"}
output_path=${2:-"/home/export/base/ycsc_chenkh/hitici_02/online1/LangBridge/eval_outputs/mgsm/debug"}
enc_tokenizer_path=${3:-"/home/export/base/ycsc_chenkh/hitici_02/online1/LangBridge/pretrained-models/mt5-xl-lm-adapt"}

_LANG=" bg da eu ha hy ja mk ne ro sr te zh af bn de fi hi id kn ml nl ru sv th ar ca en fr hr is ko mr pl sk sw uk be cs es gu hu it lb nb pt sl ta vi"
task=$(echo $_LANG | sed 's/ /,m_asdiv_mawps_/g')
task="m_asdiv_mawps_${task}"

python python_scripts/eval_langbridge.py \
  --checkpoint_path ${ckpt_path} \
  --enc_tokenizer ${enc_tokenizer_path} \
  --tasks ${task} \
  --instruction_template metamath \
  --batch_size 1 \
  --output_path ${output_path} \
  --device cuda:0 \
  --no_cache