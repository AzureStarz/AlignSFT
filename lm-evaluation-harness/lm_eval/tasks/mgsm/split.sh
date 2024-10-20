#!/bin/bash

# 输入文件
input_file="/home/export/base/ycsc_chenkh/hitici_02/online1/AlignSFT/lm-evaluation-harness/lm_eval/tasks/mgsm/metamath_cot/mgsm.yaml"

# 初始化变量
current_task=""
current_class=""

# 读取输入文件行
while IFS= read -r line; do
  # 检查是否为任务行
  if [[ $line == "  - task:"* ]]; then
    # 如果当前任务存在，则保存到文件
    if [[ -n $current_task ]]; then
      file_name="/home/export/base/ycsc_chenkh/hitici_02/online1/AlignSFT/lm-evaluation-harness/lm_eval/tasks/mgsm/metamath_cot/${current_task}.yaml"
      echo "task: $current_task" > "$file_name"
      echo "class: $current_class" >> "$file_name"
    fi
    # 提取当前任务名称
    current_task=$(echo "$line" | sed 's/.*task: //')
  elif [[ $line == "    class:"* ]]; then
    # 提取当前类名称
    current_class=$(echo "$line" | sed 's/.*class: //')
  fi
done < "$input_file"

# 写入最后一个任务
if [[ -n $current_task ]]; then
  file_name="/home/export/base/ycsc_chenkh/hitici_02/online1/AlignSFT/lm-evaluation-harness/lm_eval/tasks/mgsm/metamath_cot/${current_task}.yaml"
  echo "task: $current_task" > "$file_name"
  echo "class: $current_class" >> "$file_name"
fi