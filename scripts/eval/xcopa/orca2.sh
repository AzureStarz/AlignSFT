# export TRANSFORMERS_CACHE=/mnt/sda/dongkeun/huggingface
# export HF_DATASETS_CACHE=/mnt/sda/dongkeun/huggingface_datasets
# export CUDA_VISIBLE_DEVICES=1

python python_scripts/eval_baseline.py \
  --model hf-causal-experimental \
  --model_args dtype="bfloat16",pretrained=/home/export/base/ycsc_chenkh/hitici_02/online1/data/pretrained-models/Orca-2-7b \
  --tasks copa,xcopa_et,xcopa_ht,xcopa_it,xcopa_id,xcopa_qu,xcopa_sw,xcopa_zh,xcopa_ta,xcopa_th,xcopa_tr,xcopa_vi \
  --instruction_template orca \
  --batch_size 1 \
  --output_path eval_outputs/xcopa/Orca-2-7b  \
  --device cuda:0 \
  --no_cache