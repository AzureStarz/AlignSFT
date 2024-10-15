# export TRANSFORMERS_CACHE=/mnt/sda/dongkeun/huggingface
# export HF_DATASETS_CACHE=/mnt/sda/dongkeun/huggingface_datasets
# export CUDA_VISIBLE_DEVICES=0

python python_scripts/eval_baseline.py \
  --model hf-causal-experimental \
  --model_args dtype="bfloat16",pretrained=/home/export/base/ycsc_chenkh/hitici_02/online1/LLaMA-Factory/saves/llama2-7b/full/wmt_en_zh_bactrian-x_en_SFT_baseline \
  --tasks copa,xcopa_et,xcopa_ht,xcopa_it,xcopa_id,xcopa_qu,xcopa_sw,xcopa_zh,xcopa_ta,xcopa_th,xcopa_tr,xcopa_vi \
  --num_fewshot 8\
  --batch_size 1 \
  --output_path eval_outputs/xcopa/bactrian-x_en_SFT_baseline \
  --device cuda:0 \
  --no_cache \