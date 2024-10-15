# export TRANSFORMERS_CACHE=/mnt/sda/dongkeun/huggingface
# export HF_DATASETS_CACHE=/mnt/sda/dongkeun/huggingface_datasets
export CUDA_VISIBLE_DEVICES=0

python python_scripts/eval_langbridge.py \
  --checkpoint_path /home/export/base/ycsc_chenkh/hitici_02/online1/LangBridge/checkpoints/v3/epoch=1-step=1041 \
  --enc_tokenizer /home/export/base/ycsc_chenkh/hitici_02/online1/LangBridge/pretrained-models/mt5-xl-lm-adapt \
  --tasks copa,xcopa_et,xcopa_ht,xcopa_it,xcopa_id,xcopa_qu,xcopa_sw,xcopa_zh,xcopa_ta,xcopa_th,xcopa_tr,xcopa_vi \
  --instruction_template orca \
  --batch_size 128 \
  --output_path eval_outputs/xcopa/debug \
  --device cuda:0 \
  --no_cache \