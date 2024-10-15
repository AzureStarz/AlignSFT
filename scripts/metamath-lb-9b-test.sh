# export TRANSFORMERS_CACHE=/mnt/sda/dongkeun/huggingface
# export HF_DATASETS_CACHE=/mnt/sda/dongkeun/huggingface_datasets
# export CUDA_VISIBLE_DEVICES=0

python /home/export/base/ycsc_chenkh/hitici_02/online1/enhance_alignment/python_scripts/eval_langbridge.py \
  --checkpoint_path /home/export/base/ycsc_chenkh/hitici_02/online1/enhance_alignment/LLaMA-Factory/saves/llama2-7b/full/lb/llama2_metamath_1st-stage_full_sft \
  --enc_tokenizer /home/export/base/ycsc_chenkh/hitici_02/online1/enhance_alignment/LLaMA-Factory/saves/llama2-7b/full/lb/llama2_metamath_1st-stage_full_sft/encoder_tokenizer \
  --tasks mgsm_en,mgsm_es,mgsm_fr,mgsm_de,mgsm_ru,mgsm_zh,mgsm_ja,mgsm_th,mgsm_sw,mgsm_bn,mgsm_te\
  --instruction_template mathoctopus \
  --batch_size 1 \
  --output_path eval_outputs/mgsm/llama2_metamath_1st-stage_full_sft \
  --device cuda:0 \
  --no_cache