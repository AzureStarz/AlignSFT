model_path=${1:-"/home/export/base/ycsc_chenkh/hitici_02/online1/enhance_alignment/LLaMA-Factory/saves/llama2-7b/full/lb/llama2_metamath_1st-stage_full_sft"}
enc_tokenizer_path=${2:-"/home/export/base/ycsc_chenkh/hitici_02/online1/enhance_alignment/LLaMA-Factory/saves/llama2-7b/full/lb/llama2_metamath_1st-stage_full_sft/encoder_tokenizer"}
# extract the last name of model_path as model_name
model_name=$(echo ${model_path} | rev | cut -d'/' -f1 | rev)

tasks="mgsm_metamath_cot_en"

python python_scripts/evaluate_lb.py \
    --checkpoint_path ${model_path} \
    --enc_tokenizer ${enc_tokenizer_path} \
    --tasks ${tasks} \
    --batch_size 1 \
    --device cuda:0 \
    --log_samples \
    --output_path eval_outputs/${model_name}/mgsm