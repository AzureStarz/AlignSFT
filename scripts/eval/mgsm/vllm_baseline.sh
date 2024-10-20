# parsing args
model_path=${1:-"/home/export/base/ycsc_chenkh/hitici_02/online1/data/pretrained-models/MetaMath-7B-V1.0"}
template_name=${2:-"metamath"}
infer_framework=${3:-"vllm"}
# additonal args
model_name=$(echo ${model_path} | rev | cut -d'/' -f1 | rev)
num_of_gpu=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)
# task specific args
tasks="mgsm_zeroshot_cot_de,mgsm_zeroshot_cot_es,mgsm_zeroshot_cot_ja,mgsm_zeroshot_cot_sw,mgsm_zeroshot_cot_th,mgsm_zeroshot_cot_bn,mgsm_zeroshot_cot_en,mgsm_zeroshot_cot_fr,mgsm_zeroshot_cot_ru,mgsm_zeroshot_cot_te,mgsm_zeroshot_cot_zh"
# evaluation script
lm_eval --model ${infer_framework} \
    --model_args pretrained=${model_path},tensor_parallel_size=${num_of_gpu},dtype=half,gpu_memory_utilization=0.9,enforce_eager=True \
    --tasks ${tasks} \
    --batch_size 1 \
    --template_name ${template_name} \
    --log_samples \
    --output_path eval_outputs/mgsm/vllm_${model_name}