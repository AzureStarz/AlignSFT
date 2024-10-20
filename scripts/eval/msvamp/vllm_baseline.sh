# parsing args
model_path=${1:-"/home/export/base/ycsc_chenkh/hitici_02/online1/data/pretrained-models/MetaMath-7B-V1.0"}
template_name=${2:-"metamath"}
infer_framework=${3:-"vllm"}
# additonal args
model_name=$(echo ${model_path} | rev | cut -d'/' -f1 | rev)
num_of_gpu=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)
# task specific args
tasks="msvamp_zeroshot_cot_bn,msvamp_zeroshot_cot_en,msvamp_zeroshot_cot_fr,msvamp_zeroshot_cot_ru,msvamp_zeroshot_cot_th,msvamp_zeroshot_cot_de,msvamp_zeroshot_cot_es,msvamp_zeroshot_cot_ja,msvamp_zeroshot_cot_sw,msvamp_zeroshot_cot_zh"
# evaluation script
lm_eval --model ${infer_framework} \
    --model_args pretrained=${model_path},tensor_parallel_size=${num_of_gpu},dtype=half,gpu_memory_utilization=0.9,enforce_eager=True \
    --tasks ${tasks} \
    --batch_size 1 \
    --template_name ${template_name} \
    --log_samples \
    --output_path eval_outputs/msvamp/vllm_${model_name}