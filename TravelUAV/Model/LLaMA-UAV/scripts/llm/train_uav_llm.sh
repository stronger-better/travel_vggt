#!/bin/bash
set -euo pipefail

# Change these paths for your own server.
export WANDB_MODE=offline
export WANDB_DIR=/mnt/sfs_turbo_new/R10844/zhangpeilun/openuav_vggt/TravelUAV/Model/LLaMA-UAV/wandb
export PATH=/mnt/sfs_turbo/R10840/anaconda3/envs/travel_vggt/bin:$PATH

SITE_PACKAGES="/mnt/sfs_turbo/R10840/anaconda3/envs/travel_vggt/lib/python3.10/site-packages"
export LD_LIBRARY_PATH="${SITE_PACKAGES}/nvidia/cublas/lib:${SITE_PACKAGES}/nvidia/nccl/lib:${SITE_PACKAGES}/nvidia/cudnn/lib:${LD_LIBRARY_PATH:-}"

root_dir=/mnt/sfs_turbo_new/R10844/zhangpeilun/openuav_vggt/TravelUAV
model_dir=$root_dir/Model/LLaMA-UAV

# Optional local VGGT settings.
# 1. If VGGT_MODEL_PATH is set, the script will load local model.pt.
# 2. If FORCE_OFFLINE=1, the script will disable Hugging Face network access.
# Examples:
#   export VGGT_MODEL_PATH=/path/to/VGGT-1B/model.pt
#   export FORCE_OFFLINE=1
VGGT_MODEL_PATH=/mnt/sfs_turbo_new/R10844/zhangpeilun/openuav_vggt/TravelUAV/vggt_checkpoints/model.pt
FORCE_OFFLINE=1
VGGT_ARGS=()

if [[ -n "$VGGT_MODEL_PATH" ]]; then
    if [[ ! -f "$VGGT_MODEL_PATH" ]]; then
        echo "VGGT model file not found: $VGGT_MODEL_PATH" >&2
        echo "Please set VGGT_MODEL_PATH to a valid local VGGT model.pt path." >&2
        exit 1
    fi
    export VGGT_AUTO_DOWNLOAD=0
    VGGT_ARGS+=(--vggt_model_path "$VGGT_MODEL_PATH")
fi

if [[ "$FORCE_OFFLINE" == "1" ]]; then
    export HF_HUB_OFFLINE=1
    export TRANSFORMERS_OFFLINE=1
    export HF_DATASETS_OFFLINE=1
fi

# export CUDA_VISIBLE_DEVICES=5

deepspeed \
    --include localhost:0,1,2,3,4,5,6,7 \
    --master_port 23101 \
    $model_dir/llamavid/train/train_uav/train_uav_notice.py \
    --data_path $root_dir/data/uav_dataset/trainset.json \
    --dataset_path /mnt/sfs_turbo_new/R10844/zhangpeilun/project1/TravelUAV/dataset \
    --output_dir $model_dir/work_dirs/new-evolve-evo-qwen-vid-7b-pretrain-224-uav-full-data-lora32 \
    --deepspeed $model_dir/scripts/zero2.json \
    --ddp_find_unused_parameters True \
    --model_name_or_path /mnt/sfs_turbo_new/R10844/zhangpeilun/openuav_vggt/TravelUAV/Model/LLaMA-UAV/model_zoo/vicuna-7b-v1.5 \
    --version imgsp_uav \
    --is_multimodal True \
    --vision_tower /mnt/sfs_turbo_new/R10844/zhangpeilun/openuav_vggt/TravelUAV/Model/LLaMA-UAV/model_zoo/LAVIS/eva_vit_g.pth \
    --image_processor $model_dir/llamavid/processor/clip-patch14-224 \
    --mm_projector_type mlp2x_gelu \
    --tune_mm_mlp_adapter True \
    --tune_waypoint_predictor True \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --video_fps 1 \
    --bert_type "qformer_pretrain_freeze" \
    --num_query 32 \
    --pretrain_qformer /mnt/sfs_turbo_new/R10844/zhangpeilun/openuav_vggt/TravelUAV/Model/LLaMA-UAV/model_zoo/LAVIS/instruct_blip_vicuna7b_trimmed.pth \
    --compress_type "mean" \
    --bf16 True \
    --num_train_epochs 4 \
    --per_device_train_batch_size 4 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 2 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 1000 \
    --save_total_limit 1 \
    --learning_rate 2e-4 \
    --weight_decay 0.01 \
    --warmup_ratio 0.1 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 2048 \
    --gradient_checkpointing True \
    --dataloader_num_workers 4 \
    --lazy_preprocess True \
    --report_to wandb \
    --lora_enable True \
    "${VGGT_ARGS[@]}"
