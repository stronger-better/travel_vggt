#!/bin/bash
set -euo pipefail

# change the root_dir and dataset_path to your own path
export WANDB_MODE=offline
export WANDB_DIR=/mnt/sfs_turbo_new/R10844/zhangpeilun/openuav_vggt/TravelUAV/Model/LLaMA-UAV/wandb
export PATH=/mnt/sfs_turbo/R10840/anaconda3/envs/travel_vggt/bin:$PATH

# Resolve the current conda environment site-packages path.
SITE_PACKAGES="/mnt/sfs_turbo/R10840/anaconda3/envs/travel_vggt/lib/python3.10/site-packages"

# Prefer the PyTorch-bundled NVIDIA libraries.
export LD_LIBRARY_PATH="${SITE_PACKAGES}/nvidia/cublas/lib:${SITE_PACKAGES}/nvidia/nccl/lib:${SITE_PACKAGES}/nvidia/cudnn/lib:$LD_LIBRARY_PATH"
root_dir=/mnt/sfs_turbo_new/R10844/zhangpeilun/openuav_vggt/TravelUAV # TravelUAV directory
model_dir=$root_dir/Model/LLaMA-UAV

# Training launch values used for subsequent GeoThinker runs.
VGGT_MODEL_PATH=/mnt/sfs_turbo_new/R10844/zhangpeilun/openuav_vggt/TravelUAV/vggt_checkpoints
DEEPSPEED_INCLUDE=localhost:0,1,2,3,4,5,6,7
MASTER_PORT=23101
OUTPUT_DIR=$model_dir/work_dirs/geothinker-paperlike-vicuna7b

# GeoThinker formal training defaults baked into the launch command:
# - 8 GPUs
# - global batch size 64
# - learning rate 1e-5
# - warmup ratio 0.03
# - 2 epochs
# - cross-attention + importance gating
# - 2 SGF injection points in the later half of the 32-layer Vicuna stack
#
# The public paper text/figure indicates SGF is inserted into selected multiple
# VLM layers, but does not expose exact Vicuna-equivalent indices. We therefore
# use a paper-like two-layer setting at layers 16 and 24 by default.

# Set visible CUDA devices if needed.
# export CUDA_VISIBLE_DEVICES=5

deepspeed \
    --include "$DEEPSPEED_INCLUDE" \
    --master_port "$MASTER_PORT" \
    $model_dir/llamavid/train/train_uav/train_uav_notice.py \
    --data_path $root_dir/data/uav_dataset/trainset.json \
    --dataset_path /mnt/sfs_turbo_new/R10844/zhangpeilun/project1/TravelUAV/dataset \
    --output_dir "$OUTPUT_DIR" \
    --deepspeed $model_dir/scripts/zero2.json \
    --ddp_find_unused_parameters True \
    --model_name_or_path /mnt/sfs_turbo_new/R10844/zhangpeilun/openuav_vggt/TravelUAV/Model/LLaMA-UAV/model_zoo/vicuna-7b-v1.5 \
    --version imgsp_uav \
    --is_multimodal True \
    --vision_tower /mnt/sfs_turbo_new/R10844/zhangpeilun/openuav_vggt/TravelUAV/Model/LLaMA-UAV/model_zoo/LAVIS/eva_vit_g.pth \
    --image_processor $model_dir/llamavid/processor/clip-patch14-224 \
    --mm_projector_type mlp2x_gelu \
    --tune_mm_mlp_adapter False \
    --tune_geometry_merger True \
    --tune_feature_fusion True \
    --tune_waypoint_predictor True \
    --feature_fusion_method cross_attention \
    --fusion_attention_heads 8 \
    --fusion_num_layers 1 \
    --fusion_dropout 0.1 \
    --importance_gating True \
    --importance_gate_init 0.0 \
    --sgf_injection_layers 16,24 \
    --geometry_merge_size 4 \
    --vggt_model_path "$VGGT_MODEL_PATH" \
    --vggt_auto_download False \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --video_fps 1 \
    --bert_type "qformer_pretrain_freeze" \
    --num_query 32 \
    --pretrain_qformer /mnt/sfs_turbo_new/R10844/zhangpeilun/openuav_vggt/TravelUAV/Model/LLaMA-UAV/model_zoo/LAVIS/instruct_blip_vicuna7b_trimmed.pth \
    --compress_type "mean" \
    --bf16 True \
    --num_train_epochs 2 \
    --per_device_train_batch_size 4 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 2 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 500 \
    --save_total_limit 2 \
    --learning_rate 1e-5 \
    --weight_decay 0.0 \
    --warmup_ratio 0.03 \
    --lr_scheduler_type cosine \
    --logging_steps 10 \
    --tf32 True \
    --model_max_length 2048 \
    --gradient_checkpointing True \
    --dataloader_num_workers 4 \
    --lazy_preprocess True \
    --report_to wandb \
    --lora_enable True \
    --lora_r 64 \
    --lora_alpha 16 \
    --lora_dropout 0.05
