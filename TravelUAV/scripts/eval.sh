#!/bin/bash
# change the dataset_path to your own path

root_dir=/home/renpengzhen/zhangpeilun/travel_vggt/TravelUAV # TravelUAV directory
model_dir=$root_dir/Model/LLaMA-UAV


CUDA_VISIBLE_DEVICES=0 python -u $root_dir/src/vlnce_src/eval.py \
    --run_type eval \
    --name TravelLLM \
    --gpu_id 0 \
    --simulator_tool_port 30000 \
    --DDP_MASTER_PORT 80005 \
    --batchSize 1 \
    --always_help True \
    --use_gt True \
    --maxWaypoints 200 \
    --dataset_path /data2/uav \
    --eval_save_path /home/renpengzhen/zhangpeilun/travel_vggt/TravelUAV/eval_test \
    --model_path /data2/uav/uavcheckpoints/evo-qwen-vid-7b-pretrain-224-uav-full-data-lora32 \
    --model_base $model_dir/model_zoo/vicuna-7b-v1.5 \
    --vision_tower $model_dir/model_zoo/LAVIS/eva_vit_g.pth \
    --image_processor $model_dir/llamavid/processor/clip-patch14-224 \
    --traj_model_path /data2/uav/origin_traj_checkpoints \
    --eval_json_path /home/renpengzhen/zhangpeilun/TravelUAV/data/uav_dataset/Carla_Town06_valset.json \
    --map_spawn_area_json_path $root_dir/data/meta/map_spawnarea_info.json \
    --object_name_json_path $root_dir/data/meta/object_description.json \
    --groundingdino_config $root_dir/src/model_wrapper/utils/GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py \
    --groundingdino_model_path $root_dir/src/model_wrapper/utils/GroundingDINO/groundingdino_swint_ogc.pth