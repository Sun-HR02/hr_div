#!bin/bash

set -x

# PAPER_TABLE=ok_vqa_val2014,pope,scienceqa_img,seedbench,mmmu_val
PAPER_TABLE=nocaps_val

LOG_DIR=./logs_final
RUN_NAME=divprune_llava_1.5_7b

#To run other models use: liuhaotian/llava-v1.5-13b, liuhaotian/llava-v1.6-vicuna-7b liuhaotian/llava-v1.5-7b
CUDA_VISIBLE_DEVICES=0,1,2,3 BASELINE=OURS LAYER_INDEX=0 SUBSET_RATIO=0.098 python3 -m accelerate.commands.launch \
    --num_processes=4 \
    -m lmms_eval \
    --model llava \
    --model_args pretrained="liuhaotian/llava-v1.5-13b" \
    --tasks $PAPER_TABLE \
    --batch_size 1 \
    --log_samples \
    --log_samples_suffix $RUN_NAME \
    --output_path $LOG_DIR/$RUN_NAME