#!/bin/bash

# Activate conda environment before running training
echo "Activating conda environment..."
conda activate /data/macaulay/envs/protein2text_env

# Check if activation was successful
if [ $? -ne 0 ]; then
    echo "Error: Failed to activate conda environment /data/macaulay/envs/protein2text_env"
    exit 1
fi

echo "Environment activated successfully!"
echo "Python path: $(which python)"
echo "Ninja path: $(which ninja)"

# Change to project directory
cd /data/macaulay/second/scratch/mutation2text

echo "Starting LoRA Finetuning..."
deepspeed --num_gpus=2 ./scripts/finetune.py \
    --output_dir ./output/finetune_lora \
    --deepspeed ./configs/zero2.json \
    --model_config_file ./configs/lora_config.yaml

echo "LoRA Finetuning finished."
