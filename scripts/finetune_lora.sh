#!/bin/bash

# Script to launch LoRA finetuning

# DeepSpeed launcher
LAUNCHER="deepspeed"

# Main finetuning script
FINETUNE_SCRIPT="./scripts/finetune.py"

# Configuration files
ZERO_STAGE_CONFIG="./configs/zero2.json" # Can be the same or a different DeepSpeed config for LoRA
LORA_CONFIG="./configs/lora_config.yaml" # Parsed by finetune.py

NUM_GPUS=2 # Default to 1, adjust as needed

# Extract output_dir from LORA_CONFIG YAML
LORA_CONFIG_OUTPUT_DIR=$(python -c "import yaml; f = open('$LORA_CONFIG'); d = yaml.safe_load(f); print(d.get('training_args', {}).get('output_dir', './output/finetune_lora_fallback')); f.close()")

# Ensure output directory exists
mkdir -p $LORA_CONFIG_OUTPUT_DIR

echo "Starting LoRA Finetuning..."
echo "Output directory: $LORA_CONFIG_OUTPUT_DIR"

# $LAUNCHER --num_gpus=$NUM_GPUS $FINETUNE_SCRIPT \
CUDA_VISIBLE_DEVICES=1 $LAUNCHER  --master_port 29500 $FINETUNE_SCRIPT \
    --output_dir $LORA_CONFIG_OUTPUT_DIR \
    --deepspeed $ZERO_STAGE_CONFIG \
    --model_config_file $LORA_CONFIG \
    # Similar to pretrain, finetune.py should parse LORA_CONFIG for model, data, and training args.
    # The Hugging Face Trainer (if used in finetune.py) will use the deepspeed config path
    # specified within the training_args section of lora_config.yaml.

echo "LoRA Finetuning finished." 