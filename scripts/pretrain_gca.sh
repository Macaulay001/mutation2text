#!/bin/bash

# Script to launch pretraining for GCA, Resampler, and Projector

# DeepSpeed launcher
LAUNCHER="deepspeed"

# Main training script
TRAIN_SCRIPT="./scripts/train.py"

# Configuration files
ZERO_STAGE_CONFIG="./configs/zero2.json"
TRAIN_CONFIG="./configs/train_config.yaml" 

NUM_GPUS=1 

# Extract output_dir from TRAIN_CONFIG YAML
TRAIN_CONFIG_OUTPUT_DIR=$(python -c "import yaml; f = open('$TRAIN_CONFIG'); d = yaml.safe_load(f); print(d.get('training_args', {}).get('output_dir', './output/pretrain_gca_fallback')); f.close()")

mkdir -p $TRAIN_CONFIG_OUTPUT_DIR

echo "Starting GCA Pretraining..."
echo "Output directory: $TRAIN_CONFIG_OUTPUT_DIR"


# --num_gpus=$NUM_GPUS


CUDA_VISIBLE_DEVICES=1 $LAUNCHER  --master_port 29500 $TRAIN_SCRIPT \
    --output_dir $TRAIN_CONFIG_OUTPUT_DIR \
    --deepspeed $ZERO_STAGE_CONFIG \
    --model_config_file $TRAIN_CONFIG \

echo "Pretraining finished." 