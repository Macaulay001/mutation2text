#!/bin/bash

# Script to launch pretraining for GCA, Resampler, and Projector

# DeepSpeed launcher
LAUNCHER="deepspeed"

# Main training script
TRAIN_SCRIPT="./scripts/train.py"

# Configuration files
ZERO_STAGE_CONFIG="./configs/zero2.json"
TRAIN_CONFIG="./configs/train_config.yaml" # This will be parsed by train.py

# Training arguments from train_config.yaml are typically parsed by the script itself.
# However, some might be overridden or set here if needed.
# For example, to specify number of GPUs for DeepSpeed if not in zero_config.
NUM_GPUS=2 # Default to 1, adjust as needed or determine from environment

# Extract output_dir from TRAIN_CONFIG YAML
# This requires python and pyyaml to be available in the environment a priori.
# A more robust solution might involve a dedicated config parsing utility or tool like yq.
TRAIN_CONFIG_OUTPUT_DIR=$(python -c "import yaml; f = open('$TRAIN_CONFIG'); d = yaml.safe_load(f); print(d.get('training_args', {}).get('output_dir', './output/pretrain_gca_fallback')); f.close()")

# Ensure output directory exists (using the extracted or fallback value)
mkdir -p $TRAIN_CONFIG_OUTPUT_DIR

echo "Starting GCA Pretraining..."
echo "Output directory: $TRAIN_CONFIG_OUTPUT_DIR"

$LAUNCHER --num_gpus=$NUM_GPUS $TRAIN_SCRIPT \
    --output_dir $TRAIN_CONFIG_OUTPUT_DIR \
    --deepspeed $ZERO_STAGE_CONFIG \
    --model_config_file $TRAIN_CONFIG \
    # Additional arguments from train_config.yaml will be loaded by train.py
    # or can be passed directly, e.g.:
    # --output_dir $OUTPUT_DIR \
    # --num_train_epochs 3 \
    # --per_device_train_batch_size 8 \
    # --learning_rate 2e-4 \
    # ... etc.

# The train.py script should be designed to parse arguments from both
# command line and the yaml config file (e.g., using OmegaConf or similar).
# Hugging Face Trainer often takes these as direct command line args.

# Example if train.py uses Hugging Face Trainer directly with its args:
# (This assumes train.py is set up to accept all relevant HuggingFace TrainingArguments)

# Read values from train_config.yaml and pass them (more robust in python script)
# For a shell script, it's simpler if train.py reads the yaml.
# If train.py *only* accepts CLI args, you'd need to parse yaml here or duplicate args.

# Let's assume train.py is a standard HuggingFace Trainer script that accepts TrainingArguments
# and custom model/data args. The --model_config_file is a custom arg for train.py to load model_args and data_args.

# The `training_args` from `train_config.yaml` will be primarily used by the HuggingFace Trainer
# which is initialized within `train.py`. The `--deepspeed` flag tells Trainer to use this config.

# A more common way with HF Trainer is to pass all args directly:
# However, the setup mentions train_config.yaml and deepspeed json.
# The pretrain_gca.sh is simple and mainly invokes train.py with config paths.

# Minimal version assuming train.py handles parsing train_config.yaml for all HF args:
# deepspeed --num_gpus=$NUM_GPUS train.py --config_path $TRAIN_CONFIG
# where train.py would internally load $TRAIN_CONFIG for model, data, and training args,
# and initialize trainer with Deepspeed from training_args.deepspeed path.

# Given the structure, the provided command seems plausible if train.py expects
# --deepspeed for the json and another arg for the main yaml config.

echo "Pretraining finished." 