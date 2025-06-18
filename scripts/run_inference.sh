#!/bin/bash

# Common Configuration
MODEL_NAME="meta-llama/Meta-Llama-3.1-8B-Instruct"
PROTEIN_ENCODER="esm3_sm_open_v1"
DATA_PATH="data/mut_text_data.json"  # Your test data path
BATCH_SIZE=4
MAX_NEW_TOKENS=128
DEVICE="cuda:0"
DTYPE="bf16"  # Use bfloat16 like in training

# Generation parameters
NUM_BEAMS=1
TEMPERATURE=1.0
TOP_P=0.9
# lora parameters
# Note: These parameters are only used for finetuning, not pretraining
LORA_R=16
LORA_ALPHA=32
LORA_DROPOUT=0.05
LORA_TARGET_MODULES="q_proj,v_proj,k_proj,o_proj"  # For finetuning with LoRA

PRETRAINED_ADAPTER_PATH="/data/macaulay/second/scratch/mutation2text/output/finetune_lora/checkpoint-4000"  # Path to the latest pretrained adapter
FINETUNED_ADAPTER_PATH="/data/macaulay/second/scratch/mutation2text/output/finetune_lora"  # Path to the latest finetuned adapter

# Ensure CUDA devices are available
export CUDA_VISIBLE_DEVICES="0"

# Get model type from command line argument
MODEL_TYPE=${1:-pretrain}  # Default to pretrain if not specified

if [ "$MODEL_TYPE" = "pretrain" ]; then
    CHECKPOINT_DIR="output/pretrain_gca/checkpoint-2000"  # Latest pretrained checkpoint
    OUTPUT_PATH="output/generated_responses.jsonl"
    
    echo "Starting inference with pretrained model..."
    python ./scripts/inference.py \
        --model_type pretrain \
        --checkpoint_dir $CHECKPOINT_DIR \
        --model_name $MODEL_NAME \
        --protein_encoder_name $PROTEIN_ENCODER \
        --data_path $DATA_PATH \
        --output_path $OUTPUT_PATH \
        --batch_size $BATCH_SIZE \
        --max_new_tokens $MAX_NEW_TOKENS \
        --device $DEVICE \
        --dtype $DTYPE \
        --num_beams $NUM_BEAMS \
        --temperature $TEMPERATURE \
        --top_p $TOP_P \
        --mode inference \
        --pretrained_adapter_path $FINETUNED_ADAPTER_PATH \


elif [ "$MODEL_TYPE" = "finetune" ]; then
    CHECKPOINT_DIR="/data/macaulay/second/scratch/mutation2text/output/finetune_lora"  # Latest LoRA checkpoint directory
    OUTPUT_PATH="output/generated_responses_finetuned.jsonl"
    
    # LoRA parameters
    LORA_R=16
    LORA_ALPHA=32
    LORA_DROPOUT=0.05
    
    echo "Starting inference with finetuned model..."
    python ./scripts/inference.py \
        --model_type finetune \
        --checkpoint_dir $CHECKPOINT_DIR \
        --model_name $MODEL_NAME \
        --protein_encoder_name $PROTEIN_ENCODER \
        --lora_weights_path $CHECKPOINT_DIR \
        --data_path $DATA_PATH \
        --output_path $OUTPUT_PATH \
        --batch_size $BATCH_SIZE \
        --max_new_tokens $MAX_NEW_TOKENS \
        --device $DEVICE \
        --dtype $DTYPE \
        --num_beams $NUM_BEAMS \
        --temperature $TEMPERATURE \
        --top_p $TOP_P \
        --lora_r $LORA_R \
        --lora_alpha $LORA_ALPHA \
        --lora_dropout $LORA_DROPOUT \
        --mode inference \
        --pretrained_adapter_path $FINETUNED_ADAPTER_PATH \

else
    echo "Invalid model type. Use 'pretrain' or 'finetune'"
    exit 1
fi

echo "Inference completed. Results saved to $OUTPUT_PATH"
