# Mutation2Text: Protein Mutation Impact Understanding

This project aims to train a multimodal model capable of understanding the functional impact of protein mutations and generating relevant textual descriptions. It integrates protein sequence information (wild-type and mutated) into a Large Language Model (LLM) using a specialized protein processing pipeline.

## Architecture Overview

- **Large Language Model (LLM):** `meta-llama/Meta-Llama-3.1-8B-Instruct` is used as the textual backbone.
- **Protein Encoder:** `esm3_sm_open_v1` converts protein sequences into embeddings. This model is loaded using the `fair-esm` SDK.
- **Integration Strategy:**
    - **Gated Cross-Attention (GCA):** Compares wild-type and mutated protein embeddings to learn a "delta" representation.
    - **Perceiver Resampler:** Condenses the GCA output into a fixed number of tokens (`num_media_tokens`).
    - **Multimodal Projector:** An MLP that projects the resampled protein features into the LLM's embedding space.
- **Special Token:** A `<delta_P>` token is used in text sequences to mark the insertion point for protein information.
- **Protein Feature Selection:** The `per_residue_embedding` obtained from the ESM3 SDK is used as the primary feature representation. The `mm_protein_select_layer` configuration is noted in the architecture but effectively superseded by this specific SDK output.

## Training Process

1.  **Pretraining (Adapter Modules):**
    - The GCA, Resampler, and Projector modules are trained while the LLM and Protein Encoder backbones are kept frozen.
    - Goal: Teach the adapter modules to extract, compare, condense, and align protein mutation information for the LLM.
    - Script: `scripts/pretrain_gca.sh`
    - Configuration: `configs/train_config.yaml`

2.  **LoRA Finetuning (LLM Adaptation):**
    - The LLM is finetuned using Low-Rank Adaptation (LoRA) with the pretrained adapter modules (GCA, Resampler, Projector) and the Protein Encoder kept frozen.
    - Goal: Efficiently adapt the LLM to generate textual descriptions of protein mutation impacts.
    - Script: `scripts/finetune_lora.sh`
    - Configuration: `configs/lora_config.yaml`

## Directory Structure

```
mutation2text/
├── configs/                  # Configuration files (DeepSpeed, training, LoRA)
│   ├── zero2.json
│   ├── train_config.yaml
│   └── lora_config.yaml
├── data/                     # Data and preprocessing scripts
│   ├── mut_text_data.json    # Example raw data (JSONL format)
│   ├── preprocess.py         # Script for data preprocessing
│   └── finetune_split.py     # Script for creating train/val splits
├── llava/                    # Core model and utility code
│   ├── model/
│   │   ├── __init__.py
│   │   ├── llava_arch.py         # Core architecture (Llama + GCA/Resampler/Projector)
│   │   ├── esm_protein_encoder.py # ESM protein encoder wrapper
│   │   ├── llama_adapter.py      # Placeholder for Llama specific adaptations
│   │   └── lora_adapter.py       # LoRA model creation utilities
│   └── utils/
│       ├── __init__.py
│       ├── data_utils.py         # Dataset, collator, tokenization
│       ├── model_utils.py        # Model loading, checkpointing, freezing
│       ├── training_utils.py     # Optimizer, scheduler helpers
│       └── lora_utils.py         # Other LoRA utilities (placeholder)
├── scripts/                  # Execution scripts
│   ├── pretrain_gca.sh       # Launch pretraining
│   ├── finetune_lora.sh      # Launch LoRA finetuning
│   ├── train.py              # Main pretraining Python script
│   └── finetune.py           # Main LoRA finetuning Python script
├── requirements.txt          # Python dependencies
├── README.md                 # This file
└── setup.py                  # Project setup (placeholder)
```

## Setup

1.  **Clone the repository (if applicable):**
    ```bash
    # git clone <repository_url>
    # cd mutation2text
    ```

2.  **Create a Python virtual environment (recommended):**
    ```bash
    python -m venv .venv
    source .venv/bin/activate 
    ```

3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
    *Note: Ensure your PyTorch installation matches your CUDA version if using GPUs.*
    *The `requirements.txt` includes `fair-esm` for using the ESM3 SDK.*

4.  **Protein Encoder Model:** The protein encoder is set to `esm3_sm_open_v1`. The `llava/model/esm_protein_encoder.py` script uses the `fair-esm` SDK to load this model. Ensure you have internet access for the initial download if the model is not cached.

5.  **Prepare Data:**
    - Place your raw data (JSONL format with `wild_type_seq`, `mutation_seq`, `text_description` fields) in a file, e.g., `data/mut_text_data_raw.json`.
    - Preprocess the data:
      ```bash
      python data/preprocess.py --input_file data/mut_text_data_raw.json --output_file data/mut_text_data_processed.json
      ```
    - Update `data_path` in `configs/train_config.yaml` to point to `data/mut_text_data_processed.json`.
    - For LoRA finetuning, create train/validation splits:
      ```bash
      python data/finetune_split.py \
          --input_file data/mut_text_data_processed.json \
          --train_output_file data/finetune_train_data.json \
          --val_output_file data/finetune_eval_data.json \
          --val_size 0.1
      ```
    - Update `data_path` and `eval_data_path` in `configs/lora_config.yaml`.
    - **Important for `<delta_P>` token**: Ensure your `text_description` fields in the JSONL data include the `<delta_P>` token at the position where protein information should be contextually inserted. The `data_utils.py` expects this token for embedding replacement logic in `llava_arch.py`.

## Running the Model

### 1. Pretraining (Adapters)

- Review and adjust parameters in `configs/train_config.yaml` and `configs/zero2.json` (especially `output_dir`, `data_path`, batch sizes, learning rate).
- Launch pretraining:
  ```bash
  bash scripts/pretrain_gca.sh
  ```
- Trained adapter weights (GCA, Resampler, Projector, as part of the full model checkpoint) will be saved in the `output_dir` specified in `configs/train_config.yaml` (e.g., `./output/pretrain_gca/checkpoint-XXXX`).

### 2. LoRA Finetuning (LLM)

- Update `pretrained_adapter_path` in `configs/lora_config.yaml` to point to the checkpoint saved during pretraining (e.g., `./output/pretrain_gca/checkpoint-XXXX`).
- Review and adjust other parameters in `configs/lora_config.yaml` (e.g., LoRA ranks, learning rate, data paths).
- Launch LoRA finetuning:
  ```bash
  bash scripts/finetune_lora.sh
  ```
- The finetuned model (including LoRA weights) will be saved in the `output_dir` specified in `configs/lora_config.yaml`.

## Configuration Notes

- **DeepSpeed:** ZeRO Stage 2 is configured in `configs/zero2.json`. Ensure this is compatible with your hardware and distributed setup.
- **Model Names:** The LLM (`meta-llama/Meta-Llama-3.1-8B-Instruct`) and Protein Encoder (`esm3_sm_open_v1`) names are specified in the YAML configuration files and script defaults.
- **Paths:** Data paths and output directories are specified in the YAML configuration files and may need adjustment based on your environment.

## TODO / Assumptions

- The Protein Encoder `esm3_sm_open_v1` is loaded via the `fair-esm` SDK, as implemented in `llava/model/esm_protein_encoder.py`.
- The specific embeddings used are `per_residue_embedding` from the ESM3 SDK. The `mm_protein_select_layer` config is present but its selection mechanism is effectively handled by this SDK output.
- The exact internal dimensions for GCA input/output and Resampler output (`esm_hidden_size`, `gca_output_dim`, `resampler_output_dim` in `llava_arch.py` and `model_utils.py`) are currently placeholders or examples. These should be correctly set based on the actual ESM model used (the `esm_hidden_size` should be automatically deduced by the updated `ESMProteinEncoder`) and desired architecture.
- Error handling for file I/O and model loading can be further improved in the scripts.
- The data preprocessing script (`data/preprocess.py`) is a basic template. More sophisticated cleaning or feature engineering might be needed.
- The placement of `<delta_P>` in input text is crucial and assumed to be handled during data preparation. 