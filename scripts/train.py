import os
import sys
import yaml
import json
from dataclasses import dataclass, field, asdict
from typing import Optional, Dict, Any
import time

import torch
from transformers import (
    HfArgumentParser,
    TrainingArguments,
    Trainer,
    set_seed,
)
from custom_trainer import CustomRNGTrainer

# Add project root to sys.path to allow direct import of project modules
# This assumes the script is run from the root of the mutation2text directory or `scripts` subdir
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
print(f"[DEBUG] PROJECT_ROOT set to: {PROJECT_ROOT}")
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from llava.utils.data_utils import MutationTextDataset, custom_collate_fn
from llava.utils.model_utils import load_model_and_tokenizer

@dataclass
class ModelArguments:
    model_name_or_path: str = field(default="meta-llama/Meta-Llama-3.1-8B-Instruct")
    protein_encoder_name_or_path: str = field(default="esm3_sm_open_v1")
    mm_use_resampler_gca: bool = field(default=True)
    mm_gated_cross_attention: bool = field(default=True)
    mm_projector_type: str = field(default="mlp2x_gelu")
    num_media_tokens: int = field(default=128)
    mm_protein_select_layer: int = field(default=-2)
    tune_mm_mlp_adapter: bool = field(default=True) # Crucial for pretraining adapters
    use_mm_proj: bool = field(default=True)
    pretrained_adapter_path: Optional[str] = field(default=None) # For resuming pretraining
    esm_hidden_size: int = field(default=1280, metadata={"help": "Hidden size of the ESM protein encoder"})
    dim_head: int = field(default=64, metadata={"help": "Dimension of the attention head for GCA"})
    ff_mult: int = field(default=4, metadata={"help": "Feedforward multiplier for GCA and projector"})
    perceiver_depth: int = field(default=6, metadata={"help": "Depth of the perceiver for GCA"})
    # mode: str = field(default="train", metadata={"help": "Mode: 'train' or 'inference'"})
    lora_enable: bool = field(default=False, metadata={"help": "Disable LoRA for model pretraining"})
    # Potentially add gca_output_dim, resampler_output_dim if not auto-inferred

@dataclass
class DataArguments:
    data_path: str = field(default="/data/macaulay/second/scratch/mutation2text/data/mut_text_data.json")
    require_both_sequences: bool = field(default=True)
    max_text_len: int = field(default=512) # Max length for text tokenizer

@dataclass
class CustomTrainingArguments(TrainingArguments):
    output_dir: str = field(default="./output/pretrain_gca")
    do_train: bool = field(default=True)
    do_eval: bool = field(default=False)
    num_train_epochs: int = field(default=1)
    per_device_train_batch_size: int = field(default=8)
    per_device_eval_batch_size: int = field(default=4)
    gradient_accumulation_steps: int = field(default=1)
    evaluation_strategy: str = field(default="steps")
    eval_steps: int = field(default=500)
    save_strategy: str = field(default="steps")
    save_steps: int = field(default=1000)
    save_total_limit: int = field(default=3)
    learning_rate: float = field(default=2.0e-4)
    weight_decay: float = field(default=0.)
    warmup_ratio: float = field(default=0.03)
    lr_scheduler_type: str = field(default="cosine")
    logging_steps: int = field(default=10)
    report_to: str = field(default="tensorboard")
    mode: str = field(default="train")
    deepspeed: Optional[str] = field(default="./configs/zero2.json")
    bf16: bool = field(default=True)
    tf32: bool = field(default=True)
    gradient_checkpointing: bool = field(default=True)

    # Other important args
    dataloader_num_workers: int = field(default=4)
    remove_unused_columns: bool = field(default=False) #
    # We can add any custom training args here if needed, or rely on HF defaults
    # The deepspeed config path will be taken from the .json file specified in yaml or CLI
    

@dataclass
class ScriptArguments:
    model_config_file: Optional[str] = field(default=None,
                                             metadata={"help": "Path to a YAML file containing model_args, data_args, and training_args sections."})

def load_config_from_yaml(yaml_path: str) -> Dict[str, Any]:
    """Loads configuration from a YAML file."""
    if not os.path.exists(yaml_path):
        raise FileNotFoundError(f"YAML config file not found: {yaml_path}")
    with open(yaml_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

def main():
    start_total = time.time()

    # Parser for all argument groups including the custom script arguments like model_config_file
    parser = HfArgumentParser((ModelArguments, DataArguments, CustomTrainingArguments, ScriptArguments))
    print(f"[DEBUG] Argument parser initialized with classes: {parser}")

    # Initial parse of CLI arguments. sys.argv[1:] includes args from shell + deepspeed.
    # This parse will populate all dataclasses based on defaults and CLI overrides.
    # We use `look_for_args_file=False` to prevent HfArgumentParser's own args file loading.
    cli_model_args, cli_data_args, cli_training_args, cli_script_args = \
        parser.parse_args_into_dataclasses(args=sys.argv[1:], look_for_args_file=False)

    yaml_config_path = cli_script_args.model_config_file
    print(f"model args: {cli_model_args}, data args: {cli_data_args} , tran_arg: {cli_training_args}, script arg: {cli_script_args}")
    print(f"[CHECK] model_config_file argument: {yaml_config_path}")

    if yaml_config_path:
        if not os.path.exists(yaml_config_path):
            raise FileNotFoundError(f"YAML config file '{yaml_config_path}' not found. Passed via --model_config_file.")
        print(f"Loading configuration from YAML: {yaml_config_path}")
        yaml_configs = load_config_from_yaml(yaml_config_path)
        print(f"[DEBUG] Loaded YAML configuration: {yaml_configs}") 

        print("[DEBUG] Loading configuration from YAML file.")
        # Prepare a list of arguments: YAML values first, then CLI arguments.
        # HfArgumentParser will use the last occurrence for an argument, effectively making CLI override YAML.
        flat_yaml_args = []
        if 'model_args' in yaml_configs:
            for k, v in yaml_configs['model_args'].items(): flat_yaml_args.extend([f'--{k}', str(v)])
        if 'data_args' in yaml_configs:
            for k, v in yaml_configs['data_args'].items(): flat_yaml_args.extend([f'--{k}', str(v)])
        if 'training_args' in yaml_configs:
            for k, v in yaml_configs['training_args'].items(): flat_yaml_args.extend([f'--{k}', str(v)])
        # ScriptArguments like model_config_file are typically not in the YAML itself, but read from CLI first.

        print(flat_yaml_args)
        # Combine YAML-derived args with the original CLI args.
        final_args_to_parse = flat_yaml_args + sys.argv[1:]
        
        # Re-parse to get the final arguments with YAML defaults and CLI overrides.
        model_args, data_args, training_args, script_args = \
            parser.parse_args_into_dataclasses(args=final_args_to_parse, look_for_args_file=False)
    else:
        # No YAML file specified, so the initial CLI parse is the final set of arguments.
        model_args, data_args, training_args, script_args = \
            cli_model_args, cli_data_args, cli_training_args, cli_script_args

    print("[DEBUG] Parsing CLI arguments.")
    # Ensure training_args.deepspeed is correctly set if provided in YAML and used by Trainer
    if hasattr(training_args, 'deepspeed') and isinstance(training_args.deepspeed, str):
        if not os.path.exists(training_args.deepspeed):
            print(f"Warning: DeepSpeed config file specified in training_args ({training_args.deepspeed}) does not exist.")

    print("[DEBUG] Effective Model Arguments:", asdict(model_args))
    print("[DEBUG] Effective Data Arguments:", asdict(data_args))
    print("[DEBUG] Effective Training Arguments:", training_args)

    # Auto-detect the latest checkpoint if resuming is enabled
    if training_args.output_dir and os.path.exists(training_args.output_dir):
        checkpoints = [d for d in os.listdir(training_args.output_dir) if d.startswith('checkpoint-')]
        if checkpoints:
            latest_checkpoint = max(checkpoints, key=lambda x: int(x.split('-')[1]))
            checkpoint_path = os.path.join(training_args.output_dir, latest_checkpoint)
            print(f"[INFO] Auto-detected latest checkpoint: {checkpoint_path}")
            training_args.resume_from_checkpoint = checkpoint_path
    
    # Check DeepSpeed config for evaluation compatibility
    if training_args.deepspeed and (training_args.do_eval or training_args.evaluation_strategy != "no"):
        try:
            deepspeed_config_path = training_args.deepspeed
            if not os.path.exists(deepspeed_config_path):
                 # If path is relative to configs dir, try that, common in HF examples
                potential_path = os.path.join(PROJECT_ROOT, "configs", os.path.basename(deepspeed_config_path))
                if os.path.exists(potential_path):
                    deepspeed_config_path = potential_path
                elif os.path.exists(os.path.join(PROJECT_ROOT, deepspeed_config_path)):
                    deepspeed_config_path = os.path.join(PROJECT_ROOT, deepspeed_config_path)
                # Final check if it's an absolute path or relative to script execution that was found before warning
                if not os.path.exists(deepspeed_config_path) and os.path.exists(training_args.deepspeed):
                    deepspeed_config_path = training_args.deepspeed # revert to original if other attempts failed but original existed
            
            if os.path.exists(deepspeed_config_path):
                with open(deepspeed_config_path, 'r') as f:
                    ds_config = json.load(f)
                if ds_config.get("zero_optimization", {}).get("stage") != 3:
                    print(f"Warning: DeepSpeed is enabled with ZeRO stage {ds_config.get('zero_optimization', {}).get('stage', 'N/A')}, "
                          f"which is not Stage 3. Evaluation with DeepSpeed ZeRO inference requires Stage 3. "
                          f"Disabling evaluation for this run. To enable, use a ZeRO Stage 3 config or set evaluation_strategy to 'no'.")
                    training_args.do_eval = False
                    training_args.evaluation_strategy = "no"
            else:
                print(f"Warning: DeepSpeed config file {training_args.deepspeed} (or resolved {deepspeed_config_path}) not found. Cannot check ZeRO stage for evaluation compatibility.")

        except FileNotFoundError: # This case should be handled by os.path.exists now
            print(f"Warning: DeepSpeed config file {training_args.deepspeed} not found. Cannot check ZeRO stage for evaluation compatibility.")
        except json.JSONDecodeError:
            print(f"Warning: Could not parse DeepSpeed config file {training_args.deepspeed}. Cannot check ZeRO stage for evaluation compatibility.")
        except Exception as e:
            print(f"Warning: An unexpected error occurred while checking DeepSpeed config: {e}. Proceeding without check.")

    set_seed(training_args.seed)
    model, tokenizer = load_model_and_tokenizer(model_args, training_args)

    t0 = time.time()
    print("[DEBUG] Initializing training dataset.")
    print(f"[DEBUG] Using data path: {data_args.data_path}")
    print(f"[DEBUG] Tokenizer model: {model_args.model_name_or_path}")
    print(f"[DEBUG] Max text length: {data_args.max_text_len}")
    print(f"[DEBUG] Require both sequences: {data_args.require_both_sequences}")
    train_dataset = MutationTextDataset(
        data_path=data_args.data_path,
        tokenizer=tokenizer,  # Pass the configured tokenizer instance
        max_text_len=data_args.max_text_len,
        require_both_sequences=data_args.require_both_sequences
    )
    print(f"[DEBUG] Training dataset initialized. Number of samples: {len(train_dataset)}")
    print(f"[TIME] Dataset initialization took {time.time() - t0:.2f} seconds.")

    eval_dataset = None 
    if training_args.do_eval:
        print("Warning: Using training dataset for evaluation as no separate eval_data_path is configured in train.py.")
        eval_dataset = train_dataset 

    t1 = time.time()
    trainer = CustomRNGTrainer(
        model=model,
        args=training_args, # This now uses the fully resolved training_args
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        data_collator=custom_collate_fn,
    )
    print("[DEBUG] Hugging Face Trainer initialized.")
    print(f"[TIME] Trainer initialization took {time.time() - t1:.2f} seconds.")

    print(f"\n[DEBUG] Value of training_args.do_train before training block: {training_args.do_train}")
    if training_args.do_train:
        print("[DEBUG] Starting pretraining.")
        print(f"[DEBUG] train_dataset length: {len(train_dataset) if train_dataset else 'None'}")
        print(f"[DEBUG] training_args.num_train_epochs: {training_args.num_train_epochs}")
        print(f"[DEBUG] training_args.per_device_train_batch_size: {training_args.per_device_train_batch_size}")
        print(f"[DEBUG] training_args.gradient_accumulation_steps: {training_args.gradient_accumulation_steps}")
        t2 = time.time()
        train_result = trainer.train(resume_from_checkpoint=training_args.resume_from_checkpoint)
        print(f"[TIME] trainer.train() took {time.time() - t2:.2f} seconds.")
        print(f"[DEBUG] trainer.train() finished. Result: {train_result}")

        t3 = time.time()
        print("[DEBUG] Attempting to save model (trainer.save_model())...")
        trainer.save_model()
        print(f"[TIME] trainer.save_model() took {time.time() - t3:.2f} seconds.")
        print("[DEBUG] trainer.save_model() finished.")

        t4 = time.time()
        print("[DEBUG] Logging training metrics...")
        trainer.log_metrics("train", train_result.metrics)
        print(f"[TIME] trainer.log_metrics() took {time.time() - t4:.2f} seconds.")
        print("[DEBUG] Training metrics logged.")

        t5 = time.time()
        print("[DEBUG] Saving training metrics...")
        trainer.save_metrics("train", train_result.metrics)
        print(f"[TIME] trainer.save_metrics() took {time.time() - t5:.2f} seconds.")
        print("[DEBUG] Training metrics saved.")

        t6 = time.time()
        print("[DEBUG] Attempting to save state (trainer.save_state())...")
        trainer.save_state()
        print(f"[TIME] trainer.save_state() took {time.time() - t6:.2f} seconds.")
        print("[DEBUG] trainer.save_state() finished.")
        print("[DEBUG] Pretraining finished.")
        print(f"[TIME] Total training block took {time.time() - t2:.2f} seconds.")

    if training_args.do_eval:
        t7 = time.time()
        print("[DEBUG] Starting evaluation.")
        metrics = trainer.evaluate()
        print(f"[TIME] trainer.evaluate() took {time.time() - t7:.2f} seconds.")
        print(f"[DEBUG] trainer.evaluate() finished. Metrics: {metrics}")

        t8 = time.time()
        print("[DEBUG] Logging evaluation metrics...")
        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)
        print(f"[TIME] Logging and saving eval metrics took {time.time() - t8:.2f} seconds.")
        print("[DEBUG] Evaluation metrics saved.")

    print(f"[TIME] Total script runtime: {time.time() - start_total:.2f} seconds.")
    
    # Cleanup distributed training resources
    if torch.distributed.is_initialized():
        torch.distributed.destroy_process_group()

if __name__ == "__main__":
    # Set CUDA_VISIBLE_DEVICES to use both cuda:0 and cuda:1
    os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
    main()