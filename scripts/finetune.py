import os
import sys
import yaml
import json
from dataclasses import dataclass, field, asdict
from typing import Optional, List, Dict, Any
import time
import torch
from transformers import (
    HfArgumentParser,
    TrainingArguments,
    Trainer,
    set_seed,
)
from adapter_trainer import AdapterTrainer
# Add project root to sys.path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from llava.utils.data_utils import MutationTextDataset, custom_collate_fn
from llava.utils.model_utils import load_model_and_tokenizer

@dataclass
class ModelArguments:
    model_name_or_path: str = field(default="meta-llama/Meta-Llama-3.1-8B-Instruct")
    protein_encoder_name_or_path: str = field(default="esm3_sm_open_v1") # Reverted to ESM3 model
    mm_resampler: bool = field(default=True)
    mm_gated_cross_attention: bool = field(default=True)
    mm_projector_type: str = field(default="mlp2x_gelu")
    num_media_tokens: int = field(default=128)
    mm_protein_select_layer: int = field(default=-2)
    use_mm_proj: bool = field(default=True)
    
    # Path to pretrained GCA/Resampler/Projector (output of pretrain_gca.sh)
    pretrained_adapter_path: Optional[str] = field(default="./output/pretrain_gca/", metadata={"help": "Path to the pretrained GCA/Resampler/Projector checkpoint."})
    
    # LoRA specific arguments
    lora_enable: bool = field(default=True) # Enable LoRA for finetuning
    lora_r: int = field(default=16)
    lora_alpha: int = field(default=32)
    lora_dropout: float = field(default=0.05)
    lora_target_modules: List[str] = field(default_factory=lambda: ["q_proj", "k_proj", "v_proj", "o_proj"])
    lora_bias: str = field(default="none", metadata={"help": "Bias type for LoRA. Can be 'none', 'all' or 'lora_only'."})
    tune_mm_mlp_adapter: bool = field(default=True) # Adapters (GCA/Resampler/Projector) are trainable during LoRA
    # mode: str = field(default="train", metadata={"help": "Mode: 'train' or 'inference'"})
    esm_hidden_size: int = field(default=1536, metadata={"help": "Hidden size for ESM3 protein encoder."})
    dim_head: int = field(default=512, metadata={"help": "Dimension of the head for attention."})
    ff_mult: int = field(default=4, metadata={"help": "Feed-forward multiplier for transformer layers."})
    perceiver_depth: int = field(default=6, metadata={"help": "Depth of the Perceiver architecture."})
    gca_output_dim: int = field(default=512, metadata={"help": "Output dimension for GCA (Gated Cross-Attention)."})
    mm_gca_num_heads: int = field(default=8, metadata={"help": "Number of heads for GCA"})
    mm_resampler_num_heads: int = field(default=8, metadata={"help": "Number of heads for resampler"})
    resampler_output_dim: int = field(default=1536, metadata={"help": "Output dimension of the resampler"})

@dataclass
class DataArguments:
    data_path: str = field(default="/data/macaulay/Mutation2Text/data/finetune_train_data.json")
    eval_data_path: Optional[str] = field(default="/data/macaulay/Mutation2Text/data/finetune_eval_data.json")
    require_both_sequences: bool = field(default=True)
    max_text_len: int = field(default=512)

@dataclass
class CustomTrainingArguments(TrainingArguments):
    output_dir: str = field(default="./output/finetune_lora")
    num_train_epochs: int = field(default=2)
    per_device_train_batch_size: int = field(default=2)
    per_device_eval_batch_size: int = field(default=2)
    gradient_accumulation_steps: int = field(default=1)
    evaluation_strategy: str = field(default="steps")
    eval_steps: int = field(default=200)
    save_strategy: str = field(default="steps")
    save_steps: int = field(default=200)
    save_total_limit: int = field(default=2)
    learning_rate: float = field(default=1.0e-5)
    weight_decay: float = field(default=0.)
    warmup_ratio: float = field(default=0.03)
    lr_scheduler_type: str = field(default="cosine")
    logging_steps: int = field(default=10)
    report_to: str = field(default="tensorboard")
    do_train: bool = field(default=True)
    mode: str = field(default="train")
    

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
    parser = HfArgumentParser((ModelArguments, DataArguments, CustomTrainingArguments, ScriptArguments))

    cli_model_args, cli_data_args, cli_training_args, cli_script_args = \
        parser.parse_args_into_dataclasses(args=sys.argv[1:], look_for_args_file=False)

    yaml_config_path = cli_script_args.model_config_file

    if yaml_config_path:
        if not os.path.exists(yaml_config_path):
            raise FileNotFoundError(f"YAML config file '{yaml_config_path}' not found. Passed via --model_config_file.")
        print(f"Loading configuration from YAML: {yaml_config_path}")
        yaml_configs = load_config_from_yaml(yaml_config_path)

        flat_yaml_args = []

        def convert_yaml_section_to_cli_args(section_dict, cli_arg_list):
            if section_dict is None:
                return
            for key, value in section_dict.items():
                if isinstance(value, list):
                    cli_arg_list.append(f'--{key}')
                    for item in value:
                        cli_arg_list.append(str(item))
                # For booleans, HfArgumentParser handles "True" or "False" strings.
                # For other types (string, int, float), str(value) is appropriate.
                else:
                    cli_arg_list.extend([f'--{key}', str(value)])
        
        convert_yaml_section_to_cli_args(yaml_configs.get('model_args'), flat_yaml_args)
        convert_yaml_section_to_cli_args(yaml_configs.get('data_args'), flat_yaml_args)
        convert_yaml_section_to_cli_args(yaml_configs.get('training_args'), flat_yaml_args)

        final_args_to_parse = flat_yaml_args + sys.argv[1:]
        
        model_args, data_args, training_args, script_args = \
            parser.parse_args_into_dataclasses(args=final_args_to_parse, look_for_args_file=False)
    else:
        model_args, data_args, training_args, script_args = \
            cli_model_args, cli_data_args, cli_training_args, cli_script_args

    if hasattr(training_args, 'deepspeed') and isinstance(training_args.deepspeed, str):
        if not os.path.exists(training_args.deepspeed):
            print(f"Warning: DeepSpeed config file specified in training_args ({training_args.deepspeed}) does not exist.")

    print("Effective Model Arguments:", asdict(model_args))
    print("Effective Data Arguments:", asdict(data_args))
    print("Effective Training Arguments:", asdict(training_args))
    if training_args.deepspeed:
        print("Effective Training Arguments (deepspeed config):", training_args.deepspeed)

    if model_args.pretrained_adapter_path is None and model_args.lora_enable:
        print("Warning: LoRA finetuning enabled, but `pretrained_adapter_path` for GCA/Resampler/Projector is not set. Ensure these are correctly loaded if needed.")

    # Check DeepSpeed config for evaluation compatibility
    if training_args.deepspeed and (training_args.do_eval or training_args.evaluation_strategy != "no"):
        try:
            # Resolve DeepSpeed config path
            resolved_ds_path = training_args.deepspeed
            if not os.path.isabs(resolved_ds_path) and not os.path.exists(resolved_ds_path):
                # Try path relative to project root
                path_rel_project = os.path.join(PROJECT_ROOT, training_args.deepspeed)
                if os.path.exists(path_rel_project):
                    resolved_ds_path = path_rel_project
                else:
                    # Try path assuming it's a filename in PROJECT_ROOT/configs/
                    path_in_configs = os.path.join(PROJECT_ROOT, "configs", os.path.basename(training_args.deepspeed))
                    if os.path.exists(path_in_configs):
                        resolved_ds_path = path_in_configs
            
            if os.path.exists(resolved_ds_path):
                with open(resolved_ds_path, 'r') as f:
                    ds_config = json.load(f)
                if ds_config.get("zero_optimization", {}).get("stage") != 3:
                    print(f"Warning: DeepSpeed is enabled with ZeRO stage {ds_config.get('zero_optimization', {}).get('stage', 'N/A')}, "
                          f"which is not Stage 3. Evaluation with DeepSpeed ZeRO inference requires Stage 3. "
                          f"Disabling evaluation for this run. To enable, use a ZeRO Stage 3 config or set evaluation_strategy to 'no'.")
                    training_args.do_eval = False
                    training_args.evaluation_strategy = "no"
            else:
                print(f"Warning: DeepSpeed config file '{training_args.deepspeed}' not found at resolved path '{resolved_ds_path}'. "
                      f"Cannot check ZeRO stage for evaluation compatibility.")
        except json.JSONDecodeError:
            print(f"Warning: Could not parse DeepSpeed config file {resolved_ds_path}. Cannot check ZeRO stage for evaluation compatibility.")
        except Exception as e:
            print(f"Warning: An unexpected error occurred while checking DeepSpeed config: {e}. Proceeding without check.")

    set_seed(training_args.seed)



    #print the training args
    print(f"[DEBUG] Training arguments: {training_args}")
    print(f"[DEBUG] Model arguments: {model_args}")
    print(f"[DEBUG] Data arguments: {data_args}")

    print("[DEBUG] Loading model and tokenizer...")
    model, tokenizer = load_model_and_tokenizer(model_args, training_args)

    t1 = time.time()
    print("[DEBUG] Initializing training dataset.")
    print(f"[DEBUG] Using data path: {data_args.data_path}")
    print(f"[DEBUG] Tokenizer model: {model_args.model_name_or_path}")
    print(f"[DEBUG] Max text length: {data_args.max_text_len}")
    train_dataset = MutationTextDataset(
        data_path=data_args.data_path,
        tokenizer=tokenizer,
        max_text_len=data_args.max_text_len
    )
    print(f"[DEBUG] Training dataset initialized. Number of samples: {len(train_dataset)}")
    print(f"[TIME] Dataset initialization took {time.time() - t1:.2f} seconds.")

    eval_dataset = None
    if training_args.do_eval:
        t2 = time.time()
        if data_args.eval_data_path and os.path.exists(data_args.eval_data_path):
            print("[DEBUG] Initializing evaluation dataset...")
            eval_dataset = MutationTextDataset(
                data_path=data_args.eval_data_path,
                tokenizer=tokenizer,
                max_text_len=data_args.max_text_len,
            )
            print(f"[DEBUG] Evaluation dataset initialized. Number of samples: {len(eval_dataset)}")
            print(f"[TIME] Evaluation dataset initialization took {time.time() - t2:.2f} seconds.")
        else:
            print(f"Warning: `do_eval` is True, but `eval_data_path` ({data_args.eval_data_path}) is not provided or does not exist. No evaluation will be performed.")
            training_args.do_eval = False

    t3 = time.time()
    trainer = AdapterTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        data_collator=custom_collate_fn,
    )
    print("[DEBUG] AdapterTrainer initialized.")
    print(f"[TIME] Trainer initialization took {time.time() - t3:.2f} seconds.")

    print("\n=== DEBUG: Verifying Model State Before Training ===")
    print("Trainable parameters:")
    for name, param in model.named_parameters():
        if param.requires_grad:
            print(f"  - {name}")
    print("================================================\n")

    if training_args.do_train:
        print("[DEBUG] Starting LoRA finetuning...")
        print(f"[DEBUG] train_dataset length: {len(train_dataset) if train_dataset else 'None'}")
        print(f"[DEBUG] training_args.num_train_epochs: {training_args.num_train_epochs}")
        print(f"[DEBUG] training_args.per_device_train_batch_size: {training_args.per_device_train_batch_size}")
        print(f"[DEBUG] training_args.gradient_accumulation_steps: {training_args.gradient_accumulation_steps}")
        
        t4 = time.time()
        train_result = trainer.train(resume_from_checkpoint=training_args.resume_from_checkpoint)
        print(f"[TIME] trainer.train() took {time.time() - t4:.2f} seconds.")
        print(f"[DEBUG] trainer.train() finished. Result: {train_result}")

        t5 = time.time()
        print("[DEBUG] Saving model...")
        trainer.save_model()
        print(f"[TIME] trainer.save_model() took {time.time() - t5:.2f} seconds.")

        t6 = time.time()
        print("[DEBUG] Logging and saving metrics...")
        trainer.log_metrics("train", train_result.metrics)
        trainer.save_metrics("train", train_result.metrics)
        trainer.save_state()
        print(f"[TIME] Metrics logging and saving took {time.time() - t6:.2f} seconds.")
        print("LoRA finetuning finished. Model and state saved.")

    if training_args.do_eval and eval_dataset is not None:
        print("[DEBUG] Starting evaluation on the LoRA finetuned model...")
        t7 = time.time()
        metrics = trainer.evaluate()
        print(f"[TIME] trainer.evaluate() took {time.time() - t7:.2f} seconds.")
        
        t8 = time.time()
        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)
        print(f"[TIME] Evaluation metrics logging and saving took {time.time() - t8:.2f} seconds.")
        print("Evaluation finished. Metrics saved.")

    # print(f"\n[TIME] Total script execution took {time.time() - start_total:.2f} seconds.")

if __name__ == "__main__":
    main()