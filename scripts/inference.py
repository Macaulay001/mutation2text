import os
import sys
import json
import torch
import torch.nn as nn
import argparse
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
from tqdm import tqdm
import logging

# Add project root to path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from llava.utils.model_utils import load_model_and_tokenizer
from llava.utils.data_utils import MutationTextDataset, DELTA_TOKEN
from llava.model.esm_protein_encoder import ESMProteinEncoder
from llava.model.llava_arch import GatedCrossAttention, PerceiverResampler
import torch.nn as nn

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class InferenceConfig:
    # Model paths
    checkpoint_dir: str = field(default="output/pretrain_gca/checkpoint-10000")
    model_name: str = field(default="meta-llama/Meta-Llama-3.1-8B-Instruct")
    protein_encoder_name: str = field(default="esm3_sm_open_v1")
    
    # Model type and weights
    model_type: str = field(
        default="finetune",
        metadata={"help": "Type of model to use: 'pretrain' or 'finetune'"}
    )
    lora_weights_path: Optional[str] = field(
        default=None,
        metadata={"help": "Path to LoRA weights for finetuned model"}
    )
    
    # Data and output
    data_path: str = field(default="data/mut_text_data.json")
    output_path: str = field(default="output/generated_responses.jsonl")
    
    # Model configuration
    batch_size: int = field(default=4)
    max_new_tokens: int = field(default=128)
    device: str = field(default="cuda")
    dtype: str = field(default="bf16")
    mm_protein_select_layer: int = field(default=-2)
    
    # Architecture dimensions (can be overridden by saved config)
    esm_hidden_size: int = field(default=1536)
    gca_output_dim: int = field(default=512)
    resampler_output_dim: int = field(default=4096)
    
    # Generation parameters
    num_beams: int = field(default=1)
    temperature: float = field(default=1.0)
    top_p: float = field(default=0.9)
    do_sample: bool = field(default=False)
    
    # LoRA parameters (used when model_type is "finetune")
    lora_r: int = field(default=16)
    lora_alpha: int = field(default=32)
    lora_dropout: float = field(default=0.05)
    lora_target_modules: List[str] = field(
        default_factory=lambda: ["q_proj", "k_proj", "v_proj", "o_proj"]
    )
    # lora_bias: str = field(default="none")

    # Mode: 'train' or 'inference'
    mode: str = field(default="inference", metadata={"help": "Mode: 'train' or 'inference'"})
# Pretrained adapter path (if applicable)
    pretrained_adapter_path: str = field(
        default="output/pretrain_gca/",
        metadata={"help": "Path to pretrained adapter weights (if applicable)"}
    )

def setup_model_and_tokenizer(config: InferenceConfig):
    logger.info("Loading model and tokenizer...")
    
    # Load protein config from checkpoint
    config_path = os.path.join(config.checkpoint_dir, "config.json")
    if os.path.exists(config_path):
        with open(config_path, 'r') as f:
            saved_config = json.load(f)
            if 'protein_config' in saved_config:
                logger.info("Loading protein configuration from checkpoint")
                protein_config = saved_config['protein_config']
                config.esm_hidden_size = protein_config.get('esm_hidden_size', config.esm_hidden_size)
                config.gca_output_dim = protein_config.get('gca_output_dim', config.gca_output_dim)
                config.resampler_output_dim = protein_config.get('resampler_output_dim', config.resampler_output_dim)
                config.mm_protein_select_layer = protein_config.get('mm_protein_select_layer', config.mm_protein_select_layer)

    # Setup model arguments
    model_args = {
        "model_name_or_path": config.model_name,
        "protein_encoder_name_or_path": config.protein_encoder_name,
        "mm_use_resampler_gca": True,  # Enable for inference
        "mm_gated_cross_attention": True,  # Enable for inference
        "mm_projector_type": "mlp2x_gelu",  # Enable for inference
        "num_media_tokens": 128,
        "use_mm_proj": True,
        "mm_protein_select_layer": config.mm_protein_select_layer,
        "esm_hidden_size": config.esm_hidden_size,
        "gca_output_dim": config.gca_output_dim,
        "resampler_output_dim": config.resampler_output_dim,
        "pretrained_adapter_path": config.pretrained_adapter_path,
    }
    
    # Add LoRA configuration if in finetune mode
    print(f"Model type: {config.model_type}")
    if config.model_type == "finetune":
        model_args.update({
            "lora_enable": True,
            "lora_r": config.lora_r,
            "lora_alpha": config.lora_alpha,
            "lora_dropout": config.lora_dropout,
            "lora_target_modules": config.lora_target_modules,
            # "lora_bias": config.lora_bias,
            "tune_mm_mlp_adapter": False  # Adapters are frozen during LoRA
            
        })
    else:
        model_args.update({
            "lora_enable": False,
            "tune_mm_mlp_adapter": False  # Enable adapter training in pretrain mode
        })
    
    # Setup training arguments (minimal for inference)
    training_args = {
        "output_dir": config.checkpoint_dir,
        "do_train": False,
        "do_eval": False,
        "local_rank": -1,
        "deepspeed": None,  # Disable deepspeed for inference
        "bf16": config.dtype == "bf16",  # Set precision
        "fp16": config.dtype == "fp16",
        "mode": config.mode

    }
    
    # Convert dictionaries to objects for load_model_and_tokenizer
    model_args_obj = type("ModelArgs", (), model_args)()
    training_args_obj = type("TrainingArgs", (), training_args)()
    
    # Load base model and tokenizer (with adapters)
    model, tokenizer = load_model_and_tokenizer(model_args_obj, training_args_obj)

    # Load weights based on model type
    # if config.model_type == "pretrain":
    logger.info("Loading pretrained adapter weights...")
    adapter_weights_path = os.path.join(config.checkpoint_dir, "adapters_only.pt")
    full_checkpoint_path = None

    # if os.path.exists(adapter_weights_path):
    #     logger.info(f"Loading adapter weights from {adapter_weights_path}")
    #     checkpoint = torch.load(adapter_weights_path, map_location='cpu')
    #     adapter_state = checkpoint
    # else:
    #     logger.warning(f"Adapter weights not found at {adapter_weights_path}. Looking for full checkpoint...")
    #     # Look for a global_stepXXXX subdirectory
    #     global_step_dirs = [d for d in os.listdir(config.checkpoint_dir) if os.path.isdir(os.path.join(config.checkpoint_dir, d)) and d.startswith('global_step')]
    #     if global_step_dirs:
    #         # Find the latest global_step directory
    #         latest_global_step_dir = max(global_step_dirs, key=lambda x: int(x.replace('global_step', '')))
    #         full_checkpoint_path = os.path.join(config.checkpoint_dir, latest_global_step_dir, "mp_rank_00_model_states.pt")
            
    #         if os.path.exists(full_checkpoint_path):
    #             logger.info(f"Loading full checkpoint from {full_checkpoint_path} and extracting adapter weights")
    #             checkpoint = torch.load(full_checkpoint_path, map_location='cpu')
                
    #             # Extract adapters from full checkpoint
    #             adapter_state = {}
    #             model_state = checkpoint.get('module', checkpoint)  # Handle DeepSpeed wrapping
    #             for key, value in model_state.items():
    #                 if any(x in key for x in ['mm_gated_cross_attention', 'mm_resampler', 'mm_projector']):
    #                     adapter_state[key] = value
    #         else:
    #                 raise FileNotFoundError(
    #                 f"Neither adapter weights ({adapter_weights_path}) nor a full checkpoint ({full_checkpoint_path}) found in {config.checkpoint_dir}."
    #             )
    #     else:
    #             raise FileNotFoundError(
    #             f"Neither adapter weights ({adapter_weights_path}) nor any global_step subdirectories found in {config.checkpoint_dir}."
    #         )
    
    #     logger.info("Loading adapter weights into model...")
    #     # Load the weights into the model components
    #     if hasattr(model, 'mm_gated_cross_attention') and model.mm_gated_cross_attention is not None:
    #         gca_state = {k.replace('mm_gated_cross_attention.', ''): v for k, v in adapter_state.items() if k.startswith('mm_gated_cross_attention.')}
    #         if gca_state:
    #             model.mm_gated_cross_attention.load_state_dict(gca_state)
    #             logger.info("Loaded GCA weights")
            
    #     if hasattr(model, 'mm_resampler') and model.mm_resampler is not None:
    #         resampler_state = {k.replace('mm_resampler.', ''): v for k, v in adapter_state.items() if k.startswith('mm_resampler.')}
    #         if resampler_state:
    #             model.mm_resampler.load_state_dict(resampler_state)
    #             logger.info("Loaded Resampler weights")
            
    #     if hasattr(model, 'mm_projector') and model.mm_projector is not None:
    #         projector_state = {k.replace('mm_projector.', ''): v for k, v in adapter_state.items() if k.startswith('mm_projector.')}
    #         if projector_state:
    #             model.mm_projector.load_state_dict(projector_state)
    #             logger.info("Loaded Projector weights")
    
    if config.model_type == "finetune" and config.lora_weights_path:
        logger.info(f"Loading LoRA weights from {config.lora_weights_path}")
        from peft import PeftModel, PeftConfig
        try:
            # Load LoRA config and then the model with LoRA weights
            peft_config = PeftConfig.from_pretrained(config.lora_weights_path)
            model = PeftModel.from_pretrained(model, config.lora_weights_path)
            logger.info("Successfully loaded LoRA weights")
        except Exception as e:
            logger.error(f"Failed to load LoRA weights: {e}")
            raise
    
    elif config.model_type == "finetune" and not config.lora_weights_path:
         logger.warning("Finetune mode specified but no lora_weights_path provided. Loading base model only.")

    logger.info("Successfully loaded all weights")

    # Move model to device and set dtype
    dtype = torch.bfloat16 if config.dtype == "bf16" else torch.float16
    model = model.to(config.device).to(dtype)
    model.eval()  # Set to evaluation mode
    
    # Verify all components are properly loaded
    logger.info("Model components loaded successfully:")
    logger.info(f"- Model on device: {next(model.parameters()).device}")
    logger.info(f"- Model dtype: {next(model.parameters()).dtype}")
    logger.info(f"- Protein Encoder loaded: {model.protein_encoder is not None}")
    logger.info(f"- GCA loaded: {model.mm_gated_cross_attention is not None}")
    logger.info(f"- Resampler loaded: {model.mm_resampler is not None}")
    logger.info(f"- Projector loaded: {model.mm_projector is not None}")
    
    return model, tokenizer

def generate_prompt(human_query: str, delta_token: str) -> str:
    """Generate the prompt for mutation explanation."""
    # Format consistent with training: Human asks with delta token, Assistant responds
    system_prompt = "You are a helpful assistant that explains protein mutations."
    processed_query = human_query.replace("<wt_prot_seq>\n<mut_prot_seq>", delta_token)
    prompt = f"{system_prompt}\n\nHuman: {processed_query}\n\nAssistant:"
    return prompt

def load_test_data(data_path: str) -> List[Dict[str, Any]]:
    """Load test data from JSON file."""
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Test data file not found: {data_path}")
    
    with open(data_path, 'r') as f:
        data = json.load(f)
    return data

@torch.no_grad()
def generate_response(
    model: torch.nn.Module,
    tokenizer,
    human_query: str,
    wild_type_seq: str,
    mutation_seq: str,
    config: InferenceConfig,
    debug: bool = True
) -> str:
    """Generate a response for a mutation query using model.generate()."""
    # Debug logging
    prompt = generate_prompt(human_query, DELTA_TOKEN)
    if debug:
        logger.info("\nGenerating response for:")
        logger.info(f"Wild type sequence (len={len(wild_type_seq)}): {wild_type_seq[:50]}...")
        logger.info(f"Mutation sequence (len={len(mutation_seq)}): {mutation_seq[:50]}...")
        logger.info(f"Query: {prompt}")
        
        # Add hash of sequences to verify they're different
        wt_hash = hash(wild_type_seq)
        mut_hash = hash(mutation_seq)
        logger.info(f"Wild type sequence hash: {wt_hash}")
        logger.info(f"Mutation sequence hash: {mut_hash}")
    
    # Tokenize input
    inputs = tokenizer(prompt, return_tensors="pt").to(config.device)
    
    try:
        # Set pad token if not set
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token_id = tokenizer.eos_token_id
        
        # Use the model's generate method with protein sequences
        output_ids = model.generate(
            input_ids=inputs["input_ids"],
            wild_type_sequences=[wild_type_seq],
            mutation_sequences=[mutation_seq],
            attention_mask=inputs["attention_mask"],
            max_new_tokens=config.max_new_tokens,
            do_sample=True,  # Enable sampling to get diverse outputs
            temperature=0.7,  # Lower temperature for more focused but still diverse outputs
            top_p=0.9,
            top_k=50,  # Add top-k sampling
            num_beams=1,  # Use 1 beam with sampling for diversity
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
            use_cache=True,
            repetition_penalty=1.2,  # Add repetition penalty to discourage identical outputs
        )
        
        # Decode the generated response
        response = tokenizer.decode(output_ids[0], skip_special_tokens=True)
        
        # Extract just the assistant's response
        response = response.split("Assistant:")[-1].strip()
        
        return response
    
    except Exception as e:
        logger.error(f"Error generating response: {e}")
        import traceback
        traceback.print_exc()
        return f"Error generating response: {str(e)}"

def main():
    parser = argparse.ArgumentParser(description="Run inference with pretrained or finetuned mutation model")
    # Model paths
    parser.add_argument("--checkpoint_dir", type=str, default="output/pretrain_gca/checkpoint-10000",
                       help="Directory containing the model checkpoint")
    parser.add_argument("--model_name", type=str, default="meta-llama/Meta-Llama-3.1-8B-Instruct",
                       help="Base model name or path")
    parser.add_argument("--protein_encoder_name", type=str, default="esm3_sm_open_v1",
                       help="Protein encoder model name")

    # Model type and weights
    parser.add_argument("--model_type", type=str, default="pretrain", choices=["pretrain", "finetune"],
                       help="Type of model to use: 'pretrain' or 'finetune'")
    parser.add_argument("--lora_weights_path", type=str, default=None,
                       help="Path to LoRA weights for finetuned model")

    # Data and output
    parser.add_argument("--data_path", type=str, default="data/mut_text_data.json",
                       help="Path to test data file")
    parser.add_argument("--output_path", type=str, default="output/generated_responses.jsonl",
                       help="Path to save generated responses")

    # Model configuration
    parser.add_argument("--batch_size", type=int, default=4,
                       help="Batch size for inference")
    parser.add_argument("--max_new_tokens", type=int, default=128,
                       help="Maximum number of tokens to generate")
    parser.add_argument("--device", type=str, default="cuda",
                       help="Device to run inference on")
    parser.add_argument("--dtype", type=str, default="bf16", choices=["bf16", "fp16"],
                       help="Data type for model weights")

    # Generation parameters
    parser.add_argument("--num_beams", type=int, default=1,
                       help="Number of beams for beam search")
    parser.add_argument("--temperature", type=float, default=1.0,
                       help="Temperature for sampling")
    parser.add_argument("--top_p", type=float, default=0.9,
                       help="Top-p sampling parameter")
    parser.add_argument("--do_sample", action="store_true",
                       help="Whether to use sampling instead of greedy decoding")

    # LoRA parameters
    parser.add_argument("--lora_r", type=int, default=16,
                       help="LoRA attention dimension")
    parser.add_argument("--lora_alpha", type=int, default=32,
                       help="LoRA alpha scaling factor")
    parser.add_argument("--lora_dropout", type=float, default=0.05,
                       help="LoRA dropout probability")
    parser.add_argument("--mode", type=str, default="inference", choices=["train", "inference"],
                       help="Mode of operation: 'train' or 'inference'")
    parser.add_argument("--lora_target_modules", type=str, nargs='+', default=["q_proj", "k_proj", "v_proj", "o_proj"],
                       help="List of target modules for LoRA")
    parser.add_argument("--pretrained_adapter_path", type=str, default="output/pretrain_gca/",
                       help="Path to pretrained adapter weights (if applicable)")
    args = parser.parse_args()

    # Create config from args
    config = InferenceConfig(
        checkpoint_dir=args.checkpoint_dir,
        model_name=args.model_name,
        protein_encoder_name=args.protein_encoder_name,
        model_type=args.model_type,
        lora_weights_path=args.lora_weights_path,
        data_path=args.data_path,
        output_path=args.output_path,
        batch_size=args.batch_size,
        max_new_tokens=args.max_new_tokens,
        device=args.device,
        dtype=args.dtype,
        num_beams=args.num_beams,
        temperature=args.temperature,
        top_p=args.top_p,
        do_sample=args.do_sample,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        mode=args.mode,
        pretrained_adapter_path=args.pretrained_adapter_path,
    )

    # Load model and tokenizer
    model, tokenizer = setup_model_and_tokenizer(config)
    
    # Load test data
    test_data = load_test_data(config.data_path)
    
    # Debug info about sequence variations
    unique_wt_seqs = len(set(item["wild_type_seq"] for item in test_data if "wild_type_seq" in item))
    unique_mut_seqs = len(set(item["mutation_seq"] for item in test_data if "mutation_seq" in item))
    logger.info(f"Number of unique wild type sequences: {unique_wt_seqs}")
    logger.info(f"Number of unique mutation sequences: {unique_mut_seqs}")
    
    # Debug: Print first few examples
    for i, item in enumerate(test_data[:3]):
        wt_seq = item.get("wild_type_seq", "")
        mut_seq = item.get("mutation_seq", "")
        logger.info(f"\nExample {i+1}:")
        logger.info(f"Wild type sequence (len={len(wt_seq)}): {wt_seq[:50]}...")
        logger.info(f"Mutation sequence (len={len(mut_seq)}): {mut_seq[:50]}...")

    logger.info(f"Loaded {len(test_data)} test examples")
    
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(config.output_path), exist_ok=True)

    # Process test data and generate responses
    results = []
    required_keys = ["wild_type_seq", "mutation_seq", "conversations"]
    for idx, item in enumerate(tqdm(test_data, desc="Generating responses")):
        if not all(k in item for k in required_keys):
            logger.warning(f"Skipping item {idx} due to missing keys. Found keys: {list(item.keys())}")
            continue
        
        # Extract query from conversations
        human_query = "Describe the mutation?"  # Default query
        if item["conversations"] and len(item["conversations"]) > 0:
            human_conv = item["conversations"][0]
            if human_conv.get("from") == "human" and "value" in human_conv:
                # Use the original human conversation value without removing placeholders
                # The generate_prompt function will handle the placeholder replacement
                human_query = human_conv["value"]
        
        response = generate_response(
            model=model,
            tokenizer=tokenizer,
            human_query=human_query,
            wild_type_seq=item["wild_type_seq"],
            mutation_seq=item["mutation_seq"],
            config=config
        )
        result = {
            "id": item.get("id", f"item_{idx}"),
            "query": human_query,
            "wild_type_sequence": item["wild_type_seq"],
            "mutation_sequence": item["mutation_seq"],
            "generated_response": response,
            "expected_response": item["conversations"][1]["value"] if len(item["conversations"]) > 1 else None
        }
        results.append(result)

        # Write results to file as we go
        with open(config.output_path, 'a') as f:
            f.write(json.dumps(result) + '\n')

    logger.info(f"Inference completed. Results saved to {config.output_path}")

if __name__ == "__main__":
    main()
