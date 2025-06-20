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
import dataclasses

# Add project root to path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from llava.utils.model_utils import load_model_and_tokenizer
from llava.utils.data_utils import MutationTextDataset, DELTA_TOKEN
from llava.model.esm_protein_encoder import ESMProteinEncoder
from llava.model.llava_arch import GatedCrossAttention, PerceiverResampler, LlavaLlamaForCausalLM, MLPProjector
import torch.nn as nn
from transformers import AutoTokenizer, AutoConfig

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
    temperature: float = field(default=0.7)
    top_p: float = field(default=0.9)
    do_sample: bool = field(default=True)
    
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

    #lora weights path
    lora_weights_path: str = field(
        default="output/finetune_lora",
        metadata={"help": "Path to LoRA weights for finetuned model"}
    )

def setup_model_and_tokenizer(config: InferenceConfig):
    """
    Sets up the model and tokenizer for inference.
    1. Loads the base LLM model.
    2. Loads the pretrained GCA/Resampler/Projector weights from 'adapters_only.pt'.
    3. Loads LoRA adapter weights if specified.
    """
    logger.info("Setting up model and tokenizer...")

    # Load base model config and add our custom protein_config to it
    model_config = AutoConfig.from_pretrained(config.model_name)
    model_config.protein_config = {
        "mm_gated_cross_attention": True,
        "mm_use_resampler_gca": True,
        "use_mm_proj": True,
        "esm_hidden_size": config.esm_hidden_size,
        "gca_output_dim": config.gca_output_dim,
        "resampler_output_dim": config.resampler_output_dim,
        "mm_projector_type": "mlp2x_gelu",
        "num_media_tokens": 128,
        "mm_gca_num_heads": 8,
        "mm_resampler_num_heads": 8,
    }

    # Load the model using our custom LlavaLlamaForCausalLM class
    # This will load the base model weights and initialize our custom adapters.
    model = LlavaLlamaForCausalLM.from_pretrained(
        config.model_name,
        config=model_config,
        torch_dtype=torch.bfloat16 if config.dtype == "bf16" else torch.float16,
    )
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(config.model_name)
    
    # Add special tokens if they are not already there
    if DELTA_TOKEN not in tokenizer.additional_special_tokens:
        tokenizer.add_special_tokens({"additional_special_tokens": [DELTA_TOKEN]})
        model.resize_token_embeddings(len(tokenizer))
    if tokenizer.pad_token is None:
        print("[DEBUG] Setting pad_token to eos_token")
        tokenizer.pad_token = tokenizer.eos_token
        config.pad_token_id = tokenizer.pad_token_id
        print(f"[DEBUG] Padding token ID set to: {config.pad_token_id}")
    # Load the pretrained GCA, Resampler, and Projector weights
    adapter_weights_path = os.path.join(config.pretrained_adapter_path, "adapters_only.pt")
    if os.path.exists(adapter_weights_path):
        logger.info(f"Loading MM adapter weights from {adapter_weights_path}")
        state_dict = torch.load(adapter_weights_path, map_location="cpu")
        
        # Create a single state dict to load, which is required by the model's load_state_dict
        full_state_to_load = {}
        for module_key, module_state_dict in state_dict.items():
            if module_key in ["mm_gated_cross_attention", "mm_resampler", "mm_projector"]:
                for param_key, value in module_state_dict.items():
                    full_state_to_load[f"{module_key}.{param_key}"] = value
            
        # Load the weights, ignoring mismatches for the base model part
        model.load_state_dict(full_state_to_load, strict=False)
        logger.info("Successfully loaded adapter weights onto base model.")
    else:
         logger.warning(
            f"'adapters_only.pt' not found in {config.pretrained_adapter_path}. "
            "The model will use randomly initialized weights for the adapter modules."
        )

    # Load LoRA weights if provided
    if config.model_type == "finetune" and config.lora_weights_path:
        logger.info(f"Loading LoRA weights from {config.lora_weights_path}")
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, config.lora_weights_path)
        logger.info("Successfully merged LoRA weights")

    # Set up the external protein encoder
    model.protein_encoder = ESMProteinEncoder(config.protein_encoder_name)
    model.set_delta_token_id(tokenizer.convert_tokens_to_ids(DELTA_TOKEN))
    
    # Final setup
    dtype = torch.bfloat16 if config.dtype == "bf16" else torch.float16
    model = model.to(config.device).to(dtype).eval()
    
    logger.info("Model and tokenizer setup complete.")
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
    """Main function to run inference."""
    parser = argparse.ArgumentParser(description="Run inference on a trained model.")
    # You can still use dataclass fields to auto-generate parser arguments
    for field in dataclasses.fields(InferenceConfig):
        parser.add_argument(f"--{field.name}", type=type(field.default), default=field.default)
        
    args = parser.parse_args()
    config = InferenceConfig(**vars(args))

    # Setup model and tokenizer
    model, tokenizer = setup_model_and_tokenizer(config)
    delta_token = DELTA_TOKEN

    # Load data
    test_data = load_test_data(config.data_path)

    # Prepare for generation
    results = []
    with open(config.output_path, 'w') as f_out:
        for i in tqdm(range(0, len(test_data), config.batch_size), desc="Generating responses"):
            batch_data = test_data[i:i+config.batch_size]
            
            prompts = [generate_prompt(item['conversations'][0]['value'], delta_token) for item in batch_data]
            wild_type_seqs = [item.get('wild_type_sequence', '') for item in batch_data]
            mutation_seqs = [item.get('mutation_sequence', '') for item in batch_data]

            # Tokenize prompts
            inputs = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True)
            inputs = {k: v.to(config.device) for k, v in inputs.items()}

            # Generate responses
            outputs = model.generate(
                **inputs,
                wild_type_sequences=wild_type_seqs,
                mutation_sequences=mutation_seqs,
                max_new_tokens=config.max_new_tokens,
                num_beams=config.num_beams,
                do_sample=config.do_sample,
                temperature=config.temperature,
                top_p=config.top_p,
            )

            # Decode and save results
            decoded_outputs = tokenizer.batch_decode(outputs, skip_special_tokens=True)
            for j, item in enumerate(batch_data):
                response = decoded_outputs[j].split("Assistant:")[-1].strip()
                result_item = {
                    "id": item.get('id', f'item_{i+j}'),
                    "prompt": prompts[j],
                    "response": response,
                    "ground_truth": item['conversations'][1]['value']
                }
                results.append(result_item)
                f_out.write(json.dumps(result_item) + '\n')

    logger.info(f"Inference complete. Results saved to {config.output_path}")

if __name__ == "__main__":
    main()
