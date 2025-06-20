import torch
from transformers import AutoTokenizer, AutoConfig
from llava.model.llava_arch import LlavaLlamaForCausalLM # Relative import
from llava.model.esm_protein_encoder import ESMProteinEncoder # Relative import
from llava.model.lora_adapter import create_lora_model # Relative import for LoRA
from llava.utils.data_utils import DELTA_TOKEN # For tokenizer setup
import os
from llava.model.llava_arch import GatedCrossAttention, PerceiverResampler
import torch.nn as nn

def load_model_and_tokenizer(model_args, training_args):
    """
    Loads the LlavaLlamaForCausalLM model, ESM protein encoder, and tokenizer.
    Handles initialization for pretraining or LoRA finetuning.
    """
    print("[DEBUG] Loading model and tokenizer.")
    # Load base Llama config and tokenizer
    config = AutoConfig.from_pretrained(model_args.model_name_or_path, trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path, use_fast=True, trust_remote_code=True)

    # Configure padding token (use EOS token if pad token is not set)
    if tokenizer.pad_token is None:
        print("[DEBUG] Setting pad_token to eos_token")
        tokenizer.pad_token = tokenizer.eos_token
        config.pad_token_id = tokenizer.pad_token_id
        print(f"[DEBUG] Padding token ID set to: {config.pad_token_id}")

    # Add delta_P token if not present
    if DELTA_TOKEN not in tokenizer.additional_special_tokens:
        print(f"[DEBUG] Adding {DELTA_TOKEN} to special tokens")
        tokenizer.add_special_tokens({'additional_special_tokens': [DELTA_TOKEN]})
    delta_token_id = tokenizer.convert_tokens_to_ids(DELTA_TOKEN)
    print(f"[DEBUG] Delta token ID: {delta_token_id}")

    print(f"Config {config}")

    # Attach protein model configuration to the main Llama config
    # These would come from model_args in a training script
    config.protein_config = {
        "protein_encoder_name_or_path": model_args.protein_encoder_name_or_path,
        "mm_gated_cross_attention": model_args.mm_gated_cross_attention,
        "mm_use_resampler_gca": model_args.mm_use_resampler_gca,
        "num_media_tokens": model_args.num_media_tokens,
        "mm_projector_type": model_args.mm_projector_type,
        "use_mm_proj": getattr(model_args, "use_mm_proj", True),  # Added this
        "mm_protein_select_layer": model_args.mm_protein_select_layer,
        # These dimensions might be dynamically inferred or explicitly set in a full setup
        "esm_hidden_size": getattr(model_args, "esm_hidden_size", 1536), # Example, ideally from ESM config
        "gca_output_dim": getattr(model_args, "gca_output_dim", 512),   # Example
        "resampler_output_dim": getattr(model_args, "resampler_output_dim", config.hidden_size), # Projector might handle the final projection to LLM dim
    }


    print(f"Config {config}")
    # # Instantiate the main Llava model
    # model = LlavaLlamaForCausalLM(config)
    # model.resize_token_embeddings(len(tokenizer)) # Important after adding new tokens
    # model.set_delta_token_id(delta_token_id)

    # Instantiate and set the protein encoder
    print("[DEBUG] Configuring protein encoder.")
    print(f"[DEBUG] mode detected: {getattr(training_args, 'mode', None)}")
    if getattr(training_args, 'mode', None) == 'inference':
        # For inference mode, try cuda:1 first, if not available fall back to cuda:0, then cpu
        if torch.cuda.is_available() and torch.cuda.device_count() > 1:
            print("[DEBUG] inference mode detected, multiple GPUs available. Using cuda:1 for protein encoder.")
            device = torch.device("cuda:1")
        elif torch.cuda.is_available():
            print("[DEBUG] inference mode detected, only one GPU available. Using cuda:0 for protein encoder.")
            device = torch.device("cuda:0")
        else:
            print("[DEBUG] No CUDA devices available. Using CPU for protein encoder.")
            device = torch.device("cpu")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[DEBUG] Setting up protein encoder on device: {device}")
    
    protein_encoder = ESMProteinEncoder(model_args.protein_encoder_name_or_path)
    protein_encoder.to(device)  # Explicitly move to correct device

    # Dynamically set esm_hidden_size in config to match ESM3 encoder output
    esm_output_dim = protein_encoder.output_embedding_dim
    print(f"[DEBUG] Detected ESM3 output embedding dim: {esm_output_dim}")
    config.protein_config["esm_hidden_size"] = esm_output_dim

    # Re-instantiate the model with the correct config
    model = LlavaLlamaForCausalLM.from_pretrained(
        model_args.model_name_or_path, 
        config=config,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2"
    )
    model.resize_token_embeddings(len(tokenizer))
    model.set_delta_token_id(delta_token_id)

    # Set the protein encoder on the model
    model.set_protein_encoder(protein_encoder)
    # print(f"[DEBUG] Protein encoder set on model. {'device': device, 'output_dim': esm_output_dim}")
    # Ensure the model is in eval mode if not training
    # print(protein_encoder)
    
    
        # Freeze protein encoder by default
    if hasattr(model, 'protein_encoder') and model.protein_encoder is not None:
        print("[DEBUG] Freezing protein encoder parameters.")
        print("Freezing Protein Encoder (ESM) parameters.")
        for param in model.protein_encoder.parameters():
            param.requires_grad = False
        # Ensure protein encoder is in eval mode if frozen
        model.protein_encoder.eval()

    # Handle loading pretrained adapter weights
    if getattr(model_args, 'pretrained_adapter_path', None) is not None:
        adapter_path_dir = model_args.pretrained_adapter_path
        print(f"Attempting to load pretrained adapter weights from directory: {adapter_path_dir}")

        # Check for the adapters_only.pt file saved by AdapterTrainer
        adapters_only_path = os.path.join(adapter_path_dir, "adapters_only.pt")

        if os.path.exists(adapters_only_path):
            try:
                print(f"Loading adapter weights from {adapters_only_path}")
                adapter_state_dict = torch.load(adapters_only_path, map_location='cpu')

                # Load state dicts into the specific modules
                if "mm_gated_cross_attention" in adapter_state_dict and hasattr(model, 'mm_gated_cross_attention') and model.mm_gated_cross_attention is not None:
                    print("Loading GCA weights...")
                    model.mm_gated_cross_attention.load_state_dict(adapter_state_dict["mm_gated_cross_attention"])
                if "mm_resampler" in adapter_state_dict and hasattr(model, 'mm_resampler') and model.mm_resampler is not None:
                    print("Loading Resampler weights...")
                    model.mm_resampler.load_state_dict(adapter_state_dict["mm_resampler"])
                if "mm_projector" in adapter_state_dict and hasattr(model, 'mm_projector') and model.mm_projector is not None:
                    print("Loading Projector weights...")
                    model.mm_projector.load_state_dict(adapter_state_dict["mm_projector"])

                print(f"Successfully loaded adapter weights from {adapters_only_path}.")

            except Exception as e:
                
                print(f"Error loading adapter weights from {adapters_only_path}: {e}")
                print("Proceeding with randomly initialized or default weights for adapters.")
                pass
        else:
            
            print(f"Warning: No adapters_only.pt file found in {adapter_path_dir}, Adapter weights will not be loaded from this path.")
            pass

    # --- MODE HANDLING: skip training-only logic if in inference mode ---
    # Accepts model_args.mode ('train' or 'inference'), or model_args.is_inference (bool)
    mode = getattr(training_args, 'mode', None)
    print(f"[DEBUG] Training mode: {mode}")
    if  str(mode).lower() == 'inference':
        print("[INFO] Inference mode detected: Skipping training-only parameter freezing and checks.")
        #load weights from checkpoint
        return model, tokenizer

    # Only do pretraining/finetuning logic if in training mode
    if str(mode).lower() == 'train':
        # Configure parameter freezing based on training stage
        print("[DEBUG] Configuring trainable modules.")
        if getattr(model_args, 'tune_mm_mlp_adapter', False) and not getattr(model_args, 'lora_enable', False):
            print("Pretraining mode: Freezing LLM, LM head, and protein encoder. Training GCA, Resampler, Projector.")
            
            # Track which modules we successfully configure
            trainable_modules = []
            frozen_modules = []
            
            # 1. First freeze the LLM backbone and LM head specifically
            if hasattr(model, 'model'):  # LlamaModel
                print("Freezing LLM backbone parameters.")
                for param in model.model.parameters():
                    param.requires_grad = False
                frozen_modules.append("LLM backbone")
            
            if hasattr(model, 'lm_head'):  # Language modeling head
                print("Freezing LM head parameters.")
                for param in model.lm_head.parameters():
                    param.requires_grad = False
                frozen_modules.append("LM head")
            
            # 2. Ensure protein encoder is frozen (it should be from earlier, but let's be explicit)
            if hasattr(model, 'protein_encoder') and model.protein_encoder is not None:
                print("Ensuring protein encoder is frozen.")
                for param in model.protein_encoder.parameters():
                    param.requires_grad = False
                model.protein_encoder.eval()  # Ensure it's in eval mode
                frozen_modules.append("Protein encoder")
            
            # 3. Configure trainable modules - with better error checking
            # GCA
            if hasattr(model, 'mm_gated_cross_attention'):
                if model.mm_gated_cross_attention is not None:
                    print("Setting GCA module (mm_gated_cross_attention) as trainable.")
                    for param in model.mm_gated_cross_attention.parameters():
                        param.requires_grad = True
                    trainable_modules.append("GCA")
                else:
                    print("WARNING: mm_gated_cross_attention exists but is None")
                    pass
            else:
                print("WARNING: mm_gated_cross_attention module not found in model")
                pass
            
            # Resampler
            if hasattr(model, 'mm_resampler'):
                if model.mm_resampler is not None:
                    print("Setting Protein Resampler module (mm_resampler) as trainable.")
                    for param in model.mm_resampler.parameters():
                        param.requires_grad = True
                    trainable_modules.append("Resampler")
                else:
                    print("WARNING: mm_resampler exists but is None")
                    pass
            else:
                print("WARNING: mm_resampler module not found in model")
                pass
            
            # Projector
            if hasattr(model, 'mm_projector'):
                if model.mm_projector is not None:
                    print("Setting Multimodal Projector (mm_projector) as trainable.")
                    for param in model.mm_projector.parameters():
                        param.requires_grad = True
                    trainable_modules.append("Projector")
                else:
                    print("WARNING: mm_projector exists but is None")
                    pass
            else:
                print("WARNING: mm_projector module not found in model")
                pass

            # Summary and validation
            print(f"\nConfiguration Summary:")
            print(f"Frozen modules: {', '.join(frozen_modules)}")
            print(f"Trainable modules: {', '.join(trainable_modules)}")
            
            if not trainable_modules:
                raise ValueError(
                    "Critical Error: No trainable modules found in pretraining mode. "
                    "Expected at least one of: GCA, Resampler, or Projector to be present and trainable. "
                    "Check model configuration and initialization."
                )
            
            # Verify we have some trainable parameters
            num_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
            if num_trainable == 0:
                raise ValueError(
                    f"No trainable parameters found after configuration! "
                    f"This would cause an optimizer error. "
                    f"Modules that should be trainable: {', '.join(trainable_modules)}"
                )

        if getattr(model_args, 'lora_enable', False):
            print("[INFO] LoRA finetuning enabled. Applying LoRA to the model.")
            
            lora_config_params = {
                'r': model_args.lora_r,
                'lora_alpha': model_args.lora_alpha,
                'target_modules': model_args.lora_target_modules,
                'lora_dropout': model_args.lora_dropout,
                'bias': getattr(model_args, 'lora_bias', 'none'),
            }

            # This single call now handles applying LoRA and unfreezing adapters if needed.
            model = create_lora_model(
                model, 
                lora_config_params, 
                tune_mm_adapters=getattr(model_args, 'tune_mm_mlp_adapter', False)
            )

    # print a summary of trainable parameters at the end for verification
    print("\n--- Final Trainable Parameters Summary ---")
    total_params = sum(p.numel() for p in model.parameters())

    return model, tokenizer

# Example helper for managing model saving/loading if not using HuggingFace Trainer
def save_checkpoint(model, optimizer, epoch, loss, checkpoint_dir, model_name):
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
    checkpoint_path = os.path.join(checkpoint_dir, f"{model_name}_epoch_{epoch}.pt")
    # Save model state, not the whole model if it's large and can be reconstructed
    # For DeepSpeed, Trainer handles saving more robustly.
    state_to_save = model.state_dict() 
    # If using DeepSpeed, need to get state_dict from engine: `model_engine.module.state_dict()`
    torch.save({
        'epoch': epoch,
        'model_state_dict': state_to_save,
        'optimizer_state_dict': optimizer.state_dict() if optimizer else None, # Optimizer might be handled by DS
        'loss': loss,
    }, checkpoint_path)
    print(f"Checkpoint saved to {checkpoint_path}")

def load_checkpoint(model, optimizer, checkpoint_path):
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_path}")
    
    map_location = 'cuda' if torch.cuda.is_available() else 'cpu'
    checkpoint = torch.load(checkpoint_path, map_location=map_location)
    
    # Handle nested state dicts that might come from DataParallel or custom saving
    model_state_dict = checkpoint['model_state_dict']
    if hasattr(model, 'module') and not isinstance(model, torch.nn.parallel.DistributedDataParallel): # Check if it's a DataParallel wrapped model
        try: # Try loading into model.module
            model.module.load_state_dict(model_state_dict)
        except: # Fallback to loading directly if model.module doesn't exist or keys don't match
            model.load_state_dict(model_state_dict)
    else: # Standard model or DDP (DDP usually strips 'module.' prefix on save or handles it)
        model.load_state_dict(model_state_dict)

    if optimizer is not None and 'optimizer_state_dict' in checkpoint and checkpoint['optimizer_state_dict'] is not None:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
    epoch = checkpoint.get('epoch', 0)
    loss = checkpoint.get('loss', float('inf'))
    print(f"Checkpoint loaded from {checkpoint_path}, epoch {epoch}, loss {loss:.4f}")
    return model, optimizer, epoch, loss