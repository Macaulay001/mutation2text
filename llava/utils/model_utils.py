import torch
from transformers import AutoTokenizer, AutoConfig
from llava.model.llava_arch import LlavaLlamaForCausalLM # Relative import
from llava.model.esm_protein_encoder import ESMProteinEncoder # Relative import
from llava.model.lora_adapter import create_lora_model # Relative import for LoRA
from llava.utils.data_utils import WT_PROTEIN_START_TOKEN, WT_PROTEIN_END_TOKEN, MUT_PROTEIN_START_TOKEN, MUT_PROTEIN_END_TOKEN # For tokenizer setup
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

    # Add all special tokens at once to avoid multiple vocabulary expansions
    special_tokens_to_add = []
    if WT_PROTEIN_START_TOKEN not in tokenizer.additional_special_tokens:
        special_tokens_to_add.append(WT_PROTEIN_START_TOKEN)
    if WT_PROTEIN_END_TOKEN not in tokenizer.additional_special_tokens:
        special_tokens_to_add.append(WT_PROTEIN_END_TOKEN)
    if MUT_PROTEIN_START_TOKEN not in tokenizer.additional_special_tokens:
        special_tokens_to_add.append(MUT_PROTEIN_START_TOKEN)
    if MUT_PROTEIN_END_TOKEN not in tokenizer.additional_special_tokens:
        special_tokens_to_add.append(MUT_PROTEIN_END_TOKEN)
    
    if special_tokens_to_add:
        print(f"[DEBUG] Adding special tokens: {special_tokens_to_add}")
        tokenizer.add_special_tokens({'additional_special_tokens': special_tokens_to_add})

    wt_protein_start_token_id = tokenizer.convert_tokens_to_ids(WT_PROTEIN_START_TOKEN)
    wt_protein_end_token_id = tokenizer.convert_tokens_to_ids(WT_PROTEIN_END_TOKEN)
    mut_protein_start_token_id = tokenizer.convert_tokens_to_ids(MUT_PROTEIN_START_TOKEN)
    mut_protein_end_token_id = tokenizer.convert_tokens_to_ids(MUT_PROTEIN_END_TOKEN)

    config.wt_protein_start_token_id = wt_protein_start_token_id
    config.wt_protein_end_token_id = wt_protein_end_token_id
    config.mut_protein_start_token_id = mut_protein_start_token_id
    config.mut_protein_end_token_id = mut_protein_end_token_id

    print(f"[DEBUG] WT protein start token ID: {wt_protein_start_token_id}")
    print(f"[DEBUG] WT protein end token ID: {wt_protein_end_token_id}")
    print(f"[DEBUG] Mut protein start token ID: {mut_protein_start_token_id}")
    print(f"[DEBUG] Mut protein end token ID: {mut_protein_end_token_id}")

    # Attach protein model configuration to the main Llama config
    # These would come from model_args in a training script
    config.protein_config = {
        "protein_encoder_name_or_path": model_args.protein_encoder_name_or_path,
        "mm_gated_cross_attention": model_args.mm_gated_cross_attention,
        "mm_resampler": model_args.mm_resampler,
        "num_media_tokens": model_args.num_media_tokens,
        "mm_projector_type": model_args.mm_projector_type,
        "use_mm_proj": getattr(model_args, "use_mm_proj", True),  # Added this
        "mm_protein_select_layer": model_args.mm_protein_select_layer,
        # These dimensions might be dynamically inferred or explicitly set in a full setup
        "esm_hidden_size": getattr(model_args, "esm_hidden_size", 1536), # Example, ideally from ESM config
        "gca_output_dim": getattr(model_args, "gca_output_dim", 512),   # Example
        "resampler_output_dim": getattr(model_args, "resampler_output_dim", config.hidden_size), # Projector might handle the final projection to LLM dim
        "mm_gca_num_heads": getattr(model_args, "mm_gca_num_heads", 8),
        "mm_resampler_num_heads": getattr(model_args, "mm_resampler_num_heads", 8),
    }
    config.tune_mm_mlp_adapter = getattr(model_args, "tune_mm_mlp_adapter", False)


    print(f"Config {config}")

    # Instantiate and set the protein encoder
    print("[DEBUG] Configuring protein encoder.")
    print(f"[DEBUG] mode detected: {getattr(training_args, 'mode', None)}")



    # if getattr(training_args, 'mode', None) == 'inference':
    #     # For inference mode, try cuda:1 first, if not available fall back to cuda:0, then cpu
    #     if torch.cuda.is_available() and torch.cuda.device_count() > 1:
    #         print("[DEBUG] inference mode detected, multiple GPUs available. Using cuda:1 for protein encoder.")
    #         device = torch.device("cuda:1")
    #     elif torch.cuda.is_available():
    #         print("[DEBUG] inference mode detected, only one GPU available. Using cuda:0 for protein encoder.")
    #         device = torch.device("cuda:0")
    #     else:
    #         print("[DEBUG] No CUDA devices available. Using CPU for protein encoder.")
    #         device = torch.device("cpu")
    # else:
    #     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



    # print(f"[DEBUG] Setting up protein encoder on device: {device}")
    
    protein_encoder = ESMProteinEncoder(model_args.protein_encoder_name_or_path)
    # protein_encoder.to(device)  # Explicitly move to correct device

    # Dynamically set esm_hidden_size in config to match ESM3 encoder output
    esm_output_dim = protein_encoder.output_embedding_dim
    print(f"[DEBUG] Detected ESM3 output embedding dim: {esm_output_dim}")
    config.protein_config["esm_hidden_size"] = esm_output_dim

    # Re-instantiate the model with the correct config
    model = LlavaLlamaForCausalLM.from_pretrained(model_args.model_name_or_path, config=config)
    if special_tokens_to_add:
        print("[DEBUG] Resizing token embeddings without mean resizing")
        model.resize_token_embeddings(len(tokenizer))

    model.set_wt_protein_start_token_id(wt_protein_start_token_id)
    model.set_wt_protein_end_token_id(wt_protein_end_token_id)
    model.set_mut_protein_start_token_id(mut_protein_start_token_id)
    model.set_mut_protein_end_token_id(mut_protein_end_token_id)
    # model.set_delta_token_id(delta_token_id)

    # Set the protein encoder on the model
    model.set_protein_encoder(protein_encoder)
    
    # Freeze protein encoder 
    model.freeze_protein_related_modules()

    # If LoRA is not enabled, we are in pretraining mode for the adapters.
    # Unfreeze the adapter modules if tune_mm_mlp_adapter is True.
    if model_args.tune_mm_mlp_adapter:
        print("[Model Utils] Unfreezing GCA/Resampler/Projector")
        model.unfreeze_pretrain_adapters()
    else:
        print("[Model Utils] Freezing GCA/Resampler/Projector")
        model.freeze_pretrain_adapters()









    # Handle loading pretrained adapter weights
    if model_args.pretrained_adapter_path:
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
        else:
            print(f"Warning: No adapters_only.pt file found in {adapter_path_dir}. "
                  "Adapter weights will not be loaded from this path.")
            




    # --- MODE HANDLING: skip training-only logic if in inference mode ---
    # Accepts model_args.mode ('train' or 'inference'), or model_args.is_inference (bool)

    running_mode = getattr(training_args, 'mode', None)
    print(f"[DEBUG] Training mode: {running_mode}")
    if  str(running_mode).lower() == 'inference':
        print("[INFO] Inference mode detected: Skipping training-only parameter freezing and checks.")
        return model, tokenizer

    if str(running_mode).lower() == 'train':
        # Configure parameter freezing based on training stage
        print("[DEBUG] Configuring trainable modules.")
        if not getattr(model_args, 'lora_enable', False):
            print("Pretraining mode: Freezing LLM, LM head")
            

            
            # 1. First freeze the LLM backbone and LM head specifically
            if hasattr(model, 'model'):  # LlamaModel
                print("Freezing LLM backbone parameters.")
                for param in model.model.parameters():
                    param.requires_grad = False
                print("LLM backbone parameters frozen.")
            
            if hasattr(model, 'lm_head'):  # Language modeling head
                print("Freezing LM head parameters.")
                for param in model.lm_head.parameters():
                    param.requires_grad = False
                print("LM head parameters frozen.")

            
            
            # Verify we have some trainable parameters
            num_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
            print(f"[DEBUG] Number of trainable parameters: {num_trainable}")
            if num_trainable == 0:
                raise ValueError(
                    f"No trainable parameters found after configuration! "
                    f"This would cause an optimizer error. "
                )

        if getattr(model_args, 'lora_enable', False):
            print("LoRA finetuning mode")
            
            model = create_lora_model(model, model_args) # This function should handle freezing non-LoRA LLM parts.
            print("LoRA applied.")
            print(f"Lora target modules: {model_args.lora_target_modules}")

            # Print trainable parameters
            if hasattr(model, 'print_trainable_parameters'):
                model.print_trainable_parameters()
            
            # Verify adapter modules are trainable if tune_mm_mlp_adapter=True
            if model_args.tune_mm_mlp_adapter:
                print("\n[FINAL ADAPTER VERIFICATION]")
                adapter_trainable = False
                
                # Check if GCA, Resampler and Projector parameters are still trainable
                for name, param in model.named_parameters():
                    if param.requires_grad and any(adapter in name for adapter in ['mm_gated_cross_attention', 'mm_resampler', 'mm_projector']):
                        adapter_trainable = True
                        print(f"Adapter parameter trainable: {name}")
                
                if not adapter_trainable:
                    print("WARNING: No adapter parameters found to be trainable after LoRA application!")
                    print("Setting adapter parameters as trainable again...")
                    
                    # Force adapter modules to be trainable
                    if hasattr(model, 'base_model') and hasattr(model.base_model, 'model'):
                        base = model.base_model.model
                        if hasattr(base, 'mm_gated_cross_attention') and base.mm_gated_cross_attention is not None:
                            for param in base.mm_gated_cross_attention.parameters():
                                param.requires_grad = True
                        if hasattr(base, 'mm_resampler') and base.mm_resampler is not None:
                            for param in base.mm_resampler.parameters():
                                param.requires_grad = True
                        if hasattr(base, 'mm_projector') and base.mm_projector is not None:
                            for param in base.mm_projector.parameters():
                                param.requires_grad = True
                    
            # Count total trainable parameters
            num_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
            print(f"Number of trainable parameters after LoRA: {num_trainable}")
            if num_trainable == 0:
                raise ValueError("No trainable parameters after applying LoRA. Check LoRA configuration and target modules.")

        else:
            print("Default mode: Trainable parameters determined by initial model state and previous freezing steps (e.g., protein encoder).")

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