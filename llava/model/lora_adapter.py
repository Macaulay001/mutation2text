from peft import LoraConfig, get_peft_model, TaskType

def create_lora_model(model, lora_config_params, tune_mm_adapters=False):
    """
    Applies LoRA to the given model based on the provided configuration.

    Args:
        model: The base Hugging Face model to adapt.
        lora_config_params (dict): A dictionary containing parameters for LoraConfig.
        tune_mm_adapters (bool): If True, the multimodal adapters (GCA, Resampler, Projector)
                                 will also be unfrozen and trained alongside LoRA layers.

    Returns:
        PeftModel: The LoRA-adapted model.
    """
    # Create the LoRA config from the provided parameters
    required_params = ['r', 'lora_alpha', 'target_modules', 'lora_dropout']
    for param in required_params:
        if param not in lora_config_params:
            raise ValueError(f"Missing required LoRA config parameter: {param}")

    config = LoraConfig(
        r=lora_config_params['r'],
        lora_alpha=lora_config_params['lora_alpha'],
        target_modules=lora_config_params['target_modules'],
        lora_dropout=lora_config_params['lora_dropout'],
        bias=lora_config_params.get('bias', "none"),
        task_type=lora_config_params.get('task_type', TaskType.CAUSAL_LM)
    )

    # Apply PEFT to get the LoRA model. This freezes all original model weights.
    lora_model = get_peft_model(model, config)
    
    print("\nLoRA adapted model created. Initial trainable parameters (LoRA layers only):")
    lora_model.print_trainable_parameters()
    
    # If requested, unfreeze the multimodal adapters to allow for joint training.
    if tune_mm_adapters:
        print("\n[INFO] Unfreezing multimodal adapters for joint training...")
        adapter_names = ['mm_gated_cross_attention', 'mm_resampler', 'mm_projector']
        for name, param in lora_model.named_parameters():
            if any(adapter_name in name for adapter_name in adapter_names):
                param.requires_grad = True
    
        print("\nFinal trainable parameters summary (LoRA + MM Adapters):")
        lora_model.print_trainable_parameters()

    return lora_model
