from peft import LoraConfig, get_peft_model, TaskType

def create_lora_model(model, lora_config_params):
    """
    Applies LoRA to the given model based on the provided configuration.

    Args:
        model: The base Hugging Face model to adapt.
        lora_config_params (dict): A dictionary containing parameters for LoraConfig.
            Expected keys: r, lora_alpha, target_modules, lora_dropout, bias, task_type.

    Returns:
        PeftModel: The LoRA-adapted model.
    """
    # Ensure all necessary LoRA parameters are present on the ModelArguments object
    required_params = ['lora_r', 'lora_alpha', 'lora_target_modules', 'lora_dropout']
    for param in required_params:
        if not hasattr(lora_config_params, param):
            raise ValueError(f"Missing required LoRA config parameter: {param}")

    # Create the LoRA config from the attributes of the dataclass
    config = LoraConfig(
        r=lora_config_params.lora_r,
        lora_alpha=lora_config_params.lora_alpha,
        target_modules=lora_config_params.lora_target_modules,
        lora_dropout=lora_config_params.lora_dropout,
        bias=getattr(lora_config_params, 'lora_bias', "none"),
        task_type=getattr(lora_config_params, 'task_type', TaskType.CAUSAL_LM)
    )

    # Identify which parameters should be trainable before applying PEFT
    trainable_before_peft = {}
    adapter_names = ['mm_gated_cross_attention', 'mm_resampler', 'mm_projector']
    adapter_trainable = False
    
    for name, param in model.named_parameters():
        if param.requires_grad:
            print(f"Trainable before PEFT: {name}")
            trainable_before_peft[name] = True
            
            # Check if any of these are adapter parameters
            if any(adapter in name for adapter in adapter_names):
                adapter_trainable = True

    # Apply PEFT to get the LoRA model
    lora_model = get_peft_model(model, config)
    
    print("LoRA adapted model created with the following PEFT config:")
    print(config)
    lora_model.print_trainable_parameters()
    
    # IMPORTANT: PEFT/LoRA will reset requires_grad for all parameters
    # We need to restore requires_grad=True for the adapter parameters if they were trainable before
    if adapter_trainable:
        print("\n[RESTORING ADAPTER TRAINABLE STATE]")
        for name, param in lora_model.named_parameters():
            # Skip LoRA parameters which are already trainable
            if 'lora_' in name:
                continue
                
            # Check if this parameter was trainable before PEFT
            # Use any matching substring since PEFT prefixes with base_model.model
            was_trainable = any(orig_name in name for orig_name in trainable_before_peft.keys()
                              if any(adapter in orig_name for adapter in adapter_names))
                
            if was_trainable:
                print(f"Restoring trainable state for adapter param: {name}")
                param.requires_grad = True
    
    # Verify which parameters are trainable after our adjustments
    trainable_params = []
    for name, param in lora_model.named_parameters():
        if param.requires_grad:
            trainable_params.append(name)
    
    print(f"\nFinal trainable parameters summary:")
    print(f"Number of trainable parameters: {len(trainable_params)}")
    print(f"Sample trainable parameters: {trainable_params[:5] if trainable_params else 'None'}")
    
    return lora_model
