import torch
import transformers

def setup_optimizer_and_scheduler(model, training_args, num_training_steps_per_epoch=None, num_train_epochs=None):
    """
    Sets up the optimizer and learning rate scheduler.
    training_args should be an object with attributes like learning_rate, weight_decay, 
    lr_scheduler_type, warmup_ratio, warmup_steps etc.
    """
    # Collect parameters that require gradients
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    
    if not trainable_params:
        print("Warning: No trainable parameters found in the model. Optimizer will not be created.")
        return None, None

    optimizer = torch.optim.AdamW(
        trainable_params,
        lr=training_args.learning_rate,
        weight_decay=training_args.weight_decay
    )

    if num_training_steps_per_epoch is None or num_train_epochs is None:
        if hasattr(training_args, 'max_steps') and training_args.max_steps > 0:
            num_training_steps = training_args.max_steps
        else:
            # This is a fallback, ideally these are provided or derivable from dataset and epochs
            raise ValueError("num_training_steps_per_epoch and num_train_epochs, or max_steps must be provided to setup scheduler.")
    else:
        num_training_steps = num_training_steps_per_epoch * num_train_epochs

    # Determine warmup steps
    if training_args.warmup_steps > 0:
        num_warmup_steps = training_args.warmup_steps
    elif training_args.warmup_ratio > 0:
        num_warmup_steps = int(num_training_steps * training_args.warmup_ratio)
    else:
        num_warmup_steps = 0

    scheduler = transformers.get_scheduler(
        name=training_args.lr_scheduler_type, # e.g., "linear", "cosine"
        optimizer=optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps
    )
    
    print(f"Optimizer: AdamW with lr={training_args.learning_rate}, weight_decay={training_args.weight_decay}")
    print(f"Scheduler: {training_args.lr_scheduler_type} with {num_warmup_steps} warmup steps over {num_training_steps} total steps.")
    
    return optimizer, scheduler

def enable_gradient_checkpointing(model, model_args):
    """
    Enables gradient checkpointing for the model if specified in model_args.
    """
    if getattr(model_args, 'gradient_checkpointing', False):
        if hasattr(model, 'gradient_checkpointing_enable'):
            model.gradient_checkpointing_enable()
            # For some models, specific parts might need it enabled, e.g. model.model.gradient_checkpointing_enable()
            print("Gradient checkpointing enabled.")
        else:
            print("Warning: Model does not have a `gradient_checkpointing_enable` method. Gradient checkpointing not enabled.")

# Placeholder for other training utilities:
# - Mixed precision setup (though often handled by Trainer or DeepSpeed)
# - Logging utilities (e.g., custom metrics)
# - Early stopping logic (if not using Trainer)

# Example: How to set up mixed precision with PyTorch AMP (Automatic Mixed Precision)
# if training_args.bf16 or training_args.fp16:
#     scaler = torch.cuda.amp.GradScaler(enabled=(training_args.fp16 or training_args.bf16))
# else:
#     scaler = None

# In training loop:
# with torch.cuda.amp.autocast(enabled=(training_args.bf16 or training_args.fp16)):
#     outputs = model(...)
#     loss = outputs.loss
# if scaler:
#     scaler.scale(loss).backward()
#     scaler.step(optimizer)
#     scaler.update()
# else:
#     loss.backward()
#     optimizer.step() 