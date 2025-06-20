import os
import json
import torch
import dataclasses
from transformers import Trainer
from typing import Any, Dict, Optional, Tuple, List, Union
from transformers.training_args import TrainingArguments
from peft import PeftModel

class TrainingArgsEncoder(json.JSONEncoder):
    """Custom JSON encoder that handles special types in training arguments."""
    def default(self, obj: Any) -> Any:
        if dataclasses.is_dataclass(obj):
            return dataclasses.asdict(obj)
        if isinstance(obj, torch.dtype):
            return str(obj)
        if isinstance(obj, (set, frozenset)):
            return list(obj)
        # Handle any object with a get_state method (like PartialState)
        if hasattr(obj, 'get_state'):
            return obj.get_state()
        # For any other complex objects, try to convert to dict if possible
        if hasattr(obj, '__dict__'):
            return obj.__dict__
        return str(obj)  # Fallback to string representation

from accelerate.utils import wait_for_everyone
# HF Trainer already has self.accelerator, so you don't need to import Accelerator again.

class AdapterTrainer(Trainer):
    """
    A custom trainer to handle saving adapter weights and an overridden
    prediction step for models with custom inputs.
    """
    def __init__(self, *args, model_args=None, **kwargs):
        super().__init__(*args, **kwargs)
        if model_args is None:
            raise ValueError("AdapterTrainer requires model_args to be passed.")
        self.model_args = model_args
        # Validate trainable parameters on initialization
        self._validate_trainable_parameters()
        
    def _validate_trainable_parameters(self):
        """Validate that the model has trainable parameters before training"""
        model = self.accelerator.unwrap_model(self.model)
        trainable_params = [name for name, param in model.named_parameters() if param.requires_grad]
        
        print(f"\n======= TRAINABLE PARAMETERS VALIDATION =======")
        print(f"Total number of trainable parameters: {len(trainable_params)}")
        if trainable_params:
            print(f"Sample trainable parameter names:")
            for name in trainable_params[:10]:
                print(f"  - {name}")
            
            # Check explicitly for adapter parameters
            adapter_params = [name for name in trainable_params if any(adapter in name for adapter in 
                             ['mm_gated_cross_attention', 'mm_resampler', 'mm_projector'])]
            if adapter_params:
                print(f"\nAdapter parameters that are trainable: {len(adapter_params)}")
                for name in adapter_params[:5]:
                    print(f"  - {name}")
            else:
                print("\nNo adapter parameters (GCA/Resampler/Projector) found to be trainable.")
                
            # Check for LoRA parameters
            lora_params = [name for name in trainable_params if 'lora_' in name]
            if lora_params:
                print(f"\nLoRA parameters that are trainable: {len(lora_params)}")
                for name in lora_params[:5]:
                    print(f"  - {name}")
            else:
                print("\nNo LoRA parameters found to be trainable. This is unusual.")
        else:
            raise ValueError("No trainable parameters found in the model. Training would fail with gradient errors.")
        print(f"===============================================\n")
        
    def training_step(self, model, inputs):
        """Override training_step to add a manual gradient check."""
        model.train()
        inputs = self._prepare_inputs(inputs)

        with self.compute_loss_context_manager():
            loss = self.compute_loss(model, inputs)
        
        # Standard backward pass
        if self.args.n_gpu > 1:
            loss = loss.mean()
        self.accelerator.backward(loss)

        return loss.detach() / self.args.gradient_accumulation_steps

    def save_model(self, output_dir: Optional[str] = None, _internal_call: bool = False):
        """Saves the full model state and also saves adapter weights separately."""
        # First, save the full model using the parent class method.
        # This saves the base model, tokenizer, config, etc. as part of a checkpoint.
        super().save_model(output_dir, _internal_call)

        # Determine the correct output directory for adapters
        final_output_dir = output_dir if output_dir is not None else self.args.output_dir

        # Now, save the adapter weights separately.
        self._save_adapters_only(final_output_dir)

    def prediction_step(
        self,
        model: torch.nn.Module,
        inputs: Dict[str, Any],
        prediction_loss_only: bool,
        ignore_keys: Optional[List[str]] = None,
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
        """
        Overrides the default prediction step to handle custom model inputs during generation.
        """
        # Separate the standard inputs from our custom ones.
        # `generate` expects `input_ids` and `attention_mask` as positional args.
        standard_inputs = {
            "input_ids": inputs["input_ids"],
            "attention_mask": inputs["attention_mask"],
        }
        
        # Prepare the custom kwargs that our model's forward pass expects.
        custom_kwargs = {
            "wild_type_sequences": inputs.get("wild_type_sequences"),
            "mutation_sequences": inputs.get("mutation_sequences"),
        }

        # Run the standard prediction step to get the loss.
        loss, _, _ = super().prediction_step(model, inputs, prediction_loss_only=True, ignore_keys=ignore_keys)

        # If we are only calculating loss, we are done.
        if prediction_loss_only:
            return (loss, None, None)

        # If predict_with_generate is enabled, run generation.
        # We must manually pass our custom kwargs here.
        generated_tokens = self.model.generate(
            **standard_inputs,
            **custom_kwargs,
            max_new_tokens=128  # You can make this configurable
        )
        
        # The rest of this logic is standard from the Trainer to handle labels.
        labels = inputs.get("labels")
        if labels is not None and len(labels.shape) > 1:
             # Ensure generated tokens are padded to the same length as labels for metrics
            if generated_tokens.shape[-1] < labels.shape[-1]:
                generated_tokens = self._pad_tensors_to_max_len(generated_tokens, labels)
        
        return (loss, generated_tokens, labels)

    def _pad_tensors_to_max_len(self, tensor, target_tensor):
        """Pads a tensor to the length of a target tensor."""
        # (Your existing _pad_tensors_to_max_len implementation if you have one,
        # otherwise, a simple implementation is needed if it's called from prediction_step)
        target_len = target_tensor.shape[-1]
        current_len = tensor.shape[-1]
        if target_len > current_len:
            padding_size = target_len - current_len
            return torch.nn.functional.pad(tensor, (0, padding_size))
        return tensor

    def _save_adapters_only(self, output_dir: str):
        """
        Saves only the trainable adapter parameters to a file named 'adapters_only.pt'.
        This includes GCA, Resampler, Projector, and potentially LoRA layers if enabled.
        """
        # 1️⃣  ALWAYS unwrap first
        peft_model = self.accelerator.unwrap_model(self.model)

        # 2️⃣  Save LoRA adapter (weights + adapter_config.json)
        if isinstance(peft_model, PeftModel) and self.is_world_process_zero():
            peft_model.save_pretrained(output_dir)     # writes adapter_config.json
            print(f"[AdapterTrainer] PEFT adapter saved to {output_dir}")

        # 3️⃣  (optional) save your GCA / Resampler / Projector
        to_save = {}
        for name in ("mm_gated_cross_attention", "mm_resampler", "mm_projector"):
            module = getattr(peft_model, name, None)
            if module is not None:
                to_save[name] = module.state_dict()
        if to_save and self.is_world_process_zero():
            torch.save(to_save, os.path.join(output_dir, "adapters_only.pt"))

        # 4️⃣  Everyone else waits for rank-0
        wait_for_everyone()

        # 5️⃣  Config & training args (rank-0 only is fine)
        if self.is_world_process_zero():
            peft_model.base_model.config.save_pretrained(os.path.join(output_dir, "base_model"))
            with open(os.path.join(output_dir, "training_args.json"), "w") as f:
                json.dump(self.args.to_dict(), f, indent=2)
        #save config.json
        if self.is_world_process_zero():
            with open(os.path.join(output_dir, "config.json"), "w") as f:
                json.dump(self.model.config.to_dict(), f, indent=2, cls=TrainingArgsEncoder)
    
