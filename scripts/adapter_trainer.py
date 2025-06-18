import os
import json
import torch
import dataclasses
from transformers import Trainer
from typing import Any
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
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
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
        """Override training_step to add gradient debugging if needed"""
        try:
            return super().training_step(model, inputs)
        except RuntimeError as e:
            if "does not require grad and does not have a grad_fn" in str(e):
                print("\n\n===== GRADIENT ERROR DIAGNOSTICS =====")
                # Analyze the model's parameters
                no_grad_params = []
                trainable_params = []
                
                for name, param in model.named_parameters():
                    if param.requires_grad:
                        trainable_params.append(name)
                    else:
                        no_grad_params.append(name)
                
                print(f"Total parameters: {len(trainable_params) + len(no_grad_params)}")
                print(f"Trainable parameters: {len(trainable_params)}")
                print(f"Non-trainable parameters: {len(no_grad_params)}")
                
                print("\nFirst few trainable parameters:")
                for name in trainable_params[:5]:
                    print(f"  - {name}")
                    
                print("\nThis error usually indicates a mismatch between trainable parameters and the computation graph.")
                print("The model likely has disconnected components or tensors that don't require gradients.")
                print("Please check that all modules involved in the forward pass have their parameters set correctly.")
                print("===================================\n\n")
            # Re-raise the exception
            raise

    def save_model(self, output_dir=None, **_):
        output_dir = output_dir or self.args.output_dir
        os.makedirs(output_dir, exist_ok=True)

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
    
