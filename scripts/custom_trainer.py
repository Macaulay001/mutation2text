import os
import numpy as np
import torch
from typing import Optional
from adapter_trainer import AdapterTrainer
from torch.serialization import add_safe_globals

# Add numpy array reconstruction to safe globals
add_safe_globals(['numpy.core.multiarray._reconstruct'])

class CustomRNGTrainer(AdapterTrainer):
    def _load_rng_state(self, resume_from_checkpoint):
        """Override to handle RNG state loading with weights_only=False."""
        if resume_from_checkpoint is None:
            return

        if not os.path.isdir(resume_from_checkpoint):
            return

        # Get RNG file paths
        rng_files = [
            os.path.join(resume_from_checkpoint, f"rng_state_{process_index}.pth")
            for process_index in range(2)  # Assuming 2 GPUs
        ]

        if not os.path.exists(rng_files[0]):
            self.print(f"Didn't find RNG states at {rng_files[0]}")
            return

        try:
            # Load RNG state with weights_only=False to handle numpy arrays
            checkpoint_rng_state = torch.load(rng_files[self.process_index], weights_only=False)
            random_state = checkpoint_rng_state["python"]
            numpy_state = checkpoint_rng_state["numpy"]
            torch_state = checkpoint_rng_state["torch"]
            torch_cuda_state = checkpoint_rng_state["torch_cuda"]

            # Set RNG states
            import random
            random.setstate(random_state)
            np.random.set_state(numpy_state)
            torch.set_rng_state(torch_state)
            torch.cuda.set_rng_state(torch_cuda_state)
            
            if torch.cuda.is_available():
                if self.process_index <= torch.cuda.device_count():
                    torch.cuda.set_rng_state(torch_cuda_state)

            self.print(f"Loaded RNG state from {rng_files[self.process_index]}")
            
        except Exception as e:
            print(f"Failed to load RNG state: {e}")
