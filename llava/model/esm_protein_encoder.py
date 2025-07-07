import torch
import torch.nn as nn

# ESMProteinEncoder using ESM3 SDK from fair-esm
from esm.models.esm3 import ESM3
from esm.sdk.api import ESMProtein, SamplingConfig

class ESMProteinEncoder(nn.Module):
    def __init__(self, model_name_or_path="esm3_sm_open_v1", **kwargs): # kwargs for compatibility, may not be used
        super().__init__()
        self.model_name_or_path = model_name_or_path
        self.client = None
        self._device = torch.device("cpu") # Default device, updated by to()
        self.output_embedding_dim = None
        
        # ESM3 SDK doesn't use cache_dir or model_kwargs in the same way as HuggingFace AutoModel
        # These are kept in signature for compatibility with how it was called previously, but might be ignored.
        if kwargs:
            print(f"ESMProteinEncoder (ESM3 SDK) received unused kwargs: {kwargs}")

        # By calling _init_client here, we ensure that output_embedding_dim is available after instantiation.
        self._init_client() # This will initialize on CPU by default. .to(device) will re-init on GPU if called later.

    def _init_client(self):
        if self.client is None:
            if self._device is None:
                # This case should ideally not be hit if .to(device) is called after instantiation.
                self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                print(f"ESMProteinEncoder client device not set, defaulting to {self._device}. Call .to(device) for specific placement.")
            elif isinstance(self._device, str):
                self._device = torch.device(self._device)

            print(f"Loading ESM3 client: {self.model_name_or_path} to device: {self._device}")
            try:
                # ESM3.from_pretrained expects device argument as torch.device
                self.client = ESM3.from_pretrained(self.model_name_or_path, device=self._device)
                self.client.eval()  # Set to evaluation mode

                # Determine output_embedding_dim with a dummy sequence
                # Using a short, valid protein sequence.
                dummy_protein_seq = "M" 
                dummy_protein = ESMProtein(sequence=dummy_protein_seq)
                dummy_tensor = self.client.encode(dummy_protein)
                # Ensure dummy_tensor is on the correct device if encode doesn't guarantee it.
                # However, client itself is on self._device, so its outputs should be too.
                dummy_output = self.client.forward_and_sample(
                    dummy_tensor, 
                    SamplingConfig(return_per_residue_embeddings=True)
                )
                self.output_embedding_dim = dummy_output.per_residue_embedding.shape[-1]
                print(f"ESM3 client loaded successfully. Deduced output embedding dimension: {self.output_embedding_dim}")
            except Exception as e:
                print(f"Error loading ESM3 model '{self.model_name_or_path}' using ESM SDK: {e}")
                raise

    def forward(self, protein_sequences_list, select_layer=-1, average_embeddings=False, max_length=None):
        if self.client is None or self.output_embedding_dim is None:
            # If .to(device) was not called, this ensures client initialization on the current _device
            self._init_client() 
            if self.client is None: # Still None after trying to init
                 raise RuntimeError("ESMProteinEncoder client is not initialized. Call .to(device) on the encoder instance before using forward pass.")


        if not protein_sequences_list:
            return torch.empty(0, device=self._device)

        if select_layer != -1 and select_layer is not None: # Allow None to also mean default
            print(f"Warning: `select_layer` argument ({select_layer}) is provided but ESMProteinEncoder using ESM3 SDK "
                  "currently uses `per_residue_embedding` directly from SamplingConfig. The `select_layer` argument is ignored.")

        embeddings_list = []
        actual_lengths = []

        for i, seq_str in enumerate(protein_sequences_list):
            if not seq_str:
                # Handle empty sequences - this will likely cause problems if not filtered upstream
                # For now, raising an error is safer.
                raise ValueError(f"Empty protein sequence provided at index {i} in the batch.")
            
            protein = ESMProtein(sequence=seq_str)
            # protein_tensor is placed on the device the client was initialized with.
            protein_tensor = self.client.encode(protein)
            
            sampling_config = SamplingConfig(return_per_residue_embeddings=True)
            output = self.client.forward_and_sample(protein_tensor, sampling_config)
            
            # output.per_residue_embedding shape is (1, seq_len, embed_dim)
            current_embeddings = output.per_residue_embedding.squeeze(0)  # Shape: (seq_len, embed_dim)
            embeddings_list.append(current_embeddings)
            actual_lengths.append(current_embeddings.shape[0])

        if not embeddings_list:
            return torch.empty(0, device=self._device) # Should not happen if we raise error on empty seq_str

        # Determine padding length
        # If max_length is provided, sequences are truncated or padded to this length.
        # Otherwise, pad to the length of the longest sequence in the batch.
        if max_length is not None:
            padding_target_len = max_length
        else:
            padding_target_len = max(actual_lengths) if actual_lengths else 0
        
        if padding_target_len == 0 and protein_sequences_list: # Edge case: all sequences were empty (if not raising error) or max_length=0
            # Return a batch of zero-length sequences if that's meaningful, or handle error.
            # Assuming self.output_embedding_dim is known.
             return torch.empty((len(protein_sequences_list), 0, self.output_embedding_dim), device=self._device)


        padded_embeddings_list = []
        for emb in embeddings_list:
            seq_len = emb.shape[0]
            if seq_len > padding_target_len:  # Truncate
                emb = emb[:padding_target_len, :]
            elif seq_len < padding_target_len:  # Pad
                padding_size = padding_target_len - seq_len
                padding = torch.zeros((padding_size, self.output_embedding_dim), device=emb.device) # Use emb.device
                emb = torch.cat([emb, padding], dim=0)
            padded_embeddings_list.append(emb)
        
        batch_embeddings = torch.stack(padded_embeddings_list, dim=0) # (batch_size, padding_target_len, hidden_dim)

        if average_embeddings:
            # Create a mask based on actual lengths for correct averaging
            attention_mask = torch.zeros(batch_embeddings.shape[0], padding_target_len, device=self._device)
            for i, length in enumerate(actual_lengths):
                # Consider truncation: actual length used for mask is min(original_length, padding_target_len)
                effective_len = min(length, padding_target_len)
                attention_mask[i, :effective_len] = 1
            
            masked_sum = torch.sum(batch_embeddings * attention_mask.unsqueeze(-1), dim=1)
            num_tokens = attention_mask.sum(dim=1, keepdim=True).clamp(min=1) # Avoid division by zero
            averaged_embeddings = masked_sum / num_tokens
            return averaged_embeddings
        else:
            return batch_embeddings

    def to(self, device):
        super().to(device) # Moves registered parameters/buffers of the nn.Module
        self._device = device # Store the target device
        
        # Re-initialize client on the new device.
        # ESM3.from_pretrained takes a device argument.
        # If client exists and is on a different device, it needs to be reloaded or moved.
        # Simplest is to re-load, assuming ESM3 objects are not massive stateful nn.Modules themselves that .to() would handle.
        print(f"ESMProteinEncoder: Setting device to {self._device} and re-initializing ESM3 client.")
        self.client = None # Force re-initialization on the new device
        self._init_client() # This will load ESM3.from_pretrained(..., device=self._device)
        return self

    def eval(self):
        super().eval()
        if self.client:
            self.client.eval()
        return self