import torch
import torch.nn as nn
import math
from transformers import LlamaForCausalLM, LlamaConfig
from transformers.modeling_outputs import CausalLMOutputWithPast
import typing
import time
from torch.utils.checkpoint import checkpoint
from einops import rearrange, repeat
import einops
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence

def exists(val):
    return val is not None

class CustomLayerNorm(nn.Module):
    """
    A custom LayerNorm that performs computation in float32 for stability,
    while keeping the model in bfloat16 for performance.
    """
    def __init__(self, normalized_shape, eps=1e-5):
        super().__init__()
        if isinstance(normalized_shape, int):
            normalized_shape = (normalized_shape,)
        self.normalized_shape = torch.Size(normalized_shape)
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(self.normalized_shape))
        self.bias = nn.Parameter(torch.zeros(self.normalized_shape))

    def forward(self, x):
        orig_dtype = x.dtype
        weight = self.weight.to(torch.float32)
        bias = self.bias.to(torch.float32)
        x_float = x.to(torch.float32)
        out = F.layer_norm(x_float, self.normalized_shape, weight, bias, self.eps)
        return out.to(orig_dtype)

class FeedForward(nn.Module):
    def __init__(self, dim, mult=4):
        super().__init__()
        inner_dim = int(dim * mult)
        
        self.norm = CustomLayerNorm(dim)
        self.fc1 = nn.Linear(dim, inner_dim, bias=False)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(inner_dim, dim, bias=False)

    def forward(self, x):
        x = self.norm(x)
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        return x

# from .esm_protein_encoder import ESMProteinEncoder # Assuming this will be in the same directory

# Placeholder for DELTA_TOKEN and IGNORE_INDEX, usually defined in training scripts or data utils
# DELTA_TOKEN_ID_PLACEHOLDER = -1  # This should be dynamically set from tokenizer
IGNORE_INDEX = -100

class GatedCrossAttention(nn.Module):
    def __init__(self, query_dim, key_value_dim, output_dim, num_heads=8, dim_head=64, ff_mult=4):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim_head
        inner_dim = dim_head * num_heads
        
        # Main attention components for cross attention (mut -> wt)
        self.norm_query = CustomLayerNorm(query_dim)
        self.norm_key_value = CustomLayerNorm(key_value_dim)
        self.to_q_cross = nn.Linear(query_dim, inner_dim, bias=False)
        self.to_kv_cross = nn.Linear(key_value_dim, inner_dim * 2, bias=False)
        
        # Components for self attention (mut -> mut)
        self.to_qkv_self = nn.Linear(query_dim, inner_dim * 3, bias=False)
        
        # Attention-based gating mechanism
        self.gate_norm = CustomLayerNorm(inner_dim)
        self.gate_attention = nn.MultiheadAttention(
            embed_dim=inner_dim,
            num_heads=num_heads,
            batch_first=True,
            bias=False
        )
        
        # Learned mixing weights projector
        self.to_mixing_weights = nn.Sequential(
            nn.Linear(inner_dim, 2),
            nn.Softmax(dim=-1)
        )
        
        # Final projection to output dimension
        self.to_out = nn.Linear(inner_dim, output_dim, bias=False)
        
        # Feedforward network
        self.ff = FeedForward(dim=output_dim, mult=ff_mult)
        self.ff_gate = nn.Parameter(torch.ones(1))

    def forward(self, query_feats, kv_feats_wt, kv_feats_mut):
        # Layer normalization
        query_feats_norm = self.norm_query(query_feats)  # mutation features
        kv_feats_wt_norm = self.norm_key_value(kv_feats_wt)  # wild-type features
        
        # 1. Cross Attention (mutation query -> wild-type k/v)
        q_cross = self.to_q_cross(query_feats_norm)
        k_wt, v_wt = self.to_kv_cross(kv_feats_wt_norm).chunk(2, dim=-1)
        
        # Split heads for cross attention
        q_cross = rearrange(q_cross, 'b n (h d) -> b h n d', h=self.num_heads)
        k_wt = rearrange(k_wt, 'b n (h d) -> b h n d', h=self.num_heads)
        v_wt = rearrange(v_wt, 'b n (h d) -> b h n d', h=self.num_heads)
        
        # Cross attention computation
        scale = q_cross.shape[-1] ** -0.5
        sim_wt = torch.einsum('b h i d, b h j d -> b h i j', q_cross * scale, k_wt)
        attn_wt = F.softmax(sim_wt, dim=-1)
        cross_out = torch.einsum('b h i j, b h j d -> b h i d', attn_wt, v_wt)
        cross_out = rearrange(cross_out, 'b h n d -> b n (h d)')

        if kv_feats_mut is not None:
            # 2. Self Attention (mutation -> mutation)
            qkv_self = self.to_qkv_self(query_feats_norm)
            q_self, k_self, v_self = qkv_self.chunk(3, dim=-1)
            
            # Split heads for self attention
            q_self = rearrange(q_self, 'b n (h d) -> b h n d', h=self.num_heads)
            k_self = rearrange(k_self, 'b n (h d) -> b h n d', h=self.num_heads)
            v_self = rearrange(v_self, 'b n (h d) -> b h n d', h=self.num_heads)
            
            # Self attention computation
            sim_self = torch.einsum('b h i d, b h j d -> b h i j', q_self * scale, k_self)
            attn_self = F.softmax(sim_self, dim=-1)
            self_out = torch.einsum('b h i j, b h j d -> b h i d', attn_self, v_self)
            self_out = rearrange(self_out, 'b h n d -> b n (h d)')
            
            # Apply attention-based gating
            # 1. Normalize both attention outputs
            self_out_norm = self.gate_norm(self_out)
            cross_out_norm = self.gate_norm(cross_out)
            
            # 2. Use attention mechanism to compute interaction between self and cross attention
            gate_context, _ = self.gate_attention(
                query=self_out_norm,
                key=cross_out_norm,
                value=cross_out_norm
            )
            
            # 3. Generate dynamic mixing weights based on the attention context
            batch_size, seq_len, _ = gate_context.shape
            mixing_weights = self.to_mixing_weights(gate_context)  # [batch, seq_len, 2]
            
            # 4. Mix self and cross attention outputs using learned weights
            mixed_output = (
                mixing_weights[..., 0:1] * self_out +
                mixing_weights[..., 1:2] * cross_out
            )
            
            # Project to output dimension
            delta = self.to_out(mixed_output)
            
            # Apply feedforward
            if hasattr(self, 'ff'):
                ff_out = self.ff(delta)
                delta = delta + self.ff_gate * ff_out
            
            return delta
        else:
            # If no mutation features, just return processed cross-attention features
            out = self.to_out(cross_out)
            if hasattr(self, 'ff'):
                ff_out = self.ff(out)
                out = out + self.ff_gate * ff_out
            return out
        

class PerceiverResampler(nn.Module):
    def __init__(self, input_dim, output_dim, num_latents, num_output_tokens, num_layers=2, 
                 num_heads=8, dim_head=64, ff_mult=4):
        super().__init__()
        self.num_output_tokens = num_output_tokens
        self.num_heads = num_heads
        self.dim_head = dim_head
        self.input_dim = input_dim
        self.output_dim = output_dim
        inner_dim = dim_head * num_heads
        
        # Learnable latent vectors that will be queried against the input sequence
        self.latents = nn.Parameter(torch.randn(num_output_tokens, input_dim))
        
        # Position embeddings for the latents only - input sequence will use relative attention
        self.latent_pos_emb = nn.Parameter(torch.randn(1, num_output_tokens, input_dim))
        
        # Layer stack
        self.layers = nn.ModuleList([])
        for layer_idx in range(num_layers):
            # Create attention block
            attn_block = nn.ModuleDict({
                'norm_media': CustomLayerNorm(input_dim),
                'norm_latents': CustomLayerNorm(input_dim),
                'to_q': nn.Linear(input_dim, inner_dim, bias=False),
                'to_kv': nn.Linear(input_dim, inner_dim * 2, bias=False),
                'to_out': nn.Linear(inner_dim, input_dim, bias=False)
            })
            
            self.layers.append(nn.ModuleList([
                attn_block,
                FeedForward(dim=input_dim, mult=ff_mult)
            ]))
        
        # Final normalization and projection
        self.norm = CustomLayerNorm(input_dim)
        if input_dim != output_dim:
            self.proj_out = nn.Linear(input_dim, output_dim)
        else:
            self.proj_out = nn.Identity()
    
    def forward(self, x):
        """
        x: (batch_size, seq_len, input_dim) - Features from GCA
        Returns: (batch_size, num_output_tokens, output_dim)
        """
        batch_size = x.shape[0]
        
        # Repeat latents for batch
        latents = repeat(self.latents, 'n d -> b n d', b=batch_size)
        
        # Add positional embeddings to latents
        latents = latents + self.latent_pos_emb
        
        # Main Perceiver attention logic
        for attn_block, ff in self.layers:
            def _attn_block(x, latents):
                # Layer normalization
                norm_media = attn_block['norm_media']
                norm_latents = attn_block['norm_latents']
                
                x_norm = norm_media(x)
                latents_norm = norm_latents(latents)
                
                # Project latents to queries
                q = attn_block['to_q'](latents_norm)
                
                # Project input sequence to keys/values
                k, v = attn_block['to_kv'](x_norm).chunk(2, dim=-1)
                
                # Reshape for multi-head attention
                q = rearrange(q, 'b n (h d) -> b h n d', h=self.num_heads)
                k = rearrange(k, 'b n (h d) -> b h n d', h=self.num_heads)
                v = rearrange(v, 'b n (h d) -> b h n d', h=self.num_heads)
                
                # Attention mechanism
                scale = self.dim_head ** -0.5
                sim = torch.einsum('b h i d, b h j d -> b h i j', q, k) * scale
                attn = sim.softmax(dim=-1)
                
                # Aggregate values
                out = torch.einsum('b h i j, b h j d -> b h i d', attn, v)
                out = rearrange(out, 'b h n d -> b n (h d)')
                
                # Final projection
                out = attn_block['to_out'](out)
                return out

            # Apply attention block with residual connection
            latents = latents + _attn_block(x, latents)
            # Apply feedforward with residual connection
            latents = latents + ff(latents)
        
        # Final normalization
        latents = self.norm(latents)
        
        # Project to output dimension
        output = self.proj_out(latents)
        
        return output
        

class LlavaLlamaConfig(LlamaConfig):
    model_type = "llava_llama"


class LlavaLlamaForCausalLM(LlamaForCausalLM):
    config_class = LlavaLlamaConfig

    def __init__(self, config):
        super().__init__(config)

        self.protein_config = getattr(config, 'protein_config', None)
        #print("[DEBUG] Initializing LlavaLlamaForCausalLM with protein_config:", self.protein_config)

        if self.protein_config is not None:
            # Ensure required parameters are present with defaults
            required_params = {
                "esm_hidden_size": 1280,
                "gca_output_dim": 512,
                "resampler_output_dim": 4096,  # Same as LLM hidden size
                "num_media_tokens": 128,
                "mm_gca_num_heads": 8,
                "mm_resampler_num_heads": 8
            }
            
            for param, default in required_params.items():
                if param not in self.protein_config:
                    #print(f"[WARNING] {param} not found in protein_config. Using default: {default}")
                    self.protein_config[param] = default

            # Initialize GCA
            if self.protein_config.get("mm_gated_cross_attention", False):
                #print("[DEBUG] Initializing GCA module with parameters:")
                #print(f"  esm_hidden_size: {self.protein_config['esm_hidden_size']}")
                #print(f"  gca_output_dim: {self.protein_config['gca_output_dim']}")
                #print(f"  num_heads: {self.protein_config['mm_gca_num_heads']}")
                
                self.mm_gated_cross_attention = GatedCrossAttention(
                    query_dim=self.protein_config['esm_hidden_size'],
                    key_value_dim=self.protein_config['esm_hidden_size'],
                    output_dim=self.protein_config['gca_output_dim'],
                    num_heads=self.protein_config['mm_gca_num_heads']
                )
                assert self.mm_gated_cross_attention is not None, "GCA module failed to initialize"
                #print("[DEBUG] GCA module initialized successfully")

            # Initialize Resampler
            if self.protein_config.get("mm_use_resampler_gca", False):
                #print("[DEBUG] Initializing Resampler module with parameters:")
                #print(f"  gca_output_dim: {self.protein_config['gca_output_dim']}")
                #print(f"  resampler_output_dim: {self.protein_config['resampler_output_dim']}")
                #print(f"  num_media_tokens: {self.protein_config['num_media_tokens']}")
                #print(f"  num_heads: {self.protein_config['mm_resampler_num_heads']}")
                
                self.mm_resampler = PerceiverResampler(
                    input_dim=self.protein_config['gca_output_dim'],
                    output_dim=self.protein_config['resampler_output_dim'],
                    num_latents=self.protein_config['num_media_tokens'],
                    num_output_tokens=self.protein_config['num_media_tokens'],
                    num_heads=self.protein_config['mm_resampler_num_heads']
                )
                assert self.mm_resampler is not None, "Resampler module failed to initialize"
                
                # # Verify dimensions match between modules
                # if hasattr(self, 'mm_gated_cross_attention'):
                #     assert self.mm_gated_cross_attention.output_dim == self.mm_resampler.input_dim, \
                #         "GCA output dimension must match Resampler input dimension"
                # #print("[DEBUG] Resampler module initialized successfully")

            # Initialize Projector
            if self.protein_config.get("use_mm_proj", False):
                #print("[DEBUG] Initializing Projector module")
                projector_type = self.protein_config.get("mm_projector_type", "mlp2x_gelu")
                resampler_output_dim = self.protein_config.get("resampler_output_dim", self.config.hidden_size)
                
                #print(f"[DEBUG] Projector configuration:")
                #print(f"  type: {projector_type}")
                #print(f"  input_dim (from resampler): {resampler_output_dim}")
                #print(f"  output_dim (LLM hidden): {self.config.hidden_size}")
                
                if projector_type == "mlp2x_gelu":
                    self.mm_projector = MLPProjector(
                        input_dim=resampler_output_dim,
                        output_dim=self.config.hidden_size
                    )
                    assert self.mm_projector is not None, "Projector module failed to initialize"
                    
                    # Verify dimensions match between modules
                    if hasattr(self, 'mm_resampler'):
                        assert self.mm_resampler.output_dim == self.mm_projector.linear1.in_features, \
                            "Resampler output dimension must match Projector input dimension"
                        assert self.mm_projector.linear2.out_features == self.config.hidden_size, \
                            "Projector output dimension must match LLM hidden dimension"
                    
                    #print("[DEBUG] Projector module initialized successfully with proper dimension matching")

            # Verify at least one module is initialized
            has_trainable_module = (
                getattr(self, "mm_gated_cross_attention", None) is not None or
                getattr(self, "mm_resampler", None) is not None or
                getattr(self, "mm_projector", None) is not None
            )
            assert has_trainable_module, "No trainable modules were initialized"
        
        # Special token for delta_P, should be part of tokenizer
        self.delta_token_id = None # Will be set from tokenizer

    def set_protein_encoder(self, protein_encoder):
        """Allows injecting the protein encoder after initialization."""
        self.protein_encoder = protein_encoder

    def set_delta_token_id(self, token_id):
        """Set the delta token ID and initialize its embeddings."""
        self.delta_token_id = token_id
        
        # Initialize embeddings for the delta token
        if token_id is not None:
            # Get the embedding layer
            embed_layer = self.get_input_embeddings()
            hidden_size = embed_layer.weight.shape[1]
            
            # Initialize the embedding for the delta token with xavier uniform
            if token_id < embed_layer.weight.shape[0]:
                nn.init.xavier_uniform_(embed_layer.weight.data[token_id].view(1, hidden_size))
                #print(f"[DEBUG] Initialized embedding for delta token ID {token_id}")
            else:
                #print(f"[WARNING] Delta token ID {token_id} is out of bounds for embedding layer")
                pass

    def get_protein_encoder(self):
        return self.protein_encoder # For external access if needed, e.g. freezing/unfreezing

    def apply_GCA_resampler(self, wild_type_seqs_list, mutation_seqs_list):
        """
        Processes protein sequences through ESM, GCA, and Resampler.
        wild_type_seqs_list: List of WT AA strings.
        mutation_seqs_list: List of Mutated AA strings.
        """
        if self.protein_encoder is None:
            raise ValueError("Protein encoder is not initialized.")
        if self.mm_gated_cross_attention is None or self.mm_resampler is None:
            raise ValueError("GCA or Resampler is not configured.")

        # 1a. Protein Encoding (ESM)
        # Output: list of tensors (protein_sequence_length, esm_hidden_dimension)
        # The ESMProteinEncoder should handle batching and return padded tensors directly
        # Or return list of tensors that we pad here. Let's assume it returns padded tensors.
        
        # Get select layer from config
        select_layer = self.protein_config.get("mm_protein_select_layer", -1)
        
        # (batch_size, max_wt_protein_len, esm_hidden_dim)
        wt_protein_features = self.protein_encoder(wild_type_seqs_list, select_layer=select_layer)
        # (batch_size, max_mut_protein_len, esm_hidden_dim)
        mut_protein_features = self.protein_encoder(mutation_seqs_list, select_layer=select_layer)

        # Ensure protein features match the main model's dtype (e.g., bfloat16).
        # This is a clean cast at the boundary between the frozen encoder and the trainable adapters.
        model_dtype = next(self.mm_gated_cross_attention.parameters()).dtype
        wt_protein_features = wt_protein_features.to(dtype=model_dtype)
        mut_protein_features = mut_protein_features.to(dtype=model_dtype)

        # 1b. Padding (Handled by ESMProteinEncoder or here if it returns lists of tensors)
        max_len = max(wt_protein_features.shape[1], mut_protein_features.shape[1])
        def pad_tensor(tensor, target_len):
            padding_size = target_len - tensor.shape[1]
            if padding_size > 0:
                return nn.functional.pad(tensor, (0,0,0,padding_size))
            return tensor
        wt_protein_features_padded = pad_tensor(wt_protein_features, max_len)
        mut_protein_features_padded = pad_tensor(mut_protein_features, max_len)

        # Timing: before GCA
        before_gca_time = time.time()
        #print(f"[TIME] >>> Before GCA: {before_gca_time}")
        #print(f"[DEBUG] Before GCA:")
        #print(f"  wt_features: shape={wt_protein_features_padded.shape}, dtype={wt_protein_features_padded.dtype}")
        #print(f"  mut_features: shape={mut_protein_features_padded.shape}, dtype={mut_protein_features_padded.dtype}")

        # 1c. Gated Cross-Attention
        gca_start = time.time()
        delta_features = self.mm_gated_cross_attention(
            query_feats=mut_protein_features_padded,
            kv_feats_wt=wt_protein_features_padded,
            kv_feats_mut=mut_protein_features_padded
        )
        gca_end = time.time()
        #print(f"[TIME] <<< After GCA: {gca_end} (Duration: {gca_end - gca_start:.3f} sec)")
        #print(f"[DEBUG] After GCA:")
        #print(f"  delta_features: shape={delta_features.shape}, dtype={delta_features.dtype}")

        # 1d. Resampler
        resampler_start = time.time()
        resampled_features = self.mm_resampler(delta_features)
        resampler_end = time.time()
        #print(f"[TIME] Resampler duration: {resampler_end - resampler_start:.3f} sec")
        
        # Timing: after resampler, before next GCA (if in a loop)
        after_resampler_time = time.time()
        #print(f"[TIME] >>> After Resampler, before next GCA: {after_resampler_time}")
        
        return resampled_features

    def prepare_inputs_labels_for_multimodal(
            self, input_ids, attention_mask, labels, protein_features
    ):
        """
        Merges protein features with text embeddings.
        protein_features: (batch_size, num_media_tokens, llm_hidden_dim) - after projector
        """
        if protein_features is None or self.delta_token_id is None:
            return input_ids, attention_mask, labels, None # No protein features to merge
        
        batch_size, num_protein_tokens, llm_hidden_dim = protein_features.shape
        
        # Get LLM's token embeddings
        token_embeddings = self.get_input_embeddings()(input_ids) # (batch_size, text_seq_len, llm_hidden_dim)

        new_input_embeds = []
        new_labels = [] if labels is not None else None
        new_attention_mask = []

        for i in range(batch_size):
            # Find the position of DELTA_TOKEN_INDEX
            delta_token_indices = (input_ids[i] == self.delta_token_id).nonzero(as_tuple=True)[0]
            
            if len(delta_token_indices) == 0:
                #print(f"Warning: DELTA_TOKEN_ID {self.delta_token_id} not found in input_ids sample {i}. Protein features not inserted.")
                new_input_embeds.append(token_embeddings[i])
                if labels is not None: new_labels.append(labels[i])
                new_attention_mask.append(attention_mask[i])
                continue

            delta_token_idx = delta_token_indices[0] # Use the first occurrence

            # Get protein features for this sample - maintain 2D shape
            current_protein_features = protein_features[i]  # Shape: [num_protein_tokens, llm_hidden_dim]

            # Split text embeddings around the delta token
            pre_delta_embeds = token_embeddings[i, :delta_token_idx]
            post_delta_embeds = token_embeddings[i, delta_token_idx + 1:]
            
            # Concatenate: text_before | protein_features | text_after
            merged_embeds = torch.cat([pre_delta_embeds, current_protein_features, post_delta_embeds], dim=0)
            new_input_embeds.append(merged_embeds)
            
            if labels is not None:
                pre_delta_labels = labels[i, :delta_token_idx]
                # Protein feature labels: IGNORE_INDEX
                protein_labels = torch.full((num_protein_tokens,), IGNORE_INDEX, 
                                            device=labels.device, dtype=labels.dtype)
                post_delta_labels = labels[i, delta_token_idx + 1:]
                merged_labels = torch.cat([pre_delta_labels, protein_labels, post_delta_labels], dim=0)
                new_labels.append(merged_labels)
            
            # Update attention mask
            pre_delta_mask = attention_mask[i, :delta_token_idx]
            protein_mask = torch.ones(num_protein_tokens, device=attention_mask.device, dtype=attention_mask.dtype)
            post_delta_mask = attention_mask[i, delta_token_idx + 1:]
            merged_mask = torch.cat([pre_delta_mask, protein_mask, post_delta_mask], dim=0)
            new_attention_mask.append(merged_mask)

        # Pad sequences using the official PyTorch utility
        padded_input_embeds = pad_sequence(new_input_embeds, batch_first=True, padding_value=0)
        
        padded_attention_mask = pad_sequence(new_attention_mask, batch_first=True, padding_value=0)
        if labels is not None:
            padded_labels = pad_sequence(new_labels, batch_first=True, padding_value=IGNORE_INDEX)
        else:
            padded_labels = None
        
        return None, padded_input_embeds, padded_attention_mask, padded_labels


    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: typing.Optional[torch.Tensor] = None,
        position_ids: typing.Optional[torch.LongTensor] = None, # Added position_ids
        past_key_values: typing.Optional[typing.List[torch.FloatTensor]] = None,
        inputs_embeds: typing.Optional[torch.FloatTensor] = None,
        labels: typing.Optional[torch.LongTensor] = None,
        use_cache: typing.Optional[bool] = None,
        output_attentions: typing.Optional[bool] = None,
        output_hidden_states: typing.Optional[bool] = None,
        return_dict: typing.Optional[bool] = None,
        # Custom arguments for protein sequences
        wild_type_sequences: typing.Optional[typing.List[str]] = None,
        mutation_sequences: typing.Optional[typing.List[str]] = None,
    ) -> typing.Union[tuple, CausalLMOutputWithPast]:
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # Step 1: Handle multimodal inputs on the first forward pass (prefill)
        if past_key_values is None:
            protein_features_projected = None
            if wild_type_sequences is not None and mutation_sequences is not None and self.protein_config is not None:
                if self.protein_encoder is None:
                    raise ValueError("Protein encoder not set. Call set_protein_encoder() or ensure it's loaded via config.")
                if self.delta_token_id is None:
                    raise ValueError("Delta token ID not set. Call set_delta_token_id().")

                # Process protein sequences and project them
                protein_features_resampled = self.apply_GCA_resampler(
                    wild_type_seqs_list=wild_type_sequences,
                    mutation_seqs_list=mutation_sequences
                )
                protein_features_projected = self.mm_projector(protein_features_resampled)
                
                # Create the final `inputs_embeds`
                if inputs_embeds is None:
                    if self.training:
                        _, inputs_embeds, attention_mask, labels = self.prepare_inputs_labels_for_multimodal(
                            input_ids=input_ids,
                            attention_mask=attention_mask,
                            labels=labels,
                            protein_features=protein_features_projected
                        )
                    else:
                        _, inputs_embeds, attention_mask, _ = self.prepare_inputs_labels_for_multimodal(
                            input_ids=input_ids,
                            attention_mask=attention_mask,
                            labels=None,
                            protein_features=protein_features_projected
                        )
                    input_ids = None # Embeddings are now the source of truth
        
        # Step 2: Ensure all subsequent passes (decode) also use embeddings
        if inputs_embeds is None:
            inputs_embeds = self.get_input_embeddings()(input_ids)
            input_ids = None

        # Call the parent LlamaForCausalLM forward method
        outputs = super().forward(
            input_ids=input_ids, # Should be None if embeds are used
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            labels=labels,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        return outputs

    def freeze_protein_related_modules(self):
        """Freezes protein encoder and optionally multimodal modules based on config."""
        # Always freeze protein encoder
        if hasattr(self, 'protein_encoder') and self.protein_encoder is not None:
            for name, param in self.protein_encoder.named_parameters():
                param.requires_grad = False
            #print("Protein encoder frozen.")
        
        # Only freeze multimodal modules if not in tuning mode
        if not self.protein_config.get("tune_mm_mlp_adapter", False):
            #print("[DEBUG] Freezing multimodal modules (tune_mm_mlp_adapter=False)")
            if hasattr(self, 'mm_gated_cross_attention') and self.mm_gated_cross_attention is not None:
                for name, param in self.mm_gated_cross_attention.named_parameters():
                    param.requires_grad = False
                #print("GCA frozen.")
            
            if hasattr(self, 'mm_resampler') and self.mm_resampler is not None:
                for name, param in self.mm_resampler.named_parameters():
                    param.requires_grad = False
                #print("Resampler frozen.")
            
            if hasattr(self, 'mm_projector') and self.mm_projector is not None:
                for name, param in self.mm_projector.named_parameters():
                    param.requires_grad = False
                #print("Projector frozen.")
        else:
            #print("[DEBUG] Keeping multimodal modules trainable (tune_mm_mlp_adapter=True)")
            pass

        # #print trainable status
        for name, param in self.named_parameters():
            if any(module in name for module in ['mm_gated_cross_attention', 'mm_resampler', 'mm_projector', 'protein_encoder']):
                #print(f"[DEBUG] {name}: requires_grad = {param.requires_grad}")
                pass

    def unfreeze_protein_related_modules(self):
        """Unfreeze multimodal modules for fine-tuning."""
        # Only unfreeze if tuning is enabled
        if hasattr(self, 'protein_config') and self.protein_config.get("tune_mm_mlp_adapter", False):
            if hasattr(self, 'mm_gated_cross_attention') and self.mm_gated_cross_attention is not None:
                for name, param in self.mm_gated_cross_attention.named_parameters():
                    param.requires_grad = True
                #print("GCA unfrozen.")
            
            if hasattr(self, 'mm_resampler') and self.mm_resampler is not None:
                for name, param in self.mm_resampler.named_parameters():
                    param.requires_grad = True
                #print("Resampler unfrozen.")
            
            if hasattr(self, 'mm_projector') and self.mm_projector is not None:
                for name, param in self.mm_projector.named_parameters():
                    param.requires_grad = True
                #print("Projector unfrozen.")
            
            # Verify unfreeze status
            #print("\n[DEBUG] Module trainable status after unfreezing:")
            for name, param in self.named_parameters():
                if any(module in name for module in ['mm_gated_cross_attention', 'mm_resampler', 'mm_projector']):
                    #print(f"  {name}: requires_grad = {param.requires_grad}")
                    pass
        else:
            #print("[WARNING] tune_mm_mlp_adapter is False, keeping modules frozen. Set tune_mm_mlp_adapter=True to train these modules.")
            pass

    def freeze_all_but_lora(self):
        """Freezes base LLM and conditionally GCA/Resampler/Projector based on tune_mm_mlp_adapter flag"""
        # Explicitly freeze LLM parameters that are not LoRA layers
        for name, param in self.named_parameters():
            # Skip adapter modules if they should be trained
            if (self.protein_config.get("tune_mm_mlp_adapter", False) and 
                any(module in name for module in ['mm_gated_cross_attention', 'mm_resampler', 'mm_projector'])):
                continue
                
            # Freeze non-LoRA parameters
            if not any(lora_str in name for lora_str in ['lora_', 'adapter']):
                param.requires_grad = False
                #print(f"[DEBUG] Freezing param: {name}")
        #print("Base LLM parameters frozen except for LoRA layers.")
        
        # Conditionally freeze GCA, Resampler, Projector based on tune_mm_mlp_adapter flag
        if not self.protein_config.get("tune_mm_mlp_adapter", False):
            if hasattr(self, 'mm_gated_cross_attention') and self.mm_gated_cross_attention is not None:
                for name, param in self.mm_gated_cross_attention.named_parameters():
                    param.requires_grad = False
                #print("GCA frozen for LoRA.")
            if hasattr(self, 'mm_resampler') and self.mm_resampler is not None:
                for name, param in self.mm_resampler.named_parameters():
                    param.requires_grad = False
                #print("Resampler frozen for LoRA.")
            if hasattr(self, 'mm_projector') and self.mm_projector is not None:
                for name, param in self.mm_projector.named_parameters():
                    param.requires_grad = False
                #print("Projector frozen for LoRA.")
        else:
            #print("GCA, Resampler, and Projector will be trained alongside LoRA parameters (tune_mm_mlp_adapter=True)")
            pass
        
        # Verify proper freezing
        trainable_params = [name for name, param in self.named_parameters() if param.requires_grad]
        #print(f"[DEBUG] Trainable parameters after freezing:")
        for param in trainable_params:
            #print(f"  {param}")
            pass
        
        # Protein encoder should already be frozen from pretraining
        if hasattr(self, 'protein_encoder') and self.protein_encoder is not None:
             for param in self.protein_encoder.parameters():
                param.requires_grad = False

class MLPProjector(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        hidden_dim = input_dim * 2  # As per mlp2x_gelu
        
        # First linear layer + GELU
        self.linear1 = nn.Linear(input_dim, hidden_dim)
        self.act = nn.GELU()
        # Second linear layer
        self.linear2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = self.linear1(x)
        x = self.act(x)
        x = self.linear2(x)
        return x