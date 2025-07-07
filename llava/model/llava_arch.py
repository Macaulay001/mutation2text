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
from llava.utils.data_utils import IGNORE_INDEX
from typing import Optional, List, Union, Tuple

def exists(val):
    return val is not None

class FeedForward(nn.Module):
    def __init__(self, dim, mult=4):
        super().__init__()
        inner_dim = int(dim * mult)
        print(f"[DEBUG FF Init] Dimensions:")
        print(f"  input_dim: {dim}")
        print(f"  inner_dim: {inner_dim}")
        
        self.norm = nn.LayerNorm(dim)
        self.fc1 = nn.Linear(dim, inner_dim, bias=False)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(inner_dim, dim, bias=False)
        
        # Convert weights to float32
        self.norm.weight.data = self.norm.weight.data.to(torch.float32)
        self.norm.bias.data = self.norm.bias.data.to(torch.float32)
        self.fc1.weight.data = self.fc1.weight.data.to(torch.float32)
        self.fc2.weight.data = self.fc2.weight.data.to(torch.float32)
        
        # print("[DEBUG FF Init] Parameter dtypes:")
        # print(f"  norm.weight: {self.norm.weight.dtype}")
        # print(f"  norm.bias: {self.norm.bias.dtype}")
        # print(f"  fc1.weight: {self.fc1.weight.dtype}")
        # print(f"  fc2.weight: {self.fc2.weight.dtype}")

    def forward(self, x):
        orig_dtype = x.dtype
        # print("\n[DEBUG FF Forward] Input:")
        # print(f"  x: shape={x.shape}, dtype={x.dtype}")
        # print(f"  fc1.weight: {self.fc1.weight.dtype}")
        # print(f"  fc2.weight: {self.fc2.weight.dtype}")
        
        # Ensure LayerNorm parameters are float32 and apply LayerNorm in float32
        x_f32 = x.to(torch.float32)
        # print(f"[DEBUG FF Forward] x_f32 dtype before norm: {x_f32.dtype}")
        x = F.layer_norm(
            x_f32,
            self.norm.normalized_shape,
            self.norm.weight.to(torch.float32),
            self.norm.bias.to(torch.float32),
            self.norm.eps
        )
        # print(f"[DEBUG FF Forward] x dtype after norm: {x.dtype}")
        
        # Convert back to original dtype for the rest of processing
        x = x.to(orig_dtype)
        # print(f"[DEBUG FF Forward] x dtype after converting back: {x.dtype}")
        
        # Match input dtype to weight dtype for linear layers
        x = self.fc1(x.to(self.fc1.weight.dtype))
        # print(f"[DEBUG FF Forward] x dtype after fc1: {x.dtype}")
        
        x = self.act(x)  # GELU preserves dtype
        # print(f"[DEBUG FF Forward] x dtype after activation: {x.dtype}")
        
        x = self.fc2(x.to(self.fc2.weight.dtype))
        # print(f"[DEBUG FF Forward] x dtype after fc2: {x.dtype}")
        return x



class GatedCrossAttention(nn.Module):
    def __init__(self, query_dim, key_value_dim, output_dim, num_heads=8, dim_head=64, ff_mult=4):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim_head
        inner_dim = dim_head * num_heads
        
        # Main attention components for cross attention (mut -> wt)
        self.norm_query = nn.LayerNorm(query_dim)
        self.norm_key_value = nn.LayerNorm(key_value_dim)
        self.to_q_cross = nn.Linear(query_dim, inner_dim, bias=False)
        self.to_kv_cross = nn.Linear(key_value_dim, inner_dim * 2, bias=False)
        
        # Shared self-attention for both wild-type and mutation features
        self.to_qkv_self_attn = nn.Linear(query_dim, inner_dim * 3, bias=False)

        # Attention-based gating mechanism
        self.gate_norm = nn.LayerNorm(inner_dim)
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

    def forward(self, query_feats, kv_feats_wt, mode):
        # Handle different modes of operation for GCA
        scale = self.head_dim ** -0.5
        
        kv_feats_wt = kv_feats_wt.to(torch.bfloat16)
        kv_feats_wt_norm = self.norm_key_value(kv_feats_wt)

        # Always compute wild-type self-attention for modes that use it
        wt_out = None
        if mode in ['full', 'wt_only']:
            qkv_wt = self.to_qkv_self_attn(kv_feats_wt_norm)
            q_wt, k_wt_self, v_wt_self = qkv_wt.chunk(3, dim=-1)
            
            q_wt = rearrange(q_wt, 'b n (h d) -> b h n d', h=self.num_heads)
            k_wt_self = rearrange(k_wt_self, 'b n (h d) -> b h n d', h=self.num_heads)
            v_wt_self = rearrange(v_wt_self, 'b n (h d) -> b h n d', h=self.num_heads)
            
            sim_wt_self = torch.einsum('b h i d, b h j d -> b h i j', q_wt * scale, k_wt_self)
            attn_wt_self = F.softmax(sim_wt_self, dim=-1)
            wt_out = torch.einsum('b h i j, b h j d -> b h i d', attn_wt_self, v_wt_self)
            wt_out = rearrange(wt_out, 'b h n d -> b n (h d)')
            wt_out = self.to_out(wt_out)

        if mode == 'wt_only':
            return None, wt_out

        # For 'full' and 'delta_only', mutation features are required
        if query_feats is None:
            raise ValueError("query_feats cannot be None for 'full' or 'delta_only' modes.")
        
        query_feats = query_feats.to(torch.bfloat16)
        query_feats_norm = self.norm_query(query_feats)
        
        # Cross Attention (mut -> wt)
        q_cross = self.to_q_cross(query_feats_norm)
        # Ensure we use the kv_feats_wt that corresponds to the query_feats batch
        k_cross, v_cross = self.to_kv_cross(kv_feats_wt_norm).chunk(2, dim=-1)
        
        q_cross = rearrange(q_cross, 'b n (h d) -> b h n d', h=self.num_heads)
        k_cross = rearrange(k_cross, 'b n (h d) -> b h n d', h=self.num_heads)
        v_cross = rearrange(v_cross, 'b n (h d) -> b h n d', h=self.num_heads)
        
        sim_cross = torch.einsum('b h i d, b h j d -> b h i j', q_cross * scale, k_cross)
        attn_cross = F.softmax(sim_cross, dim=-1)
        cross_out = torch.einsum('b h i j, b h j d -> b h i d', attn_cross, v_cross)
        cross_out = rearrange(cross_out, 'b h n d -> b n (h d)')

        # Self Attention (mut -> mut)
        qkv_self = self.to_qkv_self_attn(query_feats_norm)
        q_self, k_self, v_self = qkv_self.chunk(3, dim=-1)
        
        q_self = rearrange(q_self, 'b n (h d) -> b h n d', h=self.num_heads)
        k_self = rearrange(k_self, 'b n (h d) -> b h n d', h=self.num_heads)
        v_self = rearrange(v_self, 'b n (h d) -> b h n d', h=self.num_heads)
        
        sim_self = torch.einsum('b h i d, b h j d -> b h i j', q_self * scale, k_self)
        attn_self = F.softmax(sim_self, dim=-1)
        self_out = torch.einsum('b h i j, b h j d -> b h i d', attn_self, v_self)
        self_out = rearrange(self_out, 'b h n d -> b n (h d)')

        # Gating
        self_out_norm = self.gate_norm(self_out)
        cross_out_norm = self.gate_norm(cross_out)
        gate_context, _ = self.gate_attention(query=self_out_norm, key=cross_out_norm, value=cross_out_norm)
        mixing_weights = self.to_mixing_weights(gate_context)
        
        mixed_output = (
            mixing_weights[..., 0:1] * self_out +
            mixing_weights[..., 1:2] * cross_out
        )
        
        delta = self.to_out(mixed_output)
        
        if hasattr(self, 'ff'):
            ff_out = self.ff(delta)
            delta = delta + self.ff_gate * ff_out
        
        return delta, wt_out

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
        
        # print(f"[DEBUG Resampler Init] Configuration:")
        # print(f"  input_dim: {input_dim}")
        # print(f"  output_dim: {output_dim}")
        # print(f"  num_heads: {num_heads}")
        # print(f"  dim_head: {dim_head}")
        # print(f"  inner_dim: {inner_dim}")
        # print(f"  num_output_tokens: {num_output_tokens}")
        
        # Learnable latent vectors that will be queried against the input sequence
        self.latents = nn.Parameter(torch.randn(num_output_tokens, input_dim))
        
        # Position embeddings for the latents only - input sequence will use relative attention
        self.latent_pos_emb = nn.Parameter(torch.randn(1, num_output_tokens, input_dim))
        
        # Layer stack
        self.layers = nn.ModuleList([])
        for layer_idx in range(num_layers):
            # Create attention block
            attn_block = nn.ModuleDict({
                'norm_media': nn.LayerNorm(input_dim),
                'norm_latents': nn.LayerNorm(input_dim),
                'to_q': nn.Linear(input_dim, inner_dim, bias=False),
                'to_kv': nn.Linear(input_dim, inner_dim * 2, bias=False),
                'to_out': nn.Linear(inner_dim, input_dim, bias=False)
            })
            
            self.layers.append(nn.ModuleList([
                attn_block,
                FeedForward(dim=input_dim, mult=ff_mult)
            ]))
        
        # Final normalization and projection
        self.norm = nn.LayerNorm(input_dim)
        if input_dim != output_dim:
            self.proj_out = nn.Linear(input_dim, output_dim)
        else:
            self.proj_out = nn.Identity()
    
    def forward(self, x):
        """
        x: (batch_size, seq_len, input_dim) - Features from GCA
        Returns: (batch_size, num_output_tokens, output_dim)
        """
        # Store original dtype and shape
        orig_dtype = x.dtype
        batch_size, seq_len, _ = x.shape
        
        # Expand latents for batch and add positional embeddings
        latents = repeat(self.latents, 'n d -> b n d', b=batch_size)
        latents = latents + self.latent_pos_emb
        
        # Process through layers with gradient checkpointing
        for attn_block, ff in self.layers:
            def _attn_block(x, latents):
                # Layer normalization
                x_norm = F.layer_norm(
                    x.to(torch.float32),
                    attn_block['norm_media'].normalized_shape,
                    attn_block['norm_media'].weight.to(torch.float32),
                    attn_block['norm_media'].bias.to(torch.float32),
                    attn_block['norm_media'].eps
                ).to(orig_dtype)
                
                latents_norm = F.layer_norm(
                    latents.to(torch.float32),
                    attn_block['norm_latents'].normalized_shape,
                    attn_block['norm_latents'].weight.to(torch.float32),
                    attn_block['norm_latents'].bias.to(torch.float32),
                    attn_block['norm_latents'].eps
                ).to(orig_dtype)

                # Project to queries, keys, values
                q = attn_block['to_q'](latents_norm)  # [batch, num_output_tokens, inner_dim]
                k, v = attn_block['to_kv'](x_norm).chunk(2, dim=-1)  # [batch, seq_len, inner_dim]

                # Split heads
                q = rearrange(q, 'b n (h d) -> b h n d', h=self.num_heads)
                k = rearrange(k, 'b n (h d) -> b h n d', h=self.num_heads)
                v = rearrange(v, 'b n (h d) -> b h n d', h=self.num_heads)

                # Scaled dot product attention
                scale = q.shape[-1] ** -0.5
                sim = torch.einsum('b h i d, b h j d -> b h i j', q, k) * scale
                
                # Attention softmax
                attn = F.softmax(sim, dim=-1)
                
                # Compute weighted sum
                out = torch.einsum('b h i j, b h j d -> b h i d', attn, v)
                
                # Combine heads
                out = rearrange(out, 'b h n d -> b n (h d)')
                
                # Project output
                out = out.to(attn_block['to_out'].weight.dtype)
                return attn_block['to_out'](out)
            
            # Apply attention and feedforward with gradient checkpointing
            latents = latents + checkpoint(_attn_block, x, latents, use_reentrant=False)
            
            # Apply feedforward
            latents_f32 = latents.to(torch.float32)
            latents = latents + checkpoint(ff, latents_f32, use_reentrant=False)
        
        # Final normalization
        latents = F.layer_norm(
            latents.to(torch.float32),
            self.norm.normalized_shape,
            self.norm.weight.to(torch.float32),
            self.norm.bias.to(torch.float32),
            self.norm.eps
        ).to(orig_dtype)
        
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
        print("[DEBUG] Initializing LlavaLlamaForCausalLM with protein_config:", self.protein_config)

        if self.protein_config is not None:

            # Initialize GCA
            if self.protein_config.get("mm_gated_cross_attention", False):
                print("[DEBUG] Initializing GCA module with parameters:")
                print(f"  esm_hidden_size: {self.protein_config['esm_hidden_size']}")
                print(f"  gca_output_dim: {self.protein_config['gca_output_dim']}")
                print(f"  num_heads: {self.protein_config['mm_gca_num_heads']}")
                
                self.mm_gated_cross_attention = GatedCrossAttention(
                    query_dim=self.protein_config['esm_hidden_size'],
                    key_value_dim=self.protein_config['esm_hidden_size'],
                    output_dim=self.protein_config['gca_output_dim'],
                    num_heads=self.protein_config['mm_gca_num_heads']
                )
                assert self.mm_gated_cross_attention is not None, "GCA module failed to initialize"
                print("[DEBUG] GCA module initialized successfully")

            # Initialize Resampler
            if self.protein_config.get("mm_resampler", False):
                print("[DEBUG] Initializing Resampler module with parameters:")
                print(f"  gca_output_dim: {self.protein_config['gca_output_dim']}")
                print(f"  resampler_output_dim: {self.protein_config['resampler_output_dim']}")
                print(f"  num_media_tokens: {self.protein_config['num_media_tokens']}")
                print(f"  num_heads: {self.protein_config['mm_resampler_num_heads']}")
                
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
                # print("[DEBUG] Resampler module initialized successfully")

            # Initialize Projector
            if self.protein_config.get("use_mm_proj", False):
                print("[DEBUG] Initializing Projector module")
                projector_type = self.protein_config.get("mm_projector_type", "mlp2x_gelu")
                resampler_output_dim = self.protein_config.get("resampler_output_dim", self.config.hidden_size)
                
                print(f"[DEBUG] Projector configuration:")
                print(f"  type: {projector_type}")
                print(f"  input_dim (from resampler): {resampler_output_dim}")
                print(f"  output_dim (LLM hidden): {self.config.hidden_size}")
                
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
                    
                    print("[DEBUG] Projector module initialized successfully with proper dimension matching")

            # Verify at least one module is initialized
            has_trainable_module = (
                getattr(self, "mm_gated_cross_attention", None) is not None or
                getattr(self, "mm_resampler", None) is not None or
                getattr(self, "mm_projector", None) is not None
            )
            assert has_trainable_module, "No trainable modules were initialized"
        
        # # Special token for delta_P, should be part of tokenizer
        # self.delta_token_id = None # Will be set from tokenizer
        # self.wildtype_protein_token_id = None # Will be set from tokenizer

    def set_protein_encoder(self, protein_encoder):
        """Allows injecting the protein encoder after initialization."""
        self.protein_encoder = protein_encoder

    def get_protein_encoder(self):
        return self.protein_encoder # For external access if needed, e.g. freezing/unfreezing


    def set_wt_protein_start_token_id(self, token_id):
        self.wt_protein_start_token_id = token_id

        # Initialize embeddings for the wt protein start token
        if token_id is not None:
            embed_layer = self.get_input_embeddings()
            hidden_size = embed_layer.weight.shape[1]
            if token_id < embed_layer.weight.shape[0]:
                nn.init.xavier_uniform_(embed_layer.weight.data[token_id].view(1, hidden_size))
                print(f"[DEBUG] Initialized embedding for wt protein start token ID {token_id}")
            else:
                print(f"[WARNING] WT protein start token ID {token_id} is out of bounds for embedding layer")

    def set_wt_protein_end_token_id(self, token_id):
        self.wt_protein_end_token_id = token_id

        # Initialize embeddings for the wt protein end token
        if token_id is not None:
            embed_layer = self.get_input_embeddings()
            hidden_size = embed_layer.weight.shape[1]
            if token_id < embed_layer.weight.shape[0]:
                nn.init.xavier_uniform_(embed_layer.weight.data[token_id].view(1, hidden_size))
                print(f"[DEBUG] Initialized embedding for wt protein end token ID {token_id}")
            else:
                print(f"[WARNING] WT protein end token ID {token_id} is out of bounds for embedding layer")

    def set_mut_protein_start_token_id(self, token_id): 
        self.mut_protein_start_token_id = token_id

        # Initialize embeddings for the mut protein start token
        if token_id is not None:
            embed_layer = self.get_input_embeddings()
            hidden_size = embed_layer.weight.shape[1]
            if token_id < embed_layer.weight.shape[0]:
                nn.init.xavier_uniform_(embed_layer.weight.data[token_id].view(1, hidden_size))
                print(f"[DEBUG] Initialized embedding for mut protein start token ID {token_id}")
            else:   
                print(f"[WARNING] Mut protein start token ID {token_id} is out of bounds for embedding layer")

    def set_mut_protein_end_token_id(self, token_id):
        self.mut_protein_end_token_id = token_id

        # Initialize embeddings for the mut protein end token
        if token_id is not None:
            embed_layer = self.get_input_embeddings()
            hidden_size = embed_layer.weight.shape[1]
            if token_id < embed_layer.weight.shape[0]:
                nn.init.xavier_uniform_(embed_layer.weight.data[token_id].view(1, hidden_size))
                print(f"[DEBUG] Initialized embedding for mut protein end token ID {token_id}")
            else:
                print(f"[WARNING] Mut protein end token ID {token_id} is out of bounds for embedding layer")



    def apply_GCA_resampler(self, wild_type_seqs_list, mutation_seqs_list, mode):
        """
        Processes protein sequences through ESM, GCA, and Resampler based on the mode.
        """
        if self.protein_encoder is None:
            raise ValueError("Protein encoder is not initialized.")
        if self.mm_gated_cross_attention is None or self.mm_resampler is None or self.mm_projector is None:
            raise ValueError("GCA, Resampler, or Projector is not configured.")

        # Encode wild-type sequences if needed
        wt_feats = None
        if mode in ['full', 'wt_only', 'delta_only']:
            if wild_type_seqs_list is None or not all(wild_type_seqs_list):
                 raise ValueError("Wild-type sequences must be provided for 'full', 'wt_only', or 'delta_only' mode.")
            wt_feats = self.protein_encoder(wild_type_seqs_list)
            print("\n[DEBUG] ESM Encoder Output (Wild-Type):")
            print(f"  - Shape: {wt_feats.shape}")
            print(f"  - Dtype: {wt_feats.dtype}")
            print(f"  - Contains NaN: {torch.isnan(wt_feats).any().item()}")
            print(f"  - Contains Inf: {torch.isinf(wt_feats).any().item()}")
            print(f"  - Max value: {wt_feats.max().item():.4f}")
            print(f"  - Min value: {wt_feats.min().item():.4f}")
            print(f"  - Mean value: {wt_feats.mean().item():.4f}")

        # Encode mutation sequences if needed
        mut_feats = None
        if mode in ['full', 'delta_only']:
            if mutation_seqs_list is None or not all(mutation_seqs_list):
                 raise ValueError("Mutation sequences must be provided for 'full' or 'delta_only' mode.")
            mut_feats = self.protein_encoder(mutation_seqs_list)
            print("\n[DEBUG] ESM Encoder Output (Mutation):")
            print(f"  - Shape: {mut_feats.shape}")
            print(f"  - Dtype: {mut_feats.dtype}")
            print(f"  - Contains NaN: {torch.isnan(mut_feats).any().item()}")
            print(f"  - Contains Inf: {torch.isinf(mut_feats).any().item()}")
            print(f"  - Max value: {mut_feats.max().item():.4f}")
            print(f"  - Min value: {mut_feats.min().item():.4f}")
            print(f"  - Mean value: {mut_feats.mean().item():.4f}")

        # Pad features to the same length if both are present
        if wt_feats is not None and mut_feats is not None:
            max_len = max(wt_feats.shape[1], mut_feats.shape[1])
            def pad_tensor(tensor, target_len):
                padding_size = target_len - tensor.shape[1]
                if padding_size > 0:
                    return F.pad(tensor, (0, 0, 0, padding_size))
                return tensor
            wt_feats = pad_tensor(wt_feats, max_len)
            mut_feats = pad_tensor(mut_feats, max_len)

        # Gated Cross-Attention
        delta_from_gca, wt_out_from_gca = self.mm_gated_cross_attention(
            query_feats=mut_feats, 
            kv_feats_wt=wt_feats,
            mode=mode
        )

        # Resample features
        protein_features = None
        protein_features_wt = None
        if self.mm_resampler is not None:
            if delta_from_gca is not None:
                protein_features = self.mm_resampler(delta_from_gca)
            if wt_out_from_gca is not None:
                protein_features_wt = self.mm_resampler(wt_out_from_gca)
        
        return protein_features, protein_features_wt

    def prepare_inputs_labels_for_multimodal(
        self, input_ids, attention_mask, labels, delta_features, wt_features
    ):
        if delta_features is None and wt_features is None:
            return input_ids, None, attention_mask, labels

        print("\n=== prepare_inputs_labels_for_multimodal Debug ===")
        print("Input shapes:")
        print(f"  input_ids: {input_ids.shape}")
        print(f"  attention_mask: {attention_mask.shape}")
        if labels is not None:
            print(f"  labels: {labels.shape}")
        if delta_features is not None:
            print(f"  delta_features: {delta_features.shape}")
        else:
            print(f"  delta_features: None")
        if wt_features is not None:
            print(f"  wt_features: {wt_features.shape}")
        else:
            print(f"  wt_features: None")

        token_embeddings = self.get_input_embeddings()(input_ids)
        new_input_embeds, new_labels, new_attention_mask = [], [], []
        batch_size = input_ids.shape[0]

        print("\nSpecial token counts in input_ids:")
        delta_tokens = (input_ids == self.mut_protein_start_token_id).sum()
        wt_tokens = (input_ids == self.wt_protein_start_token_id).sum()
        print(f"  Delta tokens: {delta_tokens}")
        print(f"  Wildtype tokens: {wt_tokens}")

        print(f"\nToken embeddings shape: {token_embeddings.shape}")
        print(f"Batch size: {batch_size}, Hidden dim: {token_embeddings.shape[-1]}")

        for i in range(batch_size):
            print(f"\nProcessing batch item {i}:")
            # Find all special token indices for the current sample
            wt_start_idx = (input_ids[i] == self.wt_protein_start_token_id).nonzero(as_tuple=True)[0]
            wt_end_idx = (input_ids[i] == self.wt_protein_end_token_id).nonzero(as_tuple=True)[0]
            mut_start_idx = (input_ids[i] == self.mut_protein_start_token_id).nonzero(as_tuple=True)[0]
            mut_end_idx = (input_ids[i] == self.mut_protein_end_token_id).nonzero(as_tuple=True)[0]
            
            print(f"  Found wildtype tokens at: {wt_start_idx.tolist()}")
            print(f"  Found delta tokens at: {mut_start_idx.tolist()}")

            # Create a list of parts to assemble
            embed_parts, label_parts, mask_parts = [], [], []
            last_idx = 0

            # Determine insertion order based on token positions
            insertions = []
            # Only plan to insert if BOTH start/end tokens are present AND the corresponding features were passed in.
            if len(wt_start_idx) > 0 and len(wt_end_idx) > 0 and wt_features is not None:
                insertions.append({'type': 'wt', 'start': wt_start_idx[0], 'end': wt_end_idx[0]})
            if len(mut_start_idx) > 0 and len(mut_end_idx) > 0 and delta_features is not None:
                insertions.append({'type': 'mut', 'start': mut_start_idx[0], 'end': mut_end_idx[0]})
            
            insertions.sort(key=lambda x: x['start'])

            for insert in insertions:
                start_idx, end_idx = insert['start'], insert['end']
                
                # Part 1: Text from last stop to this start
                embed_parts.append(token_embeddings[i, last_idx : start_idx + 1])
                label_parts.append(labels[i, last_idx : start_idx + 1])
                mask_parts.append(attention_mask[i, last_idx : start_idx + 1])
                
                # Part 2: Inserted features, guaranteed to exist by the check above
                if insert['type'] == 'wt':
                    features = wt_features[i]
                    num_feat_tokens = wt_features.shape[1]
                    if delta_features is None: # wt_only mode for this sample
                        print(f"  Processing wt_only mode")
                        print(f"  Merging wildtype features at index {start_idx.item()}")
                        print(f"  Number of wildtype tokens: {num_feat_tokens}")
                else: # 'mut'
                    features = delta_features[i]
                    num_feat_tokens = delta_features.shape[1]
                    print(f"  Processing delta or full mode")
                    print(f"  Delta token index: {start_idx.item()}")
                    print(f"  Number of protein tokens: {num_feat_tokens}")

                
                embed_parts.append(features)
                label_parts.append(torch.full((num_feat_tokens,), IGNORE_INDEX, device=labels.device, dtype=labels.dtype))
                mask_parts.append(torch.ones(num_feat_tokens, device=attention_mask.device, dtype=attention_mask.dtype))

                last_idx = end_idx

            # Part 3: Remaining text
            embed_parts.append(token_embeddings[i, last_idx:])
            label_parts.append(labels[i, last_idx:])
            mask_parts.append(attention_mask[i, last_idx:])

            # Assemble the final sequence
            merged_embeds = torch.cat(embed_parts, dim=0)
            if insertions and delta_features is None and any(d['type'] == 'wt' for d in insertions):
                 print(f"  Merged embeddings shape: {merged_embeds.shape}")

            new_input_embeds.append(merged_embeds)
            new_labels.append(torch.cat(label_parts, dim=0))
            new_attention_mask.append(torch.cat(mask_parts, dim=0))
            
        # Pad the entire batch to the same length
        max_len = max(len(embed) for embed in new_input_embeds)
        print(f"\nPadding all sequences to max length: {max_len}")
        
        final_embeds = torch.zeros(batch_size, max_len, token_embeddings.shape[-1], dtype=token_embeddings.dtype, device=token_embeddings.device)
        final_labels = torch.full((batch_size, max_len), IGNORE_INDEX, dtype=labels.dtype, device=labels.device)
        final_mask = torch.zeros(batch_size, max_len, dtype=attention_mask.dtype, device=attention_mask.device)

        for i in range(batch_size):
            seq_len = new_input_embeds[i].shape[0]
            final_embeds[i, :seq_len] = new_input_embeds[i]
            final_labels[i, :seq_len] = new_labels[i]
            final_mask[i, :seq_len] = new_attention_mask[i]
        
        print("\nFinal output shapes:")
        print(f"  input_embeds: {final_embeds.shape}")
        print(f"  attention_mask: {final_mask.shape}")
        print(f"  labels: {final_labels.shape}")
        print("=== End prepare_inputs_labels_for_multimodal Debug ===")

        return None, final_embeds, final_mask, final_labels

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        wild_type_sequences: Optional[List[str]] = None,
        mutation_sequences: Optional[List[str]] = None,
        attention_mode: Optional[List[str]] = None,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        print("\n=== Forward Pass Debug ===")
        print("Input shapes and types:")
        print(f"  input_ids: {input_ids.shape if input_ids is not None else 'None'}")
        print(f"  attention_mask: {attention_mask.shape if attention_mask is not None else 'None'}")
        print(f"  position_ids: {position_ids.shape if position_ids is not None else 'None'}")
        print(f"  inputs_embeds: {inputs_embeds.shape if inputs_embeds is not None else 'None'}")
        print(f"  labels: {labels.shape if labels is not None else 'None'}")

        print("\nSequence inputs:")
        print(f"  wild_type_sequences: {len(wild_type_sequences) if wild_type_sequences is not None else 0}")
        print(f"  mutation_sequences: {len(mutation_sequences) if mutation_sequences is not None else 0}")
        print(f"  attention_mode: {attention_mode}")

        has_protein_input = wild_type_sequences is not None or mutation_sequences is not None
        print(f"\nHas protein input: {has_protein_input}")


        if has_protein_input:
            # Group samples by attention_mode
            print("\nGrouping samples by attention_mode...")
            grouped_indices = {'full': [], 'delta_only': [], 'wt_only': [], 'text_only': []}
            for i, mode in enumerate(attention_mode):
                grouped_indices[mode].append(i)

            print("Grouped indices:")
            print(f"  full: {len(grouped_indices['full'])} samples")
            print(f"  delta_only: {len(grouped_indices['delta_only'])} samples")
            print(f"  wt_only: {len(grouped_indices['wt_only'])} samples")
            print(f"  text_only: {len(grouped_indices['text_only'])} samples")

            final_embeds_parts = []
            final_attention_mask_parts = []
            final_labels_parts = []
            original_indices_parts = []

            # Process each group
            for mode, indices in grouped_indices.items():
                if not indices:
                    continue
                
                print(f"\n[DEBUG] Processing {mode} group with {len(indices)} samples")
                sub_batch_indices = torch.tensor(indices, device=self.device)
                
                # Common slicing for all modes
                sub_input_ids = input_ids[sub_batch_indices]
                sub_attention_mask = attention_mask[sub_batch_indices]
                sub_labels = labels[sub_batch_indices] if labels is not None else None

                print("  Sub-batch shapes:")
                print(f"    input_ids: {sub_input_ids.shape}")
                print(f"    attention_mask: {sub_attention_mask.shape}")
                if sub_labels is not None:
                    print(f"    labels: {sub_labels.shape}")

                delta_features, wt_features = None, None

                if mode in ['full', 'delta_only', 'wt_only']:
                    sub_wt_seqs = [wild_type_sequences[i] for i in indices] if mode in ['full', 'wt_only', 'delta_only'] else None
                    sub_mut_seqs = [mutation_sequences[i] for i in indices] if mode in ['full', 'delta_only'] else None
                    
                    print(f"\nProcessing {mode} group:")
                    print(f"  Wild-type sequences: {len(sub_wt_seqs) if sub_wt_seqs is not None else 0}")
                    print(f"  Mutation sequences: {len(sub_mut_seqs) if sub_mut_seqs is not None else 0}")

                    delta_features, wt_features = self.apply_GCA_resampler(
                        wild_type_seqs_list=sub_wt_seqs,
                        mutation_seqs_list=sub_mut_seqs,
                        mode=mode
                    )
                    
                    print("  GCA+Resampler output shapes:")
                    print(f"    delta_features: {delta_features.shape if delta_features is not None else 'None'}")
                    print(f"    wt_features: {wt_features.shape if wt_features is not None else 'None'}")

                    # Project features to the LLM's embedding space
                    if delta_features is not None and hasattr(self, 'mm_projector'):
                        projected_features = self.mm_projector(delta_features)
                        print(f"\n[DEBUG] MLPProjector forward (delta):\n  Input shape: {delta_features.shape}, dtype: {delta_features.dtype}\n    After projection: {projected_features.shape}")
                        delta_features = projected_features

                    if wt_features is not None and hasattr(self, 'mm_projector'):
                        projected_features_wt = self.mm_projector(wt_features)
                        print(f"\n[DEBUG] MLPProjector forward (wt):\n  Input shape: {wt_features.shape}, dtype: {wt_features.dtype}\n    After projection (wt): {projected_features_wt.shape}")
                        wt_features = projected_features_wt
                
                if mode == 'text_only':
                    new_input_embeds = self.get_input_embeddings()(sub_input_ids)
                    new_attention_mask = sub_attention_mask
                    new_labels = sub_labels
                else:
                    # Step 3: Merge projected features with text embeddings
                    _, new_input_embeds, new_attention_mask, new_labels = self.prepare_inputs_labels_for_multimodal(
                        input_ids=sub_input_ids,
                        attention_mask=sub_attention_mask,
                        labels=sub_labels,
                        delta_features=delta_features,
                        wt_features=wt_features
                    )
                
                final_embeds_parts.append(new_input_embeds)
                final_attention_mask_parts.append(new_attention_mask)
                if new_labels is not None:
                    final_labels_parts.append(new_labels)
                original_indices_parts.extend(indices)

            print("\nCombining processed parts...")
            
            max_len = max(p.shape[1] for p in final_embeds_parts)
            print(f"\nPadding details:")
            print(f"  Max sequence length: {max_len}")
            print(f"  Number of parts to process: {len(final_embeds_parts)}")

            padded_embeds_parts = []
            padded_mask_parts = []
            padded_labels_parts = []

            for i, part in enumerate(final_embeds_parts):
                print(f"\nPart shapes before padding:\n  Part {i}:")
                print(f"    Embeddings: {part.shape}")
                print(f"    Attention mask: {final_attention_mask_parts[i].shape}")
                if final_labels_parts:
                    print(f"    Labels: {final_labels_parts[i].shape}")

                current_len = part.shape[1]
                if current_len == max_len:
                    print(f"\nPart {i} already at max length ({max_len})")
                    padded_embeds_parts.append(part)
                    padded_mask_parts.append(final_attention_mask_parts[i])
                    if final_labels_parts:
                        padded_labels_parts.append(final_labels_parts[i])
                    continue
                
                padding_len = max_len - current_len
                print(f"\nPadding part {i}:")
                print(f"  Current length: {current_len}")
                print(f"  Target length: {max_len}")
                print(f"  Padding length: {padding_len}")
                
                print("  Original shapes:")
                print(f"    Embeddings: {part.shape}")
                print(f"    Attention mask: {final_attention_mask_parts[i].shape}")
                if final_labels_parts:
                    print(f"    Labels: {final_labels_parts[i].shape}")

                pad_spec = (0, 0, 0, padding_len)
                padded_embeds = F.pad(part, pad_spec, 'constant', 0)
                padded_mask = F.pad(final_attention_mask_parts[i], (0, padding_len), 'constant', 0)
                
                padded_embeds_parts.append(padded_embeds)
                padded_mask_parts.append(padded_mask)

                if final_labels_parts:
                    padded_labels = F.pad(final_labels_parts[i], (0, padding_len), 'constant', IGNORE_INDEX)
                    padded_labels_parts.append(padded_labels)

                print("  Padded shapes:")
                print(f"    Embeddings: {padded_embeds.shape}")
                print(f"    Attention mask: {padded_mask.shape}")
                if final_labels_parts:
                    print(f"    Labels: {padded_labels.shape}")


            print("\nConcatenating padded parts:")
            print("Shapes before concatenation:")
            for i in range(len(padded_embeds_parts)):
                print(f"  Part {i}:")
                print(f"    Embeddings: {padded_embeds_parts[i].shape}, dtype: {padded_embeds_parts[i].dtype}, device: {padded_embeds_parts[i].device}")
                print(f"    Attention mask: {padded_mask_parts[i].shape}, dtype: {padded_mask_parts[i].dtype}, device: {padded_mask_parts[i].device}")
                if padded_labels_parts:
                    print(f"    Labels: {padded_labels_parts[i].shape}")


            final_inputs_embeds = torch.cat(padded_embeds_parts, dim=0)
            final_attention_mask = torch.cat(padded_mask_parts, dim=0)
            final_labels = torch.cat(padded_labels_parts, dim=0) if padded_labels_parts else None

            print("\nShapes after concatenation:")
            print(f"  Inputs embeds: {final_inputs_embeds.shape}, dtype: {final_inputs_embeds.dtype}, device: {final_inputs_embeds.device}")
            print(f"  Attention mask: {final_attention_mask.shape}, dtype: {final_attention_mask.dtype}, device: {final_attention_mask.device}")
            if final_labels is not None:
                print(f"  Labels: {final_labels.shape}")


            print("\nReordering to original batch order...")
            inverse_indices = torch.empty(len(original_indices_parts), dtype=torch.long, device=self.device)
            inverse_indices[torch.tensor(original_indices_parts, device=self.device)] = torch.arange(len(original_indices_parts), device=self.device)

            final_inputs_embeds = final_inputs_embeds[inverse_indices]
            final_attention_mask = final_attention_mask[inverse_indices]
            if final_labels is not None:
                final_labels = final_labels[inverse_indices]

            print("Final reordered shapes:")
            print(f"  inputs_embeds: {final_inputs_embeds.shape}")
            print(f"  attention_mask: {final_attention_mask.shape}")
            if final_labels is not None:
                print(f"  labels: {final_labels.shape}")

            fwd_input_ids = None
            fwd_inputs_embeds = final_inputs_embeds
            fwd_attention_mask = final_attention_mask
            fwd_labels = final_labels

        else:
            fwd_input_ids = input_ids
            fwd_inputs_embeds = inputs_embeds
            fwd_attention_mask = attention_mask
            fwd_labels = labels

        print("\nForwarding to parent class with shapes:")
        print(f"  input_ids: {'None' if fwd_input_ids is None else fwd_input_ids.shape}")
        print(f"  attention_mask: {fwd_attention_mask.shape}")
        print(f"  inputs_embeds: {'None' if fwd_inputs_embeds is None else fwd_inputs_embeds.shape}")
        print(f"  labels: {'None' if fwd_labels is None else fwd_labels.shape}")
        print("=== End Forward Pass Debug ===")

        return super().forward(
            input_ids=fwd_input_ids,
            attention_mask=fwd_attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=fwd_inputs_embeds,
            labels=fwd_labels,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )



    def freeze_protein_related_modules(self):
        """Freezes protein encoder and optionally multimodal modules based on config."""
        # Always freeze protein encoder
        if hasattr(self, 'protein_encoder') and self.protein_encoder is not None:
            for name, param in self.protein_encoder.named_parameters():
                param.requires_grad = False
            print("Protein encoder frozen.")

            # set to eval mode
            self.protein_encoder.eval()
            print("Protein encoder set to eval mode.")
        


    def unfreeze_pretrain_adapters(self):
        """Unfreeze adapters modules."""
        # Only unfreeze if tuning is enabled
        should_tune_adapter = getattr(self.config, "tune_mm_mlp_adapter", False)
        if hasattr(self, 'protein_config') and should_tune_adapter:
            if hasattr(self, 'mm_gated_cross_attention') and self.mm_gated_cross_attention is not None:
                for name, param in self.mm_gated_cross_attention.named_parameters():
                    param.requires_grad = True
                print("GCA unfrozen.")
            
            if hasattr(self, 'mm_resampler') and self.mm_resampler is not None:
                for name, param in self.mm_resampler.named_parameters():
                    param.requires_grad = True
                print("Resampler unfrozen.")
            
            if hasattr(self, 'mm_projector') and self.mm_projector is not None:
                for name, param in self.mm_projector.named_parameters():
                    param.requires_grad = True
                print("Projector unfrozen.")
            
            # Verify unfreeze status
            print("\n[DEBUG] Module trainable status after unfreezing:")
            for name, param in self.named_parameters():
                if any(module in name for module in ['mm_gated_cross_attention', 'mm_resampler', 'mm_projector']):
                    print(f"  {name}: requires_grad = {param.requires_grad}")
        else:
            print("[WARNING] tune_mm_mlp_adapter is False, keeping modules frozen. Set tune_mm_mlp_adapter=True to train these modules.")


    def freeze_pretrain_adapters(self):
        """Freeze adapters modules."""

        if hasattr(self, 'mm_gated_cross_attention') and self.mm_gated_cross_attention is not None:
            for name, param in self.mm_gated_cross_attention.named_parameters():
                param.requires_grad = False
            print("GCA frozen.")
        if hasattr(self, 'mm_resampler') and self.mm_resampler is not None:
            for name, param in self.mm_resampler.named_parameters():
                param.requires_grad = False
            print("Resampler frozen.")
        if hasattr(self, 'mm_projector') and self.mm_projector is not None:
            for name, param in self.mm_projector.named_parameters():
                param.requires_grad = False
            print("Projector frozen.")

class MLPProjector(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        hidden_dim = input_dim * 2  # As per mlp2x_gelu
        
        print(f"[DEBUG] Initializing MLPProjector:")
        print(f"  input_dim: {input_dim}")
        print(f"  hidden_dim: {hidden_dim}")
        print(f"  output_dim: {output_dim}")
        
        # First linear layer + GELU
        self.linear1 = nn.Linear(input_dim, hidden_dim)
        self.act = nn.GELU()
        # Second linear layer
        self.linear2 = nn.Linear(hidden_dim, output_dim)
        
        # Convert all weights to float32 for stability
        self.linear1.weight.data = self.linear1.weight.data.to(torch.float32)
        self.linear1.bias.data = self.linear1.bias.data.to(torch.float32)
        self.linear2.weight.data = self.linear2.weight.data.to(torch.float32)
        self.linear2.bias.data = self.linear2.bias.data.to(torch.float32)
        
        print("[DEBUG] MLPProjector parameter dtypes:")
        print(f"  linear1.weight: {self.linear1.weight.dtype}")
        print(f"  linear1.bias: {self.linear1.bias.dtype}")
        print(f"  linear2.weight: {self.linear2.weight.dtype}")
        print(f"  linear2.bias: {self.linear2.bias.dtype}")

    def forward(self, x):
        """
        x: (batch_size, 1, num_media_tokens, input_dim) from Resampler
        Returns: (batch_size, 1, num_media_tokens, output_dim)
        """
        orig_dtype = x.dtype
        # print(f"\n[DEBUG] MLPProjector forward: Input shape: {x.shape}, dtype: {x.dtype}")
        
        # Process through layers with proper dtype handling
        # Explicitly cast input to bfloat16 before linear layers
        x = self.linear1(x.to(torch.bfloat16))
        # print(f"[DEBUG] MLPProjector forward: After linear1 dtype: {x.dtype}")
        
        x = self.act(x)  # GELU preserves dtype
        # print(f"[DEBUG] MLPProjector forward: After GELU dtype: {x.dtype}")
        
        x = self.linear2(x.to(torch.bfloat16))
        # print(f"[DEBUG] MLPProjector forward: After linear2 dtype: {x.dtype}")
        
        # Return in original dtype
        out = x.to(orig_dtype)
        # print(f"[DEBUG] MLPProjector forward: Output dtype: {out.dtype}")
        return out


