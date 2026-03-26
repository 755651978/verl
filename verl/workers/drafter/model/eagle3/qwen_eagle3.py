import torch
from torch import nn
from typing import Optional
from transformers import Qwen2Config
from .base import Eagle3DraftModel

# Dynamic import to handle different transformer versions
try:
    from transformers.models.qwen3.modeling_qwen3 import Qwen3MLP, Qwen3Attention, Qwen3RMSNorm
    HAS_QWEN3 = True
except ImportError:
    HAS_QWEN3 = False

from transformers.models.qwen2.modeling_qwen2 import Qwen2MLP, Qwen2Attention, Qwen2RMSNorm

class QwenDraftLayer(nn.Module):
    """Specific Decoder Layer for Qwen-family Eagle Drafters."""
    def __init__(self, config: Qwen2Config, layer_idx: int):
        super().__init__()
        # Determine which operators to use based on config
        is_qwen3 = "Qwen3" in config.__class__.__name__
        
        if is_qwen3 and HAS_QWEN3:
            attn_cls, mlp_cls, norm_cls = Qwen3Attention, Qwen3MLP, Qwen3RMSNorm
            # Qwen3 often disables bias by default
            attn_bias = getattr(config, "attention_bias", False)
        else:
            attn_cls, mlp_cls, norm_cls = Qwen2Attention, Qwen2MLP, Qwen2RMSNorm
            attn_bias = True # Qwen2.5 default

        self.self_attn = attn_cls(config=config, layer_idx=layer_idx)
        
        # Eagle-3 Special: Input is concatenated [Embeddings; Hidden_States]
        # We re-define QKV projections to handle 2 * hidden_size input
        head_dim = getattr(config, "head_dim", config.hidden_size // config.num_attention_heads)
        self.self_attn.q_proj = nn.Linear(config.hidden_size * 2, config.num_attention_heads * head_dim, bias=attn_bias)
        self.self_attn.k_proj = nn.Linear(config.hidden_size * 2, config.num_key_value_heads * head_dim, bias=attn_bias)
        self.self_attn.v_proj = nn.Linear(config.hidden_size * 2, config.num_key_value_heads * head_dim, bias=attn_bias)

        self.mlp = mlp_cls(config)
        self.input_layernorm = norm_cls(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = norm_cls(config.hidden_size, eps=config.rms_norm_eps)
        self.hidden_norm = norm_cls(config.hidden_size, eps=config.rms_norm_eps)

    def forward(self, input_embeds, hidden_states, **kwargs):
        residual = hidden_states

        # 1. Normalize and Fuse
        input_embeds = self.input_layernorm(input_embeds)
        hidden_states = self.hidden_norm(hidden_states)
        fused_states = torch.cat([input_embeds, hidden_states], dim=-1)

        # 2. Attention
        attn_output = self.self_attn(hidden_states=fused_states, **kwargs)[0]
        hidden_states = residual + attn_output

        # 3. MLP
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        return (residual + hidden_states,)


class QwenEagle(Eagle3DraftModel):
    """Eagle-3 Model for Qwen2 / Qwen2.5 / Qwen3."""

    def __init__(self, config):
        super().__init__(config)
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, config.pad_token_id)

        is_qwen3 = "Qwen3" in config.__class__.__name__
        norm_cls = Qwen3RMSNorm if (is_qwen3 and HAS_QWEN3) else Qwen2RMSNorm
        self.norm = norm_cls(config.hidden_size, eps=config.rms_norm_eps)

        self.lm_head = nn.Linear(config.hidden_size, config.draft_vocab_size, bias=False)

        # Feature fusion from Target Model
        target_h = getattr(config, "target_hidden_size", config.hidden_size)
        self.fc = nn.Linear(target_h * 3, config.hidden_size, bias=False)

        # Initialize the 1 or 2 decoder layers
        num_layers = getattr(config, "num_layers", 1)
        self.layers = nn.ModuleList([QwenDraftLayer(config, i) for i in range(num_layers)])

        self.post_init()