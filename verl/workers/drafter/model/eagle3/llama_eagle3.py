import math
import warnings
from typing import List, Optional, Tuple

import torch
from torch import nn
import torch.nn.functional as F
from transformers.models.llama.modeling_llama import LlamaMLP, LlamaRMSNorm, apply_rotary_pos_emb
# from yunchang.comm import SeqAllToAll4D

# from specforge.modeling.draft.flex_attention import (
#     compile_friendly_create_block_mask,
#     compile_friendly_flex_attention,
#     generate_eagle3_mask,
# )
from .base import Eagle3DraftModel
# from ...distributed import get_sp_ulysses_group

try:
    from flash_attn import flash_attn_func
except ImportError:
    flash_attn_func = None

def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)

class LlamaEagle3Attention(nn.Module):
    def __init__(self, config, layer_idx=None):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.scaling = self.head_dim**-0.5

        # Eagle3 拼接输入维度为 2 * hidden_size
        self.q_proj = nn.Linear(self.hidden_size * 2, self.num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(self.hidden_size * 2, self.num_key_value_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(self.hidden_size * 2, self.num_key_value_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=False)

        # Ulysses 并行组
        # self.sp_group = get_sp_ulysses_group()

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[any] = None,
        use_cache: bool = False,
        **kwargs,
    ) -> torch.Tensor:
        bsz, q_len, _ = hidden_states.size()

        # 1. 投影
        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        # 2. 准备并行维度转换 (Ulysses All-to-All)
        # 从 [B, L/P, H*D] 转换为 [B, L, H/P, D]
        # if self.sp_group is not None:
        #     # 序列并行通信
        #     # query_states = SeqAllToAll4D.apply(self.sp_group, query_states, 2, 1)
        #     # key_states = SeqAllToAll4D.apply(self.sp_group, key_states, 2, 1)
        #     # value_states = SeqAllToAll4D.apply(self.sp_group, value_states, 2, 1)
        # else:
        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

        # 3. RoPE (此处假设 rotary_emb 逻辑已通过父模型或全局定义处理)
        # 注意：在 Ulysses 中 position_ids 需要覆盖全局范围
        # query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin, position_ids)

        n_rep = self.num_heads // self.num_key_value_heads
        key_states = repeat_kv(key_states, n_rep)   # 变为 [bsz, 32, q_len, head_dim]
        value_states = repeat_kv(value_states, n_rep) # 变为 [bsz, 32, q_len, head_dim]

        # 4. 注意力计算
        if flash_attn_func is not None and attention_mask is None:
            # 转回 Flash Attention 要求的 [B, L, H, D]
            attn_output = flash_attn_func(
                query_states.transpose(1, 2),
                key_states.transpose(1, 2),
                value_states.transpose(1, 2),
                causal=True
            ).transpose(1, 2)
        else:
            # 使用 SDPA 或 Flex Attention
            attn_output = F.scaled_dot_product_attention(
                query_states, key_states, value_states, attn_mask=attention_mask
            )

        # 5. 逆并行转换 (All-to-All)
        # if self.sp_group is not None:
        #     # attn_output = SeqAllToAll4D.apply(self.sp_group, attn_output, 1, 2)
        # else:
        attn_output = attn_output.transpose(1, 2).contiguous()

        attn_output = attn_output.view(bsz, q_len, self.hidden_size)
        return self.o_proj(attn_output)


class LlamaEagle3DecoderLayer(nn.Module):
    def __init__(self, config, layer_idx):
        super().__init__()
        self.self_attn = LlamaEagle3Attention(config, layer_idx)
        self.mlp = LlamaMLP(config)
        self.input_layernorm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.hidden_norm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
            self,
            input_emb: torch.Tensor,
            hidden_states: torch.Tensor,
            attention_mask: Optional[torch.Tensor] = None,
            position_ids: Optional[torch.LongTensor] = None,
            **kwargs,
    ) -> torch.Tensor:
        residual = hidden_states

        # Eagle3 核心逻辑：双 Norm 拼接
        normed_h = self.hidden_norm(hidden_states)
        normed_e = self.input_layernorm(input_emb)
        combined = torch.cat([normed_h, normed_e], dim=-1)

        # Self Attention
        attn_output = self.self_attn(
            hidden_states=combined,
            attention_mask=attention_mask,
            position_ids=position_ids,
            **kwargs,
        )
        hidden_states = residual + attn_output

        # MLP
        residual = hidden_states
        hidden_states = residual + self.mlp(self.post_attention_layernorm(hidden_states))

        return hidden_states

class LlamaForCausalLMEagle3(Eagle3DraftModel):
    _supports_sdpa = True
    _no_split_modules = ["LlamaEagle3DecoderLayer"]

    def __init__(self, config):
        super().__init__(config)
        self.vocab_size = config.vocab_size
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        # 通常 Eagle3 只包含一个中间层
        self.midlayer = LlamaEagle3DecoderLayer(config, layer_idx=0)

        # 投影层：将主模型的 hidden 拼接后投影
        self.fc = nn.Linear(config.hidden_size * 3, config.hidden_size, bias=False)
        self.norm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # 词表缓存映射
        self.register_buffer("t2d", torch.zeros(config.vocab_size, dtype=torch.bool))
        self.register_buffer("d2t", torch.zeros(config.vocab_size, dtype=torch.int64))

    def embed_input_ids(self, input_ids):
        return self.embed_tokens(input_ids)

    def project_hidden_states(self, hidden_states):
        # 拼接 3 层 hidden states 的输出
        return self.fc(hidden_states)

    def backbone(self, **kwargs):
        return self.midlayer(**kwargs)
    
    def compute_logits(self, hidden_states_out):
        return self.lm_head(self.norm(hidden_states_out))

    def forward(
            self,
            input_ids: torch.LongTensor,
            hidden_states: torch.Tensor,
            attention_mask: Optional[torch.Tensor] = None,
            position_ids: Optional[torch.LongTensor] = None,
            **kwargs,
    ):
        input_emb = self.embed_input_ids(input_ids)
        bsz, seq_len = input_ids.shape

        # 调用 base.py 中定义的 prepare_decoder_attention_mask (或直接传入经过转换的 4D mask)
        causal_mask = self.prepare_decoder_attention_mask(
            attention_mask=attention_mask,
            input_shape=(bsz, seq_len),
            inputs_embeds=self.embed_input_ids(input_ids),
            past_key_values_length=0
        )
        
        hidden_states = self.project_hidden_states(hidden_states)
        hidden_states_out = self.backbone(
            input_emb=input_emb,
            hidden_states=hidden_states,
            attention_mask=causal_mask,
            position_ids=position_ids,
            **kwargs,
        )

        logits = self.compute_logits(hidden_states_out)
        return {
            "logits": logits,
            "hidden_states": hidden_states_out
        }