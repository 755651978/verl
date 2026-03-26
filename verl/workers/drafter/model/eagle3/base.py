import glob
import json
import os
import torch
from huggingface_hub import snapshot_download
from transformers import PreTrainedModel
from safetensors import safe_open
from transformers.cache_utils import Cache
from typing import Optional, Tuple, List

class Eagle3DraftModel(PreTrainedModel):
    """
    Base class for Eagle-style draft models.
    Eagle differs from standard draft models by incorporating target model 
    hidden states through a feature fusion layer (FC).
    """
    def __init__(self, config):
        super().__init__(config)

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        """
        Embed the input ids.
        """

    def project_hidden_states(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        Project the concatenated hidden states from the high, medium and low layers to the target hidden size.
        """

    def backbone(
        self, 
        input_embeds: torch.Tensor,
        hidden_states: torch.Tensor,
        cache_hidden: List[List[torch.Tensor]],
        attention_mask: torch.Tensor,
        position_ids: torch.Tensor,
        past_key_values: Optional[Cache] = None,
        use_cache: bool = True,
    ) -> torch.Tensor:
        """
        The baclbone of the draft model.
        """

    def compute_logits(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        """
    
    def load_embedding(self, model_path: str, embedding_key: str = "model.embed_tokens.weight") -> None:
        """从本地或 HF 仓库加载 Embedding 权重，确保草稿模型与原模型语义对齐。"""
        if not os.path.isdir(model_path):
            model_path = snapshot_download(repo_id=model_path)
            
        index_file = glob.glob(os.path.join(model_path, "*.index.json"))
        if index_file:
            with open(index_file[0], "r") as f:
                index_json = json.load(f)
            ckpt_file = index_json["weight_map"][embedding_key]
        else:
            # 兼容单文件模型
            ckpt_file = (glob.glob(os.path.join(model_path, "*.safetensors")) or 
                         glob.glob(os.path.join(model_path, "*.bin")))[0]

        full_path = os.path.join(model_path, ckpt_file)
        if full_path.endswith(".safetensors"):
            with safe_open(full_path, framework="pt") as f:
                self.embed_tokens.weight.copy_(f.get_tensor(embedding_key))
        else:
            state_dict = torch.load(full_path, map_location="cpu")
            self.embed_tokens.weight.copy_(state_dict[embedding_key])

    def load_vocab_mapping(self, file_path: str) -> None:
        """加载词表映射（t2d/d2t），用于大词表到草稿词表的映射。"""
        data = torch.load(file_path, map_location="cpu") 
        self.t2d.copy_(data["t2d"]) 
        self.d2t.copy_(data["d2t"])

    def prepare_decoder_attention_mask(self, attention_mask, input_shape, inputs_embeds, past_key_values_length):
        # input_shape: (batch_size, query_length)
        bsz, q_len = input_shape
        dtype = inputs_embeds.dtype
        device = inputs_embeds.device
        
        # 1. 创建基础的因果掩码 (Causal Mask)
        # 形状为 (q_len, q_len + past_kv_len)
        combined_attention_mask = torch.full(
            (q_len, q_len), torch.finfo(dtype).min, device=device
        )
        mask_cond = torch.arange(combined_attention_mask.size(-1), device=device)
        combined_attention_mask.masked_fill_(mask_cond < (mask_cond + 1).view(q_len, 1), 0)
        combined_attention_mask = combined_attention_mask.to(dtype)

        # 2. 如果有 KV Cache，在左侧补 0 (表示之前的 token 全可见)
        if past_key_values_length > 0:
            combined_attention_mask = torch.cat(
                [torch.zeros(q_len, past_key_values_length, dtype=dtype, device=device), 
                combined_attention_mask], dim=-1
            )

        # 3. 广播到 4D: (bsz, 1, q_len, total_seq_len)
        # 这是 SDPA 算子要求的标准维度
        combined_attention_mask = combined_attention_mask[None, None, :, :].expand(
            bsz, 1, q_len, q_len + past_key_values_length
        )

        # 4. 合并传入的填充掩码 (Attention Mask from DataLoader)
        if attention_mask is not None:
            if attention_mask.dim() == 2:
                 # attention_mask 是 (bsz, total_seq_len)，1 是有效，0 是 padding
                # 转换为 (bsz, 1, 1, total_seq_len) 并取反相加
                expanded_attn_mask = (1.0 - attention_mask[:, None, None, :].to(dtype)) * torch.finfo(dtype).min
            else:
                expanded_attn_mask = attention_mask
           
            combined_attention_mask = combined_attention_mask + expanded_attn_mask

        return combined_attention_mask