import glob
import json
import logging
import os
from typing import Optional

import torch
import torch.nn as nn
from huggingface_hub import snapshot_download
from safetensors import safe_open
from transformers import AutoConfig


logger = logging.getLogger(__name__)


class TargetHead(nn.Module):
    """
    将离线存储的隐藏状态还原为主模型原本会输出的logits
    只提取主模型的最后的一个线性层
    """
    def __init__(self, model_path, trust_remote_code: bool = False):
        super().__init__()
        self.config = AutoConfig.from_pretrained(
            model_path, trust_remote_code=trust_remote_code
        )
        self.text_config = getattr(self.config, "text_config", self.config)

        self.hidden_size = self.text_config.hidden_size
        self.vocab_size = self.text_config.vocab_size

        self.fc = nn.Linear(self.hidden_size, self.vocab_size, bias=False)

    @classmethod
    def from_pretrained(
        cls,
        model_path,
        lm_head_key: str = "lm_head.weight",
        cache_dir: Optional[str] = None,
        trust_remote_code: bool = False,
    ) -> "TargetHead":
        target_head = cls(model_path, trust_remote_code=trust_remote_code)
        target_head.load_weights(
            model_path=model_path,
            lm_head_key=lm_head_key,
            cache_dir=cache_dir,
        )
        target_head.freeze_weights()
        target_head = target_head.eval().cuda().to(torch.bfloat16)
        return target_head

    @torch.no_grad()
    def load_weights(
        self,
        model_path,
        lm_head_key: str = "lm_head.weight",
        cache_dir: Optional[str] = None,
    ):
        if os.path.exists(model_path):
            self.model_path = model_path
        else:
            self.model_path = snapshot_download(repo_id=model_path)

        # model_path is a local directory
        # check if there is file ending with index.json
        glob_path = os.path.join(self.model_path, "*.index.json")
        index_json_path = glob.glob(glob_path)

        if len(index_json_path) == 0:
            raise FileNotFoundError(f"No index.json file found in {self.model_path}")
        if len(index_json_path) > 1:
            raise FileNotFoundError(
                f"Multiple index.json files found in {self.model_path}"
            )
        index_json_path = index_json_path[0]

        with open(index_json_path, "r") as f:
            index_json = json.load(f)

        weight_map = index_json["weight_map"]
        candidate_keys = [
            lm_head_key,
            "model.lm_head.weight",
            "base_model.model.lm_head.weight",
            "model.embed_tokens.weight",
            "embed_tokens.weight",
            "base_model.model.model.embed_tokens.weight",
        ]
        selected_key = next((key for key in candidate_keys if key in weight_map), None)
        if selected_key is None:
            available = list(weight_map.keys())
            raise KeyError(
                f"Cannot find target lm_head or tied embedding weight in {self.model_path}. "
                f"Tried {candidate_keys}. Available keys sample: {available[:20]}"
            )
        if selected_key != lm_head_key:
            logger.warning(
                "Target lm_head key %s not found in %s; using %s as target head weight.",
                lm_head_key,
                self.model_path,
                selected_key,
            )

        ckpt_file = weight_map[selected_key]

        if ckpt_file.endswith(".safetensors"):
            with safe_open(
                os.path.join(self.model_path, ckpt_file), framework="pt"
            ) as f:
                lm_head = f.get_tensor(selected_key)
        else:
            state_dict = torch.load(os.path.join(self.model_path, ckpt_file))
            lm_head = state_dict[selected_key]
        if tuple(lm_head.shape) != tuple(self.fc.weight.shape):
            raise ValueError(
                f"Target head weight shape mismatch for {selected_key}: "
                f"checkpoint={tuple(lm_head.shape)} expected={tuple(self.fc.weight.shape)}"
            )
        self.fc.weight.copy_(lm_head)

    def freeze_weights(self):
        for param in self.fc.parameters():
            param.requires_grad = False

    def forward(self, hidden_states):
        return self.fc(hidden_states)
    
    # def preprocess(self, input_ids, target, loss_mask):
    #     # apply pading
    #     target = padding(target, left=False)
    #     input_ids = padding(input_ids, left=False)
    #     loss_mask = loss_mask[..., None]
    #     return input_ids, target, loss_mask
