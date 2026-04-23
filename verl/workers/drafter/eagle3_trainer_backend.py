import logging
import os
from copy import deepcopy

import torch
import torch.nn as nn
from torch.nn import functional as F
import numpy as np

from .model.auto import AutoDraftModelConfig, AutoEagle3DraftModel
from .eagle_trainer_backend import EagleTrainerBackend
from .model.target.target_head import TargetHead


logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "INFO"))


def reconstruct_logits(target_topk_logprobs, topk, vocab_size):

    dtype = [('logprobs', 'f4'), ('idx', 'i4'), ('none', 'O')]

    flat_topk_logprobs_list = np.array([item for step in target_topk_logprobs for item in step], dtype=dtype)

    logprobs_flat = flat_topk_logprobs_list['logprobs']
    indices_flat = flat_topk_logprobs_list['idx']

    l = len(target_topk_logprobs)

    final_topk_logprobs = torch.from_numpy(logprobs_flat).reshape(l, topk)
    final_topk_indices = torch.from_numpy(indices_flat).reshape(l, topk)
    # logprob转为概率
    final_topk_logits = torch.exp(final_topk_logprobs)
    # 初始化全为0的张量
    full_logits = torch.full((l, vocab_size), float(0), device=final_topk_logits.device)
    # 构建行索引
    row_indeces = torch.arange(l, device=final_topk_logits.device).unsqueeze(1).expand(-1, topk)
    # 填入logits
    full_logits[row_indeces, final_topk_indices] = final_topk_logits

    return full_logits


class Eagle3TrainerBackend(EagleTrainerBackend):

    def __init__(
        self,
        config,
        target_model_config
    ):
        super().__init__(config, target_model_config)

        self.target_model = None
        self.vocab_size = None

    @property
    def model_type(self):
        return "eagle3"

    def build_model(self):
        """build eagle3 draft model"""
        logger.info(f"Initializing Eagle3 model with type: {self.target_model_config.model_type}")
        spec_model_path = self.config.rollout.drafter.model_path
        config_path = os.path.join(spec_model_path, "config.json")

        # 1、加载 Config
        if os.path.exists(config_path):
            drafter_config = AutoDraftModelConfig.from_file(config_path)
        else:
            drafter_config = deepcopy(self.target_model_config)
            drafter_config.num_hidden_layers = 1
            drafter_config.torch_dtype = torch.bfloat16
            drafter_config.tie_word_embeddings = False
            drafter_config.architectures = ["LlamaForCausalLMEagle3"]

        self.vocab_size = drafter_config.vocab_size

        factory_cls = AutoEagle3DraftModel
        
        drafter_module = factory_cls.from_config(drafter_config)

        # Initialize model
        if spec_model_path and os.path.exists(spec_model_path):
            drafter_module = factory_cls.from_pretrained(spec_model_path)

        
        # 复用主模型的Embedding和LM_Head
        target_model_path = self.config.model.path
            
        drafter_module.load_embedding(target_model_path)
        drafter_module.freeze_embedding()
        
        # EAGLE-3 特有逻辑：加载词表映射
        # mapping_path = getattr(eagle_cfg, "vocab_mapping_path", None)

        # if mapping_path and os.path.exists(mapping_path):
        #     drafter_module.load_vocab_mapping(mapping_path)
        #     logger.info(f"Loaded EAGLE-3 vocab mapping from {mapping_path}")

        use_logits = self.config.rollout.drafter.training.get("use_logits", False)
        if not use_logits:
            self.target_model = self._build_target_model(target_model_path)

        return drafter_module, drafter_config


    def _build_target_model(self, target_model_path: str):
        """
        构建主模型，先实现根据last_hidden_states构建主模型线性层，直接使用主模型后续看要不要实现
        """
        target_head = TargetHead.from_pretrained(
            model_path=target_model_path,
        )

        return target_head
    
    def preprocess_individual_items(self, items, device, model_config):
        """
        针对单条数据：裁剪窗口、生成Mask、确保维度对齐
        """
        res = {'ids':[], 'h_states':[], 'masks': [], 'last_h_states': []}
        max_window = 512
        pad_id = int(getattr(model_config, "pad_token_id", 0) or 0)
        h_dim = model_config.hidden_size

        for item in items:
            # 1. 搬运到GPU
            ids = item["input_ids"].to(device, non_blocking=True)

            raw_h = item["hidden_states"]

            if isinstance(raw_h, (list, tuple)):
                # 将hidden_states进行拼接
                full_h = torch.cat(raw_h, dim=-1).to(device, dtype=torch.bfloat16)
            else:
                full_h = raw_h.to(device, dtype=torch.bfloat16)

            h_states = full_h[:, :3*h_dim]
            last_h_states = full_h[:, 3*h_dim:]

            # Compute loss_mask if not present (for DataBuffer items)
            if "loss_mask" not in item:
                item_loss_mask = torch.zeros_like(ids, dtype=torch.float32)
                if "prompts" in item and "responses" in item:
                    prompt_len = item["prompts"].size(0)
                    response_len = item["responses"].size(0)
                    for j in range(response_len):
                        if item["responses"][j] != pad_id:
                            item_loss_mask[prompt_len + j] = 1.0
                elif "responses" in item:
                    response_start = full_len - item["responses"].size(0)
                    response_mask = (item["responses"] != pad_id).float()
                    item_loss_mask[response_start:] = response_mask
                else:
                    # If no response info, assume all tokens are valid
                    item_loss_mask[:] = 1.0
            else:
                item_loss_mask = item["loss_mask"]
            
            # Select window around response tokens
            nonzero = torch.nonzero(item_loss_mask)
            full_len = ids.size(0)

            if nonzero.numel() > 0:
                # 获取 Response 的起止点
                r_start, r_end = nonzero[0, 0], nonzero[-1, 0] + 1
                # 尽量让窗口覆盖 Response 区域
                start_max = max(0, full_len - max_window)
                start = torch.clamp(r_start - (max_window // 2), min=0, max=start_max).item()
                end = min(start + max_window, full_len)
            else:
                start = max(0, full_len - max_window)
                end = full_len

            res['ids'].append(ids[start:end])
            res['h_states'].append(h_states[start:end])
            res['last_h_states'].append(last_h_states[start:end])
            res['masks'].append(item_loss_mask[start:end])
        
        return res

    def compute_loss(self, model, batch, _current_pad_size):
        """
        Compute Eagle3 multi-step prediction losses
        """
        input_ids = batch["input_ids"]
        hidden_states = batch["hidden_states"]
        last_hidden_states = batch.get("last_hidden_states", None)
        attention_mask = batch["attention_mask"]
        loss_mask = batch["loss_mask"]
        position_ids = batch["position_ids"]
        use_logits = self.config.drafter.training.use_logits

        # 前向传播
        outputs = model(
            input_ids=input_ids,
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            loss_mask=loss_mask,
            position_ids=position_ids,
        )

        all_step_logits = outputs["logits"]
        all_step_position_mask = outputs["position_masks"]

        # Gather outputs if using Ulysses SP
        if getattr(self, "use_ulysses_sp", False):
            from verl.utils.ulysses import gather_outputs_and_unpad

            all_step_logits = [
                gather_outputs_and_unpad(
                    hidden_states.squeeze(0),
                    gather_dim=0,
                    unpad_dim=0,
                    padding_size=_current_pad_size,
                ).unsqueeze(0) for l in all_step_logits
            ]

            all_step_position_mask = [
                gather_outputs_and_unpad(
                    logits.squeeze(0), gather_dim=0, unpad_dim=0, padding_size=_current_pad_size
                ).unsqueeze(0) for m in all_step_position_mask
            ]

            loss_mask = gather_outputs_and_unpad(
                loss_mask.squeeze(0),
                gather_dim=0,
                unpad_dim=0,
                padding_size=_current_pad_size,
            ).unsqueeze(0)

            if use_logits:
                target_topk_logprobs = gather_outputs_and_unpad(
                    batch["target_logprobs"].squeeze(0),
                    gather_dim=0,
                    unpad_dim=0,
                    padding_size=_current_pad_size,
                ).unsqueeze(0)
                target_logits = reconstruct_logits(target_topk_logprobs[1:], topk=self.config.rollout.drafter.training.logits_topk, vocab_size=self.vocab_size)
            else:
                if last_hidden_states is None:
                    raise ValueError("last_hidden_states is required when use_target_model=False")
                last_hidden_states = gather_outputs_and_unpad(
                    last_hidden_states.squeeze(0),
                    gather_dim=0,
                    unpad_dim=0,
                    padding_size=_current_pad_size,
                ).unsqueeze(0)
                with torch.no_grad():
                    target_logits = self.target_model(last_hidden_states)
        else:
            all_step_logits = all_step_logits
            all_step_position_mask = all_step_position_mask
            loss_mask = loss_mask
            if use_logits:
                target_topk_logits = batch["target_logits"]
                target_logits = reconstruct_logits(target_topk_logits[1:], topk=self.config.drafter.training.logits_topk, vocab_size=self.vocab_size)
            else:
                if last_hidden_states is None:
                    raise ValueError("last_hidden_states is required when use_target_model=False")
                with torch.no_grad():
                    target_logits = self.target_model(last_hidden_states)
        
        length = len(all_step_logits)
        seq_length = input_ids.shape[1]

        target_p_padded, position_mask = self._compute_target_p_padded(
            target=target_logits,
            t2d=model.t2d,
            loss_mask=loss_mask,
            length=length,
        )

        # Clean up large tensors to free memory
        del target_logits

        total_local_ploss = 0
        gamma = 0.8
        
        # 预处理
        for idx in range(length):
            # 切片对齐：取当前步对应的未来目标
            # 这里的关键是 target_p 会随着 idx 往后偏移
            target_p = target_p_padded[:, idx : idx + seq_length, :].contiguous()
            logits = all_step_logits[idx]
            position_mask = all_step_position_mask[idx]
            if position_mask.dim() == 3:
                position_mask = position_mask.squeeze(-1)

            log_probs = F.log_softmax(logits, dim=-1)
            
            step_loss_sum = (-(target_p * log_probs).sum(dim=-1) * position_mask).sum()
            
            # 应用Eagle3的时间步衰减
            total_local_ploss += (gamma ** idx) * step_loss_sum

        return {
            "total_local_vloss": torch.tensor(0.0, device=input_ids.device),
            "total_local_ploss": total_local_ploss,
            "local_num_tokens": loss_mask.sum(),
            "v_weight": 0.0,
            "p_weight": 1.0 / length
        }
    
    def _compute_target_p_padded(self, target, t2d, loss_mask, length):
        with torch.no_grad():
            target_p, position_mask = self._compute_target_p(
                target=target,
                t2d=t2d,
                loss_mask=loss_mask,
            )

            assert len(target_p.shape) == 3
            target_p_padded = F.pad(
                target_p,
                pad=(0, 0, 0, length),
                mode="constant",
                # For bitwise equality with previous code
                value=1 / target_p.shape[-1],
            )

            return target_p_padded, position_mask


    def _compute_target_p(self, target, t2d, loss_mask):
        target_head = target
        target_max_token = target_head.argmax(-1)
        target_mask = t2d[target_max_token]
        target_mask = target_mask[..., None].int()
        position_mask = target_mask * loss_mask
        target_head = target_head[..., t2d]
        target_head = target_head.float()
        target_p = nn.Softmax(dim=2)(target_head)
        target_p = target_p.detach()
        return target_p, position_mask
        

        
