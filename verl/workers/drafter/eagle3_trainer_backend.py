import logging
import os
from copy import deepcopy

import torch
import torch.nn as nn
from torch.nn import functional as F

from .model.auto import AutoDraftModelConfig, AutoEagle3DraftModel
from .eagle_trainer_backend import EagleTrainerBackend
from .model.target.target_head import TargetHead
from verl.utils.fsdp_utils import get_device_id
from verl.utils.device import get_device_name


logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "INFO"))

device_name = get_device_name()


def _scatter_sparse_logprobs(logprobs: torch.Tensor, indices: torch.Tensor, vocab_size: int) -> torch.Tensor:
    full_logits = torch.full(
        (logprobs.size(0), vocab_size),
        float("-inf"),
        dtype=logprobs.dtype,
        device=logprobs.device,
    )
    if logprobs.numel() == 0:
        return full_logits

    valid = torch.isfinite(logprobs) & (indices >= 0) & (indices < vocab_size)
    if not valid.any():
        return full_logits

    row_indices = torch.arange(logprobs.size(0), device=logprobs.device).unsqueeze(1).expand_as(indices)
    full_logits[row_indices[valid], indices[valid]] = logprobs[valid]
    return full_logits


def reconstruct_sparse_logprob_view(target_topk_logprobs, topk, vocab_size):
    if isinstance(target_topk_logprobs, torch.Tensor):
        if target_topk_logprobs.dim() != 3 or target_topk_logprobs.size(-1) < 2:
            raise ValueError(
                "target_topk_logprobs must have shape [seq, topk, 2+] when reconstructing a sparse logprob view, "
                f"but got shape={tuple(target_topk_logprobs.shape)}"
            )
        if target_topk_logprobs.numel() == 0:
            return torch.empty(
                target_topk_logprobs.shape[0],
                vocab_size,
                dtype=target_topk_logprobs.dtype,
                device=target_topk_logprobs.device,
            )
        logprobs = target_topk_logprobs[..., 0]
        indices = target_topk_logprobs[..., 1].to(torch.long)
        return _scatter_sparse_logprobs(logprobs, indices, vocab_size)

    rows = []
    for step_top_logprobs in target_topk_logprobs:
        if isinstance(step_top_logprobs, dict):
            entries = list(step_top_logprobs.values())
        else:
            entries = list(step_top_logprobs or [])

        row = []
        for entry in entries[:topk]:
            if isinstance(entry, dict):
                logprob = entry.get("logprob", entry.get("log_probs", entry.get("log_prob")))
                token_id = entry.get("token_id", entry.get("idx", entry.get("id")))
            elif isinstance(entry, (list, tuple)) and len(entry) >= 2:
                logprob, token_id = entry[0], entry[1]
            else:
                continue

            try:
                row.append([float(logprob), int(token_id)])
            except (TypeError, ValueError):
                continue

        if not row:
            continue
        while len(row) < topk:
            row.append([float("-inf"), -1])
        rows.append(row)

    if not rows:
        return torch.empty((0, vocab_size), dtype=torch.float32)

    rows_tensor = torch.tensor(rows, dtype=torch.float32)
    logprobs = rows_tensor[..., 0]
    indices = rows_tensor[..., 1].to(torch.long)
    return _scatter_sparse_logprobs(logprobs, indices, vocab_size)


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
            target_device = torch.device(f"{device_name}:{get_device_id()}") if device_name != "cpu" else torch.device("cpu")
            self.target_model = self._build_target_model(target_model_path).to(target_device).eval()
            for param in self.target_model.parameters():
                param.requires_grad_(False)

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
        res = {'ids':[], 'h_states':[], 'masks': [], 'last_h_states': [], 'target_logprobs': []}
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
            full_len = ids.size(0)
            if "loss_mask" not in item:
                item_loss_mask = torch.zeros_like(ids, dtype=torch.float32)
                if "prompts" in item and "responses" in item:
                    prompt_len = item["prompts"].size(0)
                    response_len = item["responses"].size(0)
                    for j in range(response_len):
                        token_idx = prompt_len + j
                        if token_idx < full_len and item["responses"][j] != pad_id:
                            item_loss_mask[token_idx] = 1.0
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
            if item.get("target_logprobs") is not None:
                res["target_logprobs"].append(item["target_logprobs"].to(device, dtype=torch.float32)[start:end])
        
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
        use_logits = self.config.rollout.drafter.training.use_logits
        draft_model = model.module if hasattr(model, "module") else model

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
                    l.squeeze(0),
                    gather_dim=0,
                    unpad_dim=0,
                    padding_size=_current_pad_size,
                ).unsqueeze(0) for l in all_step_logits
            ]

            all_step_position_mask = [
                gather_outputs_and_unpad(
                    m.squeeze(0), gather_dim=0, unpad_dim=0, padding_size=_current_pad_size
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
                target_scores = reconstruct_sparse_logprob_view(
                    target_topk_logprobs.squeeze(0),
                    topk=self.config.rollout.drafter.training.logits_topk,
                    vocab_size=self.vocab_size,
                ).unsqueeze(0)
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
                    target_scores = self.target_model(last_hidden_states)
        else:
            all_step_logits = all_step_logits
            all_step_position_mask = all_step_position_mask
            loss_mask = loss_mask
            if use_logits:
                target_topk_logprobs = batch["target_logprobs"]
                target_scores = reconstruct_sparse_logprob_view(
                    target_topk_logprobs.squeeze(0),
                    topk=self.config.rollout.drafter.training.logits_topk,
                    vocab_size=self.vocab_size,
                ).unsqueeze(0)
            else:
                if last_hidden_states is None:
                    raise ValueError("last_hidden_states is required when use_target_model=False")
                with torch.no_grad():
                    target_scores = self.target_model(last_hidden_states)
        
        length = len(all_step_logits)
        seq_length = input_ids.shape[1]

        target_p_padded, position_mask = self._compute_target_p_padded(
            target_scores=target_scores,
            t2d=draft_model.t2d,
            loss_mask=loss_mask,
            length=length,
        )

        # Clean up large tensors to free memory
        del target_scores

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
    
    def _compute_target_p_padded(self, target_scores, t2d, loss_mask, length):
        with torch.no_grad():
            target_p, position_mask = self._compute_target_p(
                target_scores=target_scores,
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


    def _compute_target_p(self, target_scores, t2d, loss_mask):
        target_subset_scores = target_scores
        target_max_token = target_subset_scores.argmax(-1)
        target_mask = t2d[target_max_token]
        target_mask = target_mask[..., None].int()
        position_mask = target_mask * loss_mask
        target_subset_scores = target_subset_scores[..., t2d]
        target_subset_scores = target_subset_scores.float()
        target_p = nn.Softmax(dim=2)(target_subset_scores)
        target_p = target_p.detach()
        return target_p, position_mask
        

        
