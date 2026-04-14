import logging
import os
import glob
from copy import deepcopy
import safetensors

import torch
from torch.nn import SmoothL1Loss
from torch.nn import functional as F

from .model.auto import AutoDraftModelConfig, AutoEagleDraftModel

from verl.utils.torch_functional import (
    get_constant_schedule_with_warmup,
    get_cosine_schedule_with_warmup,
)


logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "INFO"))


class EagleTrainerBackend:
    def __init__(
        self,
        config,
        target_model_config
    ):
        self.config = config
        self.target_model_config = target_model_config

        self.criterion = SmoothL1Loss(reduction="none")

        # Ulysses Sequence Parallelism configuration
        self.ulysses_sequence_parallel_size = self.config.actor_rollout_ref.actor.get("ulysses_sequence_parallel_size", 1)
        self.use_ulysses_sp = self.ulysses_sequence_parallel_size > 1

    def build_model(self):
        """build draft model"""
        logger.info(f"Initializing Eagle model with type: {self.target_model_config.model_type}")
        eagle_cfg = self.config.actor_rollout_ref.drafter.eagle
        spec_model_path = eagle_cfg.spec_model_path
        config_path = os.path.join(spec_model_path, "config.json")

        # 1、加载 Config
        if os.path.exists(config_path):
            drafter_config = AutoDraftModelConfig.from_file(config_path)
        else:
            drafter_config = deepcopy(self.target_model_config)
            drafter_config.num_hidden_layers = 1
            drafter_config.torch_dtype = torch.bfloat16
            drafter_config.tie_word_embeddings = False
            drafter_config.architectures = ["LlamaForCausalLMEagle"]

        factory_cls = AutoEagleDraftModel
        
        drafter_module = factory_cls.from_config(drafter_config)

        # Initialize model
        if spec_model_path and os.path.exists(spec_model_path):
            drafter_module = factory_cls.from_pretrained(spec_model_path)

        
        # 复用主模型的Embedding和LM_Head
        target_model_path = self.config.actor_rollout_ref.model.path
        logger.info("Start load lm_head for eagle")
        drafter_module.load_lm_head(target_model_path)
        drafter_module.freeze_lm_head()
        
        drafter_module.load_embedding(target_model_path)
        drafter_module.freeze_embedding()

        del base_module

        return drafter_module, drafter_config

    def _load_checkpoint_files(self, path):
        """内部工具：支持 safetensors 和 bin 格式"""
        allow_patterns = ["*.safetensors", "*.bin", "*.pt"]
        hf_weights_files = []
        for pattern in allow_patterns:
            files = glob.glob(os.path.join(path, pattern))
            if files:
                hf_weights_files = files
                use_safetensors = (pattern == "*.safetensors")
                break

        state = {}
        if use_safetensors:
            # Load from safetensors files
            for file in hf_weights_files:
                with safetensors.safe_open(file, framework="pt", device="cpu") as f:
                    for name in f.keys():
                        state[name] = f.get_tensor(name)
        else:
            # Load from bin/pt files
            for file in hf_weights_files:
                file_state = torch.load(file, map_location="cpu", weights_only=True)
                state.update(file_state)
        return state
    
    def setup_optimizer(self, drafter_model, drafter_train_config):
        trainable_params = [p for p in drafter_model.parameters() if p.requires_grad]

        optimizer = torch.optim.AdamW(
            trainable_params,
            lr=drafter_train_config.optim.lr,
            betas=(0.9, 0.95),
            weight_decay=drafter_train_config.optim.get("weight_decay", 1e-2),
        )

        return optimizer


    def setup_scheduler(self, optimizer, train_cfg):
        total_steps = train_cfg.optim.get("total_training_steps", 0)
        num_warmup_steps = int(train_cfg.optim.get("lr_warmup_steps", 1000))
        warmup_style = train_cfg.optim.get("warmup_style", "constant")

        if warmup_style == "constant":
            return get_constant_schedule_with_warmup(
                optimizer=optimizer, num_warmup_steps=num_warmup_steps
            )
        elif warmup_style == "cosine":
            return get_cosine_schedule_with_warmup(
                optimizer=optimizer,
                num_warmup_steps=num_warmup_steps,
                num_training_steps=total_steps,
                min_lr_ratio=train_cfg.optim.get("min_lr_ratio", 0.0),
                num_cycles=train_cfg.optim.get("num_cycles", 0.5),
            )
        # elif warmup_style == "linear":
        #     return get_linear_schedule_with_warmup(
        #         optimizer=optimizer,
        #         num_warmup_steps=num_warmup_steps,
        #         num_training_steps=total_steps,
        #     )
        else:
            raise NotImplementedError(f"Warmup style {warmup_style} is not supported")
        
    def preprocess_individual_items(self, items, device, model_config):
        """
        针对单条数据：裁剪窗口、生成Mask、确保维度对齐
        """
        res = {'ids':[], 'h_states':[], 'masks': []}
        max_window = 512
        pad_id = int(getattr(model_config, "pad_token_id", 0) or 0)

        for item in items:
            # 1. 搬运到GPU
            ids = item["input_ids"].to(device, non_blocking=True)
            seq_len = ids.size(0)

            raw_h = item["hidden_states"]

            if isinstance(raw_h, (list, tuple)):
                # 将hidden_states进行拼接
                h_states = torch.cat(raw_h, dim=-1).to(device, dtype=torch.bfloat16)
            else:
                h_states = raw_h.to(device, dtype=torch.bfloat16)

            # 通过统一裁剪或填充将Hidden States与sequence length对齐
            h_len = h_states.size(0)
            if h_len < seq_len:
                # 批量 Padding
                padding = torch.zeros((seq_len - h_len, h_states.size(-1)), 
                                    device, dtype=h_states.dtype)
                h_states = torch.cat([h_states, padding], dim=0)
            else:
                h_states = h_states[:seq_len, :]

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
            res['masks'].append(item_loss_mask[start:end])
        
        return res
    
    def compute_loss(self, model, batch, _current_pad_size):
        """
        计算 Eagle 特有的V-Loss 和 P-Loss
        """
        # 前向传播
        outputs = model(
            input_ids=batch["input_ids"],
            hidden_states=batch["hidden_states"],
            attention_mask=batch["attention_mask"],
            position_ids=batch["position_ids"],
        )

        hidden_states = outputs["hidden_states"]
        logits = outputs["logits"]

        # Gather outputs if using Ulysses SP
        if getattr(self, "use_ulysses_sp", False):
            from verl.utils.ulysses import gather_outputs_and_unpad

            hidden_states = gather_outputs_and_unpad(
                hidden_states.squeeze(0),
                gather_dim=0,
                unpad_dim=0,
                padding_size=_current_pad_size,
            ).unsqueeze(0)

            logits = gather_outputs_and_unpad(
                logits.squeeze(0), gather_dim=0, unpad_dim=0, padding_size=_current_pad_size
            ).unsqueeze(0)

            target = gather_outputs_and_unpad(
                batch["target"].squeeze(0),
                gather_dim=0,
                unpad_dim=0,
                padding_size=_current_pad_size,
            ).unsqueeze(0)

            loss_mask = gather_outputs_and_unpad(
                batch["loss_mask"].squeeze(0),
                gather_dim=0,
                unpad_dim=0,
                padding_size=_current_pad_size,
            ).unsqueeze(0)
        else:
            target = batch["target"]
            loss_mask = batch["loss_mask"]

        # V-Loss：隐藏态回归损失
        vloss_all = self.criterion(hidden_states, target)  # [B,T,H]
        vloss_per_token = vloss_all.mean(dim=-1) # [B, T]

        # P-Loss: 概率分布对齐损失
        with torch.no_grad():
            target_p = F.softmax(model.lm_head(target), dim=1)

        log_prod = F.log_softmax(logits,  dim=-1)
        ploss_per_token = -(target_p * log_prod).sum(dim=-1) # [B, T]

        # 结合 Mask
        total_local_vloss = (vloss_per_token * loss_mask).sum()
        total_local_ploss = (ploss_per_token * loss_mask).sum()
        local_num_tokens = loss_mask.sum()

        # 读取权重并返回 Loss 字典
        w_v = float(self.config.get("vloss_weight", 0.5))
        w_p = float(self.config.get("ploss_weight", 0.5))

        return {
            "total_local_vloss": total_local_vloss,
            "total_local_ploss": total_local_ploss,
            "local_num_tokens": local_num_tokens,
            "v_weight": w_v,
            "p_weight": w_p
        }
