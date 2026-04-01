import logging
import os
import glob
from copy import deepcopy
import safetensors

import torch
from torch.nn import SmoothL1Loss
from torch.nn import functional as F

from .model.eagle3.eagle import LlamaForCausalLMEagle3

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
        target_model_config,
        target_model_fsdp
    ):
        self.config = config
        self.target_model_config = target_model_config
        self.target_model_fsdp = target_model_fsdp

        self.criterion = SmoothL1Loss(reduction="none")

        # Ulysses Sequence Parallelism configuration
        self.ulysses_sequence_parallel_size = self.config.actor_rollout_ref.actor.get("ulysses_sequence_parallel_size", 1)
        self.use_ulysses_sp = self.ulysses_sequence_parallel_size > 1

    def build_model(self):
        """build draft model"""
        num_layers_to_concat = getattr(self.config.actor_rollout_ref.drafter.eagle, "num_layers_to_concat", 1)
        logger.info(f"Initializing Eagle model with type: {self.target_model_config.model_type}")

        # 1、复制主模型配置并修改为单层 Eagle 结构
        config = deepcopy(self.target_model_config)
        config.num_hidden_layers = 1
        config.torch_dtype = torch.bfloat16
        config.tie_word_embeddings = False
        config.num_layers_to_concat = num_layers_to_concat

        # todo 对于eagle3，引入了draft_vocab_size
        # config.draft_vocab_size

        model_type = getattr(config, "model_type", "llama") 
        
        # 2、实例化模型
        if model_type.lower() in ["llama", "qwen2", "qwen2.5", "qwen3"]:
            model_class = LlamaForCausalLMEagle3
        else:
            raise ValueError(f"Unsupported model type for eagle: {model_type}")
        
        drafter_module = model_class(config=config)

        spec_model_path = self.config.actor_rollout_ref.drafter.eagle.spec_model_path
        # 3、权重加载与处理
        if spec_model_path and os.path.exists(spec_model_path):
            logger.info(f"Loading eagle model from checkpoint: {spec_model_path}")
            state = self._load_checkpoint_files(spec_model_path)

            # 兼容性更名
            model_dict = drafter_module.state_dict()
            matched_checkpoint = {}

            for key, value in state.items():
                # 处理 key 的命名空间兼容性
                new_key = key if (key.startswith("model.") or key.startswith("lm_head")) else f"model.{key}"

                if new_key in model_dict:
                    # 【核心修复】：检查 Checkpoint 里的权重维度和当前模型是否一致
                    if value.shape == model_dict[new_key].shape:
                        matched_checkpoint[new_key] = value
                    else:
                        # 如果维度不一致（比如 32000 vs 128256），跳过这个 key，不让它报 RuntimeError
                        logger.warning(
                            f"Dimension mismatch for {new_key}: Checkpoint {value.shape} vs Model {model_dict[new_key].shape}. "
                            f"Skipping this weight (it will be randomly initialized)."
                        )
                else:
                    logger.debug(f"Key {new_key} not found in model_class, skipping.")

            # 使用 strict=False，允许跳过刚才那些维度不匹配的词表权重
            drafter_module.load_state_dict(matched_checkpoint, strict=False)
            del state, matched_checkpoint
        else:
            logger.info("Initialized eagle model from scratch")

        # 4、复用主模型的Embedding和LM_Head
        with self.target_model_fsdp.unshard():
            base_module = self.actor_module_fsdp
            # 共享LM_Head
            if hasattr(base_module, "lm_head"):
                drafter_module.lm_head = base_module.lm_head
                for param in drafter_module.lm_head.parameters():
                    param.requires_grad = False
                logger.info("Successfully load lm_head for drafter model")

            # 共享Embedding
            base_model_obj = getattr(base_module, "model", base_module)
            drafter_model_obj = getattr(drafter_module, "model", drafter_module)

            if hasattr(base_module.model, "embed_tokens"):
                drafter_model_obj.embed_tokens = base_model_obj.embed_tokens
                for param in drafter_model_obj.embed_tokens.parameters():
                    param.requires_grad = False
                logger.info("Successfully load embed_tokens for drafter model")

        return drafter_module

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

        num_layers_to_concat = getattr(self.config.actor_rollout_ref.drafter.eagle, "num_layers_to_concat", 1)

        for item in items:
            # 1. 搬运到GPU并统一hidden states维度[L, D]
            ids = item["input_ids"].to(device, non_blocking=True)
            seq_len = ids.size(0)

            raw_h = item["hidden_states"]

            if isinstance(raw_h, (list, tuple)):
                h_states = torch.cat(raw_h[-num_layers_to_concat:], dim=-1).to(device, dtype=torch.bfloat16)
            elif raw_h.dim() == 3:
                h_states = torch.cat([raw_h[i] for i in range(-num_layers_to_concat, 0)], dim=-1).to(device, dtype=torch.bfloat16)
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
    
    def transform_to_algorithm_features(self, input_ids_concat, loss_mask_concat, hidden_states_concat):
        """
        执行 Eagle 核心的滑动窗口特征融合
        input_ids_concat: [1, Total_L]
        hidden_states_concat: [1, Total_L, D]
        """
        dev = input_ids_concat.device
        total_seq_len = input_ids_concat.size(1)
        
        # 识别当前的纵向维度（D * num_layer）
        current_dim = hidden_states_concat.size(-1)

        time_window = getattr(self.config, "time_window_size",1)

        if time_window > 1:
            # padding 维度: [1, window_size-1, 4096]
            pad = torch.zeros((1, time_window - 1, current_dim), device=dev, dtype=hidden_states_concat.dtype)
            h_extended = torch.cat([pad, hidden_states_concat], dim=1)  # [1, L + window-1, 4096]
            
            # 3. 构造滑动窗口拼接 (t, t-1, t-2...)
            chunks = []
            for i in range(time_window):
                # i=0: 当前位置 t (从偏移后的 window-1 开始)
                # i=1: 前一个位置 t-1
                start_idx = (time_window - 1) - i
                chunks.append(h_extended[:, start_idx : start_idx + total_seq_len, :])
            
            # 在最后一个维度 D 上拼接，得到 [1, L, 12288] 或 [1, L, 16384]
            full_hidden_states = torch.cat(chunks, dim=-1)
        else:
            full_hidden_states = hidden_states_concat

        input_ids = input_ids_concat[:, :-1].contiguous()
        base_h = full_hidden_states[:, :-1].contiguous()

        # 无论输入拼接多少层，我们要预测的永远只是主模型最后一层的输出向量
        base_hidden_dim = self.target_model_config.hidden_size
        target = hidden_states_concat[:, 1:, -base_hidden_dim:].contiguous()
        loss_mask = loss_mask_concat[:, 1:].contiguous()

        return {
            "input_ids": input_ids,
            "hidden_states": base_h,
            "target": target,
            "loss_mask": loss_mask,
            "attention_mask": torch.ones_like(input_ids)
        }
    
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
