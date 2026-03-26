import logging
import os
import time
import glob
from collections import deque
from typing import Optional
from copy import deepcopy
import safetensors
from omegaconf import open_dict

import torch
import torch.distributed as dist
import torch.distributed.checkpoint as dcp
from torch import optim
from torch.distributed.device_mesh import DeviceMesh
from torch.nn import SmoothL1Loss
from torch.nn import functional as F
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import StateDictType, ShardedStateDictConfig

from .model.eagle3.llama_eagle3 import LlamaForCausalLMEagle3
from .model.eagle3.qwen_eagle3 import QwenEagle
from verl.utils.data_buffer import DataBuffer
from verl.utils.fsdp_utils import (
    get_device_id,
    apply_fsdp2,
    fsdp2_load_full_state_dict,
    load_fsdp_model_to_gpu,
    load_fsdp_optimizer,
    MixedPrecisionPolicy
)
from verl.utils.torch_functional import (
    get_constant_schedule_with_warmup,
    get_cosine_schedule_with_warmup,
)
from verl.utils.ulysses import ulysses_pad_and_slice_inputs

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "INFO"))


class EagleBackgroundTrainer:
    def __init__(
        self,
        config,
        model_config,
        actor_module_fsdp,
        world_size: int,
        rollout_dp_rank: int,
        **kwargs
    ):
        self.model_config = model_config
        self.config = config
        self.actor_module_fsdp = actor_module_fsdp
        self.world_size = world_size
        self.rollout_dp_rank = rollout_dp_rank

        self.training_device_mesh = self._setup_device_mesh()
        self.model, self.optimizer, self.lr_scheduler, self.train_config = self.build_draft_model()

        self.rank = kwargs.get("rank", dist.get_rank() if dist.is_initialized() else 0)
        self.device_id = get_device_id()
        self.copy_stream = torch.cuda.Stream()

        self.is_offload_param = False
        self.is_offload_optimizer = False
        self._training_initialized = False
        self._training_active = False
        self.training_steps = 0

        self.collected_data = deque(maxlen=int(self.config.actor_rollout_ref.rollout.get("buffer_max_samples", 2000)))
        self.shared_data_buffer = None
        self.batch_size = int(self.config.actor_rollout_ref.drafter.train.get("batch_size_per_gpu", 32))

        # Initialize DataBuffer for storing data across RL steps
        buffer_max_size = int(self.config.actor_rollout_ref.drafter.train.get("data_buffer_max_size", 10000))
        # Only store hidden states in buffer if we're collecting them during generation
        collect_hidden_states_from_sgl = bool(self.config.actor_rollout_ref.drafter.train.get("collect_hidden_states_from_sgl", False))

        # todo DataBuffer define
        self.data_buffer = DataBuffer(max_size=buffer_max_size, store_hidden_states=collect_hidden_states_from_sgl)

        self.criterion = SmoothL1Loss(reduction="none")

        self.eagle_model_path = self.config.actor_rollout_ref.drafter.eagle.get("spec_model_path")
        self.checkpoint_dir = self.config.actor_rollout_ref.drafter.train.get("checkpoint_path")
        self._last_ckpt_step = -1
        # New: optional per-step barrier (default False to avoid stalls)
        self.enable_mesh_barrier = bool(self.config.actor_rollout_ref.drafter.train.get("enable_step_barrier", False))

        # Track the last pending async checkpoint save future
        self._pending_checkpoint_future = None
        self._frozen_param_names = {"model.embed_tokens.weight", "lm_head.weight"}

        # Ulysses Sequence Parallelism configuration
        self.ulysses_sequence_parallel_size = self.config.actor_rollout_ref.actor.get("ulysses_sequence_parallel_size", 1)
        self.use_ulysses_sp = self.ulysses_sequence_parallel_size > 1

    def _setup_device_mesh(self):
        infer_tp = self.config.actor_rollout_ref.rollout.tensor_model_parallel_size
        dp_size = self.world_size // infer_tp
        global_device_mesh_list = [
            DeviceMesh("cuda", list(range(i * infer_tp, (i + 1) * infer_tp))) for i in range(dp_size)
        ]
        return global_device_mesh_list[self.rollout_dp_rank]

    def build_draft_model(self):
        """build draft model"""
        logger.info(f"[Rank {self.rollout_dp_rank}] Building Eagle drafter model...")

        # 1、配置准备
        rollout_cfg = self.config.actor_rollout_ref
        if not (hasattr(rollout_cfg, "drafter") and hasattr(rollout_cfg.drafter, "eagle")):
            raise ValueError("Speculative eagle config is missing")
        
        spec_model_path = rollout_cfg.drafter.eagle.spec_model_path

        # 复制主模型配置并修改为单层 Eagle 结构
        config = deepcopy(self.model_config)
        config.num_hidden_layers = 1
        config.torch_dtype = torch.bfloat16
        config.tie_word_embeddings = False
        model_type = getattr(config, "model_type", "llama")

        # 2、实例化模型
        if model_type.lower() == "llama":
            model_class = LlamaForCausalLMEagle3
        elif model_type.lower() == "qwen2" or model_type.lower() == "qwen3":
            model_class = QwenEagle
        else:
            raise ValueError(f"Unsupported model type for eagle: {model_type}")
        
        drafter_module = model_class(config=config).cuda()

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
        base_module = self.actor_module_fsdp.unshard()
        if hasattr(base_module, "lm_head"):
            drafter_module.lm_head = base_module.lm_head
            for param in drafter_module.lm_head.parameters():
                param.requires_grad = False
            logger.info("Successfully load lm_head for drafter model")

        if hasattr(base_module, "model") and hasattr(base_module.model, "embed_tokens"):
            drafter_module.embed_tokens = base_module.model.embed_tokens
            for param in drafter_module.embed_tokens.parameters():
                param.requires_grad = False
            logger.info("Successfully load embed_tokens for drafter model")
            
        # 释放对著模型 unshared 状态的引用
        del base_module   

        # 5、Apply FSDP2
        fsdp_config = self.config.actor_rollout_ref.actor.fsdp_config
        mp_policy = MixedPrecisionPolicy(
            param_dtype=torch.bfloat16, reduce_dtype=torch.float32, cast_forward_inputs=True
        )
        fsdp_kwargs = {
            "mesh": self.training_device_mesh,
            "mp_policy": mp_policy,
            "offload_policy": None,
        }
        logger.info("Inside building drafter model (Before FSDP2)")
        full_state = drafter_module.state_dict()
        apply_fsdp2(drafter_module, fsdp_kwargs, fsdp_config)
        # Load full state dict using the same mesh as used by drafter FSDP wrapping
        fsdp2_load_full_state_dict(drafter_module, full_state, self.training_device_mesh, None)
        del full_state

        # 6、构建训练配置、优化器和调度器
        drafter_train_config = self._prepare_training_config(rollout_cfg)

        drafter_optimizer = optim.AdamW(
            [p for p in drafter_module.parameters() if p.requires_grad],
            lr=drafter_train_config.optim.lr,
            betas=(0.9, 0.95),
            weight_decay=drafter_train_config.optim.get("weight_decay", 1e-2),
        )

        drafter_lr_scheduler = self._setup_scheduler(drafter_optimizer, drafter_train_config)

        logger.info("After building drafter model")
        return drafter_module, drafter_optimizer, drafter_lr_scheduler, drafter_train_config

    def _prepare_training_config(self, rollout_config):
        """
        Prepare the training configuration for drafter module.

        Args:
            rollout_config (dict): The rollout configuration.

        Returns:
            dict: The prepared training configuration.
        """
        drafter_train_config = rollout_config['drafter']['train'].copy()

        # Open the dictionary for modification
        with open_dict(drafter_train_config):
            # Update the configuration with required values
            drafter_train_config.update(
                {
                    "spec_strategy": rollout_config['drafter']['spec_strategy'],
                    "spec_model_path": rollout_config['drafter']['eagle']['spec_model_path'],
                    "is_offload_optimizer": False,
                    "is_offload_param": False,
                    "vloss_weight": 1.0,
                    "ploss_weight": 0.1,
                    "data_augment_std": 0.2,
                }
            )

        return drafter_train_config

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

    def _setup_scheduler(self, optimizer, train_cfg):
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

    
    def _get_trainable_state_dict(self) -> dict[str, torch.Tensor]:
        """Get state dict excluding frozen layers (embed_tokens, lm_head)."""
        full_state_dict = self.model.state_dict()
        trainable_state_dict = {}

        for name, param in full_state_dict.items():
            # Skip frozen parameters
            if any(frozen_name in name for frozen_name in self._frozen_param_names):
                logger.debug(f"Skipping frozen parameter: {name}")
                continue
            trainable_state_dict[name] = param

        return trainable_state_dict

    
    def _save_checkpoint_async(self, step: int, is_final: bool = False):
        """Asynchronously save checkpoint using DCP's async_save.

        Args:
            step: Current training step
            is_final: Whether this is the final checkpoint during cleanup

        Returns:
            Future object from dcp.async_save that can be awaited or checked for completion
        """
        if not self.checkpoint_dir:
            return None

        try:
            checkpoint_path = os.path.join(self.checkpoint_dir, f"eagle_step_{step}")
            if self.rank == 0:
                os.makedirs(checkpoint_path, exist_ok=True)
            
            # Synchronization point: Ensure directory exists before any rank starts writing
            if self.training_device_mesh is not None:
                dist.barrier(group=self.training_device_mesh.get_group()) 
            
            # Optimization: Sharded save with CPU offloading to maintain GPU compute space 
            with FSDP.state_dict_type( 
                self.model, 
                state_dict_type=StateDictType.SHARDED_STATE_DICT, 
                sharded_state_dict_config=ShardedStateDictConfig(offload_to_cpu=True) 
            ): 
                state_dict = { 
                    "model": self._get_trainable_state_dict(), 
                    # "optimizer": FSDP.sharded_optim_state_dict(self.model, self.optimizer) if self.optimizer else {},
                    "step": step 
                } 
                    
            return dcp.async_save( 
                state_dict=state_dict, 
                checkpoint_id=checkpoint_path, 
                process_group=self.training_device_mesh.get_group(), 
            )

        except Exception as e:  # noqa: BLE001
            logger.warning(f"Async checkpoint save failed on rank {self.rank}: {e}")
            return None

    async def activate_training_model(
        self, device_mesh: DeviceMesh, training_ranks: list[int], base_model=None
    ) -> bool:
        # 将模型和优化器状态从CPU加载到GPU，激活草稿模型进入训练状态
        start_ts = time.time()
        try:        
            logger.warning(
                f"[EagleTrainer rank {getattr(self, 'rank', -1)}] activate_training_model enter "
                f"training_ranks={training_ranks}"
            )

            # 只有当配置了 offload 或者当前模型不在 CUDA 上时执行加载
            first_param = next(self.model.parameters(), None)
            is_on_cuda = first_param is not None and first_param.device.type == "cuda"

            if self.is_offload_param or is_on_cuda:
                # 调用工具将 FSDP 分片移动到 GPU
                load_fsdp_model_to_gpu(self.model)
                logger.debug("Loaded drafter model to GPU for training")
            
            if self.optimizer is not None:
                # 获取 device_id,否则在多卡环境优化器状态可能全部挤在 cuda:0 导致 OOM
                current_dev_id = get_device_id()
                load_fsdp_optimizer(optimizer=self.optimizer, device_id=current_dev_id)
                logger.debug("Loaded drafter optimizer to GPU for training")

            self.training_device_mesh = device_mesh

            # 先标记初始化完成，然后开启 active 开关，确保训练循环不会读到中间状态
            self._training_initialized = True
            self._training_active = True

            logger.info(
                f"Drafter training activated with device_mesh={device_mesh}, training_ranks={training_ranks}"
                f"[EagleTrainer rank {getattr(self, 'rank', -1)}] activate_training_model success "
                f"elapsed={time.time() - start_ts:.2f}s"
            )
            return True
        
        except Exception as e:
            logger.error(f"[EagleTrainer rank {getattr(self, 'rank', -1)}] activate_training_model failed: {e}")
            self._training_active = False
            return False

    def collect_online_data(self, batch: dict, hidden_states: list[torch.Tensor]):
        """Collect online data from inference for Eagle training.

        This method collects data both to the local collected_data deque (for immediate use)
        and to the DataBuffer (for cross-step data accumulation).
        """
        input_ids = batch.get("input_ids")
        if input_ids is None:
            logger.warning(
                f"[Rank {self.rank}] Non-batched data in input_ids"
            )
            return

        # 1、异步拷贝，GPU在后台进行数据搬运，避免阻塞Rollout Stream
        with torch.cuda.stream(self.copy_stream):
            cpu_input_ids = input_ids.to('cpu', non_blocking=True)
            cpu_h_states = [h.to('cpu', non_blocking=True) for h in hidden_states]
            cpu_responses = batch.get("responses").to('cpu', non_blocking=True) if "responses" in batch else None
            cpu_prompts = batch.get("prompts").to('cpu', non_blocking=True) if "prompts" in batch else None

        # 构建要存入的数据项
        data_item = {
            "input_ids": cpu_input_ids,
            "responses": cpu_responses,
            "prompts": cpu_prompts,
            "hidden_states": cpu_h_states[0] if isinstance(cpu_h_states, list) else cpu_h_states,
        }
        
        # 同步 DataBuffer
        self.data_buffer.add_batch(data_item, hidden_states)

        # 同步 collect_data (当前步训练直接使用)
        self.collected_data.append(data_item)

    def _prepare_training_batch(
        self, use_buffer_data: bool = True, buffer_steps: int = 2
    ) -> Optional[dict[str, torch.Tensor]]:
        """Prepare a batch for training using Ulysses SP to remove padding.

        Args:
            use_buffer_data: If True, use data from DataBuffer (across multiple RL steps)
            buffer_steps: Number of recent RL steps to include data from (only used if use_buffer_data=True)

        Returns:
            Dictionary containing batch tensors for training
        """
        effective_batch_size = min(self.batch_size, 4)

        # Determine data source: DataBuffer (cross-step) or collected_data (current step only)
        if use_buffer_data and len(self.data_buffer) > 0:
            # Use data from last N RL steps via DataBuffer
            available_data = self.data_buffer.get_data_from_last_n_steps(buffer_steps)
            if len(available_data) < effective_batch_size:
                if 0 < len(available_data) >= min(2, effective_batch_size // 2):
                    items = available_data
                else:
                    return None
            else:
                # Randomly sample from available data to ensure diversity
                import random

                items = random.sample(available_data, min(len(available_data), effective_batch_size))
        else:
            # Fall back to current step data only
            if len(self.collected_data) < effective_batch_size:
                if 0 < len(self.collected_data) >= min(2, effective_batch_size // 2):
                    items = list(self.collected_data)
                else:
                    return None
            else:
                items = list(self.collected_data)[:effective_batch_size]
        
        # Filter out items without hidden_states (defensive check)
        items = [item for item in items if "hidden_states" in item]
        if len(items) == 0:
            logger.warning(f"[Rank {self.rank}] No items with hidden_states found, cannot prepare batch")
            return None
        elif len(items) < min(2, effective_batch_size // 2):
            logger.warning(
                f"[Rank {self.rank}] Only {len(items)} items with hidden_states found "
                f"(need at least {min(2, effective_batch_size // 2)}), cannot prepare batch"
            )
            return None

        pad_id = int(getattr(self.model_config, "pad_token_id", 0) or 0)
        dev = next(self.model.parameters()).device
        max_window = 512

        # Collect sequences to concatenate (removing padding)
        input_ids_list, loss_mask_list, hidden_states_list = [], [], []
        cu_seqlens = [0] # 用于后续支持变长 Flash Attention

        for item in items:
            # 利用non_blocking异步搬运，且保持在GPU上操作
            ids = item["input_ids"].to(dev, non_blocking=True)
            h_states = item["hidden_states"].to(dev, dtype=torch.bfloat16, non_blocking=True)

            # 统一对齐 Hidden States
            # 统一维度：[1, L, D] -> [L, D]
            if h_states.dim() == 1:
                h_states = h_states.unsqueeze(0)
            elif h_states.dim() > 2:
                h_states = h_states.view(-1, h_states.size(-1))
            
            # 通过统一裁剪或填充将Hidden States与sequence length对齐
            h_len = h_states.size(0)
            seq_len = ids.size(0)
            if h_len < seq_len:
                # 批量 Padding
                padding = torch.zeros((seq_len - h_len, h_states.size(-1)), 
                                    device=dev, dtype=h_states.dtype)
                h_states = torch.cat([h_states, padding], dim=0)
            else:
                h_states = h_states[:seq_len, :]


            # Compute loss_mask if not present (for DataBuffer items)，在GPU上向量化生成，避免CPU循环
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
                start = torch.clamp(r_start - (max_window // 2), min=0, max=full_len - max_window).item()
                end = min(start + max_window, full_len)
            else:
                start = max(0, full_len - max_window)
                end = full_len

            # Extract the window
            actual_ids = ids[start:end]
            actual_mask = item_loss_mask[start:end]
            actual_h = h_states[start:end]

            # 使用F.pad统一长度，避免cat产生新碎片
            curr_len = actual_ids.size(0)
            if actual_h.size(0) < curr_len:
                actual_h = F.pad(actual_h, (0, 0, 0, curr_len - actual_h.size(0)))

            input_ids_list.append(actual_ids)
            loss_mask_list.append(actual_mask)
            hidden_states_list.append(actual_h)
            cu_seqlens.append(cu_seqlens[-1] + curr_len)

        # Concatenate all sequences into a single sequence
        input_ids_concat = torch.cat(input_ids_list, dim=0).unsqueeze(0)  # (1, total_seq_len)
        loss_mask_concat = torch.cat(loss_mask_list, dim=0).unsqueeze(0)  # (1, total_seq_len)
        hidden_states_concat = torch.cat(hidden_states_list, dim=0).unsqueeze(0)  # (1, total_seq_len, hidden_dim)

        # Create attention mask (all 1s since no padding)
        total_seq_len = input_ids_concat.size(1)
        attn_mask = torch.ones((1, total_seq_len), dtype=torch.long, device=dev)

        # Window Concatenation
        # 1. 自动获取窗口大小 (e.g., 12288 / 4096 = 3)
        base_hidden_dim = self.model_config.hidden_size
        target_fc_in = self.model.fc.in_features
        window_size = target_fc_in // base_hidden_dim
        
        if window_size > 1:
            # padding 维度: [1, window_size-1, 4096]
            pad = torch.zeros((1, window_size - 1, base_hidden_dim), device=dev, dtype=hidden_states_concat.dtype)
            h_extended = torch.cat([pad, hidden_states_concat], dim=1)  # [1, L + window-1, 4096]
            
            # 3. 构造滑动窗口拼接 (t, t-1, t-2...)
            chunks = []
            for i in range(window_size):
                # i=0: 当前位置 t (从偏移后的 window-1 开始)
                # i=1: 前一个位置 t-1
                start_idx = (window_size - 1) - i
                chunks.append(h_extended[:, start_idx : start_idx + total_seq_len, :])
            
            # 在最后一个维度 D 上拼接，得到 [1, L, 12288] 或 [1, L, 16384]
            hidden_states_concat = torch.cat(chunks, dim=-1)

        full_pos_ids = torch.arange(total_seq_len, device=dev).unsqueeze(0)

        # Use Ulysses SP to pad and slice if needed
        if self.use_ulysses_sp:
            # Pad to be divisible by SP size and slice across ranks
            input_ids_concat, sharded_pos_ids, pad_size = ulysses_pad_and_slice_inputs(
                input_ids_concat, position_ids_rmpad=None, sp_size=self.ulysses_sequence_parallel_size
            )
            # Pad loss_mask and hidden_states to match
            if pad_size > 0:
                loss_mask_concat = torch.nn.functional.pad(loss_mask_concat, (0, pad_size), value=0.0)
                hidden_states_concat = torch.nn.functional.pad(hidden_states_concat, (0, 0, 0, pad_size), value=0.0)
                attn_mask = torch.nn.functional.pad(attn_mask, (0, pad_size), value=0)

            # Slice for this rank
            from verl.utils.ulysses import slice_input_tensor
            loss_mask_concat = slice_input_tensor(loss_mask_concat, dim=1, padding=False)
            hidden_states_concat = slice_input_tensor(hidden_states_concat, dim=1, padding=False)
            attn_mask = slice_input_tensor(attn_mask, dim=1, padding=False)

            # Store pad_size for later gathering
            self._current_pad_size = pad_size
            position_ids_concat = sharded_pos_ids
        else:
            self._current_pad_size = 0
            position_ids_concat = full_pos_ids
        
        position_ids = position_ids_concat[:, :-1].contiguous()

        # Shift for next token prediction
        if window_size > 1:
            target = hidden_states_concat[:, 1:, :base_hidden_dim].contiguous()
        else:
            target = hidden_states_concat[:, 1:].contiguous()
        loss_mask = loss_mask_concat[:, 1:].contiguous()
        input_ids = input_ids_concat[:, :-1].contiguous()
        attn_mask = attn_mask[:, :-1].contiguous()
        base_h = hidden_states_concat[:, :-1].contiguous()

        return {
            "input_ids": input_ids,
            "position_ids": position_ids,
            "attention_mask": attn_mask,
            "hidden_states": base_h,
            "target": target,
            "loss_mask": loss_mask,
        }
    
    async def training_step(self, step: int) -> bool:
        try:
            with torch.enable_grad():
                return await self._training_step_impl(step)
        except Exception as e:  # noqa: BLE001
            logger.exception(f"Training step {step} failed with error: {e}")
            return False
        
    async def _training_step_impl(self, step: int) -> bool:
        """Execute a single training step."""
        if not self.model:
            logger.warning("No model available for training")
            return False

        # Skip training if we're not collecting hidden states (since we can't train without them)
        collect_hidden_states_from_sgl = bool(self.config.actor_rollout_ref.drafter.train.get("collect_hidden_states_from_sgl", False))
        if not collect_hidden_states_from_sgl:
            logger.debug(
                f"[EagleTrainer rank {self.rank}] Skipping training step {step} "
                f"because collect_hidden_states_from_sgl=False"
            )
            return False

        batch = self._prepare_training_batch()
        if batch is None:
            logger.debug(
                f"[EagleTrainer rank {self.rank}] Not enough data at step {step} "
                f"(have={len(self.collected_data)} need≥{min(self.batch_size, 4)})"
            )
            return False
        
        self.model.train()
        self.optimizer.zero_grad(set_to_none=True)

        # 前向传播
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            outputs = self.model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                hidden_states=batch["hidden_states"],
                position_ids=batch["position_ids"],
                output_hidden_states=True,
            )

            hidden_states = outputs["hidden_states"]
            logits = outputs["logits"]

            # 局部损失计算，不再对大张量进行 gather，直接在当前 Rank 计算局部 Loss
            target = batch["target"]
            loss_mask = batch["loss_mask"]

            # V-Loss：隐藏态回归损失
            vloss_all = self.criterion(hidden_states, target)  # [B,T,H]
            vloss_per_token = vloss_all.mean(dim=-1) # [B, T]

            # P-Loss: 概率分布对齐损失
            with torch.no_grad():
                target_p = F.softmax(self.model.lm_head(target), dim=1)

            log_prod = F.log_softmax(logits,  dim=-1)
            ploss_per_token = -(target_p * log_prod).sum(dim=-1) # [B, T]

            # 结合 Mask
            valid_mask = loss_mask > 0
            total_local_vloss = (vloss_per_token * loss_mask).sum()
            total_local_ploss = (ploss_per_token * loss_mask).sum()
            local_num_tokens = loss_mask.sum()

        # 分布式同步（Global Reduction）,如果使用序列并行，仅在这里进行一次标量同步
        if self.training_device_mesh is not None and self.training_device_mesh.size() > 1:
            metrics = torch.stack([total_local_vloss, total_local_ploss, local_num_tokens])
            dist.all_reduce(metrics, group=self.training_device_mesh.get_group())
            global_vloss, global_ploss, global_tokens = metrics[0], metrics[1], metrics[2]
        else:
            global_vloss, global_ploss, global_tokens = total_local_vloss, total_local_ploss, local_num_tokens
        
        # 最终 Loss 平滑处理
        denom = global_tokens.clamp(min=1.0)
        vloss = global_vloss / denom
        ploss = global_ploss / denom

        w_v = float(self.config.get("vloss_weight", 0.5))
        w_p = float(self.config.get("ploss_weight", 0.5))
        loss = w_v * vloss + w_p * ploss

        # 反向传播
        loss.backward()

        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

        self.optimizer.step()
        if self.lr_scheduler is not None:
            self.lr_scheduler.step()

        self.training_steps += 1
        if self.training_steps % 10 == 0:
            logger.info(
                f"Step {self.training_steps}: loss={float(loss.item()):.4f}, vloss={float(vloss.item()):.4f}, ploss={float(ploss.item()):.4f}"
            )
        # 异步进行checkpoint保存
        if self.checkpoint_dir and (step // 100) > self._last_ckpt_step:
            # Wait for previous checkpoint to complete before starting a new one
            # This avoids queuing multiple checkpoints and excessive memory usage
            if self._pending_checkpoint_future is not None:
                try:
                    self._pending_checkpoint_future.result()
                except Exception as e:  # noqa: BLE001
                    logger.warning(f"Previous checkpoint save failed: {e}")

            # Launch async checkpoint save without blocking training
            self._pending_checkpoint_future = self._save_checkpoint_async(step, is_final=False)
            self._last_ckpt_step = step // 100

        return True
    
    def increment_rl_step(self):
        """Increment the RL step counter in the data buffer.

        Should be called at the end of each RL training step to mark the boundary.
        """
        self.data_buffer.increment_step()
        logger.debug(
            f"[Rank {self.rank}] DataBuffer RL step incremented to {self.data_buffer.get_current_step()}, "
            f"total samples: {len(self.data_buffer)}"
        )
    
    def get_model_state_dict(self) -> Optional[dict[str, torch.Tensor]]:
        """Get trainable model state dict (excluding frozen layers)."""
        if not self.model:
            return None
        trainable_state = self._get_trainable_state_dict()
        return {k: v.detach().cpu() for k, v in trainable_state.items() if v.requires_grad}
