import logging
import os
from copy import deepcopy

import torch
from torch.nn import functional as F
from transformers import AutoConfig

from .model.auto import AutoDraftModelConfig, AutoEagle3DraftModel
from .eagle_trainer_backend import EagleTrainerBackend
from .model.target.target_head import TargetHead
from verl.utils.fsdp_utils import get_device_id
from verl.utils.device import get_device_name


logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "INFO"))

device_name = get_device_name()


def _scatter_topk_logprobs_with_tail(logprobs: torch.Tensor, indices: torch.Tensor, vocab_size: int) -> torch.Tensor:
    dense_logprob_view = torch.full(
        (logprobs.size(0), vocab_size),
        float("-inf"),
        dtype=logprobs.dtype,
        device=logprobs.device,
    )
    if logprobs.numel() == 0:
        return dense_logprob_view

    valid = torch.isfinite(logprobs) & (indices >= 0) & (indices < vocab_size)
    if not valid.any():
        return dense_logprob_view

    valid_count = valid.sum(dim=-1)
    has_valid = valid_count > 0
    topk_mass = torch.where(valid, logprobs.float().exp(), torch.zeros_like(logprobs, dtype=torch.float32)).sum(dim=-1)
    remaining_mass = (1.0 - topk_mass).clamp(min=torch.finfo(torch.float32).tiny)
    remaining_count = (vocab_size - valid_count).clamp(min=1).to(torch.float32)
    tail_logprob = (remaining_mass.log() - remaining_count.log()).to(logprobs.dtype)
    dense_logprob_view = torch.where(
        has_valid.unsqueeze(-1),
        tail_logprob.unsqueeze(-1).expand(-1, vocab_size),
        dense_logprob_view,
    )

    row_indices = torch.arange(logprobs.size(0), device=logprobs.device).unsqueeze(1).expand_as(indices)
    dense_logprob_view[row_indices[valid], indices[valid]] = logprobs[valid]
    return dense_logprob_view


def _masked_soft_cross_entropy(
    logits: torch.Tensor,
    target_p: torch.Tensor,
    position_mask: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    logits = logits.float()
    target_p = target_p.float()
    finite_logits = torch.isfinite(logits).all(dim=-1)
    finite_target = torch.isfinite(target_p).all(dim=-1) & (target_p.sum(dim=-1) > 0)
    valid_position = (position_mask > 0) & finite_logits & finite_target

    safe_logits = torch.where(torch.isfinite(logits), logits, torch.zeros_like(logits))
    safe_target = torch.where(torch.isfinite(target_p), target_p, torch.zeros_like(target_p))
    safe_target = torch.where(valid_position.unsqueeze(-1), safe_target, torch.zeros_like(safe_target))

    log_probs = F.log_softmax(safe_logits, dim=-1)
    per_token_ploss = -(safe_target * log_probs).sum(dim=-1)
    per_token_ploss = torch.where(valid_position, per_token_ploss, torch.zeros_like(per_token_ploss))
    return per_token_ploss, valid_position


def reconstruct_dense_logprob_view(target_topk_logprobs, topk, vocab_size):
    if topk <= 0:
        raise ValueError(f"topk must be positive when reconstructing dense logprob view, got {topk}")
    if isinstance(target_topk_logprobs, torch.Tensor):
        if target_topk_logprobs.dim() != 3 or target_topk_logprobs.size(-1) < 2:
            raise ValueError(
                "target_topk_logprobs must have shape [seq, topk, 2+] when reconstructing a dense logprob view, "
                f"but got shape={tuple(target_topk_logprobs.shape)}"
            )
        if target_topk_logprobs.numel() == 0:
            return torch.full(
                (
                    target_topk_logprobs.shape[0],
                    vocab_size,
                ),
                float("-inf"),
                dtype=target_topk_logprobs.dtype,
                device=target_topk_logprobs.device,
            )
        logprobs = target_topk_logprobs[..., 0]
        indices = target_topk_logprobs[..., 1].to(torch.long)
        return _scatter_topk_logprobs_with_tail(logprobs, indices, vocab_size)

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
            row = [[float("-inf"), -1] for _ in range(topk)]
        while len(row) < topk:
            row.append([float("-inf"), -1])
        rows.append(row)

    if not rows:
        return torch.empty((0, vocab_size), dtype=torch.float32)

    rows_tensor = torch.tensor(rows, dtype=torch.float32)
    logprobs = rows_tensor[..., 0]
    indices = rows_tensor[..., 1].to(torch.long)
    return _scatter_topk_logprobs_with_tail(logprobs, indices, vocab_size)


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

    def _get_target_hf_config(self):
        target_hf_config = getattr(self.target_model_config, "hf_config", None)
        if target_hf_config is not None:
            return target_hf_config
        if hasattr(self.target_model_config, "hidden_size") and hasattr(self.target_model_config, "vocab_size"):
            return self.target_model_config

        config_path = (
            getattr(self.target_model_config, "local_hf_config_path", None)
            or getattr(self.target_model_config, "hf_config_path", None)
            or getattr(self.target_model_config, "path", None)
        )
        if config_path is None:
            raise ValueError("Cannot resolve target HF config for EAGLE3 drafter")
        return AutoConfig.from_pretrained(
            config_path,
            trust_remote_code=bool(getattr(self.target_model_config, "trust_remote_code", False)),
        )

    def build_model(self):
        """build eagle3 draft model"""
        logger.info(f"Initializing Eagle3 model with type: {getattr(self.target_model_config, 'model_type', None)}")
        spec_model_path = self.config.rollout.drafter.model_path
        config_path = os.path.join(spec_model_path, "config.json")
        target_hf_config = self._get_target_hf_config()

        # 1、加载 Config
        if os.path.exists(config_path):
            drafter_config = AutoDraftModelConfig.from_file(config_path)
        else:
            drafter_config = deepcopy(target_hf_config)
            drafter_config.num_hidden_layers = 1
            drafter_config.torch_dtype = torch.bfloat16
            drafter_config.tie_word_embeddings = False
            drafter_config.architectures = ["LlamaForCausalLMEagle3"]

        if not hasattr(drafter_config, "draft_vocab_size"):
            drafter_config.draft_vocab_size = drafter_config.vocab_size
        if not hasattr(drafter_config, "target_hidden_size"):
            drafter_config.target_hidden_size = target_hf_config.hidden_size

        self.vocab_size = drafter_config.vocab_size

        factory_cls = AutoEagle3DraftModel
        
        drafter_module = factory_cls.from_config(drafter_config)
        checkpoint_has_vocab_mapping = False

        # Initialize model
        if spec_model_path and os.path.exists(spec_model_path):
            loaded = factory_cls.from_pretrained(spec_model_path, output_loading_info=True)
            if isinstance(loaded, tuple):
                drafter_module, loading_info = loaded
                missing_keys = set(loading_info.get("missing_keys", []))
                checkpoint_has_vocab_mapping = not {"t2d", "d2t"}.intersection(missing_keys)
            else:
                drafter_module = loaded
                checkpoint_has_vocab_mapping = self._has_valid_vocab_mapping(drafter_module)

        
        # 复用主模型的Embedding和LM_Head
        reset_rope_buffers = getattr(drafter_module, "reset_rope_buffers", None)
        if callable(reset_rope_buffers):
            reset_count = reset_rope_buffers(dtype=torch.float32)
            if reset_count:
                logger.info("Reset %s EAGLE3 rotary embedding buffers after checkpoint load", reset_count)

        target_model_path = self.config.model.path
            
        drafter_module.load_embedding(target_model_path)
        drafter_module.freeze_embedding()
        
        training_cfg = self.config.rollout.drafter.training
        if drafter_module.draft_vocab_size != drafter_module.vocab_size:
            if checkpoint_has_vocab_mapping and self._has_valid_vocab_mapping(drafter_module):
                logger.info("Using EAGLE3 vocab mapping loaded from draft checkpoint")
            else:
                raise ValueError(
                    "EAGLE3 draft_vocab_size differs from target vocab_size, but the draft checkpoint "
                    "does not provide valid t2d/d2t vocab mapping buffers"
                )
        self._validate_vocab_mapping(drafter_module)

        use_logits = training_cfg.get("use_logits", False)
        if not use_logits:
            target_device = torch.device(f"{device_name}:{get_device_id()}") if device_name != "cpu" else torch.device("cpu")
            self.target_model = self._build_target_model(target_model_path).to(target_device).eval()
            for param in self.target_model.parameters():
                param.requires_grad_(False)

        return drafter_module, drafter_config

    def _has_valid_vocab_mapping(self, drafter_module) -> bool:
        try:
            self._validate_vocab_mapping(drafter_module)
            return True
        except (AttributeError, ValueError):
            return False

    def _validate_vocab_mapping(self, drafter_module) -> None:
        if not hasattr(drafter_module, "t2d") or not hasattr(drafter_module, "d2t"):
            raise AttributeError("EAGLE3 draft model does not have t2d/d2t vocab mapping buffers")

        if drafter_module.t2d.numel() != drafter_module.vocab_size:
            raise ValueError(
                f"EAGLE3 t2d shape mismatch: expected {drafter_module.vocab_size}, "
                f"got {drafter_module.t2d.numel()}"
            )
        if drafter_module.d2t.numel() != drafter_module.draft_vocab_size:
            raise ValueError(
                f"EAGLE3 d2t shape mismatch: expected {drafter_module.draft_vocab_size}, "
                f"got {drafter_module.d2t.numel()}"
            )

        selected_vocab_size = int(drafter_module.t2d.sum().item())
        if selected_vocab_size != drafter_module.draft_vocab_size:
            raise ValueError(
                f"EAGLE3 vocab mapping selects {selected_vocab_size} tokens, "
                f"but draft_vocab_size is {drafter_module.draft_vocab_size}"
            )

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
        res = {'ids':[], 'h_states':[], 'masks': [], 'position_ids': [], 'last_h_states': [], 'target_logprobs': []}
        pad_id = int(getattr(model_config, "pad_token_id", 0) or 0)
        h_dim = getattr(model_config, "target_hidden_size", model_config.hidden_size)
        use_logits = bool(self.config.rollout.drafter.training.get("use_logits", False))

        for item in items:
            # 1. 搬运到GPU
            ids = item["input_ids"].to(device, non_blocking=True)

            raw_h = item["hidden_states"]

            if isinstance(raw_h, (list, tuple)):
                # 将hidden_states进行拼接
                full_h = torch.cat(raw_h, dim=-1).to(device, dtype=torch.bfloat16)
            else:
                full_h = raw_h.to(device, dtype=torch.bfloat16)

            min_hidden_size = 3 * h_dim if use_logits else 4 * h_dim
            if full_h.size(-1) < min_hidden_size:
                raise ValueError(
                    f"EAGLE3 expected at least {min_hidden_size} hidden dims "
                    f"({'3' if use_logits else '4'} target layers of size {h_dim}), got {full_h.size(-1)}"
                )

            h_states = full_h[:, : 3 * h_dim]
            if not use_logits:
                last_h_states = full_h[:, 3 * h_dim : 4 * h_dim]

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
                item_loss_mask = item["loss_mask"].to(device, dtype=torch.float32, non_blocking=True)
            item_position_ids = item.get("position_ids")
            if item_position_ids is None:
                item_position_ids = torch.arange(full_len, device=device, dtype=torch.long)
            else:
                item_position_ids = item_position_ids.to(device, dtype=torch.long, non_blocking=True)
            
            start = 0
            end = full_len
            res['ids'].append(ids[start:end])
            res['h_states'].append(h_states[start:end])
            res['position_ids'].append(item_position_ids[start:end])
            if not use_logits:
                res['last_h_states'].append(last_h_states[start:end])
            res['masks'].append(item_loss_mask[start:end])
            if item.get("target_logprobs") is not None:
                target_end = max(start, end - 1)
                res["target_logprobs"].append(
                    item["target_logprobs"].to(device, dtype=torch.float32)[start:target_end]
                )
        
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
        ttt_length = int(self.config.rollout.drafter.training.get("ttt_length", 1))
        if ttt_length < 1:
            raise ValueError(f"EAGLE3 ttt_length must be >= 1, got {ttt_length}")
        draft_model = model.module if hasattr(model, "module") else model

        # 前向传播
        outputs = model(
            input_ids=input_ids,
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            loss_mask=loss_mask,
            position_ids=position_ids,
            ttt_length=ttt_length,
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
                target_scores = reconstruct_dense_logprob_view(
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
                target_scores = reconstruct_dense_logprob_view(
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
        if length == 0:
            return {
                "total_local_vloss": torch.tensor(0.0, device=input_ids.device),
                "total_local_ploss": torch.tensor(0.0, device=input_ids.device),
                "local_num_tokens": torch.tensor(0.0, device=input_ids.device),
                "v_weight": 0.0,
                "p_weight": 0.0,
            }
        # With Ulysses SP, logits and masks are gathered back to full sequence
        # length above, while input_ids remains the local SP slice. Use the
        # actual logits length for target/mask alignment.
        seq_length = all_step_logits[0].shape[1]
        target_device = all_step_logits[0].device
        if target_scores.device != target_device:
            target_scores = target_scores.to(target_device)
        if loss_mask.device != target_device:
            loss_mask = loss_mask.to(target_device)

        target_p_padded, target_position_mask_padded = self._compute_target_p_padded(
            target_scores=target_scores,
            t2d=draft_model.t2d,
            loss_mask=loss_mask,
            length=length,
        )

        # Clean up large tensors to free memory
        del target_scores

        total_local_ploss = torch.tensor(0.0, device=input_ids.device, dtype=torch.float32)
        total_local_tokens = torch.tensor(0.0, device=input_ids.device, dtype=torch.float32)
        gamma = 0.8
        
        # 预处理
        for idx in range(length):
            # 切片对齐：取当前步对应的未来目标
            # 这里的关键是 target_p 会随着 idx 往后偏移
            target_p = target_p_padded[:, idx : idx + seq_length, :].contiguous()
            logits = all_step_logits[idx]
            step_position_mask = all_step_position_mask[idx]
            if step_position_mask.dim() == 3:
                step_position_mask = step_position_mask.squeeze(-1)
            target_position_mask = target_position_mask_padded[:, idx : idx + seq_length]
            if target_position_mask.dim() == 3:
                target_position_mask = target_position_mask.squeeze(-1)
            position_mask = step_position_mask * target_position_mask

            base_valid_position = position_mask > 0
            per_token_ploss, valid_position = _masked_soft_cross_entropy(
                logits=logits,
                target_p=target_p,
                position_mask=position_mask,
            )
            if base_valid_position.any() and not valid_position[base_valid_position].all():
                dropped_tokens = (base_valid_position & ~valid_position).sum()
                logger.debug(
                    "Dropping %s EAGLE3 target positions with non-finite logits or targets",
                    int(dropped_tokens.detach().cpu().item()),
                )
            step_loss_sum = per_token_ploss.sum()
            
            # 应用Eagle3的时间步衰减
            total_local_ploss += (gamma ** idx) * step_loss_sum
            total_local_tokens += valid_position.float().sum()

        return {
            "total_local_vloss": torch.tensor(0.0, device=input_ids.device),
            "total_local_ploss": total_local_ploss,
            "local_num_tokens": total_local_tokens,
            "v_weight": 0.0,
            "p_weight": 1.0
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
                # Future-shift padding is masked out by position_mask_padded.
                value=1 / target_p.shape[-1],
            )
            position_mask_padded = F.pad(
                position_mask,
                pad=(0, length),
                mode="constant",
                value=0.0,
            )

            return target_p_padded, position_mask_padded


    def _compute_target_p(self, target_scores, t2d, loss_mask):
        loss_mask = loss_mask.to(device=target_scores.device)
        t2d = t2d.to(device=target_scores.device, dtype=torch.bool)
        target_subset_scores = target_scores
        target_subset_scores = target_subset_scores[..., t2d]
        if target_subset_scores.size(-1) == 0:
            raise ValueError("EAGLE3 target-to-draft vocab mask selects zero tokens")
        finite_target_mask = torch.isfinite(target_subset_scores).any(dim=-1)
        position_mask = finite_target_mask.float() * loss_mask.float()
        target_subset_scores = target_subset_scores.float()
        finite_scores = torch.isfinite(target_subset_scores)
        finite_floor = torch.finfo(target_subset_scores.dtype).min
        target_subset_scores = torch.where(
            finite_scores,
            target_subset_scores,
            torch.full_like(target_subset_scores, finite_floor),
        )
        target_subset_scores = torch.where(
            finite_target_mask.unsqueeze(-1),
            target_subset_scores,
            torch.zeros_like(target_subset_scores),
        )
        target_p = F.softmax(target_subset_scores, dim=-1)
        target_p = torch.where(
            finite_scores & finite_target_mask.unsqueeze(-1),
            target_p,
            torch.zeros_like(target_p),
        )
        target_p = target_p.detach()
        return target_p, position_mask
        

        
