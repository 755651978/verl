# Copyright 2025 ModelBest Inc. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from __future__ import annotations

import importlib
import inspect
import logging
import os
import re
import textwrap
from functools import wraps
from types import FunctionType
from typing import Callable

import torch
import sglang.srt.entrypoints.engine
from sglang.srt.utils import MultiprocessingSerializer

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))

_TARGET_WEIGHT_LOADER_ENV = "VERL_SGLANG_TARGET_WEIGHT_LOADER"
_DRAFT_WEIGHT_LOADER_ENV = "VERL_SGLANG_DRAFT_WEIGHT_LOADER"
_EAGLE_VERIFY_MODE_ENV = "VERL_SGLANG_NPU_EAGLE_VERIFY_MODE"
_EAGLE_V1_TARGET_SAMPLING_ENV = "VERL_SGLANG_NPU_EAGLE_V1_TARGET_SAMPLING"
_EAGLE_V1_VERIFY_MODE_ENV = "VERL_SGLANG_NPU_EAGLE_V1_VERIFY_MODE"
_EAGLE_V2_VERIFY_MODE_ENV = "VERL_SGLANG_NPU_EAGLE_V2_VERIFY_MODE"

_target_weight_loader: str | None = os.environ.get(_TARGET_WEIGHT_LOADER_ENV)
_draft_weight_loader: str | None = os.environ.get(_DRAFT_WEIGHT_LOADER_ENV)
_ORIGINAL_SGLANG_RUN_SCHEDULER_PROCESS = sglang.srt.entrypoints.engine.run_scheduler_process
_ORIGINAL_SGLANG_DIRECT_RUN_SCHEDULER_PROCESS = None
_SGLANG_EAGLE_UPDATE_PATCHED = False
_SGLANG_NPU_EAGLE_SAMPLING_PATCHED = False
_SGLANG_TRANSFORMERS_EAGLE3_CAPTURE_PATCHED = False
_SGLANG_HIDDEN_STATES_TENSOR_OUTPUT_PATCHED = False
_SGLANG_SCHEDULER_PROCESS_PATCHED = False
_SCHEDULER_PROCESS_PATCH_ATTR = "_verl_patched_scheduler_process"
_SGLANG_TOP_K_ALL = 1 << 30


def configure_sglang_eagle_weight_update_patch(
    target_weight_loader: str | None,
    draft_weight_loader: str | None,
) -> None:
    global _target_weight_loader, _draft_weight_loader

    if target_weight_loader is not None:
        _target_weight_loader = target_weight_loader
        os.environ[_TARGET_WEIGHT_LOADER_ENV] = target_weight_loader
    if draft_weight_loader is not None:
        _draft_weight_loader = draft_weight_loader
        os.environ[_DRAFT_WEIGHT_LOADER_ENV] = draft_weight_loader


def _get_route_markers() -> tuple[str | None, str | None]:
    return (
        _target_weight_loader or os.environ.get(_TARGET_WEIGHT_LOADER_ENV),
        _draft_weight_loader or os.environ.get(_DRAFT_WEIGHT_LOADER_ENV),
    )


def _get_sglang_worker_tp_rank(worker) -> int:
    for obj in (
        worker,
        getattr(worker, "model_runner", None),
        getattr(worker, "draft_model_runner", None),
        getattr(worker, "target_worker", None),
        getattr(worker, "target_model_worker", None),
    ):
        if obj is None:
            continue
        for attr_name in ("tp_rank", "rank"):
            attr = getattr(obj, attr_name, None)
            if attr is not None:
                return int(attr)
    return 0


def _get_sglang_draft_runner(worker):
    for attr_name in ("draft_model_runner", "model_runner"):
        runner = getattr(worker, attr_name, None)
        if runner is not None:
            return runner
    return None


def _get_sglang_target_runner(worker):
    for worker_attr in ("target_worker", "target_model_worker"):
        target_worker = getattr(worker, worker_attr, None)
        if target_worker is None:
            continue
        runner = getattr(target_worker, "model_runner", None)
        if runner is not None:
            return runner
    return None


def _make_verl_eagle_update_weights_patch(original_update_weights):
    @wraps(original_update_weights)
    def patched_update_weights_from_tensor(self, recv_req):
        target_weight_loader, draft_weight_loader = _get_route_markers()
        load_format = getattr(recv_req, "load_format", None)
        target_only = target_weight_loader is not None and load_format == target_weight_loader
        draft_only = draft_weight_loader is not None and load_format == draft_weight_loader
        disable_draft_model = bool(getattr(recv_req, "disable_draft_model", False)) or target_only
        disable_target_model = bool(getattr(recv_req, "disable_target_model", False)) or draft_only

        if not (disable_draft_model or disable_target_model):
            return original_update_weights(self, recv_req)

        if disable_draft_model and disable_target_model:
            return False, "Both target and draft model updates are disabled."

        serialized_named_tensors = getattr(recv_req, "serialized_named_tensors", None)
        if not serialized_named_tensors:
            return True, "No tensor is provided for routed EAGLE weight update."

        tp_rank = _get_sglang_worker_tp_rank(self)
        if tp_rank >= len(serialized_named_tensors):
            return (
                False,
                "Invalid routed EAGLE update tensor shard index: "
                f"tp_rank={tp_rank}, num_shards={len(serialized_named_tensors)}.",
            )

        from sglang.srt.utils.patch_torch import monkey_patch_torch_reductions

        monkey_patch_torch_reductions()
        named_tensors = MultiprocessingSerializer.deserialize(serialized_named_tensors[tp_rank])

        # The verl-only custom loaders are route markers. After EAGLEWorker
        # selects the correct side, let that model runner use its normal loader.
        routed_load_format = None if target_only or draft_only else load_format

        if not disable_draft_model:
            draft_runner = _get_sglang_draft_runner(self)
            if draft_runner is None:
                return False, "SGLang EAGLE draft model runner is missing."
            success, message = draft_runner.update_weights_from_tensor(
                named_tensors=named_tensors,
                load_format=routed_load_format,
            )
            if not success:
                return success, message

        if not disable_target_model:
            target_runner = _get_sglang_target_runner(self)
            if target_runner is None:
                return False, "SGLang EAGLE target model runner is missing."
            success, message = target_runner.update_weights_from_tensor(
                named_tensors=named_tensors,
                load_format=routed_load_format,
            )
            if not success:
                return success, message

        return True, "Routed EAGLE weight update succeeded."

    patched_update_weights_from_tensor._verl_patched_eagle_update_weights = True
    return patched_update_weights_from_tensor


def patch_sglang_eagle_update_weights_from_tensor() -> None:
    """Patch SGLang EAGLE update so target-only and draft-only sync skip the wrong side early."""
    global _SGLANG_EAGLE_UPDATE_PATCHED
    if _SGLANG_EAGLE_UPDATE_PATCHED:
        return

    patched_classes = []
    for module_name in (
        "sglang.srt.speculative.eagle_worker",
        "sglang.srt.speculative.eagle_worker_v2",
    ):
        try:
            module = importlib.import_module(module_name)
        except Exception:
            continue

        for class_name, cls in vars(module).items():
            if not isinstance(cls, type) or not class_name.lower().startswith("eagle"):
                continue

            original_update_weights = getattr(cls, "update_weights_from_tensor", None)
            if original_update_weights is None or getattr(
                original_update_weights, "_verl_patched_eagle_update_weights", False
            ):
                continue

            cls.update_weights_from_tensor = _make_verl_eagle_update_weights_patch(original_update_weights)
            patched_classes.append(f"{module_name}.{class_name}")

    if patched_classes:
        _SGLANG_EAGLE_UPDATE_PATCHED = True
        logger.info("Patched SGLang EAGLE routed weight update for %s", ", ".join(patched_classes))


def _is_sglang_npu_backend() -> bool:
    for module_name in ("sglang.srt.utils.common", "sglang.srt.utils"):
        try:
            is_npu = getattr(importlib.import_module(module_name), "is_npu", None)
        except Exception:  # noqa: BLE001
            continue
        if callable(is_npu) and is_npu():
            return True

    return hasattr(torch, "npu") and torch.npu.is_available()


def _normalize_sglang_npu_eagle_verify_mode(mode: str | None) -> str | None:
    if mode is None:
        return None
    normalized_mode = mode.strip().lower().replace("-", "_")
    if normalized_mode in {"0", "false", "off", "greedy"}:
        return "greedy"
    if normalized_mode in {
        "1",
        "true",
        "on",
        "target",
        "target_only",
        "fastrl",
        "fastrl_like",
        "fast_rl",
        "fast_rl_like",
        "vllm",
        "vllm_like",
    }:
        return "target_only"
    return None


def _sglang_npu_eagle_verify_mode(version_env: str | None = None) -> str:
    mode = _normalize_sglang_npu_eagle_verify_mode(os.getenv(version_env)) if version_env else None
    if mode is not None:
        return mode
    mode = _normalize_sglang_npu_eagle_verify_mode(os.getenv(_EAGLE_VERIFY_MODE_ENV))
    if mode is not None:
        return mode
    if version_env == _EAGLE_V1_VERIFY_MODE_ENV:
        legacy_mode = _normalize_sglang_npu_eagle_verify_mode(os.getenv(_EAGLE_V1_TARGET_SAMPLING_ENV))
        if legacy_mode is not None:
            return legacy_mode
    return "target_only"


def _sglang_npu_eagle_v1_verify_mode() -> str:
    return _sglang_npu_eagle_verify_mode(_EAGLE_V1_VERIFY_MODE_ENV)


def _sglang_npu_eagle_v2_verify_mode() -> str:
    return _sglang_npu_eagle_verify_mode(_EAGLE_V2_VERIFY_MODE_ENV)


def _renorm_probs_by_top_k_top_p(
    probs: torch.Tensor,
    top_ks: torch.Tensor,
    top_ps: torch.Tensor,
) -> torch.Tensor:
    vocab_size = probs.shape[-1]
    top_ks = top_ks.to(device=probs.device, dtype=torch.long).view(-1)
    top_ps = top_ps.to(device=probs.device, dtype=probs.dtype).view(-1)

    vocab_size_tensor = torch.full_like(top_ks, vocab_size)
    top_ks = torch.where((top_ks <= 0) | (top_ks >= _SGLANG_TOP_K_ALL), vocab_size_tensor, top_ks)
    top_ks = torch.minimum(top_ks, vocab_size_tensor)

    if bool(torch.all(top_ks >= vocab_size).item()) and bool(torch.all(top_ps >= 1.0).item()):
        return probs

    sorted_probs, sorted_indices = torch.sort(probs, dim=-1, descending=True)
    ranks = torch.arange(vocab_size, device=probs.device).view(1, -1)
    sorted_probs = sorted_probs.masked_fill(ranks >= top_ks.view(-1, 1), 0.0)

    cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
    sorted_probs = sorted_probs.masked_fill((cumulative_probs - sorted_probs) > top_ps.view(-1, 1), 0.0)

    normalizer = sorted_probs.sum(dim=-1, keepdim=True)
    sorted_probs = torch.where(normalizer > 0, sorted_probs / normalizer.clamp_min(torch.finfo(probs.dtype).tiny), 0.0)

    renormed_probs = torch.zeros_like(probs)
    renormed_probs.scatter_(dim=1, index=sorted_indices, src=sorted_probs)
    return renormed_probs


def _top_k_renorm_prob_torch(probs: torch.Tensor, top_ks: torch.Tensor) -> torch.Tensor:
    top_ps = torch.ones((probs.shape[0],), dtype=probs.dtype, device=probs.device)
    return _renorm_probs_by_top_k_top_p(probs, top_ks, top_ps)


def _top_p_renorm_prob_torch(probs: torch.Tensor, top_ps: torch.Tensor) -> torch.Tensor:
    top_ks = torch.full((probs.shape[0],), probs.shape[-1], dtype=torch.long, device=probs.device)
    return _renorm_probs_by_top_k_top_p(probs, top_ks, top_ps)


def _sample_from_probs_with_coin(probs: torch.Tensor, coin: torch.Tensor) -> torch.Tensor:
    squeeze_output = probs.dim() == 1
    if squeeze_output:
        probs = probs.unsqueeze(0)

    totals = probs.sum(dim=-1, keepdim=True)
    probs = torch.where(totals > 0, probs, torch.ones_like(probs))
    totals = probs.sum(dim=-1, keepdim=True)
    threshold = coin.to(dtype=probs.dtype, device=probs.device).view(-1, 1) * totals
    cumulative = torch.cumsum(probs, dim=-1)
    samples = torch.argmax((cumulative > threshold).to(torch.int32), dim=-1).to(torch.int32)
    return samples[0] if squeeze_output else samples


def _try_sglang_tree_speculative_sampling_kernel(
    predicts: torch.Tensor,
    accept_index: torch.Tensor,
    accept_token_num: torch.Tensor,
    candidates: torch.Tensor,
    retrive_index: torch.Tensor,
    retrive_next_token: torch.Tensor,
    retrive_next_sibling: torch.Tensor,
    uniform_samples: torch.Tensor,
    uniform_samples_for_final_sampling: torch.Tensor,
    target_probs: torch.Tensor,
    draft_probs: torch.Tensor,
    threshold_single: float,
    threshold_acc: float,
    deterministic: bool,
) -> bool:
    if target_probs.device.type != "cuda" or draft_probs.shape != target_probs.shape:
        return False

    try:
        from sgl_kernel import tree_speculative_sampling_target_only
    except Exception as exc:  # noqa: BLE001
        logger.debug("SGLang tree speculative CUDA kernel is unavailable: %s", exc)
        return False

    try:
        tree_speculative_sampling_target_only(
            predicts=predicts,
            accept_index=accept_index,
            accept_token_num=accept_token_num,
            candidates=candidates,
            retrive_index=retrive_index,
            retrive_next_token=retrive_next_token,
            retrive_next_sibling=retrive_next_sibling,
            uniform_samples=uniform_samples,
            uniform_samples_for_final_sampling=uniform_samples_for_final_sampling,
            target_probs=target_probs,
            draft_probs=draft_probs,
            threshold_single=threshold_single,
            threshold_acc=threshold_acc,
            deterministic=deterministic,
        )
        return True
    except Exception as exc:  # noqa: BLE001
        logger.debug("SGLang tree speculative CUDA kernel failed; falling back to torch path: %s", exc)
        return False


def _tree_speculative_sampling_target_only_vectorized_torch(
    predicts: torch.Tensor,
    accept_index: torch.Tensor,
    accept_token_num: torch.Tensor,
    candidates: torch.Tensor,
    retrive_index: torch.Tensor,
    retrive_next_token: torch.Tensor,
    retrive_next_sibling: torch.Tensor,
    uniform_samples: torch.Tensor,
    uniform_samples_for_final_sampling: torch.Tensor,
    target_probs: torch.Tensor,
    draft_probs: torch.Tensor,
    threshold_single: float,
    threshold_acc: float,
) -> None:
    batch_size, num_draft_tokens = candidates.shape
    num_speculative_tokens = accept_index.shape[1]
    device = target_probs.device
    vocab_size = target_probs.shape[-1]

    threshold_acc = max(float(threshold_acc), 1e-9)
    threshold_single = float(threshold_single)

    accept_index.fill_(-1)
    accept_token_num.zero_()

    batch_indices = torch.arange(batch_size, dtype=torch.long, device=device)
    cur_prob_idx = torch.zeros((batch_size,), dtype=torch.long, device=device)
    accepted_count = torch.zeros((batch_size,), dtype=torch.long, device=device)
    active = torch.ones((batch_size,), dtype=torch.bool, device=device)

    last_accepted_retrive_idx = retrive_index[:, 0].to(torch.long)
    accept_index[:, 0].copy_(last_accepted_retrive_idx.to(dtype=accept_index.dtype))
    coin = uniform_samples[:, 0]
    residual_draft_probs = draft_probs[:, 0, :]
    residual_draft_probs.zero_()

    for _ in range(1, num_speculative_tokens):
        sibling_idx = torch.where(
            active,
            retrive_next_token[batch_indices, cur_prob_idx],
            torch.full_like(cur_prob_idx, -1),
        )
        found_idx = torch.full_like(cur_prob_idx, -1)
        prob_acc = torch.zeros((batch_size,), dtype=target_probs.dtype, device=device)

        for _ in range(num_draft_tokens):
            valid = active & (found_idx < 0) & (sibling_idx >= 0)
            safe_sibling_idx = sibling_idx.clamp_min(0)
            draft_token_id = candidates[batch_indices, safe_sibling_idx].to(torch.long)
            target_prob_single = target_probs[batch_indices, cur_prob_idx, draft_token_id]
            target_prob_single = torch.where(valid, target_prob_single, torch.zeros_like(target_prob_single))
            next_prob_acc = prob_acc + target_prob_single

            old_residual = residual_draft_probs.gather(dim=1, index=draft_token_id.view(-1, 1)).squeeze(1)
            residual_update = torch.where(valid, target_prob_single, old_residual)
            residual_draft_probs.scatter_(dim=1, index=draft_token_id.view(-1, 1), src=residual_update.view(-1, 1))

            accepted = valid & (
                (coin <= (next_prob_acc / threshold_acc)) | (target_prob_single >= threshold_single)
            )
            found_idx = torch.where(accepted, sibling_idx, found_idx)
            prob_acc = torch.where(valid, next_prob_acc, prob_acc)
            sibling_idx = torch.where(
                valid & ~accepted,
                retrive_next_sibling[batch_indices, safe_sibling_idx],
                sibling_idx,
            )

        accepted = active & (found_idx >= 0)
        safe_found_idx = found_idx.clamp_min(0)
        accepted_token_id = candidates[batch_indices, safe_found_idx].to(dtype=predicts.dtype)
        accepted_retrive_idx = retrive_index[batch_indices, safe_found_idx].to(torch.long)

        old_predicts = predicts.gather(dim=0, index=last_accepted_retrive_idx)
        predict_updates = torch.where(accepted, accepted_token_id, old_predicts)
        predicts.scatter_(dim=0, index=last_accepted_retrive_idx, src=predict_updates)

        next_accepted_count = accepted_count + accepted.to(dtype=torch.long)
        accept_index_position = next_accepted_count.clamp_max(num_speculative_tokens - 1).view(-1, 1)
        old_accept_index = accept_index.gather(dim=1, index=accept_index_position).squeeze(1)
        accept_index_updates = torch.where(
            accepted,
            accepted_retrive_idx.to(dtype=accept_index.dtype),
            old_accept_index,
        )
        accept_index.scatter_(dim=1, index=accept_index_position, src=accept_index_updates.view(-1, 1))

        cur_prob_idx = torch.where(accepted, safe_found_idx, cur_prob_idx)
        last_accepted_retrive_idx = torch.where(accepted, accepted_retrive_idx, last_accepted_retrive_idx)
        accepted_count = next_accepted_count
        coin = uniform_samples[batch_indices, cur_prob_idx]
        active = accepted
        residual_draft_probs.mul_((~accepted).to(dtype=residual_draft_probs.dtype).view(-1, 1))

    accept_token_num.copy_(accepted_count.to(dtype=accept_token_num.dtype))

    final_target_probs = target_probs[batch_indices, cur_prob_idx]
    residual_probs = torch.clamp(final_target_probs - residual_draft_probs, min=0.0)
    need_residual = accepted_count != (num_speculative_tokens - 1)
    final_probs = torch.where(need_residual.view(-1, 1), residual_probs, final_target_probs)
    final_probs = torch.where(final_probs.sum(dim=-1, keepdim=True) > 0, final_probs, final_target_probs)
    final_token_ids = _sample_from_probs_with_coin(final_probs, uniform_samples_for_final_sampling)
    predicts.scatter_(dim=0, index=last_accepted_retrive_idx, src=final_token_ids.to(dtype=predicts.dtype))


def _tree_speculative_sampling_target_only_torch(
    predicts: torch.Tensor,
    accept_index: torch.Tensor,
    accept_token_num: torch.Tensor,
    candidates: torch.Tensor,
    retrive_index: torch.Tensor,
    retrive_next_token: torch.Tensor,
    retrive_next_sibling: torch.Tensor,
    uniform_samples: torch.Tensor,
    uniform_samples_for_final_sampling: torch.Tensor,
    target_probs: torch.Tensor,
    draft_probs: torch.Tensor,
    threshold_single: float = 1.0,
    threshold_acc: float = 1.0,
    deterministic: bool = True,
) -> None:
    if _try_sglang_tree_speculative_sampling_kernel(
        predicts=predicts,
        accept_index=accept_index,
        accept_token_num=accept_token_num,
        candidates=candidates,
        retrive_index=retrive_index,
        retrive_next_token=retrive_next_token,
        retrive_next_sibling=retrive_next_sibling,
        uniform_samples=uniform_samples,
        uniform_samples_for_final_sampling=uniform_samples_for_final_sampling,
        target_probs=target_probs,
        draft_probs=draft_probs,
        threshold_single=threshold_single,
        threshold_acc=threshold_acc,
        deterministic=deterministic,
    ):
        return

    _tree_speculative_sampling_target_only_vectorized_torch(
        predicts=predicts,
        accept_index=accept_index,
        accept_token_num=accept_token_num,
        candidates=candidates,
        retrive_index=retrive_index,
        retrive_next_token=retrive_next_token,
        retrive_next_sibling=retrive_next_sibling,
        uniform_samples=uniform_samples,
        uniform_samples_for_final_sampling=uniform_samples_for_final_sampling,
        target_probs=target_probs,
        draft_probs=draft_probs,
        threshold_single=threshold_single,
        threshold_acc=threshold_acc,
    )


def _make_sglang_eagle_v2_sample_patch(original_sample):
    sample_globals = dict(original_sample.__globals__)
    sample_globals["_is_npu"] = False
    sample_globals["top_k_renorm_prob"] = _top_k_renorm_prob_torch
    sample_globals["top_p_renorm_prob"] = _top_p_renorm_prob_torch
    sample_globals["tree_speculative_sampling_target_only"] = _tree_speculative_sampling_target_only_torch
    sample = FunctionType(
        original_sample.__code__,
        sample_globals,
        original_sample.__name__,
        original_sample.__defaults__,
        original_sample.__closure__,
    )
    sample.__kwdefaults__ = getattr(original_sample, "__kwdefaults__", None)
    sample.__annotations__ = getattr(original_sample, "__annotations__", {}).copy()
    sample._verl_patched_npu_eagle_sampling = True
    return sample


def patch_sglang_npu_eagle_target_sampling() -> None:
    """Patch SGLang NPU EAGLE verification to use FastRL-style target-only sampling."""
    global _SGLANG_NPU_EAGLE_SAMPLING_PATCHED
    if _SGLANG_NPU_EAGLE_SAMPLING_PATCHED or not _is_sglang_npu_backend():
        return

    patched_targets = []

    v1_verify_mode = _sglang_npu_eagle_v1_verify_mode()
    if v1_verify_mode != "greedy":
        try:
            eagle_info = importlib.import_module("sglang.srt.speculative.eagle_info")
            eagle_info.top_k_renorm_prob = _top_k_renorm_prob_torch
            eagle_info.top_p_renorm_prob = _top_p_renorm_prob_torch
            eagle_info.tree_speculative_sampling_target_only = _tree_speculative_sampling_target_only_torch
            eagle_info.TREE_SPEC_KERNEL_AVAILABLE = True
            patched_targets.append(f"sglang.srt.speculative.eagle_info({v1_verify_mode})")
        except Exception as exc:  # noqa: BLE001
            logger.debug("Skip SGLang EAGLE v1 target sampling patch: %s", exc)
    else:
        logger.info(
            "Skip SGLang EAGLE v1 target sampling patch. Set %s=target_only to enable it.",
            _EAGLE_V1_VERIFY_MODE_ENV,
        )

    v2_verify_mode = _sglang_npu_eagle_v2_verify_mode()
    if v2_verify_mode != "greedy":
        try:
            eagle_info_v2 = importlib.import_module("sglang.srt.speculative.eagle_info_v2")
            mixin = getattr(eagle_info_v2, "EagleVerifyInputV2Mixin", None)
            original_sample = getattr(mixin, "sample", None)
            if original_sample is not None and not getattr(original_sample, "_verl_patched_npu_eagle_sampling", False):
                mixin.sample = _make_sglang_eagle_v2_sample_patch(original_sample)
                patched_targets.append(f"sglang.srt.speculative.eagle_info_v2({v2_verify_mode})")
        except Exception as exc:  # noqa: BLE001
            logger.debug("Skip SGLang EAGLE v2 target sampling patch: %s", exc)
    else:
        logger.info(
            "Skip SGLang EAGLE v2 target sampling patch. Set %s=target_only to enable it.",
            _EAGLE_V2_VERIFY_MODE_ENV,
        )

    if patched_targets:
        _SGLANG_NPU_EAGLE_SAMPLING_PATCHED = True
        logger.info("Patched SGLang NPU EAGLE target sampling for %s", ", ".join(patched_targets))


def _extract_hf_hidden_states(outputs):
    if hasattr(outputs, "hidden_states"):
        return outputs.hidden_states

    if not isinstance(outputs, (tuple, list)):
        return None

    for item in outputs[1:]:
        if not isinstance(item, (tuple, list)) or len(item) == 0:
            continue
        first = item[0]
        if torch.is_tensor(first) and first.dim() >= 3:
            return item

    return None


def _normalize_eagle3_capture_layers(layer_ids, num_layers: int) -> list[int]:
    if layer_ids is None:
        capture_layers = [2, num_layers // 2, num_layers - 3]
    else:
        # Match SGLang native Llama semantics: layer id i captures the output
        # of layer i, which is hidden_states[i + 1] in HF output_hidden_states.
        capture_layers = [int(layer_id) + 1 for layer_id in layer_ids]

    return [layer_id for layer_id in capture_layers if 0 <= layer_id <= num_layers]


def _call_sglang_forward_with_supported_kwargs(original_forward, self, *args, **kwargs):
    try:
        parameters = inspect.signature(original_forward).parameters
    except (TypeError, ValueError):
        return original_forward(self, *args, **kwargs)

    if any(param.kind == inspect.Parameter.VAR_KEYWORD for param in parameters.values()):
        supported_kwargs = kwargs
    else:
        supported_kwargs = {key: value for key, value in kwargs.items() if key in parameters}
    return original_forward(self, *args, **supported_kwargs)


def _call_module_with_supported_kwargs(module: torch.nn.Module, **kwargs):
    try:
        parameters = inspect.signature(module.forward).parameters
    except (TypeError, ValueError):
        return module(**kwargs)

    if not any(param.kind == inspect.Parameter.VAR_KEYWORD for param in parameters.values()):
        kwargs = {key: value for key, value in kwargs.items() if key in parameters}
    return module(**kwargs)


def _format_transformers_position_ids(model_obj, positions):
    if hasattr(model_obj, "_format_position_ids"):
        return model_obj._format_position_ids(positions)

    model_config = getattr(model_obj, "model_config", None)
    if getattr(model_config, "uses_mrope", False):
        return positions[:, None]
    return positions[None, ...]


def _set_transformers_eagle3_capture_layers(self, layer_ids=None):
    pp_group = getattr(self, "pp_group", None)
    if pp_group is not None and not getattr(pp_group, "is_last_rank", True):
        return

    text_config = getattr(self, "text_config", getattr(self, "config", None))
    num_layers = int(getattr(text_config, "num_hidden_layers", 0) or 0)
    self.capture_aux_hidden_states = True
    self._verl_eagle3_capture_layer_ids = _normalize_eagle3_capture_layers(layer_ids, num_layers)


def _run_transformers_hf_backbone_with_aux_hidden_states(
    model_obj,
    input_ids,
    input_embeds,
    positions,
    forward_batch=None,
    extra_kwargs: dict | None = None,
):
    hf_input_ids = None if input_ids is None else input_ids[None, ...]
    hf_input_embeds = None
    if input_embeds is not None:
        hf_input_embeds = input_embeds[None, ...]
        hf_input_ids = None

    embed_scale = getattr(model_obj, "embed_scale", None)
    if embed_scale is not None and hf_input_ids is not None and hf_input_embeds is None:
        hf_input_embeds = model_obj.model.get_input_embeddings()(hf_input_ids) * embed_scale
        hf_input_ids = None

    model_kwargs = {
        "input_ids": hf_input_ids,
        "inputs_embeds": hf_input_embeds,
        "use_cache": False,
        "position_ids": _format_transformers_position_ids(model_obj, positions),
        "return_dict": False,
        "output_hidden_states": True,
    }
    if forward_batch is not None:
        model_kwargs["forward_batch"] = forward_batch
    if hasattr(model_obj, "attention_instances"):
        model_kwargs["attention_instances"] = model_obj.attention_instances
    if extra_kwargs:
        model_kwargs.update(extra_kwargs)

    outputs = _call_module_with_supported_kwargs(model_obj.model, **model_kwargs)
    hidden_states = outputs[0][0, ...]

    all_hidden_states = _extract_hf_hidden_states(outputs)
    aux_hidden_states = []
    if all_hidden_states is not None:
        for layer_id in getattr(model_obj, "_verl_eagle3_capture_layer_ids", []):
            if layer_id >= len(all_hidden_states):
                continue
            aux_hidden = all_hidden_states[layer_id]
            if torch.is_tensor(aux_hidden):
                aux_hidden_states.append(aux_hidden[0, ...] if aux_hidden.dim() >= 3 else aux_hidden)

    model_obj._verl_last_aux_hidden_states = aux_hidden_states or None
    return hidden_states


def _call_sglang_logits_processor(logits_processor, input_ids, hidden_states, lm_head, forward_batch, aux_hidden_states):
    try:
        signature = inspect.signature(logits_processor.forward)
    except (AttributeError, TypeError, ValueError):
        try:
            signature = inspect.signature(logits_processor)
        except (TypeError, ValueError):
            signature = None

    accepts_aux_hidden_states = False
    if signature is not None:
        parameters = signature.parameters
        accepts_aux_hidden_states = (
            "aux_hidden_states" in parameters
            or any(param.kind == inspect.Parameter.VAR_POSITIONAL for param in parameters.values())
            or sum(
                param.kind
                in (inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD)
                for param in parameters.values()
            )
            >= 5
        )

    if accepts_aux_hidden_states:
        return logits_processor(input_ids, hidden_states, lm_head, forward_batch, aux_hidden_states)
    return logits_processor(input_ids, hidden_states, lm_head, forward_batch)


def _patch_sglang_transformers_base(transformers_base) -> bool:
    original_run_hf_backbone = getattr(transformers_base, "_run_hf_backbone", None)
    original_forward = getattr(transformers_base, "forward", None)
    if original_run_hf_backbone is None or original_forward is None:
        return False

    if getattr(original_forward, "_verl_patched_transformers_eagle3_capture", False):
        return True

    if not hasattr(transformers_base, "set_eagle3_layers_to_capture"):
        transformers_base.set_eagle3_layers_to_capture = _set_transformers_eagle3_capture_layers

    def patched_run_hf_backbone(self, input_ids, input_embeds, positions, forward_batch, **kwargs):
        if getattr(self, "capture_aux_hidden_states", False):
            return _run_transformers_hf_backbone_with_aux_hidden_states(
                self,
                input_ids=input_ids,
                input_embeds=input_embeds,
                positions=positions,
                forward_batch=forward_batch,
                extra_kwargs=kwargs,
            )

        self._verl_last_aux_hidden_states = None
        return original_run_hf_backbone(self, input_ids, input_embeds, positions, forward_batch, **kwargs)

    @torch.no_grad()
    def patched_forward(
        self,
        input_ids,
        positions,
        forward_batch,
        pp_proxy_tensors=None,
        input_embeds=None,
        get_embedding=False,
        **kwargs,
    ):
        if not getattr(self, "capture_aux_hidden_states", False):
            return _call_sglang_forward_with_supported_kwargs(
                original_forward,
                self,
                input_ids,
                positions,
                forward_batch,
                pp_proxy_tensors=pp_proxy_tensors,
                input_embeds=input_embeds,
                get_embedding=get_embedding,
                **kwargs,
            )

        runtime_input_ids = input_ids
        runtime_input_embeds = input_embeds
        pp_group = getattr(self, "pp_group", None)
        is_first_rank = getattr(pp_group, "is_first_rank", True)
        is_last_rank = getattr(pp_group, "is_last_rank", True)
        if not is_first_rank:
            assert pp_proxy_tensors is not None
            runtime_input_ids = None
            runtime_input_embeds = pp_proxy_tensors["hidden_states"]

        hidden_states = self._forward_hidden_states(
            input_ids=runtime_input_ids,
            positions=positions,
            forward_batch=forward_batch,
            input_embeds=runtime_input_embeds,
        )

        if not is_last_rank:
            from sglang.srt.model_executor.forward_batch_info import PPProxyTensors

            return PPProxyTensors({"hidden_states": hidden_states, "residual": hidden_states})

        if get_embedding:
            assert self.pooler is not None, "pooling is not enabled for this model class"
            return self.pooler(hidden_states, forward_batch)

        assert self.logits_processor is not None and self.lm_head is not None
        return _call_sglang_logits_processor(
            self.logits_processor,
            input_ids,
            hidden_states,
            self.lm_head,
            forward_batch,
            getattr(self, "_verl_last_aux_hidden_states", None),
        )

    patched_run_hf_backbone._verl_patched_transformers_eagle3_capture = True
    patched_forward._verl_patched_transformers_eagle3_capture = True
    transformers_base._run_hf_backbone = patched_run_hf_backbone
    transformers_base.forward = patched_forward
    return True


def _patch_transformers_module(module_name: str) -> list[str]:
    try:
        transformers_module = importlib.import_module(module_name)
    except Exception as exc:  # noqa: BLE001
        logger.debug("Skip SGLang Transformers EAGLE3 capture patch for %s: %s", module_name, exc)
        return []

    patched_classes = []
    transformers_base = getattr(transformers_module, "TransformersBase", None)
    for class_name, cls in vars(transformers_module).items():
        if not isinstance(cls, type) or not class_name.startswith("Transformers"):
            continue
        if transformers_base is not None and not issubclass(cls, transformers_base):
            continue

        if _patch_sglang_transformers_base(cls):
            patched_classes.append(f"{module_name}.{class_name}")
            continue

        if not hasattr(cls, "set_eagle3_layers_to_capture"):
            cls.set_eagle3_layers_to_capture = _set_transformers_eagle3_capture_layers
            patched_classes.append(f"{module_name}.{class_name}")

    return patched_classes


def patch_sglang_transformers_eagle3_capture() -> None:
    """Add EAGLE3 aux hidden-state capture to SGLang's Transformers fallback backend."""
    global _SGLANG_TRANSFORMERS_EAGLE3_CAPTURE_PATCHED
    if _SGLANG_TRANSFORMERS_EAGLE3_CAPTURE_PATCHED:
        return

    patched_classes = _patch_transformers_module("sglang.srt.models.transformers")
    if patched_classes:
        _SGLANG_TRANSFORMERS_EAGLE3_CAPTURE_PATCHED = True
        logger.info("Patched Transformers EAGLE3 capture for %s", ", ".join(patched_classes))


def _make_sglang_hidden_states_tensor_output_patch(original_method):
    """Patch SGLang output processors to keep hidden-state chunks as CPU tensors.

    SGLang 0.5.9 and 0.5.10 both append hidden states with
    `.cpu().clone().tolist()`. The `.tolist()` conversion serializes every
    hidden value through Python objects and dominates rollout latency when
    drafter collection is enabled. Keeping CPU tensors preserves the existing
    ownership/lifetime behavior while avoiding Python list materialization.
    """
    try:
        source = inspect.getsource(original_method)
    except (OSError, TypeError):
        return None

    source = textwrap.dedent(source)
    patched_source = re.sub(
        r"\.cpu\(\)\s*\.clone\(\)\s*\.tolist\(\)",
        '.detach().to("cpu", copy=True)',
        source,
    )
    if patched_source == source:
        return None

    namespace = {}
    exec(  # noqa: S102
        "from __future__ import annotations\n" + patched_source,
        original_method.__globals__,
        namespace,
    )
    patched_method = namespace[original_method.__name__]
    patched_method = wraps(original_method)(patched_method)
    patched_method._verl_patched_hidden_states_tensor_output = True
    return patched_method


def patch_sglang_hidden_states_tensor_output() -> None:
    """Return SGLang hidden-state chunks as CPU tensors instead of Python lists."""
    global _SGLANG_HIDDEN_STATES_TENSOR_OUTPUT_PATCHED
    if _SGLANG_HIDDEN_STATES_TENSOR_OUTPUT_PATCHED:
        return

    try:
        module = importlib.import_module("sglang.srt.managers.scheduler_output_processor_mixin")
        processor_cls = getattr(module, "SchedulerOutputProcessorMixin")
    except Exception as exc:  # noqa: BLE001
        logger.debug("Skip SGLang hidden-state tensor output patch: %s", exc)
        return

    patched_methods = []
    for method_name in ("process_batch_result_prefill", "process_batch_result_decode"):
        original_method = getattr(processor_cls, method_name, None)
        if original_method is None or getattr(
            original_method,
            "_verl_patched_hidden_states_tensor_output",
            False,
        ):
            continue

        patched_method = _make_sglang_hidden_states_tensor_output_patch(original_method)
        if patched_method is None:
            logger.debug("Skip SGLang hidden-state tensor output patch for %s", method_name)
            continue

        setattr(processor_cls, method_name, patched_method)
        patched_methods.append(method_name)

    if patched_methods:
        _SGLANG_HIDDEN_STATES_TENSOR_OUTPUT_PATCHED = True
        logger.info("Patched SGLang hidden-state tensor output for %s", ", ".join(patched_methods))


def _apply_sglang_child_process_patches() -> None:
    patch_sglang_transformers_eagle3_capture()
    patch_sglang_eagle_update_weights_from_tensor()
    patch_sglang_npu_eagle_target_sampling()
    patch_sglang_hidden_states_tensor_output()


def _run_scheduler_process_with_verl_patches(*args, **kwargs):
    _apply_sglang_child_process_patches()
    return _ORIGINAL_SGLANG_RUN_SCHEDULER_PROCESS(*args, **kwargs)


_run_scheduler_process_with_verl_patches._verl_patched_eagle_update_weights = True
setattr(_run_scheduler_process_with_verl_patches, _SCHEDULER_PROCESS_PATCH_ATTR, True)


def _run_direct_scheduler_process_with_verl_patches(*args, **kwargs):
    global _ORIGINAL_SGLANG_DIRECT_RUN_SCHEDULER_PROCESS

    _apply_sglang_child_process_patches()
    if _ORIGINAL_SGLANG_DIRECT_RUN_SCHEDULER_PROCESS is None:
        scheduler_module = importlib.import_module("sglang.srt.managers.scheduler")
        _ORIGINAL_SGLANG_DIRECT_RUN_SCHEDULER_PROCESS = scheduler_module.run_scheduler_process
    return _ORIGINAL_SGLANG_DIRECT_RUN_SCHEDULER_PROCESS(*args, **kwargs)


_run_direct_scheduler_process_with_verl_patches._verl_patched_eagle_update_weights = True
setattr(_run_direct_scheduler_process_with_verl_patches, _SCHEDULER_PROCESS_PATCH_ATTR, True)


def patch_sglang_scheduler_process_entrypoints() -> None:
    """Install child-process patches for both SGLang 0.5.9 and 0.5.10 launch paths."""
    global _SGLANG_SCHEDULER_PROCESS_PATCHED
    if _SGLANG_SCHEDULER_PROCESS_PATCHED:
        return

    patched_entrypoints = []
    modules = [sglang.srt.entrypoints.engine]
    try:
        modules.append(importlib.import_module("sglang.srt.managers.scheduler"))
    except Exception as exc:  # noqa: BLE001
        logger.debug("Skip direct scheduler entrypoint patch: %s", exc)

    for module in modules:
        original_run_scheduler_process = getattr(module, "run_scheduler_process", None)
        if original_run_scheduler_process is None or getattr(
            original_run_scheduler_process,
            _SCHEDULER_PROCESS_PATCH_ATTR,
            False,
        ):
            continue
        if module is sglang.srt.entrypoints.engine:
            module.run_scheduler_process = _run_scheduler_process_with_verl_patches
        else:
            global _ORIGINAL_SGLANG_DIRECT_RUN_SCHEDULER_PROCESS
            _ORIGINAL_SGLANG_DIRECT_RUN_SCHEDULER_PROCESS = original_run_scheduler_process
            module.run_scheduler_process = _run_direct_scheduler_process_with_verl_patches
        patched_entrypoints.append(module.__name__)

    if patched_entrypoints:
        _SGLANG_SCHEDULER_PROCESS_PATCHED = True
        logger.info("Patched SGLang scheduler entrypoints for %s", ", ".join(patched_entrypoints))

def install_sglang_verl_patches(
    set_envs_and_config: Callable | None = None,
    target_weight_loader: str | None = None,
    draft_weight_loader: str | None = None,
) -> None:
    configure_sglang_eagle_weight_update_patch(target_weight_loader, draft_weight_loader)
    patch_sglang_transformers_eagle3_capture()
    patch_sglang_eagle_update_weights_from_tensor()
    patch_sglang_npu_eagle_target_sampling()
    patch_sglang_hidden_states_tensor_output()
    patch_sglang_scheduler_process_entrypoints()

    if set_envs_and_config is not None:
        sglang.srt.entrypoints.engine._set_envs_and_config = set_envs_and_config
