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
from typing import Callable

import torch
import sglang.srt.entrypoints.engine
from sglang.srt.utils import MultiprocessingSerializer

try:
    import triton
    import triton.language as tl

    _HAS_TRITON = True
except Exception:  # noqa: BLE001
    _HAS_TRITON = False
    tl = None
    triton = None

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))

_TARGET_WEIGHT_LOADER_ENV = "VERL_SGLANG_TARGET_WEIGHT_LOADER"
_DRAFT_WEIGHT_LOADER_ENV = "VERL_SGLANG_DRAFT_WEIGHT_LOADER"
_EAGLE_VERIFY_MODE_ENV = "VERL_SGLANG_NPU_EAGLE_VERIFY_MODE"
_EAGLE_V1_TARGET_SAMPLING_ENV = "VERL_SGLANG_NPU_EAGLE_V1_TARGET_SAMPLING"
_EAGLE_V1_VERIFY_MODE_ENV = "VERL_SGLANG_NPU_EAGLE_V1_VERIFY_MODE"
_EAGLE_FORCE_FP32_SAMPLING_ENV = "VERL_SGLANG_NPU_EAGLE_FORCE_FP32_SAMPLING"
_EAGLE_LINEAR_TRITON_ENV = "VERL_SGLANG_NPU_EAGLE_LINEAR_TRITON"
_EAGLE_TOP_K_RENORM_FAST_PATH_ENV = "VERL_SGLANG_NPU_EAGLE_TOP_K_RENORM_FAST_PATH"
_DISABLE_SGLANG_PATCH_ENV = "VERL_DISABLE_SGLANG_PATCH"
_SGLANG_PATCHES_ENV = "VERL_SGLANG_PATCHES"

_target_weight_loader: str | None = os.environ.get(_TARGET_WEIGHT_LOADER_ENV)
_draft_weight_loader: str | None = os.environ.get(_DRAFT_WEIGHT_LOADER_ENV)
_ORIGINAL_SGLANG_RUN_SCHEDULER_PROCESS = sglang.srt.entrypoints.engine.run_scheduler_process
_ORIGINAL_SGLANG_DIRECT_RUN_SCHEDULER_PROCESS = None
_SGLANG_EAGLE_UPDATE_PATCHED = False
_SGLANG_NPU_EAGLE_SAMPLING_PATCHED = False
_SGLANG_HIDDEN_STATES_TENSOR_OUTPUT_PATCHED = False
_SGLANG_EAGLE_VERIFY_HIDDEN_STATES_PATCHED = False
_SGLANG_SCHEDULER_PROCESS_PATCHED = False
_SCHEDULER_PROCESS_PATCH_ATTR = "_verl_patched_scheduler_process"
_SGLANG_TOP_K_ALL = 1 << 30
_SGLANG_PATCH_ALIASES = {
    "all": "all",
    "none": "none",
    "off": "none",
    "0": "none",
    "eagle_update_weights": "eagle_update_weights",
    "eagle_weight_update": "eagle_update_weights",
    "weight_update": "eagle_update_weights",
    "weight_routing": "eagle_update_weights",
    "route_weights": "eagle_update_weights",
    "target_draft_weight_routing": "eagle_update_weights",
    "npu_eagle_target_sampling": "npu_eagle_target_sampling",
    "target_sampling": "npu_eagle_target_sampling",
    "hidden_states_tensor_output": "hidden_states_tensor_output",
    "hidden_states": "hidden_states_tensor_output",
    "hidden_state_tensor_output": "hidden_states_tensor_output",
}


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


def _env_flag_enabled(name: str, default: bool = False) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    normalized = value.strip().lower()
    if normalized in {"1", "true", "on", "yes", "y"}:
        return True
    if normalized in {"0", "false", "off", "no", "n", ""}:
        return False
    return default


def _sglang_npu_eagle_force_fp32_sampling_enabled() -> bool:
    return _env_flag_enabled(_EAGLE_FORCE_FP32_SAMPLING_ENV, default=False)


def _sglang_npu_eagle_linear_triton_enabled() -> bool:
    return _env_flag_enabled(_EAGLE_LINEAR_TRITON_ENV, default=True)


def _triton_ascend_available() -> bool:
    if not (_HAS_TRITON and _is_sglang_npu_backend()):
        return False
    try:
        properties = triton.runtime.driver.active.utils.get_device_properties(torch.npu.current_device())
    except Exception as exc:  # noqa: BLE001
        logger.debug("Triton Ascend device properties are unavailable: %s", exc)
        return False
    return int(properties.get("num_aicore", -1)) > 0 and int(properties.get("num_vectorcore", -1)) > 0


def _sglang_npu_eagle_top_k_renorm_fast_path_enabled() -> bool:
    return _env_flag_enabled(_EAGLE_TOP_K_RENORM_FAST_PATH_ENV, default=False)


def _sglang_verl_patches_disabled() -> bool:
    return _env_flag_enabled(_DISABLE_SGLANG_PATCH_ENV, default=False)


def _selected_sglang_patches() -> set[str] | None:
    raw_value = os.getenv(_SGLANG_PATCHES_ENV)
    if raw_value is None or not raw_value.strip():
        return None

    selected = set()
    for item in re.split(r"[\s,]+", raw_value.strip()):
        key = item.strip().lower().replace("-", "_")
        if not key:
            continue
        patch_name = _SGLANG_PATCH_ALIASES.get(key, key)
        if patch_name == "all":
            return None
        if patch_name == "none":
            return set()
        selected.add(patch_name)
    return selected


def _sglang_patch_enabled(patch_name: str) -> bool:
    selected = _selected_sglang_patches()
    return selected is None or patch_name in selected


def _normalize_sglang_npu_eagle_verify_mode(mode: str | None) -> str | None:
    if mode is None:
        return None
    normalized_mode = mode.strip().lower().replace("-", "_")
    if normalized_mode in {"0", "false", "off", "greedy"}:
        return "greedy"
    if normalized_mode in {"1", "true", "on", "target", "target_only"}:
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


def _as_sglang_npu_eagle_sampling_float(tensor: torch.Tensor) -> torch.Tensor:
    if _sglang_npu_eagle_force_fp32_sampling_enabled() and tensor.is_floating_point():
        return tensor.to(dtype=torch.float32)
    return tensor


def _renorm_probs_by_top_k_top_p(
    probs: torch.Tensor,
    top_ks: torch.Tensor,
    top_ps: torch.Tensor,
) -> torch.Tensor:
    vocab_size = probs.shape[-1]
    probs_for_sampling = _as_sglang_npu_eagle_sampling_float(probs)
    top_ks = top_ks.to(device=probs_for_sampling.device, dtype=torch.long).view(-1)
    top_ps = top_ps.to(device=probs_for_sampling.device, dtype=probs_for_sampling.dtype).view(-1)

    vocab_size_tensor = torch.full_like(top_ks, vocab_size)
    top_ks = torch.where((top_ks <= 0) | (top_ks >= _SGLANG_TOP_K_ALL), vocab_size_tensor, top_ks)
    top_ks = torch.minimum(top_ks, vocab_size_tensor)

    if bool(torch.all(top_ks >= vocab_size).item()) and bool(torch.all(top_ps >= 1.0).item()):
        return probs_for_sampling

    sorted_probs, sorted_indices = torch.sort(probs_for_sampling, dim=-1, descending=True)
    ranks = torch.arange(vocab_size, device=probs_for_sampling.device).view(1, -1)
    sorted_probs = sorted_probs.masked_fill(ranks >= top_ks.view(-1, 1), 0.0)

    cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
    sorted_probs = sorted_probs.masked_fill((cumulative_probs - sorted_probs) > top_ps.view(-1, 1), 0.0)

    normalizer = sorted_probs.sum(dim=-1, keepdim=True)
    sorted_probs = torch.where(
        normalizer > 0,
        sorted_probs / normalizer.clamp_min(torch.finfo(probs_for_sampling.dtype).tiny),
        0.0,
    )

    renormed_probs = torch.zeros_like(probs_for_sampling)
    renormed_probs.scatter_(dim=1, index=sorted_indices, src=sorted_probs)
    return renormed_probs


def _top_k_renorm_prob_torch_fast(probs: torch.Tensor, top_ks: torch.Tensor) -> torch.Tensor:
    vocab_size = probs.shape[-1]
    probs_for_sampling = _as_sglang_npu_eagle_sampling_float(probs)
    if probs_for_sampling.numel() == 0:
        return probs_for_sampling

    top_ks = top_ks.to(device=probs_for_sampling.device, dtype=torch.long).view(-1)
    vocab_size_tensor = torch.full_like(top_ks, vocab_size)
    top_ks = torch.where((top_ks <= 0) | (top_ks >= _SGLANG_TOP_K_ALL), vocab_size_tensor, top_ks)
    top_ks = torch.minimum(top_ks, vocab_size_tensor)

    if bool(torch.all(top_ks >= vocab_size).item()):
        return probs_for_sampling

    fast_row_indices = torch.nonzero(top_ks < vocab_size, as_tuple=False).view(-1)
    if fast_row_indices.numel() == 0:
        return probs_for_sampling

    fast_top_ks = top_ks.index_select(0, fast_row_indices)
    max_top_k = int(fast_top_ks.max().item())
    if max_top_k <= 0:
        return probs_for_sampling

    fast_probs = probs_for_sampling.index_select(0, fast_row_indices)
    topk_probs, topk_indices = torch.topk(fast_probs, max_top_k, dim=-1)
    ranks = torch.arange(max_top_k, device=probs_for_sampling.device).view(1, -1)
    topk_probs = topk_probs.masked_fill(ranks >= fast_top_ks.view(-1, 1), 0.0)

    normalizer = topk_probs.sum(dim=-1, keepdim=True)
    topk_probs = torch.where(
        normalizer > 0,
        topk_probs / normalizer.clamp_min(torch.finfo(probs_for_sampling.dtype).tiny),
        0.0,
    )

    fast_renormed_probs = torch.zeros_like(fast_probs)
    fast_renormed_probs.scatter_(dim=1, index=topk_indices, src=topk_probs)
    if fast_row_indices.numel() == probs_for_sampling.shape[0]:
        return fast_renormed_probs

    renormed_probs = probs_for_sampling.clone()
    renormed_probs.index_copy_(0, fast_row_indices, fast_renormed_probs)
    return renormed_probs


def _top_k_renorm_prob_torch(probs: torch.Tensor, top_ks: torch.Tensor) -> torch.Tensor:
    if _sglang_npu_eagle_top_k_renorm_fast_path_enabled():
        try:
            return _top_k_renorm_prob_torch_fast(probs, top_ks)
        except Exception as exc:  # noqa: BLE001
            logger.debug("SGLang NPU top-k renorm fast path failed; falling back to sort path: %s", exc)
    top_ps = torch.ones(
        (probs.shape[0],),
        dtype=torch.float32 if _sglang_npu_eagle_force_fp32_sampling_enabled() else probs.dtype,
        device=probs.device,
    )
    return _renorm_probs_by_top_k_top_p(probs, top_ks, top_ps)


def _top_p_renorm_prob_torch(probs: torch.Tensor, top_ps: torch.Tensor) -> torch.Tensor:
    top_ks = torch.full((probs.shape[0],), probs.shape[-1], dtype=torch.long, device=probs.device)
    return _renorm_probs_by_top_k_top_p(probs, top_ks, top_ps)


def _sample_from_probs_with_coin(probs: torch.Tensor, coin: torch.Tensor) -> torch.Tensor:
    squeeze_output = probs.dim() == 1
    if squeeze_output:
        probs = probs.unsqueeze(0)

    probs_for_sampling = _as_sglang_npu_eagle_sampling_float(probs)
    totals = probs_for_sampling.sum(dim=-1, keepdim=True)
    probs_for_sampling = torch.where(
        totals > 0,
        probs_for_sampling,
        torch.ones_like(probs_for_sampling),
    )
    totals = probs_for_sampling.sum(dim=-1, keepdim=True)
    threshold = coin.to(
        dtype=probs_for_sampling.dtype,
        device=probs_for_sampling.device,
    ).view(-1, 1) * totals
    cumulative = torch.cumsum(probs_for_sampling, dim=-1)
    samples = torch.argmax((cumulative > threshold).to(torch.int32), dim=-1).to(torch.int32)
    return samples[0] if squeeze_output else samples


_tree_speculative_sampling_target_only_linear_triton_kernel = None
if triton is not None:

    @triton.jit(do_not_specialize=["threshold_single", "threshold_acc"])
    def _tree_speculative_sampling_target_only_linear_triton_kernel(
        predicts_ptr,
        accept_index_ptr,
        accept_token_num_ptr,
        candidates_ptr,
        retrive_index_ptr,
        retrive_next_token_ptr,
        uniform_samples_ptr,
        uniform_samples_for_final_sampling_ptr,
        target_probs_ptr,
        threshold_single,
        threshold_acc,
        NUM_DRAFT_TOKENS: tl.constexpr,
        NUM_SPECULATIVE_TOKENS: tl.constexpr,
        VOCAB_SIZE: tl.constexpr,
        SUB_BLOCK: tl.constexpr,
        NUM_VOCAB_BLOCKS: tl.constexpr,
        SPEC_BLOCK: tl.constexpr,
    ):
        req_idx = tl.program_id(0)
        row_base = req_idx * NUM_DRAFT_TOKENS
        accept_index_base = req_idx * NUM_SPECULATIVE_TOKENS

        spec_offsets = tl.arange(0, SPEC_BLOCK)
        tl.store(
            accept_index_ptr + accept_index_base + spec_offsets,
            -1,
            mask=spec_offsets < NUM_SPECULATIVE_TOKENS,
        )
        tl.store(accept_token_num_ptr + req_idx, 0)

        threshold_acc = tl.maximum(threshold_acc, 1.0e-9)
        cur_prob_idx = 0
        accepted_count = 0
        active = True
        residual_token_id = -1
        residual_token_prob = 0.0

        last_accepted_retrive_idx = tl.load(retrive_index_ptr + row_base)
        tl.store(accept_index_ptr + accept_index_base, last_accepted_retrive_idx)
        coin = tl.load(uniform_samples_ptr + row_base).to(tl.float32)

        for _ in range(1, NUM_SPECULATIVE_TOKENS):
            next_idx = tl.load(retrive_next_token_ptr + row_base + cur_prob_idx)
            valid = active & (next_idx >= 0)
            safe_next_idx = tl.maximum(next_idx, 0)
            draft_token_id = tl.load(candidates_ptr + row_base + safe_next_idx).to(tl.int64)
            target_prob_single = tl.load(
                target_probs_ptr + (row_base + cur_prob_idx) * VOCAB_SIZE + draft_token_id,
                mask=valid,
                other=0.0,
            ).to(tl.float32)
            accepted = valid & (
                (coin <= (target_prob_single / threshold_acc)) | (target_prob_single >= threshold_single)
            )

            if accepted:
                accepted_retrive_idx = tl.load(retrive_index_ptr + row_base + safe_next_idx)
                tl.store(predicts_ptr + last_accepted_retrive_idx, draft_token_id)
                accepted_count += 1
                tl.store(
                    accept_index_ptr + accept_index_base + accepted_count,
                    accepted_retrive_idx,
                    mask=accepted_count < NUM_SPECULATIVE_TOKENS,
                )
                cur_prob_idx = safe_next_idx
                last_accepted_retrive_idx = accepted_retrive_idx
                coin = tl.load(uniform_samples_ptr + row_base + cur_prob_idx).to(tl.float32)
                residual_token_id = -1
                residual_token_prob = 0.0
            else:
                if valid:
                    residual_token_id = draft_token_id
                    residual_token_prob = target_prob_single
                active = False

        tl.store(accept_token_num_ptr + req_idx, accepted_count)

        final_row_base = (row_base + cur_prob_idx) * VOCAB_SIZE
        need_residual = (accepted_count != (NUM_SPECULATIVE_TOKENS - 1)) & (residual_token_id >= 0)
        total = 0.0
        vocab_offsets = tl.arange(0, SUB_BLOCK)
        for block_idx in range(NUM_VOCAB_BLOCKS):
            token_offsets = block_idx * SUB_BLOCK + vocab_offsets
            mask = token_offsets < VOCAB_SIZE
            probs = tl.load(target_probs_ptr + final_row_base + token_offsets, mask=mask, other=0.0).to(tl.float32)
            if need_residual:
                probs = tl.where(
                    token_offsets == residual_token_id,
                    tl.maximum(probs - residual_token_prob, 0.0),
                    probs,
                )
            total += tl.sum(probs, axis=0)

        final_coin = tl.load(uniform_samples_for_final_sampling_ptr + req_idx).to(tl.float32)
        final_token_id = VOCAB_SIZE - 1
        if total <= 0.0:
            final_token_id = tl.minimum((final_coin * VOCAB_SIZE).to(tl.int64), VOCAB_SIZE - 1)
        else:
            threshold = final_coin * total
            cumulative = 0.0
            found = tl.full((), False, tl.int1)
            for block_idx in range(NUM_VOCAB_BLOCKS):
                token_offsets = block_idx * SUB_BLOCK + vocab_offsets
                mask = token_offsets < VOCAB_SIZE
                probs = tl.load(target_probs_ptr + final_row_base + token_offsets, mask=mask, other=0.0).to(tl.float32)
                if need_residual:
                    probs = tl.where(
                        token_offsets == residual_token_id,
                        tl.maximum(probs - residual_token_prob, 0.0),
                        probs,
                    )
                cumulative_probs = tl.cumsum(probs, 0) + cumulative
                hits = (cumulative_probs > threshold) & mask
                hit_values = hits.to(tl.int32)
                has_hit = tl.max(hit_values, axis=0) > 0
                if has_hit & (~found):
                    final_token_id = block_idx * SUB_BLOCK + tl.argmax(hit_values, axis=0)
                found = found | has_hit
                cumulative += tl.sum(probs, axis=0)

        tl.store(predicts_ptr + last_accepted_retrive_idx, final_token_id)


def _triton_next_power_of_2(value: int) -> int:
    return 1 << (max(int(value), 1) - 1).bit_length()


def _try_tree_speculative_sampling_target_only_linear_triton(
    predicts: torch.Tensor,
    accept_index: torch.Tensor,
    accept_token_num: torch.Tensor,
    candidates: torch.Tensor,
    retrive_index: torch.Tensor,
    retrive_next_token: torch.Tensor,
    uniform_samples: torch.Tensor,
    uniform_samples_for_final_sampling: torch.Tensor,
    target_probs: torch.Tensor,
    threshold_single: float,
    threshold_acc: float,
) -> bool:
    if not (_sglang_npu_eagle_linear_triton_enabled() and _triton_ascend_available()):
        return False
    if _tree_speculative_sampling_target_only_linear_triton_kernel is None:
        return False
    if target_probs.device.type != "npu":
        return False
    if target_probs.ndim != 3 or candidates.ndim != 2:
        return False
    if not all(
        tensor.is_contiguous()
        for tensor in (
            predicts,
            accept_index,
            accept_token_num,
            candidates,
            retrive_index,
            retrive_next_token,
            uniform_samples,
            uniform_samples_for_final_sampling,
            target_probs,
        )
    ):
        return False

    batch_size, num_draft_tokens = candidates.shape
    num_speculative_tokens = accept_index.shape[1]
    vocab_size = target_probs.shape[-1]
    if (
        batch_size == 0
        or num_draft_tokens == 0
        or num_speculative_tokens == 0
        or target_probs.shape[:2] != candidates.shape
        or uniform_samples.shape != candidates.shape
    ):
        return False

    target_probs_for_sampling = _as_sglang_npu_eagle_sampling_float(target_probs)
    uniform_samples_for_sampling = _as_sglang_npu_eagle_sampling_float(uniform_samples)
    final_uniform_samples_for_sampling = _as_sglang_npu_eagle_sampling_float(
        uniform_samples_for_final_sampling
    )
    if not (
        target_probs_for_sampling.is_contiguous()
        and uniform_samples_for_sampling.is_contiguous()
        and final_uniform_samples_for_sampling.is_contiguous()
    ):
        return False

    sub_block = 4096
    try:
        _tree_speculative_sampling_target_only_linear_triton_kernel[(batch_size,)](
            predicts,
            accept_index,
            accept_token_num,
            candidates,
            retrive_index,
            retrive_next_token,
            uniform_samples_for_sampling,
            final_uniform_samples_for_sampling,
            target_probs_for_sampling,
            float(threshold_single),
            max(float(threshold_acc), 1.0e-9),
            NUM_DRAFT_TOKENS=int(num_draft_tokens),
            NUM_SPECULATIVE_TOKENS=int(num_speculative_tokens),
            VOCAB_SIZE=int(vocab_size),
            SUB_BLOCK=sub_block,
            NUM_VOCAB_BLOCKS=(int(vocab_size) + sub_block - 1) // sub_block,
            SPEC_BLOCK=_triton_next_power_of_2(num_speculative_tokens),
        )
        return True
    except Exception as exc:  # noqa: BLE001
        logger.debug("SGLang NPU EAGLE linear target-only Triton kernel failed: %s", exc)
        return False


def _tree_speculative_sampling_target_only_linear_torch(
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
    del draft_probs, deterministic, retrive_next_sibling

    batch_size, _ = candidates.shape
    num_speculative_tokens = accept_index.shape[1]
    target_probs_for_sampling = _as_sglang_npu_eagle_sampling_float(target_probs)
    uniform_samples_for_sampling = _as_sglang_npu_eagle_sampling_float(uniform_samples)
    final_uniform_samples_for_sampling = _as_sglang_npu_eagle_sampling_float(
        uniform_samples_for_final_sampling
    )
    device = target_probs_for_sampling.device

    threshold_acc = max(float(threshold_acc), 1e-9)
    threshold_single = float(threshold_single)

    accept_index.fill_(-1)
    accept_token_num.zero_()

    batch_indices = torch.arange(batch_size, dtype=torch.long, device=device)
    inactive_next_idx = torch.full((batch_size,), -1, dtype=torch.long, device=device)
    reset_retrive_idx = torch.full((batch_size,), -1, dtype=torch.long, device=device)
    reset_residual_prob = torch.zeros(
        (batch_size,),
        dtype=target_probs_for_sampling.dtype,
        device=device,
    )

    cur_prob_idx = torch.zeros((batch_size,), dtype=torch.long, device=device)
    accepted_count = torch.zeros((batch_size,), dtype=torch.long, device=device)
    active = torch.ones((batch_size,), dtype=torch.bool, device=device)

    last_accepted_retrive_idx = retrive_index[:, 0].to(torch.long)
    accept_index[:, 0].copy_(last_accepted_retrive_idx.to(dtype=accept_index.dtype))
    coin = uniform_samples_for_sampling[:, 0]
    residual_token_id = reset_retrive_idx.clone()
    residual_token_prob = reset_residual_prob.clone()

    for _ in range(1, num_speculative_tokens):
        next_idx = torch.where(
            active,
            retrive_next_token[batch_indices, cur_prob_idx],
            inactive_next_idx,
        )
        valid = active & (next_idx >= 0)
        safe_next_idx = next_idx.clamp_min(0)
        draft_token_id = candidates[batch_indices, safe_next_idx].to(torch.long)
        target_prob_single = target_probs_for_sampling[
            batch_indices,
            cur_prob_idx,
            draft_token_id,
        ]
        target_prob_single = torch.where(valid, target_prob_single, torch.zeros_like(target_prob_single))
        accepted = valid & (
            (coin <= (target_prob_single / threshold_acc)) | (target_prob_single >= threshold_single)
        )
        rejected = valid & ~accepted

        residual_token_id = torch.where(rejected, draft_token_id, residual_token_id)
        residual_token_prob = torch.where(rejected, target_prob_single, residual_token_prob)

        accepted_retrive_idx = retrive_index[batch_indices, safe_next_idx].to(torch.long)
        old_predicts = predicts.gather(dim=0, index=last_accepted_retrive_idx)
        predict_updates = torch.where(accepted, draft_token_id.to(dtype=predicts.dtype), old_predicts)
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

        cur_prob_idx = torch.where(accepted, safe_next_idx, cur_prob_idx)
        last_accepted_retrive_idx = torch.where(accepted, accepted_retrive_idx, last_accepted_retrive_idx)
        accepted_count = next_accepted_count
        coin = uniform_samples_for_sampling[batch_indices, cur_prob_idx]
        active = accepted
        residual_token_id = torch.where(accepted, reset_retrive_idx, residual_token_id)
        residual_token_prob = torch.where(accepted, reset_residual_prob, residual_token_prob)

    accept_token_num.copy_(accepted_count.to(dtype=accept_token_num.dtype))

    final_target_probs = target_probs_for_sampling[batch_indices, cur_prob_idx]
    need_residual = accepted_count != (num_speculative_tokens - 1)
    residual_mask = need_residual & (residual_token_id >= 0)
    if bool(residual_mask.any().item()):
        final_probs = final_target_probs.clone()
        residual_rows = torch.nonzero(residual_mask, as_tuple=False).view(-1)
        residual_cols = residual_token_id[residual_rows]
        final_probs[residual_rows, residual_cols] = torch.clamp(
            final_probs[residual_rows, residual_cols] - residual_token_prob[residual_rows],
            min=0.0,
        )
    else:
        final_probs = final_target_probs
    final_probs = torch.where(final_probs.sum(dim=-1, keepdim=True) > 0, final_probs, final_target_probs)
    final_token_ids = _sample_from_probs_with_coin(final_probs, final_uniform_samples_for_sampling)
    predicts.scatter_(dim=0, index=last_accepted_retrive_idx, src=final_token_ids.to(dtype=predicts.dtype))


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
    del draft_probs

    batch_size, num_draft_tokens = candidates.shape
    num_speculative_tokens = accept_index.shape[1]
    target_probs_for_sampling = _as_sglang_npu_eagle_sampling_float(target_probs)
    uniform_samples_for_sampling = _as_sglang_npu_eagle_sampling_float(uniform_samples)
    final_uniform_samples_for_sampling = _as_sglang_npu_eagle_sampling_float(
        uniform_samples_for_final_sampling
    )
    device = target_probs_for_sampling.device
    vocab_size = target_probs_for_sampling.shape[-1]

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
    coin = uniform_samples_for_sampling[:, 0]
    residual_draft_probs = torch.zeros(
        (batch_size, vocab_size),
        dtype=target_probs_for_sampling.dtype,
        device=device,
    )

    for _ in range(1, num_speculative_tokens):
        sibling_idx = torch.where(
            active,
            retrive_next_token[batch_indices, cur_prob_idx],
            torch.full_like(cur_prob_idx, -1),
        )
        found_idx = torch.full_like(cur_prob_idx, -1)
        prob_acc = torch.zeros(
            (batch_size,),
            dtype=target_probs_for_sampling.dtype,
            device=device,
        )

        for _ in range(num_draft_tokens):
            valid = active & (found_idx < 0) & (sibling_idx >= 0)
            safe_sibling_idx = sibling_idx.clamp_min(0)
            draft_token_id = candidates[batch_indices, safe_sibling_idx].to(torch.long)
            target_prob_single = target_probs_for_sampling[
                batch_indices,
                cur_prob_idx,
                draft_token_id,
            ]
            target_prob_single = torch.where(valid, target_prob_single, torch.zeros_like(target_prob_single))
            next_prob_acc = prob_acc + target_prob_single

            old_residual = residual_draft_probs.gather(
                dim=1,
                index=draft_token_id.view(-1, 1),
            ).squeeze(1)
            residual_update = torch.where(valid, target_prob_single, old_residual)
            residual_draft_probs.scatter_(
                dim=1,
                index=draft_token_id.view(-1, 1),
                src=residual_update.view(-1, 1),
            )

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
        coin = uniform_samples_for_sampling[batch_indices, cur_prob_idx]
        active = accepted
        residual_draft_probs.mul_((~accepted).to(dtype=residual_draft_probs.dtype).view(-1, 1))

    accept_token_num.copy_(accepted_count.to(dtype=accept_token_num.dtype))

    final_target_probs = target_probs_for_sampling[batch_indices, cur_prob_idx]
    residual_probs = torch.clamp(final_target_probs - residual_draft_probs, min=0.0)
    need_residual = accepted_count != (num_speculative_tokens - 1)
    final_probs = torch.where(need_residual.view(-1, 1), residual_probs, final_target_probs)
    final_probs = torch.where(final_probs.sum(dim=-1, keepdim=True) > 0, final_probs, final_target_probs)
    final_token_ids = _sample_from_probs_with_coin(final_probs, final_uniform_samples_for_sampling)
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
    # Linear EAGLE trees (for example spec_topk=1) can skip sibling scanning.
    if not bool(torch.any(retrive_next_sibling >= 0).item()):
        if _try_tree_speculative_sampling_target_only_linear_triton(
            predicts=predicts,
            accept_index=accept_index,
            accept_token_num=accept_token_num,
            candidates=candidates,
            retrive_index=retrive_index,
            retrive_next_token=retrive_next_token,
            uniform_samples=uniform_samples,
            uniform_samples_for_final_sampling=uniform_samples_for_final_sampling,
            target_probs=target_probs,
            threshold_single=threshold_single,
            threshold_acc=threshold_acc,
        ):
            return
        _tree_speculative_sampling_target_only_linear_torch(
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


def patch_sglang_npu_eagle_target_sampling() -> None:
    """Patch SGLang NPU EAGLE v1 verification to use target-only sampling."""
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
            patched_targets.append("sglang.srt.speculative.eagle_info(target_only)")
        except Exception as exc:  # noqa: BLE001
            logger.debug("Skip SGLang EAGLE v1 target sampling patch: %s", exc)
    else:
        logger.info(
            "Skip SGLang EAGLE v1 target sampling patch. Set %s=target_only to enable it.",
            _EAGLE_V1_VERIFY_MODE_ENV,
        )

    if patched_targets:
        _SGLANG_NPU_EAGLE_SAMPLING_PATCHED = True
        logger.warning("Patched SGLang NPU EAGLE sampling for %s", ", ".join(patched_targets))


def _append_sglang_decode_hidden_states(req, logits_output, result, req_index: int, hidden_state_offset: int) -> int:
    hidden_states = getattr(logits_output, "hidden_states", None)
    if hidden_states is None:
        return hidden_state_offset

    accept_lengths = getattr(result, "accept_length_per_req_cpu", None)
    if accept_lengths is not None and req_index < len(accept_lengths) and torch.is_tensor(hidden_states):
        rows = max(int(accept_lengths[req_index]) + 1, 1)

        if hidden_states.dim() == 3 and req_index < int(hidden_states.shape[0]):
            if int(hidden_states.shape[1]) >= rows:
                if getattr(req, "return_hidden_states", False):
                    req.hidden_states.append(hidden_states[req_index, :rows].detach().to("cpu", copy=True))
                return hidden_state_offset + rows
            if getattr(req, "return_hidden_states", False):
                raise RuntimeError(
                    "SGLang EAGLE verify hidden states are incomplete for accepted tokens: "
                    f"shape={tuple(hidden_states.shape)}, req_index={req_index}, required_rows={rows}."
                )

        expected_rows = sum(max(int(accept_len) + 1, 1) for accept_len in accept_lengths)
        end = hidden_state_offset + rows
        has_expected_rows = int(hidden_states.shape[0]) >= expected_rows and end <= int(hidden_states.shape[0])
        if hidden_states.dim() >= 2 and has_expected_rows:
            if getattr(req, "return_hidden_states", False):
                req.hidden_states.append(hidden_states[hidden_state_offset:end].detach().to("cpu", copy=True))
            return end

        if getattr(req, "return_hidden_states", False):
            raise RuntimeError(
                "SGLang EAGLE verify hidden states are incomplete for accepted tokens: "
                f"shape={tuple(hidden_states.shape)}, req_index={req_index}, "
                f"offset={hidden_state_offset}, required_rows={rows}, expected_total_rows={expected_rows}."
            )

    if not getattr(req, "return_hidden_states", False):
        return hidden_state_offset
    if torch.is_tensor(hidden_states):
        req.hidden_states.append(hidden_states[req_index].detach().to("cpu", copy=True))
    else:
        req.hidden_states.append(hidden_states[req_index])
    return hidden_state_offset


def _sglang_batch_requests_hidden_states(batch) -> bool:
    return any(bool(getattr(req, "return_hidden_states", False)) for req in getattr(batch, "reqs", []) or [])


def _ensure_sglang_eagle_verify_full_hidden_mode(batch, spec_info) -> None:
    if not _sglang_batch_requests_hidden_states(batch):
        return
    try:
        from sglang.srt.model_executor.forward_batch_info import CaptureHiddenMode
    except Exception as exc:  # noqa: BLE001
        logger.debug("Cannot import SGLang CaptureHiddenMode for EAGLE hidden-state patch: %s", exc)
        return
    spec_info.capture_hidden_mode = CaptureHiddenMode.FULL


def _sglang_hidden_state_rows(hidden_states) -> int:
    if not torch.is_tensor(hidden_states):
        try:
            return len(hidden_states)
        except TypeError:
            return 0
    if hidden_states.dim() == 0:
        return 1
    if hidden_states.dim() >= 3:
        return int(hidden_states.shape[0]) * int(hidden_states.shape[1])
    return int(hidden_states.shape[0])


def _sglang_eagle_verify_expected_hidden_rows(batch, spec_info) -> int:
    batch_size = len(getattr(batch, "reqs", []) or [])
    draft_token_num = int(getattr(spec_info, "draft_token_num", 0) or 0)
    return batch_size * draft_token_num


def _sglang_eagle_verify_hidden_states_incomplete(batch, spec_info, logits_output) -> bool:
    if not _sglang_batch_requests_hidden_states(batch):
        return False
    expected_rows = _sglang_eagle_verify_expected_hidden_rows(batch, spec_info)
    if expected_rows <= 0:
        return False
    hidden_states = getattr(logits_output, "hidden_states", None)
    return hidden_states is None or _sglang_hidden_state_rows(hidden_states) < expected_rows


def _rerun_sglang_eagle_verify_without_graph(worker, model_worker_batch):
    target_worker = getattr(worker, "target_worker", None)
    model_runner = getattr(target_worker, "model_runner", None)
    graph_runner = getattr(model_runner, "graph_runner", None)
    try:
        if model_runner is not None:
            model_runner.graph_runner = None
        return target_worker.forward_batch_generation(model_worker_batch, is_verify=True)
    finally:
        if model_runner is not None:
            model_runner.graph_runner = graph_runner


def _validate_sglang_eagle_verify_hidden_states(batch, spec_info, logits_output) -> None:
    if not _sglang_eagle_verify_hidden_states_incomplete(batch, spec_info, logits_output):
        return
    hidden_states = getattr(logits_output, "hidden_states", None)
    shape = tuple(hidden_states.shape) if torch.is_tensor(hidden_states) else None
    expected_rows = _sglang_eagle_verify_expected_hidden_rows(batch, spec_info)
    actual_rows = _sglang_hidden_state_rows(hidden_states)
    raise RuntimeError(
        "SGLang EAGLE verify did not return full hidden states for drafter training: "
        f"actual_rows={actual_rows}, expected_rows={expected_rows}, shape={shape}. "
        "This would train on partial/incorrect hidden alignment."
    )


def _make_sglang_eagle_verify_full_hidden_patch(original_method):
    try:
        source = inspect.getsource(original_method)
    except (OSError, TypeError):
        return None

    source = textwrap.dedent(source)
    patched_source = source

    old_prepare = "        spec_info.prepare_for_verify(batch, self.page_size)\n"
    new_prepare = (
        "        spec_info.prepare_for_verify(batch, self.page_size)\n"
        "        _ensure_sglang_eagle_verify_full_hidden_mode(batch, spec_info)\n"
    )
    if old_prepare in patched_source:
        patched_source = patched_source.replace(old_prepare, new_prepare, 1)
    else:
        return None

    old_forward = """        # Forward
        batch_result = self.target_worker.forward_batch_generation(
            model_worker_batch, is_verify=True
        )
        logits_output, can_run_cuda_graph = (
            batch_result.logits_output,
            batch_result.can_run_cuda_graph,
        )
"""
    new_forward = """        # Forward
        batch_result = self.target_worker.forward_batch_generation(
            model_worker_batch, is_verify=True
        )
        logits_output, can_run_cuda_graph = (
            batch_result.logits_output,
            batch_result.can_run_cuda_graph,
        )
        if _sglang_eagle_verify_hidden_states_incomplete(batch, spec_info, logits_output):
            logger.warning(
                "SGLang EAGLE verify returned incomplete hidden states; rerunning without graph for full hidden output."
            )
            batch_result = _rerun_sglang_eagle_verify_without_graph(self, model_worker_batch)
            logits_output, can_run_cuda_graph = (
                batch_result.logits_output,
                batch_result.can_run_cuda_graph,
            )
        _validate_sglang_eagle_verify_hidden_states(batch, spec_info, logits_output)
"""
    if old_forward in patched_source:
        patched_source = patched_source.replace(old_forward, new_forward, 1)
    else:
        return None

    globals_dict = original_method.__globals__
    globals_dict["logger"] = logger
    globals_dict["_ensure_sglang_eagle_verify_full_hidden_mode"] = _ensure_sglang_eagle_verify_full_hidden_mode
    globals_dict["_sglang_eagle_verify_hidden_states_incomplete"] = _sglang_eagle_verify_hidden_states_incomplete
    globals_dict["_rerun_sglang_eagle_verify_without_graph"] = _rerun_sglang_eagle_verify_without_graph
    globals_dict["_validate_sglang_eagle_verify_hidden_states"] = _validate_sglang_eagle_verify_hidden_states
    namespace = {}
    exec(  # noqa: S102
        "from __future__ import annotations\n" + patched_source,
        globals_dict,
        namespace,
    )
    patched_method = namespace[original_method.__name__]
    patched_method = wraps(original_method)(patched_method)
    patched_method._verl_patched_eagle_verify_full_hidden_states = True
    return patched_method


def patch_sglang_eagle_verify_hidden_states_full() -> None:
    """Force SGLang EAGLE v1 verify to return full per-token hidden states."""
    global _SGLANG_EAGLE_VERIFY_HIDDEN_STATES_PATCHED
    if _SGLANG_EAGLE_VERIFY_HIDDEN_STATES_PATCHED:
        return

    targets = (
        ("sglang.srt.speculative.eagle_worker", "EAGLEWorker"),
        ("sglang.srt.speculative.multi_layer_eagle_worker", "MultiLayerEagleWorker"),
    )
    patched_targets = []
    for module_name, class_name in targets:
        try:
            module = importlib.import_module(module_name)
            worker_cls = getattr(module, class_name)
            original_method = getattr(worker_cls, "verify", None)
        except Exception as exc:  # noqa: BLE001
            logger.debug("Skip SGLang EAGLE full hidden-state patch for %s.%s: %s", module_name, class_name, exc)
            continue
        if original_method is None or getattr(original_method, "_verl_patched_eagle_verify_full_hidden_states", False):
            continue
        patched_method = _make_sglang_eagle_verify_full_hidden_patch(original_method)
        if patched_method is None:
            logger.debug("Skip SGLang EAGLE full hidden-state patch for %s.%s", module_name, class_name)
            continue
        setattr(worker_cls, "verify", patched_method)
        patched_targets.append(f"{module_name}.{class_name}.verify")

    if patched_targets:
        _SGLANG_EAGLE_VERIFY_HIDDEN_STATES_PATCHED = True
        logger.info("Patched SGLang EAGLE verify full hidden states for %s", ", ".join(patched_targets))


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
    if original_method.__name__ == "process_batch_result_decode":
        loop_line = "        for i, (req, next_token_id) in enumerate(zip(batch.reqs, next_token_ids)):\n"
        new_hidden_block = """            hidden_state_offset = _append_sglang_decode_hidden_states(
                req,
                logits_output,
                result,
                i,
                hidden_state_offset,
            )
"""
        hidden_block_pattern = re.compile(
            r"(?ms)^[ \t]+if req\.return_hidden_states and logits_output\.hidden_states is not None:\r?\n"
            r"[ \t]+req\.hidden_states\.append\(\r?\n"
            r".*?"
            r"^[ \t]+\)\r?\n"
        )
        patched_source, hidden_block_count = hidden_block_pattern.subn(new_hidden_block, patched_source, count=1)
        if loop_line in patched_source and hidden_block_count > 0:
            patched_source = patched_source.replace(loop_line, "        hidden_state_offset = 0\n\n" + loop_line)
        elif hidden_block_count > 0:
            return None
        else:
            logger.warning(
                "Skip SGLang decode hidden-state full-output patch for %s: hidden append block not found.",
                original_method.__name__,
            )
            return None
    if patched_source == source:
        return None

    globals_dict = original_method.__globals__
    globals_dict["_append_sglang_decode_hidden_states"] = _append_sglang_decode_hidden_states
    namespace = {}
    exec(  # noqa: S102
        "from __future__ import annotations\n" + patched_source,
        globals_dict,
        namespace,
    )
    patched_method = namespace[original_method.__name__]
    patched_method = wraps(original_method)(patched_method)
    patched_method._verl_patched_hidden_states_tensor_output = True
    return patched_method


def patch_sglang_hidden_states_tensor_output() -> None:
    """Return SGLang hidden-state chunks as CPU tensors instead of Python lists."""
    global _SGLANG_HIDDEN_STATES_TENSOR_OUTPUT_PATCHED
    patch_sglang_eagle_verify_hidden_states_full()
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


def _apply_selected_sglang_patches() -> bool:
    patchers = (
        ("eagle_update_weights", patch_sglang_eagle_update_weights_from_tensor),
        ("npu_eagle_target_sampling", patch_sglang_npu_eagle_target_sampling),
        ("hidden_states_tensor_output", patch_sglang_hidden_states_tensor_output),
    )

    applied_any = False
    skipped = []
    for patch_name, patcher in patchers:
        if _sglang_patch_enabled(patch_name):
            patcher()
            applied_any = True
        else:
            skipped.append(patch_name)

    if skipped:
        logger.info("Skip verl SGLang patches not selected by %s: %s", _SGLANG_PATCHES_ENV, ", ".join(skipped))
    return applied_any


def _apply_sglang_child_process_patches() -> None:
    if _sglang_verl_patches_disabled():
        logger.warning("Skip all verl SGLang patches because %s=1.", _DISABLE_SGLANG_PATCH_ENV)
        return

    _apply_selected_sglang_patches()


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
    if _sglang_verl_patches_disabled():
        logger.warning("Skip installing verl SGLang patches because %s=1.", _DISABLE_SGLANG_PATCH_ENV)
        return

    if _sglang_patch_enabled("eagle_update_weights"):
        configure_sglang_eagle_weight_update_patch(target_weight_loader, draft_weight_loader)
    applied_any = _apply_selected_sglang_patches()
    if applied_any:
        patch_sglang_scheduler_process_entrypoints()

    if set_envs_and_config is not None:
        sglang.srt.entrypoints.engine._set_envs_and_config = set_envs_and_config
