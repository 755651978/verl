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
import logging
import os
from functools import wraps
from typing import Callable

import sglang.srt.entrypoints.engine
from sglang.srt.utils import MultiprocessingSerializer

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))

_TARGET_WEIGHT_LOADER_ENV = "VERL_SGLANG_TARGET_WEIGHT_LOADER"
_DRAFT_WEIGHT_LOADER_ENV = "VERL_SGLANG_DRAFT_WEIGHT_LOADER"

_target_weight_loader: str | None = os.environ.get(_TARGET_WEIGHT_LOADER_ENV)
_draft_weight_loader: str | None = os.environ.get(_DRAFT_WEIGHT_LOADER_ENV)
_ORIGINAL_SGLANG_RUN_SCHEDULER_PROCESS = sglang.srt.entrypoints.engine.run_scheduler_process
_SGLANG_EAGLE_UPDATE_PATCHED = False


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


def _run_scheduler_process_with_verl_patches(*args, **kwargs):
    patch_sglang_eagle_update_weights_from_tensor()
    return _ORIGINAL_SGLANG_RUN_SCHEDULER_PROCESS(*args, **kwargs)


_run_scheduler_process_with_verl_patches._verl_patched_eagle_update_weights = True


def install_sglang_verl_patches(
    set_envs_and_config: Callable | None = None,
    target_weight_loader: str | None = None,
    draft_weight_loader: str | None = None,
) -> None:
    configure_sglang_eagle_weight_update_patch(target_weight_loader, draft_weight_loader)
    patch_sglang_eagle_update_weights_from_tensor()

    if set_envs_and_config is not None:
        sglang.srt.entrypoints.engine._set_envs_and_config = set_envs_and_config

    if not getattr(
        sglang.srt.entrypoints.engine.run_scheduler_process,
        "_verl_patched_eagle_update_weights",
        False,
    ):
        sglang.srt.entrypoints.engine.run_scheduler_process = _run_scheduler_process_with_verl_patches
