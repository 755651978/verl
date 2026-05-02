# Copyright 2023-2024 SGLang Team
# Copyright 2025 Bytedance Ltd. and/or its affiliates
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
import asyncio
import dataclasses
import hashlib
import json
import logging
import os
import math
from typing import Any, Optional

import ray
import sglang
import sglang.srt.entrypoints.engine
import torch
from packaging import version
from ray.actor import ActorHandle
from sglang.srt.entrypoints.http_server import (
    ServerArgs,
    _GlobalState,
    app,
    set_global_state,
)
from sglang.srt.managers.io_struct import (
    ContinueGenerationReqInput,
    GenerateReqInput,
    PauseGenerationReqInput,
    ReleaseMemoryOccupationReqInput,
    ResumeMemoryOccupationReqInput,
)
from sglang.srt.managers.tokenizer_manager import ServerStatus
from sglang.srt.utils import MultiprocessingSerializer

from verl.utils.config import omega_conf_to_dataclass
from verl.utils.device import get_visible_devices_keyword
from verl.utils.net_utils import get_free_port, is_valid_ipv6_address
from verl.utils.profiler import DistProfiler, build_sglang_profiler_args
from verl.workers.drafter.sglang_patch import install_sglang_verl_patches
from verl.workers.config import HFModelConfig, RolloutConfig
from verl.workers.rollout.replica import RolloutMode, RolloutReplica, TokenOutput
from verl.workers.rollout.sglang_rollout.sglang_rollout import (
    VERL_SGLANG_DRAFT_WEIGHT_LOADER,
    VERL_SGLANG_TARGET_WEIGHT_LOADER,
    _set_envs_and_config,
)
from verl.workers.rollout.sglang_rollout.utils import SGLANG_LORA_NAME
from verl.workers.rollout.utils import get_max_position_embeddings, run_uvicorn
logger = logging.getLogger(__file__)
logger.setLevel(logging.INFO)

visible_devices_keyword = get_visible_devices_keyword()
_DEBUG_OUTPUT_TOKENS_ENV = "VERL_SGLANG_DEBUG_OUTPUT_TOKENS"
_DEBUG_OUTPUT_TOKENS_LIMIT_ENV = "VERL_SGLANG_DEBUG_OUTPUT_TOKENS_LIMIT"
_EAGLE_STATE_DEBUG_ENV = "VERL_SGLANG_EAGLE_STATE_DEBUG"


def _env_flag_enabled(name: str, default: bool = False) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() not in {"", "0", "false", "off", "no"}


def _log_sglang_eagle_server_state(args: dict, server_args: ServerArgs, config: RolloutConfig) -> None:
    if not _env_flag_enabled(_EAGLE_STATE_DEBUG_ENV):
        return

    keys = (
        "model_path",
        "load_format",
        "custom_weight_loader",
        "speculative_algorithm",
        "speculative_draft_model_path",
        "speculative_num_steps",
        "speculative_eagle_topk",
        "speculative_num_draft_tokens",
        "enable_return_hidden_states",
        "enable_weights_cpu_backup",
        "enable_draft_weights_cpu_backup",
        "disable_cuda_graph",
        "cuda_graph_max_bs",
        "skip_server_warmup",
        "attention_backend",
        "disable_overlap_schedule",
    )
    selected = {key: getattr(server_args, key, args.get(key, None)) for key in keys}
    logger.warning(
        "[SGLangEagleStateDebug] server sglang_version=%s drafter_enable=%s "
        "drafter_training=%s collect_hidden=%s target_loader=%s draft_loader=%s args=%s",
        sglang.__version__,
        bool(config.drafter.enable),
        bool(config.drafter.enable_drafter_training),
        bool(config.drafter.training.collect_hidden_states_from_sgl),
        VERL_SGLANG_TARGET_WEIGHT_LOADER,
        VERL_SGLANG_DRAFT_WEIGHT_LOADER,
        selected,
    )


def _top_logprobs_to_tensor(top_logprobs: list, topk: int) -> Optional[torch.Tensor]:
    if topk <= 0:
        return None

    rows = []
    for step_top_logprobs in top_logprobs:
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
                row.append([float(logprob), float(int(token_id))])
            except (TypeError, ValueError):
                continue

        if not row:
            row = [[-math.inf, -1.0] for _ in range(topk)]
        while len(row) < topk:
            row.append([-math.inf, -1.0])
        rows.append(row)

    if not rows:
        return None
    return torch.tensor(rows, dtype=torch.float32)


def _hidden_state_chunk_to_tensor(hidden_state_chunk: Any) -> Optional[torch.Tensor]:
    if hidden_state_chunk is None:
        return None

    if torch.is_tensor(hidden_state_chunk):
        h_states = hidden_state_chunk.detach().to(device="cpu", dtype=torch.bfloat16)
    elif isinstance(hidden_state_chunk, (bytes, bytearray, memoryview, str)):
        serialized_chunk = (
            hidden_state_chunk.tobytes()
            if isinstance(hidden_state_chunk, memoryview)
            else bytes(hidden_state_chunk)
            if isinstance(hidden_state_chunk, bytearray)
            else hidden_state_chunk
        )
        try:
            deserialized = MultiprocessingSerializer.deserialize(serialized_chunk)
        except Exception:  # noqa: BLE001
            logger.warning("Failed to deserialize hidden-state chunk from bytes; skipping this chunk")
            return None
        return _hidden_state_chunk_to_tensor(deserialized)
    else:
        h_states = torch.tensor(hidden_state_chunk, dtype=torch.bfloat16)

    if h_states.numel() == 0:
        return None
    if h_states.dim() == 1:
        h_states = h_states.unsqueeze(0)
    elif h_states.dim() == 3:
        h_states = h_states.squeeze(0)
    return h_states.contiguous()


def _iter_hidden_state_chunks(hidden_states_data: Any):
    if hidden_states_data is None:
        return []
    if torch.is_tensor(hidden_states_data) or isinstance(hidden_states_data, (bytes, bytearray, memoryview, str)):
        return [hidden_states_data]
    return hidden_states_data


class SGLangHttpServer:
    """SGLang http server in single node, this is equivalent to launch server with command line:
    ```
    python -m sglang.launch_server --node-rank 0 --nnode 1 ...
    ```

    Args:
        config (DictConfig): full config.
        rollout_mode (RolloutMode): rollout mode.
        replica_rank (int): replica rank, a replica may contain multiple nodes.
        node_rank (int): node rank.
        nnodes (int): number of nodes.
        cuda_visible_devices (str): cuda visible devices.
    """

    def __init__(
        self,
        config: RolloutConfig,
        model_config: HFModelConfig,
        rollout_mode: RolloutMode,
        workers: list[ActorHandle],
        replica_rank: int,
        node_rank: int,
        nnodes: int,
        cuda_visible_devices: str,
        base_gpu_id: int,
    ):
        print(f"SGLang http server: {rollout_mode=}, {replica_rank=}, {node_rank=}, {nnodes=}, {cuda_visible_devices=}")
        os.environ[visible_devices_keyword] = cuda_visible_devices

        self.config: RolloutConfig = omega_conf_to_dataclass(config)
        self.model_config: HFModelConfig = omega_conf_to_dataclass(model_config, dataclass_type=HFModelConfig)
        max_position_embeddings = get_max_position_embeddings(self.model_config.hf_config)
        if self.config.max_model_len is None:
            self.config.max_model_len = max_position_embeddings
        else:
            if self.config.max_model_len > max_position_embeddings:
                raise ValueError(
                    f"max_model_len ({self.config.max_model_len}) should be less than or equal to "
                    f"max_position_embeddings ({max_position_embeddings})"
                )
        self.rollout_mode = rollout_mode
        self.workers = workers

        self.replica_rank = replica_rank
        self.node_rank = node_rank
        self.nnodes = nnodes
        self.base_gpu_id = base_gpu_id
        # model weights version, set by ServerAdapter when update weights.
        self.global_steps = None
        self._drafter_collection_step = None
        self._drafter_collection_samples = 0
        self._drafter_collection_tokens = 0

        if self.rollout_mode != RolloutMode.HYBRID and self.config.load_format == "dummy":
            logger.warning(f"rollout mode is {self.rollout_mode}, load_format is dummy, set to auto")
            self.config.load_format = "auto"

        # used for http server
        self._server_address = ray.util.get_node_ip_address().strip("[]")
        self._server_port = None

        # used for controlling sglang server profiler
        profiler_config = self.config.profiler
        tool_config = None
        if profiler_config is not None:
            if profiler_config.tool in ["torch", "npu"]:
                tool_config = omega_conf_to_dataclass((profiler_config.tool_config or {}).get(profiler_config.tool))
            else:
                logger.warning(f"agent loop only support torch and npu profiler, got {profiler_config.tool}")
                profiler_config = None
        self.profiler_controller = DistProfiler(self.replica_rank, config=profiler_config, tool_config=tool_config)

        # For multi-node, we need dist_init_addr so nodes can coordinate NCCL init.
        # For single-node, let SGLang handle port selection internally via nccl_port,
        # which also avoids port conflicts.
        self._master_address = None
        self._master_port = None
        self._master_sock = None
        if self.nnodes > 1 and self.node_rank == 0:
            self._master_address = self._server_address
            self._master_port, self._master_sock = get_free_port(self._server_address, with_alive_sock=True)
            logger.info(
                f"SGLangHttpServer, replica_rank: {self.replica_rank}, "
                f"master address: {self._master_address}, port: {self._master_port}"
            )
    def get_master_address(self):
        """Get master address and port for init NCCL process group."""
        return self._master_address, self._master_port

    def get_server_address(self):
        """Get http server address and port."""
        assert self._server_port is not None, "http server is not launched, port is None"
        return self._server_address, self._server_port

    async def launch_server(self, master_address: str = None, master_port: int = None):
        if self.nnodes > 1:
            if self.node_rank != 0:
                assert master_address and master_port, "non-master node should provide master address and port"
                self._master_address = master_address
                self._master_port = master_port

        engine_kwargs = self.config.get("engine_kwargs", {}).get("sglang", {}) or {}
        attention_backend = engine_kwargs.pop("attention_backend", None)
        quantization = self.config.get("quantization", None)
        if quantization is not None:
            if quantization == "fp8":
                assert version.parse(sglang.__version__) >= version.parse("0.5.5"), (
                    "sglang>=0.5.5 is required for FP8 quantization"
                )
                FP8_BLOCK_QUANT_KWARGS = {
                    "activation_scheme": "dynamic",
                    "fmt": "e4m3",
                    "quant_method": "fp8",
                    "weight_block_size": [128, 128],
                }
                fp8_block_quant_kwargs = dict(FP8_BLOCK_QUANT_KWARGS)
            else:
                raise ValueError(f"Currently only support fp8 quantization, got: {quantization}")
        infer_tp = self.config.tensor_model_parallel_size * self.config.data_parallel_size
        args = {
            "model_path": self.model_config.local_path,
            "dtype": self.config.dtype,
            "mem_fraction_static": self.config.gpu_memory_utilization,
            "disable_cuda_graph": self.config.enforce_eager,
            "enable_memory_saver": True,
            "base_gpu_id": self.base_gpu_id,
            "gpu_id_step": 1,
            "tp_size": infer_tp,
            "dp_size": self.config.data_parallel_size,
            "ep_size": self.config.expert_parallel_size,
            "node_rank": self.node_rank,
            "load_format": self.config.load_format,
            "nnodes": self.nnodes,
            "trust_remote_code": self.model_config.trust_remote_code,
            "max_running_requests": self.config.get("max_num_seqs", None),
            "log_level": "error",
            "mm_attention_backend": "fa3",
            "attention_backend": attention_backend if attention_backend is not None else "fa3",
            "skip_tokenizer_init": self.config.skip_tokenizer_init,
            "skip_server_warmup": True,
            "quantization": quantization,
            "json_model_override_args": json.dumps({"quantization_config": fp8_block_quant_kwargs})
            if quantization == "fp8"
            else json.dumps({}),
            **engine_kwargs,
        }

        # update lora-related args
        if self.model_config.lora_rank > 0:
            args.update(
                {
                    "enable_lora": True,
                    "max_lora_rank": self.model_config.lora_rank,
                    "lora_target_modules": self.model_config.target_modules,
                }
            )
        # Only set dist_init_addr for multi-node; for single-node, let SGLang
        # handle port selection internally via nccl_port to avoid conflicts.
        if self.nnodes > 1:
            dist_init_addr = (
                f"[{self._master_address}]:{self._master_port}"
                if is_valid_ipv6_address(self._master_address)
                else f"{self._master_address}:{self._master_port}"
            )
            args["dist_init_addr"] = dist_init_addr

        if self.config.prometheus.enable:
            if self.config.prometheus.served_model_name:
                # Extract model name from path if it's a full path
                served_model_name = self.config.prometheus.served_model_name
                if "/" in served_model_name:
                    # If it's a full path, extract the last part as model name
                    served_model_name = served_model_name.split("/")[-1]
                args["served_model_name"] = served_model_name

            # start sglang metrics
            args["enable_metrics"] = True

        # enable_weights_cpu_backup is supported in sglang>=0.5.3
        if "enable_weights_cpu_backup" in [f.name for f in dataclasses.fields(ServerArgs)]:
            enable_weights_cpu_backup = (
                True if self.rollout_mode == RolloutMode.COLOCATED or self.model_config.lora_rank > 0 else False
            )
            args["enable_weights_cpu_backup"] = enable_weights_cpu_backup

        if self.config.enable_rollout_routing_replay:
            args.update({"enable_return_routed_experts": True})

        if self.config.drafter.enable and "custom_weight_loader" in [f.name for f in dataclasses.fields(ServerArgs)]:
            custom_weight_loaders = list(args.get("custom_weight_loader") or [])
            for loader in (VERL_SGLANG_TARGET_WEIGHT_LOADER, VERL_SGLANG_DRAFT_WEIGHT_LOADER):
                if loader not in custom_weight_loaders:
                    custom_weight_loaders.append(loader)
            args["custom_weight_loader"] = custom_weight_loaders

        # mtp
        if self.config.mtp.enable and self.config.mtp.enable_rollout:
            # Enable weights CPU backup for sglang >= 0.5.6
            if sglang.__version__ < "0.5.6":
                raise ValueError(f"sglang version {sglang.__version__} is not supported for MTP rollout")

            args["speculative_algorithm"] = self.config.mtp.speculative_algorithm
            args["speculative_num_steps"] = self.config.mtp.speculative_num_steps
            args["speculative_eagle_topk"] = self.config.mtp.speculative_eagle_topk
            args["speculative_num_draft_tokens"] = self.config.mtp.speculative_num_draft_tokens

            args["enable_weights_cpu_backup"] = True
            args["enable_draft_weights_cpu_backup"] = True

        # drafter
        if self.config.drafter.enable:
            args["speculative_algorithm"] = self.config.drafter.speculative_algorithm
            args["cuda_graph_max_bs"] = 32
            args["speculative_draft_model_path"] = self.config.drafter.model_path
            args["speculative_num_steps"] = self.config.drafter.rollout.spec_steps
            args["speculative_eagle_topk"] = self.config.drafter.rollout.spec_topk
            args["speculative_num_draft_tokens"] = self.config.drafter.rollout.spec_verify_tokens
            args["enable_return_hidden_states"] = bool(self.config.drafter.enable_drafter_training
                                                       and self.config.drafter.training.collect_hidden_states_from_sgl)

            args["enable_weights_cpu_backup"] = True
            args["enable_draft_weights_cpu_backup"] = True

        # NOTE: We can't directly call SGLang's launch_server since it's not an async function.
        # https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/entrypoints/http_server.py
        if self.config.drafter.enable:
            install_sglang_verl_patches(
                set_envs_and_config=_set_envs_and_config,
                target_weight_loader=VERL_SGLANG_TARGET_WEIGHT_LOADER,
                draft_weight_loader=VERL_SGLANG_DRAFT_WEIGHT_LOADER,
            )
        sglang.srt.entrypoints.engine._set_envs_and_config = _set_envs_and_config
        os.environ["SGLANG_BLOCK_NONZERO_RANK_CHILDREN"] = "0"
        server_args = ServerArgs(**args)
        _log_sglang_eagle_server_state(args, server_args, self.config)
        # For SGLang main branch or version >= 0.5.10
        # The latest main branch of SGLang has wrapped the _launch_subprocesses function inside the Engine class
        if version.parse(sglang.__version__) >= version.parse("0.5.10"):
            from sglang.srt.entrypoints.http_server import Engine

            self.tokenizer_manager, self.template_manager, self.scheduler_info, *_ = Engine._launch_subprocesses(
                server_args=server_args,
                init_tokenizer_manager_func=sglang.srt.entrypoints.engine.init_tokenizer_manager,
                run_scheduler_process_func=sglang.srt.entrypoints.engine.run_scheduler_process,
                run_detokenizer_process_func=sglang.srt.entrypoints.engine.run_detokenizer_process,
            )
        elif version.parse(sglang.__version__) >= version.parse("0.5.7"):
            from sglang.srt.entrypoints.http_server import _launch_subprocesses

            self.tokenizer_manager, self.template_manager, self.scheduler_info, *_ = _launch_subprocesses(
                server_args=server_args,
                init_tokenizer_manager_func=sglang.srt.entrypoints.engine.init_tokenizer_manager,
                run_scheduler_process_func=sglang.srt.entrypoints.engine.run_scheduler_process,
                run_detokenizer_process_func=sglang.srt.entrypoints.engine.run_detokenizer_process,
            )
        else:
            from sglang.srt.entrypoints.http_server import _launch_subprocesses

            self.tokenizer_manager, self.template_manager, self.scheduler_info, *_ = _launch_subprocesses(
                server_args=server_args
            )

        # In multi-node cases, non-zero rank nodes should not launch http server.
        if self.node_rank > 0:
            return

        set_global_state(
            _GlobalState(
                tokenizer_manager=self.tokenizer_manager,
                template_manager=self.template_manager,
                scheduler_info=self.scheduler_info,
            )
        )
        app.is_single_tokenizer_mode = True

        # Set warmup_thread_{kw}args to avoid AttributeError in lifespan function
        app.server_args = server_args
        app.warmup_thread_kwargs = {"server_args": server_args}
        app.warmup_thread_args = (server_args, None, None)

        # Manually add Prometheus middleware before starting server
        # This ensures /metrics endpoint is available immediately
        if server_args.enable_metrics:
            from sglang.srt.utils.common import add_prometheus_middleware

            add_prometheus_middleware(app)

        self._server_port, self._server_task = await run_uvicorn(app, server_args, self._server_address)
        self.tokenizer_manager.server_status = ServerStatus.Up

    async def wake_up(self):
        if self.node_rank != 0:
            return

        if self.rollout_mode == RolloutMode.HYBRID:
            # In hybrid mode, rollout is wake up in `update_weights`
            raise ValueError(f"wake_up not support rollout_mode {self.rollout_mode}")
        elif self.rollout_mode == RolloutMode.COLOCATED:
            # Directly call engine to wake up without sync weights.
            obj = ResumeMemoryOccupationReqInput(tags=["kv_cache", "weights"])
            await self.tokenizer_manager.resume_memory_occupation(obj, None)
            await self.tokenizer_manager.flush_cache()
        elif self.rollout_mode == RolloutMode.STANDALONE:
            # In standalone mode, resume kv_cache if free_cache_engine is enabled
            obj = ResumeMemoryOccupationReqInput(tags=["kv_cache"])
            await self.tokenizer_manager.resume_memory_occupation(obj, None)
            await self.tokenizer_manager.flush_cache()

    @property
    def lora_as_adapter(self) -> bool:
        return (
            self.model_config.lora_rank > 0 or self.model_config.lora.get("rank", 0) > 0
        ) and not self.model_config.lora.get("merge", False)

    async def sleep(self):
        if self.node_rank != 0 or not self.config.free_cache_engine:
            return

        # When using LoRA as adapter (merge=False), only release kv_cache —
        # keep base weights in GPU so we only need to sync adapter deltas.
        # Mirrors the vLLM sleep() pattern in vllm_async_server.py.
        if self.lora_as_adapter:
            tags = ["kv_cache"]
        else:
            tags = ["kv_cache", "weights"]

        if self.rollout_mode == RolloutMode.HYBRID:
            obj = ReleaseMemoryOccupationReqInput(tags=tags)
            await self.tokenizer_manager.release_memory_occupation(obj, None)
        elif self.rollout_mode == RolloutMode.COLOCATED:
            obj = ReleaseMemoryOccupationReqInput(tags=tags)
            await self.tokenizer_manager.release_memory_occupation(obj, None)
        elif self.rollout_mode == RolloutMode.STANDALONE:
            # In standalone mode, resume kv_cache if free_cache_engine is enabled
            obj = ReleaseMemoryOccupationReqInput(tags=["kv_cache"])
            await self.tokenizer_manager.release_memory_occupation(obj, None)

    async def clear_kv_cache(self):
        if self.node_rank == 0:
            await self.tokenizer_manager.flush_cache()

    def _reset_drafter_collection_budget_if_needed(self, collection_global_steps: Optional[int]):
        if self._drafter_collection_step == collection_global_steps:
            return
        self._drafter_collection_step = collection_global_steps
        self._drafter_collection_samples = 0
        self._drafter_collection_tokens = 0

    def _passes_drafter_collection_sampling(self, request_id: str, collection_global_steps: Optional[int]) -> bool:
        sample_rate = float(self.config.drafter.training.collection_sample_rate)
        if sample_rate <= 0:
            return False
        if sample_rate >= 1:
            return True

        sampling_key = f"{collection_global_steps}:{self.replica_rank}:{request_id}".encode()
        digest = hashlib.blake2b(sampling_key, digest_size=8).digest()
        sample_value = int.from_bytes(digest, byteorder="big", signed=False) / float(1 << 64)
        return sample_value < sample_rate

    def _reserve_drafter_collection_budget(
        self,
        request_id: str,
        collection_global_steps: Optional[int],
        max_new_tokens: int,
    ) -> bool:
        self._reset_drafter_collection_budget_if_needed(collection_global_steps)
        if not self._passes_drafter_collection_sampling(request_id, collection_global_steps):
            return False

        training_cfg = self.config.drafter.training
        max_samples = training_cfg.max_collect_samples_per_step_per_replica
        if max_samples is not None and self._drafter_collection_samples >= int(max_samples):
            return False

        max_tokens = training_cfg.max_collect_tokens_per_step_per_replica
        if max_tokens is not None and self._drafter_collection_tokens + max_new_tokens > int(max_tokens):
            return False

        self._drafter_collection_samples += 1
        self._drafter_collection_tokens += max_new_tokens
        return True

    async def generate(
        self,
        prompt_ids: torch.Tensor,
        sampling_params: dict[str, Any],
        request_id: str,
        image_data: Optional[list[Any]] = None,
        video_data: Optional[list[Any]] = None,
    ) -> TokenOutput:
        """Generate sequence with token-in-token-out."""
        skip_drafter_collection = bool(sampling_params.pop("_verl_skip_drafter_collection", False))
        request_global_steps = sampling_params.pop("_verl_global_steps", None)
        if request_global_steps is not None:
            request_global_steps = int(request_global_steps)

        # TODO(@wuxibin): switch to `/generate` http endpoint once multi-modal support ready.
        max_possible_tokens = self.config.max_model_len - len(prompt_ids) - 1

        if max_possible_tokens < 0:
            raise ValueError(
                f"Prompt length ({len(prompt_ids)}) exceeds the model's maximum context length "
                f"({self.config.max_model_len})."
            )

        if "max_new_tokens" in sampling_params:
            max_new_tokens = sampling_params.pop("max_new_tokens")
        elif "max_tokens" in sampling_params:
            # support vllm-style 'max_tokens' param
            max_new_tokens = sampling_params.pop("max_tokens")
        else:
            # Cap max_tokens by response_length to ensure tensor alignment,
            # and by remaining budget to prevent OOM in multi-turn rollouts.
            max_new_tokens = min(
                self.config.response_length, self.config.prompt_length + self.config.response_length - len(prompt_ids)
            )

        # Clamp max_new_tokens to the valid range [0, max_possible_tokens]
        max_new_tokens = max(0, min(max_new_tokens, max_possible_tokens))

        assert max_new_tokens <= max_possible_tokens, (
            f"max_new_tokens {max_new_tokens} exceeds available context space {max_possible_tokens}"
        )
        sampling_params["max_new_tokens"] = max_new_tokens
        return_logprob = sampling_params.pop("logprobs", False)

        request = {
            "rid": request_id,
            "input_ids": prompt_ids,
            "sampling_params": sampling_params,
            "return_logprob": return_logprob,
            "image_data": image_data,
            # TODO: support video input for sglang
            # video_data=video_data,
        }

        if self.config.enable_rollout_routing_replay:
            request.update({"return_routed_experts": True})

        # return hidden states
        should_collect = False
        collection_global_steps = request_global_steps if request_global_steps is not None else self.global_steps
        collect_interval = max(1, int(self.config.drafter.training.collect_interval_steps))
        collect_this_step = collection_global_steps is None or (collection_global_steps % collect_interval == 0)
        if (
            self.config.drafter.enable
            and self.config.drafter.enable_drafter_training
            and self.config.drafter.training.collect_hidden_states_from_sgl
            and collect_this_step
            and not skip_drafter_collection
            and self._reserve_drafter_collection_budget(request_id, collection_global_steps, max_new_tokens)
        ):
            should_collect = True
            request.update({"return_hidden_states": True})
            if self.config.drafter.training.use_logits:
                request.update({"return_logprob": True})
                request.update({"top_logprobs_num": self.config.drafter.training.logits_topk})

        generate_request = GenerateReqInput(**request)

        # Add lora request
        if self.model_config.lora_rank > 0:
            generate_request.lora_path = SGLANG_LORA_NAME

        output = await self.tokenizer_manager.generate_request(generate_request, None).__anext__()
        meta_info = output.get("meta_info", {})
        finish_reason = meta_info.get("finish_reason")
        finish_reason = finish_reason["type"] if finish_reason else None
        if return_logprob:
            token_ids = list(output.get("output_ids", []))
            output_token_logprobs = meta_info.get("output_token_logprobs") or []
            if output_token_logprobs and len(output_token_logprobs) == len(token_ids):
                log_probs = [float(log_prob) for log_prob, _, _ in output_token_logprobs]
            else:
                # SGLang may return mismatched lengths (e.g. max_new_tokens=0
                # produces a phantom logprob entry with empty output_ids), or
                # an abort may leave an empty logprob payload.
                assert not token_ids, (
                    f"output_token_logprobs length ({len(output_token_logprobs)}) != "
                    f"output_ids length ({len(token_ids)}) for request {request_id}"
                )
                log_probs = []
        else:
            token_ids = output["output_ids"]
            log_probs = None

        if os.getenv(_DEBUG_OUTPUT_TOKENS_ENV, "").strip().lower() not in {"", "0", "false", "off", "no"}:
            limit = int(os.getenv(_DEBUG_OUTPUT_TOKENS_LIMIT_ENV, "8"))
            head = list(token_ids[:limit])
            tail = list(token_ids[-limit:]) if len(token_ids) > limit else []
            eos_token_ids = getattr(self.model_config.hf_config, "eos_token_id", None)
            if eos_token_ids is None:
                eos_token_set = set()
            elif isinstance(eos_token_ids, (list, tuple, set)):
                eos_token_set = {int(token_id) for token_id in eos_token_ids}
            else:
                eos_token_set = {int(eos_token_ids)}
            first_eos_pos = next(
                (idx for idx, token_id in enumerate(token_ids) if int(token_id) in eos_token_set),
                None,
            )
            logger.warning(
                "[SGLangOutputDebug] rid=%s prompt_len=%s output_len=%s max_new=%s "
                "reached_max=%s stop=%s eos=%s first_eos=%s head=%s tail=%s",
                request_id,
                len(prompt_ids),
                len(token_ids),
                max_new_tokens,
                len(token_ids) >= max_new_tokens,
                finish_reason,
                sorted(eos_token_set),
                first_eos_pos,
                head,
                tail,
            )

        routed_experts = None
        if self.config.enable_rollout_routing_replay:
            if self.config.skip_tokenizer_init:
                routed_experts = output.get("meta_info", {}).get("routed_experts", None)
            else:
                from sglang.srt.layers.moe.routed_experts_capturer import extract_routed_experts_from_meta_info

                hf_config = self.model_config.hf_config
                if not hasattr(hf_config, "num_hidden_layers") or not hasattr(hf_config, "num_experts_per_tok"):
                    raise AttributeError(
                        "enable_rollout_routing_replay is set, but hf_config is missing "
                        "'num_hidden_layers' or 'num_experts_per_tok'. This feature requires an MoE model "
                        "configuration that defines these attributes."
                    )
                routed_experts = extract_routed_experts_from_meta_info(output).reshape(
                    -1, hf_config.num_hidden_layers, hf_config.num_experts_per_tok
                )

        drafter_sample = None
        if should_collect:
            target_logprobs = None
            if self.config.drafter.training.use_logits:
                input_top = output.get("meta_info", {}).get("input_top_logprobs", [])
                output_top = output.get("meta_info", {}).get("output_top_logprobs", [])
                if len(input_top) > 1:
                    target_logprobs = _top_logprobs_to_tensor(
                        input_top[1:] + output_top,
                        int(self.config.drafter.training.logits_topk),
                    )
                    if target_logprobs is None:
                        logger.warning("Failed to convert top_logprobs to tensor; skip target_logprobs collection")
                elif output_top:
                    # Keep prompt-prefix logprob collection off by default so
                    # drafter targets do not force SGLang onto a different
                    # prefix-cache/logprob path. Prompt rows are masked out
                    # during drafter training, so invalid placeholders are OK.
                    prompt_target_padding = [[] for _ in range(max(len(prompt_ids) - 1, 0))]
                    target_logprobs = _top_logprobs_to_tensor(
                        prompt_target_padding + output_top,
                        int(self.config.drafter.training.logits_topk),
                    )
                    if target_logprobs is None:
                        logger.warning("Failed to convert output top_logprobs to tensor; skip target_logprobs collection")

            hidden_states_data = output.get("meta_info", {}).get("hidden_states", [])
            hidden_states_list = []
            for hs in _iter_hidden_state_chunks(hidden_states_data):
                h_states = _hidden_state_chunk_to_tensor(hs)
                if h_states is not None:
                    hidden_states_list.append(h_states)

            if hidden_states_list:
                prompt_tensor = torch.as_tensor(prompt_ids, dtype=torch.long).detach().cpu()
                response_tensor = torch.tensor(token_ids, dtype=torch.long)
                hidden_states = torch.cat(hidden_states_list, dim=0)
                max_hidden_tokens = self.config.drafter.training.hidden_state_max_tokens_per_sample
                if max_hidden_tokens is not None and int(max_hidden_tokens) > 0:
                    hidden_states = hidden_states[-int(max_hidden_tokens):]
                drafter_sample = {
                    "input_ids": torch.cat([prompt_tensor, response_tensor], dim=0).unsqueeze(0),
                    "prompts": prompt_tensor.unsqueeze(0),
                    "responses": response_tensor.unsqueeze(0),
                    "hidden_states": hidden_states.unsqueeze(0).cpu(),
                    "target_logprobs": target_logprobs.unsqueeze(0).cpu() if target_logprobs is not None else None,
                    "global_step": collection_global_steps,
                    "replica_rank": self.replica_rank,
                }
            else:
                logger.warning("[SGLangHttpServer] No valid hidden states returned for drafter sample collection")

        extra_fields = {"global_steps": collection_global_steps}
        if drafter_sample is not None:
            extra_fields["drafter_sample"] = drafter_sample

        return TokenOutput(
            token_ids=token_ids,
            log_probs=log_probs,
            routed_experts=routed_experts,
            stop_reason=finish_reason,
            extra_fields=extra_fields,
        )

    async def set_global_steps(self, global_steps: int):
        """Set the global steps of the model weights."""
        self.global_steps = global_steps

    async def abort_all_requests(self):
        await self.tokenizer_manager.pause_generation(PauseGenerationReqInput(mode="abort"))

    async def resume_generation(self):
        await self.tokenizer_manager.continue_generation(ContinueGenerationReqInput())

    async def start_profile(self, **kwargs):
        if (
            self.profiler_controller.check_enable()
            and self.profiler_controller.check_this_rank()
            and self.profiler_controller.is_discrete_mode()
        ):
            profile_args = build_sglang_profiler_args(
                self.profiler_controller.config, self.profiler_controller.tool_config, self.replica_rank
            )
            await self.tokenizer_manager.start_profile(**profile_args)

    async def stop_profile(self):
        if (
            self.profiler_controller.check_enable()
            and self.profiler_controller.check_this_rank()
            and self.profiler_controller.is_discrete_mode()
        ):
            await self.tokenizer_manager.stop_profile()


class SGLangReplica(RolloutReplica):
    def __init__(
        self,
        replica_rank: int,
        config: RolloutConfig,
        model_config: HFModelConfig,
        gpus_per_node: int = 8,
        is_reward_model: bool = False,
        is_teacher_model: bool = False,
        name_suffix: str = "",
    ):
        super().__init__(
            replica_rank, config, model_config, gpus_per_node, is_reward_model, is_teacher_model, name_suffix
        )
        self.server_class = ray.remote(SGLangHttpServer)

    async def launch_servers(self):
        """Launch http server in each node."""
        assert len(self.workers) == self.world_size, (
            f"worker number {len(self.workers)} not equal to world size {self.world_size}"
        )

        # get (node_id, CUDA_VISIBLE_DEVICES) of all workers
        worker_infos = await asyncio.gather(
            *[
                worker.__ray_call__.remote(
                    lambda self: (ray.get_runtime_context().get_node_id(), os.environ[visible_devices_keyword])
                )
                for worker in self.workers
            ]
        )
        worker_cuda_visible_devices = [worker_info[1] for worker_info in worker_infos]
        worker_node_ids = [worker_info[0] for worker_info in worker_infos]
        base_gpu_id = 0
        infer_tp = self.config.tensor_model_parallel_size * self.config.data_parallel_size
        replica_world_size = infer_tp * self.config.pipeline_model_parallel_size
        if os.environ.get(f"RAY_EXPERIMENTAL_NOSET_{visible_devices_keyword}", None):
            logger.warning(f"RAY_EXPERIMENTAL_NOSET_{visible_devices_keyword} is set True!")
            base_gpu_id = (0 + self.replica_rank * replica_world_size) % self.gpus_per_node
        # create server actor in each node with node affinity and cuda visible devices
        for node_rank in range(self.nnodes):
            workers = self.workers[
                node_rank * self.gpus_per_replica_node : (node_rank + 1) * self.gpus_per_replica_node
            ]
            node_cuda_visible_devices_set = worker_cuda_visible_devices[
                node_rank * self.gpus_per_replica_node : (node_rank + 1) * self.gpus_per_replica_node
            ]
            node_cuda_visible_devices = ",".join(
                map(
                    str,
                    sorted(
                        set(
                            int(device)
                            for worker_devices_set in node_cuda_visible_devices_set
                            for device in worker_devices_set.split(",")
                            if device.strip()
                        )
                    ),
                )
            )

            node_id = worker_node_ids[node_rank * self.gpus_per_replica_node]
            if self.is_reward_model:
                name = f"sglang_server_reward_{self.replica_rank}_{node_rank}{self.name_suffix}"
            elif self.is_teacher_model:
                name = f"sglang_server_teacher_{self.replica_rank}_{node_rank}{self.name_suffix}"
            else:
                name = f"sglang_server_{self.replica_rank}_{node_rank}{self.name_suffix}"
            server = self.server_class.options(
                scheduling_strategy=ray.util.scheduling_strategies.NodeAffinitySchedulingStrategy(
                    node_id=node_id,
                    soft=False,
                ),
                runtime_env={"env_vars": {f"RAY_EXPERIMENTAL_NOSET_{visible_devices_keyword}": "1"}},
                name=name,
                max_concurrency=self.max_concurrency,
            ).remote(
                config=self.config,
                model_config=self.model_config,
                rollout_mode=self.rollout_mode,
                workers=workers,
                replica_rank=self.replica_rank,
                node_rank=node_rank,
                nnodes=self.nnodes,
                cuda_visible_devices=node_cuda_visible_devices,
                base_gpu_id=base_gpu_id,
            )
            self.servers.append(server)

        # launch http server in each node
        master_address, master_port = None, None
        if self.nnodes > 1:
            master_address, master_port = await self.servers[0].get_master_address.remote()
        await asyncio.gather(
            *[
                server.launch_server.remote(master_address=master_address, master_port=master_port)
                for server in self.servers
            ]
        )

        # get http server address from first server
        server_address, server_port = await self.servers[0].get_server_address.remote()
        self._server_handle = self.servers[0]
        self._server_address = (
            f"[{server_address}]:{server_port}"
            if is_valid_ipv6_address(server_address)
            else f"{server_address}:{server_port}"
        )
