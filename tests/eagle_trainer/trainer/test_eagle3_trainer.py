import asyncio
import os
import shutil
import tempfile
from pathlib import Path

import pytest
import torch
import torch.distributed as dist
from omegaconf import OmegaConf
from transformers import AutoConfig, LlamaConfig, LlamaForCausalLM

from verl.workers.drafter.base_trainer import DrafterBaseTrainer
from verl.workers.drafter.eagle3_trainer_backend import Eagle3TrainerBackend
from verl.workers.drafter.model.eagle import LlamaForCausalLMEagle3


def _init_dist() -> tuple[int, int]:
    if not torch.cuda.is_available():
        pytest.skip("EAGLE3 multi-card integration test requires CUDA")
    if "LOCAL_RANK" not in os.environ or "WORLD_SIZE" not in os.environ:
        pytest.skip("Run with torchrun, for example: torchrun --standalone --nproc-per-node=2 this_file.py")

    if not dist.is_initialized():
        dist.init_process_group(backend="nccl")

    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    return local_rank, dist.get_world_size()


def _tiny_llama_config() -> LlamaConfig:
    return LlamaConfig(
        vocab_size=64,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=1,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=128,
        pad_token_id=0,
        bos_token_id=1,
        eos_token_id=2,
        tie_word_embeddings=False,
    )


def _create_tiny_checkpoints(root: Path) -> tuple[Path, Path]:
    target_dir = root / "target"
    drafter_dir = root / "drafter"
    target_dir.mkdir(parents=True, exist_ok=True)
    drafter_dir.mkdir(parents=True, exist_ok=True)

    target_config = _tiny_llama_config()
    target_model = LlamaForCausalLM(target_config)
    target_model.save_pretrained(target_dir, safe_serialization=True, max_shard_size="10KB")

    drafter_config = _tiny_llama_config()
    drafter_config.architectures = ["LlamaForCausalLMEagle3"]
    drafter_config.draft_vocab_size = drafter_config.vocab_size
    drafter_config.target_hidden_size = target_config.hidden_size
    drafter_config.num_hidden_layers = 1
    drafter_model = LlamaForCausalLMEagle3(drafter_config)
    drafter_model.save_pretrained(drafter_dir, safe_serialization=True, max_shard_size="10KB")

    return target_dir, drafter_dir


def _create_compressed_vocab_checkpoints(root: Path) -> tuple[Path, Path, torch.Tensor, torch.Tensor]:
    target_dir = root / "target_compressed_vocab"
    drafter_dir = root / "drafter_compressed_vocab"
    target_dir.mkdir(parents=True, exist_ok=True)
    drafter_dir.mkdir(parents=True, exist_ok=True)

    target_config = _tiny_llama_config()
    target_model = LlamaForCausalLM(target_config)
    target_model.save_pretrained(target_dir, safe_serialization=True, max_shard_size="10KB")

    drafter_config = _tiny_llama_config()
    drafter_config.architectures = ["LlamaForCausalLMEagle3"]
    drafter_config.draft_vocab_size = 8
    drafter_config.target_hidden_size = target_config.hidden_size
    drafter_config.num_hidden_layers = 1
    drafter_model = LlamaForCausalLMEagle3(drafter_config)

    selected_tokens = torch.tensor([0, 2, 5, 7, 11, 13, 17, 19], dtype=torch.long)
    t2d = torch.zeros(target_config.vocab_size, dtype=torch.bool)
    t2d[selected_tokens] = True
    d2t = selected_tokens - torch.arange(drafter_config.draft_vocab_size, dtype=torch.long)
    drafter_model.t2d.copy_(t2d)
    drafter_model.d2t.copy_(d2t)
    drafter_model.save_pretrained(drafter_dir, safe_serialization=True, max_shard_size="10KB")

    return target_dir, drafter_dir, t2d, d2t


def _build_config(target_dir: Path, drafter_dir: Path):
    return OmegaConf.create(
        {
            "model": {
                "path": str(target_dir),
                "local_hf_config_path": str(target_dir),
                "trust_remote_code": False,
            },
            "rollout": {
                "tensor_model_parallel_size": 1,
                "drafter": {
                    "enable": True,
                    "enable_drafter_training": True,
                    "speculative_algorithm": "EAGLE3",
                    "model_path": str(drafter_dir),
                    "checkpoint_path": None,
                    "training": {
                        "collect_hidden_states_from_sgl": True,
                        "use_data_buffer": False,
                        "batch_size_per_gpu": 2,
                        "step": 100,
                        "lr": 1e-4,
                        "lr_warmup_steps": 0,
                        "warmup_style": "constant",
                        "use_logits": False,
                        "ttt_length": 2,
                        "current_max_samples": 16,
                        "data_buffer_max_size": 16,
                        "fsdp_config": {
                            "wrap_policy": {"min_num_params": 0},
                            "use_orig_params": True,
                            "forward_prefetch": False,
                        },
                    },
                },
            },
        }
    )


def test_eagle3_build_model_uses_checkpoint_vocab_mapping_without_mapping_path():
    root = Path(tempfile.mkdtemp(prefix="verl_eagle3_vocab_mapping_"))
    try:
        target_dir, drafter_dir, expected_t2d, expected_d2t = _create_compressed_vocab_checkpoints(root)
        config = _build_config(target_dir, drafter_dir)
        config.rollout.drafter.training.use_logits = True

        target_config = AutoConfig.from_pretrained(target_dir)
        backend = Eagle3TrainerBackend(config=config, target_model_config=target_config)
        drafter_model, _ = backend.build_model()

        assert drafter_model.draft_vocab_size == int(expected_t2d.sum().item())
        assert torch.equal(drafter_model.t2d.cpu(), expected_t2d)
        assert torch.equal(drafter_model.d2t.cpu(), expected_d2t)
    finally:
        shutil.rmtree(root, ignore_errors=True)


def _mock_rollout_batch(config: LlamaConfig, batch_size: int = 2, seq_len: int = 24, prompt_len: int = 8):
    response_len = seq_len - prompt_len
    input_ids = torch.randint(3, config.vocab_size, (batch_size, seq_len), device="cuda")
    prompts = input_ids[:, :prompt_len].contiguous()
    responses = input_ids[:, prompt_len:].contiguous()

    hidden_states = torch.randn(
        batch_size,
        seq_len,
        config.hidden_size * 4,
        device="cuda",
        dtype=torch.bfloat16,
    )
    return {
        "input_ids": input_ids,
        "prompts": prompts,
        "responses": responses,
    }, hidden_states


def test_eagle3_multigpu_model_and_training_flow():
    _, world_size = _init_dist()
    root = Path(os.environ.get("EAGLE3_TEST_TMPDIR", tempfile.gettempdir())) / (
        f"verl_eagle3_tiny_{os.environ.get('MASTER_PORT', 'standalone')}"
    )

    if dist.get_rank() == 0:
        shutil.rmtree(root, ignore_errors=True)
        target_dir, drafter_dir = _create_tiny_checkpoints(root)
    dist.barrier()
    target_dir, drafter_dir = root / "target", root / "drafter"

    config = _build_config(target_dir, drafter_dir)
    target_config = AutoConfig.from_pretrained(target_dir)
    backend = Eagle3TrainerBackend(config=config, target_model_config=target_config)
    trainer = DrafterBaseTrainer(
        config=config,
        world_size=world_size,
        rollout_dp_rank=dist.get_rank(),
        training_device_mesh=None,
        backend=backend,
        training_process_group=dist.group.WORLD,
    )

    try:
        assert asyncio.run(trainer.activate_training_model())
        assert isinstance(backend.target_model, torch.nn.Module)

        raw_model = trainer.model.module if hasattr(trainer.model, "module") else trainer.model
        assert raw_model.draft_vocab_size == target_config.vocab_size
        assert raw_model.target_hidden_size == target_config.hidden_size

        batch, hidden_states = _mock_rollout_batch(target_config)
        trainer.collect_online_data(batch, hidden_states)
        assert len(trainer.collected_data) == batch["input_ids"].size(0)

        train_batch = trainer._prepare_training_batch()
        assert train_batch is not None
        with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
            outputs = trainer.backend.compute_loss(trainer.model, train_batch, trainer._current_pad_size)
        assert outputs["total_local_ploss"].requires_grad
        assert outputs["local_num_tokens"].item() > 0

        assert asyncio.run(trainer.training_step(step=1))
        assert trainer.training_steps == 1
    finally:
        dist.barrier()
        if dist.get_rank() == 0:
            shutil.rmtree(root, ignore_errors=True)
        dist.barrier()
        if dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    test_eagle3_multigpu_model_and_training_flow()
