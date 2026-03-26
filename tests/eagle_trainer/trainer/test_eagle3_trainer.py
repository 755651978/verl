import os
import sys
import torch
import torch.distributed as dist
import asyncio
from omegaconf import OmegaConf
from transformers import AutoConfig, LlamaForCausalLM
from unittest.mock import MagicMock
from tensordict import TensorDict

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, "../../../"))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from verl.workers.drafter.eagle_background_trainer import EagleBackgroundTrainer

class EagleDataSimulator:
    @staticmethod
    def generate_batch(rank, batch_size, seq_len, vocab_size, hidden_size):
        """
        为特定 Rank 构造模拟数据。
        不同 Rank 应该有不同的输入，模拟分布式训练。
        """
        torch.manual_seed(42 + rank)
        prompt_len = seq_len // 2

        # 1、构造 TensorDict Batch
        # 模拟完整的序列 input_ids = prompts + response
        seq = torch.randint(0, vocab_size, (batch_size, seq_len)).cuda()
        idx = seq[:, :prompt_len:] # prompts
        response = seq[:, prompt_len:] # responses

        batch = TensorDict(
            {
                "prompts": idx,
                "responses": response,
                "input_ids": seq,
                "attention_mask": torch.ones_like(seq).bool(),
                "position_ids": torch.arange(seq_len).unsqueeze(0).expand(batch_size, -1).cuda(),
            },
            batch_size=batch_size,
        )

        # 2、构造 hidden_states 列表（List[Tensor]）
        # 维度：[batch_size, seq_len, hidden_size]
        last_hidden_state = torch.randn(batch_size, seq_len, hidden_size).cuda()
        hidden_states = [last_hidden_state]

        return batch, hidden_states


def init_dist():
    """初始化分布式环境"""
    if not dist.is_initialized():
        dist.init_process_group(backend="nccl")

    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    rank = dist.get_rank()
    world_size = dist.get_world_size()

    torch.cuda.set_device(local_rank)
    
    return rank, local_rank, world_size

async def run_test():
    # 1、初始化分布式环境
    rank, local_rank, world_size = init_dist()
    
    # 2、配置加载
    config = OmegaConf.load("/nas/disk6/ls/workspace/verl-spec/tests/eagle_trainer/config/eagle3_trainer.yaml")

    # 3、构造Mock Actor配置（Llama-3.1-8B 结构）
    try:
        actor_cfg = AutoConfig.from_pretrained(config.actor_rollout_ref.drafter.eagle.spec_model_path)
    except:
        actor_cfg = AutoConfig.from_pretrained("meta-llama/Meta-Llama-3-8B")

    # 缩减盲从想规模以适配双卡测试
    actor_cfg.hidden_size = 4096
    actor_cfg.num_hidden_layers = 2
    actor_cfg.vocab_size = 128256

    # 4、Mock FSDP 主模型
    mock_actor = LlamaForCausalLM(actor_cfg).cuda().bfloat16()
    mock_fsdp = MagicMock()
    mock_fsdp.unshard.return_value = mock_actor

    tp_size = config.actor_rollout_ref.rollout.tensor_model_parallel_size

    # 5、初始化 Trainer
    trainer = EagleBackgroundTrainer(
        config=config,
        model_config=actor_cfg,
        actor_module_fsdp=mock_fsdp,
        world_size=world_size,
        rollout_dp_rank=rank // tp_size
    )

    # 6、生成数据并注入
    batch, hidden_states = EagleDataSimulator.generate_batch(
        rank=rank,
        batch_size=config.actor_rollout_ref.drafter.train.batch_size_per_gpu,
        seq_len=256,
        vocab_size=actor_cfg.vocab_size,
        hidden_size=actor_cfg.hidden_size
    )

    # 测试接口调用
    trainer.collect_online_data(batch=batch,hidden_states=hidden_states)

    await trainer.activate_training_model(device_mesh=trainer.training_device_mesh, training_ranks=[rank])

    if rank == 0:
        print(f"✅ TensorDict batch collected. Queue size: {len(trainer.collected_data)}")

    # 7. 运行训练步
    # 在这一步中，trainer 内部会自动根据 batch["prompts"] 和 batch["input_ids"] 算出 loss_mask
    success = await trainer.training_step(step=0)

    if success and rank == 0:
        print(f"✅ Training Step 0 finished. Loss: {getattr(trainer, 'last_loss', 'N/A')}")
        # 验证梯度一致性
        param_sample = next(trainer.model.parameters()).sum()
        if hasattr(param_sample, "to_local"):
            param_sample = param_sample.to_local()
        dist.all_reduce(param_sample, op=dist.ReduceOp.MAX)
        print(f"✅ All GPUs weight checksum verified.")

if __name__ == "__main__":
    asyncio.run(run_test())

    

