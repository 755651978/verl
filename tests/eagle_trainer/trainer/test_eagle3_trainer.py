import os
import torch
import torch.distributed as dist
from omegaconf import OmegaConf
from unittest.mock import MagicMock
from transformers import AutoConfig
from torch.distributed.device_mesh import init_device_mesh

from verl.workers.drafter.eagle3_trainer_backend import Eagle3TrainerBackend
from verl.workers.drafter.base_trainer import DrafterBaseTrainer
from verl.workers.drafter.model.eagle import LlamaForCausalLMEagle3


def setup_real_dist():
    # 1. 初始化分布式环境（FSDP2 依赖）
    if not dist.is_initialized():
        dist.init_process_group(backend="cpu:gloo,cuda:nccl")
        local_rank = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(local_rank)
        return local_rank
        
    return dist.get_rank()
    
def test_eagle_training_flow():
    # 1、初始化分布式环境
    rank = setup_real_dist()
    device = torch.device(f"cuda:{rank}")
    world_size = dist.get_world_size()
       
    # 2、配置加载
    config = OmegaConf.load("/nas/disk6/ls/workspace/verl-spec-3/tests/eagle_trainer/config/eagle3_trainer.yaml")

    # 3、准备模型配置
    target_config = AutoConfig.from_pretrained("/nas/disk6/ls/angelslim-qwen3-8b-eagle3", trust_remote_code=True)

    # 4. 实例化 Backend
    backend = Eagle3TrainerBackend(
        config=config,
        target_model_config=target_config,
    )

    # 5. 实例化 Trainer
    trainer = DrafterBaseTrainer(
        config=config,
        world_size=world_size,
        rollout_dp_rank=rank,
        backend=backend
    )

    # 6. 数据采集测试
    print("--- 步骤 1: 采集在线数据 ---")
    # 定义 batch 大小和序列长度
    batch_size = 2  # 对应你 config 里的 batch_size_per_gpu
    seq_len = 256
    prompt_len = 128
    response_len = 128 # 确保 prompt_len + response_len = seq_len
    hidden_dim = target_config.hidden_size

    # 构造 Mock 数据，所有 Tensor 第一维必须是 batch_size
    mock_batch = {
        "input_ids": torch.randint(0, target_config.vocab_size, (batch_size, seq_len)),
        "responses": torch.randint(0, target_config.vocab_size, (batch_size, response_len)),
        "prompts": torch.randint(0, target_config.vocab_size, (batch_size, prompt_len)),
    }
    # 模拟隐藏层状态
    mock_hiddens_states = torch.randn(batch_size, seq_len, hidden_dim*4)

    trainer.collect_online_data(mock_batch, mock_hiddens_states)
    print(f"Data collect. Buffer size: {len(trainer.data_buffer)}")

     # 7. 模型初始化
    print("--- 步骤 2: 激活模型 ---")
    import asyncio
    mesh = init_device_mesh("cuda", (world_size,))
    training_ranks = list(range(world_size))
    async def activate_model():
        success = await trainer.activate_training_model(mesh, training_ranks)
        return success
    loop = asyncio.get_event_loop() 
    active_success = loop.run_until_complete(activate_model()) 
    
    if active_success:
        print("\n ✔ 激活模型成功")
        print(f"模型架构：{type(trainer.model)}")
        print(f"主模型 Head(TargetHead)：{type(backend.target_model)}")
        # 检查模型是否正确移至 GPU
        first_param_device = next(trainer.model.parameters()).device
        print(f"模型当前所在设备：{first_param_device}")
    else:
        print("\n ❌ 模型激活失败")
        return


    # 8. 训练步执行测试
    print("--- 步骤 3: 执行训练步 ---")
    async def run_step():
        success = await trainer.training_step(step=1)
        return success
    
    loop = asyncio.get_event_loop() 
    step_success = loop.run_until_complete(run_step()) 

    if step_success: 
        print("Congratulations! The training_step finished successfully.") 
        print(f"Current training steps counter: {trainer.training_steps}") 
    else: 
        print("training_step failed (check logs for 'Not enough data' or other warnings).") 
        return
    print("--- 集成测试完成！ ---")

if __name__ == "__main__":
    test_eagle_training_flow()

    

