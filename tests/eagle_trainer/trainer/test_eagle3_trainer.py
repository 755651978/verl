import os
import sys
import torch
import torch.distributed as dist
import asyncio
import unittest
from omegaconf import OmegaConf
from transformers import AutoConfig, LlamaForCausalLM
from unittest.mock import MagicMock
from tensordict import TensorDict

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, "../../../"))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from verl.workers.drafter.eagle_trainer_backend import EagleTrainerBackend
from verl.workers.drafter.base_trainer import DrafterBaseTrainer

class TestDraftTraining(unittest.TestCase):


    @classmethod
    def setUpClass(cls):
        # 1. 初始化分布式环境（FSDP2 依赖）
        if not dist.is_initialized():
            dist.init_process_group(backend="nccl")
        
        cls.rank = dist.get_rank()
        cls.world_size = dist.get_world_size()
        torch.cuda.set_device(cls.rank)

        cls.device = "cuda" if torch.cuda.is_available() else "cpu"
    
    def test_eagle_training_flow(self):
        """测试全流程：构建 -> 数据采集 -> 训练步"""
        asyncio.run(self.run_test())

    async def run_test(self):
        
        # 2、配置加载
        config = OmegaConf.load("/nas/disk6/ls/workspace/verl-spec/tests/eagle_trainer/config/eagle3_trainer.yaml")

        # 3、准备模型配置
        target_config = AutoConfig.from_pretrained("/model/Llama-3.2-1B/", trust_remote_code=True)

        # 4、Mock FSDP对象（由于测试只有一个进程，我们Mock unshard行为）
        mock_fsdp = MagicMock()
        mock_fsdp.unshard.return_value = MagicMock(__enter__=MagicMock(), __exit__=MagicMock())
        # 模拟主模型的model对象，用于Embedding共享
        mock_fsdp.model = MagicMock()
        mock_fsdp.model.embed_tokens = torch.nn.Embedding(target_config.vocab_size, target_config.hidden_size)
        mock_fsdp.lm_head = torch.nn.Linear(target_config.hidden_size, target_config.vocab_size, bias=False)

        # 5. 实例化 Backend
        backend = EagleTrainerBackend(
            config=config,
            target_model_config=target_config,
            target_model_fsdp=mock_fsdp
        )

        # 6. 实例化 Trainer
        trainer = DrafterBaseTrainer(
            config=config,
            world_size=self.world_size,
            rollout_dp_rank=self.rank,
            backend=backend
        )

        # 7. 模型初始化
        print("--- 步骤 1: 构建模型 ---")
        trainer.build_draft_model()
        self.assertIsNotNone(trainer.model)

        # 8. 数据采集测试
        print("--- 步骤 2: 采集在线数据 ---")
        # 定义 batch 大小和序列长度
        batch_size = 2  # 对应你 config 里的 batch_size_per_gpu
        seq_len = 256
        prompt_len = 128
        response_len = 128 # 确保 prompt_len + response_len = seq_len
        hidden_dim = target_config.hidden_size

        # 构造 Mock 数据，所有 Tensor 第一维必须是 batch_size
        mock_batch = {
            "input_ids": torch.randint(0, target_config.vocab_size, (batch_size, seq_len)).cuda(),
            "responses": torch.randint(0, target_config.vocab_size, (batch_size, response_len)).cuda(),
            "prompts": torch.randint(0, target_config.vocab_size, (batch_size, prompt_len)).cuda(),
            "attention_mask": torch.ones((batch_size, seq_len), dtype=torch.bool).cuda(),
            "position_ids": torch.arange(seq_len).unsqueeze(0).expand(batch_size, -1).cuda()
        }

        # hidden_states 应该是一个列表，长度等于 batch_size
        # 每个元素的形状应为 [seq_len, hidden_dim] 或 [1, seq_len, hidden_dim]
        # 注意：对于 Eagle 3，hidden_states 的最后一维需要是 hidden_size * 3 (拼接了3层)
        is_eagle3 = False # 根据你的模型判断
        multiplier = 3 if is_eagle3 else 1

        mock_h_states = [
            torch.randn(seq_len, hidden_dim * multiplier).cuda() for _ in range(batch_size)
        ]

        # 采集两组数据以满足 batch_size=2
        trainer.collect_online_data(mock_batch, mock_h_states)
        trainer.collect_online_data(mock_batch, mock_h_states)

        self.assertEqual(len(trainer.collected_data), 2)

        # 9. 训练步执行测试
        print("--- 步骤 3: 执行训练步 ---")
        # 确保优化器存在 (DrafterBaseTrainer 中由于拼写错误是 self.oprimizer)
        if hasattr(trainer, 'oprimizer'):
            trainer.optimizer = trainer.oprimizer

        success = await trainer.training_step(step=1)
        self.assertTrue(success, "训练步执行失败")

        print("--- 集成测试完成！ ---")

    @classmethod
    def tearDownClass(cls):
        if dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    unittest.main()

    

