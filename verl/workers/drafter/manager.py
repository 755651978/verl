import logging
import os
import asyncio

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "INFO"))

class RolloutDrafterManager:
    def __init__(self, rollout_config, dp_rank):
        # training
        self.train_drafter = bool(rollout_config.drafter.enable and rollout_config.drafter.enable_drafter_training)
        self.trainer_backend = None

        # step tracking
        self.current_rl_step = 1
        self.training_interval_steps = int(rollout_config.drafter.training.get("training_interval_steps", 1))
        self.collection_interval_steps = int(rollout_config.drafter.training.get("collect_interval_steps", 1))
        self.step = rollout_config.drafter.training.get("step", 100)
        self.training_active = False


    async def run_training_loop(self):
        if self.trainer_backend is None:
            logger.warning("Drafter trainer backend is not initialized; skip training loop")
            return
        if self.should_train_this_step():
            if not self.training_active:
                self.training_active = True
                success = await self.trainer_backend.activate_training_model()
                if not success:
                    logger.error(f"Worker {self.rank} failed to activate training model")
                    self.training_active = False
                    return

            for _ in range(self.step):
                success = await self.trainer_backend.training_step(self.current_rl_step)
                if success:
                    logger.info(f"Successfully trained drafter.")
            await self.trainer_backend.cleanup_training()
            self.training_active = False


    def should_train_this_step(self):
        if not self.train_drafter or self.training_interval_steps <= 0:
            return False
        return self.current_rl_step % self.training_interval_steps == 0


    def should_collect_data_this_step(self):
        if not self.train_drafter or self.collection_interval_steps <= 0:
            return False
        return self.current_rl_step % self.collection_interval_steps == 0


    def update_rl_step(self, global_step = None):
        if global_step is not None:
            self.current_rl_step = global_step
            if self.trainer_backend is not None and hasattr(self.trainer_backend, "increment_rl_step"):
                self.trainer_backend.increment_rl_step()
        logger.debug(f"RolloutDrafterManager RL step updates to {self.current_rl_step}")


    def maybe_publish(self):
        if self.trainer_backend is not None and self.should_train_this_step():
            weights = self.trainer_backend.get_model_state_dict()
            return weights
        return None