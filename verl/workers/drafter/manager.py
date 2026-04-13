import logging
import os

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "INFO"))

class RolloutDrafterManager:
    def __init__(self, rollout_config, dp_rank):
        # training
        self.train_drafter = rollout_config
        self.trainer_backend = None

        # step tracking
        self.current_rl_step = 0
        self.training_interval_steps = rollout_config.drafter.training.get("training_interval_steps")
        self.collection_interval_steps = rollout_config.drafter.get("training.collection_interval_steps")
        self.step = rollout_config.drafter.training.get("step", 100)


    async def run_training_loop(self):
        if self.should_train_this_step():
            success = self.trainer_backend.training_step(self.step)
            if success:
                logger.info(f"Successfully trained drafter.")


    def should_train_this_step(self):
        if not self.train_drafter:
            return False
        return self.current_rl_step % self.training_interval_steps == 0


    def should_collect_data_this_step(self):
        if not self.train_drafter:
            return False
        return self.current_rl_step % self.collection_interval_steps == 0


    def update_rl_step(self, global_step = None):
        if global_step is not None:
            self.current_rl_step = global_step
            self.trainer_backend.update_rl_step(self.current_rl_step)
        logger.debug(f"RolloutDrafterManager RL step updates to {self.current_rl_step}")


    def maybe_publish(self):
        if self.should_train_this_step():
            weights = self.trainer_backend.get_model_state_dict()
            return weights
        return None
