import logging
import os
import glob
from copy import deepcopy
import safetensors

import torch
from torch.nn import SmoothL1Loss
from torch.nn import functional as F

from .model.eagle.llama_eagle import LlamaForCausalLMEagle3, LlamaForCausalLMEagle
from .model.auto import AutoDraftModelConfig, AutoEagle3DraftModel, AutoEagleDraftModel
from .eagle_trainer_backend import EagleTrainerBackend

from verl.utils.torch_functional import (
    get_constant_schedule_with_warmup,
    get_cosine_schedule_with_warmup,
)


logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "INFO"))


class Eagle3TrainerBackend(EagleTrainerBackend):
    
    def compute_loss(self, model, batch, _current_pad_size):
        """
        Compute Eagle3 multi-step prediction losses
        """
        
