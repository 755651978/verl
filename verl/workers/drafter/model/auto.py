import os
import json
import torch
from typing import Union, Optional
from transformers import AutoConfig, PretrainedConfig

# Import our specific implementations
from .eagle3.llama_eagle3 import LlamaForCausalLMEagle3
from .eagle3.qwen_eagle3 import QwenEagle

class AutoDraftConfig:
    """
    Helper class to load and adapt configurations for Eagle drafters.
    """
    @classmethod
    def from_pretrained(cls, model_path: str, **kwargs) -> PretrainedConfig:
        """
        Loads the config and ensures it has the necessary Eagle-specific fields.
        """
        config = AutoConfig.from_pretrained(model_path, **kwargs)
        
        # Ensure draft_vocab_size and target_hidden_size are present
        # If not in config, we might need to infer them or set defaults
        if not hasattr(config, "draft_vocab_size"):
            config.draft_vocab_size = config.vocab_size
            
        return config

class AutoDraftModel:
    """
    Factory class to instantiate the correct Eagle model based on the config.
    """
    # Registry mapping model_type to our Eagle implementations
    _MODEL_MAPPING = {
        "llama": LlamaForCausalLMEagle3,
        "qwen2": QwenEagle,
        "qwen2_5": QwenEagle,  # Usually shares the same architecture as qwen2
        "qwen3": QwenEagle,
    }

    @classmethod
    def from_pretrained(
        cls, 
        model_path: str, 
        config: Optional[PretrainedConfig] = None,
        **kwargs
    ) -> Union[LlamaForCausalLMEagle3, QwenEagle]:
        """
        Instantiates a Drafter model from a pre-trained path.
        
        Args:
            model_path: Path to the checkpoint directory.
            config: Optional pre-loaded config object.
            kwargs: Additional arguments for model initialization (e.g., device_map).
        """
        if config is None:
            config = AutoDraftConfig.from_pretrained(model_path)

        model_type = getattr(config, "model_type", "").lower()

        # 1. Architecture Detection
        # Some configs might use 'qwen2' for qwen2.5, so we check both
        target_cls = None
        for key, model_cls in cls._MODEL_MAPPING.items():
            if key in model_type:
                target_cls = model_cls
                break

        if target_cls is None:
            raise ValueError(
                f"Model type '{model_type}' is not supported by Eagle-3 Drafter. "
                f"Supported types: {list(cls._MODEL_MAPPING.keys())}"
            )

        # 2. Instantiate the specific Eagle Model
        # This will call the __init__ of LlamaEagle or QwenEagle
        print(f"[AutoDraftModel] Initializing {target_cls.__name__} for model_type: {model_type}")

        # Note: We pass the config to the constructor
        model = target_cls(config)

        # 3. Load Weights
        # Using safe_serialization if available, else standard torch.load
        weights_path = os.path.join(model_path, "pytorch_model.bin")
        if os.path.exists(weights_path):
            state_dict = torch.load(weights_path, map_location="cpu")
            model.load_state_dict(state_dict, strict=False)
            print(f"[AutoDraftModel] Weights loaded from {weights_path}")
        else:
            # In training scenarios, weights might be initialized randomly or via FSDP
            print(f"[AutoDraftModel] No weights found at {model_path}. Initializing with random weights.")

        return model

    @classmethod
    def from_config(cls, config: PretrainedConfig) -> Union[LlamaForCausalLMEagle3, QwenEagle]:
        """
        Instantiate a model directly from a config object (useful for fresh training).
        """
        model_type = getattr(config, "model_type", "").lower()

        for key, model_cls in cls._MODEL_MAPPING.items():
            if key in model_type:
                return model_cls(config)

        raise ValueError(f"Unsupported model_type in config: {model_type}")