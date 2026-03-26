"""
Modeling layer for Eagle-style drafters.
"""

# Import the factory class
from .auto import AutoDraftModel, AutoDraftConfig

# Import specific implementations for direct access
from .eagle3.llama_eagle3 import LlamaForCausalLMEagle3
from .eagle3.qwen_eagle3 import QwenEagle
from .eagle3.base import Eagle3DraftModel

# __all__ defines what is exported when someone does 'from modeling import *'
__all__ = [
    "AutoDraftModel", 
    "AutoDraftConfig", 
    "LlamaForCausalLMEagle3", 
    "QwenEagle", 
    "Eagle3DraftModel"
]