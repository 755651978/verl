"""
Modeling layer for Eagle-style drafters.
"""

# Import specific implementations for direct access
from .eagle3.eagle import LlamaForCausalLMEagle3

# __all__ defines what is exported when someone does 'from modeling import *'
__all__ = [
    "LlamaForCausalLMEagle3", 
]