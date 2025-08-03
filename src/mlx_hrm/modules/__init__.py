"""
MLX HRM modules - Higher-level components that use foundation layers.
"""

from .rope import (
    RotaryEmbedding,
    rotate_half,
    apply_rotary_pos_emb,
    CosSin
)
from .attention import Attention
from .act import (
    HRMInnerCarry,
    HRMCarry,
    HRMConfig,
    HRMBlock,
    HRMReasoningModule
)

__all__ = [
    "RotaryEmbedding",
    "rotate_half", 
    "apply_rotary_pos_emb",
    "CosSin",
    "Attention",
    "HRMInnerCarry",
    "HRMCarry",
    "HRMConfig",
    "HRMBlock",
    "HRMReasoningModule"
]