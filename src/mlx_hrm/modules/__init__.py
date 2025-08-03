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

__all__ = [
    "RotaryEmbedding",
    "rotate_half", 
    "apply_rotary_pos_emb",
    "CosSin",
    "Attention"
]