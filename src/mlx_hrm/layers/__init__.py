"""
MLX HRM layers module.

This module provides custom layers and initialization functions for the HRM model.
"""

from .initialization import (
    truncated_normal,
    init_truncated_normal,
    LinearTruncNormal,
    EmbeddingTruncNormal,
)
from .normalization import (
    rms_norm,
    RMSNorm,
    RMSNormCompatible,
    create_rms_norm,
)
from .activations import (
    SwiGLU,
    silu,
    SwiGLUFactory,
)
from .embeddings import (
    CastedSparseEmbedding,
    SignSGD,
    SignSGDState,
    create_sparse_embedding_optimizer,
)

__all__ = [
    # Initialization
    "truncated_normal",
    "init_truncated_normal",
    "LinearTruncNormal",
    "EmbeddingTruncNormal",
    # Normalization
    "rms_norm",
    "RMSNorm",
    "RMSNormCompatible",
    "create_rms_norm",
    # Activations
    "SwiGLU",
    "silu",
    "SwiGLUFactory",
    # Embeddings
    "CastedSparseEmbedding",
    "SignSGD",
    "SignSGDState",
    "create_sparse_embedding_optimizer",
]