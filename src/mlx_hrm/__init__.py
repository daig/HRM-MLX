"""
MLX HRM: Hierarchical Reasoning Model implementation in MLX.

This package provides an optimized implementation of the HRM architecture
for Apple Silicon using the MLX framework.
"""

__version__ = "0.1.0"

# Import key components
from . import layers
from . import modules

__all__ = [
    "layers",
    "modules",
    "__version__",
]