"""
MLX HRM: Hierarchical Reasoning Model implementation in MLX.

This package provides an optimized implementation of the HRM architecture
for Apple Silicon using the MLX framework.
"""

__version__ = "0.1.0"

# Import key components for lower-level access
from . import layers
from . import modules
from . import models
from . import configs
from . import utils

# Import main user-facing APIs
from .models import (
    HRM,
    create_hrm,
    create_hrm_from_checkpoint,
    list_available_presets,
    get_model_info
)
from .configs import (
    HRM_TINY,
    HRM_SMALL,
    HRM_BASE,
    HRM_LARGE
)
from .utils import (
    save_checkpoint,
    load_checkpoint
)

__all__ = [
    # Modules
    "layers",
    "modules",
    "models",
    "configs",
    "utils",
    "__version__",
    # Main model class
    "HRM",
    # Factory functions
    "create_hrm",
    "create_hrm_from_checkpoint",
    "list_available_presets",
    "get_model_info",
    # Preset configs
    "HRM_TINY",
    "HRM_SMALL",
    "HRM_BASE", 
    "HRM_LARGE",
    # Checkpoint utilities
    "save_checkpoint",
    "load_checkpoint"
]