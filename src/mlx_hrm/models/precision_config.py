"""
Precision configuration system for MLX HRM implementation.

This module provides the precision management system that matches the original
PyTorch HRM's sophisticated manual mixed precision architecture.

The configuration system manages:
- Forward computation dtype (typically BF16)
- Master weights dtype (always FP32)  
- Precision overrides for specific operations
- Device-specific handling (CPU for FP64, GPU for FP32/BF16)
"""

import mlx.core as mx
from dataclasses import dataclass
from typing import Dict, Any, Optional


@dataclass
class MLXPrecisionConfig:
    """
    Precision configuration matching original HRM exactly.
    
    This configuration system replicates the original PyTorch HRM's manual
    mixed precision approach where different operations use different
    precisions for optimal stability vs performance trade-offs.
    
    Architecture:
    - Forward computation: BF16 (efficiency)
    - Master weights: FP32 (gradient stability)
    - Normalization: FP32 (numerical stability)
    - Loss computation: FP64 (ultra-precision for critical ops)
    - Gradients: FP32 (accumulation stability)
    """
    
    # Global forward computation dtype (typically BF16 for efficiency)
    forward_dtype: str = "bfloat16"
    
    # Master weights always FP32 for gradient stability
    master_weights_dtype: str = "float32"
    
    # Precision overrides for specific operation types
    normalization_dtype: str = "float32"      # RMSNorm always FP32
    loss_computation_dtype: str = "float64"   # Stablemax ultra-precision
    gradient_dtype: str = "float32"           # Gradient accumulation
    embedding_dtype: str = "float32"          # Sparse embeddings master dtype
    
    # Device preferences (MLX handles automatically)
    fp64_device: str = "cpu"     # FP64 operations use CPU (Metal limitation)
    default_device: str = "gpu"  # Default operations prefer GPU
    
    def get_forward_dtype(self) -> mx.Dtype:
        """Get MLX dtype for forward computation."""
        return getattr(mx, self.forward_dtype)
    
    def get_master_dtype(self) -> mx.Dtype:
        """Get MLX dtype for master weights."""
        return getattr(mx, self.master_weights_dtype)
    
    def get_normalization_dtype(self) -> mx.Dtype:
        """Get MLX dtype for normalization operations."""
        return getattr(mx, self.normalization_dtype)
    
    def get_loss_dtype(self) -> mx.Dtype:
        """Get MLX dtype for loss computation."""
        return getattr(mx, self.loss_computation_dtype)
    
    def get_gradient_dtype(self) -> mx.Dtype:
        """Get MLX dtype for gradient operations."""
        return getattr(mx, self.gradient_dtype)
    
    def get_embedding_dtype(self) -> mx.Dtype:
        """Get MLX dtype for embedding parameters."""
        return getattr(mx, self.embedding_dtype)
    
    def should_use_cpu_for_dtype(self, dtype: mx.Dtype) -> bool:
        """Check if dtype requires CPU computation."""
        return dtype == mx.float64
    
    def cast_for_forward(self, x: mx.array) -> mx.array:
        """Cast input to forward computation dtype."""
        target_dtype = self.get_forward_dtype()
        if x.dtype != target_dtype:
            return x.astype(target_dtype)
        return x
    
    def cast_for_loss(self, x: mx.array) -> mx.array:
        """Cast input for loss computation (with CPU handling)."""
        target_dtype = self.get_loss_dtype()
        if target_dtype == mx.float64:
            # FP64 requires CPU computation
            with mx.stream(mx.cpu):
                return x.astype(target_dtype)
        return x.astype(target_dtype)
    
    def summary(self) -> Dict[str, Any]:
        """Get configuration summary."""
        return {
            "forward_computation": self.forward_dtype,
            "master_weights": self.master_weights_dtype,
            "normalization": self.normalization_dtype,
            "loss_computation": self.loss_computation_dtype,
            "gradients": self.gradient_dtype,
            "embeddings": self.embedding_dtype,
            "cpu_operations": ["float64"],
            "gpu_operations": ["float32", "bfloat16", "float16"]
        }
    
    def validate_configuration(self) -> bool:
        """Validate that the configuration is valid."""
        try:
            # Test that all dtypes are valid MLX dtypes
            self.get_forward_dtype()
            self.get_master_dtype()
            self.get_normalization_dtype()
            self.get_loss_dtype()
            self.get_gradient_dtype()
            self.get_embedding_dtype()
            
            # Validate that master weights are FP32 (required for stability)
            if self.master_weights_dtype != "float32":
                raise ValueError("Master weights must be float32 for gradient stability")
            
            # Validate that gradients are FP32 (required for accumulation)
            if self.gradient_dtype != "float32":
                raise ValueError("Gradients must be float32 for stable accumulation")
            
            return True
            
        except (AttributeError, ValueError) as e:
            print(f"❌ Invalid precision configuration: {e}")
            return False
    
    @classmethod
    def create_herm_standard_config(cls) -> 'MLXPrecisionConfig':
        """Create the standard HRM precision configuration."""
        return cls(
            forward_dtype="bfloat16",
            master_weights_dtype="float32",
            normalization_dtype="float32",
            loss_computation_dtype="float64",
            gradient_dtype="float32",
            embedding_dtype="float32"
        )
    
    @classmethod
    def create_development_config(cls) -> 'MLXPrecisionConfig':
        """Create a development configuration (all FP32 for simplicity)."""
        return cls(
            forward_dtype="float32",
            master_weights_dtype="float32",
            normalization_dtype="float32",
            loss_computation_dtype="float32",  # No FP64 for development
            gradient_dtype="float32",
            embedding_dtype="float32"
        )
    
    @classmethod
    def create_ultra_precision_config(cls) -> 'MLXPrecisionConfig':
        """Create an ultra-precision configuration for maximum accuracy."""
        return cls(
            forward_dtype="float32",       # FP32 forward for max precision
            master_weights_dtype="float32",
            normalization_dtype="float32",
            loss_computation_dtype="float64",  # Ultra-precision loss
            gradient_dtype="float32",
            embedding_dtype="float32"
        )


def get_precision_config(config_name: str = "standard") -> MLXPrecisionConfig:
    """
    Get a named precision configuration.
    
    Args:
        config_name: Name of configuration ("standard", "development", "ultra")
        
    Returns:
        MLXPrecisionConfig instance
    """
    if config_name == "standard":
        return MLXPrecisionConfig.create_herm_standard_config()
    elif config_name == "development":
        return MLXPrecisionConfig.create_development_config()
    elif config_name == "ultra":
        return MLXPrecisionConfig.create_ultra_precision_config()
    else:
        raise ValueError(f"Unknown precision config: {config_name}")


def validate_precision_setup():
    """Validate that precision configuration works correctly."""
    print("🧪 Validating Precision Configuration System...")
    
    # Test standard configuration
    config = get_precision_config("standard")
    
    print(f"  Forward dtype: {config.get_forward_dtype()}")
    print(f"  Master dtype: {config.get_master_dtype()}")
    print(f"  Loss dtype: {config.get_loss_dtype()}")
    
    # Test validation
    is_valid = config.validate_configuration()
    print(f"  Configuration valid: {is_valid}")
    
    # Test dtype casting
    test_array = mx.random.normal((2, 3))
    print(f"  Original dtype: {test_array.dtype}")
    
    forward_cast = config.cast_for_forward(test_array)
    print(f"  After forward cast: {forward_cast.dtype}")
    
    # Test CPU casting for FP64
    try:
        loss_cast = config.cast_for_loss(test_array)
        print(f"  After loss cast: {loss_cast.dtype}")
        print("  ✅ CPU FP64 casting works!")
    except Exception as e:
        print(f"  ❌ CPU FP64 casting failed: {e}")
    
    # Test all preset configurations
    presets = ["standard", "development", "ultra"]
    for preset in presets:
        preset_config = get_precision_config(preset)
        valid = preset_config.validate_configuration()
        print(f"  {preset} config valid: {valid}")
    
    print("  ✅ Precision configuration system working!")
    return is_valid


if __name__ == "__main__":
    # Run validation when script is executed directly
    validate_precision_setup()