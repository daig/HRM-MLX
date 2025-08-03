"""Memory management utilities for efficient MLX HRM data loading."""

import mlx.core as mx
import psutil
import gc
from typing import Dict, Any, Optional, Tuple
from pathlib import Path
import json
import time


class MemoryMonitor:
    """Monitor memory usage during training for optimization."""
    
    def __init__(self, log_interval_batches: int = 100):
        """
        Initialize memory monitor.
        
        Args:
            log_interval_batches: Log memory stats every N batches
        """
        self.log_interval = log_interval_batches
        self.batch_count = 0
        self.memory_history = []
        self.peak_memory = 0
    
    def log_memory_usage(self, stage: str = "training") -> Dict[str, float]:
        """Log current memory usage."""
        process = psutil.Process()
        memory_info = process.memory_info()
        
        stats = {
            'stage': stage,
            'batch': self.batch_count,
            'rss_mb': memory_info.rss / 1024 / 1024,  # Resident Set Size
            'vms_mb': memory_info.vms / 1024 / 1024,  # Virtual Memory Size
            'percent': process.memory_percent(),
            'timestamp': time.time()
        }
        
        # Track peak memory
        if stats['rss_mb'] > self.peak_memory:
            self.peak_memory = stats['rss_mb']
        
        self.memory_history.append(stats)
        
        # Log periodically
        if self.batch_count % self.log_interval == 0:
            print(f"Memory [{stage}] Batch {self.batch_count}: "
                  f"RSS={stats['rss_mb']:.1f}MB, "
                  f"Peak={self.peak_memory:.1f}MB, "
                  f"CPU%={stats['percent']:.1f}%")
        
        return stats
    
    def on_batch_start(self):
        """Call at the start of each batch."""
        self.batch_count += 1
        return self.log_memory_usage("batch_start")
    
    def on_batch_end(self):
        """Call at the end of each batch."""
        return self.log_memory_usage("batch_end")
    
    def on_epoch_end(self):
        """Call at the end of each epoch."""
        # Force garbage collection
        gc.collect()
        mx.clear_cache()  # Clear MLX memory cache
        return self.log_memory_usage("epoch_end")
    
    def get_summary(self) -> Dict[str, Any]:
        """Get memory usage summary."""
        if not self.memory_history:
            return {}
        
        rss_values = [entry['rss_mb'] for entry in self.memory_history]
        return {
            'peak_memory_mb': self.peak_memory,
            'avg_memory_mb': sum(rss_values) / len(rss_values),
            'min_memory_mb': min(rss_values),
            'total_batches': self.batch_count,
            'memory_efficiency': self.peak_memory / max(rss_values) if rss_values else 0
        }


class LazyArrayLoader:
    """Lazy loading for large MLX arrays to optimize memory usage."""
    
    def __init__(self, data_path: Path, cache_size_mb: int = 512):
        """
        Initialize lazy array loader.
        
        Args:
            data_path: Path to data directory
            cache_size_mb: Maximum cache size in MB
        """
        self.data_path = data_path
        self.cache_size_mb = cache_size_mb
        self.cache = {}
        self.cache_sizes = {}
        self.access_times = {}
        self.current_cache_size = 0
    
    def _estimate_array_size_mb(self, array: mx.array) -> float:
        """Estimate memory size of MLX array in MB."""
        # MLX arrays: dtype size * number of elements
        dtype_sizes = {
            mx.int32: 4,
            mx.float32: 4,
            mx.int16: 2,
            mx.float16: 2,
            mx.int8: 1,
            mx.uint8: 1,
        }
        
        element_size = dtype_sizes.get(array.dtype, 4)  # Default to 4 bytes
        total_bytes = array.size * element_size
        return total_bytes / (1024 * 1024)
    
    def _evict_oldest(self):
        """Evict oldest cached items to make space."""
        if not self.cache:
            return
        
        # Sort by access time and remove oldest
        oldest_key = min(self.access_times.keys(), key=self.access_times.get)
        
        # Remove from cache
        evicted_size = self.cache_sizes.pop(oldest_key, 0)
        self.current_cache_size -= evicted_size
        del self.cache[oldest_key]
        del self.access_times[oldest_key]
        
        print(f"Evicted cache entry '{oldest_key}' ({evicted_size:.1f}MB)")
    
    def load_array(self, array_name: str, force_reload: bool = False) -> mx.array:
        """
        Load array with caching.
        
        Args:
            array_name: Name of array file (without .npy extension)
            force_reload: Force reload from disk
            
        Returns:
            Loaded MLX array
        """
        # Check cache first
        if array_name in self.cache and not force_reload:
            self.access_times[array_name] = time.time()
            return self.cache[array_name]
        
        # Load from disk
        array_path = self.data_path / f"{array_name}.npy"
        if not array_path.exists():
            raise FileNotFoundError(f"Array file not found: {array_path}")
        
        # Load with numpy first, then convert to MLX
        import numpy as np
        np_array = np.load(array_path)
        mlx_array = mx.array(np_array)
        
        # Estimate size
        array_size_mb = self._estimate_array_size_mb(mlx_array)
        
        # Check if we need to evict items
        while (self.current_cache_size + array_size_mb > self.cache_size_mb and 
               self.cache):
            self._evict_oldest()
        
        # Add to cache if it fits
        if array_size_mb <= self.cache_size_mb:
            self.cache[array_name] = mlx_array
            self.cache_sizes[array_name] = array_size_mb
            self.access_times[array_name] = time.time()
            self.current_cache_size += array_size_mb
            
            print(f"Cached '{array_name}' ({array_size_mb:.1f}MB), "
                  f"total cache: {self.current_cache_size:.1f}MB")
        else:
            print(f"Array '{array_name}' ({array_size_mb:.1f}MB) too large for cache")
        
        return mlx_array
    
    def clear_cache(self):
        """Clear all cached arrays."""
        self.cache.clear()
        self.cache_sizes.clear()
        self.access_times.clear()
        self.current_cache_size = 0
        gc.collect()
        mx.clear_cache()
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        return {
            'cached_arrays': len(self.cache),
            'cache_size_mb': self.current_cache_size,
            'cache_limit_mb': self.cache_size_mb,
            'cache_utilization': self.current_cache_size / self.cache_size_mb if self.cache_size_mb > 0 else 0,
            'cached_items': list(self.cache.keys())
        }


class BatchMemoryOptimizer:
    """Optimize memory usage during batch creation."""
    
    def __init__(self, target_memory_mb: int = 1024):
        """
        Initialize batch memory optimizer.
        
        Args:
            target_memory_mb: Target memory usage limit
        """
        self.target_memory_mb = target_memory_mb
        self.batch_size_history = []
        self.memory_usage_history = []
    
    def optimize_batch_size(self, base_batch_size: int, example_size_mb: float) -> int:
        """
        Optimize batch size based on memory constraints.
        
        Args:
            base_batch_size: Desired batch size
            example_size_mb: Average memory per example in MB
            
        Returns:
            Optimized batch size
        """
        # Estimate batch memory usage
        estimated_batch_memory = base_batch_size * example_size_mb
        
        # Check if batch fits in target memory
        if estimated_batch_memory <= self.target_memory_mb:
            return base_batch_size
        
        # Calculate maximum feasible batch size
        max_batch_size = int(self.target_memory_mb / example_size_mb)
        optimized_size = max(1, max_batch_size)  # At least 1
        
        if optimized_size < base_batch_size:
            print(f"Reducing batch size from {base_batch_size} to {optimized_size} "
                  f"to fit memory limit ({self.target_memory_mb}MB)")
        
        return optimized_size
    
    def create_efficient_batch(self, indices: mx.array, data_arrays: Dict[str, mx.array]) -> Dict[str, mx.array]:
        """
        Create batch with memory-efficient operations.
        
        Args:
            indices: Indices to sample
            data_arrays: Dictionary of data arrays
            
        Returns:
            Batch dictionary
        """
        batch = {}
        
        # Use MLX's efficient indexing
        for key, array in data_arrays.items():
            # Direct indexing is more memory efficient than gather operations
            batch[key] = array[indices]
        
        return batch
    
    def monitor_batch_creation(self, batch_size: int, memory_used_mb: float):
        """Monitor batch creation for optimization."""
        self.batch_size_history.append(batch_size)
        self.memory_usage_history.append(memory_used_mb)
        
        # Keep only recent history
        max_history = 100
        if len(self.batch_size_history) > max_history:
            self.batch_size_history = self.batch_size_history[-max_history:]
            self.memory_usage_history = self.memory_usage_history[-max_history:]
    
    def get_optimal_batch_size_recommendation(self) -> Optional[int]:
        """Get recommendation for optimal batch size based on history."""
        if len(self.batch_size_history) < 10:
            return None
        
        # Find batch sizes that used acceptable memory
        acceptable_memory = self.target_memory_mb * 0.8  # 80% of target
        
        good_batch_sizes = [
            bs for bs, mem in zip(self.batch_size_history, self.memory_usage_history)
            if mem <= acceptable_memory
        ]
        
        if good_batch_sizes:
            return max(good_batch_sizes)
        else:
            return min(self.batch_size_history)


def optimize_mlx_memory_usage():
    """Apply global MLX memory optimizations."""
    # Clear MLX memory cache
    mx.clear_cache()
    
    # Force garbage collection
    gc.collect()
    
    # Print current memory status
    process = psutil.Process()
    memory_mb = process.memory_info().rss / 1024 / 1024
    print(f"Memory optimization applied. Current usage: {memory_mb:.1f}MB")


def estimate_dataset_memory_requirements(dataset_path: Path) -> Dict[str, float]:
    """
    Estimate memory requirements for a dataset.
    
    Args:
        dataset_path: Path to dataset directory
        
    Returns:
        Dictionary with memory estimates
    """
    estimates = {
        'total_size_mb': 0,
        'input_ids_mb': 0,
        'labels_mb': 0,
        'puzzle_ids_mb': 0,
        'metadata_mb': 0
    }
    
    # Check for data files
    for data_file in dataset_path.glob("**/*.npy"):
        try:
            # Get file size
            file_size_mb = data_file.stat().st_size / (1024 * 1024)
            estimates['total_size_mb'] += file_size_mb
            
            # Categorize by content
            if 'input' in data_file.name:
                estimates['input_ids_mb'] += file_size_mb
            elif 'label' in data_file.name:
                estimates['labels_mb'] += file_size_mb
            elif 'puzzle' in data_file.name:
                estimates['puzzle_ids_mb'] += file_size_mb
        except Exception as e:
            print(f"Error estimating size for {data_file}: {e}")
    
    # Check for JSON metadata
    for json_file in dataset_path.glob("**/*.json"):
        try:
            file_size_mb = json_file.stat().st_size / (1024 * 1024)
            estimates['metadata_mb'] += file_size_mb
            estimates['total_size_mb'] += file_size_mb
        except Exception as e:
            print(f"Error estimating size for {json_file}: {e}")
    
    return estimates