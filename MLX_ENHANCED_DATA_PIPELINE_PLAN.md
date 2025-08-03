# Enhanced Data Pipeline Implementation Plan

## Overview
This document details the implementation plan for enhancing the MLX HRM data pipeline to match the sophistication of the original PyTorch implementation while maintaining the simplified memory model (no memory mapping needed for ~4GB datasets).

## Features to Implement

### Feature #3: Smart Batching Strategy
**Goal**: Group similar puzzles together in batches for better gradient signals and training efficiency.

#### Current State
- Simple random/sequential batching
- No puzzle grouping or intelligent sampling
- Batches may contain completely unrelated puzzle types

#### Target Implementation
```python
class SmartBatchSampler:
    """
    Intelligent batching that groups similar puzzles together.
    
    Key concepts:
    - Groups: Collections of similar puzzles (e.g., same difficulty, type)
    - Puzzles: Individual puzzle instances within groups
    - Examples: Multiple training examples per puzzle (with augmentations)
    """
    
    def __init__(
        self,
        puzzle_indices: np.ndarray,    # Start index of each puzzle
        group_indices: np.ndarray,     # Start index of each group
        batch_size: int,
        shuffle_groups: bool = True
    ):
        self.puzzle_indices = puzzle_indices
        self.group_indices = group_indices
        self.batch_size = batch_size
        self.shuffle_groups = shuffle_groups
    
    def sample_batch(self, rng: np.random.Generator) -> Tuple[np.ndarray, np.ndarray]:
        """
        Sample a batch using smart grouping strategy.
        
        Algorithm:
        1. Select a random group (or next group if sequential)
        2. Select a random puzzle from that group
        3. Sample examples from that puzzle to fill batch
        4. If batch not full, move to next group and repeat
        
        Returns:
            - example_indices: Indices of examples to include
            - puzzle_ids: Puzzle ID for each example
        """
```

#### Implementation Details
- **Group Management**: Track which puzzles belong to which groups
- **Sampling Strategy**: Prefer examples from same group/puzzle type
- **Batch Filling**: Intelligently fill batches without mixing unrelated content
- **Random State**: Proper seeding for reproducibility

#### Files to Modify
- `src/mlx_hrm/data/dataset.py`: Add SmartBatchSampler class
- `src/mlx_hrm/data/dataset.py`: Modify DataLoader to use smart sampling

---

### Feature #4: Epochs Batching
**Goal**: Batch multiple epochs together to reduce dataloader overhead and improve training efficiency.

#### Current State
- Processes one epoch at a time
- Higher overhead from frequent epoch boundaries
- Dataloader recreation overhead

#### Target Implementation
```python
class EpochBatchingIterator:
    """
    Iterator that batches multiple epochs together for efficiency.
    
    Key benefits:
    - Reduces Python overhead by processing multiple epochs at once
    - Amortizes shuffling costs across multiple epochs
    - Better GPU utilization by reducing idle time
    """
    
    def __init__(
        self,
        dataset: PuzzleDataset,
        batch_size: int,
        epochs_per_iter: int = 10,    # Batch 10 epochs together
        shuffle: bool = True
    ):
        self.dataset = dataset
        self.batch_size = batch_size
        self.epochs_per_iter = epochs_per_iter
        self.shuffle = shuffle
    
    def __iter__(self):
        """
        Yield batches across multiple epochs.
        
        Algorithm:
        1. Create shuffled order for epochs_per_iter epochs
        2. Sample batches from this extended sequence
        3. Track epoch boundaries for metrics/logging
        4. Reset when all epochs processed
        """
```

#### Implementation Details
- **Multi-Epoch Shuffling**: Create extended shuffle order across multiple epochs
- **Epoch Tracking**: Track which epoch each batch belongs to
- **Memory Efficiency**: Don't duplicate data, just extend indices
- **Configurable Batching**: Allow tuning of epochs_per_iter

#### Files to Modify
- `src/mlx_hrm/data/dataset.py`: Add EpochBatchingIterator
- `src/mlx_hrm/training/trainer.py`: Integrate with training loop

---

### Feature #5: Sophisticated Sampling
**Goal**: Implement the advanced `_sample_batch()` function that intelligently samples from puzzle groups.

#### Current State
- Simple random sampling within dataset
- No awareness of puzzle structure or relationships
- May oversample/undersample certain puzzle types

#### Target Implementation
```python
def sample_batch_advanced(
    rng: np.random.Generator,
    group_order: np.ndarray,        # Shuffled order of groups to process
    puzzle_indices: np.ndarray,     # Start indices for each puzzle
    group_indices: np.ndarray,      # Start indices for each group
    start_index: int,               # Current position in group_order
    global_batch_size: int          # Target batch size
) -> Tuple[int, np.ndarray, np.ndarray]:
    """
    Advanced batch sampling with intelligent puzzle selection.
    
    Algorithm:
    1. Process groups in shuffled order (for variety)
    2. For each group, randomly select a puzzle
    3. From that puzzle, sample examples without replacement
    4. Continue until batch is full
    5. Track puzzle IDs for each example
    
    Key features:
    - Ensures diversity across puzzle types
    - Respects puzzle boundaries (don't mix puzzle instances)
    - Efficient sampling without replacement
    - Handles variable puzzle sizes gracefully
    
    Returns:
        - new_start_index: Updated position in group order
        - example_indices: Selected example indices
        - puzzle_ids: Puzzle ID for each example
    """
```

#### Implementation Details
- **Group-First Sampling**: Always start with group selection for diversity
- **Puzzle-Level Coherence**: Keep examples from same puzzle together when possible
- **Efficient Random Selection**: Use numpy's choice() for performance
- **Boundary Handling**: Properly handle puzzle and group boundaries

#### Files to Modify
- `src/mlx_hrm/data/dataset.py`: Implement advanced sampling function
- `src/mlx_hrm/data/dataset.py`: Integrate with SmartBatchSampler

---

### Feature #6: Test vs Train Mode
**Goal**: Implement distinct iteration strategies for training (shuffled) vs evaluation (sequential).

#### Current State
- Same iteration logic for both training and evaluation
- No deterministic evaluation mode
- Can't reproduce evaluation results reliably

#### Target Implementation
```python
class PuzzleDataset:
    def __init__(self, ..., mode: str = 'train'):
        self.mode = mode  # 'train' or 'test'
    
    def _iter_train(self):
        """
        Training iteration: shuffled, grouped sampling.
        
        Features:
        - Shuffles groups each epoch for variety
        - Uses smart batching for related content
        - Maximizes training signal through diversity
        """
        
    def _iter_test(self):
        """
        Test iteration: sequential, deterministic.
        
        Features:
        - Processes examples in fixed order
        - Ensures reproducible evaluation results
        - Processes all examples exactly once
        - Suitable for validation and testing
        """
    
    def __iter__(self):
        if self.mode == 'train':
            return self._iter_train()
        else:
            return self._iter_test()
```

#### Implementation Details
- **Deterministic Test Mode**: Fixed iteration order for reproducibility
- **Comprehensive Coverage**: Ensure all examples processed in test mode
- **Mode Switching**: Easy to switch between modes for same dataset
- **Validation Integration**: Seamless integration with validation loops

#### Files to Modify
- `src/mlx_hrm/data/dataset.py`: Add mode parameter and separate iterators
- `src/mlx_hrm/training/trainer.py`: Use appropriate modes for train/val

---

### Feature #7: Worker Management
**Goal**: Implement multi-worker data loading for improved performance and reduced bottlenecks.

#### Current State
- Single-threaded data loading
- Data loading can become bottleneck during training
- No parallelization of data preprocessing

#### Target Implementation
```python
class MultiWorkerDataLoader:
    """
    Multi-worker data loader for parallel data processing.
    
    Key features:
    - Parallel data loading across multiple workers
    - Proper worker initialization and cleanup
    - Load balancing across workers
    - Prefetching for continuous GPU feeding
    """
    
    def __init__(
        self,
        dataset: PuzzleDataset,
        batch_size: int,
        num_workers: int = 4,          # Number of worker processes
        prefetch_factor: int = 2,      # Batches to prefetch per worker
        worker_init_fn: Optional[callable] = None
    ):
        self.dataset = dataset
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.prefetch_factor = prefetch_factor
        self.worker_init_fn = worker_init_fn
    
    def _worker_loop(self, worker_id: int, start_idx: int, end_idx: int):
        """
        Worker process loop for parallel data loading.
        
        Each worker:
        1. Handles a subset of the dataset
        2. Applies transformations/augmentations
        3. Sends batches to main process via queue
        4. Manages its own random state
        """
```

#### Implementation Details
- **Process Management**: Spawn/manage worker processes safely
- **Data Partitioning**: Divide dataset across workers efficiently
- **Communication**: Use queues for inter-process communication
- **Random State**: Ensure each worker has independent random state
- **Error Handling**: Graceful handling of worker failures
- **Resource Cleanup**: Proper cleanup on shutdown

#### MLX Considerations
- **MLX Arrays**: Handle MLX array serialization across processes
- **GPU Memory**: Ensure workers don't compete for GPU memory
- **Device Placement**: Proper device management with multiple workers

#### Files to Modify
- `src/mlx_hrm/data/dataset.py`: Add MultiWorkerDataLoader class
- `src/mlx_hrm/data/dataset.py`: Add worker management utilities

---

## Implementation Priority

### Phase 1: Core Intelligence (High Priority)
1. **Smart Batching Strategy (#3)** - Critical for training quality
2. **Sophisticated Sampling (#5)** - Core algorithm for batch quality
3. **Test vs Train Mode (#6)** - Essential for proper evaluation

### Phase 2: Performance Optimization (Medium Priority)
4. **Epochs Batching (#4)** - Training efficiency improvement
5. **Worker Management (#7)** - Performance scaling

## Integration Plan

### Step 1: Enhance Dataset Class
- Add group and puzzle index support
- Implement smart batching logic
- Add train/test mode switching

### Step 2: Enhance DataLoader
- Integrate smart batching
- Add epoch batching option
- Prepare for multi-worker support

### Step 3: Update Trainer
- Use appropriate modes for train/validation
- Integrate with enhanced data loading
- Add performance monitoring

### Step 4: Testing & Validation
- Unit tests for each component
- Integration tests with full training
- Performance benchmarking vs simple version

## Expected Benefits

### Training Quality
- **Better Convergence**: Smart batching improves gradient signals
- **Stable Training**: Consistent puzzle grouping reduces variance
- **Reproducible Results**: Deterministic test mode ensures consistency

### Performance
- **Reduced Overhead**: Epoch batching amortizes costs
- **Parallel Loading**: Multi-worker prevents data bottlenecks
- **Efficient Sampling**: Advanced algorithms reduce wasted computation

### Maintainability
- **Clear Separation**: Distinct train/test modes
- **Modular Design**: Each feature implemented as separate component
- **Easy Configuration**: Factory functions for common setups

## Success Criteria

1. **Functional Parity**: All original PyTorch features working in MLX
2. **Performance**: No data loading bottlenecks during training
3. **Reproducibility**: Identical results across runs in test mode
4. **Integration**: Seamless integration with existing trainer
5. **Testing**: Comprehensive test coverage for all components

## Timeline Estimate

- **Smart Batching & Sampling**: 1-2 days
- **Train/Test Modes**: 0.5-1 day  
- **Epochs Batching**: 0.5-1 day
- **Multi-Worker Loading**: 1-2 days
- **Testing & Integration**: 1 day

**Total**: 4-7 days for full implementation