"""Data loading utilities for HRM training."""

import mlx.core as mx
from typing import Dict, List, Optional, Iterator, Union, Tuple, Any
from pathlib import Path
import json
import random

# Import IGNORE_LABEL_ID from training losses
from ..training.losses import IGNORE_LABEL_ID


class PuzzleDatasetMetadata:
    """Metadata for puzzle datasets."""
    
    def __init__(
        self,
        pad_id: int,
        ignore_label_id: Optional[int],
        blank_identifier_id: int,
        vocab_size: int,
        seq_len: int,
        num_puzzle_identifiers: int,
        total_groups: int,
        mean_puzzle_examples: float,
        sets: List[str]
    ):
        self.pad_id = pad_id
        self.ignore_label_id = ignore_label_id
        self.blank_identifier_id = blank_identifier_id
        self.vocab_size = vocab_size
        self.seq_len = seq_len
        self.num_puzzle_identifiers = num_puzzle_identifiers
        self.total_groups = total_groups
        self.mean_puzzle_examples = mean_puzzle_examples
        self.sets = sets


def _sample_batch_smart(
    rng: random.Random, 
    group_order: mx.array, 
    puzzle_indices: mx.array, 
    group_indices: mx.array, 
    start_index: int, 
    global_batch_size: int
) -> Tuple[int, mx.array, mx.array]:
    """
    Sample a batch of examples from puzzle groups using smart strategy.
    
    This implements the exact PyTorch HRM batching strategy:
    1. Groups contain puzzles of similar difficulty/type
    2. We sample one group at a time
    3. From each group, we randomly select a puzzle
    4. From that puzzle, we sample examples to fill the batch
    
    This ensures batches contain related examples for better gradient signals.
    
    Args:
        rng: Random number generator for reproducibility
        group_order: Shuffled order of groups to process
        puzzle_indices: Start indices for each puzzle in the dataset
        group_indices: Start indices for each group of puzzles
        start_index: Current position in group_order
        global_batch_size: Target batch size
        
    Returns:
        Tuple of (new_start_index, example_indices, puzzle_ids)
    """
    batch = []              # Indices of examples to include
    batch_puzzle_indices = []  # Puzzle ID for each example
    current_size = 0

    while (start_index < group_order.shape[0]) and (current_size < global_batch_size):
        # Pick next group from shuffled order
        group_id = int(group_order[start_index].item())
        
        # Randomly select a puzzle from this group
        group_start = int(group_indices[group_id].item())
        group_end = int(group_indices[group_id + 1].item())
        puzzle_id = rng.randint(group_start, group_end - 1)
        start_index += 1

        # Get the range of examples for this puzzle
        puzzle_start = int(puzzle_indices[puzzle_id].item())
        puzzle_size = int(puzzle_indices[puzzle_id + 1].item()) - puzzle_start

        # Determine how many examples to take from this puzzle
        append_size = min(puzzle_size, global_batch_size - current_size)

        # Randomly sample examples from the puzzle (without replacement)
        batch_puzzle_indices.extend([puzzle_id] * append_size)
        
        # Sample indices without replacement
        if append_size == puzzle_size:
            # Take all examples
            sampled_indices = list(range(puzzle_start, puzzle_start + puzzle_size))
        else:
            # Sample without replacement
            available_indices = list(range(puzzle_start, puzzle_start + puzzle_size))
            sampled_indices = rng.sample(available_indices, append_size)
        
        batch.extend(sampled_indices)
        current_size += append_size

    return start_index, mx.array(batch), mx.array(batch_puzzle_indices)


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
        puzzle_indices: mx.array,    # Start index of each puzzle
        group_indices: mx.array,     # Start index of each group
        batch_size: int,
        shuffle_groups: bool = True,
        seed: int = 42
    ):
        """
        Initialize smart batch sampler.
        
        Args:
            puzzle_indices: Start indices for each puzzle in the dataset
            group_indices: Start indices for each group of puzzles
            batch_size: Target batch size
            shuffle_groups: Whether to shuffle group order
            seed: Random seed for reproducibility
        """
        self.puzzle_indices = puzzle_indices
        self.group_indices = group_indices
        self.batch_size = batch_size
        self.shuffle_groups = shuffle_groups
        self.seed = seed
        
        # State for iteration
        self.current_epoch = 0
        self.group_order = None
        self.start_index = 0
        
        # Initialize RNG
        self.rng = random.Random(seed)
    
    def _create_group_order(self, epochs_per_iter: int = 1) -> mx.array:
        """Create shuffled order of groups for multiple epochs."""
        num_groups = self.group_indices.shape[0] - 1
        
        if self.shuffle_groups:
            # Create shuffled order for each epoch
            group_orders = []
            for _ in range(epochs_per_iter):
                order = list(range(num_groups))
                self.rng.shuffle(order)
                group_orders.extend(order)
            return mx.array(group_orders)
        else:
            # Sequential order
            return mx.array(list(range(num_groups)) * epochs_per_iter)
    
    def start_epoch(self, epochs_per_iter: int = 1):
        """Start a new epoch with fresh group ordering."""
        self.current_epoch += 1
        
        # Update RNG seed for this epoch
        self.rng = random.Random(self.seed + self.current_epoch)
        
        # Create new group order
        self.group_order = self._create_group_order(epochs_per_iter)
        self.start_index = 0
    
    def sample_batch(self) -> Optional[Tuple[mx.array, mx.array]]:
        """
        Sample a batch using smart grouping strategy.
        
        Returns:
            Tuple of (example_indices, puzzle_ids) or None if epoch finished
        """
        if self.group_order is None or self.start_index >= self.group_order.size:
            return None
        
        # Sample batch using smart strategy
        new_start_index, example_indices, puzzle_ids = _sample_batch_smart(
            self.rng,
            group_order=self.group_order,
            puzzle_indices=self.puzzle_indices,
            group_indices=self.group_indices,
            start_index=self.start_index,
            global_batch_size=self.batch_size
        )
        
        # Update state
        self.start_index = new_start_index
        
        # Skip incomplete batches for training stability
        if len(example_indices) < self.batch_size:
            return None
        
        return example_indices, puzzle_ids
    
    def has_batches_remaining(self) -> bool:
        """Check if there are more batches in current epoch."""
        return (self.group_order is not None and 
                self.start_index < self.group_order.shape[0])


class EnhancedPuzzleDataset:
    """
    Enhanced dataset with smart batching capabilities.
    
    Supports the sophisticated batching strategy from the original PyTorch HRM:
    - Groups similar puzzles together for better training
    - Smart sampling within groups and puzzles
    - Proper train/test mode distinction
    """
    
    def __init__(
        self,
        data_path: str,
        split: str = "train",
        max_seq_len: int = 512,
        seed: int = 42,
        mode: str = "train"  # "train" or "test"
    ):
        """
        Initialize enhanced puzzle dataset.
        
        Args:
            data_path: Path to dataset directory
            split: Dataset split ('train', 'val', 'test')
            max_seq_len: Maximum sequence length
            seed: Random seed for reproducibility
            mode: Iteration mode ("train" for shuffled, "test" for sequential)
        """
        self.data_path = Path(data_path)
        self.split = split
        self.max_seq_len = max_seq_len
        self.seed = seed
        self.mode = mode
        
        # Set random seed
        random.seed(seed)
        
        # Load metadata and data
        self.metadata = self._load_metadata()
        self.examples = self._load_data()
        
        # Convert to MLX arrays for efficient processing
        self._convert_to_mlx_arrays()
        
        # Initialize smart batching structures
        self._init_smart_batching()
        
        print(f"Loaded {self.num_examples} examples from {self.data_path}")
        print(f"Smart batching: {len(self.puzzle_indices)-1} puzzles, {len(self.group_indices)-1} groups")
    
    def _init_smart_batching(self):
        """Initialize puzzle and group indices for smart batching."""
        # Create puzzle indices (start index of each puzzle's examples)
        puzzle_to_examples = {}
        
        for i in range(self.num_examples):
            puzzle_id = int(self.puzzle_ids[i].item())
            if puzzle_id not in puzzle_to_examples:
                puzzle_to_examples[puzzle_id] = []
            puzzle_to_examples[puzzle_id].append(i)
        
        # Sort puzzles by ID for consistency
        sorted_puzzles = sorted(puzzle_to_examples.keys())
        
        # Build puzzle_indices array
        puzzle_indices = [0]
        current_index = 0
        
        for puzzle_id in sorted_puzzles:
            current_index += len(puzzle_to_examples[puzzle_id])
            puzzle_indices.append(current_index)
        
        self.puzzle_indices = mx.array(puzzle_indices)
        
        # Create simple grouping strategy (can be enhanced later)
        # For now, group puzzles by type (based on puzzle_id ranges)
        self.group_indices = self._create_simple_groups(sorted_puzzles)
        
        # Store puzzle ID mapping for batch creation
        self.puzzle_id_to_index = {pid: i for i, pid in enumerate(sorted_puzzles)}
    
    def _create_simple_groups(self, sorted_puzzles: List[int]) -> mx.array:
        """Create simple grouping based on puzzle ID ranges."""
        # Group puzzles by ID ranges (rough type classification)
        # ARC: 0-999, Sudoku: 2000-2999, Maze: 3000-3999, etc.
        groups = {}
        
        for i, puzzle_id in enumerate(sorted_puzzles):
            # Determine group based on puzzle ID
            if puzzle_id < 1000:
                group_type = 0  # ARC
            elif 2000 <= puzzle_id < 3000:
                group_type = 1  # Sudoku
            elif 3000 <= puzzle_id < 4000:
                group_type = 2  # Maze
            else:
                group_type = 3  # Other
            
            if group_type not in groups:
                groups[group_type] = []
            groups[group_type].append(i)
        
        # Build group_indices array
        group_indices = [0]
        current_index = 0
        
        for group_type in sorted(groups.keys()):
            current_index += len(groups[group_type])
            group_indices.append(current_index)
        return mx.array(group_indices)
    
    def _convert_to_mlx_arrays(self):
        """Convert loaded examples to efficient MLX array format."""
        if not self.examples:
            self.num_examples = 0
            return
        
        # Pre-allocate arrays for better efficiency
        self.num_examples = len(self.examples)
        max_len = self.max_seq_len
        
        # Extract data into separate arrays
        input_ids_list = []
        labels_list = []
        puzzle_ids_list = []
        
        for example in self.examples:
            # Pad sequences to max length
            input_ids = self._pad_sequence(example['input_ids'], max_len)
            labels = self._pad_sequence(example['labels'], max_len, IGNORE_LABEL_ID)
            
            input_ids_list.append(input_ids)
            labels_list.append(labels)
            puzzle_ids_list.append(example.get('puzzle_id', 0))
        
        # Convert to MLX arrays
        self.input_ids = mx.array(input_ids_list)
        self.labels = mx.array(labels_list)
        self.puzzle_ids = mx.array(puzzle_ids_list)
        
        # Clear the original examples list to save memory
        self.examples = None
    
    def _extract_puzzle_id(self, task_id: str) -> int:
        """Extract puzzle ID from task name."""
        if task_id.startswith('task_'):
            try:
                return int(task_id.split('_')[1])
            except (IndexError, ValueError):
                pass
        # Fallback to hash for non-standard task names
        return hash(task_id) % 1000
    
    def _load_metadata(self) -> Optional[PuzzleDatasetMetadata]:
        """Load dataset metadata from JSON file."""
        metadata_path = self.data_path / self.split / "dataset.json"
        
        if not metadata_path.exists():
            # Fallback for simple formats without metadata
            print(f"No metadata found at {metadata_path}, using defaults")
            return None
        
        with open(metadata_path, 'r') as f:
            data = json.load(f)
            return PuzzleDatasetMetadata(**data)
    
    def _load_data(self) -> List[Dict[str, Any]]:
        """Load puzzle data from disk."""
        examples = []
        
        # Check for pre-processed format (like original HRM)
        if self._load_preprocessed_data(examples):
            return examples
        
        # Fallback: Try to load from common formats
        self._load_arc_format(examples)
        self._load_sudoku_format(examples)
        self._load_maze_format(examples)
        
        if not examples:
            raise ValueError(f"No data found in {self.data_path}")
        
        return examples
    
    def _load_preprocessed_data(self, examples: List[Dict]) -> bool:
        """Load pre-processed HRM format data."""
        data_dir = self.data_path / self.split
        if not data_dir.exists():
            return False
        
        # Look for HRM-style preprocessed files
        input_files = list(data_dir.glob("*__inputs.npy"))
        if not input_files:
            return False
        
        for input_file in input_files:
            set_name = input_file.name.replace("__inputs.npy", "")
            
            # Load corresponding files
            labels_file = data_dir / f"{set_name}__labels.npy"
            puzzle_ids_file = data_dir / f"{set_name}__puzzle_identifiers.npy"
            
            if not all(f.exists() for f in [labels_file, puzzle_ids_file]):
                continue
            
            # Load data (use numpy for loading, convert to lists)
            try:
                import numpy as np
                inputs = np.load(input_file)
                labels = np.load(labels_file)
                puzzle_ids = np.load(puzzle_ids_file)
            except ImportError:
                print("Warning: numpy not available, skipping preprocessed data loading")
                continue
            
            # Convert to examples
            for i in range(len(inputs)):
                examples.append({
                    'input_ids': inputs[i].tolist(),
                    'labels': labels[i].tolist(),
                    'puzzle_id': int(puzzle_ids[i]) if len(puzzle_ids.shape) == 1 else int(puzzle_ids[i])
                })
        
        return len(examples) > 0
    
    def _load_arc_format(self, examples: List[Dict]):
        """Load ARC format data."""
        arc_files = list(self.data_path.glob("**/*.json"))
        
        for json_file in arc_files:
            if self.split not in json_file.name:
                continue
            
            try:
                with open(json_file, 'r') as f:
                    data = json.load(f)
                
                # Process ARC tasks
                for task_id, task in data.items():
                    if isinstance(task, dict) and 'train' in task:
                        # Standard ARC format
                        for example in task.get('train', []) + task.get('test', []):
                            input_grid = example['input']
                            output_grid = example['output']
                            
                            # Convert grids to token sequences
                            input_tokens = self._grid_to_tokens(input_grid)
                            output_tokens = self._grid_to_tokens(output_grid)
                            
                            # Combine input and output (teacher forcing)
                            combined = input_tokens + [self._get_separator_token()] + output_tokens
                            
                            if len(combined) <= self.max_seq_len:
                                examples.append({
                                    'input_ids': combined[:-1],  # Input without last token
                                    'labels': combined[1:],      # Output shifted by 1
                                    'puzzle_id': self._extract_puzzle_id(task_id)
                                })
            except Exception as e:
                print(f"Failed to load {json_file}: {e}")
    
    def _load_sudoku_format(self, examples: List[Dict]):
        """Load Sudoku format data."""
        sudoku_files = list(self.data_path.glob("**/*.txt"))
        
        for txt_file in sudoku_files:
            if 'sudoku' not in txt_file.name.lower():
                continue
            
            try:
                with open(txt_file, 'r') as f:
                    for line_num, line in enumerate(f):
                        line = line.strip()
                        if not line or line.startswith('#'):
                            continue
                        
                        # Parse sudoku line (format: puzzle,solution)
                        if ',' in line:
                            puzzle, solution = line.split(',', 1)
                            
                            # Convert to tokens
                            puzzle_tokens = self._sudoku_to_tokens(puzzle)
                            solution_tokens = self._sudoku_to_tokens(solution)
                            
                            # Combine
                            combined = puzzle_tokens + [self._get_separator_token()] + solution_tokens
                            
                            if len(combined) <= self.max_seq_len:
                                examples.append({
                                    'input_ids': combined[:-1],
                                    'labels': combined[1:],
                                    'puzzle_id': 2000 + (line_num % 1000)  # Sudoku puzzle IDs
                                })
            except Exception as e:
                print(f"Failed to load {txt_file}: {e}")
    
    def _load_maze_format(self, examples: List[Dict]):
        """Load maze format data."""
        maze_files = list(self.data_path.glob("**/*.npz"))
        
        for npz_file in maze_files:
            if 'maze' not in npz_file.name.lower():
                continue
            
            try:
                try:
                    import numpy as np
                    data = np.load(npz_file)
                except ImportError:
                    print("Warning: numpy not available, skipping npz data loading")
                    continue
                
                if 'inputs' in data and 'labels' in data:
                    inputs = data['inputs']
                    labels = data['labels']
                    
                    for i in range(len(inputs)):
                        if len(inputs[i]) <= self.max_seq_len:
                            examples.append({
                                'input_ids': inputs[i].tolist(),
                                'labels': labels[i].tolist(),
                                'puzzle_id': 3000 + i  # Maze puzzle IDs
                            })
            except Exception as e:
                print(f"Failed to load {npz_file}: {e}")
    
    def _grid_to_tokens(self, grid: List[List[int]]) -> List[int]:
        """Convert 2D grid to token sequence."""
        tokens = []
        
        # Add beginning of grid token
        tokens.append(self._get_special_token('BOG'))
        
        for row in grid:
            # Add row tokens
            tokens.extend([min(val, 255) for val in row])  # Clamp values
            # Add end of line token
            tokens.append(self._get_special_token('EOL'))
        
        # Add end of grid token
        tokens.append(self._get_special_token('EOG'))
        
        return tokens
    
    def _sudoku_to_tokens(self, sudoku_str: str) -> List[int]:
        """Convert Sudoku string to tokens."""
        tokens = []
        
        for char in sudoku_str:
            if char.isdigit():
                # Digits 1-9 map to tokens 1-9, 0 (empty) maps to token 0
                tokens.append(int(char))
            elif char == '.':
                # Empty cell
                tokens.append(0)
        
        return tokens
    
    def _get_special_token(self, name: str) -> int:
        """Get special token ID."""
        # Simple mapping for special tokens
        special_tokens = {
            'PAD': 256,
            'BOG': 257,  # Beginning of grid
            'EOG': 258,  # End of grid
            'EOL': 259,  # End of line
            'SEP': 260,  # Separator
        }
        return special_tokens.get(name, 256)
    
    def _get_separator_token(self) -> int:
        """Get separator token between input and output."""
        return self._get_special_token('SEP')
    
    def _pad_sequence(self, seq: List[int], max_len: int, pad_value: int = None) -> List[int]:
        """Pad sequence to maximum length."""
        if pad_value is None:
            pad_value = self._get_special_token('PAD')
        
        if len(seq) > max_len:
            seq = seq[:max_len]
        else:
            seq = seq + [pad_value] * (max_len - len(seq))
        return seq
    
    def __len__(self) -> int:
        """Get dataset size."""
        return self.num_examples
    
    def __getitem__(self, idx: int) -> Dict[str, mx.array]:
        """Get a single example."""
        return {
            'input_ids': self.input_ids[idx],
            'labels': self.labels[idx],
            'puzzle_id': self.puzzle_ids[idx]
        }
    
    def create_smart_dataloader(self, batch_size: int, epochs_per_iter: int = 1) -> 'SmartDataLoader':
        """Create a smart data loader for this dataset."""
        return SmartDataLoader(
            dataset=self,
            batch_size=batch_size,
            epochs_per_iter=epochs_per_iter
        )


class SmartDataLoader:
    """
    Data loader with smart batching for enhanced training.
    
    Features:
    - Smart puzzle grouping for better gradient signals
    - Train vs test mode distinction
    - Epochs batching for efficiency
    """
    
    def __init__(
        self,
        dataset: EnhancedPuzzleDataset,
        batch_size: int,
        epochs_per_iter: int = 1
    ):
        """
        Initialize smart data loader.
        
        Args:
            dataset: Enhanced puzzle dataset
            batch_size: Batch size
            epochs_per_iter: Number of epochs to batch together
        """
        self.dataset = dataset
        self.batch_size = batch_size
        self.epochs_per_iter = epochs_per_iter
        
        # Create smart batch sampler
        self.sampler = SmartBatchSampler(
            puzzle_indices=dataset.puzzle_indices,
            group_indices=dataset.group_indices,
            batch_size=batch_size,
            shuffle_groups=(dataset.mode == 'train'),
            seed=dataset.seed
        )
        
        # Track epoch state to avoid double initialization
        self._epoch_initialized = False
        
        # Validate dataset structure
        self._validate_dataset_structure()
    
    def _validate_dataset_structure(self):
        """Validate that dataset structure can generate batches."""
        num_groups = self.dataset.group_indices.shape[0] - 1
        dataset_size = len(self.dataset)
        
        # Check if dataset is too small
        if dataset_size < self.batch_size:
            print(f"Warning: Dataset size ({dataset_size}) is smaller than batch size ({self.batch_size}). "
                  f"This may result in no complete batches in training mode.")
        
        # Check if groups are sufficient
        if num_groups == 0:
            raise ValueError("Dataset has no groups. Cannot perform smart batching.")
        
        # Check average examples per group
        avg_examples_per_group = dataset_size / num_groups
        if avg_examples_per_group < self.batch_size:
            print(f"Warning: Average examples per group ({avg_examples_per_group:.1f}) is less than batch size ({self.batch_size}). "
                  f"Smart batching may produce few complete batches.")
        
        print(f"Dataset validation: {dataset_size} examples, {num_groups} groups, batch size {self.batch_size}")
    
    def start_new_epoch(self):
        """Explicitly start a new epoch. Call this when beginning training."""
        self._epoch_initialized = True
        if self.dataset.mode == 'train':
            self.sampler.start_epoch(self.epochs_per_iter)
    
    def __iter__(self) -> Iterator[Dict[str, mx.array]]:
        """Iterate over batches using smart sampling."""
        # Initialize epoch if not already done (for convenience)
        if not self._epoch_initialized and self.dataset.mode == 'train':
            self.start_new_epoch()
        
        if self.dataset.mode == 'train':
            return self._iter_train()
        else:
            return self._iter_test()
    
    def _iter_train(self) -> Iterator[Dict[str, mx.array]]:
        """Training iteration with smart batching."""
        # Epoch should already be started by start_new_epoch() or __iter__()
        
        while self.sampler.has_batches_remaining():
            batch_result = self.sampler.sample_batch()
            if batch_result is None:
                break
            
            example_indices, puzzle_ids = batch_result
            
            # Use MLX array indexing for efficiency
            batch = {
                'input_ids': self.dataset.input_ids[example_indices],
                'labels': self.dataset.labels[example_indices],
                'puzzle_id': self.dataset.puzzle_ids[example_indices]
            }
            
            yield batch
    
    def _iter_test(self) -> Iterator[Dict[str, mx.array]]:
        """Test iteration: sequential, deterministic."""
        # Process examples sequentially
        for i in range(0, len(self.dataset), self.batch_size):
            end_idx = min(i + self.batch_size, len(self.dataset))
            batch_indices = mx.array(list(range(i, end_idx)))
            
            # Use MLX array indexing for efficiency
            batch = {
                'input_ids': self.dataset.input_ids[batch_indices],
                'labels': self.dataset.labels[batch_indices],
                'puzzle_id': self.dataset.puzzle_ids[batch_indices]
            }
            
            yield batch
    
    def __len__(self) -> int:
        """Get number of batches (approximate for training mode)."""
        return (len(self.dataset) + self.batch_size - 1) // self.batch_size


class PuzzleDatasetMetadata:
    """Metadata for puzzle datasets."""
    
    def __init__(
        self,
        pad_id: int,
        ignore_label_id: Optional[int],
        blank_identifier_id: int,
        vocab_size: int,
        seq_len: int,
        num_puzzle_identifiers: int,
        total_groups: int,
        mean_puzzle_examples: float,
        sets: List[str]
    ):
        self.pad_id = pad_id
        self.ignore_label_id = ignore_label_id
        self.blank_identifier_id = blank_identifier_id
        self.vocab_size = vocab_size
        self.seq_len = seq_len
        self.num_puzzle_identifiers = num_puzzle_identifiers
        self.total_groups = total_groups
        self.mean_puzzle_examples = mean_puzzle_examples
        self.sets = sets


class PuzzleDataset:
    """
    Dataset for loading puzzle data.
    
    Supports:
    - ARC (Abstract Reasoning Corpus)
    - Sudoku
    - Maze navigation
    - Custom puzzle formats
    
    This is a simplified version of the original PyTorch implementation,
    focused on single-device training and basic functionality.
    """
    
    def __init__(
        self,
        data_path: str,
        split: str = "train",
        max_seq_len: int = 512,
        seed: int = 42
    ):
        """
        Initialize puzzle dataset.
        
        Args:
            data_path: Path to dataset directory
            split: Dataset split ('train', 'val', 'test')
            max_seq_len: Maximum sequence length
            seed: Random seed for reproducibility
        """
        self.data_path = Path(data_path)
        self.split = split
        self.max_seq_len = max_seq_len
        self.seed = seed
        
        # Set random seed
        random.seed(seed)
        
        # Load metadata and data
        self.metadata = self._load_metadata()
        self.examples = self._load_data()
        
        print(f"Loaded {len(self.examples)} examples from {self.data_path}")
    
    def _load_metadata(self) -> Optional[PuzzleDatasetMetadata]:
        """Load dataset metadata from JSON file."""
        metadata_path = self.data_path / self.split / "dataset.json"
        
        if not metadata_path.exists():
            # Fallback for simple formats without metadata
            print(f"No metadata found at {metadata_path}, using defaults")
            return None
        
        with open(metadata_path, 'r') as f:
            data = json.load(f)
            return PuzzleDatasetMetadata(**data)
    
    def _load_data(self) -> List[Dict[str, Any]]:
        """Load puzzle data from disk."""
        examples = []
        
        # Check for pre-processed format (like original HRM)
        if self._load_preprocessed_data(examples):
            return examples
        
        # Fallback: Try to load from common formats
        self._load_arc_format(examples)
        self._load_sudoku_format(examples)
        self._load_maze_format(examples)
        
        if not examples:
            raise ValueError(f"No data found in {self.data_path}")
        
        return examples
    
    def _load_preprocessed_data(self, examples: List[Dict]) -> bool:
        """Load pre-processed HRM format data."""
        data_dir = self.data_path / self.split
        if not data_dir.exists():
            return False
        
        # Look for HRM-style preprocessed files
        input_files = list(data_dir.glob("*__inputs.npy"))
        if not input_files:
            return False
        
        for input_file in input_files:
            set_name = input_file.name.replace("__inputs.npy", "")
            
            # Load corresponding files
            labels_file = data_dir / f"{set_name}__labels.npy"
            puzzle_ids_file = data_dir / f"{set_name}__puzzle_identifiers.npy"
            
            if not all(f.exists() for f in [labels_file, puzzle_ids_file]):
                continue
            
            # Load data (use numpy for loading, convert to lists)
            try:
                import numpy as np
                inputs = np.load(input_file)
                labels = np.load(labels_file)
                puzzle_ids = np.load(puzzle_ids_file)
            except ImportError:
                print("Warning: numpy not available, skipping preprocessed data loading")
                continue
            
            # Convert to examples
            for i in range(len(inputs)):
                examples.append({
                    'input_ids': inputs[i].tolist(),
                    'labels': labels[i].tolist(),
                    'puzzle_id': int(puzzle_ids[i]) if len(puzzle_ids.shape) == 1 else int(puzzle_ids[i])
                })
        
        return len(examples) > 0
    
    def _load_arc_format(self, examples: List[Dict]):
        """Load ARC format data."""
        arc_files = list(self.data_path.glob("**/*.json"))
        
        for json_file in arc_files:
            if self.split not in json_file.name:
                continue
            
            try:
                with open(json_file, 'r') as f:
                    data = json.load(f)
                
                # Process ARC tasks
                for task_id, task in data.items():
                    if isinstance(task, dict) and 'train' in task:
                        # Standard ARC format
                        for example in task.get('train', []) + task.get('test', []):
                            input_grid = example['input']
                            output_grid = example['output']
                            
                            # Convert grids to token sequences
                            input_tokens = self._grid_to_tokens(input_grid)
                            output_tokens = self._grid_to_tokens(output_grid)
                            
                            # Combine input and output (teacher forcing)
                            combined = input_tokens + [self._get_separator_token()] + output_tokens
                            
                            if len(combined) <= self.max_seq_len:
                                examples.append({
                                    'input_ids': combined[:-1],  # Input without last token
                                    'labels': combined[1:],      # Output shifted by 1
                                    'puzzle_id': self._extract_puzzle_id(task_id)
                                })
            except Exception as e:
                print(f"Failed to load {json_file}: {e}")
    
    def _load_sudoku_format(self, examples: List[Dict]):
        """Load Sudoku format data."""
        sudoku_files = list(self.data_path.glob("**/*.txt"))
        
        for txt_file in sudoku_files:
            if 'sudoku' not in txt_file.name.lower():
                continue
            
            try:
                with open(txt_file, 'r') as f:
                    for line_num, line in enumerate(f):
                        line = line.strip()
                        if not line or line.startswith('#'):
                            continue
                        
                        # Parse sudoku line (format: puzzle,solution)
                        if ',' in line:
                            puzzle, solution = line.split(',', 1)
                            
                            # Convert to tokens
                            puzzle_tokens = self._sudoku_to_tokens(puzzle)
                            solution_tokens = self._sudoku_to_tokens(solution)
                            
                            # Combine
                            combined = puzzle_tokens + [self._get_separator_token()] + solution_tokens
                            
                            if len(combined) <= self.max_seq_len:
                                examples.append({
                                    'input_ids': combined[:-1],
                                    'labels': combined[1:],
                                    'puzzle_id': 2000 + (line_num % 1000)  # Sudoku puzzle IDs
                                })
            except Exception as e:
                print(f"Failed to load {txt_file}: {e}")
    
    def _load_maze_format(self, examples: List[Dict]):
        """Load maze format data."""
        maze_files = list(self.data_path.glob("**/*.npz"))
        
        for npz_file in maze_files:
            if 'maze' not in npz_file.name.lower():
                continue
            
            try:
                try:
                    import numpy as np
                    data = np.load(npz_file)
                except ImportError:
                    print("Warning: numpy not available, skipping npz data loading")
                    continue
                
                if 'inputs' in data and 'labels' in data:
                    inputs = data['inputs']
                    labels = data['labels']
                    
                    for i in range(len(inputs)):
                        if len(inputs[i]) <= self.max_seq_len:
                            examples.append({
                                'input_ids': inputs[i].tolist(),
                                'labels': labels[i].tolist(),
                                'puzzle_id': 3000 + i  # Maze puzzle IDs
                            })
            except Exception as e:
                print(f"Failed to load {npz_file}: {e}")
    
    def _grid_to_tokens(self, grid: List[List[int]]) -> List[int]:
        """Convert 2D grid to token sequence."""
        tokens = []
        
        # Add beginning of grid token
        tokens.append(self._get_special_token('BOG'))
        
        for row in grid:
            # Add row tokens
            tokens.extend([min(val, 255) for val in row])  # Clamp values
            # Add end of line token
            tokens.append(self._get_special_token('EOL'))
        
        # Add end of grid token
        tokens.append(self._get_special_token('EOG'))
        
        return tokens
    
    def _sudoku_to_tokens(self, sudoku_str: str) -> List[int]:
        """Convert Sudoku string to tokens."""
        tokens = []
        
        for char in sudoku_str:
            if char.isdigit():
                # Digits 1-9 map to tokens 1-9, 0 (empty) maps to token 0
                tokens.append(int(char))
            elif char == '.':
                # Empty cell
                tokens.append(0)
        
        return tokens
    
    def _get_special_token(self, name: str) -> int:
        """Get special token ID."""
        # Simple mapping for special tokens
        special_tokens = {
            'PAD': 256,
            'BOG': 257,  # Beginning of grid
            'EOG': 258,  # End of grid
            'EOL': 259,  # End of line
            'SEP': 260,  # Separator
        }
        return special_tokens.get(name, 256)
    
    def _get_separator_token(self) -> int:
        """Get separator token between input and output."""
        return self._get_special_token('SEP')
    
    def _pad_sequence(self, seq: List[int], max_len: int, pad_value: int = None) -> List[int]:
        """Pad sequence to maximum length."""
        if pad_value is None:
            pad_value = self._get_special_token('PAD')
        
        if len(seq) > max_len:
            seq = seq[:max_len]
        else:
            seq = seq + [pad_value] * (max_len - len(seq))
        return seq
    
    def __len__(self) -> int:
        """Get dataset size."""
        return len(self.examples)
    
    def __getitem__(self, idx: int) -> Dict[str, mx.array]:
        """Get a single example."""
        example = self.examples[idx]
        
        # Pad sequences to max length
        input_ids = self._pad_sequence(example['input_ids'], self.max_seq_len)
        labels = self._pad_sequence(example['labels'], self.max_seq_len, IGNORE_LABEL_ID)
        
        return {
            'input_ids': mx.array(input_ids),
            'labels': mx.array(labels),
            'puzzle_id': mx.array(example.get('puzzle_id', 0))
        }


class DataLoader:
    """
    Efficient data loader for MLX.
    
    Features:
    - Batching with automatic padding
    - Shuffling
    - Iterator interface
    """
    
    def __init__(
        self,
        dataset: PuzzleDataset,
        batch_size: int,
        shuffle: bool = True,
        drop_last: bool = False
    ):
        """
        Initialize data loader.
        
        Args:
            dataset: Dataset to load from
            batch_size: Batch size
            shuffle: Whether to shuffle data between epochs
            drop_last: Whether to drop incomplete last batch
        """
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.drop_last = drop_last
        
        # Create indices for iteration
        self.indices = list(range(len(dataset)))
    
    def __iter__(self) -> Iterator[Dict[str, mx.array]]:
        """Iterate over batches."""
        # Shuffle indices if requested
        if self.shuffle:
            random.shuffle(self.indices)
        
        # Generate batches
        for i in range(0, len(self.indices), self.batch_size):
            batch_indices = self.indices[i:i + self.batch_size]
            
            # Skip incomplete batch if drop_last is True
            if self.drop_last and len(batch_indices) < self.batch_size:
                continue
            
            # Collect batch data
            batch_data = {
                'input_ids': [],
                'labels': [],
                'puzzle_id': []
            }
            
            for idx in batch_indices:
                example = self.dataset[idx]
                for key in batch_data:
                    if key in example:
                        batch_data[key].append(example[key])
            
            # Stack into batch tensors
            batch = {}
            for key, values in batch_data.items():
                if values:
                    batch[key] = mx.stack(values)
            
            yield batch
    
    def __len__(self) -> int:
        """Get number of batches."""
        num_batches = len(self.dataset) // self.batch_size
        if not self.drop_last and len(self.dataset) % self.batch_size > 0:
            num_batches += 1
        return num_batches


def create_arc_dataset(
    data_path: str,
    split: str = "train",
    max_seq_len: int = 512,
    seed: int = 42
) -> PuzzleDataset:
    """
    Create ARC dataset with proper configuration.
    
    Args:
        data_path: Path to ARC dataset
        split: Dataset split
        max_seq_len: Maximum sequence length
        seed: Random seed
        
    Returns:
        Configured PuzzleDataset
    """
    return PuzzleDataset(
        data_path=data_path,
        split=split,
        max_seq_len=max_seq_len,
        seed=seed
    )


def create_dataloader(
    dataset: PuzzleDataset,
    batch_size: int,
    shuffle: bool = True,
    drop_last: bool = False
) -> DataLoader:
    """
    Create data loader with sensible defaults.
    
    Args:
        dataset: Dataset to load from
        batch_size: Batch size
        shuffle: Whether to shuffle
        drop_last: Whether to drop incomplete batches
        
    Returns:
        Configured DataLoader
    """
    return DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=drop_last
    )


# Utility functions for data processing

def collate_batch(examples: List[Dict[str, mx.array]]) -> Dict[str, mx.array]:
    """
    Collate a list of examples into a batch.
    
    Args:
        examples: List of example dictionaries
        
    Returns:
        Batched data
    """
    if not examples:
        return {}
    
    # Get all keys from first example
    keys = examples[0].keys()
    
    # Stack examples for each key
    batch = {}
    for key in keys:
        values = [ex[key] for ex in examples if key in ex]
        if values:
            batch[key] = mx.stack(values)
    
    return batch


def get_vocab_size(dataset: PuzzleDataset) -> int:
    """
    Get vocabulary size for the dataset.
    
    Args:
        dataset: Puzzle dataset
        
    Returns:
        Vocabulary size
    """
    if dataset.metadata and hasattr(dataset.metadata, 'vocab_size'):
        return dataset.metadata.vocab_size
    else:
        # Default vocabulary size (covers ARC grids + special tokens)
        return 512