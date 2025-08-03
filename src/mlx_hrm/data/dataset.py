"""Data loading utilities for HRM training."""

import mlx.core as mx
import numpy as np
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
        np.random.seed(seed)
        
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
            
            # Load data
            inputs = np.load(input_file)
            labels = np.load(labels_file)
            puzzle_ids = np.load(puzzle_ids_file)
            
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
                                    'puzzle_id': hash(task_id) % 1000
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
                data = np.load(npz_file)
                
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