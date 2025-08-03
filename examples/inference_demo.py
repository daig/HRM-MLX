"""
Inference demonstration for HRM in MLX.

This example shows how to use trained HRM models for inference on
reasoning tasks, particularly puzzle-solving scenarios like ARC, Sudoku, etc.
"""

import mlx.core as mx
import mlx.nn as nn
from pathlib import Path
from typing import Dict, List, Optional, Union
import time
import json

# HRM imports
from mlx_hrm import create_hrm, create_hrm_from_checkpoint
from mlx_hrm.utils.checkpoint import load_checkpoint
from mlx_hrm.modules.act import HRMCarry


def create_mock_puzzle_data() -> Dict[str, List]:
    """
    Create mock puzzle data for demonstration.
    
    In practice, this would come from actual puzzle datasets like:
    - ARC (Abstract Reasoning Corpus)
    - Sudoku puzzles
    - Maze navigation tasks
    - Logic puzzles
    
    Returns:
        Dictionary containing different puzzle types
    """
    puzzles = {
        'arc_simple': [
            # Mock ARC puzzle: pattern recognition
            {
                'input_grid': [1, 1, 0, 0, 1, 1, 0, 0, 1, 1],  # 2x5 grid flattened
                'question_token': 100,  # Special token indicating "what comes next?"
                'expected_output': [0, 0, 1, 1, 0, 0]  # Expected pattern continuation
            },
            {
                'input_grid': [2, 0, 2, 0, 2, 0, 2, 0],
                'question_token': 100,
                'expected_output': [2, 0, 2, 0]
            }
        ],
        
        'sudoku_simple': [
            # Mock Sudoku: fill in missing numbers
            {
                'puzzle': [1, 2, 0, 4, 0, 3, 2, 1, 4],  # 3x3 mini-sudoku with 0 = empty
                'question_token': 101,
                'expected_output': [3, 1]  # Missing numbers in order
            }
        ],
        
        'sequence_completion': [
            # Pattern completion tasks
            {
                'sequence': [1, 3, 5, 7, 9],
                'question_token': 102,
                'expected_output': [11, 13, 15]  # Next numbers in arithmetic sequence
            },
            {
                'sequence': [2, 4, 8, 16],
                'question_token': 102,
                'expected_output': [32, 64]  # Geometric sequence
            }
        ]
    }
    
    return puzzles


def encode_puzzle_for_model(puzzle: Dict, puzzle_type: str) -> mx.array:
    """
    Encode a puzzle into token IDs for the model.
    
    In practice, you would have a proper tokenizer that converts
    puzzle representations into token sequences that the model understands.
    
    Args:
        puzzle: Puzzle data dictionary
        puzzle_type: Type of puzzle (determines encoding strategy)
        
    Returns:
        Token sequence for model input
    """
    tokens = []
    
    if puzzle_type == 'arc_simple':
        # Encode: [START] + grid_tokens + [QUESTION] + [END]
        tokens = [1]  # START token
        tokens.extend(puzzle['input_grid'])
        tokens.append(puzzle['question_token'])
        tokens.append(2)  # END token
        
    elif puzzle_type == 'sudoku_simple':
        # Encode: [START] + puzzle_tokens + [QUESTION] + [END]
        tokens = [1]  # START token
        tokens.extend(puzzle['puzzle'])
        tokens.append(puzzle['question_token'])
        tokens.append(2)  # END token
        
    elif puzzle_type == 'sequence_completion':
        # Encode: [START] + sequence + [QUESTION] + [END]
        tokens = [1]  # START token
        tokens.extend(puzzle['sequence'])
        tokens.append(puzzle['question_token'])
        tokens.append(2)  # END token
    
    return mx.array(tokens)


def decode_model_output(output_tokens: mx.array, puzzle_type: str) -> List[int]:
    """
    Decode model output tokens back to puzzle solution.
    
    Args:
        output_tokens: Raw token predictions from model
        puzzle_type: Type of puzzle (determines decoding strategy)
        
    Returns:
        Decoded solution
    """
    # Convert to list and remove special tokens
    tokens = output_tokens.tolist()
    
    # Filter out special tokens (0, 1, 2 = PAD, START, END)
    solution_tokens = [t for t in tokens if t > 2]
    
    return solution_tokens


class HRMInferenceEngine:
    """
    Inference engine for HRM puzzle solving.
    
    Provides high-level interface for running inference on various
    puzzle types with proper ACT handling and result interpretation.
    """
    
    def __init__(
        self, 
        model: nn.Module,
        max_act_steps: int = 64,
        temperature: float = 0.1,  # Low temperature for focused reasoning
        top_k: Optional[int] = 10
    ):
        """
        Initialize inference engine.
        
        Args:
            model: Trained HRM model
            max_act_steps: Maximum ACT steps for reasoning
            temperature: Sampling temperature (lower = more focused)
            top_k: Top-k filtering for generation
        """
        self.model = model
        self.max_act_steps = max_act_steps
        self.temperature = temperature
        self.top_k = top_k
        
    def solve_puzzle(
        self, 
        puzzle: Dict, 
        puzzle_type: str,
        max_solution_length: int = 20,
        verbose: bool = True
    ) -> Dict:
        """
        Solve a single puzzle using the HRM model.
        
        Args:
            puzzle: Puzzle data
            puzzle_type: Type of puzzle
            max_solution_length: Maximum length of generated solution
            verbose: Whether to print reasoning steps
            
        Returns:
            Dictionary containing solution and reasoning information
        """
        if verbose:
            print(f"\nSolving {puzzle_type} puzzle...")
        
        # Encode puzzle
        input_tokens = encode_puzzle_for_model(puzzle, puzzle_type)
        if verbose:
            print(f"Input tokens: {input_tokens.tolist()}")
        
        # Track reasoning process
        reasoning_steps = []
        start_time = time.time()
        
        # Initial carry state
        carry = self.model.initial_carry(1)
        
        # Reasoning phase: let ACT determine how much computation is needed
        batch = {'input_ids': input_tokens.reshape(1, -1)}
        
        # Run multiple ACT steps to let the model reason
        for step in range(self.max_act_steps):
            carry, outputs = self.model(batch, carry)
            
            step_info = {
                'step': step + 1,
                'act_steps_used': carry.act_step[0].item(),
                'halted': carry.halted[0].item(),
                'q_halt_confidence': mx.sigmoid(outputs['q_halt_logits'][0]).item()
            }
            reasoning_steps.append(step_info)
            
            if verbose:
                print(f"  Step {step + 1}: ACT={step_info['act_steps_used']}, "
                      f"Halted={step_info['halted']}, "
                      f"Q-halt={step_info['q_halt_confidence']:.3f}")
            
            # Stop if model decided to halt
            if carry.halted[0]:
                if verbose:
                    print(f"  Model halted after {step + 1} reasoning steps")
                break
        
        # Generation phase: generate solution tokens
        if verbose:
            print("  Generating solution...")
        
        solution_tokens = self.model.generate(
            prompt=input_tokens,
            max_length=len(input_tokens) + max_solution_length,
            temperature=self.temperature,
            top_k=self.top_k
        )
        
        # Extract only the new tokens (solution part)
        solution_part = solution_tokens[len(input_tokens):]
        
        # Decode solution
        decoded_solution = decode_model_output(solution_part, puzzle_type)
        
        inference_time = time.time() - start_time
        
        result = {
            'solution': decoded_solution,
            'reasoning_steps': reasoning_steps,
            'total_act_steps': sum(step['act_steps_used'] for step in reasoning_steps),
            'inference_time': inference_time,
            'input_tokens': input_tokens.tolist(),
            'output_tokens': solution_tokens.tolist(),
            'expected': puzzle.get('expected_output', None)
        }
        
        if verbose:
            print(f"  Solution: {decoded_solution}")
            print(f"  Expected: {result['expected']}")
            print(f"  Total ACT steps: {result['total_act_steps']}")
            print(f"  Inference time: {inference_time:.3f}s")
        
        return result
    
    def evaluate_on_puzzles(
        self, 
        puzzles: Dict[str, List], 
        max_puzzles_per_type: Optional[int] = None
    ) -> Dict:
        """
        Evaluate model on multiple puzzles.
        
        Args:
            puzzles: Dictionary of puzzle types and examples
            max_puzzles_per_type: Limit puzzles per type (for quick testing)
            
        Returns:
            Evaluation results
        """
        print("\nEvaluating HRM on puzzle dataset...")
        print("=" * 50)
        
        results = {}
        total_correct = 0
        total_puzzles = 0
        
        for puzzle_type, puzzle_list in puzzles.items():
            print(f"\nEvaluating {puzzle_type}...")
            
            if max_puzzles_per_type:
                puzzle_list = puzzle_list[:max_puzzles_per_type]
            
            type_results = {
                'puzzles': [],
                'correct': 0,
                'total': len(puzzle_list),
                'accuracy': 0.0,
                'avg_act_steps': 0.0,
                'avg_inference_time': 0.0
            }
            
            for i, puzzle in enumerate(puzzle_list):
                print(f"\n  Puzzle {i + 1}/{len(puzzle_list)}:")
                
                result = self.solve_puzzle(
                    puzzle, 
                    puzzle_type, 
                    verbose=False  # Reduced verbosity for evaluation
                )
                
                # Check correctness
                is_correct = (
                    result['expected'] is not None and 
                    result['solution'] == result['expected']
                )
                
                if is_correct:
                    type_results['correct'] += 1
                    total_correct += 1
                
                type_results['puzzles'].append(result)
                total_puzzles += 1
                
                print(f"    Solution: {result['solution']}")
                print(f"    Expected: {result['expected']}")
                print(f"    Correct: {is_correct}")
                print(f"    ACT steps: {result['total_act_steps']}")
            
            # Compute type-level statistics
            type_results['accuracy'] = type_results['correct'] / type_results['total']
            type_results['avg_act_steps'] = sum(
                r['total_act_steps'] for r in type_results['puzzles']
            ) / len(type_results['puzzles'])
            type_results['avg_inference_time'] = sum(
                r['inference_time'] for r in type_results['puzzles']
            ) / len(type_results['puzzles'])
            
            results[puzzle_type] = type_results
            
            print(f"\n  {puzzle_type} Results:")
            print(f"    Accuracy: {type_results['accuracy']:.1%}")
            print(f"    Avg ACT steps: {type_results['avg_act_steps']:.1f}")
            print(f"    Avg time: {type_results['avg_inference_time']:.3f}s")
        
        # Overall statistics
        overall_accuracy = total_correct / total_puzzles if total_puzzles > 0 else 0.0
        
        print(f"\n" + "=" * 50)
        print(f"Overall Results:")
        print(f"  Total puzzles: {total_puzzles}")
        print(f"  Correct: {total_correct}")
        print(f"  Overall accuracy: {overall_accuracy:.1%}")
        
        results['overall'] = {
            'accuracy': overall_accuracy,
            'total_puzzles': total_puzzles,
            'correct': total_correct
        }
        
        return results


def demo_single_puzzle_inference():
    """Demonstrate inference on a single puzzle."""
    print("=" * 60)
    print("Single Puzzle Inference Demo")
    print("=" * 60)
    
    # Create model (in practice, load from checkpoint)
    print("Creating HRM model (tiny)...")
    model = create_hrm('tiny')
    print(f"Model loaded with {model.num_parameters:,} parameters")
    
    # Create inference engine
    engine = HRMInferenceEngine(model, temperature=0.1)
    
    # Create a sample puzzle
    puzzle = {
        'sequence': [2, 4, 6, 8],
        'question_token': 102,
        'expected_output': [10, 12]
    }
    
    # Solve puzzle
    result = engine.solve_puzzle(
        puzzle, 
        'sequence_completion',
        verbose=True
    )
    
    print(f"\nResult summary:")
    print(f"  Input: {puzzle['sequence']}")
    print(f"  Expected: {puzzle['expected_output']}")
    print(f"  Generated: {result['solution']}")
    print(f"  Correct: {result['solution'] == result['expected']}")


def demo_batch_evaluation():
    """Demonstrate evaluation on multiple puzzles."""
    print("\n" + "=" * 60)
    print("Batch Evaluation Demo")
    print("=" * 60)
    
    # Create model
    model = create_hrm('tiny')
    engine = HRMInferenceEngine(model, temperature=0.1)
    
    # Create test puzzles
    puzzles = create_mock_puzzle_data()
    
    # Run evaluation
    results = engine.evaluate_on_puzzles(
        puzzles, 
        max_puzzles_per_type=2  # Limit for demo
    )
    
    return results


def demo_checkpoint_loading():
    """Demonstrate loading a trained model from checkpoint."""
    print("\n" + "=" * 60)
    print("Checkpoint Loading Demo")
    print("=" * 60)
    
    checkpoint_path = "./training_output/checkpoint_final.pkl"
    
    if Path(checkpoint_path).exists():
        print(f"Loading model from {checkpoint_path}...")
        try:
            model = create_hrm_from_checkpoint(checkpoint_path)
            print("Model loaded successfully!")
            
            # Test with a simple puzzle
            engine = HRMInferenceEngine(model)
            puzzle = {
                'sequence': [1, 3, 5, 7],
                'question_token': 102,
                'expected_output': [9, 11]
            }
            
            result = engine.solve_puzzle(puzzle, 'sequence_completion')
            print(f"Inference with trained model completed")
            
        except Exception as e:
            print(f"Failed to load checkpoint: {e}")
            print("Using untrained model instead...")
            demo_single_puzzle_inference()
    else:
        print(f"Checkpoint not found at {checkpoint_path}")
        print("Run training_example.py first to create a checkpoint")
        print("Using untrained model for demonstration...")
        demo_single_puzzle_inference()


def demo_act_reasoning_analysis():
    """Analyze how ACT mechanism works during reasoning."""
    print("\n" + "=" * 60)
    print("ACT Reasoning Analysis Demo")
    print("=" * 60)
    
    model = create_hrm('tiny')
    
    # Create puzzles of varying complexity
    puzzles = [
        # Simple pattern
        {
            'sequence': [1, 2, 3],
            'complexity': 'simple'
        },
        # More complex pattern
        {
            'sequence': [1, 1, 2, 3, 5, 8],  # Fibonacci
            'complexity': 'medium'
        },
        # Very complex pattern
        {
            'sequence': [2, 3, 5, 7, 11, 13],  # Primes
            'complexity': 'hard'
        }
    ]
    
    engine = HRMInferenceEngine(model, max_act_steps=20)
    
    print("Analyzing ACT usage by puzzle complexity...\n")
    
    for i, puzzle in enumerate(puzzles):
        print(f"Puzzle {i + 1} ({puzzle['complexity']}):")
        print(f"  Sequence: {puzzle['sequence']}")
        
        puzzle_with_question = {
            'sequence': puzzle['sequence'],
            'question_token': 102,
            'expected_output': [0]  # Placeholder
        }
        
        result = engine.solve_puzzle(
            puzzle_with_question,
            'sequence_completion',
            verbose=False
        )
        
        print(f"  Total ACT steps: {result['total_act_steps']}")
        print(f"  Reasoning steps: {len(result['reasoning_steps'])}")
        print(f"  Inference time: {result['inference_time']:.3f}s")
        
        # Show ACT step progression
        act_progression = [step['act_steps_used'] for step in result['reasoning_steps']]
        print(f"  ACT progression: {act_progression}")
        print()


def main():
    """Run all inference demonstrations."""
    print("HRM MLX Inference Demonstrations")
    print("This demo shows various ways to use HRM for puzzle solving")
    
    try:
        # 1. Single puzzle demo
        demo_single_puzzle_inference()
        
        # 2. Batch evaluation demo
        demo_batch_evaluation()
        
        # 3. Checkpoint loading demo
        demo_checkpoint_loading()
        
        # 4. ACT reasoning analysis
        demo_act_reasoning_analysis()
        
        print("\n" + "=" * 60)
        print("All inference demos completed!")
        print("\nKey takeaways:")
        print("1. HRM uses ACT to adaptively compute reasoning steps")
        print("2. More complex puzzles typically require more ACT steps")
        print("3. The model can be trained on specific puzzle types")
        print("4. Inference is fast and memory-efficient")
        print("5. Temperature and top-k can control generation quality")
        
    except Exception as e:
        print(f"\nDemo failed with error: {e}")
        print("\nNote: This demo uses an untrained model.")
        print("For best results, train the model first using training_example.py")


if __name__ == "__main__":
    main()