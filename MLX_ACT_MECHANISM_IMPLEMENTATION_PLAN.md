# MLX ACT Mechanism Implementation Plan

## Executive Summary

This document provides a comprehensive implementation plan for porting the Hierarchical Reasoning Machine (HRM) Adaptive Computation Time (ACT) mechanism from PyTorch to MLX. The plan addresses all key requirements including carry state management, gradient-free forward passes, Q-learning exploration, and the two-level hierarchy architecture.

## Table of Contents

1. [Architecture Overview](#architecture-overview)
2. [Technical Requirements](#technical-requirements)
3. [Implementation Strategy](#implementation-strategy)
4. [Carry State Implementation](#carry-state-implementation)
5. [Gradient Management](#gradient-management)
6. [ACT Mechanism](#act-mechanism)
7. [Q-Learning Integration](#q-learning-integration)
8. [Hierarchical Architecture](#hierarchical-architecture)
9. [Testing Strategy](#testing-strategy)
10. [Performance Considerations](#performance-considerations)
11. [Implementation Timeline](#implementation-timeline)
12. [Example Usage](#example-usage)

## Architecture Overview

### Hierarchical Reasoning Machine (HRM)

The HRM consists of two levels:
- **H-level (High-level)**: Planning and decision-making layer
- **L-level (Low-level)**: Computation and execution layer

### Adaptive Computation Time (ACT)

ACT allows the model to dynamically adjust computation steps based on problem complexity:
- Minimum steps: 1
- Maximum steps: `halt_max_steps` (configurable)
- Halting decision based on cumulative probability threshold
- Q-learning for intelligent halting during training

### Key Components

1. **Carry States**: Persistent state information between computation steps
2. **Halting Mechanism**: Probability-based stopping with Q-learning exploration
3. **Gradient Management**: Careful handling of gradients for stable training
4. **Memory Management**: Efficient state tracking across variable steps

## Technical Requirements

### MLX-Specific Considerations

1. **Functional Programming Style**: MLX favors functional patterns over stateful operations
2. **Gradient Functions**: Use `mlx.grad()` and `mlx.value_and_grad()`
3. **Stop Gradient**: Replace PyTorch's `no_grad()` with `mlx.stop_gradient()`
4. **Dynamic Shapes**: Handle variable computation steps efficiently
5. **Memory Layout**: Column-major storage vs PyTorch's row-major

### Core Dependencies

```python
import mlx.core as mx
import mlx.nn as nn
from dataclasses import dataclass
from typing import NamedTuple, Optional, Tuple, List
import numpy as np
```

## Implementation Strategy

### Phase 1: Core Data Structures

#### Option 1: Dataclass Approach

```python
@dataclass
class CarryState:
    """Carry state for ACT mechanism in MLX"""
    hidden: mx.array
    memory: Optional[mx.array] = None
    halting_prob: Optional[mx.array] = None
    n_steps: Optional[mx.array] = None
    
    def detach(self) -> 'CarryState':
        """Create a gradient-free copy of the carry state"""
        return CarryState(
            hidden=mx.stop_gradient(self.hidden),
            memory=mx.stop_gradient(self.memory) if self.memory is not None else None,
            halting_prob=mx.stop_gradient(self.halting_prob) if self.halting_prob is not None else None,
            n_steps=mx.stop_gradient(self.n_steps) if self.n_steps is not None else None
        )
    
    def to_dict(self) -> dict:
        """Convert to dictionary for MLX operations"""
        return {
            'hidden': self.hidden,
            'memory': self.memory,
            'halting_prob': self.halting_prob,
            'n_steps': self.n_steps
        }
```

#### Option 2: NamedTuple Approach (More Functional)

```python
class CarryStateTuple(NamedTuple):
    """Immutable carry state for functional programming"""
    hidden: mx.array
    memory: Optional[mx.array] = None
    halting_prob: Optional[mx.array] = None
    n_steps: Optional[mx.array] = None
    
    def detach(self) -> 'CarryStateTuple':
        """Create a gradient-free copy"""
        return CarryStateTuple(
            hidden=mx.stop_gradient(self.hidden),
            memory=mx.stop_gradient(self.memory) if self.memory is not None else None,
            halting_prob=mx.stop_gradient(self.halting_prob) if self.halting_prob is not None else None,
            n_steps=mx.stop_gradient(self.n_steps) if self.n_steps is not None else None
        )
```

### Phase 2: ACT Controller Implementation

```python
class ACTController(nn.Module):
    """Adaptive Computation Time controller for MLX"""
    
    def __init__(self, hidden_dim: int, halt_max_steps: int = 10, 
                 halt_threshold: float = 0.99, epsilon: float = 0.01):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.halt_max_steps = halt_max_steps
        self.halt_threshold = halt_threshold
        self.epsilon = epsilon
        
        # Halting probability predictor
        self.halt_predictor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1)
        )
        
        # Q-value predictor for exploration
        self.q_predictor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1)
        )
    
    def compute_halting_probability(self, hidden: mx.array) -> mx.array:
        """Compute halting probability for current step"""
        logits = self.halt_predictor(hidden)
        return mx.sigmoid(logits)
    
    def should_halt(self, cumulative_prob: mx.array, step: int, 
                    training: bool = False) -> Tuple[bool, mx.array]:
        """Determine if computation should halt"""
        if step >= self.halt_max_steps - 1:
            return True, mx.ones_like(cumulative_prob)
        
        # Basic halting condition
        should_halt_base = cumulative_prob >= self.halt_threshold
        
        if training:
            # Q-learning exploration
            random_halt = mx.random.uniform(shape=cumulative_prob.shape) < self.epsilon
            should_halt = mx.logical_or(should_halt_base, random_halt)
        else:
            should_halt = should_halt_base
        
        return should_halt, should_halt
```

## Gradient Management

### Gradient-Free Forward Pass

```python
def forward_step_no_grad(module: nn.Module, carry_state: CarryState, 
                        input_data: mx.array) -> CarryState:
    """Execute a forward pass without gradient tracking"""
    # Detach all inputs
    detached_state = carry_state.detach()
    detached_input = mx.stop_gradient(input_data)
    
    # Forward pass with stop_gradient
    with mx.no_grad():  # If MLX implements context manager
        new_hidden = module(detached_input, detached_state.hidden)
    
    # Alternative without context manager
    new_hidden = mx.stop_gradient(module(detached_input, detached_state.hidden))
    
    return CarryState(
        hidden=new_hidden,
        memory=detached_state.memory,
        halting_prob=detached_state.halting_prob,
        n_steps=detached_state.n_steps
    )
```

### Gradient Flow Control

```python
class GradientController:
    """Manage gradient flow in ACT mechanism"""
    
    @staticmethod
    def checkpoint_state(state: CarryState, requires_grad: bool = True) -> CarryState:
        """Create a checkpoint for gradient computation"""
        if requires_grad:
            return state  # Keep gradients
        return state.detach()  # Remove gradients
    
    @staticmethod
    def accumulate_gradients(states: List[CarryState], weights: mx.array) -> mx.array:
        """Accumulate weighted gradients across computation steps"""
        accumulated = mx.zeros_like(states[0].hidden)
        
        for i, state in enumerate(states):
            weighted_hidden = state.hidden * weights[i:i+1]
            accumulated = accumulated + weighted_hidden
        
        return accumulated
```

## ACT Mechanism

### Complete ACT Implementation

```python
class AdaptiveComputationTime(nn.Module):
    """Full ACT mechanism implementation for MLX"""
    
    def __init__(self, base_module: nn.Module, controller: ACTController):
        super().__init__()
        self.base_module = base_module
        self.controller = controller
        self.halt_max_steps = controller.halt_max_steps
    
    def forward(self, x: mx.array, initial_state: CarryState, 
                training: bool = False) -> Tuple[mx.array, dict]:
        """Forward pass with adaptive computation time"""
        batch_size = x.shape[0]
        device = x.device if hasattr(x, 'device') else None
        
        # Initialize tracking variables
        states = [initial_state]
        halt_probs = []
        halted = mx.zeros((batch_size,), dtype=mx.bool_)
        cumulative_probs = mx.zeros((batch_size, 1))
        step_weights = []
        
        # Adaptive computation loop
        for step in range(self.halt_max_steps):
            current_state = states[-1]
            
            # Compute halting probability
            p_halt = self.controller.compute_halting_probability(current_state.hidden)
            halt_probs.append(p_halt)
            
            # Update cumulative probability
            step_p = p_halt * (1 - cumulative_probs)
            cumulative_probs = cumulative_probs + step_p
            
            # Check halting condition
            should_halt, halt_mask = self.controller.should_halt(
                cumulative_probs, step, training
            )
            
            # Update halted samples
            newly_halted = mx.logical_and(mx.logical_not(halted), halt_mask.squeeze())
            halted = mx.logical_or(halted, newly_halted)
            
            # Compute step weights
            weight = mx.where(newly_halted, step_p, mx.zeros_like(step_p))
            step_weights.append(weight)
            
            # Break if all samples have halted
            if mx.all(halted):
                break
            
            # Compute next state (gradient-free for intermediate steps)
            if step < self.halt_max_steps - 1:
                next_state = forward_step_no_grad(self.base_module, current_state, x)
            else:
                # Final step keeps gradients
                next_state = self.base_module(x, current_state)
            
            states.append(next_state)
        
        # Aggregate outputs
        final_output = self._aggregate_outputs(states[1:], step_weights)
        
        # Prepare auxiliary outputs
        aux_outputs = {
            'n_steps': len(states) - 1,
            'halt_probs': mx.stack(halt_probs),
            'step_weights': mx.stack(step_weights),
            'final_state': states[-1]
        }
        
        return final_output, aux_outputs
    
    def _aggregate_outputs(self, states: List[CarryState], 
                          weights: List[mx.array]) -> mx.array:
        """Aggregate outputs across computation steps"""
        weighted_sum = mx.zeros_like(states[0].hidden)
        
        for state, weight in zip(states, weights):
            weighted_sum = weighted_sum + state.hidden * weight
        
        return weighted_sum
```

## Q-Learning Integration

### Q-Learning Loss Implementation

```python
class QLearningACT:
    """Q-learning components for ACT training"""
    
    def __init__(self, gamma: float = 0.99, alpha: float = 0.001):
        self.gamma = gamma
        self.alpha = alpha
    
    def compute_q_loss(self, q_values: mx.array, rewards: mx.array, 
                      next_q_values: mx.array, dones: mx.array) -> mx.array:
        """Compute Q-learning loss for halting decisions"""
        # Target Q-values
        targets = rewards + self.gamma * next_q_values * (1 - dones)
        targets = mx.stop_gradient(targets)
        
        # MSE loss
        q_loss = mx.mean((q_values - targets) ** 2)
        
        return q_loss
    
    def compute_step_reward(self, step: int, max_steps: int, 
                           accuracy: mx.array) -> mx.array:
        """Compute reward for halting at a given step"""
        # Efficiency bonus for early stopping
        efficiency_reward = (max_steps - step) / max_steps
        
        # Accuracy component
        accuracy_reward = accuracy
        
        # Combined reward
        return 0.5 * efficiency_reward + 0.5 * accuracy_reward
```

### Training Integration

```python
def train_step_with_qlearning(model: AdaptiveComputationTime, 
                             optimizer: mx.optimizers.Optimizer,
                             x: mx.array, y: mx.array, 
                             carry_state: CarryState) -> dict:
    """Training step with Q-learning for ACT"""
    
    def loss_fn(model, x, y, carry_state):
        # Forward pass
        output, aux = model(x, carry_state, training=True)
        
        # Task loss
        task_loss = mx.mean((output - y) ** 2)  # Example: MSE
        
        # Q-learning loss
        q_controller = QLearningACT()
        q_values = aux['q_values']
        rewards = q_controller.compute_step_reward(
            aux['n_steps'], model.halt_max_steps, 
            mx.mean(output == y)  # Accuracy
        )
        q_loss = q_controller.compute_q_loss(
            q_values[:-1], rewards[:-1], 
            q_values[1:], aux['dones'][:-1]
        )
        
        # Combined loss
        total_loss = task_loss + 0.1 * q_loss
        
        return total_loss, (output, aux)
    
    # Compute gradients
    grad_fn = mx.value_and_grad(loss_fn, has_aux=True)
    (loss, (output, aux)), grads = grad_fn(model, x, y, carry_state)
    
    # Update parameters
    optimizer.update(model, grads)
    
    return {
        'loss': loss,
        'output': output,
        'aux': aux,
        'carry_state': aux['final_state']
    }
```

## Hierarchical Architecture

### H-Level Implementation

```python
class HLevelModule(nn.Module):
    """High-level planning module"""
    
    def __init__(self, input_dim: int, hidden_dim: int, 
                 num_actions: int, use_act: bool = True):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.use_act = use_act
        
        # Planning network
        self.planner = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        
        # Action predictor
        self.action_head = nn.Linear(hidden_dim, num_actions)
        
        # ACT controller if enabled
        if use_act:
            self.act_controller = ACTController(hidden_dim)
    
    def forward(self, x: mx.array, carry_state: Optional[CarryState] = None,
                training: bool = False) -> Tuple[mx.array, CarryState]:
        """Forward pass with optional ACT"""
        if carry_state is None:
            carry_state = self.init_carry_state(x.shape[0])
        
        if self.use_act:
            # Use ACT mechanism
            act_module = AdaptiveComputationTime(self.planner, self.act_controller)
            hidden, aux = act_module(x, carry_state, training)
            new_carry_state = aux['final_state']
        else:
            # Single forward pass
            hidden = self.planner(x)
            new_carry_state = CarryState(hidden=hidden)
        
        # Predict actions
        actions = self.action_head(hidden)
        
        return actions, new_carry_state
    
    def init_carry_state(self, batch_size: int) -> CarryState:
        """Initialize carry state"""
        return CarryState(
            hidden=mx.zeros((batch_size, self.hidden_dim)),
            memory=None,
            halting_prob=mx.zeros((batch_size, 1)),
            n_steps=mx.zeros((batch_size,))
        )
```

### L-Level Implementation

```python
class LLevelModule(nn.Module):
    """Low-level computation module"""
    
    def __init__(self, input_dim: int, hidden_dim: int, 
                 output_dim: int, use_act: bool = True):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.use_act = use_act
        
        # Computation network
        self.compute_net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        
        # Output projection
        self.output_proj = nn.Linear(hidden_dim, output_dim)
        
        # Memory module
        self.memory = nn.LSTM(hidden_dim, hidden_dim)
        
        # ACT controller if enabled
        if use_act:
            self.act_controller = ACTController(hidden_dim)
    
    def forward(self, x: mx.array, h_level_output: mx.array,
                carry_state: Optional[CarryState] = None,
                training: bool = False) -> Tuple[mx.array, CarryState]:
        """Forward pass with H-level guidance"""
        if carry_state is None:
            carry_state = self.init_carry_state(x.shape[0])
        
        # Combine inputs
        combined = mx.concatenate([x, h_level_output], axis=-1)
        
        if self.use_act:
            # Use ACT mechanism
            act_module = AdaptiveComputationTime(self.compute_net, self.act_controller)
            hidden, aux = act_module(combined, carry_state, training)
            new_carry_state = aux['final_state']
        else:
            # Single forward pass
            hidden = self.compute_net(combined)
            new_carry_state = CarryState(hidden=hidden, memory=carry_state.memory)
        
        # Update memory
        if new_carry_state.memory is not None:
            hidden, (h_n, c_n) = self.memory(
                hidden.unsqueeze(0), 
                (new_carry_state.memory[0], new_carry_state.memory[1])
            )
            new_carry_state = CarryState(
                hidden=hidden.squeeze(0),
                memory=(h_n, c_n),
                halting_prob=new_carry_state.halting_prob,
                n_steps=new_carry_state.n_steps
            )
        
        # Generate output
        output = self.output_proj(new_carry_state.hidden)
        
        return output, new_carry_state
    
    def init_carry_state(self, batch_size: int) -> CarryState:
        """Initialize carry state with memory"""
        return CarryState(
            hidden=mx.zeros((batch_size, self.hidden_dim)),
            memory=(mx.zeros((1, batch_size, self.hidden_dim)),
                   mx.zeros((1, batch_size, self.hidden_dim))),
            halting_prob=mx.zeros((batch_size, 1)),
            n_steps=mx.zeros((batch_size,))
        )
```

### Complete HRM Model

```python
class HierarchicalReasoningMachine(nn.Module):
    """Complete HRM with ACT mechanism"""
    
    def __init__(self, input_dim: int, h_hidden_dim: int, 
                 l_hidden_dim: int, output_dim: int, 
                 num_actions: int, use_act: bool = True):
        super().__init__()
        
        # H-level module
        self.h_level = HLevelModule(
            input_dim, h_hidden_dim, num_actions, use_act
        )
        
        # L-level module
        self.l_level = LLevelModule(
            input_dim + num_actions, l_hidden_dim, output_dim, use_act
        )
        
        self.use_act = use_act
    
    def forward(self, x: mx.array, 
                h_carry: Optional[CarryState] = None,
                l_carry: Optional[CarryState] = None,
                training: bool = False) -> Tuple[mx.array, dict]:
        """Full hierarchical forward pass"""
        # H-level planning
        h_output, new_h_carry = self.h_level(x, h_carry, training)
        
        # L-level computation
        l_output, new_l_carry = self.l_level(x, h_output, l_carry, training)
        
        # Return outputs and states
        return l_output, {
            'h_carry': new_h_carry,
            'l_carry': new_l_carry,
            'h_output': h_output
        }
```

## Testing Strategy

### Unit Tests

```python
def test_carry_state_detach():
    """Test carry state gradient detachment"""
    # Create carry state with gradients
    hidden = mx.array([[1.0, 2.0], [3.0, 4.0]], dtype=mx.float32)
    hidden.requires_grad = True
    state = CarryState(hidden=hidden)
    
    # Detach
    detached = state.detach()
    
    # Verify no gradients
    assert not detached.hidden.requires_grad
    assert mx.array_equal(detached.hidden, hidden)

def test_act_controller():
    """Test ACT controller functionality"""
    controller = ACTController(hidden_dim=64)
    hidden = mx.random.normal((4, 64))
    
    # Test halting probability
    p_halt = controller.compute_halting_probability(hidden)
    assert p_halt.shape == (4, 1)
    assert mx.all(p_halt >= 0) and mx.all(p_halt <= 1)
    
    # Test halting decision
    cumulative = mx.array([[0.5], [0.95], [0.99], [0.3]])
    should_halt, mask = controller.should_halt(cumulative, step=3)
    expected = mx.array([False, False, True, False])
    assert mx.array_equal(mask.squeeze(), expected)

def test_gradient_free_forward():
    """Test gradient-free forward pass"""
    module = nn.Linear(64, 64)
    state = CarryState(hidden=mx.random.normal((4, 64)))
    input_data = mx.random.normal((4, 32))
    
    # Forward pass
    new_state = forward_step_no_grad(module, state, input_data)
    
    # Verify no gradients
    assert not new_state.hidden.requires_grad
```

### Integration Tests

```python
def test_full_act_mechanism():
    """Test complete ACT forward pass"""
    base_module = nn.Linear(64, 64)
    controller = ACTController(64, halt_max_steps=5)
    act_module = AdaptiveComputationTime(base_module, controller)
    
    # Test input
    x = mx.random.normal((4, 64))
    initial_state = CarryState(hidden=mx.zeros((4, 64)))
    
    # Forward pass
    output, aux = act_module(x, initial_state, training=True)
    
    # Verify outputs
    assert output.shape == (4, 64)
    assert 'n_steps' in aux
    assert aux['n_steps'] <= 5
    assert aux['halt_probs'].shape[0] <= 5

def test_hierarchical_model():
    """Test full HRM model"""
    model = HierarchicalReasoningMachine(
        input_dim=32, h_hidden_dim=64, l_hidden_dim=128,
        output_dim=10, num_actions=5, use_act=True
    )
    
    # Test input
    x = mx.random.normal((4, 32))
    
    # Forward pass
    output, states = model(x, training=True)
    
    # Verify outputs
    assert output.shape == (4, 10)
    assert 'h_carry' in states
    assert 'l_carry' in states
    assert states['h_output'].shape == (4, 5)
```

### Performance Tests

```python
def benchmark_act_vs_fixed():
    """Compare ACT performance vs fixed computation"""
    import time
    
    # Models
    act_model = HierarchicalReasoningMachine(..., use_act=True)
    fixed_model = HierarchicalReasoningMachine(..., use_act=False)
    
    # Test data
    x = mx.random.normal((100, 32))
    
    # Benchmark ACT
    start = time.time()
    for _ in range(100):
        act_model(x)
    act_time = time.time() - start
    
    # Benchmark fixed
    start = time.time()
    for _ in range(100):
        fixed_model(x)
    fixed_time = time.time() - start
    
    print(f"ACT time: {act_time:.3f}s")
    print(f"Fixed time: {fixed_time:.3f}s")
    print(f"Speedup: {fixed_time/act_time:.2f}x")
```

## Performance Considerations

### Memory Optimization

1. **State Pooling**: Reuse carry states across batches
2. **Gradient Checkpointing**: Trade compute for memory
3. **Dynamic Shapes**: Use MLX's efficient dynamic shape handling

```python
class MemoryEfficientACT:
    """Memory-optimized ACT implementation"""
    
    def __init__(self, max_batch_size: int = 128):
        self.state_pool = []
        self.max_batch_size = max_batch_size
    
    def get_carry_state(self, batch_size: int, hidden_dim: int) -> CarryState:
        """Get carry state from pool or create new"""
        for state in self.state_pool:
            if state.hidden.shape == (batch_size, hidden_dim):
                self.state_pool.remove(state)
                return state
        
        # Create new if not in pool
        return CarryState(hidden=mx.zeros((batch_size, hidden_dim)))
    
    def return_carry_state(self, state: CarryState):
        """Return state to pool"""
        if len(self.state_pool) < 10:  # Limit pool size
            self.state_pool.append(state.detach())
```

### Computation Optimization

1. **Kernel Fusion**: Combine operations where possible
2. **Batch Processing**: Process multiple steps together
3. **Early Exit**: Skip computation for halted samples

```python
def optimized_act_forward(module, x, initial_state, controller):
    """Optimized ACT forward pass"""
    batch_size = x.shape[0]
    
    # Pre-allocate arrays
    max_steps = controller.halt_max_steps
    all_hidden = mx.zeros((max_steps, batch_size, initial_state.hidden.shape[-1]))
    all_probs = mx.zeros((max_steps, batch_size, 1))
    
    # Batch computation
    state = initial_state
    for step in range(max_steps):
        # Compute for all samples
        all_hidden[step] = state.hidden
        all_probs[step] = controller.compute_halting_probability(state.hidden)
        
        # Update state
        state = module(x, state)
    
    # Compute halting masks efficiently
    cumulative = mx.cumsum(all_probs, axis=0)
    halt_step = mx.argmax(cumulative >= controller.halt_threshold, axis=0)
    
    # Gather results
    output = mx.take_along_axis(all_hidden, halt_step.unsqueeze(0), axis=0)
    
    return output.squeeze(0)
```

### Distributed Training

```python
def distributed_act_training(model, data_loader, num_devices: int = 4):
    """Distributed training setup for ACT models"""
    # Model replication
    replicated_model = mx.replicate(model, num_devices)
    
    # Distributed optimizer
    optimizer = mx.optimizers.distributed_sgd(
        learning_rate=0.001,
        momentum=0.9
    )
    
    # Training loop
    for batch in data_loader:
        # Split batch across devices
        x_split = mx.split(batch['x'], num_devices)
        y_split = mx.split(batch['y'], num_devices)
        
        # Parallel forward-backward
        grads = []
        for i in range(num_devices):
            with mx.device(i):
                loss, grad = compute_loss_and_grad(
                    replicated_model[i], x_split[i], y_split[i]
                )
                grads.append(grad)
        
        # Aggregate gradients
        avg_grad = mx.mean(mx.stack(grads), axis=0)
        
        # Update model
        optimizer.update(model, avg_grad)
```

## Implementation Timeline

### Week 1: Core Data Structures
- [ ] Implement CarryState classes (both options)
- [ ] Create gradient management utilities
- [ ] Write unit tests for state management
- [ ] Benchmark memory usage

### Week 2: ACT Mechanism
- [ ] Implement ACT controller
- [ ] Create adaptive computation loop
- [ ] Add halting logic with exploration
- [ ] Test variable-length computation

### Week 3: Hierarchical Modules
- [ ] Implement H-level module
- [ ] Implement L-level module
- [ ] Integrate ACT into both levels
- [ ] Test hierarchical communication

### Week 4: Training Infrastructure
- [ ] Implement Q-learning components
- [ ] Create training loops
- [ ] Add logging and monitoring
- [ ] Implement checkpointing

### Week 5: Optimization & Testing
- [ ] Memory optimization
- [ ] Performance benchmarking
- [ ] Integration testing
- [ ] Documentation and examples

## Example Usage

### Model Initialization

```python
# Create HRM model with ACT
model = HierarchicalReasoningMachine(
    input_dim=128,
    h_hidden_dim=256,
    l_hidden_dim=512,
    output_dim=10,
    num_actions=8,
    use_act=True
)

# Initialize optimizer
optimizer = mx.optimizers.Adam(learning_rate=0.001)

# Training configuration
config = {
    'halt_threshold': 0.95,
    'halt_max_steps': 10,
    'q_learning_rate': 0.001,
    'exploration_epsilon': 0.1
}
```

### Training Loop

```python
def train_epoch(model, data_loader, optimizer, config):
    """Complete training epoch with ACT"""
    total_loss = 0.0
    h_carry, l_carry = None, None
    
    for batch in data_loader:
        x, y = batch['input'], batch['target']
        
        # Forward pass with carry states
        def loss_fn(model, x, y, h_carry, l_carry):
            output, states = model(x, h_carry, l_carry, training=True)
            
            # Task loss
            task_loss = mx.mean((output - y) ** 2)
            
            # ACT regularization
            h_steps = states['h_carry'].n_steps
            l_steps = states['l_carry'].n_steps
            step_penalty = 0.01 * (mx.mean(h_steps) + mx.mean(l_steps))
            
            return task_loss + step_penalty, states
        
        # Compute gradients
        grad_fn = mx.value_and_grad(loss_fn, has_aux=True)
        (loss, states), grads = grad_fn(model, x, y, h_carry, l_carry)
        
        # Update model
        optimizer.update(model, grads)
        
        # Update carry states
        h_carry = states['h_carry'].detach()
        l_carry = states['l_carry'].detach()
        
        total_loss += loss.item()
    
    return total_loss / len(data_loader)
```

### Inference with Adaptive Computation

```python
def adaptive_inference(model, x, verbose=False):
    """Inference showing adaptive computation"""
    # Initialize states
    h_carry = model.h_level.init_carry_state(x.shape[0])
    l_carry = model.l_level.init_carry_state(x.shape[0])
    
    # Forward pass
    output, states = model(x, h_carry, l_carry, training=False)
    
    if verbose:
        h_steps = states['h_carry'].n_steps
        l_steps = states['l_carry'].n_steps
        
        print(f"H-level steps: {h_steps.tolist()}")
        print(f"L-level steps: {l_steps.tolist()}")
        print(f"Average steps: H={h_steps.mean():.2f}, L={l_steps.mean():.2f}")
    
    return output

# Example usage
test_input = mx.random.normal((4, 128))
predictions = adaptive_inference(model, test_input, verbose=True)
```

### Monitoring ACT Behavior

```python
class ACTMonitor:
    """Monitor ACT behavior during training"""
    
    def __init__(self):
        self.step_history = []
        self.halt_prob_history = []
    
    def log_step(self, model_output):
        """Log ACT statistics"""
        states = model_output[1]
        
        # Extract steps
        h_steps = states['h_carry'].n_steps.mean().item()
        l_steps = states['l_carry'].n_steps.mean().item()
        
        self.step_history.append({
            'h_steps': h_steps,
            'l_steps': l_steps,
            'timestamp': time.time()
        })
    
    def plot_statistics(self):
        """Visualize ACT behavior"""
        import matplotlib.pyplot as plt
        
        steps = np.array(self.step_history)
        
        plt.figure(figsize=(10, 6))
        plt.plot(steps[:, 0], label='H-level steps')
        plt.plot(steps[:, 1], label='L-level steps')
        plt.xlabel('Training iteration')
        plt.ylabel('Average steps')
        plt.legend()
        plt.title('Adaptive Computation Steps Over Training')
        plt.show()
```

## Conclusion

This implementation plan provides a comprehensive roadmap for porting the HRM ACT mechanism to MLX. Key considerations:

1. **Gradient Management**: Careful use of `mx.stop_gradient()` for intermediate steps
2. **Functional Design**: Leverage MLX's functional programming style
3. **Memory Efficiency**: Pool and reuse carry states
4. **Performance**: Optimize for MLX's execution model
5. **Testing**: Comprehensive unit and integration tests

The modular design allows for incremental implementation and testing, with clear separation between the ACT mechanism, Q-learning components, and hierarchical architecture.

### Next Steps

1. Begin with Phase 1: Core data structures
2. Implement and test each component independently
3. Integrate components incrementally
4. Benchmark against PyTorch implementation
5. Optimize based on profiling results

This plan provides all necessary technical details for a successful implementation while maintaining the sophisticated features of the original ACT mechanism.