# Understanding the Adaptive Computation Time (ACT) Mechanism

The Adaptive Computation Time (ACT) mechanism is the core innovation that makes HRM powerful for reasoning tasks. This tutorial explains how ACT works, why it's important, and how to use it effectively.

## Table of Contents

1. [What is ACT?](#what-is-act)
2. [Why ACT Matters](#why-act-matters)
3. [How ACT Works in HRM](#how-act-works-in-hrm)
4. [The Q-Learning Component](#the-q-learning-component)
5. [Carry States](#carry-states)
6. [Training with ACT](#training-with-act)
7. [Monitoring ACT Behavior](#monitoring-act-behavior)
8. [Tuning ACT Parameters](#tuning-act-parameters)
9. [Common Patterns](#common-patterns)
10. [Advanced Usage](#advanced-usage)

## What is ACT?

Adaptive Computation Time (ACT) allows neural networks to dynamically determine how many computation steps to use for each input. Instead of using a fixed number of layers or steps, the model can:

- **Use fewer steps** for simple problems
- **Use more steps** for complex problems
- **Learn when to stop** computing based on confidence

This is similar to how humans solve problems - we spend more time thinking about harder questions and less time on easy ones.

### Traditional vs. ACT Models

```python
# Traditional model: Fixed computation
for layer in layers:
    x = layer(x)  # Always uses all layers
output = x

# ACT model: Adaptive computation
x = input
for step in range(max_steps):
    x = computation_step(x)
    if should_halt(x):  # Dynamic decision
        break
output = x
```

## Why ACT Matters

ACT is particularly valuable for reasoning tasks because:

1. **Variable Complexity**: Different problems require different amounts of thinking
2. **Efficiency**: Don't waste computation on simple problems
3. **Performance**: More computation where it's needed most
4. **Interpretability**: Can see how "hard" the model thinks a problem is

### Example: Puzzle Complexity

Consider these sequence completion tasks:

```python
# Simple pattern (likely needs few ACT steps)
simple = [1, 2, 3, 4, ?]  # Answer: 5

# Medium pattern (may need moderate ACT steps)  
medium = [1, 1, 2, 3, 5, 8, ?]  # Fibonacci: Answer: 13

# Complex pattern (likely needs many ACT steps)
complex = [2, 3, 5, 7, 11, 13, ?]  # Primes: Answer: 17
```

HRM with ACT can automatically allocate more computation to the harder problems.

## How ACT Works in HRM

HRM implements ACT through a sophisticated mechanism involving:

1. **Carry States**: Maintain information between ACT steps
2. **Q-Learning**: Learn when to halt computation
3. **Hierarchical Processing**: H-level and L-level modules work together

### Basic ACT Flow

```python
def hrm_with_act(input_tokens, max_steps=64):
    carry = initial_carry_state()
    
    for step in range(max_steps):
        # Run one ACT step
        carry, outputs = hrm_step(input_tokens, carry)
        
        # Check if model wants to halt
        q_halt = outputs['q_halt_logits']
        if should_halt(q_halt, step):
            break
    
    return carry, outputs
```

### The Two-Level Architecture

HRM uses a hierarchical structure within each ACT step:

```python
def hrm_step(input_tokens, carry):
    # H-level: High-level planning and reasoning
    h_carry, h_output = h_level_module(input_tokens, carry.h_state)
    
    # L-level: Low-level computation and implementation  
    l_carry, l_output = l_level_module(h_output, carry.l_state)
    
    # Combine outputs and update carry
    new_carry = update_carry_state(carry, h_carry, l_carry)
    outputs = combine_outputs(h_output, l_output)
    
    return new_carry, outputs
```

## The Q-Learning Component

ACT uses Q-learning to decide when to halt computation. The model learns two Q-values:

- **Q-halt**: Value of stopping computation now
- **Q-continue**: Value of continuing computation

### Q-Learning Training

```python
# During training, the model learns optimal halting policies
def compute_act_loss(outputs, targets, step):
    # Language modeling loss
    lm_loss = cross_entropy(outputs['logits'], targets)
    
    # Q-halt loss: binary classification of whether to halt
    q_halt_loss = binary_cross_entropy(
        outputs['q_halt_logits'], 
        should_halt_target(step)
    )
    
    # Q-continue loss: temporal difference learning
    q_continue_loss = td_loss(
        outputs['q_continue_logits'],
        future_rewards(step)
    )
    
    return lm_loss + 0.5 * (q_halt_loss + q_continue_loss)
```

### Exploration vs. Exploitation

During training, the model explores different halting strategies:

```python
def should_halt(q_halt_logits, step, training=True):
    if training:
        # Exploration: sometimes halt randomly
        if random.random() < exploration_rate:
            return random.choice([True, False])
    
    # Exploitation: use Q-values
    halt_prob = sigmoid(q_halt_logits)
    return halt_prob > 0.5
```

## Carry States

Carry states are crucial for maintaining information across ACT steps. They contain:

- **H-level state**: High-level reasoning context
- **L-level state**: Low-level computation context
- **Metadata**: Step count, halt status, etc.

### Carry State Structure

```python
@dataclass
class HRMCarry:
    # Core states
    h_state: mx.array  # H-level hidden state
    l_state: mx.array  # L-level hidden state
    
    # ACT metadata
    act_step: mx.array      # Current ACT step number
    halted: mx.array        # Whether sequence has halted
    ponder_cost: mx.array   # Accumulated computation cost
    
    # Attention states (for efficiency)
    h_attn_cache: Optional[mx.array] = None
    l_attn_cache: Optional[mx.array] = None
```

### Working with Carry States

```python
# Initialize carry for a batch
carry = model.initial_carry(batch_size=4)

# Manual ACT loop
for step in range(10):
    carry, outputs = model(batch, carry)
    
    print(f"Step {step}:")
    print(f"  ACT steps: {carry.act_step.tolist()}")
    print(f"  Halted: {carry.halted.tolist()}")
    
    # Check if all sequences halted
    if carry.halted.all():
        print("All sequences halted!")
        break
```

## Training with ACT

Training HRM with ACT requires special considerations:

### 1. Loss Function Components

```python
class ACTLossHead(nn.Module):
    def __init__(self, model, loss_type='stablemax'):
        self.model = model
        self.loss_type = loss_type
    
    def __call__(self, batch):
        carry, outputs = self.model(batch)
        
        # Main language modeling loss
        if self.loss_type == 'stablemax':
            lm_loss = stablemax_cross_entropy(outputs['logits'], batch['labels'])
        else:
            lm_loss = softmax_cross_entropy(outputs['logits'], batch['labels'])
        
        # ACT losses
        q_halt_loss = self.compute_q_halt_loss(outputs, carry)
        q_continue_loss = self.compute_q_continue_loss(outputs, carry)
        
        # Combine losses
        total_loss = lm_loss + 0.5 * (q_halt_loss + q_continue_loss)
        
        return total_loss, {
            'lm_loss': lm_loss,
            'q_halt_loss': q_halt_loss,
            'q_continue_loss': q_continue_loss,
            'total_loss': total_loss
        }
```

### 2. Gradient Handling

ACT requires careful gradient handling:

```python
# Carry states should not accumulate gradients across steps
def detach_carry(carry):
    return HRMCarry(
        h_state=mx.stop_gradient(carry.h_state),
        l_state=mx.stop_gradient(carry.l_state),
        act_step=carry.act_step,  # Keep metadata
        halted=carry.halted,
        ponder_cost=carry.ponder_cost
    )
```

### 3. Training Loop

```python
def train_step(model, batch, optimizer):
    def loss_fn(model):
        # Start with fresh carry for each training step
        carry = model.initial_carry(batch['input_ids'].shape[0])
        carry, outputs = model(batch, carry)
        
        return compute_act_loss(outputs, batch['labels'], carry)
    
    loss, grads = mx.value_and_grad(loss_fn)(model)
    optimizer.update(model, grads)
    
    return loss
```

## Monitoring ACT Behavior

Understanding how your model uses ACT is crucial for optimization:

### 1. Basic ACT Statistics

```python
def analyze_act_usage(model, test_data):
    act_steps = []
    halt_rates = []
    
    for batch in test_data:
        carry, outputs = model(batch)
        
        # Collect ACT step counts
        act_steps.extend(carry.act_step.tolist())
        
        # Collect halt rates
        halt_rates.extend(carry.halted.tolist())
    
    print(f"Average ACT steps: {sum(act_steps) / len(act_steps):.2f}")
    print(f"Max ACT steps: {max(act_steps)}")
    print(f"Min ACT steps: {min(act_steps)}")
    print(f"Halt rate: {sum(halt_rates) / len(halt_rates):.2%}")
```

### 2. ACT Distribution Visualization

```python
import matplotlib.pyplot as plt

def plot_act_distribution(act_steps):
    plt.figure(figsize=(10, 6))
    
    # Histogram of ACT steps
    plt.subplot(1, 2, 1)
    plt.hist(act_steps, bins=range(1, max(act_steps) + 2), alpha=0.7)
    plt.xlabel('ACT Steps')
    plt.ylabel('Frequency')
    plt.title('Distribution of ACT Steps')
    
    # Cumulative distribution
    plt.subplot(1, 2, 2)
    sorted_steps = sorted(act_steps)
    cumulative = [i/len(sorted_steps) for i in range(1, len(sorted_steps) + 1)]
    plt.plot(sorted_steps, cumulative)
    plt.xlabel('ACT Steps')
    plt.ylabel('Cumulative Probability')
    plt.title('Cumulative ACT Distribution')
    
    plt.tight_layout()
    plt.show()
```

### 3. Problem Complexity Analysis

```python
def analyze_complexity_vs_act(model, puzzles_by_difficulty):
    for difficulty, puzzles in puzzles_by_difficulty.items():
        act_steps = []
        
        for puzzle in puzzles:
            batch = {'input_ids': encode_puzzle(puzzle).reshape(1, -1)}
            carry, _ = model(batch)
            act_steps.append(carry.act_step[0].item())
        
        avg_steps = sum(act_steps) / len(act_steps)
        print(f"{difficulty} puzzles: {avg_steps:.1f} average ACT steps")
```

## Tuning ACT Parameters

Several parameters control ACT behavior:

### 1. Maximum ACT Steps

```python
# In model configuration
config = HRMConfig(
    halt_max_steps=64,  # Maximum ACT steps allowed
    # ... other config
)

# The model will halt after halt_max_steps even if it wants to continue
```

### 2. Halt Threshold

```python
# During inference, you can control halting sensitivity
def should_halt_with_threshold(q_halt_logits, threshold=0.5):
    halt_prob = mx.sigmoid(q_halt_logits)
    return halt_prob > threshold

# Higher threshold = more computation
# Lower threshold = less computation
```

### 3. Loss Weights

```python
# Adjust relative importance of ACT losses
total_loss = lm_loss + alpha * q_halt_loss + beta * q_continue_loss

# Typical values:
# alpha = beta = 0.5 (balanced)
# alpha = 0.1, beta = 0.1 (focus on language modeling)
# alpha = 1.0, beta = 1.0 (focus on ACT learning)
```

## Common Patterns

### 1. Early Halting for Simple Problems

```python
# Simple patterns typically halt early
simple_sequence = mx.array([[1, 2, 3, 4, 5]])
carry, _ = model({'input_ids': simple_sequence})
print(f"Simple sequence ACT steps: {carry.act_step[0].item()}")
# Often: 1-3 steps
```

### 2. Extended Computation for Complex Problems

```python
# Complex patterns use more ACT steps
complex_sequence = mx.array([[2, 3, 5, 7, 11, 13, 17]])  # Primes
carry, _ = model({'input_ids': complex_sequence})
print(f"Complex sequence ACT steps: {carry.act_step[0].item()}")
# Often: 5-20+ steps
```

### 3. Consistent Patterns Within Problem Types

```python
# Similar problems tend to use similar ACT step counts
fibonacci_sequences = [
    [1, 1, 2, 3, 5],
    [1, 1, 2, 3, 5, 8],
    [1, 1, 2, 3, 5, 8, 13]
]

for seq in fibonacci_sequences:
    batch = {'input_ids': mx.array([seq])}
    carry, _ = model(batch)
    print(f"Fibonacci {seq}: {carry.act_step[0].item()} ACT steps")
# Often similar step counts for same pattern type
```

## Advanced Usage

### 1. Manual ACT Control

```python
def run_controlled_act(model, input_ids, max_steps=10):
    """Run ACT with manual control over each step."""
    carry = model.initial_carry(1)
    batch = {'input_ids': input_ids.reshape(1, -1)}
    
    step_outputs = []
    
    for step in range(max_steps):
        carry, outputs = model(batch, carry)
        
        step_info = {
            'step': step,
            'logits': outputs['logits'],
            'q_halt': outputs['q_halt_logits'],
            'act_step': carry.act_step[0].item(),
            'halted': carry.halted[0].item()
        }
        step_outputs.append(step_info)
        
        # Manual halt decision
        if carry.halted[0] or step >= max_steps - 1:
            break
    
    return step_outputs
```

### 2. ACT-Aware Beam Search

```python
def act_beam_search(model, prompt, beam_size=5, max_length=50):
    """Beam search that considers ACT costs."""
    beams = [(prompt, 0.0, 0)]  # (sequence, log_prob, act_cost)
    
    for _ in range(max_length):
        new_beams = []
        
        for sequence, log_prob, act_cost in beams:
            # Get next token probabilities
            carry, outputs = model({'input_ids': sequence.reshape(1, -1)})
            next_logits = outputs['logits'][0, -1]
            
            # Add ACT cost penalty
            act_penalty = carry.act_step[0].item() * 0.1  # Tunable penalty
            
            # Expand beam
            top_k = mx.topk(next_logits, beam_size)
            for i in range(beam_size):
                token = top_k.indices[i]
                token_prob = mx.log_softmax(next_logits)[token]
                
                new_seq = mx.concatenate([sequence, mx.array([token])])
                new_log_prob = log_prob + token_prob.item()
                new_act_cost = act_cost + act_penalty
                
                new_beams.append((new_seq, new_log_prob - new_act_cost, new_act_cost))
        
        # Keep top beams
        beams = sorted(new_beams, key=lambda x: x[1], reverse=True)[:beam_size]
    
    return beams[0][0]  # Return best sequence
```

### 3. Curriculum Learning with ACT

```python
def act_curriculum_training(model, datasets_by_difficulty):
    """Train with curriculum based on ACT usage."""
    
    # Start with problems that need fewer ACT steps
    for difficulty in ['easy', 'medium', 'hard']:
        dataset = datasets_by_difficulty[difficulty]
        
        print(f"Training on {difficulty} problems...")
        
        # Adjust ACT-related hyperparameters by difficulty
        if difficulty == 'easy':
            max_steps = 16
            loss_weight = 0.3
        elif difficulty == 'medium':
            max_steps = 32
            loss_weight = 0.5
        else:  # hard
            max_steps = 64
            loss_weight = 0.7
        
        # Update model configuration
        model.config.halt_max_steps = max_steps
        
        # Train on this difficulty level
        train_on_dataset(model, dataset, act_loss_weight=loss_weight)
```

## Debugging ACT Issues

### 1. Model Never Halts

```python
# Check if model is learning to halt
def check_halt_learning(model, test_data):
    halt_probs = []
    
    for batch in test_data:
        carry, outputs = model(batch)
        halt_prob = mx.sigmoid(outputs['q_halt_logits']).mean()
        halt_probs.append(halt_prob.item())
    
    avg_halt_prob = sum(halt_probs) / len(halt_probs)
    print(f"Average halt probability: {avg_halt_prob:.3f}")
    
    if avg_halt_prob < 0.1:
        print("WARNING: Model rarely wants to halt!")
        print("Try: Lower halt threshold, increase Q-halt loss weight")
```

### 2. Model Always Halts Immediately

```python
# Check if model is halting too eagerly
def check_premature_halting(model, test_data):
    early_halts = 0
    total_sequences = 0
    
    for batch in test_data:
        carry, _ = model(batch)
        
        for act_steps in carry.act_step:
            if act_steps <= 1:
                early_halts += 1
            total_sequences += 1
    
    early_halt_rate = early_halts / total_sequences
    print(f"Early halt rate: {early_halt_rate:.1%}")
    
    if early_halt_rate > 0.8:
        print("WARNING: Model halts too early!")
        print("Try: Higher halt threshold, decrease Q-halt loss weight")
```

### 3. ACT Steps Don't Correlate with Difficulty

```python
def check_act_difficulty_correlation(model, easy_data, hard_data):
    easy_steps = []
    hard_steps = []
    
    for batch in easy_data:
        carry, _ = model(batch)
        easy_steps.extend(carry.act_step.tolist())
    
    for batch in hard_data:
        carry, _ = model(batch)
        hard_steps.extend(carry.act_step.tolist())
    
    easy_avg = sum(easy_steps) / len(easy_steps)
    hard_avg = sum(hard_steps) / len(hard_steps)
    
    print(f"Easy problems: {easy_avg:.1f} average ACT steps")
    print(f"Hard problems: {hard_avg:.1f} average ACT steps")
    
    if hard_avg <= easy_avg:
        print("WARNING: ACT doesn't correlate with difficulty!")
        print("Try: Better problem labeling, curriculum training")
```

## Conclusion

The ACT mechanism is what makes HRM particularly effective for reasoning tasks. Key takeaways:

1. **ACT enables adaptive computation** based on problem complexity
2. **Q-learning trains the halting policy** to be optimal
3. **Carry states maintain information** across ACT steps
4. **Monitoring ACT behavior** is crucial for understanding your model
5. **Proper training** requires balancing multiple loss components

Understanding and properly using ACT will help you get the most out of HRM for your reasoning tasks!

---

*For more details on implementation, see the source code in `src/mlx_hrm/modules/act.py` and related files.*