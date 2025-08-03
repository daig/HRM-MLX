"""
Integration tests for complete HRM model.

Tests model creation, forward pass, checkpoint operations, and API usability.
"""

import mlx.core as mx
import tempfile
from pathlib import Path

from mlx_hrm import (
    HRM,
    create_hrm,
    create_hrm_from_checkpoint,
    list_available_presets,
    get_model_info,
    save_checkpoint,
    load_checkpoint
)
from mlx_hrm.configs import HRM_TINY, HRM_SMALL
from mlx_hrm.utils import get_checkpoint_info


def create_test_batch(batch_size: int = 2, seq_len: int = 32, vocab_size: int = 1000):
    """Create a test batch with random data."""
    return {
        'input_ids': mx.random.randint(0, vocab_size, (batch_size, seq_len)),
        'labels': mx.random.randint(0, vocab_size, (batch_size, seq_len))
    }


class TestModelCreation:
    """Test model creation with different configurations."""
    
    def test_create_with_preset(self):
        """Test creating models with preset configurations."""
        for preset in ['tiny', 'small']:
            model = create_hrm(preset)
            assert isinstance(model, HRM)
            assert model.config.hidden_size > 0
            print(f"Created {preset} model: {model}")
    
    def test_create_with_config_dict(self):
        """Test creating model with dictionary configuration."""
        config_dict = {
            'hidden_size': 128,
            'num_heads': 4,
            'head_dim': 32,
            'num_key_value_heads': 4,
            'H_layers': 2,
            'L_layers': 2,
            'H_cycles': 1,
            'L_cycles': 1,
            'vocab_size': 500,
            'seq_len': 64,
            'batch_size': 4
        }
        model = create_hrm(config_dict)
        assert isinstance(model, HRM)
        assert model.config.hidden_size == 128
    
    def test_create_with_config_object(self):
        """Test creating model with HRMConfig object."""
        model = create_hrm(HRM_TINY)
        assert isinstance(model, HRM)
        assert model.config == HRM_TINY
    
    def test_list_presets(self):
        """Test listing available presets."""
        presets = list_available_presets()
        assert 'tiny' in presets
        assert 'small' in presets
        assert 'base' in presets
        assert len(presets) >= 3
    
    def test_get_model_info(self):
        """Test getting model information."""
        # Test with preset
        info = get_model_info('small')
        assert info['name'] == 'small'
        assert info['hidden_size'] == 512
        assert info['estimated_params_millions'] > 20  # ~27M
        
        # Test with config
        info = get_model_info(HRM_TINY)
        assert info['name'] == 'custom'
        assert info['hidden_size'] == 256


class TestForwardPass:
    """Test model forward pass functionality."""
    
    def test_basic_forward(self):
        """Test basic forward pass."""
        model = create_hrm('tiny')
        batch = create_test_batch(batch_size=2, seq_len=32)
        
        carry, outputs = model(batch)
        
        # Check outputs structure
        assert 'logits' in outputs
        assert 'q_halt_logits' in outputs
        assert 'q_continue_logits' in outputs
        assert 'halted' in outputs
        
        # Check shapes
        print(f"Logits shape: {outputs['logits'].shape}")
        print(f"Expected shape: (2, 32, {model.config.vocab_size})")
        assert outputs['logits'].shape == (2, 32, model.config.vocab_size)
        assert outputs['q_halt_logits'].shape == (2,)
        assert outputs['q_continue_logits'].shape == (2,)
        assert outputs['halted'].shape == (2,)
    
    def test_forward_with_carry(self):
        """Test forward pass with carry state."""
        model = create_hrm('tiny')
        batch = create_test_batch(batch_size=2, seq_len=16)
        
        # First forward pass
        carry1 = model.initial_carry(2)
        carry2, outputs1 = model(batch, carry1)
        
        # Second forward pass with carry
        carry3, outputs2 = model(batch, carry2)
        
        # Outputs should be different due to carry state
        assert not mx.allclose(outputs1['logits'], outputs2['logits'])
        
        # Check halting behavior
        assert mx.any(carry3.halted != carry2.halted) or mx.all(carry3.halted)
    
    def test_forward_single(self):
        """Test single sequence forward."""
        model = create_hrm('tiny')
        input_ids = mx.array([1, 2, 3, 4, 5])
        
        logits = model.forward_single(input_ids)
        
        assert logits.shape == (5, model.config.vocab_size)
        assert logits.dtype == mx.float32
    
    def test_different_batch_sizes(self):
        """Test model with different batch sizes."""
        model = create_hrm('tiny')
        
        for batch_size in [1, 4, 16]:
            batch = create_test_batch(batch_size=batch_size, seq_len=32)
            carry, outputs = model(batch)
            
            assert outputs['logits'].shape[0] == batch_size
            assert outputs['q_halt_logits'].shape[0] == batch_size


class TestGeneration:
    """Test text generation functionality."""
    
    def test_basic_generation(self):
        """Test basic text generation."""
        model = create_hrm('tiny')
        prompt = mx.array([1, 2, 3])
        
        generated = model.generate(prompt, max_length=10)
        
        assert generated.ndim == 1
        assert generated.shape[0] <= 10
        assert generated.shape[0] >= 3  # At least the prompt
        assert mx.all(generated[:3] == prompt)  # Prompt preserved
    
    def test_generation_with_temperature(self):
        """Test generation with different temperatures."""
        model = create_hrm('tiny')
        prompt = mx.array([1, 2, 3])
        
        # Low temperature (more deterministic)
        gen1 = model.generate(prompt, max_length=20, temperature=0.1)
        
        # High temperature (more random)
        gen2 = model.generate(prompt, max_length=20, temperature=2.0)
        
        # Both should start with prompt
        assert mx.all(gen1[:3] == prompt)
        assert mx.all(gen2[:3] == prompt)
    
    def test_generation_with_stop_token(self):
        """Test generation with stop token."""
        model = create_hrm('tiny')
        prompt = mx.array([1, 2, 3])
        stop_token = 999
        
        generated = model.generate(
            prompt, 
            max_length=50,
            stop_token_id=stop_token
        )
        
        # Should stop at or before max_length
        assert generated.shape[0] <= 50


class TestCheckpointing:
    """Test checkpoint save/load functionality."""
    
    def test_save_load_checkpoint(self):
        """Test saving and loading checkpoints."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create and save model
            model1 = create_hrm('tiny')
            checkpoint_path = Path(tmpdir) / 'test_checkpoint'
            
            save_checkpoint(
                model1, 
                str(checkpoint_path),
                metadata={'test': True, 'step': 100}
            )
            
            # Check checkpoint files exist
            assert (checkpoint_path / 'config.json').exists()
            assert (checkpoint_path / 'weights.npz').exists()
            assert (checkpoint_path / 'metadata.json').exists()
            
            # Load checkpoint
            model2 = create_hrm_from_checkpoint(str(checkpoint_path))
            
            # Compare outputs
            batch = create_test_batch(2, 32)
            outputs1 = model1(batch)[1]
            outputs2 = model2(batch)[1]
            
            assert mx.allclose(outputs1['logits'], outputs2['logits'], atol=1e-6)
    
    def test_checkpoint_info(self):
        """Test getting checkpoint information."""
        with tempfile.TemporaryDirectory() as tmpdir:
            model = create_hrm('tiny')
            checkpoint_path = Path(tmpdir) / 'test_checkpoint'
            
            save_checkpoint(
                model,
                str(checkpoint_path),
                metadata={'step': 1000, 'loss': 0.123}
            )
            
            info = get_checkpoint_info(str(checkpoint_path))
            
            assert info['step'] == 1000
            assert info['loss'] == 0.123
            assert 'timestamp' in info
            assert 'weights_size_mb' in info
    
    def test_weights_only_save_load(self):
        """Test saving and loading weights only."""
        with tempfile.TemporaryDirectory() as tmpdir:
            model1 = create_hrm('tiny')
            weights_path = Path(tmpdir) / 'weights.npz'
            
            # Save weights only
            model1.save_weights(str(weights_path))
            assert weights_path.exists()
            
            # Load into new model
            model2 = create_hrm('tiny')
            model2.load_weights(str(weights_path))
            
            # Compare outputs
            batch = create_test_batch(2, 32)
            outputs1 = model1(batch)[1]
            outputs2 = model2(batch)[1]
            
            assert mx.allclose(outputs1['logits'], outputs2['logits'], atol=1e-6)


class TestModelProperties:
    """Test model properties and methods."""
    
    def test_num_parameters(self):
        """Test parameter counting."""
        model = create_hrm('tiny')
        num_params = model.num_parameters
        
        assert num_params > 0
        assert num_params < 10_000_000  # Less than 10M for tiny
        
        # Compare with estimate
        info = get_model_info('tiny')
        estimated = info['estimated_params']
        
        # Should be within 10% of estimate
        assert abs(num_params - estimated) / estimated < 0.1
    
    def test_model_repr(self):
        """Test model string representation."""
        model = create_hrm('small')
        repr_str = repr(model)
        
        assert 'HRM' in repr_str
        assert 'hidden_size=512' in repr_str
        assert 'num_heads=8' in repr_str
        assert 'num_parameters=' in repr_str


class TestErrorHandling:
    """Test error handling and edge cases."""
    
    def test_invalid_preset(self):
        """Test error on invalid preset."""
        try:
            create_hrm('invalid_preset')
            assert False, "Should have raised ValueError"
        except ValueError as e:
            assert "Unknown preset" in str(e)
    
    def test_missing_checkpoint(self):
        """Test error on missing checkpoint."""
        try:
            create_hrm_from_checkpoint('nonexistent/path')
            assert False, "Should have raised FileNotFoundError"
        except FileNotFoundError:
            pass
    
    def test_invalid_config(self):
        """Test error on invalid configuration."""
        try:
            create_hrm({'hidden_size': 'not_a_number'})
            assert False, "Should have raised TypeError"
        except (TypeError, ValueError):
            pass


def test_full_integration():
    """Test complete integration workflow."""
    with tempfile.TemporaryDirectory() as tmpdir:
        # 1. Create model
        model = create_hrm('tiny')
        print(f"Created model: {model}")
        
        # 2. Run forward pass
        batch = create_test_batch(4, 64)
        carry, outputs = model(batch)
        assert outputs['logits'].shape == (4, 64, model.config.vocab_size)
        
        # 3. Generate text
        prompt = mx.array([1, 2, 3, 4, 5])
        generated = model.generate(prompt, max_length=20)
        assert generated.shape[0] > 5
        
        # 4. Save checkpoint
        checkpoint_path = Path(tmpdir) / 'integration_test'
        save_checkpoint(model, str(checkpoint_path), metadata={'test': 'integration'})
        
        # 5. Load checkpoint
        model2 = create_hrm_from_checkpoint(str(checkpoint_path))
        
        # 6. Verify loaded model works
        outputs2 = model2(batch)[1]
        assert mx.allclose(outputs['logits'], outputs2['logits'], atol=1e-6)
        
        print("Full integration test passed!")


if __name__ == "__main__":
    # Run key tests
    print("Testing model creation...")
    test_create = TestModelCreation()
    test_create.test_create_with_preset()
    test_create.test_get_model_info()
    
    print("\nTesting forward pass...")
    test_forward = TestForwardPass()
    test_forward.test_basic_forward()
    test_forward.test_forward_single()
    
    print("\nTesting generation...")
    test_gen = TestGeneration()
    test_gen.test_basic_generation()
    
    print("\nTesting checkpointing...")
    test_ckpt = TestCheckpointing()
    test_ckpt.test_save_load_checkpoint()
    
    print("\nRunning full integration test...")
    test_full_integration()
    
    print("\nAll integration tests passed!")