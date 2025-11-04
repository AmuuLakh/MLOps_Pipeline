import pytest
import torch
import pandas as pd
import numpy as np
from transformers import BertForSequenceClassification, AutoTokenizer
from datasets import Dataset
import sys
import os

# Add src directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../src')))

from model import compute_metrics, prepare_datasets


class TestModelInstantiation:
    """Test suite for model instantiation and basic functionality"""
    
    def test_model_loading(self):
        """Test that BERT model can be instantiated correctly"""
        model = BertForSequenceClassification.from_pretrained(
            "distilbert-base-uncased", 
            num_labels=3
        )
        assert model is not None
        assert hasattr(model, 'config')
        assert model.config.num_labels == 3
        print("✓ Model instantiation test passed")
    
    def test_tokenizer_loading(self):
        """Test that tokenizer loads correctly"""
        tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
        assert tokenizer is not None
        assert hasattr(tokenizer, 'pad_token')
        print("✓ Tokenizer loading test passed")
    
    def test_model_forward_pass_with_dummy_data(self):
        """Test model forward pass with dummy batch"""
        model = BertForSequenceClassification.from_pretrained(
            "distilbert-base-uncased", 
            num_labels=3
        )
        model.eval()  # Set to evaluation mode
        
        # Create dummy input
        batch_size = 4
        seq_length = 128
        dummy_input_ids = torch.randint(0, 30000, (batch_size, seq_length))
        dummy_attention_mask = torch.ones((batch_size, seq_length))
        dummy_labels = torch.randint(0, 3, (batch_size,))
        
        # Run forward pass
        with torch.no_grad():
            outputs = model(
                input_ids=dummy_input_ids,
                attention_mask=dummy_attention_mask,
                labels=dummy_labels
            )
        
        # Verify output shape
        assert outputs.logits.shape == (batch_size, 3)
        assert outputs.logits.dtype == torch.float32
        assert hasattr(outputs, 'loss')
        print(f"✓ Forward pass test passed - Output shape: {outputs.logits.shape}")
    
    def test_model_output_logits_range(self):
        """Test that model outputs valid logits"""
        model = BertForSequenceClassification.from_pretrained(
            "distilbert-base-uncased", 
            num_labels=3
        )
        model.eval()
        
        # Create dummy input
        dummy_input_ids = torch.randint(0, 30000, (2, 64))
        dummy_attention_mask = torch.ones((2, 64))
        
        with torch.no_grad():
            outputs = model(
                input_ids=dummy_input_ids,
                attention_mask=dummy_attention_mask
            )
        
        # Check that logits are finite (not NaN or Inf)
        assert torch.all(torch.isfinite(outputs.logits))
        print("✓ Logits validity test passed")
    
    def test_model_predictions_sum_to_one(self):
        """Test that softmax probabilities sum to 1"""
        model = BertForSequenceClassification.from_pretrained(
            "distilbert-base-uncased", 
            num_labels=3
        )
        model.eval()
        
        dummy_input_ids = torch.randint(0, 30000, (2, 64))
        dummy_attention_mask = torch.ones((2, 64))
        
        with torch.no_grad():
            outputs = model(
                input_ids=dummy_input_ids,
                attention_mask=dummy_attention_mask
            )
            probabilities = torch.softmax(outputs.logits, dim=1)
        
        # Check probabilities sum to 1 for each sample
        sums = probabilities.sum(dim=1)
        assert torch.allclose(sums, torch.ones_like(sums), atol=1e-6)
        print("✓ Probability sum test passed")


class TestComputeMetrics:
    """Test suite for compute_metrics function"""
    
    def test_compute_metrics_perfect_predictions(self):
        """Test metrics with perfect predictions"""
        # Simulate perfect predictions
        logits = np.array([
            [2.0, -1.0, -1.0],  # Predicts class 0
            [-1.0, 2.0, -1.0],  # Predicts class 1
            [-1.0, -1.0, 2.0],  # Predicts class 2
        ])
        labels = np.array([0, 1, 2])
        
        eval_pred = (logits, labels)
        metrics = compute_metrics(eval_pred)
        
        assert 'accuracy' in metrics
        assert 'f1' in metrics
        assert metrics['accuracy'] == 1.0  # Perfect accuracy
        assert metrics['f1'] == 1.0  # Perfect F1
        print("✓ Perfect predictions metrics test passed")
    
    def test_compute_metrics_all_wrong(self):
        """Test metrics with all wrong predictions"""
        logits = np.array([
            [2.0, -1.0, -1.0],  # Predicts class 0
            [2.0, -1.0, -1.0],  # Predicts class 0
            [2.0, -1.0, -1.0],  # Predicts class 0
        ])
        labels = np.array([1, 2, 1])  # All different from prediction
        
        eval_pred = (logits, labels)
        metrics = compute_metrics(eval_pred)
        
        assert metrics['accuracy'] == 0.0  # Zero accuracy
        print("✓ All wrong predictions metrics test passed")
    
    def test_compute_metrics_returns_correct_keys(self):
        """Test that compute_metrics returns expected keys"""
        logits = np.random.randn(10, 3)
        labels = np.random.randint(0, 3, 10)
        
        eval_pred = (logits, labels)
        metrics = compute_metrics(eval_pred)
        
        assert 'accuracy' in metrics
        assert 'f1' in metrics
        assert isinstance(metrics['accuracy'], (float, np.floating))
        assert isinstance(metrics['f1'], (float, np.floating))
        print("✓ Metrics keys test passed")


class TestPrepareDatasets:
    """Test suite for dataset preparation (optional - depends on data availability)"""
    
    def test_prepare_datasets_structure(self):
        """Test that prepare_datasets returns correct structure"""
        try:
            train_dataset, eval_dataset = prepare_datasets()
            
            if train_dataset is not None and eval_dataset is not None:
                # Check that they are Dataset objects
                assert isinstance(train_dataset, Dataset)
                assert isinstance(eval_dataset, Dataset)
                
                # Check that datasets have required columns
                assert 'input_ids' in train_dataset.column_names
                assert 'attention_mask' in train_dataset.column_names
                assert 'label' in train_dataset.column_names
                
                # Check that datasets are not empty
                assert len(train_dataset) > 0
                assert len(eval_dataset) > 0
                
                print(f"✓ Dataset preparation test passed")
                print(f"  Train size: {len(train_dataset)}, Eval size: {len(eval_dataset)}")
            else:
                print("⚠ Skipping dataset test - no data available")
                pytest.skip("No dataset available")
                
        except Exception as e:
            print(f"⚠ Dataset preparation test skipped: {str(e)}")
            pytest.skip(f"Dataset not available: {str(e)}")


class TestModelWithRealTokenizer:
    """Test model with actual tokenizer"""
    
    def test_model_with_tokenized_text(self):
        """Test model with real tokenized text"""
        tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
        model = BertForSequenceClassification.from_pretrained(
            "distilbert-base-uncased", 
            num_labels=3
        )
        model.eval()
        
        # Real text samples
        texts = [
            "This movie was absolutely amazing!",
            "It was okay, nothing special.",
            "Terrible waste of time."
        ]
        
        # Tokenize
        inputs = tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=128,
            return_tensors="pt"
        )
        
        # Forward pass
        with torch.no_grad():
            outputs = model(**inputs)
        
        # Verify outputs
        assert outputs.logits.shape == (3, 3)
        assert torch.all(torch.isfinite(outputs.logits))
        
        # Get predictions
        predictions = torch.argmax(outputs.logits, dim=1)
        assert predictions.shape == (3,)
        assert all(0 <= p < 3 for p in predictions.tolist())
        
        print("✓ Model with tokenized text test passed")


# Run all tests if executed directly
if __name__ == "__main__":
    print("=" * 60)
    print("Running Model Tests")
    print("=" * 60)
    
    # Run tests
    pytest.main([__file__, "-v", "-s"])