#!/usr/bin/env python3
"""
Quick validation script for the modular IBYDMT framework
"""
import sys
import os
sys.path.append('.')

# Test basic imports
try:
    from config.experiment_config import ExperimentPresets
    from run_modular_experiments import run_single_experiment
    print("✅ All imports successful")
except ImportError as e:
    print(f"❌ Import error: {e}")
    sys.exit(1)

def test_framework():
    """Test the framework with a minimal experiment"""
    print("Testing modular framework with small dataset...")
    
    # Create a small test config
    config = ExperimentPresets.basic_linear()
    config.n_samples = 300  # Small for quick test
    config.d_hidden = 15
    config.m_concepts = 5
    config.create_visualizations = False  # Skip visualizations for speed
    config.save_results = False  # Skip saving for speed
    
    print(f"Configuration: {config.n_samples} samples, {config.d_hidden} features, {config.m_concepts} concepts")
    
    # Run the experiment
    try:
        results = run_single_experiment(config, visualize=False)
        
        if results:
            print("✅ Framework working successfully!")
            overall = results.get('summary', {}).get('overall_performance', {})
            print(f"Precision: {overall.get('mean_precision', 0):.3f}")
            print(f"Recall: {overall.get('mean_recall', 0):.3f}")
            print(f"F1-Score: {overall.get('mean_f1', 0):.3f}")
            
            # Show detected concepts
            detected = results.get('true_concept_results', {})
            true_concepts = [k for k, v in detected.items() if v.get('is_important', False)]
            print(f"Detected true concepts: {true_concepts}")
            
            return True
        else:
            print("❌ Framework test failed - no results returned")
            return False
            
    except Exception as e:
        print(f"❌ Framework test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_framework()
    sys.exit(0 if success else 1)
