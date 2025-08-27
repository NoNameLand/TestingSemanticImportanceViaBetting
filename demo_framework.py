#!/usr/bin/env python3
"""
Quick demonstration of the modular framework.
This script shows basic usage and creates a simple custom experiment.
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.append(str(project_root))

from config.experiment_config import (
    ExperimentConfig, ExperimentPresets, RelationshipConfig, 
    DistributionConfig, ConceptBankConfig
)
from run_modular_experiments import run_single_experiment

def demo_preset_experiments():
    """Demonstrate running preset experiments."""
    print("=" * 80)
    print("DEMO: RUNNING PRESET EXPERIMENTS")
    print("=" * 80)
    
    # Run basic linear experiment
    print("\n1. Running basic linear experiment...")
    linear_config = ExperimentPresets.basic_linear()
    # Make it smaller and faster for demo
    linear_config.n_samples = 800
    linear_config.robust_config.n_runs = 15
    
    linear_results = run_single_experiment(linear_config, visualize=True)
    
    return linear_results

def demo_custom_experiment():
    """Demonstrate creating and running a custom experiment."""
    print("\n" + "=" * 80)
    print("DEMO: CREATING CUSTOM EXPERIMENT")
    print("=" * 80)
    
    # Create custom configuration
    custom_config = ExperimentConfig(
        name="demo_custom",
        description="Demo of custom distributions and relationships",
        n_samples=600,  # Smaller for demo
        d_hidden=25,
        m_concepts=6,
        noise_std=0.2
    )
    
    # Custom concept distributions
    custom_config.concept_distributions = {
        0: DistributionConfig.beta(alpha=2, beta=3, scale=3, shift=-1.5),  # Beta [-1.5, 1.5]
        1: DistributionConfig.uniform(low=-2, high=2),                     # Uniform [-2, 2]
        2: DistributionConfig.mixture_gaussian([-0.8, 0.8], [0.4, 0.6], [0.4, 0.6])  # Bimodal
    }
    
    # Custom relationships
    custom_config.relationships = [
        # Polynomial relationship: concepts 0,1 with cubic terms
        RelationshipConfig.polynomial([0, 1], degree=2, coefficients=[1.5, 0.8, -0.5, 1.2]),
        # Trigonometric: sin and cos functions
        RelationshipConfig.trigonometric([2, 4], ["sin", "cos"], [1.8, 2.2]),
        # Simple linear component
        RelationshipConfig.linear([5], [1.0])
    ]
    
    # Custom concept banks
    custom_config.concept_banks = [
        ConceptBankConfig.structured("structured", overlap_ratio=0.15),
        ConceptBankConfig.correlated("correlated", n_base_patterns=2, noise_level=0.4)
    ]
    
    # Faster testing for demo
    custom_config.skit_config.n_pairs = 150
    custom_config.robust_config.n_runs = 12
    
    print("Custom configuration created!")
    print(f"  Relationships: {len(custom_config.relationships)}")
    print(f"  Custom distributions: {len(custom_config.concept_distributions)}")
    print(f"  Concept banks: {len(custom_config.concept_banks)}")
    print(f"  True concepts expected: {[0, 1, 2, 4, 5]} (based on relationships)")
    
    # Run custom experiment
    custom_results = run_single_experiment(custom_config, visualize=True)
    
    return custom_results

def print_results_summary(results, experiment_name):
    """Print a summary of experiment results."""
    print(f"\n{'='*60}")
    print(f"RESULTS SUMMARY: {experiment_name}")
    print("="*60)
    
    if not results or 'summary' not in results:
        print("No results available.")
        return
    
    # Overall performance
    overall = results['summary'].get('overall_performance', {})
    print(f"Overall Performance:")
    print(f"  Precision: {overall.get('mean_precision', 0):.3f}")
    print(f"  Recall: {overall.get('mean_recall', 0):.3f}")
    print(f"  F1-Score: {overall.get('mean_f1', 0):.3f}")
    
    # Per-scenario results
    for scenario_name, scenario_data in results.get('scenarios', {}).items():
        print(f"\nScenario: {scenario_name}")
        print(f"  True concepts: {scenario_data.get('true_concepts', [])}")
        
        for bank_name, bank_results in scenario_data.get('global_skit', {}).items():
            rejected = [cid for cid, r in bank_results.items() if r['rejected']]
            true_rejected = [cid for cid in rejected if cid in scenario_data.get('true_concepts', [])]
            false_rejected = [cid for cid in rejected if cid not in scenario_data.get('true_concepts', [])]
            
            print(f"  {bank_name} bank:")
            print(f"    Rejected: {rejected}")
            print(f"    True positives: {true_rejected}")
            print(f"    False positives: {false_rejected}")

def main():
    """Main demonstration function."""
    print("MODULAR CONCEPT IMPORTANCE TESTING FRAMEWORK - DEMO")
    print("=" * 80)
    print("This demo shows how to use the modular framework for")
    print("concept importance testing with custom configurations.")
    print("=" * 80)
    
    # Create results directory
    Path("results").mkdir(exist_ok=True)
    
    try:
        # Demo 1: Preset experiments
        linear_results = demo_preset_experiments()
        print_results_summary(linear_results, "Basic Linear")
        
        # Demo 2: Custom experiment  
        custom_results = demo_custom_experiment()
        print_results_summary(custom_results, "Custom Demo")
        
        # Final summary
        print(f"\n{'='*80}")
        print("DEMO COMPLETED SUCCESSFULLY!")
        print("="*80)
        print("✅ Preset experiment: Basic linear relationships")
        print("✅ Custom experiment: Mixed distributions and relationships")
        print("\nGenerated files:")
        print("  📁 results/demo_custom/ - Custom experiment results")
        print("  📁 results/basic_linear/ - Basic linear results")
        print("  📊 results/visualizations/ - All plots and visualizations")
        print("\nFramework features demonstrated:")
        print("  ✓ Custom concept distributions (beta, uniform, mixture)")
        print("  ✓ Multiple relationship types (polynomial, trigonometric, linear)")
        print("  ✓ Different concept bank structures")
        print("  ✓ SKIT testing and robust importance analysis")
        print("  ✓ Automatic visualization generation")
        print("  ✓ Structured result saving and analysis")
        
    except Exception as e:
        print(f"\n❌ Demo failed with error: {e}")
        import traceback
        traceback.print_exc()
        print("\nIf you encounter import errors, make sure all dependencies are installed.")

if __name__ == "__main__":
    main()
