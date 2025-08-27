"""
Main experiment runner that ties everything together.
Provides high-level interface for running modular experiments.
"""

from pathlib import Path
import sys
import traceback

# Add project root to path
project_root = Path(__file__).parent
sys.path.append(str(project_root))

from config.experiment_config import ExperimentConfig, ExperimentPresets
from data_generation.generators import ExperimentDataGenerator
from testing.skit_testing import ExperimentRunner
from visualization.plotting import ExperimentVisualizer


def run_single_experiment(config: ExperimentConfig, visualize: bool = True) -> dict:
    """Run a single experiment with the given configuration."""
    print(f"\n{'='*80}")
    print(f"RUNNING EXPERIMENT: {config.name}")
    print(f"Description: {config.description}")
    print("="*80)
    
    try:
        # Generate data
        print("\n1. Generating experimental data...")
        data_generator = ExperimentDataGenerator(config)
        scenario, concept_banks = data_generator.generate_scenario(config.name)
        
        # Run tests
        print("\n2. Running SKIT tests...")
        experiment_runner = ExperimentRunner(config)
        results = experiment_runner.run_experiment(scenario, concept_banks)
        
        # Create visualizations
        if visualize and config.create_visualizations:
            print("\n3. Creating visualizations...")
            visualizer = ExperimentVisualizer(config.output_dir + "/visualizations")
            visualizer.create_all_visualizations(scenario, results, config.name)
            
            # Additional relationship analysis
            visualizer.plot_relationship_analysis(scenario, config, 
                                                visualizer.output_dir / config.name)
        
        print(f"\n✅ Experiment '{config.name}' completed successfully!")
        return results
        
    except Exception as e:
        print(f"\n❌ Experiment '{config.name}' failed with error: {e}")
        traceback.print_exc()
        return {}


def run_experiment_suite(experiment_configs: list, visualize: bool = True) -> dict:
    """Run a suite of experiments and create comparison visualizations."""
    print(f"\n{'='*80}")
    print(f"RUNNING EXPERIMENT SUITE ({len(experiment_configs)} experiments)")
    print("="*80)
    
    all_results = {}
    
    for config in experiment_configs:
        results = run_single_experiment(config, visualize)
        if results:
            all_results[config.name] = results
    
    # Create suite-level comparisons
    if len(all_results) > 1 and visualize:
        print("\nCreating suite-level comparison visualizations...")
        create_suite_comparison(all_results)
    
    print(f"\n✅ Experiment suite completed! {len(all_results)}/{len(experiment_configs)} experiments successful.")
    return all_results


def create_suite_comparison(all_results: dict):
    """Create comparison visualizations across multiple experiments."""
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    
    # Prepare data for comparison
    comparison_data = []
    
    for exp_name, exp_results in all_results.items():
        if 'summary' in exp_results and 'overall_performance' in exp_results['summary']:
            perf = exp_results['summary']['overall_performance']
            comparison_data.append({
                'Experiment': exp_name,
                'Precision': perf['mean_precision'],
                'Recall': perf['mean_recall'],
                'F1-Score': perf['mean_f1'],
                'Precision_Std': perf['std_precision'],
                'Recall_Std': perf['std_recall'],
                'F1_Std': perf['std_f1']
            })
    
    if not comparison_data:
        return
    
    df = pd.DataFrame(comparison_data)
    
    # Create comparison plot
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    
    x = np.arange(len(df))
    width = 0.25
    
    metrics = ['Precision', 'Recall', 'F1-Score']
    colors = ['red', 'blue', 'green']
    
    for i, (metric, color) in enumerate(zip(metrics, colors)):
        values = df[metric].values
        errors = df[f'{metric}_Std'].values
        
        bars = ax.bar(x + i * width, values, width, label=metric, 
                     color=color, alpha=0.7, yerr=errors, capsize=5)
        
        # Add value labels
        for bar, value in zip(bars, values):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                   f'{value:.3f}', ha='center', va='bottom', fontsize=9)
    
    ax.set_xlabel('Experiment')
    ax.set_ylabel('Score')
    ax.set_title('Performance Comparison Across Experiments')
    ax.set_xticks(x + width)
    ax.set_xticklabels(df['Experiment'], rotation=45, ha='right')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1.1)
    
    plt.tight_layout()
    plt.savefig("results/experiment_suite_comparison.pdf", dpi=300, bbox_inches='tight')
    plt.close()
    
    print("Suite comparison saved to: results/experiment_suite_comparison.pdf")


def main():
    """Main function demonstrating the modular framework."""
    print("MODULAR CONCEPT IMPORTANCE TESTING FRAMEWORK")
    print("=" * 80)
    
    # Create output directory
    Path("results").mkdir(exist_ok=True)
    
    # Example 1: Run individual preset experiments
    print("\n--- Running individual experiments ---")
    
    # Basic linear experiment
    linear_config = ExperimentPresets.basic_linear()
    linear_results = run_single_experiment(linear_config)
    
    # Nonlinear mixed experiment
    nonlinear_config = ExperimentPresets.nonlinear_mixed()
    nonlinear_results = run_single_experiment(nonlinear_config)
    
    # Example 2: Custom experiment
    print("\n--- Running custom experiment ---")
    
    from config.experiment_config import (
        ExperimentConfig, RelationshipConfig, DistributionConfig, 
        ConceptBankConfig, SKITConfig, RobustTestConfig
    )
    
    custom_config = ExperimentConfig(
        name="custom_experiment",
        description="Custom experiment with beta distributions and polynomial relationships",
        n_samples=1500,
        d_hidden=30,
        m_concepts=8,
        noise_std=0.15
    )
    
    # Custom concept distributions
    custom_config.concept_distributions = {
        0: DistributionConfig.beta(2, 5, 4, -2),  # Beta distribution
        1: DistributionConfig.gamma(3, 1),        # Gamma distribution  
        2: DistributionConfig.mixture_gaussian([-1, 1], [0.5, 0.8], [0.3, 0.7])  # Mixture
    }
    
    # Custom relationships
    custom_config.relationships = [
        RelationshipConfig.polynomial([0, 1], degree=3),  # Polynomial
        RelationshipConfig.trigonometric([2, 4], ["sin", "cos"], [2.0, 1.5]),  # Trig
        RelationshipConfig.interaction([(0, 1), (2, 4)], "product")  # Interactions
    ]
    
    # Custom concept banks
    custom_config.concept_banks = [
        ConceptBankConfig.orthogonal("orthogonal", 0.2),
        ConceptBankConfig.correlated("correlated", 2, 0.4)
    ]
    
    # Custom testing parameters
    custom_config.skit_config = SKITConfig(alpha=0.005, n_pairs=200)
    custom_config.robust_config = RobustTestConfig(n_runs=30, confidence=0.99)
    
    custom_results = run_single_experiment(custom_config)
    
    # Example 3: Run experiment suite
    print("\n--- Running experiment suite ---")
    
    suite_configs = [
        ExperimentPresets.basic_linear(),
        ExperimentPresets.nonlinear_mixed(),
        ExperimentPresets.interaction_effects(),
        ExperimentPresets.distribution_variety(),
        ExperimentPresets.neural_network_relationship()
    ]
    
    suite_results = run_experiment_suite(suite_configs)
    
    # Print final summary
    print(f"\n{'='*80}")
    print("FINAL SUMMARY")
    print("="*80)
    print("Framework demonstrated successfully!")
    print("\nKey features showcased:")
    print("✓ Modular configuration system")
    print("✓ Flexible distribution sampling")
    print("✓ Multiple relationship types (linear, polynomial, trigonometric, neural network)")
    print("✓ Various concept bank structures")
    print("✓ Comprehensive SKIT testing")
    print("✓ Robust importance analysis")
    print("✓ Automatic visualization generation")
    print("✓ Structured result saving and analysis")
    
    print(f"\nResults and visualizations saved in 'results/' directory")
    print("Check the generated PDFs and JSON files for detailed analysis!")


if __name__ == "__main__":
    main()
