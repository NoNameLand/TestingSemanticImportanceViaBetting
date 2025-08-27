#!/usr/bin/env python3
"""
Comprehensive demonstration of the modular IBYDMT framework.
Shows different configurations and their results.
"""
import sys
sys.path.append('.')

from config.experiment_config import ExperimentPresets, ExperimentConfig, RelationshipConfig, DistributionConfig
from run_modular_experiments import run_single_experiment
import numpy as np

def demo_preset_experiments():
    """Demonstrate preset experiment configurations."""
    print("="*80)
    print("DEMONSTRATING PRESET EXPERIMENTS")
    print("="*80)
    
    # Test each preset with smaller dataset for speed
    presets = [
        ("Basic Linear", ExperimentPresets.basic_linear()),
        ("Nonlinear Mixed", ExperimentPresets.nonlinear_mixed()),
    ]
    
    all_results = {}
    
    for name, config in presets:
        print(f"\n{'*'*60}")
        print(f"TESTING PRESET: {name}")
        print(f"{'*'*60}")
        
        # Make smaller for demo
        config.n_samples = 500
        config.d_hidden = 20
        config.m_concepts = 6
        config.create_visualizations = False
        config.save_results = False
        
        results = run_single_experiment(config, visualize=False)
        if results:
            all_results[name] = results
            
            # Show summary
            summary = results.get('summary', {})
            overall = summary.get('overall_performance', {})
            print(f"\n📊 SUMMARY FOR {name}:")
            print(f"   Precision: {overall.get('mean_precision', 0):.3f}")
            print(f"   Recall: {overall.get('mean_recall', 0):.3f}")
            print(f"   F1-Score: {overall.get('mean_f1', 0):.3f}")
            
            # Show concept detection
            scenario_results = results.get('scenarios', {})
            if scenario_results:
                first_scenario = list(scenario_results.values())[0]
                expected = first_scenario.get('true_concepts', [])
                
                # Get detected concepts from global SKIT
                global_results = first_scenario.get('global_skit', {})
                if global_results:
                    first_bank = list(global_results.values())[0]
                    detected = [k for k, v in first_bank.items() if v.get('rejected', False)]
                else:
                    detected = []
                
                print(f"   Expected concepts: {expected}")
                print(f"   Detected concepts: {detected}")
            else:
                print(f"   No scenario results available")
        else:
            print(f"❌ Failed to run {name}")

def demo_custom_experiments():
    """Demonstrate custom experiment configurations."""
    print("\n" + "="*80)
    print("DEMONSTRATING CUSTOM EXPERIMENTS")
    print("="*80)
    
    # Custom experiment 1: Trigonometric relationships
    print(f"\n{'*'*60}")
    print("CUSTOM EXPERIMENT 1: Trigonometric Relations")
    print(f"{'*'*60}")
    
    config1 = ExperimentConfig(
        name="custom_trig",
        description="Sine and cosine relationships with mixed distributions",
        n_samples=400,
        d_hidden=15,
        m_concepts=5,
        create_visualizations=False,
        save_results=False
    )
    
    # Set custom distributions
    config1.concept_distributions = {
        0: DistributionConfig.uniform(-np.pi, np.pi),  # Sine input
        1: DistributionConfig.normal(0, 1),
        2: DistributionConfig.uniform(-2, 2),  # Cosine input
        3: DistributionConfig.beta(2, 3, 3, -1),
        4: DistributionConfig.normal(0, 1)
    }
    
    # Set trigonometric relationships
    config1.relationships = [
        RelationshipConfig.trigonometric([0], ["sin"], [1.0]),
        RelationshipConfig.trigonometric([2], ["cos"], [1.0]),
        RelationshipConfig.linear([1], [0.5])  # Add some linear component
    ]
    
    results1 = run_single_experiment(config1, visualize=False)
    if results1:
        summary = results1.get('summary', {})
        overall = summary.get('overall_performance', {})
        print(f"📊 Trigonometric Experiment Results:")
        print(f"   Precision: {overall.get('mean_precision', 0):.3f}")
        print(f"   Recall: {overall.get('mean_recall', 0):.3f}")
        print(f"   F1-Score: {overall.get('mean_f1', 0):.3f}")
    
    # Custom experiment 2: Polynomial relationships
    print(f"\n{'*'*60}")
    print("CUSTOM EXPERIMENT 2: Polynomial Relations")
    print(f"{'*'*60}")
    
    config2 = ExperimentConfig(
        name="custom_poly",
        description="Quadratic and cubic relationships",
        n_samples=400,
        d_hidden=15,
        m_concepts=6,
        create_visualizations=False,
        save_results=False
    )
    
    # Set polynomial relationships
    config2.relationships = [
        RelationshipConfig.polynomial([0, 1], degree=2),  # Quadratic
        RelationshipConfig.polynomial([3], degree=3),     # Cubic
        RelationshipConfig.linear([4], [2.0])            # Linear component
    ]
    
    results2 = run_single_experiment(config2, visualize=False)
    if results2:
        summary = results2.get('summary', {})
        overall = summary.get('overall_performance', {})
        print(f"📊 Polynomial Experiment Results:")
        print(f"   Precision: {overall.get('mean_precision', 0):.3f}")
        print(f"   Recall: {overall.get('mean_recall', 0):.3f}")
        print(f"   F1-Score: {overall.get('mean_f1', 0):.3f}")

def demo_framework_usage():
    """Show how to use the framework components."""
    print("\n" + "="*80)
    print("FRAMEWORK USAGE EXAMPLES")
    print("="*80)
    
    print("""
The modular IBYDMT framework provides:

1. 🏗️  CONFIGURATION SYSTEM:
   - ExperimentConfig: Main experiment settings
   - DistributionConfig: Concept distributions (normal, uniform, beta, gamma, etc.)
   - RelationshipConfig: Y-Z relationships (linear, polynomial, trig, neural nets)
   - ConceptBankConfig: Concept bank generation strategies
   - SKITConfig/RobustTestConfig: Testing parameters

2. 📊 DATA GENERATION:
   - DistributionSampler: Sample from various distributions
   - RelationshipBuilder: Create complex Y-Z relationships
   - ConceptBankGenerator: Generate concept banks
   - ExperimentDataGenerator: Complete experimental scenarios

3. 🧪 TESTING FRAMEWORK:
   - SKITTester: Global SKIT and robust importance testing
   - ExperimentRunner: Complete experiment execution
   - Comprehensive metrics and performance analysis

4. 📈 VISUALIZATION & UTILITIES:
   - ExperimentVisualizer: Plots and analysis
   - Results saving and loading
   - Modular experiment suites

5. 🚀 EASY USAGE PATTERNS:
   
   # Use preset configurations
   config = ExperimentPresets.basic_linear()
   results = run_single_experiment(config)
   
   # Create custom experiments  
   config = ExperimentConfig(name="my_experiment")
   config.relationships = [RelationshipConfig.polynomial([0, 1], degree=2)]
   config.concept_distributions = {0: DistributionConfig.uniform(-2, 2)}
   results = run_single_experiment(config)
   
   # Run experiment suites
   configs = [ExperimentPresets.basic_linear(), ExperimentPresets.nonlinear_mixed()]
   suite_results = run_experiment_suite(configs, "my_suite")
""")

if __name__ == "__main__":
    print("🚀 Starting Comprehensive IBYDMT Framework Demo")
    
    try:
        # Demo preset experiments
        demo_preset_experiments()
        
        # Demo custom experiments
        demo_custom_experiments()
        
        # Show framework usage
        demo_framework_usage()
        
        print("\n" + "="*80)
        print("✅ DEMO COMPLETED SUCCESSFULLY!")
        print("The modular IBYDMT framework is ready for use!")
        print("="*80)
        
    except Exception as e:
        print(f"\n❌ Demo failed with error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
